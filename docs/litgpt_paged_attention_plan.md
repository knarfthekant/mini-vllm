### nanovllm paged attention vs. litgpt, and a plan for swappable attention

This document has two goals:

- **Explain how paged attention is implemented in `nanovllm`**, including the Triton kernel and how it is wired into the model forward pass.
- **Propose a detailed design for a swappable normal vs. paged attention implementation in `litgpt`.**

---

### 1. How paged attention is implemented in nanovllm

This is a concise, forward-pass–oriented recap; see `nanovllm_paged_attention.md` for a more exhaustive architecture overview.

#### 1.1 Global KV cache layout and per-layer views

- **Global allocation** happens in `ModelRunner.allocate_kv_cache`:
  - A single **big KV cache tensor** is allocated:
    - Shape: `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`
    - Dim 0: **K vs V**, dim 1: **layer**, dim 2: **physical block id (page)**, dim 3: **token offset inside block**.
  - `num_blocks` is chosen to (roughly) fill a target fraction of GPU memory.
- For each attention layer in the HF model (`Qwen3ForCausalLM`), `ModelRunner` walks the module tree and, for each module that has `k_cache` and `v_cache` attributes:
  - Assigns **layer-local views**:
    - `module.k_cache = self.kv_cache[0, layer_id]` → shape `[num_blocks, block_size, num_kv_heads, head_dim]`
    - `module.v_cache = self.kv_cache[1, layer_id]` → same shape.
- This gives every attention layer a **paged view** over the global KV cache, indexed by `(block_id, token_offset_within_block, head, dim)`.

#### 1.2 Logical paging: sequences, blocks, and the scheduler

- Each sequence is represented by `Sequence`, which:
  - Splits tokens into logical blocks of fixed size `block_size`.
  - Maintains a `block_table: list[int]` which maps **logical block index** → **physical block id**.
  - Tracks:
    - `num_cached_tokens`: how many tokens at the prefix are already materialized in the KV cache (shared prefix blocks, etc.).
    - `num_cached_blocks`, `last_block_num_tokens`, etc.
- `Scheduler` and `BlockManager` orchestrate:
  - **Prefill allocation** (`BlockManager.allocate`), which:
    - Walks each block of the sequence.
    - Computes a hash for full blocks and deduplicates identical blocks (prefix sharing).
    - Assigns or reuses physical blocks and appends their ids into `seq.block_table`.
    - Updates `seq.num_cached_tokens` for fully cached blocks.
  - **Decode extension** (`BlockManager.may_append`), which appends tokens block-by-block and allocates new physical blocks when crossing block boundaries.
  - **Deallocation / preemption** (`BlockManager.deallocate`), which ref-counts blocks and returns pages to the free pool when no sequence uses them.

This forms the **logical paging layer** that decides which block indices (pages) each sequence owns at each step.

#### 1.3 Context object: bridging engine and attention

- `nanovllm.utils.context` holds a global `Context` object with fields like:
  - `is_prefill: bool`
  - `cu_seqlens_q`, `cu_seqlens_k`, `max_seqlen_q`, `max_seqlen_k`
  - `slot_mapping`
  - `context_lens`
  - `block_tables`
- `ModelRunner` fills this `Context` before each `model(input_ids, positions)` call:
  - **Prefill** (`prepare_prefill`):
    - Builds flat `input_ids` and `positions` for all non-cached tokens across scheduled sequences.
    - Builds `cu_seqlens_q` / `cu_seqlens_k` and their max values for varlen FlashAttention.
    - Builds a **flat `slot_mapping`** which, for each new token, stores:
      - `slot = block_id * block_size + offset_within_block`
    - Optionally builds 2D `block_tables[batch, max_blocks]` if there is more K than Q (shared prefix cache).
    - Populates `Context` with `is_prefill=True`, the cu-seqlens, `slot_mapping`, and `block_tables`.
  - **Decode** (`prepare_decode`):
    - For each sequence in the decode batch (one token per seq), collects:
      - `input_ids` (the last token),
      - `positions` (its position in the sequence),
      - `context_lens` (full context length),
      - `slot_mapping` for the **single token being written**:
        - `slot = seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1`
    - Builds `block_tables` again for the full context.
    - Populates `Context` with `is_prefill=False`, `slot_mapping`, `context_lens`, `block_tables`.

The `Context` is thus the **contract between the engine (which knows blocks/pages) and the attention kernels (which just see flattened slots and per-seq block tables)**.

#### 1.4 Triton `store_kvcache` kernel

The core paged write kernel sits in `nanovllm/layers/attention.py`:

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

Python wrapper:

```python
def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        D,
    )
```

**Key points:**

- The kernel runs one **program** per token (`idx` indexes into the batch of tokens being written this step).
- For each token:
  - Reads `slot_mapping[idx]` to get a **linear slot index** into `[num_blocks * block_size, num_heads * head_dim]`.
  - Loads the full per-token `(num_heads * head_dim)` K and V vectors from the temporary `key` / `value` tensors.
  - Writes them contiguously into `k_cache` and `v_cache` at the flattened slot.
- `k_cache` / `v_cache` are viewed such that **flattening over `(block_id, token_offset)` matches the `slot` formula used in `ModelRunner`**.

This kernel is the only place where **logical token positions** and **physical KV pages** are coupled at the tensor level.

#### 1.5 Attention forward: integrating paged attention

The attention module in `nanovllm/layers/attention.py` looks like:

```python
class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:  # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
        else:  # decode
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            )
        return o
```

Integration details:

- **KV write path**:
  - Every forward collects the global `Context`.
  - If per-layer `k_cache` / `v_cache` are non-empty, `store_kvcache` is called to page the current K/V into the global cache using `Context.slot_mapping`.
- **Prefill read path**:
  - If `Context.is_prefill=True` and `Context.block_tables is not None`, we switch fully to reading from `k_cache` / `v_cache`, and call `flash_attn_varlen_func` with:
    - `block_table=context.block_tables` so the kernel can traverse pages per sequence.
    - `cu_seqlens_q` / `cu_seqlens_k` describing variable-length Q and K.
- **Decode read path**:
  - Uses `flash_attn_with_kvcache`, which is specialized for paged KV cache:
    - Takes `q.unsqueeze(1)` (one token per sequence, with a “batch-of-Q” dimension).
    - Takes `k_cache` / `v_cache` and `cache_seqlens=context.context_lens` plus `block_table=context.block_tables` to traverse pages.

Net result: **the HF model stays mostly unaware of paging**; attention simply sees per-layer cache views (`k_cache` / `v_cache`) and a `Context` that encodes how sequences map onto pages.

---

### 2. How litgpt currently does attention and KV caching

This section is based on `litgpt/model.py` from the upstream `Lightning-AI/litgpt` repository.

#### 2.1 Model-level KV cache management

- `GPT` maintains:
  - RoPE caches (`self.cos`, `self.sin`) sized by `max_seq_length`.
  - A `mask_cache` used for efficient masking when KV cache is enabled.
- `GPT.set_kv_cache(batch_size, max_seq_length=None, rope_cache_length=None, device=None, dtype=None)`:
  - Computes effective `max_seq_length` (default `self.max_seq_length`).
  - For each `Block` in `self.transformer.h`:
    - Calls `block.attn.build_kv_cache(...)` and assigns the returned `KVCache` object to `block.attn.kv_cache`.
  - Also pre-allocates a `mask_cache` tensor for attention masking.
- The KV cache abstraction is **centralized in the attention modules** via a `KVCache` helper, not at the engine/scheduler level.

#### 2.2 `CausalSelfAttention` KV cache usage

Key points from `class CausalSelfAttention`:

- Linear projection `self.qkv` produces concatenated Q, K, V.
- Q/K/V are reshaped to `[B, n_heads, T, head_dim]` (with GQA / MQA variants handled via `n_query_groups`).
- Rotary embeddings are applied to leading head dimensions.
- **KV cache integration in forward**:
  - `self.kv_cache: Optional[KVCache] = None` is set by `GPT.set_kv_cache`.
  - In `forward`, after Q/K/V computation and RoPE:
    - If `input_pos is not None`:
      - `k, v = self.kv_cache(input_pos, k, v)`
      - Here `KVCache.__call__` is responsible for:
        - Writing the new K/V at positions specified by `input_pos`.
        - Reading back the full K/V history up to `input_pos_maxp1` (or `max_seq_length`).
  - After optional sliding-window logic, attention is computed via `scaled_dot_product_attention`, which:
    - Uses either `torch.nn.functional.scaled_dot_product_attention` or a manual scores/mask/softmax path, **without paging**.
- The existing KV cache is **dense**:
  - `build_kv_cache` creates `KVCache` with:
    - `k_shape = (batch_size, n_query_groups, effective_cache_size, head_dim_k)`
    - `v_shape = (batch_size, n_query_groups, effective_cache_size, head_dim_v)`
  - `effective_cache_size` is either `max_seq_length` or the sliding-window size.
  - This is conceptually `[batch, head, seq_len, dim]` with a straightforward time dimension.

#### 2.3 `MultiheadLatentAttention` KV cache usage

- Similar pattern:
  - `self.kv_cache: Optional[KVCache] = None`.
  - `build_kv_cache` creates dense `[batch, head, max_seq_length, dim]` KV tensors.
  - In `forward`, when `input_pos` is provided:
    - `k, v = self.kv_cache(input_pos, k, v)` reuses the dense cache.
  - Attention is again performed with dense `scaled_dot_product_attention`, with no paging.

#### 2.4 Takeaways for a swappable normal vs. paged attention design

- **litgpt already has a clean `KVCache` abstraction**, with:
  - Centralized allocation via `GPT.set_kv_cache`.
  - Per-layer pointers `block.attn.kv_cache`.
  - A `KVCache.__call__(input_pos, k, v)` interface used inside attention forward.
- The current implementation implicitly assumes:
  - A **dense per-sequence time axis** (no paging).
  - Attention kernels that take standard `[B, n_heads, T, H]` inputs and, optionally, a dense mask.
- To integrate paged attention, the goal is to:
  - **Generalize the KV cache abstraction** so that dense and paged implementations can be swapped.
  - **Optionally introduce an engine-level scheduler** that builds paging metadata (slot mappings, block tables), similar to `nanovllm`, while minimizing invasive changes to the rest of litgpt.

---

### 3. Design: swappable normal vs. paged attention in litgpt (repo-specific)

In this repository, **most of the paged-attention machinery already exists under `src/`**:

- `src/engine/allocator.py` implements a `PagedBlockManager` that:
  - Manages shared KV blocks with prefix deduplication (`hash_to_block_id`, `PagedBlock.ref_count`, etc.).
  - Tracks per-request `block_table` and `num_cached_tokens` via `PagedRequestState`.
- `src/engine/scheduler.py`:
  - Schedules `Request` objects for prefill + decode.
  - Exports `block_tables` per batch via `SchedulerOutput.block_tables`, using the allocator’s `export_batch_state`.
- `src/worker/cache_manager.py`:
  - Defines `StandardCacheManager` (dense per-sequence KV using `GPT.set_kv_cache`).
  - Defines **`PagedCacheManager`**, which:
    - Computes a KV plan (`num_gpu_blocks`, `max_seq_length`) for the paged backend.
    - Allocates **layer-wise paged KV tensors**:
      - `key_blocks`: `[num_gpu_blocks, n_query_groups, BLOCK_SIZE, key_dim]`
      - `value_blocks`: `[num_gpu_blocks, n_query_groups, BLOCK_SIZE, value_dim]`
    - Returns a `PagedKVCacheState` with per-layer `PagedLayerKVCache`.
- `src/worker/model_runner.py`:
  - Owns the litgpt `GPT` model.
  - Plugs in the chosen cache manager (standard vs paged) based on `VllmConfig.kv_cache_manager`.
  - Drives the execution pipeline:
    - Schedules requests ⇨ `_update_states` ⇨ `_prepare_inputs` (pad `input_ids` / `positions`) ⇨ calls `self._cache_manager.forward(...)`.

Therefore, **the remaining work for swappable normal vs. paged attention is to wire the paged KV pool into the litgpt model’s forward pass**, not to reimplement a KV cache or block manager from scratch.

#### 3.1 What is already done vs. what is missing

- **Already implemented:**
  - Dense KV cache integration via `StandardCacheManager.initialize_kv_cache`:
    - Calls `GPT.set_kv_cache(batch_size, max_seq_length, device)` and uses `GPT.forward(idx, input_pos=positions)`.
  - Paged KV capacity planning and allocation via:
    - `PagedCacheManager.build_kv_cache_plan(...)`
    - `PagedCacheManager.initialize_kv_cache(...)` returning `PagedKVCacheState` with:
      - `num_gpu_blocks`, `block_size`, `max_seq_length`, `layers=[PagedLayerKVCache(...)]`.
  - Paged **logical** management and scheduling:
    - `PagedBlockManager` (prefix sharing, rolling hashes, ref-counts).
    - `Scheduler` producing:
      - `SchedulerOutput.input_ids` / `positions`
      - `SchedulerOutput.block_tables` derived from allocator state.
- **Not yet implemented (TODO):**
  - `PagedCacheManager.forward` integration with the litgpt `GPT` model:
    - It currently raises:
      - `"Paged attention allocation is configured, but paged-attention execution is not wired into the litgpt model yet."`
  - Model-side wiring so that:
    - The litgpt attention layers can **read and write** K/V using the shared paged KV tensors in `PagedKVCacheState.layers`.
    - The model forward uses `block_tables` (and optionally slot mappings) supplied by the scheduler and allocator.

The design below therefore focuses **only** on this integration layer.

#### 3.2 High-level integration strategy

Goals:

- Keep **dense mode (`StandardCacheManager`) working exactly as today**, using `GPT.set_kv_cache` and the built-in `KVCache`.
- For **paged mode (`PagedCacheManager`)**:
  - Bypass `GPT.set_kv_cache` and its dense `KVCache`.
  - Instead, give each attention layer **views into the shared paged KV tensors** from `PagedKVCacheState.layers`.
  - Provide the model with **per-batch metadata** from `SchedulerOutput` (`block_tables`, `positions`) so that:
    - Writes go to the correct `(block_id, offset)` in `key_blocks` / `value_blocks`.
    - Reads reconstruct the correct logical time axis per sequence for attention scores.

Concretely, we want:

- `StandardCacheManager.forward`:
  - Unchanged: `return runner.model(input_ids, input_pos=positions)`.
- `PagedCacheManager.forward`:
  - New behavior:
    - Use `scheduler_output.block_tables` and `model_inputs.positions` to build **paged attention metadata**.
    - Call a modified model entrypoint (or wrapper) that:
      - Writes K/V into paged blocks instead of dense per-sequence KV tensors.
      - Computes attention using the paged layout.

#### 3.3 Paged model entrypoint, metadata, and Triton kernels

To get **nanovllm-comparable performance**, we should avoid reconstructing dense K/V on every step and instead:

- **Store K/V directly into paged blocks via Triton kernels**, and
- **Run attention against the paged layout** using a custom kernel (or FlashAttention’s paged API) that understands `block_tables` and context lengths.

Architecture:

- Add a paged-aware model entrypoint in this repo (e.g. `src/worker/paged_forward.py`) that exposes:

```python
def paged_gpt_forward(
    model: GPT,
    paged_state: PagedKVCacheState,
    input_ids: torch.Tensor,       # (B, T)
    positions: torch.Tensor,       # (B, T)
    block_tables: list[list[int]], # from SchedulerOutput
    is_prefill: bool,
) -> torch.Tensor:
    ...
```

- Introduce a **small “attn metadata” struct** that mirrors `nanovllm`’s `Context` but uses your existing scheduler output:
  - `block_tables: Tensor` (padded 2D `[B, max_blocks]` from `SchedulerOutput.block_tables`).
  - `context_lens: Tensor[int32]` (per-request lengths, easily computed from `positions`).
  - Optional `cu_seqlens_q`, `cu_seqlens_k`, `max_seqlen_q`, `max_seqlen_k` for varlen kernels.
- For **KV writes**, implement Triton kernels analogous to `store_kvcache_kernel` but targeting `PagedLayerKVCache`:
  - Inputs:
    - Per-step K/V (`[N, n_heads, head_dim]` in flat form),
    - `slot_mapping` built from `(block_tables, positions)`:
      - `logical_pos = positions[b, t]`
      - `block_idx = logical_pos // BLOCK_SIZE`
      - `offset = logical_pos % BLOCK_SIZE`
      - `slot = block_id * BLOCK_SIZE + offset`, where `block_id = block_tables[b][block_idx]`.
    - Layer-local `key_blocks` / `value_blocks` flattened along `(block_id, offset)` so the kernel can do:
      - `tl.store(k_cache_ptr + slot * D + ...)`, as in nanovllm.
  - Launch with one program per token (or warp-friendly grouping) to match nanovllm’s pattern.
- For **attention reads**, avoid rebuilding a dense `[B, T, ...]`:
  - Implement either:
    - A custom Triton kernel for paged scaled-dot-product attention that:
      - Iterates the pages for each sequence via `block_tables[b]` and `context_lens[b]`,
      - Streams K/V from `key_blocks` / `value_blocks` directly, or
    - A bridge to **FlashAttention’s paged APIs** (similar to `flash_attn_with_kvcache`) using:
      - `block_table`, `cache_seqlens`, and a `[num_blocks, BLOCK_SIZE, n_heads, head_dim]` layout.

The key is that **writes and reads stay in paged form**; the only dense tensor is Q for the current batch, which is already required.

#### 3.4 Changes inside `PagedCacheManager.forward`

Implement `PagedCacheManager.forward` as the **single integration point** between the existing scheduler/allocator and the paged-attention kernels:

- Inputs:
  - `runner: ModelRunner`
  - `model_inputs: ModelExecutionInputs` with:
    - `input_ids: (B, T)` (already padded)
    - `positions: (B, T)` (already padded, monotonic per sequence)
    - `attn_metadata: None` (for now)
Additional per-batch data:

- Extend `ModelExecutionInputs` to include:
  - `block_tables: list[list[int]]`, passed from `SchedulerOutput.block_tables` in `_prepare_inputs`.
  - A lightweight `is_prefill: bool` or a step-type flag if you later implement multi-token prefill vs. single-token decode differently.

Implementation sketch:

```python
def forward(
    self,
    runner: "ModelRunner",
    model_inputs: ModelExecutionInputs,
) -> torch.Tensor:
    assert isinstance(runner.kv_cache_state, PagedKVCacheState)
    paged_state = runner.kv_cache_state
    meta = model_inputs.attn_metadata
    return paged_gpt_forward(
        runner.model,
        paged_state,
        model_inputs.input_ids,
        model_inputs.positions,
        meta.block_tables,
        meta.is_prefill,
    )
```

Adjustments needed:

- Extend `ModelExecutionInputs` to carry per-batch paged metadata:

```python
@dataclass
class ModelExecutionInputs:
    input_ids: torch.Tensor
    positions: torch.Tensor
    attn_metadata: object | None = None  # for paged: block_tables, maybe context_lens, etc.
```

- In `BaseCacheManager.prepare_model_inputs` or an override in `PagedCacheManager`:
  - Attach `scheduler_output.block_tables` into `attn_metadata` (e.g., as a simple dataclass).

This keeps the `ModelRunner.execute_model` pipeline unchanged and makes paged-vs-dense behavior entirely a concern of the cache manager + a paged-attention fork, while allowing **Triton kernels to handle the hot loops**.

#### 3.5 Model-side changes and Triton-backed attention fork

To reach nanovllm-like performance, we should **fork litgpt’s attention stack in this repo** and make it paged-aware with Triton kernels, while keeping the dense path untouched:

- Leave **`GPT.set_kv_cache` and dense `KVCache`** as-is for `StandardCacheManager`.
- For paged mode:
  - Do **not** call `GPT.set_kv_cache`; instead, rely entirely on `PagedKVCacheState.layers` plus:
    - Triton KV-store kernel(s) (per-layer) that write K/V into `key_blocks` / `value_blocks` using `slot_mapping`.
    - A **paged scaled-dot-product attention kernel** (Triton or FlashAttention paged API) that:
      - Consumes Q (`[B, n_heads, T_q, head_dim]`),
      - Reads K/V from `key_blocks` / `value_blocks` using `block_tables` and `context_lens`,
      - Applies causal masking and returns attention outputs without materializing dense K/V.
- Implement a **local fork** of `CausalSelfAttention` / `MultiheadLatentAttention` in this repo:
  - Keep Q/K/V projection and RoPE logic identical to upstream.
  - Replace `KVCache.__call__` and SDPA with:
    - Calls to the Triton KV-store kernel on each step to update paged storage.
    - Calls to the paged attention kernel to compute outputs.
  - Wire these forked attention modules into a forked `Block` / `GPT` constructor that is only used when `kv_cache_manager == "paged"`.

This approach mirrors nanovllm’s design: **the engine (scheduler + allocator + cache manager) only supplies block tables and slot mappings; the forked attention kernels, written in Triton, handle the high-throughput math on the paged layout.**

#### 3.6 Revised implementation steps (this repo, Triton-focused)

With the goal of **comparable performance to nanovllm**, the concrete, repo-specific TODOs are:

1. **Plumb batch metadata**
   - Extend `ModelExecutionInputs` and related plumbing so that, for paged mode:
     - `attn_metadata` carries `block_tables` (and optionally `is_prefill`, `context_lens`).
   - Keep standard mode behavior unchanged.
2. **Implement Triton KV-store kernel(s) over `PagedLayerKVCache`**
   - Design layouts to match nanovllm’s flattening over `(block_id, offset)` for efficient `slot_mapping`.
   - Implement and test kernels that write per-token K/V into paged storage using `(positions, block_tables)`.
3. **Fork litgpt attention with paged kernels**
   - Add local copies of `CausalSelfAttention` / `MultiheadLatentAttention` that:
     - Use the Triton KV-store kernel to update `PagedKVCacheState.layers`.
     - Invoke a paged attention kernel (Triton or FlashAttention paged) that operates directly on `key_blocks` / `value_blocks` plus `block_tables`, `context_lens`.
   - Add a forked `GPT`/`Block` constructor that wires in these modules when `kv_cache_manager == "paged"`.
4. **Wire `PagedCacheManager.forward` to `paged_gpt_forward`**
   - Retrieve `PagedKVCacheState` and `block_tables` from `runner` / `ModelExecutionInputs`.
   - Call `paged_gpt_forward` which uses the forked attention and Triton kernels.
5. **Validation and benchmarking**
   - Numerical parity tests:
     - `standard` vs `paged` on small prompts, with enough KV capacity.
   - Microbenchmarks:
     - Compare per-step latency vs nanovllm for similar model sizes and batch patterns.
   - Tune Triton launch configs (`num_warps`, block sizes) to close the performance gap.

With this plan, **standard mode** remains a faithful litgpt implementation, while **paged mode** uses a forked attention stack plus Triton kernels and your existing scheduler/allocator to target nanovllm-class throughput.

---

### 4. Summary

- **nanovllm** implements paged attention by combining:
  - A global paged KV tensor, a logical block manager/scheduler, a `Context` bridging layer, a Triton kernel (`store_kvcache`) for slot-based writes, and FlashAttention kernels that understand `block_table` and `cache_seqlens`.
- **litgpt** currently uses a dense `KVCache` abstraction inside attention blocks, which is already amenable to backend swapping.
- The proposed design for litgpt introduces:
  - A **configurable KV cache backend** (`dense` vs `paged`), a `BaseKVCache` interface with `DenseKVCache` and `PagedKVCache` implementations, a `PagedContext` plus optional scheduler, and optional paged attention kernels.
- This plan allows **incremental adoption** of paged attention in litgpt, starting from API-compatible dense emulation and evolving towards a fully paged, highly efficient inference engine similar in spirit to nanovllm.

