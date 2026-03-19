### nanovllm paged attention design

This document explains how **nanovllm** implements paged attention, going from:

- **Logical allocation** of KV cache blocks for sequences
- **Physical layout** of the KV cache tensor on GPU
- **Runtime mapping** from tokens to cache slots (`slot_mapping`, `block_tables`)
- **Attention kernels** (`flash_attn_varlen_func`, `flash_attn_with_kvcache` and the Triton `store_kvcache` kernel)

The relevant components are:

- `Sequence` (`nanovllm/engine/sequence.py`)
- `BlockManager` (`nanovllm/engine/block_manager.py`)
- `Scheduler` (`nanovllm/engine/scheduler.py`)
- `ModelRunner` (`nanovllm/engine/model_runner.py`)
- `Attention` layer (`nanovllm/layers/attention.py`)
- `Context` (`nanovllm/utils/context.py`)

---

### 1. Logical sequence layout and “blocks”

Each sequence is represented by `Sequence`:

```python
class Sequence:
    block_size = 256
    ...
    def __init__(self, token_ids, sampling_params=SamplingParams()):
        self.token_ids = copy(token_ids)
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []      # indices into the global KV cache
    ...
    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    def block(self, i):
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]
```

Key points:

- A **sequence is chunked into fixed-size blocks** of `Sequence.block_size` tokens (256).
- `block_table` is a **logical-to-physical mapping**: for each logical block index `i` of this sequence, `block_table[i]` is the **global KV cache block id** where that segment’s keys/values live.
- `num_cached_tokens` tracks how many initial tokens are already present in KV cache (prefix/paged attention).

The `block_table` is **purely logical** at this level; actual GPU allocations are managed by `BlockManager`.

---

### 2. Global KV cache and physical block allocation

#### 2.1. Allocating the big KV cache tensor

In `ModelRunner.allocate_kv_cache`:

```python
def allocate_kv_cache(self):
    ...
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    head_dim = getattr(hf_config, "head_dim",
                       hf_config.hidden_size // hf_config.num_attention_heads)
    block_bytes = (
        2 * hf_config.num_hidden_layers *
        self.block_size * num_kv_heads * head_dim *
        hf_config.torch_dtype.itemsize
    )
    config.num_kvcache_blocks = int(
        total * config.gpu_memory_utilization - used - peak + current
    ) // block_bytes
    assert config.num_kvcache_blocks > 0

    self.kv_cache = torch.empty(
        2,  # K/V
        hf_config.num_hidden_layers,
        config.num_kvcache_blocks,
        self.block_size,
        num_kv_heads,
        head_dim,
    )

    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

Interpretation:

- There is **one big KV cache tensor**:
  - Shape: `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`.
  - The first dimension indexes **K vs V**.
  - `num_blocks` is the total number of **physical blocks** available on GPU.
- For each attention layer, `module.k_cache` and `module.v_cache` are **views** into this global tensor:
  - `k_cache`: `[num_blocks, block_size, num_kv_heads, head_dim]`
  - `v_cache`: same shape

This implements the **physical paging layer**: the entire memory for KV cache is allocated up front as “pages” (blocks).

#### 2.2. BlockManager: logical→physical mapping and deduplication

`Scheduler` constructs:

```python
self.block_manager = BlockManager(config.num_kvcache_blocks,
                                  config.kvcache_block_size)
```

`BlockManager` manages small objects of type `Block`:

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id     # physical block index in kv_cache
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
```

The manager holds:

- `blocks`: list of `Block` objects of length `num_blocks`.
- `hash_to_block_id`: map from hash(prefix, token_ids) → physical block id.
- `free_block_ids` / `used_block_ids`: which pages are free vs in use.

##### 2.2.1. Allocation (`allocate`)

When a sequence is first scheduled for prefill, `Scheduler.schedule` calls:

```python
self.block_manager.allocate(seq)
```

Inside `allocate`:

- For each logical block index `i` in the sequence (`0..seq.num_blocks-1`):
  - Get `token_ids = seq.block(i)`.
  - Compute a **rolling hash** over that block, optionally using the previous hash as prefix:

    ```python
    h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
    block_id = self.hash_to_block_id.get(h, -1)
    if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
        cache_miss = True
    ```

  - If it is a **cache miss**:
    - Take a free physical block: `block_id = self.free_block_ids[0]`.
    - `_allocate_block(block_id)`:
      - Marks it used, resets metadata, sets `ref_count = 1`.
  - If it is a **cache hit**:
    - **Re-use** that physical block:
      - If already used, increment its `ref_count`.
      - Else, allocate it and set `ref_count = 1`.
    - Also update `seq.num_cached_tokens += block_size` for every full cached block.

  - If `h != -1` (full block):
    - Store `(h, token_ids)` in `Block`.
    - Update `hash_to_block_id[h] = block_id`.

  - Append to sequence’s logical mapping:

    ```python
    seq.block_table.append(block_id)
    ```

This implements **prefix sharing and deduplication**: identical prefix blocks across different sequences can share the same physical KV pages, tracked by `ref_count`.

##### 2.2.2. Deallocation (`deallocate`)

When a sequence finishes or is preempted:

```python
for block_id in reversed(seq.block_table):
    block = self.blocks[block_id]
    block.ref_count -= 1
    if block.ref_count == 0:
        self._deallocate_block(block_id)
seq.num_cached_tokens = 0
seq.block_table.clear()
```

So when `ref_count` reaches zero, that page becomes **free** and can be reused by other sequences.

##### 2.2.3. Append during decoding (`may_append`)

During decoding, sequences grow by 1 token per step. `Scheduler.schedule` (decode path) does:

```python
self.block_manager.may_append(seq)
```

`may_append`:

- If adding the new token creates a **new block** (`len(seq) % block_size == 1`):
  - Allocate a new physical block and append its id to `seq.block_table`.
- If the last block just filled up to a full block (`len(seq) % block_size == 0`):
  - Compute the hash for the last block and update `hash_to_block_id`, enabling **future prefix sharing**.
- Otherwise, the block is partially filled; `last_block.hash` remains `-1` and is not deduplicated yet.

This keeps the logical `block_table` consistent with how many tokens are in the sequence while also maintaining prefix deduplication.

---

### 3. Context object: bridging engine and attention layers

`Context` is a global structure stored in `nanovllm.utils.context`:

```python
@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
```

It carries **all per-batch metadata** needed by the attention kernels:

- **Prefill mode** (`is_prefill = True`):
  - `cu_seqlens_q`, `cu_seqlens_k`, `max_seqlen_q`, `max_seqlen_k`
  - `slot_mapping`: mapping from **new tokens** to **physical cache slots** in `k_cache` / `v_cache`
  - `block_tables`: (optional) per-sequence block tables for prefix cache
- **Decode mode** (`is_prefill = False`):
  - `slot_mapping`: slot index where the current-step token’s KV should be written
  - `context_lens`: length of context per sequence
  - `block_tables`: full block tables per sequence

`ModelRunner` is responsible for **filling the Context** before calling the model.

---

### 4. Preparing paged-attention metadata in ModelRunner

#### 4.1. Prefill (`prepare_prefill`)

For a set of sequences in prefill (initial or extended), `ModelRunner.prepare_prefill` builds:

- Flat `input_ids` and `positions` tensors.
- `cu_seqlens_q` / `cu_seqlens_k` and their maxes.
- `slot_mapping` for newly written tokens.
- Optional `block_tables` for prefix cache.

Key logic:

```python
for seq in seqs:
    seqlen = len(seq)
    # Only the non-cached tail needs to be computed this step
    input_ids.extend(seq[seq.num_cached_tokens:])
    positions.extend(range(seq.num_cached_tokens, seqlen))

    seqlen_q = seqlen - seq.num_cached_tokens  # #tokens this step
    seqlen_k = seqlen                          # total context so far
    ...
    if not seq.block_table:
        continue  # warmup (no cached blocks yet)

    # Build slot_mapping for the blocks that are not cached yet
    for i in range(seq.num_cached_blocks, seq.num_blocks):
        start = seq.block_table[i] * self.block_size
        if i != seq.num_blocks - 1:
            end = start + self.block_size
        else:
            end = start + seq.last_block_num_tokens
        slot_mapping.extend(range(start, end))
```

Interpretation:

- For sequences with existing prefix blocks (`block_table` nonempty):
  - Earlier blocks (up to `num_cached_blocks`) are already **fully materialized** in KV cache.
  - For remaining blocks, for each token index within that block, we construct a **linear slot index**:

    \[
    \text{slot} = \text{block\_id} \times \text{block\_size} + \text{offset\_within\_block}
    \]

- These slots correspond to rows in `k_cache` / `v_cache` views:
  - If `k_cache` is `[num_blocks, block_size, num_heads, head_dim]`, then flattening the first two dims yields length `num_blocks * block_size`, and `slot` indexes into it.

If there is more K than Q (i.e., **prefix cache** with shared blocks), we build dense `block_tables`:

```python
if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
    block_tables = self.prepare_block_tables(seqs)
```

`prepare_block_tables` pads each `seq.block_table` with `-1` to the same length and returns a 2D tensor `[batch_size, max_blocks]`.

Finally, everything is moved to GPU and stored in the global context:

```python
set_context(
    True,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    slot_mapping, None, block_tables
)
```

#### 4.2. Decode (`prepare_decode`)

For decode steps (one token per sequence), we need:

- The **next input token** and its position.
- The **slot** where this new KV should be stored.
- The **context length** (for attention window).
- The `block_tables` per sequence.

`prepare_decode`:

```python
for seq in seqs:
    input_ids.append(seq.last_token)
    positions.append(len(seq) - 1)
    context_lens.append(len(seq))
    slot_mapping.append(
        seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
    )
...
block_tables = self.prepare_block_tables(seqs)
set_context(
    False,
    slot_mapping=slot_mapping,
    context_lens=context_lens,
    block_tables=block_tables,
)
```

Interpretation:

- For each sequence:
  - `seq.last_block_num_tokens - 1` is the index of the last token within the last block.
  - `seq.block_table[-1]` is the **physical block id** for the last block.
  - Their combination yields the **exact slot index** where the KV for the current-step token should be written.
- `context_lens` tells the attention kernel how many tokens in the prefix to use.
- `block_tables` encodes which physical blocks (pages) belong to each sequence.

---

### 5. Attention layer: using the paged cache

The `Attention` module (`nanovllm/layers/attention.py`) receives per-layer `q`, `k`, `v` tensors produced by the model’s projections and interacts with the global `Context` and KV cache.

#### 5.1. The Triton `store_kvcache` kernel

The kernel:

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr, key_stride,
    value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr,
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
    ...
    store_kvcache_kernel[(N,)](
        key, key.stride(0),
        value, value.stride(0),
        k_cache, v_cache,
        slot_mapping, D
    )
```

Interpretation:

- `key` / `value` are **per-token** K/V with shape `[N, num_heads, head_dim]`, flattened by head dimension.
- `slot_mapping[idx]` tells which **global slot** (flattened over `[block_id, offset_in_block]`) this token’s KV should be written to.
- The kernel:
  - Reads K/V at token index `idx`.
  - Writes them into `k_cache` / `v_cache` at `cache_offsets = slot * D + ...`.
- `slot == -1` means “ignore this token” (e.g., unused padding).

This kernel implements the **paged-write** of new KV entries into the global cache tensor.

#### 5.2. Attention forward: prefill vs decode

The `forward` of `Attention`:

```python
def forward(self, q, k, v):
    context = get_context()
    k_cache, v_cache = self.k_cache, self.v_cache

    # 1) Store K/V into the global cache using slot_mapping
    if k_cache.numel() and v_cache.numel():
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

    if context.is_prefill:
        # 2a) Prefill mode
        if context.block_tables is not None:  # prefix cache
            k, v = k_cache, v_cache
        o = flash_attn_varlen_func(
            q, k, v,
            max_seqlen_q=context.max_seqlen_q,
            cu_seqlens_q=context.cu_seqlens_q,
            max_seqlen_k=context.max_seqlen_k,
            cu_seqlens_k=context.cu_seqlens_k,
            softmax_scale=self.scale,
            causal=True,
            block_table=context.block_tables,
        )
    else:
        # 2b) Decode mode
        o = flash_attn_with_kvcache(
            q.unsqueeze(1),  # [B, 1, H, D]
            k_cache, v_cache,
            cache_seqlens=context.context_lens,
            block_table=context.block_tables,
            softmax_scale=self.scale,
            causal=True,
        )
    return o
```

Flow:

1. **Write new K/Vs** into the global `k_cache` / `v_cache` using the `slot_mapping` prepared by `ModelRunner` (prefill or decode).
2. **Prefill**:
   - If `block_tables is not None`, this means we have a prefix cache; `k` and `v` for attention are **taken from the cache** (`k_cache`, `v_cache`) instead of just the per-step K/V.
   - `flash_attn_varlen_func` is called in a **variable-length, block-table-aware** mode:
     - `cu_seqlens_q` / `cu_seqlens_k` + `max_seqlen_*` describe ragged Q/K lengths.
     - `block_table` encodes the layout of prefix blocks per sequence.
3. **Decode**:
   - `flash_attn_with_kvcache` uses `k_cache`, `v_cache`, `cache_seqlens` (i.e., `context_lens`) and `block_table`:
     - It interprets `block_table[batch, block_idx]` as the **physical block index** to read from.
     - Combined with internal logic, it reconstructs the logical context for each sequence without materializing a dense `[batch, seqlen, ...]` K/V tensor.

This is the core of **paged attention**: attention operates directly over **paged KV storage** described by `block_table` and `slot_mapping`, without ever requiring contiguous per-sequence KV.

---

### 6. Scheduler: when allocation and paging happen

The `Scheduler` orchestrates when we:

- **Allocate** blocks for new sequences (prefill).
- **Append** blocks as sequences grow (decode).
- **Deallocate** or **preempt** sequences to free KV pages.

High-level:

- **Prefill phase**:

  ```python
  while self.waiting and num_seqs < self.max_num_seqs:
      seq = self.waiting[0]
      if num_batched_tokens + len(seq) > self.max_num_batched_tokens \
         or not self.block_manager.can_allocate(seq):
          break
      self.block_manager.allocate(seq)
      ...
  ```

  - New sequences are taken from `waiting`.
  - `BlockManager.allocate` builds or reuses blocks, computes `seq.block_table`, `seq.num_cached_tokens`.
  - The resulting sequences and block tables are later consumed by `ModelRunner.prepare_prefill` and ultimately attention.

- **Decode phase**:

  ```python
  while self.running and num_seqs < self.max_num_seqs:
      seq = self.running.popleft()
      while not self.block_manager.can_append(seq):
          ...
          self.preempt(...)  # deallocates blocks and moves seq to waiting
      else:
          self.block_manager.may_append(seq)
          scheduled_seqs.append(seq)
  ```

  - For each active sequence, it ensures there is capacity to append (if necessary) by possibly preempting others.
  - `may_append` updates the `block_table` to include new blocks or finalize full blocks.

- **Postprocess**:

  ```python
  for seq, token_id in zip(seqs, token_ids):
      seq.append_token(token_id)
      if finished:
          self.block_manager.deallocate(seq)
  ```

  - When sequences finish, their KV pages are deallocated.

Thus, `Scheduler` + `BlockManager` control **when and how** sequences claim and release pages in the global KV cache, while `ModelRunner` + `Attention` control **how those pages are actually populated and read** during attention.

---

### 7. End-to-end paged attention lifecycle

Putting it all together for a typical request:

1. **New request added** via `LLMEngine.add_request`:
   - Creates a `Sequence` for each prompt.
2. **Prefill scheduling** (`Scheduler.schedule`):
   - `BlockManager.allocate` creates/looks up blocks and fills `seq.block_table`.
3. **Model prefill run** (`ModelRunner.run` with `is_prefill=True`):
   - `prepare_prefill` builds `input_ids`, `positions`, `cu_seqlens_q/k`, `slot_mapping`, `block_tables`.
   - `set_context(..., is_prefill=True, ...)`.
   - The model forward calls each `Attention`:
     - `store_kvcache` writes new tokens’ KV into global cache using `slot_mapping`.
     - `flash_attn_varlen_func` consumes `q` and `k_cache` / `v_cache` using `block_tables` and `cu_seqlens_*`.
4. **Decode loop**:
   - `Scheduler.schedule` (decode path) calls `BlockManager.may_append` as sequences grow.
   - `prepare_decode` builds `slot_mapping`, `context_lens`, and `block_tables`.
   - `set_context(..., is_prefill=False, ...)`.
   - In `Attention.forward` (decode branch):
     - `store_kvcache` writes the new step’s KV into its precise slot.
     - `flash_attn_with_kvcache` reads KV via `block_table` and `cache_seqlens`, computing attention over paged storage.
5. **Completion / preemption**:
   - `Scheduler.postprocess` deallocates blocks (via `BlockManager.deallocate`) when sequences finish.
   - Preemption can also deallocate pages early to free memory.

In summary, **nanovllm’s paged attention** is implemented by:

- A **global, block-structured KV cache tensor** on GPU.
- A **logical block table per sequence** managed with prefix-sharing and reference counting.
- A **slot-based mapping** from tokens to cache slots, used by a Triton kernel to write KV.
- FlashAttention kernels that operate directly on this paged layout via `block_table`, `cu_seqlens_*`, and `cache_seqlens`, avoiding dense per-sequence KV materialization.

