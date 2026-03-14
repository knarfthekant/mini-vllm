# KV Cache Allocation: Static vs Paged Attention

This document compares how KV-cache memory is allocated and sized under **static (dense)** allocation (current implementation) vs **paged attention** (planned). It defines the main variables and walks through how allocation is determined in each strategy.

---

## 1. Variables

| Variable | Meaning |
|----------|--------|
| **`max_num_seqs`** | Maximum number of sequences that can run concurrently (batch size). Config: `VllmConfig.max_num_seqs`. |
| **`max_seq_length`** | Maximum number of tokens (context + generated) that a single sequence can use. Determined at init from available VRAM and config. |
| **`num_gpu_blocks`** | In **static**: number of blocks per sequence (= `max_seq_length / BLOCK_SIZE`). In **paged**: total number of blocks in the shared pool. |
| **`BLOCK_SIZE`** | Tokens per block (e.g. 16). Used to align lengths and, in paged attention, as the unit of allocation. |
| **`available_bytes`** | GPU memory (bytes) available for KV cache after model weights (and optional profiling) are accounted for. |
| **`bytes_for_kv`** | Subset of `available_bytes` actually used for KV (e.g. `available_bytes * kv_cache_fraction`, with fraction often 0.9). |
| **`max_model_len`** | Optional config cap on per-sequence length; limits `max_seq_length` regardless of VRAM. |

Model-dependent constants used in sizing (same in both strategies):

- `n_layer`, `n_query_groups`, `k_head_dim`, `dtype_bytes`, `cfg.block_size` (model context window).

---

## 2. Side-by-Side Comparison

| Aspect | Static (dense) allocation | Paged attention |
|--------|---------------------------|------------------|
| **Layout** | One contiguous KV region **per sequence**. Shape: `batch_size × max_seq_length` (plus layer/head dims). | One **shared pool** of blocks. Each sequence has a **block table** mapping logical positions → physical block IDs. |
| **Allocation unit** | Per-sequence slot of fixed length `max_seq_length`. | Block of `BLOCK_SIZE` tokens. Blocks are allocated/freed per sequence from the pool. |
| **Meaning of `num_gpu_blocks`** | `max_seq_length / BLOCK_SIZE` — blocks **per sequence** (same for every sequence). Not a pool size. | **Total** number of blocks in the GPU pool. Shared by all sequences. |
| **Meaning of `max_num_seqs`** | Number of pre-allocated sequence “slots”; directly sets batch size and multiplies into total KV size. | Upper bound on **concurrent** sequences; does **not** multiply into per-block size. Only constrains scheduler. |
| **Meaning of `max_seq_length`** | Fixed length of each sequence’s cache; derived from VRAM and config. | Config/model cap (e.g. context length). Pool must have enough blocks so that up to `max_num_seqs` sequences can each use up to `max_seq_length` tokens. |
| **Total KV memory** | `bytes_per_position × max_num_seqs × max_seq_length`. | `bytes_per_block × num_gpu_blocks`, with `num_gpu_blocks` = total blocks in pool. |
| **Fragmentation** | None; each sequence has a dedicated row. | Possible; short sequences free blocks that can be reused. |
| **Flexibility** | Adding a sequence requires a free “slot”; length is fixed at `max_seq_length` even if sequence is short. | Any mix of sequences up to pool capacity: `sum(blocks per seq) ≤ num_gpu_blocks`, each seq ≤ `max_seq_length`. |

---

## 3. How Allocation Is Determined (Static / Dense)

Current flow (e.g. in `ModelRunner.plan_kv_cache` with the standard backend):

1. **Total KV bytes (fixed layout)**  
   For K and V, one position per layer, per sequence, per query group, per head dimension:
   \[
   \text{total\_bytes} = 2 \times n\_layer \times \textit{max\_num\_seqs} \times n\_query\_groups \times k\_head\_dim \times \textit{max\_seq\_length} \times dtype\_bytes.
   \]

2. **Bytes per “token” (one position in each of the `max_num_seqs` sequences)**  
   “Token” here means one position along the sequence dimension **for every** concurrent sequence:
   \[
   bytes\_per\_token = 2 \times n\_layer \times \textit{max\_num\_seqs} \times n\_query\_groups \times k\_head\_dim \times dtype\_bytes.
   \]
   So `max_num_seqs` is multiplied in because we reserve one position per sequence for each logical “token” step.

3. **Solve for `max_seq_length`**  
   We set:
   \[
   \textit{max\_seq\_length} = \lfloor bytes\_for\_kv \;/\; bytes\_per\_token \rfloor.
   \]
   So:
   \[
   \textit{max\_seq\_length} = \frac{bytes\_for\_kv}{bytes\_per\_position \times \textit{max\_num\_seqs}},
   \]
   i.e. we split the total capacity evenly across `max_num_seqs` sequences.

4. **Caps and alignment**  
   - Cap `max_seq_length` by `max_model_len` (if set) and by the model’s context size `cfg.block_size`.  
   - Round down to a multiple of `BLOCK_SIZE`, with a minimum of `BLOCK_SIZE`.

5. **Blocks (static meaning)**  
   \[
   num\_gpu\_blocks = \textit{max\_seq\_length} \;/\; BLOCK\_SIZE.
   \]
   This is “max sequence length in block units,” not a pool size.

6. **Allocation**  
   Call something like `set_kv_cache(batch_size=max_num_seqs, max_seq_length=max_seq_length)` so that each of the `max_num_seqs` sequences gets a dedicated cache of length `max_seq_length`.

**Summary (static):**  
Given `available_bytes` and `max_num_seqs`, we compute `max_seq_length` so that total KV memory fits. Each sequence gets exactly that length. `num_gpu_blocks` is derived from `max_seq_length` for block-alignment and future paged compatibility; it does not represent a shared block pool.

---

## 4. How Allocation Would Be Determined (Paged)

Conceptual flow for a paged implementation:

1. **Bytes per block**  
   One block holds `BLOCK_SIZE` tokens; no batch dimension in the block itself:
   \[
   bytes\_per\_block = 2 \times n\_layer \times n\_query\_groups \times k\_head\_dim \times BLOCK\_SIZE \times dtype\_bytes.
   \]
   Notice: **no `max_num_seqs`** — the pool is shared.

2. **Total blocks in the pool**  
   \[
   num\_gpu\_blocks = \lfloor bytes\_for\_kv \;/\; bytes\_per\_block \rfloor.
   \]
   This is the **total** number of blocks available to all sequences.

3. **`max_seq_length`**  
   Set by config and model, e.g.:
   \[
   \textit{max\_seq\_length} = \min(\textit{max\_model\_len}, \; cfg.block\_size),
   \]
   aligned to `BLOCK_SIZE`. Not derived from dividing pool by `max_num_seqs`; it’s a per-sequence limit.

4. **Sizing check (optional)**  
   Ensure the pool can support the worst case where every concurrent sequence uses the full length:
   \[
   num\_gpu\_blocks \geq \textit{max\_num\_seqs} \times \lceil \textit{max\_seq\_length} \;/\; BLOCK\_SIZE \rceil.
   \]
   If not, either reduce `max_num_seqs`, reduce `max_seq_length`, or require more VRAM.

5. **Runtime allocation**  
   - Scheduler assigns blocks from the pool to sequences via block tables.  
   - A sequence of length `L` uses \(\lceil L / BLOCK\_SIZE \rceil\) blocks.  
   - When a request finishes, its blocks are returned to the pool.

**Summary (paged):**  
`num_gpu_blocks` is the size of the shared block pool. `max_num_seqs` and `max_seq_length` are limits on concurrency and per-sequence length; they do not multiply into the per-block size. Allocation is “how many blocks does this sequence need?” and “do we have that many free blocks?”.

---

## 5. Formula Summary

| Quantity | Static (dense) | Paged |
|----------|----------------|-------|
| Total KV bytes | \(bytes\_per\_position \times \textit{max\_num\_seqs} \times \textit{max\_seq\_length}\) | \(bytes\_per\_block \times num\_gpu\_blocks\) |
| Per-token/block unit | \(bytes\_per\_token = bytes\_per\_position \times \textit{max\_num\_seqs}\) | \(bytes\_per\_block = bytes\_per\_position \times BLOCK\_SIZE\) |
| Solve for | \(\textit{max\_seq\_length} = \dfrac{bytes\_for\_kv}{bytes\_per\_token}\) | \(num\_gpu\_blocks = \dfrac{bytes\_for\_kv}{bytes\_per\_block}\) |
| Role of `max_num_seqs` | In denominator when solving for `max_seq_length`; in `set_kv_cache(batch_size=...)`. | Only in sanity check and scheduler; not in pool sizing. |
| Role of `max_seq_length` | Output of sizing; length of each sequence’s cache. | Input (config/model); per-sequence limit; used in block count check. |

---

## 6. References in This Codebase

- **Config:** `src/config/vllm.py` — `max_num_seqs`, `max_model_len`, `BLOCK_SIZE`.
- **Static sizing:** `src/worker/model_runner.py` and `src/worker/attention_backends.py`.
- **Orchestration:** `src/engine/async_engine.py` — `_initialize_kv_caches()`; exposes `num_gpu_blocks` and `max_seq_length` for the scheduler.
