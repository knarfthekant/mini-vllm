from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - import guard exercised at runtime
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = exc
else:
    _TRITON_IMPORT_ERROR = None


def require_triton_paged_attention() -> None:
    if triton is None:
        detail = f": {_TRITON_IMPORT_ERROR}" if _TRITON_IMPORT_ERROR is not None else ""
        raise RuntimeError(
            "Paged attention requires Triton to be installed and importable" + detail
        )


if triton is not None:

    @triton.jit
    def _cdiv(x, y):
        return (x + y - 1) // y


    @triton.jit
    def _find_seq_idx(
        query_start_loc_ptr,
        target_idx,
        num_seqs,
        BLOCK_Q: tl.constexpr,
        use_q_block_mode: tl.constexpr,
    ):
        left: tl.int32 = 0
        right = num_seqs
        while left < right:
            mid = (left + right) // 2
            val = tl.load(query_start_loc_ptr + mid)
            mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

            if mid_val <= target_idx:
                left = mid + 1
            else:
                right = mid

        return left - 1


    @triton.jit
    def _paged_attention_kernel(
        output_ptr,  # [num_query_tokens, num_query_heads, head_size]
        query_ptr,  # [num_query_tokens, num_query_heads, head_size]
        key_cache_ptr,  # [num_blocks, block_size, num_kv_heads, head_size]
        value_cache_ptr,  # [num_blocks, block_size, num_kv_heads, head_size]
        block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
        seq_lens_ptr,  # [num_seqs]
        scale,  # float32
        num_query_heads: tl.constexpr,
        num_queries_per_kv: tl.constexpr,
        block_table_stride: tl.int64,
        query_stride_0: tl.int64,
        query_stride_1: tl.int64,
        output_stride_0: tl.int64,
        output_stride_1: tl.int64,
        BLOCK_SIZE: tl.constexpr,
        TILE_SIZE: tl.constexpr,
        HEAD_SIZE: tl.constexpr,
        HEAD_SIZE_PADDED: tl.constexpr,
        stride_k_cache_0: tl.int64,
        stride_k_cache_1: tl.int64,
        stride_k_cache_2: tl.int64,
        stride_k_cache_3: tl.int64,
        stride_v_cache_0: tl.int64,
        stride_v_cache_1: tl.int64,
        stride_v_cache_2: tl.int64,
        stride_v_cache_3: tl.int64,
        query_start_loc_ptr,  # [num_seqs + 1]
        BLOCK_Q: tl.constexpr,
        num_seqs: tl.int32,
        BLOCK_M: tl.constexpr,
    ):
        q_block_global_idx = tl.program_id(0)
        kv_head_idx = tl.program_id(1)

        seq_idx = _find_seq_idx(
            query_start_loc_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
        )

        q_block_start_idx = tl.load(query_start_loc_ptr + seq_idx) // BLOCK_Q + seq_idx
        q_block_local_idx = q_block_global_idx - q_block_start_idx

        cur_query_start = tl.load(query_start_loc_ptr + seq_idx)
        cur_query_stop = tl.load(query_start_loc_ptr + seq_idx + 1)
        cur_query_len = cur_query_stop - cur_query_start

        if q_block_local_idx * BLOCK_Q >= cur_query_len:
            return

        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_SIZE_PADDED)
        offs_t = tl.arange(0, TILE_SIZE)

        query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
        query_offset_0 = cur_query_start + query_pos
        query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
        query_offset = (
            query_offset_0[:, None] * query_stride_0
            + query_offset_1[:, None] * query_stride_1
            + offs_d[None, :]
        )

        dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
        query_mask_0 = tl.where(query_pos < cur_query_len, 1, 0).to(tl.int1)
        query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

        q = tl.load(
            query_ptr + query_offset,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            other=0.0,
        )

        block_table_offset = seq_idx * block_table_stride
        seq_len = tl.load(seq_lens_ptr + seq_idx)
        context_len = seq_len - cur_query_len

        max_seq_prefix_len = (
            context_len
            + q_block_local_idx * BLOCK_Q
            + (BLOCK_M - 1) // num_queries_per_kv
            + 1
        )
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
        num_tiles = _cdiv(max_seq_prefix_len, TILE_SIZE)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

        for tile_idx in range(0, num_tiles):
            seq_offset = tile_idx * TILE_SIZE + offs_t
            tile_mask = seq_offset < max_seq_prefix_len

            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)

            v_offset = (
                physical_block_idx[:, None] * stride_v_cache_0
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
                + kv_head_idx * stride_v_cache_2
                + offs_d[None, :] * stride_v_cache_3
            )
            k_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
                + kv_head_idx * stride_k_cache_2
                + offs_d[:, None] * stride_k_cache_3
            )

            k = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask[:, None] & tile_mask[None, :],
                other=0.0,
            )
            v = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :] & tile_mask[:, None],
                other=0.0,
            )

            query_abs_pos = context_len + query_pos[:, None]
            causal_mask = seq_offset[None, :] <= query_abs_pos

            scores = scale * tl.dot(q, k)
            scores = tl.where(
                query_mask_1[:, None] & query_mask_0[:, None] & causal_mask,
                scores,
                float("-inf"),
            )

            m_j = tl.maximum(m_i, tl.max(scores, axis=1))
            m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
            p = tl.exp(scores - m_j[:, None])
            l_j = tl.sum(p, axis=1)
            alpha = tl.exp(m_i - m_j)

            acc = acc * alpha[:, None]
            l_i = l_i * alpha + l_j
            m_i = m_j
            acc += tl.dot(p.to(v.dtype), v)

        acc = acc / l_i[:, None]

        output_offset = (
            query_offset_0[:, None] * output_stride_0
            + query_offset_1[:, None] * output_stride_1
            + offs_d[None, :]
        )

        tl.store(
            output_ptr + output_offset,
            acc,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    softmax_scale: float,
) -> torch.Tensor:
    require_triton_paged_attention()

    del max_query_len, max_seq_len

    if query.ndim != 3:
        raise ValueError(f"query must have shape [tokens, heads, dim], got {tuple(query.shape)}")
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("Paged KV cache tensors must be rank-4 block-major tensors")

    num_query_tokens, num_query_heads, head_size = query.shape
    num_kv_heads = key_cache.shape[2]
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads for paged attention")

    query = query.contiguous()
    key_cache = key_cache.contiguous()
    value_cache = value_cache.contiguous()
    block_tables = block_tables.contiguous()
    query_start_loc = query_start_loc.contiguous()
    seq_lens = seq_lens.contiguous()

    block_size = key_cache.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads
    block_m = 16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    block_q = block_m // num_queries_per_kv
    total_num_q_blocks = num_query_tokens // block_q + seq_lens.numel()
    tile_size = 32 if query.element_size() >= 2 else 64

    output = torch.empty_like(query)
    _paged_attention_kernel[(total_num_q_blocks, num_kv_heads)](
        output_ptr=output,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_tables_ptr=block_tables,
        seq_lens_ptr=seq_lens,
        scale=softmax_scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_tables.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        stride_k_cache_0=key_cache.stride(0),
        stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2),
        stride_k_cache_3=key_cache.stride(3),
        stride_v_cache_0=value_cache.stride(0),
        stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2),
        stride_v_cache_3=value_cache.stride(3),
        query_start_loc_ptr=query_start_loc,
        BLOCK_Q=block_q,
        num_seqs=seq_lens.numel(),
        BLOCK_M=block_m,
    )
    return output
