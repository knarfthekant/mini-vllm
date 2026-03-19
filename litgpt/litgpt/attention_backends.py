from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

try:
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - exercised by runtime guard
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = exc
else:
    _TRITON_IMPORT_ERROR = None

from litgpt.triton_paged_attention import paged_attention

if TYPE_CHECKING:
    from litgpt.model import CausalSelfAttention


@dataclass
class LayerPagedKVCache:
    key_blocks: torch.Tensor
    value_blocks: torch.Tensor


@dataclass
class PagedAttentionMetadata:
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    max_query_len: int
    seq_lens: torch.Tensor
    max_seq_len: int
    block_tables: torch.Tensor
    slot_mapping: torch.Tensor
    last_token_indices: torch.Tensor


def require_paged_attention_kernels() -> None:
    if triton is None:
        detail = f": {_TRITON_IMPORT_ERROR}" if _TRITON_IMPORT_ERROR is not None else ""
        raise RuntimeError(
            "Paged attention requires Triton to be installed and importable" + detail
        )


class BaseAttentionBackend(ABC):
    uses_paged_cache: bool = False

    @abstractmethod
    def forward(
        self,
        attn_module: "CausalSelfAttention",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
        input_pos: torch.Tensor | None,
        input_pos_maxp1: int | None,
        attn_metadata: PagedAttentionMetadata | None,
    ) -> torch.Tensor:
        raise NotImplementedError


class DenseAttentionBackend(BaseAttentionBackend):
    uses_paged_cache = False

    def forward(
        self,
        attn_module: "CausalSelfAttention",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
        input_pos: torch.Tensor | None,
        input_pos_maxp1: int | None,
        attn_metadata: PagedAttentionMetadata | None,
    ) -> torch.Tensor:
        del attn_metadata

        if input_pos is not None:
            from litgpt.model import KVCache

            if not isinstance(attn_module.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = attn_module.kv_cache(input_pos, k, v)

            if attn_module.apply_sliding_window_attention:
                actual_kv_len = k.size(2)
                if mask is not None and mask.size(-1) != actual_kv_len:
                    mask = mask[..., :actual_kv_len]

            if input_pos_maxp1 is not None:
                k = k[..., :input_pos_maxp1, :]
                v = v[..., :input_pos_maxp1, :]

        if attn_module.config.n_query_groups != attn_module.config.n_head and (
            input_pos is None or attn_module.config.n_query_groups != 1
        ):
            q_per_kv = attn_module.config.n_head // attn_module.config.n_query_groups
            k = k.repeat_interleave(q_per_kv, dim=1)
            v = v.repeat_interleave(q_per_kv, dim=1)

        return attn_module.scaled_dot_product_attention(q, k, v, mask)


class PagedAttentionBackend(BaseAttentionBackend):
    uses_paged_cache = True

    def __init__(self) -> None:
        self.layer_cache: LayerPagedKVCache | None = None

    def bind_kv_cache(self, layer_cache: LayerPagedKVCache) -> None:
        require_paged_attention_kernels()
        self.layer_cache = layer_cache

    def forward(
        self,
        attn_module: "CausalSelfAttention",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
        input_pos: torch.Tensor | None,
        input_pos_maxp1: int | None,
        attn_metadata: PagedAttentionMetadata | None,
    ) -> torch.Tensor:
        del mask, input_pos, input_pos_maxp1

        if self.layer_cache is None:
            raise RuntimeError("Paged attention backend is not bound to a KV cache")
        if attn_metadata is None:
            raise RuntimeError("Paged attention requires per-step attention metadata")

        require_paged_attention_kernels()

        query_lens = _query_lens(attn_metadata.query_start_loc)
        q_flat = _flatten_real_tokens(q, query_lens)
        k_flat = _flatten_real_tokens(k, query_lens)
        v_flat = _flatten_real_tokens(v, query_lens)
        _store_paged_kv(k_flat, v_flat, self.layer_cache, attn_metadata.slot_mapping)

        scale = 1.0 / torch.sqrt(
            torch.tensor(
                attn_module.config.attention_scores_scalar or attn_module.config.head_size,
                dtype=q.dtype,
                device=q.device,
            )
        ).item()

        out_flat = paged_attention(
            query=q_flat,
            key_cache=self.layer_cache.key_blocks,
            value_cache=self.layer_cache.value_blocks,
            block_tables=attn_metadata.block_tables,
            query_start_loc=attn_metadata.query_start_loc,
            seq_lens=attn_metadata.seq_lens,
            max_query_len=attn_metadata.max_query_len,
            max_seq_len=attn_metadata.max_seq_len,
            softmax_scale=scale,
        )
        return _unflatten_real_tokens(out_flat, query_lens, q.size(2))


def _query_lens(query_start_loc: torch.Tensor) -> torch.Tensor:
    if query_start_loc.numel() <= 1:
        return query_start_loc.new_zeros((0,), dtype=torch.long)
    return (query_start_loc[1:] - query_start_loc[:-1]).to(dtype=torch.long)


def _flatten_real_tokens(x: torch.Tensor, query_lens: torch.Tensor) -> torch.Tensor:
    pieces = []
    for batch_idx, query_len in enumerate(query_lens.tolist()):
        if query_len <= 0:
            continue
        pieces.append(x[batch_idx, :, :query_len, :].transpose(0, 1))
    if not pieces:
        return x.new_empty((0, x.size(1), x.size(-1)))
    return torch.cat(pieces, dim=0).contiguous()


def _unflatten_real_tokens(
    x_flat: torch.Tensor,
    query_lens: torch.Tensor,
    padded_len: int,
) -> torch.Tensor:
    batch = query_lens.numel()
    num_heads = x_flat.size(1)
    head_dim = x_flat.size(2)
    output = x_flat.new_zeros((batch, padded_len, num_heads, head_dim))
    start = 0
    for batch_idx, query_len in enumerate(query_lens.tolist()):
        if query_len <= 0:
            continue
        end = start + query_len
        output[batch_idx, :query_len] = x_flat[start:end]
        start = end
    return output


if triton is not None:

    @triton.jit
    def _store_paged_kv_kernel(
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
        offsets = tl.arange(0, D)
        key = tl.load(key_ptr + idx * key_stride + offsets)
        value = tl.load(value_ptr + idx * value_stride + offsets)
        cache_offsets = slot * D + offsets
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


def _store_paged_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_cache: LayerPagedKVCache,
    slot_mapping: torch.Tensor,
) -> None:
    if slot_mapping.numel() == 0:
        return
    require_paged_attention_kernels()

    num_tokens, num_heads, head_dim = key.shape
    flat_dim = num_heads * head_dim

    key = key.contiguous()
    value = value.contiguous()
    k_cache = layer_cache.key_blocks.contiguous().view(-1, flat_dim)
    v_cache = layer_cache.value_blocks.contiguous().view(-1, flat_dim)

    _store_paged_kv_kernel[(num_tokens,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping.contiguous(),
        flat_dim,
    )
