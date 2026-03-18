from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from litgpt.attention_backends import (
    DenseAttentionBackend,
    LayerPagedKVCache,
    PagedAttentionBackend,
    PagedAttentionMetadata,
    require_paged_attention_kernels,
)

from src.config.vllm import BLOCK_SIZE, PAGED_BLOCK_SIZE
from src.engine.allocator import AllocatorEvent, ClearDenseSlot, MoveDenseSlot
from src.worker.interface import SchedulerOutput

if TYPE_CHECKING:
    from src.worker.model_runner import ModelRunner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KVCachePlan:
    num_gpu_blocks: int
    max_seq_length: int
    backend_name: str


@dataclass
class ModelExecutionInputs:
    input_ids: torch.Tensor
    positions: torch.Tensor
    seq_lens: torch.Tensor
    last_token_indices: torch.Tensor
    attn_metadata: PagedAttentionMetadata | None = None


@dataclass(frozen=True)
class DenseKVCacheState:
    batch_size: int
    max_seq_length: int


@dataclass
class PagedKVCacheState:
    num_gpu_blocks: int
    block_size: int
    max_seq_length: int
    layers: list[LayerPagedKVCache]


class BaseCacheManager(ABC):
    name: str
    kv_cache_fraction: float = 0.9

    @abstractmethod
    def build_kv_cache_plan(
        self,
        runner: "ModelRunner",
        available_bytes: int,
    ) -> KVCachePlan:
        raise NotImplementedError

    @abstractmethod
    def initialize_kv_cache(
        self,
        runner: "ModelRunner",
        plan: KVCachePlan,
    ) -> DenseKVCacheState | PagedKVCacheState:
        raise NotImplementedError

    def update_states(
        self,
        runner: "ModelRunner",
        scheduler_output: SchedulerOutput,
    ) -> None:
        del runner, scheduler_output

    def apply_allocator_events(
        self,
        runner: "ModelRunner",
        events: list[AllocatorEvent],
    ) -> None:
        del runner, events

    def prepare_model_inputs(
        self,
        runner: "ModelRunner",
        scheduler_output: SchedulerOutput,
    ) -> ModelExecutionInputs:
        """Pad input_ids and positions to the same length to conform to litgpt model input requirements."""
        seqs = scheduler_output.input_ids
        pos_seqs = scheduler_output.positions

        if not seqs:
            raise ValueError("scheduler_output.input_ids must not be empty")
        if len(seqs) != len(pos_seqs):
            raise ValueError("input_ids and positions must have the same batch size")

        seq_lens = torch.tensor(
            [len(seq) for seq in seqs],
            dtype=torch.long,
            device=runner.device,
        )
        last_token_indices = seq_lens - 1

        max_len = max(len(seq) for seq in seqs)
        # Pad with each sequence's last token and last position so we never write
        # token_id 0 or position 0 into the KV cache (which would overwrite
        # the start of the context and cause drift / trailing padding tokens).
        padded_ids = [seq + [seq[-1]] * (max_len - len(seq)) for seq in seqs]
        padded_pos = [pos + [pos[-1]] * (max_len - len(pos)) for pos in pos_seqs]

        return ModelExecutionInputs(
            input_ids=torch.tensor(padded_ids, dtype=torch.long, device=runner.device),
            positions=torch.tensor(padded_pos, dtype=torch.long, device=runner.device),
            seq_lens=seq_lens,
            last_token_indices=last_token_indices,
        )

    @abstractmethod
    def forward(
        self,
        runner: "ModelRunner",
        model_inputs: ModelExecutionInputs,
    ) -> torch.Tensor:
        raise NotImplementedError


class StandardCacheManager(BaseCacheManager):
    name = "standard"

    def build_kv_cache_plan(
        self,
        runner: "ModelRunner",
        available_bytes: int,
    ) -> KVCachePlan:
        bytes_for_kv = int(available_bytes * self.kv_cache_fraction)
        bytes_per_token = _kv_bytes_per_token_position(runner) * runner.vllm_config.max_num_seqs
        if bytes_per_token == 0:
            raise RuntimeError("bytes_per_token is zero; model config may be invalid")

        # The max sequence length is the COMPUTED max number of tokens capped at the configured max sequence length
        max_seq_length = bytes_for_kv // bytes_per_token
        logger.debug("StandardCacheManager raw computed: %d max_seq_length = %d bytes_for_kv / %d bytes_per_token", max_seq_length, bytes_for_kv, bytes_per_token)

        max_seq_length = _align_to_block_size(min(max_seq_length, _configured_max_seq_length(runner)))
        num_gpu_blocks = max_seq_length // BLOCK_SIZE

        logger.info(
            "Standard attention KV sizing: %d blocks x %d tokens/block = %d max_seq_length",
            num_gpu_blocks,
            BLOCK_SIZE,
            max_seq_length,
        )
        return KVCachePlan(
            num_gpu_blocks=num_gpu_blocks,
            max_seq_length=max_seq_length,
            backend_name=self.name,
        )

    def initialize_kv_cache(
        self,
        runner: "ModelRunner",
        plan: KVCachePlan,
    ) -> DenseKVCacheState:
        _set_dense_attention_backends(runner)
        runner.model.max_seq_length = plan.max_seq_length
        runner.model.set_kv_cache(
            batch_size=runner.vllm_config.max_num_seqs,
            max_seq_length=plan.max_seq_length,
            device=runner.device,
        )
        logger.info(
            "Dense KV cache allocated: batch_size=%d, max_seq_length=%d (%d blocks)",
            runner.vllm_config.max_num_seqs,
            plan.max_seq_length,
            plan.num_gpu_blocks,
        )
        return DenseKVCacheState(
            batch_size=runner.vllm_config.max_num_seqs,
            max_seq_length=plan.max_seq_length,
        )

    def forward(
        self,
        runner: "ModelRunner",
        model_inputs: ModelExecutionInputs,
    ) -> torch.Tensor:
        _ = model_inputs.attn_metadata
        return runner.model(model_inputs.input_ids, input_pos=model_inputs.positions)

    def _move_sequence_cache(
        self,
        runner: "ModelRunner",
        src_slot: int,
        dst_slot: int,
    ) -> None:
        if src_slot == dst_slot:
            return
        for block in runner.model.transformer.h:
            kv = block.attn.kv_cache
            if kv is None:
                raise RuntimeError("KV cache must be initialized before moving slots")
            kv.k[dst_slot].copy_(kv.k[src_slot])
            kv.v[dst_slot].copy_(kv.v[src_slot])
            kv.k[src_slot].zero_()
            kv.v[src_slot].zero_()

    def _clear_sequence_cache(
        self,
        runner: "ModelRunner",
        slot: int,
    ) -> None:
        for block in runner.model.transformer.h:
            kv = block.attn.kv_cache
            if kv is None:
                raise RuntimeError("KV cache must be initialized before clearing slots")
            kv.k[slot].zero_()
            kv.v[slot].zero_()

    def apply_allocator_events(
        self,
        runner: "ModelRunner",
        events: list[AllocatorEvent],
    ) -> None:
        """Apply adjustments to the model's KV cache."""
        for event in events:
            if isinstance(event, MoveDenseSlot):
                self._move_sequence_cache(runner, event.src_slot, event.dst_slot)
            elif isinstance(event, ClearDenseSlot):
                self._clear_sequence_cache(runner, event.slot)


class PagedCacheManager(BaseCacheManager):
    name = "paged"

    def build_kv_cache_plan(
        self,
        runner: "ModelRunner",
        available_bytes: int,
    ) -> KVCachePlan:
        bytes_for_kv = int(available_bytes * self.kv_cache_fraction)
        bytes_per_block = _kv_bytes_per_token_position(runner) * PAGED_BLOCK_SIZE
        if bytes_per_block == 0:
            raise RuntimeError("bytes_per_block is zero; model config may be invalid")

        # Actual available blocks in the pool
        num_gpu_blocks = max(1, bytes_for_kv // bytes_per_block)
        # The max sequence length is the CONFIGURED max sequence length
        max_seq_length = _align_to_paged_block_size(_configured_max_seq_length(runner))

        blocks_per_seq = math.ceil(max_seq_length / PAGED_BLOCK_SIZE)
        required_blocks = blocks_per_seq * runner.vllm_config.max_num_seqs
        if num_gpu_blocks < required_blocks:
            if num_gpu_blocks < runner.vllm_config.max_num_seqs:
                raise RuntimeError(
                    "Paged attention KV pool is too small for the configured "
                    f"max_num_seqs={runner.vllm_config.max_num_seqs}. "
                    "Need at least one block per active sequence."
                )
            supported_blocks_per_seq = max(1, num_gpu_blocks // runner.vllm_config.max_num_seqs)
            max_seq_length = supported_blocks_per_seq * PAGED_BLOCK_SIZE
            logger.warning(
                "Paged attention pool cannot support %d sequences at %d tokens; "
                "reducing max_seq_length to %d",
                runner.vllm_config.max_num_seqs,
                blocks_per_seq * PAGED_BLOCK_SIZE,
                max_seq_length,
            )

        logger.info(
            "Paged attention KV sizing: %d shared blocks, max_seq_length=%d",
            num_gpu_blocks,
            max_seq_length,
        )
        return KVCachePlan(
            num_gpu_blocks=num_gpu_blocks,
            max_seq_length=max_seq_length,
            backend_name=self.name,
        )

    def initialize_kv_cache(
        self,
        runner: "ModelRunner",
        plan: KVCachePlan,
    ) -> PagedKVCacheState:
        require_paged_attention_kernels()
        _validate_paged_model_support(runner)
        cfg = runner.model.config
        if cfg.n_query_groups is None:
            raise RuntimeError("Config.n_query_groups must be set after model init")
        head_dim = cfg.head_size
        value_dim = cfg.head_size
        if head_dim is None or value_dim is None:
            raise RuntimeError("Config.head_size must be set after model init")
        if head_dim != value_dim:
            raise RuntimeError("Paged attention currently requires matching K/V head dimensions")

        runner.model.max_seq_length = plan.max_seq_length
        dtype = runner.model.transformer.wte.weight.dtype  # type: ignore[attr-defined]
        layers: list[LayerPagedKVCache] = []
        for _ in range(cfg.n_layer):
            layers.append(
                LayerPagedKVCache(
                    # [num_gpu_blocks, PAGED_BLOCK_SIZE, n_query_groups, head_dim]
                    key_blocks=torch.empty(
                        (
                            plan.num_gpu_blocks,
                            PAGED_BLOCK_SIZE,
                            cfg.n_query_groups,
                            head_dim,
                        ),
                        dtype=dtype,  # type: ignore[arg-type]
                        device=runner.device,
                    ),
                    value_blocks=torch.empty(
                        (
                            plan.num_gpu_blocks,
                            PAGED_BLOCK_SIZE,
                            cfg.n_query_groups,
                            value_dim,
                        ),
                        dtype=dtype,  # type: ignore[arg-type]
                        device=runner.device,
                    ),
                )
            )

        _bind_paged_attention_backends(runner, layers)

        logger.info(
            "Paged KV cache allocated: %d shared blocks, block_size=%d, max_seq_length=%d",
            plan.num_gpu_blocks,
            PAGED_BLOCK_SIZE,
            plan.max_seq_length,
        )
        return PagedKVCacheState(
            num_gpu_blocks=plan.num_gpu_blocks,
            block_size=PAGED_BLOCK_SIZE,
            max_seq_length=plan.max_seq_length,
            layers=layers,
        )

    def forward(
        self,
        runner: "ModelRunner",
        model_inputs: ModelExecutionInputs,
    ) -> torch.Tensor:
        if not isinstance(model_inputs.attn_metadata, PagedAttentionMetadata):
            raise RuntimeError("Paged attention requires PagedAttentionMetadata")
        return runner.model(
            model_inputs.input_ids,
            input_pos=model_inputs.positions,
            attn_metadata=model_inputs.attn_metadata,
        )

    def prepare_model_inputs(
        self,
        runner: "ModelRunner",
        scheduler_output: SchedulerOutput,
    ) -> ModelExecutionInputs:
        inputs = super().prepare_model_inputs(runner, scheduler_output)
        inputs.attn_metadata = _build_paged_attention_metadata(
            runner,
            scheduler_output,
            inputs.seq_lens,
            inputs.last_token_indices,
        )
        return inputs


def build_cache_manager(name: str) -> BaseCacheManager:
    logger.info("Initializing cache manager: %s", name)
    backends: dict[str, type[BaseCacheManager]] = {
        StandardCacheManager.name: StandardCacheManager,
        PagedCacheManager.name: PagedCacheManager,
    }
    try:
        return backends[name]()
    except KeyError as exc:
        supported = ", ".join(sorted(backends))
        raise ValueError(
            f"Unsupported cache manager {name!r}. Expected one of: {supported}"
        ) from exc


def _configured_max_seq_length(runner: "ModelRunner") -> int:
    configured = runner.vllm_config.max_model_len
    model_limit = runner.model.config.block_size
    if configured is None:
        return model_limit
    return min(configured, model_limit)


def _align_to_block_size(value: int) -> int:
    """Align value down to the nearest BLOCK_SIZE multiple, minimum BLOCK_SIZE."""
    return max(BLOCK_SIZE, (value // BLOCK_SIZE) * BLOCK_SIZE)


def _align_to_paged_block_size(value: int) -> int:
    """Align value down to the nearest PAGED_BLOCK_SIZE multiple, minimum PAGED_BLOCK_SIZE."""
    return max(PAGED_BLOCK_SIZE, (value // PAGED_BLOCK_SIZE) * PAGED_BLOCK_SIZE)


def _kv_bytes_per_token_position(runner: "ModelRunner") -> int:
    cfg = runner.model.config
    head_dim = _kv_key_head_dim(runner)
    dtype_bytes = runner.model.transformer.wte.weight.element_size()  # type: ignore[attr-defined]
    if cfg.head_size is None or cfg.n_query_groups is None:
        raise RuntimeError("Model config head dimensions are not initialized")

    return 2 * cfg.n_layer * cfg.n_query_groups * head_dim * dtype_bytes


def _kv_key_head_dim(runner: "ModelRunner") -> int:
    cfg = runner.model.config
    if cfg.head_size is None or cfg.n_query_groups is None:
        raise RuntimeError("Model config head dimensions are not initialized")

    rope_cache_length = runner.model.rope_cache_length()  # type: ignore[attr-defined]
    rope_n_elem = int(cfg.rope_n_elem)  # type: ignore[attr-defined]
    return rope_cache_length + cfg.head_size - rope_n_elem


def _set_dense_attention_backends(runner: "ModelRunner") -> None:
    for block in runner.model.transformer.h:
        if hasattr(block.attn, "set_attention_backend"):
            block.attn.set_attention_backend(DenseAttentionBackend())


def _bind_paged_attention_backends(
    runner: "ModelRunner",
    layers: list[LayerPagedKVCache],
) -> None:
    for layer_cache, block in zip(layers, runner.model.transformer.h, strict=True):
        if not hasattr(block.attn, "set_attention_backend"):
            raise RuntimeError("Paged attention requires CausalSelfAttention modules")
        backend = PagedAttentionBackend()
        backend.bind_kv_cache(layer_cache)
        block.attn.kv_cache = None
        block.attn.set_attention_backend(backend)


def _validate_paged_model_support(runner: "ModelRunner") -> None:
    cfg = runner.model.config
    if cfg.latent_attention is not None:
        raise RuntimeError("Paged attention is not implemented for latent attention models")
    if cfg.sliding_window_size is not None:
        raise RuntimeError("Paged attention is not implemented for sliding-window attention")
    if cfg.attention_logit_softcapping is not None:
        raise RuntimeError("Paged attention is not implemented with attention_logit_softcapping")


def _build_paged_attention_metadata(
    runner: "ModelRunner",
    scheduler_output: SchedulerOutput,
    seq_lens: torch.Tensor,
    last_token_indices: torch.Tensor,
) -> PagedAttentionMetadata:
    q_lens = [len(seq) for seq in scheduler_output.input_ids]
    if not q_lens or any(length <= 0 for length in q_lens):
        raise ValueError("Paged attention requires non-empty per-request token inputs")
    if seq_lens.numel() != len(q_lens):
        raise ValueError("seq_lens must align with the paged-attention batch")
    if last_token_indices.numel() != len(q_lens):
        raise ValueError("last_token_indices must align with the paged-attention batch")

    query_lens = seq_lens.to(device=runner.device, dtype=torch.int32)
    if query_lens.tolist() != q_lens:
        raise ValueError("seq_lens must match the real per-request query lengths")

    block_tables = scheduler_output.block_tables
    if len(block_tables) != len(q_lens):
        raise ValueError("block_tables must have the same batch size as input_ids")

    slot_mapping_values: list[int] = []
    for positions, block_table in zip(scheduler_output.positions, block_tables, strict=True):
        for pos in positions:
            block_idx = pos // PAGED_BLOCK_SIZE
            if block_idx >= len(block_table):
                raise ValueError(
                    f"Missing physical block for logical block {block_idx} at position {pos}"
                )
            block_id = block_table[block_idx]
            slot_mapping_values.append(block_id * PAGED_BLOCK_SIZE + (pos % PAGED_BLOCK_SIZE))

    max_blocks = max((len(table) for table in block_tables), default=0)
    padded_block_tables = [table + [-1] * (max_blocks - len(table)) for table in block_tables]
    block_tables_tensor = torch.tensor(
        padded_block_tables,
        dtype=torch.int32,
        device=runner.device,
    )
    slot_mapping = torch.tensor(
        slot_mapping_values,
        dtype=torch.int32,
        device=runner.device,
    )

    requests = scheduler_output.requests
    if len(requests) != len(q_lens):
        raise ValueError("SchedulerOutput.requests must align with input_ids for paged attention")

    full_seq_lens = torch.tensor(
        [request.num_tokens for request in requests],
        dtype=torch.int32,
        device=runner.device,
    )
    query_start_loc = torch.empty(
        (query_lens.numel() + 1,),
        dtype=torch.int32,
        device=runner.device,
    )
    query_start_loc[0] = 0
    query_start_loc[1:] = torch.cumsum(query_lens, dim=0)

    return PagedAttentionMetadata(
        num_actual_tokens=int(query_lens.sum().item()),
        query_start_loc=query_start_loc,
        max_query_len=int(query_lens.max().item()),
        seq_lens=full_seq_lens,
        max_seq_len=int(full_seq_lens.max().item()),
        block_tables=block_tables_tensor,
        slot_mapping=slot_mapping,
        last_token_indices=last_token_indices,
    )
