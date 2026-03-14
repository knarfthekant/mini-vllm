from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from src.config.vllm import BLOCK_SIZE
from src.worker.model_input import SchedulerOutput

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
    attn_metadata: object | None = None


@dataclass(frozen=True)
class DenseKVCacheState:
    batch_size: int
    max_seq_length: int


@dataclass
class PagedLayerKVCache:
    key_blocks: torch.Tensor
    value_blocks: torch.Tensor


@dataclass
class PagedKVCacheState:
    num_gpu_blocks: int
    block_size: int
    max_seq_length: int
    layers: list[PagedLayerKVCache]


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

    def prepare_model_inputs(
        self,
        runner: "ModelRunner",
        scheduler_output: SchedulerOutput,
    ) -> ModelExecutionInputs:
        seqs = scheduler_output.input_ids
        pos_seqs = scheduler_output.positions

        if not seqs:
            raise ValueError("scheduler_output.input_ids must not be empty")
        if len(seqs) != len(pos_seqs):
            raise ValueError("input_ids and positions must have the same batch size")

        max_len = max(len(seq) for seq in seqs)
        padded_ids = [seq + [0] * (max_len - len(seq)) for seq in seqs]
        padded_pos = [pos + [0] * (max_len - len(pos)) for pos in pos_seqs]

        return ModelExecutionInputs(
            input_ids=torch.tensor(padded_ids, dtype=torch.long, device=runner.device),
            positions=torch.tensor(padded_pos, dtype=torch.long, device=runner.device),
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

        max_seq_length = bytes_for_kv // bytes_per_token
        max_seq_length = _align_max_seq_length(runner, max_seq_length)
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


class PagedCacheManager(BaseCacheManager):
    name = "paged"

    def build_kv_cache_plan(
        self,
        runner: "ModelRunner",
        available_bytes: int,
    ) -> KVCachePlan:
        bytes_for_kv = int(available_bytes * self.kv_cache_fraction)
        bytes_per_block = _kv_bytes_per_token_position(runner) * BLOCK_SIZE
        if bytes_per_block == 0:
            raise RuntimeError("bytes_per_block is zero; model config may be invalid")

        num_gpu_blocks = max(1, bytes_for_kv // bytes_per_block)
        max_seq_length = _align_max_seq_length(
            runner,
            _configured_max_seq_length(runner),
        )

        blocks_per_seq = math.ceil(max_seq_length / BLOCK_SIZE)
        required_blocks = blocks_per_seq * runner.vllm_config.max_num_seqs
        if num_gpu_blocks < required_blocks:
            if num_gpu_blocks < runner.vllm_config.max_num_seqs:
                raise RuntimeError(
                    "Paged attention KV pool is too small for the configured "
                    f"max_num_seqs={runner.vllm_config.max_num_seqs}. "
                    "Need at least one block per active sequence."
                )
            supported_blocks_per_seq = max(1, num_gpu_blocks // runner.vllm_config.max_num_seqs)
            max_seq_length = supported_blocks_per_seq * BLOCK_SIZE
            logger.warning(
                "Paged attention pool cannot support %d sequences at %d tokens; "
                "reducing max_seq_length to %d",
                runner.vllm_config.max_num_seqs,
                blocks_per_seq * BLOCK_SIZE,
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
        cfg = runner.model.config
        if cfg.n_query_groups is None:
            raise RuntimeError("Config.n_query_groups must be set after model init")
        head_dim = _kv_key_head_dim(runner)
        value_dim = cfg.head_size
        if value_dim is None:
            raise RuntimeError("Config.head_size must be set after model init")

        runner.model.max_seq_length = plan.max_seq_length
        dtype = runner.model.transformer.wte.weight.dtype  # type: ignore[attr-defined]
        layers: list[PagedLayerKVCache] = []
        for _ in range(cfg.n_layer):
            layers.append(
                PagedLayerKVCache(
                    key_blocks=torch.empty(
                        (
                            plan.num_gpu_blocks,
                            cfg.n_query_groups,
                            BLOCK_SIZE,
                            head_dim,
                        ),
                        dtype=dtype, # type: ignore[arg-type]
                        device=runner.device, 
                    ),
                    value_blocks=torch.empty(
                        (
                            plan.num_gpu_blocks,
                            cfg.n_query_groups,
                            BLOCK_SIZE,
                            value_dim,
                        ),
                        dtype=dtype, # type: ignore[arg-type]
                        device=runner.device,
                    ),
                )
            )

        logger.info(
            "Paged KV cache allocated: %d shared blocks, block_size=%d, max_seq_length=%d",
            plan.num_gpu_blocks,
            BLOCK_SIZE,
            plan.max_seq_length,
        )
        return PagedKVCacheState(
            num_gpu_blocks=plan.num_gpu_blocks,
            block_size=BLOCK_SIZE,
            max_seq_length=plan.max_seq_length,
            layers=layers,
        )

    def forward(
        self,
        runner: "ModelRunner",
        model_inputs: ModelExecutionInputs,
    ) -> torch.Tensor:
        del runner, model_inputs
        raise NotImplementedError(
            "Paged attention allocation is configured, but paged-attention execution "
            "is not wired into the litgpt model yet."
        )


def build_cache_manager(name: str) -> BaseCacheManager:
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


def _align_max_seq_length(runner: "ModelRunner", max_seq_length: int) -> int:
    max_seq_length = min(max_seq_length, _configured_max_seq_length(runner))
    return max(BLOCK_SIZE, (max_seq_length // BLOCK_SIZE) * BLOCK_SIZE)


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
