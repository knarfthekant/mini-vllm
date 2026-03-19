"""
Configuration for mini-vllm.
"""
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

if TYPE_CHECKING:
    from litgpt.config import Config

# Number of tokens per KV-cache block for the *dense* backend. Matches nano-vllm;
# used by Request's block helpers and the dense KV-cache size computation.
#
# Note: the paged KV backend may use a different block size depending on the
# attention kernel's requirements.
BLOCK_SIZE: int = 16

# Number of tokens per block for the *paged* KV backend.
#
# Keep this separate from the dense backend's BLOCK_SIZE so we can tune paged
# kernel behavior independently without changing dense-cache behavior.
PAGED_BLOCK_SIZE: int = 16
BaseCacheManager = Literal["standard", "paged"]


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"))
class VllmConfig:
    """Configuration for mini-vllm. Minimal config for single GPU inference."""

    checkpoint_dir: Path
    """Path to the litgpt model checkpoint directory
    (containing lit_model.pth and model_config.yaml)."""

    max_num_batched_tokens: int = 16384
    """Total number of tokens (summed across all sequences) that can be batched together."""

    max_num_seqs: int = 256
    """Maximum number of sequences that can run concurrently."""

    max_model_len: Optional[int] = 8192
    """Maximum sequence length (context + generated) per request.
    Caps KV cache allocation. None = use full computed length from VRAM (can be
    slow and risk OOM on prefill). Typical: 4096, 8192, 16384."""

    kv_cache_manager: BaseCacheManager = "standard"
    """KV-cache manager used by the ModelRunner.
    ``standard`` pre-allocates dense per-sequence KV tensors.
    ``paged`` allocates a shared block pool for paged KV cache."""

    gpu_memory_utilization: float = 0.9
    """Fraction of total GPU memory reserved for the engine (weights + KV cache).
    Values in (0, 1]. Lower values leave headroom for other GPU workloads."""

    @cached_property
    def model_config(self) -> "Config":
        """LitGPT model config loaded from checkpoint_dir/model_config.yaml."""
        from litgpt.config import Config

        return Config.from_file(self.checkpoint_dir / "model_config.yaml")
