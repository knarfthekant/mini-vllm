from pathlib import Path
from typing import Optional

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

# Number of tokens per KV-cache block.  Matches nano-vllm; used by Request's
# block helpers and the KV-cache size computation.
BLOCK_SIZE: int = 16


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"))
class VllmConfig:
    """Configuration for mini-vllm. Minimal config for single GPU inference."""

    checkpoint_dir: Path
    """Path to the litgpt model checkpoint directory
    (containing lit_model.pth and model_config.yaml)."""

    max_num_seqs: int = 256
    """Maximum number of sequences that can run concurrently.
    Determines the batch_size dimension of litgpt's static KV cache."""

    gpu_memory_utilization: float = 0.9
    """Fraction of total GPU memory reserved for the engine (weights + KV cache).
    Values in (0, 1]. Lower values leave headroom for other GPU workloads."""
