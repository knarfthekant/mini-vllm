from pathlib import Path
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

# Number of tokens per KV-cache block. 
# Request's block helpers.  Will move into VllmConfig once the block manager
# is wired up.
BLOCK_SIZE: int = 256


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, extra="forbid"))
class VllmConfig:
    """Configuration for mini-vllm. Minimal config for single GPU inference."""
    checkpoint_dir: Path
    """Path to the litgpt model checkpoint directory (containing lit_model.pth and model_config.yaml)."""
