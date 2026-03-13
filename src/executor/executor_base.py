from abc import ABC, abstractmethod
from typing import Tuple

from src.config.vllm import VllmConfig


class BaseExecutor(ABC):
    """
    Abstract base for all executors.

    An Executor is the hardware-topology layer that sits between the Engine
    and the ModelRunner(s).  It owns the KV-cache lifecycle:

      1. determine_available_memory()  – profile peak activation memory and
                                         return bytes free for KV cache.
      2. compute_num_gpu_blocks()      – translate available bytes into a
                                         (num_blocks, max_seq_length) pair.
      3. initialize_kv_cache()         – allocate and bind KV-cache tensors
                                         to the model.

    This mirrors vLLM v1's AbstractExecutor interface but is stripped to the
    minimum needed for single-process execution.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self._init_executor()

    # ------------------------------------------------------------------
    # Template method: subclasses bring up their worker/model-runner here
    # ------------------------------------------------------------------

    @abstractmethod
    def _init_executor(self) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # KV-cache lifecycle (mirrors vLLM v1 executor abstract interface)
    # ------------------------------------------------------------------

    @abstractmethod
    def determine_available_memory(self) -> int:
        """
        Run a profiling forward pass and return the number of bytes on the
        device that are available for KV-cache allocation.

        Called once by the Engine during startup, before any real requests
        are processed.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_num_gpu_blocks(self, available_bytes: int) -> Tuple[int, int]:
        """
        Given the bytes available for KV cache, return
        ``(num_gpu_blocks, max_seq_length)`` using the model's layer/head
        configuration.

        ``max_seq_length`` is derived as ``num_gpu_blocks * BLOCK_SIZE`` and
        is capped at the model's own maximum context window.

        Args:
            available_bytes: Bytes available for KV-cache tensors, as
                             returned by ``determine_available_memory()``.

        Returns:
            num_gpu_blocks: Number of KV-cache blocks in the pool.
            max_seq_length: Maximum token length the KV cache can hold per
                            sequence (``num_gpu_blocks * BLOCK_SIZE``).
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_kv_cache(self, num_gpu_blocks: int, max_seq_length: int) -> None:
        """
        Allocate KV-cache tensors and bind them to the model.

        For litgpt this calls ``GPT.set_kv_cache(batch_size, max_seq_length)``,
        allocating static ``(batch_size, n_query_groups, max_seq_length,
        head_size)`` buffers per attention layer.

        Args:
            num_gpu_blocks: Pool size in blocks (stored on config for the
                            scheduler).
            max_seq_length: Maximum sequence length the cache can hold
                            (``num_gpu_blocks * BLOCK_SIZE``).
        """
        raise NotImplementedError
