from typing import Tuple

from src.executor.executor_base import BaseExecutor
from src.worker.model_runner import GPUModelRunner


class UniProcExecutor(BaseExecutor):
    """
    Single-process executor for single-GPU inference.

    Owns one ``GPUModelRunner`` and delegates all model lifecycle calls
    directly to it (no inter-process communication).  This is the simplest
    possible executor and serves as the reference implementation.
    """

    def _init_executor(self) -> None:
        self.model_runner = GPUModelRunner(self.vllm_config)
        self.model_runner.load_model()

    # ------------------------------------------------------------------
    # KV-cache lifecycle  (delegates straight to GPUModelRunner)
    # ------------------------------------------------------------------

    def determine_available_memory(self) -> int:
        """Profile the model and return bytes available for KV cache."""
        return self.model_runner.determine_available_memory()

    def compute_num_gpu_blocks(self, available_bytes: int) -> Tuple[int, int]:
        """
        Translate available bytes into ``(num_gpu_blocks, max_seq_length)``.
        """
        return self.model_runner.compute_num_gpu_blocks(available_bytes)

    def initialize_kv_cache(self, num_gpu_blocks: int, max_seq_length: int) -> None:
        """Allocate and bind KV-cache buffers on the model."""
        self.model_runner.initialize_kv_cache(num_gpu_blocks, max_seq_length)
