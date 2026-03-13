from .executor_base import BaseExecutor
from src.worker.model_runner import GPUModelRunner


class UniProcExecutor(BaseExecutor):
    """Single-process executor for single GPU inference."""

    def _init_executor(self) -> None:
        self.model_runner = GPUModelRunner(self.vllm_config)
        self.model_runner.load_model()