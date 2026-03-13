import logging

from src.config.vllm import VllmConfig
from src.executor.executor_base import BaseExecutor

logger = logging.getLogger(__name__)


class AsyncEngine:
    """
    Top-level engine orchestrator.

    Mirrors vLLM v1's EngineCore initialisation sequence:

      1. Bring up the Executor (model load happens inside _init_executor).
      2. Profile GPU memory and compute how many KV-cache blocks fit.
      3. Allocate and bind KV-cache tensors to the model.

    The scheduler and request queue will be added in subsequent steps.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[BaseExecutor],
    ) -> None:
        self.vllm_config = vllm_config
        logger.info("Initializing AsyncEngine with config: %s", self.vllm_config)

        # Step 1 – bring up executor (loads model weights)
        self.model_executor: BaseExecutor = executor_class(self.vllm_config)

        # Step 2 & 3 – profile memory, size the KV cache, allocate buffers
        self._initialize_kv_caches()

    # ------------------------------------------------------------------
    # KV-cache initialisation (mirrors vLLM v1 EngineCore._initialize_kv_caches)
    # ------------------------------------------------------------------

    def _initialize_kv_caches(self) -> None:
        """
        Orchestrate KV-cache sizing and allocation.

        Follows vLLM v1's three-step pattern:
          1. ``determine_available_memory``  – profiling forward pass; returns
                                              bytes free after weights + activations.
          2. ``compute_num_gpu_blocks``      – convert bytes → (num_blocks, max_seq_len)
                                              using the model's layer/head config.
          3. ``initialize_kv_cache``         – allocate tensors and bind to model.

        ``num_gpu_blocks`` is stored on the engine so the scheduler can use it
        to bound the number of concurrent token slots.
        """
        logger.info("Profiling GPU memory for KV cache sizing...")
        available_bytes = self.model_executor.determine_available_memory()

        num_gpu_blocks, max_seq_length = self.model_executor.compute_num_gpu_blocks(
            available_bytes
        )
        logger.info(
            "KV cache: %d blocks (max_seq_length=%d per sequence)",
            num_gpu_blocks,
            max_seq_length,
        )

        # Expose for the scheduler
        self.num_gpu_blocks: int = num_gpu_blocks
        self.max_seq_length: int = max_seq_length

        self.model_executor.initialize_kv_cache(num_gpu_blocks, max_seq_length)
        logger.info("KV cache initialisation complete.")
