import logging

from src.config.vllm import VllmConfig
from src.worker.model_runner import ModelRunner
from .scheduler import Scheduler

logger = logging.getLogger(__name__)


class AsyncEngine:
    """
    Top-level engine orchestrator.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        self.vllm_config = vllm_config
        logger.info("Initializing AsyncEngine with config: %s", self.vllm_config)

        self.model_runner = ModelRunner(self.vllm_config)
        self.model_runner.load_model()

        # Profile memory, size the KV cache, allocate buffers
        self._initialize_kv_caches()

        # Setup scheduler
        self.scheduler = Scheduler(self.vllm_config)


    def _initialize_kv_caches(self) -> None:
        """
        Initialize the model runner's cache manager and expose its KV limits
        to the scheduler layer.
        """
        logger.info(
            "Initializing %s cache manager...",
            self.model_runner.cache_manager_name,
        )
        plan = self.model_runner.initialize_kv_cache()
        logger.info(
            "KV cache: %d blocks (max_seq_length=%d per sequence)",
            plan.num_gpu_blocks,
            plan.max_seq_length,
        )

        self.num_gpu_blocks = plan.num_gpu_blocks
        self.max_seq_length = plan.max_seq_length

        logger.info("KV cache manager initialisation complete.")
