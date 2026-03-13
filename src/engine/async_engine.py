from src.config.vllm import VllmConfig
from src.executor import Executor

class AsyncEngine:
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
    ):
        pass

       
