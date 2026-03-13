from abc import ABC, abstractmethod
from src.config.vllm import VllmConfig

class BaseExecutor(ABC):
    # For now, only support uniproc executor. This class exists for
    # extensibility to support multi-process executor in the future.
    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        self.vllm_config = vllm_config
        
        self._init_executor()
        
    # Template Method Pattern
    @abstractmethod
    def _init_executor(self) -> None:
        raise NotImplementedError
    
        
        
    