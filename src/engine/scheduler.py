from src.config.vllm import VllmConfig

class Scheduler:
    def __init__(self, vllm_config: VllmConfig):
        self.max_num_seqs = vllm_config.max_num_seqs
