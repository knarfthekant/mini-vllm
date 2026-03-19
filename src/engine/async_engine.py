from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

from src.config.vllm import VllmConfig
from src.engine.allocator import DenseSlotManager, PagedBlockManager
from src.request import Request
from src.sampling_params import SamplingParams
from src.worker.model_runner import ModelRunner
from litgpt.tokenizer import Tokenizer
from .scheduler import Scheduler, SchedulerPostprocessResult

logger = logging.getLogger(__name__)

class AsyncEngine:
    """
    Top-level engine orchestrator.

    Requests may be added while the engine is already serving. The batch for the
    next model step is rebuilt from the scheduler's latest waiting/running state,
    enabling continuous batching at step granularity.
    """

    def __init__(
        self,
        vllm_config: VllmConfig
    ) -> None:
        logger.info("Initializing AsyncEngine with config: %s", vllm_config)
        self.vllm_config = vllm_config
        self._shutdown = False
        self._completed_requests: deque[Request] = deque()
        self._requests: dict[str, Request] = {}
        self.last_step_stats: dict[str, float] | None = None

        # Configuring model
        self.model_runner = ModelRunner(self.vllm_config)
        self.model_runner.load_model()

        # Configuring tokenizer
        self.tokenizer = Tokenizer(self.vllm_config.checkpoint_dir)
        self.eos_token_id = self.tokenizer.eos_id

        # Profile memory, size the KV cache, allocate buffers
        self._initialize_kv_caches()

        # Setup scheduler-side allocator and scheduler
        allocator = self._build_allocator()
        self.scheduler = Scheduler(
            self.vllm_config,
            allocator=allocator,
            max_seq_length=self.max_seq_length,
            eos_token_id=self.eos_token_id,
        )

    def _initialize_kv_caches(self) -> None:
        """
        Initialize the model runner's cache manager and expose its KV limits
        to the scheduler layer.
        """
        plan = self.model_runner.initialize_kv_cache()
        logger.info(
            "KV cache: %d blocks (max_seq_length=%d per sequence)",
            plan.num_gpu_blocks,
            plan.max_seq_length,
        )

        # Registering KV cache limits with the scheduler
        self.num_gpu_blocks = plan.num_gpu_blocks
        self.max_seq_length = plan.max_seq_length
        self.kv_cache_backend_name = plan.backend_name

        logger.info("KV cache manager initialisation complete.")

    def _build_allocator(self) -> DenseSlotManager | PagedBlockManager:
        if self.kv_cache_backend_name == "paged":
            return PagedBlockManager(self.num_gpu_blocks)
        return DenseSlotManager(max_num_seqs=self.vllm_config.max_num_seqs)

    # Request handling
    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
    ) -> Request:
        if isinstance(prompt, str):
            prompt_token_ids = self.tokenizer.encode(prompt).tolist()
        else:
            prompt_token_ids = prompt

        logger.debug("Adding request with prompt: %s, prompt_token_ids: %s, sampling_params: %s, request_id: %s", prompt, prompt_token_ids, sampling_params, request_id)
        
        request = Request(
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            request_id=request_id,
        )
        self.scheduler.add_request(request)
        self._requests[request.request_id] = request
        return request

    def get_request(self, request_id: str) -> Request | None:
        return self._requests.get(request_id)

    def abort_request(self, request_id: str) -> Request | None:
        result = self.scheduler.abort_request(request_id)
        if result is None:
            return None
        self._apply_postprocess_result(result)
        return result.finished_requests[0]

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_requests()

    def step(self) -> list[Request]:
        self.last_step_stats = None

        step_start = time.perf_counter()
        scheduler_output = self.scheduler.schedule()
        after_schedule = time.perf_counter()
        pending_result = self.scheduler.drain_pending_result()
        prefinished = self._apply_postprocess_result(pending_result)
        after_prefinished = time.perf_counter()

        if not scheduler_output.requests:
            self.last_step_stats = {
                "schedule_s": after_schedule - step_start,
                "prefinished_apply_s": after_prefinished - after_schedule,
                "execute_model_s": 0.0,
                "runner_update_states_s": 0.0,
                "runner_prepare_inputs_s": 0.0,
                "runner_forward_s": 0.0,
                "runner_sample_s": 0.0,
                "scheduler_postprocess_s": 0.0,
                "finished_apply_s": 0.0,
                "step_total_s": after_prefinished - step_start,
            }
            return prefinished

        runner_output = self.model_runner.execute_model(scheduler_output)
        after_execute = time.perf_counter()
        result = self.scheduler.postprocess(scheduler_output, runner_output)
        after_postprocess = time.perf_counter()
        finished_requests = self._apply_postprocess_result(result)
        after_apply = time.perf_counter()
        runner_stats = self.model_runner.last_execute_model_stats or {}
        self.last_step_stats = {
            "schedule_s": after_schedule - step_start,
            "prefinished_apply_s": after_prefinished - after_schedule,
            "execute_model_s": after_execute - after_prefinished,
            "runner_update_states_s": runner_stats.get("update_states_s", 0.0),
            "runner_prepare_inputs_s": runner_stats.get("prepare_inputs_s", 0.0),
            "runner_forward_s": runner_stats.get("forward_s", 0.0),
            "runner_sample_s": runner_stats.get("sample_s", 0.0),
            "scheduler_postprocess_s": after_postprocess - after_execute,
            "finished_apply_s": after_apply - after_postprocess,
            "step_total_s": after_apply - step_start,
        }
        return prefinished + finished_requests

    async def run_until_idle(self) -> list[Request]:
        completed: list[Request] = []
        while self.has_unfinished_requests():
            completed.extend(self.step())
            await asyncio.sleep(0)
        return completed

    async def serve(self, idle_sleep_s: float = 0.01) -> None:
        self._shutdown = False
        while not self._shutdown:
            if self.has_unfinished_requests():
                self.step()
                await asyncio.sleep(0)
                continue
            await asyncio.sleep(idle_sleep_s)

    def shutdown(self) -> None:
        self._shutdown = True

    def drain_completed_requests(self) -> list[Request]:
        completed = list(self._completed_requests)
        self._completed_requests.clear()
        return completed

    async def wait_for_request(
        self,
        request_id: str,
        poll_interval_s: float = 0.01,
    ) -> Request:
        while True:
            request = self._requests.get(request_id)
            if request is None:
                raise KeyError(f"unknown request {request_id!r}")
            if request.is_finished():
                return request
            await asyncio.sleep(poll_interval_s)

    def _apply_postprocess_result(
        self,
        result: SchedulerPostprocessResult,
    ) -> list[Request]:
        """Apply the postprocess result to the model runner and the completed requests."""
        self.model_runner.apply_allocator_events(result.allocator_events)
        self._completed_requests.extend(result.finished_requests)
        return result.finished_requests
