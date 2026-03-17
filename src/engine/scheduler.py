from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
from typing import Deque

from src.config.vllm import VllmConfig
from src.engine.allocator import (
    AllocationManager,
    DenseSlotManager,
    FreeResult,
    SlotMove,
)
from src.request import Request, RequestStatus
from src.worker.interface import ModelRunnerOutput, SchedulerOutput

logger = logging.getLogger(__name__)

@dataclass
class SchedulerPostprocessResult:
    finished_requests: list[Request] = field(default_factory=list)
    cleared_slots: list[int] = field(default_factory=list)
    slot_moves: list[SlotMove] = field(default_factory=list)


class Scheduler:
    def __init__(
        self,
        vllm_config: VllmConfig,
        allocator: AllocationManager,
        max_seq_length: int,
        eos_token_id: int | None = None,
    ) -> None:
        self.max_num_seqs = vllm_config.max_num_seqs
        self.max_num_batched_tokens = vllm_config.max_num_batched_tokens
        self.max_seq_length = max_seq_length
        self.eos_token_id = eos_token_id
        self.allocator = allocator

        self.waiting: Deque[Request] = deque()
        self.running: list[Request] = []
        self._pending_result = SchedulerPostprocessResult()

    def add_request(self, request: Request) -> None:
        if not request.prompt_token_ids:
            raise ValueError("request prompt must contain at least one token")
        if request.num_prompt_tokens > self.max_seq_length:
            raise ValueError(
                f"request prompt length {request.num_prompt_tokens} exceeds "
                f"max_seq_length={self.max_seq_length}"
            )
        if request.num_prompt_tokens > self.max_num_batched_tokens:
            raise ValueError(
                f"request prompt length {request.num_prompt_tokens} exceeds "
                f"max_num_batched_tokens={self.max_num_batched_tokens}; "
                "chunked prefill is not implemented yet"
            )
        request.status = RequestStatus.WAITING
        self.waiting.append(request)

    def abort_request(self, request_id: str) -> SchedulerPostprocessResult | None:
        request = self._remove_from_waiting(request_id)
        if request is not None:
            request.status = RequestStatus.FINISHED_ABORTED
            return SchedulerPostprocessResult(finished_requests=[request])

        request = self._remove_from_running(request_id)
        if request is None:
            return None

        request.status = RequestStatus.FINISHED_ABORTED
        free_result = self.allocator.free(request)
        request.stop_reason = "aborted"
        result = SchedulerPostprocessResult(finished_requests=[request])
        self._append_free_result(result, free_result)
        return result

    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> SchedulerOutput:
        scheduled: list[Request] = []
        input_ids: list[list[int]] = []
        positions: list[list[int]] = []
        block_tables: list[list[int]] = []
        budget = self.max_num_batched_tokens

        running = self._running_in_execution_order()
        scheduled_all_running = True

        # Schedule running requests (Priority)
        for request in running:
            if budget < 1:
                logger.debug("Stopping due to low budget. Budget: %d", budget)
                scheduled_all_running = False
                break
            # Cap running requests to the max sequence length
            if request.num_tokens > self.max_seq_length:
                self._finish_without_execution(
                    request,
                    RequestStatus.FINISHED_LENGTH_CAPPED,
                    "max_seq_length",
                )
                continue
            # If running requests exceed the allocator capacity, stop scheduling
            if not self.allocator.can_append(request):
                logger.debug("Stopping due to allocator capacity. Request: %s", request)
                scheduled_all_running = False
                break

            self.allocator.append(request)
            scheduled.append(request)
            input_ids.append([request.last_token])
            positions.append([request.num_tokens - 1]) # only the last token is new
            block_tables.append(list(request.block_table))
            budget -= 1

        # Allocate new requests
        if scheduled_all_running and budget > 0:
            while self.waiting and len(self.running) < self.max_num_seqs:
                request = self.waiting[0]
                num_new_tokens = request.num_tokens - request.num_cached_tokens
                if num_new_tokens <= 0:
                    self.waiting.popleft()
                    continue
                if num_new_tokens > budget:
                    break
                if not self.allocator.can_allocate(request):
                    break

                self.allocator.allocate(request)
                self.waiting.popleft()
                request.status = RequestStatus.RUNNING
                self.running.append(request)

                scheduled.append(request)
                input_ids.append(
                    request.token_ids[request.num_cached_tokens : request.num_tokens]
                )
                positions.append(
                    list(range(request.num_cached_tokens, request.num_tokens))
                )
                block_tables.append(list(request.block_table))
                budget -= num_new_tokens

        return self._build_output(scheduled, input_ids, positions, block_tables)

    def postprocess(
        self,
        scheduler_output: SchedulerOutput,
        model_output: ModelRunnerOutput,
    ) -> SchedulerPostprocessResult:
        """Update the request objects based on the model output."""
        if len(scheduler_output.requests) != len(model_output.sampled_token_ids):
            raise ValueError(
                "scheduler batch size and sampled token count must match"
            )

        result = SchedulerPostprocessResult()
        finished_set: set[str] = set()

        for request, input_ids, sampled_token in zip(
            scheduler_output.requests,
            scheduler_output.input_ids,
            model_output.sampled_token_ids,
        ):
            request.num_cached_tokens += len(input_ids)

            # Normalize to running. The scheduler output should only have running requests.
            if request.status != RequestStatus.RUNNING:
                request.status = RequestStatus.RUNNING

            # Update request with the sampled token
            request.append_token(sampled_token)

            if (
                request.status == RequestStatus.FINISHED_LENGTH_CAPPED
                and request.stop_reason is None
            ):
                request.stop_reason = "max_tokens"

            if (
                self.eos_token_id is not None
                and not request.ignore_eos
                and sampled_token == self.eos_token_id
            ):
                request.status = RequestStatus.FINISHED_STOPPED
                request.stop_reason = "eos_token"

            if request.num_tokens > self.max_seq_length:
                request.status = RequestStatus.FINISHED_LENGTH_CAPPED
                request.stop_reason = "max_seq_length"

            if request.is_finished() and request.request_id not in finished_set:
                finished_set.add(request.request_id)
                result.finished_requests.append(request)

        for request in result.finished_requests:
            self._remove_running_request_object(request)
            free_result = self.allocator.free(request)
            self._append_free_result(result, free_result)

        return result

    def drain_pending_result(self) -> SchedulerPostprocessResult:
        result = self._pending_result
        self._pending_result = SchedulerPostprocessResult()
        return result

    def _running_in_execution_order(self) -> list[Request]:
        if isinstance(self.allocator, DenseSlotManager):
            return sorted(self.running, key=self.allocator.slot_of)
        return list(self.running)

    def _build_output(
        self,
        requests: list[Request],
        input_ids: list[list[int]],
        positions: list[list[int]],
        block_tables: list[list[int]],
    ) -> SchedulerOutput:
        # logger.debug("Building output. Requests: %s, Input IDs: %s, Positions: %s, Block Tables: %s", requests, input_ids, positions, block_tables)
        return SchedulerOutput(
            requests=requests,
            input_ids=input_ids,
            positions=positions,
            block_tables=block_tables,
        )

    def _remove_from_waiting(self, request_id: str) -> Request | None:
        for index, request in enumerate(self.waiting):
            if request.request_id != request_id:
                continue
            del self.waiting[index]
            return request
        return None

    def _remove_from_running(self, request_id: str) -> Request | None:
        """Remove the request object by request_id from the running list."""
        for request in self.running:
            if request.request_id == request_id:
                self._remove_running_request_object(request)
                return request
        return None

    def _remove_running_request_object(self, request: Request) -> None:
        """Remove the request object from the running list."""
        self.running = [running for running in self.running if running is not request]

    def _append_free_result(
        self,
        result: SchedulerPostprocessResult,
        free_result: FreeResult,
    ) -> None:
        """Append allocator changes to the postprocess result."""
        if free_result.slot_move is not None:
            result.slot_moves.append(free_result.slot_move)
        if free_result.cleared_slot is not None:
            result.cleared_slots.append(free_result.cleared_slot)

    def _finish_without_execution(
        self,
        request: Request,
        status: RequestStatus,
        stop_reason: str,
    ) -> None:
        request.status = status
        request.stop_reason = stop_reason
        self._remove_running_request_object(request)
        free_result = self.allocator.free(request)
        self._pending_result.finished_requests.append(request)
        self._append_free_result(self._pending_result, free_result)
