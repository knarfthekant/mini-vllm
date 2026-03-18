from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from logging import getLogger
from typing import Protocol

import numpy as np
import xxhash

from src.config.vllm import BLOCK_SIZE
from src.request import Request

logger = getLogger(__name__)


@dataclass(frozen=True)
class MoveDenseSlot:
    src_slot: int
    dst_slot: int


@dataclass(frozen=True)
class ClearDenseSlot:
    slot: int


AllocatorEvent = MoveDenseSlot | ClearDenseSlot


@dataclass(frozen=True)
class RequestBatchState:
    block_table: list[int] = field(default_factory=list)


class AllocationManager(Protocol):
    def can_allocate(self, request: Request) -> bool: ...
    """Check if there is enough backend capacity to admit the request."""

    def allocate(self, request: Request) -> None: ...
    """Bind backend allocation state to the request."""

    def can_append(self, request: Request) -> bool: ...
    """Check if the request can advance by one scheduling step."""

    def append(self, request: Request) -> None: ...
    """Advance backend allocation state for one scheduling step."""

    def free(self, request: Request) -> list[AllocatorEvent]: ...
    """Release backend allocation state and return physical cache events."""

    def execution_key(self, request: Request) -> int: ...
    """Return the execution order key for the request."""

    def export_batch_state(self, request: Request) -> RequestBatchState: ...
    """Return backend metadata needed by the current scheduled batch."""


class PagedBlock:
    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids: list[int] = []

    def update(self, hash_value: int, token_ids: list[int]) -> None:
        self.hash = hash_value
        self.token_ids = token_ids

    def reset(self) -> None:
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


@dataclass
class PagedRequestState:
    block_table: list[int] = field(default_factory=list)
    execution_key: int = 0


class PagedBlockManager:
    """Virtual sequence block allocator for the paged KV backend."""

    def __init__(self, num_gpu_blocks: int, block_size: int = BLOCK_SIZE) -> None:
        if num_gpu_blocks < 1:
            raise ValueError(f"num_gpu_blocks must be >= 1, got {num_gpu_blocks}")
        self.block_size = block_size
        self.blocks = [PagedBlock(i) for i in range(num_gpu_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_gpu_blocks))
        self._request_states: dict[str, PagedRequestState] = {}
        self._next_execution_key = 0

        logger.info("PagedBlockManager initialized with %d blocks", num_gpu_blocks)

    @classmethod
    def _compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _claim_free_block(self) -> PagedBlock:
        if not self.free_block_ids:
            raise RuntimeError("insufficient free paged KV blocks")
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        if block.ref_count != 0:
            raise RuntimeError(
                f"internal error: block {block_id} is not free (ref_count={block.ref_count})"
            )
        block.reset()
        return block

    def _require_state(self, request: Request) -> PagedRequestState:
        try:
            return self._request_states[request.request_id]
        except KeyError as exc:
            raise KeyError(f"request {request.request_id!r} is not allocated") from exc

    def _num_required_blocks(self, request: Request) -> int:
        return (request.num_tokens + self.block_size - 1) // self.block_size

    def _num_new_blocks(self, request: Request, state: PagedRequestState) -> int:
        return self._num_required_blocks(request) - len(state.block_table)

    def _get_block_token_ids(self, request: Request, block_idx: int) -> list[int]:
        num_required_blocks = self._num_required_blocks(request)
        if not 0 <= block_idx < num_required_blocks:
            raise IndexError(
                f"block index {block_idx} out of range for {num_required_blocks} blocks"
            )
        start = block_idx * self.block_size
        return request.token_ids[start : start + self.block_size]

    def can_allocate(self, request: Request) -> bool:
        if request.request_id in self._request_states:
            return False
        return len(self.free_block_ids) >= self._num_required_blocks(request)

    def allocate(self, request: Request) -> None:
        if request.request_id in self._request_states:
            raise RuntimeError(
                f"request {request.request_id!r} is already allocated; use append()"
            )

        state = PagedRequestState(execution_key=self._next_execution_key)
        self._next_execution_key += 1

        rolling = -1
        cache_miss = False

        for block_idx in range(self._num_required_blocks(request)):
            token_ids = self._get_block_token_ids(request, block_idx)
            is_full_block = len(token_ids) == self.block_size

            if is_full_block:
                rolling = self._compute_hash(token_ids, rolling)
            else:
                rolling = -1

            if (not cache_miss) and is_full_block:
                candidate_id = self.hash_to_block_id.get(rolling)
                if (
                    candidate_id is not None
                    and self.blocks[candidate_id].hash == rolling
                    and self.blocks[candidate_id].token_ids == token_ids
                ):
                    # Cache hit: share the existing block
                    block = self.blocks[candidate_id]
                    if block.ref_count == 0:
                        try:
                            self.free_block_ids.remove(candidate_id)
                        except ValueError:
                            pass
                        block.ref_count = 1
                    else:
                        block.ref_count += 1
                    state.block_table.append(candidate_id)
                    request.num_cached_tokens += self.block_size
                    self.hash_to_block_id[rolling] = candidate_id
                    continue
                cache_miss = True

            block = self._claim_free_block()
            state.block_table.append(block.block_id)

            if is_full_block:
                block.update(rolling, token_ids)
                self.hash_to_block_id[rolling] = block.block_id

        self._request_states[request.request_id] = state

    def can_append(self, request: Request) -> bool:
        try:
            state = self._require_state(request)
        except KeyError:
            return False

        missing = self._num_new_blocks(request, state)
        if missing <= 0:
            return True
        return len(self.free_block_ids) >= missing

    def append(self, request: Request) -> None:
        state = self._require_state(request)

        if request.num_tokens % self.block_size == 0 and state.block_table:
            last_block = self.blocks[state.block_table[-1]]
            if last_block.hash == -1:
                token_ids = self._get_block_token_ids(request, len(state.block_table) - 1)
                prefix = (
                    self.blocks[state.block_table[-2]].hash
                    if len(state.block_table) > 1
                    else -1
                )
                hash_value = self._compute_hash(token_ids, prefix)
                last_block.update(hash_value, token_ids)
                self.hash_to_block_id[hash_value] = last_block.block_id

        missing = self._num_new_blocks(request, state)
        if missing <= 0:
            return
        if missing > len(self.free_block_ids):
            raise RuntimeError("insufficient free paged KV blocks to append")
        for _ in range(missing):
            block = self._claim_free_block()
            state.block_table.append(block.block_id)

    def free(self, request: Request) -> list[AllocatorEvent]:
        state = self._request_states.pop(request.request_id, None)
        if state is None:
            request.num_cached_tokens = 0
            return []

        while state.block_table:
            block_id = state.block_table.pop()
            block = self.blocks[block_id]
            if block.ref_count <= 0:
                raise RuntimeError(
                    f"internal error: freeing block {block_id} with ref_count={block.ref_count}"
                )
            block.ref_count -= 1
            if block.ref_count == 0:
                self.free_block_ids.append(block_id)

        request.num_cached_tokens = 0
        return []

    def execution_key(self, request: Request) -> int:
        return self._require_state(request).execution_key

    def export_batch_state(self, request: Request) -> RequestBatchState:
        state = self._require_state(request)
        return RequestBatchState(block_table=list(state.block_table))


class DenseSlotManager:
    """
    Virtual sequence slot allocator for the dense KV backend.
    Dense KV caches are row-based, so each active request must own a stable slot.
    This manager keeps those slot assignments compact so the active batch can
    always occupy the first ``num_running`` rows of the model KV cache.
    """

    def __init__(self, max_num_seqs: int) -> None:
        if max_num_seqs < 1:
            raise ValueError(f"max_num_seqs must be >= 1, got {max_num_seqs}")
        self.max_num_seqs = max_num_seqs
        self._request_to_slot: dict[str, int] = {}
        self._slot_to_request: list[Request | None] = [None] * max_num_seqs

    def _find_free_slot(self) -> int | None:
        for idx, request in enumerate(self._slot_to_request):
            if request is None:
                return idx
        return None

    def _highest_occupied_slot(self) -> int | None:
        for idx in range(self.max_num_seqs - 1, -1, -1):
            if self._slot_to_request[idx] is not None:
                return idx
        return None

    def can_allocate(self, request: Request) -> bool:
        return (
            request.request_id not in self._request_to_slot
            and self._find_free_slot() is not None
        )

    def allocate(self, request: Request) -> None:
        slot = self._find_free_slot()
        if slot is None:
            raise RuntimeError("no free dense KV slots available")
        if request.request_id in self._request_to_slot:
            raise RuntimeError(f"request {request.request_id!r} is already allocated")
        self._request_to_slot[request.request_id] = slot
        self._slot_to_request[slot] = request

    def can_append(self, request: Request) -> bool:
        return request.request_id in self._request_to_slot

    def append(self, request: Request) -> None:
        if request.request_id not in self._request_to_slot:
            raise RuntimeError("request cannot append in dense KV cache")

    def free(self, request: Request) -> list[AllocatorEvent]:
        """Free a dense KV slot and return operations to adjust the slot assignment."""
        slot = self._request_to_slot.pop(request.request_id, None)
        request.num_cached_tokens = 0
        if slot is None:
            return []

        self._slot_to_request[slot] = None
        tail_slot = self._highest_occupied_slot()
        if tail_slot is None or tail_slot == slot:
            return [ClearDenseSlot(slot=slot)]

        moved_request = self._slot_to_request[tail_slot]
        if moved_request is None:
            raise RuntimeError("dense slot manager is internally inconsistent")

        self._slot_to_request[slot] = moved_request
        self._slot_to_request[tail_slot] = None
        self._request_to_slot[moved_request.request_id] = slot
        return [
            MoveDenseSlot(src_slot=tail_slot, dst_slot=slot),
            ClearDenseSlot(slot=tail_slot),
        ]

    def execution_key(self, request: Request) -> int:
        return self.slot_of(request)

    def export_batch_state(self, request: Request) -> RequestBatchState:
        self.slot_of(request)
        return RequestBatchState()

    def slot_of(self, request: Request) -> int:
        try:
            return self._request_to_slot[request.request_id]
        except KeyError as exc:
            raise KeyError(f"request {request.request_id!r} is not allocated") from exc
