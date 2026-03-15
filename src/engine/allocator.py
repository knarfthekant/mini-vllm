from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.config.vllm import BLOCK_SIZE
from src.request import Request


@dataclass(frozen=True)
class SlotMove:
    src_slot: int
    dst_slot: int


@dataclass(frozen=True)
class FreeResult:
    cleared_slot: int | None = None
    slot_move: SlotMove | None = None


class AllocationManager(Protocol):
    def can_allocate(self, request: Request) -> bool: ...

    def allocate(self, request: Request) -> None: ...

    def can_append(self, request: Request) -> bool: ...

    def append(self, request: Request) -> None: ...

    def free(self, request: Request) -> FreeResult: ...


class PagedBlockManager:
    """Engine-side free-list allocator for the paged KV backend."""

    def __init__(self, num_gpu_blocks: int, block_size: int = BLOCK_SIZE) -> None:
        if num_gpu_blocks < 1:
            raise ValueError(f"num_gpu_blocks must be >= 1, got {num_gpu_blocks}")
        self.block_size = block_size
        self._free_blocks = list(range(num_gpu_blocks))

    def can_allocate(self, request: Request) -> bool:
        return len(self._free_blocks) >= self._num_missing_blocks(request)

    def allocate(self, request: Request) -> None:
        self._allocate_missing_blocks(request)

    def can_append(self, request: Request) -> bool:
        return len(self._free_blocks) >= self._num_missing_blocks(request)

    def append(self, request: Request) -> None:
        self._allocate_missing_blocks(request)

    def free(self, request: Request) -> FreeResult:
        while request.block_table:
            self._free_blocks.append(request.block_table.pop())
        request.num_cached_tokens = 0
        return FreeResult()

    def _allocate_missing_blocks(self, request: Request) -> None:
        missing = self._num_missing_blocks(request)
        if missing < 0:
            raise RuntimeError("request block table exceeds required capacity")
        if missing > len(self._free_blocks):
            raise RuntimeError("insufficient free paged KV blocks")
        for _ in range(missing):
            request.block_table.append(self._free_blocks.pop())

    def _num_missing_blocks(self, request: Request) -> int:
        required_blocks = (request.num_tokens + self.block_size - 1) // self.block_size
        return required_blocks - len(request.block_table)


class DenseSlotManager:
    """
    Scheduler-visible capacity accounting for the dense KV backend.

    Dense KV caches are row-based, so each active request must own a stable slot.
    This manager keeps those slot assignments compact so the active batch can
    always occupy the first ``num_running`` rows of the model KV cache.
    """

    def __init__(self, max_num_seqs: int, max_seq_length: int) -> None:
        if max_num_seqs < 1:
            raise ValueError(f"max_num_seqs must be >= 1, got {max_num_seqs}")
        if max_seq_length < 1:
            raise ValueError(f"max_seq_length must be >= 1, got {max_seq_length}")
        self.max_num_seqs = max_num_seqs
        self.max_seq_length = max_seq_length
        self._request_to_slot: dict[str, int] = {}
        self._slot_to_request: list[Request | None] = [None] * max_num_seqs

    def can_allocate(self, request: Request) -> bool:
        return (
            request.request_id not in self._request_to_slot
            and self._find_free_slot() is not None
            and request.num_tokens <= self.max_seq_length
        )

    def allocate(self, request: Request) -> None:
        slot = self._find_free_slot()
        if slot is None:
            raise RuntimeError("no free dense KV slots available")
        if request.num_tokens > self.max_seq_length:
            raise RuntimeError("request exceeds dense KV max_seq_length")
        self._request_to_slot[request.request_id] = slot
        self._slot_to_request[slot] = request

    def can_append(self, request: Request) -> bool:
        return (
            request.request_id in self._request_to_slot
            and request.num_tokens <= self.max_seq_length
        )

    def append(self, request: Request) -> None:
        if not self.can_append(request):
            raise RuntimeError("request cannot append in dense KV cache")

    def free(self, request: Request) -> FreeResult:
        slot = self._request_to_slot.pop(request.request_id, None)
        if slot is None:
            return FreeResult()

        self._slot_to_request[slot] = None
        tail_slot = self._highest_occupied_slot()
        if tail_slot is None:
            return FreeResult(cleared_slot=slot)
        if tail_slot == slot:
            return FreeResult(cleared_slot=slot)

        moved_request = self._slot_to_request[tail_slot]
        if moved_request is None:
            raise RuntimeError("dense slot manager is internally inconsistent")

        self._slot_to_request[slot] = moved_request
        self._slot_to_request[tail_slot] = None
        self._request_to_slot[moved_request.request_id] = slot
        return FreeResult(
            cleared_slot=tail_slot,
            slot_move=SlotMove(src_slot=tail_slot, dst_slot=slot),
        )

    def slot_of(self, request: Request) -> int:
        try:
            return self._request_to_slot[request.request_id]
        except KeyError as exc:
            raise KeyError(f"request {request.request_id!r} is not allocated") from exc

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
