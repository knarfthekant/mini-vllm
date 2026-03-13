import time
import uuid
from copy import copy
from enum import IntEnum, auto
from typing import List, Optional

from src.sampling_params import SamplingParams


class RequestStatus(IntEnum):
    """
    Lifecycle status of a Request.

    Ordering rule (mirrors vLLM v1):
        any status > PREEMPTED  →  terminal (finished)

    New statuses must be inserted before or after the PREEMPTED boundary
    to preserve the is_finished() boundary check.
    """

    WAITING = auto()    # queued; not yet scheduled
    RUNNING = auto()    # currently scheduled in the active batch
    PREEMPTED = auto()  # evicted from GPU; will re-enter WAITING

    # ---- terminal states (anything > PREEMPTED) ----
    FINISHED_STOPPED = auto()        # stop-token / stop-string matched
    FINISHED_LENGTH_CAPPED = auto()  # max_tokens limit reached
    FINISHED_ABORTED = auto()        # cancelled by the caller

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED


class Request:
    """
    Tracks the complete state of a single inference request.

    Design follows nano-vllm's Sequence: prompt and generated tokens live
    in a single growing ``token_ids`` list, avoiding repeated concatenation.
    ``num_prompt_tokens`` is fixed at construction and acts as the split point.

    Token layout
    ────────────
    token_ids = [ prompt tokens ... | generated tokens ... ]
                 ◄─ num_prompt_tokens ─►◄─ num_completion_tokens ─►

    KV-cache stubs
    ──────────────
    ``block_table`` and ``num_cached_tokens`` are present as stubs so the
    scheduler and block manager can be wired up later without changing the
    Request interface.

    Args:
        prompt_token_ids: Tokenised prompt (copied; not modified after init).
        sampling_params:  Decoding parameters; defaults to greedy, 64 tokens.
        request_id:       Caller-supplied ID; auto-generated (UUID4) if omitted.
        arrival_time:     Monotonic timestamp; recorded automatically if omitted.
    """

    def __init__(
        self,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams | None = None,
        request_id: Optional[str] = None,
        arrival_time: Optional[float] = None,
    ) -> None:
        if sampling_params is None:
            sampling_params = SamplingParams()

        self.request_id: str = request_id or str(uuid.uuid4())
        self.arrival_time: float = (
            arrival_time if arrival_time is not None else time.monotonic()
        )

        # Single token buffer: grows via append_token()
        self.token_ids: List[int] = copy(prompt_token_ids)
        self.num_prompt_tokens: int = len(prompt_token_ids)
        self.num_tokens: int = len(prompt_token_ids)  # plain int, not property
        self.last_token: int = prompt_token_ids[-1]

        # Sampling parameters (flattened from SamplingParams for hot-path access)
        self.temperature: float = sampling_params.temperature
        self.max_tokens: int = sampling_params.max_tokens
        self.ignore_eos: bool = sampling_params.ignore_eos

        # KV-cache stubs — populated by the block manager once paged attention
        # is implemented.
        self.num_cached_tokens: int = 0
        self.block_table: List[int] = []

        self.status: RequestStatus = RequestStatus.WAITING
        self.stop_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # Token views (slice-based; no extra allocations until needed)
    # ------------------------------------------------------------------

    @property
    def prompt_token_ids(self) -> List[int]:
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self) -> List[int]:
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    # ------------------------------------------------------------------
    # Block helpers (stubs; values become meaningful with paged attention)
    # ------------------------------------------------------------------

    @property
    def num_cached_blocks(self) -> int:
        from src.config.vllm import BLOCK_SIZE
        return self.num_cached_tokens // BLOCK_SIZE

    @property
    def num_blocks(self) -> int:
        from src.config.vllm import BLOCK_SIZE
        return (self.num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE

    @property
    def last_block_num_tokens(self) -> int:
        from src.config.vllm import BLOCK_SIZE
        return self.num_tokens - (self.num_blocks - 1) * BLOCK_SIZE

    def get_block_token_ids(self, block_idx: int) -> List[int]:
        """Return the token IDs that belong to block ``block_idx``."""
        from src.config.vllm import BLOCK_SIZE
        assert 0 <= block_idx < self.num_blocks
        start = block_idx * BLOCK_SIZE
        return self.token_ids[start : start + BLOCK_SIZE]

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def append_token(self, token_id: int) -> None:
        """
        Append one generated token and advance all counters.

        Automatically transitions to FINISHED_LENGTH_CAPPED when max_tokens
        is reached.  Raises if the request is already finished.
        """
        if self.is_finished():
            raise RuntimeError(
                f"Cannot append token to finished request {self.request_id!r} "
                f"(status={self.status.name})"
            )

        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

        if self.num_completion_tokens >= self.max_tokens:
            self.status = RequestStatus.FINISHED_LENGTH_CAPPED

    # ------------------------------------------------------------------
    # Sequence-protocol helpers (mirrors nano-vllm for scheduler compat)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    def __repr__(self) -> str:
        return (
            f"Request("
            f"id={self.request_id!r}, "
            f"status={self.status.name}, "
            f"prompt={self.num_prompt_tokens}, "
            f"completion={self.num_completion_tokens}, "
            f"max_tokens={self.max_tokens}"
            f")"
        )

    def __lt__(self, other: "Request") -> bool:
        """FIFO ordering for priority queues: earlier arrival first."""
        return self.arrival_time < other.arrival_time
