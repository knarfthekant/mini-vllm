from dataclasses import dataclass, field
from typing import List


@dataclass
class SchedulerOutput:
    """
    Carries the per-step batch decision from the Scheduler to the ModelRunner.

    Both lists are parallel: input_ids[i] and positions[i] describe the tokens
    and their positions for request i.  Variable-length sequences are supported;
    the ModelRunner pads them to a uniform (B, T) tensor internally.

    When paged attention is added, block_tables and slot_mappings will be added
    here so the ModelRunner can build attn_metadata.
    """

    input_ids: List[List[int]] = field(default_factory=list)
    positions: List[List[int]] = field(default_factory=list)


@dataclass
class ModelRunnerOutput:
    """
    Carries the per-step sampling result back from the ModelRunner to the
    Executor / Engine.

    sampled_token_ids[i] is the greedy next-token for request i.
    """

    sampled_token_ids: List[int] = field(default_factory=list)
