from dataclasses import dataclass, field
from typing import List


@dataclass
class SchedulerOutput:
    """
    Carries the per-step batch decision from the Scheduler to the ModelRunner.

    Both lists are parallel: input_ids[i] and positions[i] describe the tokens
    and their positions for request i.  Variable-length sequences are supported;
    the ModelRunner pads them to a uniform (B, T) tensor internally.

    ``block_tables`` and ``slot_mappings`` are optional today. Standard
    attention ignores them; paged attention will consume them once the paged
    execution path is wired into the model.
    """

    input_ids: List[List[int]] = field(default_factory=list)
    positions: List[List[int]] = field(default_factory=list)
    block_tables: List[List[int]] = field(default_factory=list)
    slot_mappings: List[List[int]] = field(default_factory=list)


@dataclass
class ModelRunnerOutput:
    """
    Carries the per-step sampling result back from the ModelRunner to the
    Executor / Engine.

    sampled_token_ids[i] is the greedy next-token for request i.
    """

    sampled_token_ids: List[int] = field(default_factory=list)
