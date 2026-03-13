from dataclasses import dataclass


@dataclass
class SamplingParams:
    """
    Controls how the model samples the next token at each decode step.

    temperature: Sampling temperature.
        0.0  → greedy argmax (deterministic).
        >0.0 → softmax with this temperature; higher = more random.
    max_tokens:  Maximum number of new tokens to generate.
    ignore_eos:  If True, generation continues past the EOS token up to
                 max_tokens; useful for benchmarking and forced decoding.
    """

    temperature: float = 0.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be >= 0.0, got {self.temperature}"
            )
        if self.max_tokens < 1:
            raise ValueError(
                f"max_tokens must be >= 1, got {self.max_tokens}"
            )

    @property
    def greedy(self) -> bool:
        """True when temperature == 0 (greedy argmax sampling)."""
        return self.temperature == 0.0
