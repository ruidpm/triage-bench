"""A single contestant's answer for a single ticket. Pure domain."""

from dataclasses import dataclass

CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0


@dataclass(frozen=True)
class Decision:
    ticket_id: int
    contestant: str
    label: str | None
    confidence: float
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error: str | None = None

    def __post_init__(self) -> None:
        if not CONFIDENCE_MIN <= self.confidence <= CONFIDENCE_MAX:
            raise ValueError(f"confidence out of range: {self.confidence}")
        for name in ("latency_ms", "input_tokens", "output_tokens", "cost_usd"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")

    @property
    def is_error(self) -> bool:
        return self.error is not None

    def is_correct(self, true_label: str) -> bool:
        return not self.is_error and self.label == true_label
