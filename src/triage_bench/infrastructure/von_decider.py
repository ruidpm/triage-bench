"""Von, the local System One model, behind the Decider protocol."""

import time
from typing import Protocol

from triage_bench.application.prompt import INSTRUCTIONS
from triage_bench.domain.decision import Decision
from triage_bench.domain.ticket import INTENTS, Ticket

CONTESTANT = "von"
WARM_UP_TEXT = "warm up"
MS_PER_SECOND = 1000.0


class VonResult(Protocol):
    choice: str
    probabilities: dict[str, float]


class VonEngine(Protocol):
    def decide(self, state: str, choices: dict[str, str], instructions: str) -> VonResult: ...


class VonDecider:
    name = CONTESTANT

    def __init__(self, engine: VonEngine) -> None:
        self._engine = engine
        self._choices = {i.label: i.description for i in INTENTS}

    @classmethod
    def from_sdk(cls) -> "VonDecider":
        import von  # local model; imported here so tests never load it

        return cls(von)

    def warm_up(self) -> None:
        self._engine.decide(state=WARM_UP_TEXT, choices=self._choices, instructions=INSTRUCTIONS)

    def decide(self, ticket: Ticket) -> Decision:
        started = time.perf_counter()
        try:
            result = self._engine.decide(
                state=ticket.text, choices=self._choices, instructions=INSTRUCTIONS
            )
        except Exception as exc:  # adapters must not raise; the runner needs every tick
            return Decision(ticket.id, CONTESTANT, None, 0.0, 0.0, 0, 0, 0.0, error=str(exc))
        latency_ms = (time.perf_counter() - started) * MS_PER_SECOND
        return Decision(
            ticket_id=ticket.id,
            contestant=CONTESTANT,
            label=result.choice,
            confidence=result.probabilities[result.choice],
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
        )
