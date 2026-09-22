from dataclasses import dataclass

from triage_bench.application.prompt import INSTRUCTIONS
from triage_bench.domain.ticket import INTENTS, Ticket
from triage_bench.infrastructure.von_decider import VonDecider

TICKET = Ticket(1, "my card still has not arrived", "card_arrival")


@dataclass
class FakeResult:
    choice: str
    probabilities: dict[str, float]


class FakeEngine:
    def __init__(self, result: FakeResult | Exception) -> None:
        self.result, self.calls = result, []

    def decide(self, state: str, choices: dict[str, str], instructions: str) -> FakeResult:
        self.calls.append((state, choices, instructions))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_maps_result_and_passes_shared_prompt() -> None:
    engine = FakeEngine(FakeResult("card_arrival", {"card_arrival": 0.93, "exchange_rate": 0.07}))
    d = VonDecider(engine).decide(TICKET)
    assert d.contestant == "von" and d.label == "card_arrival"
    assert d.confidence == 0.93 and d.cost_usd == 0.0 and d.latency_ms > 0
    state, choices, instructions = engine.calls[0]
    assert state == TICKET.text and instructions == INSTRUCTIONS
    assert choices == {i.label: i.description for i in INTENTS}


def test_engine_exception_becomes_error_decision() -> None:
    d = VonDecider(FakeEngine(RuntimeError("mps out of memory"))).decide(TICKET)
    assert d.is_error and "mps out of memory" in (d.error or "")
    assert d.label is None and d.latency_ms == 0.0
