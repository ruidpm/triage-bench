from dataclasses import dataclass

import pytest

from triage_bench.domain.task import TASKS, TRIAGE, Task
from triage_bench.domain.ticket import Ticket
from triage_bench.infrastructure.von_decider import VonDecider

TICKET = Ticket(1, "my card still has not arrived", "card_arrival")
TASK_CASES = pytest.mark.parametrize("task", TASKS.values(), ids=[t.name for t in TASKS.values()])


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


@TASK_CASES
def test_passes_the_tasks_choices_and_instructions(task: Task) -> None:
    first = task.label_names()[0]
    engine = FakeEngine(FakeResult(first, {first: 0.93}))
    d = VonDecider(engine, task).decide(TICKET)
    assert d.contestant == "von" and d.label == first
    assert d.confidence == 0.93 and d.cost_usd == 0.0 and d.latency_ms > 0
    state, choices, instructions = engine.calls[0]
    assert state == TICKET.text and instructions == task.instructions
    assert choices == task.choices()


def test_warm_up_uses_the_same_choices_and_instructions() -> None:
    engine = FakeEngine(FakeResult("card_arrival", {"card_arrival": 1.0}))
    VonDecider(engine, TRIAGE).warm_up()
    _, choices, instructions = engine.calls[0]
    assert choices == TRIAGE.choices() and instructions == TRIAGE.instructions


def test_engine_exception_becomes_error_decision() -> None:
    d = VonDecider(FakeEngine(RuntimeError("mps out of memory")), TRIAGE).decide(TICKET)
    assert d.is_error and "mps out of memory" in (d.error or "")
    assert d.label is None and d.latency_ms == 0.0


def test_missing_choice_in_probabilities_becomes_error_decision() -> None:
    engine = FakeEngine(FakeResult("card_arrival", {"exchange_rate": 1.0}))
    d = VonDecider(engine, TRIAGE).decide(TICKET)
    assert d.is_error and d.label is None and d.latency_ms == 0.0
    assert "card_arrival" in (d.error or "")


def test_confidence_slightly_above_one_becomes_error_decision() -> None:
    engine = FakeEngine(FakeResult("card_arrival", {"card_arrival": 1.0000000002}))
    d = VonDecider(engine, TRIAGE).decide(TICKET)
    assert d.is_error and d.label is None and d.latency_ms == 0.0
