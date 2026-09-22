import pytest

from triage_bench.application.runner import run
from triage_bench.domain.decision import Decision
from triage_bench.domain.ticket import Ticket

T = [Ticket(1, "card never came", "card_arrival"), Ticket(2, "rate?", "exchange_rate")]


class Scripted:
    def __init__(self, name: str, labels: list[str | None]) -> None:
        self.name, self._labels, self.calls = name, list(labels), []

    def decide(self, ticket: Ticket) -> Decision:
        self.calls.append(ticket.id)
        label = self._labels.pop(0)
        return Decision(ticket.id, self.name, label, 0.9 if label else 0.0,
                        10.0 if label else 0.0, 1, 1, 0.0, None if label else "fail")


class ListSink:
    def __init__(self) -> None:
        self.items: list[Decision] = []

    def write(self, decision: Decision) -> None:
        self.items.append(decision)


def test_lockstep_order_and_totals() -> None:
    a = Scripted("von", ["card_arrival", "card_arrival"])
    b = Scripted("haiku", ["card_arrival", None])
    sink = ListSink()
    events = list(run(T, [a, b], sink))

    assert [e.tick for e in events] == [1, 2] and events[0].total == 2
    assert a.calls == [1, 2] and b.calls == [1, 2]
    assert [d.contestant for d in sink.items] == ["von", "haiku", "von", "haiku"]
    assert events[1].totals["von"].accuracy == 0.5
    assert events[1].totals["haiku"].accuracy == 0.5 and events[1].totals["haiku"].errors == 1


def test_to_dict_is_json_ready_and_marks_correctness() -> None:
    e = next(run(T[:1], [Scripted("von", ["exchange_rate"])], ListSink()))
    payload = e.to_dict()
    assert payload["ticket"] == {"id": 1, "text": "card never came", "label": "card_arrival"}
    assert payload["decisions"]["von"]["correct"] is False
    assert payload["totals"]["von"]["accuracy"] == 0.0


def test_duplicate_names_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        list(run(T, [Scripted("von", []), Scripted("von", [])], ListSink()))


def test_empty_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        list(run([], [Scripted("von", [])], ListSink()))
    with pytest.raises(ValueError):
        list(run(T, [], ListSink()))
