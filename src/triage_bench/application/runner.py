"""Feed every ticket to every decider in lockstep and yield one event per ticket."""

from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from triage_bench.application.decider import Decider
from triage_bench.domain.decision import Decision
from triage_bench.domain.metrics import Totals, summarise
from triage_bench.domain.ticket import Ticket


class ResultSink(Protocol):
    def write(self, decision: Decision) -> None: ...


@dataclass(frozen=True)
class TickEvent:
    tick: int
    total: int
    ticket: Ticket
    decisions: dict[str, Decision]
    totals: dict[str, Totals]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "total": self.total,
            "ticket": asdict(self.ticket),
            "decisions": {
                name: {**asdict(d), "correct": d.is_correct(self.ticket.label)}
                for name, d in self.decisions.items()
            },
            "totals": {name: asdict(t) for name, t in self.totals.items()},
        }


def run(
    tickets: Sequence[Ticket], deciders: Sequence[Decider], sink: ResultSink
) -> Iterator[TickEvent]:
    if not tickets:
        raise ValueError("no tickets to run")
    if not deciders:
        raise ValueError("no deciders to run")
    names = [d.name for d in deciders]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate decider names: {names}")

    truth = {t.id: t.label for t in tickets}
    history: dict[str, list[Decision]] = {name: [] for name in names}
    for index, ticket in enumerate(tickets, start=1):
        decisions: dict[str, Decision] = {}
        for decider in deciders:
            decision = decider.decide(ticket)
            sink.write(decision)
            history[decider.name].append(decision)
            decisions[decider.name] = decision
        totals = {name: summarise(name, history[name], truth) for name in names}
        yield TickEvent(index, len(tickets), ticket, decisions, totals)
