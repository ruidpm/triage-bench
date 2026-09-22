"""Re-emit a saved run as TickEvents through the normal runner."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from triage_bench.application.runner import TickEvent, run
from triage_bench.domain.decision import Decision
from triage_bench.domain.ticket import Ticket


class _Replayer:
    def __init__(self, name: str, decisions: Sequence[Decision]) -> None:
        self.name = name
        self._by_ticket = {d.ticket_id: d for d in decisions if d.contestant == name}

    def decide(self, ticket: Ticket) -> Decision:
        return self._by_ticket[ticket.id]


class _NullSink:
    def write(self, decision: Decision) -> None:
        return None


@dataclass(frozen=True)
class CompleteRun:
    tickets: list[Ticket]
    decisions: list[Decision]
    dropped: int


def complete_tickets(
    tickets: Sequence[Ticket], decisions: Sequence[Decision], contestants: Sequence[str]
) -> CompleteRun:
    """Keep only tickets answered by every contestant (a stopped live run leaves a tail)."""
    answered = {name: {d.ticket_id for d in decisions if d.contestant == name}
                for name in contestants}
    kept_ids = {t.id for t in tickets if all(t.id in ids for ids in answered.values())}
    return CompleteRun(
        tickets=[t for t in tickets if t.id in kept_ids],
        decisions=[d for d in decisions if d.ticket_id in kept_ids and d.contestant in contestants],
        dropped=len(tickets) - len(kept_ids),
    )


def replay(
    tickets: Sequence[Ticket], decisions: Sequence[Decision], deciders_order: Sequence[str]
) -> Iterator[TickEvent]:
    deciders = [_Replayer(name, decisions) for name in deciders_order]
    return run(tickets, deciders, _NullSink())
