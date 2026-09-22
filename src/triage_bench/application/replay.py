"""Re-emit a saved run as TickEvents through the normal runner."""

from collections.abc import Iterator, Sequence

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


def replay(
    tickets: Sequence[Ticket], decisions: Sequence[Decision], deciders_order: Sequence[str]
) -> Iterator[TickEvent]:
    deciders = [_Replayer(name, decisions) for name in deciders_order]
    return run(tickets, deciders, _NullSink())
