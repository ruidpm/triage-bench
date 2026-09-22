"""What every contestant must look like to the runner."""

from typing import Protocol

from triage_bench.domain.decision import Decision
from triage_bench.domain.ticket import Ticket


class Decider(Protocol):
    name: str

    def decide(self, ticket: Ticket) -> Decision:
        """Classify one ticket. Must not raise; report failures via Decision.error."""
        ...
