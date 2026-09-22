from collections import Counter
from pathlib import Path

from triage_bench.domain.ticket import LABELS
from triage_bench.infrastructure.csv_tickets import load_tickets

TICKETS_PER_LABEL = 25


def test_committed_dataset_is_balanced_and_valid() -> None:
    tickets = load_tickets(Path("data/tickets.csv"))
    assert len(tickets) == TICKETS_PER_LABEL * len(LABELS)
    assert Counter(t.label for t in tickets) == {label: TICKETS_PER_LABEL for label in LABELS}
    assert [t.id for t in tickets] == sorted(t.id for t in tickets)
    assert len({t.id for t in tickets}) == len(tickets)
