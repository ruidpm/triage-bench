from collections import Counter
from pathlib import Path

from triage_bench.domain.task import TRIAGE
from triage_bench.infrastructure.csv_tickets import load_tickets

LABELS = TRIAGE.label_names()
TICKETS_PER_LABEL = 25
MAX_SAME_LABEL_RUN = 3
EARLY_WINDOW = 40


def _longest_run(labels: list[str]) -> int:
    """Length of the longest run of consecutive identical labels."""
    longest = 0
    current = 0
    previous: str | None = None
    for label in labels:
        current = current + 1 if label == previous else 1
        longest = max(longest, current)
        previous = label
    return longest


def test_committed_dataset_is_balanced_and_valid() -> None:
    tickets = load_tickets(Path("data/tickets.csv"), TRIAGE)
    assert len(tickets) == TICKETS_PER_LABEL * len(LABELS)
    assert Counter(t.label for t in tickets) == {label: TICKETS_PER_LABEL for label in LABELS}
    assert len({t.id for t in tickets}) == len(tickets)


def test_committed_dataset_order_is_well_mixed() -> None:
    """Rows come from PolyAI's test.csv grouped by category; a dashboard replay
    of 200 tickets in id order would show 25 identical intents in a row, which
    looks broken and makes early running accuracy meaningless. The sample must
    be written in a shuffled order instead."""
    tickets = load_tickets(Path("data/tickets.csv"), TRIAGE)
    labels = [t.label for t in tickets]
    assert _longest_run(labels) <= MAX_SAME_LABEL_RUN
    assert set(labels[:EARLY_WINDOW]) == set(LABELS)
