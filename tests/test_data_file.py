from collections import Counter
from pathlib import Path

import pytest

from triage_bench.domain.task import SENTIMENT, TRIAGE, Task
from triage_bench.infrastructure.csv_tickets import load_tickets

# (file, task, rows per label, longest same-label run allowed, rows within which every label
# has appeared). The run limit keeps a dashboard replay from showing one label many times in a
# row; the binary file gets a longer allowance because two labels alternate less.
DATASETS = [
    pytest.param(Path("data/triage.csv"), TRIAGE, 25, 3, 40, id="triage"),
    pytest.param(Path("data/sentiment.csv"), SENTIMENT, 100, 4, 10, id="sentiment"),
]


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


@pytest.mark.parametrize(("path", "task", "per_label", "max_run", "early_window"), DATASETS)
def test_committed_dataset_is_balanced_and_valid(
    path: Path, task: Task, per_label: int, max_run: int, early_window: int
) -> None:
    tickets = load_tickets(path, task)
    assert len(tickets) == per_label * len(task.labels)
    assert Counter(t.label for t in tickets) == {name: per_label for name in task.label_names()}
    assert len({t.id for t in tickets}) == len(tickets)


@pytest.mark.parametrize(("path", "task", "per_label", "max_run", "early_window"), DATASETS)
def test_committed_dataset_order_is_well_mixed(
    path: Path, task: Task, per_label: int, max_run: int, early_window: int
) -> None:
    """Source files are grouped or skewed by label; a replay in source order would show many
    identical labels in a row, which looks broken and makes early running accuracy meaningless.
    The samples are written in a shuffled order instead."""
    tickets = load_tickets(path, task)
    labels = [t.label for t in tickets]
    assert _longest_run(labels) <= max_run
    assert set(labels[:early_window]) == set(task.label_names())


def test_sentiment_rows_carry_title_and_body() -> None:
    tickets = load_tickets(Path("data/sentiment.csv"), SENTIMENT)
    assert all("\n\n" in t.text for t in tickets)
