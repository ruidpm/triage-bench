"""Load tickets from the committed CSV, validating every row."""

import csv
from pathlib import Path

from triage_bench.domain.task import Task
from triage_bench.domain.ticket import Ticket

EXPECTED_HEADER = ["id", "text", "label"]
FIRST_DATA_ROW = 2  # header is row 1


class TicketLoadError(Exception):
    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


def load_tickets(path: Path, task: Task) -> list[Ticket]:
    if not path.exists():
        raise TicketLoadError([f"tickets file not found: {path}"])
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != EXPECTED_HEADER:
            raise TicketLoadError([f"bad header {header}, expected {EXPECTED_HEADER}"])
        tickets: list[Ticket] = []
        problems: list[str] = []
        for row_number, row in enumerate(reader, start=FIRST_DATA_ROW):
            try:
                ticket_id, text, label = row
                ticket = Ticket(id=int(ticket_id), text=text, label=label)
                task.validate_label(ticket.label)
                tickets.append(ticket)
            except ValueError as exc:
                problems.append(f"row {row_number}: {exc}")
    if problems:
        raise TicketLoadError(problems)
    return tickets
