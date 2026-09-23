"""Tickets: one labelled text to classify. Pure domain, no I/O.

Which labels are valid is the task's business (domain.task); a Ticket only checks its own shape.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Ticket:
    id: int
    text: str
    label: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("ticket text is blank")
