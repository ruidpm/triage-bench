import pytest

from triage_bench.domain.ticket import Ticket


def test_ticket_keeps_its_fields() -> None:
    ticket = Ticket(id=1, text="My card never came", label="card_arrival")
    assert (ticket.id, ticket.text, ticket.label) == (1, "My card never came", "card_arrival")


def test_ticket_does_not_judge_labels() -> None:
    # Which labels are valid is the task's business (domain.task); a ticket only checks its shape.
    assert Ticket(id=1, text="Loved it.", label="positive").label == "positive"


def test_ticket_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match="blank"):
        Ticket(id=1, text="   ", label="card_arrival")
