import pytest

from triage_bench.domain.ticket import INTENTS, LABELS, Ticket


def test_eight_intents_in_spec_order() -> None:
    assert LABELS == (
        "lost_or_stolen_card", "card_arrival", "declined_card_payment",
        "refund_not_showing_up", "exchange_rate", "top_up_failed",
        "passcode_forgotten", "terminate_account",
    )
    assert all(intent.description for intent in INTENTS)


def test_ticket_accepts_known_label() -> None:
    ticket = Ticket(id=1, text="My card never came", label="card_arrival")
    assert ticket.label == "card_arrival"


def test_ticket_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="unknown label"):
        Ticket(id=1, text="hi", label="pizza")


def test_ticket_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match="blank"):
        Ticket(id=1, text="   ", label="card_arrival")
