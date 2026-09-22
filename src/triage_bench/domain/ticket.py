"""Intents and tickets. Pure domain, no I/O."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    label: str
    description: str


INTENTS: tuple[Intent, ...] = (
    Intent("lost_or_stolen_card", "Card was lost, stolen, or the customer wants it blocked"),
    Intent("card_arrival", "Asking when or whether an ordered card will arrive"),
    Intent("declined_card_payment", "A card payment was declined or refused at checkout"),
    Intent("refund_not_showing_up", "A refund was promised but has not appeared in the account"),
    Intent("exchange_rate", "Questions about exchange rates or currency conversion applied"),
    Intent("top_up_failed", "Adding money to the account failed or did not go through"),
    Intent("passcode_forgotten", "Forgot the app passcode or PIN and wants to reset it"),
    Intent("terminate_account", "Wants to close or delete their account"),
)

LABELS: tuple[str, ...] = tuple(intent.label for intent in INTENTS)


@dataclass(frozen=True)
class Ticket:
    id: int
    text: str
    label: str

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"unknown label {self.label!r}")
        if not self.text.strip():
            raise ValueError("ticket text is blank")
