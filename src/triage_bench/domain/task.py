"""Tasks: what the contestants are asked and the label set they answer from. Pure domain."""

from dataclasses import dataclass

MIN_LABELS = 2


@dataclass(frozen=True)
class Label:
    name: str
    description: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("label name is blank")
        if not self.description.strip():
            raise ValueError(f"label {self.name!r} has a blank description")


@dataclass(frozen=True)
class Task:
    name: str          # CLI value, data/run file stem, run-file header
    item_noun: str     # "ticket" / "review": dashboard copy only
    label_noun: str    # "intent" / "sentiment": wording inside the LLM system prompt
    instructions: str
    labels: tuple[Label, ...]

    def __post_init__(self) -> None:
        for field in ("name", "item_noun", "label_noun", "instructions"):
            if not getattr(self, field).strip():
                raise ValueError(f"task {self.name!r}: {field} is blank")
        if len(self.labels) < MIN_LABELS:
            raise ValueError(f"task {self.name!r} needs at least {MIN_LABELS} labels")
        names = self.label_names()
        if len(set(names)) != len(names):
            raise ValueError(f"task {self.name!r} has duplicate label names: {names}")

    def label_names(self) -> tuple[str, ...]:
        return tuple(label.name for label in self.labels)

    def choices(self) -> dict[str, str]:
        return {label.name: label.description for label in self.labels}

    def validate_label(self, label: str) -> None:
        if label not in self.label_names():
            raise ValueError(f"unknown label {label!r} for task {self.name!r}")


TRIAGE = Task(
    name="triage",
    item_noun="ticket",
    label_noun="intent",
    instructions="Classify the primary intent of this customer support message.",
    labels=(
        Label("lost_or_stolen_card", "Card was lost, stolen, or the customer wants it blocked"),
        Label("card_arrival", "Asking when or whether an ordered card will arrive"),
        Label("declined_card_payment", "A card payment was declined or refused at checkout"),
        Label("refund_not_showing_up",
              "A refund was promised but has not appeared in the account"),
        Label("exchange_rate", "Questions about exchange rates or currency conversion applied"),
        Label("top_up_failed", "Adding money to the account failed or did not go through"),
        Label("passcode_forgotten", "Forgot the app passcode or PIN and wants to reset it"),
        Label("terminate_account", "Wants to close or delete their account"),
    ),
)

SENTIMENT = Task(
    name="sentiment",
    item_noun="review",
    label_noun="sentiment",
    instructions="Classify the sentiment of this customer product review.",
    labels=(
        Label("negative", "The customer is unhappy with the product."),
        Label("positive", "The customer is happy with the product."),
    ),
)

TASKS: dict[str, Task] = {task.name: task for task in (TRIAGE, SENTIMENT)}
DEFAULT_TASK = TRIAGE
