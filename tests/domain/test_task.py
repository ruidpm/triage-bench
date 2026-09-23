import pytest

from triage_bench.domain.task import (
    DEFAULT_TASK,
    MIN_LABELS,
    SENTIMENT,
    TASKS,
    TRIAGE,
    Label,
    Task,
)

TWO_LABELS = (Label("a", "first"), Label("b", "second"))


def make_task(**overrides: object) -> Task:
    fields: dict[str, object] = {
        "name": "t", "item_noun": "item", "label_noun": "kind",
        "instructions": "do it", "labels": TWO_LABELS,
    }
    fields.update(overrides)
    return Task(**fields)  # type: ignore[arg-type]


def test_triage_keeps_the_eight_shipped_intents_in_order() -> None:
    assert TRIAGE.name == "triage" and TRIAGE.item_noun == "ticket"
    assert TRIAGE.label_noun == "intent"
    assert TRIAGE.instructions == "Classify the primary intent of this customer support message."
    assert TRIAGE.label_names() == (
        "lost_or_stolen_card", "card_arrival", "declined_card_payment",
        "refund_not_showing_up", "exchange_rate", "top_up_failed",
        "passcode_forgotten", "terminate_account",
    )
    assert TRIAGE.choices()["card_arrival"] == (
        "Asking when or whether an ordered card will arrive")


def test_sentiment_is_binary_over_reviews() -> None:
    assert SENTIMENT.name == "sentiment" and SENTIMENT.item_noun == "review"
    assert SENTIMENT.label_noun == "sentiment"
    assert SENTIMENT.instructions == "Classify the sentiment of this customer product review."
    assert SENTIMENT.choices() == {
        "negative": "The customer is unhappy with the product.",
        "positive": "The customer is happy with the product.",
    }


def test_registry_is_keyed_by_task_name_and_defaults_to_triage() -> None:
    assert TASKS == {"triage": TRIAGE, "sentiment": SENTIMENT}
    assert DEFAULT_TASK is TRIAGE


def test_validate_label_accepts_known_and_names_unknown() -> None:
    task = make_task()
    task.validate_label("a")
    with pytest.raises(ValueError, match="unknown label 'zzz' for task 't'"):
        task.validate_label("zzz")


@pytest.mark.parametrize("labels", [(), (Label("a", "first"),)])
def test_task_needs_at_least_two_labels(labels: tuple[Label, ...]) -> None:
    with pytest.raises(ValueError, match=f"at least {MIN_LABELS}"):
        make_task(labels=labels)


def test_task_rejects_duplicate_label_names() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        make_task(labels=(Label("a", "x"), Label("a", "y")))


@pytest.mark.parametrize("field", ["name", "item_noun", "label_noun", "instructions"])
def test_task_rejects_blank_text_fields(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        make_task(**{field: "   "})


@pytest.mark.parametrize(("name", "description"), [("", "x"), ("a", " ")])
def test_label_rejects_blank_name_or_description(name: str, description: str) -> None:
    with pytest.raises(ValueError, match="blank"):
        Label(name, description)
