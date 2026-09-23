import pytest

from triage_bench.application.prompt import (
    LLM_MAX_TOKENS,
    label_descriptions_text,
    llm_system_prompt,
)
from triage_bench.domain.task import TASKS, TRIAGE, Task

# The exact system prompt that produced runs/triage-sample.jsonl. It must never change.
SHIPPED_TRIAGE_PROMPT = (
    "Classify the primary intent of this customer support message.\n\n"
    "Intents:\n"
    "lost_or_stolen_card: Card was lost, stolen, or the customer wants it blocked\n"
    "card_arrival: Asking when or whether an ordered card will arrive\n"
    "declined_card_payment: A card payment was declined or refused at checkout\n"
    "refund_not_showing_up: A refund was promised but has not appeared in the account\n"
    "exchange_rate: Questions about exchange rates or currency conversion applied\n"
    "top_up_failed: Adding money to the account failed or did not go through\n"
    "passcode_forgotten: Forgot the app passcode or PIN and wants to reset it\n"
    "terminate_account: Wants to close or delete their account\n\n"
    "Answer with exactly one intent label and your confidence that it is correct, "
    "as a number from 0 to 1."
)

TASK_CASES = pytest.mark.parametrize("task", TASKS.values(), ids=[t.name for t in TASKS.values()])


@TASK_CASES
def test_descriptions_list_every_label_once(task: Task) -> None:
    text = label_descriptions_text(task)
    for label in task.labels:
        assert text.count(f"{label.name}: {label.description}") == 1


@TASK_CASES
def test_system_prompt_contains_instructions_labels_and_noun(task: Task) -> None:
    prompt = llm_system_prompt(task)
    assert task.instructions in prompt
    assert label_descriptions_text(task) in prompt
    assert f"exactly one {task.label_noun} label" in prompt
    assert "confidence" in prompt


def test_triage_prompt_is_byte_identical_to_the_shipped_run() -> None:
    assert llm_system_prompt(TRIAGE) == SHIPPED_TRIAGE_PROMPT


def test_token_cap() -> None:
    assert LLM_MAX_TOKENS == 64
