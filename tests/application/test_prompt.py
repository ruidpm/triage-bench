from triage_bench.application.prompt import (
    INSTRUCTIONS,
    LLM_MAX_TOKENS,
    label_descriptions_text,
    llm_system_prompt,
)
from triage_bench.domain.ticket import INTENTS


def test_descriptions_list_every_intent_once() -> None:
    text = label_descriptions_text()
    for intent in INTENTS:
        assert text.count(f"{intent.label}: {intent.description}") == 1


def test_system_prompt_contains_instructions_and_labels() -> None:
    prompt = llm_system_prompt()
    assert INSTRUCTIONS in prompt
    assert label_descriptions_text() in prompt
    assert "confidence" in prompt


def test_token_cap() -> None:
    assert LLM_MAX_TOKENS == 64
