"""The single instruction and label wording shared by every contestant."""

from triage_bench.domain.ticket import INTENTS

INSTRUCTIONS = "Classify the primary intent of this customer support message."
LLM_MAX_TOKENS = 64


def label_descriptions_text() -> str:
    return "\n".join(f"{i.label}: {i.description}" for i in INTENTS)


def llm_system_prompt() -> str:
    return (
        f"{INSTRUCTIONS}\n\nIntents:\n{label_descriptions_text()}\n\n"
        "Answer with exactly one intent label and your confidence that it is correct, "
        "as a number from 0 to 1."
    )
