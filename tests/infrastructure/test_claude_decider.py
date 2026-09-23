from types import SimpleNamespace

import anthropic
import httpx2 as httpx
import pytest

from triage_bench.application.prompt import LLM_MAX_TOKENS, llm_system_prompt
from triage_bench.application.verdict import verdict_model
from triage_bench.domain.task import TASKS, TRIAGE, Task
from triage_bench.domain.ticket import Ticket
from triage_bench.infrastructure.claude_decider import MODEL_ID, ClaudeDecider

TICKET = Ticket(1, "my card still has not arrived", "card_arrival")
TRIAGE_VERDICT = verdict_model(TRIAGE)
TASK_CASES = pytest.mark.parametrize("task", TASKS.values(), ids=[t.name for t in TASKS.values()])


class FakeMessages:
    def __init__(self, outcome: object) -> None:
        self.outcome, self.kwargs = outcome, None

    def parse(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def client_with(outcome: object) -> tuple[object, FakeMessages]:
    messages = FakeMessages(outcome)
    return SimpleNamespace(messages=messages), messages


def good_response() -> object:
    return SimpleNamespace(
        parsed_output=TRIAGE_VERDICT(label="card_arrival", confidence=0.85),
        usage=SimpleNamespace(input_tokens=120, output_tokens=15),
    )


@TASK_CASES
def test_builds_request_per_fairness_rules(task: Task) -> None:
    client, messages = client_with(good_response())
    ClaudeDecider(client, task).decide(TICKET)  # type: ignore[arg-type]
    assert messages.kwargs is not None
    assert messages.kwargs["model"] == MODEL_ID == "claude-haiku-4-5"
    assert messages.kwargs["max_tokens"] == LLM_MAX_TOKENS
    assert messages.kwargs["system"] == llm_system_prompt(task)
    assert messages.kwargs["messages"] == [{"role": "user", "content": TICKET.text}]
    schema = messages.kwargs["output_format"].model_json_schema()  # type: ignore[attr-defined]
    assert schema == verdict_model(task).model_json_schema()
    assert "thinking" not in messages.kwargs


def test_maps_parsed_output_tokens_and_cost() -> None:
    client, _ = client_with(good_response())
    d = ClaudeDecider(client, TRIAGE).decide(TICKET)  # type: ignore[arg-type]
    assert d.contestant == "haiku" and d.label == "card_arrival" and d.confidence == 0.85
    assert d.input_tokens == 120 and d.output_tokens == 15
    assert d.cost_usd == (120 * 1.00 + 15 * 5.00) / 1_000_000


def test_api_error_becomes_error_decision() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request)
    err = anthropic.RateLimitError("rate limited", response=response, body=None)
    client, _ = client_with(err)
    d = ClaudeDecider(client, TRIAGE).decide(TICKET)  # type: ignore[arg-type]
    assert d.is_error and "rate limited" in (d.error or "")


def test_missing_parsed_output_is_error() -> None:
    client, _ = client_with(
        SimpleNamespace(parsed_output=None, usage=SimpleNamespace(input_tokens=1, output_tokens=1))
    )
    d = ClaudeDecider(client, TRIAGE).decide(TICKET)  # type: ignore[arg-type]
    assert d.is_error and "no parsed output" in (d.error or "")


def test_out_of_range_confidence_becomes_error_decision_not_a_raise() -> None:
    # Mapping the response into a Decision (which validates on construction) must happen inside
    # decide()'s try block, otherwise decide() would raise and break the Decider contract.
    good = TRIAGE_VERDICT(label="card_arrival", confidence=0.5)
    bad_verdict = TRIAGE_VERDICT.model_construct(label=good.label, confidence=1.5)  # type: ignore[attr-defined]
    usage = SimpleNamespace(input_tokens=1, output_tokens=1)
    client, _ = client_with(SimpleNamespace(parsed_output=bad_verdict, usage=usage))
    d = ClaudeDecider(client, TRIAGE).decide(TICKET)  # type: ignore[arg-type]
    assert d.is_error and "confidence" in (d.error or "")


def test_negative_usage_tokens_becomes_error_decision_not_a_raise() -> None:
    # Same reasoning: a response with a negative usage field must not escape decide() as a
    # raised ValueError from Decision's own validation.
    client, _ = client_with(
        SimpleNamespace(
            parsed_output=TRIAGE_VERDICT(label="card_arrival", confidence=0.5),
            usage=SimpleNamespace(input_tokens=-1, output_tokens=1),
        )
    )
    d = ClaudeDecider(client, TRIAGE).decide(TICKET)  # type: ignore[arg-type]
    assert d.is_error and "input_tokens" in (d.error or "")
