from types import SimpleNamespace

import httpx
import openai
import pytest

from triage_bench.application.prompt import LLM_MAX_TOKENS, llm_system_prompt
from triage_bench.application.verdict import verdict_model
from triage_bench.domain.task import TASKS, TRIAGE, Task
from triage_bench.domain.ticket import Ticket
from triage_bench.infrastructure.openai_decider import (
    MODEL_ID,
    REASONING_EFFORT,
    OpenAIDecider,
)

TICKET = Ticket(1, "my card still has not arrived", "card_arrival")
TRIAGE_VERDICT = verdict_model(TRIAGE)
TASK_CASES = pytest.mark.parametrize("task", TASKS.values(), ids=[t.name for t in TASKS.values()])


class FakeResponses:
    def __init__(self, outcome: object) -> None:
        self.outcome, self.kwargs = outcome, None

    def parse(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def client_with(outcome: object) -> tuple[object, FakeResponses]:
    responses = FakeResponses(outcome)
    return SimpleNamespace(responses=responses), responses


def good_response() -> object:
    return SimpleNamespace(
        status="completed",
        output_parsed=TRIAGE_VERDICT(label="card_arrival", confidence=0.7),
        usage=SimpleNamespace(input_tokens=100, output_tokens=12),
    )


@TASK_CASES
def test_builds_request_per_fairness_rules(task: Task) -> None:
    client, responses = client_with(good_response())
    OpenAIDecider(client, task).decide(TICKET)  # type: ignore[arg-type]
    k = responses.kwargs
    assert k is not None
    assert k["model"] == MODEL_ID == "gpt-5.6-luna"
    assert k["max_output_tokens"] == LLM_MAX_TOKENS
    assert k["reasoning"] == {"effort": REASONING_EFFORT}
    assert k["input"][0] == {"role": "system", "content": llm_system_prompt(task)}
    assert k["text_format"].model_json_schema() == verdict_model(task).model_json_schema()  # type: ignore[attr-defined]
    assert k["input"] == [{"role": "system", "content": llm_system_prompt(task)},
                          {"role": "user", "content": TICKET.text}]


def test_maps_output_and_cost() -> None:
    client, _ = client_with(good_response())
    d = OpenAIDecider(client, TRIAGE).decide(TICKET)  # type: ignore[arg-type]
    assert d.contestant == "luna" and d.label == "card_arrival" and d.confidence == 0.7
    assert d.cost_usd == (100 * 0.20 + 12 * 1.20) / 1_000_000


def test_incomplete_response_is_error() -> None:
    incomplete = SimpleNamespace(
        status="incomplete", output_parsed=None,
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        usage=SimpleNamespace(input_tokens=1, output_tokens=64),
    )
    client, _ = client_with(incomplete)
    d = OpenAIDecider(client, TRIAGE).decide(TICKET)  # type: ignore[arg-type]
    assert d.is_error and "max_output_tokens" in (d.error or "")


def test_api_error_is_error_decision() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    err = openai.APIConnectionError(request=request)
    client, _ = client_with(err)
    d = OpenAIDecider(client, TRIAGE).decide(TICKET)  # type: ignore[arg-type]
    assert d.is_error


def test_out_of_range_confidence_becomes_error_decision_not_a_raise() -> None:
    # A malformed output_parsed whose confidence violates Decision's own range check must be
    # caught by the same error handling as the API call: mapping the response into a Decision
    # (which validates on construction) has to happen inside decide()'s try block, otherwise
    # decide() would raise and break the Decider contract (deviation guard, see claude_decider).
    good = TRIAGE_VERDICT(label="card_arrival", confidence=0.5)
    bad_verdict = TRIAGE_VERDICT.model_construct(label=good.label, confidence=1.5)  # type: ignore[attr-defined]
    response = SimpleNamespace(
        status="completed",
        output_parsed=bad_verdict,
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    client, _ = client_with(response)
    d = OpenAIDecider(client, TRIAGE).decide(TICKET)  # type: ignore[arg-type]
    assert d.is_error and "confidence" in (d.error or "")


def test_negative_usage_tokens_becomes_error_decision_not_a_raise() -> None:
    # Same reasoning: a response with a negative usage field must not escape decide() as a
    # raised ValueError from Decision's own validation.
    response = SimpleNamespace(
        status="completed",
        output_parsed=TRIAGE_VERDICT(label="card_arrival", confidence=0.5),
        usage=SimpleNamespace(input_tokens=-1, output_tokens=1),
    )
    client, _ = client_with(response)
    d = OpenAIDecider(client, TRIAGE).decide(TICKET)  # type: ignore[arg-type]
    assert d.is_error and "input_tokens" in (d.error or "")
