"""GPT-5.6 Luna behind the Decider protocol. One structured-output call, reasoning off."""

import time

import openai
from pydantic import ValidationError

from triage_bench.application.prompt import LLM_MAX_TOKENS, llm_system_prompt
from triage_bench.application.verdict import Verdict
from triage_bench.domain.decision import Decision
from triage_bench.domain.pricing import cost_usd
from triage_bench.domain.ticket import Ticket

CONTESTANT = "luna"
MODEL_ID = "gpt-5.6-luna"
REASONING_EFFORT = "none"  # if the API returns 400 for this model, change to "minimal"
STATUS_COMPLETED = "completed"
MS_PER_SECOND = 1000.0


class OpenAIDecider:
    name = CONTESTANT

    def __init__(self, client: openai.OpenAI, model: str = MODEL_ID) -> None:
        self._client = client
        self._model = model

    def decide(self, ticket: Ticket) -> Decision:
        started = time.perf_counter()
        try:
            response = self._client.responses.parse(
                model=self._model,
                input=[
                    {"role": "system", "content": llm_system_prompt()},
                    {"role": "user", "content": ticket.text},
                ],
                text_format=Verdict,
                max_output_tokens=LLM_MAX_TOKENS,
                # dict[str, str] vs. openai.types.shared_params.Reasoning (a TypedDict)
                reasoning={"effort": REASONING_EFFORT},  # type: ignore[arg-type]
            )
            if response.status != STATUS_COMPLETED:
                reason = getattr(getattr(response, "incomplete_details", None), "reason", "?")
                raise ValueError(f"{response.status}: {reason}")
            verdict = response.output_parsed
            if verdict is None:
                raise ValueError("no parsed output in response")
            usage = response.usage
            if usage is None:
                raise ValueError("response carried no usage")
            latency_ms = (time.perf_counter() - started) * MS_PER_SECOND
            # Constructing Decision (which validates confidence/token ranges) must stay inside
            # this try block: malformed SDK output must become Decision.error, never a raise.
            return Decision(
                ticket_id=ticket.id,
                contestant=CONTESTANT,
                label=verdict.label.value,
                confidence=verdict.confidence,
                latency_ms=latency_ms,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=cost_usd(CONTESTANT, usage.input_tokens, usage.output_tokens),
            )
        except (openai.APIError, ValidationError, ValueError) as exc:
            return self._error_decision(ticket.id, str(exc))

    @staticmethod
    def _error_decision(ticket_id: int, message: str) -> Decision:
        return Decision(
            ticket_id=ticket_id,
            contestant=CONTESTANT,
            label=None,
            confidence=0.0,
            latency_ms=0.0,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            error=message,
        )
