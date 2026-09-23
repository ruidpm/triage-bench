"""Claude Haiku 4.5 behind the Decider protocol. One structured-output call, no thinking."""

import time

import anthropic
from pydantic import ValidationError

from triage_bench.application.prompt import LLM_MAX_TOKENS, llm_system_prompt
from triage_bench.application.verdict import verdict_label, verdict_model
from triage_bench.domain.decision import Decision
from triage_bench.domain.pricing import cost_usd
from triage_bench.domain.task import Task
from triage_bench.domain.ticket import Ticket

CONTESTANT = "haiku"
MODEL_ID = "claude-haiku-4-5"
MS_PER_SECOND = 1000.0


class ClaudeDecider:
    name = CONTESTANT

    def __init__(self, client: anthropic.Anthropic, task: Task, model: str = MODEL_ID) -> None:
        self._client = client
        self._model = model
        self._system_prompt = llm_system_prompt(task)
        self._verdict_model = verdict_model(task)

    def decide(self, ticket: Ticket) -> Decision:
        started = time.perf_counter()
        try:
            response = self._client.messages.parse(
                model=self._model,
                max_tokens=LLM_MAX_TOKENS,
                system=self._system_prompt,
                messages=[{"role": "user", "content": ticket.text}],
                output_format=self._verdict_model,
            )
            verdict = response.parsed_output
            if verdict is None:
                raise ValueError("no parsed output in response")
            usage = response.usage
            latency_ms = (time.perf_counter() - started) * MS_PER_SECOND
            # Constructing Decision (which validates confidence/token ranges) must stay inside
            # this try block: malformed SDK output must become Decision.error, never a raise.
            # `verdict` is typed as a plain BaseModel (the class is built per task), so its
            # fields are read through model_dump rather than as attributes.
            fields = verdict.model_dump(mode="json")
            return Decision(
                ticket_id=ticket.id,
                contestant=CONTESTANT,
                label=verdict_label(verdict),
                confidence=float(fields["confidence"]),
                latency_ms=latency_ms,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=cost_usd(CONTESTANT, usage.input_tokens, usage.output_tokens),
            )
        except (anthropic.APIError, ValidationError, ValueError) as exc:
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
