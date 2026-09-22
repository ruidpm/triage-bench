"""Structured-output schema shared by the LLM contestants."""

from enum import Enum

from pydantic import BaseModel, Field

from triage_bench.domain.ticket import LABELS

IntentLabel = Enum("IntentLabel", {label: label for label in LABELS})  # type: ignore[misc]


class Verdict(BaseModel):
    label: IntentLabel
    confidence: float = Field(ge=0.0, le=1.0)
