"""Structured-output schema shared by the LLM contestants."""

from enum import Enum

from pydantic import BaseModel, Field

from triage_bench.domain.ticket import LABELS

# mypy cannot see members of an Enum built from a computed dict ("Second argument of Enum()
# must be ... literal"); building it from LABELS keeps one source of truth for the labels.
IntentLabel = Enum("IntentLabel", {label: label for label in LABELS})  # type: ignore[misc]


class Verdict(BaseModel):
    label: IntentLabel
    confidence: float = Field(ge=0.0, le=1.0)
