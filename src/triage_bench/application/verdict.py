"""Structured-output schema shared by the LLM contestants, one model per task."""

from enum import Enum

from pydantic import BaseModel, Field, create_model

from triage_bench.domain.task import Task

CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0


def verdict_model(task: Task) -> type[BaseModel]:
    """A `{label: <enum of the task's labels>, confidence: 0..1}` model for structured output."""
    prefix = task.name.capitalize()
    # mypy cannot see members of an Enum built from a computed dict ("Second argument of Enum()
    # must be ... literal"); building it from the task keeps one source of truth for the labels.
    label_enum = Enum(f"{prefix}Label", {name: name for name in task.label_names()})  # type: ignore[misc]
    return create_model(
        f"{prefix}Verdict",
        label=(label_enum, ...),
        confidence=(float, Field(ge=CONFIDENCE_MIN, le=CONFIDENCE_MAX)),
    )


def verdict_label(verdict: BaseModel) -> str:
    """The chosen label as a plain string, whichever task's enum it belongs to."""
    label = verdict.model_dump(mode="json")["label"]
    if not isinstance(label, str):
        raise ValueError(f"verdict label is not a string: {label!r}")
    return label
