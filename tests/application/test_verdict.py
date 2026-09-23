import pytest
from pydantic import ValidationError

from triage_bench.application.verdict import verdict_label, verdict_model
from triage_bench.domain.task import SENTIMENT, TASKS, TRIAGE, Task

TASK_CASES = pytest.mark.parametrize("task", TASKS.values(), ids=[t.name for t in TASKS.values()])


@TASK_CASES
def test_label_enum_matches_the_task(task: Task) -> None:
    model = verdict_model(task)
    schema = model.model_json_schema()
    assert task.label_names() == tuple(schema["$defs"][f"{task.name.capitalize()}Label"]["enum"])
    assert model.__name__ == f"{task.name.capitalize()}Verdict"


def test_accepts_a_task_label_and_reports_it_as_a_string() -> None:
    verdict = verdict_model(TRIAGE)(label="card_arrival", confidence=0.85)
    assert verdict_label(verdict) == "card_arrival"


def test_rejects_another_tasks_label() -> None:
    with pytest.raises(ValidationError):
        verdict_model(SENTIMENT)(label="card_arrival", confidence=0.5)


def test_rejects_confidence_above_one() -> None:
    with pytest.raises(ValidationError):
        verdict_model(SENTIMENT)(label="positive", confidence=1.2)


def test_label_read_from_an_unvalidated_instance_is_still_a_string() -> None:
    # Deciders may receive malformed parsed output; reading the label must not itself raise.
    model = verdict_model(TRIAGE)
    good = model(label="exchange_rate", confidence=0.5)
    unvalidated = model.model_construct(label=good.label, confidence=1.5)  # type: ignore[attr-defined]
    assert verdict_label(unvalidated) == "exchange_rate"
