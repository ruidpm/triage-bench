import pytest
from pydantic import ValidationError

from triage_bench.application.verdict import IntentLabel, Verdict
from triage_bench.domain.ticket import LABELS


def test_labels_match_domain() -> None:
    assert tuple(m.value for m in IntentLabel) == LABELS


def test_rejects_confidence_above_one() -> None:
    with pytest.raises(ValidationError):
        Verdict(label=IntentLabel.card_arrival, confidence=1.2)


def test_schema_has_enum() -> None:
    schema = Verdict.model_json_schema()
    assert "card_arrival" in str(schema)
