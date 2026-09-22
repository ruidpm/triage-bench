import pytest

from triage_bench.domain.decision import Decision


def make(**overrides: object) -> Decision:
    base: dict[str, object] = dict(
        ticket_id=1, contestant="von", label="card_arrival", confidence=0.9,
        latency_ms=100.0, input_tokens=0, output_tokens=0, cost_usd=0.0,
    )
    base.update(overrides)
    return Decision(**base)  # type: ignore[arg-type]


def test_correct_when_label_matches() -> None:
    assert make().is_correct("card_arrival")
    assert not make().is_correct("exchange_rate")


def test_error_decision_is_never_correct() -> None:
    d = make(label=None, confidence=0.0, latency_ms=0.0, error="boom")
    assert d.is_error
    assert not d.is_correct("card_arrival")


@pytest.mark.parametrize("field,value", [
    ("confidence", 1.5), ("confidence", -0.1), ("latency_ms", -1.0),
    ("input_tokens", -1), ("cost_usd", -0.01),
])
def test_rejects_out_of_range(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        make(**{field: value})
