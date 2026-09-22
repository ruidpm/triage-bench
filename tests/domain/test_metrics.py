import pytest

from triage_bench.domain.decision import Decision
from triage_bench.domain.metrics import (
    accuracy,
    expected_calibration_error,
    percentile,
    reliability_bins,
    summarise,
)


def d(tid: int, label: str | None, conf: float, ms: float = 100.0, err: str | None = None,
      cost: float = 0.0) -> Decision:
    return Decision(ticket_id=tid, contestant="von", label=label, confidence=conf,
                    latency_ms=ms, input_tokens=0, output_tokens=0, cost_usd=cost, error=err)


TRUTH = {1: "card_arrival", 2: "card_arrival", 3: "exchange_rate", 4: "exchange_rate"}


def test_accuracy_counts_errors_as_wrong() -> None:
    ds = [d(1, "card_arrival", 0.9), d(2, "exchange_rate", 0.9),
          d(3, None, 0.0, 0.0, err="timeout")]
    assert accuracy(ds, TRUTH) == pytest.approx(1 / 3)


def test_accuracy_empty_is_zero() -> None:
    assert accuracy([], TRUTH) == 0.0


def test_percentile_nearest_rank() -> None:
    assert percentile([10, 20, 30, 40], 50) == 20
    assert percentile([10, 20, 30, 40], 95) == 40
    assert percentile([], 50) == 0.0


def test_perfectly_calibrated_has_zero_ece() -> None:
    # two decisions at 0.5 confidence, one right one wrong -> bin acc 0.5 == conf 0.5
    ds = [d(1, "card_arrival", 0.5), d(2, "exchange_rate", 0.5)]
    assert expected_calibration_error(ds, TRUTH) == pytest.approx(0.0)


def test_overconfident_wrong_answers_have_high_ece() -> None:
    ds = [d(1, "exchange_rate", 0.95), d(2, "exchange_rate", 0.95)]
    assert expected_calibration_error(ds, TRUTH) == pytest.approx(0.95)


def test_ece_ignores_errors_and_handles_none() -> None:
    assert expected_calibration_error([d(1, None, 0.0, 0.0, err="x")], TRUTH) == 0.0


def test_reliability_bins_cover_unit_interval() -> None:
    bins = reliability_bins([d(1, "card_arrival", 0.05), d(3, "exchange_rate", 0.95)], TRUTH)
    assert len(bins) == 10
    assert bins[0].count == 1 and bins[0].accuracy == 1.0
    assert bins[-1].count == 1 and bins[-1].upper == 1.0
    assert sum(b.count for b in bins) == 2


def test_confidence_of_exactly_one_lands_in_last_bin() -> None:
    bins = reliability_bins([d(1, "card_arrival", 1.0)], TRUTH)
    assert bins[-1].count == 1


def test_summarise() -> None:
    ds = [d(1, "card_arrival", 0.9, 100, cost=0.01), d(2, "card_arrival", 0.8, 300, cost=0.01),
          d(3, None, 0.0, 0.0, err="boom")]
    t = summarise("von", ds, TRUTH)
    assert t.count == 3 and t.errors == 1
    assert t.accuracy == pytest.approx(2 / 3)
    assert t.p50_ms == 100 and t.p95_ms == 300
    assert t.cost_usd == pytest.approx(0.02)
    assert t.avg_confidence == pytest.approx(0.85)
