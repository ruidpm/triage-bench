import pytest

from triage_bench.domain.decision import Decision
from triage_bench.domain.routing import confidence_gated_report


def d(tid: int, who: str, label: str | None, conf: float, cost: float, err: str | None = None
      ) -> Decision:
    return Decision(ticket_id=tid, contestant=who, label=label, confidence=conf, latency_ms=1.0,
                    input_tokens=0, output_tokens=0, cost_usd=cost, error=err)


TRUTH = {1: "card_arrival", 2: "card_arrival", 3: "exchange_rate", 4: "exchange_rate"}
FALLBACK = [d(i, "haiku", TRUTH[i], 0.9, 0.001) for i in TRUTH]  # always right, costs


def test_high_confidence_stays_local() -> None:
    primary = [d(1, "von", "card_arrival", 0.95, 0.0), d(2, "von", "exchange_rate", 0.95, 0.0),
               d(3, "von", "exchange_rate", 0.3, 0.0), d(4, "von", None, 0.0, 0.0, err="x")]
    r = confidence_gated_report(primary, FALLBACK, TRUTH, threshold=0.8)
    assert r.handled_locally_share == 0.5           # tickets 1 and 2
    assert r.routed_accuracy == pytest.approx(0.75)  # 2 wrong at 0.95, 3 & 4 via haiku
    assert r.routed_cost_usd == pytest.approx(0.002)
    assert r.fallback_accuracy == 1.0
    assert r.fallback_cost_usd == pytest.approx(0.004)
    assert r.cost_saving_share == pytest.approx(0.5)


def test_threshold_one_routes_everything() -> None:
    primary = [d(i, "von", TRUTH[i], 0.99, 0.0) for i in TRUTH]
    r = confidence_gated_report(primary, FALLBACK, TRUTH, threshold=1.0)
    assert r.handled_locally_share == 0.0


def test_threshold_zero_keeps_everything_except_errors() -> None:
    primary = [d(i, "von", TRUTH[i], 0.01, 0.0) for i in TRUTH]
    r = confidence_gated_report(primary, FALLBACK, TRUTH, threshold=0.0)
    assert r.handled_locally_share == 1.0


def test_mismatched_tickets_raise() -> None:
    with pytest.raises(ValueError, match="ticket ids"):
        confidence_gated_report(FALLBACK[:2], FALLBACK, TRUTH)


def test_bad_threshold_raises() -> None:
    with pytest.raises(ValueError, match="threshold"):
        confidence_gated_report(FALLBACK, FALLBACK, TRUTH, threshold=1.5)
