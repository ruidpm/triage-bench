"""Accuracy, latency, calibration. Pure domain."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from triage_bench.domain.decision import Decision

ECE_BINS = 10
P50 = 50.0
P95 = 95.0


@dataclass(frozen=True)
class Bin:
    lower: float
    upper: float
    count: int
    avg_confidence: float
    accuracy: float


@dataclass(frozen=True)
class Totals:
    contestant: str
    count: int
    errors: int
    accuracy: float
    p50_ms: float
    p95_ms: float
    cost_usd: float
    ece: float
    avg_confidence: float


def _scorable(decisions: Sequence[Decision]) -> list[Decision]:
    return [d for d in decisions if not d.is_error]


def accuracy(decisions: Sequence[Decision], truth: Mapping[int, str]) -> float:
    if not decisions:
        return 0.0
    hits = sum(1 for d in decisions if d.is_correct(truth[d.ticket_id]))
    return hits / len(decisions)


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def _bin_index(confidence: float, bins: int) -> int:
    return min(int(confidence * bins), bins - 1)


def reliability_bins(
    decisions: Sequence[Decision], truth: Mapping[int, str], bins: int = ECE_BINS
) -> list[Bin]:
    buckets: list[list[Decision]] = [[] for _ in range(bins)]
    for d in _scorable(decisions):
        buckets[_bin_index(d.confidence, bins)].append(d)
    width = 1.0 / bins
    out: list[Bin] = []
    for i, bucket in enumerate(buckets):
        n = len(bucket)
        avg_conf = sum(d.confidence for d in bucket) / n if n else 0.0
        acc = accuracy(bucket, truth) if n else 0.0
        out.append(Bin(lower=i * width, upper=(i + 1) * width, count=n,
                       avg_confidence=avg_conf, accuracy=acc))
    return out


def expected_calibration_error(
    decisions: Sequence[Decision], truth: Mapping[int, str], bins: int = ECE_BINS
) -> float:
    total = len(_scorable(decisions))
    if total == 0:
        return 0.0
    return sum(
        (b.count / total) * abs(b.accuracy - b.avg_confidence)
        for b in reliability_bins(decisions, truth, bins)
        if b.count
    )


def summarise(
    contestant: str, decisions: Sequence[Decision], truth: Mapping[int, str]
) -> Totals:
    ok = _scorable(decisions)
    latencies = [d.latency_ms for d in ok]
    return Totals(
        contestant=contestant,
        count=len(decisions),
        errors=len(decisions) - len(ok),
        accuracy=accuracy(decisions, truth),
        p50_ms=percentile(latencies, P50),
        p95_ms=percentile(latencies, P95),
        cost_usd=sum(d.cost_usd for d in decisions),
        ece=expected_calibration_error(decisions, truth, bins=ECE_BINS),
        avg_confidence=(sum(d.confidence for d in ok) / len(ok)) if ok else 0.0,
    )
