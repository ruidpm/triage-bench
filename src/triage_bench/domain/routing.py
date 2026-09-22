"""Confidence-gated routing: keep the primary's answer when confident, else fall back."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from triage_bench.domain.decision import Decision
from triage_bench.domain.metrics import accuracy

DEFAULT_THRESHOLD = 0.80


@dataclass(frozen=True)
class RoutingReport:
    threshold: float
    primary: str
    fallback: str
    handled_locally_share: float
    routed_accuracy: float
    routed_cost_usd: float
    fallback_accuracy: float
    fallback_cost_usd: float
    cost_saving_share: float


def confidence_gated_report(
    primary: Sequence[Decision],
    fallback: Sequence[Decision],
    truth: Mapping[int, str],
    threshold: float = DEFAULT_THRESHOLD,
) -> RoutingReport:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be within [0, 1], got {threshold}")
    by_primary = {d.ticket_id: d for d in primary}
    by_fallback = {d.ticket_id: d for d in fallback}
    if by_primary.keys() != by_fallback.keys():
        raise ValueError("primary and fallback cover different ticket ids")

    routed: list[Decision] = []
    local = 0
    for tid, p in by_primary.items():
        if not p.is_error and p.confidence >= threshold:
            routed.append(p)
            local += 1
        else:
            routed.append(by_fallback[tid])

    n = len(routed)
    routed_cost = sum(d.cost_usd for d in routed)
    fallback_cost = sum(d.cost_usd for d in fallback)
    saving = (fallback_cost - routed_cost) / fallback_cost if fallback_cost else 0.0
    return RoutingReport(
        threshold=threshold,
        primary=primary[0].contestant if primary else "",
        fallback=fallback[0].contestant if fallback else "",
        handled_locally_share=local / n if n else 0.0,
        routed_accuracy=accuracy(routed, truth),
        routed_cost_usd=routed_cost,
        fallback_accuracy=accuracy(fallback, truth),
        fallback_cost_usd=fallback_cost,
        cost_saving_share=saving,
    )
