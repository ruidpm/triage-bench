from triage_bench.application.replay import complete_tickets, replay
from triage_bench.domain.decision import Decision
from triage_bench.domain.ticket import Ticket

T = [Ticket(1, "a", "card_arrival"), Ticket(2, "b", "exchange_rate")]
D = [
    Decision(1, "von", "card_arrival", 0.9, 10.0, 0, 0, 0.0),
    Decision(1, "haiku", "exchange_rate", 0.7, 400.0, 30, 10, 0.00008),
    Decision(2, "von", "exchange_rate", 0.8, 11.0, 0, 0, 0.0),
    Decision(2, "haiku", "exchange_rate", 0.9, 380.0, 30, 10, 0.00008),
]


def test_replay_rebuilds_events_in_order() -> None:
    events = list(replay(T, D, ["von", "haiku"]))
    assert [e.tick for e in events] == [1, 2]
    assert events[0].decisions["haiku"].label == "exchange_rate"
    assert events[1].totals["von"].accuracy == 1.0
    assert events[1].totals["haiku"].accuracy == 0.5


def test_complete_tickets_keeps_only_tickets_every_contestant_answered() -> None:
    # A live run stopped mid-ticket leaves ticket 2 with only Von's decision.
    partial = D[:3]
    kept = complete_tickets(T, partial, ["von", "haiku"])
    assert kept.tickets == [T[0]]
    assert kept.decisions == D[:2]
    assert kept.dropped == 1


def test_complete_tickets_leaves_a_full_run_unchanged() -> None:
    kept = complete_tickets(T, D, ["von", "haiku"])
    assert kept.tickets == T and kept.decisions == D and kept.dropped == 0


def test_replay_of_trimmed_partial_run_does_not_crash() -> None:
    kept = complete_tickets(T, D[:3], ["von", "haiku"])
    assert [e.tick for e in replay(kept.tickets, kept.decisions, ["von", "haiku"])] == [1]
