from triage_bench.application.replay import replay
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
