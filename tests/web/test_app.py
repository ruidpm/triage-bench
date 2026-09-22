import json
from collections.abc import Iterator

from fastapi.testclient import TestClient

from triage_bench.application.replay import replay
from triage_bench.application.runner import TickEvent
from triage_bench.domain.decision import Decision
from triage_bench.domain.ticket import Ticket
from triage_bench.web.app import RunMeta, create_app

T = [Ticket(1, "a", "card_arrival"), Ticket(2, "b", "exchange_rate")]
D = [Decision(1, "von", "card_arrival", 0.9, 10.0, 0, 0, 0.0),
     Decision(2, "von", "card_arrival", 0.6, 11.0, 0, 0, 0.0)]


def source() -> Iterator[TickEvent]:
    return replay(T, D, ["von"])


def make_client(mode: str = "replay") -> TestClient:
    meta = RunMeta(mode=mode, run_name="fixture", contestants=["von"])
    return TestClient(create_app(source, meta))


def test_meta() -> None:
    assert make_client().get("/api/meta").json() == {
        "mode": "replay", "run_name": "fixture", "contestants": ["von"]}


def test_stream_forwards_events_unchanged_then_done() -> None:
    with make_client().stream("GET", "/api/stream?speed=20") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())
    frames = [f for f in body.split("\n\n") if f.strip()]
    data_frames = [f for f in frames if f.startswith("data:")]
    assert len(data_frames) == 2
    first = json.loads(data_frames[0].removeprefix("data:"))
    assert first == next(source()).to_dict()
    assert frames[-1].startswith("event: done")


def test_index_served() -> None:
    response = make_client().get("/")
    assert response.status_code == 200 and "<html" in response.text.lower()
