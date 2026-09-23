import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from triage_bench.application.replay import replay
from triage_bench.application.runner import TickEvent
from triage_bench.domain.decision import Decision
from triage_bench.domain.ticket import Ticket
from triage_bench.web.app import MAX_SPEED_TPS, RunMeta, create_app

T = [Ticket(1, "a", "card_arrival"), Ticket(2, "b", "exchange_rate")]
D = [Decision(1, "von", "card_arrival", 0.9, 10.0, 0, 0, 0.0),
     Decision(2, "von", "card_arrival", 0.6, 11.0, 0, 0, 0.0)]


def source() -> Iterator[TickEvent]:
    return replay(T, D, ["von"])


def make_client(mode: str = "replay") -> TestClient:
    meta = RunMeta(mode=mode, run_name="fixture", contestants=["von"],
                   task="triage", item_noun="ticket")
    return TestClient(create_app(source, meta))


def test_meta() -> None:
    assert make_client().get("/api/meta").json() == {
        "mode": "replay", "run_name": "fixture", "contestants": ["von"],
        "task": "triage", "item_noun": "ticket", "max_speed_tps": MAX_SPEED_TPS}


def test_meta_max_speed_is_accepted_by_the_stream() -> None:
    # The dashboard streams replays at this speed and paces them itself.
    client = make_client()
    max_speed = client.get("/api/meta").json()["max_speed_tps"]
    with client.stream("GET", f"/api/stream?speed={max_speed}") as response:
        assert response.status_code == 200
        "".join(response.iter_text())


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


def test_live_mode_allows_only_one_stream_per_process() -> None:
    # A reload or second tab must not start a second paid run into the same run file.
    client = make_client(mode="live")
    with client.stream("GET", "/api/stream") as first:
        assert first.status_code == 200
        "".join(first.iter_text())
    second = client.get("/api/stream")
    assert second.status_code == 409
    assert "restart" in second.json()["detail"]


def test_replay_mode_allows_repeated_streams() -> None:
    client = make_client()
    for _ in range(2):
        with client.stream("GET", "/api/stream?speed=20") as response:
            assert response.status_code == 200
            "".join(response.iter_text())


class SourceBrokeError(RuntimeError):
    pass


def failing_source() -> Iterator[TickEvent]:
    yield next(source())
    raise SourceBrokeError("decider crashed mid-run")


def test_source_exception_mid_stream_is_not_swallowed() -> None:
    # The stream must not end with a normal "done" frame, which would hide the failure.
    meta = RunMeta(mode="replay", run_name="fixture", contestants=["von"],
                   task="triage", item_noun="ticket")
    client = TestClient(create_app(failing_source, meta))
    with pytest.raises(SourceBrokeError), client.stream("GET", "/api/stream?speed=20") as response:
        "".join(response.iter_text())


@pytest.mark.parametrize("module", ["app.js", "pacer.js", "latency-points.js"])
def test_dashboard_modules_are_served_as_javascript(module: str) -> None:
    # Browsers refuse to run an ES module served with a non-JavaScript MIME type.
    response = make_client().get(f"/static/{module}")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
