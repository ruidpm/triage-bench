import json
from dataclasses import asdict
from pathlib import Path

import pytest

from triage_bench.domain.decision import Decision
from triage_bench.domain.task import SENTIMENT, TRIAGE
from triage_bench.domain.ticket import Ticket
from triage_bench.infrastructure.jsonl_sink import JsonlSink, RunFileError, read_run

T = [Ticket(1, "card never came", "card_arrival")]
D = Decision(1, "von", "card_arrival", 0.9, 12.0, 0, 0, 0.0)


def _header(task: str = "triage", tickets: list[Ticket] = T) -> str:
    return json.dumps({"kind": "tickets", "task": task, "tickets": [asdict(t) for t in tickets]})


def test_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "run.jsonl"
    sink = JsonlSink(p)
    sink.write_header(TRIAGE, T)
    sink.write(D)
    run = read_run(p)
    assert run.task is TRIAGE and run.tickets == T and run.decisions == [D]
    header = json.loads(p.read_text().splitlines()[0])
    assert header["kind"] == "tickets" and header["task"] == "triage"


def test_sentiment_header_roundtrips(tmp_path: Path) -> None:
    p = tmp_path / "run.jsonl"
    JsonlSink(p).write_header(SENTIMENT, [Ticket(1, "Loved it.", "positive")])
    assert read_run(p).task is SENTIMENT


def test_header_without_task_reports_line_number(tmp_path: Path) -> None:
    p = tmp_path / "run.jsonl"
    p.write_text(json.dumps({"kind": "tickets", "tickets": []}) + "\n")
    with pytest.raises(RunFileError, match="line 1: .*no task"):
        read_run(p)


def test_header_with_unknown_task_reports_line_number(tmp_path: Path) -> None:
    p = tmp_path / "run.jsonl"
    p.write_text(_header(task="chess") + "\n")
    with pytest.raises(RunFileError, match="line 1: .*chess"):
        read_run(p)


def test_ticket_label_outside_the_task_reports_line_number(tmp_path: Path) -> None:
    p = tmp_path / "run.jsonl"
    p.write_text(_header(task="sentiment") + "\n")
    with pytest.raises(RunFileError, match="line 1: .*card_arrival.*sentiment"):
        read_run(p)


def test_malformed_line_reports_number(tmp_path: Path) -> None:
    p = tmp_path / "run.jsonl"
    p.write_text(_header(tickets=[]) + "\nnot json\n")
    with pytest.raises(RunFileError, match="line 2"):
        read_run(p)


def test_missing_header(tmp_path: Path) -> None:
    p = tmp_path / "run.jsonl"
    p.write_text(json.dumps({"kind": "decision"}) + "\n")
    with pytest.raises(RunFileError, match="header"):
        read_run(p)


def test_bad_ticket_value_reports_line_number(tmp_path: Path) -> None:
    p = tmp_path / "run.jsonl"
    p.write_text(json.dumps({"kind": "tickets", "task": "triage",
                             "tickets": [{"id": 1, "text": "x", "label": "nope"}]}) + "\n")
    with pytest.raises(RunFileError, match="line 1: .*nope"):
        read_run(p)


def test_bad_decision_value_reports_line_number(tmp_path: Path) -> None:
    p = tmp_path / "run.jsonl"
    bad = {"kind": "decision", **asdict(D), "confidence": 2.0}
    p.write_text(_header() + "\n" + json.dumps(bad) + "\n")
    with pytest.raises(RunFileError, match="line 2: .*confidence"):
        read_run(p)


def test_decision_missing_field_reports_line_number(tmp_path: Path) -> None:
    p = tmp_path / "run.jsonl"
    incomplete = {"kind": "decision", "ticket_id": 1, "contestant": "von"}
    p.write_text(_header() + "\n" + json.dumps(incomplete) + "\n")
    with pytest.raises(RunFileError, match="line 2: "):
        read_run(p)
