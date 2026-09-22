import json
from pathlib import Path

import pytest

from triage_bench.domain.decision import Decision
from triage_bench.domain.ticket import Ticket
from triage_bench.infrastructure.jsonl_sink import JsonlSink, RunFileError, read_run

T = [Ticket(1, "card never came", "card_arrival")]
D = Decision(1, "von", "card_arrival", 0.9, 12.0, 0, 0, 0.0)


def test_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "run.jsonl"
    sink = JsonlSink(p)
    sink.write_header(T)
    sink.write(D)
    tickets, decisions = read_run(p)
    assert tickets == T and decisions == [D]
    assert json.loads(p.read_text().splitlines()[0])["kind"] == "tickets"


def test_malformed_line_reports_number(tmp_path: Path) -> None:
    p = tmp_path / "run.jsonl"
    p.write_text('{"kind":"tickets","tickets":[]}\nnot json\n')
    with pytest.raises(RunFileError, match="line 2"):
        read_run(p)


def test_missing_header(tmp_path: Path) -> None:
    p = tmp_path / "run.jsonl"
    p.write_text(json.dumps({"kind": "decision"}) + "\n")
    with pytest.raises(RunFileError, match="header"):
        read_run(p)
