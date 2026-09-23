from pathlib import Path

import pytest

from triage_bench.domain.task import SENTIMENT, TRIAGE
from triage_bench.infrastructure.csv_tickets import TicketLoadError, load_tickets


def write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "t.csv"
    p.write_text(body, encoding="utf-8")
    return p


def test_loads_valid_rows(tmp_path: Path) -> None:
    p = write(tmp_path, 'id,text,label\n7,"Where is my card, it has been 2 weeks",card_arrival\n')
    tickets = load_tickets(p, TRIAGE)
    assert len(tickets) == 1
    assert tickets[0].id == 7 and tickets[0].label == "card_arrival"


def test_reports_every_bad_row(tmp_path: Path) -> None:
    p = write(tmp_path, "id,text,label\n1,hi,pizza\nx,hello,card_arrival\n3,ok,exchange_rate\n")
    with pytest.raises(TicketLoadError) as info:
        load_tickets(p, TRIAGE)
    assert len(info.value.problems) == 2
    assert info.value.problems[0].startswith("row 2:")


def test_label_from_another_task_is_a_bad_row(tmp_path: Path) -> None:
    p = write(tmp_path, "id,text,label\n1,my card never came,card_arrival\n")
    with pytest.raises(TicketLoadError) as info:
        load_tickets(p, SENTIMENT)
    assert info.value.problems == ["row 2: unknown label 'card_arrival' for task 'sentiment'"]


def test_wrong_header_rejected(tmp_path: Path) -> None:
    with pytest.raises(TicketLoadError, match="header"):
        load_tickets(write(tmp_path, "text,label\nhi,card_arrival\n"), TRIAGE)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(TicketLoadError, match="not found"):
        load_tickets(tmp_path / "nope.csv", TRIAGE)
