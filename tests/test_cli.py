from datetime import datetime
from pathlib import Path

import pytest

from triage_bench import cli
from triage_bench.cli import (
    EXIT_CONFIG,
    EXIT_OK,
    SDK_MAX_RETRIES,
    build_claude_client,
    build_openai_client,
    default_run_path,
    format_routing,
    format_totals_table,
    main,
    missing_keys,
)
from triage_bench.domain.decision import Decision
from triage_bench.domain.metrics import Totals
from triage_bench.domain.routing import RoutingReport
from triage_bench.domain.ticket import Ticket
from triage_bench.infrastructure.jsonl_sink import JsonlSink


def test_missing_keys_names_each_missing_variable() -> None:
    assert missing_keys({}) == ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
    assert missing_keys({"ANTHROPIC_API_KEY": "x", "OPENAI_API_KEY": ""}) == ["OPENAI_API_KEY"]
    assert missing_keys({"ANTHROPIC_API_KEY": "x", "OPENAI_API_KEY": "y"}) == []


def test_totals_table_is_markdown() -> None:
    t = Totals("von", 200, 0, 0.885, 101.0, 180.0, 0.0, 0.061, 0.83)
    table = format_totals_table([t])
    assert table.splitlines()[0].startswith("| contestant |")
    assert "| von | 88.5% | 101 | 180 | $0.0000 | 0.061 |" in table


def test_routing_text_mentions_key_numbers() -> None:
    r = RoutingReport(0.8, "von", "haiku", 0.72, 0.91, 0.031, 0.93, 0.110, 0.72)
    text = format_routing(r)
    assert "72%" in text and "0.80" in text and "haiku" in text


def test_default_run_path_is_timestamped() -> None:
    expected = Path("runs/2026-09-22T10-05-00.jsonl")
    assert default_run_path(datetime(2026, 9, 22, 10, 5, 0)) == expected


def test_live_without_keys_exits_with_config_error(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["live"]) == EXIT_CONFIG
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_report_on_missing_file_exits_with_config_error(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["report", "--run", "runs/does-not-exist.jsonl"]) == EXIT_CONFIG
    assert "not found" in capsys.readouterr().err


def test_client_factories_disable_sdk_auto_retries(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # A rate-limit or 5xx hit must surface as a visible error, not inflated latency from the
    # SDK silently retrying with backoff inside our measured decide() call.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert SDK_MAX_RETRIES == 0
    assert build_claude_client().max_retries == SDK_MAX_RETRIES
    assert build_openai_client().max_retries == SDK_MAX_RETRIES


TICKETS = [Ticket(1, "card never came", "card_arrival"), Ticket(2, "rate?", "exchange_rate")]
FULL_RUN = [
    Decision(1, "von", "card_arrival", 0.9, 10.0, 0, 0, 0.0),
    Decision(1, "haiku", "card_arrival", 0.9, 400.0, 30, 10, 0.00008),
    Decision(2, "von", "exchange_rate", 0.9, 11.0, 0, 0, 0.0),
    Decision(2, "haiku", "exchange_rate", 0.9, 380.0, 30, 10, 0.00008),
]
# A live run stopped mid-ticket: ticket 2 has Von's decision but not Haiku's.
PARTIAL_RUN = FULL_RUN[:3]


def write_run(path: Path, decisions: list[Decision]) -> Path:
    sink = JsonlSink(path)
    sink.write_header(TICKETS)
    for decision in decisions:
        sink.write(decision)
    return path


def test_report_on_partial_run_uses_complete_tickets_and_notes_drop(  # type: ignore[no-untyped-def]
    tmp_path, capsys
) -> None:
    run_file = write_run(tmp_path / "partial.jsonl", PARTIAL_RUN)
    assert main(["report", "--run", str(run_file)]) == EXIT_OK
    out, err = capsys.readouterr()
    assert "| von | 100.0% |" in out and "Routing at threshold" in out
    assert "dropped 1 of 2 tickets" in err


def test_report_on_header_only_run_exits_with_config_error(  # type: ignore[no-untyped-def]
    tmp_path, capsys
) -> None:
    run_file = write_run(tmp_path / "empty.jsonl", [])
    assert main(["report", "--run", str(run_file)]) == EXIT_CONFIG
    assert "no decisions" in capsys.readouterr().err


def test_report_with_out_of_range_threshold_exits_with_config_error(  # type: ignore[no-untyped-def]
    tmp_path, capsys
) -> None:
    run_file = write_run(tmp_path / "full.jsonl", FULL_RUN)
    assert main(["report", "--run", str(run_file), "--threshold", "1.5"]) == EXIT_CONFIG
    assert "threshold" in capsys.readouterr().err


def test_replay_on_header_only_run_exits_with_config_error(  # type: ignore[no-untyped-def]
    tmp_path, capsys, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "_serve", lambda *args: pytest.fail("must not serve"))
    run_file = write_run(tmp_path / "empty.jsonl", [])
    assert main(["replay", "--run", str(run_file)]) == EXIT_CONFIG
    assert "no decisions" in capsys.readouterr().err


def test_replay_on_partial_run_streams_complete_tickets_only(  # type: ignore[no-untyped-def]
    tmp_path, capsys, monkeypatch
) -> None:
    served = {}

    def fake_serve(source, meta, host, port):  # type: ignore[no-untyped-def]
        served["ticks"] = [event.tick for event in source()]
        return EXIT_OK

    monkeypatch.setattr(cli, "_serve", fake_serve)
    run_file = write_run(tmp_path / "partial.jsonl", PARTIAL_RUN)
    assert main(["replay", "--run", str(run_file)]) == EXIT_OK
    assert served["ticks"] == [1]
    assert "dropped 1 of 2 tickets" in capsys.readouterr().err


def test_live_rejects_an_unknown_task(capsys) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SystemExit) as info:
        main(["live", "--task", "bogus"])
    assert info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
