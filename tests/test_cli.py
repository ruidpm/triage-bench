from datetime import datetime
from pathlib import Path

from triage_bench.cli import (
    EXIT_CONFIG,
    SDK_MAX_RETRIES,
    build_claude_client,
    build_openai_client,
    default_run_path,
    format_routing,
    format_totals_table,
    main,
    missing_keys,
)
from triage_bench.domain.metrics import Totals
from triage_bench.domain.routing import RoutingReport


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
