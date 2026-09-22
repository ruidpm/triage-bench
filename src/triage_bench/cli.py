"""Command line entry point: live, replay, report."""

import argparse
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from triage_bench.application.runner import TickEvent
from triage_bench.domain.metrics import Totals, summarise
from triage_bench.domain.routing import DEFAULT_THRESHOLD, RoutingReport, confidence_gated_report
from triage_bench.infrastructure.csv_tickets import TicketLoadError, load_tickets
from triage_bench.infrastructure.jsonl_sink import JsonlSink, RunFileError, read_run

if TYPE_CHECKING:
    from triage_bench.web.app import EventSource, RunMeta

EXIT_OK = 0
EXIT_CONFIG = 2
REQUIRED_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
CONTESTANT_ORDER = ["von", "haiku", "luna"]
ROUTING_PRIMARY = "von"
ROUTING_FALLBACK = "haiku"
DEFAULT_TICKETS = Path("data/tickets.csv")
DEFAULT_SAMPLE_RUN = Path("runs/sample.jsonl")
RUNS_DIR = Path("runs")
RUN_TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%S"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def missing_keys(env: Mapping[str, str]) -> list[str]:
    return [key for key in REQUIRED_KEYS if not env.get(key)]


def default_run_path(now: datetime) -> Path:
    return RUNS_DIR / f"{now.strftime(RUN_TIMESTAMP_FORMAT)}.jsonl"


def format_totals_table(totals: Sequence[Totals]) -> str:
    header = "| contestant | accuracy | p50 ms | p95 ms | cost | ECE |\n|---|---|---|---|---|---|"
    rows = [
        f"| {t.contestant} | {t.accuracy:.1%} | {t.p50_ms:.0f} | {t.p95_ms:.0f} "
        f"| ${t.cost_usd:.4f} | {t.ece:.3f} |"
        for t in totals
    ]
    return "\n".join([header, *rows])


def format_routing(r: RoutingReport) -> str:
    return (
        f"Routing at threshold {r.threshold:.2f}: {r.primary} handled "
        f"{r.handled_locally_share:.0%} locally, rest to {r.fallback}.\n"
        f"  routed:   accuracy {r.routed_accuracy:.1%}, cost ${r.routed_cost_usd:.4f}\n"
        f"  {r.fallback} alone: accuracy {r.fallback_accuracy:.1%}, "
        f"cost ${r.fallback_cost_usd:.4f}\n"
        f"  cost saving: {r.cost_saving_share:.0%}"
    )


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return EXIT_CONFIG


def _serve(source_factory: "EventSource", meta: "RunMeta", host: str, port: int) -> int:
    import uvicorn

    from triage_bench.web.app import create_app

    uvicorn.run(create_app(source_factory, meta), host=host, port=port, log_level="warning")
    return EXIT_OK


def cmd_live(args: argparse.Namespace) -> int:
    from triage_bench.web.app import MODE_LIVE, RunMeta

    missing = missing_keys(os.environ)
    if missing:
        return _fail(f"missing environment variables: {', '.join(missing)} (see .env.example)")
    try:
        tickets = load_tickets(args.tickets)
    except TicketLoadError as exc:
        return _fail("bad tickets file:\n  " + "\n  ".join(exc.problems))

    import anthropic
    import openai

    from triage_bench.application.decider import Decider
    from triage_bench.application.runner import run
    from triage_bench.infrastructure.claude_decider import ClaudeDecider
    from triage_bench.infrastructure.openai_decider import OpenAIDecider
    from triage_bench.infrastructure.von_decider import VonDecider

    try:
        von = VonDecider.from_sdk()
        print("loading Von model (first run downloads ~1.5 GB)...", file=sys.stderr)
        von.warm_up()
    except Exception as exc:  # startup failure: report and exit, never run half a bench
        return _fail(f"Von failed to load: {exc}")

    deciders: list[Decider] = [
        von, ClaudeDecider(anthropic.Anthropic()), OpenAIDecider(openai.OpenAI())
    ]
    out_path = args.out or default_run_path(datetime.now())
    sink = JsonlSink(out_path)
    sink.write_header(tickets)
    print(f"writing run to {out_path}; open http://{args.host}:{args.port}", file=sys.stderr)

    def source() -> Iterator[TickEvent]:
        return run(tickets, deciders, sink)

    meta = RunMeta(mode=MODE_LIVE, run_name=out_path.stem, contestants=CONTESTANT_ORDER)
    return _serve(source, meta, args.host, args.port)


def cmd_replay(args: argparse.Namespace) -> int:
    from triage_bench.application.replay import replay
    from triage_bench.web.app import MODE_REPLAY, RunMeta

    try:
        tickets, decisions = read_run(args.run)
    except RunFileError as exc:
        return _fail(str(exc))
    present = [c for c in CONTESTANT_ORDER if any(d.contestant == c for d in decisions)]

    def source() -> Iterator[TickEvent]:
        return replay(tickets, decisions, present)

    print(f"replaying {args.run}; open http://{args.host}:{args.port}", file=sys.stderr)
    meta = RunMeta(mode=MODE_REPLAY, run_name=args.run.stem, contestants=present)
    return _serve(source, meta, args.host, args.port)


def cmd_report(args: argparse.Namespace) -> int:
    try:
        tickets, decisions = read_run(args.run)
    except RunFileError as exc:
        return _fail(str(exc))
    truth = {t.id: t.label for t in tickets}
    by_name = {c: [d for d in decisions if d.contestant == c] for c in CONTESTANT_ORDER}
    totals = [summarise(c, ds, truth) for c, ds in by_name.items() if ds]
    print(format_totals_table(totals))
    if by_name[ROUTING_PRIMARY] and by_name[ROUTING_FALLBACK]:
        print()
        print(format_routing(confidence_gated_report(
            by_name[ROUTING_PRIMARY], by_name[ROUTING_FALLBACK], truth, args.threshold)))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="triage-bench")
    sub = parser.add_subparsers(dest="command", required=True)

    live = sub.add_parser("live", help="run all contestants for real and serve the dashboard")
    live.add_argument("--tickets", type=Path, default=DEFAULT_TICKETS)
    live.add_argument("--out", type=Path, default=None)
    live.add_argument("--host", default=DEFAULT_HOST)
    live.add_argument("--port", type=int, default=DEFAULT_PORT)
    live.set_defaults(func=cmd_live)

    rep = sub.add_parser("replay", help="serve the dashboard from a saved run, no keys needed")
    rep.add_argument("--run", type=Path, default=DEFAULT_SAMPLE_RUN)
    rep.add_argument("--host", default=DEFAULT_HOST)
    rep.add_argument("--port", type=int, default=DEFAULT_PORT)
    rep.set_defaults(func=cmd_replay)

    report = sub.add_parser("report", help="print a markdown summary of a saved run")
    report.add_argument("--run", type=Path, default=DEFAULT_SAMPLE_RUN)
    report.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    report.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
