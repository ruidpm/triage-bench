"""Command line entry point: live, replay, report."""

import argparse
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from triage_bench.application.replay import CompleteRun, complete_tickets
from triage_bench.application.runner import TickEvent
from triage_bench.domain.metrics import Totals, summarise
from triage_bench.domain.routing import DEFAULT_THRESHOLD, RoutingReport, confidence_gated_report
from triage_bench.domain.task import DEFAULT_TASK, TASKS, Task
from triage_bench.infrastructure.csv_tickets import TicketLoadError, load_tickets
from triage_bench.infrastructure.jsonl_sink import JsonlSink, RunFileError, read_run

if TYPE_CHECKING:
    import anthropic
    import openai

    from triage_bench.web.app import EventSource, RunMeta

EXIT_OK = 0
EXIT_CONFIG = 2
REQUIRED_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
CONTESTANT_ORDER = ["von", "haiku", "luna"]
ROUTING_PRIMARY = "von"
ROUTING_FALLBACK = "haiku"
DATA_DIR = Path("data")
RUNS_DIR = Path("runs")
RUN_TIMESTAMP_FORMAT = "%Y-%m-%dT%H-%M-%S"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
# The SDKs' own retry-with-backoff on 429/5xx would otherwise run inside our measured
# decide() latency; disabling it means a rate limit surfaces as a visible error instead.
SDK_MAX_RETRIES = 0
# Routing shares (handled locally, cost saving) keep one decimal: with 200 items each item
# is 0.5%, so whole percents hide single routed items. web/static/app.js DIGITS.share mirrors it.
SHARE_DIGITS = 1


def missing_keys(env: Mapping[str, str]) -> list[str]:
    return [key for key in REQUIRED_KEYS if not env.get(key)]


def build_claude_client() -> "anthropic.Anthropic":
    import anthropic

    return anthropic.Anthropic(max_retries=SDK_MAX_RETRIES)


def build_openai_client() -> "openai.OpenAI":
    import openai

    return openai.OpenAI(max_retries=SDK_MAX_RETRIES)


def default_run_path(now: datetime) -> Path:
    return RUNS_DIR / f"{now.strftime(RUN_TIMESTAMP_FORMAT)}.jsonl"


def default_tickets_path(task: Task) -> Path:
    return DATA_DIR / f"{task.name}.csv"


def default_sample_run_path(task: Task) -> Path:
    return RUNS_DIR / f"{task.name}-sample.jsonl"


def _run_path(args: argparse.Namespace) -> Path:
    """--run if given, else the committed sample run for --task (default task otherwise)."""
    if args.run is not None:
        # args.run is already a Path (argparse parses --run with type=Path); cast because
        # argparse.Namespace attributes are typed Any.
        return cast(Path, args.run)
    return default_sample_run_path(TASKS[args.task or DEFAULT_TASK.name])


def _note_task_mismatch(requested: str | None, run_task: Task, path: Path) -> None:
    if requested is not None and requested != run_task.name:
        print(f"note: {path} is a {run_task.name} run; ignoring --task {requested}",
              file=sys.stderr)


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
        f"{r.handled_locally_share:.{SHARE_DIGITS}%} locally, rest to {r.fallback}.\n"
        f"  routed:   accuracy {r.routed_accuracy:.1%}, cost ${r.routed_cost_usd:.4f}\n"
        f"  {r.fallback} alone: accuracy {r.fallback_accuracy:.1%}, "
        f"cost ${r.fallback_cost_usd:.4f}\n"
        f"  cost saving: {r.cost_saving_share:.{SHARE_DIGITS}%}"
    )


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return EXIT_CONFIG


def _read_complete_run(path: Path) -> tuple[Task, CompleteRun]:
    """Read a run and trim it to tickets every present contestant answered.

    Raises RunFileError when the file is unreadable or has nothing to show.
    """
    run_file = read_run(path)
    present = [c for c in CONTESTANT_ORDER
               if any(d.contestant == c for d in run_file.decisions)]
    if not present:
        raise RunFileError(f"run file has no decisions: {path}")
    kept = complete_tickets(run_file.tickets, run_file.decisions, present)
    if not kept.tickets:
        raise RunFileError(f"no ticket in {path} was answered by every contestant ({present})")
    if kept.dropped:
        print(f"note: dropped {kept.dropped} of {len(run_file.tickets)} tickets not answered "
              f"by every contestant (partial run)", file=sys.stderr)
    return run_file.task, kept


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
    task = TASKS[args.task]
    tickets_path = args.tickets or default_tickets_path(task)
    try:
        tickets = load_tickets(tickets_path, task)
    except TicketLoadError as exc:
        return _fail("bad tickets file:\n  " + "\n  ".join(exc.problems))

    from triage_bench.application.decider import Decider
    from triage_bench.application.runner import run
    from triage_bench.infrastructure.claude_decider import ClaudeDecider
    from triage_bench.infrastructure.openai_decider import OpenAIDecider
    from triage_bench.infrastructure.von_decider import VonDecider

    try:
        von = VonDecider.from_sdk(task)
        print("loading Von model (first run downloads ~1.6 GB)...", file=sys.stderr)
        von.warm_up()
    except Exception as exc:  # startup failure: report and exit, never run half a bench
        return _fail(f"Von failed to load: {exc}")

    deciders: list[Decider] = [
        von, ClaudeDecider(build_claude_client(), task), OpenAIDecider(build_openai_client(), task)
    ]
    out_path = args.out or default_run_path(datetime.now())
    sink = JsonlSink(out_path)
    sink.write_header(task, tickets)
    print(f"writing run to {out_path}; open http://{args.host}:{args.port}", file=sys.stderr)

    def source() -> Iterator[TickEvent]:
        return run(tickets, deciders, sink)

    meta = RunMeta(mode=MODE_LIVE, run_name=out_path.stem, contestants=CONTESTANT_ORDER,
                   task=task.name, item_noun=task.item_noun)
    return _serve(source, meta, args.host, args.port)


def cmd_replay(args: argparse.Namespace) -> int:
    from triage_bench.application.replay import replay
    from triage_bench.web.app import MODE_REPLAY, RunMeta

    run_path = _run_path(args)
    try:
        task, run_data = _read_complete_run(run_path)
    except RunFileError as exc:
        return _fail(str(exc))
    _note_task_mismatch(args.task, task, run_path)
    present = [c for c in CONTESTANT_ORDER if any(d.contestant == c for d in run_data.decisions)]

    def source() -> Iterator[TickEvent]:
        return replay(run_data.tickets, run_data.decisions, present)

    print(f"replaying {run_path}; open http://{args.host}:{args.port}", file=sys.stderr)
    meta = RunMeta(mode=MODE_REPLAY, run_name=run_path.stem, contestants=present,
                   task=task.name, item_noun=task.item_noun)
    return _serve(source, meta, args.host, args.port)


def cmd_report(args: argparse.Namespace) -> int:
    run_path = _run_path(args)
    try:
        task, run_data = _read_complete_run(run_path)
    except RunFileError as exc:
        return _fail(str(exc))
    _note_task_mismatch(args.task, task, run_path)
    truth = {t.id: t.label for t in run_data.tickets}
    by_name = {c: [d for d in run_data.decisions if d.contestant == c] for c in CONTESTANT_ORDER}
    totals = [summarise(c, ds, truth) for c, ds in by_name.items() if ds]
    routing = None
    if by_name[ROUTING_PRIMARY] and by_name[ROUTING_FALLBACK]:
        try:
            routing = confidence_gated_report(
                by_name[ROUTING_PRIMARY], by_name[ROUTING_FALLBACK], truth, args.threshold)
        except ValueError as exc:
            return _fail(f"cannot compute routing for {run_path}: {exc}")
    print(format_totals_table(totals))
    if routing is not None:
        print()
        print(format_routing(routing))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="triage-bench")
    sub = parser.add_subparsers(dest="command", required=True)

    live = sub.add_parser("live", help="run all contestants for real and serve the dashboard")
    live.add_argument("--task", choices=sorted(TASKS), default=DEFAULT_TASK.name)
    live.add_argument("--tickets", type=Path, default=None)
    live.add_argument("--out", type=Path, default=None)
    live.add_argument("--host", default=DEFAULT_HOST)
    live.add_argument("--port", type=int, default=DEFAULT_PORT)
    live.set_defaults(func=cmd_live)

    rep = sub.add_parser("replay", help="serve the dashboard from a saved run, no keys needed")
    rep.add_argument("--task", choices=sorted(TASKS), default=None)
    rep.add_argument("--run", type=Path, default=None)
    rep.add_argument("--host", default=DEFAULT_HOST)
    rep.add_argument("--port", type=int, default=DEFAULT_PORT)
    rep.set_defaults(func=cmd_replay)

    report = sub.add_parser("report", help="print a markdown summary of a saved run")
    report.add_argument("--task", choices=sorted(TASKS), default=None)
    report.add_argument("--run", type=Path, default=None)
    report.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    report.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
