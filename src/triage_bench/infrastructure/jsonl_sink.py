"""Append-only JSONL run files: one header line of task + tickets, then one line per decision."""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from triage_bench.domain.decision import Decision
from triage_bench.domain.task import TASKS, Task
from triage_bench.domain.ticket import Ticket

KIND_TICKETS = "tickets"
KIND_DECISION = "decision"


class RunFileError(Exception):
    pass


@dataclass(frozen=True)
class RunFile:
    task: Task
    tickets: list[Ticket]
    decisions: list[Decision]


class JsonlSink:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path

    def write_header(self, task: Task, tickets: Sequence[Ticket]) -> None:
        self._append({"kind": KIND_TICKETS, "task": task.name,
                      "tickets": [asdict(t) for t in tickets]})

    def write(self, decision: Decision) -> None:
        self._append({"kind": KIND_DECISION, **asdict(decision)})

    def _append(self, record: dict[str, object]) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def _header_task(record: dict[str, object], number: int) -> Task:
    name = record.get("task")
    if name is None:
        raise RunFileError(f"line {number}: tickets header has no task")
    if not isinstance(name, str) or name not in TASKS:
        raise RunFileError(f"line {number}: unknown task {name!r} (known: {sorted(TASKS)})")
    return TASKS[name]


def read_run(path: Path) -> RunFile:
    if not path.exists():
        raise RunFileError(f"run file not found: {path}")
    task: Task | None = None
    tickets: list[Ticket] | None = None
    decisions: list[Decision] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RunFileError(f"line {number}: invalid JSON ({exc.msg})") from exc
        kind = record.pop("kind", None)
        if kind not in (KIND_TICKETS, KIND_DECISION):
            raise RunFileError(f"line {number}: unknown record kind {kind!r}")
        if kind == KIND_DECISION and tickets is None:
            raise RunFileError("run file has no tickets header line")
        try:
            if kind == KIND_TICKETS:
                task = _header_task(record, number)
                tickets = [Ticket(**t) for t in record["tickets"]]
                for ticket in tickets:
                    task.validate_label(ticket.label)
            else:
                decisions.append(Decision(**record))
        except (KeyError, TypeError, ValueError) as exc:
            raise RunFileError(f"line {number}: invalid {kind} record ({exc})") from exc
    if task is None or tickets is None:
        raise RunFileError("run file has no tickets header line")
    return RunFile(task=task, tickets=tickets, decisions=decisions)
