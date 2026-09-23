# Sentiment Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second benchmark task — binary sentiment of Amazon product reviews — to triage-bench without changing how the triage task behaves or what its published run means.

**Architecture:** A typed `Task` value (name, item noun, label noun, instructions, labels) enters the domain; the triage intents move into it verbatim. Prompt, verdict schema, the three deciders, the CSV loader and the run-file reader stop importing triage constants and take a `Task` instead. The CLI gains `--task`, derives data and sample-run paths from it, and the run file records its task in the header. The dashboard swaps the word "ticket" for the task's noun from `/api/meta` and otherwise does not change.

**Tech Stack:** Python 3.13 via `uv`; `pydantic` (`create_model`); `fastapi`; `pytest`, `ruff`, `mypy --strict`; vanilla ES-module JS with `node --test`. No new dependency.

Spec: `docs/superpowers/specs/2026-09-23-sentiment-task-design.md`. Read it before starting. The triage spec `docs/superpowers/specs/2026-09-22-triage-bench-design.md` still holds.

## Global Constraints

- Work on branch `sentiment-task` (already exists, based on `main`). Never commit to `main`.
- Run everything through `uv run` from `/Users/rui/projects/triage-bench`. Before every commit: `uv run pytest && uv run ruff check . && uv run mypy src && node --test 'tests/web/js/*.test.mjs'` — all green.
- TDD per task: write the failing test, run it and see it fail, write the minimum code, see it pass, run the full suite, commit. Production code without a test that would fail in its absence is deleted.
- Layering: `domain/` imports only stdlib (and `pydantic` where already used). `application/` imports `domain`. `infrastructure/` and `web/` import both. Nothing in `domain/` or `application/` touches files, network, `von`, `anthropic`, `openai` or `fastapi`.
- Task names are exactly `"triage"` and `"sentiment"`. Item nouns `"ticket"` and `"review"`. Label nouns `"intent"` and `"sentiment"`. Sentiment labels exactly `"negative"`, `"positive"`, with descriptions `The customer is unhappy with the product.` / `The customer is happy with the product.` and instructions `Classify the sentiment of this customer product review.`
- The LLM system prompt for the triage task must stay **byte-identical** to what produced the committed run (Task 2 pins it in a test).
- Contestant keys, model IDs, prices, `LLM_MAX_TOKENS = 64`, fairness rules and the routing report are unchanged.
- No magic numbers or strings: thresholds, page sizes, seeds, paths and nouns are named constants.
- No silent `except`. Errors are values (`Decision.error`, `TicketLoadError`, `RunFileError`) or exit with context.
- Data files: `data/triage.csv` (renamed from `data/tickets.csv`), `data/sentiment.csv`. Run files: `runs/triage-sample.jsonl` (renamed from `runs/sample.jsonl`), `runs/sentiment-sample.jsonl`. Both whitelisted in `.gitignore`; every other `runs/*` stays ignored.
- Secrets: API keys only via `uv run --env-file .env`. Never read, print or grep `.env` (the user's hook blocks it anyway). Before committing any run file: `grep -c 'sk-' <file>` must print `0`.
- Commit messages: imperative subject under 72 characters, body says why, and end with the trailer line `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Do not touch anything the task list below does not name. No "while I was here" changes.

## File Structure

```
src/triage_bench/
  domain/task.py                 NEW  Label, Task, TRIAGE, SENTIMENT, TASKS, DEFAULT_TASK, MIN_LABELS
  domain/ticket.py               MOD  Ticket only; Intent/INTENTS/LABELS and label validation removed
  application/prompt.py          MOD  label_descriptions_text(task), llm_system_prompt(task)
  application/verdict.py         MOD  verdict_model(task), verdict_label(verdict); IntentLabel/Verdict removed
  infrastructure/von_decider.py  MOD  VonDecider(engine, task), from_sdk(task)
  infrastructure/claude_decider.py MOD ClaudeDecider(client, task, model=...)
  infrastructure/openai_decider.py MOD OpenAIDecider(client, task, model=...)
  infrastructure/csv_tickets.py  MOD  load_tickets(path, task)
  infrastructure/jsonl_sink.py   MOD  header carries task; read_run -> RunFile(task, tickets, decisions)
  cli.py                         MOD  --task, default paths per task, task in meta, mismatch note
  web/app.py                     MOD  RunMeta gains task, item_noun
  web/static/item-copy.js        NEW  itemCopy(noun): every dashboard string that names the item
  web/static/index.html          MOD  ids on the elements whose text names the item; task badge
  web/static/app.js              MOD  uses itemCopy from meta.item_noun
scripts/
  sample_banking77.py            MOD  labels from TRIAGE; writes data/triage.csv
  sample_amazon_polarity.py      NEW  stdlib sampler -> data/sentiment.csv
  smoke_live.py                  MOD  takes an optional task name
data/triage.csv                  RENAMED from data/tickets.csv (content unchanged)
data/sentiment.csv               NEW  200 reviews, 100 per label
runs/triage-sample.jsonl         RENAMED from runs/sample.jsonl; header gains "task": "triage"
runs/sentiment-sample.jsonl      NEW  one real live run
docs/screenshots/sentiment/dashboard-finished.png  NEW
tests/
  domain/test_task.py            NEW
  domain/test_ticket.py          MOD
  application/test_prompt.py     MOD   parametrised over TASKS + shipped-prompt pin
  application/test_verdict.py    MOD
  infrastructure/test_von_decider.py, test_claude_decider.py, test_openai_decider.py  MOD
  infrastructure/test_csv_tickets.py, test_jsonl_sink.py  MOD
  test_cli.py, test_data_file.py, web/test_app.py  MOD
  web/js/item-copy.test.mjs      NEW
README.md, pyproject.toml (description), .gitignore  MOD
```

---

### Task 1: The `Task` domain value

**Files:**
- Create: `src/triage_bench/domain/task.py`
- Test: `tests/domain/test_task.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `Label(name: str, description: str)`; `Task(name, item_noun, label_noun, instructions, labels: tuple[Label, ...])` with `label_names() -> tuple[str, ...]`, `choices() -> dict[str, str]`, `validate_label(label: str) -> None` (raises `ValueError`); constants `TRIAGE`, `SENTIMENT`, `TASKS: dict[str, Task]`, `DEFAULT_TASK` (= `TRIAGE`), `MIN_LABELS = 2`.

The spec's `Task.validate(ticket)` is implemented as `validate_label(label: str)` so `task.py` never imports `ticket.py` (which would be circular once nothing else holds the label list). Callers pass `ticket.label`.

- [ ] **Step 1: Write the failing tests**

`tests/domain/test_task.py`:

```python
import pytest

from triage_bench.domain.task import (
    DEFAULT_TASK,
    MIN_LABELS,
    SENTIMENT,
    TASKS,
    TRIAGE,
    Label,
    Task,
)

TWO_LABELS = (Label("a", "first"), Label("b", "second"))


def make_task(**overrides: object) -> Task:
    fields: dict[str, object] = {
        "name": "t", "item_noun": "item", "label_noun": "kind",
        "instructions": "do it", "labels": TWO_LABELS,
    }
    fields.update(overrides)
    return Task(**fields)  # type: ignore[arg-type]


def test_triage_keeps_the_eight_shipped_intents_in_order() -> None:
    assert TRIAGE.name == "triage" and TRIAGE.item_noun == "ticket"
    assert TRIAGE.label_noun == "intent"
    assert TRIAGE.instructions == "Classify the primary intent of this customer support message."
    assert TRIAGE.label_names() == (
        "lost_or_stolen_card", "card_arrival", "declined_card_payment",
        "refund_not_showing_up", "exchange_rate", "top_up_failed",
        "passcode_forgotten", "terminate_account",
    )
    assert TRIAGE.choices()["card_arrival"] == (
        "Asking when or whether an ordered card will arrive")


def test_sentiment_is_binary_over_reviews() -> None:
    assert SENTIMENT.name == "sentiment" and SENTIMENT.item_noun == "review"
    assert SENTIMENT.label_noun == "sentiment"
    assert SENTIMENT.instructions == "Classify the sentiment of this customer product review."
    assert SENTIMENT.choices() == {
        "negative": "The customer is unhappy with the product.",
        "positive": "The customer is happy with the product.",
    }


def test_registry_is_keyed_by_task_name_and_defaults_to_triage() -> None:
    assert TASKS == {"triage": TRIAGE, "sentiment": SENTIMENT}
    assert DEFAULT_TASK is TRIAGE


def test_validate_label_accepts_known_and_names_unknown() -> None:
    task = make_task()
    task.validate_label("a")
    with pytest.raises(ValueError, match="unknown label 'zzz' for task 't'"):
        task.validate_label("zzz")


@pytest.mark.parametrize("labels", [(), (Label("a", "first"),)])
def test_task_needs_at_least_two_labels(labels: tuple[Label, ...]) -> None:
    with pytest.raises(ValueError, match=f"at least {MIN_LABELS}"):
        make_task(labels=labels)


def test_task_rejects_duplicate_label_names() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        make_task(labels=(Label("a", "x"), Label("a", "y")))


@pytest.mark.parametrize("field", ["name", "item_noun", "label_noun", "instructions"])
def test_task_rejects_blank_text_fields(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        make_task(**{field: "   "})


@pytest.mark.parametrize(("name", "description"), [("", "x"), ("a", " ")])
def test_label_rejects_blank_name_or_description(name: str, description: str) -> None:
    with pytest.raises(ValueError, match="blank"):
        Label(name, description)
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `uv run pytest tests/domain/test_task.py -v`
Expected: every test errors with `ModuleNotFoundError: No module named 'triage_bench.domain.task'`.

- [ ] **Step 3: Write the module**

`src/triage_bench/domain/task.py`:

```python
"""Tasks: what the contestants are asked and the label set they answer from. Pure domain."""

from dataclasses import dataclass

MIN_LABELS = 2


@dataclass(frozen=True)
class Label:
    name: str
    description: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("label name is blank")
        if not self.description.strip():
            raise ValueError(f"label {self.name!r} has a blank description")


@dataclass(frozen=True)
class Task:
    name: str          # CLI value, data/run file stem, run-file header
    item_noun: str     # "ticket" / "review": dashboard copy only
    label_noun: str    # "intent" / "sentiment": wording inside the LLM system prompt
    instructions: str
    labels: tuple[Label, ...]

    def __post_init__(self) -> None:
        for field in ("name", "item_noun", "label_noun", "instructions"):
            if not getattr(self, field).strip():
                raise ValueError(f"task {self.name!r}: {field} is blank")
        if len(self.labels) < MIN_LABELS:
            raise ValueError(f"task {self.name!r} needs at least {MIN_LABELS} labels")
        names = self.label_names()
        if len(set(names)) != len(names):
            raise ValueError(f"task {self.name!r} has duplicate label names: {names}")

    def label_names(self) -> tuple[str, ...]:
        return tuple(label.name for label in self.labels)

    def choices(self) -> dict[str, str]:
        return {label.name: label.description for label in self.labels}

    def validate_label(self, label: str) -> None:
        if label not in self.label_names():
            raise ValueError(f"unknown label {label!r} for task {self.name!r}")


TRIAGE = Task(
    name="triage",
    item_noun="ticket",
    label_noun="intent",
    instructions="Classify the primary intent of this customer support message.",
    labels=(
        Label("lost_or_stolen_card", "Card was lost, stolen, or the customer wants it blocked"),
        Label("card_arrival", "Asking when or whether an ordered card will arrive"),
        Label("declined_card_payment", "A card payment was declined or refused at checkout"),
        Label("refund_not_showing_up",
              "A refund was promised but has not appeared in the account"),
        Label("exchange_rate", "Questions about exchange rates or currency conversion applied"),
        Label("top_up_failed", "Adding money to the account failed or did not go through"),
        Label("passcode_forgotten", "Forgot the app passcode or PIN and wants to reset it"),
        Label("terminate_account", "Wants to close or delete their account"),
    ),
)

SENTIMENT = Task(
    name="sentiment",
    item_noun="review",
    label_noun="sentiment",
    instructions="Classify the sentiment of this customer product review.",
    labels=(
        Label("negative", "The customer is unhappy with the product."),
        Label("positive", "The customer is happy with the product."),
    ),
)

TASKS: dict[str, Task] = {task.name: task for task in (TRIAGE, SENTIMENT)}
DEFAULT_TASK = TRIAGE
```

The eight descriptions are copied character for character from `src/triage_bench/domain/ticket.py` `INTENTS`. Diff them (`grep -o '"[^"]*"' ` on both) before moving on; the shipped prompt depends on them.

- [ ] **Step 4: Run the tests and watch them pass**

Run: `uv run pytest tests/domain/test_task.py -v`
Expected: all PASS.

- [ ] **Step 5: Full suite, lint, types, commit**

```bash
uv run pytest && uv run ruff check . && uv run mypy src
git add src/triage_bench/domain/task.py tests/domain/test_task.py
git commit -m "Add the Task domain value with triage and sentiment

A task carries its own label set, wording and dashboard noun so a second
benchmark task can exist without branching on intents everywhere.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Prompt, verdict schema and deciders built per task

**Files:**
- Modify: `src/triage_bench/application/prompt.py`
- Modify: `src/triage_bench/application/verdict.py`
- Modify: `src/triage_bench/infrastructure/von_decider.py`
- Modify: `src/triage_bench/infrastructure/claude_decider.py`
- Modify: `src/triage_bench/infrastructure/openai_decider.py`
- Modify: `src/triage_bench/cli.py` (`cmd_live` decider construction; `--task` on `live`)
- Modify: `scripts/smoke_live.py`
- Test: `tests/application/test_prompt.py`, `tests/application/test_verdict.py`, `tests/infrastructure/test_von_decider.py`, `tests/infrastructure/test_claude_decider.py`, `tests/infrastructure/test_openai_decider.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `Task`, `TASKS`, `TRIAGE`, `DEFAULT_TASK` from Task 1.
- Produces: `label_descriptions_text(task: Task) -> str`; `llm_system_prompt(task: Task) -> str`; `verdict_model(task: Task) -> type[pydantic.BaseModel]`; `verdict_label(verdict: BaseModel) -> str`; `VonDecider(engine, task)`, `VonDecider.from_sdk(task)`; `ClaudeDecider(client, task, model=MODEL_ID)`; `OpenAIDecider(client, task, model=MODEL_ID)`; `live --task {triage,sentiment}`.

Deciders never read `ticket.label`, so the triage-labelled `TICKET` fixture is valid input for a sentiment decider too. That is why these tests can be parametrised over both tasks before Task 5 relaxes `Ticket`.

- [ ] **Step 1: Write the failing prompt tests**

Replace `tests/application/test_prompt.py`:

```python
import pytest

from triage_bench.application.prompt import (
    LLM_MAX_TOKENS,
    label_descriptions_text,
    llm_system_prompt,
)
from triage_bench.domain.task import TASKS, TRIAGE, Task

# The exact system prompt that produced runs/triage-sample.jsonl. It must never change.
SHIPPED_TRIAGE_PROMPT = (
    "Classify the primary intent of this customer support message.\n\n"
    "Intents:\n"
    "lost_or_stolen_card: Card was lost, stolen, or the customer wants it blocked\n"
    "card_arrival: Asking when or whether an ordered card will arrive\n"
    "declined_card_payment: A card payment was declined or refused at checkout\n"
    "refund_not_showing_up: A refund was promised but has not appeared in the account\n"
    "exchange_rate: Questions about exchange rates or currency conversion applied\n"
    "top_up_failed: Adding money to the account failed or did not go through\n"
    "passcode_forgotten: Forgot the app passcode or PIN and wants to reset it\n"
    "terminate_account: Wants to close or delete their account\n\n"
    "Answer with exactly one intent label and your confidence that it is correct, "
    "as a number from 0 to 1."
)

TASK_CASES = pytest.mark.parametrize("task", TASKS.values(), ids=[t.name for t in TASKS.values()])


@TASK_CASES
def test_descriptions_list_every_label_once(task: Task) -> None:
    text = label_descriptions_text(task)
    for label in task.labels:
        assert text.count(f"{label.name}: {label.description}") == 1


@TASK_CASES
def test_system_prompt_contains_instructions_labels_and_noun(task: Task) -> None:
    prompt = llm_system_prompt(task)
    assert task.instructions in prompt
    assert label_descriptions_text(task) in prompt
    assert f"exactly one {task.label_noun} label" in prompt
    assert "confidence" in prompt


def test_triage_prompt_is_byte_identical_to_the_shipped_run() -> None:
    assert llm_system_prompt(TRIAGE) == SHIPPED_TRIAGE_PROMPT


def test_token_cap() -> None:
    assert LLM_MAX_TOKENS == 64
```

- [ ] **Step 2: Run and watch them fail**

Run: `uv run pytest tests/application/test_prompt.py -v`
Expected: FAIL — `label_descriptions_text() takes 0 positional arguments but 1 was given` (and the import of `Task` works, since Task 1 landed).

- [ ] **Step 3: Rewrite prompt.py**

```python
"""The instruction and label wording shared by every contestant, built from a Task."""

from triage_bench.domain.task import Task

LLM_MAX_TOKENS = 64


def label_descriptions_text(task: Task) -> str:
    return "\n".join(f"{label.name}: {label.description}" for label in task.labels)


def llm_system_prompt(task: Task) -> str:
    heading = f"{task.label_noun.capitalize()}s"
    return (
        f"{task.instructions}\n\n{heading}:\n{label_descriptions_text(task)}\n\n"
        f"Answer with exactly one {task.label_noun} label and your confidence that it is "
        "correct, as a number from 0 to 1."
    )
```

Run: `uv run pytest tests/application/test_prompt.py -v` → PASS (the byte-identical test is the one that matters; if it fails, a description in `TRIAGE` differs from the original — fix `task.py`, not the test).

- [ ] **Step 4: Write the failing verdict tests**

Replace `tests/application/test_verdict.py`:

```python
import pytest
from pydantic import ValidationError

from triage_bench.application.verdict import verdict_label, verdict_model
from triage_bench.domain.task import SENTIMENT, TASKS, TRIAGE, Task

TASK_CASES = pytest.mark.parametrize("task", TASKS.values(), ids=[t.name for t in TASKS.values()])


@TASK_CASES
def test_label_enum_matches_the_task(task: Task) -> None:
    model = verdict_model(task)
    schema = model.model_json_schema()
    assert task.label_names() == tuple(schema["$defs"][f"{task.name.capitalize()}Label"]["enum"])
    assert model.__name__ == f"{task.name.capitalize()}Verdict"


def test_accepts_a_task_label_and_reports_it_as_a_string() -> None:
    verdict = verdict_model(TRIAGE)(label="card_arrival", confidence=0.85)
    assert verdict_label(verdict) == "card_arrival"


def test_rejects_another_tasks_label() -> None:
    with pytest.raises(ValidationError):
        verdict_model(SENTIMENT)(label="card_arrival", confidence=0.5)


def test_rejects_confidence_above_one() -> None:
    with pytest.raises(ValidationError):
        verdict_model(SENTIMENT)(label="positive", confidence=1.2)


def test_label_read_from_an_unvalidated_instance_is_still_a_string() -> None:
    # Deciders may receive malformed parsed output; reading the label must not itself raise.
    model = verdict_model(TRIAGE)
    good = model(label="exchange_rate", confidence=0.5)
    unvalidated = model.model_construct(label=good.label, confidence=1.5)  # type: ignore[attr-defined]
    assert verdict_label(unvalidated) == "exchange_rate"
```

Run: `uv run pytest tests/application/test_verdict.py -v` → FAIL with `ImportError: cannot import name 'verdict_label'`.

- [ ] **Step 5: Rewrite verdict.py**

```python
"""Structured-output schema shared by the LLM contestants, one model per task."""

from enum import Enum

from pydantic import BaseModel, Field, create_model

from triage_bench.domain.task import Task

CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0


def verdict_model(task: Task) -> type[BaseModel]:
    """A `{label: <enum of the task's labels>, confidence: 0..1}` model for structured output."""
    prefix = task.name.capitalize()
    # mypy cannot see members of an Enum built from a computed dict ("Second argument of Enum()
    # must be ... literal"); building it from the task keeps one source of truth for the labels.
    label_enum = Enum(f"{prefix}Label", {name: name for name in task.label_names()})  # type: ignore[misc]
    return create_model(
        f"{prefix}Verdict",
        label=(label_enum, ...),
        confidence=(float, Field(ge=CONFIDENCE_MIN, le=CONFIDENCE_MAX)),
    )


def verdict_label(verdict: BaseModel) -> str:
    """The chosen label as a plain string, whichever task's enum it belongs to."""
    label = verdict.model_dump(mode="json")["label"]
    if not isinstance(label, str):
        raise ValueError(f"verdict label is not a string: {label!r}")
    return label
```

Run: `uv run pytest tests/application/test_verdict.py -v` → PASS. Run `uv run mypy src` now: it will still complain about the deciders (fixed next).

- [ ] **Step 6: Write the failing Von decider tests**

Replace the imports, fixture and first test in `tests/infrastructure/test_von_decider.py`; the three error-path tests stay but pass `TRIAGE`:

```python
from dataclasses import dataclass

import pytest

from triage_bench.domain.task import TASKS, TRIAGE, Task
from triage_bench.domain.ticket import Ticket
from triage_bench.infrastructure.von_decider import VonDecider

TICKET = Ticket(1, "my card still has not arrived", "card_arrival")
TASK_CASES = pytest.mark.parametrize("task", TASKS.values(), ids=[t.name for t in TASKS.values()])


@dataclass
class FakeResult:
    choice: str
    probabilities: dict[str, float]


class FakeEngine:
    def __init__(self, result: FakeResult | Exception) -> None:
        self.result, self.calls = result, []

    def decide(self, state: str, choices: dict[str, str], instructions: str) -> FakeResult:
        self.calls.append((state, choices, instructions))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@TASK_CASES
def test_passes_the_tasks_choices_and_instructions(task: Task) -> None:
    first = task.label_names()[0]
    engine = FakeEngine(FakeResult(first, {first: 0.93}))
    d = VonDecider(engine, task).decide(TICKET)
    assert d.contestant == "von" and d.label == first
    assert d.confidence == 0.93 and d.cost_usd == 0.0 and d.latency_ms > 0
    state, choices, instructions = engine.calls[0]
    assert state == TICKET.text and instructions == task.instructions
    assert choices == task.choices()


def test_warm_up_uses_the_same_choices_and_instructions() -> None:
    engine = FakeEngine(FakeResult("card_arrival", {"card_arrival": 1.0}))
    VonDecider(engine, TRIAGE).warm_up()
    _, choices, instructions = engine.calls[0]
    assert choices == TRIAGE.choices() and instructions == TRIAGE.instructions


def test_engine_exception_becomes_error_decision() -> None:
    d = VonDecider(FakeEngine(RuntimeError("mps out of memory")), TRIAGE).decide(TICKET)
    assert d.is_error and "mps out of memory" in (d.error or "")
    assert d.label is None and d.latency_ms == 0.0


def test_missing_choice_in_probabilities_becomes_error_decision() -> None:
    engine = FakeEngine(FakeResult("card_arrival", {"exchange_rate": 1.0}))
    d = VonDecider(engine, TRIAGE).decide(TICKET)
    assert d.is_error and d.label is None and d.latency_ms == 0.0
    assert "card_arrival" in (d.error or "")


def test_confidence_slightly_above_one_becomes_error_decision() -> None:
    engine = FakeEngine(FakeResult("card_arrival", {"card_arrival": 1.0000000002}))
    d = VonDecider(engine, TRIAGE).decide(TICKET)
    assert d.is_error and d.label is None and d.latency_ms == 0.0
```

Run: `uv run pytest tests/infrastructure/test_von_decider.py -v` → FAIL (`__init__() takes 2 positional arguments but 3 were given`).

- [ ] **Step 7: Update von_decider.py**

Replace the imports and the class head; `decide()`'s body only changes the `instructions=` argument:

```python
"""Von, the local System One model, behind the Decider protocol."""

import time
from typing import Protocol

from triage_bench.domain.decision import Decision
from triage_bench.domain.task import Task
from triage_bench.domain.ticket import Ticket

CONTESTANT = "von"
WARM_UP_TEXT = "warm up"
MS_PER_SECOND = 1000.0


class VonResult(Protocol):
    choice: str
    probabilities: dict[str, float]


class VonEngine(Protocol):
    def decide(self, state: str, choices: dict[str, str], instructions: str) -> VonResult: ...


class VonDecider:
    name = CONTESTANT

    def __init__(self, engine: VonEngine, task: Task) -> None:
        self._engine = engine
        self._instructions = task.instructions
        self._choices = task.choices()

    @classmethod
    def from_sdk(cls, task: Task) -> "VonDecider":
        import von  # local model; imported here so tests never load it

        return cls(von, task)

    def warm_up(self) -> None:
        self._engine.decide(
            state=WARM_UP_TEXT, choices=self._choices, instructions=self._instructions
        )

    def decide(self, ticket: Ticket) -> Decision:
        started = time.perf_counter()
        try:
            result = self._engine.decide(
                state=ticket.text, choices=self._choices, instructions=self._instructions
            )
        except Exception as exc:  # adapters must not raise; the runner needs every tick
            return self._error_decision(ticket.id, str(exc))
        # ... the rest of decide() and _error_decision() are unchanged ...
```

Run: `uv run pytest tests/infrastructure/test_von_decider.py -v` → PASS.

- [ ] **Step 8: Write the failing Claude decider tests**

In `tests/infrastructure/test_claude_decider.py` replace the imports, `TICKET`, `good_response`, and the tests that mention `Verdict`/`IntentLabel`/`llm_system_prompt()`; the fakes and the API-error / missing-parsed-output tests keep their bodies but construct `ClaudeDecider(client, TRIAGE)`:

```python
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from triage_bench.application.prompt import LLM_MAX_TOKENS, llm_system_prompt
from triage_bench.application.verdict import verdict_model
from triage_bench.domain.task import TASKS, TRIAGE, Task
from triage_bench.domain.ticket import Ticket
from triage_bench.infrastructure.claude_decider import MODEL_ID, ClaudeDecider

TICKET = Ticket(1, "my card still has not arrived", "card_arrival")
TRIAGE_VERDICT = verdict_model(TRIAGE)
TASK_CASES = pytest.mark.parametrize("task", TASKS.values(), ids=[t.name for t in TASKS.values()])


class FakeMessages:
    def __init__(self, outcome: object) -> None:
        self.outcome, self.kwargs = outcome, None

    def parse(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def client_with(outcome: object) -> tuple[object, FakeMessages]:
    messages = FakeMessages(outcome)
    return SimpleNamespace(messages=messages), messages


def good_response() -> object:
    return SimpleNamespace(
        parsed_output=TRIAGE_VERDICT(label="card_arrival", confidence=0.85),
        usage=SimpleNamespace(input_tokens=120, output_tokens=15),
    )


@TASK_CASES
def test_builds_request_per_fairness_rules(task: Task) -> None:
    client, messages = client_with(good_response())
    ClaudeDecider(client, task).decide(TICKET)  # type: ignore[arg-type]
    assert messages.kwargs is not None
    assert messages.kwargs["model"] == MODEL_ID == "claude-haiku-4-5"
    assert messages.kwargs["max_tokens"] == LLM_MAX_TOKENS
    assert messages.kwargs["system"] == llm_system_prompt(task)
    assert messages.kwargs["messages"] == [{"role": "user", "content": TICKET.text}]
    schema = messages.kwargs["output_format"].model_json_schema()  # type: ignore[attr-defined]
    assert schema == verdict_model(task).model_json_schema()
    assert "thinking" not in messages.kwargs


def test_maps_parsed_output_tokens_and_cost() -> None:
    client, _ = client_with(good_response())
    d = ClaudeDecider(client, TRIAGE).decide(TICKET)  # type: ignore[arg-type]
    assert d.contestant == "haiku" and d.label == "card_arrival" and d.confidence == 0.85
    assert d.input_tokens == 120 and d.output_tokens == 15
    assert d.cost_usd == (120 * 1.00 + 15 * 5.00) / 1_000_000


def test_api_error_becomes_error_decision() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request)
    err = anthropic.RateLimitError("rate limited", response=response, body=None)
    client, _ = client_with(err)
    d = ClaudeDecider(client, TRIAGE).decide(TICKET)  # type: ignore[arg-type]
    assert d.is_error and "rate limited" in (d.error or "")


def test_missing_parsed_output_is_error() -> None:
    client, _ = client_with(
        SimpleNamespace(parsed_output=None, usage=SimpleNamespace(input_tokens=1, output_tokens=1))
    )
    d = ClaudeDecider(client, TRIAGE).decide(TICKET)  # type: ignore[arg-type]
    assert d.is_error and "no parsed output" in (d.error or "")


def test_out_of_range_confidence_becomes_error_decision_not_a_raise() -> None:
    # Mapping the response into a Decision (which validates on construction) must happen inside
    # decide()'s try block, otherwise decide() would raise and break the Decider contract.
    good = TRIAGE_VERDICT(label="card_arrival", confidence=0.5)
    bad_verdict = TRIAGE_VERDICT.model_construct(label=good.label, confidence=1.5)  # type: ignore[attr-defined]
    usage = SimpleNamespace(input_tokens=1, output_tokens=1)
    client, _ = client_with(SimpleNamespace(parsed_output=bad_verdict, usage=usage))
    d = ClaudeDecider(client, TRIAGE).decide(TICKET)  # type: ignore[arg-type]
    assert d.is_error and "confidence" in (d.error or "")


def test_negative_usage_tokens_becomes_error_decision_not_a_raise() -> None:
    # Same reasoning: a response with a negative usage field must not escape decide() as a
    # raised ValueError from Decision's own validation.
    client, _ = client_with(
        SimpleNamespace(
            parsed_output=TRIAGE_VERDICT(label="card_arrival", confidence=0.5),
            usage=SimpleNamespace(input_tokens=-1, output_tokens=1),
        )
    )
    d = ClaudeDecider(client, TRIAGE).decide(TICKET)  # type: ignore[arg-type]
    assert d.is_error and "input_tokens" in (d.error or "")
```

Note the current file imports `httpx2 as httpx` on line 4; replace that with `import httpx` as shown.

Run: `uv run pytest tests/infrastructure/test_claude_decider.py -v` → FAIL.

- [ ] **Step 9: Update claude_decider.py**

```python
"""Claude Haiku 4.5 behind the Decider protocol. One structured-output call, no thinking."""

import time

import anthropic
from pydantic import ValidationError

from triage_bench.application.prompt import LLM_MAX_TOKENS, llm_system_prompt
from triage_bench.application.verdict import verdict_label, verdict_model
from triage_bench.domain.decision import Decision
from triage_bench.domain.pricing import cost_usd
from triage_bench.domain.task import Task
from triage_bench.domain.ticket import Ticket

CONTESTANT = "haiku"
MODEL_ID = "claude-haiku-4-5"
MS_PER_SECOND = 1000.0


class ClaudeDecider:
    name = CONTESTANT

    def __init__(self, client: anthropic.Anthropic, task: Task, model: str = MODEL_ID) -> None:
        self._client = client
        self._model = model
        self._system_prompt = llm_system_prompt(task)
        self._verdict_model = verdict_model(task)

    def decide(self, ticket: Ticket) -> Decision:
        started = time.perf_counter()
        try:
            response = self._client.messages.parse(
                model=self._model,
                max_tokens=LLM_MAX_TOKENS,
                system=self._system_prompt,
                messages=[{"role": "user", "content": ticket.text}],
                output_format=self._verdict_model,
            )
            verdict = response.parsed_output
            if verdict is None:
                raise ValueError("no parsed output in response")
            usage = response.usage
            latency_ms = (time.perf_counter() - started) * MS_PER_SECOND
            # Constructing Decision (which validates confidence/token ranges) must stay inside
            # this try block: malformed SDK output must become Decision.error, never a raise.
            # `verdict` is typed as a plain BaseModel (the class is built per task), so its
            # fields are read through model_dump rather than as attributes.
            fields = verdict.model_dump(mode="json")
            return Decision(
                ticket_id=ticket.id,
                contestant=CONTESTANT,
                label=verdict_label(verdict),
                confidence=float(fields["confidence"]),
                latency_ms=latency_ms,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=cost_usd(CONTESTANT, usage.input_tokens, usage.output_tokens),
            )
```

`_error_decision` and the `except (anthropic.APIError, ValidationError, ValueError)` clause are unchanged.

Run: `uv run pytest tests/infrastructure/test_claude_decider.py -v` → PASS.

- [ ] **Step 10: OpenAI decider, same shape**

Update `tests/infrastructure/test_openai_decider.py` exactly as in Step 8: import `verdict_model`, `TASKS`, `TRIAGE`, `Task`; `TRIAGE_VERDICT = verdict_model(TRIAGE)`; `good_response()` uses `TRIAGE_VERDICT(label="card_arrival", confidence=0.7)`; every `OpenAIDecider(client)` becomes `OpenAIDecider(client, TRIAGE)`; `test_builds_request_per_fairness_rules` is parametrised with `TASK_CASES`, asserts `k["input"][0] == {"role": "system", "content": llm_system_prompt(task)}` and `k["text_format"].model_json_schema() == verdict_model(task).model_json_schema()`; the two "not a raise" tests build `bad_verdict` with `model_construct(label=good.label, confidence=1.5)` as in Step 8.

Run the file → FAIL. Then in `src/triage_bench/infrastructure/openai_decider.py`: import `verdict_label, verdict_model` and `Task`; `__init__(self, client, task: Task, model: str = MODEL_ID)` stores `self._system_prompt = llm_system_prompt(task)` and `self._verdict_model = verdict_model(task)`; `input=[{"role": "system", "content": self._system_prompt}, ...]`, `text_format=self._verdict_model`; after the usage check read `fields = verdict.model_dump(mode="json")` and build the Decision with `label=verdict_label(verdict), confidence=float(fields["confidence"])`. Everything else unchanged.

Run the file → PASS.

- [ ] **Step 11: CLI builds deciders for a task; `live --task`**

Add to `tests/test_cli.py`:

```python
def test_live_rejects_an_unknown_task(capsys) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SystemExit) as info:
        main(["live", "--task", "bogus"])
    assert info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
```

Run: `uv run pytest tests/test_cli.py::test_live_rejects_an_unknown_task -v` → FAIL (`unrecognized arguments: --task bogus`).

In `src/triage_bench/cli.py`: add `from triage_bench.domain.task import DEFAULT_TASK, TASKS` to the imports. In `cmd_live`, right after the missing-keys check, add `task = TASKS[args.task]`, then build `von = VonDecider.from_sdk(task)` and `deciders: list[Decider] = [von, ClaudeDecider(build_claude_client(), task), OpenAIDecider(build_openai_client(), task)]`. In `build_parser`, before `live.add_argument("--tickets", ...)`:

```python
    live.add_argument("--task", choices=sorted(TASKS), default=DEFAULT_TASK.name)
```

Run the test → PASS.

- [ ] **Step 12: smoke_live.py takes a task name**

In `scripts/smoke_live.py` add `from triage_bench.domain.task import DEFAULT_TASK, TASKS`, and in `main()`:

```python
    task = TASKS[sys.argv[1]] if len(sys.argv) > 1 else DEFAULT_TASK
    tickets = load_tickets(Path("data/tickets.csv"))[:SMOKE_TICKETS]
    von = VonDecider.from_sdk(task)
    von.warm_up()
    deciders = [von, ClaudeDecider(build_claude_client(), task),
                OpenAIDecider(build_openai_client(), task)]
```

Update its docstring usage line to `uv run --env-file .env python scripts/smoke_live.py [triage|sentiment]`. (The tickets path is fixed in Task 3.)

- [ ] **Step 13: Full suite, lint, types, commit**

```bash
uv run pytest && uv run ruff check . && uv run mypy src
git add -A src tests scripts
git commit -m "Build prompts, verdict schemas and deciders per task

The LLM system prompt, the structured-output enum and Von's choices now
come from a Task instead of module constants. The triage prompt is pinned
byte for byte so the committed run stays reproducible.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: CSV loader validates labels against a task

**Files:**
- Modify: `src/triage_bench/infrastructure/csv_tickets.py`
- Modify: `src/triage_bench/cli.py` (`cmd_live` call site)
- Modify: `scripts/smoke_live.py`
- Test: `tests/infrastructure/test_csv_tickets.py`

**Interfaces:**
- Produces: `load_tickets(path: Path, task: Task) -> list[Ticket]`.

- [ ] **Step 1: Write the failing tests**

Replace `tests/infrastructure/test_csv_tickets.py`:

```python
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
```

Run: `uv run pytest tests/infrastructure/test_csv_tickets.py -v` → FAIL (`load_tickets() takes 1 positional argument`).

- [ ] **Step 2: Update the loader**

In `csv_tickets.py`, import `from triage_bench.domain.task import Task`, change the signature to `def load_tickets(path: Path, task: Task) -> list[Ticket]:` and the row loop body to:

```python
            try:
                ticket_id, text, label = row
                ticket = Ticket(id=int(ticket_id), text=text, label=label)
                task.validate_label(ticket.label)
                tickets.append(ticket)
            except ValueError as exc:
                problems.append(f"row {row_number}: {exc}")
```

`test_label_from_another_task_is_a_bad_row` will still fail at this point because `Ticket` itself rejects `card_arrival`? No — `card_arrival` is a valid triage label and `Ticket` still validates against the triage list, so `Ticket(...)` succeeds and `task.validate_label` raises with the expected message. Confirm: run the file → all PASS.

- [ ] **Step 3: Call sites**

`cli.py` `cmd_live`: `task = TASKS[args.task]` (added in Task 2) already sits above the tickets load; change the load to `tickets = load_tickets(args.tickets, task)`. `scripts/smoke_live.py`: `tickets = load_tickets(Path("data/tickets.csv"), task)[:SMOKE_TICKETS]`.

- [ ] **Step 4: Full suite, lint, types, commit**

```bash
uv run pytest && uv run ruff check . && uv run mypy src
git add -A src tests scripts
git commit -m "Validate CSV labels against the task being loaded

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Run files record their task

**Files:**
- Modify: `src/triage_bench/infrastructure/jsonl_sink.py`
- Modify: `src/triage_bench/cli.py` (`_read_complete_run`, `cmd_live`, `cmd_replay`, `cmd_report`)
- Modify: `runs/sample.jsonl` (header line only)
- Test: `tests/infrastructure/test_jsonl_sink.py`, `tests/test_cli.py`

**Interfaces:**
- Produces: `JsonlSink.write_header(task: Task, tickets)`; `RunFile(task: Task, tickets: list[Ticket], decisions: list[Decision])`; `read_run(path) -> RunFile`; header record `{"kind": "tickets", "task": "<name>", "tickets": [...]}`.

- [ ] **Step 1: Write the failing tests**

Replace `tests/infrastructure/test_jsonl_sink.py`:

```python
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


def test_sentiment_header_roundtrips(tmp_path: Path) -> None:
    p = tmp_path / "run.jsonl"
    JsonlSink(p).write_header(SENTIMENT, [Ticket(1, "Loved it.", "positive")])
    assert read_run(p).task is SENTIMENT


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
```

Note `test_ticket_label_outside_the_task_reports_line_number`: `T` has label `card_arrival`, which `Ticket` (still triage-validating until Task 5) accepts, so the task check is what fires.

Run: `uv run pytest tests/infrastructure/test_jsonl_sink.py -v` → FAIL.

- [ ] **Step 2: Update the sink and reader**

`jsonl_sink.py`:

```python
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
```

Run the file → PASS.

- [ ] **Step 3: CLI call sites and tests**

In `tests/test_cli.py`: `from triage_bench.domain.task import TRIAGE` and in `write_run` use `sink.write_header(TRIAGE, TICKETS)`. Run `uv run pytest tests/test_cli.py` → FAIL on the run-reading tests (tuple unpacking). In `cli.py`:

```python
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
```

Import `Task` alongside `DEFAULT_TASK, TASKS`. `cmd_replay` and `cmd_report`: `task, run_data = _read_complete_run(args.run)` (the `task` is used in Tasks 6 and 7; until then keep it as `_task, run_data` so ruff does not flag an unused name). `cmd_live`: `sink.write_header(task, tickets)`.

Run `uv run pytest tests/test_cli.py` → PASS.

- [ ] **Step 4: Add the task to the committed run's header**

The committed run predates the field. Regenerate only its first line, keeping every other line byte-identical:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
path = Path("runs/sample.jsonl")
lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
header = json.loads(lines[0])
assert header["kind"] == "tickets" and "task" not in header, header.keys()
ordered = {"kind": header["kind"], "task": "triage", "tickets": header["tickets"]}
lines[0] = json.dumps(ordered) + "\n"
path.write_text("".join(lines), encoding="utf-8")
print("header now:", lines[0][:60])
PY
git diff --stat runs/sample.jsonl     # exactly 1 insertion, 1 deletion
uv run triage-bench report            # still prints the README table
```

- [ ] **Step 5: Full suite, lint, types, commit**

```bash
uv run pytest && uv run ruff check . && uv run mypy src
git add -A src tests runs/sample.jsonl
git commit -m "Record the task in run-file headers

A replay or report must know which label set a run used; the committed
triage run gets the field added to its header, decisions untouched.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: `Ticket` stops owning the label list

**Files:**
- Modify: `src/triage_bench/domain/ticket.py`
- Modify: `scripts/sample_banking77.py`
- Test: `tests/domain/test_ticket.py`, `tests/test_data_file.py`, `tests/infrastructure/test_csv_tickets.py`

After Tasks 2–4 nothing in `src/` imports `Intent`, `INTENTS` or `LABELS` from `ticket.py`. Verify: `grep -rn "INTENTS\|LABELS\|Intent\b" src scripts tests` should list only `ticket.py`, `test_ticket.py`, `test_data_file.py`, `sample_banking77.py`.

- [ ] **Step 1: Write the failing tests**

Replace `tests/domain/test_ticket.py`:

```python
import pytest

from triage_bench.domain.ticket import Ticket


def test_ticket_keeps_its_fields() -> None:
    ticket = Ticket(id=1, text="My card never came", label="card_arrival")
    assert (ticket.id, ticket.text, ticket.label) == (1, "My card never came", "card_arrival")


def test_ticket_does_not_judge_labels() -> None:
    # Which labels are valid is the task's business (domain.task); a ticket only checks its shape.
    assert Ticket(id=1, text="Loved it.", label="positive").label == "positive"


def test_ticket_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match="blank"):
        Ticket(id=1, text="   ", label="card_arrival")
```

Add to `tests/infrastructure/test_csv_tickets.py`:

```python
def test_loads_sentiment_rows_for_the_sentiment_task(tmp_path: Path) -> None:
    p = write(tmp_path, 'id,text,label\n5,"Great fit\n\nOrdered a medium, fits well.",positive\n')
    tickets = load_tickets(p, SENTIMENT)
    assert tickets[0].label == "positive" and "\n\n" in tickets[0].text
```

Run: `uv run pytest tests/domain/test_ticket.py tests/infrastructure/test_csv_tickets.py -v` → the two new tests FAIL with `unknown label 'positive'`.

- [ ] **Step 2: Slim down ticket.py**

```python
"""Tickets: one labelled text to classify. Pure domain, no I/O.

Which labels are valid is the task's business (domain.task); a Ticket only checks its own shape.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Ticket:
    id: int
    text: str
    label: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("ticket text is blank")
```

Run the two files → PASS.

- [ ] **Step 3: Fix the remaining importers**

`tests/test_data_file.py`: replace `from triage_bench.domain.ticket import LABELS` with `from triage_bench.domain.task import TRIAGE`, add `LABELS = TRIAGE.label_names()` after the imports, and call `load_tickets(Path("data/tickets.csv"), TRIAGE)` in both tests.

`scripts/sample_banking77.py`: replace `from triage_bench.domain.ticket import LABELS` with `from triage_bench.domain.task import TRIAGE` and add `LABELS = TRIAGE.label_names()` below the imports.

- [ ] **Step 4: Full suite, lint, types, commit**

```bash
uv run pytest && uv run ruff check . && uv run mypy src
git add -A src tests scripts
git commit -m "Move label ownership from Ticket to Task

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: `/api/meta` names the task and its item noun

**Files:**
- Modify: `src/triage_bench/web/app.py`
- Modify: `src/triage_bench/cli.py` (`cmd_live`, `cmd_replay`)
- Test: `tests/web/test_app.py`

**Interfaces:**
- Produces: `RunMeta(mode: str, run_name: str, contestants: list[str], task: str, item_noun: str)`; `/api/meta` JSON gains `"task"` and `"item_noun"`.

- [ ] **Step 1: Write the failing test**

In `tests/web/test_app.py` change `make_client` and `test_meta`, and the `RunMeta(...)` inside `test_source_exception_mid_stream_is_not_swallowed`:

```python
def make_client(mode: str = "replay") -> TestClient:
    meta = RunMeta(mode=mode, run_name="fixture", contestants=["von"],
                   task="triage", item_noun="ticket")
    return TestClient(create_app(source, meta))


def test_meta() -> None:
    assert make_client().get("/api/meta").json() == {
        "mode": "replay", "run_name": "fixture", "contestants": ["von"],
        "task": "triage", "item_noun": "ticket", "max_speed_tps": MAX_SPEED_TPS}
```

Run: `uv run pytest tests/web/test_app.py -v` → FAIL (`unexpected keyword argument 'task'`).

- [ ] **Step 2: Extend RunMeta**

In `web/app.py`:

```python
@dataclass(frozen=True)
class RunMeta:
    mode: str
    run_name: str
    contestants: list[str]
    task: str        # Task.name, shown beside the mode badge
    item_noun: str   # Task.item_noun, replaces "ticket" in the dashboard copy
```

Run the file → PASS.

- [ ] **Step 3: CLI passes the task**

`cmd_live`: `meta = RunMeta(mode=MODE_LIVE, run_name=out_path.stem, contestants=CONTESTANT_ORDER, task=task.name, item_noun=task.item_noun)`. `cmd_replay`: rename `_task` back to `task` and `meta = RunMeta(mode=MODE_REPLAY, run_name=args.run.stem, contestants=present, task=task.name, item_noun=task.item_noun)`.

- [ ] **Step 4: Full suite, lint, types, commit**

```bash
uv run pytest && uv run ruff check . && uv run mypy src
git add -A src tests
git commit -m "Expose the task and its item noun in run metadata

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: `--task` everywhere, files named by task

**Files:**
- Rename: `data/tickets.csv` → `data/triage.csv`; `runs/sample.jsonl` → `runs/triage-sample.jsonl`
- Modify: `.gitignore`, `src/triage_bench/cli.py`, `scripts/sample_banking77.py`, `scripts/smoke_live.py`
- Test: `tests/test_cli.py`, `tests/test_data_file.py`

**Interfaces:**
- Produces: `default_tickets_path(task: Task) -> Path` (`data/<name>.csv`); `default_sample_run_path(task: Task) -> Path` (`runs/<name>-sample.jsonl`); `replay --task`, `report --task` choose the default `--run`; a note on stderr when `--task` disagrees with the run file.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py` (import `SENTIMENT` alongside `TRIAGE`, and `default_sample_run_path, default_tickets_path` from `triage_bench.cli`):

```python
def test_default_paths_are_named_by_task() -> None:
    assert default_tickets_path(TRIAGE) == Path("data/triage.csv")
    assert default_tickets_path(SENTIMENT) == Path("data/sentiment.csv")
    assert default_sample_run_path(TRIAGE) == Path("runs/triage-sample.jsonl")
    assert default_sample_run_path(SENTIMENT) == Path("runs/sentiment-sample.jsonl")


def test_report_notes_when_task_flag_disagrees_with_the_run_file(  # type: ignore[no-untyped-def]
    tmp_path, capsys
) -> None:
    run_file = tmp_path / "s.jsonl"
    sink = JsonlSink(run_file)
    sink.write_header(SENTIMENT, [Ticket(1, "Loved it.", "positive")])
    sink.write(Decision(1, "von", "positive", 0.9, 10.0, 0, 0, 0.0))
    assert main(["report", "--task", "triage", "--run", str(run_file)]) == EXIT_OK
    out, err = capsys.readouterr()
    assert "| von | 100.0% |" in out
    assert "is a sentiment run; ignoring --task triage" in err


def test_report_default_run_follows_task(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(cli, "RUNS_DIR", Path("runs-that-do-not-exist"))
    assert main(["report", "--task", "sentiment"]) == EXIT_CONFIG
    assert "runs-that-do-not-exist/sentiment-sample.jsonl" in capsys.readouterr().err
```

Run: `uv run pytest tests/test_cli.py -v` → the three new tests FAIL.

- [ ] **Step 2: Update the CLI**

In `cli.py` remove `DEFAULT_TICKETS` and `DEFAULT_SAMPLE_RUN`; add:

```python
DATA_DIR = Path("data")
RUNS_DIR = Path("runs")


def default_tickets_path(task: Task) -> Path:
    return DATA_DIR / f"{task.name}.csv"


def default_sample_run_path(task: Task) -> Path:
    return RUNS_DIR / f"{task.name}-sample.jsonl"


def _run_path(args: argparse.Namespace) -> Path:
    """--run if given, else the committed sample run for --task (default task otherwise)."""
    if args.run is not None:
        return Path(args.run)
    return default_sample_run_path(TASKS[args.task or DEFAULT_TASK.name])


def _note_task_mismatch(requested: str | None, run_task: Task, path: Path) -> None:
    if requested is not None and requested != run_task.name:
        print(f"note: {path} is a {run_task.name} run; ignoring --task {requested}",
              file=sys.stderr)
```

`default_run_path(now)` (the timestamped live output) keeps using `RUNS_DIR`. In `cmd_live`: `tickets_path = args.tickets or default_tickets_path(task)` and load from it. In `cmd_replay` and `cmd_report`: `run_path = _run_path(args)`, read it (in `cmd_report` rename the `_task` from Task 4 back to `task`), then `_note_task_mismatch(args.task, task, run_path)`, and use `run_path` wherever `args.run` was used (messages, `run_name=run_path.stem`). In `build_parser`:

```python
    live.add_argument("--tickets", type=Path, default=None)
    ...
    rep.add_argument("--task", choices=sorted(TASKS), default=None)
    rep.add_argument("--run", type=Path, default=None)
    ...
    report.add_argument("--task", choices=sorted(TASKS), default=None)
    report.add_argument("--run", type=Path, default=None)
```

Run `uv run pytest tests/test_cli.py -v` → PASS.

- [ ] **Step 3: Rename the files and fix every reference**

```bash
git mv data/tickets.csv data/triage.csv
git mv runs/sample.jsonl runs/triage-sample.jsonl
```

`.gitignore`: replace `!runs/sample.jsonl` with two lines `!runs/triage-sample.jsonl` and `!runs/sentiment-sample.jsonl`.

`tests/test_data_file.py`: both `load_tickets(Path("data/triage.csv"), TRIAGE)`. `scripts/sample_banking77.py`: `OUT = Path("data/triage.csv")` and the docstring's first line. `scripts/smoke_live.py`: `from triage_bench.cli import ..., default_tickets_path` and `tickets = load_tickets(default_tickets_path(task), task)[:SMOKE_TICKETS]`.

`grep -rn "tickets.csv\|sample.jsonl" --exclude-dir=.venv --exclude-dir=.git .` must show only `runs/triage-sample.jsonl` / `runs/sentiment-sample.jsonl` in `.gitignore` and the spec/plan documents.

- [ ] **Step 4: Verify the README commands still work**

```bash
uv run triage-bench report                       # triage table, as in the README
uv run triage-bench report --task triage         # same
uv run triage-bench report --task sentiment      # error: run file not found: runs/sentiment-sample.jsonl (until Task 10)
```

- [ ] **Step 5: Full suite, lint, types, commit**

```bash
uv run pytest && uv run ruff check . && uv run mypy src
git add -A
git commit -m "Select the task on the command line and name files by task

data/triage.csv and runs/triage-sample.jsonl replace the unqualified
names; replay and report default to the sample run of the chosen task.
Every command in the README behaves as before.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Dashboard copy follows the item noun

**Files:**
- Create: `src/triage_bench/web/static/item-copy.js`
- Modify: `src/triage_bench/web/static/index.html`, `src/triage_bench/web/static/app.js`
- Test: `tests/web/js/item-copy.test.mjs`, `tests/web/test_app.py`

**Interfaces:**
- Produces: `itemCopy(noun)` returning `{ counterLabel, restartLabel, currentHeading, waiting, chartHeading, chartAria, chartAxis, feedHeading, routingIncomplete, progress(tick, total), finished(count), paused(tick, total), resumed(tick, total) }`.

- [ ] **Step 1: Write the failing JS test**

`tests/web/js/item-copy.test.mjs`:

```js
// Dashboard strings that name the benchmark item. Run: node --test 'tests/web/js/*.test.mjs'
import { test } from "node:test";
import assert from "node:assert/strict";

import { itemCopy } from "../../../src/triage_bench/web/static/item-copy.js";

test("ticket copy reproduces the triage dashboard's original strings", () => {
  const copy = itemCopy("ticket");
  assert.equal(copy.counterLabel, "tickets processed");
  assert.equal(copy.restartLabel, "Restart from ticket 1");
  assert.equal(copy.currentHeading, "Current ticket");
  assert.equal(copy.waiting, "Waiting for the first ticket…");
  assert.equal(copy.chartHeading, "Latency per ticket");
  assert.equal(copy.chartAria,
    "Latency per ticket, one line per contestant. The contestant cards list p50 latency.");
  assert.equal(copy.chartAxis, "ticket");
  assert.equal(copy.feedHeading, "Tickets");
  assert.equal(copy.routingIncomplete,
    "a ticket is missing a Von or Haiku decision; routing panel stopped");
  assert.equal(copy.progress(120, 200), "Ticket 120 of 200.");
  assert.equal(copy.finished(200), "Run finished after 200 tickets.");
  assert.equal(copy.paused(7, 200), "Paused at ticket 7 of 200.");
  assert.equal(copy.resumed(7, 200), "Resumed at ticket 7 of 200.");
});

test("review copy capitalises headings and pluralises", () => {
  const copy = itemCopy("review");
  assert.equal(copy.currentHeading, "Current review");
  assert.equal(copy.feedHeading, "Reviews");
  assert.equal(copy.progress(17, 200), "Review 17 of 200.");
  assert.equal(copy.finished(200), "Run finished after 200 reviews.");
});

test("every string mentions the noun", () => {
  const copy = itemCopy("review");
  for (const value of Object.values(copy)) {
    if (typeof value === "string") {
      assert.match(value.toLowerCase(), /review/);
    }
  }
});
```

Run: `node --test tests/web/js/item-copy.test.mjs` → FAIL (`Cannot find module .../item-copy.js`).

- [ ] **Step 2: Write item-copy.js**

```js
/*
 * Dashboard copy that names the benchmark item: "ticket" for triage, "review" for sentiment.
 *
 * Pure: given the noun from /api/meta it returns every string that mentions the item, so the
 * headings, ARIA labels and screen-reader announcements cannot drift apart
 * (tests/web/js/item-copy.test.mjs).
 */

export function itemCopy(noun) {
  const plural = `${noun}s`;
  const capital = noun.charAt(0).toUpperCase() + noun.slice(1);
  return {
    counterLabel: `${plural} processed`,
    restartLabel: `Restart from ${noun} 1`,
    currentHeading: `Current ${noun}`,
    waiting: `Waiting for the first ${noun}…`,
    chartHeading: `Latency per ${noun}`,
    chartAria: `Latency per ${noun}, one line per contestant. The contestant cards list p50 latency.`,
    chartAxis: noun,
    feedHeading: `${capital}s`,
    routingIncomplete: `a ${noun} is missing a Von or Haiku decision; routing panel stopped`,
    progress: (tick, total) => `${capital} ${tick} of ${total}.`,
    finished: (count) => `Run finished after ${count} ${plural}.`,
    paused: (tick, total) => `Paused at ${noun} ${tick} of ${total}.`,
    resumed: (tick, total) => `Resumed at ${noun} ${tick} of ${total}.`,
  };
}
```

Run the test → PASS.

- [ ] **Step 3: Serve it as JavaScript (Python test)**

In `tests/web/test_app.py` extend the parametrised list: `["app.js", "pacer.js", "latency-points.js", "item-copy.js"]`. Run → PASS already (StaticFiles serves any file in `static/`); it guards the MIME type going forward.

- [ ] **Step 4: Give the HTML hooks**

In `index.html`:

- After `<span id="mode" class="badge"></span>` add `<span id="task-name" class="badge"></span>`.
- `<h2>Current ticket</h2>` → `<h2 id="current-heading">Current ticket</h2>`.
- `<h2>Latency per ticket <span id="chart-caption" class="muted"></span></h2>` → `<h2><span id="chart-heading">Latency per ticket</span> <span id="chart-caption" class="muted"></span></h2>`.
- `<h2 id="feed-title">Tickets <span class="muted">newest first</span></h2>` → `<h2 id="feed-title"><span id="feed-heading">Tickets</span> <span class="muted">newest first</span></h2>`.

The `aria-label`s on `#tick`, `#restart` and `#latency` stay as sensible defaults; JS overwrites them once meta arrives.

- [ ] **Step 5: Wire app.js**

1. Next to the existing `import` lines at the top: `import { itemCopy } from "./item-copy.js";` and a constant `const DEFAULT_ITEM_NOUN = "ticket";` beside the other constants.
2. In `MESSAGES` delete the `routingIncomplete` entry; where it was used (`showBanner(MESSAGES.routingIncomplete)` near the `console.error("Routing needs both decisions for every ticket; ...")`), use `state.copy.routingIncomplete`.
3. In `els` add `taskName: $("task-name")`, `currentHeading: $("current-heading")`, `chartHeading: $("chart-heading")`, `feedHeading: $("feed-heading")`.
4. In `state` add `copy: itemCopy(DEFAULT_ITEM_NOUN),`.
5. Add:

```js
/** Rewrites every heading, label and placeholder that names the benchmark item. */
function applyItemCopy() {
  const copy = state.copy;
  els.tick.setAttribute("aria-label", copy.counterLabel);
  els.restart.setAttribute("aria-label", copy.restartLabel);
  els.currentHeading.textContent = copy.currentHeading;
  els.ticketText.textContent = copy.waiting;
  els.chartHeading.textContent = copy.chartHeading;
  els.latency.setAttribute("aria-label", copy.chartAria);
  els.feedHeading.textContent = copy.feedHeading;
}
```

6. In `applyMeta`, after `state.contestants = meta.contestants;`: `state.copy = itemCopy(meta.item_noun); els.taskName.textContent = meta.task; applyItemCopy();` — this must run before `buildChart()`.
7. In `buildChart`, the x-axis `title: { display: true, text: "ticket" }` → `text: state.copy.chartAxis`.
8. Replace the four literal announcements:

```js
// in enqueue/render of a tick (was: `Ticket ${event.tick} of ${event.total}. Accuracy: ...`)
announce(`${state.copy.progress(event.tick, event.total)} Accuracy: ${accuracySummary(event.totals)}.`);
// in finishRun (was: `Run finished after ${state.lastTick} tickets.${summary}`)
announce(`${state.copy.finished(state.lastTick)}${summary}`);
// in pauseReplay / resumeReplay
announce(state.copy.paused(state.lastTick, state.lastTotal));
announce(state.copy.resumed(state.lastTick, state.lastTotal));
```
9. In `resetDisplay`: `els.ticketText.textContent = state.copy.waiting;`.
10. `grep -n "ticket" src/triage_bench/web/static/app.js` afterwards must show only identifiers (`ticketText`, `ticketLabel`, `event.ticket`, `ticketId`) and comments — no user-visible literal containing the word.

- [ ] **Step 6: Look at it**

```bash
uv run triage-bench replay --port 8130 &
```

Open http://127.0.0.1:8130 in the built-in browser, press Start: header shows `replay` and `triage` badges, the copy still says ticket everywhere, announcements unchanged. Check at 360 px wide that the extra badge wraps without horizontal scroll. Then `kill %1`. (The sentiment replay is checked in Task 11 once its run exists.)

- [ ] **Step 7: Full suite, lint, types, JS tests, commit**

```bash
uv run pytest && uv run ruff check . && uv run mypy src && node --test 'tests/web/js/*.test.mjs'
git add -A src tests
git commit -m "Name the benchmark item from run metadata in the dashboard

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Sample Amazon Polarity into `data/sentiment.csv`

**Files:**
- Create: `scripts/sample_amazon_polarity.py`, `data/sentiment.csv`
- Test: `tests/test_data_file.py`

**Interfaces:**
- Produces: `data/sentiment.csv` with header `id,text,label`, 100 `negative` + 100 `positive`, ids = upstream test-split row index, text = `title` + `\n\n` + `content`.

- [ ] **Step 1: Write the failing data-file tests**

Replace `tests/test_data_file.py`:

```python
from collections import Counter
from pathlib import Path

import pytest

from triage_bench.domain.task import SENTIMENT, TRIAGE, Task
from triage_bench.infrastructure.csv_tickets import load_tickets

# (file, task, rows per label, longest same-label run allowed, rows within which every label
# has appeared). The run limit keeps a dashboard replay from showing one label many times in a
# row; the binary file gets a longer allowance because two labels alternate less.
DATASETS = [
    pytest.param(Path("data/triage.csv"), TRIAGE, 25, 3, 40, id="triage"),
    pytest.param(Path("data/sentiment.csv"), SENTIMENT, 100, 4, 10, id="sentiment"),
]


def _longest_run(labels: list[str]) -> int:
    """Length of the longest run of consecutive identical labels."""
    longest = 0
    current = 0
    previous: str | None = None
    for label in labels:
        current = current + 1 if label == previous else 1
        longest = max(longest, current)
        previous = label
    return longest


@pytest.mark.parametrize(("path", "task", "per_label", "max_run", "early_window"), DATASETS)
def test_committed_dataset_is_balanced_and_valid(
    path: Path, task: Task, per_label: int, max_run: int, early_window: int
) -> None:
    tickets = load_tickets(path, task)
    assert len(tickets) == per_label * len(task.labels)
    assert Counter(t.label for t in tickets) == {name: per_label for name in task.label_names()}
    assert len({t.id for t in tickets}) == len(tickets)


@pytest.mark.parametrize(("path", "task", "per_label", "max_run", "early_window"), DATASETS)
def test_committed_dataset_order_is_well_mixed(
    path: Path, task: Task, per_label: int, max_run: int, early_window: int
) -> None:
    """Source files are grouped or skewed by label; a replay in source order would show many
    identical labels in a row, which looks broken and makes early running accuracy meaningless.
    The samples are written in a shuffled order instead."""
    tickets = load_tickets(path, task)
    labels = [t.label for t in tickets]
    assert _longest_run(labels) <= max_run
    assert set(labels[:early_window]) == set(task.label_names())


def test_sentiment_rows_carry_title_and_body() -> None:
    tickets = load_tickets(Path("data/sentiment.csv"), SENTIMENT)
    assert all("\n\n" in t.text for t in tickets)
```

Run: `uv run pytest tests/test_data_file.py -v` → the sentiment cases FAIL with `tickets file not found`.

- [ ] **Step 2: Write the sampler**

`scripts/sample_amazon_polarity.py`:

```python
"""One-off: sample 100 negative + 100 positive Amazon Polarity test reviews into data/sentiment.csv.

Standard-library only, no extra dependency required.
Run: `uv run python scripts/sample_amazon_polarity.py`

Amazon Polarity is Apache-2.0 (Zhang, Zhao & LeCun 2015, built on McAuley & Leskovec 2013),
Hugging Face dataset `fancyzhx/amazon_polarity`. Labels come from star ratings upstream:
1-2 stars -> negative (0), 4-5 stars -> positive (1); 3-star reviews are excluded.

The dataset is distributed as parquet, which the standard library cannot read, so this
script pages through the Hugging Face datasets-server rows API instead. That API serves
the current revision and cannot be pinned; when this sample was drawn the dataset
revision was 9d9c45c18f8c3cf1b23a3c27917b60cbf28f3289 (last modified 2024-01-09). The
committed CSV is the source of truth, exactly as data/triage.csv is for Banking77.

The test split is shuffled upstream, so its first POOL_ROWS rows are a fair pool. Rows
are written in a seeded shuffle, redrawn until no label repeats more than
MAX_SAME_LABEL_RUN times in a row: with two labels a plain shuffle of 200 rows nearly
always contains a run of five or more, which looks broken on the live dashboard. Ids
are the upstream 0-based test-split row index.
"""

import csv
import json
import random
import urllib.request
from pathlib import Path

from triage_bench.domain.task import SENTIMENT

ROWS_API = "https://datasets-server.huggingface.co/rows"
DATASET = "fancyzhx/amazon_polarity"
CONFIG = "amazon_polarity"
SPLIT = "test"
PAGE_ROWS = 100  # the API's maximum page length
POOL_ROWS = 1000
SEED = 20260923
PER_LABEL = 100
MAX_SAME_LABEL_RUN = 4
UPSTREAM_LABELS = {0: "negative", 1: "positive"}
OUT = Path("data/sentiment.csv")

Row = tuple[int, str, str]  # (upstream row index, review text, label name)


def review_text(title: str, content: str) -> str:
    return f"{title}\n\n{content}"


def fetch_pool() -> list[Row]:
    pool: list[Row] = []
    for offset in range(0, POOL_ROWS, PAGE_ROWS):
        url = (f"{ROWS_API}?dataset={DATASET}&config={CONFIG}&split={SPLIT}"
               f"&offset={offset}&length={PAGE_ROWS}")
        with urllib.request.urlopen(url) as response:
            page = json.load(response)
        for item in page["rows"]:
            row = item["row"]
            pool.append((item["row_idx"], review_text(row["title"], row["content"]),
                         UPSTREAM_LABELS[row["label"]]))
    return pool


def longest_run(labels: list[str]) -> int:
    longest = current = 0
    previous: str | None = None
    for label in labels:
        current = current + 1 if label == previous else 1
        longest = max(longest, current)
        previous = label
    return longest


def main() -> None:
    if set(UPSTREAM_LABELS.values()) != set(SENTIMENT.label_names()):
        raise SystemExit(f"{UPSTREAM_LABELS} does not match the task's labels "
                         f"{SENTIMENT.label_names()}")
    by_label: dict[str, list[Row]] = {name: [] for name in SENTIMENT.label_names()}
    for row in fetch_pool():
        by_label[row[2]].append(row)

    rng = random.Random(SEED)
    chosen: list[Row] = []
    for label, candidates in by_label.items():
        if len(candidates) < PER_LABEL:
            raise SystemExit(f"{label}: only {len(candidates)} candidates, need {PER_LABEL}")
        chosen.extend(rng.sample(candidates, PER_LABEL))
    rng.shuffle(chosen)
    while longest_run([row[2] for row in chosen]) > MAX_SAME_LABEL_RUN:
        rng.shuffle(chosen)

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "text", "label"])
        writer.writerows(chosen)
    print(f"wrote {len(chosen)} reviews to {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it, inspect, test**

```bash
uv run python scripts/sample_amazon_polarity.py      # wrote 200 reviews to data/sentiment.csv
uv run python scripts/sample_amazon_polarity.py && git status --short data   # second run: file unchanged? (no diff) — confirms determinism
head -c 600 data/sentiment.csv
uv run pytest tests/test_data_file.py -v             # all PASS
```

If the Hugging Face API returns 429, wait a minute and rerun; do not reduce `POOL_ROWS`.

Read through at least 20 rows of the file: text is title, blank line, body; labels read right.

- [ ] **Step 4: Lint, full suite, commit**

```bash
uv run pytest && uv run ruff check . && uv run mypy src
git add scripts/sample_amazon_polarity.py data/sentiment.csv tests/test_data_file.py
git commit -m "Sample 200 Amazon Polarity reviews for the sentiment task

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: The live sentiment run

**Files:**
- Create: `runs/sentiment-sample.jsonl`

This spends real API money (≈ $0.15–0.20) and takes about eight minutes. Ask the user before starting it; they may prefer to run the command themselves.

- [ ] **Step 1: Smoke first**

```bash
uv run --env-file .env python scripts/smoke_live.py sentiment
```

Expected: three reviews, three contestants each, no `ERROR`. If Luna returns 400 for `reasoning.effort = "none"`, stop and report — do not change `REASONING_EFFORT` in this branch without asking.

- [ ] **Step 2: Run it**

```bash
uv run --env-file .env triage-bench live --task sentiment --out runs/sentiment-sample.jsonl
```

Open http://127.0.0.1:8000, press Start, wait for `200 / 200` and the Finished button. Then Ctrl-C the server.

- [ ] **Step 3: Check the file before it goes anywhere near git**

```bash
grep -c 'sk-' runs/sentiment-sample.jsonl        # must print 0
grep -c '"error": "' runs/sentiment-sample.jsonl  # decisions with an error; expect 0
wc -l runs/sentiment-sample.jsonl                 # 601 = 1 header + 200 x 3
uv run triage-bench report --task sentiment
uv run triage-bench replay --task sentiment --port 8130   # dashboard says Review / Reviews; Ctrl-C
```

Copy the report's table and routing lines aside for the README. If errors are non-zero, report the count and the messages to the user before deciding whether to re-run.

- [ ] **Step 4: Commit**

```bash
git add runs/sentiment-sample.jsonl
git commit -m "Add the committed sentiment run

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: README, screenshot, package description

**Files:**
- Create: `docs/screenshots/sentiment/dashboard-finished.png`
- Modify: `README.md`, `pyproject.toml`

- [ ] **Step 1: Capture the sentiment screenshot**

```bash
uv run triage-bench replay --task sentiment --port 8130 &
node scripts/capture_screenshots.mjs --url http://127.0.0.1:8130 --out docs/screenshots/sentiment
kill %1
rm docs/screenshots/sentiment/dashboard-mid-run.png docs/screenshots/sentiment/dashboard-threshold-095.png docs/screenshots/sentiment/dashboard-phone.png
ls docs/screenshots/sentiment      # only dashboard-finished.png
```

Open the PNG and check it shows `sentiment` beside the badge, "Current review", "Reviews", and 200 / 200.

- [ ] **Step 2: README**

Edits, in order:

1. First paragraph → `A local System One decision model against two hosted LLMs on two tasks — 8-way support-ticket intent triage and binary customer-review sentiment — streamed item by item to a live dashboard.`
2. `## Results` → keep the existing intro line, then two sub-headings:

```markdown
### Triage: 200 Banking77 tickets

<existing sentence with date, machine, cost, errors; existing table; existing routing bullets>

### Sentiment: 200 Amazon Polarity reviews

200 reviews, 100 negative and 100 positive, <date of the run>, Apple M2 16 GB (Von on MPS), total API cost ≈ $<from report>, <errors> errors. Table as printed by `uv run triage-bench report --task sentiment`:

<table from the report>

Routing summary at threshold 0.80, restated:

- Handled locally by Von: <x>%, rest to Haiku.
- Routed: accuracy <x>%, cost $<x>.
- Haiku alone: accuracy <x>%, cost $<x>.
- Cost saving: <x>%.
```

   Every number comes from the `report` output of Task 10, verbatim.
3. `## Quick start: replay` — after the `replay` line add `uv run triage-bench replay --task sentiment   # the review run`.
4. `## Quick start: live` — the bullet about own tickets becomes: `Own items: --tickets file.csv with header id,text,label; labels from the task in src/triage_bench/domain/task.py; pick the task with --task triage|sentiment.`
5. `## Screenshots` table gains a fourth column `Sentiment run` with `![Finished sentiment run at 200 of 200 with the review feed](docs/screenshots/sentiment/dashboard-finished.png)`.
6. `## Caveats` — keep every existing bullet; change the demo bullet to `200 tickets on 8 intents and 200 reviews on 2 labels are demos; no significance testing.` and add:

```markdown
- Where the model class stops, from throwaway spikes on 2026-09-23 (not part of the benchmark): on matched prompt-injection boundary pairs (72 rows from `3nesdeniz/agentic-prompt-injection-boundary-pairs`) Von scored 43%, below chance, giving attacks and their benign twins the same probability; on three-way support-ticket priority (90 rows from a synthetic set whose labels did not track the text) it scored 46% at a median confidence of 0.97. It reads labels off the words of a text; it does not judge actions or fuzzy middles.
```

7. `## Licence and attribution` — after the Banking77 block add:

```markdown
Reviews sampled from [Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity), Apache-2.0:

- Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. NIPS 2015.
- Julian McAuley, Jure Leskovec. Hidden factors and hidden topics: understanding rating dimensions with review text. RecSys 2013.
```

- [ ] **Step 3: pyproject description**

`description = "Live benchmark: a local System One decision model vs standard LLMs on support-ticket triage and review sentiment"`.

- [ ] **Step 4: Read the README top to bottom**

Every command in it must run as written from a fresh clone (`replay`, `replay --task sentiment`, `report`, `report --task sentiment`). Run each once.

- [ ] **Step 5: Full suite and commit**

```bash
uv run pytest && uv run ruff check . && uv run mypy src && node --test 'tests/web/js/*.test.mjs'
git add README.md pyproject.toml docs/screenshots/sentiment/dashboard-finished.png
git commit -m "Document the sentiment task and its results

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 6: Push and open the PR**

```bash
git push -u origin sentiment-task
```

Open a PR against `main` titled `Add a sentiment task alongside triage`, body: what changed (Task value, `--task`, dashboard noun, Amazon Polarity sample and run), the two results tables, and the spec link. End the body with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. Wait for CI green before asking for review.
