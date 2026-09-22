# Triage Bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A locally runnable web benchmark that streams, ticket by ticket, how Von (a local System One decision model) compares with Claude Haiku 4.5 and GPT-5.6 Luna on 8-way support-ticket intent classification, tracking accuracy, latency, cost and calibration.

**Architecture:** Pure domain modules (tickets, decisions, metrics, pricing, routing) with no I/O. An application layer defines a `Decider` protocol and a lockstep `run()` generator that yields one `TickEvent` per ticket. Infrastructure adapters wrap Von, the Anthropic SDK and the OpenAI SDK behind `Decider`, plus a CSV loader and a JSONL sink. A FastAPI app forwards events over Server-Sent Events to a static single-page dashboard. Replay mode re-emits a saved JSONL run through the same path with no network.

**Tech Stack:** Python 3.13 managed by `uv`; `von-sdk` 1.0.1; `anthropic`; `openai`; `pydantic`; `fastapi` + `uvicorn`; `pytest`, `ruff`, `mypy`; vanilla JS + Chart.js 4 from jsDelivr.

Spec: `docs/superpowers/specs/2026-09-22-triage-bench-design.md`. Read it before starting.

## Global Constraints

- Python `>=3.12`, `.python-version` pinned to `3.13`. All dependencies pinned via `uv.lock`.
- Package name `triage_bench`, src layout at `src/triage_bench/`. Run everything through `uv run`.
- Layering: `domain/` imports only stdlib and `pydantic`. `application/` imports `domain`. `infrastructure/` and `web/` import both. Nothing in `domain/` or `application/` imports `von`, `anthropic`, `openai`, `fastapi` or touches files or network.
- Contestant keys are exactly `"von"`, `"haiku"`, `"luna"`. Model IDs are exactly `wfzyx/von-1.0`, `claude-haiku-4-5`, `gpt-5.6-luna`.
- Prices per 1M tokens, checked 2026-09-22: Haiku 1.00 in / 5.00 out; Luna 0.20 in / 1.20 out; Von 0 / 0.
- LLMs: structured output `{label: enum, confidence: float 0..1}`, no thinking (Anthropic: omit `thinking`; OpenAI: `reasoning={"effort": "none"}`), `max_tokens` = 64.
- No magic numbers: every threshold, bin count, default speed, port and price is a named constant.
- No silent `except`. Errors are values on `Decision.error` or exit with context.
- Every task: RED test first, watch it fail, GREEN, run full suite, commit. Commit messages imperative, subject < 72 chars, end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Never commit `.env`, `runs/*.jsonl` other than `runs/sample.jsonl`, or the virtualenv.

## File Structure

```
triage-bench/
  pyproject.toml            project metadata, deps, ruff/mypy/pytest config
  .python-version           3.13
  uv.lock                   generated
  .gitignore  .env.example  LICENSE (MIT)  README.md
  .github/workflows/ci.yml  ruff + mypy + pytest, no secrets
  data/tickets.csv          200 sampled Banking77 tickets (id,text,label)
  runs/sample.jsonl         one committed real run (produced in Task 17)
  scripts/sample_banking77.py   one-off sampler, needs optional `datasets`
  scripts/smoke_live.py         3 real tickets through all deciders, not in CI
  src/triage_bench/
    __init__.py
    domain/__init__.py
    domain/ticket.py        Intent, INTENTS, LABELS, Ticket
    domain/decision.py      Decision
    domain/pricing.py       PRICES, cost_usd
    domain/metrics.py       accuracy, percentile, expected_calibration_error,
                            reliability_bins, Totals, summarise
    domain/routing.py       RoutingReport, confidence_gated_report
    application/__init__.py
    application/decider.py  Decider protocol, DeciderError
    application/prompt.py   INSTRUCTIONS, label_descriptions_text
    application/runner.py   TickEvent, ResultSink, run
    application/replay.py   replay
    infrastructure/__init__.py
    infrastructure/csv_tickets.py   load_tickets, TicketLoadError
    infrastructure/jsonl_sink.py    JsonlSink, read_run
    infrastructure/von_decider.py
    infrastructure/claude_decider.py
    infrastructure/openai_decider.py
    infrastructure/jev_decider.py
    web/__init__.py
    web/app.py              create_app, SSE endpoint
    web/static/index.html  app.js  styles.css
    cli.py                  live | replay | report
  tests/  (mirrors src; fixtures in tests/fixtures/)
```

---

### Task 0: Project scaffold

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `.env.example`, `LICENSE`, `README.md`, `.github/workflows/ci.yml`, `src/triage_bench/__init__.py`, `src/triage_bench/{domain,application,infrastructure,web}/__init__.py`, `tests/__init__.py`, `tests/test_package.py`

**Interfaces:**
- Produces: importable package `triage_bench` with `__version__ = "0.1.0"`; `uv run pytest`, `uv run ruff check .`, `uv run mypy src` all work.

- [ ] **Step 1: Write the failing test**

`tests/test_package.py`:
```python
import triage_bench


def test_package_exposes_version() -> None:
    assert triage_bench.__version__ == "0.1.0"
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[project]
name = "triage-bench"
version = "0.1.0"
description = "Live benchmark: a local System One decision model vs standard LLMs on support-ticket triage"
readme = "README.md"
license = "MIT"
requires-python = ">=3.12"
dependencies = [
    "von-sdk==1.0.1",
    "anthropic>=1.0",
    "openai>=2.0",
    "pydantic>=2.7",
    "fastapi>=0.115",
    "uvicorn>=0.30",
]

[project.optional-dependencies]
sampling = ["datasets>=3.0"]

[project.scripts]
triage-bench = "triage_bench.cli:main"

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6", "mypy>=1.11", "httpx>=0.27"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/triage_bench"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "N", "SIM"]

[tool.mypy]
python_version = "3.12"
strict = true
ignore_missing_imports = true
```

`.python-version`: `3.13`

- [ ] **Step 3: Create package files and supporting files**

`src/triage_bench/__init__.py`:
```python
"""Triage Bench: System One decision model vs LLMs on support-ticket triage."""

__version__ = "0.1.0"
```

Empty `__init__.py` in `domain/`, `application/`, `infrastructure/`, `web/`, and `tests/`.

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
.env
runs/*
!runs/sample.jsonl
.mypy_cache/
.ruff_cache/
.pytest_cache/
dist/
```

`.env.example`:
```
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
```

`LICENSE`: standard MIT text, copyright 2026 Rui Magalhaes.

`README.md`: title line and one sentence; filled in Task 18.

`.github/workflows/ci.yml`:
```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --all-groups
      - run: uv run ruff check .
      - run: uv run mypy src
      - run: uv run pytest
```

- [ ] **Step 4: Sync and run**

Run: `cd ~/projects/triage-bench && uv sync --all-groups && uv run pytest tests/test_package.py -v`
Expected: PASS. (The sync downloads torch via von-sdk; a few minutes on first run.)

Run: `uv run ruff check . && uv run mypy src`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "Scaffold triage-bench project with uv, ruff, mypy, pytest and CI"
```
(Add the Co-Authored-By trailer to every commit; omitted from the snippets below for brevity.)

---

### Task 1: Domain — intents and tickets

**Files:**
- Create: `src/triage_bench/domain/ticket.py`, `tests/domain/test_ticket.py`, `tests/domain/__init__.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Intent: label: str; description: str`
  - `INTENTS: tuple[Intent, ...]` — the eight from the spec, in spec order
  - `LABELS: tuple[str, ...]` — labels in the same order
  - `@dataclass(frozen=True) class Ticket: id: int; text: str; label: str` with `__post_init__` raising `ValueError` if `label not in LABELS` or `text` is blank.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from triage_bench.domain.ticket import INTENTS, LABELS, Ticket


def test_eight_intents_in_spec_order() -> None:
    assert LABELS == (
        "lost_or_stolen_card", "card_arrival", "declined_card_payment",
        "refund_not_showing_up", "exchange_rate", "top_up_failed",
        "passcode_forgotten", "terminate_account",
    )
    assert all(intent.description for intent in INTENTS)


def test_ticket_accepts_known_label() -> None:
    ticket = Ticket(id=1, text="My card never came", label="card_arrival")
    assert ticket.label == "card_arrival"


def test_ticket_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="unknown label"):
        Ticket(id=1, text="hi", label="pizza")


def test_ticket_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match="blank"):
        Ticket(id=1, text="   ", label="card_arrival")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/domain/test_ticket.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
"""Intents and tickets. Pure domain, no I/O."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    label: str
    description: str


INTENTS: tuple[Intent, ...] = (
    Intent("lost_or_stolen_card", "Card was lost, stolen, or the customer wants it blocked"),
    Intent("card_arrival", "Asking when or whether an ordered card will arrive"),
    Intent("declined_card_payment", "A card payment was declined or refused at checkout"),
    Intent("refund_not_showing_up", "A refund was promised but has not appeared in the account"),
    Intent("exchange_rate", "Questions about exchange rates or currency conversion applied"),
    Intent("top_up_failed", "Adding money to the account failed or did not go through"),
    Intent("passcode_forgotten", "Forgot the app passcode or PIN and wants to reset it"),
    Intent("terminate_account", "Wants to close or delete their account"),
)

LABELS: tuple[str, ...] = tuple(intent.label for intent in INTENTS)


@dataclass(frozen=True)
class Ticket:
    id: int
    text: str
    label: str

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"unknown label {self.label!r}")
        if not self.text.strip():
            raise ValueError("ticket text is blank")
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/domain -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/triage_bench/domain/ticket.py tests/domain
git commit -m "Add Intent and Ticket domain types with the eight Banking77 intents"
```

---

### Task 2: Domain — Decision

**Files:**
- Create: `src/triage_bench/domain/decision.py`, `tests/domain/test_decision.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class Decision:
      ticket_id: int
      contestant: str          # "von" | "haiku" | "luna"
      label: str | None        # None when error
      confidence: float        # 0.0 when error
      latency_ms: float        # 0.0 when error
      input_tokens: int
      output_tokens: int
      cost_usd: float
      error: str | None = None

      def is_correct(self, true_label: str) -> bool  # False when error
      @property
      def is_error(self) -> bool
  ```
  Validation: confidence outside [0, 1] or negative latency/tokens/cost raises `ValueError`.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from triage_bench.domain.decision import Decision


def make(**overrides: object) -> Decision:
    base: dict[str, object] = dict(
        ticket_id=1, contestant="von", label="card_arrival", confidence=0.9,
        latency_ms=100.0, input_tokens=0, output_tokens=0, cost_usd=0.0,
    )
    base.update(overrides)
    return Decision(**base)  # type: ignore[arg-type]


def test_correct_when_label_matches() -> None:
    assert make().is_correct("card_arrival")
    assert not make().is_correct("exchange_rate")


def test_error_decision_is_never_correct() -> None:
    d = make(label=None, confidence=0.0, latency_ms=0.0, error="boom")
    assert d.is_error
    assert not d.is_correct("card_arrival")


@pytest.mark.parametrize("field,value", [
    ("confidence", 1.5), ("confidence", -0.1), ("latency_ms", -1.0),
    ("input_tokens", -1), ("cost_usd", -0.01),
])
def test_rejects_out_of_range(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        make(**{field: value})
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/domain/test_decision.py -v` → FAIL, module not found.

- [ ] **Step 3: Implement**

```python
"""A single contestant's answer for a single ticket. Pure domain."""

from dataclasses import dataclass

CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0


@dataclass(frozen=True)
class Decision:
    ticket_id: int
    contestant: str
    label: str | None
    confidence: float
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error: str | None = None

    def __post_init__(self) -> None:
        if not CONFIDENCE_MIN <= self.confidence <= CONFIDENCE_MAX:
            raise ValueError(f"confidence out of range: {self.confidence}")
        for name in ("latency_ms", "input_tokens", "output_tokens", "cost_usd"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")

    @property
    def is_error(self) -> bool:
        return self.error is not None

    def is_correct(self, true_label: str) -> bool:
        return not self.is_error and self.label == true_label
```

- [ ] **Step 4: Run** `uv run pytest tests/domain -v` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/triage_bench/domain/decision.py tests/domain/test_decision.py
git commit -m "Add Decision domain type with correctness and error semantics"
```

---
### Task 3: Domain — pricing

**Files:**
- Create: `src/triage_bench/domain/pricing.py`, `tests/domain/test_pricing.py`

**Interfaces:**
- Produces:
  - `PRICES_CHECKED_ON = "2026-09-22"`
  - `@dataclass(frozen=True) class Price: input_per_million: float; output_per_million: float`
  - `PRICES: dict[str, Price]` keyed by contestant `"von"`, `"haiku"`, `"luna"`
  - `def cost_usd(contestant: str, input_tokens: int, output_tokens: int) -> float` — raises `KeyError` for unknown contestant.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from triage_bench.domain.pricing import PRICES, cost_usd


def test_haiku_price_per_million() -> None:
    assert PRICES["haiku"].input_per_million == 1.00
    assert PRICES["haiku"].output_per_million == 5.00


def test_luna_price_per_million() -> None:
    assert PRICES["luna"].input_per_million == 0.20
    assert PRICES["luna"].output_per_million == 1.20


def test_von_is_free() -> None:
    assert cost_usd("von", 10_000, 10_000) == 0.0


def test_cost_scales_linearly() -> None:
    # 1M input + 1M output tokens on haiku = 1.00 + 5.00
    assert cost_usd("haiku", 1_000_000, 1_000_000) == pytest.approx(6.00)
    assert cost_usd("luna", 500_000, 0) == pytest.approx(0.10)


def test_unknown_contestant_raises() -> None:
    with pytest.raises(KeyError):
        cost_usd("gpt-9", 1, 1)
```

- [ ] **Step 2: Run** → FAIL, module not found.

- [ ] **Step 3: Implement**

```python
"""Per-token prices and cost computation. Pure domain."""

from dataclasses import dataclass

PRICES_CHECKED_ON = "2026-09-22"
TOKENS_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class Price:
    input_per_million: float
    output_per_million: float


PRICES: dict[str, Price] = {
    "von": Price(0.0, 0.0),
    "haiku": Price(1.00, 5.00),
    "luna": Price(0.20, 1.20),
}


def cost_usd(contestant: str, input_tokens: int, output_tokens: int) -> float:
    price = PRICES[contestant]
    return (
        input_tokens * price.input_per_million + output_tokens * price.output_per_million
    ) / TOKENS_PER_MILLION
```

- [ ] **Step 4: Run** `uv run pytest tests/domain -v` → PASS.

- [ ] **Step 5: Commit** `git commit -m "Add contestant price table and cost computation"`

---

### Task 4: Domain — metrics and totals

**Files:**
- Create: `src/triage_bench/domain/metrics.py`, `tests/domain/test_metrics.py`

**Interfaces:**
- Consumes: `Decision` (Task 2), `Ticket` (Task 1).
- Produces:
  - `ECE_BINS = 10`
  - `def accuracy(decisions: Sequence[Decision], truth: Mapping[int, str]) -> float` — 0.0 for empty input; errors count as wrong.
  - `def percentile(values: Sequence[float], pct: float) -> float` — nearest-rank; 0.0 for empty.
  - `def reliability_bins(decisions, truth, bins: int = ECE_BINS) -> list[Bin]` where `@dataclass(frozen=True) class Bin: lower: float; upper: float; count: int; avg_confidence: float; accuracy: float`. Error decisions are excluded.
  - `def expected_calibration_error(decisions, truth, bins: int = ECE_BINS) -> float` — weighted |acc − conf| over bins; 0.0 when no scorable decisions.
  - `@dataclass(frozen=True) class Totals: contestant: str; count: int; errors: int; accuracy: float; p50_ms: float; p95_ms: float; cost_usd: float; ece: float; avg_confidence: float`
  - `def summarise(contestant: str, decisions: Sequence[Decision], truth: Mapping[int, str]) -> Totals` — latency percentiles over non-error decisions only.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from triage_bench.domain.decision import Decision
from triage_bench.domain.metrics import (
    accuracy, expected_calibration_error, percentile, reliability_bins, summarise,
)


def d(tid: int, label: str | None, conf: float, ms: float = 100.0, err: str | None = None,
      cost: float = 0.0) -> Decision:
    return Decision(ticket_id=tid, contestant="von", label=label, confidence=conf,
                    latency_ms=ms, input_tokens=0, output_tokens=0, cost_usd=cost, error=err)


TRUTH = {1: "card_arrival", 2: "card_arrival", 3: "exchange_rate", 4: "exchange_rate"}


def test_accuracy_counts_errors_as_wrong() -> None:
    ds = [d(1, "card_arrival", 0.9), d(2, "exchange_rate", 0.9),
          d(3, None, 0.0, 0.0, err="timeout")]
    assert accuracy(ds, TRUTH) == pytest.approx(1 / 3)


def test_accuracy_empty_is_zero() -> None:
    assert accuracy([], TRUTH) == 0.0


def test_percentile_nearest_rank() -> None:
    assert percentile([10, 20, 30, 40], 50) == 20
    assert percentile([10, 20, 30, 40], 95) == 40
    assert percentile([], 50) == 0.0


def test_perfectly_calibrated_has_zero_ece() -> None:
    # two decisions at 0.5 confidence, one right one wrong -> bin acc 0.5 == conf 0.5
    ds = [d(1, "card_arrival", 0.5), d(2, "exchange_rate", 0.5)]
    assert expected_calibration_error(ds, TRUTH) == pytest.approx(0.0)


def test_overconfident_wrong_answers_have_high_ece() -> None:
    ds = [d(1, "exchange_rate", 0.95), d(2, "exchange_rate", 0.95)]
    assert expected_calibration_error(ds, TRUTH) == pytest.approx(0.95)


def test_ece_ignores_errors_and_handles_none() -> None:
    assert expected_calibration_error([d(1, None, 0.0, 0.0, err="x")], TRUTH) == 0.0


def test_reliability_bins_cover_unit_interval() -> None:
    bins = reliability_bins([d(1, "card_arrival", 0.05), d(3, "exchange_rate", 0.95)], TRUTH)
    assert len(bins) == 10
    assert bins[0].count == 1 and bins[0].accuracy == 1.0
    assert bins[-1].count == 1 and bins[-1].upper == 1.0
    assert sum(b.count for b in bins) == 2


def test_confidence_of_exactly_one_lands_in_last_bin() -> None:
    bins = reliability_bins([d(1, "card_arrival", 1.0)], TRUTH)
    assert bins[-1].count == 1


def test_summarise() -> None:
    ds = [d(1, "card_arrival", 0.9, 100, cost=0.01), d(2, "card_arrival", 0.8, 300, cost=0.01),
          d(3, None, 0.0, 0.0, err="boom")]
    t = summarise("von", ds, TRUTH)
    assert t.count == 3 and t.errors == 1
    assert t.accuracy == pytest.approx(2 / 3)
    assert t.p50_ms == 100 and t.p95_ms == 300
    assert t.cost_usd == pytest.approx(0.02)
    assert t.avg_confidence == pytest.approx(0.85)
```

- [ ] **Step 2: Run** → FAIL, module not found.

- [ ] **Step 3: Implement**

```python
"""Accuracy, latency, calibration. Pure domain."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from triage_bench.domain.decision import Decision

ECE_BINS = 10
P50 = 50.0
P95 = 95.0


@dataclass(frozen=True)
class Bin:
    lower: float
    upper: float
    count: int
    avg_confidence: float
    accuracy: float


@dataclass(frozen=True)
class Totals:
    contestant: str
    count: int
    errors: int
    accuracy: float
    p50_ms: float
    p95_ms: float
    cost_usd: float
    ece: float
    avg_confidence: float


def _scorable(decisions: Sequence[Decision]) -> list[Decision]:
    return [d for d in decisions if not d.is_error]


def accuracy(decisions: Sequence[Decision], truth: Mapping[int, str]) -> float:
    if not decisions:
        return 0.0
    hits = sum(1 for d in decisions if d.is_correct(truth[d.ticket_id]))
    return hits / len(decisions)


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def _bin_index(confidence: float, bins: int) -> int:
    return min(int(confidence * bins), bins - 1)


def reliability_bins(
    decisions: Sequence[Decision], truth: Mapping[int, str], bins: int = ECE_BINS
) -> list[Bin]:
    buckets: list[list[Decision]] = [[] for _ in range(bins)]
    for d in _scorable(decisions):
        buckets[_bin_index(d.confidence, bins)].append(d)
    width = 1.0 / bins
    out: list[Bin] = []
    for i, bucket in enumerate(buckets):
        n = len(bucket)
        avg_conf = sum(d.confidence for d in bucket) / n if n else 0.0
        acc = accuracy(bucket, truth) if n else 0.0
        out.append(Bin(lower=i * width, upper=(i + 1) * width, count=n,
                       avg_confidence=avg_conf, accuracy=acc))
    return out


def expected_calibration_error(
    decisions: Sequence[Decision], truth: Mapping[int, str], bins: int = ECE_BINS
) -> float:
    total = len(_scorable(decisions))
    if total == 0:
        return 0.0
    return sum(
        (b.count / total) * abs(b.accuracy - b.avg_confidence)
        for b in reliability_bins(decisions, truth, bins)
        if b.count
    )


def summarise(
    contestant: str, decisions: Sequence[Decision], truth: Mapping[int, str]
) -> Totals:
    ok = _scorable(decisions)
    latencies = [d.latency_ms for d in ok]
    return Totals(
        contestant=contestant,
        count=len(decisions),
        errors=len(decisions) - len(ok),
        accuracy=accuracy(decisions, truth),
        p50_ms=percentile(latencies, P50),
        p95_ms=percentile(latencies, P95),
        cost_usd=sum(d.cost_usd for d in decisions),
        ece=expected_calibration_error(decisions, truth, bins=ECE_BINS),
        avg_confidence=(sum(d.confidence for d in ok) / len(ok)) if ok else 0.0,
    )
```

- [ ] **Step 4: Run** `uv run pytest tests/domain -v` → PASS. Run `uv run mypy src` → clean.

- [ ] **Step 5: Commit** `git commit -m "Add accuracy, latency percentile and calibration metrics"`

---

### Task 5: Domain — confidence-gated routing report

**Files:**
- Create: `src/triage_bench/domain/routing.py`, `tests/domain/test_routing.py`

**Interfaces:**
- Consumes: `Decision`, `accuracy`.
- Produces:
  - `DEFAULT_THRESHOLD = 0.80`
  - `@dataclass(frozen=True) class RoutingReport: threshold: float; primary: str; fallback: str; handled_locally_share: float; routed_accuracy: float; routed_cost_usd: float; fallback_accuracy: float; fallback_cost_usd: float; cost_saving_share: float`
  - `def confidence_gated_report(primary: Sequence[Decision], fallback: Sequence[Decision], truth: Mapping[int, str], threshold: float = DEFAULT_THRESHOLD) -> RoutingReport` — pairs by `ticket_id`; a primary error always routes to fallback; raises `ValueError` if the two sequences cover different ticket ids or threshold outside [0, 1].

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from triage_bench.domain.decision import Decision
from triage_bench.domain.routing import confidence_gated_report


def d(tid: int, who: str, label: str | None, conf: float, cost: float, err: str | None = None
      ) -> Decision:
    return Decision(ticket_id=tid, contestant=who, label=label, confidence=conf, latency_ms=1.0,
                    input_tokens=0, output_tokens=0, cost_usd=cost, error=err)


TRUTH = {1: "card_arrival", 2: "card_arrival", 3: "exchange_rate", 4: "exchange_rate"}
FALLBACK = [d(i, "haiku", TRUTH[i], 0.9, 0.001) for i in TRUTH]  # always right, costs


def test_high_confidence_stays_local() -> None:
    primary = [d(1, "von", "card_arrival", 0.95, 0.0), d(2, "von", "exchange_rate", 0.95, 0.0),
               d(3, "von", "exchange_rate", 0.3, 0.0), d(4, "von", None, 0.0, 0.0, err="x")]
    r = confidence_gated_report(primary, FALLBACK, TRUTH, threshold=0.8)
    assert r.handled_locally_share == 0.5           # tickets 1 and 2
    assert r.routed_accuracy == pytest.approx(0.75)  # 2 wrong at 0.95, 3 & 4 via haiku
    assert r.routed_cost_usd == pytest.approx(0.002)
    assert r.fallback_accuracy == 1.0
    assert r.fallback_cost_usd == pytest.approx(0.004)
    assert r.cost_saving_share == pytest.approx(0.5)


def test_threshold_one_routes_everything() -> None:
    primary = [d(i, "von", TRUTH[i], 0.99, 0.0) for i in TRUTH]
    r = confidence_gated_report(primary, FALLBACK, TRUTH, threshold=1.0)
    assert r.handled_locally_share == 0.0


def test_threshold_zero_keeps_everything_except_errors() -> None:
    primary = [d(i, "von", TRUTH[i], 0.01, 0.0) for i in TRUTH]
    r = confidence_gated_report(primary, FALLBACK, TRUTH, threshold=0.0)
    assert r.handled_locally_share == 1.0


def test_mismatched_tickets_raise() -> None:
    with pytest.raises(ValueError, match="ticket ids"):
        confidence_gated_report(FALLBACK[:2], FALLBACK, TRUTH)


def test_bad_threshold_raises() -> None:
    with pytest.raises(ValueError, match="threshold"):
        confidence_gated_report(FALLBACK, FALLBACK, TRUTH, threshold=1.5)
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

```python
"""Confidence-gated routing: keep the primary's answer when confident, else fall back."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from triage_bench.domain.decision import Decision
from triage_bench.domain.metrics import accuracy

DEFAULT_THRESHOLD = 0.80


@dataclass(frozen=True)
class RoutingReport:
    threshold: float
    primary: str
    fallback: str
    handled_locally_share: float
    routed_accuracy: float
    routed_cost_usd: float
    fallback_accuracy: float
    fallback_cost_usd: float
    cost_saving_share: float


def confidence_gated_report(
    primary: Sequence[Decision],
    fallback: Sequence[Decision],
    truth: Mapping[int, str],
    threshold: float = DEFAULT_THRESHOLD,
) -> RoutingReport:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be within [0, 1], got {threshold}")
    by_primary = {d.ticket_id: d for d in primary}
    by_fallback = {d.ticket_id: d for d in fallback}
    if by_primary.keys() != by_fallback.keys():
        raise ValueError("primary and fallback cover different ticket ids")

    routed: list[Decision] = []
    local = 0
    for tid, p in by_primary.items():
        if not p.is_error and p.confidence >= threshold:
            routed.append(p)
            local += 1
        else:
            routed.append(by_fallback[tid])

    n = len(routed)
    routed_cost = sum(d.cost_usd for d in routed)
    fallback_cost = sum(d.cost_usd for d in fallback)
    saving = (fallback_cost - routed_cost) / fallback_cost if fallback_cost else 0.0
    return RoutingReport(
        threshold=threshold,
        primary=primary[0].contestant if primary else "",
        fallback=fallback[0].contestant if fallback else "",
        handled_locally_share=local / n if n else 0.0,
        routed_accuracy=accuracy(routed, truth),
        routed_cost_usd=routed_cost,
        fallback_accuracy=accuracy(fallback, truth),
        fallback_cost_usd=fallback_cost,
        cost_saving_share=saving,
    )
```

- [ ] **Step 4: Run** `uv run pytest -v && uv run mypy src` → PASS, clean.

- [ ] **Step 5: Commit** `git commit -m "Add confidence-gated routing report"`

---
### Task 6: Application — Decider protocol and shared prompt

**Files:**
- Create: `src/triage_bench/application/decider.py`, `src/triage_bench/application/prompt.py`, `tests/application/__init__.py`, `tests/application/test_prompt.py`

**Interfaces:**
- Consumes: `INTENTS`, `LABELS`, `Ticket`.
- Produces:
  ```python
  class Decider(Protocol):
      name: str                       # "von" | "haiku" | "luna"
      def decide(self, ticket: Ticket) -> Decision: ...
  ```
  Adapters must never raise from `decide`; they return a Decision with `error` set.
  - `INSTRUCTIONS: str` — the one instruction sentence every contestant receives.
  - `def label_descriptions_text() -> str` — one line per intent: `label: description`.
  - `def llm_system_prompt() -> str` — INSTRUCTIONS + descriptions + the rule to answer with exactly one label and a confidence from 0 to 1.
  - `LLM_MAX_TOKENS = 64`

- [ ] **Step 1: Write the failing tests**

```python
from triage_bench.application.prompt import (
    INSTRUCTIONS, LLM_MAX_TOKENS, label_descriptions_text, llm_system_prompt,
)
from triage_bench.domain.ticket import INTENTS


def test_descriptions_list_every_intent_once() -> None:
    text = label_descriptions_text()
    for intent in INTENTS:
        assert text.count(f"{intent.label}: {intent.description}") == 1


def test_system_prompt_contains_instructions_and_labels() -> None:
    prompt = llm_system_prompt()
    assert INSTRUCTIONS in prompt
    assert label_descriptions_text() in prompt
    assert "confidence" in prompt


def test_token_cap() -> None:
    assert LLM_MAX_TOKENS == 64
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

`application/decider.py`:
```python
"""What every contestant must look like to the runner."""

from typing import Protocol

from triage_bench.domain.decision import Decision
from triage_bench.domain.ticket import Ticket


class Decider(Protocol):
    name: str

    def decide(self, ticket: Ticket) -> Decision:
        """Classify one ticket. Must not raise; report failures via Decision.error."""
        ...
```

`application/prompt.py`:
```python
"""The single instruction and label wording shared by every contestant."""

from triage_bench.domain.ticket import INTENTS

INSTRUCTIONS = "Classify the primary intent of this customer support message."
LLM_MAX_TOKENS = 64


def label_descriptions_text() -> str:
    return "\n".join(f"{i.label}: {i.description}" for i in INTENTS)


def llm_system_prompt() -> str:
    return (
        f"{INSTRUCTIONS}\n\nIntents:\n{label_descriptions_text()}\n\n"
        "Answer with exactly one intent label and your confidence that it is correct, "
        "as a number from 0 to 1."
    )
```

- [ ] **Step 4: Run** → PASS.

- [ ] **Step 5: Commit** `git commit -m "Add Decider protocol and shared classification prompt"`

---

### Task 7: Application — lockstep runner

**Files:**
- Create: `src/triage_bench/application/runner.py`, `tests/application/test_runner.py`

**Interfaces:**
- Consumes: `Decider`, `Ticket`, `Decision`, `summarise`, `Totals`.
- Produces:
  ```python
  class ResultSink(Protocol):
      def write(self, decision: Decision) -> None: ...

  @dataclass(frozen=True)
  class TickEvent:
      tick: int                         # 1-based
      total: int
      ticket: Ticket
      decisions: dict[str, Decision]    # keyed by contestant name
      totals: dict[str, Totals]
      def to_dict(self) -> dict[str, Any]   # JSON-ready; Decision gains "correct": bool

  def run(tickets: Sequence[Ticket], deciders: Sequence[Decider], sink: ResultSink
          ) -> Iterator[TickEvent]
  ```
  Order: for each ticket, call deciders in the given order, write each Decision to the sink immediately, then yield the event. Raises `ValueError` for duplicate decider names or empty inputs.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from triage_bench.application.runner import TickEvent, run
from triage_bench.domain.decision import Decision
from triage_bench.domain.ticket import Ticket

T = [Ticket(1, "card never came", "card_arrival"), Ticket(2, "rate?", "exchange_rate")]


class Scripted:
    def __init__(self, name: str, labels: list[str | None]) -> None:
        self.name, self._labels, self.calls = name, list(labels), []

    def decide(self, ticket: Ticket) -> Decision:
        self.calls.append(ticket.id)
        label = self._labels.pop(0)
        return Decision(ticket.id, self.name, label, 0.9 if label else 0.0,
                        10.0 if label else 0.0, 1, 1, 0.0, None if label else "fail")


class ListSink:
    def __init__(self) -> None:
        self.items: list[Decision] = []

    def write(self, decision: Decision) -> None:
        self.items.append(decision)


def test_lockstep_order_and_totals() -> None:
    a = Scripted("von", ["card_arrival", "card_arrival"])
    b = Scripted("haiku", ["card_arrival", None])
    sink = ListSink()
    events = list(run(T, [a, b], sink))

    assert [e.tick for e in events] == [1, 2] and events[0].total == 2
    assert a.calls == [1, 2] and b.calls == [1, 2]
    assert [d.contestant for d in sink.items] == ["von", "haiku", "von", "haiku"]
    assert events[1].totals["von"].accuracy == 0.5
    assert events[1].totals["haiku"].accuracy == 0.5 and events[1].totals["haiku"].errors == 1


def test_to_dict_is_json_ready_and_marks_correctness() -> None:
    e = next(run(T[:1], [Scripted("von", ["exchange_rate"])], ListSink()))
    payload = e.to_dict()
    assert payload["ticket"] == {"id": 1, "text": "card never came", "label": "card_arrival"}
    assert payload["decisions"]["von"]["correct"] is False
    assert payload["totals"]["von"]["accuracy"] == 0.0


def test_duplicate_names_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        list(run(T, [Scripted("von", []), Scripted("von", [])], ListSink()))


def test_empty_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        list(run([], [Scripted("von", [])], ListSink()))
    with pytest.raises(ValueError):
        list(run(T, [], ListSink()))
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

```python
"""Feed every ticket to every decider in lockstep and yield one event per ticket."""

from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from triage_bench.application.decider import Decider
from triage_bench.domain.decision import Decision
from triage_bench.domain.metrics import Totals, summarise
from triage_bench.domain.ticket import Ticket


class ResultSink(Protocol):
    def write(self, decision: Decision) -> None: ...


@dataclass(frozen=True)
class TickEvent:
    tick: int
    total: int
    ticket: Ticket
    decisions: dict[str, Decision]
    totals: dict[str, Totals]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "total": self.total,
            "ticket": asdict(self.ticket),
            "decisions": {
                name: {**asdict(d), "correct": d.is_correct(self.ticket.label)}
                for name, d in self.decisions.items()
            },
            "totals": {name: asdict(t) for name, t in self.totals.items()},
        }


def run(
    tickets: Sequence[Ticket], deciders: Sequence[Decider], sink: ResultSink
) -> Iterator[TickEvent]:
    if not tickets:
        raise ValueError("no tickets to run")
    if not deciders:
        raise ValueError("no deciders to run")
    names = [d.name for d in deciders]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate decider names: {names}")

    truth = {t.id: t.label for t in tickets}
    history: dict[str, list[Decision]] = {name: [] for name in names}
    for index, ticket in enumerate(tickets, start=1):
        decisions: dict[str, Decision] = {}
        for decider in deciders:
            decision = decider.decide(ticket)
            sink.write(decision)
            history[decider.name].append(decision)
            decisions[decider.name] = decision
        totals = {name: summarise(name, history[name], truth) for name in names}
        yield TickEvent(index, len(tickets), ticket, decisions, totals)
```

- [ ] **Step 4: Run** `uv run pytest -v && uv run mypy src && uv run ruff check .` → clean.

- [ ] **Step 5: Commit** `git commit -m "Add lockstep runner yielding per-ticket events"`

---

### Task 8: Infrastructure — CSV ticket loader

**Files:**
- Create: `src/triage_bench/infrastructure/csv_tickets.py`, `tests/infrastructure/__init__.py`, `tests/infrastructure/test_csv_tickets.py`

**Interfaces:**
- Produces:
  - `class TicketLoadError(Exception)` with attribute `problems: list[str]` (one per bad row, `"row N: reason"`).
  - `def load_tickets(path: Path) -> list[Ticket]` — requires header exactly `id,text,label`; collects all row problems then raises once; raises `TicketLoadError` for a missing file too.

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

import pytest

from triage_bench.infrastructure.csv_tickets import TicketLoadError, load_tickets


def write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "t.csv"
    p.write_text(body, encoding="utf-8")
    return p


def test_loads_valid_rows(tmp_path: Path) -> None:
    p = write(tmp_path, 'id,text,label\n7,"Where is my card, it has been 2 weeks",card_arrival\n')
    tickets = load_tickets(p)
    assert len(tickets) == 1
    assert tickets[0].id == 7 and tickets[0].label == "card_arrival"


def test_reports_every_bad_row(tmp_path: Path) -> None:
    p = write(tmp_path, "id,text,label\n1,hi,pizza\nx,hello,card_arrival\n3,ok,exchange_rate\n")
    with pytest.raises(TicketLoadError) as info:
        load_tickets(p)
    assert len(info.value.problems) == 2
    assert info.value.problems[0].startswith("row 2:")


def test_wrong_header_rejected(tmp_path: Path) -> None:
    with pytest.raises(TicketLoadError, match="header"):
        load_tickets(write(tmp_path, "text,label\nhi,card_arrival\n"))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(TicketLoadError, match="not found"):
        load_tickets(tmp_path / "nope.csv")
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

```python
"""Load tickets from the committed CSV, validating every row."""

import csv
from pathlib import Path

from triage_bench.domain.ticket import Ticket

EXPECTED_HEADER = ["id", "text", "label"]
FIRST_DATA_ROW = 2  # header is row 1


class TicketLoadError(Exception):
    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


def load_tickets(path: Path) -> list[Ticket]:
    if not path.exists():
        raise TicketLoadError([f"tickets file not found: {path}"])
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != EXPECTED_HEADER:
            raise TicketLoadError([f"bad header {header}, expected {EXPECTED_HEADER}"])
        tickets: list[Ticket] = []
        problems: list[str] = []
        for row_number, row in enumerate(reader, start=FIRST_DATA_ROW):
            try:
                ticket_id, text, label = row
                tickets.append(Ticket(id=int(ticket_id), text=text, label=label))
            except ValueError as exc:
                problems.append(f"row {row_number}: {exc}")
    if problems:
        raise TicketLoadError(problems)
    return tickets
```

- [ ] **Step 4: Run** → PASS.

- [ ] **Step 5: Commit** `git commit -m "Add validating CSV ticket loader"`

---

### Task 9: Infrastructure — JSONL sink, run reader, and replay

**Files:**
- Create: `src/triage_bench/infrastructure/jsonl_sink.py`, `src/triage_bench/application/replay.py`, `tests/infrastructure/test_jsonl_sink.py`, `tests/application/test_replay.py`

**Interfaces:**
- Produces:
  - `class JsonlSink: def __init__(self, path: Path)`; `write(decision)` appends one JSON line `{"kind": "decision", **asdict(decision)}`, flushing each write; `write_header(tickets: Sequence[Ticket])` writes a first line `{"kind": "tickets", "tickets": [asdict(t), ...]}` so a run file is self-contained.
  - `class RunFileError(Exception)`; `def read_run(path: Path) -> tuple[list[Ticket], list[Decision]]` — raises `RunFileError` with the line number for malformed JSON or a missing header.
  - `def replay(tickets, decisions, deciders_order: Sequence[str]) -> Iterator[TickEvent]` — rebuilds events by feeding stored decisions through `run()` via a `Replayer` decider that looks decisions up by `(contestant, ticket_id)`. Pacing is the web layer's job, not this one.

- [ ] **Step 1: Write the failing tests**

`tests/infrastructure/test_jsonl_sink.py`:
```python
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
```

`tests/application/test_replay.py`:
```python
from triage_bench.application.replay import replay
from triage_bench.domain.decision import Decision
from triage_bench.domain.ticket import Ticket

T = [Ticket(1, "a", "card_arrival"), Ticket(2, "b", "exchange_rate")]
D = [
    Decision(1, "von", "card_arrival", 0.9, 10.0, 0, 0, 0.0),
    Decision(1, "haiku", "exchange_rate", 0.7, 400.0, 30, 10, 0.00008),
    Decision(2, "von", "exchange_rate", 0.8, 11.0, 0, 0, 0.0),
    Decision(2, "haiku", "exchange_rate", 0.9, 380.0, 30, 10, 0.00008),
]


def test_replay_rebuilds_events_in_order() -> None:
    events = list(replay(T, D, ["von", "haiku"]))
    assert [e.tick for e in events] == [1, 2]
    assert events[0].decisions["haiku"].label == "exchange_rate"
    assert events[1].totals["von"].accuracy == 1.0
    assert events[1].totals["haiku"].accuracy == 0.5
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

`infrastructure/jsonl_sink.py`:
```python
"""Append-only JSONL run files: one header line of tickets, then one line per decision."""

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from triage_bench.domain.decision import Decision
from triage_bench.domain.ticket import Ticket

KIND_TICKETS = "tickets"
KIND_DECISION = "decision"


class RunFileError(Exception):
    pass


class JsonlSink:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path

    def write_header(self, tickets: Sequence[Ticket]) -> None:
        self._append({"kind": KIND_TICKETS, "tickets": [asdict(t) for t in tickets]})

    def write(self, decision: Decision) -> None:
        self._append({"kind": KIND_DECISION, **asdict(decision)})

    def _append(self, record: dict[str, object]) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def read_run(path: Path) -> tuple[list[Ticket], list[Decision]]:
    if not path.exists():
        raise RunFileError(f"run file not found: {path}")
    tickets: list[Ticket] | None = None
    decisions: list[Decision] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RunFileError(f"line {number}: invalid JSON ({exc.msg})") from exc
        kind = record.pop("kind", None)
        if kind == KIND_TICKETS:
            tickets = [Ticket(**t) for t in record["tickets"]]
        elif kind == KIND_DECISION:
            decisions.append(Decision(**record))
        else:
            raise RunFileError(f"line {number}: unknown record kind {kind!r}")
    if tickets is None:
        raise RunFileError("run file has no tickets header line")
    return tickets, decisions
```

`application/replay.py`:
```python
"""Re-emit a saved run as TickEvents through the normal runner."""

from collections.abc import Iterator, Sequence

from triage_bench.application.runner import TickEvent, run
from triage_bench.domain.decision import Decision
from triage_bench.domain.ticket import Ticket


class _Replayer:
    def __init__(self, name: str, decisions: Sequence[Decision]) -> None:
        self.name = name
        self._by_ticket = {d.ticket_id: d for d in decisions if d.contestant == name}

    def decide(self, ticket: Ticket) -> Decision:
        return self._by_ticket[ticket.id]


class _NullSink:
    def write(self, decision: Decision) -> None:
        return None


def replay(
    tickets: Sequence[Ticket], decisions: Sequence[Decision], deciders_order: Sequence[str]
) -> Iterator[TickEvent]:
    deciders = [_Replayer(name, decisions) for name in deciders_order]
    return run(tickets, deciders, _NullSink())
```

- [ ] **Step 4: Run** `uv run pytest -v && uv run mypy src && uv run ruff check .` → clean.

- [ ] **Step 5: Commit** `git commit -m "Add JSONL run files and replay through the runner"`

---
### Task 10: Application — Verdict schema; Infrastructure — Von adapter

**Files:**
- Create: `src/triage_bench/application/verdict.py`, `src/triage_bench/infrastructure/von_decider.py`, `tests/application/test_verdict.py`, `tests/infrastructure/test_von_decider.py`

**Interfaces:**
- Produces:
  - `IntentLabel = Enum("IntentLabel", {label: label for label in LABELS})` and `class Verdict(BaseModel): label: IntentLabel; confidence: float = Field(ge=0, le=1)`. This is the structured-output schema both LLM adapters use.
  - `class VonDecider: name = "von"`; `__init__(self, engine: VonEngine)` where `VonEngine` is a `Protocol` with `decide(state: str, choices: dict[str, str], instructions: str) -> VonResult` and `VonResult` is a `Protocol` with `choice: str`, `probabilities: dict[str, float]`. `@classmethod def from_sdk(cls) -> "VonDecider"` imports `von` and wraps the module. Confidence recorded is `probabilities[choice]` (a probability, comparable with the LLMs' self-reported number), not Von's top-two gap.
  - `def warm_up(self) -> None` — one throwaway decide so model load is not counted in tick latency.

- [ ] **Step 1: Write the failing tests**

`tests/application/test_verdict.py`:
```python
import pytest
from pydantic import ValidationError

from triage_bench.application.verdict import IntentLabel, Verdict
from triage_bench.domain.ticket import LABELS


def test_labels_match_domain() -> None:
    assert tuple(m.value for m in IntentLabel) == LABELS


def test_rejects_confidence_above_one() -> None:
    with pytest.raises(ValidationError):
        Verdict(label=IntentLabel.card_arrival, confidence=1.2)


def test_schema_has_enum() -> None:
    schema = Verdict.model_json_schema()
    assert "card_arrival" in str(schema)
```

`tests/infrastructure/test_von_decider.py`:
```python
from dataclasses import dataclass

from triage_bench.application.prompt import INSTRUCTIONS
from triage_bench.domain.ticket import INTENTS, Ticket
from triage_bench.infrastructure.von_decider import VonDecider

TICKET = Ticket(1, "my card still has not arrived", "card_arrival")


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


def test_maps_result_and_passes_shared_prompt() -> None:
    engine = FakeEngine(FakeResult("card_arrival", {"card_arrival": 0.93, "exchange_rate": 0.07}))
    d = VonDecider(engine).decide(TICKET)
    assert d.contestant == "von" and d.label == "card_arrival"
    assert d.confidence == 0.93 and d.cost_usd == 0.0 and d.latency_ms > 0
    state, choices, instructions = engine.calls[0]
    assert state == TICKET.text and instructions == INSTRUCTIONS
    assert choices == {i.label: i.description for i in INTENTS}


def test_engine_exception_becomes_error_decision() -> None:
    d = VonDecider(FakeEngine(RuntimeError("mps out of memory"))).decide(TICKET)
    assert d.is_error and "mps out of memory" in (d.error or "")
    assert d.label is None and d.latency_ms == 0.0
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

`application/verdict.py`:
```python
"""Structured-output schema shared by the LLM contestants."""

from enum import Enum

from pydantic import BaseModel, Field

from triage_bench.domain.ticket import LABELS

IntentLabel = Enum("IntentLabel", {label: label for label in LABELS})  # type: ignore[misc]


class Verdict(BaseModel):
    label: IntentLabel  # type: ignore[valid-type]
    confidence: float = Field(ge=0.0, le=1.0)
```

`infrastructure/von_decider.py`:
```python
"""Von, the local System One model, behind the Decider protocol."""

import time
from typing import Protocol

from triage_bench.application.prompt import INSTRUCTIONS
from triage_bench.domain.decision import Decision
from triage_bench.domain.ticket import INTENTS, Ticket

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

    def __init__(self, engine: VonEngine) -> None:
        self._engine = engine
        self._choices = {i.label: i.description for i in INTENTS}

    @classmethod
    def from_sdk(cls) -> "VonDecider":
        import von  # local model; imported here so tests never load it

        return cls(von)

    def warm_up(self) -> None:
        self._engine.decide(state=WARM_UP_TEXT, choices=self._choices, instructions=INSTRUCTIONS)

    def decide(self, ticket: Ticket) -> Decision:
        started = time.perf_counter()
        try:
            result = self._engine.decide(
                state=ticket.text, choices=self._choices, instructions=INSTRUCTIONS
            )
        except Exception as exc:  # adapters must not raise; the runner needs every tick
            return Decision(ticket.id, CONTESTANT, None, 0.0, 0.0, 0, 0, 0.0, error=str(exc))
        latency_ms = (time.perf_counter() - started) * MS_PER_SECOND
        return Decision(
            ticket_id=ticket.id,
            contestant=CONTESTANT,
            label=result.choice,
            confidence=result.probabilities[result.choice],
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
        )
```

- [ ] **Step 4: Run** `uv run pytest -v && uv run mypy src && uv run ruff check .` → clean. If ruff flags the broad `except Exception` (rule BLE001 is not in the selected set, so it should not), keep it: the comment explains why.

- [ ] **Step 5: Commit** `git commit -m "Add Verdict schema and Von adapter"`

---

### Task 11: Infrastructure — Claude Haiku adapter

**Files:**
- Create: `src/triage_bench/infrastructure/claude_decider.py`, `tests/infrastructure/test_claude_decider.py`

**Interfaces:**
- Consumes: `Verdict`, `llm_system_prompt()`, `LLM_MAX_TOKENS`, `cost_usd`.
- Produces: `class ClaudeDecider: name = "haiku"`; `__init__(self, client: anthropic.Anthropic, model: str = MODEL_ID)`; `MODEL_ID = "claude-haiku-4-5"`. Calls `client.messages.parse(model=..., max_tokens=LLM_MAX_TOKENS, system=llm_system_prompt(), messages=[{"role": "user", "content": ticket.text}], output_format=Verdict)`. Reads `response.parsed_output`, `response.usage.input_tokens`, `response.usage.output_tokens`. No `thinking` parameter. Catches `anthropic.APIError`, `pydantic.ValidationError` and `ValueError` into `Decision.error`.

- [ ] **Step 1: Write the failing tests**

```python
from types import SimpleNamespace

import anthropic
import httpx2 as httpx

from triage_bench.application.prompt import LLM_MAX_TOKENS, llm_system_prompt
from triage_bench.application.verdict import IntentLabel, Verdict
from triage_bench.domain.ticket import Ticket
from triage_bench.infrastructure.claude_decider import MODEL_ID, ClaudeDecider

TICKET = Ticket(1, "my card still has not arrived", "card_arrival")


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
        parsed_output=Verdict(label=IntentLabel.card_arrival, confidence=0.85),
        usage=SimpleNamespace(input_tokens=120, output_tokens=15),
    )


def test_builds_request_per_fairness_rules() -> None:
    client, messages = client_with(good_response())
    ClaudeDecider(client).decide(TICKET)  # type: ignore[arg-type]
    assert messages.kwargs is not None
    assert messages.kwargs["model"] == MODEL_ID == "claude-haiku-4-5"
    assert messages.kwargs["max_tokens"] == LLM_MAX_TOKENS
    assert messages.kwargs["system"] == llm_system_prompt()
    assert messages.kwargs["messages"] == [{"role": "user", "content": TICKET.text}]
    assert messages.kwargs["output_format"] is Verdict
    assert "thinking" not in messages.kwargs


def test_maps_parsed_output_tokens_and_cost() -> None:
    client, _ = client_with(good_response())
    d = ClaudeDecider(client).decide(TICKET)  # type: ignore[arg-type]
    assert d.contestant == "haiku" and d.label == "card_arrival" and d.confidence == 0.85
    assert d.input_tokens == 120 and d.output_tokens == 15
    assert d.cost_usd == (120 * 1.00 + 15 * 5.00) / 1_000_000


def test_api_error_becomes_error_decision() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request)
    err = anthropic.RateLimitError("rate limited", response=response, body=None)
    client, _ = client_with(err)
    d = ClaudeDecider(client).decide(TICKET)  # type: ignore[arg-type]
    assert d.is_error and "rate limited" in (d.error or "")


def test_missing_parsed_output_is_error() -> None:
    client, _ = client_with(SimpleNamespace(parsed_output=None,
                                            usage=SimpleNamespace(input_tokens=1, output_tokens=1)))
    d = ClaudeDecider(client).decide(TICKET)  # type: ignore[arg-type]
    assert d.is_error and "no parsed output" in (d.error or "")
```

- [ ] **Step 2: Run** → FAIL. (If `import httpx2` fails, check `uv run python -c "import anthropic, httpx2"`; the anthropic 1.x SDK depends on `httpx2`.)

- [ ] **Step 3: Implement**

```python
"""Claude Haiku 4.5 behind the Decider protocol. One structured-output call, no thinking."""

import time

import anthropic
from pydantic import ValidationError

from triage_bench.application.prompt import LLM_MAX_TOKENS, llm_system_prompt
from triage_bench.application.verdict import Verdict
from triage_bench.domain.decision import Decision
from triage_bench.domain.pricing import cost_usd
from triage_bench.domain.ticket import Ticket

CONTESTANT = "haiku"
MODEL_ID = "claude-haiku-4-5"
MS_PER_SECOND = 1000.0


class ClaudeDecider:
    name = CONTESTANT

    def __init__(self, client: anthropic.Anthropic, model: str = MODEL_ID) -> None:
        self._client = client
        self._model = model

    def decide(self, ticket: Ticket) -> Decision:
        started = time.perf_counter()
        try:
            response = self._client.messages.parse(
                model=self._model,
                max_tokens=LLM_MAX_TOKENS,
                system=llm_system_prompt(),
                messages=[{"role": "user", "content": ticket.text}],
                output_format=Verdict,
            )
            verdict = response.parsed_output
            if verdict is None:
                raise ValueError("no parsed output in response")
        except (anthropic.APIError, ValidationError, ValueError) as exc:
            return Decision(ticket.id, CONTESTANT, None, 0.0, 0.0, 0, 0, 0.0, error=str(exc))
        latency_ms = (time.perf_counter() - started) * MS_PER_SECOND
        usage = response.usage
        return Decision(
            ticket_id=ticket.id,
            contestant=CONTESTANT,
            label=verdict.label.value,
            confidence=verdict.confidence,
            latency_ms=latency_ms,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=cost_usd(CONTESTANT, usage.input_tokens, usage.output_tokens),
        )
```

- [ ] **Step 4: Run** `uv run pytest -v && uv run mypy src && uv run ruff check .` → clean.

- [ ] **Step 5: Commit** `git commit -m "Add Claude Haiku adapter with structured output"`

---

### Task 12: Infrastructure — OpenAI Luna adapter and Jev stub

**Files:**
- Create: `src/triage_bench/infrastructure/openai_decider.py`, `src/triage_bench/infrastructure/jev_decider.py`, `tests/infrastructure/test_openai_decider.py`, `tests/infrastructure/test_jev_decider.py`

**Interfaces:**
- Produces:
  - `class OpenAIDecider: name = "luna"`; `__init__(self, client: openai.OpenAI, model: str = MODEL_ID)`; `MODEL_ID = "gpt-5.6-luna"`; `REASONING_EFFORT = "none"`. Calls `client.responses.parse(model=..., input=[{"role": "system", "content": llm_system_prompt()}, {"role": "user", "content": ticket.text}], text_format=Verdict, max_output_tokens=LLM_MAX_TOKENS, reasoning={"effort": REASONING_EFFORT})`. Reads `response.output_parsed`, `response.usage.input_tokens`, `response.usage.output_tokens`. A response with `status == "incomplete"` is an error (`"incomplete: <reason>"`). Catches `openai.APIError`, `ValidationError`, `ValueError`.
  - `class JevDecider: name = "jev"`; constructing it raises `NotConfiguredError` whose message points to `docs/superpowers/specs/...` section 4 and says TypeSafe signups are closed. It exists so the slot is visible in code.

- [ ] **Step 1: Write the failing tests**

`tests/infrastructure/test_openai_decider.py`:
```python
from types import SimpleNamespace

import httpx
import openai

from triage_bench.application.prompt import LLM_MAX_TOKENS, llm_system_prompt
from triage_bench.application.verdict import IntentLabel, Verdict
from triage_bench.domain.ticket import Ticket
from triage_bench.infrastructure.openai_decider import (
    MODEL_ID, REASONING_EFFORT, OpenAIDecider,
)

TICKET = Ticket(1, "my card still has not arrived", "card_arrival")


class FakeResponses:
    def __init__(self, outcome: object) -> None:
        self.outcome, self.kwargs = outcome, None

    def parse(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def client_with(outcome: object) -> tuple[object, FakeResponses]:
    responses = FakeResponses(outcome)
    return SimpleNamespace(responses=responses), responses


def good_response() -> object:
    return SimpleNamespace(
        status="completed",
        output_parsed=Verdict(label=IntentLabel.card_arrival, confidence=0.7),
        usage=SimpleNamespace(input_tokens=100, output_tokens=12),
    )


def test_builds_request_per_fairness_rules() -> None:
    client, responses = client_with(good_response())
    OpenAIDecider(client).decide(TICKET)  # type: ignore[arg-type]
    k = responses.kwargs
    assert k is not None
    assert k["model"] == MODEL_ID == "gpt-5.6-luna"
    assert k["max_output_tokens"] == LLM_MAX_TOKENS
    assert k["reasoning"] == {"effort": REASONING_EFFORT}
    assert k["text_format"] is Verdict
    assert k["input"] == [{"role": "system", "content": llm_system_prompt()},
                          {"role": "user", "content": TICKET.text}]


def test_maps_output_and_cost() -> None:
    client, _ = client_with(good_response())
    d = OpenAIDecider(client).decide(TICKET)  # type: ignore[arg-type]
    assert d.contestant == "luna" and d.label == "card_arrival" and d.confidence == 0.7
    assert d.cost_usd == (100 * 0.20 + 12 * 1.20) / 1_000_000


def test_incomplete_response_is_error() -> None:
    incomplete = SimpleNamespace(
        status="incomplete", output_parsed=None,
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        usage=SimpleNamespace(input_tokens=1, output_tokens=64),
    )
    client, _ = client_with(incomplete)
    d = OpenAIDecider(client).decide(TICKET)  # type: ignore[arg-type]
    assert d.is_error and "max_output_tokens" in (d.error or "")


def test_api_error_is_error_decision() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    err = openai.APIConnectionError(request=request)
    client, _ = client_with(err)
    d = OpenAIDecider(client).decide(TICKET)  # type: ignore[arg-type]
    assert d.is_error
```

`tests/infrastructure/test_jev_decider.py`:
```python
import pytest

from triage_bench.infrastructure.jev_decider import JevDecider, NotConfiguredError


def test_jev_slot_is_explicitly_unavailable() -> None:
    with pytest.raises(NotConfiguredError, match="signups"):
        JevDecider()
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

`infrastructure/openai_decider.py`:
```python
"""GPT-5.6 Luna behind the Decider protocol. One structured-output call, reasoning off."""

import time

import openai
from pydantic import ValidationError

from triage_bench.application.prompt import LLM_MAX_TOKENS, llm_system_prompt
from triage_bench.application.verdict import Verdict
from triage_bench.domain.decision import Decision
from triage_bench.domain.pricing import cost_usd
from triage_bench.domain.ticket import Ticket

CONTESTANT = "luna"
MODEL_ID = "gpt-5.6-luna"
REASONING_EFFORT = "none"  # if the API returns 400 for this model, change to "minimal"
STATUS_COMPLETED = "completed"
MS_PER_SECOND = 1000.0


class OpenAIDecider:
    name = CONTESTANT

    def __init__(self, client: openai.OpenAI, model: str = MODEL_ID) -> None:
        self._client = client
        self._model = model

    def decide(self, ticket: Ticket) -> Decision:
        started = time.perf_counter()
        try:
            response = self._client.responses.parse(
                model=self._model,
                input=[
                    {"role": "system", "content": llm_system_prompt()},
                    {"role": "user", "content": ticket.text},
                ],
                text_format=Verdict,
                max_output_tokens=LLM_MAX_TOKENS,
                reasoning={"effort": REASONING_EFFORT},
            )
            if response.status != STATUS_COMPLETED:
                reason = getattr(getattr(response, "incomplete_details", None), "reason", "?")
                raise ValueError(f"{response.status}: {reason}")
            verdict = response.output_parsed
            if verdict is None:
                raise ValueError("no parsed output in response")
        except (openai.APIError, ValidationError, ValueError) as exc:
            return Decision(ticket.id, CONTESTANT, None, 0.0, 0.0, 0, 0, 0.0, error=str(exc))
        latency_ms = (time.perf_counter() - started) * MS_PER_SECOND
        usage = response.usage
        if usage is None:
            return Decision(ticket.id, CONTESTANT, None, 0.0, 0.0, 0, 0, 0.0,
                            error="response carried no usage")
        return Decision(
            ticket_id=ticket.id,
            contestant=CONTESTANT,
            label=verdict.label.value,
            confidence=verdict.confidence,
            latency_ms=latency_ms,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=cost_usd(CONTESTANT, usage.input_tokens, usage.output_tokens),
        )
```

`infrastructure/jev_decider.py`:
```python
"""Placeholder slot for TypeSafe's Jev. Not wired: TypeSafe closed signups on 2026-09-22."""

CONTESTANT = "jev"
SPEC_SECTION = "docs/superpowers/specs/2026-09-22-triage-bench-design.md section 4"


class NotConfiguredError(Exception):
    pass


class JevDecider:
    name = CONTESTANT

    def __init__(self) -> None:
        raise NotConfiguredError(
            "Jev is not configured: TypeSafe is not accepting signups. "
            f"When a key is available, implement decide() per {SPEC_SECTION} "
            "using the same Verdict schema, and add 'jev' to the price table."
        )
```

- [ ] **Step 4: Run** `uv run pytest -v && uv run mypy src && uv run ruff check .` → clean. If mypy complains about the `reasoning` literal type, wrap it: `reasoning={"effort": REASONING_EFFORT}  # type: ignore[typeddict-item]` with a comment naming the SDK type it fails against.

- [ ] **Step 5: Commit** `git commit -m "Add OpenAI Luna adapter and explicit Jev placeholder"`

---
### Task 13: Web — FastAPI app with SSE stream

**Files:**
- Create: `src/triage_bench/web/app.py`, `tests/web/__init__.py`, `tests/web/test_app.py`

**Interfaces:**
- Consumes: `TickEvent.to_dict()`.
- Produces:
  ```python
  EventSource = Callable[[], Iterator[TickEvent]]   # called once per client connection

  @dataclass(frozen=True)
  class RunMeta:
      mode: str            # "live" | "replay"
      run_name: str
      contestants: list[str]

  DEFAULT_SPEED_TPS = 2.0     # replay ticks per second
  MAX_SPEED_TPS = 20.0

  def create_app(source: EventSource, meta: RunMeta) -> FastAPI
  ```
  Routes: `GET /` serves `static/index.html`; `/static/*` serves the folder; `GET /api/meta` returns `RunMeta` as JSON; `GET /api/stream?speed=<tps>` is `text/event-stream`, one `data: <event json>\n\n` per tick, then `event: done\ndata: {}\n\n`. In `replay` mode the stream sleeps `1/speed` between ticks (clamped to `MAX_SPEED_TPS`); in `live` mode `speed` is ignored because real calls set the pace. The blocking iterator is advanced with `asyncio.to_thread` so the event loop stays responsive.

- [ ] **Step 1: Write the failing tests**

```python
import json
from collections.abc import Iterator

from fastapi.testclient import TestClient

from triage_bench.application.replay import replay
from triage_bench.application.runner import TickEvent
from triage_bench.domain.decision import Decision
from triage_bench.domain.ticket import Ticket
from triage_bench.web.app import RunMeta, create_app

T = [Ticket(1, "a", "card_arrival"), Ticket(2, "b", "exchange_rate")]
D = [Decision(1, "von", "card_arrival", 0.9, 10.0, 0, 0, 0.0),
     Decision(2, "von", "card_arrival", 0.6, 11.0, 0, 0, 0.0)]


def source() -> Iterator[TickEvent]:
    return replay(T, D, ["von"])


def make_client(mode: str = "replay") -> TestClient:
    return TestClient(create_app(source, RunMeta(mode=mode, run_name="fixture", contestants=["von"])))


def test_meta() -> None:
    assert make_client().get("/api/meta").json() == {
        "mode": "replay", "run_name": "fixture", "contestants": ["von"]}


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
```

- [ ] **Step 2: Run** → FAIL (no module; index.html also missing, so create a minimal placeholder `web/static/index.html` containing `<!doctype html><html><body>Triage Bench</body></html>` in this task; Task 15 replaces it).

- [ ] **Step 3: Implement**

```python
"""FastAPI app: static dashboard plus a Server-Sent Events stream of tick events."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from triage_bench.application.runner import TickEvent

EventSource = Callable[[], Iterator[TickEvent]]

STATIC_DIR = Path(__file__).parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"
DEFAULT_SPEED_TPS = 2.0
MIN_SPEED_TPS = 0.1
MAX_SPEED_TPS = 20.0
MODE_REPLAY = "replay"
MODE_LIVE = "live"
SSE_MEDIA_TYPE = "text/event-stream"
_END = object()


@dataclass(frozen=True)
class RunMeta:
    mode: str
    run_name: str
    contestants: list[str]


def _sse(data: dict[str, object], event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data)}\n\n"


def create_app(source: EventSource, meta: RunMeta) -> FastAPI:
    app = FastAPI(title="Triage Bench")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(INDEX_FILE)

    @app.get("/api/meta")
    async def get_meta() -> dict[str, object]:
        return asdict(meta)

    @app.get("/api/stream")
    async def stream(
        speed: float = Query(DEFAULT_SPEED_TPS, ge=MIN_SPEED_TPS, le=MAX_SPEED_TPS),
    ) -> StreamingResponse:
        pause_s = 1.0 / speed if meta.mode == MODE_REPLAY else 0.0

        async def events() -> AsyncIterator[str]:
            iterator = source()
            while True:
                item = await asyncio.to_thread(next, iterator, _END)
                if item is _END:
                    break
                assert isinstance(item, TickEvent)
                yield _sse(item.to_dict())
                if pause_s:
                    await asyncio.sleep(pause_s)
            yield _sse({}, event="done")

        return StreamingResponse(events(), media_type=SSE_MEDIA_TYPE)

    return app
```

- [ ] **Step 4: Run** `uv run pytest -v && uv run mypy src && uv run ruff check .` → clean.

- [ ] **Step 5: Commit** `git commit -m "Add FastAPI app streaming tick events over SSE"`

---

### Task 14: CLI — live, replay, report

**Files:**
- Create: `src/triage_bench/cli.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces `main(argv: list[str] | None = None) -> int` with subcommands:
  - `live [--tickets PATH] [--out PATH] [--port N] [--host H]` — checks `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`, loads tickets, builds `[VonDecider.from_sdk(), ClaudeDecider(anthropic.Anthropic()), OpenAIDecider(openai.OpenAI())]`, warms up Von, writes the tickets header to the JSONL sink, serves the app. The run begins when a browser connects to `/api/stream`. One browser tab per live run.
  - `replay [--run PATH] [--port N] [--host H]` — reads the run file and serves the app in replay mode. Default run is `runs/sample.jsonl`.
  - `report [--run PATH] [--threshold T]` — prints a markdown totals table and the routing report.
  - Pure helpers, unit tested: `missing_keys(env: Mapping[str, str]) -> list[str]`, `format_totals_table(totals: Sequence[Totals]) -> str`, `format_routing(report: RoutingReport) -> str`, `default_run_path(now: datetime) -> Path`.
  - Exit codes: `EXIT_OK = 0`, `EXIT_CONFIG = 2`. Every failure path prints one line to stderr naming the cause.

- [ ] **Step 1: Write the failing tests**

```python
from datetime import datetime
from pathlib import Path

from triage_bench.cli import (
    EXIT_CONFIG, default_run_path, format_routing, format_totals_table, main, missing_keys,
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
    assert default_run_path(datetime(2026, 9, 22, 10, 5, 0)) == Path("runs/2026-09-22T10-05-00.jsonl")


def test_live_without_keys_exits_with_config_error(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["live"]) == EXIT_CONFIG
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_report_on_missing_file_exits_with_config_error(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["report", "--run", "runs/does-not-exist.jsonl"]) == EXIT_CONFIG
    assert "not found" in capsys.readouterr().err
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

```python
"""Command line entry point: live, replay, report."""

import argparse
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path

from triage_bench.application.runner import TickEvent
from triage_bench.domain.metrics import Totals, summarise
from triage_bench.domain.routing import DEFAULT_THRESHOLD, RoutingReport, confidence_gated_report
from triage_bench.infrastructure.csv_tickets import TicketLoadError, load_tickets
from triage_bench.infrastructure.jsonl_sink import JsonlSink, RunFileError, read_run

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


def _serve(source_factory, meta, host: str, port: int) -> int:  # type: ignore[no-untyped-def]
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

    deciders = [von, ClaudeDecider(anthropic.Anthropic()), OpenAIDecider(openai.OpenAI())]
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
```

- [ ] **Step 4: Run** `uv run pytest -v && uv run mypy src && uv run ruff check .` → clean. Replace the `# type: ignore[no-untyped-def]` on `_serve` with real types (`EventSource`, `RunMeta` imported under `TYPE_CHECKING`) if mypy accepts them; do not leave the ignore without the explanatory comment.

- [ ] **Step 5: Commit** `git commit -m "Add CLI with live, replay and report commands"`

---
### Task 15: Web — the dashboard frontend

**Files:**
- Create: `src/triage_bench/web/static/index.html`, `src/triage_bench/web/static/styles.css`, `src/triage_bench/web/static/app.js`
- Modify: replace the placeholder `index.html` from Task 13.

**Interfaces:**
- Consumes: `GET /api/meta` → `{mode, run_name, contestants}`; `GET /api/stream?speed=` SSE frames shaped like `TickEvent.to_dict()`; `event: done`.
- Produces: no Python interface. Manual verification is the test for this task (see Step 4). The frontend does **no metric math**; it renders `totals` from each event. The one exception is the routing panel, which needs a client-side pass over the tickets seen so far: it keeps every event's `decisions` in memory and recomputes the split when the slider moves. That logic mirrors `confidence_gated_report` and must stay identical in behaviour: primary kept when `!error && confidence >= threshold`, else fallback.

Before writing any CSS or chart code, **load the `dataviz` skill** for palette, chart form and accessibility rules; this task fixes structure and behaviour only.

- [ ] **Step 1: Structure (`index.html`)**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Triage Bench</title>
  <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
  <header class="bar">
    <h1>Triage Bench</h1>
    <span id="mode" class="badge"></span>
    <span id="run-name" class="muted"></span>
    <span id="tick" class="tick">0 / 0</span>
    <div class="controls">
      <button id="play" type="button" aria-pressed="false">Start</button>
      <label id="speed-wrap">Speed <input id="speed" type="range" min="0.5" max="10" step="0.5" value="2">
        <output id="speed-out">2×</output></label>
      <label>Routing threshold
        <input id="threshold" type="range" min="0" max="1" step="0.05" value="0.8">
        <output id="threshold-out">0.80</output></label>
    </div>
  </header>

  <main>
    <section id="cards" class="cards" aria-live="polite"></section>

    <section class="ticket">
      <h2>Current ticket</h2>
      <p id="ticket-text" class="ticket-text">Waiting for the first ticket…</p>
      <p class="muted">True label: <code id="ticket-label">–</code></p>
    </section>

    <section class="chart">
      <h2>Latency per ticket (ms, log scale, last 50)</h2>
      <canvas id="latency" height="220" role="img" aria-label="Latency per ticket"></canvas>
    </section>

    <section class="routing" id="routing">
      <h2>Confidence-gated routing</h2>
      <p id="routing-text" class="muted">Von answers when confident, otherwise Haiku answers.</p>
      <dl class="kv">
        <dt>Handled locally</dt><dd id="r-local">–</dd>
        <dt>Routed accuracy</dt><dd id="r-acc">–</dd>
        <dt>Routed cost</dt><dd id="r-cost">–</dd>
        <dt>Haiku alone</dt><dd id="r-fallback">–</dd>
        <dt>Cost saving</dt><dd id="r-saving">–</dd>
      </dl>
    </section>

    <section class="feed">
      <h2>Recent tickets</h2>
      <table id="feed">
        <thead><tr><th>#</th><th>Message</th><th>Truth</th><th id="feed-heads"></th></tr></thead>
        <tbody></tbody>
      </table>
    </section>
  </main>

  <template id="card-template">
    <article class="card" data-contestant="">
      <h3 class="card-name"></h3>
      <p class="verdict"><span class="verdict-label">–</span> <span class="verdict-conf muted"></span></p>
      <div class="ring" role="img"><span class="ring-value">–</span><span class="ring-caption">accuracy</span></div>
      <dl class="kv">
        <dt>p50 latency</dt><dd class="p50">–</dd>
        <dt>cost so far</dt><dd class="cost">–</dd>
        <dt>ECE</dt><dd class="ece">–</dd>
        <dt>errors</dt><dd class="errors">0</dd>
      </dl>
    </article>
  </template>

  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <script src="/static/app.js" defer></script>
</body>
</html>
```

Display names and the model line under each card come from a constant in `app.js`:
```js
const CONTESTANTS = {
  von:   { title: "Von",              model: "wfzyx/von-1.0 · local",   free: true },
  haiku: { title: "Claude Haiku 4.5", model: "claude-haiku-4-5 · API" },
  luna:  { title: "GPT-5.6 Luna",     model: "gpt-5.6-luna · API" },
};
```

- [ ] **Step 2: Behaviour (`app.js`)**

Requirements the implementation must meet, in plain terms:

1. On load, fetch `/api/meta`; render `mode` badge, run name, and one card per contestant in `meta.contestants` order using the template. Hide the speed control when mode is `live`.
2. **Start** opens `new EventSource('/api/stream?speed=' + speed)`. The button becomes **Pause**. Pause closes the EventSource and freezes the display. Start again reopens the stream from the beginning in replay mode (server restarts the iterator), and the UI resets its state first. In live mode the button is disabled after the first start with the title "a live run cannot be restarted from the browser".
3. Each `message` event: parse JSON; update `tick / total`; set the ticket text and true label; for each contestant update verdict label, `confidence` as a percentage, `correct` colour class (`ok` / `miss` / `err`), accuracy ring (conic-gradient driven by a CSS custom property `--pct`), p50, cost formatted as `$0.0000` or the word `local` for `free` contestants, ECE to three decimals, error count.
4. Push each contestant's `latency_ms` into the Chart.js line chart (one dataset per contestant, x = tick, y log scale, keep the last `MAX_POINTS = 50`); error decisions add `null` so the line breaks rather than drawing zero.
5. Prepend a row to the feed with `#`, truncated message (`MAX_FEED_CHARS = 90`), truth, and one cell per contestant coloured by correctness; keep `MAX_FEED_ROWS = 12`.
6. Keep `history = []` of `{ticket, decisions}`. Recompute the routing panel on every event and whenever the threshold slider moves, using primary `von` and fallback `haiku` when both exist (otherwise hide the panel). Local share, routed accuracy, routed cost, fallback accuracy and cost, saving share, formatted like the CLI's `format_routing`.
7. `done` event: close the EventSource, set the button to **Finished** and disable it.
8. `error` on the EventSource: show a visible banner "stream disconnected" in the header; do not silently retry.

No framework, no build step. Use `const` for every magic value (`MAX_POINTS`, `MAX_FEED_ROWS`, `MAX_FEED_CHARS`, `CONTESTANTS`).

- [ ] **Step 3: Style (`styles.css`)**

Requirements:
- Mobile first: single column by default; at `min-width: 768px` the cards become three across and the ticket/chart/routing sections form a two-column grid. Touch targets at least 44 px; range inputs have visible labels.
- Dark theme by default (recording target), respecting `prefers-color-scheme: light` with a light variant. All colours as tokens on `:root`, chosen per the `dataviz` skill; correctness uses colour **and** a glyph (`✓` / `✗` / `!`) so it reads without colour.
- 16:9 friendly: at `min-width: 1280px` cap `main` at 1400 px and centre it; the feed sits below the fold so the top of the screen holds header, cards, current ticket, chart and routing panel.
- No horizontal page scroll at 360 px.

- [ ] **Step 4: Manual verification (this task's test)**

Fixture run for the check: create `tests/fixtures/mini_run.jsonl` with a tickets header of 6 tickets and 18 decisions (3 contestants × 6, at least one error decision and one wrong answer per contestant), then:

Run: `uv run triage-bench replay --run tests/fixtures/mini_run.jsonl --port 8000`

Open `http://127.0.0.1:8000` in the built-in browser and confirm, taking a screenshot for each:
1. Desktop width: three cards across, Start plays six ticks at 2 ticks/s, rings and numbers change, chart draws three lines with a gap at the error, feed shows six rows, routing panel updates when the slider moves, button shows **Finished** at the end.
2. Pause mid-run freezes the display; Start restarts from tick 1.
3. Viewport 360 px wide: single column, no horizontal scroll, controls reachable.
4. Console has zero errors.

Fix anything that fails; then commit.

- [ ] **Step 5: Commit**

```bash
git add src/triage_bench/web/static tests/fixtures/mini_run.jsonl
git commit -m "Add the live dashboard frontend"
```

---

### Task 16: Data — sample 200 Banking77 tickets

**Files:**
- Create: `scripts/sample_banking77.py`, `data/tickets.csv`, `tests/test_data_file.py`

**Interfaces:**
- Consumes: `LABELS`, `load_tickets`.
- Produces: `data/tickets.csv` with exactly 200 rows, 25 per label, `id` = the row's index in the Banking77 **test** split (stable across re-runs), sorted by id. Note that Banking77 spells one label `Refund_not_showing_up` with a capital R; the script lower-cases label names before matching.

- [ ] **Step 1: Write the failing test**

```python
from collections import Counter
from pathlib import Path

from triage_bench.domain.ticket import LABELS
from triage_bench.infrastructure.csv_tickets import load_tickets

TICKETS_PER_LABEL = 25


def test_committed_dataset_is_balanced_and_valid() -> None:
    tickets = load_tickets(Path("data/tickets.csv"))
    assert len(tickets) == TICKETS_PER_LABEL * len(LABELS)
    assert Counter(t.label for t in tickets) == {label: TICKETS_PER_LABEL for label in LABELS}
    assert [t.id for t in tickets] == sorted(t.id for t in tickets)
    assert len({t.id for t in tickets}) == len(tickets)
```

- [ ] **Step 2: Run** → FAIL, file not found.

- [ ] **Step 3: Write and run the sampler**

```python
"""One-off: sample 25 tickets per intent from the Banking77 test split into data/tickets.csv.

Requires the optional `sampling` extra:  uv sync --extra sampling
Banking77 is CC-BY-4.0, PolyAI (Casanueva et al., 2020).
"""

import csv
import random
from pathlib import Path

from datasets import load_dataset

from triage_bench.domain.ticket import LABELS

DATASET = "PolyAI/banking77"
SPLIT = "test"
SEED = 20260922
PER_LABEL = 25
OUT = Path("data/tickets.csv")


def main() -> None:
    rows = load_dataset(DATASET, split=SPLIT)
    names = [n.lower() for n in rows.features["label"].names]
    wanted = set(LABELS)
    by_label: dict[str, list[tuple[int, str]]] = {label: [] for label in LABELS}
    for index, row in enumerate(rows):
        label = names[row["label"]]
        if label in wanted:
            by_label[label].append((index, row["text"]))

    rng = random.Random(SEED)
    chosen: list[tuple[int, str, str]] = []
    for label, candidates in by_label.items():
        if len(candidates) < PER_LABEL:
            raise SystemExit(f"{label}: only {len(candidates)} candidates, need {PER_LABEL}")
        for index, text in rng.sample(candidates, PER_LABEL):
            chosen.append((index, text, label))
    chosen.sort()

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "text", "label"])
        writer.writerows(chosen)
    print(f"wrote {len(chosen)} tickets to {OUT}")


if __name__ == "__main__":
    main()
```

Run: `uv sync --extra sampling && uv run python scripts/sample_banking77.py`
Expected: `wrote 200 tickets to data/tickets.csv`. Open the CSV and eyeball ten rows to confirm the labels look right.

- [ ] **Step 4: Run** `uv run pytest -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/sample_banking77.py data/tickets.csv tests/test_data_file.py
git commit -m "Add 200-ticket Banking77 sample with sampling script"
```

---
### Task 17: Live smoke test and the committed sample run

**Needs the user:** both API keys in a local `.env` (never committed). Everything before this task runs without keys. If the keys are not available, finish Tasks 18's README with the results table marked "pending first live run" and tell the user exactly which command to run.

**Files:**
- Create: `scripts/smoke_live.py`, `runs/sample.jsonl`

**Interfaces:**
- Consumes: all three adapters, `load_tickets`.
- Produces: `runs/sample.jsonl`, a full 200-ticket real run, which `replay` and `report` default to.

- [ ] **Step 1: Write the smoke script**

```python
"""Three real tickets through every contestant. Costs a fraction of a cent. Not run in CI.

Usage:  set -a; source .env; set +a; uv run python scripts/smoke_live.py
"""

import sys
from pathlib import Path

import anthropic
import openai

from triage_bench.infrastructure.claude_decider import ClaudeDecider
from triage_bench.infrastructure.csv_tickets import load_tickets
from triage_bench.infrastructure.openai_decider import OpenAIDecider
from triage_bench.infrastructure.von_decider import VonDecider

SMOKE_TICKETS = 3


def main() -> int:
    tickets = load_tickets(Path("data/tickets.csv"))[:SMOKE_TICKETS]
    von = VonDecider.from_sdk()
    von.warm_up()
    deciders = [von, ClaudeDecider(anthropic.Anthropic()), OpenAIDecider(openai.OpenAI())]
    failures = 0
    for ticket in tickets:
        print(f"\n#{ticket.id} [{ticket.label}] {ticket.text}")
        for decider in deciders:
            d = decider.decide(ticket)
            status = "ERROR " + (d.error or "") if d.is_error else (
                f"{d.label} conf={d.confidence:.2f} {d.latency_ms:.0f}ms ${d.cost_usd:.6f}")
            print(f"  {decider.name:5s} {status}")
            failures += d.is_error
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the smoke test**

Run: `set -a; source .env; set +a; uv run python scripts/smoke_live.py`
Expected: nine decisions, zero `ERROR` lines.

Known thing to watch: if every `luna` line says `ERROR` with an HTTP 400 mentioning `reasoning.effort`, the model does not accept `"none"`. Change `REASONING_EFFORT` in `openai_decider.py` to `"minimal"`, update the test in Task 12 accordingly, re-run. If `luna` lines say `incomplete: max_output_tokens`, reasoning tokens are eating the 64-token cap: that confirms reasoning is still on and the same fix applies. Do not raise the cap.

- [ ] **Step 3: Produce the sample run**

Run: `set -a; source .env; set +a; uv run triage-bench live --out runs/sample.jsonl`
Then open `http://127.0.0.1:8000` and press Start. The run takes roughly 200 × (Von ~0.1 s + Haiku ~1 s + Luna ~1 s) ≈ 7 minutes. Watch for error verdicts; a handful of transient API errors is acceptable and will show in the report's error column, but more than 5 for one contestant means something is wrong: stop, fix, delete the file, rerun.

Run: `uv run triage-bench report`
Expected: a markdown table with three rows and the routing paragraph. Paste the output into the README in Task 18.

- [ ] **Step 4: Verify replay works from the committed file with no keys**

Run: `env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY uv run triage-bench replay`
Open the page, press Start, watch a few ticks at speed 10. Confirm the run name badge reads `sample`.

- [ ] **Step 5: Commit**

```bash
git add scripts/smoke_live.py runs/sample.jsonl
git commit -m "Add live smoke script and a committed 200-ticket sample run"
```

---

### Task 18: README, caveats, and the final verification pass

**Files:**
- Modify: `README.md`
- Create: `docs/dashboard.png` (screenshot from Task 15 or 17, desktop width, mid-run)

**Interfaces:** none. This is the public face of the repo; write it for an engineer who has never heard of System One models.

- [ ] **Step 1: Write the README**

Sections, in this order, with the content each must contain:

1. **Title and one-paragraph pitch.** What it does, in two sentences. The dashboard screenshot right under it.
2. **What is being compared.** The table from the spec's section 4: contestant, model, where it runs, price. One paragraph on what a System One model is (returns a decision plus a probability, no text generation) and that Von is an open-source 395M ModernBERT stand-in for TypeSafe's Jev, whose signups were closed when this was built.
3. **Results.** The `report` output pasted verbatim, with the date of the run and the machine (Apple M2, 16 GB). One sentence per column explaining what it means, including that ECE is expected calibration error and lower is better.
4. **Quick start, replay (no keys, no network).**
   ```bash
   git clone https://github.com/<user>/triage-bench && cd triage-bench
   curl -LsSf https://astral.sh/uv/install.sh | sh     # if you do not have uv
   uv sync
   uv run triage-bench replay
   ```
   then open `http://127.0.0.1:8000` and press Start. Note that `uv sync` installs PyTorch (about 300 MB) because Von needs it even for replay; that is the one heavy dependency.
5. **Quick start, live.**
   - Copy `.env.example` to `.env`, fill both keys, and load it into the shell (`set -a; source .env; set +a`).
   - **Model download:** the first `live` run downloads the Von weights, about 1.5 GB, from Hugging Face into `~/.cache/huggingface/hub/models--wfzyx--von-1.0`. It happens once. Delete that folder to free the space. No Hugging Face account is needed.
   - **Hardware:** runs on CPU or Apple Silicon (MPS) out of the box; an NVIDIA GPU is picked up automatically if PyTorch sees one. On an M2 a decision takes 60 to 200 ms; the Von README's 18 ms figure is a GPU number.
   - `uv run triage-bench live`, open the page, press Start. One browser tab per live run; the run file lands in `runs/`.
   - Cost of a full 200-ticket run: under $0.10 total for both APIs. Give the actual figure from the sample run.
6. **Reading the dashboard.** One line per element: cards, ring, current ticket, latency chart, routing panel and slider, feed.
7. **Use your own tickets.** The CSV format `id,text,label`, the eight labels, and the fact that changing the label set means editing `domain/ticket.py` (labels and descriptions in one place) and the loader validates the file for you.
8. **How it stays fair.** The bullet list from spec section 4, verbatim in spirit: same wording, structured output, no thinking, 64-token cap, client-side latency, cost from real token counts.
9. **Caveats.** Plain statements:
   - Von is a stand-in, not Jev. It is about 100x smaller and self-reports 72% on a broad benchmark where Jev reports 97%.
   - Laptop latency is not the README's GPU latency.
   - LLM confidence is self-reported, not a probability the model computed. Its calibration is exactly what the ECE column measures, and poor calibration is a finding, not a bug.
   - 200 tickets is a demo, not a paper. No significance testing.
   - Live-mode latency includes network time from wherever you run it.
10. **Development.** `uv sync --all-groups`, `uv run pytest`, `uv run ruff check .`, `uv run mypy src`. Layering rules in one paragraph. Pointer to the spec and plan under `docs/superpowers/`.
11. **Next experiments.** Three bullets: Haiku with thinking on as a fourth column; Banking77 with all 77 intents; the real Jev adapter when signups reopen.
12. **Licence and attribution.** MIT for this repo. Banking77 by PolyAI, CC-BY-4.0, with the citation. Von by wfzyx, Apache-2.0.

- [ ] **Step 2: Take the screenshot**

With `replay` running, wait until about tick 120, take a 1280-wide screenshot of the top of the page, save as `docs/dashboard.png`, and reference it from the README.

- [ ] **Step 3: Full verification pass**

Run, in a fresh shell without the keys loaded:
```bash
uv sync --all-groups
uv run ruff check .
uv run mypy src
uv run pytest
uv run triage-bench report
env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY uv run triage-bench replay --port 8001
```
Expected: lint clean, types clean, all tests pass, report prints, replay serves. Walk the README's replay quick start literally, command by command, in a temporary clone (`git clone . /tmp/tb-check`) to catch anything that only works in the original checkout.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/dashboard.png
git commit -m "Write README with setup, results, fairness rules and caveats"
```

- [ ] **Step 5: Publish (user decision)**

Only when the user says go: `gh repo create <user>/triage-bench --public --source . --push`. Then confirm CI is green on GitHub and paste the CI badge into the README in a follow-up commit.

---

## Self-review against the spec

- Section 3 dataset: Task 16 (sampler, 8 intents × 25, committed CSV, loader validation in Task 8). ✓
- Section 4 contestants, prices, fairness: Tasks 3, 6, 10, 11, 12; lockstep and client-side latency in Task 7. ✓
- Section 5 architecture and event stream: Tasks 1–9 and 13; `to_dict` shape in Task 7 matches the spec's event example (`decisions.*.correct`, `totals.*`). ✓
- Section 5 modes: live and replay in Task 14 with the pacing in Task 13; report in Task 14. ✓
- Section 5 routing report: Task 5 (domain), Task 14 (CLI), Task 15 (slider). ✓
- Section 6 UI: Task 15 covers header, cards, current ticket, latency chart, feed, routing panel, mobile single column, dark theme. ✓
- Section 7 errors: adapters return `Decision.error` (Tasks 10–12); startup exits in Task 14; CSV problems Task 8; run file line numbers Task 9. ✓
- Section 8 tests: every domain/application/infrastructure task has RED→GREEN tests with fakes; web SSE test Task 13; nothing in CI touches network or loads Von (`from_sdk` is only called by the CLI and smoke script). ✓
- Section 9 hygiene: Task 0 (uv, lockfile, gitignore, .env.example, MIT, CI) and Task 18 (README, attribution, caveats). ✓
- Type consistency: `Decision` field order `(ticket_id, contestant, label, confidence, latency_ms, input_tokens, output_tokens, cost_usd, error)` is used positionally in Tasks 7, 9, 10, 11, 12, 13 tests in that order. `Totals` positional order in Task 14's test matches Task 4's dataclass. `RunMeta`, `EventSource`, `MODE_LIVE`, `MODE_REPLAY` are defined in Task 13 and imported in Task 14. `REASONING_EFFORT`, `MODEL_ID` names match between Task 12 code and tests. ✓
- Placeholder scan: the only deferred item is Task 17's dependency on the user's keys, which is called out explicitly with the fallback for the README. ✓
