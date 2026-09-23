# Sentiment task — design

Date: 2026-09-23
Status: awaiting review
Extends: `2026-09-22-triage-bench-design.md` (the triage task; everything there still holds)

## 1. Purpose

Add a second task to triage-bench so the benchmark shows the System One model on
two problems of the kind it is built for, not one. The task is binary sentiment
of customer product reviews. Same three contestants, same metrics, same routing
report, same dashboard, same fairness rules.

The README gains a second results table and one sentence of thesis: Von's
competence is reading a label off the words of a text, and it holds across intent
and sentiment. The caveats section states, with numbers, where that competence
stops.

Audience and repo status are unchanged: engineers at the author's company, public
GitHub repo.

## 2. Evidence behind the choice

Throwaway spikes run on 2026-09-23 with `von.decide`, one instruction and one
label set each, no repo code:

| task | rows | Von | chance | confidence informative? |
|---|---|---|---|---|
| ticket intent, 8-way (the shipped run) | 200 | 97.0% | 12.5% | yes, ECE 0.025 |
| Amazon Polarity, binary | 200 | 95.0% | 50% | yes: 98% of rows ≥ 0.80 at 95.9%; the 4 below at 50% |
| SST-2 sentiment, binary | 200 | 94.0% | 50% | yes |
| support-ticket priority, 3-way ordinal | 90 | 45.6% | 33% | no: median confidence 0.97 while wrong (labels also suspect) |
| prompt-injection boundary pairs, binary | 72 | 43.1% | 50% | no signal: mean p(attack) 0.32 on attacks, 0.33 on benign |
| sudoku, digit per cell | 58 | 13.8% | 11.1% | — |

Amazon Polarity is chosen over SST-2 because it is Apache-2.0 (SST-2's card says
`license: unknown`) and because customer reviews are closer to the audience's
work than film criticism. Binary is chosen over three-way or five-star because it
is the measured shape, and because the priority spike shows Von confidently wrong
on fuzzy middle classes.

## 3. Non-goals

- A third task, task-specific metrics or panels, or user-uploaded tasks.
- Changes to contestants, prices, fairness rules or the routing report.
- Renaming the repository or the `Ticket` dataclass (a review is a ticket
  internally; only the dashboard copy changes).
- Three-way or ordinal sentiment.

## 4. Dataset

Source: Amazon Polarity (`fancyzhx/amazon_polarity` on Hugging Face, Apache-2.0;
Zhang, Zhao & LeCun 2015, built on McAuley & Leskovec 2013). Binary labels derived
upstream from star ratings: 1–2 stars → `negative`, 4–5 stars → `positive`,
3-star reviews excluded. The test split has 400,000 rows and is already shuffled.

Subset: 100 `negative` + 100 `positive`, 200 reviews, sampled once with a fixed
seed by `scripts/sample_amazon_polarity.py` and committed as `data/sentiment.csv`.
Nobody needs the hub to reproduce results.

The script mirrors `scripts/sample_banking77.py`: standard library only. It reads
the first 1,000 rows of the test split through the Hugging Face datasets-server
rows API (10 paged calls of 100; the parquet files would need `pyarrow`, a new
dependency), samples 100 per label with `random.Random(SEED)`, shuffles with the
same RNG, and writes `id,text,label`. `id` is the upstream 0-based test-split row
index. `text` is `title`, a blank line, then `content`, exactly as spiked. The
rows API cannot be pinned to a revision, so the script records the dataset
revision SHA at sampling time (`9d9c45c18f8c3cf1b23a3c27917b60cbf28f3289`, last
modified 2024-01-09) in its docstring; the committed CSV is the source of truth,
as it is for triage.

`data/tickets.csv` is renamed `data/triage.csv` (`git mv`) so both files are
named by task. `scripts/sample_banking77.py` is updated to import its labels from
`domain.task.TRIAGE` and to write `data/triage.csv`; its seed and logic do not
change, so re-running it reproduces the committed file byte for byte.

Wording every contestant receives:

| | text |
|---|---|
| instructions | `Classify the sentiment of this customer product review.` |
| `negative` | `The customer is unhappy with the product.` |
| `positive` | `The customer is happy with the product.` |

## 5. Domain: `Task`

New module `domain/task.py`:

```python
@dataclass(frozen=True)
class Label:
    name: str
    description: str

@dataclass(frozen=True)
class Task:
    name: str                  # "triage" | "sentiment": CLI value, data/run file stem, run header
    item_noun: str             # "ticket" | "review": dashboard copy only
    instructions: str
    labels: tuple[Label, ...]

    def label_names(self) -> tuple[str, ...]: ...
    def validate(self, ticket: Ticket) -> None:
        """Raise ValueError naming the label when ticket.label is not one of labels."""

TRIAGE = Task(name="triage", item_noun="ticket",
              instructions="Classify the primary intent of this customer support message.",
              labels=(...the eight intents, moved verbatim from ticket.py...))
SENTIMENT = Task(name="sentiment", item_noun="review", instructions=..., labels=(...))
TASKS: dict[str, Task] = {t.name: t for t in (TRIAGE, SENTIMENT)}
DEFAULT_TASK = TRIAGE
```

`Task.__post_init__` rejects an empty label tuple, duplicate label names, and
blank names or descriptions.

`domain/ticket.py` keeps `Ticket(id, text, label)` and the blank-text check, and
stops validating the label: `Intent`, `INTENTS` and `LABELS` are removed from it.
Label validity is the task's business, checked at the boundaries (CSV loader, run
file reader). `Decision`, `metrics`, `routing` and `pricing` are untouched.

## 6. Application layer

- `prompt.py`: `label_descriptions_text(task)` and `llm_system_prompt(task)`.
  `LLM_MAX_TOKENS` unchanged.
- `verdict.py`: `verdict_model(task) -> type[BaseModel]` builds the structured
  output schema `{label: Enum of task.label_names(), confidence: 0..1}` with
  `pydantic.create_model`, so the enum still has one source of truth (the task).
  The module-level `IntentLabel` and `Verdict` go away.
- `runner.py`, `replay.py`, `decider.py`: unchanged. The runner never knew about
  intents.

## 7. Infrastructure

- `VonDecider(engine, task)`, `ClaudeDecider(client, task, model=...)`,
  `OpenAIDecider(client, task, model=...)`: each builds its choices / system
  prompt / verdict model once in `__init__`; `decide()` bodies are unchanged.
  `VonDecider.from_sdk(task)`.
- `csv_tickets.load_tickets(path, task)`: calls `task.validate(ticket)` per row;
  a bad label still fails with the row number in `TicketLoadError`.
- `jsonl_sink`: the header record becomes
  `{"kind": "tickets", "task": "<name>", "tickets": [...]}`. `write_header(task,
  tickets)`. `read_run(path)` returns `(task, tickets, decisions)`; a header
  without `task`, or with a name not in `TASKS`, is a `RunFileError` with the line
  number; every ticket in the header is validated against the task. The committed
  `runs/sample.jsonl` gets `"task": "triage"` added to its header line (decisions
  untouched) and is renamed `runs/triage-sample.jsonl`. No legacy branch for
  headers without a task.

## 8. CLI

- `live --task {triage,sentiment}` (default `triage`). `--tickets` defaults to
  `data/<task>.csv`. Deciders are built for that task; the run header records it.
- `replay --task ...` and `report --task ...` only pick the default `--run`
  (`runs/<task>-sample.jsonl`); the task actually used comes from the run file.
  Passing `--run` for a file whose header names a different task than `--task`
  is not an error: the file wins, and the CLI prints a note.
- Every command in the current README works unchanged: `triage-bench replay`
  replays triage.

## 9. Web

`/api/meta` adds `task` (name) and `item_noun`. `RunMeta` carries both.

The page substitutes the noun in every place that currently says "ticket": the
tick counter's label, the Restart button label, the current-item panel heading
and placeholder, the chart axis title, the feed heading, and the four screen-reader
announcements. The task name appears beside the mode badge. No new panels, no new
metrics; the frontend still does no metric math. If the substitution needs a
helper (e.g. capitalising the noun for headings), it lives in its own small module
with a Node test, like `pacer.js`.

## 10. Errors

Unchanged in kind, all errors are values or exit-with-context:

- Unknown label for the task in a CSV row → `TicketLoadError` listing rows.
- Run header without `task` / unknown task / ticket label not in the task →
  `RunFileError` with the line number.
- `--task` not in `TASKS` → argparse rejects it (choices).
- Everything else as in the triage spec (§7).

## 11. README

- Title line: "…on two tasks: 8-way support-ticket intent triage and binary
  customer-review sentiment…".
- Results: two tables, each with its dataset, date, machine and cost line, both
  printed by `triage-bench report`. Routing summary for each.
- Quick start unchanged, plus one line: `uv run triage-bench replay --task sentiment`.
- Licence and attribution: Amazon Polarity, Apache-2.0, with the two citations
  from its card.
- Caveats: keep the existing ones; add two sentences with the spike numbers for
  prompt-injection boundary pairs (43%, no signal) and ticket priority (46%,
  labels suspect), so the "where it stops" claim has evidence; state that both
  datasets are 200-row demos.
- Screenshots: one of the sentiment run added to the table.

## 12. Testing

TDD, RED → GREEN → REFACTOR per task. New or changed tests:

- Domain `task`: `validate` accepts a known label and raises naming an unknown
  one; `Task` rejects empty labels, duplicate names, blank text; `TASKS` keys
  equal task names; `TRIAGE.labels` are the eight intents in the shipped order.
- `ticket`: no longer raises on an unknown label (test updated); still raises on
  blank text.
- `prompt`, `verdict`: parametrised over `TASKS.values()` — each label appears
  once, enum members equal `label_names()`, confidence bounds, schema has the
  enum.
- Deciders: parametrised over both tasks for the request built (system prompt,
  schema class, choices dict for Von); error paths unchanged.
- `csv_tickets`: a sentiment label in the triage file fails with the row number,
  and vice versa.
- `jsonl_sink`: header round-trips the task; missing/unknown task is
  `RunFileError` with line number.
- Data files: `data/sentiment.csv` has 100/100, unique ids, longest same-label
  run ≤ 3, both labels in the first 10 rows; the existing triage test moves to
  `data/triage.csv`.
- CLI: `--task sentiment` picks `data/sentiment.csv` and
  `runs/sentiment-sample.jsonl`; the file's task wins over `--task` on replay.
- Web: `/api/meta` carries `task` and `item_noun`.
- Any new JS helper gets a `node --test` file.

No test touches the network or loads Von. The live sentiment run that produces
`runs/sentiment-sample.jsonl` is a manual step with keys via
`uv run --env-file .env`, done after the code lands; expected API cost
≈ $0.15–0.20.

## 13. Repository hygiene

- No new runtime dependency. The sampling script is standard library.
- `.gitignore` keeps ignoring `runs/*` and whitelists `runs/triage-sample.jsonl`
  and `runs/sentiment-sample.jsonl`.
- Commits: refactor to `Task` first (all existing tests green at the end of it),
  then the sentiment task, data, run file, README and screenshots, each atomic.
- Branch `sentiment-task`; PR to `main`.

## 14. Resolved during brainstorming

- Prompt injection (3nesdeniz boundary pairs) rejected as the second task: Von is
  at chance because both halves of a pair share topic and vocabulary and differ
  only in the requested action; the finding is kept as a caveat.
- Support-ticket priority rejected: the only sizeable corpus found is synthetic
  with priorities that do not track the text.
- SST-2 rejected on licence. Three-way and five-star sentiment rejected as
  unmeasured and off the measured shape.
- Task-as-data-files and parallel-modules approaches rejected in favour of a
  typed `Task` value.
