# Triage Bench — design

Date: 2026-09-22
Status: approved for planning

## 1. Purpose

Show, live and on video, how a "System One" decision model compares with standard
LLMs on a narrow, realistic task: classifying customer support messages into a
fixed set of intents. Three contestants answer the same question on the same
tickets. A web dashboard tracks accuracy, latency, cost and calibration as the run
progresses.

The System One contestant is **Von**, an open-source 395M-parameter encoder model
that runs locally. It stands in for TypeSafe's Jev, which is not accepting signups
at the time of writing. The design keeps a slot for Jev behind the same interface.

Audience: engineers at the author's company evaluating whether this model class
fits their workflows. The repo is public.

## 2. Non-goals for v1

- More than three contestants (Jev adapter is a stub with a documented interface).
- User-uploaded datasets through the UI. A documented CSV format is enough.
- Persisting runs anywhere except local JSONL files.
- Authentication, multi-user, deployment. This runs on one laptop.
- Statistical significance testing. 200 tickets is a demo, not a paper.

## 3. Dataset

Source: Banking77 (PolyAI, CC-BY-4.0), 13,083 real customer messages labelled with
77 fine-grained intents. Attribution goes in the README.

Subset: 8 intents, 25 messages each, 200 tickets total, sampled once with a fixed
seed by a script under `scripts/`. The sampled CSV is committed so every run uses
identical tickets and nobody needs the Hugging Face hub to reproduce results.

Chosen intents and the description every contestant receives:

| label | description |
|---|---|
| `lost_or_stolen_card` | Card was lost, stolen, or the customer wants it blocked |
| `card_arrival` | Asking when or whether an ordered card will arrive |
| `declined_card_payment` | A card payment was declined or refused at checkout |
| `refund_not_showing_up` | A refund was promised but has not appeared in the account |
| `exchange_rate` | Questions about exchange rates or currency conversion applied |
| `top_up_failed` | Adding money to the account failed or did not go through |
| `passcode_forgotten` | Forgot the app passcode or PIN and wants to reset it |
| `terminate_account` | Wants to close or delete their account |

CSV format (`data/tickets.csv`): `id,text,label`. `id` is a stable integer,
`label` is one of the eight strings above. The loader validates every row against
this schema and rejects the file with a clear error listing bad rows.

## 4. Contestants

| name | model | where it runs | price (in / out per 1M tokens) |
|---|---|---|---|
| Von | `wfzyx/von-1.0` via `von-sdk` 1.0.1 | in-process on the laptop | 0 / 0 |
| Claude Haiku 4.5 | `claude-haiku-4-5` | Anthropic API | 1.00 / 5.00 USD |
| GPT-5.6 Luna | `gpt-5.6-luna` | OpenAI API | 0.20 / 1.20 USD |

Prices are named constants in one module with the date they were checked.

Fairness rules, all enforced by the shared runner:

- Identical instruction text and identical label descriptions to every contestant.
- LLMs answer through structured output with the schema
  `{label: enum, confidence: number 0..1}`. No extended thinking, no tools,
  `max_tokens` capped at 64.
- Latency is wall-clock measured in the runner around each `decide` call, so
  network time counts for the APIs and model time counts for Von. Von's one-off
  model load happens before the run starts and is not counted.
- Cost is computed from token usage returned by each API. Von's cost is zero by
  construction and the UI labels it "local".
- Tickets are processed in the same order for every contestant, one ticket at a
  time, all three contestants per tick, so the dashboard advances in lockstep.

## 5. Architecture

Layers follow the dependency rule: inner layers know nothing about outer ones.

```
src/triage_bench/
  domain/          pure, no I/O, fully unit tested
    ticket.py      Ticket, Intent, INTENTS (label + description)
    decision.py    Decision (ticket_id, contestant, label, confidence,
                   latency_ms, input_tokens, output_tokens, cost_usd, error)
    metrics.py     accuracy, percentile, expected_calibration_error (10 bins),
                   reliability_bins
    pricing.py     price table + cost_usd(model, in_tokens, out_tokens)
    routing.py     confidence_gated_report(primary, fallback, threshold)
  application/
    decider.py     Decider protocol: name, decide(ticket) -> Decision
    runner.py      iterates tickets, calls each decider, yields per-tick
                   events, appends to a ResultSink
    replay.py      reads a JSONL run and yields the same events with pacing
  infrastructure/
    von_decider.py       wraps von.decide
    claude_decider.py    Anthropic SDK, structured output
    openai_decider.py    OpenAI SDK, structured output
    jev_decider.py       stub raising NotConfigured with a pointer to docs
    csv_tickets.py       load_tickets(path) -> list[Ticket] | LoadError
    jsonl_sink.py        append-only writer, one Decision per line per contestant
  web/
    app.py         FastAPI: serves static/ and the SSE stream
    static/        index.html, app.js, styles.css
  cli.py           `triage-bench live|replay|report`
```

### Event stream

The runner yields one `TickEvent` per ticket:

```
{ "tick": 17, "total": 200,
  "ticket": {"id": 4021, "text": "...", "label": "top_up_failed"},
  "decisions": {
     "von":   {"label": "...", "confidence": 0.91, "latency_ms": 98, "cost_usd": 0, "correct": true},
     "haiku": {...}, "luna": {...} },
  "totals": { "von": {"accuracy": 0.88, "p50_ms": 101, "p95_ms": 180,
                      "cost_usd": 0, "ece": 0.06}, ... } }
```

`totals` are recomputed by the domain metrics on every tick from the decisions so
far. The web layer forwards events verbatim as SSE `data:` lines. The frontend
does no metric math.

### Modes

- **live**: real calls. Requires `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`, checked
  at startup with a clear message naming the missing one. Writes
  `runs/<timestamp>.jsonl` as it goes.
- **replay**: reads a JSONL run and emits the same events, paced by
  `speed` (ticks per second, default 2). Needs no keys or network. The committed
  `runs/sample.jsonl` makes the repo demoable straight after cloning.
- **report**: prints a markdown summary table and the routing report for a run
  file. Used to fill the README numbers.

### Confidence-gated routing report

Computed after the fact from a run file, no new calls. For a threshold `t`
(default 0.80): Von's answer is kept where Von's confidence ≥ t, otherwise the
fallback's answer (Haiku by default) is used. Report: share handled locally,
resulting accuracy, resulting cost, and the same three numbers for "fallback
alone" so the saving is explicit. Threshold is a CLI flag and a UI slider.

## 6. Web UI

Single static page, no build step, served by FastAPI. Vanilla JavaScript, one
chart library loaded from a CDN. Dark theme designed for a 16:9 screen recording,
with a single-column layout below 768 px so it also works on a phone.

Sections, top to bottom:

1. **Header**: run name, mode badge (LIVE / REPLAY), tick counter, play/pause,
   speed control (replay only), routing-threshold slider.
2. **Contestant cards**, three across: name, current verdict with confidence,
   green or red border for the current ticket, accuracy ring, p50 latency,
   running cost, ECE.
3. **Current ticket**: the message text and its true label.
4. **Latency chart**: per-tick latency for all three, log scale, last 50 ticks.
5. **Feed**: last 12 tickets, each with the three verdicts coloured by correctness.
6. **Routing panel** (toggle): handled-locally share, accuracy and cost with and
   without routing at the chosen threshold.

Colours and typography are decided at implementation time following the
`dataviz` skill; the spec only fixes structure and behaviour.

## 7. Error handling

Errors are values in the domain. A `Decision` carries an optional `error` string.

- API failure after SDK retries: the Decision records the error, counts as
  incorrect, contributes no latency or cost, and the card shows a grey "error"
  verdict. The run continues.
- Structured-output response that fails schema validation: treated as an API
  failure for that ticket, logged with the raw response.
- Von model fails to load: the CLI exits before the run with the underlying error.
- Missing API key in live mode: exit before the run naming the variable.
- Bad tickets CSV: exit before the run listing the offending rows.
- Replay file missing or malformed: exit with the line number that failed.

No catch block swallows an error silently. Each one either records it on the
Decision or exits with context.

## 8. Testing

- Domain: table-driven pytest for every metric, pricing and routing function,
  including edge cases (zero decisions, all errors, one bin, threshold 0 and 1).
- Application: runner tested with three fake deciders that return scripted
  Decisions, asserting event shape, lockstep ordering and sink contents. Replay
  tested against a fixture JSONL.
- Infrastructure: adapters tested with the SDK clients replaced by fakes,
  asserting the request built (schema, max_tokens, no thinking) and the parsing of
  a good response, a schema-violating response and an API exception. The CSV
  loader tested with good and bad files.
- Web: one test that the SSE endpoint streams the fixture run's events unchanged.
- No test touches the network or loads the Von model. A separate opt-in
  `scripts/smoke_live.py` runs three real tickets and is not part of CI.

## 9. Repository hygiene

- Python 3.12+, `uv` for environment and lockfile, all dependencies pinned.
- Dependencies: `von-sdk`, `anthropic`, `openai`, `fastapi`, `uvicorn`,
  `pydantic`; dev: `pytest`, `ruff`, `mypy`. The one-off sampling script uses
  only the standard library: it fetches PolyAI's canonical Banking77 `test.csv`
  from GitHub, pinned to one commit, and writes rows in a seeded shuffled order
  (ids are the upstream row index).
- `.env.example` with the two key names. `.gitignore` covers `.env`, `runs/*`
  except `runs/sample.jsonl`, and the virtualenv.
- MIT license. Banking77 attribution and CC-BY-4.0 notice in the README.
- GitHub Actions: ruff, mypy, pytest on push and PR. No secrets in CI.
- README: what it is, a screenshot or GIF of the dashboard, quick start for
  replay (no keys) and live (keys), the results table from `runs/sample.jsonl`,
  the routing punchline, and a **Caveats** section stating plainly that Von is a
  395M stand-in, not Jev; that laptop latency is 60 to 200 ms, not the README's
  GPU numbers; and that LLM self-reported confidence is not a calibrated
  probability.

## 10. Open items resolved during brainstorming

- Game-based demos rejected: Von is a text encoder and fails at shell semantics and
  spatial reasoning (measured: it rated a disk-wipe command "safe" at 0.93).
- Terminal UI rejected in favour of the web dashboard for video capture.
- Jev deferred: TypeSafe closed signups on 2026-09-22.
