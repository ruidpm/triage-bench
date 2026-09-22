# Triage Bench

Triage Bench sends the same customer support messages, one at a time, to a small local decision model and to two hosted LLMs, then streams the results to a live web dashboard: accuracy, latency, cost and calibration, ticket by ticket. The task is intent triage on 200 real banking messages across 8 intents, and a recorded run is included so you can replay it without API keys.

![Triage Bench dashboard at ticket 120 of 200, replaying the sample run: three contestant cards (Von 96.7% accuracy at 176 ms p50, Claude Haiku 4.5 100.0% at 906 ms, GPT-5.6 Luna 99.2% at 1214 ms), a log-scale latency chart with Von's line far below the two LLMs, the current ticket, and the confidence-gated routing panel showing Von handling 95% of tickets locally](docs/screenshots/dashboard-mid-run.png)

## What is being compared

| contestant | model | where it runs | price per 1M tokens (in / out) |
|---|---|---|---|
| Von | [`wfzyx/von-1.0`](https://huggingface.co/wfzyx/von-1.0) via `von-sdk` 1.0.1 | in-process on your machine | free (local) |
| Claude Haiku 4.5 | `claude-haiku-4-5` | Anthropic API | $1.00 / $5.00 |
| GPT-5.6 Luna | `gpt-5.6-luna` | OpenAI API | $0.20 / $1.20 |

Prices were checked on 2026-09-22 and live in one module, `src/triage_bench/domain/pricing.py`.

A **System One model** is a classifier-style decision model: you give it some text, a set of choices and an instruction, and it returns one choice plus a probability for each choice in a single forward pass. It does not generate text, so there is nothing to parse and no output tokens to pay for. [Von](https://github.com/wfzyx/von) is an open-source, 395M-parameter ModernBERT model of this kind. Here it stands in for TypeSafe's Jev, the commercial model in the same class, because Jev was not accepting signups when this was built. The Jev slot exists (`src/triage_bench/infrastructure/jev_decider.py`) but is not wired up.

## Results

The committed run, `runs/sample.jsonl`, was recorded on 2026-09-22 on an Apple M2 laptop with 16 GB of memory. Von ran on the M2's GPU through PyTorch MPS, and both LLMs were called over a home internet connection. This is `uv run triage-bench report`, pasted verbatim:

```
| contestant | accuracy | p50 ms | p95 ms | cost | ECE |
|---|---|---|---|---|---|
| von | 97.0% | 174 | 244 | $0.0000 | 0.025 |
| haiku | 100.0% | 897 | 1243 | $0.1256 | 0.053 |
| luna | 99.5% | 1199 | 1941 | $0.0164 | 0.007 |

Routing at threshold 0.80: von handled 97% locally, rest to haiku.
  routed:   accuracy 97.5%, cost $0.0038
  haiku alone: accuracy 100.0%, cost $0.1256
  cost saving: 97%
```

- **accuracy**: share of the 200 tickets where the answer matched the true label. An API error counts as wrong. There were no errors in this run.
- **p50 ms / p95 ms**: median and 95th-percentile latency per ticket, in milliseconds, measured on the client. For the APIs this includes the network round trip.
- **cost**: US dollars for the whole run, computed from the token counts each API reported. Von is local, so its cost is zero.
- **ECE**: expected calibration error, which measures how far a contestant's stated confidence is from how often it is actually right (10 equal-width confidence bins). 0 means perfectly calibrated. Lower is better.
- **Routing**: a what-if computed from the same run, with no new calls. Use Von's answer when its confidence is at least the threshold, otherwise use Haiku's. At 0.80, Von answers 97% of tickets, accuracy lands between the two (97.5%), and the Haiku bill falls by 97%.

![The finished sample run at 200 of 200 with the Start button reading Finished: final totals on all three cards match the report above, the latency chart shows the last 50 tickets, and the recent-tickets feed lists each contestant's verdict per ticket](docs/screenshots/dashboard-finished.png)

The routing threshold is a slider in the dashboard's header. Moving it re-computes the routing panel in the browser, using the same rule as the `report` command (`--threshold`). At 0.95, Von keeps 93% of tickets, routed accuracy rises to 98.5%, and the saving drops to 93%:

![The finished run with the routing threshold slider moved to 0.95: the routing panel reads 93% handled locally, 98.5% routed accuracy, $0.0088 routed cost against $0.1256 for Haiku alone, a 93% cost saving](docs/screenshots/dashboard-threshold-095.png)

## Quick start: replay (no keys, no network)

```bash
git clone https://github.com/ruidpm/triage-bench && cd triage-bench
curl -LsSf https://astral.sh/uv/install.sh | sh    # only if you do not have uv
uv sync
uv run triage-bench replay
```

Open <http://127.0.0.1:8000> and press **Start**. The Speed slider sets replay pace (ticks per second) before you start. Use `--port` if 8000 is taken, or `--run path/to/run.jsonl` to replay a different run.

`uv sync` creates a virtualenv of about 800 MB, most of it PyTorch. Von needs PyTorch, and it is a regular dependency, so replay installs it too even though replay never loads the model. PyTorch is the one heavy dependency.

To print the results table instead of opening the dashboard, run `uv run triage-bench report`. Add `--threshold 0.9` to change the routing threshold.

## Quick start: live

1. Copy `.env.example` to `.env` and fill in `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`. `.env` is git-ignored.
2. Start a live run. uv loads the keys from the file, so they never go into your shell:

   ```bash
   uv run --env-file .env triage-bench live
   ```

3. Open <http://127.0.0.1:8000> in **one** tab and press **Start**. Each connection to the page's stream starts the whole run again and pays for it again, so the page won't restart a live run. Decisions are written to `runs/<timestamp>.jsonl` as they arrive (or to `--out path`), and you can replay that file afterwards.

What to expect:

- **Model download**: the first live run downloads Von's weights (about 1.6 GB) from Hugging Face into your Hugging Face cache (`~/.cache/huggingface/hub/` by default). This happens only once, and you don't need a Hugging Face account. To free the space later, run `uv run hf cache rm model/wfzyx/von-1.0`.
- **Hardware**: Von runs on the CPU or on Apple Silicon (MPS) out of the box, and uses an NVIDIA GPU automatically if PyTorch can see one. On the M2 above, a decision took 142 to 302 ms (p50 174 ms). The roughly 18 ms in Von's model card is a GPU figure.
- **Time**: calls are sequential, one ticket at a time and one contestant at a time. That works out to about 25 requests per minute per provider, which is under typical rate limits. A full 200-ticket run took about 8.5 minutes.
- **Cost**: the sample run cost about $0.14 in total ($0.1256 for Haiku, $0.0164 for Luna).
- Use `--tickets path.csv` to run your own tickets (see below).

**Troubleshooting**

- `error: missing environment variables: ...` means a key is not set. Check that `.env` has both keys and that you passed `--env-file .env`.
- Every Haiku ticket fails with `400 ... This API key is not scoped to a workspace, so this request must include the anthropic-workspace-id header`: create the Anthropic API key inside a workspace in the Anthropic Console and use that one.
- If a provider rate-limits you, the affected tickets show as errors (a grey "!" verdict). The run continues. This is deliberate (see "How it stays fair").

## Reading the dashboard

- **Header**: run name, mode badge (LIVE or REPLAY), tickets processed, Start/Pause, the replay speed slider, and the routing threshold slider.
- **Contestant cards**: the current ticket's verdict with a correct/wrong mark and the stated confidence, then an accuracy ring, p50 latency, cost so far ("local" for Von), ECE and error count, all cumulative.
- **Current ticket**: the message being classified and its true label.
- **Latency chart**: per-ticket latency for each contestant over the last 50 tickets, on a log scale so Von's line and the LLMs' lines both stay readable. Gaps are errors.
- **Confidence-gated routing**: what the Von-first, Haiku-fallback setup would have scored so far at the chosen threshold, next to Haiku alone.
- **Recent tickets**: the last 12 tickets, each contestant's answer marked correct or wrong. An error shows its message inline.

The page uses a single column on phones:

<img src="docs/screenshots/dashboard-phone.png" width="300" alt="The dashboard on a 390-pixel-wide phone at ticket 120 of 200: header controls stacked above the three contestant cards, followed by the current ticket, the latency chart and the routing panel in one column">

## Use your own tickets

Tickets are a CSV with the header `id,text,label`:

```csv
id,text,label
1834,My card payment did not complete.,declined_card_payment
11,How long does a card delivery take?,card_arrival
```

`id` is an integer, `text` must not be blank, and `label` must be one of the eight intents: `lost_or_stolen_card`, `card_arrival`, `declined_card_payment`, `refund_not_showing_up`, `exchange_rate`, `top_up_failed`, `passcode_forgotten`, `terminate_account`. Run it with `uv run --env-file .env triage-bench live --tickets my_tickets.csv`. The loader checks every row before any call is made, and if a row is bad it exits with the row numbers and the problems.

The labels and the one-line description each contestant sees are defined in one place, `src/triage_bench/domain/ticket.py`. To use a different label set, edit `INTENTS` there. The prompt, Von's choices and the LLMs' output schema are all built from it.

The bundled `data/tickets.csv` is 25 messages per intent from the Banking77 test split. `scripts/sample_banking77.py` samples them with a fixed seed from PolyAI's canonical `test.csv`, pinned to one upstream commit, using only the standard library. Rows are written in a seeded shuffled order so the dashboard doesn't show 25 tickets of the same intent in a row. Each `id` is the row's index in the upstream file.

## How it stays fair

- Every contestant gets the same instruction text and the same eight label descriptions (`src/triage_bench/application/prompt.py`).
- Both LLMs answer through structured output with the same schema, `{label: one of the 8 intents, confidence: number 0..1}`. There is no extended thinking (Anthropic: no `thinking` parameter; OpenAI: `reasoning.effort = "none"`), no tools, and `max_tokens` is 64.
- Latency is wall-clock time measured on the client around each call, so network time counts for the APIs and model time counts for Von. Von's one-off model load and warm-up happen before the run and are not counted.
- The SDKs' automatic retries are turned off (`SDK_MAX_RETRIES = 0` in `cli.py`). A rate limit or server error shows up as a visible error that counts as wrong, instead of a silent retry that inflates the measured latency.
- Cost comes from the token counts each API returns, multiplied by the price table. Von's cost is zero, and the dashboard labels it "local".
- All contestants see the tickets in the same order, one ticket at a time, and the dashboard advances only when all three have answered.

## Caveats

- **Von is a stand-in, not Jev.** Nothing here measures Jev. Von's own model card reports 72.0% macro accuracy on the broad 49-task jabr v2 benchmark, so a narrow 8-intent task like this one flatters it.
- **Laptop latency is not GPU latency.** Von's 174 ms p50 here is Apple M2 through MPS. The ~18 ms in its model card is measured on a GPU.
- **LLM confidence is self-reported.** The LLMs write their confidence as a number in their answer. It is not a probability the model computed. The ECE column measures how well those numbers track reality, so poor LLM calibration would be a result, not a bug.
- **200 tickets is a demo, not a paper.** There is no significance testing. One ticket is half a percentage point of accuracy.
- **Live latency includes your network.** API latency depends on where you run it from, and on the providers' load at that moment.

## Development

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run mypy src
```

CI runs the same three checks on every push and pull request. No test touches the network or loads the Von model.

The code follows a strict layering. `domain/` (tickets, decisions, metrics, pricing, routing) imports only the standard library and pydantic. `application/` (the shared prompt, the lockstep runner, replay) imports only `domain/`. `infrastructure/` (the Von, Claude and OpenAI adapters, CSV and JSONL files) and `web/` (FastAPI and the static dashboard) depend inward on both, never the other way round. Adapters never raise: a failed call becomes a `Decision` with its `error` set.

The design spec and the implementation plan are in `docs/superpowers/`. The README screenshots are produced by `scripts/capture_screenshots.mjs`, a dependency-free Node script that drives headless Chrome against a replay (usage is in its header).

## Next experiments

- Haiku with extended thinking on, as a fourth column, to show what the extra latency and cost buy.
- The full Banking77 label set (77 intents) instead of 8, which should be much harder for a small model.
- A real Jev adapter in the existing slot, once TypeSafe reopens signups.

## Licence and attribution

This repository is MIT licensed (see `LICENSE`).

The tickets come from **Banking77** by PolyAI, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and taken from [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets). `data/tickets.csv` is a 200-message sample of the test split, with message text and labels unchanged.

```bibtex
@inproceedings{Casanueva2020,
  author    = {I{\~{n}}igo Casanueva and Tadas Temcinas and Daniela Gerz and Matthew Henderson and Ivan Vulic},
  title     = {Efficient Intent Detection with Dual Sentence Encoders},
  year      = {2020},
  booktitle = {Proceedings of the 2nd Workshop on NLP for ConvAI - ACL 2020},
  url       = {https://arxiv.org/abs/2003.04807}
}
```

**Von** (`wfzyx/von-1.0`, `von-sdk`) is by wfzyx, licensed under Apache-2.0. Its weights are downloaded at runtime and are not redistributed here.
