# Triage Bench

A local System One decision model against two hosted LLMs on two tasks — 8-way support-ticket intent triage and binary customer-review sentiment — streamed item by item to a live dashboard.

![Dashboard at ticket 120 of 200: cards for Von (96.7%, 176 ms p50), Claude Haiku 4.5 (100.0%, 906 ms) and GPT-5.6 Luna (99.2%, 1214 ms), a log-scale latency chart, and the routing panel with Von handling 95% locally](docs/screenshots/dashboard-mid-run.png)

## Contestants

| contestant | model | runs | price per 1M tokens (in / out) |
|---|---|---|---|
| Von | [`wfzyx/von-1.0`](https://huggingface.co/wfzyx/von-1.0) (`von-sdk` 1.0.1) | locally | free |
| Claude Haiku 4.5 | `claude-haiku-4-5` | Anthropic API | $1.00 / $5.00 |
| GPT-5.6 Luna | `gpt-5.6-luna` | OpenAI API | $0.20 / $1.20 |

Von is an open-source 395M ModernBERT System One model (returns a choice plus probabilities, no text generation), standing in for TypeSafe's Jev, whose signups are closed.

## Results

ECE is expected calibration error: lower is better. Routing keeps Von's answer when its confidence is at least the threshold, otherwise uses Haiku's.

### Triage: 200 Banking77 tickets

200 Banking77 tickets, 2026-09-22, Apple M2 16 GB (Von on MPS), total API cost ≈ $0.14, 0 errors. Table as printed by `uv run triage-bench report`:

| contestant | accuracy | p50 ms | p95 ms | cost | ECE |
|---|---|---|---|---|---|
| von | 97.0% | 174 | 244 | $0.0000 | 0.025 |
| haiku | 100.0% | 897 | 1243 | $0.1256 | 0.053 |
| luna | 99.5% | 1199 | 1941 | $0.0164 | 0.007 |

The report's routing summary at threshold 0.80, restated:

- Handled locally by Von: 97%, rest to Haiku.
- Routed: accuracy 97.5%, cost $0.0038.
- Haiku alone: accuracy 100.0%, cost $0.1256.
- Cost saving: 97%.

### Sentiment: 200 Amazon Polarity reviews

200 reviews, 100 negative and 100 positive, 2026-09-23, Apple M2 16 GB (Von on MPS), total API cost ≈ $0.12, 0 errors. Table as printed by `uv run triage-bench report --task sentiment`:

| contestant | accuracy | p50 ms | p95 ms | cost | ECE |
|---|---|---|---|---|---|
| von | 98.0% | 136 | 216 | $0.0000 | 0.024 |
| haiku | 97.0% | 1049 | 1465 | $0.1071 | 0.039 |
| luna | 98.5% | 1136 | 1873 | $0.0146 | 0.007 |

Routing summary at threshold 0.80, restated:

- Handled locally by Von: 99.5% (199 of 200), rest to Haiku.
- Routed: accuracy 98.0%, cost $0.0006.
- Haiku alone: accuracy 97.0%, cost $0.1071.
- Cost saving: 99%.

## Quick start: replay (no keys)

```bash
git clone https://github.com/ruidpm/triage-bench && cd triage-bench
uv sync
uv run triage-bench replay        # open http://127.0.0.1:8000, press Start
uv run triage-bench replay --task sentiment   # the review run
```

`uv sync` installs PyTorch (Von's dependency): about 800 MB on macOS; several GB on Linux (CUDA PyTorch).

## Quick start: live

```bash
cp .env.example .env              # fill in ANTHROPIC_API_KEY and OPENAI_API_KEY
uv run --env-file .env triage-bench live
```

- The Anthropic key must be scoped to a workspace; an unscoped key fails with a 400 asking for `anthropic-workspace-id`.
- The first run downloads Von's weights (~1.6 GB) to `~/.cache/huggingface/hub/`; remove them with `uv run hf cache rm model/wfzyx/von-1.0`.
- One live run per process: a reload or second tab is refused; restart the command for another run. Results go to `runs/<timestamp>.jsonl`.
- Own items: `--tickets file.csv` with header `id,text,label`; labels from the task in `src/triage_bench/domain/task.py`; pick the task with `--task triage|sentiment`.

## Screenshots

| Finished run | Threshold at 0.95 | Phone (390 px) | Sentiment run |
|---|---|---|---|
| ![Finished run at 200 of 200 with final totals, latency chart of all 200 tickets and the scrollable tickets feed](docs/screenshots/dashboard-finished.png) | ![Routing threshold slider at 0.95: 93% handled locally, 98.5% routed accuracy, 93% cost saving](docs/screenshots/dashboard-threshold-095.png) | ![Single-column phone layout at ticket 120: controls, stacked cards, chart and routing panel](docs/screenshots/dashboard-phone.png) | ![Finished sentiment run at 200 of 200 with the review feed](docs/screenshots/sentiment/dashboard-finished.png) |

## Fairness

- Same instruction and label descriptions for every contestant.
- LLMs: structured output `{label, confidence}`, no thinking, no tools, 64 max tokens.
- Latency is client-side wall clock per call; Von's model load is excluded.
- SDK auto-retries off, so rate limits show as errors, not inflated latency.
- Cost from each API's reported token counts.
- Same ticket order, sequential calls, all three answer before the next ticket.

## Caveats

- Von is a stand-in; nothing here measures Jev.
- Von's model card cites ~18 ms on GPU; this laptop measured 174 ms p50.
- LLM confidence is self-reported, not a computed probability.
- 200 tickets on 8 intents and 200 reviews on 2 labels are demos; no significance testing.
- Live API latency includes your network.
- Where the model class stops, from throwaway spikes on 2026-09-23 (not part of the benchmark): on matched prompt-injection boundary pairs (72 rows from `3nesdeniz/agentic-prompt-injection-boundary-pairs`) Von scored 43%, below chance, giving attacks and their benign twins the same probability; on three-way support-ticket priority (90 rows from a synthetic set whose labels did not track the text) it scored 46% at a median confidence of 0.97. It reads labels off the words of a text; it does not judge actions or fuzzy middles.

## Development

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run mypy src
node --test 'tests/web/js/*.test.mjs'   # dashboard JS, Node 22+
```

## Licence and attribution

MIT. Tickets sampled from [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) by PolyAI, CC BY 4.0:

```bibtex
@inproceedings{Casanueva2020,
  author    = {I{\~{n}}igo Casanueva and Tadas Temcinas and Daniela Gerz and Matthew Henderson and Ivan Vulic},
  title     = {Efficient Intent Detection with Dual Sentence Encoders},
  year      = {2020},
  booktitle = {Proceedings of the 2nd Workshop on NLP for ConvAI - ACL 2020},
  url       = {https://arxiv.org/abs/2003.04807}
}
```

Reviews sampled from [Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity), Apache-2.0:

- Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. NIPS 2015.
- Julian McAuley, Jure Leskovec. Hidden factors and hidden topics: understanding rating dimensions with review text. RecSys 2013.

Von (`wfzyx/von-1.0`, `von-sdk`) by wfzyx, Apache-2.0.
