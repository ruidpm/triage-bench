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
