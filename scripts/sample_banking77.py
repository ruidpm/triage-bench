"""One-off: sample 25 tickets per intent from the Banking77 test split into data/triage.csv.

Standard-library only, no extra dependency required.
Run: `uv run python scripts/sample_banking77.py`

Banking77 is CC-BY-4.0, PolyAI (Casanueva et al., 2020),
source: https://github.com/PolyAI-LDN/task-specific-datasets

The Hugging Face `PolyAI/banking77` dataset card wraps this same CSV in a
Python loading script, which `datasets>=4` refuses to execute (and the repo
has no `refs/convert/parquet` fallback branch either). So this script fetches
PolyAI's canonical CSV directly from GitHub, pinned to the one commit that
has ever touched it, instead of depending on the `datasets` package.

PolyAI's test.csv is grouped by category, so writing rows in id order would
replay 25 identical intents in a row on the live dashboard. Rows are written
in a deterministic seeded shuffle instead (ids themselves stay the original
0-based row index from the source file).
"""

import csv
import io
import random
import urllib.request
from pathlib import Path

from triage_bench.domain.task import TRIAGE

LABELS = TRIAGE.label_names()
REPO = "PolyAI-LDN/task-specific-datasets"
# The only commit that has ever touched banking_data/test.csv (verified via
# `GET https://api.github.com/repos/PolyAI-LDN/task-specific-datasets/commits?path=banking_data/test.csv`),
# pinned here so the sample is reproducible regardless of upstream changes.
COMMIT_SHA = "9d081458ff52e53cf7e848f414e6e9344e4e6696"
TEST_CSV_URL = f"https://raw.githubusercontent.com/{REPO}/{COMMIT_SHA}/banking_data/test.csv"
SEED = 20260922
PER_LABEL = 25
OUT = Path("data/triage.csv")


def main() -> None:
    with urllib.request.urlopen(TEST_CSV_URL) as response:
        text = response.read().decode("utf-8")

    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    if header != ["text", "category"]:
        raise SystemExit(f"unexpected header {header}, expected ['text', 'category']")

    wanted = set(LABELS)
    by_label: dict[str, list[tuple[int, str]]] = {label: [] for label in LABELS}
    for index, row in enumerate(reader):
        text_, category = row
        label = category.lower()
        if label in wanted:
            by_label[label].append((index, text_))

    rng = random.Random(SEED)
    chosen: list[tuple[int, str, str]] = []
    for label, candidates in by_label.items():
        if len(candidates) < PER_LABEL:
            raise SystemExit(f"{label}: only {len(candidates)} candidates, need {PER_LABEL}")
        for index, sampled_text in rng.sample(candidates, PER_LABEL):
            chosen.append((index, sampled_text, label))
    # Source rows are grouped by category; shuffle with the same seeded RNG
    # (continuing its state after sampling) so the written order is mixed
    # but still fully reproducible from SEED alone. Ids remain the original
    # 0-based row index from the source file.
    rng.shuffle(chosen)

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "text", "label"])
        writer.writerows(chosen)
    print(f"wrote {len(chosen)} tickets to {OUT}")


if __name__ == "__main__":
    main()
