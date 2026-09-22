"""Three real tickets through every contestant. Costs a fraction of a cent. Not run in CI.

Usage:  uv run --env-file .env python scripts/smoke_live.py
"""

import sys
from pathlib import Path

from triage_bench.cli import build_claude_client, build_openai_client
from triage_bench.infrastructure.claude_decider import ClaudeDecider
from triage_bench.infrastructure.csv_tickets import load_tickets
from triage_bench.infrastructure.openai_decider import OpenAIDecider
from triage_bench.infrastructure.von_decider import VonDecider

SMOKE_TICKETS = 3


def main() -> int:
    tickets = load_tickets(Path("data/tickets.csv"))[:SMOKE_TICKETS]
    von = VonDecider.from_sdk()
    von.warm_up()
    deciders = [von, ClaudeDecider(build_claude_client()), OpenAIDecider(build_openai_client())]
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
