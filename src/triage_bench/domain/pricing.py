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
