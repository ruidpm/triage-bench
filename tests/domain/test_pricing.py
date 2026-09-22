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
