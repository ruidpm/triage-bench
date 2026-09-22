import pytest

from triage_bench.infrastructure.jev_decider import JevDecider, NotConfiguredError


def test_jev_slot_is_explicitly_unavailable() -> None:
    with pytest.raises(NotConfiguredError, match="signups"):
        JevDecider()
