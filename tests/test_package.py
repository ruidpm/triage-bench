import triage_bench


def test_package_exposes_version() -> None:
    assert triage_bench.__version__ == "0.1.0"
