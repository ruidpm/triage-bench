"""Placeholder slot for TypeSafe's Jev. Not wired: TypeSafe closed signups on 2026-09-22."""

CONTESTANT = "jev"
SPEC_SECTION = "docs/superpowers/specs/2026-09-22-triage-bench-design.md section 4"


class NotConfiguredError(Exception):
    pass


class JevDecider:
    name = CONTESTANT

    def __init__(self) -> None:
        raise NotConfiguredError(
            "Jev is not configured: TypeSafe is not accepting signups. "
            f"When a key is available, implement decide() per {SPEC_SECTION} "
            "using the same Verdict schema, and add 'jev' to the price table."
        )
