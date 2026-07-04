"""PII detection interface.

ai-flight-recorder ships a zero-dependency `RegexDetector` by default. A `PresidioDetector`
is available behind the `ai-flight-recorder[presidio]` extra and implements the same
`Detector` protocol, so the rule engine and redaction logic are detector-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class PIIMatch:
    """One PII hit. `value` is the raw matched text — callers must not log or
    propagate it outside of building a redacted replacement."""

    entity_type: str
    start: int
    end: int
    value: str


@runtime_checkable
class Detector(Protocol):
    def detect(self, text: str) -> list[PIIMatch]:
        """Return every PII match found in `text`. Empty list if none."""
        ...


def redact_matches(text: str, matches: list[PIIMatch]) -> str:
    """Replace every match with `[REDACTED_<ENTITY_TYPE>]`. Shared by every
    `Detector` implementation's `redact()` so the placeholder format is consistent."""
    ordered = sorted(matches, key=lambda m: m.start)
    if not ordered:
        return text

    out: list[str] = []
    cursor = 0
    for m in ordered:
        if m.start < cursor:
            # Overlapping match (e.g. an SSN also caught as a credit card):
            # already covered by the redaction just emitted, but extend the
            # cursor to the overlap's full extent so any tail past the prior
            # match's end doesn't fall through to the trailing `text[cursor:]`
            # and leak unredacted.
            cursor = max(cursor, m.end)
            continue
        out.append(text[cursor : m.start])
        out.append(f"[REDACTED_{m.entity_type}]")
        cursor = m.end
    out.append(text[cursor:])
    return "".join(out)
