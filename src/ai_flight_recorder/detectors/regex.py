"""Zero-dependency regex-based PII detector.

Covers the common, high-confidence cases (email, phone, credit card with a Luhn
check, SSN/SIN, IP address) plus caller-supplied custom patterns. This is a
pragmatic net, not an exhaustive PII scanner — for higher recall, use the
`PresidioDetector` behind the `ai-flight-recorder[presidio]` extra instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ai_flight_recorder.detectors import Detector, PIIMatch, redact_matches

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"
)

# 13-19 digits, optionally grouped by spaces or dashes. Luhn-validated below to
# cut down on false positives against arbitrary long numbers.
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")

_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")  # US Social Security Number
_SIN_RE = re.compile(r"(?<!\d)\d{3}[ -]\d{3}[ -]\d{3}(?!\d)")  # Canadian SIN

_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)


def _luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass(frozen=True)
class CustomPattern:
    entity_type: str
    pattern: re.Pattern[str]


class RegexDetector(Detector):
    """Detects email, phone, credit card, SSN/SIN, and IPv4 by default.

    Pass `custom_patterns` to add caller-defined entity types, e.g.:

        RegexDetector(custom_patterns=[
            CustomPattern("employee_id", re.compile(r"\\bEMP-\\d{6}\\b")),
        ])
    """

    def __init__(self, custom_patterns: list[CustomPattern] | None = None) -> None:
        self._custom_patterns = custom_patterns or []

    def detect(self, text: str) -> list[PIIMatch]:
        if not text:
            return []

        matches: list[PIIMatch] = []

        for m in _EMAIL_RE.finditer(text):
            matches.append(PIIMatch("EMAIL", m.start(), m.end(), m.group()))

        for m in _PHONE_RE.finditer(text):
            matches.append(PIIMatch("PHONE", m.start(), m.end(), m.group()))

        for m in _SSN_RE.finditer(text):
            matches.append(PIIMatch("SSN", m.start(), m.end(), m.group()))

        for m in _SIN_RE.finditer(text):
            # A SIN and a plain 9-digit grouped number look identical; only the
            # separator/grouping differs from SSN's 3-2-4 grouping, so this is
            # a best-effort disambiguation, not a guarantee.
            if not _SSN_RE.fullmatch(m.group()):
                matches.append(PIIMatch("SIN", m.start(), m.end(), m.group()))

        for m in _CARD_RE.finditer(text):
            digits = re.sub(r"[ -]", "", m.group())
            if 13 <= len(digits) <= 19 and _luhn_valid(digits):
                matches.append(PIIMatch("CREDIT_CARD", m.start(), m.end(), m.group()))

        for m in _IPV4_RE.finditer(text):
            matches.append(PIIMatch("IP_ADDRESS", m.start(), m.end(), m.group()))

        for custom in self._custom_patterns:
            for m in custom.pattern.finditer(text):
                matches.append(PIIMatch(custom.entity_type, m.start(), m.end(), m.group()))

        return matches

    def redact(self, text: str) -> str:
        """Replace every detected match with `[REDACTED_<ENTITY_TYPE>]`."""
        return redact_matches(text, self.detect(text))
