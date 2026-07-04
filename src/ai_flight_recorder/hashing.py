"""Integrity hashing for `DecisionRecord`.

The hash proves the *captured* decision was not altered after the fact. It does
NOT cover the human review verdict — that verdict never lives on the
`DecisionRecord` itself; it is recorded later, directly in Langfuse, as a score
a reviewer sets via the annotation UI. Excluding `integrity_hash` from its own
input is therefore sufficient to exclude everything added after capture.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ai_flight_recorder.schema import DecisionRecord


def _canonical_json(record: DecisionRecord) -> str:
    payload = record.model_dump(mode="json", exclude={"integrity_hash"})
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def compute_integrity_hash(record: DecisionRecord) -> str:
    """sha256 over every captured field except `integrity_hash` itself."""
    return hashlib.sha256(_canonical_json(record).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    expected: str
    actual: str | None


def verify(record: DecisionRecord) -> VerifyResult:
    """Recompute the integrity hash and compare it against `record.integrity_hash`.

    Reports a mismatch rather than raising, so callers can decide how to react
    (e.g. flag the record, log, alert) instead of the library dictating policy.
    """
    expected = compute_integrity_hash(record)
    actual = record.integrity_hash
    return VerifyResult(ok=(actual == expected), expected=expected, actual=actual)
