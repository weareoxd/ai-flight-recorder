from datetime import datetime, timezone

from ai_flight_recorder.hashing import compute_integrity_hash, verify
from ai_flight_recorder.schema import DecisionRecord


def _make_record(**overrides) -> DecisionRecord:
    defaults = dict(
        decision_id="dec-001",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        decision_type="loan_eligibility",
        subject_ref="cust_12345",
        options=["approve", "deny"],
        outcome="approve",
    )
    defaults.update(overrides)
    return DecisionRecord(**defaults)


def test_hash_is_deterministic_for_identical_content():
    r1 = _make_record()
    r2 = _make_record()
    assert compute_integrity_hash(r1) == compute_integrity_hash(r2)


def test_hash_changes_when_captured_field_changes():
    r1 = _make_record(outcome="approve")
    r2 = _make_record(outcome="deny")
    assert compute_integrity_hash(r1) != compute_integrity_hash(r2)


def test_hash_excludes_the_integrity_hash_field_itself():
    r = _make_record()
    h = compute_integrity_hash(r)
    r.integrity_hash = h
    # Recomputing after storing the hash must be stable — the stored value
    # itself is not part of its own input.
    assert compute_integrity_hash(r) == h

    r.integrity_hash = "some-other-stale-value"
    assert compute_integrity_hash(r) == h


def test_verify_ok_when_untampered():
    r = _make_record()
    r.integrity_hash = compute_integrity_hash(r)
    result = verify(r)
    assert result.ok is True
    assert result.expected == r.integrity_hash


def test_verify_detects_tampering():
    r = _make_record()
    r.integrity_hash = compute_integrity_hash(r)

    r.outcome = "deny"  # tamper with a captured field after the fact
    result = verify(r)
    assert result.ok is False
    assert result.actual == r.integrity_hash
    assert result.expected != r.integrity_hash


def test_verify_fails_when_no_hash_recorded_yet():
    r = _make_record()
    result = verify(r)
    assert result.ok is False
    assert result.actual is None
