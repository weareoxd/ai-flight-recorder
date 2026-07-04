from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from ai_flight_recorder.schema import ConfidenceSignal, ConfidenceSource, DecisionRecord, Flag


def test_minimal_record_gets_sane_defaults():
    record = DecisionRecord(decision_type="loan_eligibility", subject_ref="cust_1")
    assert record.decision_id
    assert record.timestamp.tzinfo is not None
    assert record.options == []
    assert record.outcome is None
    assert record.confidence == []
    assert record.flags == []
    assert record.review_required is False
    assert record.integrity_hash is None


def test_decision_type_and_subject_ref_are_required():
    with pytest.raises(ValidationError):
        DecisionRecord(subject_ref="cust_1")
    with pytest.raises(ValidationError):
        DecisionRecord(decision_type="loan_eligibility")


def test_decision_ids_are_unique_by_default():
    r1 = DecisionRecord(decision_type="x", subject_ref="cust_1")
    r2 = DecisionRecord(decision_type="x", subject_ref="cust_1")
    assert r1.decision_id != r2.decision_id


def test_naive_timestamp_is_rejected():
    with pytest.raises(ValidationError):
        DecisionRecord(
            decision_type="x", subject_ref="cust_1", timestamp=datetime(2026, 1, 1)
        )


def test_non_utc_timestamp_is_normalized_to_utc():
    from datetime import timedelta

    tz = timezone(timedelta(hours=5))
    record = DecisionRecord(
        decision_type="x", subject_ref="cust_1", timestamp=datetime(2026, 1, 1, tzinfo=tz)
    )
    assert record.timestamp.tzinfo == timezone.utc


@pytest.mark.parametrize(
    "raw_subject_ref",
    [
        "jane.doe@example.com",
        "555-123-4567",
        "123-45-6789",
    ],
)
def test_subject_ref_rejects_raw_pii(raw_subject_ref):
    with pytest.raises(ValidationError) as exc_info:
        DecisionRecord(decision_type="x", subject_ref=raw_subject_ref)
    assert raw_subject_ref not in str(exc_info.value)


def test_subject_ref_accepts_opaque_identifier():
    record = DecisionRecord(decision_type="x", subject_ref="cust_abc123")
    assert record.subject_ref == "cust_abc123"


def test_confidence_signal_none_source_forbids_value():
    with pytest.raises(ValidationError):
        ConfidenceSignal(source=ConfidenceSource.NONE, value=0.5)


def test_confidence_signal_none_source_allows_missing_value():
    signal = ConfidenceSignal(source=ConfidenceSource.NONE)
    assert signal.value is None


@pytest.mark.parametrize("source", [ConfidenceSource.LOGPROBS, ConfidenceSource.SELF_REPORT])
def test_confidence_signal_non_none_source_requires_value(source):
    with pytest.raises(ValidationError):
        ConfidenceSignal(source=source)


def test_confidence_signal_non_none_source_accepts_value():
    signal = ConfidenceSignal(source=ConfidenceSource.LOGPROBS, value=0.9)
    assert signal.value == 0.9


def test_flag_requires_rule_name_and_reason():
    flag = Flag(rule_name="low_confidence", reason="below threshold")
    assert flag.rule_name == "low_confidence"
    assert flag.reason == "below threshold"
    with pytest.raises(ValidationError):
        Flag(rule_name="low_confidence")
