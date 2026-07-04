import logging
from datetime import datetime, timezone

from ai_flight_recorder.emit import MAX_PROPAGATED_VALUE_LEN, emit
from ai_flight_recorder.schema import ConfidenceSignal, ConfidenceSource, DecisionRecord, Flag


def _make_record(**overrides) -> DecisionRecord:
    defaults = dict(
        decision_id="dec-001",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        decision_type="loan_eligibility",
        subject_ref="cust_12345",
        options=["approve", "deny"],
        outcome="approve",
        jurisdiction="CA-ON",
        integrity_hash="deadbeef" * 8,
    )
    defaults.update(overrides)
    return DecisionRecord(**defaults)


def test_trace_id_derived_from_decision_id_seed(fake_langfuse_client):
    record = _make_record(decision_id="dec-abc")
    emit(record, fake_langfuse_client)
    assert fake_langfuse_client.create_trace_id_calls == ["dec-abc"]


def test_same_decision_id_yields_same_trace_id(fake_langfuse_client):
    r1 = _make_record(decision_id="dec-xyz")
    r2 = _make_record(decision_id="dec-xyz")
    result1 = emit(r1, fake_langfuse_client)
    result2 = emit(r2, fake_langfuse_client)
    assert result1.trace_id == result2.trace_id


def test_propagated_metadata_bucket_a(fake_langfuse_client):
    record = _make_record(flags=[Flag(rule_name="low_confidence", reason="x")])
    emit(record, fake_langfuse_client)
    call = fake_langfuse_client.propagate_attributes_calls[0]
    metadata = call["metadata"]
    assert metadata["decision_id"] == record.decision_id
    assert metadata["decision_type"] == "loan_eligibility"
    assert metadata["subject_ref"] == "cust_12345"
    assert metadata["jurisdiction"] == "CA-ON"
    assert metadata["integrity_hash"] == record.integrity_hash
    assert metadata["flags"] == "low_confidence"
    assert all(isinstance(v, str) for v in metadata.values())


def test_propagated_metadata_omits_absent_optional_fields(fake_langfuse_client):
    record = _make_record(jurisdiction=None, integrity_hash=None, flags=[])
    emit(record, fake_langfuse_client)
    metadata = fake_langfuse_client.propagate_attributes_calls[0]["metadata"]
    assert "jurisdiction" not in metadata
    assert "integrity_hash" not in metadata
    assert "flags" not in metadata


def test_oversized_value_is_truncated_with_marker_and_warned(fake_langfuse_client, caplog):
    long_jurisdiction = "X" * 500
    record = _make_record(jurisdiction=long_jurisdiction)
    with caplog.at_level(logging.WARNING, logger="ai_flight_recorder.emit"):
        emit(record, fake_langfuse_client)
    metadata = fake_langfuse_client.propagate_attributes_calls[0]["metadata"]
    assert len(metadata["jurisdiction"]) <= MAX_PROPAGATED_VALUE_LEN
    assert metadata["jurisdiction"].endswith("...[truncated]")
    assert "jurisdiction" not in metadata["jurisdiction"]  # not silently identical to input
    assert any("exceeding" in r.message for r in caplog.records)


def test_observation_input_output_bucket_b(fake_langfuse_client):
    record = _make_record(
        prompt=[{"role": "user", "content": "should I approve?"}],
        options=["approve", "deny"],
        outcome="approve",
    )
    emit(record, fake_langfuse_client)
    call = fake_langfuse_client.start_as_current_observation_calls[0]
    assert call["input"] == {
        "prompt": [{"role": "user", "content": "should I approve?"}],
        "options": ["approve", "deny"],
    }
    assert call["output"] == "approve"


def test_redacted_prompt_used_as_input_when_present(fake_langfuse_client):
    record = _make_record(
        prompt="contact jane.doe@example.com",
        redacted_prompt="contact [REDACTED_EMAIL]",
    )
    emit(record, fake_langfuse_client)
    call = fake_langfuse_client.start_as_current_observation_calls[0]
    assert call["input"]["prompt"] == "contact [REDACTED_EMAIL]"


def test_regular_metadata_bucket_d(fake_langfuse_client):
    record = _make_record(
        confidence=[ConfidenceSignal(source=ConfidenceSource.LOGPROBS, value=0.42)],
        flags=[Flag(rule_name="low_confidence", reason="confidence 0.42 below threshold")],
        destination="downstream_underwriting_system",
        model_version="2024-07-18",
    )
    emit(record, fake_langfuse_client)
    call = fake_langfuse_client.start_as_current_observation_calls[0]
    metadata = call["metadata"]
    assert metadata["confidence"] == [{"source": "logprobs", "value": 0.42, "note": None}]
    assert metadata["flags"] == [
        {"rule_name": "low_confidence", "reason": "confidence 0.42 below threshold"}
    ]
    assert metadata["destination"] == "downstream_underwriting_system"
    assert metadata["model_version"] == "2024-07-18"


def test_model_passed_as_native_generation_field(fake_langfuse_client):
    record = _make_record(model="gpt-4o-mini")
    emit(record, fake_langfuse_client)
    call = fake_langfuse_client.start_as_current_observation_calls[0]
    assert call["model"] == "gpt-4o-mini"
    assert call["as_type"] == "generation"


def test_review_required_score_written_with_idempotency_id(fake_langfuse_client):
    record = _make_record(decision_id="dec-777", review_required=True)
    result = emit(record, fake_langfuse_client)
    score_call = fake_langfuse_client.create_score_calls[0]
    assert score_call["name"] == "review_required"
    assert score_call["value"] == 1.0
    assert score_call["data_type"] == "BOOLEAN"
    assert score_call["score_id"] == "dec-777-review_required"
    assert score_call["trace_id"] == result.trace_id


def test_review_required_false_writes_zero(fake_langfuse_client):
    record = _make_record(review_required=False)
    emit(record, fake_langfuse_client)
    assert fake_langfuse_client.create_score_calls[0]["value"] == 0.0


def test_reemitting_same_decision_id_reuses_idempotency_id(fake_langfuse_client):
    record = _make_record(decision_id="dec-999")
    emit(record, fake_langfuse_client)
    emit(record, fake_langfuse_client)
    ids = [c["score_id"] for c in fake_langfuse_client.create_score_calls]
    assert ids[0] == ids[1] == "dec-999-review_required"


def test_emit_result_matches_fake_client(fake_langfuse_client):
    record = _make_record()
    result = emit(record, fake_langfuse_client)
    call = fake_langfuse_client.start_as_current_observation_calls[0]
    assert result.trace_id == call["trace_context"]["trace_id"]
    assert result.observation_id == "obs-1"


def test_emit_does_not_flush(fake_langfuse_client):
    record = _make_record()
    emit(record, fake_langfuse_client)
    assert fake_langfuse_client.flush_calls == 0
