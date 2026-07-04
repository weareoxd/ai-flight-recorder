"""Cross-cutting assertion: raw PII must never surface in propagated metadata,
regular metadata, logs, exception messages, or subject_ref — regardless of
which layer (schema, rules, emit) is responsible for handling it."""

import json
import logging

import pytest
from pydantic import ValidationError

from ai_flight_recorder.capture import decision
from ai_flight_recorder.emit import emit
from ai_flight_recorder.schema import DecisionRecord

RAW_EMAIL = "jane.doe@example.com"
RAW_SSN = "123-45-6789"
RAW_PHONE = "555-123-4567"


def test_subject_ref_construction_error_never_echoes_raw_value():
    with pytest.raises(ValidationError) as exc_info:
        DecisionRecord(decision_type="x", subject_ref=RAW_EMAIL)
    assert RAW_EMAIL not in str(exc_info.value)


def test_pii_in_prompt_never_reaches_propagated_metadata(fake_langfuse_client):
    with decision(
        decision_type="loan_eligibility", subject_ref="cust_1", client=fake_langfuse_client
    ) as d:
        d.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"SSN {RAW_SSN}, email {RAW_EMAIL}"}],
            mock_response="approve",
        )
        d.outcome = "approve"

    metadata = fake_langfuse_client.propagate_attributes_calls[0]["metadata"]
    serialized = json.dumps(metadata)
    assert RAW_SSN not in serialized
    assert RAW_EMAIL not in serialized


def test_pii_in_prompt_never_reaches_flag_reason_in_regular_metadata(fake_langfuse_client):
    with decision(
        decision_type="loan_eligibility", subject_ref="cust_1", client=fake_langfuse_client
    ) as d:
        d.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"call me at {RAW_PHONE}"}],
            mock_response="approve",
        )
        d.outcome = "approve"

    call = fake_langfuse_client.start_as_current_observation_calls[0]
    serialized = json.dumps(call["metadata"])
    assert RAW_PHONE not in serialized


def test_redacted_input_sent_to_langfuse_never_contains_raw_pii(fake_langfuse_client):
    with decision(
        decision_type="loan_eligibility", subject_ref="cust_1", client=fake_langfuse_client
    ) as d:
        d.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"email {RAW_EMAIL}"}],
            mock_response="approve",
        )
        d.outcome = "approve"

    call = fake_langfuse_client.start_as_current_observation_calls[0]
    assert RAW_EMAIL not in json.dumps(call["input"])


def test_truncation_warning_log_never_contains_the_raw_value(fake_langfuse_client, caplog):
    long_value_with_pii = ("prefix " + f"{RAW_EMAIL} " * 20).strip()
    record = DecisionRecord(
        decision_type="x", subject_ref="cust_1", jurisdiction=long_value_with_pii
    )
    with caplog.at_level(logging.WARNING, logger="ai_flight_recorder.emit"):
        emit(record, fake_langfuse_client)

    for r in caplog.records:
        assert RAW_EMAIL not in r.getMessage()
