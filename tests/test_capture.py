from types import SimpleNamespace

import pytest

from ai_flight_recorder.capture import Decision, audit_decision, decision
from ai_flight_recorder.config import GovernanceConfig
from ai_flight_recorder.hashing import verify
from ai_flight_recorder.schema import ConfidenceSource
from tests.conftest import FakeAnnotationQueue


def _fake_response(content, logprobs_values=None, model="gpt-4o-mini"):
    logprobs = None
    if logprobs_values is not None:
        logprobs = SimpleNamespace(content=[SimpleNamespace(logprob=v) for v in logprobs_values])
    choice = SimpleNamespace(message=SimpleNamespace(content=content), logprobs=logprobs)
    return SimpleNamespace(model=model, choices=[choice])


def test_context_manager_captures_model_and_latency(fake_langfuse_client):
    with decision(
        decision_type="loan_eligibility",
        subject_ref="cust_123",
        options=["approve", "deny"],
        config=GovernanceConfig(),
        client=fake_langfuse_client,
    ) as d:
        resp = d.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "should I approve?"}],
            mock_response="approve",
        )
        d.outcome = resp.choices[0].message.content

    assert d.record.model == "gpt-4o-mini"
    assert d.record.latency_ms is not None and d.record.latency_ms >= 0
    assert d.record.outcome == "approve"
    assert d.record.subject_ref == "cust_123"
    assert d.record.integrity_hash is not None
    assert verify(d.record).ok is True
    assert d.emit_result is not None
    assert fake_langfuse_client.flush_calls == 1


def test_confidence_source_none_when_no_logprobs_and_no_self_report(fake_langfuse_client):
    with decision(
        decision_type="loan_eligibility", subject_ref="cust_123", client=fake_langfuse_client
    ) as d:
        d.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], mock_response="approve")
        d.outcome = "approve"

    assert len(d.record.confidence) == 1
    assert d.record.confidence[0].source == ConfidenceSource.NONE
    assert d.record.confidence[0].value is None


def test_confidence_source_logprobs_when_response_exposes_them(monkeypatch, fake_langfuse_client):
    monkeypatch.setattr(
        "ai_flight_recorder.capture.litellm.completion",
        lambda **kwargs: _fake_response("approve", logprobs_values=[-0.05, -0.1, -0.02]),
    )
    with decision(
        decision_type="loan_eligibility", subject_ref="cust_123", client=fake_langfuse_client
    ) as d:
        d.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
        d.outcome = "approve"

    assert len(d.record.confidence) == 1
    signal = d.record.confidence[0]
    assert signal.source == ConfidenceSource.LOGPROBS
    assert 0.0 < signal.value <= 1.0


def test_confidence_source_self_report_when_opted_in(monkeypatch, fake_langfuse_client):
    calls = {"n": 0}

    def fake_completion(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _fake_response("approve")  # no logprobs -> falls through to self_report
        return _fake_response("0.8")  # the follow-up self-rating call

    monkeypatch.setattr("ai_flight_recorder.capture.litellm.completion", fake_completion)

    with decision(
        decision_type="loan_eligibility",
        subject_ref="cust_123",
        confidence_method="self_report",
        client=fake_langfuse_client,
    ) as d:
        d.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
        d.outcome = "approve"

    assert calls["n"] == 2
    assert len(d.record.confidence) == 1
    signal = d.record.confidence[0]
    assert signal.source == ConfidenceSource.SELF_REPORT
    assert signal.value == 0.8


def test_self_report_falls_back_to_none_when_unparseable(monkeypatch, fake_langfuse_client):
    monkeypatch.setattr(
        "ai_flight_recorder.capture.litellm.completion",
        lambda **kwargs: _fake_response("not a number"),
    )
    with decision(
        decision_type="loan_eligibility",
        subject_ref="cust_123",
        confidence_method="self_report",
        client=fake_langfuse_client,
    ) as d:
        d.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
        d.outcome = "approve"

    signal = d.record.confidence[0]
    assert signal.source == ConfidenceSource.NONE
    assert signal.value is None


def test_self_report_follow_up_failure_falls_back_to_none_without_raising(monkeypatch, fake_langfuse_client):
    """A provider rejecting the follow-up call (e.g. Azure requiring `api_version`,
    which isn't in the self-report passthrough list) must not fail the decision
    the caller is actually trying to capture."""
    calls = {"n": 0}

    def fake_completion(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _fake_response("approve")  # no logprobs -> falls through to self_report
        raise RuntimeError("provider rejected a required kwarg")  # the follow-up call fails

    monkeypatch.setattr("ai_flight_recorder.capture.litellm.completion", fake_completion)

    with decision(
        decision_type="loan_eligibility",
        subject_ref="cust_123",
        confidence_method="self_report",
        client=fake_langfuse_client,
    ) as d:
        d.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
        d.outcome = "approve"

    assert calls["n"] == 2
    assert d.record.outcome == "approve"
    signal = d.record.confidence[0]
    assert signal.source == ConfidenceSource.NONE
    assert signal.value is None
    assert fake_langfuse_client.flush_calls == 1


@pytest.mark.asyncio
async def test_self_report_follow_up_failure_async_falls_back_to_none(monkeypatch, fake_langfuse_client):
    calls = {"n": 0}

    async def fake_acompletion(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _fake_response("approve")
        raise RuntimeError("provider rejected a required kwarg")

    monkeypatch.setattr("ai_flight_recorder.capture.litellm.acompletion", fake_acompletion)

    async with decision(
        decision_type="loan_eligibility",
        subject_ref="cust_123",
        confidence_method="self_report",
        client=fake_langfuse_client,
    ) as d:
        await d.acompletion(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
        d.outcome = "approve"

    assert calls["n"] == 2
    signal = d.record.confidence[0]
    assert signal.source == ConfidenceSource.NONE
    assert signal.value is None


def test_pii_in_inputs_flag_and_redaction_applied_on_exit(fake_langfuse_client):
    with decision(
        decision_type="loan_eligibility", subject_ref="cust_123", client=fake_langfuse_client
    ) as d:
        d.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "my email is jane.doe@example.com"}],
            mock_response="approve",
        )
        d.outcome = "approve"

    assert any(f.rule_name == "pii_in_inputs" for f in d.record.flags)
    assert d.record.review_required is True
    assert "jane.doe@example.com" not in str(d.record.redacted_prompt)


def test_exception_inside_block_is_not_recorded_and_propagates(fake_langfuse_client):
    with pytest.raises(RuntimeError):
        with decision(
            decision_type="loan_eligibility", subject_ref="cust_123", client=fake_langfuse_client
        ) as d:
            d.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], mock_response="x")
            raise RuntimeError("boom")

    assert fake_langfuse_client.flush_calls == 0
    assert fake_langfuse_client.create_score_calls == []


def test_decorator_produces_equivalent_record_to_context_manager(fake_langfuse_client):
    captured = {}

    @audit_decision(
        decision_type="loan_eligibility",
        subject_ref=lambda applicant_id: applicant_id,
        options=["approve", "deny"],
        client=fake_langfuse_client,
    )
    def evaluate(d: Decision, applicant_id: str) -> str:
        d.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"evaluate {applicant_id}"}],
            mock_response="approve",
        )
        captured["record"] = d.record
        return "approve"

    result = evaluate("cust_999")

    assert result == "approve"
    record = captured["record"]
    assert record.subject_ref == "cust_999"
    assert record.outcome == "approve"  # fell back to the function's return value
    assert record.options == ["approve", "deny"]
    assert record.model == "gpt-4o-mini"
    assert record.integrity_hash is not None


def test_decorator_respects_explicit_outcome_over_return_value(fake_langfuse_client):
    @audit_decision(decision_type="loan_eligibility", subject_ref="cust_123", client=fake_langfuse_client)
    def evaluate(d: Decision) -> str:
        d.outcome = "deny"
        return "approve"  # should be ignored since outcome was set explicitly

    evaluate()
    # re-run and inspect via a fresh capture since evaluate doesn't expose d directly here
    result_holder = {}

    @audit_decision(decision_type="loan_eligibility", subject_ref="cust_123", client=fake_langfuse_client)
    def evaluate2(d: Decision) -> str:
        d.outcome = "deny"
        result_holder["record"] = d.record
        return "approve"

    evaluate2()
    assert result_holder["record"].outcome == "deny"


def test_flagged_decision_is_enqueued_after_flush(fake_langfuse_client):
    fake_langfuse_client.api.annotation_queues._existing.append(
        FakeAnnotationQueue(id="queue-1", name="human-review")
    )
    config = GovernanceConfig(review_queue_name="human-review")

    call_order = []
    original_flush = fake_langfuse_client.flush
    original_create_item = fake_langfuse_client.api.annotation_queues.create_queue_item

    def tracking_flush():
        call_order.append("flush")
        return original_flush()

    def tracking_create_item(**kwargs):
        call_order.append("enqueue")
        return original_create_item(**kwargs)

    fake_langfuse_client.flush = tracking_flush
    fake_langfuse_client.api.annotation_queues.create_queue_item = tracking_create_item

    with decision(
        decision_type="loan_eligibility",
        subject_ref="cust_123",
        config=config,
        client=fake_langfuse_client,
    ) as d:
        d.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "my email is jane.doe@example.com"}],
            mock_response="approve",
        )
        d.outcome = "approve"

    assert d.record.review_required is True
    assert call_order == ["flush", "enqueue"]
    calls = fake_langfuse_client.api.annotation_queues.create_queue_item_calls
    assert len(calls) == 1
    assert calls[0]["queue_id"] == "queue-1"
    assert calls[0]["object_id"] == d.emit_result.trace_id
    assert calls[0]["object_type"] == "TRACE"


def test_clean_decision_is_not_enqueued(fake_langfuse_client):
    fake_langfuse_client.api.annotation_queues._existing.append(
        FakeAnnotationQueue(id="queue-1", name="human-review")
    )
    config = GovernanceConfig(review_queue_name="human-review")

    with decision(
        decision_type="content_recommendation",
        subject_ref="cust_123",
        config=config,
        client=fake_langfuse_client,
    ) as d:
        d.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], mock_response="show")
        d.outcome = "show"

    assert d.record.review_required is False
    assert fake_langfuse_client.api.annotation_queues.create_queue_item_calls == []


def test_no_queue_name_configured_never_enqueues_even_when_flagged(fake_langfuse_client):
    config = GovernanceConfig()  # review_queue_name defaults to None

    with decision(
        decision_type="loan_eligibility",
        subject_ref="cust_123",
        config=config,
        client=fake_langfuse_client,
    ) as d:
        d.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "my email is jane.doe@example.com"}],
            mock_response="approve",
        )
        d.outcome = "approve"

    assert d.record.review_required is True
    assert fake_langfuse_client.api.annotation_queues.create_queue_item_calls == []


def test_auto_flush_false_skips_flush_for_a_clean_decision(fake_langfuse_client):
    config = GovernanceConfig(auto_flush=False)  # no review_queue_name -> nothing to enqueue

    with decision(
        decision_type="content_recommendation",
        subject_ref="cust_123",
        config=config,
        client=fake_langfuse_client,
    ) as d:
        d.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], mock_response="show")
        d.outcome = "show"

    assert d.record.review_required is False
    assert fake_langfuse_client.flush_calls == 0
    assert d.emit_result is not None  # emit still happens; only the flush is skipped


def test_auto_flush_false_still_flushes_before_enqueueing_a_flagged_decision(fake_langfuse_client):
    fake_langfuse_client.api.annotation_queues._existing.append(
        FakeAnnotationQueue(id="queue-1", name="human-review")
    )
    config = GovernanceConfig(review_queue_name="human-review", auto_flush=False)

    call_order = []
    original_flush = fake_langfuse_client.flush
    original_create_item = fake_langfuse_client.api.annotation_queues.create_queue_item

    def tracking_flush():
        call_order.append("flush")
        return original_flush()

    def tracking_create_item(**kwargs):
        call_order.append("enqueue")
        return original_create_item(**kwargs)

    fake_langfuse_client.flush = tracking_flush
    fake_langfuse_client.api.annotation_queues.create_queue_item = tracking_create_item

    with decision(
        decision_type="loan_eligibility",
        subject_ref="cust_123",
        config=config,
        client=fake_langfuse_client,
    ) as d:
        d.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "my email is jane.doe@example.com"}],
            mock_response="approve",
        )
        d.outcome = "approve"

    assert d.record.review_required is True
    assert fake_langfuse_client.flush_calls == 1
    assert call_order == ["flush", "enqueue"]


def test_auto_flush_false_skips_flush_when_flagged_but_auto_enqueue_disabled(fake_langfuse_client):
    fake_langfuse_client.api.annotation_queues._existing.append(
        FakeAnnotationQueue(id="queue-1", name="human-review")
    )
    config = GovernanceConfig(review_queue_name="human-review", auto_flush=False, auto_enqueue=False)

    with decision(
        decision_type="loan_eligibility",
        subject_ref="cust_123",
        config=config,
        client=fake_langfuse_client,
    ) as d:
        d.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "my email is jane.doe@example.com"}],
            mock_response="approve",
        )
        d.outcome = "approve"

    assert d.record.review_required is True
    assert fake_langfuse_client.flush_calls == 0
    assert fake_langfuse_client.api.annotation_queues.create_queue_item_calls == []


@pytest.mark.asyncio
async def test_async_decorator_and_acompletion(fake_langfuse_client):
    @audit_decision(decision_type="loan_eligibility", subject_ref="cust_123", client=fake_langfuse_client)
    async def evaluate(d: Decision) -> str:
        await d.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="approve",
        )
        return "approve"

    result = await evaluate()
    assert result == "approve"
