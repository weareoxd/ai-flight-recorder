import logging

from ai_flight_recorder.annotation_queue import enqueue_if_flagged
from ai_flight_recorder.config import GovernanceConfig
from ai_flight_recorder.schema import DecisionRecord
from tests.conftest import FakeAnnotationQueue


def _make_record(**overrides) -> DecisionRecord:
    defaults = dict(
        decision_id="dec-001",
        decision_type="loan_eligibility",
        subject_ref="cust_12345",
        review_required=True,
    )
    defaults.update(overrides)
    return DecisionRecord(**defaults)


def _configured(client, queue_name="human-review", **overrides) -> GovernanceConfig:
    client.api.annotation_queues._existing.append(FakeAnnotationQueue(id="queue-1", name=queue_name))
    return GovernanceConfig(review_queue_name=queue_name, **overrides)


def test_enqueues_flagged_decision_with_trace_id_and_trace_object_type(fake_langfuse_client):
    config = _configured(fake_langfuse_client)
    record = _make_record()

    enqueue_if_flagged(record, "trace-abc", fake_langfuse_client, config)

    calls = fake_langfuse_client.api.annotation_queues.create_queue_item_calls
    assert len(calls) == 1
    assert calls[0]["queue_id"] == "queue-1"
    assert calls[0]["object_id"] == "trace-abc"
    assert calls[0]["object_type"] == "TRACE"


def test_not_enqueued_when_review_not_required(fake_langfuse_client):
    config = _configured(fake_langfuse_client)
    record = _make_record(review_required=False)

    enqueue_if_flagged(record, "trace-abc", fake_langfuse_client, config)

    assert fake_langfuse_client.api.annotation_queues.create_queue_item_calls == []


def test_not_enqueued_when_no_queue_name_configured(fake_langfuse_client):
    config = GovernanceConfig()
    record = _make_record()

    enqueue_if_flagged(record, "trace-abc", fake_langfuse_client, config)

    assert fake_langfuse_client.api.annotation_queues.create_queue_item_calls == []


def test_not_enqueued_when_auto_enqueue_disabled(fake_langfuse_client):
    config = _configured(fake_langfuse_client, auto_enqueue=False)
    record = _make_record()

    enqueue_if_flagged(record, "trace-abc", fake_langfuse_client, config)

    assert fake_langfuse_client.api.annotation_queues.create_queue_item_calls == []


def test_missing_queue_logs_warning_and_skips_without_raising(fake_langfuse_client, caplog):
    config = GovernanceConfig(review_queue_name="does-not-exist")
    record = _make_record()

    with caplog.at_level(logging.WARNING, logger="ai_flight_recorder.annotation_queue"):
        enqueue_if_flagged(record, "trace-abc", fake_langfuse_client, config)

    assert fake_langfuse_client.api.annotation_queues.create_queue_item_calls == []
    assert any("does not exist" in r.message for r in caplog.records)


def test_enqueue_api_failure_is_swallowed_and_logged(fake_langfuse_client, caplog):
    config = _configured(fake_langfuse_client)
    fake_langfuse_client.api.annotation_queues.raise_on_create_item = RuntimeError("boom")
    record = _make_record()

    with caplog.at_level(logging.WARNING, logger="ai_flight_recorder.annotation_queue"):
        enqueue_if_flagged(record, "trace-abc", fake_langfuse_client, config)  # must not raise

    assert any("failed to enqueue" in r.message for r in caplog.records)


def test_idempotent_does_not_enqueue_same_decision_twice(fake_langfuse_client):
    config = _configured(fake_langfuse_client)
    record = _make_record()

    enqueue_if_flagged(record, "trace-abc", fake_langfuse_client, config)
    enqueue_if_flagged(record, "trace-abc", fake_langfuse_client, config)

    assert len(fake_langfuse_client.api.annotation_queues.create_queue_item_calls) == 1


def test_queue_item_request_carries_no_pii(fake_langfuse_client):
    config = _configured(fake_langfuse_client)
    record = _make_record(subject_ref="cust_12345")

    enqueue_if_flagged(record, "trace-abc", fake_langfuse_client, config)

    call = fake_langfuse_client.api.annotation_queues.create_queue_item_calls[0]
    # CreateAnnotationQueueItemRequest only carries object_id (the trace id,
    # already an opaque hash) and object_type; there is no free-form
    # metadata field on the item for raw PII to leak into.
    assert set(call.keys()) == {"queue_id", "object_id", "object_type"}
    assert call["object_id"] == "trace-abc"
