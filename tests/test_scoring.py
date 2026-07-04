from ai_flight_recorder.scoring import (
    HUMAN_REVIEW_VERDICT_CONFIG_NAME,
    REVIEW_REQUIRED_CONFIG_NAME,
    ensure_review_queue,
    setup_score_configs,
)
from tests.conftest import FakeAnnotationQueue, FakeScoreConfig


def test_creates_both_configs_when_none_exist(fake_langfuse_client):
    results = setup_score_configs(fake_langfuse_client)
    names = {r.name for r in results}
    assert names == {REVIEW_REQUIRED_CONFIG_NAME, HUMAN_REVIEW_VERDICT_CONFIG_NAME}
    assert len(fake_langfuse_client.api.score_configs.create_calls) == 2


def test_review_required_created_as_boolean_with_no_categories(fake_langfuse_client):
    setup_score_configs(fake_langfuse_client)
    request = next(
        r
        for r in fake_langfuse_client.api.score_configs.create_calls
        if r.name == REVIEW_REQUIRED_CONFIG_NAME
    )
    assert request.data_type == "BOOLEAN"
    assert request.categories is None


def test_human_review_verdict_created_as_categorical_with_expected_categories(fake_langfuse_client):
    setup_score_configs(fake_langfuse_client)
    request = next(
        r
        for r in fake_langfuse_client.api.score_configs.create_calls
        if r.name == HUMAN_REVIEW_VERDICT_CONFIG_NAME
    )
    assert request.data_type == "CATEGORICAL"
    labels = {c.label for c in request.categories}
    assert labels == {"approved", "overridden", "escalated"}


def test_idempotent_skips_existing_active_configs(fake_langfuse_client):
    fake_langfuse_client.api.score_configs._existing.append(
        FakeScoreConfig(name=REVIEW_REQUIRED_CONFIG_NAME, is_archived=False)
    )
    results = setup_score_configs(fake_langfuse_client)
    created_names = {r.name for r in fake_langfuse_client.api.score_configs.create_calls}
    assert REVIEW_REQUIRED_CONFIG_NAME not in created_names
    assert HUMAN_REVIEW_VERDICT_CONFIG_NAME in created_names
    assert {r.name for r in results} == {REVIEW_REQUIRED_CONFIG_NAME, HUMAN_REVIEW_VERDICT_CONFIG_NAME}


def test_archived_config_is_recreated(fake_langfuse_client):
    fake_langfuse_client.api.score_configs._existing.append(
        FakeScoreConfig(name=REVIEW_REQUIRED_CONFIG_NAME, is_archived=True)
    )
    setup_score_configs(fake_langfuse_client)
    created_names = {r.name for r in fake_langfuse_client.api.score_configs.create_calls}
    assert REVIEW_REQUIRED_CONFIG_NAME in created_names


def test_calling_twice_does_not_duplicate(fake_langfuse_client):
    setup_score_configs(fake_langfuse_client)
    setup_score_configs(fake_langfuse_client)
    assert len(fake_langfuse_client.api.score_configs.create_calls) == 2


def test_ensure_review_queue_creates_queue_with_both_score_configs_attached(fake_langfuse_client):
    queue = ensure_review_queue(fake_langfuse_client, queue_name="human-review")

    assert queue.name == "human-review"
    created_config_ids = {c.id for c in fake_langfuse_client.api.score_configs._existing}
    assert set(queue.score_config_ids) == created_config_ids
    assert len(fake_langfuse_client.api.annotation_queues.create_queue_calls) == 1


def test_ensure_review_queue_is_idempotent_when_queue_already_exists(fake_langfuse_client):
    fake_langfuse_client.api.annotation_queues._existing.append(
        FakeAnnotationQueue(id="queue-99", name="human-review")
    )
    queue = ensure_review_queue(fake_langfuse_client, queue_name="human-review")

    assert queue.id == "queue-99"
    assert fake_langfuse_client.api.annotation_queues.create_queue_calls == []


def test_ensure_review_queue_calling_twice_does_not_duplicate(fake_langfuse_client):
    ensure_review_queue(fake_langfuse_client, queue_name="human-review")
    ensure_review_queue(fake_langfuse_client, queue_name="human-review")
    assert len(fake_langfuse_client.api.annotation_queues.create_queue_calls) == 1
