"""Auto-enqueueing flagged decisions into a Langfuse annotation queue.

Rules flag decisions for review (see `rules.py`), but a flag by itself is just
a score sitting on a trace, with nothing to put it in front of a reviewer.
This module adds the missing step: after a flagged decision's trace has been
emitted *and flushed*, it is added to a Langfuse annotation queue, so
reviewers get a worklist instead of hunting through the trace table.

Ordering is load-bearing: a queue item references its trace by id, and
Langfuse has nothing to attach the item to until that trace is actually
persisted. `enqueue_if_flagged()` must only ever be called after
`client.flush()` has returned (see `capture.py`).

Enqueueing is a side effect, never a requirement: every Langfuse call in this
module is wrapped so a missing queue or a failing API call is logged and
swallowed, not raised. A broken review-queue integration must never take down
decision capture.

Only trace-level items (`objectType: "TRACE"`) are used. Session items are
avoided deliberately: recent Langfuse SDKs have a known response-validation
bug for that item type.
"""

from __future__ import annotations

import logging
from typing import Any

from langfuse.api.resources.annotation_queues.types.annotation_queue_object_type import (
    AnnotationQueueObjectType,
)
from langfuse.api.resources.annotation_queues.types.create_annotation_queue_item_request import (
    CreateAnnotationQueueItemRequest,
)

from ai_flight_recorder.config import GovernanceConfig
from ai_flight_recorder.schema import DecisionRecord

logger = logging.getLogger("ai_flight_recorder.annotation_queue")

_LIST_QUEUES_PAGE_SIZE = 100


def _list_all_queues(client: Any) -> list[Any]:
    """Fetch every annotation queue, paginating past the API's per-page cap so
    a queue that exists but sits on a later page isn't mistaken for missing."""
    results: list[Any] = []
    page = 1
    while True:
        batch = client.api.annotation_queues.list_queues(page=page, limit=_LIST_QUEUES_PAGE_SIZE).data
        results.extend(batch)
        if len(batch) < _LIST_QUEUES_PAGE_SIZE:
            return results
        page += 1


def _resolve_queue_id(client: Any, queue_name: str, config: GovernanceConfig) -> str | None:
    """Look up a queue's id by name, caching hits on `config` so repeated
    decisions against the same config don't re-list queues every time."""
    cached = config._queue_id_cache.get(queue_name)
    if cached is not None:
        return cached

    queues = _list_all_queues(client)
    match = next((q for q in queues if q.name == queue_name), None)
    if match is None:
        logger.warning(
            "ai_flight_recorder: review queue '%s' does not exist in Langfuse; skipping "
            "enqueue for this decision. Run ensure_review_queue() to create it.",
            queue_name,
        )
        return None

    config._queue_id_cache[queue_name] = match.id
    return match.id


def enqueue_if_flagged(
    record: DecisionRecord, trace_id: str, client: Any, config: GovernanceConfig
) -> None:
    """If `record` was flagged for review and a queue is configured, add its
    trace to that annotation queue. Call only after the trace has been
    flushed to Langfuse; the queue item points at `trace_id`, which must
    already exist.

    Never raises. Skips (rather than enqueueing) when: enqueueing is
    disabled, no queue is configured, the decision wasn't flagged, the
    decision was already enqueued in this config's lifetime, the named queue
    doesn't exist, or the API call itself fails.
    """
    if not config.auto_enqueue or not config.review_queue_name:
        return
    if not record.review_required:
        return
    if record.decision_id in config._enqueued_decision_ids:
        return

    try:
        queue_id = _resolve_queue_id(client, config.review_queue_name, config)
        if queue_id is None:
            return
        client.api.annotation_queues.create_queue_item(
            queue_id=queue_id,
            request=CreateAnnotationQueueItemRequest(
                object_id=trace_id,
                object_type=AnnotationQueueObjectType.TRACE,
            ),
        )
        config._mark_enqueued(record.decision_id)
    except Exception:
        logger.warning(
            "ai_flight_recorder: failed to enqueue decision '%s' (trace '%s') into review "
            "queue '%s'; the decision itself was still captured normally.",
            record.decision_id,
            trace_id,
            config.review_queue_name,
            exc_info=True,
        )
