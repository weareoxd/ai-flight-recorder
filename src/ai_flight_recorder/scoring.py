"""Idempotently create the Langfuse ScoreConfigs, and the review annotation
queue, that standardize governance review across a team.

We do not build a review UI: reviewers work the annotation queue this module
sets up (see `annotation_queue.py` for how decisions get enqueued into it) and
record `human_review_verdict` directly in Langfuse's annotation UI, using the
categories this sets up. ai-flight-recorder itself only ever writes
`review_required` (see `emit.py`) — it never writes `human_review_verdict`.
"""

from __future__ import annotations

from typing import Any

from langfuse.api.resources.annotation_queues.types.create_annotation_queue_request import (
    CreateAnnotationQueueRequest,
)
from langfuse.api.resources.commons.types.config_category import ConfigCategory
from langfuse.api.resources.score_configs.types.create_score_config_request import (
    CreateScoreConfigRequest,
)

from ai_flight_recorder.annotation_queue import _list_all_queues

REVIEW_REQUIRED_CONFIG_NAME = "review_required"
HUMAN_REVIEW_VERDICT_CONFIG_NAME = "human_review_verdict"

_HUMAN_REVIEW_VERDICT_CATEGORIES = [
    ("approved", 0.0),
    ("overridden", 1.0),
    ("escalated", 2.0),
]

_SCORE_CONFIG_SPECS: list[dict[str, Any]] = [
    {
        "name": REVIEW_REQUIRED_CONFIG_NAME,
        "data_type": "BOOLEAN",
        "description": (
            "Whether ai-flight-recorder's rule engine flagged this decision for human "
            "review. Set automatically at capture time — routes attention, is "
            "not a judgment of decision quality."
        ),
    },
    {
        "name": HUMAN_REVIEW_VERDICT_CONFIG_NAME,
        "data_type": "CATEGORICAL",
        "categories": _HUMAN_REVIEW_VERDICT_CATEGORIES,
        "description": (
            "The human reviewer's verdict, recorded via Langfuse's annotation "
            "UI. Never written by ai-flight-recorder itself."
        ),
    },
]


_LIST_SCORE_CONFIGS_PAGE_SIZE = 100


def _get_default_client() -> Any:
    from langfuse import get_client

    return get_client()


def _list_all_score_configs(client: Any) -> list[Any]:
    """Fetch every score config, paginating past the API's per-page cap so a
    config that exists but sits on a later page isn't mistaken for missing."""
    results: list[Any] = []
    page = 1
    while True:
        batch = client.api.score_configs.get(page=page, limit=_LIST_SCORE_CONFIGS_PAGE_SIZE).data
        results.extend(batch)
        if len(batch) < _LIST_SCORE_CONFIGS_PAGE_SIZE:
            return results
        page += 1


def setup_score_configs(client: Any | None = None) -> list[Any]:
    """Create the `review_required` and `human_review_verdict` ScoreConfigs if
    they don't already exist (by name, ignoring archived configs). Safe to call
    repeatedly — existing configs are returned as-is, never duplicated."""
    client = client or _get_default_client()

    existing = _list_all_score_configs(client)
    existing_by_name = {c.name: c for c in existing if not c.is_archived}

    results = []
    for spec in _SCORE_CONFIG_SPECS:
        if spec["name"] in existing_by_name:
            results.append(existing_by_name[spec["name"]])
            continue

        # CreateScoreConfigRequest.dict() only omits a field when it was never
        # passed to the constructor (exclude_unset), not when it's None — so
        # `categories=None` would still serialize as `"categories": null`,
        # which the API rejects for non-categorical score types. Only pass the
        # kwarg at all when there are categories to send.
        categories = spec.get("categories")
        kwargs: dict[str, Any] = {
            "name": spec["name"],
            "dataType": spec["data_type"],
            "description": spec.get("description"),
        }
        if categories:
            kwargs["categories"] = [
                ConfigCategory(value=value, label=label) for label, value in categories
            ]
        request = CreateScoreConfigRequest(**kwargs)
        results.append(client.api.score_configs.create(request=request))

    return results


def ensure_review_queue(
    client: Any | None = None,
    *,
    queue_name: str,
    description: str | None = None,
) -> Any:
    """Create the annotation queue flagged decisions are enqueued into (see
    `annotation_queue.enqueue_if_flagged`) if it doesn't already exist, by
    name. Safe to call repeatedly; an existing queue is returned as-is.

    Also runs `setup_score_configs()` first and attaches both
    `review_required` and `human_review_verdict` to the queue, so reviewers
    working it are scoring the right dimensions.
    """
    client = client or _get_default_client()
    score_configs = setup_score_configs(client)

    existing = _list_all_queues(client)
    match = next((q for q in existing if q.name == queue_name), None)
    if match is not None:
        return match

    # Same exclude_unset caveat as CreateScoreConfigRequest above: only pass
    # description when there is one, rather than passing description=None.
    kwargs: dict[str, Any] = {
        "name": queue_name,
        "scoreConfigIds": [c.id for c in score_configs],
    }
    if description:
        kwargs["description"] = description
    request = CreateAnnotationQueueRequest(**kwargs)
    return client.api.annotation_queues.create_queue(request=request)


def main() -> None:
    """Console-script entry point (`ai-flight-recorder-setup-scores`). Creates the
    ScoreConfigs against LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY/LANGFUSE_HOST
    from the environment (or a `.env` file — see README)."""
    for config in setup_score_configs():
        print(f"  {config.name}: {config.data_type}")
