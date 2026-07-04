"""Shared test fixtures. No test in this suite hits litellm or langfuse over the
network: litellm calls use `mock_response`, and the Langfuse client is this
lightweight fake implementing the same subset of the v3 SDK's interface."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
from typing import Any

import pytest

# `ai_flight_recorder/__init__.py` re-exports `emit` (the function), which shadows the
# `emit` submodule attribute on the package — `import ai_flight_recorder.emit as x`
# would resolve to that function instead of the module, so look it up via
# `sys.modules` directly.
emit_module = importlib.import_module("ai_flight_recorder.emit")


class FakeObservation:
    def __init__(self, id: str, trace_id: str) -> None:
        self.id = id
        self.trace_id = trace_id


class FakeScoreConfig:
    """Stand-in for langfuse.api's ScoreConfig — only the attributes
    ai-flight-recorder actually reads (`id`, `name`, `is_archived`)."""

    def __init__(self, name: str, is_archived: bool = False, id: str | None = None) -> None:
        self.id = id or f"score-config-{name}"
        self.name = name
        self.is_archived = is_archived


class FakeScoreConfigsPage:
    def __init__(self, data: list[FakeScoreConfig]) -> None:
        self.data = data


class FakeScoreConfigsAPI:
    def __init__(self, existing: list[FakeScoreConfig] | None = None) -> None:
        self._existing = existing or []
        self.create_calls: list[Any] = []

    def get(self, *, page: int | None = None, limit: int | None = None, **kwargs):
        return FakeScoreConfigsPage(list(self._existing))

    def create(self, *, request: Any):
        self.create_calls.append(request)
        created = FakeScoreConfig(name=request.name, is_archived=False)
        self._existing.append(created)
        return created


class FakeAnnotationQueue:
    """Stand-in for langfuse.api's AnnotationQueue — only the attributes
    ai-flight-recorder actually reads (`id`, `name`)."""

    def __init__(self, id: str, name: str, score_config_ids: list[str] | None = None) -> None:
        self.id = id
        self.name = name
        self.score_config_ids = list(score_config_ids or [])


class FakeAnnotationQueueItem:
    def __init__(self, id: str, queue_id: str, object_id: str, object_type: str) -> None:
        self.id = id
        self.queue_id = queue_id
        self.object_id = object_id
        self.object_type = object_type


class FakeAnnotationQueuesPage:
    def __init__(self, data: list[FakeAnnotationQueue]) -> None:
        self.data = data


class FakeAnnotationQueuesAPI:
    def __init__(self, existing: list[FakeAnnotationQueue] | None = None) -> None:
        self._existing = existing or []
        self.create_queue_calls: list[Any] = []
        self.create_queue_item_calls: list[dict] = []
        # Set to an exception instance to make create_queue_item raise it,
        # simulating a failing annotation-queue API call.
        self.raise_on_create_item: Exception | None = None
        self._next_queue_id = 0
        self._next_item_id = 0

    def list_queues(self, *, page: int | None = None, limit: int | None = None, **kwargs):
        return FakeAnnotationQueuesPage(list(self._existing))

    def create_queue(self, *, request: Any):
        self.create_queue_calls.append(request)
        self._next_queue_id += 1
        created = FakeAnnotationQueue(
            id=f"queue-{self._next_queue_id}",
            name=request.name,
            score_config_ids=request.score_config_ids,
        )
        self._existing.append(created)
        return created

    def create_queue_item(self, *, queue_id: str, request: Any):
        self.create_queue_item_calls.append(
            {"queue_id": queue_id, "object_id": request.object_id, "object_type": request.object_type}
        )
        if self.raise_on_create_item is not None:
            raise self.raise_on_create_item
        self._next_item_id += 1
        return FakeAnnotationQueueItem(
            id=f"item-{self._next_item_id}",
            queue_id=queue_id,
            object_id=request.object_id,
            object_type=request.object_type,
        )


class FakeAPI:
    def __init__(
        self,
        existing_score_configs: list[FakeScoreConfig] | None = None,
        existing_annotation_queues: list[FakeAnnotationQueue] | None = None,
    ) -> None:
        self.score_configs = FakeScoreConfigsAPI(existing_score_configs)
        self.annotation_queues = FakeAnnotationQueuesAPI(existing_annotation_queues)


class FakeLangfuseClient:
    """Records every call it receives so tests can assert on the exact A/B/C/D
    payloads ai-flight-recorder sends, without touching the network.

    `propagate_attributes` isn't a client method on the real SDK (it's a
    module-level context manager in `langfuse`), so `propagate_attributes_calls`
    is populated by the `fake_langfuse_client` fixture monkeypatching
    `ai_flight_recorder.emit.propagate_attributes` instead of by a method here.
    """

    def __init__(self) -> None:
        self.create_trace_id_calls: list[str | None] = []
        self.propagate_attributes_calls: list[dict] = []
        self.start_as_current_observation_calls: list[dict] = []
        self.create_score_calls: list[dict] = []
        self.flush_calls = 0
        self._next_obs_id = 0
        self.api = FakeAPI()

    def create_trace_id(self, *, seed: str | None = None) -> str:
        self.create_trace_id_calls.append(seed)
        # deterministic stand-in for langfuse's real seeded id derivation
        return hashlib.sha256((seed or "").encode()).hexdigest()[:32]

    def start_as_current_observation(
        self,
        *,
        trace_context,
        name,
        as_type="span",
        input=None,
        output=None,
        metadata=None,
        model=None,
        **kwargs,
    ):
        self.start_as_current_observation_calls.append(
            dict(
                trace_context=trace_context,
                name=name,
                as_type=as_type,
                input=input,
                output=output,
                metadata=metadata,
                model=model,
            )
        )
        self._next_obs_id += 1
        observation = FakeObservation(
            id=f"obs-{self._next_obs_id}", trace_id=trace_context["trace_id"]
        )
        return contextlib.nullcontext(observation)

    def create_score(self, **kwargs):
        self.create_score_calls.append(kwargs)

    def flush(self) -> None:
        self.flush_calls += 1


@pytest.fixture
def fake_langfuse_client(monkeypatch: pytest.MonkeyPatch) -> FakeLangfuseClient:
    client = FakeLangfuseClient()

    def fake_propagate_attributes(*, metadata: dict | None = None, **kwargs: Any):
        client.propagate_attributes_calls.append({"metadata": metadata, **kwargs})
        return contextlib.nullcontext()

    monkeypatch.setattr(emit_module, "propagate_attributes", fake_propagate_attributes)
    return client
