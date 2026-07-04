"""ai-flight-recorder quickstart.

Runs three decision scenarios across two decision_types, some flagged for
review and some not, and prints the resulting Langfuse trace ids so you can
open them in the Langfuse UI. Flagged decisions are also auto-enqueued into a
Langfuse annotation queue (`REVIEW_QUEUE_NAME` below); the clean scenario is
not, since it was never flagged.

Two things are mocked by default, and each turns real once you set the
matching env var:

  - the LLM calls use `mock_response` UNLESS you set OPENAI_API_KEY, in which
    case they call the real OpenAI API instead.
  - the Langfuse client is a local, in-memory mock UNLESS you set
    LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (and optionally LANGFUSE_HOST)
    in your environment, in which case this script uses the real client and
    prints real trace URLs you can open.

Run it with:

    python examples/quickstart.py
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from types import SimpleNamespace
from typing import Any

import ai_flight_recorder
from ai_flight_recorder import GovernanceConfig
from ai_flight_recorder.schema import ConfidenceSignal, ConfidenceSource

REVIEW_QUEUE_NAME = "ai-flight-recorder-review"


class _MockObservation:
    def __init__(self, id: str, trace_id: str) -> None:
        self.id = id
        self.trace_id = trace_id


class _MockAnnotationQueuesAPI:
    """No-ops the annotation-queue API so the zero-setup local run still
    exercises the auto-enqueue path, without a real Langfuse project to
    enqueue into. `list_queues` reports the review queue as already existing
    (skipping the "queue not found" warning); `create_queue_item` just
    records the trace id it *would have* enqueued."""

    def __init__(self, queue_name: str) -> None:
        self._queue = SimpleNamespace(id="mock-queue-id", name=queue_name)
        self.would_have_enqueued: list[str] = []

    def list_queues(self, **kwargs: Any) -> Any:
        return SimpleNamespace(data=[self._queue])

    def create_queue_item(self, *, queue_id: str, request: Any) -> None:
        self.would_have_enqueued.append(request.object_id)


class _MockLangfuseClient:
    """Zero-setup stand-in for the real Langfuse SDK client. Prints nothing on
    its own; `run_scenario` below prints what ai-flight-recorder captured."""

    def __init__(self) -> None:
        self._next_obs_id = 0
        self.api = SimpleNamespace(annotation_queues=_MockAnnotationQueuesAPI(REVIEW_QUEUE_NAME))

    def create_trace_id(self, *, seed: str | None = None) -> str:
        return hashlib.sha256((seed or "").encode()).hexdigest()[:32]

    def start_as_current_observation(self, *, trace_context, name, **kwargs: Any):
        self._next_obs_id += 1
        return contextlib.nullcontext(
            _MockObservation(id=f"obs-{self._next_obs_id}", trace_id=trace_context["trace_id"])
        )

    def create_score(self, **kwargs: Any) -> None:
        pass

    def flush(self) -> None:
        pass

    def get_trace_url(self, *, trace_id: str) -> str | None:
        return None


def _build_client():
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        from langfuse import get_client

        print("Using a real Langfuse client (LANGFUSE_PUBLIC_KEY is set).\n")
        return get_client()

    print(
        "No LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY set, using a local mock "
        "Langfuse client. Set those env vars (and optionally LANGFUSE_HOST) to "
        "run this against a real instance and get openable trace URLs.\n"
    )
    return _MockLangfuseClient()


def _print_result(d, client) -> None:
    flags = [f.rule_name for f in d.record.flags]
    print(f"  decision_id       {d.record.decision_id}")
    print(f"  decision_type     {d.record.decision_type}")
    print(f"  subject_ref       {d.record.subject_ref}")
    print(f"  outcome           {d.record.outcome}")
    print(f"  flags             {flags or '(none)'}")
    print(f"  review_required   {d.record.review_required}")
    print(f"  trace_id          {d.emit_result.trace_id}")
    if hasattr(client, "auth_check"):  # a real Langfuse client
        url = client.get_trace_url(trace_id=d.emit_result.trace_id)
        if url:
            print(f"  trace_url         {url}")
    # review_required alone doesn't mean it landed in the queue (that needs
    # review_queue_name configured too), but here it always is, so the two
    # line up. The clean scenario is deliberately never enqueued.
    if d.record.review_required:
        print(f"  enqueued          yes, into '{REVIEW_QUEUE_NAME}'")
    else:
        print("  enqueued          no (not flagged for review)")
    print()


def _outcome_from_response(response: Any, options: list[str]) -> str:
    """Extracts the model's one-word answer from a real completion response,
    normalizing it to one of the configured `options` when it matches
    (case-insensitively) so it lines up with the sensitive-outcomes checks
    in `rules.py`."""
    text = (response.choices[0].message.content or "").strip()
    lowered = text.lower()
    for option in options:
        if option.lower() in lowered:
            return option
    return text


def scenario_high_stakes_with_pii(client, config, use_real_llm: bool) -> None:
    """loan_eligibility, flagged for two independent reasons: it's a
    configured high-stakes decision_type, and the prompt contains an email
    address that the pii_in_inputs rule catches and redacts."""
    print("Scenario 1: loan_eligibility with PII in the prompt")
    with ai_flight_recorder.decision(
        decision_type="loan_eligibility",
        subject_ref="cust_1001",
        options=["approve", "deny"],
        config=config,
        client=client,
    ) as d:
        kwargs: dict[str, Any] = dict(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a loan underwriting assistant. Review the application "
                        "and respond with exactly one word: approve or deny."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Applicant: Jane Doe (jane.doe@example.com)\n"
                        "Annual income: $85,000\n"
                        "Requested loan amount: $20,000\n"
                        "Purpose: debt consolidation\n"
                        "Credit score: 710\n"
                        "Existing monthly debt payments: $650\n\n"
                        "Approve or deny this application?"
                    ),
                },
            ],
        )
        if not use_real_llm:
            kwargs["mock_response"] = "approve"
        response = d.completion(**kwargs)
        d.outcome = _outcome_from_response(response, d.record.options) if use_real_llm else "approve"
    _print_result(d, client)
    print(f"  redacted_prompt   {d.record.redacted_prompt}\n")


def scenario_low_confidence(client, config, use_real_llm: bool) -> None:
    """loan_eligibility again, this time flagged for low confidence.

    Without a real LLM key, `mock_response` is given a dict shaped like a
    real OpenAI-compatible response (`choices[0].logprobs.content`) instead
    of a plain string, since a plain string produces no logprobs at all.
    That dict is the same shape a real `logprobs=True` call returns, so
    ai-flight-recorder captures the confidence signal through its normal path in
    `capture.py` either way; no `ConfidenceSignal` is appended by hand.

    With a real key, the actual response's logprobs drive the signal, so
    whether this trips the low_confidence flag depends on how confident the
    model actually is.
    """
    print("Scenario 2: loan_eligibility with a low-confidence signal from logprobs")
    with ai_flight_recorder.decision(
        decision_type="loan_eligibility",
        subject_ref="cust_1002",
        options=["approve", "deny"],
        config=config,
        client=client,
    ) as d:
        kwargs: dict[str, Any] = dict(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a loan underwriting assistant. Review the application "
                        "and respond with exactly one word: approve or deny."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Applicant: John Smith\n"
                        "Annual income: $42,000\n"
                        "Requested loan amount: $35,000\n"
                        "Purpose: home renovation\n"
                        "Credit score: 640\n"
                        "Existing monthly debt payments: $1,100\n\n"
                        "The requested amount is high relative to income, but there have "
                        "been no missed payments in the last three years. Approve or deny "
                        "this application?"
                    ),
                },
            ],
            logprobs=True,
        )
        if not use_real_llm:
            kwargs["mock_response"] = {
                "id": "mock-low-confidence",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {"role": "assistant", "content": "deny"},
                        "logprobs": {"content": [{"token": "deny", "logprob": -1.05, "top_logprobs": []}]},
                    }
                ],
            }
        response = d.completion(**kwargs)
        d.outcome = _outcome_from_response(response, d.record.options) if use_real_llm else "deny"
    _print_result(d, client)


def scenario_clean_pass(client, config, use_real_llm: bool) -> None:
    """content_recommendation, not high-stakes, no PII, no low confidence.
    Should not be flagged for review."""
    print("Scenario 3: content_recommendation, nothing flagged")
    with ai_flight_recorder.decision(
        decision_type="content_recommendation",
        subject_ref="cust_2001",
        options=["show", "suppress"],
        config=config,
        client=client,  # ai-flight-recorder will capture logprobs if the model emits them
    ) as d:
        kwargs: dict[str, Any] = dict(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a content recommendation assistant for a news feed. "
                        "Respond with exactly one word: show or suppress."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        'Article: "City Council Approves New Bike Lane Network"\n'
                        "Source: Local Times, published today\n"
                        "Summary: The city council voted 5-3 to approve 12 miles of "
                        "protected bike lanes downtown, with reactions from cyclists and "
                        "local business owners.\n\n"
                        "Should we recommend this article to users who follow local news "
                        "and transportation topics?"
                    ),
                },
            ],
        )
        if use_real_llm:
            kwargs["logprobs"] = True
        else:
            kwargs["mock_response"] = "show"
        response = d.completion(**kwargs)
        if not use_real_llm:
            d.record.confidence.append(
                ConfidenceSignal(source=ConfidenceSource.LOGPROBS, value=0.94, note="simulated for this example")
            )
        d.outcome = _outcome_from_response(response, d.record.options) if use_real_llm else "show"
    _print_result(d, client)


def main() -> None:
    client = _build_client()
    use_real_llm = bool(os.environ.get("OPENAI_API_KEY"))
    if use_real_llm:
        print("OPENAI_API_KEY is set, calling the real OpenAI API instead of mock_response.\n")
    config = GovernanceConfig(
        high_stakes_types={"loan_eligibility"},
        low_confidence_threshold=0.5,
        review_queue_name=REVIEW_QUEUE_NAME,
    )

    is_real_client = hasattr(client, "auth_check")
    if is_real_client:
        # Must happen before the scenarios run: a decision can only be
        # enqueued into a review queue that already exists.
        ai_flight_recorder.ensure_review_queue(client, queue_name=REVIEW_QUEUE_NAME)
        print(
            f"Review queue '{REVIEW_QUEUE_NAME}' ensured, with the review_required "
            "and human_review_verdict ScoreConfigs attached.\n"
        )

    scenario_high_stakes_with_pii(client, config, use_real_llm)
    scenario_low_confidence(client, config, use_real_llm)
    scenario_clean_pass(client, config, use_real_llm)

    if is_real_client:
        print("Open the trace URLs above to review these decisions in Langfuse.")
        print(f"Or go to Human Annotation > {REVIEW_QUEUE_NAME} to see the flagged ones waiting.")


if __name__ == "__main__":
    main()
