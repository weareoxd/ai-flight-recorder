"""The LiteLLM capture layer: `decision()` context manager and `audit_decision` decorator.

Both integration styles build the same `DecisionRecord` and finish the same way
on a normal exit: run the rule engine (sets flags/review_required, and may
populate redacted_prompt), compute the integrity hash over the now-finalized
record, emit to Langfuse, then flush.

Note on ordering vs. the hash: the hash is computed *after* the rule engine
runs (not before) so that it covers the fully-finalized captured record,
including any PII redaction the pii_in_inputs rule performs. Flags and
review_required are set once by the rule engine and never mutated again at the
DecisionRecord level, so including them in the hash is safe — the only thing
ever added after this point is the human review verdict, which lives purely in
Langfuse as a score and was never part of this schema.

If the `with` block raises, the decision is not recorded — an incomplete
capture (e.g. outcome never set) isn't a decision worth emitting. The
exception always propagates.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import math
import time
from typing import Any, Callable

import litellm

from ai_flight_recorder.annotation_queue import enqueue_if_flagged
from ai_flight_recorder.config import GovernanceConfig
from ai_flight_recorder.emit import EmitResult, emit
from ai_flight_recorder.hashing import compute_integrity_hash
from ai_flight_recorder.rules import RuleEngine
from ai_flight_recorder.schema import ConfidenceSignal, ConfidenceSource, DecisionRecord

logger = logging.getLogger("ai_flight_recorder.capture")

_SELF_REPORT_PROMPT = (
    "On a scale from 0 to 1, how confident are you in the response you just gave? "
    "Reply with only a number between 0 and 1."
)
_LOGPROBS_NOTE = (
    "geometric mean of per-token logprobs; a crude proxy, not a calibrated probability"
)
_SELF_REPORT_NOTE = (
    "model self-reported its own confidence via a follow-up prompt; "
    "unreliable, not a calibrated probability"
)
_SELF_REPORT_UNPARSEABLE_NOTE = (
    "self_report requested but the model's reply could not be parsed as a 0-1 value"
)
_SELF_REPORT_CALL_FAILED_NOTE = (
    "self_report requested but the follow-up confidence-rating call itself failed"
)
_SELF_REPORT_PASSTHROUGH_KWARGS = ("api_key", "api_base", "custom_llm_provider", "mock_response")


def _get_default_client() -> Any:
    from langfuse import get_client

    return get_client()


def _confidence_from_logprobs(response: Any) -> float | None:
    """Geometric mean of per-token logprobs, if the response exposes them."""
    try:
        content = response.choices[0].logprobs.content
    except (AttributeError, IndexError, TypeError):
        return None
    if not content:
        return None
    logprobs = [tok.logprob for tok in content]
    if not logprobs:
        return None
    return math.exp(sum(logprobs) / len(logprobs))


def _parse_confidence_value(text: str | None) -> float | None:
    if not text:
        return None
    try:
        value = float(text.strip().split()[0])
    except (ValueError, IndexError):
        return None
    return value if 0.0 <= value <= 1.0 else None


class Decision:
    """The handle yielded by `decision()` and injected into `audit_decision` functions."""

    def __init__(self, record: DecisionRecord, confidence_method: str | None) -> None:
        self.record = record
        self.emit_result: EmitResult | None = None
        self._confidence_method = confidence_method

    @property
    def outcome(self) -> Any:
        return self.record.outcome

    @outcome.setter
    def outcome(self, value: Any) -> None:
        self.record.outcome = value

    def completion(self, **kwargs: Any) -> Any:
        """Wraps `litellm.completion`, auto-capturing model, model_version,
        latency_ms, and a confidence signal."""
        self._append_prompt(kwargs.get("messages"))
        start = time.perf_counter()
        response = litellm.completion(**kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        confidence = self._apply_response_metadata(response, kwargs, elapsed_ms)
        if confidence is not None:
            self.record.confidence.append(
                ConfidenceSignal(source=ConfidenceSource.LOGPROBS, value=confidence, note=_LOGPROBS_NOTE)
            )
        elif self._confidence_method == "self_report":
            self.record.confidence.append(self._self_report_confidence(response, kwargs))
        else:
            self.record.confidence.append(ConfidenceSignal(source=ConfidenceSource.NONE))
        return response

    async def acompletion(self, **kwargs: Any) -> Any:
        """Async counterpart of `completion`, wraps `litellm.acompletion`."""
        self._append_prompt(kwargs.get("messages"))
        start = time.perf_counter()
        response = await litellm.acompletion(**kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        confidence = self._apply_response_metadata(response, kwargs, elapsed_ms)
        if confidence is not None:
            self.record.confidence.append(
                ConfidenceSignal(source=ConfidenceSource.LOGPROBS, value=confidence, note=_LOGPROBS_NOTE)
            )
        elif self._confidence_method == "self_report":
            self.record.confidence.append(await self._self_report_confidence_async(response, kwargs))
        else:
            self.record.confidence.append(ConfidenceSignal(source=ConfidenceSource.NONE))
        return response

    def _append_prompt(self, messages: Any) -> None:
        """Accumulate messages across multiple completion()/acompletion() calls
        within one decision, so pii_in_inputs scans everything actually sent
        to the model instead of only the most recent call."""
        if messages is None:
            return
        existing = self.record.prompt
        if isinstance(existing, list) and isinstance(messages, list):
            self.record.prompt = existing + messages
        else:
            self.record.prompt = messages

    def _apply_response_metadata(self, response: Any, kwargs: dict, elapsed_ms: float) -> float | None:
        self.record.model = getattr(response, "model", kwargs.get("model"))
        self.record.model_version = getattr(response, "system_fingerprint", None)
        self.record.latency_ms = elapsed_ms
        return _confidence_from_logprobs(response)

    def _self_report_kwargs(self, response: Any, kwargs: dict) -> dict:
        messages = list(kwargs.get("messages") or [])
        try:
            assistant_content = response.choices[0].message.content
        except (AttributeError, IndexError):
            assistant_content = ""
        followup = {k: kwargs[k] for k in _SELF_REPORT_PASSTHROUGH_KWARGS if k in kwargs}
        followup["model"] = kwargs.get("model")
        followup["messages"] = messages + [
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": _SELF_REPORT_PROMPT},
        ]
        followup["max_tokens"] = 5
        return followup

    def _self_report_confidence(self, response: Any, kwargs: dict) -> ConfidenceSignal:
        try:
            followup_response = litellm.completion(**self._self_report_kwargs(response, kwargs))
        except Exception:
            return self._self_report_failed_signal()
        return self._build_self_report_signal(followup_response)

    async def _self_report_confidence_async(self, response: Any, kwargs: dict) -> ConfidenceSignal:
        try:
            followup_response = await litellm.acompletion(**self._self_report_kwargs(response, kwargs))
        except Exception:
            return self._self_report_failed_signal()
        return self._build_self_report_signal(followup_response)

    def _self_report_failed_signal(self) -> ConfidenceSignal:
        """The self-report follow-up call is an opt-in, best-effort side call, not
        part of the primary completion. A provider rejecting one of its kwargs
        (e.g. Azure requiring `api_version`, which isn't in the passthrough list)
        or a transient network error must not fail the decision the caller is
        actually trying to capture."""
        logger.warning(
            "ai_flight_recorder: self_report confidence follow-up call failed; recording "
            "no confidence signal for this decision instead of failing it.",
            exc_info=True,
        )
        return ConfidenceSignal(source=ConfidenceSource.NONE, note=_SELF_REPORT_CALL_FAILED_NOTE)

    @staticmethod
    def _build_self_report_signal(followup_response: Any) -> ConfidenceSignal:
        text = followup_response.choices[0].message.content
        value = _parse_confidence_value(text)
        if value is None:
            return ConfidenceSignal(source=ConfidenceSource.NONE, note=_SELF_REPORT_UNPARSEABLE_NOTE)
        return ConfidenceSignal(source=ConfidenceSource.SELF_REPORT, value=value, note=_SELF_REPORT_NOTE)


class _DecisionContext:
    def __init__(
        self,
        *,
        decision_type: str,
        subject_ref: str,
        options: list[Any] | None,
        decision_id: str | None,
        jurisdiction: str | None,
        destination: str | None,
        confidence_method: str | None,
        config: GovernanceConfig | None,
        client: Any | None,
    ) -> None:
        self._config = config or GovernanceConfig()
        self._client = client or _get_default_client()
        record_kwargs: dict[str, Any] = dict(
            decision_type=decision_type,
            subject_ref=subject_ref,
            options=options or [],
            jurisdiction=jurisdiction,
            destination=destination,
        )
        if decision_id is not None:
            record_kwargs["decision_id"] = decision_id
        self._decision = Decision(DecisionRecord(**record_kwargs), confidence_method)

    def __enter__(self) -> Decision:
        return self._decision

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> None:
        if exc_type is not None:
            return None
        self._finalize()
        return None

    async def __aenter__(self) -> Decision:
        return self._decision

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any
    ) -> None:
        if exc_type is not None:
            return None
        # _finalize() ends in a blocking client.flush() network call; running it
        # via to_thread keeps an `async with`/async-decorator exit from stalling
        # the event loop for every other coroutine scheduled on it.
        await asyncio.to_thread(self._finalize)
        return None

    def _will_enqueue(self, record: DecisionRecord) -> bool:
        return bool(
            record.review_required and self._config.review_queue_name and self._config.auto_enqueue
        )

    def _finalize(self) -> None:
        record = self._decision.record
        try:
            RuleEngine(self._config).run(record)
            record.integrity_hash = compute_integrity_hash(record)
            self._decision.emit_result = emit(record, self._client)
            # auto_flush=True (the default) always flushes here, which is the
            # safe choice for scripts/serverless. With auto_flush=False, still
            # flush when this decision is about to be enqueued: annotation
            # queue items reference the trace by id, so the trace has to exist
            # in Langfuse before we can point a queue item at it. See
            # annotation_queue.py for the guarded, side-effect only enqueue
            # step this never lets bubble into the decision flow.
            if self._config.auto_flush or self._will_enqueue(record):
                self._client.flush()
            enqueue_if_flagged(record, self._decision.emit_result.trace_id, self._client, self._config)
        except Exception:
            # A failure here (e.g. Langfuse unreachable during flush) must not
            # clobber a business result the wrapped function already computed
            # successfully — log and move on rather than letting it propagate
            # out of __exit__/__aexit__ mid-`return`.
            logger.warning(
                "ai_flight_recorder: failed to finalize decision '%s'; the wrapped "
                "function's own result is unaffected, but this decision was not "
                "fully captured in Langfuse.",
                record.decision_id,
                exc_info=True,
            )


def decision(
    *,
    decision_type: str,
    subject_ref: str,
    options: list[Any] | None = None,
    decision_id: str | None = None,
    jurisdiction: str | None = None,
    destination: str | None = None,
    confidence_method: str | None = None,
    config: GovernanceConfig | None = None,
    client: Any | None = None,
) -> _DecisionContext:
    """Open a governance decision:

        with ai_flight_recorder.decision(decision_type="loan_eligibility", subject_ref="cust_123") as d:
            resp = d.completion(model="gpt-4o-mini", messages=[...])
            d.outcome = "approve"

    `confidence_method="self_report"` is opt-in and off by default — it costs
    one extra LLM call and the result is not a reliability guarantee. If that
    follow-up call itself fails (e.g. a provider rejecting one of its kwargs),
    ai-flight-recorder records `ConfidenceSource.NONE` for that decision instead of
    aborting it.
    """
    return _DecisionContext(
        decision_type=decision_type,
        subject_ref=subject_ref,
        options=options,
        decision_id=decision_id,
        jurisdiction=jurisdiction,
        destination=destination,
        confidence_method=confidence_method,
        config=config,
        client=client,
    )


def audit_decision(
    *,
    decision_type: str,
    subject_ref: Callable[..., str] | str,
    options: Callable[..., list[Any]] | list[Any] | None = None,
    decision_id: Callable[..., str] | str | None = None,
    jurisdiction: str | None = None,
    destination: str | None = None,
    confidence_method: str | None = None,
    config: GovernanceConfig | None = None,
    client: Any | None = None,
) -> Callable[[Callable], Callable]:
    """Decorate a function whose first parameter receives the `Decision` handle.

    `subject_ref`, `options`, and `decision_id` may be a static value or a
    callable invoked with the wrapped function's own call arguments:

        @audit_decision(decision_type="loan_eligibility", subject_ref=lambda applicant_id: applicant_id)
        def evaluate(d, applicant_id):
            resp = d.completion(model="gpt-4o-mini", messages=[...])
            return "approve"

        evaluate("cust_123")

    If the function doesn't set `d.outcome` explicitly, its return value is
    used as the outcome. Works with `async def` functions too.
    """

    def _resolve(value: Any, args: tuple, kwargs: dict) -> Any:
        return value(*args, **kwargs) if callable(value) else value

    def decorator(fn: Callable) -> Callable:
        def _open(args: tuple, kwargs: dict) -> _DecisionContext:
            return decision(
                decision_type=decision_type,
                subject_ref=_resolve(subject_ref, args, kwargs),
                options=_resolve(options, args, kwargs) or [],
                decision_id=_resolve(decision_id, args, kwargs),
                jurisdiction=jurisdiction,
                destination=destination,
                confidence_method=confidence_method,
                config=config,
                client=client,
            )

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async with _open(args, kwargs) as d:
                    result = await fn(d, *args, **kwargs)
                    if d.outcome is None:
                        d.outcome = result
                    return result

            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with _open(args, kwargs) as d:
                result = fn(d, *args, **kwargs)
                if d.outcome is None:
                    d.outcome = result
                return result

        return wrapper

    return decorator
