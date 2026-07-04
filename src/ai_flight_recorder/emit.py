"""The Langfuse emit layer: the A/B/C(/D) field split in action.

    (A) propagated metadata — short scalars, <=200 chars, via propagate_attributes()
    (B) observation input/output — prompt/options as input, outcome as output
    (C) scores — review_required (boolean), written now with a deterministic id
    (D) regular (non-propagated) observation metadata — confidence, full flags,
        destination, model_version; no length cap since it isn't baggage

All three(/four) writes land on the same trace because the trace id is derived
deterministically from decision_id via `client.create_trace_id(seed=...)`.

Emitting does NOT flush. Callers (see `capture.py`) must call `client.flush()`
themselves, especially in short-lived scripts, or buffered events are lost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langfuse import propagate_attributes

from ai_flight_recorder.schema import DecisionRecord

logger = logging.getLogger("ai_flight_recorder.emit")

MAX_PROPAGATED_VALUE_LEN = 200
_TRUNCATION_MARKER = "...[truncated]"

REVIEW_REQUIRED_SCORE_NAME = "review_required"


def _guard_scalar(key: str, value: str) -> str:
    """Enforce Langfuse's <=200-char propagated-attribute limit without silently
    dropping data: truncate with a visible marker and warn. The untruncated
    value still lives in bucket (D)/(B), so nothing is actually lost."""
    if len(value) <= MAX_PROPAGATED_VALUE_LEN:
        return value
    truncated = value[: MAX_PROPAGATED_VALUE_LEN - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER
    logger.warning(
        "ai_flight_recorder: propagated metadata field '%s' is %d chars, exceeding the "
        "%d-char Langfuse limit; truncated for the filterable field. The full "
        "value is still recorded in observation metadata/input.",
        key,
        len(value),
        MAX_PROPAGATED_VALUE_LEN,
    )
    return truncated


def build_propagated_metadata(record: DecisionRecord) -> dict[str, str]:
    """(A) Short scalar strings only — the filterable fields in the Langfuse UI."""
    metadata: dict[str, str] = {
        "decision_id": _guard_scalar("decision_id", record.decision_id),
        "decision_type": _guard_scalar("decision_type", record.decision_type),
        "subject_ref": _guard_scalar("subject_ref", record.subject_ref),
    }
    if record.jurisdiction:
        metadata["jurisdiction"] = _guard_scalar("jurisdiction", record.jurisdiction)
    if record.integrity_hash:
        metadata["integrity_hash"] = _guard_scalar("integrity_hash", record.integrity_hash)
    if record.flags:
        names = list(dict.fromkeys(f.rule_name for f in record.flags))
        metadata["flags"] = _guard_scalar("flags", ",".join(names))
    return metadata


def build_regular_metadata(record: DecisionRecord) -> dict[str, Any]:
    """(D) Regular (non-propagated) observation metadata — no length cap."""
    metadata: dict[str, Any] = {
        "confidence": [signal.model_dump(mode="json") for signal in record.confidence],
        "flags": [flag.model_dump() for flag in record.flags],
    }
    if record.destination:
        metadata["destination"] = record.destination
    if record.model_version:
        metadata["model_version"] = record.model_version
    return metadata


def build_input(record: DecisionRecord) -> dict[str, Any]:
    """(B) input — redacted_prompt takes precedence over raw prompt when set."""
    prompt = record.redacted_prompt if record.redacted_prompt is not None else record.prompt
    return {"prompt": prompt, "options": record.options}


@dataclass(frozen=True)
class EmitResult:
    trace_id: str
    observation_id: str


def emit(record: DecisionRecord, client: Any) -> EmitResult:
    """Write a finished `DecisionRecord` to Langfuse across the A/B/C(/D) split.

    `client` is a Langfuse SDK v3 client (or anything implementing the same
    subset of its interface). Does not call `client.flush()` — the caller owns
    that, since batching flush across multiple decisions is often desirable.
    """
    trace_id = client.create_trace_id(seed=record.decision_id)

    propagated = build_propagated_metadata(record)
    regular_metadata = build_regular_metadata(record)
    input_ = build_input(record)

    with propagate_attributes(metadata=propagated):
        with client.start_as_current_observation(
            trace_context={"trace_id": trace_id},
            name=record.decision_type,
            as_type="generation",
            input=input_,
            output=record.outcome,
            metadata=regular_metadata,
            model=record.model,
        ) as observation:
            client.create_score(
                name=REVIEW_REQUIRED_SCORE_NAME,
                value=1.0 if record.review_required else 0.0,
                data_type="BOOLEAN",
                trace_id=trace_id,
                score_id=f"{record.decision_id}-{REVIEW_REQUIRED_SCORE_NAME}",
            )
            observation_id = observation.id

    return EmitResult(trace_id=trace_id, observation_id=observation_id)
