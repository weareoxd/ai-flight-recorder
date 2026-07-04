"""The governance decision schema.

This module is the spec: a `DecisionRecord` is the opinionated, governance-shaped
record ai-flight-recorder captures for every AI-influenced decision. Nothing here is
Langfuse-specific — see `emit.py` for how these fields map onto Langfuse's trace
metadata / observation input-output / scores (the A/B/C split documented there).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Imported lazily-shaped to avoid a hard dependency loop: the regex detector is
# used only to *reject* obviously-raw-PII subject_ref values as a guardrail, not
# to redact anything here.
from ai_flight_recorder.detectors.regex import RegexDetector

_SUBJECT_REF_GUARD = RegexDetector()


class ConfidenceSource(str, Enum):
    """Where a confidence value came from. Not a reliability ranking."""

    LOGPROBS = "logprobs"
    SELF_REPORT = "self_report"
    NONE = "none"


class ConfidenceSignal(BaseModel):
    """A single recorded confidence signal.

    This is a traceability artifact, not a guarantee the decision was correct.
    """

    source: ConfidenceSource
    value: float | None = Field(default=None, validate_default=True)
    note: str | None = None

    @field_validator("value")
    @classmethod
    def _value_required_unless_none_source(cls, v: float | None, info) -> float | None:
        source = info.data.get("source")
        if source == ConfidenceSource.NONE and v is not None:
            raise ValueError("confidence value must be None when source is 'none'")
        if source != ConfidenceSource.NONE and v is None:
            raise ValueError(f"confidence value is required when source is '{source}'")
        return v


class Flag(BaseModel):
    """A single rule-engine finding. Routes attention; never a verdict."""

    rule_name: str
    reason: str


class DecisionRecord(BaseModel):
    """A governance-shaped record of one AI-influenced decision.

    Field groups map onto Langfuse as documented in `emit.py`:
      (A) propagated metadata, (B) observation input/output, (C) scores,
      (D) regular (non-propagated) observation metadata.
    """

    # hide_input_in_errors: pydantic's default ValidationError.__str__ echoes
    # the raw field value back (e.g. `input_value='jane.doe@example.com'`) —
    # that would defeat the subject_ref PII guardrail below the moment anyone
    # logs or prints the exception. Note this only covers str()/repr(); the
    # structured `.errors()` API still contains the raw input value.
    #
    # validate_assignment: without it, the subject_ref guardrail below only
    # runs at construction time — `record.subject_ref = "jane@example.com"`
    # would otherwise silently bypass it.
    model_config = ConfigDict(hide_input_in_errors=True, validate_assignment=True)

    # identity
    decision_id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # classification
    decision_type: str
    subject_ref: str

    # decision content
    options: list[Any] = Field(default_factory=list)
    outcome: Any | None = None

    # execution metadata, auto-captured from the LiteLLM response
    model: str | None = None
    model_version: str | None = None
    latency_ms: float | None = None

    # confidence signals — recorded, not a reliability guarantee
    confidence: list[ConfidenceSignal] = Field(default_factory=list)

    # context
    jurisdiction: str | None = None
    destination: str | None = None

    # prompt capture
    prompt: str | list[dict[str, Any]] | None = None
    redacted_prompt: str | list[dict[str, Any]] | None = None

    # governance
    flags: list[Flag] = Field(default_factory=list)
    review_required: bool = False

    # integrity, computed last over everything above (see hashing.py)
    integrity_hash: str | None = None

    @field_validator("timestamp")
    @classmethod
    def _require_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware (UTC)")
        return v.astimezone(timezone.utc)

    @field_validator("subject_ref")
    @classmethod
    def _reject_raw_pii(cls, v: str) -> str:
        """Guardrail, not a guarantee: catches the common case of a caller
        accidentally passing an email/phone/SSN/credit-card/IP as subject_ref
        instead of an opaque identifier."""
        hits = _SUBJECT_REF_GUARD.detect(v)
        if hits:
            entities = ", ".join(sorted({h.entity_type for h in hits}))
            raise ValueError(
                f"subject_ref looks like it contains raw PII ({entities}); "
                "use an opaque identifier instead (e.g. an internal customer id)"
            )
        return v
