"""The rule engine: flags -> review_required.

Rules only route human ATTENTION. They never render an approve/reject/quality
verdict on the decision itself — that judgment is for the human reviewer in
Langfuse's annotation UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ai_flight_recorder.config import GovernanceConfig
from ai_flight_recorder.detectors import PIIMatch, redact_matches
from ai_flight_recorder.schema import ConfidenceSource, DecisionRecord, Flag

RulePredicate = Callable[[DecisionRecord, GovernanceConfig], "str | None"]


@dataclass(frozen=True)
class Rule:
    """A named predicate over the finished record.

    `predicate` returns a reason string if the rule fires, or None otherwise.
    It receives the live record and may mutate it (the built-in pii_in_inputs
    rule uses this to populate `redacted_prompt`).
    """

    name: str
    predicate: RulePredicate
    triggers_review: bool = True


def _prompt_segments(prompt: Any) -> list[str]:
    """Every string-content chunk in `prompt`, in order. A bare string prompt
    yields itself as the only segment."""
    if prompt is None:
        return []
    if isinstance(prompt, str):
        return [prompt]
    return [
        msg["content"]
        for msg in prompt
        if isinstance(msg, dict) and isinstance(msg.get("content"), str)
    ]


def _redact_prompt(prompt: Any, hits_per_segment: list[list[PIIMatch]]) -> Any:
    if isinstance(prompt, str):
        return redact_matches(prompt, hits_per_segment[0])
    redacted = []
    segment_index = 0
    for msg in prompt:
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            redacted.append(
                {**msg, "content": redact_matches(msg["content"], hits_per_segment[segment_index])}
            )
            segment_index += 1
        else:
            redacted.append(msg)
    return redacted


def _rule_low_confidence(record: DecisionRecord, config: GovernanceConfig) -> str | None:
    for signal in record.confidence:
        if signal.source == ConfidenceSource.NONE or signal.value is None:
            continue
        if signal.value < config.low_confidence_threshold:
            return (
                f"confidence {signal.value} from source '{signal.source.value}' "
                f"is below threshold {config.low_confidence_threshold}"
            )
    return None


def _rule_high_stakes_type(record: DecisionRecord, config: GovernanceConfig) -> str | None:
    if record.decision_type in config.high_stakes_types:
        return f"decision_type '{record.decision_type}' is configured as high-stakes"
    return None


def _rule_sensitive_outcome(record: DecisionRecord, config: GovernanceConfig) -> str | None:
    try:
        hit = record.outcome in config.sensitive_outcomes
    except TypeError:
        return None  # unhashable outcome (e.g. dict/list); rule doesn't apply
    if hit:
        return f"outcome '{record.outcome}' is configured as sensitive"
    return None


def _rule_pii_in_inputs(record: DecisionRecord, config: GovernanceConfig) -> str | None:
    segments = _prompt_segments(record.prompt)
    if not segments:
        return None

    # Detect once per segment and reuse the same matches for both the flag's
    # entity-type summary and redaction, instead of detecting the joined text
    # once and then having redaction re-detect per message from scratch.
    hits_per_segment = [config.pii_detector.detect(segment) for segment in segments]
    all_hits = [hit for hits in hits_per_segment for hit in hits]
    if not all_hits:
        return None

    if config.redact_pii_in_input:
        record.redacted_prompt = _redact_prompt(record.prompt, hits_per_segment)

    entity_types = sorted({h.entity_type for h in all_hits})
    return f"detected PII entity types in prompt: {', '.join(entity_types)}"


BUILTIN_RULES: dict[str, Rule] = {
    "low_confidence": Rule("low_confidence", _rule_low_confidence),
    "high_stakes_type": Rule("high_stakes_type", _rule_high_stakes_type),
    "sensitive_outcome": Rule("sensitive_outcome", _rule_sensitive_outcome),
    "pii_in_inputs": Rule("pii_in_inputs", _rule_pii_in_inputs),
}


class RuleEngine:
    """Runs a set of rules over a finished `DecisionRecord`, appending flags
    and setting `review_required` in place. Emits no quality judgment."""

    def __init__(self, config: GovernanceConfig, extra_rules: list[Rule] | None = None) -> None:
        self._config = config
        self._rules: list[Rule] = [
            rule for name, rule in BUILTIN_RULES.items() if name in config.enabled_rules
        ]
        self._rules.extend(extra_rules or [])

    def run(self, record: DecisionRecord) -> DecisionRecord:
        for rule in self._rules:
            reason = rule.predicate(record, self._config)
            if reason is None:
                continue
            record.flags.append(Flag(rule_name=rule.name, reason=reason))
            if rule.triggers_review:
                record.review_required = True
        return record
