"""Governance configuration: thresholds, toggles, and the active PII detector.

This is deliberately plain Python (not pydantic) since it holds a `Detector`
instance, which is a runtime dependency, not serializable config.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from ai_flight_recorder.detectors import Detector
from ai_flight_recorder.detectors.regex import RegexDetector

DEFAULT_ENABLED_RULES = frozenset(
    {"low_confidence", "high_stakes_type", "sensitive_outcome", "pii_in_inputs"}
)

# Cap on the enqueue-dedup cache below, so a long-lived GovernanceConfig used
# across many decisions doesn't grow this without bound.
_MAX_ENQUEUE_DEDUP_ENTRIES = 10_000


@dataclass
class GovernanceConfig:
    # low_confidence fires when a recorded confidence value (source != "none")
    # falls below this threshold.
    low_confidence_threshold: float = 0.5

    # high_stakes_type fires when decision_type is in this set.
    high_stakes_types: set[str] = field(default_factory=set)

    # sensitive_outcome fires when outcome equals a member of this set.
    sensitive_outcomes: set[str] = field(default_factory=set)

    # Which built-in (or custom) rule names are active.
    enabled_rules: set[str] = field(default_factory=lambda: set(DEFAULT_ENABLED_RULES))

    # Detector used by the pii_in_inputs rule. Swap in a PresidioDetector or a
    # RegexDetector with custom_patterns for domain-specific entities.
    pii_detector: Detector = field(default_factory=RegexDetector)

    # When pii_in_inputs fires, populate record.redacted_prompt and prefer it
    # as the Langfuse observation input over the raw prompt.
    redact_pii_in_input: bool = True

    # Name of the Langfuse annotation queue flagged decisions are enqueued
    # into (see annotation_queue.py). None disables enqueueing entirely.
    review_queue_name: str | None = None

    # Only enqueue when review_queue_name is set. Exists as a separate switch
    # so a queue name can stay configured while enqueueing is toggled off,
    # e.g. temporarily, without clearing the name.
    auto_enqueue: bool = True

    # Flush the Langfuse client after every decision. True (the default) is
    # the safe choice for scripts/serverless, where the process may exit
    # before Langfuse's own background thread would otherwise send buffered
    # events. In a long-running service, set to False to let the SDK batch
    # flushes on its own schedule instead of paying a network round trip on
    # every decision. capture.py still flushes immediately regardless of
    # this flag when a decision is about to be enqueued into a review queue,
    # since that step needs the trace to already exist in Langfuse.
    auto_flush: bool = True

    # Internal caches for annotation-queue enqueueing (queue name -> resolved
    # queueId; decision_ids already enqueued). Scoped to this config instance,
    # not process-global, so a fresh GovernanceConfig starts with a clean
    # dedup slate. Not meant to be set by callers.
    _queue_id_cache: dict[str, str] = field(default_factory=dict, init=False, repr=False, compare=False)
    # Bounded FIFO: capped at _MAX_ENQUEUE_DEDUP_ENTRIES via _mark_enqueued()
    # below, so a long-running process reusing one config doesn't leak memory
    # proportional to total flagged-decision volume.
    _enqueued_decision_ids: "OrderedDict[str, None]" = field(
        default_factory=OrderedDict, init=False, repr=False, compare=False
    )

    def _mark_enqueued(self, decision_id: str) -> None:
        self._enqueued_decision_ids[decision_id] = None
        if len(self._enqueued_decision_ids) > _MAX_ENQUEUE_DEDUP_ENTRIES:
            self._enqueued_decision_ids.popitem(last=False)
