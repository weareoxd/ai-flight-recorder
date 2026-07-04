"""ai-flight-recorder: a thin governance layer on top of Langfuse.

Wraps an AI-influenced decision made via the LiteLLM SDK, captures a
governance-shaped decision record, and emits it into Langfuse using Langfuse's
own SDK. Human review happens in Langfuse's annotation UI — ai-flight-recorder does
not build its own storage or viewer.
"""

from ai_flight_recorder.capture import Decision, audit_decision, decision
from ai_flight_recorder.config import GovernanceConfig
from ai_flight_recorder.detectors import Detector, PIIMatch
from ai_flight_recorder.detectors.regex import CustomPattern, RegexDetector
from ai_flight_recorder.emit import EmitResult, emit
from ai_flight_recorder.hashing import VerifyResult, compute_integrity_hash, verify
from ai_flight_recorder.rules import Rule, RuleEngine
from ai_flight_recorder.schema import ConfidenceSignal, ConfidenceSource, DecisionRecord, Flag
from ai_flight_recorder.scoring import ensure_review_queue, setup_score_configs

__all__ = [
    "Decision",
    "audit_decision",
    "decision",
    "GovernanceConfig",
    "Detector",
    "PIIMatch",
    "CustomPattern",
    "RegexDetector",
    "EmitResult",
    "emit",
    "VerifyResult",
    "compute_integrity_hash",
    "verify",
    "Rule",
    "RuleEngine",
    "ConfidenceSignal",
    "ConfidenceSource",
    "DecisionRecord",
    "Flag",
    "ensure_review_queue",
    "setup_score_configs",
]
