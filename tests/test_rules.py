from datetime import datetime, timezone

from ai_flight_recorder.config import GovernanceConfig
from ai_flight_recorder.rules import Rule, RuleEngine
from ai_flight_recorder.schema import ConfidenceSignal, ConfidenceSource, DecisionRecord


def _make_record(**overrides) -> DecisionRecord:
    defaults = dict(
        decision_id="dec-001",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        decision_type="loan_eligibility",
        subject_ref="cust_12345",
        options=["approve", "deny"],
        outcome="approve",
    )
    defaults.update(overrides)
    return DecisionRecord(**defaults)


def test_no_rules_fire_on_a_clean_record():
    record = _make_record()
    engine = RuleEngine(GovernanceConfig())
    engine.run(record)
    assert record.flags == []
    assert record.review_required is False


def test_low_confidence_rule_fires_below_threshold():
    record = _make_record(
        confidence=[ConfidenceSignal(source=ConfidenceSource.LOGPROBS, value=0.2)]
    )
    engine = RuleEngine(GovernanceConfig(low_confidence_threshold=0.5))
    engine.run(record)
    assert any(f.rule_name == "low_confidence" for f in record.flags)
    assert record.review_required is True


def test_low_confidence_rule_does_not_fire_above_threshold():
    record = _make_record(
        confidence=[ConfidenceSignal(source=ConfidenceSource.LOGPROBS, value=0.9)]
    )
    engine = RuleEngine(GovernanceConfig(low_confidence_threshold=0.5))
    engine.run(record)
    assert not any(f.rule_name == "low_confidence" for f in record.flags)


def test_low_confidence_rule_ignores_none_source():
    record = _make_record(confidence=[ConfidenceSignal(source=ConfidenceSource.NONE)])
    engine = RuleEngine(GovernanceConfig(low_confidence_threshold=0.9))
    engine.run(record)
    assert not any(f.rule_name == "low_confidence" for f in record.flags)


def test_high_stakes_type_rule_fires():
    record = _make_record(decision_type="loan_eligibility")
    engine = RuleEngine(GovernanceConfig(high_stakes_types={"loan_eligibility"}))
    engine.run(record)
    assert any(f.rule_name == "high_stakes_type" for f in record.flags)
    assert record.review_required is True


def test_high_stakes_type_rule_does_not_fire_for_other_types():
    record = _make_record(decision_type="content_recommendation")
    engine = RuleEngine(GovernanceConfig(high_stakes_types={"loan_eligibility"}))
    engine.run(record)
    assert not any(f.rule_name == "high_stakes_type" for f in record.flags)


def test_sensitive_outcome_rule_fires():
    record = _make_record(outcome="deny")
    engine = RuleEngine(GovernanceConfig(sensitive_outcomes={"deny"}))
    engine.run(record)
    assert any(f.rule_name == "sensitive_outcome" for f in record.flags)
    assert record.review_required is True


def test_sensitive_outcome_rule_skips_unhashable_outcome():
    record = _make_record(outcome=["deny", "escalate"])
    engine = RuleEngine(GovernanceConfig(sensitive_outcomes={"deny"}))
    # must not raise even though list outcomes aren't hashable
    engine.run(record)
    assert not any(f.rule_name == "sensitive_outcome" for f in record.flags)


def test_pii_in_inputs_rule_fires_and_redacts():
    record = _make_record(
        prompt=[{"role": "user", "content": "my email is jane.doe@example.com"}]
    )
    engine = RuleEngine(GovernanceConfig())
    engine.run(record)
    flag = next(f for f in record.flags if f.rule_name == "pii_in_inputs")
    assert "EMAIL" in flag.reason
    assert "jane.doe@example.com" not in flag.reason  # reason never carries raw PII
    assert record.redacted_prompt is not None
    assert "jane.doe@example.com" not in str(record.redacted_prompt)
    assert record.review_required is True


def test_pii_in_inputs_rule_respects_redact_toggle():
    record = _make_record(prompt="contact jane.doe@example.com")
    engine = RuleEngine(GovernanceConfig(redact_pii_in_input=False))
    engine.run(record)
    assert any(f.rule_name == "pii_in_inputs" for f in record.flags)
    assert record.redacted_prompt is None


def test_pii_in_inputs_rule_does_not_fire_on_clean_prompt():
    record = _make_record(prompt="just a normal question")
    engine = RuleEngine(GovernanceConfig())
    engine.run(record)
    assert not any(f.rule_name == "pii_in_inputs" for f in record.flags)


def test_disabled_rules_do_not_run():
    record = _make_record(decision_type="loan_eligibility")
    config = GovernanceConfig(
        high_stakes_types={"loan_eligibility"},
        enabled_rules={"low_confidence"},  # high_stakes_type excluded
    )
    engine = RuleEngine(config)
    engine.run(record)
    assert not any(f.rule_name == "high_stakes_type" for f in record.flags)


def test_custom_extra_rule_runs_alongside_builtins():
    record = _make_record()
    custom = Rule(name="always_fires", predicate=lambda r, c: "custom reason")
    engine = RuleEngine(GovernanceConfig(), extra_rules=[custom])
    engine.run(record)
    assert any(f.rule_name == "always_fires" and f.reason == "custom reason" for f in record.flags)


def test_custom_rule_can_opt_out_of_triggering_review():
    record = _make_record()
    custom = Rule(name="informational", predicate=lambda r, c: "fyi", triggers_review=False)
    engine = RuleEngine(GovernanceConfig(), extra_rules=[custom])
    engine.run(record)
    assert any(f.rule_name == "informational" for f in record.flags)
    assert record.review_required is False


def test_multiple_rules_all_flagged_and_review_required_true():
    record = _make_record(
        decision_type="loan_eligibility",
        outcome="deny",
        confidence=[ConfidenceSignal(source=ConfidenceSource.LOGPROBS, value=0.1)],
    )
    config = GovernanceConfig(
        high_stakes_types={"loan_eligibility"},
        sensitive_outcomes={"deny"},
        low_confidence_threshold=0.5,
    )
    engine = RuleEngine(config)
    engine.run(record)
    fired = {f.rule_name for f in record.flags}
    assert fired == {"high_stakes_type", "sensitive_outcome", "low_confidence"}
    assert record.review_required is True
