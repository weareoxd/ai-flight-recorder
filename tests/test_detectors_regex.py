import re

from ai_flight_recorder.detectors.regex import CustomPattern, RegexDetector


def test_no_pii_returns_empty():
    assert RegexDetector().detect("just a normal sentence with no secrets") == []


def test_detects_email():
    hits = RegexDetector().detect("contact me at jane.doe@example.com please")
    assert any(h.entity_type == "EMAIL" and h.value == "jane.doe@example.com" for h in hits)


def test_detects_phone():
    hits = RegexDetector().detect("call 555-123-4567 tomorrow")
    assert any(h.entity_type == "PHONE" and h.value == "555-123-4567" for h in hits)


def test_detects_credit_card_valid_luhn():
    # well-known Visa test number, passes Luhn
    hits = RegexDetector().detect("card number 4111111111111111 on file")
    assert any(h.entity_type == "CREDIT_CARD" for h in hits)


def test_rejects_credit_card_invalid_luhn():
    # same number with the last digit changed, fails Luhn
    hits = RegexDetector().detect("card number 4111111111111112 on file")
    assert not any(h.entity_type == "CREDIT_CARD" for h in hits)


def test_detects_ssn():
    hits = RegexDetector().detect("SSN on file: 123-45-6789")
    assert any(h.entity_type == "SSN" and h.value == "123-45-6789" for h in hits)


def test_detects_sin_not_misclassified_as_ssn():
    hits = RegexDetector().detect("SIN on file: 123-456-789")
    entity_types = {h.entity_type for h in hits}
    assert "SIN" in entity_types
    assert "SSN" not in entity_types


def test_detects_ip_address():
    hits = RegexDetector().detect("request came from 192.168.1.1 today")
    assert any(h.entity_type == "IP_ADDRESS" and h.value == "192.168.1.1" for h in hits)


def test_custom_pattern():
    detector = RegexDetector(
        custom_patterns=[CustomPattern("employee_id", re.compile(r"\bEMP-\d{6}\b"))]
    )
    hits = detector.detect("assigned to EMP-004821 for review")
    assert any(h.entity_type == "employee_id" and h.value == "EMP-004821" for h in hits)


def test_redact_replaces_all_matches_and_hides_raw_value():
    text = "email jane.doe@example.com or call 555-123-4567"
    redacted = RegexDetector().redact(text)
    assert "jane.doe@example.com" not in redacted
    assert "555-123-4567" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted


def test_redact_noop_when_no_pii():
    text = "nothing sensitive here"
    assert RegexDetector().redact(text) == text
