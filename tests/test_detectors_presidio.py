"""These tests never import the real presidio-analyzer package — a fake
analyzer is injected via the `analyzer=` constructor param, so this test file
runs regardless of whether the optional `presidio` extra is installed."""

import sys
from types import SimpleNamespace

import pytest

from ai_flight_recorder.detectors.presidio import PresidioDetector


class FakeAnalyzer:
    def __init__(self, results):
        self._results = results
        self.analyze_calls = []

    def analyze(self, *, text, language, score_threshold):
        self.analyze_calls.append(
            {"text": text, "language": language, "score_threshold": score_threshold}
        )
        return self._results


def _result(entity_type, start, end, score=0.9):
    return SimpleNamespace(entity_type=entity_type, start=start, end=end, score=score)


def test_detect_maps_presidio_results_to_pii_matches():
    text = "contact jane.doe@example.com now"
    start = text.index("jane.doe@example.com")
    end = start + len("jane.doe@example.com")
    analyzer = FakeAnalyzer([_result("EMAIL_ADDRESS", start, end)])

    hits = PresidioDetector(analyzer=analyzer).detect(text)

    assert len(hits) == 1
    assert hits[0].entity_type == "EMAIL_ADDRESS"
    assert hits[0].value == "jane.doe@example.com"


def test_detect_empty_text_returns_empty_without_calling_analyzer():
    analyzer = FakeAnalyzer([])
    assert PresidioDetector(analyzer=analyzer).detect("") == []
    assert analyzer.analyze_calls == []


def test_redact_replaces_matches_and_hides_raw_value():
    text = "contact jane.doe@example.com now"
    start = text.index("jane.doe@example.com")
    end = start + len("jane.doe@example.com")
    analyzer = FakeAnalyzer([_result("EMAIL_ADDRESS", start, end)])

    redacted = PresidioDetector(analyzer=analyzer).redact(text)

    assert "jane.doe@example.com" not in redacted
    assert "[REDACTED_EMAIL_ADDRESS]" in redacted


def test_language_and_score_threshold_passed_through():
    analyzer = FakeAnalyzer([])
    PresidioDetector(analyzer=analyzer, language="fr", score_threshold=0.7).detect("bonjour")
    assert analyzer.analyze_calls[0]["language"] == "fr"
    assert analyzer.analyze_calls[0]["score_threshold"] == 0.7


def test_raises_helpful_error_when_presidio_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "presidio_analyzer", None)
    with pytest.raises(ImportError, match=r"ai-flight-recorder\[presidio\]"):
        PresidioDetector()
