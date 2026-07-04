"""Optional, higher-recall PII detector backed by Microsoft Presidio.

Not imported by anything else in ai-flight-recorder — installing the base package
never pulls in presidio-analyzer or its NLP model dependencies. Install with:

    pip install "ai-flight-recorder[presidio]"

Presidio also needs an NLP model available at runtime (spaCy by default), e.g.:

    python -m spacy download en_core_web_lg

That's a Presidio/spaCy setup step, not something ai-flight-recorder manages.
"""

from __future__ import annotations

from typing import Any

from ai_flight_recorder.detectors import Detector, PIIMatch, redact_matches

_INSTALL_HINT = (
    'PresidioDetector requires the "presidio" extra: pip install "ai-flight-recorder[presidio]". '
    "It also needs a spaCy model, e.g.: python -m spacy download en_core_web_lg"
)


class PresidioDetector(Detector):
    """Implements the same `Detector` interface as `RegexDetector`, backed by
    Presidio's `AnalyzerEngine`."""

    def __init__(
        self,
        analyzer: Any | None = None,
        language: str = "en",
        score_threshold: float = 0.5,
    ) -> None:
        if analyzer is None:
            try:
                from presidio_analyzer import AnalyzerEngine
            except ImportError as exc:
                raise ImportError(_INSTALL_HINT) from exc
            analyzer = AnalyzerEngine()
        self._analyzer = analyzer
        self._language = language
        self._score_threshold = score_threshold

    def detect(self, text: str) -> list[PIIMatch]:
        if not text:
            return []
        results = self._analyzer.analyze(
            text=text, language=self._language, score_threshold=self._score_threshold
        )
        return [
            PIIMatch(entity_type=r.entity_type, start=r.start, end=r.end, value=text[r.start : r.end])
            for r in results
        ]

    def redact(self, text: str) -> str:
        return redact_matches(text, self.detect(text))
