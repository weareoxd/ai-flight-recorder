# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e ".[dev]"   # or: uv sync --extra dev
pytest                     # run the full suite
pytest tests/test_rules.py -k pii_in_inputs   # single test/file
uv run python examples/quickstart.py          # end-to-end demo, no API keys required
```

All tests are fully mocked (`tests/conftest.py`'s `FakeLangfuseClient`, and LiteLLM's
`mock_response`) — nothing hits the network. There is no configured lint/format command.

## Architecture

ai-flight-recorder is a thin governance layer over Langfuse: it builds a `DecisionRecord` for
one AI-influenced decision and maps it onto Langfuse's own data model (trace metadata,
observation input/output, scores). It does not implement its own storage, viewer, or
review UI — Langfuse's annotation UI is where humans review flagged decisions.

Pipeline, in call order, all driven from `capture.py`:

1. **Capture** (`capture.py`) — `decision()` (context manager) and `audit_decision()`
   (decorator) both build a `_DecisionContext` wrapping a `Decision` handle around a
   `DecisionRecord`. `Decision.completion()`/`acompletion()` wrap `litellm.completion`/
   `acompletion`, auto-populating `model`, `model_version`, `latency_ms`, and a
   `ConfidenceSignal` (from response logprobs, or an opt-in extra self-report call, or
   `none`).
2. **Rules** (`rules.py`) — on normal exit (not on exception — an incomplete capture is
   never emitted), `RuleEngine.run()` walks the enabled `BUILTIN_RULES` predicates
   against the finished record, appending `Flag`s and setting `review_required`. Rules
   only route attention; they never render a quality verdict. `pii_in_inputs` is the one
   rule with a side effect: it populates `record.redacted_prompt` via the configured
   `Detector` (`RegexDetector` by default, or `PresidioDetector` behind the `presidio`
   extra — see `detectors/`).
3. **Hashing** (`hashing.py`) — `compute_integrity_hash()` runs *after* the rule engine
   (so it covers any PII redaction) and is a sha256 over every `DecisionRecord` field
   except itself. It never covers the human review verdict, since that verdict is never
   part of the schema — it's written later, directly in Langfuse, as a score.
4. **Emit** (`emit.py`) — maps the record onto Langfuse across four buckets: (A)
   propagated metadata — short scalars, hard-capped at 200 chars by Langfuse itself, with
   `_guard_scalar` truncating (with a visible marker + warning) rather than letting
   Langfuse silently drop oversized values; (B) observation input/output; (C) scores —
   writes `review_required` with a deterministic `score_id` so re-emitting the same
   decision updates rather than duplicates; (D) regular (uncapped) observation metadata.
   All fields land on one trace because the trace id is deterministically derived from
   `decision_id` via `client.create_trace_id(seed=...)`. `emit()` does not flush; the
   context manager/decorator call `client.flush()` on normal exit.
5. **Scoring** (`scoring.py`) — `setup_score_configs()` (console script
   `ai-flight-recorder-setup-scores`) idempotently creates the `review_required` and
   `human_review_verdict` Langfuse ScoreConfigs a project needs before recording
   decisions. ai-flight-recorder only ever writes `review_required`; `human_review_verdict` is
   set by a human reviewer directly in Langfuse's annotation UI.

`schema.py` (`DecisionRecord`) is the spec everything else maps onto; it has no Langfuse
awareness. Its `subject_ref` field is guarded (via `RegexDetector`, imported directly to
avoid a dependency on `GovernanceConfig`) to reject values that look like raw PII —
callers must pass an opaque id, never an email/phone/SSN/etc.

`config.py`'s `GovernanceConfig` is deliberately a plain dataclass, not pydantic, because
it holds a live `Detector` instance (a runtime dependency), not serializable data.

### Detectors

`detectors/__init__.py` defines the `Detector` protocol (`detect(text) -> list[PIIMatch]`)
and the shared `redact_matches()` used by every implementation. `RegexDetector` (default)
covers email/phone/credit-card(Luhn-checked)/SSN/SIN/IPv4 plus custom patterns, with zero
extra dependencies. `PresidioDetector` (`detectors/presidio.py`) implements the same
protocol at higher recall but is never imported unless the `presidio` extra is installed
and the caller opts in — installing the base package never pulls in Presidio or its NLP
models.

### Testing conventions

`tests/conftest.py`'s `FakeLangfuseClient` implements the same subset of the Langfuse v3
SDK interface as `emit.py` uses, recording every call so tests can assert on exact A/B/C/D
payloads. Note `ai_flight_recorder/__init__.py` re-exports `emit` (the function), which shadows
the `emit` submodule attribute — tests reach the module itself via
`importlib.import_module("ai_flight_recorder.emit")`, not `import ai_flight_recorder.emit`.
