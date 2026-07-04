# Contributing

## Setup

```bash
git clone git@github.com:weareoxd/ai-flight-recorder.git
cd ai-flight-recorder
pip install -e ".[dev]"   # or: uv sync --extra dev
```

## Running tests

```bash
pytest                                          # full suite
pytest tests/test_rules.py -k pii_in_inputs     # single test/file
uv run python examples/quickstart.py            # end-to-end demo, no API keys required
```

All tests are fully mocked (`tests/conftest.py`'s `FakeLangfuseClient`, and LiteLLM's
`mock_response`) — nothing hits the network, and no API keys are needed to run the suite.
There is no configured lint/format command.

## Before opening a PR

- Add or update tests for any behavior change. `tests/conftest.py`'s `FakeLangfuseClient`
  records every call it receives, so most emit-path changes can be asserted against exact
  A/B/C/D payloads (see the "Testing conventions" section of [CLAUDE.md](CLAUDE.md)).
- Keep `pytest` green.
- If you touch the `DecisionRecord` schema (`src/ai_flight_recorder/schema.py`), check
  whether `compute_integrity_hash()` (`hashing.py`) and the Langfuse field mapping
  (`emit.py`) need corresponding updates — the hash covers every field except itself, and
  a new field that isn't mapped in `emit.py` will silently never reach Langfuse.
- If you add a rule to `rules.py`, note whether it has side effects on the record (like
  `pii_in_inputs` populating `redacted_prompt`) — rules should otherwise only route
  attention (`review_required`, `Flag`s), never render a quality verdict.

## Architecture

See [CLAUDE.md](CLAUDE.md) for the pipeline overview (capture → rules → hashing → emit →
scoring) and the design rationale behind each stage, and [docs/architecture.md](docs/architecture.md)
for the full `DecisionRecord` schema and its Langfuse field mapping.

## Reporting bugs and requesting features

Open a [GitHub issue](https://github.com/weareoxd/ai-flight-recorder/issues). For
security vulnerabilities, see [SECURITY.md](SECURITY.md) instead of filing a public issue.

## License

By contributing, you agree that your contributions will be licensed under the project's
[MIT License](LICENSE).
