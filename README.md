# ai-flight-recorder

A thin governance layer on top of [Langfuse](https://langfuse.com) for AI-influenced
decisions. It wraps an LLM call made via [LiteLLM](https://docs.litellm.ai/), captures
a governance-shaped decision record, and writes it into Langfuse using Langfuse's own
SDK: trace metadata, observation input/output, and scores. Human review happens in
Langfuse's existing annotation UI. There is no separate storage layer and no separate
viewer.

## Why

Langfuse traces show that a call happened, with what input and output. They don't mark
a specific trace as a **decision** about a person (a loan, a moderation call, a
recommendation), with who it concerned, what evidence backed it, and whether a human
needs to sign off before it takes effect.

ai-flight-recorder adds a decision-record schema and the code that writes it into Langfuse
the same way every time, so decisions end up in a consistent shape instead of whatever
ad hoc metadata each call site happened to log.

**ai-flight-recorder records and surfaces decisions; it does not certify that any of them
were correct.** Confidence values are recorded for traceability, not as reliability
guarantees, and rule flags route human attention rather than render a quality verdict.

## Documentation

- **[User guide](docs/user-guide.md)**: install, configuration, how a decision flows
  from your code into Langfuse and a review queue, and how the rule engine decides
  what needs human review.
- **[Architecture](docs/architecture.md)**: how the pipeline is built internally, the
  full `DecisionRecord` schema and its Langfuse field mapping, and known limitations.

## Quickstart

With [uv](https://docs.astral.sh/uv/), no separate install step is needed. `uv run`
creates the environment and runs the command in one step. This runs three scenarios
end-to-end, no API keys required (LiteLLM and Langfuse both fall back to mocks unless
you set `OPENAI_API_KEY` / `LANGFUSE_PUBLIC_KEY`):

```bash
git clone git@github.com:weareoxd/ai-flight-recorder.git
cd ai-flight-recorder
uv run python examples/quickstart.py
```

```python
import ai_flight_recorder

with ai_flight_recorder.decision(
    decision_type="loan_eligibility",
    subject_ref="cust_1001",       # opaque id, not raw PII
    options=["approve", "deny"],
) as d:
    resp = d.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "..."}])
    d.outcome = "approve"

print(d.emit_result.trace_id)
```

See the [user guide](docs/user-guide.md) for the decorator-based integration style,
connecting to a real Langfuse project, and configuring thresholds, PII detection, and
the review queue.

## Development

```bash
pip install -e ".[dev]"
pytest
```

All tests are fully mocked; no test hits litellm or Langfuse over the network.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor workflow, and
[SECURITY.md](SECURITY.md) to report a vulnerability. This project follows the
[Code of Conduct](CODE_OF_CONDUCT.md).
