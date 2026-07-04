# User guide

How to install ai-flight-recorder, configure it for your project, and understand what
happens to a decision between your code calling it and a human seeing it in Langfuse's
review queue.

For how the pieces are built internally, see [Architecture](architecture.md).

## Install

Not published to PyPI. Install from a local checkout. With
[uv](https://docs.astral.sh/uv/), there's no venv or activate step: `uv run` creates
`.venv`, installs the project and its dependencies, and runs the given command inside
that environment.

```bash
git clone <this-repo> ai-flight-recorder
cd ai-flight-recorder
uv run python examples/quickstart.py
# optional, higher-recall PII detection:
uv run --extra presidio python examples/quickstart.py
```

Re-run `uv run` any time you change code or dependencies; it re-syncs the environment
first.

Without uv, create the virtual environment and install with `pip` instead:

```bash
git clone <this-repo> ai-flight-recorder
cd ai-flight-recorder
python -m venv .venv
source .venv/bin/activate
pip install -e .
# optional, higher-recall PII detection:
pip install -e ".[presidio]"
```

Requires Python 3.11+. Core dependencies: `litellm`, `langfuse` (pinned to the v3 SDK,
`>=3.3.0,<4.0.0`), `pydantic`.

## Configuration

### Environment

Configure the Langfuse SDK the way you normally would, typically via environment
variables:

```bash
export LANGFUSE_PUBLIC_KEY=pk-...
export LANGFUSE_SECRET_KEY=sk-...
export LANGFUSE_HOST=https://cloud.langfuse.com   # or your self-hosted URL
```

Or copy [`.env.example`](../.env.example) to `.env` and fill in your values:

```bash
cp .env.example .env
```

`litellm` loads `.env` files automatically on import (via `python-dotenv`), and
ai-flight-recorder imports `litellm` directly, so no extra setup is needed. `.env.example` also
has a spot for `OPENAI_API_KEY`; it's optional and only used by
`examples/quickstart.py` to call the real OpenAI API instead of a mock response.

This auto-load only happens when `LITELLM_MODE` is unset or `DEV` (the default). If you
set `LITELLM_MODE=PRODUCTION`, call `dotenv.load_dotenv()` yourself before importing
`ai_flight_recorder`.

If you're self-hosting Langfuse, run platform version 3.125.0 or later: the first
version with full support for the v3 SDK features ai-flight-recorder relies on
(`start_as_current_observation`, `propagate_attributes`, idempotent score ids via
`score_id`). Pin a recent `langfuse` SDK release too: older `create_score` releases had
bugs around idempotent score updates, and older annotation-queue endpoints have a
response-validation bug for `SESSION`-type items. ai-flight-recorder only ever enqueues
`TRACE` items, so this doesn't affect it directly, but pin a recent SDK anyway.

### Set up ScoreConfigs

Required once per Langfuse project, before recording any decisions. ai-flight-recorder
doesn't create these on its own, and without them Langfuse has no `review_required` or
`human_review_verdict` fields to score against. With `LANGFUSE_PUBLIC_KEY` /
`LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` set (env vars or `.env`, per above):

```bash
uv run ai-flight-recorder-setup-scores
```

This is idempotent, safe to run again without duplicating configs that already exist,
and creates:

- `review_required`: `BOOLEAN`. Written automatically by ai-flight-recorder's rule engine.
- `human_review_verdict`: `CATEGORICAL`, categories `approved`, `overridden`, or
  `escalated`. **ai-flight-recorder does not write this.** Reviewers record it themselves,
  directly in Langfuse's annotation UI, against the trace ai-flight-recorder created.

To run it from Python instead, for example as part of a setup script:

```python
from ai_flight_recorder import setup_score_configs

setup_score_configs()  # or setup_score_configs(my_client)
```

### Set up the review queue (optional)

If you want flagged decisions to land in a Langfuse annotation queue automatically
(see [How decisions reach a reviewer](#how-decisions-reach-a-reviewer) below), create
that queue once per project:

```python
from ai_flight_recorder import ensure_review_queue

ensure_review_queue(queue_name="ai-flight-recorder-review")  # or ensure_review_queue(my_client, queue_name=...)
```

This also runs `setup_score_configs()` and attaches both `review_required` and
`human_review_verdict` to the queue. It's idempotent: an existing queue (matched by
name) is returned as-is, not duplicated.

### GovernanceConfig

`GovernanceConfig` (from `ai_flight_recorder`) holds every threshold and toggle. Pass it to
`decision()` or `@audit_decision(...)` via `config=`; the default is a plain
`GovernanceConfig()` with no high-stakes types or sensitive outcomes configured.

```python
from ai_flight_recorder import GovernanceConfig, RegexDetector, CustomPattern
import re

config = GovernanceConfig(
    low_confidence_threshold=0.6,
    high_stakes_types={"loan_eligibility", "content_moderation"},
    sensitive_outcomes={"deny", "ban"},
    enabled_rules={"low_confidence", "high_stakes_type", "sensitive_outcome", "pii_in_inputs"},
    pii_detector=RegexDetector(custom_patterns=[
        CustomPattern("employee_id", re.compile(r"\bEMP-\d{6}\b")),
    ]),
    redact_pii_in_input=True,
    review_queue_name="ai-flight-recorder-review",
    auto_enqueue=True,
    auto_flush=True,
)
```

| Field | Default | What it controls |
|---|---|---|
| `low_confidence_threshold` | `0.5` | Below this, the `low_confidence` rule fires (only for confidence signals with `source != "none"`). |
| `high_stakes_types` | empty set | `decision_type` values that trigger review. |
| `sensitive_outcomes` | empty set | `outcome` values that trigger review. |
| `enabled_rules` | all four built-in rules | Which rules run. Drop a name to disable it. |
| `pii_detector` | `RegexDetector()` | The `Detector` the `pii_in_inputs` rule uses. Swap in `PresidioDetector` or a `RegexDetector` with `custom_patterns` for domain-specific entities. |
| `redact_pii_in_input` | `True` | When PII is found, send the redacted prompt to Langfuse instead of the raw one. |
| `review_queue_name` | `None` | Annotation queue flagged decisions are enqueued into. `None` disables enqueueing entirely. |
| `auto_enqueue` | `True` | Set to `False` to keep `review_queue_name` configured but pause enqueueing without clearing the name. |
| `auto_flush` | `True` | Set to `False` in a long-running service to skip the per-decision `client.flush()` round trip and let the Langfuse SDK batch on its own schedule. Not safe for short-lived scripts/serverless (see [Flushing](#flushing) below). Ignored (flush still happens) when a decision is about to be enqueued into a review queue. |

### Running Langfuse locally for end-to-end testing

`examples/quickstart.py` works with no Langfuse instance at all (it falls back to an
in-memory mock client), but to see real traces land in the Langfuse UI, run Langfuse
locally with Docker Compose:

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up
```

Give it two to three minutes for `langfuse-web-1` to log `Ready`, then open
`http://localhost:3000`. Sign up for a local account, create an organization and a
project, and generate an API key pair from the project's settings page. Export those
into your shell, or put them in a `.env` file (see [Environment](#environment) above):

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=http://localhost:3000
```

Then, back in ai-flight-recorder, run the quickstart:

```bash
uv run python examples/quickstart.py
python examples/quickstart.py  # or, inside an activated venv
```

With those env vars set, the script uses the real Langfuse client instead of the mock,
prints an openable trace URL for each scenario, and registers the `review_required` and
`human_review_verdict` ScoreConfigs. Open a trace URL to see the decision record and,
on the `loan_eligibility` scenario, the PII redaction that ran before the record was
sent.

## Basic usage

Both integration styles produce the same kind of record.

### Context manager

```python
import ai_flight_recorder

with ai_flight_recorder.decision(
    decision_type="loan_eligibility",
    subject_ref="cust_1001",       # opaque id, not raw PII (see below)
    options=["approve", "deny"],
) as d:
    resp = d.completion(              # wraps litellm.completion
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Should we approve this loan?"}],
    )
    d.outcome = "approve"
# on exit: rule engine runs, integrity hash is computed, the record is
# emitted to Langfuse, and the client is flushed.

print(d.emit_result.trace_id)
```

There's an `async` counterpart, `d.acompletion(...)`, wrapping `litellm.acompletion`.

### Decorator

```python
from ai_flight_recorder import audit_decision

@audit_decision(
    decision_type="loan_eligibility",
    subject_ref=lambda applicant_id: applicant_id,   # static value or a callable of the call args
    options=["approve", "deny"],
)
def evaluate(d, applicant_id: str) -> str:
    d.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "..."}])
    return "approve"   # used as the outcome, unless d.outcome is set explicitly

evaluate("cust_1001")
```

The wrapped function's first parameter receives the `Decision` handle. Works with
`async def` functions too.

### Flushing

`flush()` is called automatically when a `with ai_flight_recorder.decision(...)` block or an
`@audit_decision`-wrapped call exits normally, as long as `GovernanceConfig.auto_flush`
is `True` (the default). If you call `ai_flight_recorder.emit(...)` directly (bypassing both
integration styles), you own calling `client.flush()` yourself. In short-lived scripts
or serverless functions, buffered events are otherwise lost, since the process may exit
before Langfuse's own background thread would have sent them.

In a long-running service, that per-decision flush is a synchronous network round trip
on every decision. Set `auto_flush=False` to skip it and let the Langfuse SDK batch
flushes on its own schedule instead (and flush yourself, e.g. on shutdown). ai-flight-recorder
still flushes immediately regardless of `auto_flush` when a decision is about to be
enqueued into a review queue, since that step needs the trace to already exist in
Langfuse.

If the `with` block raises, the decision is **not** recorded. An incomplete capture
isn't a decision worth emitting, and the exception propagates unchanged.

## Data flow

What happens, in order, between your code opening a decision and a reviewer seeing it:

1. **You open the decision.** `ai_flight_recorder.decision(...)` (or an
   `@audit_decision`-wrapped call) creates a `DecisionRecord` in memory. Nothing has
   touched the network yet.
2. **You call the model.** `d.completion(...)` calls `litellm.completion` and records
   `model`, `model_version`, `latency_ms`, and a confidence signal onto the record.
   This only talks to the LLM provider, so there's still no Langfuse traffic.
3. **You set the outcome.** Either `d.outcome = ...` directly, or (with the decorator)
   your function's return value.
4. **The `with` block exits normally.** This is where everything below happens, in
   order, inside `_DecisionContext.__exit__`:
   1. `RuleEngine.run()` walks the enabled rules against the finished record. Any rule
      that fires appends a `Flag` and may set `review_required = True`. If
      `pii_in_inputs` fires, `record.redacted_prompt` is populated here.
   2. `compute_integrity_hash()` runs over the now-finished record (redaction
      included) and is stored as `record.integrity_hash`.
   3. `emit()` writes the record to Langfuse: a trace, an observation with
      input/output, propagated metadata, regular metadata, and a `review_required`
      score. This is the first point anything reaches Langfuse.
   4. `client.flush()` sends the buffered events over the network.
   5. `enqueue_if_flagged()` runs last, and only after the flush completed: a queue
      item points at a trace by id, so the trace has to exist before anything can
      point at it. See [How decisions reach a
      reviewer](#how-decisions-reach-a-reviewer) below.
5. **A reviewer works the queue** (if one is configured and the decision was flagged),
   or browses the trace directly in Langfuse if not. They record `human_review_verdict`
   through Langfuse's annotation UI, a step entirely outside ai-flight-recorder.

If the `with` block raises at any point before step 4, none of the above runs, and the
decision is dropped rather than partially recorded.

## The decision-making process

"Decision-making" here means how ai-flight-recorder decides whether *a decision needs human
review*. It does not judge whether the decision itself was right.

### What causes a flag

`RuleEngine` runs the rules named in `enabled_rules` (all four below are on by default)
against the finished record, and sets `review_required=True` the first time one fires:

| Rule | Fires when | Config it reads |
|---|---|---|
| `low_confidence` | A confidence signal (`source` other than `"none"`) is below the threshold | `low_confidence_threshold` |
| `high_stakes_type` | `decision_type` is in the configured set | `high_stakes_types` |
| `sensitive_outcome` | `outcome` is in the configured set | `sensitive_outcomes` |
| `pii_in_inputs` | The prompt contains PII | `pii_detector`, `redact_pii_in_input` |

Drop a rule's name from `enabled_rules` to turn it off. Every rule that fires also
appends a `Flag(rule_name, reason)` to the record, visible in Langfuse's metadata.

Flagging and enqueueing are two different steps: a flagged decision gets
`review_required=True` and shows up as that score on its trace in Langfuse regardless
of queue configuration, while `review_queue_name` and `auto_enqueue` only decide what
happens next.

### How decisions reach a reviewer

Set `review_queue_name` on `GovernanceConfig` and ai-flight-recorder handles this part too:
whenever a decision comes out of `RuleEngine` with `review_required=True`, its trace is
added to that Langfuse annotation queue automatically. Reviewers work the queue from
Langfuse's **Human Annotation** section instead of hunting through the trace table, and
record `human_review_verdict` there.

```python
config = GovernanceConfig(
    high_stakes_types={"loan_eligibility"},
    review_queue_name="ai-flight-recorder-review",  # None (the default) disables enqueueing
)
```

Create the queue once per project first, with `ensure_review_queue()` (see [Set up the
review queue](#set-up-the-review-queue-optional) above). A decision can only be
enqueued into a queue that already exists.

A few behaviors worth knowing about:

- **Ordering.** Enqueueing happens strictly after the trace is flushed, since a queue
  item points at a trace by id and Langfuse has nothing to attach it to otherwise. If
  the enqueue call fails for any reason (the named queue doesn't exist, the API call
  errors), the failure is logged and swallowed rather than raised. The trace was
  already written before the enqueue was attempted, so decision capture is unaffected.
- **Only flagged decisions are enqueued.** A clean decision (`review_required=False`)
  does not touch the queue, no matter how `review_queue_name` is set.
- **A flag doesn't guarantee a queue item.** The reverse is true too: if
  `review_queue_name` is `None` or `auto_enqueue` is `False`, a flagged decision still
  gets `review_required=True` as a score on its trace, but nothing puts it in front of
  a reviewer.
- **`auto_enqueue=False`** keeps `review_queue_name` configured but pauses enqueueing,
  without clearing the name.
- **Idempotent within a process.** Enqueueing the same decision twice in the same
  process is a no-op. Duplicate protection across process restarts depends on the
  annotation queue's own dedup behavior, not on ai-flight-recorder.

### PII detection

`pii_in_inputs` runs a `Detector` over the prompt text on every decision (unless
disabled via `enabled_rules`). Two implementations ship:

- `RegexDetector` (default, zero extra dependencies): email, phone, credit card
  (Luhn-checked), SSN/SIN, IPv4, plus any `custom_patterns` you supply.
- `PresidioDetector` (`pip install ai-flight-recorder[presidio]`): higher recall, same
  `Detector` interface. `presidio` is only imported if you opt in.

When PII is found, the rule engine populates `redacted_prompt` (each match replaced
with `[REDACTED_<ENTITY_TYPE>]`) and, if `GovernanceConfig.redact_pii_in_input` is
`True` (the default), that redacted version, not the raw prompt, is what's sent to
Langfuse as the observation input.

`subject_ref` is checked separately, at record-creation time: a validator rejects
values that look like an email, phone number, SSN, credit card, or IP address, since
`subject_ref` is meant to be an opaque id, not raw PII. This is a guardrail against
the common mistake, not a guarantee, so pick a real opaque identifier.

## Where to go next

- [Architecture](architecture.md) covers how the pipeline is implemented internally,
  the full `DecisionRecord` schema and its Langfuse field mapping, the integrity hash,
  and known limitations.
