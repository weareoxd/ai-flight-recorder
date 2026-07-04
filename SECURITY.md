# Security Policy

## Supported Versions

ai-flight-recorder is pre-1.0. Security fixes are made against the latest release only.

## Reporting a Vulnerability

Please do not open a public GitHub issue for security vulnerabilities.

Instead, use GitHub's private reporting for this repository:
[Report a vulnerability](https://github.com/weareoxd/ai-flight-recorder/security/advisories/new).
This opens a draft security advisory visible only to the maintainers.

Include, as applicable:

- The affected version(s) or commit.
- A description of the vulnerability and its impact.
- Steps to reproduce, or a minimal proof of concept.

We'll acknowledge new reports and follow up with next steps as we investigate.

## Scope notes specific to this project

ai-flight-recorder handles PII in two places worth flagging explicitly when reporting:

- **PII detection** (`src/ai_flight_recorder/detectors/`): a false negative in
  `RegexDetector` or `PresidioDetector` means unredacted PII reaches Langfuse via
  `record.redacted_prompt`. Detector recall gaps are a legitimate security concern, not
  just a correctness bug.
- **`subject_ref` validation** (`schema.py`): this field is guarded against values that
  look like raw PII (email/phone/SSN/etc.) via `RegexDetector`. A bypass of that guard is
  also in scope.

Issues in `RegexDetector`'s pattern coverage that you believe are exploitable to leak PII
should go through this same private reporting process rather than a public issue.
