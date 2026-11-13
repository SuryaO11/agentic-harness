---
phase: 01-shared-core-contracts-pluggable-llm-client
plan: 01
subsystem: contracts
status: completed
files_modified:
  - src/harness/__init__.py
  - src/harness/contracts.py
  - src/harness/exceptions.py
---

# Plan 01-01 Summary: Shared Core Contracts & Type Definitions

## Accomplishments
1. Established the immutable core contracts in `src/harness/contracts.py`:
   - `FailureTypeEnum`: Standardized diagnostic failure categories (`schema_syntax`, `regex_mismatch`, `semantic_drift`, etc.).
   - `SeverityEnum`: Uniform severity indicators (`critical`, `warning`, `info`).
   - `ValidatorResult`: Frozen Pydantic contract shared across declarative and custom validators.
   - `DiagnosticItem` and `CriticFeedback`: Structured critic outputs replacing unconstrained prose.
   - `LLMResponse`: Unified container for token counts, latencies, provider metadata, and output content.
   - `TaskSpec`: Declarative task and validator specification schema.
2. Built the exception hierarchy in `src/harness/exceptions.py`:
   - `HarnessError` as base exception.
   - `ProviderError` for API failures.
   - `RateLimitError` with dedicated parsing and storage of the HTTP 429 `retry-after` header in seconds.
3. Exported all public types cleanly via `src/harness/__init__.py`.

## Verification Evidence
- Instantiated `ValidatorResult` with `SeverityEnum` and `FailureTypeEnum` - passed.
- Instantiated `RateLimitError` and verified `retry_after == 12.5` - passed.
- Zero type coercion or validation errors encountered.
