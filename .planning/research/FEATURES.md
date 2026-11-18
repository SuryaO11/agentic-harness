# Features Research

**Domain:** LLM Output Validation & Self-Correction Harness  
**Researched:** 2026-11-13  
**Confidence:** HIGH  

## Feature Landscape

### Table Stakes (Expected Baseline)
- **Deterministic Schema Checks**: JSON schema, Pydantic type validation, regex pattern matching, required/prohibited keys.
- **Uniform Validator Result**: Consistent `ValidatorResult(passed, reason, severity)` interface across all checks.
- **Retry Mechanism**: Ability to re-query the model upon validation failure.
- **Audit Logging**: Recording validation status and output history.

### Differentiating Features (Agentic Harness Advantages)
- **Closed-Loop Critic Steering**: Structured diagnostic critique (failure type enum + severity + specific correction guidance) rather than unconstrained free text.
- **Multi-Turn History Injection**: Appending precise failure reasons and prior output into conversation history, keeping retry prompt tokens minimal (0.30 retries/query).
- **Secondary Prompt Escalation**: Automated system prompt negative constraint injection after 2 consecutive failures.
- **Independent Semantic Similarity Scoring**: Decoupled sentence embedding similarity check to ensure schema-valid outputs do not drift from the prompt's semantic intent.
- **Selective High-Reliability SQLite Cache**: Caches only responses meeting Reliability Score ≥ 0.80, with cache keys hashing prompt + model + active validator set.
- **Interactive Streamlit Observability**: Visual trace breakdown showing step-by-step retry progression, validator pass/fail badges, and latency benchmarks.

## Prior Art Analysis
- **Guardrails AI**: Focuses heavily on schema validation and RAIL specifications; lacks integrated multi-turn critic steering, post-repair stability checks, and selective high-reliability response caching.
- **llm-output-guard**: Good Pydantic wrapper with retry feedback, but lacks independent semantic embedding scoring and real-time dashboard observability.

