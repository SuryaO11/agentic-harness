# Pitfalls Research

**Domain:** LLM Output Validation & Self-Correction Harness  
**Researched:** 2026-11-13  
**Confidence:** HIGH  

## Critical Pitfalls & Mitigation Strategies

### 1. Free-Text Critic Hallucinations & False Positives
- **Pitfall**: Research shows unconstrained LLM evaluators evaluating LLM outputs have high false positive rates (76%–98% in some unstructured benchmark studies) and hallucinate critique rules.
- **Mitigation**: Constrain the critic to structured diagnostic outputs with strict enums (`failure_type`), severity levels, and explicit references to failed `ValidatorResult` records. Never allow the critic to invent new requirements outside the `TaskSpec`.

### 2. Repair-Induced Cascade Failure
- **Pitfall**: When correcting one specific validator failure (e.g. fixing a missing regex pattern or formatting issue), the model often breaks a previously passing field (e.g. changing field names or dropping attributes).
- **Mitigation**: Post-repair stability check. After every retry generation, all validators (including previously passed ones) must re-execute from scratch. A retry is only considered passing if the entire suite passes simultaneously.

### 3. Silent Transport Retry Multiplication
- **Pitfall**: SDKs (such as Groq SDK or httpx) often have built-in retry mechanisms enabled by default (e.g. 2–3 transport retries per call). If the harness performs 3 algorithmic retries, the system may actually make 9–12 expensive HTTP calls.
- **Mitigation**: Explicitly instantiate client sessions with `max_retries=0` to ensure that every network call is tracked and accounted for under the harness's measured 0.30 retries/query budget.

### 4. Conflating Schema Validation with Semantic Correctness
- **Pitfall**: An output can perfectly match a Pydantic JSON schema while being complete factual nonsense or completely answering a different question.
- **Mitigation**: Maintain distinct, decoupled layers: deterministic validators handle structure and syntactic integrity; Sentence Transformers handles semantic alignment; the critic evaluates task-level domain compliance.

### 5. Stale Cache Serving Outdated Validation Rules
- **Pitfall**: If a developer changes a regex pattern or adds a new required field to a `TaskSpec`, naive prompt-only cache lookups return previously cached responses that violate the new rules.
- **Mitigation**: Include the canonical hash of the active validator set in the SHA256 cache key. Any change to validator definitions automatically yields a cache miss.

