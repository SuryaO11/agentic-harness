# Requirements: Agentic Harness v1.2

**Defined:** 2026-11-13  
**Core Value:** Turn stochastic, hallucination-prone LLM outputs into deterministic, verifiable, and explainable responses through targeted, self-correcting feedback loops and selective high-reliability caching.  

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Core Contracts & LLM Client (CLIENT)
- [ ] **CLIENT-01**: User can query Llama 3.1 8B Instant via a unified `LLMClient` interface with Groq as default provider and Ollama as local fallback.
- [ ] **CLIENT-02**: Developer can switch inference providers via environment variable (`LLM_PROVIDER=groq|ollama`) without altering application code.
- [ ] **CLIENT-03**: System disables SDK transport retries (`max_retries=0`) to ensure all API calls are strictly attributed to and observable within the harness.
- [ ] **CLIENT-04**: System preserves fixed temperature (0.0) across all calls and retry cycles to maintain reproducible reliability scoring.

### Dual Validator System (VALID)
- [ ] **VALID-01**: Developer can register declarative validators using Pydantic models, JSON schema, regex patterns, and keyword constraints.
- [ ] **VALID-02**: Developer can define custom deterministic Python functions returning `ValidatorResult(passed, reason, severity, failure_type)`.
- [ ] **VALID-03**: System executes declarative validators first (cheap fail-fast) before executing custom Python callables.
- [ ] **VALID-04**: System prohibits network and LLM calls inside custom validator functions to enforce pure determinism.

### Semantic Similarity Scoring (SCORE)
- [ ] **SCORE-01**: System computes semantic similarity using local Sentence Transformers (`all-MiniLM-L6-v2`) embeddings.
- [ ] **SCORE-02**: System evaluates semantic alignment between generated response and query intent/ground truth, producing a normalized score [0.0, 1.0].
- [ ] **SCORE-03**: System keeps semantic similarity scoring decoupled from schema validation so syntax validity does not mask semantic drift.

### Critic Evaluator (CRITIC)
- [ ] **CRITIC-01**: System executes critic evaluation producing structured diagnostics (`failure_type` enum, severity, explicit references to failed validators).
- [ ] **CRITIC-02**: System generates concise, actionable natural-language guidance explaining exactly what must be corrected.
- [ ] **CRITIC-03**: System prevents unconstrained critic hallucinations by restricting critic outputs to validated schema boundaries.

### Targeted Retry Orchestrator (RETRY)
- [ ] **RETRY-01**: System orchestrates targeted retries by appending specific validator failure reasons, failed text, and critic guidance into conversation history.
- [ ] **RETRY-02**: System executes post-repair stability check by re-running the entire validator suite on retried outputs to detect cascade regressions.
- [ ] **RETRY-03**: System automatically escalates to prompt refinement with negative constraints if validation fails across 2 consecutive retries.
- [ ] **RETRY-04**: System enforces a hard ceiling of max 3 retries, terminating with an explicit failure summary if unrecoverable.

### Selective SQLite Cache (CACHE)
- [ ] **CACHE-01**: System evaluates response composite reliability score and caches only responses meeting `Reliability Score >= 0.80`.
- [ ] **CACHE-02**: System generates deterministic SHA256 cache keys combining provider, model, prompt text, and active validator set hash.
- [ ] **CACHE-03**: System returns cached responses in microsecond latency upon repeat queries without triggering upstream LLM calls.
- [ ] **CACHE-04**: System invalidates cache lookup automatically whenever task validator definitions or schemas change.

### Streamlit Observability Dashboard (UI)
- [ ] **UI-01**: User can interactively input queries, select task specs, and view real-time generation and retry progression.
- [ ] **UI-02**: User can inspect step-by-step validator pass/fail badges, severity ratings, and error reasons for every retry attempt.
- [ ] **UI-03**: User can view the critic's structured diagnostic feedback and corrective guidance in an explainable audit card.
- [ ] **UI-04**: User can monitor cache hit/miss status, execution latency, token utilization, and final composite reliability metrics.

### Benchmark & Evaluation Suite (BENCH)
- [ ] **BENCH-01**: Developer can execute the 54-sample benchmark suite comparing baseline zero-shot vs harnessed execution.
- [ ] **BENCH-02**: System verifies baseline to harnessed improvement (1.5% zero-shot success -> >= 95% harnessed success).
- [ ] **BENCH-03**: System verifies reliability score improvement (0.816 baseline -> >= 0.942 harnessed).
- [ ] **BENCH-04**: System verifies autonomous recovery rate >= 81.8% at <= 0.30 retries/query.

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Enterprise & Multimodal
- **ENT-01**: Multi-user role-based access control and persistent team cache sharing.
- **ENT-02**: Multimodal image/document validator support for Vision-Language models.
- **ENT-03**: Redis/PostgreSQL distributed cache backend for clustered deployments.

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Temperature/Sampling Tuning on Retry | Non-deterministic sampling breaks reproducible reliability scores and invalidates determinism guarantees. |
| Remote/Network Calls in Custom Validators | Validators must remain pure and local to preserve sub-millisecond execution and avoid distributed failures. |
| Unbounded Retries | Uncapped retries risk runaway API costs and infinite loops; capped strictly at 3 attempts. |
| Free-Text Unconstrained Critic | Research shows unconstrained evaluators produce high false-positive rates; critic must output structured diagnostics. |

## Traceability

Which phases cover which requirements. (To be mapped in ROADMAP.md).

| Requirement | Phase | Status |
|-------------|-------|--------|
| CLIENT-01..04 | Phase 1 | Pending |
| VALID-01..04  | Phase 2 | Pending |
| SCORE-01..03  | Phase 3 | Pending |
| CRITIC-01..03 | Phase 4 | Pending |
| RETRY-01..04  | Phase 5 | Pending |
| CACHE-01..04  | Phase 6 | Pending |
| UI-01..04     | Phase 6 | Pending |
| BENCH-01..04  | Phase 7 | Pending |
