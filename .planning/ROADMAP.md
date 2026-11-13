# Roadmap: Agentic Harness v1.2

## Overview

Agentic Harness v1.2 transforms stochastic Llama 3.1 8B Instant generation into a deterministic, explainable, self-correcting system. The roadmap follows a contract-first horizontal architecture: we lock core data schemas and the pluggable LLMClient in Phase 1, build dual deterministic validation in Phase 2, integrate local semantic similarity scoring in Phase 3, implement structured critic diagnostics in Phase 4, establish closed-loop targeted retry orchestration with post-repair stability checks in Phase 5, provide SQLite selective caching and interactive Streamlit observability in Phase 6, and validate the full system against the 54-sample benchmark in Phase 7.

## Phases

- [ ] **Phase 1: Shared Core Contracts & Pluggable LLM Client** - Lock contracts (`ValidatorResult`, `CriticFeedback`, `LLMResponse`) and build the Groq/Ollama client interface with deterministic SDK settings (`max_retries=0`).
- [ ] **Phase 2: Dual Validator Engine & Registry** - Implement declarative schemas (Pydantic/regex) and pure custom Python validation functions under a unified registry.
- [ ] **Phase 3: Semantic Similarity Scoring** - Integrate local Sentence Transformers (`all-MiniLM-L6-v2`) embeddings and cosine similarity scoring independent of syntactic validation.
- [ ] **Phase 4: Structured Critic Evaluator** - Construct the diagnostic critic generating structured failure enums, severity ratings, and actionable prompt guidance.
- [ ] **Phase 5: Targeted Retry Orchestrator & Stability Engine** - Implement multi-turn error injection, post-repair cascade stability checks, prompt refinement escalation, and max 3-attempt ceiling.
- [ ] **Phase 6: Selective SQLite Cache & Streamlit Observability Dashboard** - Implement high-reliability caching (Reliability >= 0.80) with validator-hashed keys and interactive Streamlit trace monitoring.
- [ ] **Phase 7: 54-Sample Benchmark & Verification Suite** - Execute full benchmark suite verifying 95% success, 0.30 retries/query, and >= 81.8% autonomous recovery.

## Phase Details

### Phase 1: Shared Core Contracts & Pluggable LLM Client
**Goal**: Establish immutable data contracts across all subsystems and implement the pluggable LLM inference client with zero transport retries and fixed temperature.
**Depends on**: Nothing (first phase)
**Requirements**: CLIENT-01, CLIENT-02, CLIENT-03, CLIENT-04
**Success Criteria**:
  1. Core data classes (`LLMResponse`, `ValidatorResult`, `CriticFeedback`, `DiagnosticItem`, `TaskSpec`) are defined with strict type safety.
  2. `LLMClient` protocol supports Groq Cloud (`llama-3.1-8b-instant`) as default with automatic/configurable Ollama local fallback.
  3. Client sessions explicitly set `max_retries=0` to guarantee zero silent HTTP transport retries.
  4. Temperature is strictly fixed at 0.0 across all calls to ensure reproducible reliability metrics.
**Plans**: 3 plans

Wave 1:
- [ ] 01-01: Shared Core Contracts & Type Definitions (Pydantic schemas, enums, RateLimitError)

Wave 2 *(blocked on Wave 1)*:
- [ ] 01-02: Pluggable LLM Client Implementation (LLMClient ABC, GroqClient with max_retries=0, OllamaClient)

Wave 3 *(blocked on Waves 1 & 2)*:
- [ ] 01-03: Contract & Client Verification Suite (pytest suite for contracts and mock transport clients)

### Phase 2: Dual Validator Engine & Registry
**Goal**: Deliver a two-tier validator engine supporting zero-code declarative schemas and pure custom Python functions sharing the `ValidatorResult` contract.
**Depends on**: Phase 1
**Requirements**: VALID-01, VALID-02, VALID-03, VALID-04
**Success Criteria**:
  1. Declarative validator parses Pydantic models, JSON Schema, regex patterns, and keyword blacklists/whitelists.
  2. Custom validator registry accepts pure functions `validate(output, context) -> ValidatorResult` with enforcement against network/unseeded calls.
  3. Execution pipeline runs cheap declarative checks first, failing fast before invoking custom callables.
  4. Active validator set produces a deterministic hash for downstream cache key composition.
**Plans**: 3 plans (TBD)

Plans:
- [ ] 02-01: Build declarative Pydantic and regex validation engine
- [ ] 02-02: Build custom function validator runner with purity enforcement
- [ ] 02-03: Implement unified validator registry and two-stage pipeline execution

### Phase 3: Semantic Similarity Scoring
**Goal**: Implement local sentence embedding comparisons using Sentence Transformers to detect semantic drift independently of schema compliance.
**Depends on**: Phase 1
**Requirements**: SCORE-01, SCORE-02, SCORE-03
**Success Criteria**:
  1. Sentence Transformers model (`all-MiniLM-L6-v2`) loads locally with cached weights and sub-15ms inference latency.
  2. Cosine similarity calculates normalized match score [0.0, 1.0] against reference intent or ground truth.
  3. Semantic scoring executes as a distinct decoupled phase, preventing syntactically valid hallucinations from passing unflagged.
**Plans**: 2 plans (TBD)

Plans:
- [ ] 03-01: Integrate Sentence Transformers model with local embedding cache
- [ ] 03-02: Build semantic similarity scorer and threshold evaluation logic

### Phase 4: Structured Critic Evaluator
**Goal**: Build a constrained critic evaluator that transforms validator failures and semantic drift into structured diagnostic guidance for targeted retries.
**Depends on**: Phase 1, Phase 2, Phase 3
**Requirements**: CRITIC-01, CRITIC-02, CRITIC-03
**Success Criteria**:
  1. Critic outputs conform strictly to `CriticFeedback` schema with standardized `failure_type` enums and severity levels.
  2. Critic generates concrete, natural-language correction instructions tied directly to the failed validator evidence.
  3. Critic never produces unconstrained free-text assertions, eliminating false-positive evaluator hallucinations.
**Plans**: 2 plans (TBD)

Plans:
- [ ] 04-01: Implement critic prompt templates and structured output parser
- [ ] 04-02: Build diagnostic aggregator linking validator results to actionable guidance

### Phase 5: Targeted Retry Orchestrator & Stability Engine
**Goal**: Orchestrate multi-turn self-correction loops with post-repair stability checks, prompt refinement escalation, and a hard retry ceiling.
**Depends on**: Phase 1, Phase 2, Phase 3, Phase 4
**Requirements**: RETRY-01, RETRY-02, RETRY-03, RETRY-04
**Success Criteria**:
  1. Retry messages append precise validator reasons, failed prior output, and critic guidance to conversation history without altering sampling temperature.
  2. Post-repair stability check re-runs the entire validator suite on retried outputs, catching repair-induced cascade regressions.
  3. Persistent failures across 2 consecutive retries automatically escalate to prompt refinement with negative constraints.
  4. Retry loops terminate strictly within max 3 attempts, returning an audited execution failure trace if unrecoverable.
**Plans**: 3 plans (TBD)

Plans:
- [ ] 05-01: Build multi-turn retry message constructor and conversation history manager
- [ ] 05-02: Implement post-repair stability check and prompt escalation engine
- [ ] 05-03: Build closed-loop orchestrator with execution tracing and retry budget guards

### Phase 6: Selective SQLite Cache & Streamlit Observability Dashboard
**Goal**: Deploy selective high-reliability caching (Reliability >= 0.80) and an interactive Streamlit UI for real-time generation and audit trace inspection.
**Depends on**: Phase 5
**Requirements**: CACHE-01, CACHE-02, CACHE-03, CACHE-04, UI-01, UI-02, UI-03, UI-04
**Success Criteria**:
  1. SQLite caching layer stores only verified responses with Reliability Score >= 0.80.
  2. Cache keys deterministically incorporate SHA256 of provider + model + prompt + active validator set hash.
  3. Streamlit web application renders live input testing, retry timelines, and validator pass/fail badges.
  4. Audit panel displays critic feedback cards, token/latency metrics, and cache hit/miss indicators.
**Plans**: 3 plans (TBD)

Plans:
- [ ] 06-01: Implement SQLite cache manager with selective insertion and validator-hashed keys
- [ ] 06-02: Develop Streamlit interactive query runner and real-time generation panel
- [ ] 06-03: Build Streamlit audit trail dashboard with validator breakdowns and cache metrics

### Phase 7: 54-Sample Benchmark & Verification Suite
**Goal**: Run comprehensive automated evaluation across 54 diverse benchmark samples to prove measured reliability, recovery, and retry targets.
**Depends on**: Phase 6
**Requirements**: BENCH-01, BENCH-02, BENCH-03, BENCH-04
**Success Criteria**:
  1. Benchmark suite executes 54 diverse challenging test cases against unharnessed baseline vs Agentic Harness.
  2. Success rate increases from 1.5% zero-shot to >= 95% harnessed.
  3. Average reliability score improves from 0.816 baseline to >= 0.942.
  4. Autonomous recovery rate achieves >= 81.8% at average <= 0.30 retries/query.
**Plans**: 2 plans (TBD)

Plans:
- [ ] 07-01: Assemble 54-sample benchmark dataset and automated test runner
- [ ] 07-02: Execute evaluation runs, generate comparative metrics report, and verify all invariant targets

