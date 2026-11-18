# Agentic Harness v1.2

## What This Is

An explainable, self-correcting reliability layer for LLMs, delivered as both an importable Python library and an interactive Streamlit web dashboard. It wraps Llama 3.1 8B Instant in a closed feedback loop that pairs deterministic validators, semantic similarity scoring, and a critic-based evaluator to drive targeted multi-turn retries, logging every decision along the way.

## Core Value

Turn stochastic, hallucination-prone LLM outputs into deterministic, verifiable, and explainable responses through targeted, self-correcting feedback loops and selective high-reliability caching.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

(None yet — ship to validate)

### Active

<!-- Current scope. Building toward these. -->

- [ ] **Pluggable LLM Client**: Unified interface wrapping Llama 3.1 8B Instant via Groq Cloud API as primary/default, with fallback support for local Ollama and OpenAI-compatible providers.
- [ ] **Dual Validator System**: Support for declarative validators (Pydantic models / JSON Schema, regex patterns, keyword blacklists/whitelists, structure bounds) and custom deterministic Python functions (`validate(output, context) -> ValidatorResult`).
- [ ] **Unified Validator Result Contract**: Standardized `ValidatorResult(passed, reason, severity)` returned by all validators so the retry engine and dashboard treat them uniformly.
- [ ] **Semantic Similarity Scoring**: Embedding and cosine similarity comparison using Sentence Transformers to verify semantic consistency against ground truth or query intent.
- [ ] **Critic & Targeted Retry Orchestration**: Closed-loop retry engine using multi-turn error injection (injecting specific validator failures, exact failed output, and critic guidance) without temperature drift.
- [ ] **Escalation Layer**: Automatic prompt refinement with negative constraints when validation fails across 2 consecutive attempts.
- [ ] **Selective SQLite Cache**: High-reliability caching layer storing only responses with Reliability Score ≥ 0.80, keyed on provider, model, prompt hash, and active validator set.
- [ ] **Streamlit Interactive Dashboard**: Real-time visual monitoring displaying raw prompts, output traces, validation pass/fail breakdowns, critic explanations, retry progression, and cache hit/miss stats.
- [ ] **Benchmark & Evaluation Suite**: Evaluation harness supporting the 54-sample benchmark + challenge suite to measure pass rate, reliability score progression, and autonomous recovery rates.

### Out of Scope

- **Temperature / Sampling Tuning on Retry**: Excluded by design because non-deterministic sampling prevents reproducible reliability scoring and degrades deterministic guarantees.
- **Network / Remote Calls inside Custom Validators**: Excluded to ensure custom validators remain pure, fast, and deterministic.
- **Unbounded Retries**: Hard upper bound on retry attempts (max 3) to prevent runaway token costs and infinite loops.
- **Multi-Tenant User Management**: v1 is a developer-focused local library and dashboard, not a hosted SaaS platform.

## Context

- Target Model: Meta Llama 3.1 8B Instant (known for extreme speed but occasional formatting drift or schema hallucinations under zero-shot prompting).
- Primary Inference: Groq Cloud API (`llama-3.1-8b-instant`) with sub-second time-to-first-token.
- Target Benchmarks: 1.5% zero-shot success → 95% harnessed success, 0.816 → 0.942 reliability score, 81.8% autonomous recovery at 0.30 retries/query.
- Persistence: SQLite database for logging evaluation decisions and selective caching.

## Constraints

- **Language & Stack**: Python 3.11+, Streamlit, SQLite, Sentence Transformers (`all-MiniLM-L6-v2` or similar lightweight model).
- **Inference Determinism**: Keep temperature and sampling parameters fixed throughout the retry loop.
- **Cache Invalidation**: Cache keys must include the hash of the active validator set so configuration changes invalidate stale cached items.
- **Performance**: Declarative validators execute first before custom functions to fail fast on cheap checks.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Pluggable LLMClient (Groq default + Ollama fallback) | Maximizes inference speed on Groq while allowing offline local development via Ollama | — Pending |
| Dual Validator System with `ValidatorResult` contract | Allows fast zero-code schema validation plus rich domain logic under a single unified interface | — Pending |
| Multi-turn error injection over prompt rewrite | Feeds exact failure context to the model while keeping token overhead minimal (0.30 retries/query average) | — Pending |
| Prompt refinement escalation after 2 failures | Prevents infinite retry loops on stubborn constraint violations by enforcing negative constraints | — Pending |
| Cache cutoff at Reliability ≥ 0.80 | Ensures only high-confidence, verified responses populate cache for instant repeat retrieval | — Pending |
| Fixed temperature during retry | Preserves reproducible reliability scoring and strict evaluation integrity | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-11-13 after initialization*

