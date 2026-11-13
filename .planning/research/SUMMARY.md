# Research Summary: LLM Reliability Harness

**Project:** Agentic Harness v1.2  
**Completed:** 2026-11-13  
**Status:** Validated & Ready for Requirements  

## Executive Summary
This targeted research pass confirms that the architecture and locked core contracts established in initialization are sound and align with modern 2026 LLM validation best practices. Crucially, 5 key pitfalls were identified and addressed before writing requirements:

1. **Structured Critic Outputs**: Eliminates unconstrained evaluator drift by binding critic diagnostics to strict failure enums and severity ratings.
2. **Post-Repair Stability Invariant**: Mandates full validator suite re-execution on every retry to catch repair-induced cascade regressions.
3. **Deterministic Transport**: Explicit `max_retries=0` on client SDKs keeps the 0.30 retries/query metric honest and observable.
4. **Decoupled Verification Layers**: Preserves syntax/schema checks, semantic similarity scoring, and critic evaluation as independent sequential stages.
5. **Validator-Aware Caching**: Embeds active validator hashes into cache keys to prevent stale hits after rule adjustments.

## Core Stack Consensus
- **Python 3.11+**, **Streamlit**, **SQLite3**, **Pydantic v2**, **Sentence Transformers** (`all-MiniLM-L6-v2`), and **Groq Cloud API** (Llama 3.1 8B Instant) with **Ollama** fallback.
