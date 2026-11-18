# Architecture Research

**Domain:** LLM Self-Correcting Reliability Harness  
**Researched:** 2026-11-13  
**Confidence:** HIGH  

## System Topology & Contracts

```
                +-------------------------------------------------+
                |                   User Query                    |
                +-------------------------------------------------+
                                         |
                                         v
                         +-------------------------------+
                         |   SQLite Cache Lookup Gate    |
                         | (Key: Prompt + Model + ValSet)|
                         +-------------------------------+
                                  /             \
                   [Cache Hit ≥0.80]           [Cache Miss]
                                /                 \
                               v                   v
                     +---------------+    +------------------------+
                     | Instant Return|    |   LLMClient.generate   |
                     |  (Microsecond)|    | (Llama 3.1 8B Instant) |
                     +---------------+    +------------------------+
                                                       |
                                                       v
                                          +-------------------------+
                                          | Phase 1: Fast Checks    |
                                          | (Declarative Validators)|
                                          +-------------------------+
                                                       |
                                                       v
                                          +-------------------------+
                                          | Phase 2: Domain Logic   |
                                          | (Custom Python Callables|
                                          +-------------------------+
                                                       |
                                                       v
                                          +-------------------------+
                                          | Phase 3: Semantic Score |
                                          | (Sentence Transformers) |
                                          +-------------------------+
                                                       |
                                        [Any Failed or Rel < 0.80?]
                                                  /          \
                                            [Yes]             [No]
                                              /                 \
                                             v                   v
                              +--------------------+    +--------------------+
                              |  Critic Evaluator  |    | Store in Cache (if |
                              | (Structured Diag)  |    |  Reliability≥0.80) |
                              +--------------------+    +--------------------+
                                       |                           |
                                       v                           v
                              +--------------------+    +--------------------+
                              | Retry Orchestrator |    | Streamlit Decision |
                              | (Multi-turn inject)|    |     Dashboard      |
                              +--------------------+    +--------------------+
```

## Shared Core Contracts (To Lock in Phase 1)

1. **`LLMClient`**:
   - `generate(messages: list[dict], max_tokens: int, temperature: float = 0.0) -> LLMResponse`
   - Implementations: `GroqClient`, `OllamaClient`, `OpenAICompatibleClient`.

2. **`ValidatorResult`**:
   - `passed: bool`
   - `reason: str`
   - `severity: Literal["critical", "warning", "info"]`
   - `failure_type: FailureTypeEnum` (e.g. `SCHEMA_SYNTAX`, `REGEX_MISMATCH`, `SEMANTIC_DRIFT`, `CONSTRAINT_VIOLATION`)

3. **`CriticFeedback`**:
   - `passed: bool`
   - `diagnostics: list[DiagnosticItem]`
   - `guidance: str` (specific actionable correction for the LLM)
   - `escalate_to_negative_constraints: bool` (true after 2 consecutive failures)

4. **`CacheRecord`**:
   - `cache_key: str` (SHA256 of provider + model + prompt + sorted validator names/hashes)
   - `response_text: str`
   - `reliability_score: float` (must be ≥ 0.80 to be inserted)
   - `created_at: str`

