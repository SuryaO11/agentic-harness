# Stack Research

**Domain:** LLM Self-Correcting Reliability Harness & Analytics  
**Researched:** 2026-11-13  
**Confidence:** HIGH  

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **Python** | 3.11+ | Core language | Stable typing, performance, sentence-transformers compatibility |
| **Streamlit** | ^1.39.0 | Interactive web dashboard | Fast reactive UI for inspecting generation, validation, and retry trees |
| **SQLite3** | standard library | High-reliability cache & evaluation log | Zero-setup, ACID-compliant, microsecond reads for cached query hits |
| **Groq Python SDK** | ^0.11.0 | Fast Llama 3.1 8B inference | Sub-second TTFT allows multi-turn retries to remain interactive |
| **Sentence-Transformers** | ^3.2.0 | Semantic similarity scoring | Local vector embeddings (`all-MiniLM-L6-v2`) without third-party API dependencies |
| **Pydantic** | ^2.9.0 | Declarative schemas & validation | High-speed Rust core (pydantic-core), structured error extraction |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **Ollama Python SDK** | ^0.3.3 | Local inference fallback | When `LLM_PROVIDER=ollama` or running offline |
| **httpx** | ^0.27.0 | Underlying HTTP client for providers | Explicit connection pooling; `max_retries=0` to avoid multiplied retries |
| **pytest** | ^8.3.0 | Test runner | Verification of contracts, validators, and caching logic |

### Installation

```bash
pip install streamlit pydantic sentence-transformers groq ollama pytest
```

### Critical Configuration Rules
- **Explicit SDK Retries**: Pass `max_retries=0` to the Groq/Ollama client instances to ensure the SDK does not perform silent transport retries that artificially inflate retry counts.
