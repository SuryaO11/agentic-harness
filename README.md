# 🤖 Agentic Harness v1.2

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-119%20passed-brightgreen.svg)]()
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An **explainable, self-correcting reliability framework** for Large Language Models. It turns stochastic, hallucination-prone LLM outputs into deterministic, verifiable, and explainable responses through targeted feedback loops and selective high-reliability caching.

---

## 🌟 Key Features

- **Dual Validation Engine**:
  - *Declarative Validators*: Rapid fail-fast schema checks using Pydantic, JSON Schema, Regex patterns, keyword whitelists/blacklists, and length constraints.
  - *Deterministic Custom Validators*: Pure Python validation functions with enforced purity (no network/LLM calls).
- **Decoupled Semantic Similarity**: Local embedding-based scoring (`all-MiniLM-L6-v2` via Sentence Transformers) detects semantic drift independently of syntactic validity.
- **Structured Critic Evaluator**: Generates typed diagnostics (`failure_type` enum, severity levels, actionable guidance) without unconstrained prose or evaluator hallucinations.
- **Targeted Retry Orchestration**: Multi-turn conversation feedback injecting failure context, post-repair cascade stability checks, and automatic escalation to negative constraints (capped at max 3 retries).
- **High-Reliability SQLite Cache**: Automatically caches verified responses with composite `Reliability Score >= 0.80` using deterministic SHA256 keys (hashing provider, model, prompt, and active validator set).
- **Interactive Streamlit Observability Dashboard**: Real-time generation progression, validator pass/fail badges, critic diagnostic cards, and cache hit/miss telemetry.
- **54-Sample Benchmark Suite**: Verifies baseline zero-shot improvement from **1.5%** to **≥ 95%** harnessed success rate and **≥ 81.8%** autonomous recovery.

---

## 🏗️ Architecture

```
User Prompt
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                   AGENTIC HARNESS                        │
│                                                          │
│  ┌──────────┐    ┌────────────┐    ┌──────────────────┐  │
│  │ LLM      │───▶│ Validator  │───▶│ Semantic Scorer  │  │
│  │ Client   │    │ Engine     │    │ (Sentence Trans) │  │
│  └──────────┘    └────────────┘    └──────────────────┘  │
│       ▲                │                    │             │
│       │         ┌──────▼────────────────────▼──────┐     │
│       │         │         Critic Evaluator          │     │
│       │         │   (structured diagnostic output)  │     │
│       │         └──────────────────────────────────┘     │
│       │                      │                            │
│       │         ┌────────────▼─────────────────────┐     │
│       └─────────│    Retry Orchestrator             │     │
│                 │  (inject failure + retry up to 3) │     │
│                 └──────────────────────────────────-┘     │
│                                │                          │
│                 ┌──────────────▼──────────────────────┐  │
│                 │     SQLite Cache (Reliability≥0.80)  │  │
│                 └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Streamlit Observability Dashboard
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/SuryaO11/agentic-harness.git
cd agentic-harness
```

### 2. Set Up Virtual Environment

```bash
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1

# Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install streamlit pydantic jsonschema httpx sentence-transformers pytest
```

### 4. Launch Observability Dashboard

```bash
streamlit run run_dashboard.py
```

Open your browser to `http://localhost:8501`.

---

## 🛠️ Usage Example

```python
from harness.contracts import TaskSpec
from harness.orchestrator import RetryOrchestrator
from harness.client import get_llm_client

# Define your task schema and constraints
task = TaskSpec(
    task_name="invoice_extractor",
    json_schema={
        "type": "object",
        "properties": {
            "customer": {"type": "string"},
            "invoice_number": {"type": "string"},
            "total": {"type": "number"},
            "currency": {"type": "string"}
        },
        "required": ["customer", "invoice_number", "total", "currency"]
    }
)

# Initialize orchestrator (Groq, Ollama, or Mock)
orchestrator = RetryOrchestrator(
    llm_client=get_llm_client(provider="groq", model_name="llama-3.1-8b-instant"),
    max_retries=3,
    temperature=0.0
)

# Execute query with self-correcting validation loop
result = orchestrator.execute(
    task_spec=task,
    prompt="Extract invoice details for Acme Corp, invoice INV-202401, total 1250.00 USD."
)

print(f"Success: {result.success}")
print(f"Reliability Score: {result.reliability_score:.3f}")
print(f"Retries: {result.retry_count}")
print(f"Output: {result.final_output}")
```

---

## 🧪 Testing & Verification

Run the full automated test suite (119 unit tests):

```bash
# Set PYTHONPATH to src
$env:PYTHONPATH="src"  # Windows PowerShell
export PYTHONPATH="src"  # Linux / macOS

pytest tests/ -v
```

Run the 54-sample comparative benchmark suite:

```bash
python -m harness.benchmark
```

---

## 📜 Architectural Invariants

| Invariant | Specification |
|---|---|
| **Deterministic Sampling** | Temperature fixed strictly at `0.0` across all attempts. |
| **Transport Transparency** | Client SDK `max_retries=0` to prevent silent retry multiplication. |
| **Structured Critic** | Critic outputs enum failure types and severity, eliminating hallucinated guidance. |
| **Post-Repair Stability** | Complete validator suite re-executed on retried outputs to detect regressions. |
| **High-Reliability Caching** | Only responses with `Reliability Score >= 0.80` stored; keys hash prompt + model + active validator set. |

---

## 📄 License

This project is licensed under the MIT License.

