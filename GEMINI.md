# Agentic Harness v1.2 — Development Guidelines

This project uses the **Get Shit Done (GSD)** execution framework for planning and verifying all milestones and phases.

## Project Context
- **Core Value**: Turn stochastic, hallucination-prone LLM outputs into deterministic, verifiable, and explainable responses through targeted, self-correcting feedback loops and selective high-reliability caching.
- **Project State**: Consult [.planning/STATE.md](file:///s:/porttttt/agentic/.planning/STATE.md) for current phase and plan progress.
- **Roadmap**: Consult [.planning/ROADMAP.md](file:///s:/porttttt/agentic/.planning/ROADMAP.md) for all phases and requirements.
- **Requirements**: Consult [.planning/REQUIREMENTS.md](file:///s:/porttttt/agentic/.planning/REQUIREMENTS.md) for full v1 specifications.

## Architectural Invariants (Must Preserve)
1. **Locked Contracts**: `LLMResponse`, `ValidatorResult`, `CriticFeedback`, and `TaskSpec` defined in Phase 1 are immutable shared interfaces.
2. **Deterministic Sampling**: Temperature must be fixed at 0.0 across all inference and retries.
3. **Transport Transparency**: Client SDKs must be configured with `max_retries=0` to prevent silent retry multiplication.
4. **Structured Critic**: The critic must output structured diagnostic enums (`failure_type`) and severity levels, not unconstrained prose.
5. **Post-Repair Stability**: Every retry must re-run the complete validator suite to detect cascade regressions.
6. **High-Reliability Caching**: SQLite cache stores only responses with composite Reliability Score >= 0.80, with keys hashing prompt + model + active validator set.

## GSD Workflow Commands
- `/gsd-discuss-phase <phase>`: Discuss approach and context before planning a phase.
- `/gsd-plan-phase <phase>`: Create task-level execution plans.
- `/gsd-execute-phase <phase>`: Execute plans for a phase.
- `/gsd-verify-work <phase>`: Formally verify deliverables and metrics.
- `/gsd-progress`: Inspect current progress and velocity.

