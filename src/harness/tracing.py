"""Execution Tracing and Budget Guards for Agentic Harness v1.2.

Provides comprehensive execution tracing, retry budget management,
and integration with the full retry orchestration pipeline.
"""
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .client import LLMClient, get_llm_client
from .contracts import (
    CriticFeedback,
    FailureTypeEnum,
    LLMResponse,
    SeverityEnum,
    TaskSpec,
    ValidatorResult,
)
from .aggregator import DiagnosticAggregator, create_diagnostic_aggregator
from .critic import CriticEvaluator, create_critic_evaluator
from .escalation import PromptRefiner, create_prompt_refiner
from .orchestrator import (
    ConversationHistory,
    OrchestrationResult,
    RetryAttempt,
    RetryOrchestrator,
)
from .scorer import SemanticScorer, create_semantic_scorer
from .validators import ValidatorRegistry


@dataclass
class TraceEvent:
    """Single event in the execution trace."""
    event_id: str
    timestamp: float
    event_type: str  # "generation", "validation", "scoring", "critique", "retry", "escalation", "success", "failure"
    attempt_number: int
    data: Dict[str, Any]
    latency_ms: float


@dataclass
class ExecutionTrace:
    """Complete execution trace for a task run."""
    trace_id: str
    task_name: str
    start_time: float
    end_time: Optional[float] = None
    events: List[TraceEvent] = field(default_factory=list)
    final_result: Optional[OrchestrationResult] = None

    def add_event(
        self,
        event_type: str,
        attempt_number: int,
        data: Dict[str, Any],
        latency_ms: float = 0.0,
    ) -> TraceEvent:
        """Add an event to the trace."""
        event = TraceEvent(
            event_id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            event_type=event_type,
            attempt_number=attempt_number,
            data=data,
            latency_ms=latency_ms,
        )
        self.events.append(event)
        return event

    def to_dict(self) -> Dict[str, Any]:
        """Serialize trace to dictionary."""
        return {
            "trace_id": self.trace_id,
            "task_name": self.task_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": (self.end_time - self.start_time) * 1000 if self.end_time else None,
            "events": [asdict(e) for e in self.events],
            "final_result": {
                "success": self.final_result.success,
                "total_attempts": self.final_result.total_attempts,
                "total_latency_ms": self.final_result.total_latency_ms,
                "reliability_score": self.final_result.reliability_score,
            } if self.final_result else None,
        }

    def to_json(self) -> str:
        """Serialize trace to JSON."""
        return json.dumps(self.to_dict(), indent=2)


class RetryBudget:
    """Manages retry budget with configurable limits."""

    def __init__(
        self,
        max_retries: int = 3,
        max_total_latency_ms: float = 60000,  # 60 seconds
        max_total_tokens: int = 10000,
    ):
        self.max_retries = max_retries
        self.max_total_latency_ms = max_total_latency_ms
        self.max_total_tokens = max_total_tokens

        self.retries_used = 0
        self.total_latency_ms = 0.0
        self.total_tokens = 0

    def can_retry(self) -> bool:
        """Check if another retry is allowed."""
        return (
            self.retries_used < self.max_retries and
            self.total_latency_ms < self.max_total_latency_ms and
            self.total_tokens < self.max_total_tokens
        )

    def consume(self, latency_ms: float, tokens: int = 0) -> None:
        """Consume budget for an attempt."""
        self.retries_used += 1
        self.total_latency_ms += latency_ms
        self.total_tokens += tokens

    def remaining_retries(self) -> int:
        """Get remaining retry count."""
        return max(0, self.max_retries - self.retries_used)

    def reset(self) -> None:
        """Reset budget."""
        self.retries_used = 0
        self.total_latency_ms = 0.0
        self.total_tokens = 0

    def get_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        return {
            "retries_used": self.retries_used,
            "max_retries": self.max_retries,
            "remaining_retries": self.remaining_retries(),
            "total_latency_ms": self.total_latency_ms,
            "max_total_latency_ms": self.max_total_latency_ms,
            "total_tokens": self.total_tokens,
            "max_total_tokens": self.max_total_tokens,
            "can_retry": self.can_retry(),
        }


class TracedRetryOrchestrator:
    """Retry orchestrator with full execution tracing and budget guards.

    Extends the base RetryOrchestrator with:
    - Comprehensive event tracing
    - Retry budget enforcement
    - Escalation integration
    - Detailed audit trail
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        validator_registry: Optional[ValidatorRegistry] = None,
        semantic_scorer: Optional[SemanticScorer] = None,
        critic: Optional[CriticEvaluator] = None,
        aggregator: Optional[DiagnosticAggregator] = None,
        prompt_refiner: Optional[PromptRefiner] = None,
        max_retries: int = 3,
        temperature: float = 0.0,
        budget: Optional[RetryBudget] = None,
    ):
        self.llm_client = llm_client or get_llm_client()
        self.validator_registry = validator_registry or ValidatorRegistry()
        self.semantic_scorer = semantic_scorer or create_semantic_scorer()
        self.critic = critic or create_critic_evaluator(llm_client=self.llm_client)
        self.aggregator = aggregator or create_diagnostic_aggregator(critic=self.critic)
        self.prompt_refiner = prompt_refiner or create_prompt_refiner()
        self.max_retries = max_retries
        self.temperature = temperature
        self.budget = budget or RetryBudget(max_retries=max_retries)

        # Base orchestrator for core logic
        self._base_orchestrator = RetryOrchestrator(
            llm_client=self.llm_client,
            validator_registry=self.validator_registry,
            semantic_scorer=self.semantic_scorer,
            critic=self.critic,
            aggregator=self.aggregator,
            max_retries=max_retries,
            temperature=temperature,
        )

    def execute(
        self,
        task_spec: TaskSpec,
        prompt: str,
        ground_truth: Optional[str] = None,
        trace: Optional[ExecutionTrace] = None,
    ) -> OrchestrationResult:
        """Execute with full tracing and budget management."""
        # Create trace if not provided
        if trace is None:
            trace = ExecutionTrace(
                trace_id=str(uuid.uuid4())[:8],
                task_name=task_spec.task_name,
                start_time=time.time(),
            )

        # Register task spec
        self.validator_registry.register_declarative(task_spec)
        self._base_orchestrator.validator_registry = self.validator_registry

        # Reset state
        self.prompt_refiner.reset()
        self.budget.reset()

        # Initialize conversation history
        history = ConversationHistory(system_prompt=task_spec.system_prompt)
        history.add_user_message(prompt)

        retry_attempts = []
        prior_failures = []

        trace.add_event("start", 0, {"prompt": prompt, "task": task_spec.task_name})

        for attempt in range(self.max_retries + 1):
            # Check budget
            if not self.budget.can_retry() and attempt > 0:
                trace.add_event("budget_exhausted", attempt, self.budget.get_status())
                break

            start_time = time.perf_counter()

            # Generate response
            response = self.llm_client.generate(
                messages=history.get_messages(),
                temperature=self.temperature,
                max_tokens=2048,
            )
            generated_output = response.content
            history.add_assistant_message(generated_output)

            gen_latency = (time.perf_counter() - start_time) * 1000
            trace.add_event("generation", attempt + 1, {
                "output_length": len(generated_output),
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
            }, gen_latency)

            # Run validators
            val_start = time.perf_counter()
            validator_results = self.validator_registry.validate(generated_output)
            val_latency = (time.perf_counter() - val_start) * 1000

            trace.add_event("validation", attempt + 1, {
                "results": [
                    {
                        "passed": r.passed,
                        "reason": r.reason,
                        "severity": r.severity.value,
                        "failure_type": r.failure_type.value if r.failure_type else None,
                    }
                    for r in validator_results
                ],
                "all_passed": all(r.passed for r in validator_results),
            }, val_latency)

            # Run semantic scorer if ground truth provided
            semantic_result = None
            if ground_truth:
                sem_start = time.perf_counter()
                semantic_result = self.semantic_scorer.evaluate(generated_output, ground_truth)
                sem_latency = (time.perf_counter() - sem_start) * 1000

                trace.add_event("scoring", attempt + 1, {
                    "passed": semantic_result.passed,
                    "reason": semantic_result.reason,
                    "failure_type": semantic_result.failure_type.value if semantic_result.failure_type else None,
                }, sem_latency)

            # Get critic feedback
            crit_start = time.perf_counter()
            critic_feedback = self.aggregator.evaluate_with_critic(
                task_spec=task_spec,
                generated_output=generated_output,
                validator_results=validator_results,
                semantic_result=semantic_result,
                ground_truth=ground_truth,
                retry_count=attempt,
                prior_failures=prior_failures,
            )
            crit_latency = (time.perf_counter() - crit_start) * 1000

            trace.add_event("critique", attempt + 1, {
                "passed": critic_feedback.passed,
                "guidance": critic_feedback.guidance,
                "constraints": critic_feedback.constraints,
                "escalate": critic_feedback.escalate_to_negative_constraints,
                "diagnostics_count": len(critic_feedback.diagnostics),
            }, crit_latency)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Record attempt
            retry_attempt = RetryAttempt(
                attempt_number=attempt + 1,
                prompt_messages=history.get_messages(),
                generated_output=generated_output,
                validator_results=validator_results,
                semantic_result=semantic_result,
                critic_feedback=critic_feedback,
                latency_ms=elapsed_ms,
            )
            retry_attempts.append(retry_attempt)

            # Check if all passed
            all_passed = all(r.passed for r in validator_results)
            semantic_passed = semantic_result.passed if semantic_result else True

            # Post-repair stability check
            if attempt > 0 and all_passed and semantic_passed:
                stab_start = time.perf_counter()
                stability_passed = self._base_orchestrator._post_repair_stability_check(
                    generated_output, task_spec, ground_truth
                )
                stab_latency = (time.perf_counter() - stab_start) * 1000

                trace.add_event("stability_check", attempt + 1, {
                    "passed": stability_passed,
                }, stab_latency)

                if not stability_passed:
                    all_passed = False
                    semantic_passed = False
                    critic_feedback.diagnostics.append(
                        type(critic_feedback.diagnostics[0])(
                            failure_type=FailureTypeEnum.CONSTRAINT_VIOLATION,
                            severity=SeverityEnum.CRITICAL,
                            validator_name="post_repair_stability",
                            message="Post-repair stability check failed: new regressions detected",
                            suggested_correction="Ensure corrections don't introduce new failures",
                        )
                    )

            if all_passed and semantic_passed:
                # Success!
                reliability_score = self._base_orchestrator._compute_reliability_score(
                    validator_results, semantic_result, critic_feedback
                )

                trace.add_event("success", attempt + 1, {
                    "reliability_score": reliability_score,
                })
                trace.end_time = time.time()

                return OrchestrationResult(
                    success=True,
                    final_output=generated_output,
                    final_validator_results=validator_results,
                    final_semantic_result=semantic_result,
                    final_critic_feedback=critic_feedback,
                    retry_attempts=retry_attempts,
                    total_attempts=attempt + 1,
                    total_latency_ms=sum(r.latency_ms for r in retry_attempts),
                    reliability_score=reliability_score,
                )

            # Consume budget
            self.budget.consume(elapsed_ms, response.prompt_tokens + response.completion_tokens)

            trace.add_event("retry", attempt + 1, {
                "budget_status": self.budget.get_status(),
            })

            # Prepare for next retry
            if attempt < self.max_retries:
                # Refine prompt if escalation needed
                refined_prompt = self.prompt_refiner.refine(
                    prompt=prompt,
                    task_spec=task_spec,
                    validator_results=validator_results,
                    semantic_result=semantic_result,
                    critic_constraints=critic_feedback.constraints,
                )

                if refined_prompt != prompt:
                    trace.add_event("escalation", attempt + 1, {
                        "persistent_failures": list(self.prompt_refiner.escalation_engine.get_persistent_failures()),
                    })
                    prompt = refined_prompt
                    history.clear()
                    history.add_user_message(prompt)
                else:
                    # Build retry prompt with failure context
                    retry_prompt = history.build_retry_prompt(
                        original_prompt=prompt,
                        failed_output=generated_output,
                        critic_feedback=critic_feedback,
                        validator_results=validator_results,
                    )
                    history.add_user_message(retry_prompt)

                # Track failures for escalation
                prior_failures.append({
                    "diagnostics": [
                        {
                            "failure_type": d.failure_type.value,
                            "severity": d.severity.value,
                            "validator_name": d.validator_name,
                            "message": d.message,
                        }
                        for d in critic_feedback.diagnostics
                    ]
                })

        # Max retries exceeded or budget exhausted
        trace.add_event("failure", len(retry_attempts), {
            "reason": "max_retries_exceeded" if self.budget.retries_used >= self.max_retries else "budget_exhausted",
            "budget_status": self.budget.get_status(),
        })
        trace.end_time = time.time()

        return OrchestrationResult(
            success=False,
            final_output=retry_attempts[-1].generated_output if retry_attempts else "",
            final_validator_results=retry_attempts[-1].validator_results if retry_attempts else [],
            final_semantic_result=retry_attempts[-1].semantic_result if retry_attempts else None,
            final_critic_feedback=retry_attempts[-1].critic_feedback if retry_attempts else CriticFeedback(passed=False, diagnostics=[]),
            retry_attempts=retry_attempts,
            total_attempts=len(retry_attempts),
            total_latency_ms=sum(r.latency_ms for r in retry_attempts),
            reliability_score=0.0,
            failure_summary=f"Failed after {len(retry_attempts)} attempts. Budget: {self.budget.get_status()}",
        )


def create_traced_orchestrator(
    llm_client: Optional[LLMClient] = None,
    validator_registry: Optional[ValidatorRegistry] = None,
    semantic_scorer: Optional[SemanticScorer] = None,
    critic: Optional[CriticEvaluator] = None,
    aggregator: Optional[DiagnosticAggregator] = None,
    prompt_refiner: Optional[PromptRefiner] = None,
    max_retries: int = 3,
    budget: Optional[RetryBudget] = None,
) -> TracedRetryOrchestrator:
    """Factory function to create a TracedRetryOrchestrator with default settings."""
    return TracedRetryOrchestrator(
        llm_client=llm_client,
        validator_registry=validator_registry,
        semantic_scorer=semantic_scorer,
        critic=critic,
        aggregator=aggregator,
        prompt_refiner=prompt_refiner,
        max_retries=max_retries,
        budget=budget,
    )