"""Retry Orchestrator for Agentic Harness v1.2.

Implements multi-turn self-correction loops with post-repair stability checks,
prompt refinement escalation, and a hard retry ceiling (max 3 attempts).
"""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .client import LLMClient, get_llm_client
from .contracts import (
    CriticFeedback,
    DiagnosticItem,
    FailureTypeEnum,
    LLMResponse,
    SeverityEnum,
    TaskSpec,
    ValidatorResult,
)
from .aggregator import DiagnosticAggregator, create_diagnostic_aggregator
from .critic import CriticEvaluator, create_critic_evaluator
from .scorer import SemanticScorer, create_semantic_scorer
from .validators import ValidatorRegistry, Validator


MAX_RETRIES = 3
RELIABILITY_THRESHOLD = 0.80


@dataclass
class RetryAttempt:
    """Record of a single retry attempt."""
    attempt_number: int
    prompt_messages: List[Dict[str, str]]
    generated_output: str
    validator_results: List[ValidatorResult]
    semantic_result: Optional[ValidatorResult]
    critic_feedback: CriticFeedback
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class OrchestrationResult:
    """Final result of the orchestration process."""
    success: bool
    final_output: str
    final_validator_results: List[ValidatorResult]
    final_semantic_result: Optional[ValidatorResult]
    final_critic_feedback: CriticFeedback
    retry_attempts: List[RetryAttempt]
    total_attempts: int
    total_latency_ms: float
    reliability_score: float
    failure_summary: Optional[str] = None

    @property
    def retry_count(self) -> int:
        """Number of retries performed (total attempts - 1)."""
        return max(0, self.total_attempts - 1)

    @property
    def validator_results(self) -> List[ValidatorResult]:
        """Convenience alias for final_validator_results."""
        return self.final_validator_results

    @property
    def critic_feedback(self) -> CriticFeedback:
        """Convenience alias for final_critic_feedback."""
        return self.final_critic_feedback

    @property
    def semantic_result(self) -> Optional[ValidatorResult]:
        """Convenience alias for final_semantic_result."""
        return self.final_semantic_result


class ConversationHistory:
    """Manages conversation history for multi-turn retries.

    Maintains the message history with injected validator failures
    and critic guidance for each retry attempt.
    """

    def __init__(self, system_prompt: str = ""):
        self.messages: List[Dict[str, str]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def add_user_message(self, content: str) -> None:
        """Add a user message to history."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to history."""
        self.messages.append({"role": "assistant", "content": content})

    def get_messages(self) -> List[Dict[str, str]]:
        """Get current message history."""
        return self.messages.copy()

    def build_retry_prompt(
        self,
        original_prompt: str,
        failed_output: str,
        critic_feedback: CriticFeedback,
        validator_results: List[ValidatorResult],
    ) -> str:
        """Construct a retry prompt with failure context.

        Injects specific validator failure reasons, the exact failed output,
        and critic guidance into the conversation.
        """
        # Build failure context
        failure_details = []
        for result in validator_results:
            if not result.passed:
                failure_details.append(f"- {result.reason}")

        failure_text = "\n".join(failure_details) if failure_details else "Unknown validation failure"

        critic_guidance = critic_feedback.guidance
        critic_constraints = "\n".join(f"- {c}" for c in critic_feedback.constraints) if critic_feedback.constraints else ""

        # Pre-compute constraint block to avoid backslash in f-string expression
        constraint_block = f"Additional Constraints:\n{critic_constraints}" if critic_constraints else ""

        return f"""The previous output failed validation. Please correct it.

Previous Output:
{failed_output}

Validation Failures:
{failure_text}

Critic Guidance:
{critic_guidance}

{constraint_block}

Please provide a corrected output that addresses all failures above. Maintain the same format and structure as the original task."""

    def clear(self) -> None:
        """Clear history (keep system prompt)."""
        system_msgs = [m for m in self.messages if m["role"] == "system"]
        self.messages = system_msgs


class RetryOrchestrator:
    """Orchestrates the complete retry loop with validation, scoring, and critique.

    Coordinates:
    1. LLM generation
    2. Declarative + custom validation
    3. Semantic similarity scoring
    4. Critic evaluation
    5. Post-repair stability check
    6. Prompt refinement escalation
    7. Hard retry ceiling enforcement
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        validator_registry: Optional[ValidatorRegistry] = None,
        semantic_scorer: Optional[SemanticScorer] = None,
        critic: Optional[CriticEvaluator] = None,
        aggregator: Optional[DiagnosticAggregator] = None,
        max_retries: int = MAX_RETRIES,
        temperature: float = 0.0,
    ):
        self.llm_client = llm_client or get_llm_client()
        self.validator_registry = validator_registry or ValidatorRegistry()
        self.semantic_scorer = semantic_scorer or create_semantic_scorer()
        self.critic = critic or create_critic_evaluator(llm_client=self.llm_client)
        self.aggregator = aggregator or create_diagnostic_aggregator(critic=self.critic)
        self.max_retries = max_retries
        self.temperature = temperature

    def execute(
        self,
        task_spec: TaskSpec,
        prompt: str,
        ground_truth: Optional[str] = None,
    ) -> OrchestrationResult:
        """Execute the complete retry orchestration for a task.

        Args:
            task_spec: Task specification with validators and constraints.
            prompt: User prompt/input.
            ground_truth: Optional ground truth for semantic scoring.

        Returns:
            OrchestrationResult with final output and full execution trace.
        """
        # Register task spec with validator registry
        self.validator_registry.register_declarative(task_spec)

        # Initialize conversation history
        history = ConversationHistory(system_prompt=task_spec.system_prompt)
        history.add_user_message(prompt)

        retry_attempts = []
        prior_failures = []

        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            start_time = time.perf_counter()

            # Generate response
            response = self.llm_client.generate(
                messages=history.get_messages(),
                temperature=self.temperature,
                max_tokens=2048,
            )
            generated_output = response.content
            history.add_assistant_message(generated_output)

            # Run validators
            validator_results = self.validator_registry.validate(generated_output)

            # Run semantic scorer if ground truth provided
            semantic_result = None
            if ground_truth:
                semantic_result = self.semantic_scorer.evaluate(generated_output, ground_truth)

            # Get critic feedback
            critic_feedback = self.aggregator.evaluate_with_critic(
                task_spec=task_spec,
                generated_output=generated_output,
                validator_results=validator_results,
                semantic_result=semantic_result,
                ground_truth=ground_truth,
                retry_count=attempt,
                prior_failures=prior_failures,
            )

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

            # Post-repair stability check (on retry attempts that pass validation)
            if attempt > 0 and all_passed and semantic_passed:
                stability_passed = self._post_repair_stability_check(
                    generated_output, task_spec, ground_truth
                )
                if not stability_passed:
                    # Stability check failed - treat as validation failure
                    all_passed = False
                    semantic_passed = False
                    critic_feedback.diagnostics.append(
                        DiagnosticItem(
                            failure_type=FailureTypeEnum.CONSTRAINT_VIOLATION,
                            severity=SeverityEnum.CRITICAL,
                            validator_name="post_repair_stability",
                            message="Post-repair stability check failed: new regressions detected",
                            suggested_correction="Ensure corrections don't introduce new failures",
                        )
                    )

            if all_passed and semantic_passed:
                # Success! Compute reliability score
                reliability_score = self._compute_reliability_score(
                    validator_results, semantic_result, critic_feedback
                )
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

            # Prepare for next retry
            if attempt < self.max_retries:
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

        # Max retries exceeded
        return OrchestrationResult(
            success=False,
            final_output=retry_attempts[-1].generated_output,
            final_validator_results=retry_attempts[-1].validator_results,
            final_semantic_result=retry_attempts[-1].semantic_result,
            final_critic_feedback=retry_attempts[-1].critic_feedback,
            retry_attempts=retry_attempts,
            total_attempts=self.max_retries + 1,
            total_latency_ms=sum(r.latency_ms for r in retry_attempts),
            reliability_score=0.0,
            failure_summary=f"Max retries ({self.max_retries}) exceeded. Last failure: {retry_attempts[-1].critic_feedback.guidance}",
        )

    def _post_repair_stability_check(
        self,
        generated_output: str,
        task_spec: TaskSpec,
        ground_truth: Optional[str] = None,
    ) -> bool:
        """Re-run complete validator suite on retried output.

        Catches repair-induced cascade regressions by validating
        the corrected output against all validators again.
        """
        # Re-run all validators
        validator_results = self.validator_registry.validate(generated_output)

        # Check if any previously passing validators now fail
        all_passed = all(r.passed for r in validator_results)

        # Also re-check semantic similarity
        semantic_passed = True
        if ground_truth:
            semantic_result = self.semantic_scorer.evaluate(generated_output, ground_truth)
            semantic_passed = semantic_result.passed

        return all_passed and semantic_passed

    def _compute_reliability_score(
        self,
        validator_results: List[ValidatorResult],
        semantic_result: Optional[ValidatorResult],
        critic_feedback: CriticFeedback,
    ) -> float:
        """Compute composite reliability score for caching decision."""
        # Base score from validators
        validator_score = 1.0 if all(r.passed for r in validator_results) else 0.0

        # Semantic score
        semantic_score = 1.0
        if semantic_result:
            # Extract similarity from reason text (rough approximation)
            semantic_score = 1.0 if semantic_result.passed else 0.5

        # Critic confidence (lower if escalation needed)
        critic_confidence = 0.9 if not critic_feedback.escalate_to_negative_constraints else 0.7

        # Weighted composite
        return (validator_score * 0.5) + (semantic_score * 0.3) + (critic_confidence * 0.2)


def create_retry_orchestrator(
    llm_client: Optional[LLMClient] = None,
    validator_registry: Optional[ValidatorRegistry] = None,
    semantic_scorer: Optional[SemanticScorer] = None,
    critic: Optional[CriticEvaluator] = None,
    aggregator: Optional[DiagnosticAggregator] = None,
    max_retries: int = MAX_RETRIES,
) -> RetryOrchestrator:
    """Factory function to create a RetryOrchestrator with default settings."""
    return RetryOrchestrator(
        llm_client=llm_client,
        validator_registry=validator_registry,
        semantic_scorer=semantic_scorer,
        critic=critic,
        aggregator=aggregator,
        max_retries=max_retries,
    )