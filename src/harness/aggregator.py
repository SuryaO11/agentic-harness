"""Diagnostic aggregator for Agentic Harness v1.2.

Links validator results to actionable guidance for the critic and retry orchestrator.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .contracts import (
    DiagnosticItem,
    FailureTypeEnum,
    SeverityEnum,
    ValidatorResult,
    TaskSpec,
)
from .critic import CriticEvaluator, create_critic_evaluator


@dataclass
class AggregatedDiagnostics:
    """Aggregated diagnostic information for a validation run."""
    validator_results: List[ValidatorResult]
    semantic_result: Optional[ValidatorResult] = None
    failed_validators: List[ValidatorResult] = field(default_factory=list)
    failure_types: List[FailureTypeEnum] = field(default_factory=list)
    max_severity: SeverityEnum = SeverityEnum.INFO
    passed: bool = True

    def __post_init__(self):
        self.failed_validators = [r for r in self.validator_results if not r.passed]
        self.failure_types = [r.failure_type for r in self.failed_validators if r.failure_type]
        if self.failed_validators:
            self.passed = False
            severities = [r.severity for r in self.failed_validators]
            if SeverityEnum.CRITICAL in severities:
                self.max_severity = SeverityEnum.CRITICAL
            elif SeverityEnum.WARNING in severities:
                self.max_severity = SeverityEnum.WARNING
        if self.semantic_result and not self.semantic_result.passed:
            self.passed = False
            if self.semantic_result.failure_type:
                self.failure_types.append(self.semantic_result.failure_type)
            if self.semantic_result.severity == SeverityEnum.CRITICAL:
                self.max_severity = SeverityEnum.CRITICAL


class DiagnosticAggregator:
    """Aggregates validator results and prepares structured input for critic evaluation.

    This class bridges the validator engine and the critic, providing a unified
    view of all failures with context for actionable guidance generation.
    """

    def __init__(self, critic: Optional[CriticEvaluator] = None):
        self.critic = critic or create_critic_evaluator()

    def aggregate(
        self,
        task_spec: TaskSpec,
        generated_output: str,
        validator_results: List[ValidatorResult],
        semantic_result: Optional[ValidatorResult] = None,
        ground_truth: Optional[str] = None,
        retry_count: int = 0,
        prior_failures: Optional[List[Dict[str, Any]]] = None,
    ) -> AggregatedDiagnostics:
        """Aggregate all diagnostic information from a validation run.

        Args:
            task_spec: The task specification.
            generated_output: The LLM-generated output.
            validator_results: Results from all validators (declarative + custom).
            semantic_result: Optional semantic similarity result.
            ground_truth: Optional reference for semantic comparison.
            retry_count: Current retry attempt number.
            prior_failures: History of prior aggregated failures.

        Returns:
            AggregatedDiagnostics with all failure information.
        """
        return AggregatedDiagnostics(
            validator_results=validator_results,
            semantic_result=semantic_result,
        )

    def evaluate_with_critic(
        self,
        task_spec: TaskSpec,
        generated_output: str,
        validator_results: List[ValidatorResult],
        semantic_result: Optional[ValidatorResult] = None,
        ground_truth: Optional[str] = None,
        retry_count: int = 0,
        prior_failures: Optional[List[Dict[str, Any]]] = None,
    ):
        """Run full diagnostic aggregation and critic evaluation.

        This is the main entry point for getting structured critic feedback
        from a complete validation run.

        Returns:
            CriticFeedback with diagnostics, guidance, and constraints.
        """
        aggregated = self.aggregate(
            task_spec=task_spec,
            generated_output=generated_output,
            validator_results=validator_results,
            semantic_result=semantic_result,
            ground_truth=ground_truth,
            retry_count=retry_count,
            prior_failures=prior_failures,
        )

        # Convert prior_failures to format expected by critic
        critic_prior_failures = None
        if prior_failures:
            critic_prior_failures = [
                {"diagnostics": [
                    {"failure_type": d.failure_type.value if hasattr(d, 'failure_type') else d.get("failure_type")}
                    for d in pf.get("diagnostics", [])
                ]}
                for pf in prior_failures
            ]

        return self.critic.evaluate(
            task_spec=task_spec,
            generated_output=generated_output,
            validator_results=validator_results,
            semantic_result=semantic_result,
            ground_truth=ground_truth,
            retry_count=retry_count,
            prior_failures=critic_prior_failures,
        )

    def get_failure_summary(self, aggregated: AggregatedDiagnostics) -> Dict[str, Any]:
        """Generate a summary of failures for logging/debugging."""
        return {
            "total_validators": len(aggregated.validator_results),
            "failed_count": len(aggregated.failed_validators),
            "failure_types": [ft.value for ft in aggregated.failure_types],
            "max_severity": aggregated.max_severity.value,
            "passed": aggregated.passed,
            "has_semantic_drift": FailureTypeEnum.SEMANTIC_DRIFT in aggregated.failure_types,
        }


def create_diagnostic_aggregator(critic: Optional[CriticEvaluator] = None) -> DiagnosticAggregator:
    """Factory function to create a DiagnosticAggregator with default settings."""
    return DiagnosticAggregator(critic=critic)