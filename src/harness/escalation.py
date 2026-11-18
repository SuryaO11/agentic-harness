"""Prompt Escalation Engine for Agentic Harness v1.2.

Automatically refines prompts with negative constraints when validation fails
across consecutive retries, preventing infinite retry loops.
"""
from typing import Any, Dict, List, Optional, Set

from .contracts import (
    FailureTypeEnum,
    SeverityEnum,
    TaskSpec,
    ValidatorResult,
)


class PromptEscalationEngine:
    """Generates refined prompts with negative constraints for persistent failures.

    When the same failure types persist across 2+ consecutive retries,
    this engine creates a new prompt that explicitly forbids the problematic
    patterns, guiding the model away from repeated mistakes.
    """

    def __init__(self):
        self.failure_history: List[Dict[str, Any]] = []

    def record_failure(
        self,
        validator_results: List[ValidatorResult],
        semantic_result: Optional[ValidatorResult] = None,
    ) -> None:
        """Record a failure for escalation tracking."""
        failure_types = set()
        for r in validator_results:
            if not r.passed and r.failure_type:
                failure_types.add(r.failure_type)
        if semantic_result and not semantic_result.passed and semantic_result.failure_type:
            failure_types.add(semantic_result.failure_type)

        self.failure_history.append({
            "failure_types": failure_types,
            "timestamp": len(self.failure_history),
        })

    def should_escalate(self, min_consecutive: int = 2) -> bool:
        """Check if escalation is needed based on consecutive similar failures."""
        if len(self.failure_history) < min_consecutive:
            return False

        # Check last N failures for overlapping failure types
        recent = self.failure_history[-min_consecutive:]
        common_types = set(recent[0]["failure_types"])
        for entry in recent[1:]:
            common_types &= entry["failure_types"]

        return len(common_types) > 0

    def get_persistent_failures(self, min_consecutive: int = 2) -> Set[FailureTypeEnum]:
        """Get the failure types that have persisted across consecutive retries."""
        if len(self.failure_history) < min_consecutive:
            return set()

        recent = self.failure_history[-min_consecutive:]
        common_types = set(recent[0]["failure_types"])
        for entry in recent[1:]:
            common_types &= entry["failure_types"]

        return common_types

    def generate_escalated_prompt(
        self,
        original_prompt: str,
        task_spec: TaskSpec,
        persistent_failures: Set[FailureTypeEnum],
        critic_constraints: List[str],
    ) -> str:
        """Generate an escalated prompt with negative constraints.

        Adds explicit "DO NOT" instructions for persistent failure patterns.
        """
        escalation_lines = [
            "\n=== ESCALATED CONSTRAINTS (Previous attempts failed) ===",
            "The following patterns have caused repeated failures. You MUST avoid them:",
        ]

        # Add negative constraints for each persistent failure type
        for failure_type in persistent_failures:
            if failure_type == FailureTypeEnum.REGEX_MISMATCH:
                for pattern in task_spec.regex_patterns:
                    escalation_lines.append(f"  - DO NOT violate pattern: {pattern}")
            elif failure_type == FailureTypeEnum.KEYWORD_VIOLATION:
                for kw in task_spec.keyword_blacklist:
                    escalation_lines.append(f"  - DO NOT use forbidden keyword: '{kw}'")
                for kw in task_spec.keyword_whitelist:
                    escalation_lines.append(f"  - MUST include required keyword: '{kw}'")
            elif failure_type == FailureTypeEnum.CONSTRAINT_VIOLATION:
                if task_spec.min_length is not None:
                    escalation_lines.append(f"  - DO NOT write less than {task_spec.min_length} characters")
                if task_spec.max_length is not None:
                    escalation_lines.append(f"  - DO NOT exceed {task_spec.max_length} characters")
            elif failure_type == FailureTypeEnum.SCHEMA_SYNTAX:
                escalation_lines.append("  - DO NOT produce invalid JSON. Output MUST be valid JSON per schema.")
            elif failure_type == FailureTypeEnum.SEMANTIC_DRIFT:
                escalation_lines.append("  - DO NOT deviate from the reference intent. Stay semantically aligned.")

        # Add critic-generated constraints
        for constraint in critic_constraints:
            if constraint not in escalation_lines:
                escalation_lines.append(f"  - {constraint}")

        escalation_lines.append("=== END ESCALATED CONSTRAINTS ===\n")

        return original_prompt + "\n" + "\n".join(escalation_lines)

    def clear_history(self) -> None:
        """Clear failure history (e.g., after successful attempt)."""
        self.failure_history.clear()


class PromptRefiner:
    """Refines prompts iteratively based on failure patterns."""

    def __init__(self, escalation_engine: Optional[PromptEscalationEngine] = None):
        self.escalation_engine = escalation_engine or PromptEscalationEngine()

    def refine(
        self,
        prompt: str,
        task_spec: TaskSpec,
        validator_results: List[ValidatorResult],
        semantic_result: Optional[ValidatorResult] = None,
        critic_constraints: Optional[List[str]] = None,
    ) -> str:
        """Refine prompt based on validation results.

        Returns original prompt if no escalation needed, otherwise
        returns escalated prompt with negative constraints.
        """
        critic_constraints = critic_constraints or []

        # Record this failure
        self.escalation_engine.record_failure(validator_results, semantic_result)

        # Check if we should escalate
        if self.escalation_engine.should_escalate():
            persistent = self.escalation_engine.get_persistent_failures()
            return self.escalation_engine.generate_escalated_prompt(
                prompt, task_spec, persistent, critic_constraints
            )

        return prompt

    def reset(self) -> None:
        """Reset the refiner state."""
        self.escalation_engine.clear_history()


def create_prompt_escalation_engine() -> PromptEscalationEngine:
    """Factory function to create a PromptEscalationEngine."""
    return PromptEscalationEngine()


def create_prompt_refiner() -> PromptRefiner:
    """Factory function to create a PromptRefiner."""
    return PromptRefiner()