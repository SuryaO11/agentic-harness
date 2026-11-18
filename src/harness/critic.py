"""Structured Critic Evaluator for Agentic Harness v1.2.

Generates structured diagnostic feedback from validator failures and semantic drift,
producing actionable guidance for targeted retries without hallucination.
"""
import json
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ValidationError

from .client import get_llm_client, LLMClient
from .contracts import (
    CriticFeedback,
    DiagnosticItem,
    FailureTypeEnum,
    SeverityEnum,
    ValidatorResult,
    TaskSpec,
)


CRITIC_SYSTEM_PROMPT = """You are a structured diagnostic evaluator for LLM outputs.
Your task is to analyze validator failures and semantic drift, then produce structured feedback
that guides targeted correction. You MUST output ONLY valid JSON matching the CriticFeedback schema.

Constraints:
- failure_type must be one of: schema_syntax, regex_mismatch, keyword_violation, semantic_drift, constraint_violation, rate_limit, timeout, other
- severity must be one of: critical, warning, info
- guidance must be concise, actionable natural language
- constraints must be specific negative constraints (e.g., "do not include X", "must match pattern Y")
- escalate_to_negative_constraints: true only if 2+ consecutive failures of same type

Output format (JSON only):
{
  "passed": boolean,
  "diagnostics": [
    {
      "failure_type": "enum_value",
      "severity": "enum_value",
      "validator_name": "string",
      "message": "string",
      "suggested_correction": "string"
    }
  ],
  "guidance": "string",
  "constraints": ["string"],
  "escalate_to_negative_constraints": boolean
}"""


class CriticEvaluator:
    """Structured critic that transforms validation failures into actionable diagnostics."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.llm_client = llm_client or get_llm_client()
        self.model_name = model_name
        self.temperature = temperature

    def evaluate(
        self,
        task_spec: TaskSpec,
        generated_output: str,
        validator_results: List[ValidatorResult],
        semantic_result: Optional[ValidatorResult] = None,
        ground_truth: Optional[str] = None,
        retry_count: int = 0,
        prior_failures: Optional[List[Dict[str, Any]]] = None,
    ) -> CriticFeedback:
        """Generate structured critic feedback from validation results.

        Args:
            task_spec: The task specification with constraints.
            generated_output: The LLM-generated output that was validated.
            validator_results: Results from declarative and custom validators.
            semantic_result: Optional semantic similarity result.
            ground_truth: Optional reference/ground truth for semantic comparison.
            retry_count: Current retry attempt (0 = first try).
            prior_failures: History of prior failure diagnostics for escalation detection.

        Returns:
            Structured CriticFeedback with diagnostics, guidance, and constraints.
        """
        # Collect all failed validators
        failed_validators = [r for r in validator_results if not r.passed]
        diagnostics = []

        # Convert validator failures to diagnostics
        for result in failed_validators:
            diagnostics.append(DiagnosticItem(
                failure_type=result.failure_type or FailureTypeEnum.OTHER,
                severity=result.severity,
                validator_name="declarative",  # Could be enhanced to track specific validator
                message=result.reason,
                suggested_correction=self._generate_correction(result),
            ))

        # Add semantic drift if present
        if semantic_result and not semantic_result.passed:
            diagnostics.append(DiagnosticItem(
                failure_type=FailureTypeEnum.SEMANTIC_DRIFT,
                severity=semantic_result.severity,
                validator_name="semantic_scorer",
                message=semantic_result.reason,
                suggested_correction="Align output semantics with reference intent",
            ))

        # Determine if we should escalate
        escalate = self._should_escalate(diagnostics, prior_failures, retry_count)

        # Build guidance and constraints
        guidance = self._generate_guidance(diagnostics, task_spec, generated_output)
        constraints = self._generate_constraints(diagnostics, task_spec)

        # If no failures, return passing feedback
        if not diagnostics:
            return CriticFeedback(
                passed=True,
                diagnostics=[],
                guidance="All validators passed",
                constraints=[],
                escalate_to_negative_constraints=False,
            )

        # For complex cases, use LLM to refine feedback
        if len(diagnostics) > 1 or escalate:
            return self._llm_refine_feedback(
                task_spec, generated_output, diagnostics, guidance, constraints, escalate,
                ground_truth, retry_count
            )

        return CriticFeedback(
            passed=False,
            diagnostics=diagnostics,
            guidance=guidance,
            constraints=constraints,
            escalate_to_negative_constraints=escalate,
        )

    def _generate_correction(self, result: ValidatorResult) -> str:
        """Generate suggested correction from validator result."""
        if result.failure_type == FailureTypeEnum.SCHEMA_SYNTAX:
            return "Ensure output is valid JSON matching the required schema"
        elif result.failure_type == FailureTypeEnum.REGEX_MISMATCH:
            return "Match the required regex pattern"
        elif result.failure_type == FailureTypeEnum.KEYWORD_VIOLATION:
            if "blacklist" in result.reason.lower():
                return "Remove forbidden keywords from output"
            return "Include all required keywords"
        elif result.failure_type == FailureTypeEnum.CONSTRAINT_VIOLATION:
            return "Satisfy length or format constraints"
        elif result.failure_type == FailureTypeEnum.SEMANTIC_DRIFT:
            return "Align output meaning with reference/ground truth"
        return "Address the validation failure"

    def _should_escalate(
        self,
        diagnostics: List[DiagnosticItem],
        prior_failures: Optional[List[Dict[str, Any]]],
        retry_count: int,
    ) -> bool:
        """Check if we should escalate to negative constraints."""
        if retry_count < 2:
            return False

        if not prior_failures:
            return False

        # Check if same failure types persist across retries
        current_types = {d.failure_type.value for d in diagnostics}
        for prior in prior_failures[-2:]:  # Check last 2 attempts
            prior_types = {d.get("failure_type") for d in prior.get("diagnostics", [])}
            if current_types & prior_types:  # Intersection not empty
                return True

        return False

    def _generate_guidance(
        self,
        diagnostics: List[DiagnosticItem],
        task_spec: TaskSpec,
        generated_output: str,
    ) -> str:
        """Generate concise actionable guidance."""
        if not diagnostics:
            return "All validators passed"

        primary = diagnostics[0]
        if primary.failure_type == FailureTypeEnum.SCHEMA_SYNTAX:
            return f"Fix JSON syntax: {primary.message}"
        elif primary.failure_type == FailureTypeEnum.REGEX_MISMATCH:
            return f"Output must match pattern: {primary.message}"
        elif primary.failure_type == FailureTypeEnum.KEYWORD_VIOLATION:
            return f"Keyword constraint violated: {primary.message}"
        elif primary.failure_type == FailureTypeEnum.CONSTRAINT_VIOLATION:
            return f"Constraint violation: {primary.message}"
        elif primary.failure_type == FailureTypeEnum.SEMANTIC_DRIFT:
            return "Output meaning diverges from expected intent; realign content"
        return f"Fix: {primary.message}"

    def _generate_constraints(
        self,
        diagnostics: List[DiagnosticItem],
        task_spec: TaskSpec,
    ) -> List[str]:
        """Generate negative constraints from diagnostics."""
        constraints = []

        for diag in diagnostics:
            if diag.failure_type == FailureTypeEnum.KEYWORD_VIOLATION:
                if "blacklist" in diag.message.lower():
                    # Extract blacklisted keywords
                    import re
                    matches = re.findall(r"'([^']+)'", diag.message)
                    for kw in matches:
                        constraints.append(f"DO NOT include '{kw}'")
                else:
                    constraints.append(f"MUST include required keywords")
            elif diag.failure_type == FailureTypeEnum.REGEX_MISMATCH:
                # Extract pattern from message
                import re
                pattern_match = re.search(r"pattern:?\s*([^\s]+)", diag.message)
                if pattern_match:
                    constraints.append(f"Output MUST match regex: {pattern_match.group(1)}")
            elif diag.failure_type == FailureTypeEnum.CONSTRAINT_VIOLATION:
                if "minimum" in diag.message.lower():
                    constraints.append("Output MUST meet minimum length")
                elif "maximum" in diag.message.lower() or "exceeds" in diag.message.lower():
                    constraints.append("Output MUST NOT exceed maximum length")
            elif diag.failure_type == FailureTypeEnum.SCHEMA_SYNTAX:
                constraints.append("Output MUST be valid JSON per schema")
            elif diag.failure_type == FailureTypeEnum.SEMANTIC_DRIFT:
                constraints.append("Output semantics MUST align with reference intent")

        # Add task-level constraints
        if task_spec.keyword_blacklist:
            for kw in task_spec.keyword_blacklist:
                constraints.append(f"DO NOT use '{kw}'")
        if task_spec.regex_patterns:
            for pat in task_spec.regex_patterns:
                constraints.append(f"MUST match: {pat}")

        return list(dict.fromkeys(constraints))  # Deduplicate preserving order

    def _llm_refine_feedback(
        self,
        task_spec: TaskSpec,
        generated_output: str,
        diagnostics: List[DiagnosticItem],
        guidance: str,
        constraints: List[str],
        escalate: bool,
        ground_truth: Optional[str],
        retry_count: int,
    ) -> CriticFeedback:
        """Use LLM to refine critic feedback for complex cases."""
        # Build prompt for LLM
        prompt = self._build_refinement_prompt(
            task_spec, generated_output, diagnostics, guidance, constraints,
            escalate, ground_truth, retry_count
        )

        try:
            response = self.llm_client.generate(
                messages=[
                    {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                model=self.model_name,
                max_tokens=1024,
            )

            # Parse LLM response as CriticFeedback
            feedback_data = json.loads(response.content)
            return CriticFeedback(**feedback_data)

        except (json.JSONDecodeError, ValidationError, Exception):
            # Fallback to heuristic feedback if LLM fails
            return CriticFeedback(
                passed=False,
                diagnostics=diagnostics,
                guidance=guidance,
                constraints=constraints,
                escalate_to_negative_constraints=escalate,
            )

    def _build_refinement_prompt(
        self,
        task_spec: TaskSpec,
        generated_output: str,
        diagnostics: List[DiagnosticItem],
        guidance: str,
        constraints: List[str],
        escalate: bool,
        ground_truth: Optional[str],
        retry_count: int,
    ) -> str:
        """Build prompt for LLM-based feedback refinement."""
        diag_json = [d.model_dump() for d in diagnostics]

        return f"""Task: {task_spec.task_name}
Description: {task_spec.description}
System Prompt: {task_spec.system_prompt}

Generated Output:
{generated_output}

{f"Ground Truth/Reference: {ground_truth}" if ground_truth else ""}

Retry Attempt: {retry_count + 1}

Current Diagnostics:
{json.dumps(diag_json, indent=2)}

Current Guidance: {guidance}
Current Constraints: {json.dumps(constraints)}
Escalate: {str(escalate).lower()}

Refine the feedback to be more specific and actionable. Output ONLY valid JSON per the schema."""


def create_critic_evaluator(
    llm_client: Optional[LLMClient] = None,
    model_name: Optional[str] = None,
) -> CriticEvaluator:
    """Factory function to create a CriticEvaluator with default settings."""
    return CriticEvaluator(llm_client=llm_client, model_name=model_name)