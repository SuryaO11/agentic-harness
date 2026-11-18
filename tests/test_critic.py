"""Unit tests for structured critic evaluator."""
import pytest
from unittest.mock import MagicMock, patch

from src.harness.contracts import (
    CriticFeedback,
    DiagnosticItem,
    FailureTypeEnum,
    SeverityEnum,
    TaskSpec,
    ValidatorResult,
)
from src.harness.critic import CriticEvaluator, create_critic_evaluator


def test_critic_evaluator_creation():
    """CriticEvaluator can be created with defaults."""
    critic = create_critic_evaluator()
    assert critic.temperature == 0.0


def test_critic_evaluator_all_pass():
    """Critic returns passing feedback when all validators pass."""
    critic = create_critic_evaluator()

    spec = TaskSpec(task_name="test")
    validator_results = [
        ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO),
    ]

    result = critic.evaluate(
        task_spec=spec,
        generated_output="valid output",
        validator_results=validator_results,
    )

    assert result.passed is True
    assert result.diagnostics == []
    assert result.guidance == "All validators passed"


def test_critic_evaluator_schema_failure():
    """Critic generates diagnostics for schema failure."""
    critic = create_critic_evaluator()

    spec = TaskSpec(task_name="json_parser")
    validator_results = [
        ValidatorResult(
            passed=False,
            reason="JSON schema validation failed: missing field 'id'",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.SCHEMA_SYNTAX,
        ),
    ]

    result = critic.evaluate(
        task_spec=spec,
        generated_output='{"name": "test"}',
        validator_results=validator_results,
    )

    assert result.passed is False
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].failure_type == FailureTypeEnum.SCHEMA_SYNTAX
    assert "JSON" in result.guidance


def test_critic_evaluator_regex_failure():
    """Critic generates diagnostics for regex failure."""
    critic = create_critic_evaluator()

    spec = TaskSpec(task_name="id_parser", regex_patterns=[r"^INV-\d{6}$"])
    validator_results = [
        ValidatorResult(
            passed=False,
            reason="Output does not match required pattern: ^INV-\\d{6}$",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.REGEX_MISMATCH,
        ),
    ]

    result = critic.evaluate(
        task_spec=spec,
        generated_output="INV-123",
        validator_results=validator_results,
    )

    assert result.passed is False
    assert result.diagnostics[0].failure_type == FailureTypeEnum.REGEX_MISMATCH
    assert "regex" in result.constraints[0].lower()


def test_critic_evaluator_keyword_blacklist():
    """Critic generates diagnostics for keyword blacklist violation."""
    critic = create_critic_evaluator()

    spec = TaskSpec(task_name="safe", keyword_blacklist=["password", "secret"])
    validator_results = [
        ValidatorResult(
            passed=False,
            reason="Blacklisted keywords found: password",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.KEYWORD_VIOLATION,
        ),
    ]

    result = critic.evaluate(
        task_spec=spec,
        generated_output="The password is 123",
        validator_results=validator_results,
    )

    assert result.passed is False
    assert any("password" in c.lower() for c in result.constraints)


def test_critic_evaluator_semantic_drift():
    """Critic includes semantic drift in diagnostics."""
    critic = create_critic_evaluator()

    spec = TaskSpec(task_name="semantic_test")
    validator_results = [
        ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO),
    ]
    semantic_result = ValidatorResult(
        passed=False,
        reason="Semantic similarity 0.35 below threshold 0.75",
        severity=SeverityEnum.CRITICAL,
        failure_type=FailureTypeEnum.SEMANTIC_DRIFT,
    )

    result = critic.evaluate(
        task_spec=spec,
        generated_output="The stock market crashed",
        validator_results=validator_results,
        semantic_result=semantic_result,
        ground_truth="The cat sat on the mat",
    )

    assert result.passed is False
    assert any(d.failure_type == FailureTypeEnum.SEMANTIC_DRIFT for d in result.diagnostics)


def test_critic_evaluator_multiple_failures():
    """Critic handles multiple validator failures."""
    critic = create_critic_evaluator()

    spec = TaskSpec(task_name="multi")
    validator_results = [
        ValidatorResult(
            passed=False,
            reason="JSON schema validation failed",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.SCHEMA_SYNTAX,
        ),
        ValidatorResult(
            passed=False,
            reason="Output does not match required pattern",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.REGEX_MISMATCH,
        ),
    ]

    result = critic.evaluate(
        task_spec=spec,
        generated_output="invalid",
        validator_results=validator_results,
    )

    assert result.passed is False
    assert len(result.diagnostics) == 2


def test_critic_evaluator_escalation_detection():
    """Critic detects escalation after 2 retries with same failure."""
    critic = create_critic_evaluator()

    spec = TaskSpec(task_name="escalate")
    validator_results = [
        ValidatorResult(
            passed=False,
            reason="Pattern mismatch",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.REGEX_MISMATCH,
        ),
    ]
    prior_failures = [
        {"diagnostics": [{"failure_type": "regex_mismatch"}]},
        {"diagnostics": [{"failure_type": "regex_mismatch"}]},
    ]

    result = critic.evaluate(
        task_spec=spec,
        generated_output="bad",
        validator_results=validator_results,
        retry_count=2,  # Third attempt (0-indexed)
        prior_failures=prior_failures,
    )

    assert result.escalate_to_negative_constraints is True


def test_critic_evaluator_no_escalation_first_retry():
    """No escalation on first retry."""
    critic = create_critic_evaluator()

    spec = TaskSpec(task_name="no_escalate")
    validator_results = [
        ValidatorResult(
            passed=False,
            reason="Pattern mismatch",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.REGEX_MISMATCH,
        ),
    ]
    prior_failures = [
        {"diagnostics": [{"failure_type": "regex_mismatch"}]},
    ]

    result = critic.evaluate(
        task_spec=spec,
        generated_output="bad",
        validator_results=validator_results,
        retry_count=0,  # First retry
        prior_failures=prior_failures,
    )

    assert result.escalate_to_negative_constraints is False


def test_critic_evaluator_constraint_generation():
    """Critic generates negative constraints from task spec."""
    critic = create_critic_evaluator()

    spec = TaskSpec(
        task_name="constrained",
        keyword_blacklist=["secret"],
        regex_patterns=[r"^ID-\d+$"],
        min_length=10,
        max_length=100,
    )
    validator_results = [
        ValidatorResult(
            passed=False,
            reason="Output length 5 below minimum 10",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.CONSTRAINT_VIOLATION,
        ),
    ]

    result = critic.evaluate(
        task_spec=spec,
        generated_output="short",
        validator_results=validator_results,
    )

    assert any("minimum" in c.lower() for c in result.constraints)
    assert any("secret" in c.lower() for c in result.constraints)
    assert any("ID-" in c for c in result.constraints)


def test_critic_evaluator_llm_refinement_mock():
    """Critic uses LLM refinement for complex cases."""
    critic = create_critic_evaluator()

    spec = TaskSpec(task_name="complex")
    validator_results = [
        ValidatorResult(
            passed=False,
            reason="Schema error",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.SCHEMA_SYNTAX,
        ),
        ValidatorResult(
            passed=False,
            reason="Regex mismatch",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.REGEX_MISMATCH,
        ),
    ]

    # Mock LLM client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps({
        "passed": False,
        "diagnostics": [
            {"failure_type": "schema_syntax", "severity": "critical", "validator_name": "schema", "message": "Invalid JSON"},
            {"failure_type": "regex_mismatch", "severity": "critical", "validator_name": "regex", "message": "Pattern mismatch"}
        ],
        "guidance": "Fix JSON and match pattern",
        "constraints": ["Must be valid JSON", "Must match pattern"],
        "escalate_to_negative_constraints": False
    })
    mock_client.generate.return_value = mock_response

    critic.llm_client = mock_client

    result = critic.evaluate(
        task_spec=spec,
        generated_output="bad output",
        validator_results=validator_results,
    )

    assert result.passed is False
    mock_client.generate.assert_called_once()


import json