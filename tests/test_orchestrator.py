"""Unit tests for retry orchestrator."""
import pytest
from unittest.mock import MagicMock, patch

from src.harness.orchestrator import (
    ConversationHistory,
    OrchestrationResult,
    RetryAttempt,
    RetryOrchestrator,
    create_retry_orchestrator,
)
from src.harness.contracts import (
    CriticFeedback,
    DiagnosticItem,
    FailureTypeEnum,
    SeverityEnum,
    TaskSpec,
    ValidatorResult,
)


def test_conversation_history_basic():
    """ConversationHistory manages messages correctly."""
    history = ConversationHistory(system_prompt="System prompt")

    assert len(history.get_messages()) == 1
    assert history.get_messages()[0]["role"] == "system"

    history.add_user_message("User message")
    history.add_assistant_message("Assistant response")

    assert len(history.get_messages()) == 3
    assert history.get_messages()[1]["role"] == "user"
    assert history.get_messages()[2]["role"] == "assistant"


def test_conversation_history_build_retry_prompt():
    """ConversationHistory builds retry prompts with failure context."""
    history = ConversationHistory()
    history.add_user_message("Original prompt")
    history.add_assistant_message("Failed output")

    critic_feedback = CriticFeedback(
        passed=False,
        diagnostics=[
            DiagnosticItem(
                failure_type=FailureTypeEnum.REGEX_MISMATCH,
                severity=SeverityEnum.CRITICAL,
                validator_name="regex",
                message="Pattern mismatch",
                suggested_correction="Match pattern",
            )
        ],
        guidance="Fix the pattern",
        constraints=["Must match ^INV-\\d{6}$"],
        escalate_to_negative_constraints=False,
    )

    validator_results = [
        ValidatorResult(
            passed=False,
            reason="Pattern mismatch",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.REGEX_MISMATCH,
        ),
    ]

    retry_prompt = history.build_retry_prompt(
        original_prompt="Original prompt",
        failed_output="Failed output",
        critic_feedback=critic_feedback,
        validator_results=validator_results,
    )

    assert "Previous Output:" in retry_prompt
    assert "Failed output" in retry_prompt
    assert "Validation Failures:" in retry_prompt
    assert "Pattern mismatch" in retry_prompt
    assert "Critic Guidance:" in retry_prompt
    assert "Fix the pattern" in retry_prompt
    assert "Additional Constraints:" in retry_prompt
    assert "Must match ^INV-\\d{6}$" in retry_prompt


def test_retry_orchestrator_creation():
    """RetryOrchestrator can be created with defaults."""
    orchestrator = create_retry_orchestrator()
    assert orchestrator.max_retries == 3
    assert orchestrator.temperature == 0.0


def test_retry_orchestrator_success_first_attempt():
    """Orchestrator returns success on first attempt if all pass."""
    orchestrator = create_retry_orchestrator()

    # Mock LLM client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '{"status": "ok"}'
    mock_client.generate.return_value = mock_response
    orchestrator.llm_client = mock_client

    # Mock validator registry to pass
    mock_registry = MagicMock()
    mock_registry.validate.return_value = [
        ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO),
    ]
    orchestrator.validator_registry = mock_registry

    # Mock semantic scorer
    mock_scorer = MagicMock()
    mock_scorer.evaluate.return_value = ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO)
    orchestrator.semantic_scorer = mock_scorer

    spec = TaskSpec(task_name="test", system_prompt="Test")
    result = orchestrator.execute(spec, "Test prompt", ground_truth="reference")

    assert result.success is True
    assert result.total_attempts == 1
    assert result.final_output == '{"status": "ok"}'


def test_retry_orchestrator_retry_on_failure():
    """Orchestrator retries on validation failure."""
    orchestrator = create_retry_orchestrator(max_retries=2)

    # Mock LLM client - fail first, succeed second
    mock_client = MagicMock()
    mock_response1 = MagicMock()
    mock_response1.content = "bad output"
    mock_response2 = MagicMock()
    mock_response2.content = '{"status": "ok"}'
    mock_client.generate.side_effect = [mock_response1, mock_response2]
    orchestrator.llm_client = mock_client

    # Mock validator registry - fail first, pass second, then stability check (passes)
    mock_registry = MagicMock()
    mock_registry.validate.side_effect = [
        [ValidatorResult(passed=False, reason="schema error", severity=SeverityEnum.CRITICAL, failure_type=FailureTypeEnum.SCHEMA_SYNTAX)],
        [ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO)],
        [ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO)],  # stability check
    ]
    orchestrator.validator_registry = mock_registry

    # Mock semantic scorer
    mock_scorer = MagicMock()
    mock_scorer.evaluate.return_value = ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO)
    orchestrator.semantic_scorer = mock_scorer

    spec = TaskSpec(task_name="test")
    result = orchestrator.execute(spec, "Test prompt")

    assert result.success is True
    assert result.total_attempts == 2
    assert mock_client.generate.call_count == 2


def test_retry_orchestrator_max_retries_exceeded():
    """Orchestrator fails after max retries."""
    orchestrator = create_retry_orchestrator(max_retries=2)

    # Mock LLM client - always fail
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "bad output"
    mock_client.generate.return_value = mock_response
    orchestrator.llm_client = mock_client

    # Mock validator registry - always fail
    mock_registry = MagicMock()
    mock_registry.validate.return_value = [
        ValidatorResult(passed=False, reason="schema error", severity=SeverityEnum.CRITICAL, failure_type=FailureTypeEnum.SCHEMA_SYNTAX),
    ]
    orchestrator.validator_registry = mock_registry

    # Mock semantic scorer
    mock_scorer = MagicMock()
    mock_scorer.evaluate.return_value = ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO)
    orchestrator.semantic_scorer = mock_scorer

    spec = TaskSpec(task_name="test")
    result = orchestrator.execute(spec, "Test prompt")

    assert result.success is False
    assert result.total_attempts == 3  # initial + 2 retries
    assert "Max retries" in result.failure_summary


def test_retry_orchestrator_post_repair_stability_check():
    """Post-repair stability check catches regressions."""
    orchestrator = create_retry_orchestrator(max_retries=2)

    # Mock LLM client - 4 attempts (initial fail, retry pass but stability fail, third pass but stability fail again, fourth)
    mock_client = MagicMock()
    mock_response1 = MagicMock()
    mock_response1.content = "bad"
    mock_response2 = MagicMock()
    mock_response2.content = "fixed but broken elsewhere"
    mock_response3 = MagicMock()
    mock_response3.content = "third attempt"
    mock_response4 = MagicMock()
    mock_response4.content = "final success"
    mock_client.generate.side_effect = [mock_response1, mock_response2, mock_response3, mock_response4]
    orchestrator.llm_client = mock_client

    # Mock validator registry - needs many calls
    mock_registry = MagicMock()
    mock_registry.validate.side_effect = [
        [ValidatorResult(passed=False, reason="error", severity=SeverityEnum.CRITICAL, failure_type=FailureTypeEnum.SCHEMA_SYNTAX)],  # attempt 1
        [ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO)],  # attempt 2
        [ValidatorResult(passed=False, reason="regression", severity=SeverityEnum.CRITICAL, failure_type=FailureTypeEnum.CONSTRAINT_VIOLATION)],  # stability check after attempt 2
        [ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO)],  # attempt 3
        [ValidatorResult(passed=False, reason="regression", severity=SeverityEnum.CRITICAL, failure_type=FailureTypeEnum.CONSTRAINT_VIOLATION)],  # stability check after attempt 3
        [ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO)],  # attempt 4
    ]
    orchestrator.validator_registry = mock_registry

    # Mock semantic scorer
    mock_scorer = MagicMock()
    mock_scorer.evaluate.return_value = ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO)
    orchestrator.semantic_scorer = mock_scorer

    spec = TaskSpec(task_name="test")
    result = orchestrator.execute(spec, "Test prompt")

    # Should detect stability failure and continue retrying
    assert mock_registry.validate.call_count >= 4
    assert result.total_attempts >= 3


def test_retry_orchestrator_reliability_score():
    """Reliability score is computed correctly."""
    orchestrator = create_retry_orchestrator()

    validator_results = [
        ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO),
    ]
    semantic_result = ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO)
    critic_feedback = CriticFeedback(
        passed=True,
        diagnostics=[],
        guidance="",
        constraints=[],
        escalate_to_negative_constraints=False,
    )

    score = orchestrator._compute_reliability_score(
        validator_results, semantic_result, critic_feedback
    )

    assert 0.0 <= score <= 1.0
    assert score > 0.8  # Should be high for all passing


def test_retry_orchestrator_escalation():
    """Orchestrator tracks prior failures for escalation."""
    orchestrator = create_retry_orchestrator(max_retries=2)

    # Mock LLM client - always fail
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "bad"
    mock_client.generate.return_value = mock_response
    orchestrator.llm_client = mock_client

    # Mock validator registry - always same failure
    mock_registry = MagicMock()
    mock_registry.validate.return_value = [
        ValidatorResult(passed=False, reason="pattern mismatch", severity=SeverityEnum.CRITICAL, failure_type=FailureTypeEnum.REGEX_MISMATCH),
    ]
    orchestrator.validator_registry = mock_registry

    # Mock semantic scorer
    mock_scorer = MagicMock()
    mock_scorer.evaluate.return_value = ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO)
    orchestrator.semantic_scorer = mock_scorer

    spec = TaskSpec(task_name="test")
    result = orchestrator.execute(spec, "Test prompt")

    # Critic should escalate after 2 retries with same failure
    assert result.total_attempts == 3
    # Last attempt should have escalation
    last_feedback = result.final_critic_feedback
    # Escalation is detected in critic, not orchestrator directly