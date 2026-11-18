"""Unit tests for prompt escalation engine."""
import pytest

from src.harness.contracts import (
    FailureTypeEnum,
    SeverityEnum,
    TaskSpec,
    ValidatorResult,
)
from src.harness.escalation import (
    PromptEscalationEngine,
    PromptRefiner,
    create_prompt_escalation_engine,
    create_prompt_refiner,
)


def test_prompt_escalation_engine_creation():
    """PromptEscalationEngine can be created."""
    engine = create_prompt_escalation_engine()
    assert engine is not None


def test_record_failure():
    """Engine records failure types."""
    engine = create_prompt_escalation_engine()

    results = [
        ValidatorResult(
            passed=False,
            reason="regex mismatch",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.REGEX_MISMATCH,
        ),
    ]

    engine.record_failure(results)
    assert len(engine.failure_history) == 1
    assert FailureTypeEnum.REGEX_MISMATCH in engine.failure_history[0]["failure_types"]


def test_should_escalate_false_insufficient_history():
    """No escalation with insufficient history."""
    engine = create_prompt_escalation_engine()

    results = [
        ValidatorResult(
            passed=False,
            reason="error",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.REGEX_MISMATCH,
        ),
    ]
    engine.record_failure(results)

    assert engine.should_escalate(min_consecutive=2) is False


def test_should_escalate_true_persistent_failure():
    """Escalation triggered after 2 consecutive same failures."""
    engine = create_prompt_escalation_engine()

    results = [
        ValidatorResult(
            passed=False,
            reason="regex mismatch",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.REGEX_MISMATCH,
        ),
    ]

    # First failure
    engine.record_failure(results)
    # Second failure (same type)
    engine.record_failure(results)

    assert engine.should_escalate(min_consecutive=2) is True


def test_should_escalate_false_different_failures():
    """No escalation if failures are different types."""
    engine = create_prompt_escalation_engine()

    results1 = [
        ValidatorResult(
            passed=False,
            reason="regex mismatch",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.REGEX_MISMATCH,
        ),
    ]
    results2 = [
        ValidatorResult(
            passed=False,
            reason="keyword violation",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.KEYWORD_VIOLATION,
        ),
    ]

    engine.record_failure(results1)
    engine.record_failure(results2)

    assert engine.should_escalate(min_consecutive=2) is False


def test_get_persistent_failures():
    """Get persistent failure types."""
    engine = create_prompt_escalation_engine()

    results = [
        ValidatorResult(
            passed=False,
            reason="regex mismatch",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.REGEX_MISMATCH,
        ),
        ValidatorResult(
            passed=False,
            reason="keyword violation",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.KEYWORD_VIOLATION,
        ),
    ]

    engine.record_failure(results)
    engine.record_failure(results)  # Same failures again

    persistent = engine.get_persistent_failures(min_consecutive=2)
    assert FailureTypeEnum.REGEX_MISMATCH in persistent
    assert FailureTypeEnum.KEYWORD_VIOLATION in persistent


def test_generate_escalated_prompt_regex():
    """Escalated prompt includes regex constraints."""
    engine = create_prompt_escalation_engine()

    task_spec = TaskSpec(
        task_name="test",
        regex_patterns=[r"^INV-\d{6}$"],
    )

    persistent = {FailureTypeEnum.REGEX_MISMATCH}
    escalated = engine.generate_escalated_prompt(
        original_prompt="Generate invoice",
        task_spec=task_spec,
        persistent_failures=persistent,
        critic_constraints=[],
    )

    assert "ESCALATED CONSTRAINTS" in escalated
    assert "DO NOT violate pattern" in escalated
    assert "^INV-\\d{6}$" in escalated


def test_generate_escalated_prompt_keywords():
    """Escalated prompt includes keyword constraints."""
    engine = create_prompt_escalation_engine()

    task_spec = TaskSpec(
        task_name="test",
        keyword_blacklist=["password", "secret"],
        keyword_whitelist=["total"],
    )

    persistent = {FailureTypeEnum.KEYWORD_VIOLATION}
    escalated = engine.generate_escalated_prompt(
        original_prompt="Generate output",
        task_spec=task_spec,
        persistent_failures=persistent,
        critic_constraints=[],
    )

    assert "DO NOT use forbidden keyword: 'password'" in escalated
    assert "DO NOT use forbidden keyword: 'secret'" in escalated
    assert "MUST include required keyword: 'total'" in escalated


def test_generate_escalated_prompt_length():
    """Escalated prompt includes length constraints."""
    engine = create_prompt_escalation_engine()

    task_spec = TaskSpec(
        task_name="test",
        min_length=10,
        max_length=100,
    )

    persistent = {FailureTypeEnum.CONSTRAINT_VIOLATION}
    escalated = engine.generate_escalated_prompt(
        original_prompt="Generate output",
        task_spec=task_spec,
        persistent_failures=persistent,
        critic_constraints=[],
    )

    assert "DO NOT write less than 10 characters" in escalated
    assert "DO NOT exceed 100 characters" in escalated


def test_generate_escalated_prompt_schema():
    """Escalated prompt includes schema constraint."""
    engine = create_prompt_escalation_engine()

    task_spec = TaskSpec(task_name="test")

    persistent = {FailureTypeEnum.SCHEMA_SYNTAX}
    escalated = engine.generate_escalated_prompt(
        original_prompt="Generate JSON",
        task_spec=task_spec,
        persistent_failures=persistent,
        critic_constraints=[],
    )

    assert "DO NOT produce invalid JSON" in escalated


def test_generate_escalated_prompt_semantic_drift():
    """Escalated prompt includes semantic drift constraint."""
    engine = create_prompt_escalation_engine()

    task_spec = TaskSpec(task_name="test")

    persistent = {FailureTypeEnum.SEMANTIC_DRIFT}
    escalated = engine.generate_escalated_prompt(
        original_prompt="Generate output",
        task_spec=task_spec,
        persistent_failures=persistent,
        critic_constraints=[],
    )

    assert "DO NOT deviate from the reference intent" in escalated


def test_generate_escalated_prompt_includes_critic_constraints():
    """Escalated prompt includes critic-generated constraints."""
    engine = create_prompt_escalation_engine()

    task_spec = TaskSpec(task_name="test")

    persistent = {FailureTypeEnum.REGEX_MISMATCH}
    critic_constraints = ["Must match ^INV-\\d{6}$", "No special characters"]

    escalated = engine.generate_escalated_prompt(
        original_prompt="Generate output",
        task_spec=task_spec,
        persistent_failures=persistent,
        critic_constraints=critic_constraints,
    )

    assert "Must match ^INV-\\d{6}$" in escalated
    assert "No special characters" in escalated


def test_clear_history():
    """Clear failure history."""
    engine = create_prompt_escalation_engine()

    results = [
        ValidatorResult(
            passed=False,
            reason="error",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.REGEX_MISMATCH,
        ),
    ]

    engine.record_failure(results)
    engine.record_failure(results)
    assert engine.should_escalate()

    engine.clear_history()
    assert len(engine.failure_history) == 0
    assert engine.should_escalate() is False


def test_prompt_refiner_refine_no_escalation():
    """Refiner returns original prompt when no escalation needed."""
    refiner = create_prompt_refiner()

    task_spec = TaskSpec(task_name="test")
    results = [
        ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO),
    ]

    refined = refiner.refine(
        prompt="Original prompt",
        task_spec=task_spec,
        validator_results=results,
    )

    assert refined == "Original prompt"


def test_prompt_refiner_refine_escalation():
    """Refiner returns escalated prompt when escalation triggered."""
    refiner = create_prompt_refiner()

    task_spec = TaskSpec(
        task_name="test",
        regex_patterns=[r"^INV-\d{6}$"],
    )
    results = [
        ValidatorResult(
            passed=False,
            reason="regex mismatch",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.REGEX_MISMATCH,
        ),
    ]

    # First failure
    refined1 = refiner.refine(
        prompt="Generate invoice",
        task_spec=task_spec,
        validator_results=results,
    )
    assert refined1 == "Generate invoice"  # No escalation yet

    # Second failure (same type) - should escalate
    refined2 = refiner.refine(
        prompt="Generate invoice",
        task_spec=task_spec,
        validator_results=results,
    )

    assert "ESCALATED CONSTRAINTS" in refined2
    assert "DO NOT violate pattern" in refined2


def test_prompt_refiner_reset():
    """Refiner reset clears escalation state."""
    refiner = create_prompt_refiner()

    task_spec = TaskSpec(task_name="test")
    results = [
        ValidatorResult(
            passed=False,
            reason="regex mismatch",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.REGEX_MISMATCH,
        ),
    ]

    # Two failures to trigger escalation
    refiner.refine("prompt", task_spec, results)
    refiner.refine("prompt", task_spec, results)

    # Reset
    refiner.reset()

    # Should not escalate anymore
    refined = refiner.refine("prompt", task_spec, results)
    assert refined == "prompt"