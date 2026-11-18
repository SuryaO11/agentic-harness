"""Unit tests for diagnostic aggregator."""
import pytest

from src.harness.aggregator import (
    AggregatedDiagnostics,
    DiagnosticAggregator,
    create_diagnostic_aggregator,
)
from src.harness.contracts import (
    FailureTypeEnum,
    SeverityEnum,
    TaskSpec,
    ValidatorResult,
)


def test_aggregated_diagnostics_all_pass():
    """AggregatedDiagnostics shows passed when all validators pass."""
    results = [
        ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO),
        ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO),
    ]
    agg = AggregatedDiagnostics(validator_results=results)

    assert agg.passed is True
    assert agg.failed_validators == []
    assert agg.failure_types == []
    assert agg.max_severity == SeverityEnum.INFO


def test_aggregated_diagnostics_some_fail():
    """AggregatedDiagnostics tracks failures correctly."""
    results = [
        ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO),
        ValidatorResult(
            passed=False,
            reason="regex mismatch",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.REGEX_MISMATCH,
        ),
        ValidatorResult(
            passed=False,
            reason="keyword violation",
            severity=SeverityEnum.WARNING,
            failure_type=FailureTypeEnum.KEYWORD_VIOLATION,
        ),
    ]
    agg = AggregatedDiagnostics(validator_results=results)

    assert agg.passed is False
    assert len(agg.failed_validators) == 2
    assert FailureTypeEnum.REGEX_MISMATCH in agg.failure_types
    assert FailureTypeEnum.KEYWORD_VIOLATION in agg.failure_types
    assert agg.max_severity == SeverityEnum.CRITICAL


def test_aggregated_diagnostics_with_semantic_drift():
    """AggregatedDiagnostics includes semantic drift."""
    results = [
        ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO),
    ]
    semantic = ValidatorResult(
        passed=False,
        reason="semantic drift detected",
        severity=SeverityEnum.CRITICAL,
        failure_type=FailureTypeEnum.SEMANTIC_DRIFT,
    )
    agg = AggregatedDiagnostics(validator_results=results, semantic_result=semantic)

    assert agg.passed is False
    assert FailureTypeEnum.SEMANTIC_DRIFT in agg.failure_types
    assert agg.max_severity == SeverityEnum.CRITICAL


def test_diagnostic_aggregator_creation():
    """DiagnosticAggregator can be created."""
    agg = create_diagnostic_aggregator()
    assert agg is not None


def test_diagnostic_aggregator_aggregate():
    """DiagnosticAggregator.aggregate returns AggregatedDiagnostics."""
    aggregator = create_diagnostic_aggregator()

    spec = TaskSpec(task_name="test")
    results = [
        ValidatorResult(
            passed=False,
            reason="schema error",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.SCHEMA_SYNTAX,
        ),
    ]

    agg = aggregator.aggregate(
        task_spec=spec,
        generated_output="bad output",
        validator_results=results,
    )

    assert isinstance(agg, AggregatedDiagnostics)
    assert agg.passed is False
    assert len(agg.failed_validators) == 1


def test_diagnostic_aggregator_failure_summary():
    """DiagnosticAggregator.get_failure_summary returns structured summary."""
    aggregator = create_diagnostic_aggregator()

    results = [
        ValidatorResult(
            passed=False,
            reason="schema error",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.SCHEMA_SYNTAX,
        ),
        ValidatorResult(
            passed=False,
            reason="regex mismatch",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.REGEX_MISMATCH,
        ),
    ]
    agg = AggregatedDiagnostics(validator_results=results)

    summary = aggregator.get_failure_summary(agg)

    assert summary["total_validators"] == 2
    assert summary["failed_count"] == 2
    assert "schema_syntax" in summary["failure_types"]
    assert "regex_mismatch" in summary["failure_types"]
    assert summary["max_severity"] == "critical"
    assert summary["passed"] is False
    assert summary["has_semantic_drift"] is False


def test_diagnostic_aggregator_failure_summary_with_semantic():
    """Failure summary includes semantic drift flag."""
    aggregator = create_diagnostic_aggregator()

    results = [
        ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO),
    ]
    semantic = ValidatorResult(
        passed=False,
        reason="drift",
        severity=SeverityEnum.CRITICAL,
        failure_type=FailureTypeEnum.SEMANTIC_DRIFT,
    )
    agg = AggregatedDiagnostics(validator_results=results, semantic_result=semantic)

    summary = aggregator.get_failure_summary(agg)

    assert summary["has_semantic_drift"] is True


def test_diagnostic_aggregator_evaluate_with_critic():
    """DiagnosticAggregator can run full critic evaluation."""
    aggregator = create_diagnostic_aggregator()

    spec = TaskSpec(task_name="test")
    results = [
        ValidatorResult(
            passed=False,
            reason="schema error",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.SCHEMA_SYNTAX,
        ),
    ]

    # This will use the default critic which may call LLM
    # For unit test, we just verify it runs without error
    feedback = aggregator.evaluate_with_critic(
        task_spec=spec,
        generated_output="bad",
        validator_results=results,
        retry_count=0,
    )

    assert feedback is not None
    assert feedback.passed is False
    assert len(feedback.diagnostics) >= 1