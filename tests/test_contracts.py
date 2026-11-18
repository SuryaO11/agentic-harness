"""Unit tests for shared core contracts and data models."""
import pytest
from pydantic import ValidationError

from src.harness.contracts import (
    CriticFeedback,
    DiagnosticItem,
    FailureTypeEnum,
    LLMResponse,
    SeverityEnum,
    TaskSpec,
    ValidatorResult,
)
from src.harness.exceptions import HarnessError, ProviderError, RateLimitError


def test_validator_result_valid_and_frozen():
    result = ValidatorResult(
        passed=True,
        reason="All schema checks passed",
        severity=SeverityEnum.INFO,
        failure_type=None,
    )
    assert result.passed is True
    assert result.reason == "All schema checks passed"
    assert result.severity == SeverityEnum.INFO
    assert result.failure_type is None

    # Test immutability (frozen config)
    with pytest.raises(ValidationError):
        result.passed = False  # type: ignore


def test_validator_result_failure():
    result = ValidatorResult(
        passed=False,
        reason="Field 'total' must be >= 0",
        severity=SeverityEnum.CRITICAL,
        failure_type=FailureTypeEnum.CONSTRAINT_VIOLATION,
    )
    assert result.passed is False
    assert result.failure_type == FailureTypeEnum.CONSTRAINT_VIOLATION
    assert result.severity == SeverityEnum.CRITICAL


def test_diagnostic_item_and_critic_feedback():
    diag = DiagnosticItem(
        failure_type=FailureTypeEnum.REGEX_MISMATCH,
        severity=SeverityEnum.CRITICAL,
        validator_name="invoice_id_regex",
        message="ID must match ^INV-\\d{6}$",
        suggested_correction="Prefix with INV- followed by 6 digits",
    )
    feedback = CriticFeedback(
        passed=False,
        diagnostics=[diag],
        guidance="Fix the invoice_id format",
        constraints=["invoice_id must match ^INV-\\d{6}$"],
        escalate_to_negative_constraints=False,
    )
    assert feedback.passed is False
    assert len(feedback.diagnostics) == 1
    assert feedback.diagnostics[0].failure_type == FailureTypeEnum.REGEX_MISMATCH

    # Test JSON round-trip
    json_data = feedback.model_dump_json()
    reloaded = CriticFeedback.model_validate_json(json_data)
    assert reloaded == feedback


def test_llm_response():
    resp = LLMResponse(
        content='{"status": "ok"}',
        model="openai/gpt-oss-20b",
        provider="groq",
        prompt_tokens=45,
        completion_tokens=12,
        latency_ms=210.5,
    )
    assert resp.content == '{"status": "ok"}'
    assert resp.latency_ms == 210.5
    assert resp.provider == "groq"

    # Immutability
    with pytest.raises(ValidationError):
        resp.content = "modified"  # type: ignore


def test_task_spec():
    spec = TaskSpec(
        task_name="invoice_parser",
        description="Extract invoice data",
        system_prompt="You are a strict data extraction assistant.",
        regex_patterns=[r"^INV-\d{6}$"],
        keyword_blacklist=["confidential"],
        min_length=10,
        max_length=5000,
        custom_validators=["validate_totals"],
    )
    assert spec.task_name == "invoice_parser"
    assert len(spec.regex_patterns) == 1
    assert "confidential" in spec.keyword_blacklist


def test_rate_limit_error_retry_after():
    err = RateLimitError("Rate limit reached", retry_after=30.0)
    assert isinstance(err, HarnessError)
    assert err.retry_after == 30.0

    err_no_retry = RateLimitError("Rate limit without header")
    assert err_no_retry.retry_after is None

