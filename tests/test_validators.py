"""Unit tests for declarative and custom validator engine."""
import pytest
from pydantic import BaseModel, ValidationError

from src.harness.contracts import (
    FailureTypeEnum,
    SeverityEnum,
    TaskSpec,
    ValidatorResult,
)
from src.harness.validators import (
    CustomValidator,
    DeclarativeValidator,
    ValidatorRegistry,
    check_purity,
)


# Module-level functions for purity testing (inspect.getsource doesn't work for nested functions)
def _impure_validator_httpx(output: str, context: dict | None = None) -> ValidatorResult:
    import httpx
    httpx.get("http://example.com")
    return ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO)


def _pure_validator(output: str, context: dict | None = None) -> ValidatorResult:
    return ValidatorResult(passed=True, reason="pure ok", severity=SeverityEnum.INFO)


def _positive_check_validator(output: str, context: dict | None = None) -> ValidatorResult:
    try:
        value = float(output.strip())
        if value > 0:
            return ValidatorResult(passed=True, reason="Positive number", severity=SeverityEnum.INFO)
        return ValidatorResult(
            passed=False,
            reason="Not positive",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.CONSTRAINT_VIOLATION,
        )
    except ValueError:
        return ValidatorResult(
            passed=False,
            reason="Not a number",
            severity=SeverityEnum.CRITICAL,
            failure_type=FailureTypeEnum.SCHEMA_SYNTAX,
        )


def _buggy_validator(output: str, context: dict | None = None) -> ValidatorResult:
    raise RuntimeError("Internal error")


def _custom1_validator(output: str, context: dict | None = None) -> ValidatorResult:
    return ValidatorResult(passed=True, reason="c1 ok", severity=SeverityEnum.INFO)


def _custom2_validator(output: str, context: dict | None = None) -> ValidatorResult:
    return ValidatorResult(passed=False, reason="c2 fail", severity=SeverityEnum.CRITICAL,
                           failure_type=FailureTypeEnum.CONSTRAINT_VIOLATION)


def _custom_validator_for_hash(output: str, context: dict | None = None) -> ValidatorResult:
    return ValidatorResult(passed=True, reason="", severity=SeverityEnum.INFO)


def _validator_func_for_hash(output: str, context: dict | None = None) -> ValidatorResult:
    return ValidatorResult(passed=True, reason="", severity=SeverityEnum.INFO)


def test_purity_check_clean():
    """Clean function passes purity check."""
    source = '''
def validator(output, context):
    return ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO)
'''
    violations = check_purity(source)
    assert violations == []


def test_purity_check_forbidden_import():
    """Function with forbidden import fails purity check."""
    source = '''
import httpx
def validator(output, context):
    return ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO)
'''
    violations = check_purity(source)
    assert any("httpx" in v for v in violations)


def test_purity_check_forbidden_call():
    """Function with forbidden call pattern fails purity check."""
    source = '''
def validator(output, context):
    httpx.post("http://example.com")
    return ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO)
'''
    violations = check_purity(source)
    assert any("post" in v for v in violations)


def test_purity_check_requests():
    """Function using requests fails purity check."""
    source = '''
import requests
def validator(output, context):
    requests.get("http://example.com")
    return ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO)
'''
    violations = check_purity(source)
    assert len(violations) >= 2  # import + call


def test_custom_validator_purity_enforcement():
    """Custom validator rejects impure functions."""
    validator = CustomValidator("impure", _impure_validator_httpx, enforce_purity=True)
    assert not validator.is_pure
    assert len(validator.purity_violations) > 0

    result = validator.validate("test")
    assert result.passed is False
    assert "purity" in result.reason.lower()


def test_custom_validator_pure_passes():
    """Pure custom validator works normally."""
    validator = CustomValidator("pure", _pure_validator, enforce_purity=True)
    assert validator.is_pure
    assert validator.purity_violations == []

    result = validator.validate("test")
    assert result.passed is True


def test_custom_validator_purity_can_be_disabled():
    """Purity enforcement can be disabled."""
    validator = CustomValidator("impure", _impure_validator_httpx, enforce_purity=False)
    result = validator.validate("test")
    assert result.passed is True


def test_declarative_validator_no_constraints():
    """Validator with no constraints passes everything."""
    spec = TaskSpec(task_name="free_form")
    validator = DeclarativeValidator(spec)
    result = validator.validate("any output")
    assert result.passed is True


def test_declarative_validator_regex():
    """Regex pattern validation."""
    spec = TaskSpec(task_name="id_parser", regex_patterns=[r"^INV-\d{6}$"])
    validator = DeclarativeValidator(spec)

    # Valid
    result = validator.validate("INV-123456")
    assert result.passed is True

    # Invalid
    result = validator.validate("INV-123")
    assert result.passed is False
    assert result.failure_type == FailureTypeEnum.REGEX_MISMATCH


def test_declarative_validator_keyword_blacklist():
    """Keyword blacklist validation."""
    spec = TaskSpec(task_name="safe_output", keyword_blacklist=["password", "secret"])
    validator = DeclarativeValidator(spec)

    # Clean
    result = validator.validate("The result is 42")
    assert result.passed is True

    # Violates blacklist
    result = validator.validate("The password is 123")
    assert result.passed is False
    assert result.failure_type == FailureTypeEnum.KEYWORD_VIOLATION


def test_declarative_validator_keyword_whitelist():
    """Keyword whitelist validation (all must be present)."""
    spec = TaskSpec(task_name="structured", keyword_whitelist=["total", "currency"])
    validator = DeclarativeValidator(spec)

    # All present
    result = validator.validate("total: 100, currency: USD")
    assert result.passed is True

    # Missing one
    result = validator.validate("total: 100")
    assert result.passed is False
    assert result.failure_type == FailureTypeEnum.KEYWORD_VIOLATION


def test_declarative_validator_length_bounds():
    """Min/max length validation."""
    spec = TaskSpec(task_name="bounded", min_length=10, max_length=50)
    validator = DeclarativeValidator(spec)

    # Within bounds
    result = validator.validate("This is a valid output string")
    assert result.passed is True

    # Too short
    result = validator.validate("short")
    assert result.passed is False
    assert result.failure_type == FailureTypeEnum.CONSTRAINT_VIOLATION

    # Too long
    result = validator.validate("x" * 60)
    assert result.passed is False
    assert result.failure_type == FailureTypeEnum.CONSTRAINT_VIOLATION


def test_declarative_validator_json_schema():
    """JSON schema validation via Pydantic."""

    class Invoice(BaseModel):
        invoice_id: str
        amount: float

    spec = TaskSpec(
        task_name="invoice_parser",
        json_schema=Invoice.model_json_schema(),
    )
    validator = DeclarativeValidator(spec)

    # Valid JSON
    result = validator.validate('{"invoice_id": "INV-001", "amount": 99.5}')
    assert result.passed is True

    # Invalid JSON (missing required field)
    result = validator.validate('{"amount": 99.5}')
    assert result.passed is False
    assert result.failure_type == FailureTypeEnum.SCHEMA_SYNTAX


def test_declarative_validator_hash_deterministic():
    """Validator hash is deterministic for same config."""
    spec1 = TaskSpec(task_name="test", regex_patterns=[r"\d+"])
    spec2 = TaskSpec(task_name="test", regex_patterns=[r"\d+"])
    v1 = DeclarativeValidator(spec1)
    v2 = DeclarativeValidator(spec2)
    assert v1.validator_hash == v2.validator_hash


def test_custom_validator():
    """Custom validator function wrapper."""
    validator = CustomValidator("positive_check", _positive_check_validator)

    # Valid
    result = validator.validate("42.5")
    assert result.passed is True

    # Invalid
    result = validator.validate("-5")
    assert result.passed is False
    assert result.failure_type == FailureTypeEnum.CONSTRAINT_VIOLATION


def test_custom_validator_exception_handling():
    """Custom validator handles exceptions gracefully."""
    validator = CustomValidator("buggy", _buggy_validator)
    result = validator.validate("anything")
    assert result.passed is False
    assert "exception" in result.reason.lower()


def test_validator_registry_declarative_only():
    """Registry with only declarative validator."""
    registry = ValidatorRegistry()
    spec = TaskSpec(task_name="test", regex_patterns=[r"^OK$"])
    registry.register_declarative(spec)

    results = registry.validate("OK")
    assert len(results) == 1
    assert results[0].passed is True

    results = registry.validate("FAIL")
    assert len(results) == 1
    assert results[0].passed is False


def test_validator_registry_fail_fast_declarative():
    """Registry fails fast on declarative failure, skips custom."""
    registry = ValidatorRegistry()
    spec = TaskSpec(task_name="test", regex_patterns=[r"^OK$"])
    registry.register_declarative(spec)

    custom_called = False
    def custom(output, context):
        nonlocal custom_called
        custom_called = True
        return ValidatorResult(passed=True, reason="custom ok", severity=SeverityEnum.INFO)

    registry.register_custom("custom", custom)

    # Declarative fails - custom should not be called
    results = registry.validate("FAIL")
    assert len(results) == 1
    assert results[0].passed is False
    assert custom_called is False


def test_validator_registry_runs_custom_after_declarative_pass():
    """Registry runs custom validators after declarative passes."""
    registry = ValidatorRegistry()
    spec = TaskSpec(task_name="test", regex_patterns=[r"^\d+$"])
    registry.register_declarative(spec)

    registry.register_custom("custom1", _custom1_validator)
    registry.register_custom("custom2", _custom2_validator)

    results = registry.validate("123")
    assert len(results) == 3  # declarative + 2 custom
    assert results[0].passed is True  # declarative
    assert results[1].passed is True  # custom1
    assert results[2].passed is False  # custom2


def test_validator_registry_hash_composition():
    """Registry hash includes all validators."""
    registry = ValidatorRegistry()
    spec = TaskSpec(task_name="test", regex_patterns=[r"\d+"])
    registry.register_declarative(spec)
    registry.register_custom("custom1", _custom_validator_for_hash)

    hash1 = registry.validator_set_hash

    registry2 = ValidatorRegistry()
    registry2.register_declarative(spec)
    registry2.register_custom("custom1", _custom_validator_for_hash)

    hash2 = registry2.validator_set_hash
    assert hash1 == hash2
    assert len(hash1) == 16  # SHA256 truncated to 16 chars


def test_custom_validator_hash_deterministic():
    """Custom validator hash is deterministic."""
    v1 = CustomValidator("same_name", _validator_func_for_hash)
    v2 = CustomValidator("same_name", _validator_func_for_hash)
    assert v1.validator_hash == v2.validator_hash