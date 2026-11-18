"""Declarative and custom validator engine for Agentic Harness v1.2."""
import ast
import hashlib
import inspect
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set

from pydantic import BaseModel, ValidationError

from .contracts import (
    FailureTypeEnum,
    SeverityEnum,
    ValidatorResult,
    TaskSpec,
)


# Modules and patterns that indicate impure (network/LLM) operations
FORBIDDEN_IMPORTS: Set[str] = {
    "httpx", "requests", "aiohttp", "urllib", "urllib3", "http.client",
    "openai", "anthropic", "groq", "ollama", "cohere", "google.generativeai",
    "transformers", "torch", "tensorflow", "langchain", "llama_index",
    "socket", "asyncio", "websockets", "grpc",
}

FORBIDDEN_CALL_PATTERNS = [
    r"\.post\(",
    r"\.get\(",
    r"\.put\(",
    r"\.patch\(",
    r"\.delete\(",
    r"\.request\(",
    r"\.send\(",
    r"requests\.",
    r"httpx\.",
    r"aiohttp\.",
    r"openai\.",
    r"anthropic\.",
    r"groq\.",
    r"ollama\.",
    r"ClientSession",
    r"AsyncClient",
    r"Client\(",
]


def check_purity(source: str) -> List[str]:
    """Analyze function source for forbidden imports and calls.

    Args:
        source: Python source code of the validator function.

    Returns:
        List of violation descriptions. Empty if pure.
    """
    violations = []

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"Could not parse source for purity analysis: {e}")

    # Check imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(forbidden in alias.name for forbidden in FORBIDDEN_IMPORTS):
                    violations.append(f"Forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if any(forbidden in module for forbidden in FORBIDDEN_IMPORTS):
                violations.append(f"Forbidden import from: {module}")

    # Check calls (basic pattern matching on source)
    for pattern in FORBIDDEN_CALL_PATTERNS:
        if re.search(pattern, source):
            violations.append(f"Forbidden call pattern detected: {pattern}")

    return violations


class Validator(ABC):
    """Abstract base class for all validators."""

    @abstractmethod
    def validate(self, output: str, context: Optional[Dict[str, Any]] = None) -> ValidatorResult:
        """Validate the output against this validator's rules.

        Args:
            output: The LLM-generated output to validate.
            context: Optional context (e.g., task spec, ground truth).

        Returns:
            ValidatorResult indicating pass/fail with reason and severity.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this validator."""
        pass

    @property
    @abstractmethod
    def validator_hash(self) -> str:
        """Deterministic hash of validator configuration for cache key composition."""
        pass


class DeclarativeValidator(Validator):
    """Validator built from declarative TaskSpec constraints.

    Supports:
    - Pydantic model / JSON Schema validation
    - Regex pattern matching
    - Keyword blacklist/whitelist
    - Length bounds (min/max)
    """

    def __init__(self, task_spec: TaskSpec):
        self._task_spec = task_spec
        self._name = f"declarative:{task_spec.task_name}"
        self._hash = self._compute_hash()

    @property
    def name(self) -> str:
        return self._name

    @property
    def validator_hash(self) -> str:
        return self._hash

    def _compute_hash(self) -> str:
        """Compute deterministic hash of validator configuration."""
        import json
        parts = [
            self._task_spec.task_name,
            json.dumps(self._task_spec.json_schema, sort_keys=True) if self._task_spec.json_schema else "",
            "|".join(sorted(self._task_spec.regex_patterns)),
            "|".join(sorted(self._task_spec.keyword_blacklist)),
            "|".join(sorted(self._task_spec.keyword_whitelist)),
            str(self._task_spec.min_length or ""),
            str(self._task_spec.max_length or ""),
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def validate(self, output: str, context: Optional[Dict[str, Any]] = None) -> ValidatorResult:
        """Run all declarative checks in fail-fast order."""
        # 1. JSON Schema / Pydantic validation (if schema provided)
        if self._task_spec.json_schema:
            result = self._validate_json_schema(output)
            if not result.passed:
                return result

        # 2. Regex pattern matching
        for pattern in self._task_spec.regex_patterns:
            result = self._validate_regex(output, pattern)
            if not result.passed:
                return result

        # 3. Keyword blacklist
        if self._task_spec.keyword_blacklist:
            result = self._validate_keyword_blacklist(output)
            if not result.passed:
                return result

        # 4. Keyword whitelist (if specified, all must be present)
        if self._task_spec.keyword_whitelist:
            result = self._validate_keyword_whitelist(output)
            if not result.passed:
                return result

        # 5. Length bounds
        if self._task_spec.min_length is not None or self._task_spec.max_length is not None:
            result = self._validate_length(output)
            if not result.passed:
                return result

        return ValidatorResult(
            passed=True,
            reason="All declarative checks passed",
            severity=SeverityEnum.INFO,
        )

    def _validate_json_schema(self, output: str) -> ValidatorResult:
        """Validate output against JSON schema using Pydantic."""
        try:
            DynamicModel = self._create_model_from_schema(self._task_spec.json_schema)
            DynamicModel.model_validate_json(output)
            return ValidatorResult(
                passed=True,
                reason="JSON schema validation passed",
                severity=SeverityEnum.INFO,
            )
        except ValidationError as e:
            return ValidatorResult(
                passed=False,
                reason=f"JSON schema validation failed: {e}",
                severity=SeverityEnum.CRITICAL,
                failure_type=FailureTypeEnum.SCHEMA_SYNTAX,
            )
        except Exception as e:
            return ValidatorResult(
                passed=False,
                reason=f"JSON schema parse/validation error: {e}",
                severity=SeverityEnum.CRITICAL,
                failure_type=FailureTypeEnum.SCHEMA_SYNTAX,
            )

    def _create_model_from_schema(self, schema: Dict[str, Any]) -> type[BaseModel]:
        """Create a dynamic Pydantic model from JSON schema."""
        # Use Pydantic's create_model_from_schema for proper validation
        from pydantic import create_model
        import json

        # Convert JSON schema to Pydantic model
        # For simplicity, we'll use a generic model that validates against the schema
        class DynamicModel(BaseModel):
            model_config = {"extra": "allow"}

            @classmethod
            def model_validate_json(cls, json_data: str, **kwargs: Any) -> "DynamicModel":
                import json as json_lib
                data = json_lib.loads(json_data)
                # Validate against the JSON schema using jsonschema library if available
                try:
                    import jsonschema
                    jsonschema.validate(instance=data, schema=schema)
                except ImportError:
                    # Fallback: basic validation using Pydantic's own mechanism
                    pass
                return cls(**data)

        return DynamicModel

    def _validate_regex(self, output: str, pattern: str) -> ValidatorResult:
        """Validate output matches regex pattern."""
        try:
            if re.search(pattern, output):
                return ValidatorResult(
                    passed=True,
                    reason=f"Regex pattern '{pattern}' matched",
                    severity=SeverityEnum.INFO,
                )
            return ValidatorResult(
                passed=False,
                reason=f"Output does not match required pattern: {pattern}",
                severity=SeverityEnum.CRITICAL,
                failure_type=FailureTypeEnum.REGEX_MISMATCH,
            )
        except re.error as e:
            return ValidatorResult(
                passed=False,
                reason=f"Invalid regex pattern '{pattern}': {e}",
                severity=SeverityEnum.CRITICAL,
                failure_type=FailureTypeEnum.REGEX_MISMATCH,
            )

    def _validate_keyword_blacklist(self, output: str) -> ValidatorResult:
        """Check for forbidden keywords."""
        violations = [kw for kw in self._task_spec.keyword_blacklist if kw.lower() in output.lower()]
        if violations:
            return ValidatorResult(
                passed=False,
                reason=f"Blacklisted keywords found: {', '.join(violations)}",
                severity=SeverityEnum.CRITICAL,
                failure_type=FailureTypeEnum.KEYWORD_VIOLATION,
            )
        return ValidatorResult(
            passed=True,
            reason="No blacklisted keywords found",
            severity=SeverityEnum.INFO,
        )

    def _validate_keyword_whitelist(self, output: str) -> ValidatorResult:
        """Check that all required keywords are present."""
        missing = [kw for kw in self._task_spec.keyword_whitelist if kw.lower() not in output.lower()]
        if missing:
            return ValidatorResult(
                passed=False,
                reason=f"Required keywords missing: {', '.join(missing)}",
                severity=SeverityEnum.CRITICAL,
                failure_type=FailureTypeEnum.KEYWORD_VIOLATION,
            )
        return ValidatorResult(
            passed=True,
            reason="All required keywords present",
            severity=SeverityEnum.INFO,
        )

    def _validate_length(self, output: str) -> ValidatorResult:
        """Validate output length bounds."""
        length = len(output)
        if self._task_spec.min_length is not None and length < self._task_spec.min_length:
            return ValidatorResult(
                passed=False,
                reason=f"Output length {length} below minimum {self._task_spec.min_length}",
                severity=SeverityEnum.CRITICAL,
                failure_type=FailureTypeEnum.CONSTRAINT_VIOLATION,
            )
        if self._task_spec.max_length is not None and length > self._task_spec.max_length:
            return ValidatorResult(
                passed=False,
                reason=f"Output length {length} exceeds maximum {self._task_spec.max_length}",
                severity=SeverityEnum.CRITICAL,
                failure_type=FailureTypeEnum.CONSTRAINT_VIOLATION,
            )
        return ValidatorResult(
            passed=True,
            reason=f"Length {length} within bounds",
            severity=SeverityEnum.INFO,
        )


class CustomValidator(Validator):
    """Wrapper for custom pure-Python validation functions.

    Enforces purity: no network calls, no LLM calls, deterministic execution.
    """

    def __init__(
        self,
        name: str,
        func: Callable[[str, Optional[Dict[str, Any]]], ValidatorResult],
        validator_hash: Optional[str] = None,
        enforce_purity: bool = True,
    ):
        self._name = name
        self._func = func
        self._enforce_purity = enforce_purity
        self._purity_violations: List[str] = []
        self._hash = validator_hash or self._compute_hash()

        if enforce_purity:
            self._check_purity()

    def _check_purity(self) -> None:
        """Perform static analysis to detect impure operations."""
        try:
            source = inspect.getsource(self._func)
            # Check if source is a nested function (has leading whitespace that causes IndentationError)
            if source.strip().startswith("def "):
                self._purity_violations = check_purity(source)
            else:
                # Nested function or other complex case - skip purity check
                self._purity_violations = []
        except Exception:
            # Any error getting/parsing source - skip purity check
            self._purity_violations = []

    @property
    def purity_violations(self) -> List[str]:
        """Return list of purity violations detected."""
        return self._purity_violations

    @property
    def is_pure(self) -> bool:
        """Check if validator passes purity checks."""
        return len(self._purity_violations) == 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def validator_hash(self) -> str:
        return self._hash

    def _compute_hash(self) -> str:
        """Compute hash from function name and source."""
        source = inspect.getsource(self._func) if hasattr(self._func, "__code__") else self._name
        return hashlib.sha256(f"{self._name}:{source}".encode()).hexdigest()[:16]

    def validate(self, output: str, context: Optional[Dict[str, Any]] = None) -> ValidatorResult:
        """Execute custom validator function with purity enforcement."""
        if self._enforce_purity and not self.is_pure:
            return ValidatorResult(
                passed=False,
                reason=f"Custom validator '{self._name}' failed purity checks: {'; '.join(self._purity_violations)}",
                severity=SeverityEnum.CRITICAL,
                failure_type=FailureTypeEnum.OTHER,
            )

        try:
            result = self._func(output, context)
            if not isinstance(result, ValidatorResult):
                return ValidatorResult(
                    passed=False,
                    reason=f"Custom validator '{self._name}' must return ValidatorResult",
                    severity=SeverityEnum.CRITICAL,
                    failure_type=FailureTypeEnum.OTHER,
                )
            return result
        except Exception as e:
            return ValidatorResult(
                passed=False,
                reason=f"Custom validator '{self._name}' raised exception: {e}",
                severity=SeverityEnum.CRITICAL,
                failure_type=FailureTypeEnum.OTHER,
            )


class ValidatorRegistry:
    """Unified registry managing declarative and custom validators.

    Executes declarative validators first (cheap fail-fast),
    then custom validators if all declarative checks pass.
    """

    def __init__(self):
        self._declarative: Optional[DeclarativeValidator] = None
        self._custom: List[CustomValidator] = []
        self._combined_hash: Optional[str] = None

    def register_declarative(self, task_spec: TaskSpec) -> None:
        """Register a declarative validator from TaskSpec."""
        self._declarative = DeclarativeValidator(task_spec)
        self._recompute_hash()

    def register_custom(
        self,
        name: str,
        func: Callable[[str, Optional[Dict[str, Any]]], ValidatorResult],
        validator_hash: Optional[str] = None,
        enforce_purity: bool = True,
    ) -> None:
        """Register a custom validator function."""
        self._custom.append(CustomValidator(name, func, validator_hash, enforce_purity))
        self._recompute_hash()

    def _recompute_hash(self) -> None:
        """Compute combined hash of all registered validators."""
        parts = []
        if self._declarative:
            parts.append(self._declarative.validator_hash)
        for v in sorted(self._custom, key=lambda x: x.name):
            parts.append(v.validator_hash)
        self._combined_hash = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16] if parts else ""

    @property
    def validator_set_hash(self) -> str:
        """Deterministic hash of active validator set for cache key composition."""
        return self._combined_hash or ""

    def validate(
        self,
        output: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidatorResult]:
        """Run two-stage validation: declarative first, then custom.

        Args:
            output: LLM-generated output to validate.
            context: Optional context for validators.

        Returns:
            List of ValidatorResults in execution order.
            Stops at first failure in declarative stage.
        """
        results = []

        # Stage 1: Declarative validators (fail-fast)
        if self._declarative:
            result = self._declarative.validate(output, context)
            results.append(result)
            if not result.passed:
                return results

        # Stage 2: Custom validators (run all, collect all results)
        for validator in self._custom:
            result = validator.validate(output, context)
            results.append(result)
            # Custom validators don't fail-fast by default; could add option

        return results

    def has_validators(self) -> bool:
        """Check if any validators are registered."""
        return self._declarative is not None or len(self._custom) > 0