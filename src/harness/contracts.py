"""Immutable shared data contracts and types for Agentic Harness v1.2."""
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class FailureTypeEnum(str, Enum):
    """Standardized failure categories for diagnostics and evaluation."""
    SCHEMA_SYNTAX = "schema_syntax"
    REGEX_MISMATCH = "regex_mismatch"
    KEYWORD_VIOLATION = "keyword_violation"
    SEMANTIC_DRIFT = "semantic_drift"
    CONSTRAINT_VIOLATION = "constraint_violation"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    OTHER = "other"


class SeverityEnum(str, Enum):
    """Severity ratings for validator and critic diagnostics."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class ValidatorResult(BaseModel):
    """Uniform result contract returned by both declarative and custom validators."""
    model_config = ConfigDict(frozen=True)

    passed: bool
    reason: str
    severity: SeverityEnum = SeverityEnum.CRITICAL
    failure_type: Optional[FailureTypeEnum] = None


class DiagnosticItem(BaseModel):
    """Detailed diagnostic element within critic feedback."""
    model_config = ConfigDict(frozen=True)

    failure_type: FailureTypeEnum
    severity: SeverityEnum
    validator_name: str
    message: str
    suggested_correction: Optional[str] = None


class CriticFeedback(BaseModel):
    """Structured diagnostic evaluation returned by the critic."""
    model_config = ConfigDict(frozen=True)

    passed: bool
    diagnostics: List[DiagnosticItem] = Field(default_factory=list)
    guidance: str = ""
    constraints: List[str] = Field(default_factory=list)
    escalate_to_negative_constraints: bool = False


class LLMResponse(BaseModel):
    """Standardized response container returned by all LLMClient implementations."""
    model_config = ConfigDict(frozen=True)

    content: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None


class TaskSpec(BaseModel):
    """Specification defining a task, its schemas, constraints, and active validators."""
    model_config = ConfigDict(frozen=True)

    task_name: str
    description: str = ""
    system_prompt: str = ""
    json_schema: Optional[Dict[str, Any]] = None
    regex_patterns: List[str] = Field(default_factory=list)
    keyword_blacklist: List[str] = Field(default_factory=list)
    keyword_whitelist: List[str] = Field(default_factory=list)
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    custom_validators: List[str] = Field(default_factory=list)
