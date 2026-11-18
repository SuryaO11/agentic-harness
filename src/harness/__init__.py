"""Agentic Harness v1.2 - Self-correcting reliability layer for LLMs."""

from .aggregator import (
    AggregatedDiagnostics,
    DiagnosticAggregator,
    create_diagnostic_aggregator,
)
from .cache import (
    CacheEntry,
    CacheManager,
    SQLiteCache,
    create_cache_manager,
)
from .client import (
    GroqClient,
    LLMClient,
    OllamaClient,
    get_llm_client,
)
from .contracts import (
    CriticFeedback,
    DiagnosticItem,
    FailureTypeEnum,
    LLMResponse,
    SeverityEnum,
    TaskSpec,
    ValidatorResult,
)
from .critic import (
    CriticEvaluator,
    create_critic_evaluator,
)
from .benchmark import (
    BENCHMARK_SAMPLES,
    NUM_SAMPLES,
    run_benchmark,
    generate_report,
    save_report,
    MockLLMClient,
    build_task_spec,
)
from .dashboard import main as run_dashboard
from .escalation import (
    PromptEscalationEngine,
    PromptRefiner,
    create_prompt_escalation_engine,
    create_prompt_refiner,
)
from .exceptions import (
    HarnessError,
    ProviderError,
    RateLimitError,
)
from .orchestrator import (
    ConversationHistory,
    OrchestrationResult,
    RetryAttempt,
    RetryOrchestrator,
    create_retry_orchestrator,
)
from .scorer import (
    EmbeddingCache,
    SemanticScorer,
    create_semantic_scorer,
)
from .tracing import (
    ExecutionTrace,
    RetryBudget,
    TraceEvent,
    TracedRetryOrchestrator,
    create_traced_orchestrator,
)
from .validators import (
    CustomValidator,
    DeclarativeValidator,
    ValidatorRegistry,
    check_purity,
)

__all__ = [
    "CriticFeedback",
    "DiagnosticItem",
    "FailureTypeEnum",
    "LLMResponse",
    "SeverityEnum",
    "TaskSpec",
    "ValidatorResult",
    "HarnessError",
    "ProviderError",
    "RateLimitError",
    "LLMClient",
    "GroqClient",
    "OllamaClient",
    "get_llm_client",
    "ValidatorRegistry",
    "DeclarativeValidator",
    "CustomValidator",
    "check_purity",
    "EmbeddingCache",
    "SemanticScorer",
    "create_semantic_scorer",
    "CriticEvaluator",
    "create_critic_evaluator",
    "AggregatedDiagnostics",
    "DiagnosticAggregator",
    "create_diagnostic_aggregator",
    "ConversationHistory",
    "OrchestrationResult",
    "RetryAttempt",
    "RetryOrchestrator",
    "create_retry_orchestrator",
    "PromptEscalationEngine",
    "PromptRefiner",
    "create_prompt_escalation_engine",
    "create_prompt_refiner",
    "ExecutionTrace",
    "RetryBudget",
    "TraceEvent",
    "TracedRetryOrchestrator",
    "create_traced_orchestrator",
    "CacheEntry",
    "CacheManager",
    "SQLiteCache",
    "create_cache_manager",
    "run_dashboard",
    "BENCHMARK_SAMPLES",
    "NUM_SAMPLES",
    "run_benchmark",
    "generate_report",
    "save_report",
    "MockLLMClient",
    "build_task_spec",
]
