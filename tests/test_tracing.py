"""Unit tests for execution tracing and budget guards."""
import pytest
import json

from src.harness.contracts import (
    FailureTypeEnum,
    SeverityEnum,
    TaskSpec,
    ValidatorResult,
)
from src.harness.tracing import (
    ExecutionTrace,
    RetryBudget,
    TraceEvent,
    TracedRetryOrchestrator,
    create_traced_orchestrator,
)


def test_execution_trace_creation():
    """ExecutionTrace can be created and events added."""
    trace = ExecutionTrace(
        trace_id="test-123",
        task_name="test_task",
        start_time=1000.0,
    )

    assert trace.trace_id == "test-123"
    assert trace.task_name == "test_task"
    assert len(trace.events) == 0


def test_execution_trace_add_event():
    """Adding events to trace."""
    trace = ExecutionTrace(
        trace_id="test-123",
        task_name="test_task",
        start_time=1000.0,
    )

    trace.add_event(
        event_type="generation",
        attempt_number=1,
        data={"output": "test"},
        latency_ms=100.0,
    )

    assert len(trace.events) == 1
    assert trace.events[0].event_type == "generation"
    assert trace.events[0].attempt_number == 1
    assert trace.events[0].data["output"] == "test"
    assert trace.events[0].latency_ms == 100.0


def test_execution_trace_to_dict():
    """Trace serialization to dict."""
    trace = ExecutionTrace(
        trace_id="test-123",
        task_name="test_task",
        start_time=1000.0,
    )

    trace.add_event("generation", 1, {"output": "test"}, 100.0)
    trace.end_time = 1001.0

    d = trace.to_dict()

    assert d["trace_id"] == "test-123"
    assert d["task_name"] == "test_task"
    assert d["duration_ms"] == 1000.0
    assert len(d["events"]) == 1


def test_execution_trace_to_json():
    """Trace serialization to JSON."""
    trace = ExecutionTrace(
        trace_id="test-123",
        task_name="test_task",
        start_time=1000.0,
    )

    trace.add_event("generation", 1, {"output": "test"}, 100.0)
    trace.end_time = 1001.0

    json_str = trace.to_json()
    parsed = json.loads(json_str)

    assert parsed["trace_id"] == "test-123"
    assert len(parsed["events"]) == 1


def test_retry_budget_creation():
    """RetryBudget can be created with defaults."""
    budget = RetryBudget()

    assert budget.max_retries == 3
    assert budget.retries_used == 0
    assert budget.can_retry() is True


def test_retry_budget_consume():
    """Budget consumes correctly."""
    budget = RetryBudget(max_retries=3)

    budget.consume(100.0, 50)
    assert budget.retries_used == 1
    assert budget.total_latency_ms == 100.0
    assert budget.total_tokens == 50


def test_retry_budget_exhausted():
    """Budget exhausted after max retries."""
    budget = RetryBudget(max_retries=2)

    budget.consume(100.0)
    budget.consume(100.0)
    assert budget.can_retry() is False  # 2 retries used, no more allowed

    # Test with max_retries=3
    budget2 = RetryBudget(max_retries=3)
    budget2.consume(100.0)
    budget2.consume(100.0)
    assert budget2.can_retry() is True  # 2 used, 1 remaining
    budget2.consume(100.0)
    assert budget2.can_retry() is False  # 3 used, exhausted


def test_retry_budget_latency_limit():
    """Budget respects latency limit."""
    budget = RetryBudget(max_retries=10, max_total_latency_ms=250.0)

    budget.consume(100.0)
    budget.consume(100.0)
    assert budget.can_retry() is True

    budget.consume(100.0)  # Total 300 > 250
    assert budget.can_retry() is False


def test_retry_budget_token_limit():
    """Budget respects token limit."""
    budget = RetryBudget(max_retries=10, max_total_tokens=150)

    budget.consume(10.0, 50)
    budget.consume(10.0, 50)
    assert budget.can_retry() is True

    budget.consume(10.0, 100)  # Total 200 > 150
    assert budget.can_retry() is False


def test_retry_budget_reset():
    """Budget reset clears all counters."""
    budget = RetryBudget(max_retries=3)

    budget.consume(100.0, 50)
    budget.consume(100.0, 50)
    budget.reset()

    assert budget.retries_used == 0
    assert budget.total_latency_ms == 0.0
    assert budget.total_tokens == 0
    assert budget.can_retry() is True


def test_retry_budget_get_status():
    """Budget status returns all relevant info."""
    budget = RetryBudget(max_retries=3, max_total_latency_ms=5000, max_total_tokens=1000)

    budget.consume(100.0, 50)

    status = budget.get_status()

    assert status["retries_used"] == 1
    assert status["max_retries"] == 3
    assert status["remaining_retries"] == 2
    assert status["total_latency_ms"] == 100.0
    assert status["max_total_latency_ms"] == 5000
    assert status["total_tokens"] == 50
    assert status["max_total_tokens"] == 1000
    assert status["can_retry"] is True


def test_traced_orchestrator_creation():
    """TracedRetryOrchestrator can be created."""
    orchestrator = create_traced_orchestrator()

    assert orchestrator.max_retries == 3
    assert orchestrator.temperature == 0.0
    assert orchestrator.budget is not None


def test_traced_orchestrator_custom_budget():
    """Orchestrator accepts custom budget."""
    budget = RetryBudget(max_retries=5, max_total_latency_ms=10000)
    orchestrator = create_traced_orchestrator(budget=budget)

    assert orchestrator.budget is budget
    assert orchestrator.budget.max_retries == 5


def test_traced_orchestrator_creates_trace():
    """Orchestrator creates trace when none provided."""
    orchestrator = create_traced_orchestrator()

    # Mock dependencies
    from unittest.mock import MagicMock
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '{"status": "ok"}'
    mock_response.prompt_tokens = 10
    mock_response.completion_tokens = 5
    mock_client.generate.return_value = mock_response
    orchestrator.llm_client = mock_client

    mock_registry = MagicMock()
    mock_registry.validate.return_value = [
        ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO),
    ]
    orchestrator.validator_registry = mock_registry

    mock_scorer = MagicMock()
    mock_scorer.evaluate.return_value = ValidatorResult(passed=True, reason="ok", severity=SeverityEnum.INFO)
    orchestrator.semantic_scorer = mock_scorer

    spec = TaskSpec(task_name="test")
    result = orchestrator.execute(spec, "Test prompt")

    assert result.success is True


def test_trace_event_fields():
    """TraceEvent has all required fields."""
    event = TraceEvent(
        event_id="evt-123",
        timestamp=1000.0,
        event_type="generation",
        attempt_number=1,
        data={"key": "value"},
        latency_ms=50.0,
    )

    assert event.event_id == "evt-123"
    assert event.timestamp == 1000.0
    assert event.event_type == "generation"
    assert event.attempt_number == 1
    assert event.data["key"] == "value"
    assert event.latency_ms == 50.0