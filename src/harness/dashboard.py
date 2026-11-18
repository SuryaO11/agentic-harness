"""Streamlit Observability Dashboard for Agentic Harness v1.2.

Provides interactive query runner, real-time generation panel,
audit trail with validator breakdowns, critic feedback cards,
cache metrics, and full observability.
"""
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

try:
    # When running from src/ as package
    from .contracts import (
        CriticFeedback,
        DiagnosticItem,
        FailureTypeEnum,
        LLMResponse,
        SeverityEnum,
        TaskSpec,
        ValidatorResult,
    )
    from .orchestrator import OrchestrationResult, RetryOrchestrator
    from .critic import CriticEvaluator
    from .cache import CacheManager, create_cache_manager
except ImportError:
    # When running from project root
    from harness.contracts import (
        CriticFeedback,
        DiagnosticItem,
        FailureTypeEnum,
        LLMResponse,
        SeverityEnum,
        TaskSpec,
        ValidatorResult,
    )
    from harness.orchestrator import OrchestrationResult, RetryOrchestrator
    from harness.critic import CriticEvaluator
    from harness.cache import CacheManager, create_cache_manager


# Page configuration
st.set_page_config(
    page_title="Agentic Harness v1.2 - Observability Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .validator-badge-pass {
        background: #dcfce7;
        color: #166534;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.25rem;
    }
    .validator-badge-fail {
        background: #fee2e2;
        color: #991b1b;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.25rem;
    }
    .severity-critical {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .severity-warning {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .severity-info {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .cache-hit {
        background: #dcfce7;
        color: #166534;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-weight: 600;
        font-size: 0.875rem;
    }
    .cache-miss {
        background: #fef3c7;
        color: #92400e;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-weight: 600;
        font-size: 0.875rem;
    }
    .retry-timeline {
        border-left: 2px solid #e5e7eb;
        padding-left: 1.5rem;
        margin-left: 1rem;
    }
    .retry-step {
        position: relative;
        padding-bottom: 1.5rem;
    }
    .retry-step::before {
        content: "";
        position: absolute;
        left: -1.65rem;
        top: 0.25rem;
        width: 0.75rem;
        height: 0.75rem;
        border-radius: 50%;
        background: #3b82f6;
        border: 2px solid white;
    }
    .retry-step.final::before {
        background: #10b981;
    }
    .retry-step.failed::before {
        background: #ef4444;
    }
    .audit-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.75rem 0;
    }
    .audit-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }
    .audit-card-title {
        font-weight: 600;
        color: #1f2937;
        font-size: 1rem;
    }
    .attempt-badge {
        background: #e5e7eb;
        color: #374151;
        padding: 0.125rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .critic-guidance {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .constraint-box {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        font-family: monospace;
    }
    .success-rate {
        font-size: 1.875rem;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 1.5rem;
        background: #f3f4f6;
        border-radius: 0.5rem 0.5rem 0 0;
    }
    .stTabs [aria-selected="true"] {
        background: #1f2937;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = None
    if "cache_manager" not in st.session_state:
        st.session_state.cache_manager = create_cache_manager()
    if "critic" not in st.session_state:
        st.session_state.critic = CriticEvaluator()
    if "execution_history" not in st.session_state:
        st.session_state.execution_history = []
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    if "current_trace" not in st.session_state:
        st.session_state.current_trace = None
    if "task_spec" not in st.session_state:
        st.session_state.task_spec = None
    if "cache_hit" not in st.session_state:
        st.session_state.cache_hit = False


def render_header():
    """Render the main header."""
    st.markdown('<h1 class="main-header">🤖 Agentic Harness v1.2</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explainable Self-Correcting Reliability Framework for LLMs</p>', unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Provider selection
        provider = st.selectbox(
            "LLM Provider",
            ["groq", "ollama"],
            index=0,
            help="Select inference provider",
        )

        # Model selection based on provider
        if provider == "groq":
            model = st.selectbox(
                "Model",
                ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
                index=0,
            )
        else:
            model = st.text_input("Ollama Model", value="llama3.1:8b")

        # Temperature (fixed at 0.0)
        st.number_input(
            "Temperature",
            value=0.0,
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            disabled=True,
            help="Fixed at 0.0 for deterministic outputs",
        )

        # Max retries
        max_retries = st.number_input("Max Retries", value=3, min_value=1, max_value=5, step=1)

        # Cache settings
        st.subheader("💾 Cache")
        use_cache = st.checkbox("Enable Cache", value=True)
        cache_threshold = st.slider("Reliability Threshold", 0.0, 1.0, 0.8, 0.05)

        # Initialize orchestrator button
        use_mock = st.checkbox("Mock Mode (no API key needed)", value=True)

        if st.button("🔄 Initialize Orchestrator", type="primary"):
            with st.spinner("Initializing..."):
                try:
                    if use_mock:
                        from harness.benchmark import MockLLMClient
                        llm_client = MockLLMClient()
                    else:
                        from harness.client import get_llm_client
                        llm_client = get_llm_client(provider=provider, model_name=model)

                    orchestrator = RetryOrchestrator(
                        llm_client=llm_client,
                        max_retries=max_retries,
                        temperature=0.0,
                    )
                    st.session_state.orchestrator = orchestrator
                    st.session_state.cache_manager = create_cache_manager()
                    st.session_state.cache_manager.reliability_threshold = cache_threshold
                    mode_text = "Mock Mode" if use_mock else f"{provider}/{model}"
                    st.success(f"Orchestrator initialized! ({mode_text})")
                except Exception as e:
                    st.error(f"Initialization failed: {str(e)}")
                    st.info("Tip: Enable 'Mock Mode' to test without an API key.")

        # Show cache stats
        if st.session_state.cache_manager:
            stats = st.session_state.cache_manager.get_stats()
            st.subheader("📊 Cache Stats")
            st.metric("Total Entries", stats["total_entries"])
            st.metric("Above Threshold", stats["entries_above_threshold"])
            st.metric("Avg Reliability", f"{stats['average_reliability']:.2f}")
            st.metric("Size", f"{stats['size_mb']:.2f} MB")

        # Clear history
        if st.button("🗑️ Clear History"):
            st.session_state.execution_history = []
            st.session_state.current_result = None
            st.session_state.current_trace = None
            st.session_state.cache_hit = False
            st.rerun()

        return {
            "provider": provider,
            "model": model,
            "max_retries": max_retries,
            "use_cache": use_cache,
        }


def render_task_spec_input():
    """Render task specification input panel."""
    st.subheader("📋 Task Specification")

    col1, col2 = st.columns([2, 1])

    with col1:
        prompt = st.text_area(
            "Prompt / Query",
            placeholder="Enter your query or task description...",
            height=150,
            help="The input prompt to send to the LLM",
        )

    with col2:
        st.write("**Validator Configuration**")

        use_pydantic = st.checkbox("Pydantic Schema", value=False)
        if use_pydantic:
            schema_text = st.text_area(
                "JSON Schema",
                placeholder='{"type": "object", "properties": {"result": {"type": "string"}}}',
                height=100,
            )

        use_regex = st.checkbox("Regex Pattern", value=False)
        if use_regex:
            regex_pattern = st.text_input("Regex", placeholder="^\\d+$")

        use_keywords = st.checkbox("Keyword Constraints", value=False)
        if use_keywords:
            col_kw1, col_kw2 = st.columns(2)
            with col_kw1:
                whitelist = st.text_input("Whitelist (comma-separated)", placeholder="success, valid, true")
            with col_kw2:
                blacklist = st.text_input("Blacklist (comma-separated)", placeholder="error, fail, null")

        use_length = st.checkbox("Length Bounds", value=False)
        if use_length:
            col_len1, col_len2 = st.columns(2)
            with col_len1:
                min_len = st.number_input("Min Length", value=0, min_value=0)
            with col_len2:
                max_len = st.number_input("Max Length", value=10000, min_value=1)

    return {
        "prompt": prompt,
        "use_pydantic": use_pydantic,
        "schema": schema_text if use_pydantic else None,
        "use_regex": use_regex,
        "regex": regex_pattern if use_regex else None,
        "use_keywords": use_keywords,
        "whitelist": [w.strip() for w in whitelist.split(",")] if use_keywords and whitelist else [],
        "blacklist": [b.strip() for b in blacklist.split(",")] if use_keywords and blacklist else [],
        "use_length": use_length,
        "min_length": min_len if use_length else 0,
        "max_length": max_len if use_length else 10000,
    }


def run_orchestration(task_spec: Dict[str, Any], config: Dict[str, Any]) -> Optional[OrchestrationResult]:
    """Run the orchestration with the given task spec."""
    if not task_spec["prompt"].strip():
        st.error("Please enter a prompt!")
        return None

    # Auto-initialize orchestrator if not yet initialized
    if not st.session_state.orchestrator:
        try:
            from harness.benchmark import MockLLMClient
            from harness.orchestrator import RetryOrchestrator
            st.session_state.orchestrator = RetryOrchestrator(
                llm_client=MockLLMClient(),
                max_retries=config.get("max_retries", 3),
                temperature=0.0,
            )
            st.info("ℹ️ Orchestrator automatically initialized in Mock Mode.")
        except Exception as e:
            st.error(f"Failed to auto-initialize orchestrator: {str(e)}")
            return None

    from harness.contracts import TaskSpec
    from harness.validators import ValidatorRegistry

    validators = ValidatorRegistry()

    # Build JSON schema if Pydantic/JSON schema enabled
    json_schema = None
    if task_spec["use_pydantic"] and task_spec["schema"]:
        try:
            json_schema = json.loads(task_spec["schema"])
        except json.JSONDecodeError:
            st.error("Invalid JSON schema")
            return None

    # Build regex patterns list
    regex_patterns = []
    if task_spec["use_regex"] and task_spec["regex"]:
        regex_patterns = [task_spec["regex"]]

    # Build TaskSpec with proper fields
    task = TaskSpec(
        task_name="dashboard_task",
        description=task_spec.get("category", ""),
        system_prompt="",
        json_schema=json_schema,
        regex_patterns=regex_patterns,
        keyword_blacklist=task_spec.get("blacklist", []),
        keyword_whitelist=task_spec.get("whitelist", []),
        min_length=task_spec.get("min_length") if task_spec.get("use_length") else None,
        max_length=task_spec.get("max_length") if task_spec.get("use_length") else None,
        custom_validators=[],
    )
    st.session_state.task_spec = task

    # Register declarative validators with the registry
    validators.register_declarative(task)

    with st.spinner("Running orchestration..."):
        try:
            cache_manager = st.session_state.cache_manager
            validator_set_hash = validators.validator_set_hash
            provider = config.get("provider", "mock")
            model = config.get("model", "llama-3.1-8b-instant")
            prompt = task_spec["prompt"]

            cached_resp = None
            if config.get("use_cache", True) and cache_manager:
                cached_resp = cache_manager.try_get_cached(
                    prompt=prompt,
                    model=model,
                    provider=provider,
                    validator_set_hash=validator_set_hash,
                )

            if cached_resp:
                st.session_state.cache_hit = True
                result = OrchestrationResult(
                    success=True,
                    final_output=cached_resp.content,
                    final_validator_results=[
                        ValidatorResult(passed=True, reason="Loaded from selective SQLite cache", severity=SeverityEnum.INFO)
                    ],
                    final_semantic_result=None,
                    final_critic_feedback=CriticFeedback(passed=True, guidance="Cached high-reliability response (Reliability >= 0.80)"),
                    retry_attempts=[],
                    total_attempts=1,
                    total_latency_ms=cached_resp.latency_ms,
                    reliability_score=1.0,
                )
            else:
                st.session_state.cache_hit = False
                result = st.session_state.orchestrator.execute(task, prompt)
                if config.get("use_cache", True) and cache_manager and result.success:
                    cache_manager.try_cache_result(
                        prompt=prompt,
                        model=model,
                        provider=provider,
                        validator_set_hash=validator_set_hash,
                        result=result,
                    )

            st.session_state.current_result = result
            st.session_state.execution_history.append({
                "timestamp": datetime.now().isoformat(),
                "task": prompt[:100],
                "result": result,
            })
            return result
        except Exception as e:
            st.error(f"Orchestration failed: {str(e)}")
            return None


def render_generation_panel(result: OrchestrationResult):
    """Render real-time generation and retry panel."""
    st.subheader("🔄 Generation & Retry Timeline")

    st.markdown("**Final Output**")
    st.code(result.final_output, language="json")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Success", "✅" if result.success else "❌")
    with col2:
        st.metric("Retries", result.retry_count)
    with col3:
        st.metric("Reliability", f"{result.reliability_score:.3f}")
    with col4:
        st.metric("Latency", f"{result.total_latency_ms:.0f} ms")

    if hasattr(result, 'trace') and result.trace:
        render_retry_timeline(result.trace)
    elif result.retry_attempts:
        st.markdown(f"**Execution History ({len(result.retry_attempts)} Attempt{'s' if len(result.retry_attempts) > 1 else ''})**")
        for attempt in result.retry_attempts:
            is_initial = attempt.attempt_number == 1
            label = "Attempt 1 (Initial Generation)" if is_initial else f"Attempt {attempt.attempt_number} (Retry #{attempt.attempt_number - 1})"
            all_passed = all(v.passed for v in attempt.validator_results)
            status_icon = "✅ Passed" if all_passed else "❌ Failed"

            with st.expander(f"{label} — {status_icon} — ⏱️ {attempt.latency_ms:.0f}ms"):
                st.markdown("**Generated Output:**")
                st.code(attempt.generated_output, language="json")
                if attempt.critic_feedback and attempt.critic_feedback.guidance:
                    st.info(f"**Critic Guidance:** {attempt.critic_feedback.guidance}")
    elif result.retry_count > 0:
        st.info("Retry details available in Audit Trail tab")


def render_retry_timeline(trace):
    """Render the retry timeline visualization."""
    st.markdown("**Retry Timeline**")

    for i, event in enumerate(trace.events):
        is_final = i == len(trace.events) - 1
        is_failed = event.event_type == "retry_failed"

        step_class = "retry-step"
        if is_final and not is_failed:
            step_class += " final"
        elif is_failed:
            step_class += " failed"

        with st.container():
            st.markdown(f'<div class="{step_class}">', unsafe_allow_html=True)

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{event.event_type.replace('_', ' ').title()}**")
                if event.details:
                    for key, value in event.details.items():
                        st.caption(f"{key}: {value}")
            with col2:
                st.caption(f"⏱️ {event.timestamp:.3f}s")
                if event.latency_ms:
                    st.caption(f"Latency: {event.latency_ms:.0f}ms")

            st.markdown('</div>', unsafe_allow_html=True)


def render_validator_panel(result: OrchestrationResult):
    """Render validator pass/fail badges and detailed breakdowns."""
    st.subheader("✅ Validator Results")

    val_results = getattr(result, "final_validator_results", getattr(result, "validator_results", []))
    if not val_results:
        st.info("No validator results available")
        return

    passed = sum(1 for v in val_results if v.passed)
    failed = len(val_results) - passed

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Passed", passed)
    with col2:
        st.metric("Failed", failed)
    with col3:
        st.metric("Pass Rate", f"{passed / max(1, len(val_results)) * 100:.0f}%")

    st.markdown("**Validator Details**")
    for idx, vr in enumerate(val_results):
        v_name = getattr(vr, "validator_name", None)
        if not v_name:
            if vr.failure_type:
                v_name = f"Check: {vr.failure_type.value}"
            else:
                v_name = f"Validator #{idx + 1}"
        status = "✅ PASS" if vr.passed else "❌ FAIL"
        sev_val = vr.severity.value if hasattr(vr.severity, "value") else str(vr.severity)

        with st.expander(f"{status} — {v_name} ({sev_val})"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Reason:** {vr.reason}")
                st.write(f"**Severity:** {sev_val}")
            with col_b:
                if vr.failure_type:
                    ft_val = vr.failure_type.value if hasattr(vr.failure_type, "value") else str(vr.failure_type)
                    st.write(f"**Failure Type:** {ft_val}")
            details = getattr(vr, "details", None)
            if details:
                st.json(details)

    if getattr(result, "final_semantic_result", None):
        sem = result.final_semantic_result
        st.markdown("**Semantic Scorer**")
        sem_status = "✅ PASS" if sem.passed else "❌ FAIL"
        with st.expander(f"{sem_status} — Semantic Similarity Assessment"):
            st.write(f"**Reason:** {sem.reason}")
            st.write(f"**Severity:** {sem.severity.value if hasattr(sem.severity, 'value') else str(sem.severity)}")


def render_critic_panel(result: OrchestrationResult):
    """Render critic diagnostic feedback cards."""
    st.subheader("🔍 Critic Diagnostic Feedback")

    cf: Optional[CriticFeedback] = getattr(result, "final_critic_feedback", getattr(result, "critic_feedback", None))
    if not cf:
        st.info("No critic feedback available")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        if cf.passed:
            st.success("✅ Critic Assessment: Output passed validation checks cleanly!")
        else:
            st.warning("⚠️ Critic Assessment: Corrective guidance applied during evaluation.")
    with col2:
        if cf.escalate_to_negative_constraints:
            st.markdown('<span class="validator-badge-fail">🚨 Negative Constraints Escalated</span>', unsafe_allow_html=True)

    if cf.guidance:
        st.markdown(f'**Guidance:** <div class="critic-guidance">{cf.guidance}</div>', unsafe_allow_html=True)

    if cf.constraints:
        st.markdown("**Active Constraints:**")
        for c in cf.constraints:
            st.markdown(f'<div class="constraint-box">{c}</div>', unsafe_allow_html=True)

    if cf.diagnostics:
        st.markdown(f"**Diagnostic Items ({len(cf.diagnostics)})**")
        for idx, diag in enumerate(cf.diagnostics):
            sev_str = diag.severity.value if hasattr(diag.severity, "value") else str(diag.severity)
            ft_str = diag.failure_type.value if hasattr(diag.failure_type, "value") else str(diag.failure_type)

            with st.container():
                st.markdown('<div class="audit-card">', unsafe_allow_html=True)
                st.markdown('<div class="audit-card-header">', unsafe_allow_html=True)
                st.markdown(f'<span class="audit-card-title">Diagnostic #{idx + 1}: {diag.validator_name}</span>', unsafe_allow_html=True)
                st.markdown(f'<span class="attempt-badge">Severity: {sev_str}</span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                col_d1, col_d2 = st.columns([2, 1])
                with col_d1:
                    st.markdown(f"**Failure Type:** `{ft_str}`")
                    st.markdown(f"**Message:** {diag.message}")
                with col_d2:
                    if diag.suggested_correction:
                        st.markdown(f"**Suggested Correction:** {diag.suggested_correction}")

                st.markdown('</div>', unsafe_allow_html=True)
    elif cf.passed:
        st.info("No diagnostic warnings or failures detected.")

    # Show retry feedback history if multiple attempts occurred
    if result.retry_attempts and len(result.retry_attempts) > 1:
        with st.expander(f"📜 View Critic Feedback Across All {len(result.retry_attempts)} Attempts"):
            for att in result.retry_attempts:
                st.markdown(f"**Attempt {att.attempt_number} (Latency: {att.latency_ms:.0f}ms)**")
                if att.critic_feedback:
                    st.caption(f"Guidance: {att.critic_feedback.guidance or 'None'}")
                    if att.critic_feedback.diagnostics:
                        for d in att.critic_feedback.diagnostics:
                            st.caption(f"• [{d.severity.value}] {d.validator_name}: {d.message}")
                st.divider()


def render_cache_panel(result: OrchestrationResult, config: Dict[str, Any]):
    """Render cache metrics and hit/miss status."""
    st.subheader("💾 Cache Metrics")

    cache_manager = st.session_state.cache_manager
    if cache_manager and config["use_cache"]:
        stats = cache_manager.get_stats()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Cache Entries", stats["total_entries"])
        with col2:
            hit_rate = (stats["entries_above_threshold"] / max(1, stats["total_entries"]) * 100)
            st.metric("Hit Rate", f"{hit_rate:.1f}%")
        with col3:
            st.metric("Avg Reliability", f"{stats['average_reliability']:.3f}")
        with col4:
            st.metric("Size", f"{stats['size_mb']:.2f} MB")

        st.markdown("**Cache Status for Last Query**")
        cache_status = st.session_state.cache_hit
        if cache_status:
            st.markdown('<span class="cache-hit">✅ CACHE HIT</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="cache-miss">❌ CACHE MISS</span>', unsafe_allow_html=True)

        if st.button("🗑️ Clear Cache"):
            cache_manager.cache.clear()
            st.success("Cache cleared!")
            st.rerun()
    else:
        st.info("Cache disabled")

    st.markdown("**Cache Configuration**")
    st.write(f"Reliability Threshold: {cache_manager.reliability_threshold if cache_manager else 'N/A'}")
    st.write(f"DB Path: {cache_manager.cache.db_path if cache_manager and cache_manager.cache else 'N/A'}")


def render_execution_history():
    """Render execution history."""
    if not st.session_state.execution_history:
        st.info("No execution history yet")
        return

    st.subheader("📜 Execution History")

    for i, entry in enumerate(reversed(st.session_state.execution_history)):
        result = entry["result"]
        with st.expander(
            f"{entry['timestamp'][:19]} | {entry['task']} | "
            f"{'✅' if result.success else '❌'} | {result.reliability_score:.3f} | "
            f"{result.retry_count} retries"
        ):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Retries", result.retry_count)
            with col2:
                st.metric("Reliability", f"{result.reliability_score:.3f}")
            with col3:
                st.metric("Latency", f"{result.total_latency_ms:.0f}ms")

            if st.button(f"Load #{len(st.session_state.execution_history) - i}", key=f"load_{i}"):
                st.session_state.current_result = result
                st.rerun()


def render_summary_dashboard():
    """Render overall summary dashboard."""
    st.subheader("📈 Overall Summary")

    history = st.session_state.execution_history
    if not history:
        st.info("No execution history available")
        return

    total = len(history)
    successful = sum(1 for e in history if e["result"].success)
    failed = total - successful
    avg_reliability = sum(e["result"].reliability_score for e in history) / total
    avg_retries = sum(e["result"].retry_count for e in history) / total
    avg_latency = sum(e["result"].total_latency_ms for e in history) / total

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Executions", total)
    with col2:
        st.metric("Success Rate", f"{successful / total * 100:.1f}%")
    with col3:
        st.metric("Avg Reliability", f"{avg_reliability:.3f}")
    with col4:
        st.metric("Avg Retries", f"{avg_retries:.2f}")
    with col5:
        st.metric("Avg Latency", f"{avg_latency:.0f} ms")


def main():
    """Main Streamlit application."""
    init_session_state()
    render_header()

    config = render_sidebar()

    render_summary_dashboard()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Query Runner",
        "🔄 Generation",
        "✅ Validators",
        "🔍 Critic Audit",
        "💾 Cache & Metrics"
    ])

    with tab1:
        st.markdown("### Interactive Query Runner")
        task_spec = render_task_spec_input()

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("▶️ Execute", type="primary", use_container_width=True):
                result = run_orchestration(task_spec, config)
                if result:
                    st.rerun()
        with col2:
            if st.session_state.current_result:
                result = st.session_state.current_result
                status = "✅ Success" if result.success else "❌ Failed"
                st.info(f"Last run: {status} | Reliability: {result.reliability_score:.3f} | Retries: {result.retry_count}")

    with tab2:
        if st.session_state.current_result:
            render_generation_panel(st.session_state.current_result)
        else:
            st.info("Execute a query to see generation timeline")

    with tab3:
        if st.session_state.current_result:
            render_validator_panel(st.session_state.current_result)
        else:
            st.info("Execute a query to see validator results")

    with tab4:
        if st.session_state.current_result:
            render_critic_panel(st.session_state.current_result)
        else:
            st.info("Execute a query to see critic feedback")

    with tab5:
        if st.session_state.current_result:
            render_cache_panel(st.session_state.current_result, config)
        else:
            st.info("Execute a query to see cache metrics")

        render_execution_history()


if __name__ == "__main__":
    main()