"""Unit tests for pluggable LLMClient implementations and factory."""
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.harness.client import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_OLLAMA_MODEL,
    GroqClient,
    OllamaClient,
    get_llm_client,
)
from src.harness.contracts import LLMResponse
from src.harness.exceptions import ProviderError, RateLimitError


def test_temperature_determinism_violation():
    client = GroqClient(api_key="mock_key")
    with pytest.raises(ValueError, match="Determinism violation: temperature must be 0.0"):
        client.generate(messages=[{"role": "user", "content": "hello"}], temperature=0.7)


def test_groq_client_zero_transport_retries():
    client = GroqClient(api_key="mock_key")
    assert isinstance(client._transport, httpx.HTTPTransport)
    # Verify the underlying transport was constructed with retries=0
    assert client._transport._pool._retries == 0


def test_groq_client_mock_successful_generate():
    client = GroqClient(api_key="mock_key", model_name="openai/gpt-oss-20b")

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Deterministic output"}}],
        "usage": {"prompt_tokens": 15, "completion_tokens": 5},
    }

    with patch.object(client._http_client, "post", return_value=mock_response):
        resp = client.generate(messages=[{"role": "user", "content": "hi"}], temperature=0.0)
        assert isinstance(resp, LLMResponse)
        assert resp.content == "Deterministic output"
        assert resp.model == "openai/gpt-oss-20b"
        assert resp.provider == "groq"
        assert resp.prompt_tokens == 15
        assert resp.completion_tokens == 5
        assert resp.latency_ms >= 0.0


def test_groq_client_mock_rate_limit_429():
    client = GroqClient(api_key="mock_key")

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 429
    mock_response.text = "Rate limit reached"
    mock_response.headers = {"retry-after": "42"}

    with patch.object(client._http_client, "post", return_value=mock_response):
        with pytest.raises(RateLimitError) as exc_info:
            client.generate(messages=[{"role": "user", "content": "hi"}], temperature=0.0)
        assert exc_info.value.retry_after == 42.0


def test_ollama_client_mock_successful_generate():
    client = OllamaClient(model_name="llama3.1:8b")

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {"content": "Ollama local response"},
        "prompt_eval_count": 20,
        "eval_count": 10,
    }

    with patch.object(client._http_client, "post", return_value=mock_response):
        resp = client.generate(messages=[{"role": "user", "content": "ping"}], temperature=0.0)
        assert isinstance(resp, LLMResponse)
        assert resp.content == "Ollama local response"
        assert resp.provider == "ollama"
        assert resp.prompt_tokens == 20
        assert resp.completion_tokens == 10


def test_factory_get_llm_client():
    # Test default Groq
    with patch.dict(os.environ, {"LLM_PROVIDER": "groq"}):
        c1 = get_llm_client()
        assert isinstance(c1, GroqClient)
        assert c1.default_model == DEFAULT_GROQ_MODEL

    # Test Ollama switch
    with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}):
        c2 = get_llm_client()
        assert isinstance(c2, OllamaClient)
        assert c2.default_model == DEFAULT_OLLAMA_MODEL

    # Test custom model override
    c3 = get_llm_client(provider="groq", model_name="llama-3.1-8b-instant")
    assert isinstance(c3, GroqClient)
    assert c3.default_model == "llama-3.1-8b-instant"

    # Test invalid provider
    with pytest.raises(ValueError, match="Unsupported LLM provider 'unknown'"):
        get_llm_client(provider="unknown")

