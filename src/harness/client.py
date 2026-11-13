"""Pluggable LLM inference clients for Agentic Harness v1.2."""
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx

from .contracts import LLMResponse
from .exceptions import ProviderError, RateLimitError

# Model note: In mid-2026, Groq deprecated llama-3.1-8b-instant and llama-3.3-70b-versatile.
# openai/gpt-oss-20b is the recommended fast, low-cost default with structured output support.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"


class LLMClient(ABC):
    """Abstract base class for all pluggable LLM inference clients."""

    def __init__(self, default_model: str):
        self.default_model = default_model

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """Generate a response from the LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            max_tokens: Maximum tokens in completion.
            temperature: Sampling temperature. MUST be 0.0 for determinism.
            model: Optional model override. Defaults to self.default_model.

        Returns:
            LLMResponse containing generated content, metadata, and token usage.

        Raises:
            ValueError: If temperature != 0.0 (determinism invariant).
            RateLimitError: If provider returns HTTP 429 (Too Many Requests).
            ProviderError: If inference fails.
        """
        pass

    def _verify_temperature(self, temperature: float) -> None:
        """Enforces temperature=0.0 determinism invariant."""
        if temperature != 0.0:
            raise ValueError(
                f"Determinism violation: temperature must be 0.0, got {temperature}. "
                "Non-zero temperature breaks reproducible reliability scoring."
            )


class GroqClient(LLMClient):
    """Inference client wrapping Groq Cloud API with zero transport retries."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_GROQ_MODEL,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout: float = 60.0,
    ):
        super().__init__(default_model=model_name)
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # Explicitly configure HTTP client with retries=0 to prevent transport multiplication
        self._transport = httpx.HTTPTransport(retries=0)
        self._http_client = httpx.Client(
            transport=self._transport,
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        model: Optional[str] = None,
    ) -> LLMResponse:
        self._verify_temperature(temperature)
        active_model = model or self.default_model

        payload = {
            "model": active_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }

        start_time = time.perf_counter()
        try:
            response = self._http_client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
        except Exception as exc:
            raise ProviderError(f"Groq network connection error: {exc}") from exc

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if response.status_code == 429:
            retry_after_header = response.headers.get("retry-after")
            retry_after_val: Optional[float] = None
            if retry_after_header:
                try:
                    retry_after_val = float(retry_after_header)
                except ValueError:
                    retry_after_val = None
            raise RateLimitError(
                f"Groq rate limit exceeded (HTTP 429): {response.text}",
                retry_after=retry_after_val,
            )

        if response.status_code != 200:
            raise ProviderError(
                f"Groq API returned HTTP {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            return LLMResponse(
                content=content,
                model=active_model,
                provider="groq",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=elapsed_ms,
                raw_response=data,
            )
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderError(f"Failed to parse Groq response structure: {exc}") from exc

    def close(self) -> None:
        self._http_client.close()


class OllamaClient(LLMClient):
    """Inference client wrapping local Ollama instance with zero transport retries."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/api",
        model_name: str = DEFAULT_OLLAMA_MODEL,
        timeout: float = 120.0,
    ):
        super().__init__(default_model=model_name)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._transport = httpx.HTTPTransport(retries=0)
        self._http_client = httpx.Client(
            transport=self._transport,
            timeout=self.timeout,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        model: Optional[str] = None,
    ) -> LLMResponse:
        self._verify_temperature(temperature)
        active_model = model or self.default_model

        payload = {
            "model": active_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": max_tokens,
            },
        }

        start_time = time.perf_counter()
        try:
            response = self._http_client.post(
                f"{self.base_url}/chat",
                json=payload,
            )
        except Exception as exc:
            raise ProviderError(f"Ollama connection error: {exc}") from exc

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if response.status_code != 200:
            raise ProviderError(
                f"Ollama returned HTTP {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
            content = data["message"]["content"]
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)

            return LLMResponse(
                content=content,
                model=active_model,
                provider="ollama",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=elapsed_ms,
                raw_response=data,
            )
        except (KeyError, ValueError) as exc:
            raise ProviderError(f"Failed to parse Ollama response: {exc}") from exc

    def close(self) -> None:
        self._http_client.close()


def get_llm_client(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> LLMClient:
    """Factory creating an LLMClient instance based on configuration or environment.
    
    Args:
        provider: 'groq' or 'ollama'. Defaults to LLM_PROVIDER env var, then 'groq'.
        model_name: Optional model override.
        **kwargs: Additional provider-specific kwargs.

    Returns:
        LLMClient instance.
    """
    active_provider = (
        provider or os.environ.get("LLM_PROVIDER", "groq")
    ).strip().lower()

    if active_provider == "groq":
        return GroqClient(model_name=model_name or DEFAULT_GROQ_MODEL, **kwargs)
    elif active_provider == "ollama":
        return OllamaClient(model_name=model_name or DEFAULT_OLLAMA_MODEL, **kwargs)
    else:
        raise ValueError(
            f"Unsupported LLM provider '{active_provider}'. Supported: 'groq', 'ollama'."
        )
