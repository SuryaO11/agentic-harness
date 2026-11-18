"""Custom exception hierarchy for Agentic Harness."""
from typing import Optional


class HarnessError(Exception):
    """Base exception for all Agentic Harness errors."""
    pass


class ProviderError(HarnessError):
    """Raised when an external inference provider fails or returns an error."""
    pass


class RateLimitError(HarnessError):
    """Raised when an inference provider returns HTTP 429 (Too Many Requests).
    
    Attributes:
        retry_after: Number of seconds to wait before retrying (extracted from Retry-After header).
    """

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after

