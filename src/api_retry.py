"""Retry policy for transient T-Invest gRPC failures."""

from functools import wraps
from typing import Any, Callable, TypeVar

import grpc
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

try:
    from t_tech.invest.exceptions import RequestError
except ImportError:
    RequestError = Exception


ReturnT = TypeVar("ReturnT")


class ApiRateLimitError(RuntimeError):
    """Raised when T-Invest reports RESOURCE_EXHAUSTED."""


def _status_code(error: BaseException):
    """Extract a gRPC status code from an SDK RequestError."""
    status = getattr(error, "status_code", None)
    if status is not None:
        return status
    if getattr(error, "args", None):
        return error.args[0]
    return None


def _is_retryable(error: BaseException) -> bool:
    """Return true only for transient unavailable/deadline errors."""
    status = _status_code(error)
    if status == grpc.StatusCode.RESOURCE_EXHAUSTED:
        return False
    return status in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED)


def _raise_rate_limit(error: BaseException) -> None:
    """Convert RESOURCE_EXHAUSTED into a clear application exception."""
    if _status_code(error) == grpc.StatusCode.RESOURCE_EXHAUSTED:
        raise ApiRateLimitError("T-Invest API rate limit reached") from error


def retry_api_call(function: Callable[..., ReturnT]) -> Callable[..., ReturnT]:
    """Retry a single SDK call up to four times with 1/2/4/8 second backoff."""
    @wraps(function)
    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def wrapped(*args: Any, **kwargs: Any) -> ReturnT:
        try:
            return function(*args, **kwargs)
        except RequestError as error:
            _raise_rate_limit(error)
            raise

    return wrapped
