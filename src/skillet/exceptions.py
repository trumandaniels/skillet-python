from __future__ import annotations

from typing import Any


class SkilletError(Exception):
    """Base exception for all Skillet SDK errors.

    All exceptions raised by the SDK inherit from this class, so you can catch
    any Skillet error with a single ``except SkilletError`` clause.

    Args:
        message: Human-readable error description.
        status_code: HTTP status code returned by the API, if applicable.
        detail: Raw detail payload from the API response body.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        detail: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class AuthenticationError(SkilletError):
    """Raised when the request lacks valid credentials (HTTP 401).

    Occurs when no API key or bearer token is provided, or the credentials
    have been revoked or expired.
    """


class AuthorizationError(SkilletError):
    """Raised when the credentials are valid but lack the required permissions (HTTP 403).

    Occurs when, for example, a bearer token and API key from different tenants
    are sent together.
    """


class RateLimitError(SkilletError):
    """Raised when the monthly plan quota has been exhausted (HTTP 402 or 429).

    Inspect ``error.status_code`` to distinguish between quota exhaustion (402)
    and transient rate limiting (429).
    """


class ValidationError(SkilletError):
    """Raised when the request body fails server-side validation (HTTP 422).

    ``error.detail`` contains a list of Pydantic validation errors describing
    which fields are invalid and why.
    """


class ApiError(SkilletError):
    """Raised for unexpected API errors (HTTP 5xx or unrecognised 4xx).

    Indicates a server-side problem or an unhandled edge case.
    """


class TransportError(SkilletError):
    """Raised when the HTTP request cannot be completed due to a network issue.

    Wraps ``httpx.RequestError`` (connection refused, timeout, DNS failure, etc.).
    The original exception is available as ``error.detail``.
    """

    def __init__(self, message: str, *, detail: Any | None = None) -> None:
        super().__init__(message, detail=detail)
