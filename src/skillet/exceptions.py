from __future__ import annotations

from typing import Any


class SkilletError(Exception):
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
    pass


class AuthorizationError(SkilletError):
    pass


class RateLimitError(SkilletError):
    pass


class ValidationError(SkilletError):
    pass


class ApiError(SkilletError):
    pass


class TransportError(SkilletError):
    def __init__(self, message: str, *, detail: Any | None = None) -> None:
        super().__init__(message, detail=detail)
