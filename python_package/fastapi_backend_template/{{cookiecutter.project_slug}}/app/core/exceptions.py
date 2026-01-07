"""Custom exception classes."""
from typing import Any


class AppException(Exception):
    """Base application exception."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)


class NotFoundException(AppException):
    """Resource not found exception."""

    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(message, status_code=404)


class BadRequestException(AppException):
    """Bad request exception."""

    def __init__(self, message: str = "Bad request") -> None:
        super().__init__(message, status_code=400)


class UnauthorizedException(AppException):
    """Unauthorized exception."""

    def __init__(self, message: str = "Unauthorized") -> None:
        super().__init__(message, status_code=401)


class ForbiddenException(AppException):
    """Forbidden exception."""

    def __init__(self, message: str = "Forbidden") -> None:
        super().__init__(message, status_code=403)


class ConflictException(AppException):
    """Conflict exception."""

    def __init__(self, message: str = "Resource already exists") -> None:
        super().__init__(message, status_code=409)


class ValidationException(AppException):
    """Validation exception."""

    def __init__(self, message: str = "Validation error", details: dict[str, Any] | None = None) -> None:
        super().__init__(message, status_code=422, details=details)
