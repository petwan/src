"""Common utility functions."""
from typing import Any


def mask_email(email: str) -> str:
    """Mask email address."""
    username, domain = email.split("@")
    if len(username) <= 2:
        masked = username[0] + "*" * (len(username) - 1)
    else:
        masked = username[0] + "*" * (len(username) - 2) + username[-1]
    return f"{masked}@{domain}"


def mask_phone(phone: str) -> str:
    """Mask phone number."""
    if len(phone) < 7:
        return phone
    return phone[:3] + "*" * (len(phone) - 6) + phone[-3:]


def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to max length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def safe_get(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    """Safely get nested value from dictionary."""
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current
