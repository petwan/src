"""Authentication schemas."""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    """Login request."""

    email: EmailStr
    password: str = Field(min_length=8)


class UserResponse(BaseModel):
    """User response."""

    id: int
    email: str
    full_name: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class LoginResponse(BaseModel):
    """Login response."""

    access_token: str
    refresh_token: str
    user: UserResponse


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""

    refresh_token: str


class RefreshTokenResponse(BaseModel):
    """Refresh token response."""

    access_token: str
