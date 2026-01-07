"""User schemas."""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field, field_validator


class UserBase(BaseModel):
    """Base user schema."""

    email: EmailStr
    full_name: str = Field(min_length=1, max_length=255)


class UserCreate(UserBase):
    """User creation schema."""

    password: str = Field(min_length=8, max_length=100)

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserUpdate(BaseModel):
    """User update schema."""

    full_name: str | None = Field(None, min_length=1, max_length=255)
    password: str | None = Field(None, min_length=8, max_length=100)


class UserResponse(UserBase):
    """User response schema."""

    id: int
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    """User list response schema."""

    items: list[UserResponse]
    total: int
    skip: int
    limit: int
