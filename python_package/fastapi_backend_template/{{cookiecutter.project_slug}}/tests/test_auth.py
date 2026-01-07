"""Authentication endpoint tests."""
import pytest

from httpx import AsyncClient


@pytest.mark.asyncio
async def test_login_success(client: AsyncClient) -> None:
    """Test successful login."""
    # Create user first
    await client.post(
        "/api/v1/users",
        json={
            "email": "login@example.com",
            "full_name": "Login User",
            "password": "TestPass123",
        },
    )
    
    # Login
    response = await client.post(
        "/api/v1/auth/login",
        json={
            "email": "login@example.com",
            "password": "TestPass123",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert "user" in data


@pytest.mark.asyncio
async def test_login_invalid_credentials(client: AsyncClient) -> None:
    """Test login with invalid credentials."""
    response = await client.post(
        "/api/v1/auth/login",
        json={
            "email": "nonexistent@example.com",
            "password": "WrongPass123",
        },
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_unauthorized(client: AsyncClient) -> None:
    """Test getting current user without authentication."""
    response = await client.get("/api/v1/users/me")
    assert response.status_code == 401
