"""User endpoint tests."""
import pytest

from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_user(client: AsyncClient) -> None:
    """Test user creation."""
    response = await client.post(
        "/api/v1/users",
        json={
            "email": "test@example.com",
            "full_name": "Test User",
            "password": "TestPass123",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
    assert data["full_name"] == "Test User"
    assert "id" in data


@pytest.mark.asyncio
async def test_create_user_duplicate_email(client: AsyncClient) -> None:
    """Test duplicate email handling."""
    user_data = {
        "email": "duplicate@example.com",
        "full_name": "Duplicate User",
        "password": "TestPass123",
    }
    
    # Create first user
    response1 = await client.post("/api/v1/users", json=user_data)
    assert response1.status_code == 201
    
    # Try to create duplicate
    response2 = await client.post("/api/v1/users", json=user_data)
    assert response2.status_code == 409


@pytest.mark.asyncio
async def test_get_users(client: AsyncClient) -> None:
    """Test getting users list."""
    # Create test users
    for i in range(3):
        await client.post(
            "/api/v1/users",
            json={
                "email": f"user{i}@example.com",
                "full_name": f"User {i}",
                "password": "TestPass123",
            },
        )
    
    response = await client.get("/api/v1/users")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert len(data["items"]) >= 3


@pytest.mark.asyncio
async def test_get_user_by_id(client: AsyncClient) -> None:
    """Test getting user by ID."""
    # Create user
    create_response = await client.post(
        "/api/v1/users",
        json={
            "email": "getbyid@example.com",
            "full_name": "Get By ID",
            "password": "TestPass123",
        },
    )
    user_id = create_response.json()["id"]
    
    # Get user
    response = await client.get(f"/api/v1/users/{user_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == user_id
    assert data["email"] == "getbyid@example.com"


@pytest.mark.asyncio
async def test_update_user(client: AsyncClient) -> None:
    """Test updating user."""
    # Create user
    create_response = await client.post(
        "/api/v1/users",
        json={
            "email": "update@example.com",
            "full_name": "Update Me",
            "password": "TestPass123",
        },
    )
    user_id = create_response.json()["id"]
    
    # Update user (without auth, this should fail)
    response = await client.patch(
        f"/api/v1/users/{user_id}",
        json={"full_name": "Updated Name"},
    )
    assert response.status_code == 401  # Unauthorized


@pytest.mark.asyncio
async def test_weak_password_validation(client: AsyncClient) -> None:
    """Test weak password validation."""
    response = await client.post(
        "/api/v1/users",
        json={
            "email": "weak@example.com",
            "full_name": "Weak Password",
            "password": "weak",
        },
    )
    assert response.status_code == 422
