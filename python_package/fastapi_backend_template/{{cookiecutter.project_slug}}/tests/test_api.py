"""API endpoint tests."""
from httpx import AsyncClient


async def test_root(client: AsyncClient) -> None:
    """Test root endpoint."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


async def test_health_check(client: AsyncClient) -> None:
    """Test health check endpoint."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


async def test_api_health(client: AsyncClient) -> None:
    """Test API health check endpoint."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
