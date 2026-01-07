# API Documentation

## Authentication

Most endpoints require authentication via JWT token in the Authorization header:

```
Authorization: Bearer <your_access_token>
```

## Endpoints

### Health

#### GET /
Root endpoint

**Response:**
```json
{
  "message": "Welcome to {{ cookiecutter.project_name }}",
  "docs": "{{ cookiecutter.api_prefix }}/docs",
  "version": "0.1.0"
}
```

#### GET /health
Health check endpoint

**Response:**
```json
{
  "status": "healthy"
}
```

### Authentication

#### POST {{ cookiecutter.api_prefix }}/auth/login
Login with email and password

**Request:**
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**Response:**
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "user": {
    "id": 1,
    "email": "user@example.com",
    "full_name": "John Doe",
    "is_active": true,
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

#### POST {{ cookiecutter.api_prefix }}/auth/refresh
Refresh access token

**Request:**
```json
{
  "refresh_token": "eyJ..."
}
```

**Response:**
```json
{
  "access_token": "eyJ..."
}
```

### Users

#### GET {{ cookiecutter.api_prefix }}/users
Get list of users (paginated)

**Query Parameters:**
- `skip` (optional, default: 0) - Number of users to skip
- `limit` (optional, default: 10, max: 100) - Number of users to return

**Response:**
```json
{
  "items": [
    {
      "id": 1,
      "email": "user@example.com",
      "full_name": "John Doe",
      "is_active": true,
      "is_superuser": false,
      "created_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-01T00:00:00Z"
    }
  ],
  "total": 1,
  "skip": 0,
  "limit": 10
}
```

#### GET {{ cookiecutter.api_prefix }}/users/me
Get current user (requires authentication)

**Response:**
```json
{
  "id": 1,
  "email": "user@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "is_superuser": false,
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

#### GET {{ cookiecutter.api_prefix }}/users/{user_id}
Get user by ID

**Response:**
```json
{
  "id": 1,
  "email": "user@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "is_superuser": false,
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

#### POST {{ cookiecutter.api_prefix }}/users
Create new user

**Request:**
```json
{
  "email": "user@example.com",
  "full_name": "John Doe",
  "password": "SecurePass123"
}
```

**Response:**
```json
{
  "id": 1,
  "email": "user@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "is_superuser": false,
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:00Z"
}
```

#### PATCH {{ cookiecutter.api_prefix }}/users/{user_id}
Update user (requires authentication and ownership)

**Request:**
```json
{
  "full_name": "Jane Doe"
}
```

**Response:**
```json
{
  "id": 1,
  "email": "user@example.com",
  "full_name": "Jane Doe",
  "is_active": true,
  "is_superuser": false,
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T12:00:00Z"
}
```

#### DELETE {{ cookiecutter.api_prefix }}/users/{user_id}
Delete user (requires authentication and ownership)

**Response:** 204 No Content

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Bad request"
}
```

### 401 Unauthorized
```json
{
  "detail": "Could not validate credentials"
}
```

### 403 Forbidden
```json
{
  "detail": "Not enough permissions"
}
```

### 404 Not Found
```json
{
  "detail": "Resource not found"
}
```

### 409 Conflict
```json
{
  "detail": "User with this email already exists"
}
```

### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "email"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```
