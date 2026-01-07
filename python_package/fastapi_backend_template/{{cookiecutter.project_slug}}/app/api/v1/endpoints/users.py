"""User endpoints."""
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import CurrentUserDep, DatabaseDep
from app.schemas.user import UserCreate, UserResponse, UserUpdate, UserListResponse
from app.services.user_service import UserService

router = APIRouter()


@router.get("", response_model=UserListResponse)
async def get_users(
    db: DatabaseDep,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
) -> UserListResponse:
    """Get list of users."""
    user_service = UserService(db)
    users, total = await user_service.get_multi(skip=skip, limit=limit)
    return UserListResponse(items=users, total=total, skip=skip, limit=limit)


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    current_user_id: CurrentUserDep,
    db: DatabaseDep,
) -> UserResponse:
    """Get current user."""
    user_service = UserService(db)
    user = await user_service.get(current_user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return UserResponse.model_validate(user)


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: DatabaseDep,
) -> UserResponse:
    """Get user by ID."""
    user_service = UserService(db)
    user = await user_service.get(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return UserResponse.model_validate(user)


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_in: UserCreate,
    db: DatabaseDep,
) -> UserResponse:
    """Create new user."""
    user_service = UserService(db)
    user = await user_service.create(user_in)
    return UserResponse.model_validate(user)


@router.patch("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_in: UserUpdate,
    db: DatabaseDep,
    current_user_id: CurrentUserDep,
) -> UserResponse:
    """Update user."""
    if current_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    
    user_service = UserService(db)
    user = await user_service.update(user_id, user_in)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return UserResponse.model_validate(user)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    db: DatabaseDep,
    current_user_id: CurrentUserDep,
) -> None:
    """Delete user."""
    if current_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    
    user_service = UserService(db)
    success = await user_service.delete(user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
