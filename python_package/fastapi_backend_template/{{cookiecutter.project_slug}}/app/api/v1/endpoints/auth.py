"""Authentication endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import DatabaseDep
from app.core.security import create_access_token, create_refresh_token, verify_password
from app.schemas.auth import LoginRequest, LoginResponse, RefreshTokenRequest, RefreshTokenResponse, UserResponse
from app.services.user_service import UserService

router = APIRouter()


@router.post("/login", response_model=LoginResponse)
async def login(
    credentials: LoginRequest,
    db: DatabaseDep,
) -> LoginResponse:
    """Login endpoint."""
    user_service = UserService(db)
    user = await user_service.get_by_email(credentials.email)
    
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    
    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token(user.id)
    
    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserResponse.model_validate(user),
    )


@router.post("/refresh", response_model=RefreshTokenResponse)
async def refresh_token(
    refresh_data: RefreshTokenRequest,
) -> RefreshTokenResponse:
    """Refresh access token."""
    from jose import jwt, ExpiredSignatureError
    from app.core.config import settings
    
    try:
        payload = jwt.decode(
            refresh_data.refresh_token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )
        
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )
        
        access_token = create_access_token(user_id)
        
        return RefreshTokenResponse(access_token=access_token)
    
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired",
        )
