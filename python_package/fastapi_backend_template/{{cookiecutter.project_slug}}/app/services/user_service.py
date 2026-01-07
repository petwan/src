"""User service."""
from typing import Any, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictException, NotFoundException
from app.core.security import get_password_hash
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate


class UserService:
    """User service class."""

    def __init__(self, db: AsyncSession) -> None:
        """Initialize user service."""
        self.db = db

    async def get(self, user_id: int) -> User | None:
        """Get user by ID."""
        result = await self.db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        """Get user by email."""
        result = await self.db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def get_multi(
        self,
        skip: int = 0,
        limit: int = 10,
    ) -> Tuple[list[User], int]:
        """Get multiple users with pagination."""
        # Get total count
        count_result = await self.db.execute(select(User).count())
        total = count_result.scalar() or 0
        
        # Get users with pagination
        result = await self.db.execute(
            select(User).offset(skip).limit(limit).order_by(User.created_at.desc()),
        )
        users = result.scalars().all()
        
        return list(users), total

    async def create(self, user_in: UserCreate) -> User:
        """Create new user."""
        # Check if user already exists
        existing_user = await self.get_by_email(user_in.email)
        if existing_user:
            raise ConflictException("User with this email already exists")
        
        # Create new user
        user = User(
            email=user_in.email,
            full_name=user_in.full_name,
            hashed_password=get_password_hash(user_in.password),
        )
        
        self.db.add(user)
        await self.db.flush()
        await self.db.refresh(user)
        
        return user

    async def update(self, user_id: int, user_in: UserUpdate) -> User | None:
        """Update user."""
        user = await self.get(user_id)
        if not user:
            return None
        
        update_data = user_in.model_dump(exclude_unset=True)
        
        if "password" in update_data and update_data["password"]:
            update_data["hashed_password"] = get_password_hash(update_data.pop("password"))
        
        for field, value in update_data.items():
            setattr(user, field, value)
        
        await self.db.flush()
        await self.db.refresh(user)
        
        return user

    async def delete(self, user_id: int) -> bool:
        """Delete user."""
        user = await self.get(user_id)
        if not user:
            return False
        
        await self.db.delete(user)
        await self.db.flush()
        
        return True

    async def authenticate(self, email: str, password: str) -> User | None:
        """Authenticate user."""
        from app.core.security import verify_password
        
        user = await self.get_by_email(email)
        if not user:
            return None
        
        if not verify_password(password, user.hashed_password):
            return None
        
        return user
