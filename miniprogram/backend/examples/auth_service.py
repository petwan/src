"""
示例：认证服务

展示如何实现完整的认证和授权逻辑，包括用户登录、令牌验证等。
"""

from datetime import datetime
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.dependencies import get_db
from app.core.exception import UnauthorizedException
from app.core.response import BaseResponse
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_password_hash,
    verify_password,
)
from examples.user_crud import CRUDUser
from examples.user_model import (
    AuthSchema,
    TokenResponse,
    User,
    UserCreate,
    UserLogin,
    UserResponse,
)

# OAuth2 密码流
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


class AuthService:
    """
    认证服务类

    处理用户登录、注册、令牌验证等认证相关逻辑。
    """

    def __init__(self, db: AsyncSession):
        """
        初始化认证服务

        Args:
            db: 数据库会话
        """
        self.db = db
        self.auth = AuthSchema(db=db)
        self.user_crud = CRUDUser(User, self.auth)

    async def register(self, user_data: UserCreate) -> User:
        """
        用户注册

        验证用户信息并创建新用户。

        Args:
            user_data: 用户注册数据

        Returns:
            User: 创建的用户对象

        Raises:
            HTTPException: 用户名或邮箱已存在时抛出
        """
        # 检查用户名是否已存在
        existing_user = await self.user_crud.get_by_username(user_data.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户名已存在",
            )

        # 检查邮箱是否已存在
        existing_user = await self.user_crud.get_by_email(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="邮箱已被注册",
            )

        # 创建用户（密码哈希处理）
        user_dict = user_data.model_dump()
        user_dict["hashed_password"] = get_password_hash(user_dict.pop("password"))

        new_user = await self.user_crud.create(user_dict)

        return new_user

    async def login(self, login_data: UserLogin) -> TokenResponse:
        """
        用户登录

        验证用户凭据并生成令牌。

        Args:
            login_data: 登录数据（用户名/邮箱 + 密码）

        Returns:
            TokenResponse: 包含访问令牌和刷新令牌

        Raises:
            HTTPException: 用户名或密码错误时抛出
        """
        # 查找用户（支持用户名或邮箱登录）
        user = await self.user_crud.get_by_username(login_data.username)
        if not user:
            user = await self.user_crud.get_by_email(login_data.username)

        # 验证密码
        if not user or not verify_password(login_data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户名或密码错误",
            )

        # 检查用户状态
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="账号已被禁用",
            )

        # 生成令牌
        access_token = create_access_token(subject=user.id)
        refresh_token = create_refresh_token(subject=user.id)

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
        )

    async def refresh_token(self, refresh_token: str) -> TokenResponse:
        """
        刷新访问令牌

        使用刷新令牌获取新的访问令牌。

        Args:
            refresh_token: 刷新令牌

        Returns:
            TokenResponse: 新的令牌对

        Raises:
            HTTPException: 令牌无效时抛出
        """
        # 解码刷新令牌
        payload = decode_token(refresh_token)
        if not payload or "sub" not in payload or payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的刷新令牌",
            )

        user_id = int(payload["sub"])

        # 验证用户是否存在
        user = await self.user_crud.get(id=user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用户不存在",
            )

        # 生成新令牌
        new_access_token = create_access_token(subject=user.id)
        new_refresh_token = create_refresh_token(subject=user.id)

        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
        )

    async def get_current_user(self, token: str) -> User:
        """
        获取当前登录用户

        验证令牌并返回用户信息。

        Args:
            token: 访问令牌

        Returns:
            User: 当前用户对象

        Raises:
            HTTPException: 令牌无效时抛出
        """
        # 解码令牌
        payload = decode_token(token)
        if not payload or "sub" not in payload:
            raise UnauthorizedException("无效的认证凭据")

        user_id = int(payload["sub"])

        # 获取用户
        user = await self.user_crud.get(id=user_id)
        if not user:
            raise UnauthorizedException("用户不存在")

        # 检查用户状态
        if not user.is_active:
            raise UnauthorizedException("账号已被禁用")

        return user

    async def change_password(
        self,
        user: User,
        old_password: str,
        new_password: str,
    ) -> None:
        """
        修改密码

        验证旧密码并更新为新密码。

        Args:
            user: 当前用户
            old_password: 旧密码
            new_password: 新密码

        Raises:
            HTTPException: 旧密码错误时抛出
        """
        # 验证旧密码
        if not verify_password(old_password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="旧密码错误",
            )

        # 更新密码
        from examples.user_model import UserUpdate

        await self.user_crud.update(
            user.id,
            UserUpdate(
                hashed_password=get_password_hash(new_password),
            ),
        )


# ============================================
# 依赖注入函数
# ============================================

async def get_auth_service(
    db: AsyncSession = Depends(get_db),
) -> AuthService:
    """
    获取认证服务实例（依赖注入）

    Args:
        db: 数据库会话

    Returns:
        AuthService: 认证服务实例
    """
    return AuthService(db)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    auth_service: AuthService = Depends(get_auth_service),
) -> User:
    """
    获取当前登录用户（依赖注入）

    用于在路由中获取当前登录的用户信息。

    Args:
        token: JWT 令牌
        auth_service: 认证服务

    Returns:
        User: 当前用户对象

    示例:
        @router.get("/profile")
        async def get_profile(current_user: User = Depends(get_current_user)):
            return current_user
    """
    return await auth_service.get_current_user(token)


# ============================================
# 使用示例
# ============================================

async def example_usage():
    """
    认证服务使用示例
    """

    from sqlalchemy.ext.asyncio import AsyncSession
    from app.database.session import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        auth_service = AuthService(db)

        # ========== 用户注册 ==========
        user_create = UserCreate(
            username="zhangsan",
            email="zhangsan@example.com",
            password="123456",
            full_name="张三",
        )
        user = await auth_service.register(user_create)
        print(f"注册成功: {user.username}")

        # ========== 用户登录 ==========
        login_data = UserLogin(username="zhangsan", password="123456")
        tokens = await auth_service.login(login_data)
        print(f"登录成功，访问令牌: {tokens.access_token[:20]}...")

        # ========== 获取当前用户 ==========
        current_user = await auth_service.get_current_user(tokens.access_token)
        print(f"当前用户: {current_user.full_name}")

        # ========== 刷新令牌 ==========
        new_tokens = await auth_service.refresh_token(tokens.refresh_token)
        print(f"刷新令牌成功: {new_tokens.access_token[:20]}...")

        # ========== 修改密码 ==========
        await auth_service.change_password(current_user, "123456", "newpass123")
        print("密码修改成功")


if __name__ == "__main__":
    print("""
    认证服务示例

    主要功能:
    - 用户注册
    - 用户登录
    - 刷新令牌
    - 获取当前用户
    - 修改密码

    使用方法:
    1. 创建认证服务实例
    2. 调用相应方法完成认证操作

    注意: 需要先配置数据库连接才能运行示例
    """)
