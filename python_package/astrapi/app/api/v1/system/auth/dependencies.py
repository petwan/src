"""
认证模块 - 依赖注入

提供获取当前用户、认证服务等依赖注入函数。
"""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.system.auth.models import User
from app.api.v1.system.auth.service import WechatAuthService, PhoneAuthService
from app.core.dependencies import get_db
from app.api.v1.system.auth.schemas import AuthSchema
from app.core.exception import UnauthorizedException
from app.core.security import decode_token

# OAuth2 密码流（用于获取令牌）
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/wechat/login",
    auto_error=False
)


# ============================================
# 认证服务依赖
# ============================================

async def get_wechat_auth_service(
    db: AsyncSession = Depends(get_db),
) -> WechatAuthService:
    """
    获取微信认证服务实例

    Args:
        db: 数据库会话

    Returns:
        WechatAuthService: 微信认证服务实例

    示例:
        @router.post("/wechat/login")
        async def wechat_login(
            login_request: WechatLoginRequest,
            auth_service: WechatAuthService = Depends(get_wechat_auth_service),
        ):
            return await auth_service.wechat_login(login_request.code)
    """
    return WechatAuthService(db)


async def get_phone_auth_service(
    db: AsyncSession = Depends(get_db),
) -> PhoneAuthService:
    """
    获取手机号认证服务实例

    Args:
        db: 数据库会话

    Returns:
        PhoneAuthService: 手机号认证服务实例

    示例:
        @router.post("/phone/login")
        async def phone_login(
            login_request: PhoneLoginRequest,
            auth_service: PhoneAuthService = Depends(get_phone_auth_service),
        ):
            return await auth_service.phone_login(login_request.phone, login_request.code)
    """
    return PhoneAuthService(db)


# ============================================
# 用户认证依赖
# ============================================

async def get_current_user_optional(
    token: str | None = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """
    获取当前登录用户（可选认证）

    如果未提供令牌或令牌无效，返回 None 而不是抛出异常。
    适用于某些接口既支持登录用户也支持匿名访问的场景。

    Args:
        token: JWT 令牌（可选）
        db: 数据库会话

    Returns:
        User | None: 当前用户对象，未登录返回 None

    示例:
        @router.get("/profile")
        async def get_profile(
            current_user: User | None = Depends(get_current_user_optional),
        ):
            if current_user:
                return current_user
            else:
                return {"message": "未登录"}
    """
    if not token:
        return None

    return await get_current_user(token, db)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    获取当前登录用户（必需认证）

    从 JWT 令牌中解析用户信息并验证，获取当前登录用户。

    Args:
        token: JWT 令牌
        db: 数据库会话

    Returns:
        User: 当前用户对象

    Raises:
        UnauthorizedException: 令牌无效或用户不存在时抛出

    示例:
        @router.get("/profile")
        async def get_profile(
            current_user: User = Depends(get_current_user),
        ):
            return current_user
    """
    if not token:
        raise UnauthorizedException("缺少认证令牌")

    # 解码令牌
    payload = decode_token(token)
    if not payload or "sub" not in payload:
        raise UnauthorizedException("无效的认证令牌")

    user_id = int(payload["sub"])

    # 从数据库获取用户
    auth = AuthSchema(db=db)
    from app.api.v1.system.auth.crud import CRUDUser

    user_crud = CRUDUser(User, auth)
    user = await user_crud.get(id=user_id)

    if not user:
        raise UnauthorizedException("用户不存在")

    # 检查用户状态
    if not user.is_active:
        raise UnauthorizedException("账号已被禁用")

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    获取当前活跃用户

    验证用户是否处于活跃状态（未禁用）。

    Args:
        current_user: 当前用户对象

    Returns:
        User: 活跃的用户对象

    Raises:
        HTTPException: 用户被禁用时抛出

    示例:
        @router.get("/protected")
        async def protected_endpoint(
            current_user: User = Depends(get_current_active_user),
        ):
            return {"message": f"Hello, {current_user.nickname}!"}
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="账号已被禁用",
        )
    return current_user


# ============================================
# 类型别名（用于简化依赖类型注解）
# ============================================

# 当前用户依赖（可选）
CurrentUserOptional = Annotated[User | None, Depends(get_current_user_optional)]

# 当前用户依赖（必需）
CurrentUser = Annotated[User, Depends(get_current_user)]

# 当前活跃用户依赖
CurrentActiveUser = Annotated[User, Depends(get_current_active_user)]

# 微信认证服务依赖
WechatAuth = Annotated[WechatAuthService, Depends(get_wechat_auth_service)]

# 手机号认证服务依赖
PhoneAuth = Annotated[PhoneAuthService, Depends(get_phone_auth_service)]
