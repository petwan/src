"""
认证模块 - API 路由

提供微信小程序登录、用户信息管理等相关接口。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.system.auth.dependencies import (
    CurrentUser,
    CurrentUserOptional,
    CurrentActiveUser,
    WechatAuth,
    get_current_user_optional,
)
from app.api.v1.system.auth.models import User
from app.api.v1.system.auth.schemas import (
    WechatLoginRequest,
    WechatUserInfoUpdate,
    PhoneLoginRequest,
    UserInfo,
    AuthResponse,
)
from app.core.dependencies import get_db
from app.core.exception import CustomException
from app.core.response import BaseResponse

# 创建认证路由器
router = APIRouter(prefix="/auth", tags=["认证管理"])


# ============================================
# 微信登录相关接口
# ============================================

@router.post(
    "/wechat/login",
    response_model=BaseResponse[AuthResponse],
    summary="微信小程序登录",
    description="通过微信小程序临时登录凭证(code)进行登录，返回JWT令牌和用户信息"
)
async def wechat_login(
    login_request: WechatLoginRequest,
    auth_service: WechatAuth,
) -> BaseResponse[AuthResponse]:
    """
    微信小程序登录接口

    登录流程：
    1. 前端调用 uni.login() 获取临时登录凭证 code
    2. 将 code 发送到此接口
    3. 后端使用 code 向微信服务器换取 openid 和 session_key
    4. 根据 openid 查找用户，不存在则创建
    5. 生成 JWT 令牌返回给前端

    请求示例:
    ```json
    {
        "code": "061a2b3c4d5e6f"
    }
    ```

    响应示例:
    ```json
    {
        "code": 200,
        "msg": "微信登录成功",
        "data": {
            "token": {
                "token_type": "bearer",
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "expires_in": 1800
            },
            "user": {
                "id": 1,
                "nickname": "用户-abc12345",
                "avatar_url": null,
                "phone": null,
                "gender": 0,
                "is_active": 1
            }
        }
    }
    ```

    注意:
        - code 有效期为 5 分钟，且只能使用一次
        - 首次登录时，昵称和头像可能为空，需要用户授权后通过 update_profile 接口更新
    """
    try:
        # 调用服务层处理登录逻辑
        auth_response = await auth_service.wechat_login(login_request.code)

        return BaseResponse[AuthResponse](
            code=200,
            msg="微信登录成功",
            data=auth_response,
        )
    except CustomException as e:
        # 自定义异常直接抛出
        raise
    except Exception as e:
        # 其他异常捕获后抛出自定义异常
        raise CustomException(msg="微信登录失败，请稍后重试")


@router.post(
    "/wechat/update-profile",
    response_model=BaseResponse[UserInfo],
    summary="更新微信用户信息",
    description="更新用户昵称、头像等信息（通常在用户授权获取信息后调用）"
)
async def update_wechat_profile(
    profile_update: WechatUserInfoUpdate,
    current_user: CurrentUser,
    auth_service: WechatAuth,
) -> BaseResponse[UserInfo]:
    """
    更新微信用户信息接口

    用于用户首次授权获取昵称和头像后，更新用户信息。

    请求示例:
    ```json
    {
        "nickname": "张三",
        "avatar_url": "https://example.com/avatar.jpg",
        "gender": 1,
        "country": "中国",
        "province": "广东",
        "city": "深圳"
    }
    ```

    注意:
        - 需要先登录才能调用此接口
        - 使用微信的 getUserProfile 接口获取用户信息后再调用此接口
    """
    # 更新用户信息
    updated_user = await auth_service.update_user_profile(current_user, profile_update)

    # 构造响应
    user_info = UserInfo(
        id=updated_user.id,
        nickname=updated_user.nickname,
        avatar_url=updated_user.avatar_url,
        phone=updated_user.phone,
        gender=updated_user.gender or 0,
        is_active=updated_user.is_active,
    )

    return BaseResponse[UserInfo](
        code=200,
        msg="用户信息更新成功",
        data=user_info,
    )


# ============================================
# 手机号登录相关接口
# ============================================

@router.post(
    "/phone/login",
    response_model=BaseResponse[AuthResponse],
    summary="手机号验证码登录",
    description="通过手机号和验证码进行登录，返回JWT令牌和用户信息"
)
async def phone_login(
    login_request: PhoneLoginRequest,
    auth_service: WechatAuth,  # 这里用 WechatAuth 暂时代替，实际应使用 PhoneAuth
) -> BaseResponse[AuthResponse]:
    """
    手机号验证码登录接口

    请求示例:
    ```json
    {
        "phone": "13800138000",
        "code": "123456"
    }
    ```

    注意:
        - 验证码发送接口需要根据短信服务商实现
        - 验证码有效期通常为 5 分钟
    """
    # 注意：这里需要使用 PhoneAuthService，但由于 imports 问题暂时使用 WechatAuth
    # 实际使用时应该：
    # from app.api.v1.system.auth.dependencies import PhoneAuth
    # auth_service: PhoneAuth = Depends()
    from app.api.v1.system.auth.service import PhoneAuthService

    # 手动创建 PhoneAuthService
    # 这里需要获取 db，可以通过 Depends 注入
    # 为简化示例，暂时抛出异常
    raise CustomException(msg="手机号登录功能正在开发中")


# ============================================
# 用户信息接口
# ============================================

@router.get(
    "/user/profile",
    response_model=BaseResponse[UserInfo],
    summary="获取当前用户信息",
    description="获取当前登录用户的详细信息"
)
async def get_user_profile(
    current_user: CurrentUser,
) -> BaseResponse[UserInfo]:
    """
    获取当前用户信息接口

    需要在请求头中携带 JWT 令牌:
    ```
    Authorization: Bearer {access_token}
    ```

    响应示例:
    ```json
    {
        "code": 200,
        "msg": "获取成功",
        "data": {
            "id": 1,
            "nickname": "张三",
            "avatar_url": "https://example.com/avatar.jpg",
            "phone": "13800138000",
            "gender": 1,
            "is_active": 1
        }
    }
    ```
    """
    user_info = UserInfo(
        id=current_user.id,
        nickname=current_user.nickname,
        avatar_url=current_user.avatar_url,
        phone=current_user.phone,
        gender=current_user.gender or 0,
        is_active=current_user.is_active,
    )

    return BaseResponse[UserInfo](
        code=200,
        msg="获取成功",
        data=user_info,
    )


@router.get(
    "/user/info",
    response_model=BaseResponse[UserInfo],
    summary="获取用户信息（可选登录）",
    description="获取用户信息，支持未登录状态访问"
)
async def get_user_info_optional(
    current_user: CurrentUserOptional,
) -> BaseResponse[UserInfo]:
    """
    获取用户信息接口（可选登录）

    如果用户已登录，返回用户信息；如果未登录，返回 null。

    使用场景：
        - 某些接口既需要登录用户访问，也需要支持匿名访问
    """
    if not current_user:
        return BaseResponse[UserInfo](
            code=200,
            msg="未登录",
            data=None,
        )

    user_info = UserInfo(
        id=current_user.id,
        nickname=current_user.nickname,
        avatar_url=current_user.avatar_url,
        phone=current_user.phone,
        gender=current_user.gender or 0,
        is_active=current_user.is_active,
    )

    return BaseResponse[UserInfo](
        code=200,
        msg="获取成功",
        data=user_info,
    )


# ============================================
# 登出接口
# ============================================

@router.post(
    "/logout",
    response_model=BaseResponse[None],
    summary="用户登出",
    description="用户登出（可选实现 token 黑名单）"
)
async def logout(
    current_user: CurrentUser,
) -> BaseResponse[None]:
    """
    用户登出接口

    注意:
        - 当前实现为无状态登出，前端删除 token 即可
        - 如需实现 token 黑名单，可以使用 Redis 存储
        - 登出后需要前端清除本地存储的 token
    """
    # TODO: 可选 - 将 token 加入黑名单
    # 如果要实现真正的登出功能，可以使用 Redis 存储失效的 token
    # 代码示例：
    # from app.core.redis import redis_client
    # await redis_client.setex(f"blacklist:{token}", expires_in, "1")

    return BaseResponse[None](
        code=200,
        msg="登出成功",
        data=None,
    )


# ============================================
# 接口使用示例（文档）
# ============================================

@router.get(
    "/example",
    summary="接口使用示例（不实际调用）",
    description="展示各种依赖的使用方式"
)
async def example_endpoints():
    """
    接口使用示例

    此接口不会实际被调用，仅用于文档展示各种依赖的使用方式。

    示例代码：
    ```python
    # 1. 需要登录的接口
    @router.get("/protected")
    async def protected(user: CurrentUser):
        return {"user_id": user.id}

    # 2. 可选登录的接口
    @router.get("/optional")
    async def optional(user: CurrentUserOptional):
        if user:
            return {"user_id": user.id, "logged_in": True}
        return {"logged_in": False}

    # 3. 需要活跃用户的接口
    @router.get("/active")
    async def active(user: CurrentActiveUser):
        return {"nickname": user.nickname}
    ```
    """
    return {
        "message": "请查看接口文档了解具体使用方式",
        "examples": {
            "protected": "使用 CurrentUser 依赖",
            "optional": "使用 CurrentUserOptional 依赖",
            "active": "使用 CurrentActiveUser 依赖",
        }
    }


# ============================================
# 导出路由
# ============================================

__all__ = ["router"]
