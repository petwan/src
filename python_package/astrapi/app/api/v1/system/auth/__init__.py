"""
认证模块

提供微信小程序登录、用户认证、用户信息管理等功能。

主要功能:
    - 微信小程序一键登录
    - 手机号验证码登录
    - 用户信息管理
    - JWT 令牌认证
    - 权限验证

模块结构:
    - models.py: 用户数据模型
    - schemas.py: Pydantic Schema 定义
    - crud.py: 数据访问层
    - service.py: 业务逻辑层
    - dependencies.py: 依赖注入
    - router.py: API 路由

使用示例:
    # 1. 在主路由中注册
    from app.api.v1.system.auth.router import router as auth_router

    api_router.include_router(auth_router)

    # 2. 前端调用示例
    # 微信登录
    uni.login({
        provider: 'weixin',
        success: (res) => {
            // 将 res.code 发送到后端
            axios.post('/api/v1/auth/wechat/login', { code: res.code })
                .then(response => {
                    const { token, user } = response.data.data
                    // 保存 token
                    uni.setStorageSync('token', token.access_token)
                })
        }
    })

    # 获取用户信息
    axios.get('/api/v1/auth/user/profile', {
        headers: {
            'Authorization': 'Bearer ' + token
        }
    })
"""

from app.api.v1.system.auth.models import User
from app.api.v1.system.auth.schemas import (
    WechatLoginRequest,
    WechatUserInfoUpdate,
    PhoneLoginRequest,
    TokenInfo,
    UserInfo,
    AuthResponse,
    UserCreate,
    UserUpdate,
)
from app.api.v1.system.auth.crud import CRUDUser
from app.api.v1.system.auth.service import (
    WechatAuthService,
    PhoneAuthService,
)
from app.api.v1.system.auth.dependencies import (
    get_wechat_auth_service,
    get_phone_auth_service,
    get_current_user,
    get_current_user_optional,
    get_current_active_user,
    CurrentUser,
    CurrentUserOptional,
    CurrentActiveUser,
    WechatAuth,
    PhoneAuth,
)
from app.api.v1.system.auth.router import router

__all__ = [
    # Models
    "User",
    # Schemas
    "WechatLoginRequest",
    "WechatUserInfoUpdate",
    "PhoneLoginRequest",
    "TokenInfo",
    "UserInfo",
    "AuthResponse",
    "UserCreate",
    "UserUpdate",
    # CRUD
    "CRUDUser",
    # Services
    "WechatAuthService",
    "PhoneAuthService",
    # Dependencies
    "get_wechat_auth_service",
    "get_phone_auth_service",
    "get_current_user",
    "get_current_user_optional",
    "get_current_active_user",
    "CurrentUser",
    "CurrentUserOptional",
    "CurrentActiveUser",
    "WechatAuth",
    "PhoneAuth",
    # Router
    "router",
]
