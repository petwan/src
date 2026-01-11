"""
认证模块 - Pydantic Schema 定义

定义用户相关的请求和响应数据结构。
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, EmailStr


# ============================================
# 通用响应 Schema
# ============================================
# 注意：统一使用 app.core.response.BaseResponse 作为 API 响应格式
# 不再在此处定义独立的响应类


# ============================================
# 微信登录 Schema
# ============================================

class WechatLoginRequest(BaseModel):
    """
    微信小程序登录请求

    用户通过微信小程序登录时需要提供的参数。

    Attributes:
        code: 微信小程序临时登录凭证，通过 uni.login() 获取
    """

    code: str = Field(..., min_length=1, description="微信小程序临时登录凭证")

    model_config = {
        "json_schema_extra": {
            "example": {
                "code": "061a2b3c4d5e6f",
            }
        }
    }


class WechatUserInfoUpdate(BaseModel):
    """
    微信用户信息更新请求

    用于更新用户的昵称、头像等信息。

    Attributes:
        nickname: 用户昵称
        avatar_url: 用户头像 URL
        gender: 性别
        country: 国家
        province: 省份
        city: 城市
    """

    nickname: str | None = Field(None, max_length=100, description="昵称")
    avatar_url: str | None = Field(None, max_length=500, description="头像URL")
    gender: int | None = Field(None, ge=0, le=2, description="性别: 0=未知, 1=男, 2=女")
    country: str | None = Field(None, max_length=50, description="国家")
    province: str | None = Field(None, max_length=50, description="省份")
    city: str | None = Field(None, max_length=50, description="城市")

    model_config = {
        "json_schema_extra": {
            "example": {
                "nickname": "张三",
                "avatar_url": "https://example.com/avatar.jpg",
                "gender": 1,
                "country": "中国",
                "province": "广东",
                "city": "深圳",
            }
        }
    }


# ============================================
# 手机号登录 Schema
# ============================================

class PhoneLoginRequest(BaseModel):
    """
    手机号登录请求

    通过手机号和验证码登录。

    Attributes:
        phone: 手机号
        code: 验证码
    """

    phone: str = Field(..., pattern=r"^1[3-9]\d{9}$", description="手机号")
    code: str = Field(..., min_length=6, max_length=6, description="验证码")

    model_config = {
        "json_schema_extra": {
            "example": {
                "phone": "13800138000",
                "code": "123456",
            }
        }
    }


# ============================================
# Token Schema
# ============================================

class TokenInfo(BaseModel):
    """
    令牌信息

    登录成功后返回的 JWT 令牌信息。

    Attributes:
        token_type: 令牌类型，通常为 "bearer"
        access_token: 访问令牌
        refresh_token: 刷新令牌（可选）
        expires_in: 过期时间（秒）
    """

    token_type: str = Field(default="bearer", description="令牌类型")
    access_token: str = Field(..., description="访问令牌")
    refresh_token: str | None = Field(None, description="刷新令牌")
    expires_in: int | None = Field(None, description="过期时间（秒）")

    model_config = {
        "json_schema_extra": {
            "example": {
                "token_type": "bearer",
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "expires_in": 1800,
            }
        }
    }


# ============================================
# 用户信息 Schema
# ============================================

class UserInfo(BaseModel):
    """
    用户基本信息

    返回给前端的用户信息。

    Attributes:
        id: 用户ID
        nickname: 昵称
        avatar_url: 头像URL
        phone: 手机号
        gender: 性别
        is_active: 是否激活
    """

    id: int = Field(..., description="用户ID")
    nickname: str | None = Field(None, description="昵称")
    avatar_url: str | None = Field(None, description="头像URL")
    phone: str | None = Field(None, description="手机号")
    gender: int = Field(default=0, description="性别: 0=未知, 1=男, 2=女")
    is_active: int = Field(default=1, description="是否激活")

    model_config = {"from_attributes": True}


class AuthResponse(BaseModel):
    """
    认证响应

    登录成功后返回的完整响应，包含令牌和用户信息。

    Attributes:
        token: 令牌信息
        user: 用户信息
    """

    token: TokenInfo = Field(..., description="令牌信息")
    user: UserInfo = Field(..., description="用户信息")

    model_config = {
        "json_schema_extra": {
            "example": {
                "token": {
                    "token_type": "bearer",
                    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "expires_in": 1800,
                },
                "user": {
                    "id": 1,
                    "nickname": "张三",
                    "avatar_url": "https://example.com/avatar.jpg",
                    "phone": "13800138000",
                    "gender": 1,
                    "is_active": 1,
                },
            }
        }
    }


# ============================================
# 用户创建和更新 Schema
# ============================================

class UserCreate(BaseModel):
    """
    创建用户请求

    用于管理员创建新用户。

    Attributes:
        openid: 微信 openid（必填）
        nickname: 昵称
        avatar_url: 头像URL
        phone: 手机号
    """

    openid: str = Field(..., min_length=1, max_length=100, description="微信openid")
    nickname: str | None = Field(None, max_length=100, description="昵称")
    avatar_url: str | None = Field(None, max_length=500, description="头像URL")
    phone: str | None = Field(None, pattern=r"^1[3-9]\d{9}$", description="手机号")


class UserUpdate(BaseModel):
    """
    更新用户请求

    用于更新用户信息。

    Attributes:
        nickname: 昵称
        avatar_url: 头像URL
        phone: 手机号
        is_active: 是否激活
    """

    nickname: str | None = Field(None, max_length=100, description="昵称")
    avatar_url: str | None = Field(None, max_length=500, description="头像URL")
    phone: str | None = Field(None, pattern=r"^1[3-9]\d{9}$", description="手机号")
    is_active: int | None = Field(None, ge=0, le=1, description="是否激活: 1=激活, 0=禁用")


# ============================================
# 认证上下文 Schema（用于 CRUD）
# ============================================

class AuthSchema:
    """
    认证信息 Schema

    用于在 CRUD 操作中传递认证上下文和数据库会话。

    Attributes:
        db: 数据库会话
        user: 当前用户对象（可选）
    """

    def __init__(self, db: Any, user: Any = None):
        """
        初始化认证信息

        Args:
            db: 数据库会话
            user: 当前用户对象（可选）
        """
        self.db = db
        self.user = user
