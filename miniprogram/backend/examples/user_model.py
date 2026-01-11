"""
示例：用户模型

展示如何定义数据库模型、Schema 和 CRUD 操作。
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import relationship

from app.database.model import Base


class User(Base):
    """
    用户模型

    用户表存储系统用户的基本信息。

    表名: users
    """

    __repr_attrs__ = ["username", "email"]

    # 主键
    id = Column(Integer, primary_key=True, index=True, comment="用户ID")

    # 基本信息
    username = Column(String(50), unique=True, index=True, nullable=False, comment="用户名")
    email = Column(String(100), unique=True, index=True, nullable=False, comment="邮箱")
    hashed_password = Column(String(200), nullable=False, comment="密码哈希")

    # 扩展信息
    full_name = Column(String(100), comment="全名")
    phone = Column(String(20), comment="手机号")
    avatar = Column(String(255), comment="头像URL")
    bio = Column(Text, comment="个人简介")

    # 状态字段
    is_active = Column(Integer, default=1, comment="是否激活: 1=激活, 0=禁用")
    is_superuser = Column(Integer, default=0, comment="是否超级管理员: 1=是, 0=否")

    # 审计字段
    created_id = Column(Integer, comment="创建人ID")
    updated_id = Column(Integer, comment="更新人ID")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")

    # 关联关系
    # roles = relationship("Role", secondary="user_roles", back_populates="users")


# ============================================
# Pydantic Schema 定义（用于 API 请求/响应）
# ============================================

class UserBase(BaseModel):
    """用户基础 Schema"""

    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    email: EmailStr = Field(..., description="邮箱")
    full_name: str | None = Field(None, max_length=100, description="全名")
    phone: str | None = Field(None, max_length=20, description="手机号")


class UserCreate(UserBase):
    """创建用户 Schema"""

    password: str = Field(..., min_length=6, max_length=100, description="密码")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "username": "zhangsan",
                    "email": "zhangsan@example.com",
                    "password": "123456",
                    "full_name": "张三",
                }
            ]
        }
    }


class UserUpdate(BaseModel):
    """更新用户 Schema"""

    username: str | None = Field(None, min_length=3, max_length=50, description="用户名")
    email: EmailStr | None = Field(None, description="邮箱")
    full_name: str | None = Field(None, max_length=100, description="全名")
    phone: str | None = Field(None, max_length=20, description="手机号")
    bio: str | None = Field(None, description="个人简介")
    is_active: int | None = Field(None, description="是否激活")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "full_name": "李四",
                    "phone": "13800138000",
                }
            ]
        }
    }


class UserResponse(UserBase):
    """用户响应 Schema"""

    id: int
    avatar: str | None = None
    bio: str | None = None
    is_active: int
    is_superuser: int
    created_at: datetime

    model_config = {"from_attributes": True}


class UserLogin(BaseModel):
    """用户登录 Schema"""

    username: str = Field(..., description="用户名或邮箱")
    password: str = Field(..., description="密码")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "username": "zhangsan",
                    "password": "123456",
                }
            ]
        }
    }


class TokenResponse(BaseModel):
    """令牌响应 Schema"""

    access_token: str = Field(..., description="访问令牌")
    refresh_token: str = Field(..., description="刷新令牌")
    token_type: str = Field(default="bearer", description="令牌类型")


# ============================================
# Auth Schema（用于权限传递）
# ============================================

class AuthSchema:
    """
    认证信息 Schema

    用于在 CRUD 操作中传递认证上下文和数据库会话。
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
