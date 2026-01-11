"""
示例模块

展示如何使用 Astrapi 框架构建完整的功能模块。

包含以下示例:
    - user_model.py: 数据模型和 Schema 定义
    - user_crud.py: CRUD 操作
    - user_router.py: API 路由

使用方法:
    1. 将示例代码复制到 app/api/v1/system/users/ 目录
    2. 在 app/api/v1/__init__.py 中注册路由
    3. 运行数据库迁移创建表
    4. 测试 API 接口
"""

from .user_crud import CRUDUser
from .user_model import (
    AuthSchema,
    TokenResponse,
    User,
    UserCreate,
    UserLogin,
    UserResponse,
    UserUpdate,
)
from .user_router import router

__all__ = [
    "User",
    "UserCreate",
    "UserUpdate",
    "UserResponse",
    "UserLogin",
    "TokenResponse",
    "AuthSchema",
    "CRUDUser",
    "router",
]
