---
title: ☀️ FastAPI 工程化后端搭建思考
date: 2026-01-05
tags: [Python]
description: FastAPI 工程化后端搭建思考
draft: false
---

# ☀️ FastAPI 工程化后端搭建思考

之前工作中使用 FastAPI 搭建后端，基本都是简单的几个endpoints即可，更多地是为了做基础性的 PoC 验证（例如相似性搜索服务等），没有太多工程化上的考量，这个文章挑选了部分[FastAPI 最佳实践](https://github.com/zhanymkanov/fastapi-best-practices/blob/master/README_ZH.md)的内容，同时总结实际的需求并记录对应的工程化实践的思考。**建议阅读原文以全面理解FastAPI的最佳实践。**

## 1. FastAPI 后端服务目标
设计一个适用于中大型 FastAPI 项目的模块化、高内聚低耦合、支持自动化代码生成、便于团队协作和长期维护的项目架构。

关键设计原则：
1. **modules 平铺式模块结构**：每个模块是自治单元，包含了完整的 MVC-like层，git 合并冲突少，可独立开发、测试、部署。
2. **严格分层：Model → CRUD → Service → Router**：
   1. Model 仅 ORM 映射，无业务逻辑
   2. CRUD  纯数据操作（增删修查），不处理业务规则
   3. Service 封装业务逻辑，依赖 CRUD，可调用多个CRUD完成一个业务逻辑
   4. Router 只做请求解析、调用 service、返回响应
3. **依赖注入集中管理**：例如所有模块通过 `Depends(get_db)` 获取会话。
4. **配置外部化**：可以使用环境变量和配置文件管理，例如 `.env.dev` 用于开发环境，`.env.prod` 用于生产环境。
5. **自动化代码生成**：自动生成 CRUD、Service、Router 代码，减少重复劳动。
6. **核心能力下沉到 core**：所有跨模块共享能力放在这里，例如数据库会话、依赖注入、配置管理、日志记录、异常处理（`@app.exception_handler(CustomException)`）、中间件（认证、限流、审计日志）等
7. **模块开关**：例如在 `settings.py` 中配置 ENABLED_MODULES = ["user", "order"]，动态加载路由
8. **OpenAPI分组**：使用 `tags` 参数为每个路由指定分组，方便文档展示
9. **命令行工具**：使用 [Typer](https://github.com/tiangolo/typer) 创建命令行工具，加速开发流程。

## 2. 项目结构
许多示例项目和教程按文件类型（如crud、routers、models）划分项目，这种方式对于微服务或范围较小的项目很有效。但是，这种方法并不适合包含许多领域和模块的单体应用。

针对中大型单体项目，采用模块化平铺式结构更为合适，每个模块包含完整的功能组件，便于独立开发和维护。比较优秀的开源项目参考是 Netflix 的 [Dispatch](https://github.com/Netflix/dispatch/tree/main)


```console
- app/main.py: 项目根文件
- <module>
  - router.py - 每个模块的核心，包含所有端点
  - schemas.py - 用于pydantic模型
  - models.py - 用于数据库模型
  - service.py - 模块特定的业务逻辑
  - dependencies.py - 路由依赖项
  - constants.py - 模块特定的常量和错误代码
  - config.py - 例如环境变量
  - utils.py - 非业务逻辑函数，例如响应规范化、数据丰富等
  - exceptions.py - 模块特定的异常，例如PostNotFound、InvalidUserData
- config.py : 全局配置文件
- models.py : 全局数据库模型
- exceptions.py : 全局异常处理
```

## 3. BaseSettings 拆分
BaseSettings是读取环境变量的一项伟大创新，但为整个应用使用单个BaseSettings随着时间的推移可能会变得混乱。为了提高可维护性和组织性，我们将BaseSettings拆分到不同的模块和领域中。
```python
# src.auth.config
from datetime import timedelta

from pydantic_settings import BaseSettings

class AuthConfig(BaseSettings):
    JWT_ALG: str
    JWT_SECRET: str
    JWT_EXP: int = 5  # 分钟

    REFRESH_TOKEN_KEY: str
    REFRESH_TOKEN_EXP: timedelta = timedelta(days=30)

    SECURE_COOKIES: bool = True

auth_settings = AuthConfig()

# src.config
from pydantic import PostgresDsn, RedisDsn, model_validator
from pydantic_settings import BaseSettings

from src.constants import Environment

class Config(BaseSettings):
    DATABASE_URL: PostgresDsn
    REDIS_URL: RedisDsn

    SITE_DOMAIN: str = "myapp.com"

    ENVIRONMENT: Environment = Environment.PRODUCTION

    SENTRY_DSN: str | None = None

    CORS_ORIGINS: list[str]
    CORS_ORIGINS_REGEX: str | None = None
    CORS_HEADERS: list[str]

    APP_VERSION: str = "1.0"

settings = Config()
```

## 4. 依赖注入与链式依赖
依赖项可以使用其他依赖项，避免类似逻辑的代码重复。
```python
# dependencies.py
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

async def valid_post_id(post_id: UUID4) -> dict[str, Any]:
    post = await service.get_by_id(post_id)
    if not post:
        raise PostNotFound()

    return post

async def parse_jwt_data(
    token: str = Depends(OAuth2PasswordBearer(tokenUrl="/auth/token"))
) -> dict[str, Any]:
    try:
        payload = jwt.decode(token, "JWT_SECRET", algorithms=["HS256"])
    except JWTError:
        raise InvalidCredentials()

    return {"user_id": payload["id"]}

async def valid_owned_post(
    post: dict[str, Any] = Depends(valid_post_id), 
    token_data: dict[str, Any] = Depends(parse_jwt_data),
) -> dict[str, Any]:
    if post["creator_id"] != token_data["user_id"]:
        raise UserNotOwner()

    return post

# router.py
@router.get("/users/{user_id}/posts/{post_id}", response_model=PostResponse)
async def get_user_post(post: dict[str, Any] = Depends(valid_owned_post)):
    return
```

**优先使用 `async` 依赖项**

## 5. 规范化响应
```python
from fastapi import APIRouter, status

router = APIRouter()

@router.post(
    "/endpoints",
    response_model=DefaultResponseModel,  # default response pydantic model 
    status_code=status.HTTP_201_CREATED,  # default status code
    description="Description of the well documented endpoint",
    tags=["Endpoint Category"],
    summary="Summary of the Endpoint",
    responses={
        status.HTTP_200_OK: {
            "model": OkResponse, # custom pydantic model for 200 response
            "description": "Ok Response",
        },
        status.HTTP_201_CREATED: {
            "model": CreatedResponse,  # custom pydantic model for 201 response
            "description": "Creates something from user request",
        },
        status.HTTP_202_ACCEPTED: {
            "model": AcceptedResponse,  # custom pydantic model for 202 response
            "description": "Accepts request and handles it later",
        },
    },
)
async def documented_route():
    pass
```

## 6. alembic
为新迁移设置人类可读的文件模板。我们使用date*_*slug*.py模式，例如2022-08-24_post_content_idx.py
```ini
# alembic.ini
file_template = %%(year)d-%%(month).2d-%%(day).2d_%%(slug)s
```

## SQLALchemy Base类

基于 SQLALchemy 的 DeclarativeBase 类，构建一个 Base 类。

FastAPI 开发中经常需要打印模型实例（如日志、断点调试），默认SQLAlchemy 模型的 __repr__ 是 <User object at 0x7f8b1c2d3e40>，毫无信息量。

同时，SQLAlchemy中可以通过 resolve_table_name 避免重复写 __tablename__

使用 @declared_attr.directive 符合 SQLAlchemy 2.0+ 规范


## Exception
FastAPI 不会自动将异常转换为友好 JSON 响应。如果不注册全局异常处理器：

用户会看到 500 Internal Server Error（无细节）
开发者无法统一日志格式
客户端得不到结构化错误信息（如 code, msg, data）

因此需要自定义一个 异常处理器

## CRUD Base
针对service，所有模型共用一套增删改查方法，避免每个 Service 写 100 行重复 SQL

```python
# -*- coding: utf-8 -*-
"""
异步 SQLAlchemy Repository 基类（无状态、类型安全、轻量）

命名说明：
- 使用 `AsyncBaseRepository` 作为基类名，符合业界主流命名习惯（如 Clean Architecture / DDD）
- 所有方法显式接收 `session`，确保无状态、线程/协程安全
- 兼容 SQLAlchemy 2.0+，充分利用其新特性（如 session.get 支持 options）

泛型约束：
- ModelType 必须继承自 sqlalchemy.orm.DeclarativeBase
- 这样可兼容任何合法的 SQLAlchemy 模型（无论你项目中的 Base 是如何定义的）
"""

from typing import (
    TypeVar,
    Generic,
    Optional,
    List,
    Any,
    Sequence,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import Select


# 定义泛型变量：ModelType 必须是 DeclarativeBase 的子类
# 注意：这里用 DeclarativeBase 而不是你自定义的 Base，
#       因为所有 SQLAlchemy 模型最终都继承自它，保证通用性
ModelType = TypeVar("ModelType", bound=DeclarativeBase)


class AsyncBaseRepository(Generic[ModelType]):
    """
    异步数据访问基类（Repository 模式）

    设计原则：
    - 无状态：不持有 session，由调用方传入
    - 类型安全：通过泛型确保返回值类型正确
    - 灵活查询：支持 where 条件、排序、分页、预加载等
    - 高性能：优先使用 session.get() 利用 identity map 缓存
    """

    def __init__(self, model: type[ModelType]) -> None:
        """
        初始化 Repository
        
        :param model: 对应的 SQLAlchemy 模型类（如 User, Product）
        """
        self.model = model

    # ===================================================================
    # 🔍 查询方法（Query Methods）
    # ===================================================================

    async def get_by_id(
        self,
        session: AsyncSession,
        id: Any,
        options: Optional[List[Any]] = None
    ) -> Optional[ModelType]:
        """
        根据主键获取单个对象（推荐方式）
        
        ✅ 优势：
          - 自动识别主键字段（无需硬编码 "id"）
          - 支持复合主键（传入元组即可）
          - 利用 SQLAlchemy 的 identity map 缓存（若对象已在 session 中则不查数据库）
          - 原生支持预加载（options），避免 N+1 问题
        
        📌 示例：
          user = await repo.get_by_id(session, 123)
          user_with_posts = await repo.get_by_id(session, 123, options=[selectinload(User.posts)])
          record = await repo.get_by_id(session, ("user_001", "2025-01-01"))  # 复合主键
        """
        return await session.get(self.model, id, options=options)

    async def find_one(
        self,
        session: AsyncSession,
        *where_clauses,
        options: Optional[List[Any]] = None
    ) -> Optional[ModelType]:
        """
        查找满足条件的第一个对象
        
        :param where_clauses: SQLAlchemy 的 WHERE 条件（如 User.email == 'a@example.com'）
        :param options: 预加载选项（如 [joinedload(User.profile)]）
        :return: 匹配的对象，或 None
        
        📌 示例：
          user = await user_repo.find_one(session, User.email == "test@example.com")
        """
        stmt = select(self.model).where(*where_clauses)
        if options:
            stmt = stmt.options(*options)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def find_all(
        self,
        session: AsyncSession,
        *where_clauses,
        options: Optional[List[Any]] = None,
        order_by: Optional[List[Any]] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None
    ) -> Sequence[ModelType]:
        """
        查找所有满足条件的对象（支持分页、排序、预加载）
        
        📌 示例：
          active_users = await user_repo.find_all(
              session,
              User.is_active.is_(True),
              order_by=[User.created_at.desc()],
              limit=10,
              offset=0,
              options=[selectinload(User.roles)]
          )
        """
        stmt = select(self.model).where(*where_clauses)
        if order_by:
            stmt = stmt.order_by(*order_by)
        if options:
            stmt = stmt.options(*options)
        if offset is not None:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        return result.scalars().all()

    async def exists(
        self,
        session: AsyncSession,
        *where_clauses
    ) -> bool:
        """
        检查是否存在满足条件的记录（高效，只查 1 行）
        
        📌 示例：
          if await user_repo.exists(session, User.email == email):
              raise ValueError("Email already registered")
        """
        stmt = select(1).select_from(self.model).where(*where_clauses).limit(1)
        result = await session.execute(stmt)
        return result.scalar() is not None

    async def count(
        self,
        session: AsyncSession,
        *where_clauses
    ) -> int:
        """
        统计满足条件的记录数量
        
        📌 示例：
          total = await user_repo.count(session, User.is_active.is_(True))
        """
        stmt = select(func.count()).select_from(self.model).where(*where_clauses)
        result = await session.execute(stmt)
        return result.scalar() or 0

    # ===================================================================
    # ✏️ 写入方法（Write Methods）
    # ===================================================================

    async def create(
        self,
        session: AsyncSession,
        **data
    ) -> ModelType:
        """
        创建新对象
        
        :param data: 模型字段的键值对（如 name="Alice", email="a@example.com"）
        :return: 创建后的对象（已刷新，包含数据库生成的字段如 ID）
        
        📌 示例：
          user = await user_repo.create(session, name="Alice", email="a@example.com")
        """
        obj = self.model(**data)
        session.add(obj)
        await session.flush()      # 触发 INSERT，获取自增 ID 等
        await session.refresh(obj) # 从 DB 重新加载（确保拿到最新值）
        return obj

    async def update(
        self,
        session: AsyncSession,
        obj: ModelType,
        **data
    ) -> ModelType:
        """
        更新现有对象
        
        :param obj: 已从数据库加载的对象实例
        :param data: 要更新的字段（仅更新存在的属性）
        :return: 更新并刷新后的对象
        
        📌 示例：
          updated_user = await user_repo.update(session, user, name="New Name")
        """
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
        await session.flush()
        await session.refresh(obj)
        return obj

    async def delete(
        self,
        session: AsyncSession,
        obj: ModelType
    ) -> None:
        """
        删除对象
        
        📌 示例：
          user = await repo.get_by_id(session, 123)
          if user:
              await repo.delete(session, user)
        """
        await session.delete(obj)
        await session.flush()

    async def delete_by_id(
        self,
        session: AsyncSession,
        id: Any
    ) -> bool:
        """
        根据主键删除对象
        
        :return: 是否成功删除（True/False）
        
        📌 示例：
          success = await user_repo.delete_by_id(session, 123)
        """
        obj = await self.get_by_id(session, id)
        if obj:
            await self.delete(session, obj)
            return True
        return False

    # ===================================================================
    # 🧪 批量操作（Batch Operations）—— 按需使用
    # ===================================================================

    async def bulk_create(
        self,
        session: AsyncSession,
        data_list: List[dict]
    ) -> List[ModelType]:
        """
        批量创建对象（注意：不触发 ORM 事件，慎用于有默认值/触发器的字段）
        
        ⚠️ 警告：
          - 不会调用 __init__ 或监听器（如 @event.listens_for）
          - 不会自动处理关系（需手动处理外键）
          - 适合简单、高性能插入场景
        
        📌 示例：
          users = await user_repo.bulk_create(session, [
              {"name": "A", "email": "a@example.com"},
              {"name": "B", "email": "b@example.com"}
          ])
        """
        objects = [self.model(**data) for data in data_list]
        session.add_all(objects)
        await session.flush()
        # 刷新每个对象以获取数据库生成的字段（如 ID）
        for obj in objects:
            await session.refresh(obj)
        return objects
```
## 不要在async def 中使用阻塞操作
举个例子：
```python
@app.get('/')
def endpoint():
    time.sleep(10)
```

## Pydantic中进行类型校验

## PRD Template

## 开发可以使用uvicorn，量产环境采用  gunicorn 带worker-class 参数
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

gunicorn main:app -k uvicorn.workers.UvicornWorker -c gunicorn_config.py





## Reference
1. https://github.com/zhanymkanov/fastapi-best-practices/blob/master/README_ZH.md