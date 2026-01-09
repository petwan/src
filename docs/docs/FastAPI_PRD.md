# FastAPI企业级项目需求文档

## 1. 项目概述

### 1.1 项目名称
Astrapi


### 1.2 项目描述
基于FastAPI构建的企业级后端服务框架，采用模块化架构设计，支持高并发、异步处理、数据验证等功能，提供完整的用户认证、权限管理、数据持久化、API版本控制等功能。

### 1.3 项目目标
- 提供高性能、可扩展的Web API服务
- 支持微服务架构
- 实现安全可靠的认证和授权机制
- 提供完整的错误处理和日志系统
- 支持容器化部署

## 2. 技术栈

### 2.1 主要技术
- Python 3.11+
- FastAPI 0.104+
- SQLAlchemy 2.0+
- PostgreSQL/MySQL
- Redis
- Celery
- Docker
- Poetry

### 2.2 开发工具
- Ruff (代码检查和格式化)
- MyPy (类型检查)
- PyTest (测试)


### 2.3 部署技术
- Docker
- Docker Compose
- Kubernetes (可选)

## 3. 架构设计
### 3.1 模块化架构
项目采用模块化架构，每个业务功能模块包含其完整的组件：

```console
modules/
├── auth/             # 认证模块
│   ├── __init__.py
│   ├── router.py     # 路由
│   ├── models.py     # 数据模型
│   ├── schemas.py    # 序列化模型
│   ├── crud.py       # 数据访问层
│   ├── service.py    # 业务逻辑
│   ├── dependencies.py # 依赖项
│   └── tests/
└── users/            # 用户模块
    ├── __init__.py
    ├── router.py
    ├── models.py
    ├── schemas.py
    ├── crud.py
    ├── service.py
    ├── dependencies.py
    └── tests/
```

### 3.2 核心组件
- **core/**: 核心功能（配置、安全、中间件、异常处理）
- **api/**: API版本控制和路由聚合

## 4. 项目核心代码示例
### 4.1 核心配置
配置代码结构如下，并按照项目实际需求添加配置项，并生成对应的 .env.example 文件。
```python
# app/core/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # App
    APP_NAME: str = "Enterprise FastAPI App"
    APP_VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"

    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Database
    DATABASE_URL: str
    DATABASE_POOL_SIZE: int = 20
    DATABASE_POOL_OVERFLOW: int = 10

    # CORS
    ALLOWED_ORIGINS: list[str]

    # Logging
    LOG_LEVEL: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
```

### 4.2 logging模块示例
```python
# app/core/logging.py
"""Loguru 日志配置"""

from loguru import logger

from app.core.config import settings

# 定义日志格式
LOG_FORMAT_DEBUG = "<level>{level}</level>: {message}: {name}: {function}:{line}"


# 日志级别映射
LOG_LEVELS = {
    "INFO": "INFO",
    "WARN": "WARNING",
    "ERROR": "ERROR",
    "DEBUG": "DEBUG",
}


def configure_logging():
    """配置 Loguru 日志"""
    log_level = settings.LOG_LEVEL.upper()

    # 验证日志级别是否有效
    if log_level not in LOG_LEVELS:
        # 使用 ERROR 作为默认日志级别
        logger.remove()
        logger.add(lambda msg: None, level="ERROR")
        return

    # 移除默认的 handler
    logger.remove()

    # 配置日志级别
    level = LOG_LEVELS.get(log_level, "ERROR")

    if log_level == "DEBUG":
        # DEBUG 模式使用详细格式
        logger.add(
            lambda msg: print(msg, end=""),
            level=level,
            format=LOG_FORMAT_DEBUG,
        )
    else:
        # 其他模式使用简单格式
        logger.add(
            lambda msg: print(msg, end=""),
            level=level,
            format="<level>{level}</level>: {message}",
        )
```

### 4.3 响应体结构定义
创建标准的 API 响应体结构，code 使用 HTTP 状态码，并根据实际业务需求添加额外的结构体定义。
```python
# app/core/response.py
from fastapi import status
from pydantic import BaseModel, ConfigDict, Field
from typing import Generic, TypeVar

T = TypeVar("T")

class BaseResponse(BaseModel, Generic[T]):
    """
    标准 API 响应结构（code 使用 HTTP 状态码）

    示例（成功）：
        {
            "code": 200,
            "msg": "Success",
            "data": { ... }
        }

    示例（错误）：
        {
            "code": 400,
            "msg": "用户名或密码错误",
            "data": null
        }
    """

    code: int = Field(default=status.HTTP_200_OK, description="HTTP 状态码")
    msg: str = Field(default="Success", description="人类可读的提示信息")
    data: T | None = Field(default=None, description="业务数据")

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "code": 200,
                "msg": "操作成功",
                "data": {"id": 1, "name": "示例用户"},
            }
        },
    )
```

### 4.4 alembic 数据库管理
后端数据库采用postgresql，使用 alembic 进行数据库管理。

在项目根目录下使用 alembic init ./app/alembic 创建数据库迁移脚本目录，并设置默认的`sqlalchemy.url = postgresql+asyncpg://user:password@localhost:5432/fastapi_db`

在 app/database 目录下保存如下内容：
1. 异步数据库连接，用于FastAPI进行依赖注入，示例代码如下
   ```python
   # app/database/session.py 
    from collections.abc import AsyncGenerator

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from app.core.config import settings

    from .model import Base

    # 创建异步数据库引擎
    engine = create_async_engine(
        settings.DATABASE_URL,
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_MAX_OVERFLOW,
        echo=settings.DEBUG,
    )

    # 创建异步会话工厂
    AsyncSessionLocal = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )


    async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
        async with AsyncSessionLocal() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()


    # 仅在测试和开发模式下初始化数据库
    async def init_db():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)


    async def check_db_connection():
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        print("✅ 数据库连接正常")

   ```
2. 创建Base类，继承自 AsyncAttrs和 DeclarativeBase，为后续的数据库模型提供基类，示例代码如下：
   ```python
    import re
    from typing import ClassVar

    from sqlalchemy import inspect
    from sqlalchemy.ext.asyncio import AsyncAttrs  # ← 新增
    from sqlalchemy.orm import DeclarativeBase, declared_attr


    def resolve_table_name(name: str) -> str:
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


    class Base(AsyncAttrs, DeclarativeBase):  # ← AsyncAttrs 在前
        __abstract__ = True

        __repr_attrs__: ClassVar[list[str]] = []
        __repr_max_length__: ClassVar[int] = 15

        @declared_attr.directive
        def __tablename__(cls) -> str:
            return resolve_table_name(cls.__name__)

        @property
        def _id_str(self) -> str:
            identity = inspect(self).identity
            if identity is None:
                return "None"
            return "-".join(str(x) for x in identity) if len(identity) > 1 else str(identity[0])

        @property
        def _repr_attrs_str(self) -> str:
            if not self.__repr_attrs__:
                return ""
            values = []
            single = len(self.__repr_attrs__) == 1
            for key in self.__repr_attrs__:
                if not hasattr(self, key):
                    raise AttributeError(
                        f"{self.__class__.__name__} has invalid __repr_attrs__ key: {key}"
                    )
                val = getattr(self, key)
                s = str(val)
                if len(s) > self.__repr_max_length__:
                    s = s[: self.__repr_max_length__] + "..."
                if isinstance(val, str):
                    s = f"'{s}'"
                values.append(s if single else f"{key}:{s}")
            return " ".join(values)

        def __repr__(self) -> str:
            id_part = f"#{self._id_str}" if self._id_str != "None" else ""
            attrs = f" {self._repr_attrs_str}" if self._repr_attrs_str else ""
            # 简化输出，避免多余的空白
            if not id_part and not attrs:
                return f"<{self.__class__.__name__}>"

            return f"<{self.__class__.__name__}{id_part}{attrs}>"
   ```
### 4.5 创建基础的CRUD操作类封装
示例代码如下，必要的话，可以进行代码的优化：
```python
"""
通用异步 CRUD 基类，配合 Base 使用
"""

import builtins
from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from app.api.v1.module_system.auth.schema import AuthSchema
from app.core.permission import Permission
from pydantic import BaseModel
from sqlalchemy import Select, asc, delete, desc, func, select, update
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import Result
from sqlalchemy.orm import selectinload
from sqlalchemy.sql.elements import ColumnElement

from app.core.exception import CustomException

from .model import Base

ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)
OutSchemaType = TypeVar("OutSchemaType", bound=BaseModel)


class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """基础数据层"""

    def __init__(self, model: type[ModelType], auth: AuthSchema) -> None:
        self.model = model
        self.auth = auth

    async def get(
        self, preload: list[str | Any] | None = None, **kwargs
    ) -> ModelType | None:
        try:
            conditions = await self.__build_conditions(**kwargs)
            sql = select(self.model).where(*conditions)
            for opt in self.__loader_options(preload):
                sql = sql.options(opt)
            sql = await self.__filter_permissions(sql)
            result: Result = await self.auth.db.execute(sql)
            return result.scalars().first()
        except Exception as e:
            raise CustomException(msg=f"获取查询失败: {str(e)}")

    async def list(
        self,
        search: dict | None = None,
        order_by: list[dict[str, str]] | None = None,
        preload: list[str | Any] | None = None,
    ) -> Sequence[ModelType]:
        try:
            conditions = await self.__build_conditions(**search) if search else []
            order = order_by or [{"id": "asc"}]
            sql = select(self.model).where(*conditions).order_by(*self.__order_by(order))
            for opt in self.__loader_options(preload):
                sql = sql.options(opt)
            sql = await self.__filter_permissions(sql)
            result: Result = await self.auth.db.execute(sql)
            return result.scalars().all()
        except Exception as e:
            raise CustomException(msg=f"列表查询失败: {str(e)}")

    async def tree_list(
        self,
        search: dict | None = None,
        order_by: builtins.list[dict[str, str]] | None = None,
        children_attr: str = "children",
        preload: builtins.list[str | Any] | None = None,
    ) -> Sequence[ModelType]:
        try:
            conditions = await self.__build_conditions(**search) if search else []
            order = order_by or [{"id": "asc"}]
            sql = select(self.model).where(*conditions).order_by(*self.__order_by(order))

            final_preload = preload
            if preload is None and children_attr and hasattr(self.model, children_attr):
                model_defaults = getattr(self.model, "__loader_options__", [])
                final_preload = list(model_defaults) + [children_attr]

            for opt in self.__loader_options(final_preload):
                sql = sql.options(opt)

            sql = await self.__filter_permissions(sql)
            result: Result = await self.auth.db.execute(sql)
            return result.scalars().all()
        except Exception as e:
            raise CustomException(msg=f"树形列表查询失败: {str(e)}")

    async def page(
        self,
        offset: int,
        limit: int,
        order_by: builtins.list[dict[str, str]],
        search: dict,
        out_schema: type[OutSchemaType],
        preload: builtins.list[str | Any] | None = None,
    ) -> dict[str, Any]:
        try:
            conditions = await self.__build_conditions(**search) if search else []
            order = order_by or [{"id": "asc"}]
            sql = select(self.model).where(*conditions).order_by(*self.__order_by(order))
            for opt in self.__loader_options(preload):
                sql = sql.options(opt)
            sql = await self.__filter_permissions(sql)

            mapper = sa_inspect(self.model)
            pk_cols = list(getattr(mapper, "primary_key", []))
            if pk_cols:
                count_sql = select(func.count(pk_cols[0])).select_from(self.model)
            else:
                count_sql = select(func.count()).select_from(self.model)
            if conditions:
                count_sql = count_sql.where(*conditions)
            count_sql = await self.__filter_permissions(count_sql)

            total_result = await self.auth.db.execute(count_sql)
            total = total_result.scalar() or 0

            result: Result = await self.auth.db.execute(sql.offset(offset).limit(limit))
            objs = result.scalars().all()

            return {
                "page_no": offset // limit + 1 if limit else 1,
                "page_size": limit if limit else 10,
                "total": total,
                "has_next": offset + limit < total,
                "items": [out_schema.model_validate(obj).model_dump() for obj in objs],
            }
        except Exception as e:
            raise CustomException(msg=f"分页查询失败: {str(e)}")

    async def create(self, data: CreateSchemaType | dict) -> ModelType:
        try:
            obj_dict = data if isinstance(data, dict) else data.model_dump()
            obj = self.model(**obj_dict)

            if self.auth.user:
                if hasattr(obj, "created_id"):
                    obj.created_id = self.auth.user.id
                if hasattr(obj, "updated_id"):
                    obj.updated_id = self.auth.user.id

            self.auth.db.add(obj)
            await self.auth.db.flush()
            await self.auth.db.refresh(obj)
            return obj
        except Exception as e:
            raise CustomException(msg=f"创建失败: {str(e)}")

    async def update(self, id: int, data: UpdateSchemaType | dict) -> ModelType:
        try:
            obj_dict = (
                data
                if isinstance(data, dict)
                else data.model_dump(exclude_unset=True, exclude={"id"})
            )
            obj = await self.get(id=id)
            if not obj:
                raise CustomException(msg="更新对象不存在")

            if self.auth.user and hasattr(obj, "updated_id"):
                obj.updated_id = self.auth.user.id

            for key, value in obj_dict.items():
                if hasattr(obj, key):
                    setattr(obj, key, value)

            await self.auth.db.flush()
            await self.auth.db.refresh(obj)

            verify_obj = await self.get(id=id)
            if not verify_obj:
                raise CustomException(msg="更新失败，对象不存在或无权限访问")

            return obj
        except Exception as e:
            raise CustomException(msg=f"更新失败: {str(e)}")

    async def delete(self, ids: builtins.list[int]) -> None:
        try:
            mapper = sa_inspect(self.model)
            pk_cols = list(getattr(mapper, "primary_key", []))
            if not pk_cols:
                raise CustomException(msg="模型缺少主键，无法删除")
            if len(pk_cols) > 1:
                raise CustomException(msg="暂不支持复合主键的批量删除")

            sql = delete(self.model).where(pk_cols[0].in_(ids))
            await self.auth.db.execute(sql)
            await self.auth.db.flush()
        except Exception as e:
            raise CustomException(msg=f"删除失败: {str(e)}")

    async def clear(self) -> None:
        try:
            sql = delete(self.model)
            await self.auth.db.execute(sql)
            await self.auth.db.flush()
        except Exception as e:
            raise CustomException(msg=f"清空失败: {str(e)}")

    async def set(self, ids: builtins.list[int], **kwargs) -> None:
        try:
            mapper = sa_inspect(self.model)
            pk_cols = list(getattr(mapper, "primary_key", []))
            if not pk_cols:
                raise CustomException(msg="模型缺少主键，无法更新")
            if len(pk_cols) > 1:
                raise CustomException(msg="暂不支持复合主键的批量更新")

            sql = update(self.model).where(pk_cols[0].in_(ids)).values(**kwargs)
            await self.auth.db.execute(sql)
            await self.auth.db.flush()
        except CustomException:
            raise
        except Exception as e:
            raise CustomException(msg=f"批量更新失败: {str(e)}")

    async def __filter_permissions(self, sql: Select) -> Select:
        filter = Permission(model=self.model, auth=self.auth)
        return await filter.filter_query(sql)

    async def __build_conditions(self, **kwargs) -> builtins.list[ColumnElement]:
        conditions = []
        for key, value in kwargs.items():
            if value is None or value == "":
                continue

            attr = getattr(self.model, key)
            if isinstance(value, tuple):
                seq, val = value
                if seq == "None":
                    conditions.append(attr.is_(None))
                elif seq == "not None":
                    conditions.append(attr.isnot(None))
                elif seq == "date" and val:
                    conditions.append(func.date_format(attr, "%Y-%m-%d") == val)
                elif seq == "month" and val:
                    conditions.append(func.date_format(attr, "%Y-%m") == val)
                elif seq == "like" and val:
                    conditions.append(attr.like(f"%{val}%"))
                elif seq == "in" and val:
                    conditions.append(attr.in_(val))
                elif seq == "between" and isinstance(val, (list, tuple)) and len(val) == 2:
                    conditions.append(attr.between(val[0], val[1]))
                elif seq in ("!=", "ne") and val:
                    conditions.append(attr != val)
                elif seq in (">", "gt") and val:
                    conditions.append(attr > val)
                elif seq in (">=", "ge") and val:
                    conditions.append(attr >= val)
                elif seq in ("<", "lt") and val:
                    conditions.append(attr < val)
                elif seq in ("<=", "le") and val:
                    conditions.append(attr <= val)
                elif seq in ("==", "eq") and val:
                    conditions.append(attr == val)
            else:
                conditions.append(attr == value)
        return conditions

    def __order_by(self, order_by: builtins.list[dict[str, str]]) -> builtins.list[ColumnElement]:
        columns = []
        for order in order_by:
            for field, direction in order.items():
                column = getattr(self.model, field)
                columns.append(desc(column) if direction.lower() == "desc" else asc(column))
        return columns

    def __loader_options(self, preload: builtins.list[str | Any] | None = None) -> builtins.list[Any]:
        options = []
        model_loader_options = getattr(self.model, "__loader_options__", [])
        all_preloads = set(model_loader_options)

        if preload is not None:
            if preload == []:
                all_preloads = set()
            else:
                for opt in preload:
                    if isinstance(opt, str):
                        all_preloads.add(opt)

        for opt in all_preloads:
            if isinstance(opt, str):
                if hasattr(self.model, opt):
                    options.append(selectinload(getattr(self.model, opt)))
            else:
                options.append(opt)

        return options
```
### 4.6 API接口
示例代码：
```python
from fastapi import APIRouter

api_router = APIRouter(
    # default_response_class=JSONResponse,
    # responses={
    #     400: {"model": ErrorResponse},
    #     401: {"model": ErrorResponse},
    #     403: {"model": ErrorResponse},
    #     404: {"model": ErrorResponse},
    #     500: {"model": ErrorResponse},
    # },
)
# api_router.include_router(


@api_router.get("/healthcheck", include_in_schema=False)
def healthcheck():
    """Simple Healthcheck endpoint"""
    return {"status": "ok"}
```

### 4.7 main.py
```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api import api_router
from app.core.config import settings
from app.core.exception import handle_exception
from app.database.core import check_db_connection, engine, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """项目启动时执行"""
    logger.info(f"{settings.APP_NAME} {settings.APP_VERSION} 启动中...")
    # 数据库连接测试
    await check_db_connection()
    # 对于开发模式下初始化数据库
    if settings.RUN_ENVIRONMENT:
        logger.info("测试开发环境初始化数据库...")
        await init_db()
    yield

    logger.info(f"{settings.APP_NAME} {settings.APP_VERSION} 关闭中...")

    # 关闭数据库连接池
    try:
        await engine.dispose()
        logger.info("数据库连接池已关闭")
    except Exception as e:
        logger.error(f"数据库连接池关闭失败: {e}")

    logger.info("应用已关闭")


app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

handle_exception(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(api_router)
```
### 4.8 异常处理
代码应放置于 app/core/exception.py 中


## 5. 权限管理
- 基于角色的权限控制(RBAC)
- API端点访问控制
- 数据级权限控制

## 6. 非功能性需求

### 6.1 性能需求
- 并发用户支持：1000+
- 响应时间：< 200ms
- 吞吐量：1000 req/s

### 6.2 可靠性需求
- 系统可用性：99.9%
- 数据一致性保证
- 错误恢复机制

