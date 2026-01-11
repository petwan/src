"""
数据库会话管理模块

提供异步数据库连接引擎、会话工厂和数据库操作辅助函数。

使用示例:
    from app.database.session import get_async_session, init_db, check_db_connection
    from sqlalchemy.ext.asyncio import AsyncSession

    # 获取数据库会话（在路由依赖中使用）
    @app.get("/users")
    async def get_users(db: AsyncSession = Depends(get_async_session)):
        result = await db.execute(select(User))
        return result.scalars().all()

    # 初始化数据库
    await init_db()

    # 检查数据库连接
    await check_db_connection()
"""

from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

from .model import Base


# 创建异步数据库引擎
# 使用 asyncpg 驱动连接 PostgreSQL
engine = create_async_engine(
    settings.DATABASE_URL,  # 数据库连接 URL
    pool_size=settings.DATABASE_POOL_SIZE,  # 连接池大小
    max_overflow=settings.DATABASE_MAX_OVERFLOW,  # 连接池最大溢出数量
    echo=settings.DEBUG,  # 是否输出 SQL 日志（调试模式）
)

# 创建异步会话工厂
# 用于创建数据库会话实例
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,  # 使用异步会话
    expire_on_commit=False,  # 提交后不使对象过期
    autocommit=False,  # 不自动提交
    autoflush=False,  # 不自动刷新
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    获取数据库会话（依赖注入函数）

    这是 FastAPI 依赖注入函数，用于在路由中获取数据库会话。
    自动处理会话的生命周期、异常回滚和会话关闭。

    使用场景:
        - 在 FastAPI 路由中作为依赖项使用
        - 确保每个请求使用独立的会话
        - 异常时自动回滚事务

    Returns:
        AsyncGenerator[AsyncSession, None]: 数据库会话生成器

    示例:
        @app.get("/users/{id}")
        async def get_user(id: int, db: AsyncSession = Depends(get_async_session)):
            user = await db.get(User, id)
            return user
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session  # 将会话注入到路由函数中
        except Exception:
            # 发生异常时回滚事务
            await session.rollback()
            raise
        finally:
            # 确保会话被关闭
            await session.close()


# 仅在测试和开发模式下初始化数据库
async def init_db() -> None:
    """
    初始化数据库表结构

    根据 Base 子类的定义创建所有表。
    仅在开发和测试环境使用，生产环境应使用 Alembic 迁移。

    使用场景:
        - 开发环境下快速创建表结构
        - 测试时初始化测试数据库

    示例:
        await init_db()  # 创建所有表
    """
    async with engine.begin() as conn:
        # 同步执行创建所有表
        await conn.run_sync(Base.metadata.create_all)


async def check_db_connection() -> None:
    """
    检查数据库连接是否正常

    通过执行简单的 SQL 查询来验证数据库连接。

    使用场景:
        - 应用启动时检查数据库连接
        - 健康检查端点

    示例:
        try:
            await check_db_connection()
            print("数据库连接成功")
        except Exception as e:
            print(f"数据库连接失败: {e}")
    """
    async with engine.connect() as conn:
        # 执行简单查询验证连接
        await conn.execute(text("SELECT 1"))
    print("✅ 数据库连接正常")
