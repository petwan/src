"""
依赖注入模块

提供 FastAPI 依赖注入函数，用于在路由中获取数据库会话、认证信息等。

使用示例:
    from app.core.dependencies import get_db

    @app.get("/users")
    async def get_users(db: AsyncSession = Depends(get_db)):
        result = await db.execute(select(User))
        return result.scalars().all()
"""

from collections.abc import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_async_session


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    获取数据库会话（依赖注入函数）

    这是一个包装函数，用于简化在路由中获取数据库会话的调用。

    使用场景:
        - 在 FastAPI 路由中作为依赖项使用
        - 确保每个请求使用独立的数据库会话
        - 自动处理会话的生命周期、异常回滚和会话关闭

    Returns:
        AsyncGenerator[AsyncSession, None]: 数据库会话生成器

    示例:
        @app.get("/users/{id}")
        async def get_user(id: int, db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(User).where(User.id == id))
            return result.scalar_one_or_none()

    注意:
        此函数实际上是对 get_async_session 的包装，两者功能相同。
        可以根据项目习惯选择使用 get_db 或 get_async_session。
    """
    async for session in get_async_session():
        yield session
