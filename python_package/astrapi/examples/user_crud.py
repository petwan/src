"""
示例：用户 CRUD 操作

展示如何使用 CRUDBase 创建用户的增删改查操作。
"""

from typing import Any

from app.database.crud import CRUDBase
from app.database.model import Base
from examples.user_model import UserCreate, UserUpdate

from .user_model import User


class CRUDUser(CRUDBase[User, UserCreate, UserUpdate]):
    """
    用户 CRUD 操作类

    继承 CRUDBase，提供用户特定的数据访问方法。
    """

    async def get_by_username(self, username: str) -> User | None:
        """
        根据用户名获取用户

        Args:
            username: 用户名

        Returns:
            User | None: 用户对象，不存在返回 None

        示例:
            >>> user = await crud.get_by_username("zhangsan")
        """
        return await self.get(username=username)

    async def get_by_email(self, email: str) -> User | None:
        """
        根据邮箱获取用户

        Args:
            email: 邮箱地址

        Returns:
            User | None: 用户对象，不存在返回 None

        示例:
            >>> user = await crud.get_by_email("zhangsan@example.com")
        """
        return await self.get(email=email)

    async def search_users(
        self,
        keyword: str | None = None,
        is_active: int | None = None,
        offset: int = 0,
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        搜索用户（分页）

        Args:
            keyword: 搜索关键词（匹配用户名、邮箱、全名）
            is_active: 是否激活
            offset: 偏移量
            limit: 每页数量

        Returns:
            dict: 分页结果

        示例:
            >>> result = await crud.search_users(keyword="张", is_active=1)
            >>> print(result["total"])
            25
        """
        # 构建搜索条件
        search: dict = {}

        if keyword:
            # 使用 OR 条件实现多字段模糊搜索
            # 这里简化处理，实际可以使用 or_ 函数
            search["username"] = ("like", keyword)

        if is_active is not None:
            search["is_active"] = is_active

        # 调用父类的 page 方法
        return await self.page(
            offset=offset,
            limit=limit,
            order_by=[{"id": "desc"}],
            search=search,
            out_schema=None,  # 可以传入 UserResponse 进行序列化
        )


# ============================================
# 使用示例
# ============================================

async def example_usage():
    """
    CRUD 操作使用示例

    展示如何使用 CRUDUser 进行数据操作。
    """

    from sqlalchemy.ext.asyncio import AsyncSession
    from app.database.session import AsyncSessionLocal

    # 创建数据库会话
    async with AsyncSessionLocal() as db:
        # 创建认证上下文
        auth = type("Auth", (), {"db": db, "user": None})()

        # 创建 CRUD 实例
        crud = CRUDUser(User, auth)

        # ========== 创建用户 ==========
        user_create = UserCreate(
            username="zhangsan",
            email="zhangsan@example.com",
            password="123456",
            full_name="张三",
        )
        new_user = await crud.create(user_create)
        print(f"创建用户: {new_user}")

        # ========== 获取用户 ==========
        user = await crud.get(id=new_user.id)
        print(f"获取用户: {user}")

        # ========== 根据用户名获取 ==========
        user = await crud.get_by_username("zhangsan")
        print(f"根据用户名获取: {user}")

        # ========== 列表查询 ==========
        users = await crud.list(search={"is_active": 1})
        print(f"激活用户列表: {len(users)} 个")

        # ========== 搜索用户 ==========
        result = await crud.search_users(keyword="张", limit=5)
        print(f"搜索结果: {result['total']} 条")

        # ========== 更新用户 ==========
        user_update = UserUpdate(full_name="李四", phone="13800138000")
        updated_user = await crud.update(new_user.id, user_update)
        print(f"更新用户: {updated_user}")

        # ========== 删除用户 ==========
        await crud.delete([new_user.id])
        print("用户已删除")


if __name__ == "__main__":
    # 运行示例
    import asyncio

    # 注意：需要先配置数据库连接才能运行此示例
    # asyncio.run(example_usage())
    print("请先配置数据库连接后再运行此示例")
