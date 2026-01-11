"""
认证模块 - CRUD 操作

提供用户相关的数据访问层操作。
"""

from typing import Any

from app.database.crud import CRUDBase
from app.database.model import Base
from app.api.v1.auth.schemas import UserCreate, UserUpdate
from app.api.v1.auth.models import User


class CRUDUser(CRUDBase[User, UserCreate, UserUpdate]):
    """
    用户 CRUD 操作类

    继承 CRUDBase，提供用户特定的数据访问方法。
    """

    async def get_by_openid(self, openid: str) -> User | None:
        """
        根据 openid 获取用户

        Args:
            openid: 微信 openid

        Returns:
            User | None: 用户对象，不存在返回 None

        示例:
            >>> user = await crud.get_by_openid("oXYZ123...")
        """
        return await self.get(openid=openid)

    async def get_by_phone(self, phone: str) -> User | None:
        """
        根据手机号获取用户

        Args:
            phone: 手机号

        Returns:
            User | None: 用户对象，不存在返回 None

        示例:
            >>> user = await crud.get_by_phone("13800138000")
        """
        return await self.get(phone=phone)

    async def create_wechat_user(
        self,
        openid: str,
        nickname: str | None = None,
        avatar_url: str | None = None,
    ) -> User:
        """
        创建微信用户

        用于微信小程序登录时创建新用户。

        Args:
            openid: 微信 openid
            nickname: 昵称（可选）
            avatar_url: 头像URL（可选）

        Returns:
            User: 创建的用户对象

        示例:
            >>> user = await crud.create_wechat_user(
            ...     openid="oXYZ123...",
            ...     nickname="张三",
            ...     avatar_url="https://..."
            ... )
        """
        user_data = {
            "openid": openid,
            "nickname": nickname,
            "avatar_url": avatar_url,
        }
        return await self.create(user_data)

    async def update_user_info(
        self,
        user_id: int,
        nickname: str | None = None,
        avatar_url: str | None = None,
        **kwargs,
    ) -> User:
        """
        更新用户信息

        更新用户的基本信息，通常用于用户首次授权后更新昵称和头像。

        Args:
            user_id: 用户ID
            nickname: 昵称（可选）
            avatar_url: 头像URL（可选）
            **kwargs: 其他可更新的字段

        Returns:
            User: 更新后的用户对象

        示例:
            >>> user = await crud.update_user_info(
            ...     user_id=1,
            ...     nickname="李四",
            ...     avatar_url="https://..."
            ... )
        """
        update_data: dict = {}
        if nickname is not None:
            update_data["nickname"] = nickname
        if avatar_url is not None:
            update_data["avatar_url"] = avatar_url
        update_data.update(kwargs)

        return await self.update(user_id, UserUpdate(**update_data))

    async def update_last_login(self, user_id: int) -> None:
        """
        更新用户最后登录时间

        Args:
            user_id: 用户ID

        示例:
            >>> await crud.update_last_login(user_id=1)
        """
        # 使用 set 方法更新字段
        await self.set([user_id], last_login_at="now()")