"""
数据模型基类模块

定义 SQLAlchemy ORM 模型的基类，提供表名自动映射和友好的字符串表示。

使用示例:
    from app.database.model import Base
    from sqlalchemy import Column, Integer, String

    class User(Base):
        __tablename__ = "users"
        __repr_attrs__ = ["username", "email"]

        id = Column(Integer, primary_key=True)
        username = Column(String(50))
        email = Column(String(100))

    # 表名自动从类名转换: User -> users
    # 打印对象: <User#1 username:'zhangsan' email:'zhang@...'>
"""

import re
from typing import ClassVar

from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, declared_attr


def resolve_table_name(name: str) -> str:
    """
    将驼峰命名转换为蛇形命名

    例如:
        "User" -> "user"
        "UserProfile" -> "user_profile"
        "APIResponse" -> "api_response"

    Args:
        name: 驼峰命名的字符串

    Returns:
        str: 蛇形命名的字符串

    示例:
        >>> resolve_table_name("UserProfile")
        'user_profile'
    """
    # 第一步: 在大写字母前添加下划线（"UserProfile" -> "_User_Profile"）
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    # 第二步: 在小写或数字和大写字母之间添加下划线，然后转为小写
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


class Base(AsyncAttrs, DeclarativeBase):
    """
    数据库模型基类

    所有 ORM 模型都应继承此基类，提供以下功能:
        - 自动根据类名生成表名（驼峰 -> 蛇形）
        - 友好的对象字符串表示（__repr__）
        - 支持异步操作（AsyncAttrs）

    属性:
        __abstract__: 标记为抽象基类
        __repr_attrs__: 用于 __repr__ 显示的字段列表
        __repr_max_length__: 字符串字段的最大显示长度

    示例:
        >>> class User(Base):
        ...     __repr_attrs__ = ["username"]
        ...     id = Column(Integer, primary_key=True)
        ...     username = Column(String(50))
        >>> user = User(id=1, username="zhangsan")
        >>> print(user)
        <User#1 username:'zhangsan'>
    """

    __abstract__ = True  # 标记为抽象类，不会在数据库中创建表

    # 用于 __repr__ 显示的字段列表
    __repr_attrs__: ClassVar[list[str]] = []

    # 字符串字段的最大显示长度
    __repr_max_length__: ClassVar[int] = 15

    @declared_attr.directive
    def __tablename__(cls) -> str:
        """
        自动生成表名

        使用 resolve_table_name 函数将类名转换为表名。

        Returns:
            str: 转换后的表名

        示例:
            >>> class UserProfile(Base): pass
            >>> UserProfile.__tablename__
            'user_profile'
        """
        return resolve_table_name(cls.__name__)

    @property
    def _id_str(self) -> str:
        """
        获取对象的 ID 字符串

        支持单主键和复合主键。

        Returns:
            str: ID 字符串，主键为多个时用 "-" 连接

        示例:
            >>> user = User(id=1)
            >>> user._id_str
            '1'
        """
        identity = inspect(self).identity
        if identity is None:
            return "None"
        # 复合主键: (1, 2) -> "1-2"
        return "-".join(str(x) for x in identity) if len(identity) > 1 else str(identity[0])

    @property
    def _repr_attrs_str(self) -> str:
        """
        生成 __repr__ 中的属性字符串

        根据 __repr_attrs__ 配置生成属性显示字符串。

        Returns:
            str: 属性字符串，如 "username:'zhangsan' email:'zhang@...'"

        示例:
            >>> user = User(id=1, username="verylongusername@example.com")
            >>> print(user._repr_attrs_str)
            username:'verylongusernam...'
        """
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
            # 截断过长的字符串
            if len(s) > self.__repr_max_length__:
                s = s[: self.__repr_max_length__] + "..."
            # 字符串值用单引号包裹
            if isinstance(val, str):
                s = f"'{s}'"
            values.append(s if single else f"{key}:{s}")
        return " ".join(values)

    def __repr__(self) -> str:
        """
        生成对象的字符串表示

        格式: <ClassName#id attr1:value1 attr2:value2>

        Returns:
            str: 对象的字符串表示

        示例:
            >>> user = User(id=1, username="zhangsan", email="zhang@example.com")
            >>> print(user)
            <User#1 username:'zhangsan' email:'zhang@ex...'>
        """
        id_part = f"#{self._id_str}" if self._id_str != "None" else ""
        attrs = f" {self._repr_attrs_str}" if self.__repr_attrs_str else ""
        if not id_part and not attrs:
            return f"<{self.__class__.__name__}>"

        return f"<{self.__class__.__name__}{id_part}{attrs}>"
