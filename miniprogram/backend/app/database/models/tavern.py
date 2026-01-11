"""
酒馆管理模块数据模型

包含:
- User: 用户表（微信小程序用户）
- Tavern: 酒馆表
- TavernMember: 酒馆成员表（包含管理员和访客）
"""

from datetime import datetime
from enum import Enum as PyEnum
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.model import Base

if TYPE_CHECKING:
    pass


class UserRole(str, PyEnum):
    """用户角色枚举"""
    CREATOR = "creator"  # 创建者
    ADMIN = "admin"      # 管理员
    GUEST = "guest"      # 访客


class User(Base):
    """用户表

    存储微信小程序用户信息
    """
    __tablename__ = "user"
    __repr_attrs__ = ["nickname", "openid"]

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, comment="用户ID")
    openid: Mapped[str] = mapped_column(String(100), unique=True, index=True, comment="微信OpenID")
    unionid: Mapped[str | None] = mapped_column(String(100), index=True, comment="微信UnionID")
    nickname: Mapped[str] = mapped_column(String(50), comment="用户昵称")
    avatar_url: Mapped[str | None] = mapped_column(String(500), comment="头像URL")
    gender: Mapped[int | None] = mapped_column(Integer, comment="性别：0-未知，1-男，2-女")
    country: Mapped[str | None] = mapped_column(String(50), comment="国家")
    province: Mapped[str | None] = mapped_column(String(50), comment="省份")
    city: Mapped[str | None] = mapped_column(String(50), comment="城市")
    language: Mapped[str | None] = mapped_column(String(20), comment="语言")
    phone: Mapped[str | None] = mapped_column(String(20), comment="手机号")
    is_active: Mapped[bool] = mapped_column(default=True, comment="是否激活")
    last_login_time: Mapped[datetime | None] = mapped_column(
        DateTime, comment="最后登录时间"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), comment="创建时间"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now(), comment="更新时间"
    )

    # 关系
    owned_taverns = relationship(
        "Tavern",
        back_populates="owner",
        foreign_keys="Tavern.owner_id"
    )
    memberships = relationship(
        "TavernMember",
        back_populates="user",
        cascade="all, delete-orphan"
    )


class Tavern(Base):
    """酒馆表

    用户的"家庭酒吧"，酒馆管理模块的核心实体
    """
    __tablename__ = "tavern"
    __repr_attrs__ = ["name"]

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, comment="酒馆ID")
    name: Mapped[str] = mapped_column(String(50), comment="酒馆名称")
    cover_url: Mapped[str | None] = mapped_column(String(500), comment="封面图URL")
    description: Mapped[str | None] = mapped_column(Text, comment="酒馆简介")
    welcome_message: Mapped[str | None] = mapped_column(String(200), comment="欢迎语")
    owner_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("user.id", ondelete="CASCADE"),
        comment="创建者ID"
    )
    is_active: Mapped[bool] = mapped_column(default=True, comment="是否启用")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), comment="创建时间"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now(), comment="更新时间"
    )

    # 关系
    owner = relationship("User", back_populates="owned_taverns", foreign_keys=[owner_id])
    members = relationship(
        "TavernMember",
        back_populates="tavern",
        cascade="all, delete-orphan"
    )


class TavernMember(Base):
    """酒馆成员表

    记录酒馆的创建者、管理员和访客
    """
    __tablename__ = "tavern_member"
    __repr_attrs__ = ["role"]

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, comment="成员ID")
    tavern_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("tavern.id", ondelete="CASCADE"),
        comment="酒馆ID"
    )
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("user.id", ondelete="CASCADE"),
        comment="用户ID"
    )
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole),
        default=UserRole.GUEST,
        comment="角色：creator-创建者，admin-管理员，guest-访客"
    )
    visit_count: Mapped[int] = mapped_column(Integer, default=0, comment="来访次数")
    last_visit_time: Mapped[datetime | None] = mapped_column(
        DateTime, comment="最后访问时间"
    )
    joined_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), comment="加入时间"
    )
    invited_by: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("user.id"),
        comment="邀请人ID"
    )
    is_active: Mapped[bool] = mapped_column(default=True, comment="是否有效")

    # 关系
    tavern = relationship("Tavern", back_populates="members")
    user = relationship("User", back_populates="memberships")
