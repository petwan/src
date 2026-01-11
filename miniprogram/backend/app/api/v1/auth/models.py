"""
认证模块 - 用户数据模型

支持微信小程序登录的用户模型，包含微信 openid 和用户基本信息。
"""

from datetime import datetime
from typing import ClassVar

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import relationship

from app.database.model import Base


class User(Base):
    """
    用户模型

    支持微信小程序登录的用户表。

    表名: users
    """

    __repr_attrs__ = ["nickname", "openid"]

    # 主键
    id = Column(Integer, primary_key=True, index=True, comment="用户ID")

    # 微信相关信息
    openid = Column(
        String(100), unique=True, index=True, nullable=False, comment="微信openid"
    )
    unionid = Column(String(100), index=True, comment="微信unionid（开放平台）")

    # 基本信息
    nickname = Column(String(100), comment="昵称")
    avatar_url = Column(String(500), comment="头像URL")
    phone = Column(String(20), comment="手机号")

    # 扩展信息
    gender = Column(Integer, default=0, comment="性别: 0=未知, 1=男, 2=女")
    country = Column(String(50), comment="国家")
    province = Column(String(50), comment="省份")
    city = Column(String(50), comment="城市")
    language = Column(String(20), default="zh_CN", comment="语言")

    # 状态字段
    is_active = Column(Integer, default=1, comment="是否激活: 1=激活, 0=禁用")
    is_superuser = Column(Integer, default=0, comment="是否超级管理员: 1=是, 0=否")

    # 审计字段
    last_login_at = Column(DateTime, comment="最后登录时间")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
    updated_at = Column(
        DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间"
    )

    # 备注
    remark = Column(Text, comment="备注")
