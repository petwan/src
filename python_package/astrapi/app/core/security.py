"""
安全相关工具函数模块

提供密码哈希、JWT 令牌生成与解码等安全相关的功能。

使用示例:
    from app.core.security import (
        get_password_hash,
        verify_password,
        create_access_token,
        create_refresh_token,
        decode_token
    )

    # 密码处理
    hashed = get_password_hash("123456")
    is_valid = verify_password("123456", hashed)

    # JWT 令牌
    token = create_access_token(user_id=1)
    payload = decode_token(token)
"""

import datetime
from typing import Any

from jose import jwt
from passlib.context import CryptContext

from app.core.config import settings

# 创建密码上下文，使用 bcrypt 算法进行密码哈希
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证密码

    使用 bcrypt 算法验证明文密码与哈希密码是否匹配。

    Args:
        plain_password: 明文密码
        hashed_password: 哈希密码（存储在数据库中的密码）

    Returns:
        bool: 密码是否匹配

    示例:
        >>> stored_hash = "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5N8yE1qK3QZ4m"
        >>> verify_password("123456", stored_hash)
        True
        >>> verify_password("wrong", stored_hash)
        False
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    生成密码哈希

    使用 bcrypt 算法对明文密码进行哈希处理，用于存储到数据库。

    Args:
        password: 明文密码

    Returns:
        str: 哈希后的密码

    示例:
        >>> hashed = get_password_hash("123456")
        >>> print(hashed[:20])
        $2b$12$LQv3c1yqBWV
    """
    return pwd_context.hash(password)


def create_access_token(subject: str | Any, expires_delta: datetime.timedelta | None = None) -> str:
    """
    创建访问令牌（Access Token）

    生成用于身份验证的 JWT 令牌，默认有效期为配置中的 ACCESS_TOKEN_EXPIRE_MINUTES。

    Args:
        subject: 令牌主题，通常是用户 ID 或用户名
        expires_delta: 自定义过期时间增量，为 None 时使用默认配置

    Returns:
        str: JWT 访问令牌

    示例:
        >>> # 使用默认有效期（30分钟）
        >>> token = create_access_token(user_id=1)
        >>> print(token[:30])
        eyJhbGciOiJIUzI1NiIsInR5cCI6...

        >>> # 自定义有效期（1小时）
        >>> token = create_access_token(user_id=1, expires_delta=datetime.timedelta(hours=1))
    """
    # 计算令牌过期时间
    if expires_delta:
        # 使用自定义过期时间
        expire = datetime.datetime.now(datetime.timezone.utc) + expires_delta
    else:
        # 使用配置文件中的默认过期时间
        expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    # 构建 JWT 载荷（payload）
    to_encode = {"exp": expire, "sub": str(subject)}

    # 使用配置的密钥和算法编码 JWT
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def create_refresh_token(subject: str | Any) -> str:
    """
    创建刷新令牌（Refresh Token）

    生成用于刷新访问令牌的 JWT，有效期较长（默认 7 天）。

    Args:
        subject: 令牌主题，通常是用户 ID 或用户名

    Returns:
        str: JWT 刷新令牌

    示例:
        >>> token = create_refresh_token(user_id=1)
        >>> print(token[:30])
        eyJhbGciOiJIUzI1NiIsInR5cCI6...
    """
    # 计算令牌过期时间（使用配置中的刷新令牌有效期）
    expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        days=settings.REFRESH_TOKEN_EXPIRE_DAYS
    )

    # 构建 JWT 载荷，包含类型标识 "refresh"
    to_encode = {"exp": expire, "sub": str(subject), "type": "refresh"}

    # 使用配置的密钥和算法编码 JWT
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> dict:
    """
    解码 JWT 令牌

    解析 JWT 令牌并返回载荷内容。

    Args:
        token: JWT 令牌字符串

    Returns:
        dict: 解码后的载荷（payload），解码失败返回空字典

    载荷结构:
        {
            "exp": 1234567890,  # 过期时间（Unix 时间戳）
            "sub": "1",         # 主题（通常是用户 ID）
            "type": "access"    # 令牌类型（可选）
        }

    示例:
        >>> token = create_access_token(user_id=1)
        >>> payload = decode_token(token)
        >>> print(payload["sub"])
        1
    """
    try:
        # 使用配置的密钥和算法解码 JWT
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except jwt.JWTError:
        # 令牌无效或已过期，返回空字典
        return {}
