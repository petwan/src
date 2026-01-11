"""
应用配置管理模块

使用 Pydantic Settings 进行配置管理，支持从环境变量和 .env 文件读取配置。

使用示例:
    from app.core.config import settings

    # 获取应用名称
    app_name = settings.APP_NAME

    # 获取数据库URL
    db_url = settings.DATABASE_URL
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置类

    继承自 Pydantic BaseSettings，支持从环境变量和 .env 文件加载配置。

    配置项说明:
        - APP_NAME: 应用名称
        - APP_VERSION: 应用版本
        - DATABASE_URL: 数据库连接URL
        - SECRET_KEY: JWT密钥（必须设置）
        - DEBUG: 调试模式开关
    """

    # Pydantic 配置
    model_config = SettingsConfigDict(
        env_file=".env",           # 从 .env 文件读取配置
        env_file_encoding="utf-8", # 文件编码
        case_sensitive=True,       # 区分大小写
        extra="ignore",            # 忽略额外的字段
    )

    # ========== 应用配置 ==========
    APP_NAME: str = "Enterprise FastAPI App"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Enterprise-grade FastAPI application"
    API_V1_PREFIX: str = "/api/v1"  # API路由前缀
    DEBUG: bool = False              # 调试模式
    RUN_ENVIRONMENT: str = "development"  # 运行环境

    # ========== 安全配置 ==========
    SECRET_KEY: str  # JWT签名密钥（必须配置，建议使用 openssl rand -hex 32 生成）
    ALGORITHM: str = "HS256"  # JWT算法
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30  # 访问令牌有效期（分钟）
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7     # 刷新令牌有效期（天）

    # ========== 数据库配置 ==========
    DATABASE_URL: str ='postgresql+asyncpg://user:password@localhost:5432/fastapi_db'
    DATABASE_POOL_SIZE: int = 20        # 数据库连接池大小
    DATABASE_MAX_OVERFLOW: int = 10     # 连接池最大溢出数量

    # ========== CORS配置 ==========
    ALLOW_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8000"]
    # 允许跨域访问的源列表，生产环境需要配置具体域名

    # ========== 日志配置 ==========
    LOG_LEVEL: str = "INFO"  # 日志级别: DEBUG, INFO, WARNING, ERROR

    # ========== Redis配置 ==========
    REDIS_URL: str = "redis://localhost:6379/0"  # Redis连接URL

    # ========== Celery配置 ==========
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"  # Celery消息代理URL
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"  # Celery结果存储URL

    # ========== 微信小程序配置 ==========
    WECHAT_APP_ID: str = ""  # 微信小程序 AppID
    WECHAT_APP_SECRET: str = ""  # 微信小程序 AppSecret


@lru_cache
def get_settings() -> Settings:
    """
    获取配置实例（单例模式）

    使用 lru_cache 缓存，确保每次调用返回相同的实例，
    避免重复从 .env 文件读取配置。

    Returns:
        Settings: 配置实例

    示例:
        settings = get_settings()
        print(settings.APP_NAME)
    """
    return Settings()


# 全局配置实例，其他模块直接导入使用
settings = get_settings()
