"""
FastAPI 应用主入口文件

配置和启动 FastAPI 应用，包含路由注册、中间件配置、异常处理等。

使用示例:
    # 直接运行
    python -m app.main

    # 或使用 uvicorn
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api import api_router
from app.core.config import settings
from app.core.exception import handle_exception
from app.core.logging import configure_logging
from app.database.session import check_db_connection, engine, init_db

# 配置日志系统
configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理函数

    处理应用启动和关闭时的初始化和清理工作。

    启动时执行:
        1. 检查数据库连接
        2. 开发环境下初始化数据库表

    关闭时执行:
        1. 关闭数据库连接池

    Args:
        app: FastAPI 应用实例

    示例:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # 启动逻辑
            logger.info("应用启动中...")
            yield
            # 关闭逻辑
            logger.info("应用关闭中...")
    """
    # ========== 应用启动 ==========
    logger.info(f"{settings.APP_NAME} {settings.APP_VERSION} 启动中...")

    # 测试数据库连接
    await check_db_connection()

    # 开发环境下初始化数据库（生产环境应使用 Alembic 迁移）
    if settings.RUN_ENVIRONMENT == "development":
        logger.info("开发环境初始化数据库...")
        await init_db()

    # 交出控制权，应用开始接收请求
    yield

    # ========== 应用关闭 ==========
    logger.info(f"{settings.APP_NAME} {settings.APP_VERSION} 关闭中...")

    # 关闭数据库连接池
    try:
        await engine.dispose()
        logger.info("数据库连接池已关闭")
    except Exception as e:
        logger.error(f"数据库连接池关闭失败: {e}")

    logger.info("应用已关闭")


# 创建 FastAPI 应用实例
app = FastAPI(
    title=settings.APP_NAME,                          # 应用标题
    description=settings.APP_DESCRIPTION,              # 应用描述
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",  # OpenAPI JSON 文档路径
    docs_url="/docs",                                  # Swagger UI 文档路径
    redoc_url="/redoc",                                # ReDoc 文档路径
    lifespan=lifespan,                                 # 生命周期管理函数
)

# 注册全局异常处理器
# 这将自动捕获应用中的所有异常并返回统一的错误格式
handle_exception(app)

# 配置 CORS（跨域资源共享）中间件
# 允许前端应用从不同域名访问 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOW_ORIGINS,  # 允许的源列表
    allow_credentials=True,                # 允许携带 Cookie
    allow_methods=["*"],                    # 允许的 HTTP 方法
    allow_headers=["*"],                    # 允许的请求头
)

# 注册 API 路由
# 将 api_router 挂载到 API_V1_PREFIX 路径下
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


# 直接运行此文件时启动开发服务器
if __name__ == "__main__":
    import uvicorn

    # 使用 uvicorn 启动 ASGI 服务器
    uvicorn.run(
        "app.main:app",           # 模块路径:应用实例
        host="0.0.0.0",           # 监听所有网络接口
        port=8000,                # 监听端口
        reload=settings.DEBUG,    # 调试模式下启用热重载
    )
