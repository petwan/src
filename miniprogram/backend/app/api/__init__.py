"""API routes"""

from fastapi import APIRouter

# 创建主路由器
api_router = APIRouter()


@api_router.get("/healthcheck", include_in_schema=False)
def healthcheck():
    """Simple Healthcheck endpoint"""
    return {"status": "ok"}


# 导入并注册子路由
from app.api.v1.auth.router import router as system_auth_router

# 注册系统认证模块路由
api_router.include_router(system_auth_router)
