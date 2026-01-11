"""API routes"""

from fastapi import APIRouter

# 创建主路由器
api_router = APIRouter()


@api_router.get("/healthcheck", include_in_schema=False)
def healthcheck():
    """Simple Healthcheck endpoint"""
    return {"status": "ok"}


# 导入并注册子路由
from app.api.v1.system.auth.router import router as auth_router

# 注册认证模块路由
api_router.include_router(auth_router)
