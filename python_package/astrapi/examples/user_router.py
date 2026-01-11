"""
示例：用户 API 路由

展示如何创建 FastAPI 路由处理 HTTP 请求。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db
from app.core.response import BaseResponse
from app.core.security import (
    create_access_token,
    create_refresh_token,
    get_password_hash,
    verify_password,
)
from examples.user_crud import CRUDUser
from examples.user_model import (
    AuthSchema,
    TokenResponse,
    User,
    UserCreate,
    UserResponse,
    UserUpdate,
)

# 创建路由器
router = APIRouter(prefix="/users", tags=["用户管理"])

# OAuth2 密码流
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/users/login")


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    获取当前登录用户（依赖注入函数）

    从 JWT 令牌中解析用户信息。

    Args:
        token: JWT 令牌
        db: 数据库会话

    Returns:
        User: 当前用户对象

    Raises:
        HTTPException: 令牌无效时抛出 401 错误
    """

    from app.core.security import decode_token

    # 解码令牌
    payload = decode_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭据",
        )

    user_id = int(payload["sub"])

    # 从数据库获取用户
    auth = AuthSchema(db=db)
    crud = CRUDUser(User, auth)
    user = await crud.get(id=user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在",
        )

    return user


# ============================================
# 用户注册和登录
# ============================================

@router.post(
    "/register",
    response_model=BaseResponse[UserResponse],
    summary="用户注册",
    description="创建新用户账号"
)
async def register(
    user_create: UserCreate,
    db: AsyncSession = Depends(get_db),
) -> BaseResponse[UserResponse]:
    """
    用户注册接口

    创建新用户，密码会自动哈希存储。
    """
    # 检查用户名是否已存在
    auth = AuthSchema(db=db)
    crud = CRUDUser(User, auth)

    existing_user = await crud.get_by_username(user_create.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户名已存在",
        )

    # 检查邮箱是否已存在
    existing_user = await crud.get_by_email(user_create.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="邮箱已被注册",
        )

    # 创建用户（密码哈希处理）
    user_data = user_create.model_dump()
    user_data["hashed_password"] = get_password_hash(user_data.pop("password"))

    new_user = await crud.create(user_data)

    return BaseResponse[UserResponse](
        code=status.HTTP_201_CREATED,
        msg="注册成功",
        data=UserResponse.model_validate(new_user),
    )


@router.post(
    "/login",
    response_model=BaseResponse[TokenResponse],
    summary="用户登录",
    description="用户登录获取访问令牌"
)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: AsyncSession = Depends(get_db),
) -> BaseResponse[TokenResponse]:
    """
    用户登录接口

    验证用户名和密码，返回 JWT 令牌。
    """
    # 查找用户
    auth = AuthSchema(db=db)
    crud = CRUDUser(User, auth)

    # 尝试用用户名查找
    user = await crud.get_by_username(form_data.username)

    # 如果找不到，尝试用邮箱查找
    if not user:
        user = await crud.get_by_email(form_data.username)

    # 验证用户和密码
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )

    # 检查用户是否激活
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="账号已被禁用",
        )

    # 生成令牌
    access_token = create_access_token(subject=user.id)
    refresh_token = create_refresh_token(subject=user.id)

    return BaseResponse[TokenResponse](
        msg="登录成功",
        data=TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
        ),
    )


# ============================================
# 用户 CRUD 接口
# ============================================

@router.get(
    "/me",
    response_model=BaseResponse[UserResponse],
    summary="获取当前用户信息",
    description="获取当前登录用户的详细信息"
)
async def get_me(
    current_user: Annotated[User, Depends(get_current_user)],
) -> BaseResponse[UserResponse]:
    """
    获取当前用户信息
    """
    return BaseResponse[UserResponse](data=UserResponse.model_validate(current_user))


@router.get(
    "/{user_id}",
    response_model=BaseResponse[UserResponse],
    summary="获取用户详情",
    description="根据用户 ID 获取用户信息"
)
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: Annotated[User, Depends(get_current_user)],
) -> BaseResponse[UserResponse]:
    """
    获取用户详情
    """
    auth = AuthSchema(db=db, user=current_user)
    crud = CRUDUser(User, auth)

    user = await crud.get(id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在",
        )

    return BaseResponse[UserResponse](data=UserResponse.model_validate(user))


@router.get(
    "/",
    response_model=BaseResponse[dict],
    summary="获取用户列表",
    description="分页获取用户列表，支持搜索和过滤"
)
async def list_users(
    offset: int = 0,
    limit: int = 10,
    keyword: str | None = None,
    is_active: int | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: Annotated[User, Depends(get_current_user)],
) -> BaseResponse[dict]:
    """
    获取用户列表（分页）
    """
    auth = AuthSchema(db=db, user=current_user)
    crud = CRUDUser(User, auth)

    result = await crud.search_users(
        keyword=keyword,
        is_active=is_active,
        offset=offset,
        limit=limit,
    )

    return BaseResponse[dict](data=result)


@router.put(
    "/{user_id}",
    response_model=BaseResponse[UserResponse],
    summary="更新用户信息",
    description="更新指定用户的信息"
)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: Annotated[User, Depends(get_current_user)],
) -> BaseResponse[UserResponse]:
    """
    更新用户信息
    """
    auth = AuthSchema(db=db, user=current_user)
    crud = CRUDUser(User, auth)

    # 权限检查：只能更新自己的信息（除非是超级管理员）
    if user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权修改其他用户信息",
        )

    updated_user = await crud.update(user_id, user_update)

    return BaseResponse[UserResponse](
        msg="更新成功",
        data=UserResponse.model_validate(updated_user),
    )


@router.delete(
    "/{user_id}",
    response_model=BaseResponse[None],
    summary="删除用户",
    description="删除指定用户（仅超级管理员可用）"
)
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: Annotated[User, Depends(get_current_user)],
) -> BaseResponse[None]:
    """
    删除用户
    """
    # 仅超级管理员可以删除用户
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="仅超级管理员可以删除用户",
        )

    auth = AuthSchema(db=db, user=current_user)
    crud = CRUDUser(User, auth)

    await crud.delete([user_id])

    return BaseResponse[None](msg="删除成功")


# ============================================
# 使用示例
# ============================================

if __name__ == "__main__":
    print("""
    用户 API 路由示例

    端点列表:
    - POST   /api/v1/users/register   用户注册
    - POST   /api/v1/users/login      用户登录
    - GET    /api/v1/users/me         获取当前用户信息
    - GET    /api/v1/users/{id}       获取用户详情
    - GET    /api/v1/users/           获取用户列表
    - PUT    /api/v1/users/{id}       更新用户信息
    - DELETE /api/v1/users/{id}       删除用户

    测试请求示例:

    # 注册用户
    curl -X POST http://localhost:8000/api/v1/users/register \\
      -H "Content-Type: application/json" \\
      -d '{"username":"zhangsan","email":"zhangsan@example.com","password":"123456"}'

    # 登录
    curl -X POST http://localhost:8000/api/v1/users/login \\
      -H "Content-Type: application/x-www-form-urlencoded" \\
      -d "username=zhangsan&password=123456"

    # 获取用户列表
    curl -X GET "http://localhost:8000/api/v1/users/?offset=0&limit=10" \\
      -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
    """)
