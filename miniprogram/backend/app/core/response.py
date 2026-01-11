"""
API 响应模型模块

定义统一的 API 响应结构，确保所有接口返回数据格式一致。

使用示例:
    from app.core.response import BaseResponse

    # 成功响应
    response = BaseResponse[int](code=200, msg="操作成功", data=123)

    # 失败响应
    response = BaseResponse[None](code=400, msg="参数错误", data=None)

    # 在路由中使用
    @router.get("/users/{id}")
    async def get_user(id: int) -> BaseResponse[UserSchema]:
        user = await get_user_by_id(id)
        return BaseResponse[UserSchema](data=user)
"""

from fastapi import status
from pydantic import BaseModel, ConfigDict, Field
from typing import Generic, TypeVar

T = TypeVar("T")  # 泛型类型变量，用于表示任意数据类型


class BaseResponse(BaseModel, Generic[T]):
    """
    标准 API 响应模型

    所有 API 接口的返回值都应使用此结构，确保格式统一。

    通用响应结构:
        {
            "code": 200,           # HTTP 状态码
            "msg": "Success",      # 提示信息
            "data": { ... }        # 业务数据
        }

    类型参数:
        T: 业务数据的类型，可以为任意类型（UserSchema, List[UserSchema], dict 等）

    Attributes:
        code: HTTP 状态码，默认 200
        msg: 提示信息，默认 "Success"
        data: 业务数据，默认 None

    示例（成功）:
        >>> response = BaseResponse[dict](
        ...     code=200,
        ...     msg="操作成功",
        ...     data={"id": 1, "name": "张三"}
        ... )
        >>> print(response.model_dump())
        {'code': 200, 'msg': '操作成功', 'data': {'id': 1, 'name': '张三'}}

    示例（错误）:
        >>> response = BaseResponse[None](
        ...     code=400,
        ...     msg="用户名或密码错误",
        ...     data=None
        ... )
    """

    # HTTP 状态码，表示请求的处理结果
    code: int = Field(
        default=status.HTTP_200_OK,
        description="HTTP 状态码，如 200 成功、400 请求错误、404 未找到等"
    )

    # 提示信息，用于向用户展示操作结果
    msg: str = Field(
        default="Success",
        description="人类可读的提示信息，描述操作结果"
    )

    # 业务数据，可以是任意类型（对象、列表、字典等）
    data: T | None = Field(
        default=None,
        description="业务数据，成功时返回数据，失败时返回 null"
    )

    # Pydantic 模型配置
    model_config = ConfigDict(
        from_attributes=True,      # 允许从 ORM 对象创建
        populate_by_name=True,    # 允许使用字段别名
        json_schema_extra={
            "example": {
                "code": 200,
                "msg": "操作成功",
                "data": {"id": 1, "name": "示例用户"},
            }
        },
    )
