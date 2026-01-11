"""
异常处理模块

定义自定义异常类和异常处理器，统一处理应用中的错误情况。

使用示例:
    from app.core.exception import (
        CustomException,
        NotFoundException,
        UnauthorizedException,
        ForbiddenException
    )

    # 抛出自定义异常
    raise NotFoundException("用户不存在")
    raise UnauthorizedException("未登录")
    raise ForbiddenException("权限不足")
"""

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.response import BaseResponse


class CustomException(Exception):
    """
    自定义异常基类

    所有业务异常都应继承此基类，便于统一处理。

    Attributes:
        msg: 错误提示信息
        code: HTTP 状态码，默认 400
        data: 额外的错误数据，可选

    示例:
        >>> raise CustomException(msg="操作失败", code=400)
        Traceback (most recent call last):
          ...
        app.core.exception.CustomException: 操作失败
    """

    def __init__(self, msg: str, code: int = status.HTTP_400_BAD_REQUEST, data: Any = None):
        """
        初始化自定义异常

        Args:
            msg: 错误提示信息
            code: HTTP 状态码，默认 400（BAD_REQUEST）
            data: 额外的错误数据，可以是字典、列表等
        """
        self.msg = msg
        self.code = code
        self.data = data
        super().__init__(msg)


class NotFoundException(CustomException):
    """
    资源未找到异常 (404)

    当请求的资源不存在时抛出此异常。

    示例:
        >>> raise NotFoundException("用户 ID 123 不存在")
    """

    def __init__(self, msg: str = "资源未找到"):
        """
        初始化资源未找到异常

        Args:
            msg: 错误提示信息，默认 "资源未找到"
        """
        super().__init__(msg=msg, code=status.HTTP_404_NOT_FOUND)


class UnauthorizedException(CustomException):
    """
    未授权异常 (401)

    当用户未认证或认证信息无效时抛出此异常。

    示例:
        >>> raise UnauthorizedException("Token 已过期")
    """

    def __init__(self, msg: str = "未授权访问"):
        """
        初始化未授权异常

        Args:
            msg: 错误提示信息，默认 "未授权访问"
        """
        super().__init__(msg=msg, code=status.HTTP_401_UNAUTHORIZED)


class ForbiddenException(CustomException):
    """
    禁止访问异常 (403)

    当用户已认证但无权限访问资源时抛出此异常。

    示例:
        >>> raise ForbiddenException("您没有权限访问此资源")
    """

    def __init__(self, msg: str = "禁止访问"):
        """
        初始化禁止访问异常

        Args:
            msg: 错误提示信息，默认 "禁止访问"
        """
        super().__init__(msg=msg, code=status.HTTP_403_FORBIDDEN)


class ValidationException(CustomException):
    """
    数据验证异常 (422)

    当请求数据验证失败时抛出此异常。

    示例:
        >>> raise ValidationException("邮箱格式不正确")
    """

    def __init__(self, msg: str = "数据验证失败"):
        """
        初始化数据验证异常

        Args:
            msg: 错误提示信息，默认 "数据验证失败"
        """
        super().__init__(msg=msg, code=status.HTTP_422_UNPROCESSABLE_ENTITY)


def handle_exception(app: FastAPI) -> None:
    """
    注册全局异常处理器

    将自定义异常处理器绑定到 FastAPI 应用实例，统一处理所有异常。

    此函数会在 main.py 中被调用，应用启动时完成注册。

    Args:
        app: FastAPI 应用实例

    示例:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> handle_exception(app)
    """

    @app.exception_handler(CustomException)
    async def custom_exception_handler(request: Request, exc: CustomException) -> JSONResponse:
        """
        处理自定义异常

        当抛出继承自 CustomException 的异常时，自动调用此处理器。
        返回符合 BaseResponse 格式的 JSON 响应。

        Args:
            request: 请求对象
            exc: 自定义异常实例

        Returns:
            JSONResponse: 包含错误信息的 JSON 响应

        响应格式:
            {
                "code": 400,
                "msg": "错误信息",
                "data": null
            }
        """
        return JSONResponse(
            status_code=exc.code,
            content=BaseResponse[Any](code=exc.code, msg=exc.msg, data=exc.data).model_dump(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        处理通用异常（未捕获的异常）

        处理所有未被其他处理器捕获的异常，用于捕获系统级错误。

        生产环境返回通用错误信息，调试环境返回详细错误堆栈。

        Args:
            request: 请求对象
            exc: 异常实例

        Returns:
            JSONResponse: 包含错误信息的 JSON 响应（500 状态码）

        响应格式:
            {
                "code": 500,
                "msg": "服务器内部错误",
                "data": null
            }
        """
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=BaseResponse[Any](
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                # 生产环境隐藏详细错误信息，调试环境显示完整错误
                msg="服务器内部错误" if not app.debug else str(exc),
                data=None,
            ).model_dump(),
        )
