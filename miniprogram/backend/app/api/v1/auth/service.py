"""
认证模块 - 业务逻辑服务

处理微信登录、令牌生成等业务逻辑。
"""

from datetime import datetime, timedelta
from typing import Any, Dict

import httpx
from loguru import logger

from app.api.v1.auth.crud import CRUDUser
from app.api.v1.auth.models import User
from app.api.v1.auth.schemas import (
    AuthSchema,
    AuthResponse,
    TokenInfo,
    UserInfo,
    WechatUserInfoUpdate,
)
from app.core.config import settings
from app.core.exception import CustomException
from app.core.security import create_access_token, create_refresh_token


class WechatAuthService:
    """
    微信认证服务

    处理微信小程序登录相关的业务逻辑。
    """

    def __init__(self, db: Any):
        """
        初始化微信认证服务

        Args:
            db: 数据库会话
        """
        self.db = db
        self.auth = AuthSchema(db=db)
        self.user_crud = CRUDUser(User, self.auth)

    async def wechat_login(self, code: str) -> AuthResponse:
        """
        微信小程序登录

        核心流程：
        1. 使用 code 向微信服务器换取 openid 和 session_key
        2. 根据 openid 查找用户
        3. 如果用户不存在，创建新用户
        4. 如果用户存在，更新最后登录时间
        5. 生成 JWT 令牌并返回

        Args:
            code: 微信小程序临时登录凭证，通过 uni.login() 获取

        Returns:
            AuthResponse: 认证响应，包含令牌和用户信息

        Raises:
            CustomException: 微信配置错误或登录失败时抛出

        示例:
            >>> service = WechatAuthService(db)
            >>> response = await service.wechat_login("061a2b3c4d5e6f")
        """
        try:
            # 检查微信配置
            if not settings.WECHAT_APP_ID or not settings.WECHAT_APP_SECRET:
                logger.error(
                    "微信配置缺失，请检查 .env 文件中的 WECHAT_APP_ID 和 WECHAT_APP_SECRET 配置"
                )
                raise CustomException(msg="微信登录配置错误，请联系管理员")

            # 1. 向微信服务器请求换取 openid 和 session_key
            wx_data = await self._get_wechat_openid(code)

            openid = wx_data.get("openid")
            session_key = wx_data.get("session_key")
            unionid = wx_data.get("unionid")

            if not openid:
                raise CustomException(msg="微信授权失败，无法获取 openid")

            # 2. 查找用户
            user = await self.user_crud.get_by_openid(openid)

            # 3. 如果用户不存在，创建新用户
            if not user:
                user = await self._create_wechat_user(openid, session_key, unionid)
            else:
                # 4. 如果用户存在，更新 session_key 和最后登录时间
                await self._update_wechat_user(user, session_key, unionid)

            # 5. 生成 JWT 令牌
            token_info = self._generate_token(user)

            # 6. 构造用户信息
            user_info = self._build_user_info(user)

            # 业务流程完成，提交事务
            await self.db.commit()

            return AuthResponse(token=token_info, user=user_info)
        except Exception:
            # 发生异常时回滚事务
            await self.db.rollback()
            raise

    async def update_user_profile(
        self,
        user: User,
        profile_update: WechatUserInfoUpdate,
    ) -> User:
        """
        更新用户资料

        用于用户首次授权后更新昵称、头像等信息。

        Args:
            user: 当前用户对象
            profile_update: 用户信息更新数据

        Returns:
            User: 更新后的用户对象

        示例:
            >>> service = WechatAuthService(db)
            >>> updated_user = await service.update_user_profile(
            ...     user,
            ...     WechatUserInfoUpdate(nickname="张三", avatar_url="https://...")
            ... )
        """
        try:
            update_data = profile_update.model_dump(
                exclude_unset=True, exclude_none=True
            )

            if not update_data:
                return user

            updated_user = await self.user_crud.update_user_info(user.id, **update_data)

            # 业务流程完成，提交事务
            await self.db.commit()

            return updated_user
        except Exception:
            # 发生异常时回滚事务
            await self.db.rollback()
            raise

    async def _get_wechat_openid(self, code: str) -> Dict[str, Any]:
        """
        向微信服务器获取 openid 和 session_key

        Args:
            code: 微信小程序临时登录凭证

        Returns:
            Dict: 微信服务器返回的数据，包含 openid、session_key 等

        Raises:
            CustomException: 微信 API 调用失败时抛出

        示例:
            >>> data = await self._get_wechat_openid("061a2b3c4d5e6f")
            >>> print(data["openid"])
            oXYZ123...
        """
        # 构造微信 API 请求 URL
        wechat_api_url = (
            f"https://api.weixin.qq.com/sns/jscode2session?"
            f"appid={settings.WECHAT_APP_ID}&"
            f"secret={settings.WECHAT_APP_SECRET}&"
            f"js_code={code}&"
            f"grant_type=authorization_code"
        )

        try:
            # 向微信服务器发起异步请求
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(wechat_api_url)
                wx_res = response.json()
        except httpx.TimeoutException:
            logger.error("微信服务器请求超时")
            raise CustomException(msg="微信服务器响应超时，请稍后重试")
        except Exception as e:
            logger.error(f"微信服务器请求失败: {str(e)}")
            raise CustomException(msg="微信服务器请求失败")

        # 检查微信 API 响应错误
        if "errcode" in wx_res:
            errcode = wx_res["errcode"]
            errmsg = wx_res.get("errmsg", "未知错误")

            logger.error(f"微信登录失败 [errcode: {errcode}]: {errmsg}")

            # 根据错误码提供具体的错误信息
            error_map = {
                40013: "微信 AppID 无效",
                40125: "微信 AppSecret 无效",
                40029: "授权码(code)无效或已过期",
                45011: "调用太频繁，请稍后再试",
                40002: "grant_type 参数错误",
                40163: "code 已被使用",
            }
            detail = error_map.get(errcode, f"微信登录失败: {errmsg}")

            raise CustomException(msg=detail)

        if "openid" not in wx_res:
            logger.error(f"微信登录失败，响应中缺少 openid: {wx_res}")
            raise CustomException(msg="微信授权失败，无效的授权码")

        return wx_res

    async def _create_wechat_user(
        self,
        openid: str,
        session_key: str | None = None,
        unionid: str | None = None,
    ) -> User:
        """
        创建微信用户

        Args:
            openid: 微信 openid
            session_key: 微信 session_key
            unionid: 微信 unionid

        Returns:
            User: 创建的用户对象
        """
        # 创建用户，昵称使用默认值，头像等由用户授权后更新
        user = await self.user_crud.create_wechat_user(
            openid=openid,
            nickname=None,  # 首次登录可能获取不到昵称
            avatar_url=None,
        )

        # 更新 session_key 和 unionid
        if session_key:
            user.session_key = session_key
        if unionid:
            user.unionid = unionid

        # 仅 flush，不 commit（由外层 wechat_login 控制）
        await self.db.flush()
        await self.db.refresh(user)

        logger.info(f"创建新用户: {user.id}, openid: {openid}")

        return user

    async def _update_wechat_user(
        self,
        user: User,
        session_key: str | None = None,
        unionid: str | None = None,
    ) -> None:
        """
        更新微信用户信息

        Args:
            user: 用户对象
            session_key: 新的 session_key
            unionid: unionid
        """
        # 更新 session_key（每次登录都会变化）
        if session_key:
            user.session_key = session_key

        # 更新 unionid（如果之前没有）
        if unionid and not user.unionid:
            user.unionid = unionid

        # 更新最后登录时间
        user.last_login_at = datetime.now()

        # 仅 flush，不 commit（由外层 wechat_login 控制）
        await self.db.flush()
        await self.db.refresh(user)

        logger.info(f"更新用户信息: {user.id}, last_login: {user.last_login_at}")

    def _generate_token(self, user: User) -> TokenInfo:
        """
        生成 JWT 令牌

        Args:
            user: 用户对象

        Returns:
            TokenInfo: 令牌信息

        示例:
            >>> token_info = self._generate_token(user)
            >>> print(token_info.access_token)
            eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
        """
        # 生成访问令牌
        access_token = create_access_token(
            subject=str(user.id),
            expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        )

        # 生成刷新令牌
        refresh_token = create_refresh_token(subject=str(user.id))

        return TokenInfo(
            token_type="bearer",
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    @staticmethod
    def _build_user_info(user: User) -> UserInfo:
        """
        构造用户信息响应

        Args:
            user: 用户对象

        Returns:
            UserInfo: 用户信息
        """
        return UserInfo(
            id=user.id,
            nickname=user.nickname,
            avatar_url=user.avatar_url,
            phone=user.phone,
            gender=user.gender or 0,
            is_active=user.is_active,
        )


class PhoneAuthService:
    """
    手机号认证服务

    处理手机号验证码登录的业务逻辑。

    注意：验证码发送和验证逻辑需要根据实际短信服务商实现。
    """

    def __init__(self, db: Any):
        """
        初始化手机号认证服务

        Args:
            db: 数据库会话
        """
        self.db = db
        self.auth = AuthSchema(db=db)
        self.user_crud = CRUDUser(User, self.auth)

    async def phone_login(self, phone: str, code: str) -> AuthResponse:
        """
        手机号验证码登录

        Args:
            phone: 手机号
            code: 验证码

        Returns:
            AuthResponse: 认证响应

        注意:
            验证码验证逻辑需要根据实际实现补充。
            这里仅提供框架，实际使用时需要接入短信服务。
        """
        # TODO: 验证验证码是否正确
        # 这里需要实现验证码验证逻辑
        # 验证码可以存储在 Redis 中，设置过期时间

        # 查找用户
        user = await self.user_crud.get_by_phone(phone)

        # 如果用户不存在，创建新用户
        if not user:
            # 这里需要决定是否需要 openid
            # 对于纯手机号用户，可以使用空字符串或生成临时 openid
            user = await self.user_crud.create_wechat_user(
                openid=f"phone_{phone}",  # 临时 openid
                nickname=phone,  # 使用手机号作为昵称
            )
        else:
            # 更新最后登录时间
            await self.user_crud.update_last_login(user.id)

        # 生成令牌
        access_token = create_access_token(
            subject=str(user.id),
            expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        )

        token_info = TokenInfo(
            token_type="bearer",
            access_token=access_token,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

        user_info = UserInfo(
            id=user.id,
            nickname=user.nickname,
            avatar_url=user.avatar_url,
            phone=user.phone,
            gender=user.gender or 0,
            is_active=user.is_active,
        )

        return AuthResponse(token=token_info, user=user_info)
