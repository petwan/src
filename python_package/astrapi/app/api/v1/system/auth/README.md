# 认证模块 (Auth Module)

提供完整的用户认证和授权功能，特别针对微信小程序进行了优化。

## 功能特性

- ✅ 微信小程序一键登录
- ✅ 手机号验证码登录
- ✅ JWT 令牌认证
- ✅ 用户信息管理
- ✅ 可选认证（支持匿名访问）
- ✅ 活跃用户验证

## 目录结构

```
auth/
├── __init__.py        # 模块导出
├── models.py          # 用户数据模型
├── schemas.py         # Pydantic Schema 定义
├── crud.py           # 数据访问层
├── service.py        # 业务逻辑层
├── dependencies.py    # 依赖注入
└── router.py         # API 路由
```

## API 接口

### 1. 微信小程序登录

**接口**: `POST /api/v1/auth/wechat/login`

**请求**:
```json
{
    "code": "061a2b3c4d5e6f"
}
```

**响应**:
```json
{
    "code": 200,
    "msg": "微信登录成功",
    "data": {
        "token": {
            "token_type": "bearer",
            "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            "expires_in": 1800
        },
        "user": {
            "id": 1,
            "nickname": "用户-abc12345",
            "avatar_url": null,
            "phone": null,
            "gender": 0,
            "is_active": 1
        }
    }
}
```

### 2. 更新用户信息

**接口**: `POST /api/v1/auth/wechat/update-profile`

**请求**:
```json
{
    "nickname": "张三",
    "avatar_url": "https://example.com/avatar.jpg",
    "gender": 1,
    "country": "中国",
    "province": "广东",
    "city": "深圳"
}
```

**请求头**:
```
Authorization: Bearer {access_token}
```

### 3. 获取用户信息

**接口**: `GET /api/v1/auth/user/profile`

**请求头**:
```
Authorization: Bearer {access_token}
```

**响应**:
```json
{
    "code": 200,
    "message": "获取成功",
    "result": {
        "id": 1,
        "nickname": "张三",
        "avatar_url": "https://example.com/avatar.jpg",
        "phone": "13800138000",
        "gender": 1,
        "is_active": 1
    }
}
```

### 4. 用户登出

**接口**: `POST /api/v1/auth/logout`

**请求头**:
```
Authorization: Bearer {access_token}
```

**响应**:
```json
{
    "code": 200,
    "message": "登出成功",
    "result": null
}
```

## 微信小程序集成

### 前端代码示例

```javascript
// 1. 微信登录获取 code
uni.login({
    provider: 'weixin',
    success: (res) => {
        // 2. 将 code 发送到后端
        axios.post('/api/v1/auth/wechat/login', {
            code: res.code
        }).then(response => {
            const { token, user } = response.data.result

            // 3. 保存 token
            uni.setStorageSync('token', token.access_token)
            uni.setStorageSync('user', user)

            // 4. 如果昵称为空，提示用户授权
            if (!user.nickname) {
                uni.getUserProfile({
                    desc: '用于完善用户资料',
                    success: (profileRes) => {
                        // 5. 更新用户信息
                        axios.post('/api/v1/auth/wechat/update-profile', {
                            nickname: profileRes.userInfo.nickName,
                            avatar_url: profileRes.userInfo.avatarUrl,
                            gender: profileRes.userInfo.gender
                        }, {
                            headers: {
                                'Authorization': 'Bearer ' + token.access_token
                            }
                        })
                    }
                })
            }
        })
    }
})

// 6. 携带 token 发送请求
axios.get('/api/v1/auth/user/profile', {
    headers: {
        'Authorization': 'Bearer ' + uni.getStorageSync('token')
    }
})
```

### 微信小程序配置

1. 在微信公众平台获取 AppID 和 AppSecret
2. 在 `.env` 文件中配置：
```env
WECHAT_APP_ID=wx1234567890abcdef
WECHAT_APP_SECRET=abc123def456...
```

## 依赖注入使用

### 需要登录的接口

```python
from fastapi import APIRouter, Depends
from app.api.v1.system.auth.dependencies import CurrentUser

router = APIRouter()

@router.get("/protected")
async def protected_endpoint(
    current_user: CurrentUser
):
    return {
        "user_id": current_user.id,
        "nickname": current_user.nickname
    }
```

### 可选登录的接口

```python
from app.api.v1.system.auth.dependencies import CurrentUserOptional

@router.get("/optional")
async def optional_endpoint(
    current_user: CurrentUserOptional
):
    if current_user:
        return {
            "logged_in": True,
            "user_id": current_user.id
        }
    return {
        "logged_in": False,
        "message": "请先登录"
    }
```

### 需要活跃用户的接口

```python
from app.api.v1.system.auth.dependencies import CurrentActiveUser

@router.get("/active")
async def active_endpoint(
    current_user: CurrentActiveUser
):
    return {
        "nickname": current_user.nickname
    }
```

### 使用认证服务

```python
from app.api.v1.system.auth.dependencies import WechatAuth

@router.post("/custom-login")
async def custom_login(
    code: str,
    auth_service: WechatAuth
):
    response = await auth_service.wechat_login(code)
    return response
```

## 数据库模型

### User 表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | Integer | 用户ID（主键）|
| openid | String(100) | 微信 openid（唯一）|
| unionid | String(100) | 微信 unionid |
| session_key | String(100) | 微信 session_key |
| nickname | String(100) | 昵称 |
| avatar_url | String(500) | 头像 URL |
| phone | String(20) | 手机号 |
| gender | Integer | 性别 |
| country | String(50) | 国家 |
| province | String(50) | 省份 |
| city | String(50) | 城市 |
| is_active | Integer | 是否激活 |
| last_login_at | DateTime | 最后登录时间 |
| created_at | DateTime | 创建时间 |
| updated_at | DateTime | 更新时间 |

## 错误码说明

| HTTP 状态码 | 说明 |
|-------------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 401 | 未授权（token 无效或过期）|
| 403 | 禁止访问（账号被禁用）|
| 429 | 请求过于频繁 |
| 500 | 服务器内部错误 |

### 微信登录错误码

| errcode | 说明 |
|---------|------|
| 40013 | 微信 AppID 无效 |
| 40125 | 微信 AppSecret 无效 |
| 40029 | 授权码(code)无效或已过期 |
| 45011 | 调用太频繁，请稍后再试 |
| 40163 | code 已被使用 |

## 配置项

在 `.env` 文件中添加：

```env
# 微信小程序配置
WECHAT_APP_ID=wx1234567890abcdef
WECHAT_APP_SECRET=abc123def456...
```

## 安全建议

1. **JWT 密钥**: 使用强密钥，建议使用 `openssl rand -hex 32` 生成
2. **HTTPS**: 生产环境必须使用 HTTPS
3. **令牌有效期**: 根据业务需求调整 `ACCESS_TOKEN_EXPIRE_MINUTES`
4. **Token 黑名单**: 可选实现 token 黑名单功能，实现真正的登出
5. **敏感数据**: 不要在 token 中存储敏感信息

## 扩展功能

### 实现手机号验证码登录

1. 在 `service.py` 中实现 `PhoneAuthService` 的验证逻辑
2. 接入短信服务商（如阿里云、腾讯云）
3. 使用 Redis 存储验证码，设置过期时间

### 实现 Token 黑名单

```python
from app.core.redis import redis_client

async def logout(token: str):
    # 将 token 加入黑名单
    await redis_client.setex(
        f"blacklist:{token}",
        expires_in * 60,
        "1"
    )
```

## 常见问题

### Q: 首次登录昵称为空怎么办？
A: 首次登录时，微信不会直接返回用户昵称和头像，需要用户授权。前端应调用 `uni.getUserProfile()` 获取用户信息后，调用 `/wechat/update-profile` 接口更新。

### Q: 如何验证 token 是否有效？
A: 在需要认证的接口中使用 `CurrentUser` 依赖注入，它会自动验证 token 并返回用户对象。

### Q: 如何实现记住登录状态？
A: 前端将 access_token 存储在 localStorage 中，每次请求时在 Authorization header 中携带。

### Q: 微信登录失败怎么办？
A: 检查以下几点：
1. 微信 AppID 和 AppSecret 是否配置正确
2. code 是否有效（5 分钟内，只能使用一次）
3. 网络是否正常，能否访问微信 API

## 相关文档

- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- [微信小程序登录文档](https://developers.weixin.qq.com/miniprogram/dev/framework/open-ability/login.html)
- [JWT 官方文档](https://jwt.io/)
