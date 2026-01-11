# Astrapi 示例文档

本文档展示如何使用 Astrapi 框架构建完整的功能模块。

## 快速开始

### 1. 创建数据库迁移

```bash
# 创建迁移文件
alembic revision --autogenerate -m "添加用户表"

# 执行迁移
alembic upgrade head
```

### 2. 注册路由

在 `app/api/v1/__init__.py` 中添加:

```python
from examples.user_router import router as user_router

api_router.include_router(user_router)
```

### 3. 测试 API

```bash
# 启动服务
python -m app.main

# 访问文档
http://localhost:8000/docs
```

## 示例代码说明

### 1. 数据模型 (user_model.py)

定义用户表结构和 Pydantic Schema:

```python
class User(Base):
    """用户模型"""
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True)
    email = Column(String(100), unique=True)
    # ...

class UserCreate(BaseModel):
    """创建用户 Schema"""
    username: str
    email: EmailStr
    password: str
```

### 2. CRUD 操作 (user_crud.py)

继承 CRUDBase 实现数据访问:

```python
class CRUDUser(CRUDBase[User, UserCreate, UserUpdate]):
    """用户 CRUD 操作"""

    async def get_by_username(self, username: str) -> User | None:
        """根据用户名获取用户"""
        return await self.get(username=username)
```

### 3. API 路由 (user_router.py)

创建 HTTP 接口:

```python
@router.post("/register")
async def register(user_create: UserCreate, db: AsyncSession = Depends(get_db)):
    """用户注册"""
    # 创建用户逻辑
    return BaseResponse[UserResponse](data=user)
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/users/register` | 用户注册 |
| POST | `/users/login` | 用户登录 |
| GET | `/users/me` | 获取当前用户信息 |
| GET | `/users/{id}` | 获取用户详情 |
| GET | `/users/` | 获取用户列表（分页） |
| PUT | `/users/{id}` | 更新用户信息 |
| DELETE | `/users/{id}` | 删除用户 |

## 测试示例

### 注册用户

```bash
curl -X POST http://localhost:8000/api/v1/users/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "zhangsan",
    "email": "zhangsan@example.com",
    "password": "123456",
    "full_name": "张三"
  }'
```

### 登录

```bash
curl -X POST http://localhost:8000/api/v1/users/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=zhangsan&password=123456"
```

### 获取用户列表

```bash
curl -X GET "http://localhost:8000/api/v1/users/?offset=0&limit=10" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## 扩展功能

### 添加权限过滤

在 `app/core/permission.py` 中扩展权限逻辑:

```python
async def filter_query(self, sql: Select) -> Select:
    if not self.auth.user.is_superuser:
        sql = sql.where(self.model.created_id == self.auth.user.id)
    return sql
```

### 添加搜索过滤

在 CRUD 类中添加自定义搜索方法:

```python
async def search_users(self, keyword: str):
    search = {"username": ("like", keyword)}
    return await self.list(search=search)
```

## 常见问题

### Q: 如何处理密码加密？
A: 使用 `get_password_hash()` 和 `verify_password()` 函数。

### Q: 如何实现分页查询？
A: 使用 `crud.page(offset, limit, order_by, search, out_schema)` 方法。

### Q: 如何预加载关联数据？
A: 使用 `preload=["roles", "permissions"]` 参数。

## 更多示例

更多示例和最佳实践请参考项目文档。
