"""
通用异步 CRUD 基类，配合 Base 使用

提供数据库的增删改查操作，支持分页、排序、条件查询等功能。

使用示例:
    from pydantic import BaseModel
    from app.database.model import Base
    from app.database.crud import CRUDBase

    class User(Base):
        id: int
        name: str

    class UserCreate(BaseModel):
        name: str

    class UserUpdate(BaseModel):
        name: str | None = None

    class CRUDUser(CRUDBase[User, UserCreate, UserUpdate]):
        pass

    # 使用 CRUD
    crud = CRUDUser(User, auth)
    user = await crud.get(id=1)
    await crud.create(UserCreate(name="张三"))
"""

import builtins
from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from pydantic import BaseModel
from sqlalchemy import Select, asc, delete, desc, func, select, update
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.engine import Result
from sqlalchemy.orm import selectinload
from sqlalchemy.sql.elements import ColumnElement

from app.core.exception import CustomException

from .model import Base

# 定义泛型类型变量
ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)
OutSchemaType = TypeVar("OutSchemaType", bound=BaseModel)


class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    基础数据层 CRUD 操作类

    提供通用的增删改查功能，所有业务 CRUD 类都应继承此类。

    类型参数:
        ModelType: ORM 模型类型
        CreateSchemaType: 创建 Schema 类型
        UpdateSchemaType: 更新 Schema 类型

    主要方法:
        - get: 获取单个对象
        - list: 获取列表
        - page: 分页查询
        - create: 创建对象
        - update: 更新对象
        - delete: 删除对象
    """

    def __init__(self, model: type[ModelType], auth: Any) -> None:
        """
        初始化 CRUD 实例

        Args:
            model: ORM 模型类
            auth: 认证信息对象，包含数据库会话和用户信息
        """
        self.model = model
        self.auth = auth

    async def get(
        self, preload: list[str | Any] | None = None, **kwargs
    ) -> ModelType | None:
        """
        获取单个对象

        根据查询条件获取单个对象，支持预加载关联对象。

        Args:
            preload: 预加载的关联字段列表
            **kwargs: 查询条件，如 id=1, name="张三"

        Returns:
            ModelType | None: 找到的对象，未找到返回 None

        示例:
            >>> user = await crud.get(id=1)
            >>> user = await crud.get(username="zhangsan")
            >>> user = await crud.get(id=1, preload=["roles"])
        """
        try:
            # 构建查询条件
            conditions = await self.__build_conditions(**kwargs)
            sql = select(self.model).where(*conditions)

            # 添加预加载选项
            for opt in self.__loader_options(preload):
                sql = sql.options(opt)

            # 应用权限过滤
            sql = await self.__filter_permissions(sql)

            # 执行查询
            result: Result = await self.auth.db.execute(sql)
            return result.scalars().first()
        except Exception as e:
            raise CustomException(msg=f"获取查询失败: {str(e)}")

    async def list(
        self,
        search: dict | None = None,
        order_by: list[dict[str, str]] | None = None,
        preload: list[str | Any] | None = None,
    ) -> Sequence[ModelType]:
        """
        获取对象列表

        支持条件查询、排序、预加载关联对象。

        Args:
            search: 查询条件字典，如 {"status": 1, ("age", "gt", 18)}
            order_by: 排序规则列表，如 [{"id": "desc"}, {"name": "asc"}]
            preload: 预加载的关联字段列表

        Returns:
            Sequence[ModelType]: 对象列表

        示例:
            >>> users = await crud.list()
            >>> users = await crud.list(search={"status": 1})
            >>> users = await crud.list(order_by=[{"id": "desc"}])
        """
        try:
            conditions = await self.__build_conditions(**search) if search else []
            order = order_by or [{"id": "asc"}]
            sql = select(self.model).where(*conditions).order_by(*self.__order_by(order))

            # 添加预加载选项
            for opt in self.__loader_options(preload):
                sql = sql.options(opt)

            # 应用权限过滤
            sql = await self.__filter_permissions(sql)

            result: Result = await self.auth.db.execute(sql)
            return result.scalars().all()
        except Exception as e:
            raise CustomException(msg=f"列表查询失败: {str(e)}")

    async def tree_list(
        self,
        search: dict | None = None,
        order_by: builtins.list[dict[str, str]] | None = None,
        children_attr: str = "children",
        preload: builtins.list[str | Any] | None = None,
    ) -> Sequence[ModelType]:
        """
        获取树形结构列表

        用于需要父子关系的数据结构，自动预加载子节点。

        Args:
            search: 查询条件字典
            order_by: 排序规则列表
            children_attr: 子节点关联属性名，默认 "children"
            preload: 额外的预加载字段列表

        Returns:
            Sequence[ModelType]: 对象列表

        示例:
            >>> categories = await crud.tree_list()
        """
        try:
            conditions = await self.__build_conditions(**search) if search else []
            order = order_by or [{"id": "asc"}]
            sql = select(self.model).where(*conditions).order_by(*self.__order_by(order))

            # 自动预加载子节点
            final_preload = preload
            if preload is None and children_attr and hasattr(self.model, children_attr):
                model_defaults = getattr(self.model, "__loader_options__", [])
                final_preload = list(model_defaults) + [children_attr]

            # 添加预加载选项
            for opt in self.__loader_options(final_preload):
                sql = sql.options(opt)

            # 应用权限过滤
            sql = await self.__filter_permissions(sql)

            result: Result = await self.auth.db.execute(sql)
            return result.scalars().all()
        except Exception as e:
            raise CustomException(msg=f"树形列表查询失败: {str(e)}")

    async def page(
        self,
        offset: int,
        limit: int,
        order_by: builtins.list[dict[str, str]],
        search: dict,
        out_schema: type[OutSchemaType],
        preload: builtins.list[str | Any] | None = None,
    ) -> dict[str, Any]:
        """
        分页查询

        支持条件查询、排序、预加载，自动计算总数和分页信息。

        Args:
            offset: 偏移量（起始位置）
            limit: 每页数量
            order_by: 排序规则列表
            search: 查询条件字典
            out_schema: 输出 Schema 类型
            preload: 预加载的关联字段列表

        Returns:
            dict: 分页结果，包含:
                - page_no: 当前页码
                - page_size: 每页数量
                - total: 总记录数
                - has_next: 是否有下一页
                - items: 当前页数据列表

        示例:
            >>> result = await crud.page(
            ...     offset=0, limit=10,
            ...     order_by=[{"id": "desc"}],
            ...     search={"status": 1},
            ...     out_schema=UserOutSchema
            ... )
            >>> print(result["total"])
            100
        """
        try:
            conditions = await self.__build_conditions(**search) if search else []
            order = order_by or [{"id": "asc"}]
            sql = select(self.model).where(*conditions).order_by(*self.__order_by(order))

            # 添加预加载选项
            for opt in self.__loader_options(preload):
                sql = sql.options(opt)

            # 应用权限过滤
            sql = await self.__filter_permissions(sql)

            # 构建计数查询
            mapper = sa_inspect(self.model)
            pk_cols = list(getattr(mapper, "primary_key", []))
            if pk_cols:
                count_sql = select(func.count(pk_cols[0])).select_from(self.model)
            else:
                count_sql = select(func.count()).select_from(self.model)
            if conditions:
                count_sql = count_sql.where(*conditions)
            count_sql = await self.__filter_permissions(count_sql)

            # 执行计数查询
            total_result = await self.auth.db.execute(count_sql)
            total = total_result.scalar() or 0

            # 执行分页查询
            result: Result = await self.auth.db.execute(sql.offset(offset).limit(limit))
            objs = result.scalars().all()

            # 返回分页结果
            return {
                "page_no": offset // limit + 1 if limit else 1,
                "page_size": limit if limit else 10,
                "total": total,
                "has_next": offset + limit < total,
                "items": [out_schema.model_validate(obj).model_dump() for obj in objs],
            }
        except Exception as e:
            raise CustomException(msg=f"分页查询失败: {str(e)}")

    async def create(self, data: CreateSchemaType | dict) -> ModelType:
        """
        创建新对象

        创建新的数据库记录，自动记录创建者和更新者信息。

        Args:
            data: 创建数据，可以是 Schema 或字典

        Returns:
            ModelType: 创建的对象

        示例:
            >>> user = await crud.create(UserCreate(name="张三"))
        """
        try:
            obj_dict = data if isinstance(data, dict) else data.model_dump()
            obj = self.model(**obj_dict)

            # 自动设置创建者和更新者
            if self.auth.user:
                if hasattr(obj, "created_id"):
                    obj.created_id = self.auth.user.id
                if hasattr(obj, "updated_id"):
                    obj.updated_id = self.auth.user.id

            self.auth.db.add(obj)
            await self.auth.db.flush()
            await self.auth.db.refresh(obj)
            # 注意：事务提交由调用方控制，CRUD 层不负责提交
            # 这样可以支持多个操作的原子性事务
            return obj
        except Exception as e:
            raise CustomException(msg=f"创建失败: {str(e)}")

    async def update(self, id: int, data: UpdateSchemaType | dict) -> ModelType:
        """
        更新对象

        更新指定 ID 的对象，自动记录更新者信息。

        Args:
            id: 对象 ID
            data: 更新数据，可以是 Schema 或字典

        Returns:
            ModelType: 更新后的对象

        示例:
            >>> user = await crud.update(1, UserUpdate(name="李四"))
        """
        try:
            obj_dict = (
                data
                if isinstance(data, dict)
                else data.model_dump(exclude_unset=True, exclude={"id"})
            )
            obj = await self.get(id=id)
            if not obj:
                raise CustomException(msg="更新对象不存在")

            # 自动设置更新者
            if self.auth.user and hasattr(obj, "updated_id"):
                obj.updated_id = self.auth.user.id

            # 更新字段
            for key, value in obj_dict.items():
                if hasattr(obj, key):
                    setattr(obj, key, value)

            await self.auth.db.flush()
            await self.auth.db.refresh(obj)

            # 注意：事务提交由调用方控制，CRUD 层不负责提交
            # 这样可以支持多个操作的原子性事务

            return obj
        except Exception as e:
            raise CustomException(msg=f"更新失败: {str(e)}")

    async def delete(self, ids: builtins.list[int]) -> None:
        """
        批量删除对象

        根据主键 ID 删除多个对象。

        Args:
            ids: 要删除的对象 ID 列表

        示例:
            >>> await crud.delete([1, 2, 3])
        """
        try:
            mapper = sa_inspect(self.model)
            pk_cols = list(getattr(mapper, "primary_key", []))
            if not pk_cols:
                raise CustomException(msg="模型缺少主键，无法删除")
            if len(pk_cols) > 1:
                raise CustomException(msg="暂不支持复合主键的批量删除")

            sql = delete(self.model).where(pk_cols[0].in_(ids))
            await self.auth.db.execute(sql)
            await self.auth.db.flush()
            # 注意：事务提交由调用方控制，CRUD 层不负责提交
        except Exception as e:
            raise CustomException(msg=f"删除失败: {str(e)}")

    async def clear(self) -> None:
        """
        清空所有数据

        删除表中的所有记录，谨慎使用。

        示例:
            >>> await crud.clear()
        """
        try:
            sql = delete(self.model)
            await self.auth.db.execute(sql)
            await self.auth.db.flush()
        except Exception as e:
            raise CustomException(msg=f"清空失败: {str(e)}")

    async def set(self, ids: builtins.list[int], **kwargs) -> None:
        """
        批量更新对象

        更新指定 ID 列表的多个对象的特定字段。

        Args:
            ids: 要更新的对象 ID 列表
            **kwargs: 要更新的字段和值

        示例:
            >>> await crud.set([1, 2, 3], status=2)
        """
        try:
            mapper = sa_inspect(self.model)
            pk_cols = list(getattr(mapper, "primary_key", []))
            if not pk_cols:
                raise CustomException(msg="模型缺少主键，无法更新")
            if len(pk_cols) > 1:
                raise CustomException(msg="暂不支持复合主键的批量更新")

            sql = update(self.model).where(pk_cols[0].in_(ids)).values(**kwargs)
            await self.auth.db.execute(sql)
            await self.auth.db.flush()
            # 注意：事务提交由调用方控制，CRUD 层不负责提交
        except CustomException:
            raise
        except Exception as e:
            raise CustomException(msg=f"批量更新失败: {str(e)}")

    async def __filter_permissions(self, sql: Select) -> Select:
        """
        权限过滤（内部方法）

        根据当前用户的权限过滤查询结果。

        Args:
            sql: 查询对象

        Returns:
            Select: 过滤后的查询对象
        """
        # 简化权限过滤，后续可扩展
        return sql

    async def __build_conditions(self, **kwargs) -> builtins.list[ColumnElement]:
        """
        构建查询条件（内部方法）

        支持多种条件操作符:
            - ("like", "keyword"): 模糊查询
            - ("in", [1, 2, 3]): IN 查询
            - ("between", [1, 10]): 范围查询
            - ("gt", 18): 大于
            - ("lt", 60): 小于
            - ("ge", 18): 大于等于
            - ("le", 60): 小于等于
            - ("ne", 0): 不等于

        Args:
            **kwargs: 查询条件

        Returns:
            list[ColumnElement]: 条件列表

        示例:
            >>> conditions = await crud._CRUDBase__build_conditions(
            ...     name=("like", "张"),
            ...     age=("gt", 18)
            ... )
        """
        conditions = []
        for key, value in kwargs.items():
            if value is None or value == "":
                continue

            attr = getattr(self.model, key)
            if isinstance(value, tuple):
                seq, val = value
                if seq == "None":
                    conditions.append(attr.is_(None))
                elif seq == "not None":
                    conditions.append(attr.isnot(None))
                elif seq == "date" and val:
                    conditions.append(func.date_format(attr, "%Y-%m-%d") == val)
                elif seq == "month" and val:
                    conditions.append(func.date_format(attr, "%Y-%m") == val)
                elif seq == "like" and val:
                    conditions.append(attr.like(f"%{val}%"))
                elif seq == "in" and val:
                    conditions.append(attr.in_(val))
                elif seq == "between" and isinstance(val, (list, tuple)) and len(val) == 2:
                    conditions.append(attr.between(val[0], val[1]))
                elif seq in ("!=", "ne") and val:
                    conditions.append(attr != val)
                elif seq in (">", "gt") and val:
                    conditions.append(attr > val)
                elif seq in (">=", "ge") and val:
                    conditions.append(attr >= val)
                elif seq in ("<", "lt") and val:
                    conditions.append(attr < val)
                elif seq in ("<=", "le") and val:
                    conditions.append(attr <= val)
                elif seq in ("==", "eq") and val:
                    conditions.append(attr == val)
            else:
                conditions.append(attr == value)
        return conditions

    def __order_by(self, order_by: builtins.list[dict[str, str]]) -> builtins.list[ColumnElement]:
        """
        构建排序条件（内部方法）

        Args:
            order_by: 排序规则列表，如 [{"id": "desc"}, {"name": "asc"}]

        Returns:
            list[ColumnElement]: 排序条件列表

        示例:
            >>> order_by = [{"id": "desc"}, {"name": "asc"}]
            >>> conditions = crud._CRUDBase__order_by(order_by)
        """
        columns = []
        for order in order_by:
            for field, direction in order.items():
                column = getattr(self.model, field)
                columns.append(desc(column) if direction.lower() == "desc" else asc(column))
        return columns

    def __loader_options(self, preload: builtins.list[str | Any] | None = None) -> builtins.list[Any]:
        """
        构建预加载选项（内部方法）

        支持预加载关联对象，避免 N+1 查询问题。

        Args:
            preload: 预加载字段列表，空列表表示禁用预加载

        Returns:
            list[Any]: 预加载选项列表

        示例:
            >>> options = crud._CRUDBase__loader_options(["roles", "permissions"])
        """
        options = []
        model_loader_options = getattr(self.model, "__loader_options__", [])
        all_preloads = set(model_loader_options)

        if preload is not None:
            if preload == []:
                all_preloads = set()
            else:
                for opt in preload:
                    if isinstance(opt, str):
                        all_preloads.add(opt)

        for opt in all_preloads:
            if isinstance(opt, str):
                if hasattr(self.model, opt):
                    options.append(selectinload(getattr(self.model, opt)))
            else:
                options.append(opt)

        return options
