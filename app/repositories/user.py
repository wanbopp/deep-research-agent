"""用户数据访问边界."""

from uuid import UUID

from sqlmodel import select

from app.models import User
from app.repositories.base import RepositoryBase


class UserRepository(RepositoryBase):
    """封装 User 的最小异步写入与查询操作."""

    async def create(self, *, email: str, password_hash: str) -> User:
        """在当前事务中保存邮箱和已生成的 credential，但不提交事务.

        Repository 不导入 ``SecretStr``，也不调用密码算法。这个签名故意只接受
        ``password_hash``，让明文密码无法沿正常调用链进入 ORM 或数据库层。
        """
        return await self._persist(
            User(email=email, password_hash=password_hash),
            resource="user",
        )

    async def get_by_id(self, user_id: UUID) -> User | None:
        """按主键查询用户；不存在时返回 ``None``."""
        statement = select(User).where(User.id == user_id)
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> User | None:
        """按规范化后的邮箱查询用户；不存在时返回 ``None``."""
        statement = select(User).where(User.email == email)
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def list_all(self) -> tuple[User, ...]:
        """稳定地列出全部用户；调用方必须在授权后的管理用例中使用."""
        statement = select(User).order_by("created_at", "id")
        result = await self._session.execute(statement)
        return tuple(result.scalars().all())
