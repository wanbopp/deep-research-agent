"""跨 Repository 的业务事务协调服务."""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories import (
    ChatSessionRepository,
    RepositoryConflictError,
    UserRepository,
)


class PersistenceServiceError(RuntimeError):
    """持久化 application service 可预期业务错误的基类."""


class UserAlreadyExistsError(PersistenceServiceError):
    """注册邮箱已经被其他用户占用."""

    def __init__(self) -> None:
        """使用固定错误文本，避免异常或日志意外包含用户邮箱."""
        super().__init__("User email already exists")


class UserNotFoundError(PersistenceServiceError):
    """业务操作引用了不存在的用户."""

    def __init__(self) -> None:
        """使用固定错误文本；HTTP 映射应由未来的 route 层完成."""
        super().__init__("User was not found")


@dataclass(frozen=True, slots=True)
class CreatedUserWorkspace:
    """创建用户和首个会话后的稳定 application service 输出.

    service 不把可变、绑定 Session 的 ORM 对象传入 Agent state，而是返回可以
    安全跨层传递的业务标识。后续 route 可以再把它转换为 Pydantic 响应模型。
    """

    user_id: UUID
    chat_session_id: UUID


class UserWorkspaceService:
    """协调用户与聊天会话 Repository 的事务型业务操作.

    Route / Agent Node
        -> UserWorkspaceService
        -> async with session.begin()
        -> 多个 Repository 共用同一个 AsyncSession
        -> Repository 执行 add + flush + refresh
        -> 全部成功：begin() 自动 commit
        -> 任意失败：begin() 自动 rollback
    """

    def __init__(self, session: AsyncSession) -> None:
        """让多个 Repository 共享同一个短生命周期 Session."""
        self._session = session
        self._users = UserRepository(session)
        self._chat_sessions = ChatSessionRepository(session)

    async def create_user_workspace(
        self,
        *,
        email: str,
        password_hash: str,
        title: str = "New chat",
    ) -> CreatedUserWorkspace:
        """原子地创建用户和首个聊天会话.

        ``session.begin()`` 是事务所有者：两个 INSERT 都成功时，退出上下文会
        commit；任意一步抛出异常时，退出上下文会 rollback。Repository 只 flush，
        因而不会出现“用户已经提交、会话却创建失败”的半完成状态。

        ``password_hash`` 必须由上层 PasswordHasher 生成。本 service 只协调事务，
        不接收明文密码，也不重复决定 Argon2 参数。Checkpoint 9D 的 AuthService
        会先执行 hash，再调用这里或 UserRepository。

        Raises:
            ValueError: 邮箱、credential 或标题为空，或者字段超过数据库长度。
            UserAlreadyExistsError: 邮箱唯一约束冲突。
            RepositoryConflictError: 首个会话违反数据库约束。
        """
        normalized_email = email.strip().casefold()
        normalized_title = title.strip()
        if not normalized_email:
            raise ValueError("email must not be empty")
        if not password_hash:
            raise ValueError("password_hash must not be empty")
        if len(password_hash) > 255:
            raise ValueError("password_hash must not exceed 255 characters")
        if not normalized_title:
            raise ValueError("title must not be empty")
        if len(normalized_title) > 200:
            raise ValueError("title must not exceed 200 characters")

        async with self._session.begin():
            try:
                user = await self._users.create(
                    email=normalized_email,
                    password_hash=password_hash,
                )
            except RepositoryConflictError as exc:
                # 这里只转换用户 INSERT 的约束冲突。会话 INSERT 若失败则保持原始
                # RepositoryConflictError，避免把外键错误误报成“邮箱重复”。
                raise UserAlreadyExistsError from exc

            chat_session = await self._chat_sessions.create(
                user_id=user.id,
                title=normalized_title,
            )

        # 到达这里说明 begin() 已成功 commit；只返回跨层稳定的 UUID。
        return CreatedUserWorkspace(
            user_id=user.id,
            chat_session_id=chat_session.id,
        )

    async def create_chat_session(
        self,
        *,
        user_id: UUID,
        title: str = "New chat",
    ) -> UUID:
        """为已存在的用户创建会话，并把“未找到”转换为业务错误."""
        normalized_title = title.strip()
        if not normalized_title:
            raise ValueError("title must not be empty")
        if len(normalized_title) > 200:
            raise ValueError("title must not exceed 200 characters")

        async with self._session.begin():
            user = await self._users.get_by_id(user_id)
            if user is None:
                # Repository 用 None 表示查询结果；service 根据当前用例判定这是一个
                # 业务错误。此异常离开 begin() 时会结束当前事务且不会产生写入。
                raise UserNotFoundError

            chat_session = await self._chat_sessions.create(
                user_id=user.id,
                title=normalized_title,
            )

        return chat_session.id
