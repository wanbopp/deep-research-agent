"""Verify the real FastAPI infrastructure startup and shutdown lifecycle.

本脚本直接进入当前 FastAPI app 的 lifespan context，覆盖完整路径：

1. startup 创建并打开共享资源；
2. 三个真实依赖探针完成预热；
3. 两个独立 ORM Session 通过共享 Engine 并发执行 ``SELECT 1``；
4. 资源发布到 ``app.state``，HTTP 请求可以在该阶段运行；
5. shutdown 移除 ``app.state`` 引用并执行所有清理回调。

脚本不调用模型，不输出地址、凭据、连接串、响应正文或底层异常文本。


核心知识：
- AsyncExitStack 保证 startup 失败也会清理资源。
- required PostgreSQL 失败会阻止应用启动。
- optional Neo4j/Redis 失败不会直接决定 startup。
- app.state.resources 只在 yield 期间存在。
- HTTP/Agent 请求复用同一个 ApplicationResources。
- AsyncEngine/sessionmaker 可以共享，具体 AsyncSession 不能共享。
- shutdown 后 accessor 必须拒绝继续读取资源。


调用链
FastAPI lifespan
  -> factory
  -> 登记清理回调
  -> pool.open()
  -> 并发 probes
  -> required 判断
  -> app.state.resources
  -> yield
  -> HTTP 请求
  -> 删除 app.state.resources
  -> 逆序关闭资源
"""

import asyncio
import json
import selectors
from time import perf_counter
from typing import Protocol, cast

from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from app.core.config import settings
from app.infrastructure.lifespan import get_application_resources
from app.main import app

TOTAL_TIMEOUT_SECONDS = 15.0


class _CheckedOutPool(Protocol):
    """描述 QueuePool 验收所需的最小只读接口."""

    _max_overflow: int

    def size(self) -> int:
        """返回连接池的常驻连接预算."""
        ...

    def checkedout(self) -> int:
        """返回当前仍被 Session 持有的连接数量."""
        ...


def _elapsed_ms(started_at: float) -> float:
    """Return elapsed milliseconds without exposing infrastructure settings."""
    return round((perf_counter() - started_at) * 1000, 2)


def _selector_loop_factory() -> asyncio.AbstractEventLoop:
    """Create the selector event loop required by psycopg async on Windows."""
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


async def _run_smoke() -> dict[str, object]:
    """Enter the real lifespan, inspect ownership, then verify shutdown state."""
    started_at = perf_counter()
    state_absent_before_startup = not hasattr(app.state, "resources")

    # 外层预算覆盖 startup probes、一次本地 ASGI 请求以及 shutdown。
    async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
        async with app.router.lifespan_context(app):
            # accessor 是业务代码读取资源的唯一入口。连续读取应返回同一组引用，
            # 证明请求期间复用的是 lifespan 创建的资源，而不是每次重新构造。
            first_resources = get_application_resources(app)
            second_resources = get_application_resources(app)

            state_available_during_lifespan = hasattr(app.state, "resources")
            resource_identity_is_stable = first_resources is second_resources
            postgres_pool_open_during_lifespan = not first_resources.postgres_pool.closed
            orm_engine_identity_is_stable = first_resources.orm_engine is second_resources.orm_engine
            orm_session_factory_identity_is_stable = (
                first_resources.orm_session_factory is second_resources.orm_session_factory
            )
            orm_dialect_is_async_psycopg = (
                first_resources.orm_engine.dialect.is_async and first_resources.orm_engine.driver == "psycopg"
            )
            orm_pool = cast(_CheckedOutPool, first_resources.orm_engine.pool)
            orm_pool_budget_matches = (
                orm_pool.size() == settings.POSTGRES_ORM_POOL_SIZE
                and orm_pool._max_overflow == settings.POSTGRES_ORM_MAX_OVERFLOW
            )

            # sessionmaker 是可共享的“Session 工厂”，每次调用则必须产生新 Session。
            # 两个 Session 并发查询能证明它们不会共享事务状态，也会让连接池按需
            # checkout 两个连接。这里只执行 SELECT 1，不读取或修改任何业务表。
            async with (
                first_resources.orm_session_factory() as first_session,
                first_resources.orm_session_factory() as second_session,
            ):
                orm_sessions_are_distinct = first_session is not second_session
                sessions_share_engine = (
                    first_session.bind is first_resources.orm_engine
                    and second_session.bind is first_resources.orm_engine
                )
                first_value, second_value = await asyncio.gather(
                    first_session.scalar(text("SELECT 1")),
                    second_session.scalar(text("SELECT 1")),
                )

            orm_select_succeeded = first_value == 1 and second_value == 1
            orm_connections_returned_before_shutdown = orm_pool.checkedout() == 0

            # ASGITransport 不自行触发 lifespan；此处已经手动进入真实 lifespan，
            # 因而该请求发生在 yield 期间，等价于应用正常运行阶段。
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                # Uvicorn 会为每个请求创建独立 task。ASGITransport 默认在当前 task
                # 直接调用 app，因此这里显式创建子 task，让 request_id 等 ContextVar
                # 留在请求上下文中，不污染负责 startup/shutdown 的父 task。
                response = await asyncio.create_task(client.get("/api/v1/health"))

            health_status_code = response.status_code

        # 离开 context 后，lifespan 的 finally 与 AsyncExitStack 均已执行。
        state_removed_after_shutdown = not hasattr(app.state, "resources")
        postgres_pool_closed_after_shutdown = first_resources.postgres_pool.closed
        # dispose() 会关闭池内空闲连接；此处至少还要保证没有连接被 Session 遗留占用。
        orm_connections_released_after_shutdown = orm_pool.checkedout() == 0

        try:
            get_application_resources(app)
        except RuntimeError:
            accessor_rejects_after_shutdown = True
        else:
            accessor_rejects_after_shutdown = False

    elapsed_ms = _elapsed_ms(started_at)
    within_total_budget = elapsed_ms <= TOTAL_TIMEOUT_SECONDS * 1000
    ok = all(
        (
            state_absent_before_startup,
            state_available_during_lifespan,
            resource_identity_is_stable,
            postgres_pool_open_during_lifespan,
            orm_engine_identity_is_stable,
            orm_session_factory_identity_is_stable,
            orm_dialect_is_async_psycopg,
            orm_pool_budget_matches,
            orm_sessions_are_distinct,
            sessions_share_engine,
            orm_select_succeeded,
            orm_connections_returned_before_shutdown,
            health_status_code == 200,
            state_removed_after_shutdown,
            postgres_pool_closed_after_shutdown,
            orm_connections_released_after_shutdown,
            accessor_rejects_after_shutdown,
            within_total_budget,
        )
    )

    return {
        "ok": ok,
        "state_absent_before_startup": state_absent_before_startup,
        "state_available_during_lifespan": state_available_during_lifespan,
        "resource_identity_is_stable": resource_identity_is_stable,
        "postgres_pool_open_during_lifespan": postgres_pool_open_during_lifespan,
        "orm_engine_identity_is_stable": orm_engine_identity_is_stable,
        "orm_session_factory_identity_is_stable": orm_session_factory_identity_is_stable,
        "orm_dialect_is_async_psycopg": orm_dialect_is_async_psycopg,
        "orm_pool_budget_matches": orm_pool_budget_matches,
        "orm_sessions_are_distinct": orm_sessions_are_distinct,
        "sessions_share_engine": sessions_share_engine,
        "orm_select_succeeded": orm_select_succeeded,
        "orm_connections_returned_before_shutdown": orm_connections_returned_before_shutdown,
        "health_status_code": health_status_code,
        "state_removed_after_shutdown": state_removed_after_shutdown,
        "postgres_pool_closed_after_shutdown": postgres_pool_closed_after_shutdown,
        "orm_connections_released_after_shutdown": orm_connections_released_after_shutdown,
        "accessor_rejects_after_shutdown": accessor_rejects_after_shutdown,
        "within_total_budget": within_total_budget,
        "elapsed_ms": elapsed_ms,
    }


def main() -> int:
    """Run the lifecycle smoke and print one safe JSON summary."""
    started_at = perf_counter()

    try:
        summary = asyncio.run(
            _run_smoke(),
            loop_factory=_selector_loop_factory,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "stage": "application_lifespan",
                    "error_type": type(exc).__name__,
                    "elapsed_ms": _elapsed_ms(started_at),
                }
            )
        )
        return 1

    print(json.dumps(summary))
    return 0 if summary["ok"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
