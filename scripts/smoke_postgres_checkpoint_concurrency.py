"""观察未加执行 guard 时，同一 LangGraph thread 的真实 PostgreSQL 并发行为.

Checkpoint 10E 不能只根据 saver 源码推测竞态。本脚本使用一个确定性、无模型的
StateGraph，让两个执行被强制从同一个父 checkpoint 开始，再检查 PostgreSQL 中形成
的父子版本关系以及最终默认可见状态。

教学心智模型：

1. 先执行一次 seed，得到稳定的共同父 checkpoint；
2. saver 在两个并发执行分别读完该父版本后把它们一起放行；
3. 两个执行各自写入自己的新 checkpoint，不使用任何应用级 guard；
4. 查询真实 ``checkpoints`` 表，判断是否形成同父多子分支；
5. 读取默认 latest state，判断线性会话视图是否只看见其中一条分支。

脚本不会调用 LLM、工具或业务 Repository，也不会输出数据库名、连接串、thread ID、
checkpoint ID、状态内容或凭据。所有表都创建在随机临时数据库中，并在 finally 中删除。
"""

import asyncio
import json
import selectors
from collections.abc import Sequence
from dataclasses import dataclass
from operator import add
from time import perf_counter
from typing import Annotated, TypedDict, cast
from uuid import uuid4

import psycopg
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg import AsyncConnection, sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from app.core.config import settings

CONNECTION_TIMEOUT_SECONDS = 10
OBSERVATION_TIMEOUT_SECONDS = 30.0

# 这些值只用于脚本内部判断两条分支是否被线性合并。最终 JSON 只输出布尔值，
# 不输出状态正文。固定值也能让失败可复现，不需要使用真实用户内容。
SEED_VALUE = "seed"
FIRST_INPUT = "concurrent-a"
SECOND_INPUT = "concurrent-b"


class _ObservationState(TypedDict):
    """无模型观察图的最小状态.

    ``Annotated[..., add]`` 表示每个节点只返回本次新增值，LangGraph 使用列表加法
    把增量并入历史。它与 ChatState 的 ``add_messages`` 心智模型相同，但没有消息、
    Prompt 或 provider，因此更适合隔离观察 checkpointer 的并发机制。
    """

    observations: Annotated[list[str], add]


@dataclass(frozen=True, slots=True)
class _CheckpointTreeStats:
    """从真实 checkpoints 表提取的安全版本树统计."""

    checkpoint_count: int
    branch_parent_count: int
    max_sibling_count: int


class _ReadBarrierPostgresSaver(AsyncPostgresSaver):
    """只为 smoke 增加“读取后暂停”能力的真实 PostgreSQL saver.

    该类不改写 ``aput``、SQL 或 checkpoint 内容。两个调用仍使用父类的真实
    ``aget_tuple`` 读取 PostgreSQL；只有读取完成后的返回时机被 Event 协调。
    这保证两个 Graph 执行都拿到同一个父快照后，才开始各自推进。
    """

    def __init__(
        self,
        pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    ) -> None:
        """保存真实连接池并初始化尚未启用的两方读取屏障.

        Args:
            pool: 指向随机临时数据库的异步 psycopg 连接池。父类负责通过该池
                执行所有 saver SQL；本类不创建第二个连接池。
        """
        super().__init__(pool)
        self._reads_remaining = 0
        self._release_reads = asyncio.Event()
        self._observed_checkpoint_ids: list[str | None] = []

    @property
    def observed_checkpoint_ids(self) -> tuple[str | None, ...]:
        """返回屏障捕获的父 checkpoint ID，仅供内存中的相等性判断."""
        return tuple(self._observed_checkpoint_ids)

    def pause_next_reads(self, count: int) -> None:
        """要求接下来的若干次 ``aget_tuple`` 在全部读完后一起返回.

        Args:
            count: 需要同步的读取次数。本 smoke 固定传入 2，分别对应两个 Graph。

        Raises:
            ValueError: count 小于 2，无法形成并发比较。
            RuntimeError: 上一次屏障仍在工作，避免混淆两轮观察结果。
        """
        if count < 2:
            raise ValueError("read barrier requires at least two participants")
        if self._reads_remaining:
            raise RuntimeError("read barrier is already active")

        self._reads_remaining = count
        self._observed_checkpoint_ids.clear()
        self._release_reads.clear()

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """真实读取 checkpoint，并在观察阶段等待另一个执行完成读取.

        Args:
            config: LangGraph 传入的运行配置，其中包含内部 thread ID。

        Returns:
            父类从 PostgreSQL 反序列化出的 checkpoint；没有记录时返回 None。

        Raises:
            TimeoutError: 另一个并发执行没有在预算内到达读取屏障。
            Exception: 父类 PostgreSQL 读取错误原样向上传播。
        """
        checkpoint_tuple = await super().aget_tuple(config)

        # seed 和最终 snapshot 读取时屏障未启用，直接保持父类行为。
        if self._reads_remaining == 0:
            return checkpoint_tuple

        checkpoint_id: str | None = None
        if checkpoint_tuple is not None:
            # RunnableConfig 的类型把 configurable 声明成可选键。checkpoint tuple
            # 正常会带它，但 smoke 仍在边界上安全读取，不能让观察工具依赖隐式假设。
            configurable = checkpoint_tuple.config.get("configurable", {})
            raw_checkpoint_id = configurable.get("checkpoint_id")
            if isinstance(raw_checkpoint_id, str):
                checkpoint_id = raw_checkpoint_id

        # 这段更新之间没有 await，因此同一个 event loop 中不会丢失计数。
        self._observed_checkpoint_ids.append(checkpoint_id)
        self._reads_remaining -= 1
        if self._reads_remaining == 0:
            self._release_reads.set()

        # 第一个 Graph 会停在这里；第二个 Graph 读完同一父版本后 set Event，
        # 两者才继续进入 Pregel 执行和 saver.aput 写入阶段。
        await asyncio.wait_for(
            self._release_reads.wait(),
            timeout=OBSERVATION_TIMEOUT_SECONDS,
        )
        return checkpoint_tuple


def _record_latest_input(state: _ObservationState) -> _ObservationState:
    """为当前 Graph 输入追加一个确定性的处理结果.

    Args:
        state: saver 恢复的历史与本次输入合并后的当前状态。

    Returns:
        仅包含一个新增值的状态增量。列表 reducer 会把它追加到当前分支。

    Notes:
        这个节点不访问网络、模型、工具或数据库。LangGraph 在节点外围自动调用
        saver，因此观察到的分支只来自框架并发，不来自节点副作用。
    """
    latest_input = state["observations"][-1]
    return {"observations": [f"processed:{latest_input}"]}


def _elapsed_ms(started_at: float) -> float:
    """返回耗时毫秒数，不暴露基础设施配置."""
    return round((perf_counter() - started_at) * 1000, 2)


def _conninfo(database: str) -> str:
    """为指定数据库构造只交给驱动使用、绝不输出的连接串.

    Args:
        database: 管理数据库或本次随机临时数据库名称。

    Returns:
        psycopg 正确转义后的连接参数字符串。
    """
    return make_conninfo(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=database,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        connect_timeout=CONNECTION_TIMEOUT_SECONDS,
    )


def _create_database(admin_database: str, test_database: str) -> None:
    """在事务外创建随机临时数据库.

    Args:
        admin_database: 已存在、仅作为 CREATE DATABASE 入口的数据库。
        test_database: 本 smoke 生成的随机数据库名。
    """
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_database)))


def _drop_database(admin_database: str, test_database: str) -> None:
    """终止临时连接，并且只删除本 smoke 的随机数据库.

    Args:
        admin_database: 已存在的管理入口数据库。
        test_database: 需要清理的随机数据库名。
    """
    with psycopg.connect(_conninfo(admin_database), autocommit=True) as connection:
        connection.execute(
            """
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = %s AND pid <> pg_backend_pid()
            """,
            (test_database,),
        )
        connection.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(test_database)))


def _selector_loop_factory() -> asyncio.AbstractEventLoop:
    """创建 Windows 下 psycopg 异步连接需要的 Selector event loop."""
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


async def _checkpoint_tree_stats(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    *,
    thread_id: str,
) -> _CheckpointTreeStats:
    """读取一个内部 thread 的版本数量与同父分支统计.

    Args:
        pool: saver 使用的同一个临时 PostgreSQL 连接池。
        thread_id: 随机内部 thread ID，只作为 SQL 参数使用，不会输出。

    Returns:
        checkpoint 总数、产生多个子节点的父节点数量，以及最大兄弟节点数。
    """
    async with pool.connection() as connection:
        count_cursor = await connection.execute(
            """
            SELECT COUNT(*) AS checkpoint_count
            FROM checkpoints
            WHERE thread_id = %s AND checkpoint_ns = ''
            """,
            (thread_id,),
        )
        count_row = await count_cursor.fetchone()
        if count_row is None:
            raise RuntimeError("checkpoint count query returned no row")

        branch_cursor = await connection.execute(
            """
            SELECT COUNT(*) AS branch_parent_count,
                   COALESCE(MAX(child_count), 0) AS max_sibling_count
            FROM (
                SELECT parent_checkpoint_id, COUNT(*) AS child_count
                FROM checkpoints
                WHERE thread_id = %s
                  AND checkpoint_ns = ''
                  AND parent_checkpoint_id IS NOT NULL
                GROUP BY parent_checkpoint_id
                HAVING COUNT(*) > 1
            ) AS branches
            """,
            (thread_id,),
        )
        branch_row = await branch_cursor.fetchone()
        if branch_row is None:
            raise RuntimeError("checkpoint branch query returned no row")

    return _CheckpointTreeStats(
        checkpoint_count=int(count_row["checkpoint_count"]),
        branch_parent_count=int(branch_row["branch_parent_count"]),
        max_sibling_count=int(branch_row["max_sibling_count"]),
    )


def _contains(values: Sequence[str], expected: str) -> bool:
    """判断内部状态是否包含某个固定观察值，不返回或记录状态正文."""
    return expected in values


async def _exercise_unprotected_concurrency(
    database: str,
) -> dict[str, bool | int | float]:
    """在随机数据库中执行同父 checkpoint 的两次未保护 Graph 调用.

    Args:
        database: 已创建但尚未包含任何表的随机临时数据库。

    Returns:
        只包含布尔值、计数和耗时的并发观察摘要。

    Raises:
        Exception: saver setup、Graph 执行或 SQL 观察失败。顶层只输出异常类型，
            并仍在 finally 中关闭 pool 和删除临时数据库。
    """
    started_at = perf_counter()
    pool: AsyncConnectionPool[AsyncConnection[DictRow]] = AsyncConnectionPool(
        conninfo=_conninfo(database),
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        min_size=0,
        max_size=4,
        open=False,
        timeout=CONNECTION_TIMEOUT_SECONDS,
    )

    try:
        await pool.open()
        saver = _ReadBarrierPostgresSaver(pool)
        await saver.setup()

        # 这是一个单节点确定性图。编译后的同一个 Graph 对象可以并发调用；
        # 真正决定持久化身份的是 config.configurable.thread_id。
        builder = StateGraph(_ObservationState)
        builder.add_node("record", _record_latest_input)
        builder.add_edge(START, "record")
        builder.add_edge("record", END)
        graph = builder.compile(checkpointer=saver)

        internal_thread_id = f"checkpoint-concurrency-{uuid4().hex}"
        config: RunnableConfig = {
            "configurable": {
                "thread_id": internal_thread_id,
            }
        }

        # seed 先建立共同父历史。此时屏障尚未启用，所以它是普通线性执行。
        seed_result = cast(
            _ObservationState,
            await graph.ainvoke(
                {"observations": [SEED_VALUE]},
                config=config,
            ),
        )
        before_stats = await _checkpoint_tree_stats(
            pool,
            thread_id=internal_thread_id,
        )

        # 只拦截接下来的两次初始 checkpoint 读取。两个 task 都读完共同父版本后，
        # saver 才让它们继续，稳定制造“同一个父 checkpoint 被同时推进”。
        saver.pause_next_reads(2)
        first_task = asyncio.create_task(
            graph.ainvoke(
                {"observations": [FIRST_INPUT]},
                config=config,
            )
        )
        second_task = asyncio.create_task(
            graph.ainvoke(
                {"observations": [SECOND_INPUT]},
                config=config,
            )
        )
        first_raw, second_raw = await asyncio.wait_for(
            asyncio.gather(first_task, second_task),
            timeout=OBSERVATION_TIMEOUT_SECONDS,
        )
        first_result = cast(_ObservationState, first_raw)
        second_result = cast(_ObservationState, second_raw)

        # 不指定 checkpoint_id 时，aget_state 会按 saver 默认规则读取 latest。
        # 如果形成两个分支，这个视图通常只会选择其中一个叶子，而不是自动合并。
        latest_snapshot = await graph.aget_state(config)
        latest_values = cast(
            list[str],
            latest_snapshot.values.get("observations", []),
        )
        after_stats = await _checkpoint_tree_stats(
            pool,
            thread_id=internal_thread_id,
        )

        observed_parent_ids = saver.observed_checkpoint_ids
        both_reads_used_same_parent = (
            len(observed_parent_ids) == 2
            and observed_parent_ids[0] is not None
            and observed_parent_ids[0] == observed_parent_ids[1]
        )
        seed_completed = _contains(seed_result["observations"], SEED_VALUE)
        both_invocations_completed = _contains(first_result["observations"], FIRST_INPUT) and _contains(
            second_result["observations"], SECOND_INPUT
        )
        result_histories_diverged = not _contains(first_result["observations"], SECOND_INPUT) and not _contains(
            second_result["observations"], FIRST_INPUT
        )
        version_branch_detected = after_stats.branch_parent_count >= 1 and after_stats.max_sibling_count >= 2
        latest_contains_exactly_one_competing_input = (
            int(_contains(latest_values, FIRST_INPUT)) + int(_contains(latest_values, SECOND_INPUT)) == 1
        )
        checkpoint_rows_increased = after_stats.checkpoint_count > before_stats.checkpoint_count
        unprotected_linear_history_is_unsafe = (
            both_reads_used_same_parent
            and both_invocations_completed
            and result_histories_diverged
            and version_branch_detected
            and latest_contains_exactly_one_competing_input
        )

        return {
            "model_call_count": 0,
            "seed_completed": seed_completed,
            "both_reads_used_same_parent": both_reads_used_same_parent,
            "both_invocations_completed": both_invocations_completed,
            "result_histories_diverged": result_histories_diverged,
            "version_branch_detected": version_branch_detected,
            "latest_contains_exactly_one_competing_input": (latest_contains_exactly_one_competing_input),
            "checkpoint_rows_increased": checkpoint_rows_increased,
            "unprotected_linear_history_is_unsafe": (unprotected_linear_history_is_unsafe),
            "checkpoint_count_before": before_stats.checkpoint_count,
            "checkpoint_count_after": after_stats.checkpoint_count,
            "branch_parent_count": after_stats.branch_parent_count,
            "max_sibling_count": after_stats.max_sibling_count,
            "elapsed_ms": _elapsed_ms(started_at),
        }
    finally:
        await pool.close()


def _run_smoke() -> dict[str, object]:
    """创建临时数据库、执行并发观察，并保证清理.

    Returns:
        包含总判定、机制证据、清理状态和总耗时的安全 JSON 字典。

    Raises:
        RuntimeError: 已创建的随机数据库无法清理。
        Exception: PostgreSQL 或 LangGraph 错误在清理后继续传播。
    """
    started_at = perf_counter()
    admin_database = settings.POSTGRES_DB
    test_database = f"deep_research_concurrency_{uuid4().hex[:10]}"
    database_created = False
    cleanup_ok = False
    checks: dict[str, bool | int | float]

    try:
        _create_database(admin_database, test_database)
        database_created = True
        checks = asyncio.run(
            _exercise_unprotected_concurrency(test_database),
            loop_factory=_selector_loop_factory,
        )
    finally:
        if database_created:
            try:
                _drop_database(admin_database, test_database)
            except Exception:
                cleanup_ok = False
            else:
                cleanup_ok = True

    if database_created and not cleanup_ok:
        raise RuntimeError("temporary concurrency database cleanup failed")

    required_checks = (
        checks["model_call_count"] == 0,
        checks["seed_completed"] is True,
        checks["both_reads_used_same_parent"] is True,
        checks["both_invocations_completed"] is True,
        checks["result_histories_diverged"] is True,
        checks["version_branch_detected"] is True,
        checks["latest_contains_exactly_one_competing_input"] is True,
        checks["checkpoint_rows_increased"] is True,
        checks["unprotected_linear_history_is_unsafe"] is True,
        cleanup_ok,
    )
    return {
        "ok": all(required_checks),
        **checks,
        "cleanup_ok": cleanup_ok,
        "total_elapsed_ms": _elapsed_ms(started_at),
    }


def main() -> int:
    """打印单行安全 JSON，并返回适合 PowerShell/CI 的退出码."""
    started_at = perf_counter()
    try:
        summary = _run_smoke()
    except Exception as error:
        # PostgreSQL 异常字符串可能包含地址、用户名或 SQL 参数；顶层只公开类型。
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "elapsed_ms": _elapsed_ms(started_at),
                }
            )
        )
        return 1

    print(json.dumps(summary))
    return 0 if summary["ok"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
