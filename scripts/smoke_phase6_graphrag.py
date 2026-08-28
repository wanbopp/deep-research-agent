"""Phase 6 真实 GraphRAG 总 Gate.

本脚本使用固定、无敏感内容的十段短文完成以下真实链路：

1. 真实 structured LLM 抽取候选实体和关系；
2. 服务端绑定 chunk 身份与原文 span，再执行保守实体消歧；
3. 真实 Neo4j 参数化、幂等写入和用户隔离查询；
4. connected-components fallback 社区检测与真实 LLM 社区摘要；
5. 真实查询实体提取、Local 路径、真实 embedding Global 匹配；
6. 真实 LLM map-reduce，并验证引用只能来自原始 chunk。

脚本最终只输出模型名称、计数、耗时和布尔门禁，不输出 API key、Base URL、
完整 prompt、原始文档或模型回答。每次运行使用随机 user_id，并在 finally 清理图数据。
"""

import asyncio
import json
from hashlib import sha256
from time import perf_counter
from uuid import UUID, uuid4

from neo4j import AsyncGraphDatabase, RoutingControl

from app.core.config import settings
from app.graphrag.runtime import create_graphrag_runtime

_CORPUS = (
    "Orion Labs created the Nova Engine.",
    "The Nova Engine uses GraphStore.",
    "GraphStore is located in the Atlas Cloud.",
    "Orion Labs produces the Research Console.",
    "The Research Console uses the Nova Engine.",
    "Mira Chen works for Orion Labs.",
    "Mira Chen created the Evidence Protocol.",
    "The Evidence Protocol is part of the Nova Engine.",
    "The Atlas Cloud uses the Evidence Protocol.",
    "The Knowledge Atlas uses GraphStore.",
)
_LOCAL_QUERY = "How is Orion Labs connected to the Nova Engine?"
_GLOBAL_QUERY = "What overall system connects Orion Labs, GraphStore, and the Evidence Protocol?"


async def _count_user_graph(driver, user_id: UUID) -> tuple[int, int]:  # noqa: ANN001
    """读取 smoke 用户的节点和事实数量，不接触其他用户 namespace."""
    records, _, _ = await driver.execute_query(
        """
MATCH (n {user_id: $user_id})
WHERE n:GraphDocument OR n:GraphChunk OR n:Entity OR n:EntityMention OR n:Community
WITH count(n) AS node_count
OPTIONAL MATCH ()-[f:FACT {user_id: $user_id}]->()
RETURN node_count, count(f) AS fact_count
""",
        {"user_id": str(user_id)},
        routing_=RoutingControl.READ,
    )
    return int(records[0]["node_count"]), int(records[0]["fact_count"])


async def run_smoke() -> None:
    """执行真实 Phase 6 Gate，并始终清理本次随机用户图."""
    started_at = perf_counter()
    user_id = uuid4()
    outsider_id = uuid4()
    driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )
    runtime = create_graphrag_runtime(config=settings, neo4j_driver=driver)
    cleanup_ok = False
    try:
        await driver.verify_connectivity()
        await runtime.repository.setup_schema()
        await runtime.repository.delete_user_graph(user_id=user_id)

        # 抽取请求彼此没有共享状态，可以并发发送；消歧和写入保持原始文档顺序，
        # 这样后一个 chunk 能看到前一个 chunk 已建立的 canonical entity。
        semaphore = asyncio.Semaphore(3)

        async def extract_one(index: int, content: str):  # noqa: ANN202
            """在明确并发预算内调用真实模型，并返回可信业务 ID 快照."""
            document_id = uuid4()
            chunk_id = uuid4()
            async with semaphore:
                graph = await runtime.extractor.extract(
                    user_id=user_id,
                    document_id=document_id,
                    chunk_id=chunk_id,
                    content=content,
                    content_sha256=sha256(content.encode("utf-8")).hexdigest(),
                )
            return index, graph

        extracted = await asyncio.gather(*(extract_one(index, content) for index, content in enumerate(_CORPUS)))
        resolved_graphs = []
        for _, graph in sorted(extracted, key=lambda item: item[0]):
            resolved = await runtime.resolver.resolve(graph)
            await runtime.repository.replace_graph_document(resolved)
            resolved_graphs.append(resolved)

        before_repeat = await _count_user_graph(driver, user_id)
        # 不再次调用模型，直接重放同一已解析结果，精确验证 Neo4j 幂等边界。
        await runtime.repository.replace_graph_document(resolved_graphs[0])
        after_repeat = await _count_user_graph(driver, user_id)

        communities = await runtime.communities.rebuild(user_id=user_id)
        local = await runtime.local.search(user_id=user_id, query=_LOCAL_QUERY, depth=2, limit=20)
        global_result = await runtime.global_retriever.search(
            user_id=user_id,
            query=_GLOBAL_QUERY,
            top_k=3,
        )
        global_answer = await runtime.global_answerer.answer(
            query=_GLOBAL_QUERY,
            retrieval=global_result,
        )
        context = runtime.context.assemble(local=local, global_result=global_result)

        # 即使知道另一个用户的 canonical entity UUID，Cypher 仍同时过滤 user_id。
        outsider_paths = await runtime.repository.local_paths(
            user_id=outsider_id,
            entity_ids=local.linked_entity_ids,
            depth=2,
            limit=20,
        )
        known_chunk_ids = {graph.source_chunk_id for graph in resolved_graphs}
        citations_are_grounded = set(global_answer.source_chunk_ids).issubset(known_chunk_ids) and {
            item.source_chunk_id for item in context.citations
        }.issubset(known_chunk_ids)
        ok = all(
            (
                len(resolved_graphs) == len(_CORPUS),
                all(graph.entities and graph.relations for graph in resolved_graphs),
                before_repeat == after_repeat,
                before_repeat[1] >= len(_CORPUS),
                bool(communities),
                bool(local.linked_entity_ids),
                bool(local.paths),
                not local.fallback_required,
                bool(global_result.communities),
                bool(global_answer.answer.strip()),
                citations_are_grounded,
                not outsider_paths,
            )
        )
        print(
            json.dumps(
                {
                    "ok": ok,
                    "model": settings.DEFAULT_LLM_MODEL,
                    "document_count": len(_CORPUS),
                    "entity_count": len({entity.entity_id for graph in resolved_graphs for entity in graph.entities}),
                    "fact_count": before_repeat[1],
                    "idempotent_counts": before_repeat == after_repeat,
                    "community_count": len(communities),
                    "local_path_count": len(local.paths),
                    "global_community_count": len(global_result.communities),
                    "global_answer_has_sources": bool(global_answer.source_chunk_ids),
                    "citations_are_grounded": citations_are_grounded,
                    "owner_isolation": not outsider_paths,
                    "elapsed_ms": round((perf_counter() - started_at) * 1000, 2),
                },
                ensure_ascii=False,
            )
        )
        if not ok:
            raise SystemExit(1)
    finally:
        try:
            await runtime.repository.delete_user_graph(user_id=user_id)
            cleanup_ok = await _count_user_graph(driver, user_id) == (0, 0)
        finally:
            await driver.close()
        if not cleanup_ok:
            raise RuntimeError("Phase 6 smoke graph cleanup failed")


if __name__ == "__main__":
    asyncio.run(run_smoke())
