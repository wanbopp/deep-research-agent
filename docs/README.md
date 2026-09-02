# DeepResearch 文档索引

本目录收纳适合随代码仓库长期维护的架构与设计文档。项目总览、系统架构和本地启动方式请先阅读根目录 [`README.md`](../README.md)。

## 设计文档

| 文档 | 内容 |
|---|---|
| [`enterprise-memory-architecture.md`](enterprise-memory-architecture.md) | 长期记忆的分层职责、身份边界、写入与检索链路 |
| [`memory-retrieval-comparison.md`](memory-retrieval-comparison.md) | 长期记忆候选召回、排序与上下文注入方案对比 |
| [`../deploy/README.md`](../deploy/README.md) | PostgreSQL/pgvector、Redis、Neo4j 的 VMware 与 systemd 运维方式 |

## 从架构定位到代码

| 架构模块 | 代码入口 |
|---|---|
| FastAPI 与 lifespan | [`app/main.py`](../app/main.py)、[`app/infrastructure/lifespan.py`](../app/infrastructure/lifespan.py) |
| Chat Graph 与 HITL | [`app/agents/chat/graph.py`](../app/agents/chat/graph.py)、[`app/services/chat.py`](../app/services/chat.py) |
| Research Graph | [`app/agents/research/graph.py`](../app/agents/research/graph.py)、[`app/agents/research/runtime.py`](../app/agents/research/runtime.py) |
| Hybrid RAG | [`app/rag/pipeline.py`](../app/rag/pipeline.py)、[`app/rag/hybrid.py`](../app/rag/hybrid.py) |
| GraphRAG | [`app/graphrag/pipeline.py`](../app/graphrag/pipeline.py)、[`app/graphrag/runtime.py`](../app/graphrag/runtime.py) |
| 知识索引 | [`app/services/index_worker.py`](../app/services/index_worker.py)、[`app/entrypoints/index_worker.py`](../app/entrypoints/index_worker.py) |
| 持久研究任务 | [`app/services/research.py`](../app/services/research.py)、[`app/workers/research_task.py`](../app/workers/research_task.py) |
| Prompt Registry | [`app/agents/prompts/loader.py`](../app/agents/prompts/loader.py) |
| Metrics 与 Tracing | [`app/observability/metrics.py`](../app/observability/metrics.py)、[`app/observability/tracing.py`](../app/observability/tracing.py) |
| 统一 Runtime | [`app/entrypoints/runtime.py`](../app/entrypoints/runtime.py) |

## 运行时契约

- OpenAPI：启动后访问 <http://127.0.0.1:8000/api/v1/openapi.json>
- Swagger UI：启动后访问 <http://127.0.0.1:8000/docs>
- Prometheus Metrics：启动后访问 <http://127.0.0.1:8000/metrics>
- 环境变量：以 [`.env.example`](../.env.example) 为公开配置契约，真实密钥只保存在未提交的环境文件或 Secret 管理系统中。

新增稳定的跨模块设计时，请在本目录添加专题文档并同步本索引；短期实验记录和包含环境细节的验收日志不应作为公开架构文档长期保留。
