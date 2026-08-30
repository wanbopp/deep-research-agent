# DeepResearch

DeepResearch is an intelligent research assistant built with FastAPI, LangGraph,
hybrid RAG, and GraphRAG.

The project is being developed as a step-by-step engineering course. The current
foundation includes multi-environment configuration, structured logging, request
correlation, FastAPI middleware, and versioned health routes.

- Product vision: [`../docs/DeepSearch.md`](../docs/DeepSearch.md)
- Authoritative learning plan and current progress: [`../docs/LABS.MD`](../docs/LABS.MD)
- Full phase-by-phase curriculum: [`../docs/phases/README.md`](../docs/phases/README.md)
- Current lesson: [`../docs/lessons/lab-05-model-configuration-registry.md`](../docs/lessons/lab-05-model-configuration-registry.md)

## Local setup

Run the following commands in PowerShell from this `deep-research` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,test]"
```

The application has development defaults, so Phase 1 does not require an LLM
key, database, Redis, Neo4j, or Docker. To customize local settings, copy the
environment template:

```powershell
Copy-Item .env.example .env.development
```

## Run the Web runtime

The default command starts FastAPI, the durable Research Worker, and the
long-running Index Scheduler. This prevents either task queue from accepting
work while no consumer is running:

```powershell
Set-Location E:\workspace\Agent\DeepResearch\deep-research
.\.venv\Scripts\deep-research-runtime.exe
```

Running without arguments is equivalent to `--mode all`. The same installed
entrypoint retains component-only modes for diagnostics and future scaling:

```powershell
.\.venv\Scripts\deep-research-runtime.exe --mode all
.\.venv\Scripts\deep-research-runtime.exe --mode api
.\.venv\Scripts\deep-research-runtime.exe --mode worker
.\.venv\Scripts\deep-research-runtime.exe --mode index
.\.venv\Scripts\deep-research-runtime.exe --mode index --until-idle
```

`all` is the supported default for the current local and cloud deployment. The
Supervisor stops the whole runtime if either background consumer fails, rather
than leaving a healthy-looking API with permanently pending tasks. `api`,
`worker`, and `index` are operational diagnostics; they are not the default
Web mode. `index --until-idle` preserves the previous one-shot maintenance
behavior.

Application JSONL logs are split by finite runtime component under `LOG_DIR`:

```text
development-runtime-YYYY-MM-DD.jsonl
development-api-YYYY-MM-DD.jsonl
development-research-worker-YYYY-MM-DD.jsonl
development-index-worker-YYYY-MM-DD.jsonl
```

The console remains an aggregated developer view. Component routing never uses
user IDs, task IDs, prompts, filenames, or document content as file names.

### Runtime 与日志规范

以下约定是当前 Web 模式的统一运行规范，开发、测试和部署脚本都应遵守：

1. **默认入口唯一。** 后端统一使用 `deep-research-runtime` 启动；无参数即为
   `--mode all`，同时运行 API、Research Worker 和 Index Scheduler。前端在
   `deep-research-web` 中使用 `npm run dev` 单独启动。
2. **当前部署不拆服务。** 本地和云端默认都使用 `all`。`api`、`worker`、`index`
   只用于故障诊断、维护和将来独立扩容，不作为当前常规部署组合。
3. **禁止重复消费者。** 运行 `all` 时，不得再启动独立 Research Worker、Index
   Scheduler 或旧 Worker 脚本，否则会产生额外消费者并增加排障复杂度。旧脚本只保留
   兼容性；一次性索引维护统一使用 `--mode index --until-idle`。
4. **统一停止。** 前台运行时使用 `Ctrl+C`；Supervisor 会停止接收新请求、取消后台
   消费循环，并等待 FastAPI lifespan 和各组件释放数据库、Neo4j 与 tracing 资源。
   不应通过强制结束单个子组件来终止 `all` 模式。
5. **组件失败即整体失败。** Research Worker 或 Index Scheduler 意外退出时，整个
   Runtime 必须以失败状态退出，由进程管理器重启，不能留下“API 健康但任务无人领取”
   的半健康实例。
6. **日志按固定组件分流。** JSONL 只允许写入 `runtime`、`api`、
   `research-worker`、`index-worker` 四类文件；控制台用于聚合查看。组件名必须来自
   代码中的有限集合，禁止根据请求参数动态创建日志文件。
7. **日志不得泄密。** 文件名、事件字段和日志消息不得记录密码、Token、连接串、完整
   Prompt、文档正文或其他敏感数据。请求 ID、任务 ID 等关联字段可写入 JSONL 事件，
   但不得用作文件名，也不得形成无界日志标签。
8. **保持任务持久化语义。** 统一进程只合并运行入口，不把 Research 或 Index 任务改成
   API 进程内存队列；任务领取、租约、heartbeat、checkpoint、重试和恢复仍以持久化
   存储为准。

### Prompt 安全与版本规范

所有模型提示词必须遵守以下规则：

1. **固定系统指令。** System Prompt 必须来自 `app/agents/prompts/` 中已注册的版本化
   Markdown 资源，禁止把用户主题、聊天正文、证据、记忆或工具输出拼入 SystemMessage。
2. **低信任数据隔离。** 用户输入和检索内容通过严格字段契约编码为 HumanMessage JSON；
   Prompt Registry 会同时拒绝缺失字段和额外字段，错误日志不得包含字段值。
3. **禁止散落 Prompt。** 业务节点只能通过逻辑名称加载 Prompt，不应在 Python 文件中新增
   大段系统提示词。升级内容时新增版本文件并修改中央注册项。
4. **模型输出不直接可信。** 优先使用 Pydantic structured output；证据 ID、引用、身份、权限、
   来源和持久化状态仍必须由服务端确定性校验或绑定。
5. **版本可复现。** 每个 Prompt 工件包含逻辑名称、发布版本和正文 SHA-256。应用启动时会
   验证全部注册资源，防止 wheel 漏打包或空文件延迟到首次请求才失败。
6. **可观测但不采集正文。** 日志和 Trace 只记录 Prompt 名称、版本与哈希；不得记录完整
   System Prompt、HumanMessage、证据正文、记忆正文或模型隐藏推理。Prompt 版本不得作为
   包含用户值的动态指标标签。
7. **修改必须经过门禁。** Prompt 变更至少通过字段契约、信任层级、注入隔离、结构化输出、
   全量回归和相关真实 Provider smoke；不能只根据单次回答的主观观感发布。

当前注册项覆盖 Chat、Research Planner/Validator/Writer、GraphRAG extraction/repair/linking/
community/map/reduce、会话标题和长期记忆提取。Prompt Registry 与调用方式见
`app/agents/prompts/loader.py`。

- Health check: <http://127.0.0.1:8000/api/v1/health>
- Swagger UI: <http://127.0.0.1:8000/docs>
- OpenAPI JSON: <http://127.0.0.1:8000/api/v1/openapi.json>

Application startup enters the FastAPI lifespan first. The required PostgreSQL
dependency must be reachable before Uvicorn starts accepting HTTP requests.
On Windows, the unified entrypoint explicitly creates the Selector event loop
required by psycopg async, so it does not depend on Uvicorn `--reload` to obtain
a compatible child process. Run the React frontend separately with `npm run
dev` from the `deep-research-web` project.

## Quality checks

Run each command independently so a failure identifies the affected stage:

```powershell
python -m ruff check .\app .\tests
python -m ruff format --check .\app .\tests
python -m pyright --pythonpath ".\.venv\Scripts\python.exe"
python -m pytest .\tests -v
```

The optional `Makefile` exposes the same workflow for environments that provide
`make`; PowerShell users can use the commands above directly.
