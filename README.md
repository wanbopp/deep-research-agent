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
