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

## Run the API

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

- Health check: <http://127.0.0.1:8000/api/v1/health>
- Swagger UI: <http://127.0.0.1:8000/api/v1/docs>
- OpenAPI JSON: <http://127.0.0.1:8000/api/v1/openapi.json>

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
