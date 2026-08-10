# ============================================================================
# DeepResearch Makefile
# 用法: make <target> [ENV=development]
# 注意: Windows 下需要安装 make（可通过 choco install make 或 Git Bash）
# ============================================================================

.DEFAULT_GOAL := help

ENV ?= development
PYTHON ?= python

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
install:
	$(PYTHON) -m pip install -e ".[dev,test]"
	$(PYTHON) -m pre_commit install

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
dev:
	$(PYTHON) -m uvicorn app.main:app --reload --port 8000

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------
lint:
	$(PYTHON) -m ruff check ./app ./tests

format:
	$(PYTHON) -m ruff format ./app ./tests

format-check:
	$(PYTHON) -m ruff format --check ./app ./tests

typecheck:
	$(PYTHON) -m pyright --pythonpath ./.venv/Scripts/python.exe

test:
	$(PYTHON) -m pytest ./tests -v

check: lint format-check typecheck test
	@echo "All checks passed"

# ---------------------------------------------------------------------------
# Config verification (Lab 01)
# ---------------------------------------------------------------------------
verify-config:
	$(PYTHON) -c "from app.core.config import settings; print(f'Config loaded: {settings.PROJECT_NAME} v{settings.VERSION} [{settings.ENVIRONMENT.value}]')"

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Setup:"
	@echo "  install         Install deps + pre-commit hooks"
	@echo ""
	@echo "Server:"
	@echo "  dev             Dev server with hot reload (port 8000)"
	@echo ""
	@echo "Code quality:"
	@echo "  lint            Ruff lint check"
	@echo "  format          Ruff format"
	@echo "  format-check    Verify formatting without changing files"
	@echo "  typecheck       Pyright static type check"
	@echo "  test            Run the test suite"
	@echo "  check           Run lint + format-check + typecheck + test"
	@echo ""
	@echo "Config:"
	@echo "  verify-config   Verify config.py loads correctly (Lab 01)"

.PHONY: install dev lint format format-check typecheck test check verify-config help
