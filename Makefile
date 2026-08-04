# ============================================================================
# DeepResearch Makefile
# 用法: make <target> [ENV=development]
# 注意: Windows 下需要安装 make（可通过 choco install make 或 Git Bash）
# ============================================================================

.DEFAULT_GOAL := help

ENV ?= development

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
install:
	pip install uv
	uv sync
	uv run pre-commit install

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
dev:
	uv run uvicorn app.main:app --reload --port 8000

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------
lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run pyright

check: lint typecheck
	@echo "All checks passed"

# ---------------------------------------------------------------------------
# Config verification (Lab 01)
# ---------------------------------------------------------------------------
verify-config:
	uv run python -c "from app.core.config import settings; print(f'Config loaded: {settings.PROJECT_NAME} v{settings.VERSION} [{settings.ENVIRONMENT.value}]')"

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
clean:
	rm -rf .venv __pycache__ .pytest_cache

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
	@echo "  typecheck       Pyright static type check"
	@echo "  check           Run lint + typecheck"
	@echo ""
	@echo "Config:"
	@echo "  verify-config   Verify config.py loads correctly (Lab 01)"
	@echo ""
	@echo "Misc:"
	@echo "  clean           Remove .venv, __pycache__, .pytest_cache"

.PHONY: install dev lint format typecheck check verify-config clean help
