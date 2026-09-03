.PHONY: help install sync lint format format-check typecheck check test test-fast build clean

.DEFAULT_GOAL := help

help: ## Show available Makefile targets
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: sync ## Alias for sync

sync: ## Sync dependencies and development environment via uv
	uv sync

lint: ## Run code linters (ruff)
	uv run ruff check .

format: ## Format source files (ruff)
	uv run ruff format .

format-check: ## Check formatting without modifying files
	uv run ruff format --check .

typecheck: ## Run strict MyPy type analysis
	uv run mypy --strict src/metrifid .github/scripts/validate_ci_evidence.py

check: lint format-check typecheck ## Run all static quality gates (lint, format, typecheck)

test: ## Run unit and contract tests with pytest
	uv run pytest -q

test-fast: ## Run tests in parallel with xdist
	uv run pytest -q -n auto

build: ## Build distribution artifacts (wheel and sdist)
	uv build

clean: ## Remove build artifacts and temporary caches
	rm -rf dist/ build/ *.egg-info .pytest_cache/ .mypy_cache/ .ruff_cache/ .wheel-test-venv/
