.PHONY: all install install-dev run dev test lint format build push clean help

# =============================================================================
# Variables
# =============================================================================
DOCKER := docker
# The tag the platform deployment pulls (platform-deployment pim.images.FOODCHAT)
IMGTAG := wisefood/foodchat:latest
PYTHON := python3
PIP := pip3
PORT ?= 8000

# Load .env file if it exists
ifneq (,$(wildcard ./.env))
	include .env
	export
endif

.DEFAULT_GOAL := help

# =============================================================================
# Local development
# =============================================================================

## Install production dependencies
install:
	$(PIP) install -r requirements.txt

## Install development dependencies
install-dev: install
	$(PIP) install pytest pytest-cov black isort ruff mypy watchfiles

## Run the API server
run:
	cd src && uvicorn main:app --host 0.0.0.0 --port $(PORT)

## Run the API server with auto-reload
dev:
	cd src && uvicorn main:app --host 0.0.0.0 --port $(PORT) --reload

## Run tests (LLM-free; no API keys needed)
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

## Run linters
lint:
	ruff check src/
	mypy src/ --ignore-missing-imports

## Format code
format:
	black src/ tests/
	isort src/ tests/

# =============================================================================
# Docker (production image, same style as the sibling wisefood repos)
# =============================================================================

## Build the production image
build:
	$(DOCKER) build --target prod -t $(IMGTAG) .

## Push the image to the registry
push:
	$(DOCKER) push $(IMGTAG)

# =============================================================================
# Housekeeping
# =============================================================================

## Remove caches and build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov

help:
	@printf '%s\n' \
	  'Local development:' \
	  '  make install        Install dependencies' \
	  '  make install-dev    Install dev dependencies' \
	  '  make run            Run the API server' \
	  '  make dev            Run with auto-reload' \
	  '  make test           Run tests' \
	  '  make lint           Run linters' \
	  '  make format         Format code' \
	  '' \
	  'Docker:' \
	  '  make build          Build $(IMGTAG)' \
	  '  make push           Push $(IMGTAG)' \
	  '' \
	  'Housekeeping:' \
	  '  make clean          Remove caches and build artifacts'
