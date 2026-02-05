.PHONY: all install install-dev run dev test lint format clean help
.PHONY: docker-build docker-build-dev docker-build-prod push
.PHONY: docker-run docker-dev docker-prod docker-stop docker-logs docker-shell
.PHONY: docker-up docker-up-dev docker-up-prod docker-down

# =============================================================================
# Variables
# =============================================================================
PYTHON := python3
PIP := pip3
APP_NAME := foodchat
DOCKER_IMAGE := $(APP_NAME)
DOCKER_CONTAINER := $(APP_NAME)-api
PORT := 8000

# Load .env file if exists
ifneq (,$(wildcard ./.env))
	include .env
	export
endif

# Default target
.DEFAULT_GOAL := help

# =============================================================================
# Quick Start Commands
# =============================================================================

## Setup everything and run development server (first time setup)
all: install-dev dev

# =============================================================================
# Local Development
# =============================================================================

## Install production dependencies
install:
	$(PIP) install -r requirements.txt

## Install development dependencies
install-dev: install
	$(PIP) install pytest pytest-cov black isort ruff mypy watchfiles

## Run the API server (production mode)
run:
	cd src && uvicorn main:app --host 0.0.0.0 --port $(PORT)

## Run the API server with auto-reload (development mode)
dev:
	cd src && uvicorn main:app --host 0.0.0.0 --port $(PORT) --reload

## Run tests
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

## Run linter
lint:
	ruff check src/
	mypy src/ --ignore-missing-imports

## Format code
format:
	black src/
	isort src/

## Clean up cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# =============================================================================
# Docker Build Commands
# =============================================================================

## Build Docker image (default: production)
docker-build: docker-build-prod

## Build Docker development image
docker-build-dev:
	docker build --target dev -t $(DOCKER_IMAGE):dev .

## Build Docker production image
docker-build-prod:
	docker build --target prod -t $(DOCKER_IMAGE):prod .

## Push Docker image to wisefood/foodchat:latest
push: docker-build-prod
	docker tag $(DOCKER_IMAGE):prod wisefood/foodchat:latest
	docker push wisefood/foodchat:latest

# =============================================================================
# Docker Run Commands
# =============================================================================

## Run Docker container (default: production)
docker-run: docker-prod

## Run development container with hot reload and volume mounts
docker-dev: docker-stop
	docker run -d \
		--name $(DOCKER_CONTAINER)-dev \
		-p $(PORT):8000 \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/KG_neo4j:/app/KG_neo4j \
		--env-file .env \
		-e LOG_LEVEL=DEBUG \
		$(DOCKER_IMAGE):dev
	@echo "Development container started at http://localhost:$(PORT)"
	@echo "Hot reload enabled - edit src/ files and see changes immediately"

## Run production container
docker-prod: docker-stop
	docker run -d \
		--name $(DOCKER_CONTAINER) \
		-p $(PORT):8000 \
		--env-file .env \
		-e LOG_LEVEL=INFO \
		--restart unless-stopped \
		$(DOCKER_IMAGE):prod
	@echo "Production container started at http://localhost:$(PORT)"

## Stop and remove Docker container
docker-stop:
	docker stop $(DOCKER_CONTAINER) 2>/dev/null || true
	docker stop $(DOCKER_CONTAINER)-dev 2>/dev/null || true
	docker rm $(DOCKER_CONTAINER) 2>/dev/null || true
	docker rm $(DOCKER_CONTAINER)-dev 2>/dev/null || true

## View Docker logs
docker-logs:
	docker logs -f $(DOCKER_CONTAINER) 2>/dev/null || docker logs -f $(DOCKER_CONTAINER)-dev

## Open shell in running container
docker-shell:
	docker exec -it $(DOCKER_CONTAINER) /bin/bash 2>/dev/null || \
	docker exec -it $(DOCKER_CONTAINER)-dev /bin/bash

## Restart Docker container
docker-restart: docker-stop docker-run

# =============================================================================
# Docker Compose-style Commands
# =============================================================================

## Build and run (default: production)
docker-up: docker-up-prod

## Build and run development container
docker-up-dev: docker-build-dev docker-dev

## Build and run production container
docker-up-prod: docker-build-prod docker-prod

## Stop all containers and clean up
docker-down: docker-stop
	docker rmi $(DOCKER_IMAGE):dev 2>/dev/null || true
	docker rmi $(DOCKER_IMAGE):prod 2>/dev/null || true

# =============================================================================
# Help
# =============================================================================

## Show help
help:
	@echo "============================================================================="
	@echo "FoodChat API - Available commands"
	@echo "============================================================================="
	@echo ""
	@echo "Quick Start:"
	@echo "  make all            Install deps and run dev server (first time setup)"
	@echo "  make dev            Run the API server with auto-reload"
	@echo ""
	@echo "Local Development:"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo "  make run            Run the API server (production mode)"
	@echo "  make dev            Run the API server with auto-reload"
	@echo "  make test           Run tests"
	@echo "  make lint           Run linter"
	@echo "  make format         Format code"
	@echo "  make clean          Clean up cache files"
	@echo ""
	@echo "Docker Build:"
	@echo "  make docker-build       Build production image (default)"
	@echo "  make docker-build-dev   Build development image"
	@echo "  make docker-build-prod  Build production image"
	@echo "  make push               Build and push to wisefood/foodchat:latest"
	@echo ""
	@echo "Docker Run:"
	@echo "  make docker-dev     Run dev container (hot reload, volume mounts)"
	@echo "  make docker-prod    Run production container"
	@echo "  make docker-stop    Stop and remove containers"
	@echo "  make docker-logs    View container logs"
	@echo "  make docker-shell   Open shell in container"
	@echo "  make docker-restart Restart container"
	@echo ""
	@echo "Docker Compose-style:"
	@echo "  make docker-up      Build and run (production)"
	@echo "  make docker-up-dev  Build and run (development)"
	@echo "  make docker-up-prod Build and run (production)"
	@echo "  make docker-down    Stop all and remove images"
	@echo ""
	@echo "============================================================================="
	@echo "First Time Setup:"
	@echo "  1. cp .env.example .env && edit .env"
	@echo "  2. make all              # Local dev with hot reload"
	@echo "  -- OR --"
	@echo "  2. make docker-up-dev    # Docker dev with hot reload"
	@echo "============================================================================="
