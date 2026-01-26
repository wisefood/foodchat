# FoodChat Makefile
# Build, test, and deploy automation

# ==============================================================================
# Configuration
# ==============================================================================
IMAGE_NAME ?= foodchat
IMAGE_TAG ?= latest
REGISTRY ?= docker.io
REGISTRY_USER ?= $(shell whoami)
FULL_IMAGE_NAME = $(REGISTRY)/$(REGISTRY_USER)/$(IMAGE_NAME):$(IMAGE_TAG)

# Container runtime (docker or podman)
CONTAINER_RUNTIME ?= docker

# Application settings
PORT ?= 8000
HOST ?= 0.0.0.0

# ==============================================================================
# Help
# ==============================================================================
.PHONY: help
help: ## Show this help message
	@echo "FoodChat Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ==============================================================================
# Development
# ==============================================================================
.PHONY: install
install: ## Install dependencies
	pip install -r requirements.txt
	pip install uvicorn[standard] fastapi python-dotenv

.PHONY: install-dev
install-dev: install ## Install development dependencies
	pip install pytest pytest-cov black isort flake8 mypy

.PHONY: run
run: ## Run the API locally
	python -m uvicorn foodchat_api:app --host $(HOST) --port $(PORT) --reload

.PHONY: run-gradio
run-gradio: ## Run the Gradio UI locally
	python main.py

.PHONY: lint
lint: ## Run linters
	black --check .
	isort --check-only .
	flake8 .

.PHONY: format
format: ## Format code
	black .
	isort .

.PHONY: test
test: ## Run tests
	pytest -v

.PHONY: test-cov
test-cov: ## Run tests with coverage
	pytest -v --cov=. --cov-report=html --cov-report=term

# ==============================================================================
# Docker Build
# ==============================================================================
.PHONY: build
build: ## Build Docker image
	$(CONTAINER_RUNTIME) build -t $(IMAGE_NAME):$(IMAGE_TAG) .

.PHONY: build-no-cache
build-no-cache: ## Build Docker image without cache
	$(CONTAINER_RUNTIME) build --no-cache -t $(IMAGE_NAME):$(IMAGE_TAG) .

.PHONY: tag
tag: ## Tag image for registry
	$(CONTAINER_RUNTIME) tag $(IMAGE_NAME):$(IMAGE_TAG) $(FULL_IMAGE_NAME)

# ==============================================================================
# Docker Registry
# ==============================================================================
.PHONY: login
login: ## Login to container registry
	$(CONTAINER_RUNTIME) login $(REGISTRY)

.PHONY: push
push: tag ## Push image to registry
	$(CONTAINER_RUNTIME) push $(FULL_IMAGE_NAME)

.PHONY: pull
pull: ## Pull image from registry
	$(CONTAINER_RUNTIME) pull $(FULL_IMAGE_NAME)

# ==============================================================================
# Docker Run
# ==============================================================================
.PHONY: docker-run
docker-run: ## Run container locally
	$(CONTAINER_RUNTIME) run -it --rm \
		-p $(PORT):$(PORT) \
		--env-file .env \
		--name $(IMAGE_NAME) \
		$(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: docker-run-detached
docker-run-detached: ## Run container in background
	$(CONTAINER_RUNTIME) run -d \
		-p $(PORT):$(PORT) \
		--env-file .env \
		--name $(IMAGE_NAME) \
		--restart unless-stopped \
		$(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: docker-stop
docker-stop: ## Stop running container
	$(CONTAINER_RUNTIME) stop $(IMAGE_NAME) || true
	$(CONTAINER_RUNTIME) rm $(IMAGE_NAME) || true

.PHONY: docker-logs
docker-logs: ## Show container logs
	$(CONTAINER_RUNTIME) logs -f $(IMAGE_NAME)

.PHONY: docker-shell
docker-shell: ## Open shell in running container
	$(CONTAINER_RUNTIME) exec -it $(IMAGE_NAME) /bin/bash

# ==============================================================================
# Docker Compose
# ==============================================================================
.PHONY: up
up: ## Start all services with docker-compose
	$(CONTAINER_RUNTIME)-compose up -d

.PHONY: down
down: ## Stop all services
	$(CONTAINER_RUNTIME)-compose down

.PHONY: logs
logs: ## Show logs for all services
	$(CONTAINER_RUNTIME)-compose logs -f

# ==============================================================================
# Cleanup
# ==============================================================================
.PHONY: clean
clean: ## Clean up build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage 2>/dev/null || true

.PHONY: clean-docker
clean-docker: ## Remove Docker image
	$(CONTAINER_RUNTIME) rmi $(IMAGE_NAME):$(IMAGE_TAG) || true
	$(CONTAINER_RUNTIME) rmi $(FULL_IMAGE_NAME) || true

.PHONY: clean-all
clean-all: clean clean-docker ## Clean everything

# ==============================================================================
# Release
# ==============================================================================
.PHONY: release
release: build push ## Build and push to registry
	@echo "Released $(FULL_IMAGE_NAME)"

.PHONY: release-version
release-version: ## Release with version tag (usage: make release-version VERSION=1.0.0)
ifndef VERSION
	$(error VERSION is not set. Usage: make release-version VERSION=1.0.0)
endif
	$(CONTAINER_RUNTIME) build -t $(IMAGE_NAME):$(VERSION) .
	$(CONTAINER_RUNTIME) tag $(IMAGE_NAME):$(VERSION) $(REGISTRY)/$(REGISTRY_USER)/$(IMAGE_NAME):$(VERSION)
	$(CONTAINER_RUNTIME) push $(REGISTRY)/$(REGISTRY_USER)/$(IMAGE_NAME):$(VERSION)
	@echo "Released $(REGISTRY)/$(REGISTRY_USER)/$(IMAGE_NAME):$(VERSION)"

# ==============================================================================
# Info
# ==============================================================================
.PHONY: info
info: ## Show build information
	@echo "Image Name:      $(IMAGE_NAME)"
	@echo "Image Tag:       $(IMAGE_TAG)"
	@echo "Registry:        $(REGISTRY)"
	@echo "Registry User:   $(REGISTRY_USER)"
	@echo "Full Image Name: $(FULL_IMAGE_NAME)"
	@echo "Container Runtime: $(CONTAINER_RUNTIME)"

.DEFAULT_GOAL := help
