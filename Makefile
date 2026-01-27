.PHONY: install install-dev run dev test lint format clean docker-build docker-run docker-stop help

# Variables
PYTHON := python3
PIP := pip3
APP_NAME := foodchat
DOCKER_IMAGE := $(APP_NAME):latest
DOCKER_CONTAINER := $(APP_NAME)-api
PORT := 8000

# Default target
.DEFAULT_GOAL := help

## Install production dependencies
install:
	$(PIP) install -r requirements.txt

## Install development dependencies
install-dev: install
	$(PIP) install pytest pytest-cov black isort ruff mypy

## Run the API server
run:
	cd src && uvicorn main:app --host 0.0.0.0 --port $(PORT)

## Run the API server with auto-reload (development)
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

## Build Docker image
docker-build:
	docker build -t $(DOCKER_IMAGE) .

## Run Docker container
docker-run:
	docker run -d \
		--name $(DOCKER_CONTAINER) \
		-p $(PORT):8000 \
		-e WISEFOOD_API_URL=$${WISEFOOD_API_URL} \
		-e WISEFOOD_USERNAME=$${WISEFOOD_USERNAME} \
		-e WISEFOOD_PASSWORD=$${WISEFOOD_PASSWORD} \
		-e DATASET=$${DATASET:-hummus} \
		-e MODEL=$${MODEL:-Llama_FoodChat} \
		$(DOCKER_IMAGE)

## Stop and remove Docker container
docker-stop:
	docker stop $(DOCKER_CONTAINER) 2>/dev/null || true
	docker rm $(DOCKER_CONTAINER) 2>/dev/null || true

## View Docker logs
docker-logs:
	docker logs -f $(DOCKER_CONTAINER)

## Restart Docker container
docker-restart: docker-stop docker-run

## Build and run Docker
docker-up: docker-build docker-run

## Show help
help:
	@echo "FoodChat API - Available commands:"
	@echo ""
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo "  make run            Run the API server"
	@echo "  make dev            Run the API server with auto-reload"
	@echo "  make test           Run tests"
	@echo "  make lint           Run linter"
	@echo "  make format         Format code"
	@echo "  make clean          Clean up cache files"
	@echo ""
	@echo "Docker commands:"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-run     Run Docker container"
	@echo "  make docker-stop    Stop and remove Docker container"
	@echo "  make docker-logs    View Docker logs"
	@echo "  make docker-restart Restart Docker container"
	@echo "  make docker-up      Build and run Docker"
	@echo ""
	@echo "Environment variables required:"
	@echo "  WISEFOOD_API_URL    WiseFood API base URL"
	@echo "  WISEFOOD_USERNAME   WiseFood API username"
	@echo "  WISEFOOD_PASSWORD   WiseFood API password"
