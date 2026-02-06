# =============================================================================
# FoodChat API - Multi-stage Dockerfile
# =============================================================================
# Build: docker build -t foodchat .
# Dev:   docker build --target dev -t foodchat:dev .
# Prod:  docker build --target prod -t foodchat:prod .
# =============================================================================

# -----------------------------------------------------------------------------
# Base stage - common dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Development stage - with hot reload and dev tools
# -----------------------------------------------------------------------------
FROM base AS dev

# Install dev dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    isort \
    ruff \
    mypy \
    watchfiles

# Copy application code
COPY src/ ./src/
COPY KG_neo4j/ ./KG_neo4j/

# Set working directory to src
WORKDIR /app/src

# Expose port
EXPOSE 8000

# Environment defaults for development
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8000
ENV LOG_LEVEL=DEBUG
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Run with hot reload for development
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# -----------------------------------------------------------------------------
# Production stage - optimized for deployment
# -----------------------------------------------------------------------------
FROM base AS prod

# Create non-root user for security
RUN groupadd -r foodchat && useradd -r -g foodchat foodchat

# Copy application code
COPY src/ ./src/
COPY KG_neo4j/ ./KG_neo4j/

# Create data directory (mount volume in prod)
RUN mkdir -p ./data

# Set ownership
RUN chown -R foodchat:foodchat /app

# Switch to non-root user
USER foodchat

# Set working directory to src
WORKDIR /app/src

# Expose port
EXPOSE 8000

# Environment defaults for production
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8000
ENV LOG_LEVEL=INFO
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/foodchat/health || exit 1

# Run with multiple workers for production
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
