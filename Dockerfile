# FoodChat API Dockerfile
# Multi-stage build for optimized image size

# ==============================================================================
# Stage 1: Builder
# ==============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir uvicorn[standard] fastapi

# ==============================================================================
# Stage 2: Runtime
# ==============================================================================
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY config/ ./config/
COPY KG_neo4j/ ./KG_neo4j/
COPY data/ ./data/
COPY *.py ./
COPY prompts.py schemas.py ./

# Create necessary directories
RUN mkdir -p /app/VECTORSTORE /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables with defaults
ENV HOST=0.0.0.0 \
    PORT=8000 \
    DEBUG=false \
    WORKERS=1 \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the API
CMD ["python", "-m", "uvicorn", "foodchat_api:app", "--host", "0.0.0.0", "--port", "8000"]
