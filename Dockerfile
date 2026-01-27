FROM python:3.11-slim

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

# Copy application code
COPY src/ ./src/
COPY KG_neo4j/ ./KG_neo4j/
COPY data/ ./data/

# Set working directory to src
WORKDIR /app/src

# Expose port
EXPOSE 8000

# Environment variables (override at runtime)
ENV WISEFOOD_API_URL=""
ENV WISEFOOD_USERNAME=""
ENV WISEFOOD_PASSWORD=""
ENV DATASET="hummus"
ENV MODEL="Llama_FoodChat"
ENV DATA_TYPE="csv"
ENV EMBEDDINGS="nomic-embed-text:latest"
ENV VECTORSTORE="chroma"
ENV MAX_RETRIEVAL="3"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/foodchat/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
