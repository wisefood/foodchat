# FoodChat

AI-powered meal planning

## Quick Start

```bash
# Copy environment config
cp .env.example .env

# Install dependencies
make install

# Run locally
make run          # API on port 8000
make run-gradio   # Gradio UI on port 7860
```

## Docker

```bash
# Build and run
make build
make docker-run

# Or use docker-compose (includes Neo4j)
make up
```

## Configuration

All settings are configured via environment variables. Copy `.env.example` to `.env` and customize.

### Server
| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | 0.0.0.0 | API host |
| `PORT` | 8000 | API port |
| `DEBUG` | false | Debug mode |
| `WORKERS` | 1 | Uvicorn workers |

### LLM (Ollama)
| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | http://localhost:11434 | Ollama API URL |
| `LLM_MODEL` | llama3.2:latest | Main chat model |
| `LLM_TEMPERATURE` | 0.0 | LLM temperature |
| `ROUTER_MODEL` | llama3.2 | Query router model |
| `GRADER_MODEL` | llama3.2 | Document grader model |
| `CHATBOT_MODEL` | llama3.2:latest | Simple chatbot model |
| `CHATBOT_TEMPERATURE` | 0.7 | Chatbot temperature |
| `EVALUATOR_MODEL` | llama3.2:3b-instruct-q8_0 | Evaluator model |

### Embeddings
| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | nomic-embed-text:latest | Ollama embedding model |
| `HUGGINGFACE_EMBEDDING_MODEL` | davanstrien/autotrain-recipes-2451975973 | HF model |

### Retrieval
| Variable | Default | Description |
|----------|---------|-------------|
| `RETRIEVAL_METHOD` | similarity | Method: similarity, mmr, similarity_score_threshold |
| `MAX_RETRIEVAL` | 3 | Max documents to retrieve |
| `QUERY_METHOD` | standard | Query method: standard, multiquery |
| `MMR_LAMBDA` | 0.5 | MMR lambda (0-1) |
| `HYBRID_SEARCH_ENABLED` | false | Enable hybrid search |
| `KEYWORD_SEARCH_WEIGHT` | 0.5 | BM25 weight |
| `VECTOR_SEARCH_WEIGHT` | 0.5 | Vector search weight |

### Data
| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_TYPE` | csv | Data source: csv, pdf |
| `DATASET` | hummus | Dataset: hummus, culinary |
| `VECTORSTORE_TYPE` | chroma | Vectorstore: chroma, faiss, milvus |
| `PDF_SOURCE_NAME` | 20_International_Recipes | PDF filename (no ext) |
| `CHUNK_SIZE` | 1000 | PDF chunk size |
| `CHUNK_OVERLAP` | 100 | PDF chunk overlap |

### Neo4j
| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | bolt://localhost:7687 | Neo4j connection URI |
| `NEO4J_USERNAME` | neo4j | Neo4j username |
| `NEO4J_PASSWORD` | password | Neo4j password |
| `NEO4J_DATABASE` | neo4j | Neo4j database |

### RAG
| Variable | Default | Description |
|----------|---------|-------------|
| `ADAPTIVE_RAG_ENABLED` | false | Enable adaptive RAG |
| `MAX_RAG_ITERATIONS` | 3 | Max RAG iterations |

### Gradio UI
| Variable | Default | Description |
|----------|---------|-------------|
| `GRADIO_SERVER_NAME` | 0.0.0.0 | Gradio host |
| `GRADIO_SERVER_PORT` | 7860 | Gradio port |
| `GRADIO_SHARE` | false | Enable public sharing |

### Logging
| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | INFO | Log level |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/chat` | POST | Chat with FoodChat |

## Makefile Targets

```bash
make help         # Show all targets
make build        # Build Docker image
make push         # Push to registry
make release      # Build and push
make clean        # Clean build artifacts
```

## Project Structure

```
foodchat/
├── config/           # Centralized configuration
├── KG_neo4j/         # Neo4j knowledge graph
├── data/             # Recipe datasets
├── Dockerfile        # Container build
├── docker-compose.yml
├── Makefile
└── foodchat_api.py   # API entry point
```
