# FoodChat API

Session-based meal planning chat API powered by RAG (Retrieval-Augmented Generation).

## Overview

FoodChat is a conversational meal planning assistant that generates personalized daily meal plans based on user dietary preferences, allergies, and nutritional goals. It integrates with the WiseFood platform for user profile management.

## Project Structure

```
foodchat/
├── src/
│   ├── main.py                 # FastAPI application entry point
│   ├── foodchat.py             # Core FoodChat class and RAG chains
│   ├── foodchat_init.py        # System initialization and config
│   ├── agents.py               # LLM agents (QueryClassifier, DocumentGrader, etc.)
│   ├── prompts.py              # System and user prompts
│   ├── schemas.py              # Pydantic schemas for LLM responses
│   ├── utils.py                # Utility functions and embeddings
│   ├── csv_processor.py        # CSV recipe data processor
│   ├── pdf_processor.py        # PDF recipe data processor
│   ├── multiple_evaluation.py  # Response evaluation
│   ├── ragas_eval.py           # RAG evaluation framework
│   ├── models/
│   │   ├── __init__.py
│   │   └── session.py          # Session, Message, MealPlan models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── session_service.py  # In-memory session management
│   │   ├── chat_service.py     # Chat orchestration and RAG flow
│   │   ├── profile_service.py  # WiseFood profile integration
│   │   └── wisefood_client.py  # WiseFood API client pool
│   └── routers/
│       ├── __init__.py
│       └── foodchat_router.py  # API endpoint definitions
├── KG_neo4j/                   # Knowledge graph queries
├── data/                       # Recipe datasets
├── Dockerfile
├── Makefile
├── requirements.txt
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/foodchat/sessions` | Create a new chat session |
| `GET` | `/foodchat/sessions/{id}` | Get session state |
| `DELETE` | `/foodchat/sessions/{id}` | Delete a session |
| `POST` | `/foodchat/sessions/{id}/messages` | Send message, get response |
| `GET` | `/foodchat/sessions/{id}/messages` | Get message history |
| `GET` | `/foodchat/sessions/{id}/meal-plans` | Get generated meal plans |
| `GET` | `/foodchat/health` | Health check |

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `WISEFOOD_API_URL` | WiseFood API base URL | Yes |
| `WISEFOOD_USERNAME` | WiseFood API username | Yes |
| `WISEFOOD_PASSWORD` | WiseFood API password | Yes |
| `DATASET` | Dataset to use (`hummus` or `culinary`) | No (default: `hummus`) |
| `MODEL` | LLM model name | No (default: `Llama_FoodChat`) |
| `DATA_TYPE` | Data source type (`csv` or `pdf`) | No (default: `csv`) |
| `EMBEDDINGS` | Embedding model | No (default: `nomic-embed-text:latest`) |
| `VECTORSTORE` | Vector store type | No (default: `chroma`) |
| `MAX_RETRIEVAL` | Max documents to retrieve | No (default: `3`) |

## Quick Start

### Local Development

```bash
# Install dependencies
make install

# Set environment variables
export WISEFOOD_API_URL="https://api.wisefood.com/rest"
export WISEFOOD_USERNAME="your-username"
export WISEFOOD_PASSWORD="your-password"

# Run the API
make run
```

### Docker

```bash
# Build the image
make docker-build

# Run the container
make docker-run
```

### Using the API

```bash
# Create a session
curl -X POST http://localhost:8000/foodchat/sessions \
  -H "Content-Type: application/json" \
  -d '{"member_id": "member-123"}'

# Send a message
curl -X POST http://localhost:8000/foodchat/sessions/{session_id}/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "I want a healthy meal plan for tomorrow"}'

# Get meal plans
curl http://localhost:8000/foodchat/sessions/{session_id}/meal-plans
```

## Session Flow

1. **Create Session**: Client creates a session with a `member_id`. The API fetches the user's profile from WiseFood.

2. **Send Messages**: Client sends messages to the session. The API:
   - Routes queries (meal planning vs general chat)
   - May ask clarifying questions (`needs_clarification: true`)
   - Returns meal plan recommendations

3. **Clarification Flow**: If the API needs more information:
   - Response includes `needs_clarification: true`
   - Client sends follow-up message with the answer
   - Process continues until clarification is complete

4. **Retrieve Meal Plans**: Client can fetch all generated meal plans for a session.

## Development

```bash
# Install dev dependencies
make install-dev

# Run linter
make lint

# Run tests
make test

# Format code
make format
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
├─────────────────────────────────────────────────────────────┤
│  Routers                                                     │
│  └── foodchat_router.py (REST endpoints)                    │
├─────────────────────────────────────────────────────────────┤
│  Services                                                    │
│  ├── ChatService (orchestrates RAG flow)                    │
│  ├── SessionService (in-memory session storage)             │
│  └── ProfileService (WiseFood integration)                  │
├─────────────────────────────────────────────────────────────┤
│  Core                                                        │
│  ├── FoodChat (RAG chains)                                  │
│  ├── Agents (QueryClassifier, DocumentGrader, etc.)         │
│  └── Retriever (vector/keyword search)                      │
├─────────────────────────────────────────────────────────────┤
│  External                                                    │
│  ├── WiseFood API (user profiles)                           │
│  ├── Ollama (LLM inference)                                 │
│  ├── ChromaDB (vector store)                                │
│  └── Neo4j (knowledge graph)                                │
└─────────────────────────────────────────────────────────────┘
```

## License

See [LICENSE](LICENSE) file.
