# FoodChat

A conversational meal planning assistant that creates personalized daily meal plans based on your dietary preferences, allergies, and goals. Built with RAG (Retrieval-Augmented Generation) and integrated with the WiseFood platform.

## Quick Start

```bash
# Install and run locally
make install
export WISEFOOD_API_URL="https://wisefood.gr/rest"
export WISEFOOD_USERNAME="your-username"
export WISEFOOD_PASSWORD="your-password"
make run

# Or use Docker
make docker-build && make docker-run
```

## How It Works

1. Create a session with your `member_id` - the API fetches your profile from WiseFood
2. Chat naturally about what you want to eat
3. Answer any clarifying questions the assistant asks
4. Get personalized meal plan recommendations

```bash
# Example usage
curl -X POST http://localhost:8000/foodchat/sessions \
  -H "Content-Type: application/json" \
  -d '{"member_id": "member-123"}'

curl -X POST http://localhost:8000/foodchat/sessions/{session_id}/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "I want a healthy meal plan for tomorrow"}'
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/foodchat/sessions` | Create session |
| GET | `/foodchat/sessions/{id}` | Get session |
| DELETE | `/foodchat/sessions/{id}` | Delete session |
| POST | `/foodchat/sessions/{id}/messages` | Send message |
| GET | `/foodchat/sessions/{id}/messages` | Get history |
| GET | `/foodchat/sessions/{id}/meal-plans` | Get meal plans |

## Configuration

Required:
- `WISEFOOD_API_URL` - WiseFood API base URL
- `WISEFOOD_USERNAME` / `WISEFOOD_PASSWORD` - API credentials

Optional:
- `DATASET` - `hummus` (default) or `culinary`
- `MODEL` - LLM model (default: `Llama_FoodChat`)
- `DATA_TYPE` - `csv` (default) or `pdf`
- `MAX_RETRIEVAL` - Documents to retrieve (default: 3)

## Development

```bash
make install-dev  # Dev dependencies
make test         # Run tests
make lint         # Lint code
make format       # Format code
```

## Project Structure

```
src/
├── main.py              # FastAPI entry point
├── foodchat.py          # Core FoodChat class and RAG chains
├── agents.py            # LLM agents (QueryClassifier, DocumentGrader, etc.)
├── prompts.py           # System and user prompts
├── schemas.py           # Pydantic schemas
├── csv_processor.py     # CSV recipe processor
├── pdf_processor.py     # PDF recipe processor
├── models/
│   └── session.py       # Session, Message, MealPlan models
├── services/
│   ├── session_service.py   # In-memory session management
│   ├── chat_service.py      # Chat orchestration and RAG flow
│   └── profile_service.py   # WiseFood profile integration
└── routers/
    └── foodchat_router.py   # API endpoints
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
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
