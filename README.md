# FoodChat

Conversational meal-planning service of the WiseFood platform. Creates and
refines personalized daily/weekly meal plans through natural-language chat,
grounded in the member's profile (diet, allergies, likes/dislikes) and the
RecipeWrangler recipe knowledge graph.

FoodChat runs behind the **wisefood-api gateway**, which authenticates users
(Keycloak) and passes the household member's `member_id` as data — FoodChat
itself is network-internal and trusts the gateway (see *Security model* below).

## Quick start

```bash
make install
cp .env.example .env        # fill in GROQ_API_KEY + WiseFood credentials
make run                    # uvicorn on :8000

# Smoke test
curl -X POST http://localhost:8000/foodchat/sessions \
  -H "Content-Type: application/json" -d '{"member_id": "member-123"}'

curl -X POST http://localhost:8000/foodchat/sessions/{session_id}/chat \
  -H "Content-Type: application/json" \
  -d '{"member_id": "member-123", "content": "I want a healthy meal plan for tomorrow"}'
```

Docker: `make docker-build && make docker-run`. The service needs **no data
files, vector stores, or embedding models** — external dependencies are the
Groq API, RecipeWrangler, the WiseFood API, and a `DATABASE_URL`.

## API

All conversational traffic goes through the unified chat endpoint; intent is
classified server-side (daily plan / weekly plan / refinement / plan-type
switch / small talk).

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/foodchat/sessions` | Create session (fetches member profile) |
| GET/DELETE | `/foodchat/sessions/{id}` | Session metadata / delete (owner only) |
| GET | `/foodchat/members/{member_id}/sessions` | All sessions of a member |
| POST | `/foodchat/sessions/{id}/chat` | **Unified chat** — any message |
| GET | `/foodchat/sessions/{id}/conversation` | Paginated history (cursor) |
| GET | `/foodchat/sessions/{id}/meal-plans[/current\|/history]` | Daily plan canvas |
| GET | `/foodchat/sessions/{id}/weekly-meal-plans[/current\|/history]` | Weekly plan canvas |
| POST | `/foodchat/sessions/{id}/messages/{mid}/feedback` | 👍/👎 on assistant messages |
| GET | `/foodchat/health` | Health check |

Full message-flow diagrams: [CHAT_ENDPOINT_PIPELINE.md](CHAT_ENDPOINT_PIPELINE.md).
Milestone history and handoff notes: [CHANGES.md](CHANGES.md).

## Architecture

```
POST /sessions/{id}/chat
        │
        ▼
OrchestratorService ──── OrchestratorAgent (ONE intent classification per turn)
        │
        ├─ daily_plan / refine_plan ──► ChatService
        │       │ ClarificationManager  (persisted state machine — asks only
        │       │                        genuinely missing info, restart-safe)
        │       │ PlanningPipeline      (RecipeWrangler candidates → LLM-graded
        │       │                        breakfast×lunch×dinner combinations)
        │       └ quality metrics       (variety / diversity / guideline scores)
        │
        ├─ weekly_plan / refine ─────► WeeklyPlanService (7-day MDP over
        │                              per-day candidate pools)
        ├─ switch_plan_type ─────────► freeze canvas, start the other type
        └─ chat ─────────────────────► SimpleChatBot (per-session history)
        │
        ▼
SessionService ──► SQLite/PostgreSQL (sessions, messages, versioned plans,
                                      feedback, clarification state)
```

Key modules (each carries a docstring with its contracts):

| Module | Role |
|---|---|
| `src/routers/foodchat_router.py` | HTTP surface + member-ownership guards |
| `src/services/orchestrator_service.py` | Intent routing (single router) |
| `src/services/chat_service.py` | Daily-plan flows + metrics |
| `src/services/clarification.py` | Restart-safe clarification state machine |
| `src/services/planning_pipeline.py` | Candidates → graded plans |
| `src/services/candidates_client.py` | RecipeWrangler HTTP client (only recipe source) |
| `src/services/weekly_plan_service.py` + `weekly_planner/` | 7-day planning MDP |
| `src/services/session_service.py` + `src/db.py` | Persistence (canvas versioning) |
| `src/services/profile_service.py` + `src/backend/platform.py` | WiseFood profiles |
| `src/agents.py` + `src/prompts.py` + `src/schemas.py` | LLM agents / prompts / output schemas |

**Plan canvases:** each plan type (daily/weekly) is a versioned lineage —
refinements create `version+1` with `parent_id`, switching types freezes the
old canvas without deleting it, and full history stays retrievable.

## Security model

FoodChat performs **no token authentication** — it must only be reachable from
the gateway network. Every session-scoped endpoint requires the owning
`member_id` and returns 404 (never 403) on mismatch. If you expose FoodChat
directly, `member_id` is forgeable; don't.

## Development

```bash
make install-dev
make test          # pytest — unit tests are LLM-free (fakes), no API keys needed
make lint && make format
```

Engineering standards are binding for contributions: every module documents
its role and contracts, cross-service payloads are typed Pydantic models, unit
tests never call an LLM, and removed endpoints are cleaned across service →
gateway → UI in the same change. Changes that alter message flow must update
`CHAT_ENDPOINT_PIPELINE.md`; every milestone appends a handoff entry to
[CHANGES.md](CHANGES.md).
