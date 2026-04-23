# FoodChat — Canvas & Version History Refactor

> **Date:** 2026-04-22  
> **Branch:** main  
> Prepared for handoff. All changes are backward-compatible with existing sessions
> (the DB migration is idempotent and adds columns with safe defaults).

---

## What changed and why

### Problem being solved

The `/chat` endpoint existed but the UX was flat: every refinement produced a
disconnected plan, switching between daily and weekly would silently overwrite the
previous context, and there was no way to retrieve older versions of a plan.

The changes below turn each plan type into a **canvas** — a live, versioned document
the user edits with natural language. Old versions are always retrievable. Switching
plan types (daily ↔ weekly) is a first-class operation that preserves both canvases
independently.

---

## Files changed

### `src/models/session.py`

| What | Detail |
|------|--------|
| `MealPlan.version` | Integer, starts at 1. Incremented by 1 on each refinement. |
| `MealPlan.parent_id` | UUID of the previous plan version. `None` for v1. |
| `WeeklyMealPlan.version` / `parent_id` | Same fields added to the weekly plan model. |
| `PlanCanvas` (new dataclass) | Tracks `plan_type`, `current_id` (latest version shown), and `root_id` (first version in lineage). |
| `Session.daily_canvas` | Replaces the old `active_context`. Tracks the live daily canvas. |
| `Session.weekly_canvas` | Independent canvas for the weekly plan. |
| `Session.active_canvas` (property) | Returns whichever canvas was most recently updated — used by the orchestrator for `refine_plan` routing. |
| `Session.get_current_daily_plan()` | Returns the `MealPlan` object pointed to by `daily_canvas.current_id`. |
| `Session.get_current_weekly_plan()` | Same for weekly. |
| `Session.active_context` | **Backward-compat shim** — returns a duck-typed object so any code still reading `.active_context.plan_type` keeps working. |
| `ActiveContext` | **Removed** — replaced by `PlanCanvas`. The shim above covers callers. |

---

### `src/db.py`

| What | Detail |
|------|--------|
| `SessionRow.daily_canvas` | New TEXT column (JSON blob). |
| `SessionRow.weekly_canvas` | New TEXT column (JSON blob). |
| `SessionRow.active_context` | **Left in place but ignored** — not dropped to avoid a destructive migration. |
| `MealPlanRow.version` | New INTEGER column, `DEFAULT 1`. |
| `MealPlanRow.parent_id` | New TEXT column, nullable. |
| `db_update_canvases()` | New helper — persists both canvas blobs in one DB write. |
| `db_get_plan_lineage()` | New helper — returns all versions in a plan lineage ordered oldest-first. Uses a recursive CTE (SQLite ≥ 3.35 / PostgreSQL); falls back to a Python-side walk on older SQLite. |
| `db_update_active_context()` | **Deprecated no-op stub** — kept so any leftover callers don't crash. |
| `init_db()` → `_migrate_existing_db()` | Called at startup. Adds missing columns to existing databases without touching data. Fully idempotent. |
| `db_save_meal_plan()` | Updated signature: now accepts `version` and `parent_id`. |

---

### `src/services/session_service.py`

| What | Detail |
|------|--------|
| `add_meal_plan()` | Creates a v1 daily plan and opens a fresh `daily_canvas`. |
| `refine_meal_plan()` (new) | Creates a new `MealPlan` with `version = parent.version + 1` and `parent_id = parent.id`. Advances `daily_canvas.current_id`; `root_id` stays fixed. |
| `add_weekly_meal_plan()` | Creates a v1 weekly plan and opens a fresh `weekly_canvas`. |
| `refine_weekly_meal_plan()` (new) | Same versioning logic for weekly plans. |
| `get_daily_plan_history()` (new) | Returns all `MealPlan` objects for the session, sorted oldest-first. |
| `get_weekly_plan_history()` (new) | Same for weekly. |
| `_persist_canvases()` | Internal helper — writes both canvas blobs to DB after every plan mutation. |
| `_load_from_db()` | Updated to restore `daily_canvas` and `weekly_canvas` from the new columns, and to deserialise `version`/`parent_id` from plan payloads. |
| `_serialize_meal_plan()` / `_deserialize_meal_plan()` | Updated to include `version` and `parent_id` fields. |

---

### `src/schemas.py`

| What | Detail |
|------|--------|
| `OrchestratorSchema.intent` | Added `"switch_plan_type"` as a valid literal. |
| `OrchestratorSchema.target_plan_type` | New optional field — only populated when intent is `switch_plan_type`. Value is `"daily"` or `"weekly"`. |

---

### `src/prompts.py`

| What | Detail |
|------|--------|
| `ORCHESTRATOR_SYSTEM_INSTRUCTIONS` | Added the `switch_plan_type` intent with a clear definition and examples. Updated rules section. Added `target_plan_type` to the output format specification. |

---

### `src/agents.py` — `OrchestratorAgent.classify()`

| What | Detail |
|------|--------|
| Return type | Changed from `str` to `dict`: `{"intent": str, "target_plan_type": str | None}`. |
| Valid intents | Now accepts `"switch_plan_type"` in addition to the original four. |
| `target_plan_type` | Extracted from the LLM response and returned only when intent is `switch_plan_type`. |

---

### `src/services/orchestrator_service.py`

| What | Detail |
|------|--------|
| `ChatTurn.plan_version` | New field — version number of the plan just produced. |
| `ChatTurn.plan_parent_id` | New field — parent plan ID (for the UI to detect refinements). |
| `process()` | Updated to unpack the new `dict` from `OrchestratorAgent.classify()`. Routing now passes `is_refinement=True/False` to sub-services. |
| `_handle_chat()` | Now passes `is_refinement` to `ChatService.process_message()`. |
| `_handle_weekly()` | Now passes `is_refinement` to `WeeklyPlanService.process_message()`. |
| `_handle_switch()` (new) | Handles `switch_plan_type` intent. Sends an acknowledgement message, then routes to the target service as a fresh plan. The old canvas is preserved untouched in memory and DB. |

---

### `src/services/chat_service.py`

| What | Detail |
|------|--------|
| `process_message(is_refinement=False)` | New parameter. When `True` and a `daily_canvas` exists, the current plan is serialised as a text block and prepended to the user message before the RAG chain runs. This gives the LLM full context of what it is being asked to change. |
| `_run_post_clarification(is_refinement=False)` | Calls `session_service.refine_meal_plan()` instead of `add_meal_plan()` when `is_refinement=True`. |
| Response text | Version label appended: `"Here is your meal plan for today (version 3):"` on refinements. |

---

### `src/services/weekly_plan_service.py`

| What | Detail |
|------|--------|
| `process_message(is_refinement=False)` | Same pattern as `ChatService`. When refining, the current weekly canvas plan is serialised and prepended to the query. |
| Calls `refine_weekly_meal_plan()` vs `add_weekly_meal_plan()` | Based on `is_refinement`. |
| Response text | Includes version number on refinements. |

---

### `src/routers/foodchat_router.py`

| What | Detail |
|------|--------|
| `MealPlanResponse.version` / `parent_id` | New fields in the daily plan response model. |
| `WeeklyMealPlanResponse.version` / `parent_id` | Same for weekly. |
| `ChatTurnResponse.plan_version` / `plan_parent_id` | New fields — the UI uses these to know whether a canvas was just updated vs. a fresh plan created. |
| `GET /sessions/{id}/meal-plans/current` | **New endpoint** — returns only the latest daily plan on the canvas. Requires `member_id` query param. |
| `GET /sessions/{id}/meal-plans/history` | **New endpoint** — returns all daily plan versions ordered oldest-first. Requires `member_id`. |
| `GET /sessions/{id}/weekly-meal-plans/current` | **New endpoint** — returns only the latest weekly plan on the canvas. |
| `GET /sessions/{id}/weekly-meal-plans/history` | **New endpoint** — returns all weekly plan versions ordered oldest-first. |
| Existing `/meal-plans` and `/weekly-meal-plans` endpoints | **Unchanged** — still return all plans (same as history). |

---

## New API endpoints summary

```
GET  /foodchat/sessions/{session_id}/meal-plans/current
     ?member_id=<id>
     → MealPlanResponse | null

GET  /foodchat/sessions/{session_id}/meal-plans/history
     ?member_id=<id>
     → List[MealPlanResponse]   (version, parent_id fields populated)

GET  /foodchat/sessions/{session_id}/weekly-meal-plans/current
     ?member_id=<id>
     → WeeklyMealPlanResponse | null

GET  /foodchat/sessions/{session_id}/weekly-meal-plans/history
     ?member_id=<id>
     → List[WeeklyMealPlanResponse]
```

---

## New intent: `switch_plan_type`

Triggered when the user says things like:
- *"Forget the daily plan, let's do a weekly one instead"*
- *"Actually, never mind the week — just give me today"*
- *"Switch to a weekly plan"*

**Behaviour:**
1. The orchestrator classifies the message as `switch_plan_type` and sets `target_plan_type`.
2. `OrchestratorService._handle_switch()` sends an acknowledgement assistant message.
3. It then calls the target service as a **fresh plan** (not a refinement).
4. Both canvases remain in memory and DB — the user can ask for history of either at any time.

---

## Database migration

`init_db()` (called at startup) runs `_migrate_existing_db()` which:
- Adds `daily_canvas TEXT` and `weekly_canvas TEXT` columns to `sessions` if missing.
- Adds `version INTEGER NOT NULL DEFAULT 1` and `parent_id TEXT` columns to `meal_plans` if missing.
- Is **fully idempotent** — safe to run multiple times, never drops data.

Existing rows get `version = 1` and `parent_id = NULL` automatically via the column defaults.

---

## Deployment checklist

- [ ] Deploy new code (no environment variable changes required)
- [ ] Restart the service — `init_db()` will auto-migrate the existing `foodchat.db`
- [ ] Verify with `GET /foodchat/health` → `{"status": "ok"}`
- [ ] Smoke test: create a session, send a daily plan request, refine it, check that `/meal-plans/history` returns 2 entries with `version=1` and `version=2`
- [ ] Smoke test: say "forget the daily plan, let's do a weekly one" — confirm `intent=switch_plan_type` in the response and a weekly plan is returned
