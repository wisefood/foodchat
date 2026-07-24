# Dynamic meal generation — design & implementation plan

**Status: proposed (2026-07-24). Not started.** Supersedes the "Multi-plate
meals & dynamic meal structure" and "N-day plan horizon" entries in IDEAS.md.

Scope: let a plan have **a variable number of meals per day** (not just the
fixed breakfast/lunch/dinner) and **more than one plate per meal** ("pasta
and a salad for lunch"), on **both the daily and weekly horizons**, and let
the daily horizon span **N days** ("plan meals for 2 days"). Everything is
**opt-in** — the default stays exactly today's 3-meal, 1-plate-per-meal,
1-day / 7-day plan, so the demo path is unchanged.

This is a model-v2 change, not a feature flag. The doc is deliberately
staged so each phase ships independently and the default never regresses.

---

## 1. Vocabulary (settle this first — the whole plan hangs on it)

Today the code conflates three concepts under "meal":

- **Slot** — a labelled time-of-day position in a day. Today fixed to
  `breakfast | lunch | dinner`. Becomes a dynamic ordered list per day.
- **Meal** — everything eaten at one slot. Today exactly one recipe.
  Becomes an ordered list of **plates**.
- **Plate** — one recipe within a meal, with a **role**: `main | side |
  dessert | drink`. Today the meal *is* the single main plate.

New canonical shape (JSON-serializable, the existing storage pattern):

```
Day   = { "day": int, "slots": [Slot, ...] }          # 1-based day index
Slot  = { "meal_type": str, "plates": [Plate, ...] }  # meal_type is now free-form-ish (see §3)
Plate = { "recipe_id", "title", "ingredients", "directions",
          "role": "main|side|dessert|drink",
          "nutrition"?, "image_url"?, "match_reasons"? }
```

A default daily plan is one `Day` with three slots, each one `main` plate —
byte-identical in intent to today's `breakfast/lunch/dinner`.

**Invariant that keeps nutrition honest:** a *meal* is the sum of its
plates. Calorie/macro budgets are per-**meal** (and per-day), never
per-plate. Splitting "pasta + salad" across two plates must not double the
lunch calorie allowance — the meal target is shared across its plates. This
single rule drives most of the grader/tracker changes (§6).

---

## 2. Phasing (each phase is shippable and default-safe)

| Phase | Delivers | Risk | Depends on |
|---|---|---|---|
| **0** | Acknowledge dropped extra dishes (interim, no model change) | trivial | — |
| **1** | Core model: plates-as-list + dynamic slots, default = today | high (touches every reader) | — |
| **2** | Daily generation of multi-plate meals + N-day horizon | med | 1 |
| **3** | Weekly generation with dynamic days × meals × plates | high | 1, 2 |
| **4** | Verified editing at (day, slot, plate) + add/remove plate | med | 1 |
| **5** | Transparency (per-plate chips/ledger/summaries) | low | 1 |
| **6** | Controls: intent parsing + slider-card "meals per day" | med | 2 |
| **7** | UI across all surfaces | high (largest single chunk) | 1–6 |

Ship 0 anytime. 1 is the gate for everything else. 2 and 4/5 can proceed in
parallel once 1 lands. 3 is the hardest and should follow 2. 7 trails the
backend it renders.

---

## 3. Phase 1 — the core model (the gate)

Files: `src/models/session.py`, `src/services/session_service.py`,
`src/db.py` (serialization only — no column migration; plans are JSON blobs).

### 3.1 New dataclasses

- Keep `MealCourse` as the **plate** payload; add `role: str = "main"` to it.
  (Renaming to `Plate` is churn; alias in a comment instead.)
- `MealPlan` becomes day-oriented:
  ```python
  @dataclass
  class Meal:
      meal_type: str
      plates: list[MealCourse]          # >= 1
  @dataclass
  class DayPlan:
      day: int                          # 1-based
      meals: list[Meal]                 # ordered
  @dataclass
  class MealPlan:                       # now spans 1..N days
      id: str; created_at; days: list[DayPlan]
      reasoning: str; version; parent_id; ...metrics...
  ```
- `WeeklyMealPlan.entries` stays a `list[dict]` but each entry gains
  `"plate_idx"` and `"role"`, and `meal_idx` is no longer capped at 0-2.
  A weekly plan is then just a `MealPlan` with `len(days) == 7` — **consider
  collapsing the two types into one** `MealPlan` with a `horizon` field
  (`daily | weekly | custom`) during this phase; it removes a whole class of
  "daily vs weekly" duplication the code review already flagged. (Decision
  point — see §11.)

### 3.2 Backward compatibility (non-negotiable)

- `MealPlan.from_courses([b, l, d], ...)` keeps working: it builds one
  `DayPlan` with three single-plate meals. Add a new
  `MealPlan.from_days(days, ...)` for the general path.
- Deserialization: a stored plan with the **old** `{breakfast, lunch,
  dinner}` shape is read by a shim that lifts it into `days=[DayPlan(...)]`.
  New plans serialize the new shape. Old plans never rewrite unless edited.
- Response models (`routers/foodchat_router.py`) expose the new shape
  **additively**: keep `breakfast/lunch/dinner` populated for a
  1-day/1-plate plan (UI back-compat) AND add `days: [...]`. The UI migrates
  to `days` in Phase 7; until then old UI keeps working.

### 3.3 `from_courses` guard

`from_courses` currently raises on `len != 3`. Keep that guard for the
legacy call, but route all new generation through `from_days`, which
imposes only `len(meals) >= 1` and `len(plates) >= 1`.

**Test:** round-trip old blob → model → response → old UI shape unchanged;
new blob → model → response carries `days`.

---

## 4. Phase 2 — daily generation (multi-plate + N-day)

Files: `src/services/chat_service.py` (`process_plan_request`,
`_generate_and_store`), `src/services/planning_pipeline.py`,
`src/services/candidates_client.py`.

### 4.1 The plan spec

Generation takes a **PlanSpec** derived from the request (see §8 for how it's
parsed), defaulting to today's shape:

```python
PlanSpec = {
  "num_days": 1,                        # N-day horizon
  "meals": ["breakfast","lunch","dinner"],   # ordered slot labels
  "plates": {"lunch": ["main","side"]},      # slot -> roles; default ["main"]
}
```

### 4.2 Candidate fetch & selection

- `PlanningPipeline.generate` today builds `candidates = {slot -> [recipe]}`
  and composes one recipe per slot. Generalize to
  `candidates = {(slot, role) -> [recipe]}`:
  - `main` plates pull from the current candidate query.
  - `side`/`dessert`/`drink` plates pull from **role-scoped** RW queries
    (e.g. side → salads/vegetables; dessert → desserts). This needs a
    `role` hint on the RW candidate fetch — check whether
    `candidates_client` can filter by dish_type/course; if not, this is the
    one RW-side dependency (fall back to keyword-biased queries).
- **Kcal split (the honest-nutrition rule):** the meal's calorie target is
  divided across its plates by role weight (e.g. main 0.7 / side 0.3), so
  `main + side ≈ one meal`, not two. The pipeline already budgets per meal;
  change the unit it scores against from "the meal recipe" to "the sum of
  the meal's plates".
- Pinned/manual picks (§ compose) address `(slot, plate_idx)`.

### 4.3 N-day horizon

- Loop day generation `num_days` times, reusing the daily pipeline per day,
  with **cross-day variety** carried in the exclude set (don't repeat the
  same main two days running). This is the daily analogue of the weekly
  spread and can share the weekly variety tracker (§6).
- `num_days == 1` is the exact current path (no behaviour change).
- A 2–6 day plan is a `MealPlan` with `len(days) == num_days`; 7 days is
  still routed to the weekly planner (§3.1 collapse decision affects this).

**Tests:** default request → 1 day / 3 meals / 1 plate each (unchanged);
"pasta and a salad for lunch" → lunch has main+side, lunch kcal ≈ one meal;
"plan 2 days" → 2 days, variety across days.

---

## 5. Phase 3 — weekly generation (the hard one)

Files: `src/services/weekly_planner/{planner,environment,state_tracking,
reward_logic,action_adapter,explainability}.py`,
`src/services/weekly_plan_service.py`.

The weekly planner hard-codes `TOTAL_SLOTS = 21`, `meal_idx` 0-2, and
`current_meal_idx >= 3` wrap (`environment.py`). Generalize:

- **Environment** iterates a per-day `meals` list and, within each meal, a
  `plates` list. `TOTAL_SLOTS` becomes `sum(len(day.meals)*plates_per_meal)`.
  The step loop advances plate → meal → day.
- **Trackers rescale to the actual denominators** (`state_tracking.py`):
  - meat limit and calorie budget scale to `num_days` and to the number of
    **meals** (not plates — a salad side isn't a meat meal).
  - the tracker counts **per meal**, folding a meal's plates before it
    tests meat/nutrition, so "chicken main + salad side" is one meat meal.
- **Guideline checklist denominators** (`explainability.py`): rules like
  "most meals plant-based" and "fish 1–2×/week" are frequency counts over
  the **meal** denominator; when meals/day or days change, the targets must
  recompute against the real meal count, and the honesty note ("checked over
  N days / M meals") must state it.
- **reward_logic / action_adapter**: pre-selection nutrition enrichment and
  the constraint score operate per plate but aggregate per meal for the
  budget test.

This phase is where the RL-ish structure is most sensitive; do it last and
lean on the deterministic-since-M6 behaviour to keep it testable.

**Tests:** default weekly → 21 single-plate slots, byte-identical metrics;
"add a salad to every dinner" → 7 dinners each main+side, meat count
unchanged, calorie budget still per-meal; 4-meals-a-day week → checklist
denominators rescaled.

---

## 6. Phase 4 — verified editing & Phase 5 — transparency

### Editing (`src/services/edit_service.py`)
- Slot address becomes `(day, meal_type, plate_idx)`. The extractor
  (`EDIT_COMMAND_EXTRACTOR`) gains `plate` disambiguation ("swap the salad,
  not the pasta") and two new directives: **add a plate** ("add a side to
  lunch") and **remove a plate** ("drop the dessert").
- `changed_slots` proof rows gain `plate_idx`/`role`; the before/after kcal
  proof compares **meal totals** (add-a-plate raises the meal's kcal — show
  that honestly).
- Manual-pick unpin logic (`_drop_manual_picks_for_slots`) keys on
  `(day, meal_type, plate_idx)`.

### Transparency (`src/services/transparency.py`, `explainability.py`)
- Reason chips, constraint ledger rows, and day summaries become
  **per-plate** where it matters (a pinned side gets its own "requested by
  you" chip) but roll up to per-meal for the day headline.
- Personalization summary counts unchanged (per plan).

---

## 7. Phase 6 — how the user asks for it

Two channels, both opt-in, per the no-interrogation rule:

### Conversational (intent → PlanSpec)
- Extend the seed/intent extraction so "pasta **and** a salad for lunch"
  yields two plates on the lunch slot (today the second seed is dropped —
  this is also **Phase 0**'s cheap interim: at minimum acknowledge it).
- "4 meals a day", "just breakfast and dinner", "plan 2 days" set
  `num_days` / `meals`. Parse in the orchestrator classifier or a small
  dedicated extractor; never ask a follow-up question for it.

### Slider card (`src/services/plan_parameters.py`)
- Add a **"meals per day"** scale (2–5, default 3) to `PARAMETER_DEFS` and a
  **"plates per meal"** choice (single / main+side), per the rule that
  parameter-style choices live on the card, not in chat questions.
- These ride the existing deterministic apply path
  (`apply_plan_parameters`), which already refines the addressed plan.

---

## 8. Phase 7 — UI (largest single chunk)

Files: `wisefood-ui` — `app/services/foodchatApi.ts` (types),
`app/pages/foodchat.vue` (daily cards, weekly rows, compose),
`app/components/foodchat/MealScheduleCard.vue`, the dashboard "today"
widget, and the apply-to-dashboard flow.

- **Types:** `MealPlan` gains `days: DayPlan[]`; `Meal` has `plates`.
  Keep `breakfast/lunch/dinner` optional for back-compat during rollout.
- **Daily canvas:** render `days[].meals[]` instead of three fixed cards;
  each meal shows its plates (main prominent, sides smaller). The ⋮
  replace/adapt menu addresses a plate.
- **Weekly canvas:** the day row already groups entries; group by
  `(meal_type)` then list plates within. The collapsible-day work done
  recently makes this a smaller lift than it was.
- **Manual compose:** a slot gains "add a plate" (role picker + the existing
  autocomplete); `draftPicks` becomes `slotKey -> Plate[]`.
- **Dashboard / apply:** the "today" widget and the member-current-plans
  serializer iterate `breakfast/lunch/dinner` as scalars — migrate to
  `days[].meals[].plates[]`.

Roll out behind the additive response fields: old UI renders the legacy
scalars, new UI renders `days`, and neither breaks mid-deploy.

---

## 9. API / gateway contract

- `ComposePick` gains `plate_idx`/`role`; `ComposeRequest` gains an optional
  `spec` (meals, plates). `PlanParametersRequest` already flows through.
- `ChatTurnResponse.meal_plan` / `weekly_meal_plan` gain `days`; the gateway
  (`wisefood-api`) passes them through (it's a thin proxy — `Dict[str,Any]`
  blobs, so mostly no change beyond response typing).
- Everything additive; a stale gateway keeps proxying the legacy fields.

---

## 10. Testing strategy

- **Golden default:** a suite asserting that with no dynamic options, every
  layer produces byte-identical output to today (the regression fence).
- **Model round-trip:** old blob ↔ new model ↔ response, both directions.
- **Nutrition honesty:** main+side lunch kcal ≈ one meal, not two
  (the core invariant); add-a-plate raises meal kcal by the plate.
- **Denominator rescale:** weekly checklist targets recompute for
  meals/day ≠ 3 and days ≠ 7.
- **Editing:** plate-addressed swap, add-plate, remove-plate, with correct
  before/after meal totals.
- All LLM-free (fakes), matching the existing suite's discipline.

---

## 11. Open decisions (resolve before Phase 1)

1. **Collapse `MealPlan` and `WeeklyMealPlan` into one** day-list model with
   a `horizon` field? Pro: kills the daily/weekly duplication the review
   flagged and makes N-day fall out for free. Con: bigger blast radius in
   Phase 1. **Recommendation: yes** — do it now while the model is already
   being rewritten; it's cheaper than a second migration later.
2. **RW role-scoped candidate fetch** — does `candidates_client` /
   RecipeWrangler support filtering by course/dish_type (for `side`,
   `dessert`)? If not, Phase 2 sides fall back to keyword-biased queries and
   we file an RW ask. **Verify before scoping Phase 2.**
3. **Plate roles set** — `main | side | dessert | drink` enough for the
   living labs, or is "starter" needed? Affects the kcal-weight table.
4. **Max plates/meal and meals/day caps** — propose 3 plates/meal, 5
   meals/day, to bound the planner search and the UI.

---

## 12. Non-goals (explicit)

- Per-plate scheduling/timing (a meal is one sitting).
- Portion scaling per plate (that's RecipeWrangler's adaptation surface).
- Ingredient-level "half portions" — out of scope; plate granularity only.
- Changing the default plan shape — the 3-meal / 1-plate / 1-day plan stays
  the untouched default and the demo path.

---

## 13. Recommended first step

Land **Phase 0** now (acknowledge the dropped second dish — a few lines in
the seed path, honest and demo-safe), then do **Phase 1 + decision §11.1**
as a single focused PR behind the additive response fields, with the golden
default suite as the fence. Nothing user-visible changes until Phase 2, so
Phase 1 can merge without touching the demo.
