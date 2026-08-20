# Pantry-driven planning ("cook from what I have") — design & implementation plan

**Status: Phases 1–3 and the badge half of Phase 5 SHIPPED (2026-08-18) —
see the CHANGES.md entry.** Implemented Tier A only (per-item fan-out, no
RecipeWrangler change). Still open: Phase 4 (the RW `boost` ask), the
durable consent-gated pantry and dedicated pantry endpoint from Phase 5,
weekly anchor persistence for named-dish swaps, and quantity awareness /
leftover chaining (out of scope by design).

Scope: let a member state ingredients they already have — "I've got zucchini,
half a bag of spinach and some ground beef" — and have the pipeline build or
refine a plan that **maximizes use of those ingredients**, to reduce food
waste. Works on both horizons (daily and weekly), integrates with verified
slot edits ("swap dinner for something that uses the zucchini"), and reports
honestly which items it managed to use and which it could not. Everything is
opt-in: a member who never mentions their fridge gets exactly today's flow.

---

## 1. The core semantic problem (settle this first)

The obvious wire-up — forward the pantry as `include_ingredients` to
`/api/v2/tools/plan_meals` — is wrong, and the codebase already documents why
twice (`planning_pipeline._fetch_candidate_pool`,
`weekly_planner/action_adapter._fetch_candidate_pool`):

> `plan_meals` treats `include_ingredients` as a **requirement** (hard AND).
> Demanding chickpeas in every breakfast empties the slot.

What food-waste planning needs is different semantics: **"require none,
reward coverage"** — a soft, rank-by-usage objective over the *whole plan*,
not a per-recipe filter. That signal does not exist on the current
RecipeWrangler surface, so the plan is two-tiered:

- **Tier A (FoodChat-only, ships first):** source pantry-matching candidates
  through the existing per-ingredient search surfaces and rank coverage
  client-side. No RecipeWrangler change; some request fan-out.
- **Tier B (coordinated RW enhancement, later):** add
  `include_ingredients_mode: "all" | "any" | "boost"` (or a separate
  `boost_ingredients` field) to `plan_meals`, and delete the fan-out. The v1
  candidate endpoint *ranked* by `include_ingredients`, so the semantics have
  upstream precedent.

Also out of scope, stated honestly rather than faked: **quantities**. We
match ingredient *presence*, not amounts — "uses your zucchini" never claims
"uses it all up". Leftover chaining across days (Monday's dish uses half the
cabbage, Tuesday's the rest) is a stretch goal, not this plan.

---

## 2. Vocabulary

- **Pantry** — the list of on-hand ingredients the member has told us about.
  Session-scoped planning state (like anchors), *not* a durable profile
  field. "Always plan around what I log" as a durable behaviour stays behind
  the M3 memory-consent nudge, later.
- **Coverage** — for one recipe: which pantry items appear in its ingredient
  text (word-boundary matching, same mechanism as `allergen_conflict`). For
  a plan: the union over its recipes.
- Pantry items are **soft seeds at ingredient granularity**. The seeded
  (M2) architecture — extract → resolve → pin → plan around → describe —
  transfers almost verbatim; only "resolve to one recipe and pin it" becomes
  "resolve to a coverage-ranked candidate pool".

---

## 3. Phasing (each phase shippable, default-safe)

| Phase | Delivers | Risk | Depends on |
|---|---|---|---|
| **1** | Pantry state + extraction + daily sourcing/ranking + honest reporting | med | — |
| **2** | Verified edits: "swap X for something using ⟨ingredient⟩" | low | 1 |
| **3** | Weekly horizon: coverage term in the MDP reward | med | 1 |
| **4** | RecipeWrangler `boost` semantics; delete the fan-out | low (FoodChat side) | RW change |
| **5** | UI/gateway affordances (pantry chips, "uses your…" badges); durable pantry behind memory consent | med | 1–3 |

---

## 4. Phase 1 — daily pantry planning (the MVP)

### 4.1 State (`src/models/planning_state.py`)

Add to `PlanningState`:

```python
pantry: tuple[str, ...] = ()   # normalized ingredient strings, insertion order
```

and mirror it in `PlanningStateDelta` (`pantry: Optional[tuple[str, ...]]`,
absent = not mentioned), `merge`, `describe`, `to_dict`/`from_dict`. Merge
rule: additive like `notes`; an explicit removal ("I used up the zucchini" /
"forget the pantry") clears items — model that as a delta carrying the *new
full list* or a paired `pantry_remove` tuple, whichever reads cleaner, but
keep "silence is not a retraction" intact. `reset()` clears it like
everything else.

### 4.2 Extraction (`src/agents.py`, `src/prompts.py`, `src/schemas.py`)

Extend `SeedExtractionSchema` with an `ingredients: list[str]` field so **one
existing LLM call** returns both named dishes and raw ingredients — no new
per-turn model call. The prompt must draw the line explicitly:

- "pastitsio for dinner" → seed (dish name).
- "I have zucchini and ground beef" → pantry ingredients.
- "chicken soup with the chicken I have" → seed *and* pantry item.

The orchestrator already runs seed extraction on plan turns; it folds
`ingredients` into a `PlanningStateDelta(pantry=…)` alongside anchors. A pure
inventory statement with no plan request ("btw I have leftover rice")
classifies as `preference_update` today — that handler acknowledges and moves
on; teach it to also store the pantry delta and offer to plan. **No new
orchestrator intent.**

Safety: pantry items are member-supplied free text. They are *inputs to
search*, never bypasses — every fetch below still carries allergens, diet and
dislikes, so "I have peanuts" from a peanut-allergic member simply finds
nothing usable (and the reply says so).

### 4.3 Candidate sourcing (`src/services/planning_pipeline.py`)

In `generate()`, when `state.pantry` is non-empty:

1. Fetch the ordinary pool exactly as today (constraints, cuisines, tier).
2. **Per pantry item** (cap: first 6), fetch matching recipes under the same
   hard constraints. Preferred call: `plan_meals(count_per_slot=3,
   include_ingredients=[item])` — a *single*-item hard AND is exactly "must
   use this one thing", and the result stays slot-grouped and
   planning-tier-aware. Fallback: `PLANNER.find_recipes(q=item, …)` with
   slot assignment via dish-types (the seed-service pattern). Run the
   per-item calls in parallel (`httpx` async or a thread pool) and treat any
   individual failure as "that item found nothing" — best-effort, like every
   other enrichment.
3. Merge into per-slot pools, dedupe by recipe id, compute each candidate's
   coverage (word-boundary matching of pantry terms against the ingredient
   text — reuse the `allergen_conflict` matching mechanics, factored into a
   shared helper). Order: coverage desc, then the deterministic upstream
   order. Keep pools at `CANDIDATE_LIMIT` so the grading space stays 8³.
4. Grade as today, with one addition to the grader input: "prefer
   combinations that together use as many of: ⟨pantry⟩". The **deterministic
   coverage computation, not the model, is the source of truth** for every
   user-facing claim.

New env knobs (documented in `.env.example`): `FOODCHAT_PANTRY_ITEM_LIMIT`
(default 6), `FOODCHAT_PANTRY_PER_ITEM_CANDIDATES` (default 3).

### 4.4 Transparency & response honesty

- `match_reasons` gains a `{"kind": "pantry", "label": "uses your zucchini"}`
  entry per matched course (`src/services/transparency.py`).
- The facts dict for the `ResponseWriter` gains
  `pantry: {"items": […], "used": {item: [recipe titles]}, "unused": […]}`.
  Unused items get a sentence, not silence: *"Nothing that fits your profile
  uses the durian — it's still on your list; tell me if you'd rather I drop
  it."* Same honesty contract as `seed_service.describe`.
- `ChatTurnResponse` needs no schema change for the MVP (prose + existing
  `match_reasons` carry it); a structured `pantry` block is Phase 5.

### 4.5 Tests (LLM-free, per `tests/conftest.py`)

- `PlanningState` merge/round-trip with pantry (add/remove/reset).
- Coverage matcher: word boundaries ("cream" must not match "ice cream"
  only when boundaries actually prevent it — pick and pin the behaviour),
  plurals, multi-word items.
- Pipeline with a fake `PLANNER`: pantry candidates merged, ranked, capped;
  per-item fetch failure degrades to the plain pool; unused items reported.
- Extractor schema handling with a faked LLM response (dishes vs
  ingredients vs both).

Update `CHAT_ENDPOINT_PIPELINE.md` (message flow changes) and append the
`CHANGES.md` handoff entry, per the engineering standards.

---

## 5. Phase 2 — verified pantry edits (`src/services/edit_service.py`)

Add a `DirectivePredicate` kind `uses_ingredient` (parse "with/using
⟨ingredient⟩", plus pantry-item mentions when state has a pantry):

- **Verifiable**: passes iff the term matches the replacement's ingredient
  text (details are already batch-fetched for the predicate check; fail
  closed on missing text, like the other quantitative predicates).
- Candidate sourcing: the single-item `plan_meals` call from §4.3 first,
  the ordinary `slot_candidates` filtered by the matcher as fallback.
- Success text states the proof: "Verified: it uses your zucchini." Failure
  uses the existing nearest-miss/honest-failure path.

This also fixes today's behaviour where "something with zucchini" falls into
the unverified branch and gets full-text-searched *as a dish name*.

While in this file, close two review gaps (small, independent commits):
- Swap candidate fetches should also exclude
  `PlanningState.excluded_recipe_ids` — today a recipe the member already
  rejected can come back through a swap.
- Weekly named-dish swaps should record the anchor in `PlanningState` the
  way `_edit_daily` does, so a later regeneration can't lose it (the
  entry-level `"pinned": True` only protects the current canvas).

---

## 6. Phase 3 — weekly horizon

- `weekly_plan_service.process` reads `state.pantry` and passes it down.
- `action_adapter`'s pool fetch gains the same per-item merge (§4.3); each
  candidate action carries a precomputed `pantry_matches: [items]`.
- The MDP reward adds a coverage bonus with **diminishing returns per item**
  (first use of the zucchini is worth a lot, third use worth little) so the
  planner spreads the pantry across the week instead of stuffing every
  match into Monday. Weight it below hard-preference terms — coverage must
  never beat variety into degeneracy.
- `build_weekly_explainability` reports the coverage ledger: which items
  were used, where, and which weren't.

---

## 7. Phase 4 — the RecipeWrangler ask

Propose upstream: `include_ingredients_mode: "all" (default) | "any" |
"boost"` on `plan_meals` (or a `boost_ingredients` list). With `boost`,
FoodChat sends the pantry once per pool fetch, RW ranks by coverage inside
the tier/Nutri-Score order, and §4.3's fan-out collapses to one call. The
client-side coverage matcher **stays** as the source of truth for
user-facing claims — same defense-in-depth reasoning as the allergen
backstop. Gate the new field on the tool manifest (`planning_options`) so
FoodChat degrades to Tier A against an older RW.

---

## 8. Phase 5 — product surface

- Gateway/UI: pantry chip row on the plan card (add/remove items → a
  `PlanningStateDelta`, either via chat or a small
  `POST /sessions/{id}/pantry` endpoint mirroring the plan-parameters
  pattern); "uses your ⟨item⟩" badges from `match_reasons`.
- Durable household pantry ("always plan around what I log") — only behind
  the M3 memory-consent nudge.
- Stretch: leftover chaining across days; quantity awareness.

---

## 9. Risks

- **String matching lies both ways.** "Cream" vs "ice cream", "rice" vs
  "rice vinegar". Word-boundary matching plus the cap on claims (only
  deterministic matches are ever stated) keeps errors on the harmless side:
  a missed match under-reports coverage, it never fabricates it.
- **Latency.** Up to 6 extra RW calls per generation in Tier A. Parallelize,
  cap, and treat failures as best-effort; Phase 4 removes the cost.
- **Coverage vs quality tension.** A plan built only from scraps can be a
  bad plan. Coverage is a *ranking* signal inside pools that already respect
  every hard constraint and preference — never a filter that shrinks the
  pool below what the grader needs.
