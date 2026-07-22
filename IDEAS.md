# Weekly planning improvements

**Status: implemented 2026-07-20** (see the CHANGES.md entry "Weekly
planner: constraints steer selection + per-day summaries"). Phase 3 was
resolved by REMOVING the per-step LLM call; the "no candidates survive"
question from Phase 2 was decided as relax-with-warning; Phase 4 shipped as
diet-derived + preference-string override (no slider card yet — that
remains open if the UI wants it).

Diagnosis and fix plan for constraint violations in the 7-day planner
(`src/services/weekly_plan_service.py` + `src/services/weekly_planner/`).

## Diagnosis

The planner's constraint/reward machinery is fully implemented but never wired
into selection — it only logs after the fact.

1. **Reward is computed after the pick, not used to make it.**
   `WeeklyPlanner.generate_full_plan` (`planner.py:96-102`) chooses a candidate
   via `build_preference_scorer` (favorites + liked ingredients + variety —
   no constraints), then calls `env.step()`, which computes
   `RewardCalculator.calculate_step_reward()` (`reward_logic.py:107-141`) and
   stores it on the entry purely for display/logging
   (`environment.py:99-119`). Nothing ever reads that reward back to influence
   a choice, current or future.

2. **The calorie/macro constraint compares against numbers that are always
   zero.** `RecipeActionSpace.get_candidate_actions` (`action_adapter.py:56-65`)
   returns candidates with no `nutrition` field. Nutrition is only attached
   *after* the full 7-day plan is generated, as a batch enrichment call
   (`weekly_plan_service.py:134-142`). So during generation
   `chosen_recipe.get("nutrition")` is always `None`,
   `WeeklyNutritionalTracker.weekly_calories` never moves off `0.0`
   (`state_tracking.py:62-76`), and `remaining["calories"]` never goes
   negative (`state_tracking.py:101-104`). The calorie constraint cannot ever
   trigger.

3. **The meat limit is hardcoded to 3/week for every profile**
   (`state_tracking.py:35`), not derived from the member's actual diet/profile.
   Meat detection itself works correctly (keyword match over ingredient text),
   but per (1), breaching the limit changes nothing about what gets picked
   next.

4. **Wasted LLM spend as a side effect.** `RewardCalculator.get_llm_feedback`
   fires one Groq call per step (21 per plan) to grade a recipe that has
   *already* been committed to the plan — cost and latency with zero effect
   on the output.

Net effect: only the hard filters applied at RecipeWrangler fetch time
(allergens, diet tags, already-selected exclusion —
`action_adapter.py:43-51`) are actually enforced. Everything downstream of
that (meat limit, calorie/macro targets, LLM preference grading) is
decorative.

## Plan

### Phase 1 — Make nutrition available during generation, not just after

- Have `RecipeActionSpace.get_candidate_actions` attach nutrition to each
  candidate dict at fetch time (via `CANDIDATES`, same client used for the
  post-hoc enrichment call), instead of leaving `weekly_plan_service.py`'s
  batch enrichment as the first time nutrition data exists.
- This unblocks `WeeklyNutritionalTracker` actually accumulating real
  calories/macros during the 21-step loop, which (2) depends on.
- Keep (or drop, see Phase 3) the existing post-plan batch enrichment call —
  it's still useful for image/nutrition-chip data on the final response, just
  no longer the *only* source of nutrition.

### Phase 2 — Turn constraints into a selection signal, not a log line

- Add a method to score/filter a *candidate list* against the tracker's
  current state, e.g. `RewardCalculator.filter_and_rank(candidates, tracker,
  preferences)`, called from `WeeklyPlanner` before a pick is made — not
  after.
- Hard constraints (should exclude, not just penalize):
  - Meat limit: if `remaining["meat_limit_left"] <= 0`, drop meat candidates
    from the pool for remaining slots (fall back to non-meat only).
  - Calorie budget: if a candidate would push cumulative calories far past
    the weekly target (e.g. > 110%), deprioritize/exclude it, especially in
    the final days of the week when there's less room to correct course.
- Soft constraints (weight into `build_preference_scorer` instead of
  discarding): macro proximity to daily/weekly targets, so the scorer choice
  and the constraint check are the same decision instead of two disconnected
  systems.
- Decide whether "no candidates survive the hard filter" should relax the
  constraint (with a note in the response) or raise, matching the existing
  `ValueError` policy for empty slots (`planner.py:89-95`) — needs a product
  decision, not just a code change.

### Phase 3 — Resolve the per-step LLM call

Pick one:
- **Remove it.** It currently buys nothing (graded after the fact, discarded).
  Simplest option if the preference scorer covers "does this match what the
  user asked for" well enough on its own.
- **Repurpose it.** Run it *before* selection, over the ~10 candidates for a
  slot (not once per committed step), and fold its score into the same
  ranking used in Phase 2 — this is the only way it actually earns its cost.

### Phase 4 — Make the meat limit configurable

- Replace the hardcoded `meat_limit: 3` (`state_tracking.py:35`) with a value
  derived from the member's profile/preferences (similar to how the calorie
  target is already parsed from a preference string in
  `profile_service.py:342` / `state_tracking.py:40-44`), so vegetarians
  aren't the only profiles for which this number is actually correct.
- Consider exposing it as a `plan_parameters.py` slider (existing pattern for
  LLM-free, deterministic refinements) rather than a free-text preference
  string, for consistency with how other plan parameters are surfaced to the
  UI.

### Phase 5 — Verify

- Add a test alongside `tests/test_seeded_planning.py` /
  `tests/test_plan_parameters.py` that builds a profile with a low meat
  limit and fake candidates skewed toward meat, and asserts the generated
  week's meat-meal count is within the limit — this is the regression test
  that would have caught the current bug.
- Add a calorie-budget equivalent: fake candidates with known calorie values,
  assert the generated week stays within tolerance of the weekly target.
- Confirm unit tests stay LLM-free per `tests/conftest.py` (fake
  `CANDIDATES`/`RewardCalculator` as needed, per existing test patterns).

## Notes

- None of this requires the "MDP" framing to become a real RL setup (Q-table,
  policy learning, discounting) — it's currently single-step greedy selection
  with reward computed and thrown away; the fix is closing that loop, not
  building out training infrastructure.
- Per `CHAT_ENDPOINT_PIPELINE.md` and the engineering standards in
  `CLAUDE.md`, any change to weekly-plan message flow should be reflected in
  `CHAT_ENDPOINT_PIPELINE.md`'s weekly plan branch section, and get a
  `CHANGES.md` handoff entry once implemented.

---

# Weekly plan presentation — per-day summaries

**Status: implemented 2026-07-20** with Option A (additive `day_summaries`
field; `entries` unchanged). Phase 3's composition templates shipped as the
starting-point rules below — still worth iterating against real generated
weeks. Phase 7 (summaries in the chat reply facts) was included.

Plan for grouping the weekly plan by day
and attaching a short (2-3 word) descriptive summary per day (e.g. "dinner
with red meat", "light vegetarian day") before the plan is returned to the
caller.

## Where this fits

"After filtering/scoring, before presenting" = the tail of
`WeeklyPlanService.process_message` (`weekly_plan_service.py:128-186`),
specifically right after the M4 enrichment loop
(`weekly_plan_service.py:134-142`) that already fetches nutrition/image data
for the 21 recipes and right before the plan is persisted
(`add_weekly_meal_plan` / `refine_weekly_meal_plan`) and handed to
`response_writer`.

Useful discovery: `RecipeEnrichment` (`models/recipe.py:30-62`, the same
object the M4 enrichment call already returns) carries `tags`, `dish_types`,
and `kcal` — but the current copy loop only pulls `nutrition` and
`image_url` off it (`weekly_plan_service.py:140-142`) and discards `tags`
and `dish_types`. Those are RecipeWrangler's own authoritative
vegetarian/vegan/pescatarian tags (`VALID_RW_DIET_TAGS`,
`candidates_client.py:36-39`) — classifying meals from them is free (no new
fetch, no LLM call) and more reliable than re-deriving diet info from
ingredient text.

## Plan

### Phase 1 — Response shape decision (needs a call, not just code)

Two ways to expose grouping, differing in blast radius:

- **Option A — additive (recommended default).** Keep
  `WeeklyMealPlanResponse.entries` exactly as-is (flat, unchanged) so any
  existing gateway/UI consumer keeps working untouched, and add a sibling
  field, e.g. `day_summaries: dict[int, str]` (day → phrase). The UI already
  has to group `entries` by `day` to render a week view (see
  `_format_weekly_plan_as_context`, `weekly_plan_service.py:19-34`, for the
  same grouping done server-side for refinement context) — this just gives
  it a headline string per group. Mirrors how `nutrition`/`image_url` were
  added to `MealCourse` as pure-additive optional fields in M4.
- **Option B — restructuring.** Replace `entries` with a nested
  `days: [{day, summary, meals: [...]}]` shape. More "correct" long-term but
  is a breaking response-shape change, which per this repo's standard
  ("removed endpoints are cleaned across service → gateway → UI in the same
  change") means coordinating the wisefood-api gateway and UI in the same
  change, not just FoodChat.

Recommendation: ship Option A first (low-risk, immediately useful), consider
B only if the flat `entries` shape becomes a real UI pain point later.

### Phase 2 — Stop discarding tags/dish_types during enrichment

Extend the copy loop at `weekly_plan_service.py:138-142` to also set
`entry["recipe"]["tags"] = rich.tags` and
`entry["recipe"]["dish_types"] = rich.dish_types` (falling back to `[]`),
alongside the existing `nutrition`/`image_url` copy. Zero new network calls —
this data is already in `enrichment`, just unused today.

### Phase 3 — Meal classifier (new small module, LLM-free)

New module, e.g. `src/services/weekly_planner/day_summary.py`. Two functions:

- `classify_meal(recipe: dict) -> MealCategory` — priority order:
  1. RecipeWrangler tags from Phase 2 (`vegetarian`/`vegan` →
     `vegetarian`; `pescatarian`/`pescatarian_safe` → `fish`).
  2. Ingredient-text keyword backstop when no tag is present — reuse
     `ALLERGEN_SYNONYMS["fish"]` / `["shellfish"]`
     (`candidates_client.py:80-83`) for fish/seafood detection, and a
     meat-keyword set for red meat vs. poultry.
  3. `kcal` (already on `entry["recipe"]["nutrition"]` since Phase 1 of the
     [constraints plan](#plan) above) for a `light`/`hearty` qualifier when
     available.
  - **Cleanup while touching this:** `state_tracking.py:79-84`
    (`MEAT_KEYWORDS`) and `candidates_client.py:74-90`
    (`ALLERGEN_SYNONYMS`) each maintain their own overlapping
    meat/fish/shellfish keyword lists today. Worth consolidating into one
    shared taxonomy (this module, or a new `services/food_taxonomy.py`) so
    the day-summary classifier and the weekly meat-limit constraint (see the
    constraints plan, Phase 4 above) read from the same source instead of
    two lists that can silently drift apart.
- `summarize_day(meals: list[dict]) -> str` — template composer over the 3
  classified meals for a day, e.g.:
  - all 3 vegetarian/vegan → `"vegetarian day"` (+ `"light"` if kcal
    supports it)
  - one meal clearly stands out → name it, matching the user's own example:
    `"{meal_type} with {category}"` (e.g. "dinner with red meat")
  - mixed/no clear signal → a generic fallback (`"varied meals"`)

  **This composition logic is the fuzziest part of the plan** — the
  rules above are a starting point, not a spec. Worth prototyping against a
  handful of real generated weeks before locking the template set, since
  "what's notable about this day" is a judgment call that's hard to get
  right on paper.

### Phase 4 — Wire it into `process_message`

After Phase 2's enrichment copy and after `overlay_weekly_entries`
(`weekly_plan_service.py:145`, so summaries reflect the member's adapted
recipes, not the originals), group `plan_entries` by `day` and call
`summarize_day` per day to build `day_summaries: dict[int, str]`.

### Phase 5 — Persistence (additive, no DB migration)

- Add `day_summaries: dict = field(default_factory=dict)` to the
  `WeeklyMealPlan` dataclass (`models/session.py:132-139`), same pattern as
  `MealPlan.constraints_applied` / `personalization_summary`.
- `_serialize_weekly_plan` / `_deserialize_weekly_plan`
  (`session_service.py:574-580`, `:618-624`) need the new key added to the
  JSON payload dict. Since `meal_plans` is stored as a JSON blob (not typed
  columns), this needs no schema migration — just `.get("day_summaries",
  {})` on deserialize for plans written before this change.

### Phase 6 — API response

Add `day_summaries: dict[int, str]` to `WeeklyMealPlanResponse`
(`foodchat_router.py:145-160`) and populate it in `from_weekly_meal_plan`
(per the Phase 1 decision).

### Phase 7 (optional) — Feed summaries into the chat reply text

`response_writer.write(facts, ...)` already receives a `facts` dict
(`weekly_plan_service.py:172-182`) that becomes the assistant's chat
message. Adding `day_summaries` to `facts` would let the reply mention e.g.
"Tuesday's a light vegetarian day" instead of the summaries only showing up
in the structured plan payload. Nice-to-have, not required for the
presentation refactor itself.

### Phase 8 — Tests

- Unit tests for `classify_meal` / `summarize_day` with hand-built recipe
  dicts covering: RW-tagged vegetarian, untagged-but-fish-by-ingredients,
  red meat, poultry, mixed day, missing nutrition data. Fully deterministic,
  no LLM, no network — fits `tests/conftest.py`'s constraints directly.
- One service-level test (style of `tests/test_seeded_planning.py`) with a
  faked `CANDIDATES.fetch_details` asserting `WeeklyMealPlan.day_summaries`
  has all 7 days populated.

### Phase 9 — Docs

Update `CHAT_ENDPOINT_PIPELINE.md` section 5 (weekly plan branch) to mention
the day-summary step, and add a `CHANGES.md` handoff entry once implemented,
per this repo's engineering standards.

## Scope note

Only the weekly plan is in scope here (that's what was asked for). The
daily plan (`MealPlan`, 3 courses) doesn't need day-grouping since it's a
single day already, but a one-line "day summary" for it would reuse the same
`classify_meal`/`summarize_day` functions from Phase 3 almost for free, if
ever wanted later.

---

# Weekly plan explainability — show the user how the plan resulted

**Status: implemented 2026-07-22** (see the CHANGES.md entry "Weekly plan
explainability"). Phases 1, 2, 4, 5 shipped as planned; Phase 3 shipped
deterministic-only — the "optional LLM grades for parity" bullet was
deliberately skipped (LLM-free where possible), the frequency checklist
and category distribution cover it. Selection events are recorded at
decision time in `apply_hard_constraints` (prunes + relaxations); the
`status: "relaxed"`/`"violated"` values and a `detail` string are additive
on ledger rows — UI should treat unknown statuses as informational.

Bring the daily plan's
transparency/metrics story to the weekly plan — and go further, because
since the constraints rework (see the CHANGES.md entry of 2026-07-20) the
weekly planner is fully deterministic: we can record the ACTUAL reasons
each pick won at decision time, instead of asking an LLM to rationalize a
finished plan the way parts of the daily metrics do.

## What daily plans have today (the baseline to mirror)

- Four quality metrics (`chat_service._compute_metrics`,
  `chat_service.py:360-385`): `llm_score`/`llm_reasoning` from the grader,
  `fvs_count` (deterministic unique-ingredient count), LLM diversity
  score, LLM guideline-adherence score (graded against
  `belgium_dietary_guidelines_augmentation.cypher` when present).
- `services/transparency.py` (pure functions, no LLM/IO): per-course
  `match_reasons` chips (kinds: pinned | favorite | memory | profile |
  feedback | diner | guideline), plan-level `constraints_applied` ledger
  (hard/soft rows with `source`), `personalization_summary` counts.
- The ledger's top rows feed the ResponseWriter facts
  (`constraints_honored`), so the chat reply mentions them.

The weekly plan currently exposes none of this — only `day_summaries` and
the internal per-entry `reward` scalar.

## Plan

### Phase 1 — Direct ports (reuse `transparency.py`, no new logic)

- `constraints_ledger(profile, downvoted_count)` and
  `personalization_summary(profile, feedback_lines)` are profile-driven
  pure functions — call them from `WeeklyPlanService.process_message` and
  store on new additive fields `WeeklyMealPlan.constraints_applied` /
  `.personalization_summary` (same JSON-blob pattern as `day_summaries`:
  no DB migration, `.get(..., default)` on deserialize, additive on
  `WeeklyMealPlanResponse`).
- Per-entry `match_reasons`: `match_reasons(recipe_id, ingredients_text,
  profile, pinned_ids)` needs nothing an entry doesn't have. Attach as
  `entry["recipe"]["match_reasons"]`, mirroring `MealCourse`. Existing
  markers map to existing chip kinds: `recipe["pinned"]` → the "requested
  by you" chip, `recipe["adapted"]` → the `ADAPTED_REASON` chip
  (`adapted_recipes.py:23` — daily overlay adds it, weekly overlay
  currently only sets the flag; unify while here).

### Phase 2 — Measured constraint ledger (the weekly-specific win)

The daily ledger statically declares `status: "satisfied"`. The weekly
tracker has real numbers, so the weekly ledger can REPORT instead of
declare:

- Meat limit row from the tracker: "Meat limit (3/week): 3 of 3 used —
  Thursday onward planned meat-free" (`meat_meals_count` vs
  `targets["meat_limit"]`, plus the slot where `apply_hard_constraints`
  first pruned).
- Calorie budget row: "13,450 of 14,000 kcal planned (96%)"
  (`weekly_calories` vs `targets["calories"]`).
- A new `status: "relaxed"` value for honest failure: when the
  all-meat-pool fallback fires (`reward_logic.apply_hard_constraints` —
  today it only logs a warning), the ledger should say "meat limit
  couldn't be fully honored on Friday dinner — every available candidate
  contained meat". Same honesty principle as the edit service's
  nearest-miss responses. NOTE: `status` today is only ever "satisfied" —
  check the UI tolerates a new value before shipping (additive enum).

To keep this truthful rather than reconstructed, the planner loop should
record small "selection events" AS IT DECIDES (pool pruned by the meat
filter + how many dropped; calorie score flipped the argmax; tiebreak
among N equals; relaxation fired). `WeeklyPlanner.generate_full_plan` has
all of this in hand at pick time — return it alongside the entries (or
accumulate on the env) rather than re-deriving post-hoc, otherwise
explanations can diverge from actual causes (e.g. random tiebreaks).

### Phase 3 — Weekly quality metrics

- **Deterministic variety:** extend the daily FVS (unique-ingredient
  count) to 21 meals; state "21 distinct recipes" (already guaranteed by
  `mark_selected` exclusion — say so instead of leaving it implicit); and
  the freebie: a category distribution from `day_summary.classify_meal`
  ("9 vegetarian, 3 fish, 2 red meat, …") — zero cost, arguably the most
  user-meaningful weekly variety statement.
- **Deterministic guideline checks:** food-based dietary guidelines
  (including the Belgian set the daily grader uses) are largely WEEKLY
  frequency rules — "fish 1-2×/week", "limit red meat per week". A day
  can barely be graded against those; a week genuinely can, and the
  frequency-type rules are checkable straight from the category counts,
  no LLM. Ship a small deterministic checklist (rule, target, actual,
  met?) for the frequency rules.
- **Optional LLM grades for parity:** one diversity + one adherence call
  per weekly plan over a compact plan text — the day summaries are a good
  compact input. 1-2 calls per plan vs the 21 removed in the constraints
  rework; decide if the parity is worth the cost/latency (the
  deterministic checklist above may be enough).
- **Nutrition summary from the tracker:** weekly totals + per-day average
  vs target (kcal, protein). Daily plans can't offer this; weekly can.
  Caveat: totals are only as complete as enrichment coverage — carry a
  "based on N of 21 meals with nutrition data" qualifier when coverage is
  partial.

### Phase 4 — Chat reply integration

Feed the measured ledger + metrics into the ResponseWriter facts (same
pattern as `constraints_honored` on daily and `day_summaries` on weekly),
so the reply itself can say "kept within your 3-meat-meal limit, 96% of
your calorie budget."

### Phase 5 — Tests + docs

- Unit tests: ledger rows from a tracker in known states (satisfied /
  reached / relaxed), category-distribution + guideline-checklist math,
  match_reasons on entry dicts (pinned/adapted/favorite/like). All
  deterministic, LLM-free per `tests/conftest.py`.
- Service-level test extending `tests/test_day_summary.py`'s faked
  `CANDIDATES` setup: assert ledger/summary fields populated, persisted,
  and exposed on `WeeklyMealPlanResponse`.
- `CHAT_ENDPOINT_PIPELINE.md` section 5/6 + `CHANGES.md` handoff entry
  once implemented.

## Explicitly NOT doing

- Surfacing the per-entry `reward` scalar in the UI — internal penalty
  number, meaningless to users (kept in the payload for compatibility).
- Per-meal LLM grading — same reason it was removed from the planning
  loop: cost with no decision value.