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
---

# Multi-plate meals & dynamic meal structure

**Status: open (recorded 2026-07-23).** User-requested during demo prep.

The ask: "I want pasta AND a salad for lunch" — meals composed of several
plates, and days with a user-chosen number of meals (2-5 instead of the
fixed breakfast/lunch/dinner). More realistic than one-recipe-per-slot.

## Why this is a deep change, not a feature flag

The single-course slot assumption is load-bearing across every layer:

- `MealPlan` model: exactly `breakfast/lunch/dinner`, each ONE `MealCourse`
  (weekly: `meal_idx` 0-2). Multi-plate needs `slot -> list[course]` and
  dynamic slot lists, with a DB-compatible serialization (JSON blobs are
  already the pattern — no migration needed, but every reader changes).
- Planning pipeline & weekly planner: candidate selection, kcal budgeting,
  and the diversity/guideline graders all assume 3 courses/day (21/week).
  Plate counts change nutrition math (a plate is not a meal — kcal targets
  must split across plates, not multiply).
- Verified slot editing: predicates address (day, meal_type); they'd need
  (day, meal_type, plate_idx) plus "remove/add a plate" directives.
- Transparency: per-meal chips/ledger/summaries become per-plate.
- UI: daily cards, weekly day rows, manual-mode compose slots, dashboard
  "today" widget, apply-to-dashboard flow.

## Suggested staging

1. **Cheap conversational start**: multiple seeds per slot already parse
   ("pastitsio and a salad for lunch" yields two seeds); today the second
   pin is dropped. Interim: fold the extra dish into the query as a soft
   side-dish request so the reply acknowledges it honestly.
2. **Model v2**: courses-as-list with `plate` role (main/side/dessert),
   fixed 3 meals — unlocks "pasta + salad" without touching meal count.
   Manual mode gets "add a plate" on a slot.
3. **Dynamic meal count** (2-5 meals/day) last — it perturbs guideline
   grading ("most meals plant-based" et al. are frequency rules over a
   meal denominator) and the RL-ish weekly tracker the most.

Slider-card tie-in: "meals per day" belongs on the plan-parameter card
(scale 2-5), NOT in chat questions, per the no-interrogation rule.

---

# N-day plan horizon ("plan meals for 2 days")

**Status: open (recorded 2026-07-23).**

Today there is no middle ground between one day and seven: the weekly
planner is hard-coded to 21 slots (`planner.py: TOTAL_SLOTS = 21`), so a
"plan meals for 2 days" request gets classified as either daily_plan
(one day) or weekly_plan (a full week) — the "2 days" is silently
ignored either way.

Tractability: HIGH compared to multi-plate meals. The redesigned UI day
list already renders whatever days exist; seeds/pins are day-addressed;
the planner loop can take `num_days`. The real work:
- classifier/orchestrator: extract the horizon (1-7) from the request
  and pass it through (or put "days" on the plan-parameter slider card
  — consistent with the no-interrogation rule);
- trackers/budgets: meat limit and calorie budget must scale to N days
  instead of assuming 7;
- explainability: the guideline frequency checklist ("fish 1-2× a week")
  must either rescale or honestly annotate "checked over N days";
- weekly refinement context and day summaries already iterate actual
  entries — should hold as-is.

---

# Localize weekly explainability prose

**Status: open (recorded 2026-07-23).** Day summaries ("fish and red
meat"), the guideline checklist rules, the variety sentence, and the
coverage note are backend-generated English strings — Hungarian and
Slovenian UIs show English fragments inside a localized page. Options:
key-based payloads the UI translates (preferred; the checklist is already
structured enough), or backend localization by household region. Fine for
the EN demo video; needed before the living-lab study.

---

# Weekly refinements discard verified slot edits

**Status: open (recorded 2026-07-23, from the code review).**

Any weekly refinement — text ("make it lighter") or a slider apply —
regenerates all 21 slots. Only stored manual picks are re-pinned, so a
slot the member approved via a verified edit ("swap Tuesday's dinner for
something lighter") is silently replaced.

The daily flow has the same shape but hurts less (3 slots, cheap to
redo). Options, roughly in order of appeal:
1. Record verified-edit results as manual picks for their slot — the
   member approved that dish, so it becomes an anchor. Cheap: the pick
   machinery already exists. Risk: makes edited slots sticky against
   later plan-wide instructions (same tension as
   `_seeds_for_refinement` re-pinning vs "make the whole day
   vegetarian"), so it needs the same "I kept X" honesty line.
2. Weekly refinement that only re-plans slots the request touches —
   needs the planner to accept a frozen-slot set, which is close to the
   pinned-slot mechanism it already has.

---

# Weekly plans that look like how people actually cook

**Status: components 1 and 2 shipped 2026-09-04; component 3 open.**
(Recorded 2026-08-27, reconstructed from a design conversation that was
never written down. The staging below was a proposal; components 1 and 2
have since been built and the entries say what actually happened.)

A generated week is 21 independently chosen recipes, and that is exactly
what it reads like: 21 shopping lists, 21 things to cook, no dish ever
seen twice. Real households do the opposite — they buy a cabbage and use
it twice, they eat the same breakfast most mornings, and they cook once
for two meals. Three changes, in increasing order of structural cost.

## What already exists (do not rebuild it)

- `build_preference_scorer` (`weekly_planner/planner.py`) already scores
  **shared perishables with meals already chosen**, weighted by the
  `waste_mode` slider (`off` / `reuse` 0.8 / `strict` 1.6 per shared
  token, capped at 4). Component 1 is a strengthening of this axis, not
  a new mechanism.
- `pantry_service.fetch_pantry_candidates` / `merge_pantry_pool` already
  turn a list of ingredient strings into candidates that contain them,
  coverage-first, with allergens/diet/cuisine/`max_minutes` riding along
  (`action_adapter.py:93-117`). Anything that can produce a list of
  ingredient strings gets recipe sourcing for free.
- `matched_items` is the only sanctioned source of a user-facing "uses
  your X" claim. The pantry module's standing rule — every claim comes
  from the matcher — governs all three components below.

## 1. Share ingredients across days

**Status: DONE.** Scoring half shipped 2026-08-27, sourcing half
2026-09-04 (CHANGES.md: "Weekly plans reuse ingredients without repeating
them" and "Weekly plans go looking for what they already buy, and
breakfast may come back").

The flat token set became a day-aware `IngredientBasket`; reuse is
rewarded at a gap of two days or more, penalised on the same or the next
day, and capped at two meals per ingredient. The monotony half applies at
every food-waste setting. No shelf life is modelled — nothing records
expiry, so no gap is ever "too old to count". Cross-day reuse is surfaced
as its own chip kind (`shared_ingredient`, "also uses Monday's cabbage")
and its own ledger row (`source: "the plan"`), kept apart from the
member's stated pantry ("uses your tomatoes", `source: "your pantry"`).

The sourcing half went in behind `strict`, as predicted below: before each
new day is fetched, `IngredientBasket.reusable_items` offers up to three
ingredients (gap and cap identical to the scorer's, so sourcing asks for
exactly what scoring would reward) and they get the pantry fan-out. The
member's own items are excluded from it — they have their own fan-out —
and the member's merge runs last so their coverage ranking still decides
the top of the pool. Both outcomes are recorded on `selection_events`
(`derived_pantry_sourced` per day, `derived_pantry_skipped` once), so
"why did this week reuse nothing" is answerable from the stored plan.

**The ask:** buying a bunch of dill for one Tuesday recipe is waste; the
week should route it through two or three meals.

**What shipped:** the offer is made by the planner
(`offer_derived_pantry`) before each new day's pool is fetched, and the
action space decides whether to spend the requests. Gated on `strict`
exactly as predicted — the fan-out is one HTTP request per item per day,
and making it fire on every weekly plan would be a latency cost paid by
everyone to strengthen an axis most members leave `off`.

One thing the plan did not anticipate: the ingredients had to be *named*
before they could be searched for. `perishable_tokens` splits on
whitespace, and "self" is not a search term any more than it is a chip.
So `nameable_phrases` moved out of `explainability` into `planner` and now
serves both — an ingredient worth naming to a member is exactly an
ingredient worth searching RecipeWrangler for, and two definitions of "an
ingredient" would have drifted apart within a release.

**What makes this honest rather than a lie:**

- We match ingredient *presence*, not amounts
  (`PANTRY_PLANNING_PLAN.md` §Risks). "Both meals use dill" is
  measurable. "Uses up the rest of the dill" is not, and must not be
  said.
- A derived pantry item is not a member statement. **Done:** the chips
  read as plan-internal ("also uses Tuesday's dill") and never as
  "uses your dill", which stays reserved for what the member actually
  told us. Keep it that way if the sourcing half lands.
- Shelf life is not modelled and should not be faked. Cabbage keeps three
  weeks, basil three days, and nothing in the state records a purchase
  date either way — so the spacing that shipped is justified as *variety*,
  never as freshness, and no wording should imply otherwise. A real
  perishability tier would be a separate piece of work with a real data
  source behind it.

**Cost (actual):** low, as estimated. The fan-out already existed; the
work was in the naming and in keeping the two merges in the right order.

## 2. Repeat favourites and breakfasts on non-adjacent days

**Status: DONE for breakfast, 2026-09-04** (CHANGES.md: "Weekly plans go
looking for what they already buy, and breakfast may come back"). Lunch
and dinner keep the original never-repeat rule; extending the cooldown to
them was not attempted, and dinner-twice-a-week remains open.

**The ask:** nobody eats seven different breakfasts. A favourite dinner
twice a week is a feature, not a failure.

**The blocker is a hard contract, in three places:**

- `RecipeActionSpace` excludes every committed id from every subsequent
  fetch (`exclude_recipe_ids=list(self._selected_ids)`,
  `action_adapter.py:90` and `:101`). Repeats are impossible at the
  *source*, not merely disfavoured.
- The module docstring asserts "a 7-day plan never repeats a recipe"
  (`action_adapter.py:5-6`), and `CHAT_ENDPOINT_PIPELINE.md:191` says
  the same.
- `variety_metrics` (`explainability.py:119-141`) scores distinctness
  and the prose says "All 21 meals are distinct recipes" as praise. An
  intentional repeat would render as a *degraded* week.

**What shipped:** a slot-scoped cooldown, breakfast only — a breakfast
may return after ≥ 2 days, at most twice in a week, never in another slot,
and never if it was pinned or downvoted (`mark_selected` still means
never).

The blocker turned out to be softer than this section assumed. A per-slot
cooldown looked like it needed per-slot fetches (3× the RecipeWrangler
calls), but the fetch is per *day* and serves all three slots: fetch with
the loosest exclusion any slot needs, apply the per-slot rule at selection
time. No extra requests at all, and there is a test asserting one fetch
per day so a regression to per-slot fetching cannot pass silently.

Two things had to be added that this section did not foresee:

- **The variety penalty had to stop fighting the cooldown.** −2 per shared
  title token, against an exact repeat, scales with how many words the
  recipe happens to be called — always enough to beat the cooldown. A
  sanctioned repeat is now exempt from its own earlier title, and from
  nothing else.
- **And the flat repeat penalty that replaced it had to go to zero.** It
  shipped at −1.0 to stop a repeat winning a coin flip. On a real profile
  there is no coin flip: with no favourites and no liked *ingredients*,
  almost every candidate scores exactly 0.0, so any penalty at all is a
  veto and the cooldown never fires. First live week: seven distinct
  breakfasts. At zero it measures ~1.25 repeats per week. The lesson
  generalises — this scorer's numbers only mean something
  relative to a spread, and a bare profile has none.
- **A repeat had to stop collecting the reuse bonus.** It shares every
  ingredient with its own earlier serving, so at `strict` the ingredient
  axis made repeating the cheapest possible way to score — observed: a
  strict week repeated a breakfast at the first legal opportunity, every
  time. A repeat now sits out that axis entirely.

**How the open questions were answered:**

- *Earned or merely allowed?* Both happen, and both are labelled.
  `repeat_source` is `member_request` when the recipe is one the member
  starred and `plan` otherwise, set at the only point that can justify it
  and carried through the chip, the ledger row, the variety metric, the
  prose and the response-writer facts. A week that repeated because the
  pool was thin says so.
- *`variety_metrics` and monotony.* It now reports `planned_repeats` and
  `repeats_by_source` separately from `unexplained_repeats` — a duplicate
  with no recorded reason (a pinned dish, a slot edit) is never folded
  into the sanctioned count, and gets its own `violated` ledger row.
  Repeats are also measured against the policy rather than asserted from
  it, so a repeat that reached the plate some other way is reported as
  out of policy.
- *The documented guarantee.* Retired in the same change:
  `action_adapter.py`'s docstring, `CHAT_ENDPOINT_PIPELINE.md`, and this
  file.

**Cost (actual):** medium, as estimated, but the work landed in a
different place than expected — the metric and prose changes were
straightforward, and the scorer interactions above were where the time
went.

**Still open here:** **resolved 2026-09-09** for the slots — lunch and
dinner repeats shipped as the member-set `repeat_meals` control; see the
section after component 3. A repeat earned by something other than a star
remains open (a member-stated liked dish is not a repeat authority).

## 3. Day N's dinner becomes day N+1's lunch

**Status: DONE, 2026-09-09** (CHANGES.md: "Cook once, eat twice — and
repeats become the member's dial"). Shipped together with component 2's
open half, because they turned out to be one mechanism.

**The ask:** cook once, eat twice. The single most common real-world
weekly pattern, and the one the model could not express at all.

**The decision this section was blocked on:** a leftover is a **flag on
an ordinary entry, holding the whole recipe** — not a new entry kind and
not a reference.

The reference model was the wrong shape for two reasons. The narrow one:
`WeeklyMealPlanEntryResponse.recipe` is a required dict, so a stub would
have moved service, gateway and UI in the same release for no gain. The
real one: the member *eats that dish*. Its nutrition, its ingredients and
its card are the dish's own, so an entry that holds them is the more
truthful representation, not merely the more convenient one. And the
repeat machinery from component 2 was already exactly "a flag on an
ordinary entry" — `repeat_of_day` + `repeat_source`, carried from the
candidate to the chip, the ledger row, the metric and the prose. A
leftover is a repeat with a slot transition, one day instead of two, and
a cap of its own. It inherits all of that rather than growing a parallel
path beside it.

**The predicted ripple, against what it cost:**

- *Planner skips the slot.* It doesn't. The leftover is one more
  candidate in the day's pool, rebuilt from the committed action rather
  than fetched — so it costs no request, is the dish on the plate by
  construction, and hard constraints, the tracker and the reward all
  work on it unchanged. It is **appended** to the pool, never
  substituted for it: the member asked that leftovers be possible, not
  that lunch stop being planned.
- *The tracker must count the meal while the shopping must not.* One
  branch, and it is the single place in the codebase where those two
  numbers had to diverge: `basket.add` is skipped. Everything else
  already counted meals rather than baskets. Adding it would have spent
  the ingredient's reuse allowance on a portion nobody bought, and
  charged the following day's dishes monotony for overlapping with a
  meal the member deliberately asked to eat twice.
- *Explainability iterates entries as independent dishes.* It still can
  — every entry has a real recipe. Three additions: the chip names the
  slot the dish was **cooked** in ("the same lunch as Monday" about a
  dinner is a lie the member catches in one glance), verification is
  against `(day, slot)` rather than the day alone, and leftovers stay
  out of `min_gap_days` — one is a day after its source by definition,
  so averaging it in would have made the ledger read its own feature as
  a violation.
- *Edits.* It cascades, and the reply says which day followed. Leaving
  the lunch alone was the alternative and is strictly worse: it would go
  on serving a dish the week no longer cooks, which every measurement
  reads as an unexplained duplicate. Dependents are matched on the
  recipe id *and* the slot, so a stale marker can never overwrite an
  unrelated meal.
- *Refinements.* Nothing to do — they rebuild all 21 slots, so leftovers
  are rebuilt with the week.
- *UI.* No change. A leftover card is not a different card; it gains a
  chip.

**Portion arithmetic: still not done, and now said out loud.** This
section called the version without it "weaker but *honest*, and might be
the right v1". It was. Nothing in the service records quantities,
purchase dates or shelf lives, and the pantry work refused quantity
claims for exactly that reason — so the wording everywhere is "Monday's
dinner again", never "the rest of it", "a double portion" or "half the
batch" (a test asserts the absence of each). The ledger row states the
gap itself: the plan doesn't track portions, so cook enough for two
meals if you want this. The response-writer facts carry
`portions_not_tracked` for the same reason, since a writer can only
decline to claim what it is told it does not know.

**Cost (actual):** medium, not the estimated high — and none of it was
where this section expected. The shape decision was most of the work;
after it, the mechanism was a candidate builder and a gate. The gateway
and the UI did not have to move at all.

## Component 2's other half, also done

Lunch and dinner repeats shipped here, as
`plan_parameters.repeat_meals`: one ordered control, `off` → `breakfast`
→ `all` → `leftovers`, each stop a superset of the one below it.

This section had it right that the mechanism was one line and the
frequency was a preference question. What it did not say is that the
preference question has no single answer: how often a *dinner* may recur
before a week reads as lazy rather than familiar differs by household,
not by request, and nothing in "plan my week" reveals it. So it is asked
rather than guessed — the same reasoning `food_waste` is built on, and
the same single-ordered-control shape, for the same reason ("repeats:
off, leftovers: on" would be a settings bug shipped as a feature).

The default is `breakfast`, which is exactly what the planner already
did, so an untouched card changes nothing. The gap and the cap stayed in
`action_adapter`: they are what stops a thin candidate pool from cashing
the member's setting in for monotony, and that is not a preference.

One thing this section did not anticipate at all: the cooldown decides
whether a recipe MAY come back, but whether it ever got the CHANCE was
RecipeWrangler's ranking. A day pool is a fresh fetch and the source
ranks unseen recipes, so on a real week an eligible dish simply never
reappeared — a live plan with the control set to its loosest stop
allowed a breakfast repeat at all four remaining slots and was offered
one at none of them. Fixed 2026-09-09: on an EXPLICIT setting the dish
is put in the pool rather than waited for, and paid enough to win a tie
(`_REPEAT_BONUS`). A default still waits for the source, because a
default is not a request. See CHANGES.md, "A repeat you asked for is
offered, not waited for".

**Still open here:** a repeat earned by something other than a star. A
member-stated liked dish is still not a repeat authority, so it comes
back labelled as the plan's own doing rather than the member's.

## Suggested staging

1. ~~**Component 1**~~ — done (scoring 2026-08-27, sourcing 2026-09-04).
2. ~~**Component 2**~~ — done for breakfast 2026-09-04; lunch, dinner and
   the member-facing control 2026-09-09.
3. ~~**Component 3**~~ — done 2026-09-09, without portion arithmetic and
   saying so.

# Plan scorer — paste a plan, get the metrics

**Status: implemented in this service 2026-09-15** — the `score_plan`
intent, steps 1–5, persistence and `POST /sessions/{id}/score-plan`
(CHANGES.md, "Plan scorer, steps 1–3" and "Plan scorer, steps 4–5"). Still
open: the gateway route and the UI text box and card (not in this
repository), the external guidelines endpoint, attaching the weekly judges to
generated weeks, and "adopt this plan".

Deviations from the text below, each for a reason found while building it:
- grounding uses a new unfiltered `SeedService.find_dish` rather than a flag
  on `_finalize_resolution` — the member's filters are applied in the search
  itself, one layer earlier;
- `ScoredPlan` is unchanged; a partial pasted day gets `None` from
  `DailyScoringInput.as_scored_plan()`, and the daily metrics are computed
  from course lists with the same functions;
- weekly metrics call the explainability functions one by one instead of
  `build_weekly_explainability`, which assumes seven days and counts snacks
  as meals; targets are scaled to the days pasted;
- a daily pasted plan also gets a calorie metric (`nutrition_metrics` for one
  day), because the routine existed and a day without it hid the obvious;
- an approximate match's catalogue tags are not used for categories or diet
  checks — the member's words outrank a recipe that merely resembles them;
- the explicit "rate this" bypass requires a structured listing and no
  request verb, so "what do you think of adding salmon for dinner and oats
  for breakfast?" still goes to the classifier;
- the three judges are ONE call (`agents.PlanJudge`), because three cost
  three prompts and three reasoning passes over the same plan — about 3,400
  input tokens against 2,100, and three requests against one;
- a dish no recipe gives calories to gets a typical serving written by a
  small model and profiled in RecipeWrangler's composition tables
  (`nutrition_source: "typical_ingredients"`), or, when that fails, the
  model's own calorie guess (`"model_estimate"`), labelled everywhere it is
  shown; the same guessed serving fills the food-variety count for a dish
  with no ingredient list, and each grounding row names its guesses in
  `guess_remarks`;
- a close title is only a match when it is the same dish: it keeps every
  part the member named and adds no food, diet or cuisine word; what the member wrote
  outranks a matched recipe's ingredients, and catalogue calories too low to
  be a meal are set aside;
- spellings of one dish ("lasagne"/"lasagna") normalise before comparison,
  and the other spelling is searched when the first query finds no match.

A member pastes a daily or weekly meal plan they wrote themselves — or got
from anywhere else — and FoodChat scores it with the same metrics it uses to
judge its own plans, with the reasoning behind each number. The score is a
third kind of card in the session, shown alongside the daily and weekly plan
cards, and reachable two ways:

- **as an intent.** A message that *contains* a plan ("here is what I eat
  this week: Mon breakfast oats, lunch …") or asks to score one is routed
  by the orchestrator to the scorer, like any other turn;
- **as an endpoint.** A dedicated text box in the UI posts to
  `POST /foodchat/sessions/{id}/score-plan`, skipping intent
  classification, exactly the way the compose canvas and the slider card
  already bypass it.

Both paths call one `PlanScorerService`, produce one payload, and persist
one assistant message with the score attached, so the card survives a
reload the way FoodScholar citations do. The point is comparability: a
plan FoodChat produced and a plan the member typed are graded by the same
code, in the same session, with the same profile.

## What already exists (reuse it, do not rebuild it)

The evaluation routines are all written; they are just buried inside the
two planning flows and take planner-internal objects as input. The scorer
is mostly an adapter that gets free text into those objects.

| Metric | Where it lives today | Input it expects | LLM? |
|---|---|---|---|
| Holistic fit score (1–5) | `agents.DocumentGrader.grade_daily_plans` + `prompts.GRADER_SYSTEM` | a *batch* of `(breakfast, lunch, dinner)` candidate triples, the query, the profile | yes |
| Food variety score (unique ingredient count) | `chat_service._food_variety_score` | `ScoredPlan` with per-course `ingredients` text | no |
| Meal diversity (1–5) | `agents.MealDiversityGrader.score` | rendered plan text | yes |
| Guideline adherence (1–5) | `agents.GuidelineAdherenceGrader.score` | rendered plan text + guidelines text | yes |
| Daily plan text rendering | `chat_service._plan_as_text` | `ScoredPlan` | no |
| Weekly variety (distinct recipes, repeats, unique ingredients, category mix) | `weekly_planner.explainability.variety_metrics` | entry dicts `{day, meal_idx, recipe: {...}}` | no |
| Weekly guideline checklist (fish 1–2×, red meat ≤3, ≥half plant-based) | `explainability.guideline_checklist` | category counts from `variety_metrics` | no |
| Weekly nutrition vs budget (totals, daily average, over/under/on-track, coverage) | `explainability.nutrition_metrics` + `calorie_budget_status` | entry dicts with `recipe.nutrition`, targets from `WeeklyNutritionalTracker(profile).targets` | no |
| Meal category / meat detection | `day_summary.classify_meal`, `day_summary.is_meat_meal` | `{title, ingredients, tags?}` dict | no |
| Profile constraints ledger (allergies, diets, dislikes) | `transparency.constraints_ledger`, `weekly_constraints_ledger` | profile dict | no |
| Allergen check | `candidates_client.allergen_conflict(text, allergies)` | ingredient text + allergy list | no |
| Diet-tag check | `candidates_client.normalize_diet_tags`, `diet_tag_status` | recipe tags + profile diet | no |
| Ingredient normalization | `chat_service._extract_ingredient_names` **and** `explainability._ingredient_names` (two copies of the same regex) | free text | no |
| Dish-name → recipe resolution | `seed_service.SeedService.resolve_seeds` (autocomplete → `fetch_recipe`, allergen-checked) | `[{name, meal_type?, day?}]` | no (HTTP to RecipeWrangler) |
| Per-recipe nutrition lookup | `candidates_client.CANDIDATES.fetch_details(ids)` | recipe ids | no (HTTP) |
| Non-chat entry that still yields a chat turn | `foodchat_router.compose_plan`, `apply_plan_parameters` | request body → `ChatTurnResponse` | — |
| Structured payload persisted on a message | `Message.attribution` (FoodScholar citations) | JSON column on `messages` | — |
| Persisted clarification with a service-specific `kind` | `FoodScholarService.CLARIFICATION_KIND`, `"edit_slot"`, `"favorites_offer"` | `sessions.clarification_state` | — |

Two of these shape the design:

- **`build_weekly_explainability` is already "score a finished week".** It
  takes an entry list and a profile, nothing from the planner's runtime (its
  docstring says so: it exists so slot-edited plans can be re-scored with no
  environment). With `selection_events=[]` it is the weekly scorer. The
  deterministic half of the weekly feature is: build entry dicts from text,
  call that function, hand back `metrics` + `constraints_applied` +
  `reasoning`.
- **`DocumentGrader` is comparative, not absolute.** It grades a batch, and
  its rubric leans on "the user's immediate query" and feedback history —
  neither of which a pasted plan has. The scorer's fit score is a separate
  judge that shares the rubric but grades one plan against the *profile*.
  See step 4.

## Guidelines: out of scope here

The daily adherence judge reads `chat_service.GUIDELINES_PATH`, which
points at a file that is not in the repo, so it scores with an empty
guidelines text today. That is left as it is. The working assumption is
that guideline text will come from an external endpoint later, so the
scorer reads it through one function, `plan_scoring.guidelines_text(scope)`
with `scope` in `{"daily", "weekly"}`, which for now returns the file read
(or `""`) for both scopes. When the endpoint exists, that function is the
only thing that changes. The two scopes are separate from day one because
the weekly judge is meant to be given frequency rules (fish twice a week,
red meat at most three times) that make no sense for a single day, and the
daily judge the per-day ones.

## The shape of the feature

```text
                 chat turn                          text box
  POST /sessions/{id}/chat                POST /sessions/{id}/score-plan
  { member_id, content }                  { member_id, plan_text,
        │                                   plan_type?: daily|weekly|auto,
        ▼                                   context?: "trying to eat less meat" }
  OrchestratorService ── intent ──┐               │
     "score_plan"                 │               │  no intent classification
                                  ▼               ▼
                        [PlanScorerService.score(session, text, plan_type, context)]
                                  │
        ├─ 0. profile    session.user_profile (fetched from WiseFood by
        │                member_id at session creation, merged with diners)
        │
        ├─ 1. parse      regex pre-pass + PlanTextParser (one FAST_MODEL call)
        │                → ParsedPlan{plan_type, days[{day, meals[{slot, title,
        │                  ingredients?}]}], unparsed[]}
        │                nothing parsed / ambiguous shape → clarification,
        │                kind="score_plan", pasted text kept in the state
        │
        ├─ 2. ground     SeedService resolution per dish (autocomplete →
        │                fetch_recipe) → recipe_id, ingredients, nutrition,
        │                tags   — matched | approximate | unresolved
        │
        ├─ 3. build      daily  → ScoredPlan-shaped courses
        │                weekly → entry dicts [{day, meal_idx, recipe}]
        │
        ├─ 4. score      hard constraints  allergen_conflict, diet tags (code)
        │                daily   _food_variety_score, MealDiversityGrader,
        │                        GuidelineAdherenceGrader, PlanFitGrader
        │                weekly  build_weekly_explainability,
        │                        WeeklyMealDiversityGrader,
        │                        WeeklyGuidelineAdherenceGrader, PlanFitGrader
        │
        └─ 5. respond    user message + assistant message persisted, the
                         PlanScore payload attached to the assistant message;
                         ChatTurnResponse{intent="score_plan", plan_score=…}
```

Nothing is written to a canvas. The pasted plan is scored, not adopted:
refine / edit intents keep targeting the member's own canvas, and
`plan_version` / `plan_parent_id` stay `None` on the turn. "Adopt this
plan" — promoting a scored plan onto the daily or weekly canvas as version
1 so it can be refined — is a natural follow-up and is explicitly not in
this cut (it needs a placement policy for unresolved dishes and a decision
about what a canvas built from someone else's plan claims in its ledger).

### The intent

`score_plan` becomes the tenth intent in `OrchestratorSchema`, the
`Intent` literal in `models/session.py`, and the orchestrator prompt. The
prompt rule, and what it must be told apart from:

- **`score_plan`** — the message itself contains a meal listing (dishes
  named per slot, optionally per day) that the member is presenting rather
  than requesting, or asks to score/rate/evaluate such a listing given in
  this or the previous message. "Here's my week, how does it look?",
  "rate this: breakfast oats, lunch lentil soup, dinner salmon".
- not `daily_plan` / `weekly_plan` — those *ask for* a plan; a message
  that *brings* one is a score request even when it also asks "is this
  ok?".
- not `nutrition_question` — "is my current plan healthy?" about the
  canvas goes to FoodScholar as before. A pasted plan is not the canvas.
  If the member asks a health question *and* pastes a plan, the plan
  wins: score it, and the summary can point at FoodScholar for the
  medical part.
- not `plan_question` — that is a lookup in the existing canvas.
- not `compose` — compose picks recipes by id on a blank canvas; the
  scorer never touches a canvas.

The orchestrator hands the raw message to the scorer; it does not extract
the plan itself (one LLM call per turn for routing stays the rule). The
turn also runs the usual memory-nudge extraction, since "I always have
oats on Monday" is a preference whether or not it arrived inside a plan.

### The endpoint

`POST /foodchat/sessions/{session_id}/score-plan`, body
`{member_id, plan_text, plan_type?, context?}`, response `ChatTurnResponse`
— same as `/compose` and `/plan-parameters`. Session-scoped, because the
card lives in the conversation and the profile comes from the session.
Same ownership guard (404 on mismatch) and error mapping as every other
session endpoint. `plan_type` defaults to `"auto"`; when the UI has a
daily/weekly toggle next to the box, it sends the explicit value and the
parser is told rather than asked. `context` is the optional free-text
"what I'm trying to do" that feeds the fit score.

The text box is the same message as the chat path with intent
classification skipped, so it also persists the user's pasted text as a
user message. The conversation then shows what was scored.

### Step 1 — parse free text into a plan (`PlanTextParser`)

New agent in `agents.py`, `ParsedPlanSchema` in `schemas.py`, prompt in
`prompts.py` registered through `_reg`. Runs on `FAST_MODEL`: span-picking,
not judgment, the same category as the plan-spec and seed extractors.

```text
plan_type:  "daily" | "weekly"      # what the text looks like, not what was asked
days:       [{ day: 1..7 | null, label: "Monday" | "Day 2" | null,
               meals: [{ slot: breakfast|lunch|dinner|snack|other,
                         title: str,
                         ingredients: str | null,     # only if the user wrote them
                         quantity_note: str | null }] }]
unparsed:   [str]                    # lines it could not place, verbatim
```

Rules the prompt must state, because each one is a way to lie about the
plan otherwise:

- Never invent ingredients. If the user wrote "chicken curry",
  `ingredients` is null; grounding fills it in from a real recipe or it
  stays unknown, and the metrics say what they were computed on.
- A day with one or two meals listed is a partial day, not an error.
  Unmentioned slots are absent, not "skipped" and not filled.
- `plan_type` follows the structure: two or more labelled days → weekly. A
  3-day plan is a "weekly" with 3 days; the checklist scales its targets by
  `total_meals` already.
- Anything it cannot place goes to `unparsed`, verbatim. Silent dropping
  is the failure mode to design against.

Deterministic pre-pass before the LLM: split on day-name / "Day N" headers
and `slot:` prefixes with a regex and pass the structure as a hint. Most
pasted plans are already shaped like that, and the regex path makes the
common case testable with no fake LLM.

**Clarification.** Two outcomes stop the flow and ask instead of guessing,
through the persisted state machine with `kind="score_plan"` and the
pasted text stored in the state so the member does not re-paste:

- zero meals parsed ("I couldn't find any meals in that — could you list
  them as `breakfast: …`, `lunch: …`?");
- `plan_type="auto"`, one unlabeled block of more than three meals (is it
  one day with snacks, or several days?).

The member's answer is routed by `kind` alone, no intent classification,
like every other clarification. A second failure returns the `unparsed`
lines in the reply and ends the loop; nobody gets asked three times.

### Step 2 — ground dishes against RecipeWrangler

Reuse `SeedService`'s resolution path, factored so it can be called
without the seed-extraction LLM step: `resolve_seeds` already takes
`[{name, meal_type, day}]` dicts, which is exactly the parsed output. Each
dish ends up in one of three states, and the response says which:

| state | meaning | what gets scored |
|---|---|---|
| `matched` | autocomplete hit with a confident title match | the RecipeWrangler recipe: its ingredients, nutrition, tags |
| `approximate` | a hit, but title similarity below a threshold | the recipe's nutrition and tags, but the user's own ingredients if given; flagged |
| `unresolved` | nothing found | title + whatever ingredients the user wrote; no nutrition; category from keywords only |

The threshold is the part `SeedService` does not have today — for seeding
a near-miss is fine because the member sees the pinned dish and can
object; for scoring, a near-miss silently changes the number. Normalized
token overlap on the title is enough to start; the matched title is in
the response so the member can see what was assumed.

`SeedService` today *drops* a dish that conflicts with an allergy. The
scorer must not: it grounds the dish anyway and reports the conflict (step
4). Factor the allergen decision out of `_finalize_resolution` behind a
flag rather than duplicating the resolution code.

### Step 3 — build the objects the existing scorers want

- **Daily.** `MealCourse(recipe_id, title, ingredients, directions="",
  nutrition, …)` per slot → a `ScoredPlan` so `_plan_as_text` and
  `_food_variety_score` work unchanged. Missing slots: `ScoredPlan` assumes
  three courses. Either allow `None` there or build a placeholder course
  the text renderer omits and the variety count ignores. The first is
  cleaner; do it only if `ScoredPlan` has few consumers (check first).
- **Weekly.** Entry dicts `{day, meal_idx, recipe: {recipe_id, title,
  ingredients, nutrition, tags}}` in the shape `environment.py` produces;
  `explainability._recipe` / `_title` / `_ingredients` read both key
  spellings already. No `pinned`, no `repeat_of`, no `selection_events`,
  so every duplicate would come out `unexplained`. That is wrong for a
  plan the member wrote (they *chose* the repeat), so tag repeats with a
  new source label, `REPEAT_BY_AUTHOR`, and a ledger row that says the
  repeats are the member's own. Do not reuse `REPEAT_BY_MEMBER`, which
  means "a starred recipe came back".
- **Profile.** `session.user_profile` — fetched from WiseFood by
  `member_id` when the session was created and merged with any diners
  since. Allergies, diet, dislikes, likes, goals, nutrition profile and
  the calorie target all come from there, as they do for the planners.
  No profile-less mode: the endpoint is session-scoped and a session
  always has a member.

### Step 4 — score

**Hard constraints first, in code, not in a prompt.** Over the grounded
dishes:

- `allergen_conflict(ingredients, profile.allergies)` per dish → a
  `violated` ledger row naming the dish, the allergen, and (household)
  the diner it protects, from `constraint_origins`.
- diet: `classify_meal` / `diet_tag_status` against `profile.diet` → a
  `violated` row for a meat dish on a vegetarian profile, etc.
- `food_dislikes` present in grounded ingredients → a `soft` row,
  `violated`.

These rows go into `constraints_applied` next to the usual profile rows
from `constraints_ledger`. Today the daily ledger only ever says
`satisfied` because the planner filtered at fetch time; the scorer is the
first daily consumer that can produce `violated`, and the UI already
treats unknown statuses as informational.

**Fit score — `PlanFitGrader`.** A new judge, not `DocumentGrader` with a
batch of one: it grades a single plan against the profile, on the same
1–5 rubric text factored out of `GRADER_SYSTEM` into a shared constant so
the two prompts cannot drift. Inputs, in this order of weight:

1. allergies — listed as *hard*: the prompt is told any presence means
   the plan fails, and the code caps the score at 1 whenever step 4's
   allergen rows are non-empty, whatever the model returned. A hard
   constraint is not a matter of judgment;
2. diet — same treatment, cap at 2 (mirrors the existing rubric's
   "implausible slot → at most 2");
3. preferences, likes, dislikes, dietary goals, nutrition profile and the
   calorie target — soft, weighed by the model;
4. `context` — the member's stated aim for this plan, if any; otherwise
   the prompt says none was given and the goals stand in for it.

Same `ScoringSchema` output. One prompt for both plan types; the user
message says how many days the plan spans and groups the text by day. The
reasoning must name the dish behind every deduction, as the daily grader's
already must.

**Daily**, in the order `ChatService._compute_metrics` uses, so the numbers
are the numbers the daily canvas carries:

1. `_food_variety_score` (LLM-free)
2. `MealDiversityGrader.score(plan_text)`
3. `GuidelineAdherenceGrader.score(plan_text, guidelines_text("daily"))`
4. `PlanFitGrader.score(plan_text, profile, context)`

Hoist `_compute_metrics` out of `ChatService` into
`plan_scoring.compute_daily_metrics(plan, guidelines_text, graders)` so the
chat flow and the scorer call one function. Same for the duplicate
ingredient normalizer: keep `explainability._ingredient_names`, delete
`chat_service._extract_ingredient_names`, import the one.

**Weekly**:

1. `build_weekly_explainability(entries, profile, selection_events=[])` →
   `metrics` (`variety`, `guideline_checklist`, `nutrition`, `days`),
   `constraints_applied`, `reasoning`. LLM-free, unchanged.
2. `WeeklyMealDiversityGrader` — a new prompt, not the daily one with a
   preamble. The daily prompt reasons about three meals; a week is 21 and
   the questions differ: does the same protein source carry every dinner,
   do cuisines rotate or sit in one place, does produce variety hold
   across days rather than within one, are breakfasts a rut or a
   sensible routine (a repeated breakfast is not a diversity failure —
   the planner's own `repeat_meals` setting says so). Output stays
   `ScoringSchema` so the metric card is identical in shape.
3. `WeeklyGuidelineAdherenceGrader` — its own prompt, fed
   `guidelines_text("weekly")`. Where the daily judge asks "does this
   day have enough vegetables and whole grains", the weekly judge asks
   the frequency questions the checklist already asks deterministically
   — fish, red meat, legumes, plant-based share — plus balance across
   the week. The deterministic checklist is passed in as facts so the
   judge explains rather than recounts; its score must not contradict a
   checklist row it was handed.
4. `PlanFitGrader` on the week.

Both weekly judges are constructor-parameterized variants of the existing
classes (`MealDiversityGrader(prompt=WEEKLY_MEAL_DIVERSITY_SYSTEM)`), so
existing constructor calls stay unchanged. Once they exist,
`WeeklyPlanService` can attach them to the plans it generates so a
generated week and a pasted week carry the same numbers — that is a
separate change, since it adds two Groq calls to every weekly plan.

**What the judges see.** Grounded ingredients when a dish is `matched`,
the member's own text otherwise, and the plan text handed to every judge
marks which is which ("ingredients from recipe *X*" vs "as written"). No
judge is ever told a grounded ingredient list is what the member wrote.

### Step 5 — response and persistence

```text
PlanScore                                   # attached to the assistant Message,
  plan_type:        "daily" | "weekly"      # JSON column like `attribution`
  days_scored:      int
  meals_scored:     int
  scored_plan:      MealPlanResponse | WeeklyMealPlanResponse shape,
                    origin="pasted", no id lineage   # so the plan card component renders it
  metrics: [ { key: "fvs" | "diversity" | "guideline_adherence" | "fit"
                    | "weekly_variety" | "weekly_guidelines" | "weekly_nutrition",
               label: str,
               score: number | null,
               kind: "likert5" | "count" | "percent" | "checklist" | "status",
               reasoning: str,
               detail: dict } ]             # checklist rows, totals, coverage, cap applied
  constraints_applied: [ledger rows]        # profile rows + violated rows from step 4
  grounding: [ { slot, day, title_given, title_matched, recipe_id, state,
                 has_nutrition } ]
  unparsed:  [str]
  warnings:  [str]                          # "3 of 21 meals had no nutrition data", ...
```

`ChatTurnResponse` gains `plan_score: Optional[PlanScoreResponse]`, and
`/conversation` returns it on the message it belongs to, the way
`attribution` rides today. `content` is the summary prose: 2–4 sentences
from `ResponseWriter` over a facts dict (metric reasonings, violated
rows, warnings) — the writer's contract already forbids claiming what it
was not told, which is the property a scorer summary needs most.

`metrics` is a list of uniformly shaped items rather than the flat
`fvs_count` / `fvs_reasoning` / … fields `MealPlanResponse` has, because
the score card renders N rows and should not know the metric names in
advance. `scored_plan` reuses the existing plan-card shape precisely so
the UI can show the pasted plan next to the daily/weekly cards with the
component it already has, with `origin="pasted"` to suppress the
refine/edit affordances.

### Gateway and UI

New endpoint → gateway route → UI text box, in one change. The gateway
also needs to know this turn costs 4–5 Groq calls with no candidate fetch
in front of it; the session message cap applies as usual since both paths
persist messages. `CHAT_ENDPOINT_PIPELINE.md` gets the `score_plan` branch
in the intent table and the `/score-plan` entry next to `/compose`.

## Tests (LLM-free, as always)

- `tests/test_orchestrator_routing.py`: a message carrying a meal listing
  routes to `score_plan`; "is my plan healthy?" with no listing still
  routes to `nutrition_question`; a clarification with `kind="score_plan"`
  bypasses classification.
- `tests/test_plan_scorer.py`
  - regex pre-pass: day headers in three styles (`Monday`, `Day 2`,
    `Tue:`), slot prefixes, a plan with no day headers → daily.
  - parser fake returning a fixed `ParsedPlanSchema`; `unparsed` survives
    to the response verbatim; zero meals → clarification state with the
    pasted text stored; the answer continues without re-pasting.
  - grounding with a fake `RecipeCandidatesClient`: matched / approximate
    / unresolved; an unresolved dish still gets a category from
    `classify_meal` on its title; an allergen-conflicting dish is grounded,
    not dropped.
  - allergen in a pasted dish vs a profile with that allergy → a
    `violated` row naming both, and the fit score capped at 1 whatever
    the fake grader returned; a meat dish on a vegetarian profile → cap 2.
  - weekly with a deliberate repeat → `planned_repeats == 1`,
    `unexplained_repeats == 0`, source is the author's.
  - weekly with 3 days → checklist targets scaled to 9 meals.
  - fake graders (same pattern as the `ChatService` tests) → the daily
    `metrics` list has exactly the four keys in the documented order; the
    weekly list has its four.
  - judge input text marks grounded vs as-written ingredients.
  - both entry points persist a user message and an assistant message with
    `plan_score` attached; `plan_version` is `None`; no canvas is created.
- `tests/test_weekly_explainability.py`: one case calling
  `build_weekly_explainability` on hand-built entries with no
  `selection_events`, if there is not one already — it is the contract the
  scorer depends on.
- Regression: `ChatService._compute_metrics` callers unchanged after the
  hoist.

## Decided

- Fit score is graded against the profile fetched by member id, with
  allergens hard (cap 1, in code), diet hard (cap 2), everything else
  soft, and `context` as the optional query.
- Judges see grounded ingredients when matched and the member's text
  otherwise, labelled as such.
- No cache.
- Weekly diversity and weekly adherence get their own prompts; weekly
  adherence is designed for a weekly-scope rules text that does not exist
  yet.
- Guidelines stay as they are behind `guidelines_text(scope)`; an external
  endpoint will supply them later.

## Staging

1. Hoist `compute_daily_metrics`, dedupe the ingredient normalizer, add
   `guidelines_text(scope)`. Pure refactor with existing tests as the net.
2. Parser (regex pre-pass + `PlanTextParser`) + grounding + the
   `score_plan` clarification kind, returning `grounding` / `unparsed` /
   `warnings` only — no scores yet. This is the part with the most ways to
   be wrong, and it is checkable by eye.
3. Hard-constraint rows + `PlanFitGrader` with the shared rubric.
4. Weekly scoring: `build_weekly_explainability` + author repeats + the two
   weekly judges.
5. Daily scoring via the hoisted function.
6. `score_plan` intent in the orchestrator; `/score-plan` endpoint;
   `plan_score` on the turn and the message; gateway; UI text box and
   card; `CHANGES.md` entry; `CHAT_ENDPOINT_PIPELINE.md`.
7. Follow-ups, not in this cut: "adopt this plan" onto a canvas; attach the
   weekly judges to generated weeks.
