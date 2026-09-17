# FoodChat — Change Log (newest first)

---

# Guideline adherence: a judge that fits the minute, and scores that cannot fail a plan

> **Date:** 2026-09-17
> **Branch:** main
> No API change. New optional env var `GUIDELINES_MAX_CHARS` (default 3500).
> A quality score whose judge call fails is now `0` (not scored) instead of the
> whole turn answering 500.

Found by running the previous change through the app end to end, on the demo
gateway and Groq's on-demand tier.

| What | Detail |
|---|---|
| `plan_quality.metrics` → `_judged` | Each judge call (diversity, guideline adherence) is caught on its own. Before, a rate-limited judge raised through `_compute_metrics` and the member got a 500 for a plan that had already been chosen and graded. Now a failed judge's score is `0` with empty reasoning, which is what the judges already return for a reply they cannot parse, and the other scores and the plan stand. Weekly already caught this around its whole verification block; daily and structured did not. |
| `guidelines_service.render` | Budget `GUIDELINES_MAX_CHARS` = 3,500 characters of rule text (was 7,000). Live, the 77-rule Irish daily set made the adherence request 3,930 tokens, on top of 4,321 already spent grading candidates, against an 8,000-per-minute limit, and the score came back empty after ~30 s of retries. Rules that state a frequency ("a day", "each meal", "once", `frequency` set) now come before those that do not, then by type. A rule too long for the remaining budget is skipped rather than ending the list. The Irish daily list is now 45 rules (~4.3k characters): the frequency rules for grains, dairy, protein, vegetables and fruit, fluids and treats first, and portion definitions last. |
| `.gitignore` | `.wisefood_demo_token` (a live bearer token) and `.idea/`. |

## Verification

New tests: a failed judge leaves the other scores; a frequency-stating untyped
rule outranks a typed one that states none; an over-long rule is skipped, not
the end of the list. Full suite, rebased on main: 2,325 passed, 1 skipped. `test_platform_client` passes on wisefood
0.0.26, the version `requirements.txt` pins; its earlier failure here was a
local install of 0.0.8/0.0.9. Ruff clean.

**Live** (local demo harness, demo data API, on-demand Groq): a daily plan
returned 200 with FVS 40 and guideline adherence 2, citing G1–G4, G8, G5/G11
and G16. The diversity judge was rate-limited that turn and scored 0 instead
of failing the turn. A 7-day weekly plan returned 200 with the three built-in
checklist rows (all met), FVS 146, diversity 4, and guideline adherence 2
against the 81 weekly Irish rules, citing them. The harness's own stand-in
client was brought level with `PlanClient` (`tags`, `offset`,
`deepen_multiplate_only`, and `find_recipes` filters) so plans could be
generated at all. It is untracked and not part of this commit.

---

# Guideline adherence reads the WiseFood data catalog

> **Date:** 2026-09-17
> **Branch:** main
> No API change. `guideline_adherence_score` / `_reasoning` are now judged
> against real rules, and weekly `metrics.guideline_checklist` rows can come
> from the catalog. New env var `GUIDELINES_DEFAULT_REGION` (default `IE`);
> `DATA_API_URL` now takes the data API's root (demo:
> `https://demo.wisefood-project.eu/dc`).

Every guideline-adherence judge graded against nothing: the daily and weekly
quality calls passed no guideline text, and the scorer's came back empty. The
catalog client written for this matched nothing in the real catalog, and it
read the SDK's `Response` object as the payload, so it returned `[]` on every
call. Probed against the live demo first (plan and API facts: IDEAS.md,
"Guideline adherence from the WiseFood data catalog").

| What | Detail |
|---|---|
| `models/guidelines.py` (new) | `GuidelineScope`: regions, life stage, plan type, guide URNs, rule ids. It is the one place filter strings are built, in the syntax verified live: `guide_region:(IE)`, `life_stage:(adulthood) OR (*:* -life_stage:*)` (the shorter `OR -life_stage:*` matches nothing), quoted URNs, `status:active`, no activity rules, and no weekly or monthly rules for a single day. A `rule_ids` scope ignores every other facet. `Guideline`: the typed row. |
| `backend/catalog.py` | `search(scope) -> list[Guideline]` replaces `search_guidelines(filters)`. Reads `.json()`, pages to `total`, dedupes identical wordings, caches per scope. A failure is not cached: it starts a 30 s backoff, so a catalog that is down does not cost every plan turn a timeout. Never sends `fl` (500 on the API). Always sends one facet field, because facets cannot be turned off. Logs in with `backend.platform.credentials_from_env`, the same identity as member profiles; username/password now works, where it previously required client credentials. |
| `backend/platform.py` | `credentials_from_env()` hoisted out of `WiseFoodPool._get_credentials`. Behaviour unchanged. |
| `services/guidelines_service.py` | `resolve_scope(profile, plan_type, override)` defaults to Ireland/adults and reads `region` and `age_group` / `life_stage` from the profile once it carries them. `guidelines_text` → `render`: numbered `[G1]…` rules under a header naming the scope and guides, food-based rules first, capped at 90 rules / 7,000 characters (the Irish adult set, 77–81 rules, fits whole), `""` on any failure. `facets_for` and `prose_context` are gone. |
| Weekly checklist rows | `split` keeps only rules a meal-category count can check: fish, red meat or poultry (what `classify_meal` counts), read from `topic` or from text naming exactly one of them, with a weekly floor ("at least once"), ceiling ("limit to 3", "no more than 2") or range ("1-2 times"). A bare count is not a bound: "Offer oily fish once a week" would fail a week with two fish dinners, and "Offer red meat 3 times a week" a week with one. Those rules stay prose for the judge, so the Irish adult set yields no rows and the checklist keeps its three built-in rules, as members see today. Before, "vegetables" and "fruit" rules would have been counted against categories nobody counts and reported `actual: 0`. Structured-quantity operators follow the catalog enum (`lt/lte/gt/gte/eq/approx`). "Limit red meat to 3 times a week" reads as a ceiling. |
| `services/plan_scoring.py` | `guidelines_text` is `guidelines_service.guidelines_text`. The local guideline file path and its read are removed. |
| `ChatService._compute_metrics(session_id, plan, profile, plan_type="daily")` | Passes the member's daily rules. The structured path passes `"weekly"` for a plan of more than one day. |
| `weekly_plan_service` | Weekly `plan_quality.metrics` gets the weekly rules. |
| `PastedPlanScorer` | `guidelines(plan_type, profile)`: the pasted plan is judged against the member's rules. |
| `weekly_planner/explainability.guideline_checklist` | Asks for the weekly scope. The three built-in rules remain the fallback. |
| Prompts | Unchanged. The rendered text says to cite rule ids and that a rule a plan cannot show is not a failure. Pinned hashes and Langfuse prompt names stand. |
| `.env.example`, `CHAT_ENDPOINT_PIPELINE.md` | Data API section rewritten; the guideline-text lines now name the catalog. |

## Verification

`tests/test_guidelines.py` was rewritten (75 tests). The old tests passed
with filters that matched nothing, because their rows were hand-written.
Parsing now runs against a recorded, trimmed demo response
(`tests/fixtures/guidelines_search.json`, 9 IE/HU rules), and the scope's
filter strings are pinned exactly as the live API accepted them. The tests
also cover: paging, dedupe, the failure backoff, an unconfigured
catalog asking nobody, the render header / order / cap, the daily and N-day
metrics receiving the right rules, checklist rows from the real rules, and
the Irish set leaving the built-in checklist in place. The two tests that stub
`_compute_metrics` follow its new signature.
Full suite: 2,220 passed, 1 skipped, 1 failed — `test_platform_client`, as on
main (the installed client does not take `telemetry`). Ruff clean on the
touched files.

**Live** (demo data API, `.env` username/password): Ireland/adults returns 77
daily and 81 weekly rules; Hungary 162; a two-id subset 2; Slovenia for a
`senior` profile 34. One
`GuidelineAdherenceGrader` call on a fry-up / ham baguette / burger day with
the Irish daily text (6,985 characters, 77 rules) scored it 1 and cited rules
G5–G76 by id.

**Live, through the app** (local demo harness; the `telemetry` SDK mismatch
shimmed in a throwaway launcher): `/score-plan` for a 3-day plan fetched the
81 weekly Irish rules; the judge scored adherence 3 and cited G11, and the
checklist kept its three built-in rows (all met). A fry-up day fetched the 77
daily rules and scored 1, citing G6. Plan generation could not be exercised:
the untracked demo harness fails on `plan_meals(tags=…)` before any
generation step, including the metrics.

Not done: client-credential login against `/dc`, region and age group in
the profile, a way to switch scope from outside, and per-rule verdicts
(IDEAS.md, "Left to do").

---

# Plan scorer: merged with three weeks of planning

> **Date:** 2026-09-17
> **Branch:** main
> No API change beyond the plan scorer's own.

The scorer was built on a main that had since moved 53 commits (the security
model, turn guards, `plan_quality`, agent tools). What changed to fit it in:

| What | Detail |
|---|---|
| `services/plan_scoring.py` | Main had hoisted the planner's metrics into `services/plan_quality.py` independently. The scorer's module no longer carries a copy: `ingredient_names` IS `plan_quality.extract_ingredient_names`, and `food_variety_score` counts the same items over any list of dishes. `compute_daily_metrics` and `plan_as_text` are gone; `chat_service` and `weekly_planner/explainability` are exactly main's. |
| `POST /score-plan` | `_require_member` first (identity before service availability), and the turn returns through `_finalize_turn`, so turn extras persist like on every other turn endpoint. |
| `OrchestratorService.score_plan` | The fifth turn entry point: one in-flight turn per session, a turn budget, and the intake memo cleared, like `process`, `apply_plan_parameters`, `regenerate` and `compose_plan`. |
| `messages.plan_score` | Beside main's `messages.extras`; both columns migrate. |
| `PlanJudge`, `PlanTextParser`, `DishIngredientEstimator` | Routed through `as_json_messages`, replacing their own "json" check, and added to `scripts/smoke_agents.py`. |
| Tests | Main's structural tests now count five turn entry points and audit `score_plan` for identity and ownership; the scorer's non-regression tests compare against `plan_quality` instead of a verbatim copy of the old `ChatService`. |

## Verification

Full suite: 2,193 passed, 1 skipped, 1 failed — `test_platform_client`, which
fails identically on a clean checkout of main (the installed WiseFood client
does not take `telemetry`). Ruff: no findings beyond main's. Live after the
merge: a daily plan over `/score-plan` and a 3-day week over chat scored as
before the merge.

---

# Plan scorer: what a live battery of pasted plans got wrong

> **Date:** 2026-09-17
> **Branch:** main
> `plan_score.grounding[].guess_remarks` can carry two new sentences (a
> catalogue figure set aside; a profiled figure far off the guess). Some dishes
> that were `matched` are now `approximate`. No schema change.

Eleven live cases against Groq and the WiseFood demo catalogue, over HTTP:
daily and weekly plans, dish names only and full pasted recipes, `/score-plan`
and chat (with and without a scoring word), plus three messages that must not
be scored. All returned 200 with every metric scored and every intent right;
reading the scored dishes showed what follows. Fixed, then run again.

## Which recipe a dish is

| What | Detail |
|---|---|
| `grounding.same_dish`, `parsing.dish_heads` | A title at Dice ≥ 0.75 is `matched` only when the recipe's name keeps every part the member named and adds only `DESCRIPTIVE_WORDS`. Live: "roast chicken with potatoes" matched "Roast potatoes" (64 kcal, and the vegetarian check lost the chicken); "caesar salad" matched "Chicken Caesar salad"; "avocado toast" "Avocado ricotta toast"; "beef tacos" "Beef and kimchi tacos". A generic word at the end of a part ("salad", "soup") passes to the word before it, so "Tuna nicoise salad" is still "Tuna Niçoise". Hits that are the same dish rank above closer-worded ones. |
| Diet and cuisine words are not descriptive | Tried and removed within the battery: "caesar salad" then matched "Vegan caesar salad" and failed a tree-nut allergy on its cashews, and after that "Mexican Caesar salad", counted as red meat. |
| `grounding.borrowable` | A generic name lends calories only when it also keeps every part: "Hummus" no longer stands in for "houmous and pitta" (47 kcal), nor "Roasted vegetables" for "baked cod with roasted vegetables". |
| Member-written ingredients win | A matched recipe no longer replaces what the member wrote: "Banana oat pancakes (2 bananas, 2 eggs, 50 g oat flour)" was failed for the milk in the catalogue's "Banana Pancakes". The recipe's allergens, ingredients and tags are then not used; its calories still are. |
| `parsing._meal_from_segment` | "(150 g yogurt, 20 g walnuts, 1 tsp honey)" is ingredients. It started with a digit and was read as an amount, so a whole pasted week had no ingredient lists. A bracket with a comma is a list. |

## Which calories to believe

| What | Detail |
|---|---|
| `MIN_MEAL_KCAL` (120), `MIN_SNACK_KCAL` (40) | Catalogue per-serving figures below these are set aside and the dish estimated instead (`rejected_recipe_kcal`, with a remark). Live: Quick Chili 12, Tomato Basil Soup 24, Crunchy fruit and yoghurt 47, Quick Chicken Stir-Fry 49, Scrambled egg on toast 61, Pancakes 97, Quinoa Salad 11, Mushroom Stroganoff 7. |
| `MAX_PROFILE_DISAGREEMENT` (2.0) | A profiled typical serving more than double off the model's own guess is not used; the guess is, with a remark naming the discarded figure. Probe: pho profiled at 2,025 kcal (guess 600), minestrone 1,138 (guess 500), both with full coverage. |
| `MAX_ESTIMATED_DISHES` (21) | The one estimator call covers a week's dishes; only profiling stays at `MAX_PROFILE_CALLS`. A 21-dish week had left its last three dishes with no calories. |
| Parser prompt, estimator prompt | "Serves 2" in a pasted recipe goes to `quantity_note`; the estimator divides a whole recipe's quantities down to one serving and lists each ingredient once (it had repeated "black pudding"). |
| `PROFILE_TIMEOUT_SECONDS` 12 → 15 | Four parallel calls returned at 11.9–12.0 s on the demo gateway. |

## Allergies and the reply

| What | Detail |
|---|---|
| `PLANT_DAIRY` in `allergen_conflicts` | "coconut milk" was lactose and dairy. The mask the diet check already used now applies to dairy and lactose allergies too, and only to them: peanut butter is still peanuts. |
| Possible allergens in the judge's conflicts | Marked `POSSIBLE ONLY` and "not a broken hard constraint", and the judge prompt says so. Live, muesli that only a catalogue recipe put almonds in scored fit 1 "for breaking your allergy"; rerun, fit 4 with the warning. |
| `service.allergen_sentences` | One sentence per certainty and set of allergens, naming every dish: "“Walnut yogurt”, “Cashew stir fry” and “Almond porridge” contain tree nuts". A three-day week had repeated the sentence three times. A reply writing "tree‑nut" with a non-breaking hyphen counts as naming it. |
| `service.ensure_calorie_caveat` | A reply quoting kcal without a word that they are partly known or estimated gets "Calories for 2 of 3 dishes are estimates, not recipe figures." (or "known for only N of M dishes"). The writer had the instruction and ignored it twice. |

## Verification

Tests for every row above in `tests/test_plan_scorer.py` and
`tests/test_plan_scoring.py`. Full suite: 951 passed. Ruff: no new findings.

Live, rerun after the fixes: routing still right in all eleven (planner
request -> daily_plan, "adding salmon for dinner" -> not scored, small talk ->
chat). Confirmed on the live data:

- 7-day week: "roast chicken with potatoes" now breaks the vegetarian diet as
  chicken; all 21 dishes have calories (was 18); 61% of target, was 69% with
  the 7–97 kcal catalogue figures in it;
- pasted recipes: no dairy verdicts on coconut milk or on a matched recipe's
  milk (fit 1 -> 4); the "serves 2" curry 438 kcal a serving (was 900 and
  then 1,250);
- a possible allergen (muesli) keeps its warning and no longer sets fit to 1;
- pho 619 kcal (was 1,349); replies quoting calories now say they are
  estimated; the tree-nut warning appears once.

Still imperfect, and not changed here: the catalogue's own figures above the
floor are taken as given (Thai green curry 1,304 kcal), and the weekly ledger's
default "at most N meat meals" row reads to the writer as the member's own limit.

---

# Plan scorer: calories per dish, and guessed servings in the variety count

> **Date:** 2026-09-17
> **Branch:** main
> `plan_score.grounding[]` rows gain `kcal`, `typical_ingredients` and
> `guess_remarks`; `fvs` detail gains `dishes_with_guessed_ingredients`, as
> does `weekly_variety` detail when it applies. Additive; rows stored before
> this read back with the defaults.

| What | Detail |
|---|---|
| `GroundedMeal.kcal` → row `kcal` | Calories per serving, rounded, from whichever source `nutrition_source` names; `null` when unknown. The UI can show where a day's total comes from. |
| `GroundedMeal.typical_ingredients` → row `typical_ingredients` | The estimator's serving, `[{name, quantity}]`. It is now written for dishes missing ingredients too, not only calories (one call either way; dishes missing calories are first under `MAX_PROFILE_CALLS`). A dish that already has calories is not profiled. |
| `GroundedMeal.guess_remarks()` → row `guess_remarks` | One plain sentence per guess in the row: calories estimated from a typical serving; calories a rough model guess; ingredients a typical serving that counts towards variety but is not checked against allergies, diet or dislikes. Empty when nothing is guessed. |
| `scoring._daily_measured` (`fvs`) | A dish with no ingredient list counts its typical serving's ingredient names. The sentence names those dishes: "2 dish(es) (“fried eggs”, “pasta with zucchini”) are counted with a typical serving's ingredients — a guess, not what you wrote or a recipe". Without guesses the count is computed exactly as before. |
| `scoring.with_guesses` (`weekly_variety`) | The same for a week's unique-ingredient count over main meals. `variety_metrics` is the planner's and is not changed: only `unique_ingredients` and its sentence are recounted; distinct recipes and meal categories still come from the member's words and matched recipes. |
| `scoring._ingredient_line` | The judge reads a guessed serving as "Ingredients not known; a typical serving might contain (a guess, not the user's words): …". |
| Guesses are not evidence | `ingredients`, which the allergy, diet and dislike rows read, is never filled from a guess. A test pins that a guessed "peanuts" leaves the peanut allergy row satisfied. |

## Verification

New tests: the row's kcal, serving and both remarks; the rough-guess remark;
the member's own ingredients kept over the guess; a dish with calories but no
ingredients getting a serving without a profiling call; an empty row; dishes
missing calories first under the cap; `fvs` and weekly counts with guesses and
their sentences; the judge's line; a guessed allergen producing no verdict.
Full suite: 918 passed. Ruff: no new findings.

**Live** (`POST /score-plan`, same three dishes): 200 in 9.8 s, 3 model calls,
3,350 tokens. Fried eggs `kcal: 180`, `typical_ingredients`, two remarks. The
demo profiler answered 503 for the pasta this time, so it carried
`nutrition_source: "model_estimate"`, `kcal: 450` and the rough-guess remark,
which is the guess path running live for the first time. The soup `kcal: 345`
from its recipe, no remarks. `fvs` went from 10 to 14 and names both guessed
dishes.

---

# Plan scorer: calories from typical ingredients

> **Date:** 2026-09-17
> **Branch:** main
> `plan_score.grounding[].nutrition_source` values change: `profiled` is gone,
> `typical_ingredients` and `model_estimate` are new. No other API change.

A live paste of fried eggs, pasta with zucchini and chicken noodle soup came
back with calories for one dish of three. The two loose matches were different
dishes, so they rightly lent nothing; the profiling fallback then timed out
twice at 60 s and held the turn for two minutes. Once the endpoint answered,
three more faults showed, and the reply called the one-dish figure "a large
calorie shortfall".

## Where the calories come from

| What | Detail |
|---|---|
| `agents.DishIngredientEstimator` + `DISH_ESTIMATOR_SYSTEM` / `_USER` | One call on the fast model (`openai/gpt-oss-20b`, a separate Groq budget from the judge's) for every dish still without calories: a typical single serving with quantities, keeping the member's own ingredients and amount, plus a calorie guess. `schemas.DishEstimatesSchema`. Returns `{}` on any failure. |
| `grounding.DishGrounder._estimate_missing` | Replaces `_profile_missing`. Each ingredient list goes to the profiler as `Title / Serves 1 / one ingredient per line`. Reliable figures become the dish's nutrition (`typical_ingredients`); otherwise a guess between 20 and 2,500 kcal is used (`model_estimate`, calories only); otherwise nothing. A repeated dish is estimated once. Still capped at `MAX_PROFILE_CALLS` distinct dishes. No estimator passed, no estimate and no call: the orchestrator and the default `PlanScorerService` wire one in. |
| Why not send the dish name alone | The profiler invents a recipe for a bare title and weighed the pasta in "pasta with zucchini" at 0 g. Given ingredient lines it looks each one up in composition tables, which is the figure worth having. |
| `grounding._profile_all` | Profiling calls run in parallel (4 workers) with `PROFILE_TIMEOUT_SECONDS` (12 s) each. The first `candidates_client.ProfilingTimeout` stops calls not yet started; those dishes fall back to the guess. A healthy pipeline answers in 1–7 s. |

## Reading the profiler

| What | Detail |
|---|---|
| `candidates_client.profile_nutrition` | Rewritten against a real response. Reads `profiling_totals.total_energy_kcal_per_serving_<table>` and its protein, carbohydrate and fat siblings (the table named by `nutrition_source_key` wins), falling back to `full_profile.nutrition_summary.energy_kcal_per_serving`. The guessing reader it replaces took whole-recipe calories, which doubled a two-serving dish, and macros from the first ingredient. |
| `models.recipe.ProfiledNutrition` | `nutrition`, `coverage`, `low_coverage`; `reliable` is false when the profiler flags low coverage or matched under `MIN_PROFILE_COVERAGE` (0.6) of the ingredients. |
| `profile_recipe(..., timeout=None)` | Both the production client and the demo client take a timeout and raise `ProfilingTimeout` on it; other failures still return None. `DemoSession.call` gains an optional `timeout`. The demo client already sent the bearer token; the earlier failures were a stalled pipeline, not authentication (without a token the gateway answers 401 in 0.2 s). |

## The fixes around it

| What | Detail |
|---|---|
| Labels | The calorie metric, the plan-score warnings and the judge's plan text say "estimated from typical ingredients" or "a rough guess", never "as written". |
| `service.summary_facts` → `calories` | When some dishes have no calories, the writer is told in so many words not to state or compare the plan's total, or call it short of or over a target. When all are known but some estimated, it is told to say so if it mentions calories. |
| `service.allergen_sentences` | One sentence per dish and certainty: "“Pasta with zucchini” may contain lactose and dairy, which are on your allergy list." instead of one sentence per allergen. `ensure_allergen_warnings` appends it when the reply leaves out any of its allergens. |
| `scoring.constraint_rows` → `violations["possible_allergens"]` | An allergen only the closest recipe has now reaches the judge's conflicts as "may contain …, a possibility, not a fact". Live, the judge had written "no allergens" beside the warning. It still caps nothing. |

## Verification

`tests/test_plan_scorer.py`: typical ingredients profiled with the member's own
ingredients passed on; low coverage, the profiler's own flag, an implausible
guess, no estimator, no endpoint, a failing profiler, a failing estimator; one
timeout stopping the rest; a repeated dish estimated once; the cap; the reader
on the live response shape (per serving, not whole recipe; named table; summary
fallback; no calories); the estimator's parsing of partial and malformed
output. `tests/test_plan_scoring.py`: both labels in the metric and the judge's
text, possible allergens reaching the judge without capping fit, the merged
warning, the calorie facts. Full suite: 909 passed. Ruff: no new findings.

**Live**, `POST /score-plan` over HTTP against Groq and the WiseFood demo
gateway, the same three dishes, a member allergic to lactose and dairy:

| | |
|---|---|
| Result | 200 in 10.2 s; 3 model calls, 3,310 tokens (estimator 570 on the fast model, judge 1,899, writer 841) |
| Fried eggs | closest recipe a pumpkin rösti, so estimated: 2 eggs, 1 tsp olive oil → 180 kcal, coverage 1.0, profiled in 3.3 s |
| Pasta with zucchini | closest recipe a creamy bacon pasta, so estimated: 80 g dry pasta, 150 g zucchini, 1 tsp oil → 360 kcal, coverage 1.0, 5.6 s |
| Chicken noodle soup | matched, the recipe's own figures |
| Day | about 885 kcal against 1,800, every dish counted |
| Reply | one merged "may contain lactose and dairy" warning; the judge's fit reasoning says dairy presence is uncertain |

Not exercised live: a profiler timeout, and the guess path.

---

# A pasted plan reaches the scorer without the classifier

> **Date:** 2026-09-16
> **Branch:** main
> `OrchestratorAgent.classify` now returns `failed: True` when every attempt
> failed; the intent it returns is unchanged. No API change.

Live, this message was answered as small talk:

> Rate for me a daily plan consisting of fried eggs for breakfast, pasta with
> zucchini for lunch and chicken noodle soup for dinner

Two independent causes, both fixed here.

| What | Detail |
|---|---|
| The classifier was never asked successfully | Three attempts, three 429s on the day's token budget, and `classify` defaults to `chat` on failure. The member got the small-talk bot, which succeeded because its prompt is smaller than the classifier's 2,105 tokens. `classify` now reports the failure, and `_classify_and_route` scores a message that lists meals instead of chatting at it. Every other message still falls back to `chat` exactly as before. |
| The bypass should have skipped the classifier | It required `slot:` lines or bullets, and this plan is prose. `looks_like_plan_listing(text, structured_only=False)` counts the prose form too, and the new `OrchestratorService.looks_like_a_pasted_plan` pairs it with the request-verb guard, which now also catches "adding", "including" and "put". So "what do you think of adding salmon for dinner and oats for breakfast?" is still a request for the classifier, and a pasted plan with a scoring word costs no classifier call at all. |

## Verification

`tests/test_plan_scoring.py`: the live message bypasses the classifier; a
listing reaches the scorer when classification fails; "adding salmon for
dinner", "what's for dinner tonight?" and "thanks, that looks great" still
fall back to chat; `classify` reports `failed` after its retries. Full suite:
891 passed, no new lint findings. Not re-run live — the day's Groq token
budget is spent.

---

# Plan scorer: one judge call, estimated calories, and one spelling per dish

> **Date:** 2026-09-16
> **Branch:** main
> `plan_score.grounding` rows gain `nutrition_source`. No other API change.
> The planner's own judges are restored to exactly their committed form.

Three changes to the scorer, all from what the live runs cost and got wrong.

## One judge call instead of three

| What | Detail |
|---|---|
| `agents.PlanJudge` + `PLAN_JUDGE_DAILY_SYSTEM` / `PLAN_JUDGE_WEEKLY_SYSTEM` / `PLAN_JUDGE_USER` | Diversity, guideline adherence and fit come back from one call as one JSON object (`schemas.PlanJudgementSchema`). The daily and weekly prompts differ only in how the two food criteria are judged. |
| Measured cost | Three calls sent the plan, the profile and a system prompt three times: about 3,400 input tokens and three outputs. One call is about 2,100 input tokens and one output, and one request instead of three. On the Groq on-demand tier (8,000 tokens a minute, 200,000 a day) that is the difference between a paste fitting in a minute's allowance and not. |
| `agents.MealDiversityGrader`, `GuidelineAdherenceGrader` | Back to their committed form: no `system_prompt`, `run_name` or `facts` parameters. The planner's daily flow is byte-identical again, and `PLAN_FIT_SYSTEM`, `WEEKLY_MEAL_DIVERSITY_SYSTEM` and `WEEKLY_GUIDELINE_ADHERENCE_SYSTEM` are gone. |
| `scoring.run_judges` removed | With one call there is no fan-out, so the thread pool and its context copying go too. `retrying` stays: one retry when the call raises or returns no usable score. |
| Partial answers survive | A payload missing one section, or carrying a score off the 1–5 scale, leaves that one metric ungraded instead of discarding the other two. |

## Calories when no recipe matches

| What | Detail |
|---|---|
| `candidates_client.profile_recipe` | `POST {RECIPEWRANGLER_API_URL}/api/v1/recipes/profile` ("Run parsing + profiling pipeline on raw recipe text"), through the gateway at `/api/v1/recipewrangler/recipes/profile`. Sends the dish as the member wrote it: title, their amount, their ingredients. |
| `candidates_client.profile_nutrition` | Reads per-serving calories out of the response. **The response shape is not in the gateway's OpenAPI document and the demo deployment answers 503 (`upstream/unavailable`), so this has never seen a real payload.** It therefore looks up the figures by name at any nesting depth, accepts the plausible spellings (`kcal_per_serving`, `calories`, …), and returns None when it finds no calorie figure. |
| `grounding.DishGrounder._profile_missing` | Runs after the details batch, for dishes that still have no calories. One call per distinct dish, capped at `MAX_PROFILE_CALLS` (10) per plan, each failure silent. A deployment without the endpoint passes `profile_client=None` and simply gets no estimates. |
| `GroundedMeal.nutrition_source` | `recipe` \| `closest_recipe` \| `profiled` \| `""`. An estimate is labelled in the grounding row, in the calorie metric's sentence, in the plan-score warnings and in the text the judge reads, so it is never presented as a measurement. |

## One spelling per dish

| What | Detail |
|---|---|
| `parsing.SPELLING_VARIANTS`, `fold_accents` | "lasagne"/"lasagna", "yoghurt"/"yogurt", "houmous"/"hummus" and similar normalise to one form on both sides before comparison, and accents fold ("crème fraîche" = "creme fraiche"). Live, the knowledge-graph search offered "Roasted vegetable lasagne" for "vegetable lasagna" and the literal matcher rejected it. |
| `parsing.spelling_variants` | The other spellings of a title, for searching. `grounding.DishGrounder._search` uses them only when the first query found nothing that scores as a match, so a hit costs one request as before. |

## Verification

`tests/test_plan_scorer.py` and `tests/test_plan_scoring.py` cover the merged
judge (one call carries the guideline text, the measured facts and the plan;
weekly gets the weekly prompt; a retry; a missing section; a failed call
leaving the measured metrics), the fallback (estimated calories and their
label, a matched recipe's own figures kept, a missing endpoint, a failing
profiler, the cap) and the spellings (normalisation, similarity, variant
search only when needed). Full suite: 885 passed. Ruff: no new findings.

**Live:** both pastes went through `POST /score-plan` over HTTP against Groq
and the WiseFood demo catalogue. Each returned 200 `application/json` with all
its metrics scored, in **two** model calls instead of four:

| Paste | Calls | Input | Output (incl. reasoning) | Total | Time |
|---|---|---|---|---|---|
| 3 days | judge + writer | 2,980 | 466 (281) | 3,446 | 10.5 s |
| 1 day | judge + writer | 2,157 | 529 (277) | 2,686 | 5.3 s |

The judge returned all three scores both times, and the caps still bit: the
day with peanut noodles and a chicken caesar salad scored fit 1 with the
allergy and the diet named.

**Not verified live:** the calorie fallback. The demo's profiling pipeline
answers 503 (`upstream/unavailable`) for every dish, which the run logged and
carried on from, so no estimate has ever come back. Its response shape is
still guessed, and `profile_nutrition` is written to tolerate that.

---

# Plan scorer, steps 4–5: a pasted plan gets FoodChat's scores

> **Date:** 2026-09-15
> **Branch:** main
> New endpoint `POST /foodchat/sessions/{id}/score-plan`. `ChatTurnResponse.plan_score`
> is now fully populated and typed; `/conversation` messages gain `plan_score`.
> New nullable column `messages.plan_score` (added by `init_db`'s migration).
> Generated daily and weekly plans are unchanged — see "The planner, fenced".
> **Not in this change:** the wisefood-api gateway route and the UI text box
> and card live in other repositories and still need the matching change.

A member can now paste a daily or weekly plan — into the chat or into a text
box — and get it scored with the metrics FoodChat uses on its own plans, with
the reasoning behind each number, the constraints it breaks, and a short
reply. Nothing is written to a canvas; the score card is stored with the reply
so it survives a reload.

## The planner, fenced

| What | Detail |
|---|---|
| `services/plan_scoring.py` (new) | `ingredient_names`, `food_variety_score`, `plan_as_text`, `compute_daily_metrics`, `guidelines_text(scope)` — moved out of `ChatService`, bodies unchanged. `ChatService._compute_metrics` delegates to it; the old private names stay as thin aliases. `weekly_planner/explainability` imports the same `ingredient_names` instead of keeping a copy (and drops the now-unused `re` import). |
| `prompts.GRADER_SYSTEM_INSTRUCTIONS` | The rubric, slot-plausibility and assessor paragraphs were extracted into `PLAN_SCORING_RUBRIC`, `SLOT_PLAUSIBILITY_RULES` and `ASSESSOR_STANCE` by a script that cut the exact substrings, so the fit judge shares them. The assembled grader prompt is byte-identical; its hash is pinned. |
| `agents.MealDiversityGrader`, `GuidelineAdherenceGrader` | Optional `system_prompt` / `run_name` (and `facts` on the guideline judge). With the defaults the planner gets the same prompt, the same message text and the same trace name — asserted against a recording client. |
| `tests/test_plan_scoring.py` | Verbatim copies of the pre-hoist functions as an oracle; SHA-256 of the seven planner prompts; the default judges' messages; eleven everyday planner messages that must not be taken for an explicit score request, and must still reach the classifier. |

## Step 4 — scoring (`services/plan_scorer/scoring.py`, new)

| What | Detail |
|---|---|
| Constraint rows | `transparency.constraints_ledger(profile)` rows — same wording and household attribution as a generated plan — re-measured dish by dish. Allergy, checkable diets (vegetarian, vegan, pescatarian, gluten-, dairy-, nut-free) and dislikes become `violated` naming the dishes, or `satisfied` with a note when some dishes could only be checked by name. Goals and non-checkable diets are `unchecked`. Catalogue tags count only for a matched dish. |
| Daily metrics | `fvs`, `daily_nutrition` (`nutrition_metrics` against a one-day target), `diversity` and `guideline_adherence` with the planner's own judges and prompts, `fit`. |
| Weekly metrics | `weekly_variety`, `weekly_guidelines`, `weekly_nutrition`, `diversity`, `guideline_adherence`, `fit`. The explainability functions are called one by one rather than through `build_weekly_explainability`, which divides by seven and counts snacks as meals: targets are scaled to the days pasted, snacks count for calories but not meals, and "eat fish 1–2 times a week" is not applicable to a shorter plan with no fish. Measured rows (meat limit, calories, repeats) say "over these N days". |
| `WEEKLY_MEAL_DIVERSITY_SYSTEM`, `WEEKLY_GUIDELINE_ADHERENCE_SYSTEM` (new prompts) | Judge across days — protein rotation, produce across the week, repeats as routine versus monotony. The weekly guideline judge is handed the measured checklist as facts and told not to contradict it. |
| `agents.PlanFitGrader` + `PLAN_FIT_SYSTEM/USER` (new) | One plan against the profile: allergies and diet as hard constraints, the conflicts code found, likes, dislikes, goals, nutrition targets, and the member's aim (the text box's `context` plus what they wrote around the listing). |
| Caps, in code | An allergen caps fit at 1 — and sets 1 even if the fit judge failed; a broken checkable diet caps it at 2. The reasoning names the dish. |
| Judges | Run concurrently, each in a copy of the turn's context so Langfuse still groups them. A failed judge gives `score: None` and a sentence, never 0. |

## Step 5 — reply, persistence, endpoint

| What | Detail |
|---|---|
| `plan_scorer/service.py` | `ResponseWriter` writes the reply from facts (scores, broken constraints, dishes not found, close matches); the deterministic fallback reads and scores. Any allergen the reply does not name is appended as a fixed sentence. The clarification state now carries `plan_type` and `context`. |
| Payload | `metrics[{key, label, score, kind, reasoning, detail}]`, `constraints_applied`, `grounding`, `unparsed`, `warnings`, `scored_plan{origin: "pasted", plan_type, days, entries}`, `context`. `scored_plan.days` uses `DayPlanResponse`, which can show a partial or two-plate day; `entries` uses the weekly card's entry shape. |
| `db.MessageRow.plan_score`, `SessionService.add_message(plan_score=)`, `Message.plan_score` | Stored with the reply, loaded with the session, returned by `get_messages_page` and `/conversation`. Idempotent `ALTER TABLE` in `_migrate_existing_db`. |
| `POST /sessions/{id}/score-plan` + `OrchestratorService.score_plan` | `{member_id, plan_text (≤ 8000), plan_type: auto|daily|weekly, context? (≤ 500)}` → `ChatTurnResponse`. Ownership 404, message cap, blank text 400; supersedes a pending clarification; no classification; memory nudges still run on the pasted text. |
| Typed wire models | `PlanScoreMetric`, `PastedPlanView`, `ScorePlanRequest`; `PlanScoreResponse.metrics` is typed. |
| `OrchestratorService.is_explicit_score_request` | Now requires a *structured* listing (`slot:` lines or bullets, not "X for dinner" prose) and no request verb (make, swap, change, add, plan my…). "What do you think of adding salmon for dinner and oats for breakfast?" goes to the classifier, as it did before the scorer existed. |
| `plan_scorer` steps 1–3 | Parser keeps the lines around a listing as `notes`; grounding makes one details call for every grounded dish it uses (images for matched dishes). |

## Close matches, corrected after the first live run

The first run against the WiseFood demo catalogue produced confident, wrong
claims that no unit test had caught, because the fakes returned tidy titles.

| What the live run showed | Fix |
|---|---|
| "porridge with banana and honey" **matched** "Honey banana cups" (0.8). `cups` was on the stop-word list, so the catalogue title was compared as "honey banana". | `parsing.MEASURE_WORDS` split from `STOPWORDS`; `title_similarity` drops amount words from the member's title only, keeping them in the catalogue title, where they name the dish. The pair now scores 0.67. |
| "vegetable lasagna" read as a beef "Lasagna": the vegetarian row said **violated by vegetable lasagna (red meat)**, the meat count rose, and the reply repeated it. "peanut noodles" read as a salmon noodle recipe was reported as fish. | An approximate match **never lends ingredients or tags**. Diet, category and meat checks read only the member's words unless the dish is matched. |
| "an apple" read as "Apple strudel", with the strudel's calories and 20-odd ingredients in the variety count. | An approximate match lends **nutrition only** when its name is a more generic form of the member's (`grounding.borrowable`: "Lasagna" for "vegetable lasagna", "Hummus" for "hummus and carrots"), never when it names something they did not write. `GroundedMeal.borrows_nutrition` and the grounding row say which; the reply names both cases differently. |
| An allergen present only in a resembling recipe would have capped the fit score. | New evidence `closest_recipe`: the reply says "may contain", the allergy row is `unchecked` with the dish named, and no cap applies. |
| The daily diversity judge and the weekly fit judge returned nothing while the router battery ran in parallel; alone, sequentially and three at a time, every call succeeded. | Each judge gets one retry on an exception or an unusable score (`scoring.retrying`); the failure is logged with its type. |
| Similarity was computed on the search hit's title and shown with the fetched record's title. | Recomputed on the fetched title when they differ. Not the cause of the porridge match, which was the stop word, but the number must describe the recipe actually used. |

## Verification

**Unit tests (LLM-free):** 876 passed — the 712 that existed before the scorer
work, plus the two scorer files. Ruff reports exactly the findings it reports
on `HEAD`; nothing new. mypy is not installed in this environment and was not run.

**Live, WiseFood demo catalogue + Groq:** the first run, before the close-match
fixes, sent a pasted day and a pasted three-day plan through `/score-plan` and
produced the wrong claims listed under "Close matches". After the fixes the same
pastes were sent over HTTP to the real router (FastAPI `TestClient`): both
returned 200 `application/json` (20.8 kB weekly, 5.8 kB daily) with every
`plan_score` field; grounding, the measured metrics and the constraint rows were
correct (vegetarian broken only by the chicken caesar salad and the grilled
salmon; the lasagna lends calories only); `/conversation` returned both cards.
The daily token quota was still spent, so one judge answered (weekly diversity,
4/5, 997 input and 354 output tokens of which 88 reasoning) and the rest returned
`score: null` with the deterministic reply, as designed.

**Live, not verified:** the router prompt battery (planner messages must route as
the committed prompt routes them; pasted plans must route to `score_plan`). The
run hit the Groq on-demand daily token limit for `openai/gpt-oss-120b` (200,000
tokens a day): 218 router calls failed and fell back to `chat` for all three
prompt variants — the committed one included — so it shows nothing about the
prompts. The first live run's chat paste and its ordinary daily-plan request
both landed in small talk, most likely for the same reason. Both checks need a
rerun with quota to spare (`scratchpad` scripts `live_router.py`, `live_e2e.py`).

**Cost note:** a pasted plan costs up to four calls on the reasoning model
(three judges and the reply writer) plus one fast-model parser call when the
text is not structured, and a failed judge is retried once.

## Still open

- The gateway route for `/score-plan` and the UI text box and score card (other repositories).
- `guidelines_text` still reads a file that is not in this repository and returns `""`; the external guidelines endpoint replaces that one function.
- Generated weeks do not yet carry the weekly diversity and guideline judges; attaching them adds two Groq calls to every weekly plan.
- "Adopt this plan" onto a canvas.
- The same dish twice on one day of a pasted week still counts as an unexplained duplicate.

---

# Plan scorer, steps 1–3: a plan the member wrote becomes a reading

> **Date:** 2026-09-14
> **Branch:** main
> New intent `score_plan`. `ChatTurnResponse` gains an optional `plan_score`.
> No canvas, storage, or existing-intent behaviour changes. No metric is
> computed yet — that is steps 4–5 of IDEAS.md "Plan scorer".

A member can paste a daily or weekly plan they wrote into the chat. FoodChat
now recognises that as its own intent, reads the text into days, slots and
dishes, matches each dish to the recipe catalogue, and builds the same objects
the planners produce, so the existing evaluation routines can grade it in the
next step. The reply says what was read, which dishes matched only roughly or
not at all, which dishes carry one of the member's allergens, and that scores
are not available yet. Nothing is written to a canvas.

## Recognising the intent apart from the others

| What | Detail |
|---|---|
| `schemas.OrchestratorSchema`, `OrchestratorAgent.VALID_INTENTS`, `models.session.Intent` | `score_plan` added. |
| `prompts.ORCHESTRATOR_SYSTEM` | Tenth intent, and rule 8 to keep it apart from its neighbours: a message that *brings* a listing is `score_plan`, even with "is this healthy?" attached; a request *for* a plan that names a few dishes stays `daily_plan`/`weekly_plan`; a question about the plan FoodChat made stays `plan_question`/`nutrition_question`. Rule 5 now defers to rule 8. |
| `OrchestratorAgent.system_prompt` + `prompts.SCORE_PLAN_INTENT_ADDENDUM` | The live router prompt is a managed Langfuse copy that predates this intent, and a deploy never overwrites it. When the compiled text does not mention `score_plan`, the addendum is appended — otherwise production could never emit the intent while every local test passed. |
| `OrchestratorService.is_explicit_score_request` | A scoring word ("rate", "score", "how does this look") plus a listing in at least two meal slots skips the classifier, like an explicit FoodScholar consult, and supersedes a pending clarification. A FoodScholar mention keeps its own bypass; "rate my week" with no listing still goes to the classifier. |
| `OrchestratorService._route` | `score_plan` is checked first and goes to `PlanScorerService`. |
| `OrchestratorService._handle_clarification_turn` | `kind == "score_plan"` resumes the scorer. A reply that answers nothing is routed as a fresh turn with `score_may_ask=False`, so the member is never asked a score question twice in a row. |

## Step 1 — parsing

| What | Detail |
|---|---|
| `models/pasted_plan.py` (new) | `PastedMeal`, `PastedDay`, `PastedPlan`, `GroundedMeal`; JSON round-trip so a parsed plan can ride inside clarification state. |
| `services/plan_scorer/parsing.py` (new) | A line scanner reads day headings ("Monday", "Day 2", "Tue:"), `slot:` prefixes, bullets under a slot heading, trailing "(ingredients)", and "oats for breakfast" prose. A heading must end like a heading, so "Sun-dried tomato pasta" is not Sunday. Preamble and questions are ignored; any other line it cannot place is kept verbatim in `unparsed`. |
| No model call for structured text | When the scanner reads the text from structure alone and places every line, its reading is used as is. Most pasted plans are shaped that way. |
| `agents.PlanTextParser` + `prompts.PLAN_TEXT_PARSER_*` (new) | Fast model, called only for text the scanner could not fully read, with the scanner's reading as a hint. A new prompt name, so it syncs to Langfuse. |
| Checked against the text, in code | A title the text does not support is dropped; an ingredient list with any item the member did not write is dropped whole; an `unparsed` line must be a verbatim substring. Each drop leaves a warning. A parser failure leaves the scanner's reading. |
| Shape question | One block with no day names in which two or more meals repeat asks "one day or several?" (`days_from_answer`: "3 days", "the whole week", "just one day", "several days"). Two dinners alone is one dinner on two plates, not a question. |
| `parsing.singular` | Its own symmetric stem. `pantry_service.singular` maps "berries" to "berri" and leaves "berry" alone, by design, for regex re-inflection — so "Berry Oatmeal" never matched "oatmeal with berries". |

## Step 2 — grounding

| What | Detail |
|---|---|
| `SeedService.find_dish` (new) | The seed path's search and tolerant autocomplete, **not** filtered by the member's allergens, diet or dislikes, with no detail fetch, overlay or allergy gate. IDEAS.md suggested a flag on `_finalize_resolution`; filtering happens in the search, one layer earlier, so a flag there would still have swapped "peanut noodles" for a peanut-free recipe. |
| `services/plan_scorer/grounding.py` (new) | Dice similarity over content words picks the best candidate: ≥ 0.75 matched (recipe ingredients, nutrition, tags), ≥ 0.40 approximate (member's ingredients when written, recipe nutrition and tags), else unresolved (member's words only). One lookup per distinct title; one `fetch_details` batch fills missing nutrition. Lookup failures leave a dish unresolved, never fail the turn. |
| `allergen_conflicts` | Every profile allergen found in a dish, with `evidence` `as_written` or `recipe`, using the same synonym-expanded matcher as the seed gate. Reported and kept — never used to drop the dish. |

## Step 3 — building

| What | Detail |
|---|---|
| `services/plan_scorer/building.py` (new) | Weekly: entry dicts in the `WeeklyMealPlanEnv` shape, so `build_weekly_explainability` runs on a pasted week unchanged; snacks and "other" kept in `extras`, because the guideline checklist counts meals. Daily: course lists per slot; `as_scored_plan()` only for a complete one-dish-per-slot day. `ScoredPlan` is unchanged — its three consumers all assume a full day. |
| Dish identity | A matched dish keeps the catalogue id. An approximate or unresolved dish gets a stable `pasted:<title words>` id, so "chicken curry" and "Thai green curry" landing near one recipe are not a repeat. |
| `weekly_planner/explainability.REPEAT_BY_AUTHOR` (new) | A dish on an earlier day is the member's own repeat. Chip: "the same breakfast as Monday, as you planned it". When every repeat is the author's, the ledger row is "repeats are your own choice", source "your own plan", status satisfied — not measured against the planner's cooldown. The prose names them too. |

## The turn

| What | Detail |
|---|---|
| `services/plan_scorer/service.py` (new) | Persists the member's text and a deterministic reply (intent `score_plan`). Clarification states `{kind: "score_plan", reason: "no_meals" | "shape", pasted_text, plan?}` keep the pasted text so nobody re-pastes. |
| `ChatTurn.plan_score` / `routers.PlanScoreResponse` (new) | `plan_type`, `days_scored`, `meals_scored`, `grounding[]`, `unparsed[]`, `warnings[]`; `metrics` and `constraints_applied` are empty lists until step 4. |
| `CHAT_ENDPOINT_PIPELINE.md` | Section 1 (bypass, clarification kind, route), new section 5b, section 6 field. |

## Verification

`tests/test_plan_scorer.py` (new, 56 tests, LLM-free): scanner heading styles,
bullets, parentheses, preamble vs stray lines, "Sun-dried" and "haddock";
parser output checked against the text; shape question and answers; grounding
states, unfiltered lookup, allergen kept with its evidence, one lookup per
dish, one nutrition batch, lookup failure; weekly entries, author repeats
through `variety_metrics` and `build_weekly_explainability`, three-day
checklist scaling; daily completeness; the turn (no canvas, persisted
messages, both clarification reasons, unanswered replies); routing (classified,
explicit bypass, superseding a pending question, no bypass without a listing,
never asking twice, the stale-prompt guard, the wire model). Full suite: 768
passed.

## Still open

- Steps 4–5: metrics, hard-constraint ledger rows, fit score, summary prose.
- `plan_score` is returned on the turn but not stored with the message, so the card does not survive a conversation reload yet.
- The `/sessions/{id}/score-plan` endpoint for the text box, and the gateway and UI.
- The same dish twice on one day of a pasted week cannot carry `repeat_of_day` and still counts as an unexplained duplicate.

---

# A repeat you asked for is offered, not waited for

> **Date:** 2026-09-09
> **Branch:** main
> No API change. New `plan_parameters.repeat_mode_is_explicit`; the
> `repeat_offered` selection event gains an optional `injected: N`. Behaviour
> changes **only** for members who have set `repeat_meals` themselves.

A live week with `repeat_meals` set to "cook once, eat twice" came back with
one repeated breakfast. The policy was not the limiter. Replaying that week's
commitments through the real action space:

```
day 4 breakfast: policy would allow ['granola']
day 5 breakfast: policy would allow ['granola']
day 6 breakfast: policy would allow ['granola', 'muffins']
day 7 breakfast: policy would allow ['granola', 'muffins', 'muesli']
```

Tuesday's granola was eligible at every remaining breakfast slot, and
`_fetch_exclusions` deliberately kept it *in* the fetch so RecipeWrangler
could return it. There is exactly one `repeat_offered` event for breakfast in
that plan — day 3 — and none afterwards.

So the cooldown decided a recipe *may* come back, and then whether it ever got
the chance was RecipeWrangler's ranking. A day's pool is a fresh fetch and the
source ranks recipes the member has not seen; on a real week it simply does
not return them. The member's setting was being quietly vetoed by the search
provider, and `repeat_offered` is the only reason that was visible at all —
which is exactly the question it was added to answer.

## Two halves, because one is not enough

Injection alone barely moves the needle. A repeat scores 0.0 like every other
bare candidate, so putting it in a 10-candidate pool buys it a 1-in-11 share
of the random tiebreak:

| pool | injected | chance per slot | expected over 5 eligible slots |
|---|---|---|---|
| 10 | 1 | 9% | 0.5 |
| 10 | 2 | 17% | 0.8 |
| 10 | 3 | 23% | 1.2 |

1.2 is not distinguishable from the 1 the member already got. So the dish has
to be *offered* and it has to be able to *win*.

| What | Detail |
|---|---|
| `repeat_mode_is_explicit(values)` | The gate. `repeat_mode` cannot answer this — it returns `"breakfast"` both for a member who chose "Repeat breakfasts" and for one who has never seen the card, and those deserve different planning. An unrecognised stored value is NOT explicit: `repeat_mode` degrades it to the default, and acting on a value we could not read would be inventing a request out of a typo. |
| `action_adapter._injected_repeats` | An eligible earlier dish the day's pool does not contain, rebuilt from `_served` and added to it — the same trick the leftover uses, at the same price of zero requests, offering the dish that was actually eaten rather than one that resembles it. |
| `_second_serving` | Extracted, and now shared by the leftover and the injected repeat, so the two cannot drift in what they carry or — more to the point — in what they drop (`pinned`, `match_reasons`, and the source's own repeat labels). |
| Every ordinary rule still decides | `_repeatable_from` is the eligibility test, so the slot, the gap, `MAX_APPEARANCES` and `mark_selected` all still apply unchanged. A dish the source *did* return is skipped rather than added twice. |
| `INJECTED_REPEATS_PER_SLOT = 2` | Everything eligible would be legal. A slot whose pool is a third old dishes is a repetitive week arriving by the back door rather than by the member's setting. Newest first: a routine is built out of what the week is already in the habit of. |
| `planner._REPEAT_BONUS = 1.0` | Zero was the right price for a repeat the planner merely *allowed* — the reasoning `_REPEAT_PENALTY` was taken to zero for. It is the wrong price for one the member *requested*. Sized on the documented ladder: below a favourite (+5), a stated pantry item (+3) and a leftover (+2.0), and at parity with a single liked ingredient (+1.0) so a fresh dish the member has a reason to like still ties rather than losing. At `strict` food waste a fresh candidate reusing one ingredient scores 1.6 and beats it, which keeps that axis meaning what it says. |
| One control, one payment | A leftover takes `_LEFTOVER_BONUS` **instead of** `_REPEAT_BONUS`, never on top. Both exist because the member moved the same single control, and stacking them priced one setting twice — enough, as it happened, to put a leftover level with a stated pantry item, which the ladder says must still win its slot. Caught by a test, not by review. |
| The diagnostic survives | `repeat_offered` gains `injected: N`, kept apart from `count` and never folded into it. That event's whole job is to separate "the week repeated nothing" from "the source never offered anything", and once the plan puts dishes in the pool itself, that question can only be answered by a number saying how many it put there. |

## Measured

Against a source with a healthy 40-recipe pool that never volunteers a served
dish back — the behaviour that produced the report — 25 seeded weeks each:

| `repeat_meals` | breakfast repeats / week | all repeats / week |
|---|---|---|
| **default (card untouched)** | **0.00** | **0.00** |
| explicit `"off"` | 0.00 | 0.00 |
| explicit `"breakfast"` | 3.00 | 3.00 |
| explicit `"all"` | 3.00 | 9.00 |

3 is the policy's own ceiling for 7 slots (`MAX_APPEARANCES` 2, gap 2 → four
distinct breakfasts, three repeats), so an explicit setting now saturates what
the caps allow instead of being vetoed by the source. Against a *generous*
source (a thin pool the source does re-offer from) the default is unchanged at
3.00 either way — the default path is genuinely untouched, not merely
untouched when nothing was available.

## Verification

`tests/test_leftovers.py` grows 60 → 89. The load-bearing ones:

- the default week against the never-re-offering source is still **0 repeats**
  — the week the report was written about, deliberately still that week;
- an explicit `"breakfast"` produces repeats against that same source, and
  produces them in **no other slot**;
- `"off"` still yields 21 distinct recipes;
- the caps and the gap still hold over a full injected week, and
  `unexplained_repeats` is 0 — every injected second serving still passed
  through a rule that recorded a reason for it;
- the scoring ladder is pinned in both directions, including the
  paid-once-not-twice case that the stacking bug tripped.

**712 passing** (683 before). `src/` ruff unchanged at its pre-existing 19.

## Still open

Unchanged by this: `flexitarian` does nothing; the ResponseWriter still
soft-pedals a violated calorie row; and `day_summary.classify_meal` trusts
RecipeWrangler's tags in both directions — a `vegetarian_or_vegan` tag on a
snapper dish erased a real fish meal from the guideline checklist, while
"fish sauce" in a vegetarian dish invented one and spent a meat allowance.
That classifier is now the highest-value thing outstanding: it is one
function, it changes what gets planned, and one of its failure modes touches
an allergy claim.

---

# Cook once, eat twice — and repeats become the member's dial

> **Date:** 2026-09-09
> **Branch:** main
> Additive API. A weekly entry's `recipe` may now carry `leftover_of {day,
> meal_type}` alongside the existing `repeat_of_day` / `repeat_source`; a new
> `repeat_source` value `"leftover"`; a new plan parameter `repeat_meals`
> (weekly cards only); new metric keys `variety.leftover_meals` and
> `repeats.leftovers`. Nothing is removed, and every default preserves the
> previous behaviour exactly.

Component 3 of the natural-weekly-planning plan (`IDEAS.md`), plus the lunch
and dinner repeats component 2 left open. They ship together because they are
one mechanism: a leftover **is** a repeat with a slot transition.

`IDEAS.md` blocked component 3 on a plan-shape decision — "is a leftover a new
entry kind, or a flag on an ordinary entry?" — and predicted a ripple across
the planner, the tracker, explainability, edits, refinements and the UI.

**It is a flag on an ordinary entry, holding the whole recipe.** The member
really does eat that dish, so its nutrition, its ingredients and its card are
the dish's own; a stub referencing another slot would be a less truthful
representation, not a more compact one. `WeeklyMealPlanEntryResponse.recipe`
is a required dict, so a stub would also have broken the gateway and the UI in
the same release — and the existing repeat machinery is already exactly "a
flag on an ordinary entry", with a chip, a ledger row, a metric and a prose
sentence to inherit.

That decision dissolved most of the predicted ripple. What was left:

| Predicted ripple | What it actually cost |
|---|---|
| Planner skips the slot | It does not. The leftover is one more candidate in the day's pool, scored against the rest — so hard constraints, the tracker and the reward all work unchanged. |
| Tracker must count the meal but not the shopping | One branch. `basket.add` is skipped for a leftover; everything else already counted meals, not baskets. |
| Explainability iterates entries as independent dishes | It still can — every entry has a real recipe. Three additions: the chip names the slot it was cooked in, verification checks (day, slot) rather than the day, and leftovers stay out of `min_gap_days`. |
| Edits invalidate the dependent slot | It cascades, and says so. |
| Refinements rebuild all 21 slots | Nothing to do; leftovers are rebuilt with the week. |
| UI renders a different card | It does not. The card is a card; it gains a chip. |
| Portion arithmetic | Not attempted, and said so out loud — see below. |

## What shipped

| What | Detail |
|---|---|
| `plan_parameters.repeat_meals` | One ordered control, `off` → `breakfast` → `all` → `leftovers`, each stop a superset of the one before. Answers component 2's open question ("how often may a *dinner* recur before a week reads as lazy?") by asking rather than guessing — the answer differs by household, not by request. Default `breakfast`, which is exactly what the planner did before, so an untouched card changes nothing. Weekly cards only: "the same breakfast may come back later in the week" is not a setting a one-day plan can honour, and a control that visibly does nothing teaches members that none of them work. |
| `repeat_mode` / `repeats_allowed` / `leftovers_allowed` | The accessors every reader goes through, so an unrecognised stored value degrades to the default instead of disabling the policy — a profile written by another release must not silently become "all different". |
| `action_adapter._slot_repeats_allowed` | The one place the member's setting and the hard `ALL_REPEATABLE_SLOTS` list are resolved. `REPEATABLE_SLOTS` stays as the default-setting constant other modules read the M9 contract from. |
| `action_adapter._leftover_action` | Yesterday's dinner as today's lunch, rebuilt from `_served` (the committed action, now kept by `mark_committed`) rather than re-fetched: no request, and the candidate is the dish on the plate by construction. Gated on the setting, the cap (`MAX_LEFTOVER_MEALS` = 3), `MAX_APPEARANCES`, the exact one-day gap, and `mark_selected` — a pinned or downvoted dish keeps its "no way back". |
| It carries BOTH markers | `repeat_of_day` so every M9 rule applies with no parallel code path (the scorer's own-title exemption, the ingredient-axis exclusion, the cap, the measured ledger row), and `leftover_of` so the chip, the ledger and the prose can say *dinner at lunch* instead of "the same lunch as Monday", which would be false. |
| Appended to the pool, never substituted | The member asked that leftovers be *possible*, not that lunch stop being planned. If the leftover loses on score, nothing is lost. |
| `_LEFTOVER_BONUS = 2.0` | Positive, unlike `_REPEAT_PENALTY`, because the member turned a control to ask for it — the same standing as the stated-pantry boost, which is also always on precisely because it was asked for in words. Below a favourite (+5) and below a stated pantry item (+3), both of which still win their slot. Needed at all for the reason `_REPEAT_PENALTY` was taken to zero, read the other way: a bare profile leaves nearly every candidate at exactly 0.0, so an unweighted leftover would never actually be picked. |
| A leftover buys nothing | The one place "eaten twice" and "bought once" have to be different numbers. `IngredientBasket.add` is skipped: adding it would spend the ingredient's `_MAX_REWARDED_USES` allowance on a portion nobody bought, so a genuine later meal using it would score as overuse — and every dish the following day would be charged monotony for sharing ingredients with a meal the member deliberately asked to eat twice. |
| Verified against (day, **slot**) | A stale marker left by an edit points at a slot that has moved on. Checking only the day would let "Monday's dinner again" through when Monday now serves that dish at *lunch* — a false sentence the member can catch in one glance at their own plan. |
| Kept out of `min_gap_days` | A leftover is one day after its source by definition. Averaging it in would report the week's repeats as closer together than the cooldown allows, and the ledger would then read its own feature as a violation. |
| Its own ledger row | `"cook once, eat twice"` — the member turned a control, and the ledger is where they check it happened without having to read the repeat row's parenthesis. The row also states the limit of the claim (below). |
| The edit cascade | `edit_service._leftover_dependents` — changing a dinner a later lunch is eating changes that lunch too, and the reply says which day followed. Matched on the recipe id *and* the slot, so a stale marker never causes an unrelated meal to be overwritten. The lunch is not marked `pinned`: the member asked to change a dinner, not to anchor its lunch. Leaving the lunch alone was the alternative, and it is worse — it would keep serving a dish the week no longer cooks, which every measurement reads as an unexplained duplicate. |

## What is deliberately not claimed

`IDEAS.md` flagged portion arithmetic as the thing this feature seems to need
and cannot have: nothing in this service records quantities, purchase dates or
shelf lives, and the pantry work refused to make quantity claims for exactly
that reason. So the honest version is "eat the same dinner again tomorrow at
noon", and that is what everything says:

- the chip reads "Monday's dinner again — cook once, eat twice", never "the
  rest of Monday's dinner", "a double portion" or "half the batch". A test
  asserts the absence of each of those words;
- the ledger row says outright that the plan doesn't track portions, and tells
  the member to cook enough for two meals if they want this;
- the response-writer facts carry `portions_not_tracked: true`, because a
  writer can only decline to claim what it is told it does not know.

## Verification

`tests/test_leftovers.py`, 60 tests, over the real action space, the real
scorer, the real explainability layer and the real edit service — only the
RecipeWrangler edges are stubbed. The load-bearing ones are invariants rather
than examples:

- the control is **monotone** — each stop's allowed slots are a superset of
  the stop below it, checked across the whole scale, so "more repeats" cannot
  quietly become a different feature at some stop;
- over a whole generated week, every leftover's `(day - 1, "dinner")` actually
  holds the same recipe id, no dish is served more than twice however it came
  back, the cap holds, and `unexplained_repeats` is 0 — every second serving
  on the plate passed through a rule that recorded a reason for it;
- the shopping-list spy counts `21 - len(leftovers)` basket additions, so the
  double-count cannot come back silently;
- the default week has no leftovers at all, and `repeat_meals: "off"` produces
  21 distinct recipes — the pre-M9 rule is still reachable;
- the marker survives a database round-trip (read back through a service with
  an empty cache) and reaches the gateway through
  `WeeklyMealPlanResponse`, chip included.

**683 passing** (623 before this change, of which 2 were the parameter-key
contract tests updated here). `src/` ruff unchanged at its pre-existing 19;
the new test file is clean.

## Still open

Component 2's other remainder is untouched: a repeat earned by something other
than a star. A member-stated liked dish is still not a repeat authority, so it
comes back as `"plan"` rather than `"member_request"`.

---

# The ledger stops claiming diets it never applied

> **Date:** 2026-09-04
> **Branch:** main
> No API change. A diet row can now be `relaxed` (and carry a `detail`), and a
> non-restrictive label renders as `soft` rather than `hard`. New
> `candidates_client.diet_tag_status`; `normalize_diet_tags` is unchanged in
> behaviour and now delegates to it.

`constraints_ledger` emitted `{"type": "hard", "status": "satisfied"}` for
every value in `profile["diet"]`, unconditionally. `normalize_diet_tags`
forwards only the values RecipeWrangler has a filter for — everything else is
dropped, with a log line nobody reads.

So a profile saying **flexitarian** produced a satisfied hard constraint for a
word that appears nowhere in this service: not in `DIET_TAG_MAP`, not in
`VALID_RW_DIET_TAGS`, and not in `state_tracking`'s diet-aware meat limit,
which special-cases only vegetarian, vegan and pescatarian. Nothing was
excluded for it and nothing counted it. The ledger row was the one place a
member could have found that out, and it said the opposite.

| What | Detail |
|---|---|
| New `diet_tag_status(value)` | Returns `("filter", tag)`, `("not_restrictive", None)` or `("unknown", None)` — the classification already inside `normalize_diet_tags`, minus its log line, so the ledger can report which of the three happened instead of assuming the first. `normalize_diet_tags` now delegates to it, so the query and the ledger cannot drift. |
| Forwarded → unchanged | `hard` / `satisfied`, as before, and now only when true. |
| Unknown → `relaxed` | With a detail saying the recipe service has no filter for it, so no dishes were excluded. `relaxed` puts it in `constraints_not_honored`, which obliges the reply to say so rather than list it as an honoured request. |
| Non-restrictive label → `soft` | "omnivore", "mediterranean" and friends are deliberately not forwarded — as hard filters they would empty every slot. Nothing was excluded and nothing was meant to be, so they render as the description they are, with a detail saying so. |
| The row is never dropped | Deleting an unknown value's row would trade a false claim for a silent one — the failure this module exists to prevent, in its other direction. A test pins it. |

The load-bearing test is the invariant rather than the examples: for an
arbitrary diet list, a row may say `hard` + `satisfied` **only** for a value
that `normalize_diet_tags` actually forwarded. That is what stops the two
sides drifting the next time the tag vocabulary changes.

## Verification

Six mutations — collapsing the three outcomes back to one, dropping the row,
reporting unknown as satisfied, reporting a label as an enforced rule, and
both halves of the classifier. All six caught; the last two by the existing
`test_candidates_client.py` tests, which is the check that
`normalize_diet_tags` still behaves exactly as it did.

`tests/test_ledger_honesty.py` grows from 11 to 20. **621 passing**; ruff
unchanged at its pre-existing 19.

## Not fixed here

`flexitarian` still *does* nothing — it is reported honestly now, but the
plan does not act on it. Honouring it would mean a diet-aware meat limit in
`state_tracking` (which already does this for vegetarian/vegan), and the
number is a product decision, not a code one.

---

# A calorie budget with only a ceiling is not a budget check

> **Date:** 2026-09-04
> **Branch:** main
> No API change. `metrics.nutrition` gains `budget_status`; the ledger row is
> renamed `weekly calorie budget` → `weekly calorie target` and can now be
> `violated` from below. `coverage.meals_with_data` will read lower on plans
> containing recipes RecipeWrangler has no composition data for — that is the
> fix, not a regression. New `day_summary.recipe_kcal`;
> `reward_logic.candidate_kcal` keeps its name and delegates to it.

Two bugs with one root, both visible in a single generated week: a plan that
fed the member **933 kcal a day** and reported the calorie constraint as
*satisfied*, alongside `meals_with_data: 21 of 21` for a week where two
recipes had no nutrition data at all.

## A reported zero is missing data, not a zero-calorie meal

RecipeWrangler returns `kcal: 0` for recipes it has no composition data for.
Every reader treated that as a measurement.

| What | Detail |
|---|---|
| One canonical reader | `day_summary.recipe_kcal` returns `None` for a missing *or zero* value. `reward_logic.candidate_kcal` delegates to it, so "unknown" means the same thing to the calorie constraint, the day headline and the weekly metrics. The rule is stated once instead of in each place that divides by it. |
| Coverage stops overstating itself | The week that prompted this goes from `21 of 21` to `19 of 21`, and the *"based on N meals with nutrition data"* note — which had been suppressed precisely when it was most needed — now fires. |
| Day headlines stop averaging in a number nobody measured | `summarize_day` scales the known meals up to the day's meal count. Counting a zero as known dragged the estimate down and could label a hearty day light. |

No recipe is 0 kcal, so the guard is `value > 0`.

## The budget had no floor

`status = "satisfied" if planned <= target * 1.05 else "violated"` — nothing
ever asked whether the member would be fed. `split_ledger` then handed the row
to the response writer under `constraints_honored`, so a half-fed week was
described to the member as an honoured request.

| What | Detail |
|---|---|
| New `calorie_budget_status` | Returns `"over"` / `"under"` / `"on_track"`, or `None` when there is no target or nothing measured. One function, called by `nutrition_metrics`, so the metric, the ledger row and the prose cannot drift apart — each reads the result instead of re-deriving it. |
| `CALORIE_FLOOR = 0.85` | Looser than the 1.05 ceiling on purpose: per-serving figures from a recipe database are approximate and real weeks vary, so this catches a week that is *wrong*, not one that is merely light. |
| The floor is measured against the meals we have data for | Judging a 19-of-21 week by the full weekly target would report "short of target" for two missing data points — a claim about our coverage, dressed up as a claim about the member's food. A week whose 11 known meals are on target reads `on_track` even though its total is half the weekly figure. |
| A missing status is not a violation | Stored plans from before this change, and hand-built payloads, carry no `budget_status`; the ledger derives it rather than reading its absence as a failure. That would be the same mistake in the other direction, and a test pins it. |
| The row says which way it went | Renamed to "weekly calorie target" — "budget" reads as a ceiling, and it is now checked both ways — with `; short of your target for the meals we have data for` or `; over your target` in the detail, and a matching clause in the week's justification. |

## On the week that prompted this

```
coverage      : 21/21    -> 19/21
budget status : (absent) -> "under"
note          : ""       -> "based on 19 of 21 meals with nutrition data"
ledger        : [satisfied] weekly calorie budget
             -> [violated]  weekly calorie target
                6,531 of 14,000 kcal planned (47%), based on 19 of 21 meals
                with nutrition data; short of your target for the meals we
                have data for
```

## Verification

Eight mutations, each neutering one part — the zero guard, the day
qualifier's use of it, `candidate_kcal`'s delegation, the floor, the
coverage-relative expectation, the ledger's use of the status, the
missing-status fallback, and the prose clause. All eight caught.

`tests/test_weekly_explainability.py` grows from 21 to 33, including the
property that keeps the floor honest: a week short only because of missing
data is *not* flagged. **612 passing**; ruff unchanged at its pre-existing 19.

---

# Three things a real week got wrong

> **Date:** 2026-09-04
> **Branch:** main
> No API change. `metrics.selection_events` gains `repeat_offered`. Weekly
> plans pick differently (breakfast repeats can now actually happen), the
> pantry extractor now fires on phrasings it used to miss, and fewer meals
> count toward the weekly meat limit.

Follow-up to *"Weekly plans go looking for what they already buy, and
breakfast may come back"*, from running it and reading the output. Nothing
here was visible from the tests; all three were visible in one generated week.

## The repeat could never fire

`planned_repeats: 0`, seven distinct breakfasts. Not a wiring fault — the
number was wrong.

`_REPEAT_PENALTY` shipped at 1.0, justified as "enough that a repeat does not
win a coin flip". That assumed candidates are spread over a range of scores.
They are not. The member had no favourites, and both their `food_likes` were
cuisines — which `split_cuisines` removes before the scorer sees them — so the
liked-ingredient boost had nothing to match and **almost every candidate scored
exactly 0.0**. Against a flat field, −1.0 is not a tiebreak, it is a veto: the
repeat lost to every fresh candidate that existed.

| What | Detail |
|---|---|
| `_REPEAT_PENALTY` → 0.0 | A legal repeat joins the tie pool and takes a proportional share of slots. It was never the real control: the cooldown decides *whether* a repeat is legal and the cap decides *how often*, and both still hold. The constant stays as the dial to turn if repeats get too frequent. |
| The branch still matters at zero | Sitting out the ingredient axis is the load-bearing half — without it a repeat collects the full reuse bonus for overlapping with its own earlier serving, and at `strict` repeating becomes the cheapest way to score. |
| Measured | 40 simulated weeks, 10 candidates/slot/day, bare profile, food waste off: **~1.2 repeats per week** (over two 40-week samples: 0 in 4–7 weeks, 1 in 20–22, 2 in 13–14). Lunch/dinner uniqueness, the 2-day gap and the 2-serving cap held on every run. |

## "The week repeated nothing" and "nothing was offered" looked identical

Two causes with opposite fixes — one in the scorer, one at RecipeWrangler —
and no way to tell them apart from a stored plan.

| What | Detail |
|---|---|
| New `repeat_offered` selection event | `{"type", "day", "meal_type", "count", "recipe_ids"}`, recorded when a slot's pool contains a legal repeat, whether or not one is chosen. Once per slot. |

## The member's pantry was silently ignored

The query was *"…and I already have avocado, tomatoes and pasta"*. No pantry
chips, no `"your pantry"` ledger row, and the pantry never reached the planner.

| What | Detail |
|---|---|
| The hint gate required subject and verb adjacent | `\bi\s+have\b` cannot span "already" — about the most natural way anyone says this. "still", "just", "only", "also" and "now" failed the same way. Subject and verb may now be up to two words apart. |
| The trade was always in this direction | The gate's own comment says a false positive costs one abstaining LLM call. A miss costs the whole feature — and worse than silently: the `ResponseWriter` still wrote *"using your avocado, tomatoes and pasta"* with no `facts["pantry"]` to support it. |

## A chickpea burger was a red-meat meal

Not just a wording problem. It spent one of three weekly meat allowances,
pruned meat from every later candidate pool, forced a `meat_limit_relaxed`
event at 4-of-3, and had the reply apologise that Thursday's dinner "required
red meat to fit the cuisine mix".

| What | Detail |
|---|---|
| `vegetarian_or_vegan` added to `VEG_TAGS` | RecipeWrangler's most-used veg tag spelling was not listed, so the authoritative signal was being ignored on exactly the recipes that need it. The module docstring already says tags are authoritative; this is one more spelling of the same tag, not a new policy. |
| It classifies as `vegetarian`, not `vegan` | The tag cannot say which. Calling a vegan dish vegetarian understates it; the reverse would be a claim about a recipe nobody made. |
| New `meat_text` — qualified uses stripped before matching | A keyword match cannot read a qualifier. "chickpea burger", "veggie burger", "meat-free chilli", "vegan sausage", "lentil meatballs" are no longer meat. One vocabulary, stripped once, rather than a growing exception list inside each caller. |
| Two kinds of qualifier, with different reach | An outright one ("vegan", "mock", "tofu") may qualify any meat word. A vegetable ("chickpea", "mushroom") may only qualify a word describing a *shape* — burger, sausage, meatball. A vegetable beside an animal is a dish containing the animal, so "mushroom chicken" stays poultry. Under-counting meat is the worse error when the point is a meat limit. |
| A qualifier covers the same word named again | RecipeWrangler ingredient strings repeat the noun bare — the chickpea burger's read *"Burger patties burger patty"*. Every occurrence of a qualified word goes; only that word, so "Beef burger with a veggie sausage" still counts on "beef". |
| The food database's hen's egg is not poultry | *"eggs, chicken, whole, raw"* is how the composition database writes an egg. Read literally it made every shakshuka a poultry meal. The pattern only fires when "chicken" is a bare comma-delimited fragment, so "2 eggs, chicken breast, flour" is still poultry. |

Replayed over the exact week that prompted this: meat meals **4 → 3** (at the
limit, so no relaxation and no apology), and Thursday's headline goes from
*"dinner with red meat"* to *"light vegetarian day"*. The other six days and
all three fish/poultry detections are unchanged.

## Verification

Eleven further mutations, each neutering one of the changes above — the penalty,
the ingredient-axis exemption, both halves of the offer recording, the gate in
each direction, the veg tag, the qualifier scope, the egg patterns, and
`meat_text` itself. All eleven are caught, and the previous change's sweep still passes — one of
its 25 became moot when the penalty went to zero, so it runs 24 now.
One mutation found dead code: the qualified-phrase substitution was redundant
once the qualifier-scope loop existed, and is gone.

Repeats make selection less deterministic, so the suite was run 20 times
rather than once. That found a flaky test of my own: the end-to-end sourcing
test asserted that *Sunday* still had an ingredient worth reusing, which
depends on which recipes happened to land where — by day 7 every ingredient
in the fixture may have had its two meals, which is the cap working, not the
offer failing. It now asserts that some day was offered something.

`tests/test_day_summary.py` grows to 38, `tests/test_pantry.py` and
`tests/test_repeat_policy.py` gain the gate and offer cases. **600 passing**,
20 runs in a row; ruff unchanged at its pre-existing 19.

## Not fixed here

The `ResponseWriter` still writes claims the facts do not carry — the run that
prompted this said *"using your avocado, tomatoes and pasta"* with no pantry in
`facts`, and *"14 kcal weekly budget"*. Widening the pantry gate removes the
occasion for the first one but not the ability. Separately, the reuse sentence
now in `explainability["reasoning"]` reaches the plan payload but not the chat
message, because the writer composes its own prose.

---

# Weekly plans go looking for what they already buy, and breakfast may come back

> **Date:** 2026-09-04
> **Branch:** main
> No breaking API change. Additive on the weekly payload: entries may carry
> `recipe.repeat_of_day` / `recipe.repeat_source`, `match_reasons` gains
> `kind: "repeat"` (with a `source` field), `metrics.variety` gains
> `planned_repeats` / `repeats_by_source` / `unexplained_repeats`,
> `metrics.repeats` is new, and `metrics.selection_events` gains three event
> types. **A 7-day plan can now contain the same breakfast twice** — the
> "21 distinct recipes" guarantee is retired for that one slot.
> Extra RecipeWrangler calls only when the food-waste slider is at `strict`.

Components 1 (sourcing half) and 2 of *"Weekly plans that look like how
people actually cook"* (`IDEAS.md`). Two mechanisms, one theme: a generated
week was 21 independently chosen recipes, and read like 21 shopping lists.

## The sourcing half of cross-day reuse

The scoring half shipped in *"Weekly reuse gets a clock"* and could only
reorder the pool it was handed. A day's pool is fetched knowing nothing
about what the week has already bought, so reuse happened when a shared
ingredient turned up by luck.

| What | Detail |
|---|---|
| The planner offers, the action space decides | Before each new day's pool is fetched, `WeeklyPlanner` calls `offer_derived_pantry` with the ingredients `IngredientBasket.reusable_items` says are still worth another meal. Duck-typed, so the fakes in the tests and the edit path's action space never receive the offer. |
| Sourcing asks for exactly what scoring rewards | `reusable_items` applies the same two rules the reuse bonus scores by — a gap of ≥ 2 days, at most 2 meals per ingredient. Sourcing something the scorer then penalises would buy latency and a worse week. |
| Gated on `strict` | One `plan_meals` request per ingredient per day. Capped at three ingredients. On by default it would be a latency cost paid by everyone to strengthen an axis most members leave `off`. |
| The member's items are left to their own fan-out | Anything matching the stated pantry is dropped from the derived list — it is already being searched for, and searching twice buys only latency. |
| The member's pantry still ranks the pool | Both merges sort coverage-first, so whichever runs last decides the top of the day's pool. The derived merge runs **first**, deliberately: what the member told us they have must outrank what the plan inferred. |
| Ingredients had to be named before they could be searched | `perishable_tokens` splits on whitespace, and "self" is no more a search term than it is a chip. `nameable_phrases` moved out of `explainability` into `planner` and now serves both — two definitions of "an ingredient" would have drifted apart within a release. |
| Both outcomes are recorded | `derived_pantry_sourced` per day that searched (with what it searched for), or `derived_pantry_skipped` once with the setting that skipped it. "This week reused nothing" and "this week never looked" are different answers. |

The ledger reports the search (`"look for recipes using ingredients the week
already buys"`, source `"food-waste setting"`) separately from the reuse,
which is still measured over the finished week by `shared_ingredient_facts`.
A search that found nothing usable is not a saving.

## Breakfast may come back

`RecipeActionSpace` excluded every committed id from every later fetch, so a
repeat was impossible at the *source*. Nobody eats seven different
breakfasts.

| What | Detail |
|---|---|
| A slot-scoped cooldown, breakfast only | A breakfast may return after ≥ 2 days, at most twice in the week, never in another slot. Lunch and dinner keep the original rule exactly. `mark_selected` still means never — pinned anchors and downvoted dishes have no way back. |
| It costs no extra requests | The fetch is per *day* and serves all three slots, so it uses the loosest exclusion any slot needs and the per-slot rule is applied at selection time. A test asserts one pool fetch per day, so a regression to per-slot fetching (3× the calls) cannot pass silently. |
| `mark_selected` split from `mark_committed` | The first means "never, at all" and is what the service calls for anchors and downvotes. The second carries the day and the slot, which is what a cooldown needs — without them a commitment can only mean "never again". |
| The variety penalty had to stop fighting the cooldown | −2 per shared title token, against an exact repeat, scales with how many words the recipe happens to be called — always enough to beat the cooldown. A sanctioned repeat is exempt from its **own** earlier title and from nothing else, and pays a flat `_REPEAT_PENALTY = 1.0` instead, so it loses to an equally good new dish. |
| A repeat earns no reuse bonus from its own ingredients | It shares everything with its earlier serving, so at `strict` the ingredient axis made repeating the cheapest way to score. Observed before the fix: a strict week repeated a breakfast at the first legal opportunity, every time. A repeat now sits out that axis entirely. |
| A thin pool repeats instead of failing | Four breakfasts across seven days used to raise `PlanGenerationError` on day 5. The cooldown strictly improves fillability — it only ever loosens an exclusion. |

## Which repeats were asked for, and which were not

The failure mode is not "too few repeats". It is a thin candidate pool
quietly producing a repetitive week that the plan then describes as a
feature. So every repeat carries the authority it repeated under, set at the
only point that can justify it and carried through unchanged.

| What | Detail |
|---|---|
| `repeat_source` on the candidate | `"member_request"` when the recipe is one the member starred, `"plan"` otherwise. Set in `get_candidate_actions`, and rides on the candidate onto the stored entry — the scorer, the environment and the explainability layer all read the same flag. |
| Two different chips | `"back from Monday, a favorite of yours"` vs `"the same breakfast as Tuesday"`, both `kind: "repeat"`, both carrying `source` for machine consumers. |
| A repeat is not also billed as ingredient reuse | A meal that IS an earlier meal shares every ingredient with it; showing "the same breakfast as Monday" beside "also uses Monday's rolled oats" says one thing twice and inflates the reuse count by the repeat count. |
| `variety_metrics` learned the difference | `planned_repeats` and `repeats_by_source` are reported apart from `unexplained_repeats` — a duplicate with no recorded reason (a pinned dish, a slot edit) is never folded into the sanctioned count and gets its own `violated` ledger row. |
| Measured against the policy, not asserted from it | The ledger row's status comes from the gap and appearance counts observed in the finished week, so a duplicate that reached the plate without passing the cooldown is reported as out of policy rather than described as intended. |
| The prose says it too | `_compose_reasoning` names the count and the split ("*2 meal(s) repeat earlier in the week … 1 you'd starred and 1 the plan's own choice, never closer together than 2 day(s)*"), and the response writer gets the same split in `facts["repeats"]` rather than a total. |

`annotate_shared_ingredients` now also **appends** its sentence to
`explainability["reasoning"]`. It runs last so its chips are additive, which
meant the whole-week justification was composed before anyone had measured
the reuse — so the one axis a member is most likely to ask about was the one
the prose never mentioned.

## Fixed on the way

| What | Detail |
|---|---|
| A counting word could impersonate an ingredient | `half` clears the length filter in `perishable_tokens`, and was stripped from a chip's *display* but not from its *stems*. "half a cabbage" and "half a pumpkin" therefore shared an ingredient, and the later meal was credited with reusing the cabbage. Measurements and counting words are now dropped before the stems are taken. |
| `env.reset()` orphaned the event list | The action space holds a reference to the same `selection_events` list, so rebinding it on reset would have silently dropped every sourcing event it recorded afterwards. Cleared in place. |
| `planner.py` had no module docstring | Contrary to the standing rule in `CLAUDE.md`. Added, covering the loop, the basket, the naming, and the scorer. |
| A repeat flag could outlive its partner | `edit_service` replaces a slot with a freshly built recipe dict, which clears the flag on the slot it edits and leaves the *other* serving still claiming to repeat a day that no longer has it. `repeat_facts` now checks each flag against the days that recipe is actually on, so a stale flag cannot become a ledger row. |
| A `strict` member could be recorded as having skipped the search | The skip branch was reached by falling through the sourcing condition, which also fails when the day pool comes back empty. The setting is now re-checked rather than inferred. |
| `metrics.repeats` carried list indices as keys | The payload is stored as JSON, so they would return as strings on the next read. Only the counts are stored; which meal is a repeat is already on the entry as `recipe.repeat_of_day`. |

## Verification

Every mechanism above was checked by neutering it and confirming a test
fails — 25 mutations (the cooldown's slot, gap and cap; the exclusion
loosening; the repeat labelling; the scorer's two exemptions and its penalty;
the sourcing gate, cap and de-duplication; the recording of both outcomes;
the stale-flag guard; each ledger row; the variety split; the prose). All 25
are caught. The first sweep found three that were not, including a test whose
fixture was derived from the constant it was testing — so raising the cap to
99 moved its goalposts with it and the assertion still held.

`tests/test_repeat_policy.py` (36) and `tests/test_reuse_sourcing.py` (22)
are new; `tests/test_shared_ingredients.py` grows from 21 to 24. Repeat
behaviour is asserted against the real `RecipeActionSpace` with only its
network edges stubbed, including a source that ignores `exclude_recipe_ids`
entirely — that is a request to RecipeWrangler, not a guarantee, and the
per-slot rule is the only thing between a downvoted recipe and the member's
plate. **565 passing**; ruff unchanged at its pre-existing 19.

---

# Cross-day reuse names the whole ingredient

> **Date:** 2026-08-27
> **Branch:** main
> No API change. Chip labels and the `"the plan"` ledger row change wording;
> `kind: "shared_ingredient"` and the row's shape are unchanged.

Follow-up to *"Weekly plans reuse ingredients without repeating them"*,
from reading its output on a live plan. The chips were naming tokens, and
a token is not an ingredient:

> *"also uses Thursday's raising and Thursday's self"*

That is one bag of self raising flour, reported as two ingredients, neither
of which is a thing you can buy. Same shape as the ones the previous change
caught (*"Monday's green"*, *"Wednesday's brown"*, *"Monday's leaf"*) — the
stoplist was treating symptoms, since whitespace tokenising will keep
producing new ones.

| What | Detail |
|---|---|
| A share is **found** by token and **named** by phrase | The overlap detection was never the problem — token matching is what makes "tomatoes"/"tomato" meet. Naming now uses the comma-separated phrase the token came from, as the *earlier* day wrote it, which is the day the label credits. |
| Several tokens from one phrase collapse into one item | "self" and "raising" both resolve to "self raising flour", so a chip names one ingredient instead of two fragments. This is what actually fixes the class of bug rather than the instances. |
| A phrase that names a staple is not a saving | "self raising flour" *is* flour, "brown sugar" *is* sugar, "macadamia nut oil" *is* oil, "thyme leaf" *is* thyme. `_PANTRY_STAPLES` already said sharing those saves nothing; per-token filtering dropped the staple and kept its modifiers, which is precisely how the modifiers ended up impersonating ingredients. |
| A phrase touching the member's pantry is dropped **whole** | "eggplant aubergine" anchors on a word the member never said, so it survived the per-stem filter — and would have been shown to a member who had just told us about their eggplants, crediting the plan for their own fridge. |
| Measurements, counting words and blobs | `_UNITS` and a new `_QUANTITY_WORDS` are stripped from the display ("half a cabbage" → "cabbage", "2 cups chopped cabbage" → "cabbage"), and a phrase over four words is a run-together blob ("brown sugar light brown cane sugar"), not a name — not reported. |

Replayed over three stored plans, every label now names something real:
*Tuesday's broccoli floret*, *Wednesday's old fashioned rolled oat*,
*Monday's bean pinto*, *Wednesday's mixed beans*, *Thursday's avocado*,
*Monday's capsicum*. Nothing reads as a fragment.

`tests/test_shared_ingredients.py` grows to 21 tests, including the flour
case and the pantry-phrase case verbatim. 504 passing.

---

# Weekly reuse gets a clock

> **Date:** 2026-08-27
> **Branch:** main
> No API change, no schema change, no new network calls. Weekly plans pick
> differently — see the note on `off` below, which now means slightly less
> than it used to. `pantry_service._singular` is now public as `singular`;
> nothing outside that module called it before.

Stage 1 of *"Weekly plans that look like how people actually cook"*
(`IDEAS.md`): sharing ingredients across days, without the sharing turning
into the same dinner four nights running.

The food-waste axis already rewarded a candidate for overlapping with what
the plan had already bought. It did it from a flat `set` of ingredient
tokens, which knows *whether* an ingredient has been used and nothing about
*when* — so reusing Monday's cabbage on Wednesday and reusing it at Monday's
dinner scored identically, and the set only ever grew. By day five almost
everything was in it, every candidate matched, and the signal meant to
separate reuse from repetition separated nothing.

No shelf life is modelled, here or anywhere else in the service: nothing
records when an ingredient was bought or when it spoils. The spacing below
is about the week being worth eating, and makes no claim about freshness.

| What | Detail |
|---|---|
| `weekly_planner/planner.py` — new `IngredientBasket` | Replaces the flat `set` the planner carried. Records each committed meal's ingredients against the **day** it lands on, so the scorer can ask how long ago one was eaten and how many times. Pinned slots go in too: a member's anchor puts food in the basket like any other meal. |
| `weekly_planner/planner.py` — new `_reuse_and_monotony()` | Returns reuse and monotony as **two** numbers, because they answer to different masters. Per shared ingredient: same day −1.0, the next day −0.5, two or more days later rewarded at the slider's weight. That is the whole rule — there is no upper gap, because a five-day gap saves the same shopping-list line a two-day gap does. |
| An ingredient is worth rewarding twice | Buy it once, cook it twice. A third appearance is not reuse, it is a theme — so it scores as monotony instead, however well spaced. Reuse should shorten the shopping list, not pick the week's flavour. |
| Monotony is **not** gated on the food-waste slider | `off` has always meant "sharing ingredients earns nothing", and it still does. It never meant "serve the same vegetable three days running" — the flat basket simply had no way to notice that it had. This is the one behaviour change for members who never touched the control. |
| The penalty is capped at −3.0, separately from the reuse cap | Uncapped, the scorer would quietly prefer recipes with short ingredient lists: they have less to collide with. That is a bias about recipe-writing style, not about food. Capped below the favourites bonus (+5) on purpose — same ordering rule the reuse bonus already followed: it nudges, it does not veto. |
| A stated pantry is spaced like any other reuse | Found by running it: a member who said *"I already have tomatoes, pasta, cabbage, eggplants"* got tomatoes in **15 of 21 meals, on all seven days**. The pantry boost is +3 per item, up to +6 — it outranks the −3.0 monotony cap on its own, so the spacing never got to bite. `_pantry_item_wanted()` now applies the same two rules the reuse bonus follows: not on an adjacent day, and not once the item has had two meals. "I have tomatoes" is a request to use them up, not to eat them daily. |
| The basket is number-symmetric | `perishable_tokens` does no stemming, so the same plan stored `tomatoes` (days 1–2) and `tomato` (days 1–7) as **two different ingredients** — and the monotony penalty compared each against only half its own history. Now keyed by `pantry_service.singular()`, the stem the pantry matcher already uses, which is the same asymmetry commit `7c3159f` fixed one module over. `tokens()` still returns the unstemmed set, so a pre-M8 scorer matches on exactly what it used to. |
| Cross-day reuse is a **different chip** from the pantry | `explainability.annotate_shared_ingredients()` adds `kind: "shared_ingredient"` — *"also uses Monday's cabbage — reducing food waste"* — beside the existing `kind: "pantry"` chip *"uses your tomatoes"*. A member can act on those differently: the first is the planner's doing about an ingredient nobody mentioned, the second is theirs and they can go check the fridge. An item the member named never carries the cross-day chip, so neither claim takes the other's ground. |
| Two ledger rows, attributed apart | `source: "your pantry"` keeps *"using N of M on-hand ingredient(s)"*; the new `source: "the plan"` row reports *"N meal(s) reuse an ingredient from an earlier day"* with the items in `detail`. The cross-day row appears whether or not a pantry was stated — the member said nothing about these ingredients, which is the whole point. |
| Naming is held to a stricter standard than ranking | `perishable_tokens` splits on whitespace, so "green beans, bay leaf, balsamic vinegar" yields `green`, `leaf`, `balsamic`. Harmless in the scorer — two meals sharing "green" really are a bit more alike, and averaging absorbs it — but the first run of this produced chips reading *"also uses Monday's green"*, *"Wednesday's brown"*, *"Monday's leaf"*. `_UNNAMEABLE` now drops colours, generic categories and preparation adjectives, and a share whose only overlap is unnameable **is not counted at all**. Under-reporting is the safe direction, the same posture the pantry matcher takes. |
| `WeeklyPlanner.generate_full_plan` dispatches on scorer arity | The scorer grew a fourth argument (the day being planned). Each arity is now called with exactly what it accepts — 4 gets the basket and the day, 3 gets `basket.tokens()` and behaves exactly as it did before this change, 2 is untouched. Adding the day could otherwise have broken a 3-argument scorer silently, which is what the old `>= 3` check would have done. |

The pre-M8 flat-basket contract still works if you pass a plain set: reward on
overlap, no spacing, no penalty. That path is tested, not merely left in.

`tests/test_food_waste.py` gains 19 tests. Verified as regressions rather
than decoration: zeroing the same-day and adjacent-day weights fails 6, and
the end-to-end spacing test then reports the shared ingredient landing on
days `[1, 2, 3, 4, 5, 6, 7]` — the exact failure the change exists to
prevent. With the weights in place it lands on `[1, 3]`: reused once,
spaced, then done. Removing the pantry gate fails 3 more, and un-stemming
the basket fails 5. `tests/test_shared_ingredients.py` is new (16 tests) and
covers the separation itself: a member-stated item must never carry the
cross-day chip, and vice versa. 499 passing (457 + 42).

Simulated against a tomato-heavy Italian pool (7 of 10 candidates carry the
stated item, as they did in the plan that exposed this): **21 of 21 slots
before, 2 of 21 after, on days 1 and 3**. What the scorer cannot fix is a
pool where nearly every candidate carries the ingredient anyway — Italian
cuisine and tomatoes — which is the sourcing half's problem, not this one's.

**Deliberately not in this change**, both recorded in `IDEAS.md`:

- *The sourcing half of stage 1.* Feeding a committed day's ingredients back
  through `pantry_service.fetch_pantry_candidates` would make reuse stronger
  by putting matching recipes in the pool rather than hoping they are there.
  It also fires a per-item HTTP fan-out on **every** weekly plan, where today
  that fan-out only runs when the member actually stated a pantry — a
  latency regression for every user, to strengthen an axis most of them
  leave `off`. Worth doing behind the slider, not worth doing blind.
Replayed over the plan that exposed the pantry bug, the two now read apart:
9 pantry chips naming tomatoes/pasta/cabbage/eggplants, and 8 cross-day
chips naming bread, capsicum, zucchini, courgette, carrots, celery,
aubergine, beans and almond — with `13 meal(s)` dropping to `8` once the
unnameable shares stopped counting.

---

# A named dish obeys the same constraints as a plan (Phase 1d)

> **Date:** 2026-08-21
> **Branch:** fix/stated-diet-and-honest-constraints
> Pairs with RecipeWrangler `fix(tools): find_recipes honours the favourites it
> already accepted`. Independent of it — foodchat sending `favorite_recipe_ids`
> to a RecipeWrangler that ignores them is harmless.

`find_recipes` resolves "I want pancakes". It took the member's allergens and
diet — added precisely so a seed the member cannot eat is never offered — but
not the **Nutri-Score floor** or the **cooking-time slider**, both of which
applied to every other fetch in the service. So a member with a 20-minute limit
could have a 90-minute dish anchored into their plan, by a lookup that ignored
the constraint the plan itself was built under. Now sent, along with the
member's favourites and standing exclusions, at all three call sites: seed
resolution, the edit path's named-dish lookup, and the pantry boost.

**A test that read source instead of running it hid a NameError.** The first
version asserted `min_nutri_score` appeared in `pantry_boost_ids`' source, and
passed — while the function raised `NameError` on every call, because the local
import block it needed was in a *different* function. `ruff` caught it (`F821
Undefined name`), not the suite. The test now executes the path with a stubbed
client and asserts the arguments that arrive; reverting the import makes it
fail. A source assertion cannot see an undefined name.

611 passing.

---

# The sliders and standing answers that did nothing (Phase 1c)

> **Date:** 2026-08-21
> **Branch:** fix/stated-diet-and-honest-constraints
> No API change. `difficulty=easy` now narrows the search, so plans for members
> with that setting will change.

Four things the member had already told us, which reached nothing.

**A goal accepted mid-conversation left the plan identical.** The session
mirror set `dietary_goals`, `preferences` and `nutrition_profile` — but not
`min_nutri_score`, which is the **only** goal-derived value that reaches
`plan_meals` (`nutrition_profile` has no parameter to travel on). So accepting
"lose weight" changed three fields, one of which nothing reads, and the next
plan was the same plan. Now mirrored: accepting `lose_weight` sets the floor to
`B` immediately.

**"Not that one" was recorded, persisted, and ignored.**
`state.excluded_recipe_ids` reached only the structured path. On the classic
daily path — the default — and on the weekly path, a recipe the member had
explicitly rejected came back on the next regeneration. Both now send it, the
weekly path through the same `mark_selected` channel downvotes already used.

**"No thanks" to the favourites offer held on daily only.** Weekly kept adding
`+5` per favourite and putting them in the week. The code comment describing
this exact bug as fixed was written for the daily path; the weekly path had
never been connected. A member who says no and sees their favourite anyway has
been told their answer does not matter.

**The difficulty slider was a pure no-op.** `grep -ri difficulty` across
RecipeWrangler's source returns **nothing** — no field, no tag, no vocabulary.
So it was prose for a grader that two of the three planning paths never run.
`easy` does have honest proxies in the corpus (`30_minutes_or_less` 2809
recipes, `5_ingredients_or_less` 563) and now uses them. `medium` and `hard`
map to nothing, deliberately: there is no "elaborate" annotation to ask for and
inventing one would empty every slot. They stay selectable — removing an option
is a UI contract change — but they apply nothing instead of pretending to.

605 passing.

---

# Claim tags reach the search (Phase 1b)

> **Date:** 2026-08-21
> **Branch:** fix/stated-diet-and-honest-constraints
> **Requires RecipeWrangler** to gain the `tags` parameter — but is safe to
> deploy in either order: the key is only sent when the live manifest advertises
> the vocabulary, so an older RecipeWrangler never sees it.

Nutrition claims — "high protein", "low carb" — had nowhere to go. They are not
diets (no recipe carries one as a `diet_tag`, so sending one as a diet filter
empties every slot, which is the outage found yesterday), and the field they
DO belong on did not exist upstream.

**RecipeWrangler** now takes `tags` on `plan_meals`, filtering the corpus's
human-authored claim field: `high_protein` (1676 recipes),
`30_minutes_or_less` (2809), `healthy_and_nutritious` (2535), `low_fat` (1081),
`5_ingredients_or_less` (563), `high_fibre` (457), `low_calorie` (184).

The design decision worth keeping: **`tags` leads the relaxation ladder.** A
claim is the softest thing a caller can ask for and the scarcest annotation in
the corpus — `high_fibre` is on 10% of recipes, so two claims ANDed across 21
slots would starve most of them. Dropping it first means "high protein and high
fibre" narrows the search when it can and widens when it cannot, instead of
returning an empty week. The vocabulary is published in the manifest so a caller
can avoid asking for a claim nothing carries, and it is an *open* field, so an
unlisted value is reported and still applied — it relaxes first, so it cannot
strand a slot.

**FoodChat** now sends them, from two sources:

| Source | Example |
|---|---|
| The slider goal | `energy` → `high_protein` + `high_fibre`, alongside the `hearty` mood |
| A claim stated in words | "high protein please" → `high_protein` |

**This fixes a dead end I shipped yesterday.** Claims were routed to
`PlanningState.notes` and described as reaching "the grader as soft signals".
`notes` is **write-only** — read solely by `describe()`, which is only logged. So
a claim was correctly saved from becoming an empty filter and then dropped on the
floor. They now live in `PlanningState.claim_tags` and reach the request. The two
tests that asserted the `notes` behaviour have been corrected, and one now
asserts `notes == ()` so nothing is routed there again.

**The capability gate.** RecipeWrangler's request model is `extra="forbid"` — an
unknown field is a 422, not a shrug — so foodchat only adds `tags` to the payload
when `GET /api/v2/tools` advertises the vocabulary. The vocabulary IS the
capability flag, it is already cached, and it costs nothing per call. Verified
both ways: advertised → sent, not advertised → withheld with the rest of the
request untouched.

Also on the RecipeWrangler side: `applied` now reports `tags` (it is the
service's own account of what it filtered on, and a missing entry would make the
plan unexplainable downstream), and `never_relaxed` no longer under-declares —
it listed three constraints while `include_ingredients`, `min_nutri_score`,
`sources`, `exclude_recipe_ids` and `course_types` were equally hard. A new test
asserts the two lists cannot overlap, since together they are the whole contract.

597 passing in foodchat, 19 new tests in RecipeWrangler.

---

# "Energy boost meal plan for today" now matches something (Phase 1a)

> **Date:** 2026-08-21
> **Branch:** fix/stated-diet-and-honest-constraints
> No RecipeWrangler change — these are parameters it has always accepted.
> Wire-compatible: the four facet tuples are additive on the planning-state
> blob and absent on anything stored earlier.

That request matched nothing, for three reasons stacked on top of each other:

1. `plan_client.plan_meals` **declares** `moods`, `flavor_profiles` and
   `food_groups`; RecipeWrangler accepts all three, describes them in its
   manifest, and puts them **first in its relaxation ladder** so they degrade
   gracefully. **No caller ever passed any of them.** `cuisines` was the only
   facet ever sent, and only from the stored profile.
2. There is **no cuisine extractor anywhere**. "Something Thai tonight" never
   became a `cuisines` filter on any path.
3. **"energy" is in no vocabulary at all** — not a mood, not a flavour, not a
   food group. It is a `plan_parameters.goal` value that only ever became prose
   for a grader that two of the three planning paths do not even run.

So the words reached the grader as text over a pool that had never been shaped
by them, and the plan came back indistinguishable from one with no request.

| Piece | What it does |
|---|---|
| `CANDIDATES.split_preferences` | Generalises `split_cuisines` to all four families, keeping both properties that made it work: the vocabulary is fetched **live** from RW's manifest, and the sort happens at **read** time so existing profiles are fixed with no migration. A stored "comfort" now drives a mood instead of being searched for as an ingredient. |
| `PlanningState.cuisines/moods/flavor_profiles/food_groups` | Standing session state, mirroring `diet_tags`: additive, never cleared by silence, with `facets_remove` for an explicit take-back — which is also what the UI's removable chips will call. |
| `PlanIntentExtractor` | A **new** agent under **new** prompt names, because `DietaryIntentExtractor`'s prompt is Langfuse-managed and extending it would ship dead. The live vocabulary is injected into the prompt *and* re-validated after the model answers. |
| `intent_facets.facet_kwargs` | One `**` replaces one `cuisines=` at every fetch site, so no site had to learn about the other three families. |
| `GOAL_FACETS` / `GOAL_CLAIM_TAGS` | Slider goals map onto vocabulary that exists. **`energy` → the `hearty` mood + the `high_protein` and `high_fibre` claim tags** — read as sustaining food rather than inventing an "energising" facet the corpus does not carry. The judgement is written down in the table instead of buried in a prompt. |

**The rule this is all built around:** never send a value the corpus does not
carry. RecipeWrangler ANDs facet values and does not relax an unlisted one to
nothing — it matches no recipe. So a hallucinated mood does not soften the
search, it empties it, and the member is told no meals exist. That is the same
failure shape as the `low-carb` outage found yesterday, and the reason the
vocabulary is checked twice.

Facets now reach the request on the classic daily pool, the structured path, the
weekly pool, the pantry fan-out and slot candidates — verified by capturing the
actual `plan_meals` kwargs.

`tests/test_intent_facets.py` is new (25 tests, LLM-free), verified regressive:
stopping the facet merge fails 3. 589 passing.

**Not yet wired**: the claim tags. `plan_meals` has no `tags` parameter, so
`claim_tags_for()` returns the right answer and nothing can send it — that is
the RecipeWrangler half of Phase 1, and it deploys first because the request
model is `extra="forbid"`.

---

# Never claim a constraint we did not enforce (P0)

> **Date:** 2026-08-21
> **Branch:** fix/stated-diet-and-honest-constraints
> No API change. One UI-visible addition: `constraints_applied` rows can now
> carry `status: "unsupported"`, which clients must render — an unknown status
> falling through to "satisfied" styling would reinstate the bug.

`normalize_diet_tags` drops **26 of the gateway's 37 dietary groups** — RecipeWrangler
has no diet tag for `peanut_free`, `halal`, `kosher`, `keto`, `low_sodium` and
the rest. `constraints_ledger` then rendered **every** raw profile diet value as
`type: hard, status: satisfied`.

So a member who selected `peanut_free` in their profile was shown a plan header
asserting a peanut-free guarantee, with no filter behind it — and no allergen
backstop either, because the backstop keys on plain-English allergen names and
never saw the slug. Of everything found in the full-stack sweep this is the only
item that is not merely a missing feature.

| Fix | Detail |
|---|---|
| `classify_diet_tags` returns `(filterable, unsupported)` | Unsupported values are handed back to the caller instead of dying in a log line. `normalize_diet_tags` is now a thin wrapper, so every existing call site keeps working. |
| A new `unsupported` ledger status | The row says the catalogue has no filter for this and it did not narrow the search. Never `satisfied`. |
| Free-from slugs reach the ingredient backstop | `FREE_FROM_TO_ALLERGEN` maps `peanut_free → peanuts`, `egg_free → eggs`, `shellfish_free → shellfish` and six more onto the existing allergen synonyms, and `screening_allergens(profile)` unions them into the screen at all ten sites that already screen. It cannot invent an upstream filter; it can make the defence that exists cover the slug. |
| Non-restrictive labels get no row at all | `omnivore` was listed as a *satisfied hard constraint* — claiming the plan honoured something never asked of it, on a row the member cannot act on. |
| `unsupported` is in neither half of `split_ledger` | Calling it honoured is the lie this exists to stop; putting it in the reply as "couldn't honour peanut_free" would over-alarm a member whose peanuts **are** screened. The ledger row carries the nuance; prose does not flatten it. |

**Two corrections to work shipped earlier today.**

`GATEWAY_DIET_GROUPS` was limited to the five values the UI picker offers. The
gateway enum also holds `gluten_free`, `dairy_free` and `nut_free` — *exactly*
the three diets FoodChat can filter on. So "remember I'm gluten-free" was
offered, accepted by the member, refused at the write, and returned
`applied: false`: the three it could act on were the three it would not persist.
The set now matches the gateway enum exactly, with a test asserting parity in
both directions (a missing value silently refuses a legitimate memory; an extra
one 422s at the boundary).

The chatbot persona told the member *"You can steer by cuisine, mood, flavour,
food group, cooking time, Nutri-Score and calorie or protein targets."* Four of
those seven are not implemented — mood, flavour and food group are never sent to
RecipeWrangler, and the endpoint has no macro parameter. I had earlier reported
this promise as harmless because `describe_options()` has no callers; that was
wrong. The same claim sits in the persona, which is the one place a member
actually reads it. Corrected under a **new prompt name** (`chatbot_system_v2`) —
a deploy never overwrites an existing Langfuse copy, so editing the in-code text
would have left the false promise live in production forever.

`tests/test_unenforced_constraints.py` is new (47 tests, LLM-free), verified
regressive: restoring the always-satisfied behaviour fails 27 of them. Every
mapped allergen is asserted expandable by the synonym table, because a mapping
to a name the table does not know would screen nothing and silently reopen the
hole. 564 passing.

---

# A local tool surface for the agent

> **Date:** 2026-08-21
> **Branch:** fix/stated-diet-and-honest-constraints
> Additive: two new endpoints, no change to any existing one. No UI change
> required — the UI can call a tool without a chat turn if it wants to.

The agent was a fixed chain: one classification per turn picked one handler,
and anything that handler could not do was unreachable. "Summarise my week"
and "redo Thursday" had no path at all — the nearest available action was a
full refinement, which regenerates all 21 slots and silently discards a slot
edit the member had already approved.

`src/tools/` is a declarative registry speaking **the same protocol FoodChat
already consumes from RecipeWrangler** — `GET /foodchat/tools` for a manifest,
`POST /foodchat/tools/{name}` to invoke. Rather than invent a second shape, the
service now speaks the one it already understands, MCP-shaped so a model can be
handed the manifest directly.

| Tool | What it does |
|---|---|
| `summarize_week` | Every day with its meals and calories, week totals, the guideline checklist and variety metrics read back from the stored plan, and the ledger split into what held and what was relaxed. Read-only, no model call. |
| `summarize_day` | One day in detail: ingredients, per-meal and whole-day nutrition, and the reason chips for each dish. |
| `plan_totals` | Sums a plan's calories and macros — per plate, per day, per week. **This total did not exist before**: the daily path had no summation at all and the prose prompt asked the model to notice when a day "sums far outside a sensible intake". |
| `replace_day` | Regenerates one day and pins the other eighteen slots, so the rest of the week survives byte for byte. The surgical alternative to refining the week. Excludes everything already in the plan, everything downvoted, and the day being replaced, so the new day is genuinely new. |
| `swap_meal` | The existing verified slot edit, exposed as a callable tool on either canvas. |

Every reader is LLM-free and every total reports how many meals actually
carried nutrition data, rather than implying a complete figure.

**The plan analyst now gets the arithmetic done for it.** It was handed 21
per-meal nutrition strings and no total, so any question about a whole day or
week made it add up numbers in prose — the one thing a model should not be
trusted with here. `_summarize_active_plan` now appends the summed totals with
an explicit "already summed — do not re-add", plus the coverage caveat. No new
intent, no prompt change, no extra model call.

Design notes worth keeping:

- Ownership is enforced in the **router**, not the tool: a tool trusts that its
  caller proved the member owns the session, the same contract every service
  here follows. A session the caller cannot see returns 404, never 403.
- `ToolError` is the member-facing failure — a bad day number, no plan yet —
  and becomes a 400 carrying prose. Anything else is a 500 and a log line.
- Argument validation lives in the registry, so a wrong day fails with a
  readable sentence instead of surfacing from inside a planner.
- `mutates` and `uses_model` are declared per tool, so a caller can decide
  whether it can afford one inside a turn that has already spent grading.

`tests/test_tools.py` is new (27 tests, LLM-free). The load-bearing one asserts
the planner still bypasses selection for pinned slots — if that ever stops
being true, `replace_day` silently becomes a full regeneration and starts
eating approved edits. 515 passing.

**Agent-side selection is deliberately not wired.** The orchestrator's intent
list lives in a Langfuse-managed prompt, and a deploy never overwrites an
existing copy — adding an intent there would work locally and ship dead to
production. Reaching these tools from a chat turn needs either a new prompt
name (the `PantryExtractor` pattern) or a manual Langfuse version push. Until
then they are reachable from the API and from code, and the analyst already
benefits.

---

# A diet you state in chat is a diet we plan with (Phase A)

> **Date:** 2026-08-21
> **Branch:** fix/stated-diet-and-honest-constraints
> Ships with wisefood-api (`feat/foodchat-plan-library-proxy` branch gains the
> gateway fixes below). No UI change. Wire-compatible: `diet_tags` is additive
> on the planning-state blob and absent on anything stored earlier.

The transcript that started this:

    member    "i need something vegetarian"
    assistant "The current plan includes chicken, pork, and meatballs, which
               conflict with your request … adjust the plan to be fully
               vegetarian?"
    member    "yes please"
    assistant "I couldn't find enough recipes … (diet: omnivore; allergens
               excluded: nuts, peanuts; avoiding: mushrooms)."

Three independent causes, all seams:

| Cause | Fix |
|---|---|
| `DietaryIntentExtractor` was wired **only** into the weekly service, so a daily plan never read diet from the message. | New `services/diet_intent.py` runs it on the RAW message every planning turn; `PlanningState.diet_tags` makes it **standing** for the session (silence is not a retraction), and `candidates_client.effective_diet` unions it with the profile at **every** fetch site — base pool, pantry fan-out, seed lookup, edit swap. |
| Nothing carried the resolved diet out of the conflict question: `merged = {**profile, **reconciliation}` merges four keys and none is `diet`, so "yes please" was discarded. | The tags were already captured from the original message, so "yes" needs no plumbing. Only a refusal has to act: `is_conflict_refusal` retracts them on "no" / "follow my profile". Deliberately narrow — an unrecognised answer KEEPS what the member said out loud. No reconciler prompt change (it is Langfuse-managed; an edit ships dead). |
| No memory kind could set a diet — `constraint` lands in free-text history. | New `diet` kind writes `dietary_groups` through the one lossless gateway path, replacing a non-restrictive `omnivore` rather than sitting beside it. The nudge is built **deterministically** from planning state — no LLM call and no `preference_extractor` prompt edit — and carries the member's own sentence as `evidence`. |

**A live outage found on the way.** `low-carb`, `low-fat` and `high-protein`
were in `VALID_RW_DIET_TAGS` and mapped straight through as hard diet filters.
Censused against the corpus dump (n=4500) they appear on **zero** recipes —
they live on RecipeWrangler's separate claim field. RW ANDs diet tags and never
relaxes them, so "I want a low-carb week" was not a narrow search, it was a
guaranteed-empty one, and the member was told no recipes exist. Already live on
the weekly path; threading diet into daily would have spread it. They are now
routed to the grader as soft signals via `split_diet_intent`, and become real
numeric targets when the planning surface grows nutrition targets.

Also fixed:

- **The apology named a constraint it never applied.** It listed the raw
  profile, so the blocker read `diet: omnivore` — dropped before the request —
  with no mention of the vegetarian filter. It now reports what was *sent*, and
  says plainly when a stored value is not a restriction. With nothing to
  contrast against, the note is omitted rather than accusing a setting that did
  nothing.
- **The weekly tracker budgeted a stated vegetarian three meat meals** and
  counted every fish meal against it, because it read only the stored profile.
- **Guest memory acceptance was a silent no-op**: a guest household has no
  profile ROW, the gateway 404s, and the SDK swallows it — so the member said
  yes, we agreed, and stored nothing. The row is now created first.
- Deleted the fictions: the `.cypher` guideline read (the file is not in the
  repo — every adherence score has come from an empty context, and had it
  existed it would have pasted raw Cypher into a prompt; rules come from the
  data catalog as faceted `rule_text`); `supports_macro_targets: True` and "hit
  calorie or protein targets", which told the agent it could promise something
  `plan_meals` has no parameter for; and `eat_healthier`, which sat in a
  Nutri-Score map but is rejected at the write gate so could never arrive.
- **wisefood-api**: `diabetic_friendly` added to the drifted `sql.py` enum (a
  PATCH carrying it passed Pydantic then 500'd); `properties` — dietary goals,
  standing seeds, memory log — no longer silently dropped when a profile row is
  created; the member-PATCH path now invalidates the profile cache it mutates.

`tests/test_stated_diet.py` is new (22 tests, LLM-free) and verified regressive:
reverting `effective_diet` to profile-only fails 2. 488 passing.

Deferred to the reasoning phase, on purpose: quality metrics still run only on
the classic daily path. The structured path builds a `MealPlan` with no
`ScoredPlan`, so a metrics adapter written now would be replaced by the
verifier that unifies metrics across all three paths.

---

# Sessions name themselves

> **Date:** 2026-08-21
> **Branch:** main
> Pairs with the gateway proxy (wisefood-api `feat/foodchat-plan-library-proxy`)
> that makes rename reachable from the UI. Wire-compatible: `title` was already
> nullable in every response.

Sessions were only ever named by an explicit rename, which almost nobody does —
so the session picker showed a wall of timestamps, and a saved plan inherited
no name at all (the save path borrows the session title). The router comment
even claimed the client falls back to the first user message; it never did —
it falls back to `created_at`.

| Piece | Behaviour |
|---|---|
| `SessionTitler` (agents.py) | Names the conversation from its opening message. Fast tier, plain text (the whole answer IS the title — a schema would only add a wrapper), 3–6 words, `NONE` sentinel for unnameable openings. A rambling or over-long answer is rejected rather than truncated into a half-name. |
| Prompts | `session_title_system` / `session_title_user` — NEW managed-prompt names, per the standing rule: deploys never overwrite an existing Langfuse copy, so extending an existing prompt ships dead to production. |
| Wiring (orchestrator `process()`) | Fires once, AFTER the turn completes, only when the session has no title and no prior user message — so it can never delay or break an answer, a member rename always wins, and it never re-fires. Failure is a log line, never a 500. |

The first-turn signal is read BEFORE routing: handlers append the message, after
which "no user messages yet" is no longer true.

457 passing (LLM-free: the titler is constructed against the fake key like
every agent; `_clean` is covered by direct calls).

---

# Pantry review follow-ups

> **Date:** 2026-08-20
> **Branch:** main
> No API change. One UI-visible contract change: the `match_reasons` chip
> `"cooked from your leftovers"` is gone (see below) — clients render chips
> generically, so nothing breaks.

Fixes from the review of PR #1, applied after the merge.

| Fix | Why it mattered |
|---|---|
| The `uses_ingredient` directive no longer captures greedily. | `([a-z\s-]{1,40})` swallowed whatever followed the ingredient — "zucchini please", "chicken instead", "zucchini and spinach". A junk term hard-fails: the single-item include finds nothing, the text match finds nothing, and the member is told no candidate satisfies a request that previously produced an ordinary swap. Now at most two words, with trailing filler stripped and comparative openers ("less salt") rejected back to `unverified`. A two-item request verifies the first item rather than searching for an ingredient literally named "zucchini and spinach". |
| The matcher is number-symmetric. | The pattern appended an optional `s`, which only worked one way: a member who said "tomatoes" — the natural way to name fridge contents — never matched a recipe listing "tomato". The plan used the item while the reply said *"I couldn't work in your tomatoes"* and the ledger recorded the coverage `relaxed`. Under-reporting is supposed to be the safe direction; here it produced an affirmative false claim. Each word is now stemmed and re-inflected. "rice" still does not match "price". |
| The fan-out carries the caller's `cuisines` and `max_minutes`. | The pantry pool is merged into the ordinary one and sorted coverage-first, so a constraint the fan-out dropped did not merely appear — it appeared at the **top**. A member with the cooking-time slider at 20 minutes who mentioned a courgette got a 90-minute bake ranked first. All three call sites now pass what their own base pool passes; the edit path matches `slot_candidates` exactly rather than letting its two branches offer differently-constrained pools. |
| The `"cooked from your leftovers"` chip is removed. | It was emitted on every match, unconditionally. "I picked up courgettes today" is a pantry statement too, and nothing in the state records whether an item is a leftover — so the claim had no measurement behind it, in a module whose stated rule is that every user-facing claim comes from the matcher. The food-waste chip already carries the intent without asserting the item's history. |

Also folded in from the merge: `PantryExtractor` moved to `FOODCHAT_FAST_MODEL`
with an inheritable temperature. Its hunks did not overlap the ones that moved
the other five extractors, so git merged it cleanly onto the 120b reasoning
model with `FOODCHAT_LLM_TEMPERATURE` shadowed — the exact bug the model
migration had just removed.

`tests/test_pantry_followups.py` is new (21 tests). Verified as regressions,
not decoration: reverting the stemmer fails 2, reverting the capture fails 1.
457 passing.

Still open from that review, deliberately: the pantry branch in
`_handle_preference_update` returns early on any pantry delta, so *"I love
spicy food and I've got leftover rice"* replies about the rice and drops the
preference acknowledgment plus its memory nudge; and `plan_structured` spends
`favorite_recipe_ids` — the structured path's only soft rank signal — entirely
on pantry ids instead of merging both lists. Neither is a false claim to the
member, which is why they waited.

---

# Memory provenance + per-diner constraint attribution

> **Date:** 2026-08-20
> **Branch:** fix/model-migration-config-chain
> Ships with wisefood-ui (the memory panel renders `evidence`). Wire-compatible:
> `evidence`, `members` and `constraint_origins` are all additive and absent on
> anything stored earlier.

Two questions the plan could not answer before: *why am I seeing this memory*,
and *which of us is this constraint for*.

| Area | Change |
|---|---|
| Provenance | The extractor's own justification for a suggested memory was computed and then dropped. It now rides accept/decline and is stored with the memory, so the panel can say what it was inferred from. |
| Attribution | Ledger rows carry `members`, sourced from a new `constraint_origins` map built during `merge_profiles`. Listing every diner on every row read as if the whole table were allergic and hid the one person the row exists for. Attribution is suppressed for solo plans, where it says nothing. |
| Goals | `goal_reconciliation` records which diner asked for each goal and whether it became a numeric target or was demoted to a soft preference. A goal that steered the plan invisibly, and a goal dropped without a trace, were the two failure modes. |

Fixes found while reviewing the above:

- **A relaxed constraint was announced as honoured.** `constraints_applied[:4]`
  fed the response writer under `constraints_honored`, but the ledger mixes
  `satisfied`, `relaxed` and (weekly) `violated`. The writer is told to mention
  "an honored request" and believed the key, so a reply could claim a goal was
  met while the ledger beside it said otherwise — reproduced with `increase
  protein` landing fourth. `transparency.split_ledger` now splits by status for
  both the daily and weekly paths, and passes `constraints_not_honored` too, so
  the prompt's standing instruction to "say plainly" what could not be honoured
  finally has a fact behind it. A row with an unrecognised status is claimed
  neither way — plans stored before the field existed must not be asserted
  either direction.
- **A memory accepted mid-session was invisible to the ledger.**
  `_apply_to_session_profile` bypasses `merge_profiles`, so a newly accepted
  allergy rendered with `members: []` beside attributed siblings, and an
  accepted goal got no row at all. Both records are now updated on accept, and
  deliberately not for solo sessions, where writing them would invent a
  household of one.

`tests/test_ledger_honesty.py` is new (11 tests): three fail without the
attribution fix, and the slicing test asserts the old `[:4]` behaviour produced
the bug it replaces. 405 passing.

Known, not fixed here: the goal `detail` strings describe a numeric-target
mechanism that does not exist yet (`nutrition_profile` is written but never
read — `plan_meals` has no macro parameter); a demoted slug missing from
`GOAL_PREFERENCE_STRINGS` is still recorded `applied: "soft"`; sessions created
before this deploy show no attribution until diners are re-set; and `evidence`
is client-echoed text stored verbatim into durable provenance while
`kind`/`value` are re-validated — a trust-model decision, not a bug fix.

---

# Model migration + one honest config chain

> **Date:** 2026-08-20
> **Branch:** fix/model-migration-config-chain
> Ships with platform-deployment (`lib/foodchat.libsonnet` gains the model
> block). No wisefood-api or wisefood-ui change. Deploy the two together: the
> image default alone is correct, the manifest just makes it explicit.

Groq shut down `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` on
2026-08-16. Both were FoodChat's defaults, so every LLM call had been failing
since — silently, because each agent catches the error and falls back: plans
lost their grader ranking, extractors returned empty, prose writers emitted
canned text. Nothing 500'd, so nothing alerted.

| Area | Change |
|---|---|
| Models | Reasoning tier → `openai/gpt-oss-120b`, matching foodscholar so the platform runs one reasoning family. The five structured-output extractors (dietary tags, plan spec, seeds, preferences, edit commands) → `openai/gpt-oss-20b` via the new `FOODCHAT_FAST_MODEL`: they pick spans rather than reason, and several run per planning turn. `qwen/qwen3.6-27b` is the documented cheaper fallback. |
| Reasoning families | New `backend/model_profiles.py`. `gpt-oss`/`qwen3`/`deepseek-r1` return their deliberation inside `content` at the provider default, which breaks every `json.loads` in agents.py into a silent fallback — the exact failure mode above, with a live model. The pool now injects `reasoning_format="hidden"` and floors `max_tokens` per family, and drops the reasoning params for families that 400 on them (`ChatGroq` is `extra="ignore"`, so a wrong knob fails one layer away at the provider). A `RETIRED` table warns on a shut-down id with its date and replacement — this is what would have caught 08-16. |
| Config chain | `GROQ_DEFAULT_MODEL` and `GROQ_DEFAULT_TEMPERATURE` were unreachable dead config: documented, read, and shadowed at every call site, because a second `os.getenv` with its own literal fallback is never `None` so `model or GROQ_DEFAULT_MODEL` could never fire. Callers now read `os.getenv(NAME)` with no default; `backend/groq.py` holds the only literal. Resolution is narrowest-wins: ctor arg → `FOODCHAT_FAST_MODEL` → `FOODCHAT_LLM_MODEL` → `GROQ_DEFAULT_*`. |
| Temperature | `FOODCHAT_LLM_TEMPERATURE` reached only 6 of 13 agents — five extractors hardcoded `temperature: float = 0.0` in their signature, so "not None" always won. All signatures are now `= None`; an unparseable value warns and inherits instead of raising. `FOODCHAT_CHATBOT_TEMPERATURE` keeps its own literal: prose is a separate setting, not a shadowed default (it governs `SimpleChatBot`, `ResponseWriter`, `PlanAnalyst`). |
| `LOG_LEVEL` | Set by both Dockerfile stages and read by nothing, so the dev image logged at INFO like production. Now honoured, validated against a fixed set so a typo falls back to INFO rather than failing the boot. |
| Docs | `.env.example` states the resolution order and forbids adding a literal at any other level — the sprawl was never the count (33 vars, all documented, all read), it was that three of them silently did nothing. |

`tests/test_model_profiles.py` is new (20 tests, LLM-free): reasoning stays
hidden for every reasoning family, the token floor raises but never lowers, a
caller's explicit knob beats the family default, non-reasoning families get the
params dropped, and profiled configs never collide on one cached pool client.
394 passing.

Not in scope: the `saved-plans` / session-rename routes the UI calls and the
gateway does not proxy (three dead UI features — separate change, needs a
`patch` verb on the gateway's foodchat client), and the pantry-planning
follow-ups tracked on PR #1.
# Fix: intermittent 500 "Object of type PlanSpec is not JSON serializable"

> **Date:** 2026-08-18
> **Branch:** main
> No API change. Fixes a live intermittent 500 on plan turns.

`process_plan_request` stashed the standing plan shape in the session profile
snapshot as a live `PlanSpec` dataclass (`profile["_plan_spec"] = state.spec`).
That snapshot rides inside `ClarificationState` and is `json.dumps`-ed onto
the session row — so **any turn that asked a clarifying question died** while
trying to store the question it had just asked, and the member got a 500.

It looked random because *whether a turn clarifies* is an LLM decision
(`QueryReconciler` + the specificity check): the same sentence clarifies on
one turn and not the next, and only the clarifying path serializes. Turns that
planned straight through popped `_plan_spec` in `_generate_and_store` before
anything touched JSON, so retrying "worked".

| What | Detail |
|---|---|
| `src/models/plan_spec.py` | `PlanSpec.to_dict()` (the JSON-safe form `from_spec` already read back) and `PlanSpec.coerce()`, which accepts a stored dict or a live instance — an in-process session can still hold the object. |
| `src/services/chat_service.py` | Stores `state.spec.to_dict()`; `_generate_and_store` coerces it back, so the shape still survives a clarification round-trip and still routes to the structured path. |
| `src/models/planning_state.py` | `to_dict()` reuses `PlanSpec.to_dict()` instead of open-coding the same three fields — the two can no longer drift. |
| `tests/test_clarification.py` | `TestClarificationStateIsPersistable` — a clarifying plan turn with a non-default spec must persist, and the stored state must round-trip to an equal, still-non-default spec. Verified to fail with the original `TypeError` before the fix. |

Introduced by `0c7b96a` (multi-plate/N-day dispatch). Audited the other
transient snapshot keys (`_pinned_slots`, `_seed_note`, `_excluded_recipe_ids`,
`_pantry`, `_favorites_declined`) — all plain JSON types.

---

# Pantry planning — "cook from what I have" (food waste)

> **Date:** 2026-08-18
> **Branch:** main
> FoodChat-only (PANTRY_PLANNING_PLAN.md Tier A — no RecipeWrangler change
> assumed). Wire-compatible: no schema change; badges ride the existing
> `match_reasons`/ledger contracts (new chip kind `"pantry"`).

A member states ingredients they have at home ("I've got zucchini, spinach
and some ground beef"); both horizons boost recipes that use them and report
measured coverage — used AND unused items — honestly. The core constraint:
`plan_meals` ANDs `include_ingredients`, so the whole pantry at once would
empty every slot; sourcing is a capped per-item fan-out (single-item hard
include = "must use this one thing"), merged coverage-first, pool size
unchanged. Every user-facing claim comes from a deterministic word-boundary
matcher, never the LLM.

| What | Detail |
|---|---|
| `src/services/pantry_service.py` (new) | Matcher, regex-gated extraction → `PlanningStateDelta`, threaded per-item candidate fetch (best-effort), coverage-first pool merge, coverage facts, badge/ledger annotation for both plan shapes, honest coverage prose. Knobs: `FOODCHAT_PANTRY_ITEM_LIMIT` (6), `FOODCHAT_PANTRY_PER_ITEM_CANDIDATES` (3) — in `.env.example`. |
| `src/models/planning_state.py` | `PlanningState.pantry` (durable, additive; "used up the X" removes; reset clears) + delta `pantry_add`/`pantry_remove`. Silence is not a retraction. |
| `src/agents.py`, `src/prompts.py`, `src/schemas.py` | `PantryExtractor` with its OWN prompt name (`pantry_extractor_*`) — extending the seed extractor's managed prompt would stay silently disabled in production (Langfuse copies are never overwritten; the PlanSpecExtractor "json" incident). Regex-gated so most turns pay no LLM call. Abstains via `mentioned: false`. |
| `src/services/chat_service.py` | Merges the pantry delta from the RAW message (refinement context text is never mistaken for the fridge), stashes `profile["_pantry"]` (survives clarification), and after generation attaches badges + `facts["pantry"] {used, unused, note}` on both the graded and structured paths. |
| `src/services/planning_pipeline.py` | `generate()`: pantry fan-out merged coverage-first + grader "prefer using …" hint. `plan_structured()`: pantry-matching ids ride the `favorite_recipe_ids` float (the structured path's only soft rank signal); structured plates now carry ingredients/directions text from the envelope (was `""` — which also blinded the matcher). |
| `src/services/weekly_plan_service.py` + `weekly_planner/` | Weekly reads the SAME PlanningState (whichever horizon hears about the zucchini, both honour it). `RecipeActionSpace(pantry=…)` folds per-item matches into every day's pool (diet normalised at the call site, incl. query-level tags); `build_preference_scorer(pantry=…)` adds +3 per matched item capped at 2 — above a favourite (+5 < 6), below variety-flattening. Entries annotated after explainability (chips appended, not overwritten). |
| `src/services/edit_service.py` | New verifiable directive `uses_ingredient` ("something with zucchini"): candidates via single-item include, verified against candidate ingredient text — fixes the old behaviour where the phrase was full-text-searched as a dish name. Also: swap candidate fetches now exclude `PlanningState.excluded_recipe_ids` (a rejected recipe could come back through a swap). |
| `src/services/orchestrator_service.py` | `preference_update` turns capture pure inventory statements ("I have leftover rice") into planning state and answer deterministically with an offer to plan; falls back to smalltalk on any failure. |
| UI badges | Per-course/entry `match_reasons` kind `"pantry"`: "uses your zucchini — reducing food waste" + "cooked from your leftovers"; plan-level ledger row "using N of M on-hand ingredient(s)" (status `relaxed` when items went unused). Unused items get a sentence in the reply, never silence. |
| Docs/tests | `CHAT_ENDPOINT_PIPELINE.md` updated (daily §2/§4, weekly §5). `tests/test_pantry.py` (29 tests, LLM-free). 397 passing. |

Known and deliberate: presence-matching only — no quantity awareness, and
"uses your zucchini" never claims "uses it all up". Weekly named-dish swap
anchors still aren't persisted to PlanningState (daily's are) — the
entry-level pin protects the current canvas only; keyed weekly anchors are
future work. The local demo harness (`wisefood_demo_client.py`, untracked)
gained an ingredient-verified slot filler so `include_ingredients` works
against the demo gateway too.

---

# Review fixes — pick lifecycle, card addressing, turn concerns

> **Date:** 2026-07-23
> **Branch:** main
> Ships with wisefood-api + wisefood-ui (`plan_type` passthrough) — rebuild
> the trio. Wire-compatible: `plan_type` is optional everywhere and falls
> back to the previous active-canvas behaviour.

Fixes from a multi-agent review of the manual-mode/weekly-card work.

| Area | Fix |
|---|---|
| Pick lifecycle | Picks are reconciled AFTER generation against the plan that came back (`_settle_manual_picks`), never cleared before it. A failed generation no longer strands the still-active plan without anchors; picks the pipeline refused (dead id, allergy conflict) aren't stored, so they're never retried or re-apologized for; a pick displaced by an explicit seed stops resurrecting next turn. |
| Diners | `PUT /diners` rebuilt the profile from member records, wiping `manual_picks`, `plan_parameters` and session-collected `history`. `ProfileService.carry_session_state` carries them over. |
| Clarification | Compose and weekly generation clear a pending clarification trap, which otherwise ate the member's next message and buried the plan they'd just made. |
| Card addressing | The slider card carries `plan_type` and the client echoes it back, so values refine the plan the card was rendered with rather than whichever canvas is newest at click time. |
| Honesty | A re-injected pin now says "I kept X … say the word if you'd like those changed too" instead of claiming the member asked for it this turn (`SeedResolution.kept`). |
| Errors | Only `SessionAccessError` maps to 404; planner `ValueError`s are 500s, not "session not found" (also fixes `/chat`). |
| Contracts | `ComposePick` declares its constraints (`Literal` meal type from the shared `MEAL_SLOTS`, day 1-7) → 422 instead of silent drops; day-less weekly picks are no longer deduped away from spread placement. |
| Analyst | Weekly day labels were 0-indexed against 1-based entries — every day was named one late in the plan-analyst context. |
| Structure | Shared entry-point guards replace three copies of the ownership/limit block; compose runs the every-turn memory-nudge hook; profile writes are once per operation. |

Known and deliberate: a weekly slider apply regenerates the week, so
verified slot edits are lost — same as any weekly text refinement (see
IDEAS.md). Tests: 204 passing (`test_session_state_integrity.py` new).

---

# Manual mode + canvas interactions

> **Date:** 2026-07-23
> **Branch:** main
> Ships with wisefood-api (compose proxy) and wisefood-ui (compose card,
> weekly canvas redesign, slot menus, adapt popup) — rebuild the trio.

- `POST /sessions/{id}/compose` — manual mode: hand-picked recipes from a
  blank canvas become slot-addressed seed anchors (id-resolved,
  allergy/diet re-checked, pinned exactly), then generation fills the
  remaining slots with no classifier and no clarification. `plan_type`
  daily|weekly; weekly picks carry `day` 1-7 into the planner's pinned
  slots. Chat text may ride along as the completion query — the UI routes
  the next message through compose while picks are staged ("fill out the
  rest, keep it light"). Tests: `tests/test_compose.py`.
- UI (wisefood-ui): weekly canvas is now collapsible day rows (expanded by
  default) with the M7 measured ledger, per-meal reason chips, and a
  "Week at a glance" panel; meal cards/cells have a ⋮ menu — Replace
  prefills the verified-edit phrasing, Adapt opens RecipeWrangler's
  adaptation assistant in an in-page popup that saves the same
  adapted-version record. Day labels are localized weekday names from the
  1-based index (fixes the off-by-one "Day N" grid labels).
- Manual picks PERSIST: stored per plan type on the session profile,
  re-injected (re-resolved, safety-rechecked) into every refinement with
  no explicit seeds, cleared by a fresh plan request, and unpinned per
  slot when a verified edit swaps that slot. The slider card also attaches
  to fresh weekly plans, and applying values with a weekly canvas active
  refines that weekly plan. Compose pickers show a favorites shortlist;
  the weekly canvas gained the plan vote + personalization line.

---

# Weekly plan explainability — measured ledger, metrics, per-meal reasons

> **Date:** 2026-07-22
> **Branch:** main
> Weekly-plan scope only; daily-plan flow untouched. All response fields
> are additive (empty defaults for pre-change plans) — gateway/UI keep
> working untouched. One heads-up for the UI: weekly ledger rows can now
> carry `status: "relaxed" | "violated"` (daily rows only ever say
> "satisfied") plus an optional `detail` string — treat unknown statuses
> as informational. Entirely LLM-free: IDEAS.md's "optional LLM grades
> for parity" phase was deliberately skipped (the deterministic checklist
> covers it).

Implements the "Weekly plan explainability" plan from IDEAS.md. Because
the weekly planner is deterministic since the M6 constraints rework, the
ledger REPORTS measured numbers (meat meals used, kcal planned vs budget)
instead of declaring "satisfied", and constraint relaxations are recorded
AT DECISION TIME by the planner rather than reconstructed afterwards.

| File | What / Detail |
|---|---|
| `weekly_planner/explainability.py` | **New module**, pure functions (no LLM/IO). `build_weekly_explainability` is the entry point: attaches per-entry `recipe.match_reasons` chips in place (reusing `transparency.match_reasons`; `pinned` flag → "requested by you", `adapted` flag → the `ADAPTED_REASON` chip — the weekly overlay previously only set the flag), and returns `constraints_applied` (profile rows from `transparency.constraints_ledger` + measured weekly rows: meat limit with `satisfied`/`relaxed`/`violated` status and slot-level detail, soft calorie-budget row with % used and coverage caveat), `personalization_summary`, `metrics` (variety: distinct recipes / unique ingredients / category distribution; deterministic weekly guideline frequency checklist: fish 1–2×, red meat ≤3, mostly plant-based; nutrition trackers: weekly totals + daily average vs target with "based on N of 21 meals" honesty; per-day breakdown with headline, kcal, and reason highlights; raw selection events), and `reasoning` — a composed whole-week justification. Targets re-derived from the profile via `WeeklyNutritionalTracker`, so it also works for patched plans with no env. Nutrition totals computed from the FINAL entries (post-enrichment, post-adapted-overlay), not the selection-time tracker. |
| `weekly_planner/reward_logic.py` | `apply_hard_constraints` optionally records selection events (`meat_pool_pruned` with dropped count, `meat_limit_relaxed`) into a caller-provided list with the slot attached — at decision time, so explanations reflect actual causes. Signature is backward-compatible (new optional `events`/`slot` params). |
| `weekly_planner/environment.py`, `weekly_planner/planner.py` | `env.selection_events` accumulator (reset per cycle); the planner passes it + the current slot into `apply_hard_constraints`. |
| `weekly_plan_service.py` | Calls `build_weekly_explainability` after enrichment/overlay/day-summaries; passes it to persistence; ResponseWriter facts gain `constraints_honored` (same key the daily flow uses) and `week_summary` so the reply can say "kept within your 3-meat-meal limit, 90% of your calorie budget". |
| `edit_service.py` | Weekly slot patches recompute explainability so it never goes stale (no selection events there; statuses come from final counts alone, and feedback rows stay out because a patch doesn't consult feedback exclusions — claiming them would be unverified). |
| `models/session.py`, `session_service.py` | Additive `WeeklyMealPlan.constraints_applied` / `.personalization_summary` / `.metrics` / `.reasoning` (same JSON-blob pattern as `day_summaries`: no DB migration, empty defaults on deserialize for pre-change plans). |
| `routers/foodchat_router.py` | The four fields exposed additively on `WeeklyMealPlanResponse`; per-meal `match_reasons` ride inside each entry's `recipe` dict (no entry-model change). |

Tests: `tests/test_weekly_explainability.py` — chip attachment
(pinned/favorite/like/adapted, adapted display keys), variety/category
math, guideline checklist, nutrition coverage honesty, measured ledger
statuses (satisfied/relaxed/violated, calorie % detail), decision-time
event recording, per-day breakdown, full-payload build, pescatarian meat
counting, and service-level wiring (populated → persisted → exposed;
pre-change plans deserialize empty). All LLM-free; suite: 172 passed.

---

# Weekly planner: constraints steer selection + per-day summaries

> **Date:** 2026-07-20
> **Branch:** main
> Weekly-plan scope only; daily-plan flow untouched. `WeeklyMealPlanResponse`
> gains an additive `day_summaries` field — flat `entries` unchanged, so the
> gateway/UI keep working without a coordinated change (rendering the
> headlines is opt-in).

Two changes from IDEAS.md, both LLM-free.

## Constraints actually enforced (was: computed and thrown away)

Pre-change, the weekly planner computed constraint penalties strictly AFTER
each pick and only logged them; nutrition never existed during generation,
so the calorie constraint compared against zeros; the meat limit was
hardcoded to 3 for every profile; and one Groq call fired per committed
slot (21 per plan) to grade a recipe already locked in.

| File | What / Detail |
|---|---|
| `weekly_planner/action_adapter.py` | Each day's candidate pool is enriched with one batch `fetch_details` call at fetch time — candidates carry `nutrition`/`tags`/`dish_types` during selection (best-effort: a failed call degrades constraints to neutral, never blocks the plan). 7 HTTP calls per plan, replacing 21 LLM calls. |
| `weekly_planner/reward_logic.py` | Per-step LLM grading REMOVED (it never affected the output). New pre-selection functions: `apply_hard_constraints` (drops meat candidates once the weekly limit is spent; relaxes with a warning rather than failing if the pool would empty) and `constraint_score` (soft penalty for kcal above the fair per-slot share of the remaining weekly budget). `calculate_step_reward` kept, now the deterministic negative penalty — the `reward` field on entries/API stays populated. |
| `weekly_planner/planner.py` | Selection = preference score + constraint score over the hard-filtered pool (argmax, random tiebreak). Without scorer and nutrition, behavior stays uniform random as before. |
| `weekly_planner/state_tracking.py` | Meat limit is diet-aware instead of hardcoded 3: vegetarian/vegan → 0, pescatarian profiles stop counting fish, and a "meat limit N" / "N meat meals" preference string overrides. Tracker accepts enrichment-style nutrition keys (`kcal`/`protein_g`/…) alongside the generic ones, and vegetarian/vegan RW tags override keyword meat detection. Meat/fish keywords moved to the shared taxonomy in `day_summary.py` (was a second drifting copy). |
| `weekly_planner/environment.py` | Passes candidate `nutrition` + `tags` through to the tracker. |

## Per-day summaries (presentation)

| File | What / Detail |
|---|---|
| `weekly_planner/day_summary.py` | **New module.** Shared meat/poultry/fish taxonomy (fish reuses `ALLERGEN_SYNONYMS`, word-boundary matching — "meatless" no longer counts as meat), `classify_meal` (RW diet tags first, ingredient keywords as backstop), `summarize_day` ("dinner with red meat", "light vegetarian day", "fish day", "fish and red meat", fallback "varied meals"), `build_day_summaries` → `{day: headline}`. Composition templates are a deliberate starting point — iterate against real weeks (see IDEAS.md). |
| `weekly_plan_service.py` | Post-plan enrichment now also copies `tags`/`dish_types` onto entries (previously discarded); `build_day_summaries` runs after enrichment + adapted-recipe overlay; summaries persist with the plan and ride into the ResponseWriter facts ("Monday: fish and red meat"). Also fixed a day-name off-by-one in the refinement context (day 1 labeled Tuesday, day 7 "Day 8"). |
| `models/session.py`, `session_service.py` | Additive `WeeklyMealPlan.day_summaries` (default `{}`); JSON-blob persistence, no DB migration; deserialize restores int day keys and yields `{}` for pre-change plans. |
| `routers/foodchat_router.py` | `WeeklyMealPlanResponse.day_summaries: Dict[int, str]` (additive, `{}` default). |
| `edit_service.py` | Weekly slot patches recompute `day_summaries` and now copy the replacement's enrichment (nutrition/image/tags) onto the patched entry. |

Tests: `tests/test_weekly_constraints.py` (meat limit enforced against a
meat-preferring scorer, unsatisfiable-limit relaxation, diet-derived
limits, pescatarian fish exemption, calorie budget kept, deterministic
reward) and `tests/test_day_summary.py` (classifier/summarizer units +
service-level wiring: enrichment tags reach entries, summaries survive a
DB round-trip, pre-change plans deserialize with `{}`). Still LLM-free.

---

# dietary_goal memory kind — worries/objectives in chat steer planning

> **Date:** 2026-07-13
> **Branch:** main
> Companion change in the foodscholar repo: worry→goal mapping in its
> qa-memory-extractor prompt fallback. NOTE: the deployed FoodScholar reads
> that prompt from Langfuse (existing prompts are never overwritten on
> startup) — push the updated text as a new version of
> `foodscholar/qa-memory-extractor` in the Langfuse UI or the change stays
> dormant.

FoodScholar already runs a full consent loop for goals expressed in Q&A
(its own extractor + chips + `POST /qa/memory` → `properties.dietary_goals`,
which the planner reads since "Planner: apply dietary_goals"). This change
gives FoodChat the same ear: the PreferenceExtractor now detects
`dietary_goal` candidates ("my cholesterol is high", "I want to build
muscle") with canonical planner slugs, the nudge policy validates the slug
(off-list values are dropped) and dedupes against existing goals, and an
accepted nudge writes `properties.dietary_goals` — the SAME field FoodScholar
writes, so both apps converge on one goal store. The live session profile is
synced exactly as a fresh profile fetch would map it (slug + soft preference
string + hard diet tag where applicable), so the very next plan honors it.
Tests: `tests/test_member_memory_bridge.py`.

---

# Interactive plan-parameter card — sliders instead of questions

> **Date:** 2026-07-13
> **Branch:** main
> Ships together with a wisefood-api proxy route and the wisefood-ui card
> component — rebuild foodchat + gateway + UI together.

The old textual clarification questions about cooking time / difficulty /
goal (tuned out entirely during demo hardening — "DEFAULT TO NOT ASKING")
return as an OPTIONAL slider card attached to every fresh daily plan turn:

- `services/plan_parameters.py` — static card definition (cooking_time
  10–90 min scale; difficulty and goal as discrete choices), value
  sanitization (clamp/snap/whitelist), canonical refinement text, and the
  known-facts history line. Fully deterministic, no LLM.
- `ChatTurn.plan_parameters` / `ChatTurnResponse.plan_parameters` — card
  payload on fresh daily plans (not on text refinements; clarification
  completions included). Not persisted in conversation history — live
  responses only, like memory_suggestions.
- `POST /sessions/{id}/plan-parameters` — applies chosen values as a
  deterministic refinement: no intent classification, no clarification
  round (`process_plan_request(skip_clarification=True)`). Values merge
  into `user_profile["plan_parameters"]` (card shows current settings) and
  append to the profile history (reconciler treats them as known facts).
  Ownership 404s like /chat; unusable values → 400.
- Reconciler prompt now bans asking about cooking time / difficulty / goal
  outright — the card owns those topics; textual clarification remains for
  dietary conflicts and food-direction-on-bare-query only.
- Gateway: `POST /api/v1/foodchat/sessions/{id}/plan-parameters`
  (auth + member access check, extra-long timeout — it generates).
- UI: `FoodchatPlanParameterCard` renders inside the assistant bubble
  (draggable knobs, touched-only apply, dismissible, only the newest card
  stays interactive); store grafts the card client-side like attribution.

Tests: `tests/test_plan_parameters.py` (sanitize/card/describe + apply-flow
wiring with a recording fake — still LLM-free).

---

# Demo hardening — live-testing fixes

> **Date:** 2026-07-08
> **Branch:** main
> Fixes driven by live demo testing on demo.wisefood-project.eu. Rebuild the
> image to deploy.

## Member-scoped current plans (dashboard widget)

`GET /members/{member_id}/current-plans` returns the member's most recently
updated plan canvases across ALL their sessions (daily and/or weekly plus the
session's `cooking_for` diners). Replaces the UI dashboard's dependency on
the legacy per-date member meal-plan store, which nothing populates anymore —
FoodChat plans are versioned canvases, not calendar entries. Backed by
`SessionService.get_member_current_plans` /
`Session.active_canvas_updated_at` (recency = the active canvas's current
plan timestamp).

## Allergen defense-in-depth (SAFETY)

Live incident: RecipeWrangler served "Almond crumbed chicken" to a tree-nut-
allergic member — the recipe's graph node has NO allergen edges and is tagged
`nut_free`, so RW's hard filters passed it (recipe 9319107827; data-quality
issue reported to INFILI). FoodChat no longer trusts upstream tags with
safety data:

- `candidates_client.allergen_conflict()` — synonym-expanded ("tree nuts" →
  almond/walnut/cashew/…; shellfish → shrimp/prawn/crab/…; dairy, gluten,
  eggs, soy, sesame, fish), word-boundary ingredient/title scan.
- `fetch_candidates()` post-filters every parsed candidate against the
  member's allergies and logs each drop (`Allergen backstop dropped …`).
- `SeedService._allergy_conflict` uses the same expansion, so "pastitsio with
  almonds" can't be pinned for a tree-nut-allergic diner either. Previously a
  plain substring check ("tree nuts" never matched "almond").

## plan_question intent — "does my meal plan adhere to that?"

Live incident: asking whether the plan met the protein guidance just
discussed was classified `refine_plan` and silently regenerated the plan.
New eighth intent `plan_question` (question ABOUT the plan ≠ request to
change it) answered by the new `PlanAnalyst` agent: grounded in the active
canvas serialized WITH per-meal nutrition enrichment plus recent conversation
(so "that" resolves), read-only by design, honest about missing nutrition
data. No active canvas → falls through to FoodScholar as a plain nutrition
question.

## Memory nudges: same-kind dedupe + contradiction resolution

Live incident: "I think I don't like chicken" produced no nudge because
"chicken" sat in food_LIKES and the suggestion filter used one flat "known"
set across kinds. Now each kind dedupes only against its own field, a
like↔dislike contradiction gets an explicit callout statement ("…currently in
your likes, but it sounds like you've gone off it — update your profile?"),
and accepting removes the value from the opposite list (both in the durable
profile and the live session).

## Seed resolution tolerates trailing typos

"bolognesse" found nothing (RW autocomplete is a non-fuzzy ES prefix match).
`SeedService._autocomplete_tolerant` retries with up to 3 trailing characters
cut, recovering the common trailing-typo case ("bolognes" prefix-matches
"bolognese"). Proper fuzziness belongs in the RW endpoint.

## Tests

91 passing (was 80): allergen synonym matching + word boundaries, backstop
drop in `fetch_candidates`, untagged-almond seed conflict, typo-tolerant
resolution, plan_question routing (answered not refined; FoodScholar
fallback without a canvas).

---

# M5 — Platform Hardening & Demo Readiness

> **Date:** 2026-07-07
> **Branch:** main
> Deployment: foodchat moves to the platform PostgreSQL (dedicated
> `foodchat` database — created idempotently by the core-components init
> script; re-run init-db or create it manually on existing clusters) and
> gains Langfuse tracing env. Rebuild the image (new deps: psycopg2-binary,
> langfuse).

## Postgres-ready persistence

- **Timezone-aware everywhere**: all column defaults and domain-model
  timestamps are aware UTC (`DateTime(timezone=True)`); pre-M5 naive rows
  are coerced on load (`_aware`) so mixed sorts can't raise.
- **Replica-safe mutations**: every SessionService mutator is load-through —
  a write landing on a replica that never saw the session loads it from the
  DB instead of raising (8 call sites).
- **Canvas clears persist**: `db_update_canvases` now NULLs cleared
  canvases (pre-M5 a cleared canvas resurrected after restart).
- Bounded, verified connection pool for PostgreSQL
  (`pool_pre_ping`, `DB_POOL_SIZE`/`DB_MAX_OVERFLOW`/`DB_POOL_RECYCLE`).
- SQLite remains the zero-config dev default; tests run on it unchanged.

## Observability

- **Langfuse tracing on every Groq call** (`backend/observability.py`):
  env-gated (`LANGFUSE_PUBLIC_KEY`/`SECRET_KEY`/`HOST`), attaches a LangChain
  callback to the pooled clients — orchestrator, graders, extractors, and the
  response writer all appear as traces in the platform Langfuse (same
  instance FoodScholar reports to). No keys → silent no-op, never affects chat.

## Demo readiness

- `scripts/seed_demo.py` — idempotent gateway-driven seeding of the demo
  household (Dimitris omnivore/high-protein, Anna vegetarian, Tom child with
  a peanut allergy) + favorites for the Greek anchor dishes; doubles as the
  preflight check that pastitsio/fakes/moussaka resolve in RecipeWrangler
  (non-zero exit when a prerequisite is missing).
- `DEMO_SCRIPT.md` — the beat-by-beat RecSys walkthrough with feature
  mapping, failure-mode recovery lines, and research talking points.
- FoodScholar `Dockerfile` EXPOSE fixed to the deployed port (8001).

## Tests

+6 (cold-replica mutations for messages/plans/clarification, canvas-clear
persistence, naive→aware coercion, mixed-timestamp sorting) — 78 total.

---

# M4 — Rich Plans, Transparency, Verified Slot Editing, Natural Voice

> **Date:** 2026-07-07
> **Branch:** main
> No DB migration (plan payloads gain optional fields; old payloads
> deserialize with nulls). RecipeWrangler gains POST /recipes/details.

## Rich, explained plans

- **Enrichment**: every plan course now carries per-serving `nutrition`
  (kcal/protein/carbs/fat, Nutri-Score label) and `image_url`, fetched in ONE
  RecipeWrangler batch call (`POST /api/v1/recipes/details`, cached
  server-side). Weekly entries enriched the same way (21 recipes, one call).
- **Transparency (structured, not prose)** — `services/transparency.py`:
  per-course `match_reasons` chips (pinned / favorite / memory / profile /
  feedback / diner), a plan-level `constraints_applied` ledger (hard/soft,
  with diner attribution), and `personalization_summary` counts linking to
  the memory panel. The four quality scores were already returned — the UI
  now renders them as a plan-quality card.
- **Weekly selection is preference-aware**: `build_preference_scorer`
  replaces uniform-random candidate choice — favorites dominate (+5), liked
  ingredients boost (+1 each), title-token overlap with already-planned
  meals penalizes (−2/token) for variety. Zero extra LLM cost; the per-step
  LLM reward is still recorded per entry.

## Verified slot editing ("swap Tuesday's dinner for something lighter")

- New `edit_plan_slot` intent (one targeted meal ≠ whole-plan `refine_plan`)
  + `EditCommandExtractor` (slot + directive; one conversational follow-up
  when the slot is ambiguous, persisted as clarification kind="edit_slot").
- **Directive predicates** (`services/edit_service.py`): measurable
  directives are verified against RecipeWrangler nutrition BEFORE selection
  — lighter ⇒ kcal ≤ 0.85×old, more protein ⇒ strictly greater, quicker ⇒
  shorter duration, diet words ⇒ tag present. Missing measurements FAIL
  CLOSED (an unverifiable swap never claims compliance). Unmeasurable
  directives ("more festive") pick best-effort and say so.
- **Honest failure**: when nothing passes, the reply says so and offers the
  nearest miss with numbers ("closest is X at 610 kcal — want it?").
- **Patch semantics**: the new version keeps every untouched slot (daily:
  courses carried over with their enrichment; weekly: 20 entries copied,
  ONE replaced — no more 21-meal regeneration on a single swap). Response
  carries `changed_slots` with the before/after kcal proof; the UI renders
  the diff chip ("700 → 420 kcal, verified").

## Natural voice + never-ask-twice

- **ResponseWriter agent**: every plan/edit reply is composed from
  structured facts (action, meals, seed notes, diners, constraints honored,
  swap proof) — grounded persona prose with a canned fallback on LLM
  failure. The era of "Here's your meal plan for today!" ×100 is over.
- **Never-ask-twice clarifier**: the reconciler now receives the profile's
  known facts and is instructed not to mark them missing; clarification
  questions are capped at 2 per request.

## Tests

+12 (predicates incl. fail-closed, daily/weekly patch edits, ambiguity
round-trip, honest failure with nearest miss, transparency attachment,
weekly scorer) — 72 total.

---

# M3 — Consented Memory, Feedback Loop, Household Diners (+ platform consent bar)

> **Date:** 2026-07-07
> **Branch:** main
> No FoodChat DB migration. Gateway gains the `user_consent` table (init-db
> re-run needed on existing deployments, same as member_favorite) and proxies
> for the new /memory and /diners endpoints.

## Consented memory ("remember this?")

Principle: session adaptation is automatic; **durable memory requires an
explicit yes**, and everything remembered is visible and deletable.

- `PreferenceExtractor` agent detects durable preference candidates per user
  turn (likes/dislikes/cuisines/allergy hints/standing dishes/constraints —
  never one-off requests). `services/memory_service.py` applies the nudge
  policy: only explicit high-confidence candidates are suggested (allergy
  hints at any confidence — and they are the ONLY path that ever touches the
  allergies field); known values and previously-declined values
  (`properties.memory_optouts`) are never re-suggested; max 2 nudges/turn.
- `ChatTurnResponse.memory_suggestions[]` + `POST /sessions/{id}/memory`
  {decision, suggestion}. Accept → durable profile write via the SDK with
  provenance (`properties.memory_log[{kind, value, source, session_id,
  recorded_at}]`) AND immediate effect in the live session. Decline →
  opt-out recorded.
- UI: nudge chips under assistant messages ([Remember]/[No thanks]) and a
  **memory panel** on my-profile ("What WiseFood remembers") with per-entry
  forget. FoodScholar reads the same profile → accepted memories personalize
  its answers too.
- **Standing seeds** (deferred from M2): "always include pastitsio" →
  consent nudge → `properties.standing_seeds`; fresh weekly plans auto-pin
  them when no explicit dishes compete.

## Feedback finally drives recommendations

- `services/feedback_service.py` joins feedback → messages.plan_id →
  meal_plans across the member's sessions: recipes with more downvotes than
  upvotes are excluded from candidate fetches (daily + weekly), and the
  rating history (with comments) replaces the hardcoded `""` in the daily
  grader prompts.

## Household diners ("who are we cooking for?")

- `CreateSessionRequest.cooking_for` + `PUT /sessions/{id}/diners` rebuild
  the session profile via `ProfileService.merge_profiles`: **hard = union**
  (allergies, diets, dislikes-as-exclusions) — one vegetarian diner makes the
  plan vegetarian, any diner's allergy excludes everywhere; **soft =
  weighted** (owner's likes lead, other diners' follow); macro targets and
  favorites stay the owner's. UI: avatar-chip diner picker + "Cooking for:"
  banner; hidden for single-member households.

## Platform consent bar (gateway + UI)

- `wisefood.user_consent` append-only ledger (user_id = Keycloak sub,
  consent_type `service_data_processing`, version, granted_at, ip_address
  from X-Forwarded-For) + `GET/POST /api/v1/users/me/consent`.
- UI: small fixed bottom bar after login (any user incl. guests) — "cookies
  + personal data processed solely to provide the service", Privacy Policy
  link, one Accept button; hidden once the current consent version is
  granted; sessionStorage cache prevents flicker.

## Tests

+13 (nudge policy incl. allergy exception and opt-outs, decisions with
session/DB persistence, feedback aggregation up/down, diner merge) — 60 total.

---

# M2 — Favorites & Seeded Planning

> **Date:** 2026-07-07
> **Branch:** main
> No FoodChat DB migration. Gateway gains the `member_favorite` table
> (DDL is `IF NOT EXISTS`; existing deployments must re-run init-db or apply
> it manually — the gateway has no migration framework). RecipeWrangler and
> the UI updated in the same release.

## What changed

Plans no longer start from a blank slate — favorites and user-named dishes
become starting points:

- **Server-side favorites** (gateway): `member_favorite` table +
  `GET/PUT/DELETE /api/v1/members/{id}/favorites[/{recipe_id}]` (idempotent,
  owner/agent/admin authz). The UI recipe store is now API-backed with a
  one-time localStorage migration.
- **Favorites boost in candidates**: FoodChat fetches the member's favorites
  at session creation (`profile["favorite_recipe_ids"]`, best-effort) and
  passes them to RecipeWrangler's `foodchat_candidates`, which ranks
  favorites to the top of their slot (weight 10 vs 1 per include-ingredient
  hit). Hard filters always win — an allergy-violating favorite never appears.
- **Seeded / anchored planning**: new `SeedExtractor` agent pulls named
  dishes ("pastitsio and fakes in my weekly meals") from plan requests;
  `services/seed_service.py` resolves them via RecipeWrangler
  (autocomplete → detail), enforces the allergy hard constraint (a
  conflicting seed is skipped with an explanation, never pinned), and places
  them: explicit hints ("Sunday dinner") honored, otherwise dish-type tags
  decide the slot and weekly anchors spread across the week.
- **Pinned slots**: daily pipeline — a pinned slot's anchor is the sole
  candidate, so every graded combination contains it; weekly planner — pinned
  (day, meal) slots bypass candidate selection, anchors are excluded from
  the pools so they never repeat, and entries carry `"pinned": true`.
  Daily pins survive mid-clarification restarts (they ride inside the
  persisted profile snapshot under `_pinned_slots`).
- **Proactive favorites offer**: on the first plan request of a session — if
  the member has favorites, named no dishes, and wasn't asked before — the
  assistant offers once ("I noticed you've favorited Pastitsio… work them
  in? yes/no") via a `favorites_offer` clarification state. Yes → favorites
  pinned as anchors; anything else → the original request proceeds
  unchanged. Dedupe is the persisted `favorites_offer` intent tag on the
  offer message. Affirmative detection is a deliberate keyword heuristic
  for M2 (misreads cost only the boost).
- **Tests:** +13 (resolution, allergy conflicts, weekly spread + hint
  placement, pipeline/planner pinning, offer lifecycle) — 47 total.

---

# M1 — FoodScholar Bridge

> **Date:** 2026-07-07
> **Branch:** main
> No DB migration. New env var: `FOODSCHOLAR_API_URL` (+ optional
> `FOODSCHOLAR_TIMEOUT`, `FOODSCHOLAR_TOP_K`). Gateway and UI updated in the
> same release (attribution passthrough / rendering).

## What changed

FoodChat no longer refuses nutrition-science questions — it answers them
**via FoodScholar** and shows its sources:

- **New intent `nutrition_question`** (orchestrator prompt + schema): factual
  questions about nutrition, diets, ingredients, or health effects of food.
  A request FOR a plan is never a nutrition_question; a question ABOUT a
  diet is.
- **`services/foodscholar_service.py`** — bridge to FoodScholar
  `POST /api/v1/qa/ask` (mode=simple, member_id passed through so FoodScholar
  personalizes with the same profile). Handles FoodScholar's clarification
  flow: the question is surfaced conversationally (options flattened into the
  text) and the pending `qa_thread_id` is persisted in the session's
  clarification state as `{"kind": "foodscholar", ...}` — restart-safe, same
  mechanism as the plan flow. FoodScholar unreachable → graceful in-chat
  apology, never a 500.
- **`models/attribution.py`** + `ChatTurnResponse.attribution` —
  `{source, confidence, citations[{title, source_type, url, label}],
  learn_more_url}`. `learn_more_url` is a UI-relative deep link
  (`/foodscholar?q=<question>`); the frontend prefills and auto-asks.
- **SimpleChatBot prompt rewritten** — never claims it "can't answer";
  warmly steers toward planning; nutrition questions never reach it anymore.
- **Gateway (wisefood-api):** `FoodChatAttribution`/`FoodChatCitation`
  mirrored on the proxied chat-turn model; fixed a latent bug where
  FoodScholar session creation with a `member_id` called nonexistent methods
  on `HOUSEHOLD` (now `HOUSEHOLD_MEMBER.get/get_member_profile`) and 500'd.
- **UI (wisefood-ui):** "Answered with FoodScholar" badge, citation chips,
  "Learn more in FoodScholar →" link on attributed messages; `/foodscholar`
  accepts `?q=` to prefill + auto-ask.
- **Deployment:** `FOODSCHOLAR_API_URL=http://foodscholar:8001` added to the
  foodchat container env (tk-validated).
- **Tests:** +6 (answer/attribution mapping, clarification round-trip incl.
  restart, graceful degradation, orchestrator routing + classifier bypass
  while clarifying).

---

# M0 — Clean Foundation

> **Date:** 2026-07-07
> **Branch:** main
> Prepared for handoff. DB migration is idempotent (adds `sessions.clarification_state`).
> Removed endpoints were deleted from the wisefood-api gateway in the same change.

## Why

Structural debt removal before the RecSys '26 feature milestones: the service
carried a dead local-RAG stack that gated startup, an unserializable
clarification flow, a cross-user memory leak, and double intent
classification. Full rationale lives in the internal roadmap (milestone M0).

## Removed

- **Legacy RAG stack** — `src/foodchat.py` (Retriever/Chroma/BM25/MMR chains),
  `src/utils.py` (embedding backends), `src/csv_processor.py`,
  `src/pdf_processor.py`, `src/foodchat_init.py`, `src/VECTORSTORE/`,
  `Modelfile`, `migrate.py`. Recipe candidates come exclusively from
  RecipeWrangler. The service now boots with **no data files**; the
  503-at-startup failure mode is gone.
- **`KG_neo4j/`** — direct-Neo4j fallback + importer. Survivor: the RW API
  client, rewritten as `src/services/candidates_client.py` (typed).
  `RECIPE_SOURCE`, `NEO4J_*`, `CHROMA_*` env vars no longer exist.
- **Offline eval harnesses** — `src/multiple_evaluation.py`, `src/ragas_eval.py`,
  `llm_eval_res.json`, `trace.json` (Ollama-era; recoverable from git history)
  and their agents (`FoodChatResponseEvaluator`, `QueryRewriter`, `FeedBackRewriter`).
- **`QueryClassifier`** — the orchestrator is now the single intent router;
  ChatService no longer re-classifies.
- **Legacy endpoints** — `POST/GET /sessions/{id}/messages`,
  `POST/GET /sessions/{id}/weekly` (superseded by `/chat` + `/conversation`;
  gateway proxies removed in lockstep).
- Dead shims: `db_update_active_context`, `Session.active_context`,
  `Session.messages`/`weekly_messages` aliases, `MealCourse.from_list`.
- Dependencies: chromadb, rank-bm25, langchain/-community/-ollama, neo4j,
  pandas, numpy, pdfplumber, unstructured, colorama dropped from requirements.

## Added / changed

- **`services/clarification.py`** — clarification is now an explicit,
  JSON-serializable state machine (`ClarificationState` persisted in the new
  `sessions.clarification_state` column). Mid-clarification sessions survive
  restarts and replicas. Same conversational behaviour as the old generator.
- **`services/planning_pipeline.py`** — typed replacement for the LangChain
  runnable chains: candidates → LLM-graded combinations → `ScoredPlan`s.
- **`models/recipe.py`** — `CandidateRecipe` / `ScoredPlan` domain models;
  tuple plumbing eliminated end to end (`MealPlan.from_courses`).
- **`SimpleChatBot` is stateless** — history passed per call from the session
  conversation. Pre-M0 it held ONE process-global memory shared by all users
  (cross-user context leak).
- **ChatService split** — `process_plan_request` / `process_smalltalk` /
  `continue_clarification`; orchestrator routes to them by intent.
- **Bug fix:** `QUERY_CHECKER_USER_INSTRUCTIONS` had unescaped `{...}` braces,
  so the query-specificity check raised `KeyError` on every call and was
  silently swallowed by a broad except — it never actually ran. Fixed and
  covered by tests; all remaining templates are format-validated in CI.
- **Tests** — new `tests/` suite (28 tests, LLM-free via fakes): session
  lifecycle/ownership, canvas versioning, clarification state machine +
  restart scenarios, candidates client contract, pipeline plumbing.
- Import scheme standardized (src-rooted, no `src.` prefixes; the old mixed
  scheme only worked via a `sys.path` hack in the deleted `foodchat.py`).
- Docs rewritten: `README.md`, `CHAT_ENDPOINT_PIPELINE.md`, `.env.example`.

## Deployment notes

- Env vars removed from `platform-deployment/lib/foodchat.libsonnet`:
  `CHROMA_*`, `NEO4J_*`, `CSV_HUMMUS_PATH`, `RECIPE_SOURCE`; added
  `DATABASE_URL`. Validated with `tk eval`.
- Rebuild the image (`wisefood/foodchat`) — it is substantially smaller
  (torch/chromadb/pandas gone).

---

# Canvas & Version History Refactor

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
