# FoodChat — Change Log (newest first)

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
