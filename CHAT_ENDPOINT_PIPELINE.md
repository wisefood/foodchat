# FoodChat `/chat` Endpoint Pipeline

How a user message flows through `POST /foodchat/sessions/{session_id}/chat`
(post-M0 architecture — single intent router, persisted clarification state).

## 1. End-to-end request flow

```text
Client (wisefood-api gateway)
  |
  | POST /foodchat/sessions/{session_id}/chat
  | body: { "member_id": "...", "content": "..." }
  v
[foodchat_router.unified_chat]
  |
  +--> _require_orchestrator_service()          -> 503 if startup failed
  |
  +--> orch_svc.process(session_id, member_id, content)
         |
         v
      [OrchestratorService.process]
         |
         +--> session_service.get_session(session_id, member_id)
         |      +--> missing / wrong owner -> ValueError -> HTTP 404
         |
         +--> session.is_at_message_limit ?
         |      +--> yes -> ChatTurn(at_message_limit=True)
         |
         +--> explicit score request ?  (a scoring word — "rate", "score", "how
         |      |                         does this look" — plus a meal listing in
         |      |                         at least two slots, prose included
         |      |                         ("eggs for breakfast, pasta for lunch");
         |      |                         no request to make or change a plan; no
         |      |                         FoodScholar mention)
         |      +--> yes -> clear any pending clarification,
         |                  PlanScorerService.process  (NO classification, section 5b)
         |
         +--> session.state == "clarifying" ?
         |      |
         |      +--> yes -> route by persisted clarification kind
         |      |          (NO intent classification — the user is answering
         |      |           our question)
         |      |          +--> kind == "foodscholar" -> FoodScholarService.continue_clarification
         |      |          +--> kind == "score_plan"  -> PlanScorerService.continue_clarification
         |      |                                        (a reply that answers nothing -> state
         |      |                                         cleared, routed as a fresh turn, and a
         |      |                                         score_plan there never asks again)
         |      |          +--> else (plan flow)      -> ChatService.continue_clarification
         |      |                                        (origin intent restored from state)
         |      |
         |      +--> no  -> OrchestratorAgent.classify(message, last 12 turns)
         |                  (the ONLY intent classification in the pipeline)
         |                  classify failed (outage, spent API budget)? it
         |                  returns {"intent": "chat", "failed": True} — and a
         |                  message that lists meals in two slots is scored
         |                  instead, rather than answered as small talk
         |
         +--> route by intent
                |
                +--> "daily_plan"          -> ChatService.process_plan_request(is_refinement=False)
                +--> "weekly_plan"         -> WeeklyPlanService.process_message(is_refinement=False)
                +--> "refine_plan"
                |      +--> no active canvas      -> fresh daily plan
                |      +--> active weekly canvas  -> WeeklyPlanService(is_refinement=True)
                |      +--> active daily canvas   -> ChatService(is_refinement=True)
                +--> "switch_plan_type"    -> ack message, then fresh plan of target type
                |                             (old canvas frozen, history retained)
                +--> "nutrition_question"  -> FoodScholarService.process_question
                |                             (see section 5a — cited answer + attribution)
                +--> "score_plan"          -> PlanScorerService.process (checked first;
                |                             section 5b — a plan the member wrote,
                |                             never written to a canvas)
                +--> "chat" / fallback     -> ChatService.process_smalltalk
```

## 2. Daily plan branch

```text
[ChatService.process_plan_request]
  |
  +--> add user message
  |
  +--> is_refinement and daily canvas exists ?
  |      +--> yes -> prepend current plan text to the effective message
  |
  +--> standing PlanningState merge (spec / anchors / favorites / exclusions)
  |      + pantry_service.extract_pantry_delta(RAW message)
  |        (regex-gated PantryExtractor: "I have zucchini and spinach" ->
  |         state.pantry; "used up the zucchini" removes it; persisted on
  |         the session, rides the profile snapshot as profile["_pantry"].
  |         The gate allows up to two words between subject and verb, so
  |         "I already have ..." reaches the extractor)
  |
  +--> ClarificationManager.start(effective_message, session.user_profile, origin_intent)
         |
         |  (reconciles query vs profile; merged profile rides on the outcome)
         |
         +--> outcome.question set ?
         |      |
         |      +--> yes -> session_service.set_clarification_state(state.to_dict())
         |      |          -> persisted to sessions.clarification_state (JSON)
         |      |          -> return question, needs_clarification=True
         |      |
         |      +--> no  -> _generate_and_store(final_query, outcome.profile)
```

## 3. Clarification loop (restart-safe)

```text
Any subsequent message while state == "clarifying":

[ChatService.continue_clarification]
  |
  +--> session.clarification (dict) missing ?
  |      +--> yes -> reset to "ready", treat message as a fresh plan request
  |                  (recovers sessions stranded by old bugs / manual edits)
  |
  +--> state = ClarificationState.from_dict(session.clarification)
  +--> ClarificationManager.step(state, user_answer)
         |
         +--> more questions -> persist updated state, ask next
         +--> done           -> clear state, _generate_and_store(final_query, profile)

Because the state is plain JSON on the session row, this loop continues
correctly after a process restart or on a different replica.
```

## 4. Generation and storage

```text
[_generate_and_store]
  |
  +--> PlanningPipeline.generate(final_query, profile)
  |      |
  |      +--> candidate pool from /api/v2/tools/plan_meals
  |      |      hard filters server-side at RecipeWrangler:
  |      |      allergens (FoodOn taxonomy), diet tags, exclude ingredients/ids
  |      |
  |      +--> profile["_pantry"] set ? (food waste)
  |      |      +--> per-item plan_meals fan-out (include_ingredients=[ONE
  |      |           item] each — the endpoint ANDs the list, so the whole
  |      |           pantry at once would empty every slot), merged into the
  |      |           pool coverage-first, pool size unchanged; grader query
  |      |           gains a "prefer using: ..." hint
  |      |
  |      +--> any slot empty ? -> [] -> apology response, no plan
  |      |
  |      +--> DocumentGrader.grade_daily_plans
  |             sample <= FOODCHAT_MAX_PLANS_TO_SCORE combos of B x L x D,
  |             one LLM grade each -> top 3 ScoredPlans
  |
  +--> metrics for the best plan:
  |      llm_score / llm_reasoning            (from grading)
  |      fvs_count / fvs_reasoning            (unique-ingredient count, no LLM)
  |      diversity_llm_score / _reasoning     (MealDiversityGrader)
  |      guideline_adherence_score / _reason  (GuidelineAdherenceGrader)
  |
  +--> store:
  |      is_refinement -> session_service.refine_meal_plan  (version+1, parent_id)
  |      else          -> session_service.add_meal_plan     (version 1, fresh canvas)
  |
  +--> pantry coverage (deterministic word-boundary matcher, never the LLM):
  |      per-course match_reasons chips (kind "pantry": "uses your zucchini —
  |      reducing food waste" / "cooked from your leftovers"), a ledger row
  |      ("using N of M on-hand ingredient(s)", status relaxed when items
  |      went unused), and facts["pantry"] {used, unused, note} for the
  |      ResponseWriter — unused items get a sentence, never silence
  |
  +--> add assistant message, return (text, needs_clarification=False, meal_plan)
```

## 5a. Nutrition question branch (FoodScholar bridge)

```text
[FoodScholarService.process_question]
  |
  +--> add user message
  +--> POST {FOODSCHOLAR_API_URL}/api/v1/qa/ask
  |      { question, mode: "simple", member_id, top_k }
  |      (FoodScholar personalizes with the same member profile)
  |
  +--> HTTP failure ?
  |      +--> yes -> in-chat apology, no attribution, never a 500
  |
  +--> needs_clarification ?
  |      |
  |      +--> yes -> render question (+ options flattened into text)
  |      |          -> persist {"kind": "foodscholar", qa_thread_id,
  |      |                      clarification_id, original_question}
  |      |             in sessions.clarification_state
  |      |          -> return needs_clarification=True
  |      |          (next user turn resumes the thread with the answer
  |      |           as free text — restart-safe)
  |      |
  |      +--> no  -> answer text (markdown) + Attribution:
  |                  { source: "foodscholar", confidence,
  |                    citations[{title, source_type, url, label}],
  |                    learn_more_url: "/foodscholar?q=<question>" }
```

## 5b. Plan scorer branch (score_plan)

A plan the member WROTE — pasted into the chat, or into the text box that
posts to `POST /foodchat/sessions/{session_id}/score-plan`. Scored, never
adopted: no canvas, no plan version, and refine/edit turns keep targeting the
member's own plan.

```text
POST /sessions/{id}/score-plan                 chat message classified (or
{ member_id, plan_text (<= 8000 chars),        explicitly detected) as
  plan_type: auto|daily|weekly, context? }     score_plan — section 1
        |                                              |
[foodchat_router.score_plan]                           |
  blank text -> 400                                    |
[OrchestratorService.score_plan]                       |
  ownership -> 404 | message cap -> limit turn         |
  pending clarification -> cleared (superseded)        |
  NO classification                                    |
        +----------------------+-----------------------+
                               v
[PlanScorerService.process]   (services/plan_scorer/service.py)
  |
  +--> add user message
  |
  +--> 1. parse_plan_text  (parsing.py)
  |      scan(): day headings ("Monday", "Day 2", "Tue:"), "slot:" prefixes,
  |      bullets under a slot heading, trailing "(ingredients)", "oats for
  |      breakfast" prose; lines around the listing are kept as notes (the
  |      member's own words about the plan). A structure-only reading that
  |      placed every line is used as is -> NO model call. Otherwise
  |      PlanTextParser (FAST_MODEL) with the scan as a hint; its titles,
  |      ingredient lists and unparsed lines are checked against the text.
  |
  +--> nothing read ?          -> clarification {kind: "score_plan", reason:
  |                                "no_meals", pasted_text, plan_type, context}
  +--> auto, one block, no day
  |    names, 2+ meals repeat ? -> clarification {..., reason: "shape", plan}
  |      (plan_type daily/weekly from the text box settles it without asking;
  |       "3 days" / "just one day" / a re-pasted listing continues; anything
  |       else is unresolved -> routed as a fresh turn, never asked twice)
  |
  +--> 2. DishGrounder.ground  (grounding.py)
  |      SeedService.find_dish once per distinct title: search, then tolerant
  |      autocomplete, NOT filtered by the member's allergens, diet or
  |      dislikes, and no allergy gate (the dish was already eaten; filtering
  |      would hide the allergen the score must report). When nothing scores
  |      as a match, the title's other spellings are searched too
  |      ("lasagne" for "lasagna" — parsing.SPELLING_VARIANTS, which also
  |      normalises both sides before comparison, accents folded).
  |      Dice title similarity, measured on the recipe actually fetched:
  |        >= 0.75 and same_dish  matched: recipe nutrition and image; its
  |                              ingredients and tags only when the member
  |                              wrote none (what they wrote is the dish).
  |                              same_dish: the name keeps every part the
  |                              member named (parsing.dish_heads — "roast
  |                              chicken WITH potatoes" needs chicken and
  |                              potato) and adds only DESCRIPTIVE_WORDS
  |                              ("Roasted", "Smoked"; never a food, diet or
  |                              cuisine word — "Vegan caesar salad" is cashews)
  |        >= 0.40  approximate  the member's words only; the recipe's
  |                              calories only when its name is a more
  |                              generic form of theirs keeping every part
  |                              ("Lasagna" for "vegetable lasagna", not
  |                              "Hummus" for "houmous and pitta")
  |        else     unresolved   the member's words only
  |      Recipe calories under MIN_MEAL_KCAL (120, main meals) or
  |      MIN_SNACK_KCAL (40) are set aside (rejected_recipe_kcal) and the dish
  |      is estimated instead.
  |      allergen_conflicts per dish {allergen, evidence: as_written |
  |      recipe | closest_recipe}; closest_recipe is a warning, never a
  |      verdict; plant milks and nut butters are not dairy (PLANT_DAIRY).
  |      One fetch_details batch fills what the search did not carry.
  |      Calories or ingredients still missing -> ONE
  |      agents.DishIngredientEstimator call (fast model) writes a typical
  |      single serving with quantities for every such dish (those missing
  |      calories first), keeping the member's own ingredients and amount,
  |      kept as typical_ingredients. A dish missing calories is profiled:
  |      each list goes to RecipeWrangler's profiler (POST
  |      /api/v1/recipes/profile) in parallel, PROFILE_TIMEOUT_SECONDS each,
  |      no new calls after the first timeout. Per-serving totals with
  |      coverage >= 0.6 and within 2x of the model's guess ->
  |      typical_ingredients; otherwise the model's own calorie guess
  |      (20-2,500 kcal) -> model_estimate, calories only. The model call
  |      covers up to MAX_ESTIMATED_DISHES (21); profiling is capped at
  |      MAX_PROFILE_CALLS (10) and the rest keep the guess. Best-effort.
  |      nutrition_source (recipe | closest_recipe | typical_ingredients |
  |      model_estimate | "") keeps an estimate from reading as a measurement.
  |      Each grounding row carries kcal (per serving, rounded),
  |      typical_ingredients and guess_remarks — a sentence for each thing in
  |      the row that is a guess.
  |
  +--> 3. build_scoring_input  (building.py)
  |      daily  -> course lists per slot (+ extras); entry_dicts for measuring
  |      weekly -> entry dicts {day, meal_idx, meal_type, recipe, reward};
  |                snack/other in extras; a dish on an earlier day is
  |                repeat_source "author" ("repeats are your own choice")
  |      matched: catalogue id, tags, image; approximate/unresolved: stable
  |      "pasted:<title words>" id and no catalogue tags
  |
  +--> 4. PastedPlanScorer.score  (scoring.py)
  |      constraint rows: transparency.constraints_ledger(profile), each row
  |        re-measured dish by dish — allergy, checkable diet (vegetarian,
  |        vegan, pescatarian, gluten/dairy/nut free), dislike -> violated |
  |        satisfied; goals and other diets -> unchecked
  |      measured in code, no model:
  |        (a dish with no ingredient list counts its typical serving, and
  |        the sentence names those dishes as guesses; allergy, diet and
  |        dislike rows never read a guess)
  |        daily : fvs (plan_scoring.food_variety_score) | daily_nutrition
  |                (nutrition_metrics, one-day target)
  |        weekly: weekly_variety (variety_metrics over main meals) |
  |                weekly_guidelines (guideline_checklist; "eat fish" is not
  |                applicable under 7 days) | weekly_nutrition (targets
  |                scaled to the days pasted, snacks counted) | measured
  |                ledger rows (meat limit, calories, repeats)
  |      ONE PlanJudge call returns diversity + guideline_adherence + fit
  |        (agents.PlanJudge, daily or weekly system prompt, the guideline
  |        text and the measured checklist as facts). Three separate calls
  |        sent the same plan three times and reasoned over it three times —
  |        most of a minute's token allowance on the on-demand tier.
  |        Retried once; a call that still fails leaves those three scores
  |        None, never 0, and the measured metrics stand.
  |      fit shares PLAN_SCORING_RUBRIC with the planner's grader and is
  |        capped in code: allergen -> 1 (even with no judge), broken diet
  |        -> 2; the reasoning names the dish
  |      guideline text = plan_scoring.guidelines_text(scope): "" today (the
  |        file is absent), an external endpoint later
  |
  +--> 5. reply: ResponseWriter over facts (scores, broken constraints, dishes
  |       not found, close matches) with a deterministic fallback; then
  |       ensure_calorie_caveat (a reply quoting kcal without saying they are
  |       partly known or estimated gets that sentence) and
  |       ensure_allergen_warnings (one sentence per allergen set, naming every
  |       dish that has it, for any allergen the reply does not name)
  |
  +--> assistant message (intent score_plan) with messages.plan_score = payload
       -> returned on ChatTurn.plan_score and on /conversation messages
```

## 5. Weekly plan branch

```text
[WeeklyPlanService.process_message]
  |
  +--> add user message; inject current weekly canvas text when refining
  +--> DietaryIntentExtractor.extract(content)      -> query-level diet tags
  +--> pantry_service.extract_pantry_delta(RAW message) merged into the
  |      standing PlanningState (same state the daily flow reads — whichever
  |      horizon hears about the zucchini, both honour it)
  +--> RecipeActionSpace(profile, extra diet tags, pantry) -> per-day pools
  |      (RecipeWrangler fetch ONCE PER DAY, serving all three of that day's
  |       slots; each day's pool enriched with one batch details call ->
  |       candidates carry nutrition + diet tags during selection; a stated
  |       pantry folds per-item matches into every day's pool coverage-first,
  |       and build_preference_scorer adds +3 per matched item, capped at 2)
  |
  |      Repeats: a slot-scoped cooldown -- a recipe may return after >= 2
  |      days, at most twice in the week, never in another slot, and never
  |      if it was pinned or downvoted (mark_selected still means never).
  |      WHICH slots may repeat is the member's plan_parameters.repeat_meals
  |      setting: off (21 distinct recipes) -> breakfast (the default) ->
  |      all -> leftovers, each stop a superset of the one before it. The
  |      day's fetch uses the loosest exclusion any of its three slots needs
  |      and the per-slot rule is applied afterwards, so this costs no extra
  |      requests. A candidate allowed back carries repeat_of_day +
  |      repeat_source ("member_request" for a starred recipe, "plan"
  |      otherwise) from the pool to the stored entry.
  |      A repeat_offered event records what a slot COULD have repeated,
  |      taken or not -- "the week repeated nothing" and "the source never
  |      offered anything" have opposite fixes and look identical without it.
  |
  |      Offering vs waiting: when the member SET repeat_meals themselves
  |      (plan_parameters.repeat_mode_is_explicit -- a chosen "breakfast",
  |      not an inherited one), an eligible earlier dish the pool does not
  |      contain is REBUILT from what was committed and added to it, up to
  |      INJECTED_REPEATS_PER_SLOT (2), newest first. Costs no request. Every
  |      ordinary rule still decides eligibility (slot, gap, cap,
  |      mark_selected), and a dish the source did return is never added
  |      twice. On a DEFAULT setting nothing is injected: the week keeps
  |      waiting for the source, exactly as before, because a default is not
  |      a request. The repeat_offered event gains `injected: N` so the
  |      diagnostic above still separates the plan's own additions from the
  |      source's.
  |
  |      Leftovers (repeat_meals = "leftovers" only): day N's dinner is
  |      offered as day N+1's lunch, REBUILT from the committed action
  |      rather than re-fetched, so it costs no request and is the dish on
  |      the plate by construction. It is a repeat with a slot transition,
  |      not a new entry kind: the entry holds the whole recipe (the member
  |      eats that dish, so its nutrition and card are the dish's own) and
  |      carries repeat_of_day + repeat_source "leftover" + leftover_of
  |      {day, meal_type}. Yesterday only, lunch only, never from a pinned
  |      or downvoted dish, at most MAX_LEFTOVER_MEALS (3) a week, and it
  |      still counts toward MAX_APPEARANCES. Appended to the day's pool,
  |      never substituted for it -- the lunch is still planned, and the
  |      leftover only wins if it scores.
  |      No portion arithmetic anywhere: nothing records quantities, so the
  |      claim is "Monday's dinner again", never "the rest of it".
  |
  |      Sourcing (food waste = strict only): before each new day is
  |      fetched the planner offers the ingredients the week has already
  |      bought (IngredientBasket.reusable_items -- >= 2 days old, used
  |      once, named as whole phrases, capped at 3, the member's own items
  |      left to their own fan-out). Those get the same per-item plan_meals
  |      fan-out a stated pantry gets, merged BEFORE the member's pantry so
  |      the member's coverage ranking still decides the top of the pool.
  |      Recorded either way on selection_events: derived_pantry_sourced
  |      per day, or derived_pantry_skipped once with the setting that
  |      skipped it.
  +--> WeeklyMealPlanEnv + WeeklyPlanner.generate_full_plan
  |      21 steps (7 days x 3 meals), fully LLM-free:
  |      apply_hard_constraints prunes the pool (weekly meat limit —
  |      diet-aware, relaxes with a warning if it would empty the pool;
  |      prunes/relaxations recorded on env.selection_events at decision
  |      time), then preference score + soft calorie-budget score pick the
  |      recipe; the tracker accumulates real kcal/macros as slots commit.
  |      An IngredientBasket carries each committed meal's ingredients WITH
  |      the day they land on, so the scorer judges overlap by when the
  |      ingredient was last eaten: same day -1.0 / next day -0.5
  |      (monotony, applied at every food-waste setting), two or more days
  |      later rewarded at the slider's weight, and only twice per
  |      ingredient. No shelf life is modelled — nothing records expiry.
  |      A sanctioned repeat sits out that axis entirely (it shares its
  |      ingredients with itself) and is not charged the variety penalty for
  |      its own earlier title, so it competes on equal terms with a new
  |      dish; the cooldown and the cap are what govern it. A repeat the
  |      member ASKED for (explicit repeat_meals) earns +1.0 instead of 0.0 --
  |      on a bare profile nearly every candidate scores exactly 0.0, so
  |      "equal terms" in a pool of ten is a one-in-eleven share, and a
  |      member who set "Repeat breakfasts" got one. It ties with a liked
  |      ingredient and still loses to a favourite (+5) and a stated pantry
  |      (+3). A LEFTOVER earns +2.0 INSTEAD (never on top -- one control,
  |      one payment) and, uniquely, is NOT
  |      added to the IngredientBasket: the basket is the shopping list and
  |      a leftover buys nothing. That is the one place "eaten twice" and
  |      "bought once" have to be different numbers
  +--> batch enrichment on the final 21 entries (nutrition, image, tags)
  |      + adapted-recipe overlay
  +--> build_day_summaries(entries) -> {day: "dinner with fish" headline}
  +--> build_weekly_explainability(entries, profile, selection_events, ...)
  |      then annotate_weekly_entries (pantry chips, "uses your tomatoes",
  |      ledger source "your pantry") and annotate_shared_ingredients
  |      (cross-day chips, "also uses Monday's cabbage", ledger source
  |      "the plan"). Two kinds, never merged: the first is what the member
  |      told us they had, the second is reuse the plan introduced on its
  |      own. A member-stated item never carries the cross-day chip, and a
  |      share the tokeniser cannot name is not counted at all.
  |      A THIRD kind rides alongside: the repeat chip (kind "repeat", with
  |      a `source` field) -- "back from Monday, a favorite of yours" vs
  |      "the same breakfast as Tuesday" vs, for a leftover, "Monday's
  |      dinner again -- cook once, eat twice" (it names the slot it was
  |      COOKED in; "the same lunch as Monday" would be false). A repeated
  |      meal gets no cross-day reuse chip: it shares its ingredients with
  |      itself, so the two would say the same thing twice and inflate the
  |      reuse count. A leftover is verified against its (day, slot), not
  |      just the day, and is kept out of min_gap_days -- it is one day
  |      after its source by definition, and averaging it in would report
  |      the week's repeats as tighter than the cooldown allows.
  |      annotate_shared_ingredients also APPENDS its sentence to
  |      explainability["reasoning"], because it runs after the prose is
  |      composed
  |      LLM-free: attaches per-entry recipe.match_reasons chips, builds
  |      the MEASURED constraint ledger (meat count / calorie target with
  |      status satisfied | relaxed | violated -- the calorie row is checked
  |      in BOTH directions, and its floor is measured against the meals that
  |      have nutrition data, so a coverage gap never reads as an underfed
  |      week; a reported kcal of 0 is missing data, not a free meal), personalization counts,
  |      weekly metrics (variety + category distribution, deterministic
  |      guideline frequency checklist, nutrition trackers with coverage,
  |      per-day breakdown) and the whole-week justification prose
  +--> pantry_service.annotate_weekly_entries (after explainability, so its
  |      chips are appended to, not overwritten): per-entry pantry badges,
  |      a coverage ledger row, facts["pantry"] {used, unused, note}
  +--> store (refine -> version+1 | fresh -> version 1) + assistant message
  |      (ResponseWriter facts include constraints_honored + week_summary)
```

## 6. Router response assembly

```text
ChatTurnResponse {
  role, content, intent,
  needs_clarification,
  meal_plan?              (id, courses, reasoning, 4 quality metrics,
                           version, parent_id),
  weekly_meal_plan?       (id, entries[day, meal_type, recipe{...,
                           match_reasons, repeat_of_day?, repeat_source?,
                           leftover_of?{day, meal_type}}, reward],
                           day_summaries{day -> headline},
                           constraints_applied[{constraint, type, status,
                           source, detail?}], personalization_summary,
                           metrics{variety (distinct_recipes,
                           planned_repeats, repeats_by_source,
                           leftover_meals, unexplained_repeats, ...),
                           guideline_checklist,
                           nutrition, days, repeats, selection_events},
                           reasoning,
                           version, parent_id),
  at_message_limit,
  plan_version, plan_parent_id,
  plan_score?             (score_plan only: plan_type, days_scored,
                           meals_scored, metrics[{key, label, score, kind:
                           likert5|count|percent|checklist, reasoning,
                           detail}], constraints_applied[{constraint, type,
                           status: satisfied|violated|unchecked, source,
                           detail, members}], grounding[{day, slot,
                           title_given, title_matched, recipe_id, state,
                           ingredients_source, has_nutrition,
                           allergen_conflicts}], unparsed[], warnings[],
                           scored_plan{origin: "pasted", plan_type,
                           days[DayPlanResponse], entries[] (weekly)},
                           context)

GET /sessions/{id}/conversation -> messages[{..., attribution, plan_score}]
  (plan_score is stored with the score_plan reply, so the card survives reloads)
}
```

## 7. Error mapping at the router boundary

```text
ValueError   -> HTTP 404   (session missing / access denied)
RuntimeError -> HTTP 429   (message limit)
Exception    -> HTTP 500
```

## 8. Source files behind this diagram

- [src/main.py](src/main.py)
- [src/routers/foodchat_router.py](src/routers/foodchat_router.py)
- [src/services/orchestrator_service.py](src/services/orchestrator_service.py)
- [src/services/chat_service.py](src/services/chat_service.py)
- [src/services/clarification.py](src/services/clarification.py)
- [src/services/planning_pipeline.py](src/services/planning_pipeline.py)
- [src/services/pantry_service.py](src/services/pantry_service.py)
- [src/services/candidates_client.py](src/services/candidates_client.py)
- [src/services/weekly_plan_service.py](src/services/weekly_plan_service.py)
- [src/services/plan_scorer/](src/services/plan_scorer/) · [src/models/pasted_plan.py](src/models/pasted_plan.py)
- [src/services/plan_scoring.py](src/services/plan_scoring.py) (metrics shared by the daily planner and the scorer)
- [src/services/session_service.py](src/services/session_service.py)
- [src/models/session.py](src/models/session.py) · [src/models/recipe.py](src/models/recipe.py)
- [src/agents.py](src/agents.py)
