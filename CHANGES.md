# FoodChat — Change Log (newest first)

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
