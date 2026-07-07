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
         +--> session.state == "clarifying" ?
         |      |
         |      +--> yes -> route by persisted clarification kind
         |      |          (NO intent classification — the user is answering
         |      |           our question)
         |      |          +--> kind == "foodscholar" -> FoodScholarService.continue_clarification
         |      |          +--> else (plan flow)      -> ChatService.continue_clarification
         |      |                                        (origin intent restored from state)
         |      |
         |      +--> no  -> OrchestratorAgent.classify(message, last 12 turns)
         |                  (the ONLY intent classification in the pipeline)
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
  |      +--> RecipeCandidatesClient.fetch_candidates(...)
  |      |      hard filters server-side at RecipeWrangler:
  |      |      allergens (FoodOn taxonomy), diet tags, exclude ingredients/ids
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

## 5. Weekly plan branch

```text
[WeeklyPlanService.process_message]
  |
  +--> add user message; inject current weekly canvas text when refining
  +--> DietaryIntentExtractor.extract(content)      -> query-level diet tags
  +--> RecipeActionSpace(profile, extra diet tags)  -> per-day candidate pools
  |      (RecipeWrangler fetch once per day, selected ids excluded -> no repeats)
  +--> WeeklyMealPlanEnv + WeeklyPlanner.generate_full_plan
  |      21 steps (7 days x 3 meals); RewardCalculator scores each step
  +--> store (refine -> version+1 | fresh -> version 1) + assistant message
```

## 6. Router response assembly

```text
ChatTurnResponse {
  role, content, intent,
  needs_clarification,
  meal_plan?              (id, courses, reasoning, 4 quality metrics,
                           version, parent_id),
  weekly_meal_plan?       (id, entries[day, meal_type, recipe, reward],
                           version, parent_id),
  at_message_limit,
  plan_version, plan_parent_id
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
- [src/services/candidates_client.py](src/services/candidates_client.py)
- [src/services/weekly_plan_service.py](src/services/weekly_plan_service.py)
- [src/services/session_service.py](src/services/session_service.py)
- [src/models/session.py](src/models/session.py) · [src/models/recipe.py](src/models/recipe.py)
- [src/agents.py](src/agents.py)
