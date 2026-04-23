# FoodChat `/chat` Endpoint Pipeline

This document illustrates how a user message flows through `POST /foodchat/sessions/{session_id}/chat`.

## 1. End-to-End Request Flow

```text
Client
  |
  | POST /foodchat/sessions/{session_id}/chat
  | body: { "member_id": "...", "content": "..." }
  v
[foodchat_router.unified_chat]
  |
  +--> _require_orchestrator_service()
  |      |
  |      +--> orchestrator missing
  |             -> HTTP 503
  |
  +--> orch_svc.process(session_id, member_id, content)
         |
         v
      [OrchestratorService.process]
         |
         +--> session_service.get_session(session_id, member_id)
         |      |
         |      +--> session missing / wrong owner
         |             -> ValueError -> router returns HTTP 404
         |
         +--> session.is_at_message_limit ?
         |      |
         |      +--> yes
         |             -> return ChatTurn(
         |                  role="assistant",
         |                  intent="chat",
         |                  at_message_limit=True
         |                )
         |
         +--> session.state == "clarifying" ?
         |      |
         |      +--> yes
         |      |      -> force intent = "daily_plan"
         |      |
         |      +--> no
         |             -> OrchestratorAgent.classify(message, recent history snapshot)
         |
         +--> route by intent
                |
                +--> "daily_plan"
                |      -> ChatService.process_message(..., is_refinement=False)
                |
                +--> "weekly_plan"
                |      -> WeeklyPlanService.process_message(..., is_refinement=False)
                |
                +--> "refine_plan"
                |      |
                |      +--> no active canvas
                |      |      -> ChatService fresh daily plan
                |      |
                |      +--> active weekly canvas
                |      |      -> WeeklyPlanService.process_message(..., is_refinement=True)
                |      |
                |      +--> active daily canvas
                |             -> ChatService.process_message(..., is_refinement=True)
                |
                +--> "switch_plan_type"
                |      |
                |      +--> add assistant acknowledgement
                |      |
                |      +--> target = "weekly"
                |      |      -> WeeklyPlanService fresh weekly plan
                |      |
                |      +--> target = "daily"
                |             -> ChatService fresh daily plan
                |
                +--> "chat" or fallback
                       -> ChatService.process_message(..., is_refinement=False)
```

## 2. Daily Plan and General Chat Branch

```text
[ChatService.process_message]
  |
  +--> session_service.get_session(session_id)
  |
  +--> session_service.add_message("user", message)
  |
  +--> session.state == "clarifying" ?
  |      |
  |      +--> yes
  |             -> _handle_clarification(session_id, message)
  |
  +--> is_refinement and current daily canvas exists ?
  |      |
  |      +--> yes
  |             -> inject current daily plan into effective_message
  |
  +--> QueryClassifier.router(effective_message)
         |
         +--> source = "chatbot"
         |      |
         |      +--> SimpleChatBot.chat(message)
         |      +--> session_service.add_message("assistant", response)
         |      +--> return plain chat response
         |
         +--> source = "vectorstore"
                |
                +--> _handle_rag_query(session_id, effective_message, is_refinement)
```

## 3. Daily Plan RAG and Clarification Loop

```text
[_handle_rag_query]
  |
  +--> rag_chain_part1.invoke({
  |      "question": message,
  |      "user_id": session.member_id,
  |      "user_profile_data": session.user_profile
  |    })
  |
  +--> RAGReadyPreparator.query_check(...)
         |
         +--> clarification needed ?
         |      |
         |      +--> yes
         |             -> yield warning / follow-up question(s)
         |             -> session_service.set_clarification_state(...)
         |             -> session_service.add_message("assistant", first_question)
         |             -> return needs_clarification=True
         |
         +--> no
                -> final_query ready
                -> _run_post_clarification(...)


[_handle_clarification]
  |
  +--> generator.send(user_reply)
         |
         +--> more questions
         |      -> session_service.add_message("assistant", next_question)
         |      -> return needs_clarification=True
         |
         +--> StopIteration(final_query)
                -> _run_post_clarification(...)


[_run_post_clarification]
  |
  +--> session_service.clear_clarification_state()
  |
  +--> rag_chain_part2.invoke({
  |      ...pending_data,
  |      "reformulated_query": final_query
  |    })
  |
  +--> docs found ?
         |
         +--> no
         |      -> apology response
         |      -> session_service.add_message("assistant", msg)
         |      -> return no plan
         |
         +--> yes
                |
                +--> choose top candidate plan
                +--> compute metrics:
                |      - llm_score / llm_reasoning
                |      - fvs_count / fvs_reasoning
                |      - diversity_llm_score / diversity_llm_reasoning
                |      - guideline_adherence_score / reasoning
                |
                +--> store plan
                |      |
                |      +--> is_refinement=True
                |      |      -> session_service.refine_meal_plan(...)
                |      |
                |      +--> is_refinement=False
                |             -> session_service.add_meal_plan(...)
                |
                +--> format assistant response text
                +--> session_service.add_message("assistant", formatted)
                +--> return meal_plan + version + parent_id
```

## 4. Weekly Plan Branch

```text
[WeeklyPlanService.process_message]
  |
  +--> session_service.get_session(session_id)
  |
  +--> session_service.add_weekly_message("user", content)
  |
  +--> is_refinement and current weekly canvas exists ?
  |      |
  |      +--> yes
  |             -> inject current weekly plan into effective_query
  |
  +--> RecipeActionSpace(session.user_profile)
  +--> WeeklyMealPlanEnv(user_profile, action_space, reward_calculator, user_query)
  +--> WeeklyPlanner.generate_full_plan(user_query)
  |
  +--> store weekly plan
  |      |
  |      +--> is_refinement=True
  |      |      -> session_service.refine_weekly_meal_plan(...)
  |      |
  |      +--> is_refinement=False
  |             -> session_service.add_weekly_meal_plan(...)
  |
  +--> session_service.add_weekly_message("assistant", response_text)
  +--> return weekly_meal_plan + version + parent_id
```

## 5. Router Response Assembly

```text
[foodchat_router.unified_chat]
  |
  +--> if turn.meal_plan exists
  |      -> MealPlanResponse.from_meal_plan(...)
  |
  +--> if turn.weekly_meal_plan exists
  |      -> WeeklyMealPlanResponse.from_weekly_meal_plan(...)
  |
  +--> return ChatTurnResponse {
         role,
         content,
         intent,
         needs_clarification,
         meal_plan?,
         weekly_meal_plan?,
         at_message_limit,
         plan_version,
         plan_parent_id
      }
```

## 6. Error Mapping at the Router Boundary

```text
ValueError   -> HTTP 404
RuntimeError -> HTTP 429
Exception    -> HTTP 500
```

## 7. Source Files Behind This Diagram

- [src/main.py](src/main.py)
- [src/routers/foodchat_router.py](src/routers/foodchat_router.py)
- [src/services/orchestrator_service.py](src/services/orchestrator_service.py)
- [src/services/chat_service.py](src/services/chat_service.py)
- [src/services/weekly_plan_service.py](src/services/weekly_plan_service.py)
- [src/services/session_service.py](src/services/session_service.py)
- [src/models/session.py](src/models/session.py)
- [src/agents.py](src/agents.py)
