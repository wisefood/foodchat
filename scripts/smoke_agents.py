#!/usr/bin/env python3
"""
Every agent, once, against the real Groq API.

    GROQ_API_KEY=... python scripts/smoke_agents.py
    GROQ_API_KEY=... python scripts/smoke_agents.py --only meal_judge

Nothing in the test suite calls Groq — deliberately, because a unit suite that
needs a network and a credential is a suite people stop running. That leaves
one class of failure with no coverage: the call itself. A prompt served from
Langfuse without the word "json", a schema the model will not fill, a reasoning
model whose deliberation lands in `content` and breaks `json.loads` — every one
of those is swallowed by an agent's own `except` and shows up as a feature that
quietly does nothing.

`tests/test_prompt_contracts.py` covers what can be checked offline. This covers
the rest, and it exists so that check is ONE COMMAND rather than an afternoon of
poking at a REPL.

Each agent gets one realistic input. The script reports what came back and
whether it is usable — not whether it is *right*, which is a judgement no script
makes. A blank or fallback answer is the signal: that is precisely what a
swallowed 400 looks like from the outside.

Exit code 1 if any agent came back empty, so it can gate a deploy.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

VOCAB = {
    "cuisines": ["thai", "italian", "greek"],
    "moods": ["comforting", "light", "hearty"],
    "flavor_profiles": ["spicy", "fresh"],
    "food_groups": ["vegetables", "legumes"],
}


def _plan_text() -> str:
    return (
        "Breakfast: Porridge with berries\nIngredients: oats, milk, blueberries\n"
        "Directions: simmer\n\n"
        "Lunch: Lentil soup\nIngredients: lentils, carrot, onion\nDirections: simmer\n\n"
        "Dinner: Grilled salmon with green salad\n"
        "Ingredients: salmon, lettuce, lemon\nDirections: grill\n"
    )


def _candidates():
    from models.recipe import CandidateRecipe

    return {
        slot: [
            CandidateRecipe(f"{slot}-{n}", f"{slot.title()} option {n}",
                            "beans, rice, tomato", "cook it")
            for n in range(2)
        ]
        for slot in ("breakfast", "lunch", "dinner")
    }


def _checks() -> list[tuple[str, callable]]:
    """(name, run) — each returns something falsy when the agent gave nothing."""
    import agents

    return [
        ("orchestrator", lambda: agents.OrchestratorAgent().classify(
            "plan me a week of dinners", [],
        ).get("intent")),
        ("plan_spec", lambda: agents.PlanSpecExtractor().extract(
            "three days, dinner with a salad on the side",
        ).describe()),
        ("dietary_intent", lambda: agents.DietaryIntentExtractor().extract(
            "keep it vegetarian and high protein",
        )),
        ("pantry", lambda: agents.PantryExtractor().extract(
            "I have spinach and half a jar of olives to use up",
        )),
        ("plan_intent", lambda: agents.PlanIntentExtractor().extract(
            "something comforting and Thai tonight", VOCAB,
        )),
        ("seed", lambda: agents.SeedExtractor().extract(
            "I want apple pie for breakfast",
        )),
        ("preference", lambda: agents.PreferenceExtractor().extract(
            "remember that I don't like mushrooms",
        )),
        ("edit_command", lambda: agents.EditCommandExtractor().extract(
            "swap Tuesday's dinner for something lighter", "weekly",
        )),
        ("tool_selector", lambda: agents.ToolSelector().choose(
            "how many calories is my week?",
            plan_type="weekly", plan_shape="7 day(s), 21 meals",
            manifest="- plan_totals(plan_type, session_id): Add up a plan's calories.",
            allowed={"plan_totals"},
        )),
        ("grader", lambda: agents.DocumentGrader().grade_daily_plans(
            "a light vegetarian day", _candidates(), {"diet": ["vegetarian"]}, [],
        )),
        ("diversity", lambda: agents.MealDiversityGrader().score(_plan_text())),
        ("guidelines", lambda: agents.GuidelineAdherenceGrader().score(
            _plan_text(), "Eat fish twice a week. Limit red meat.",
        )),
        ("query_reconciler", lambda: agents.QueryReconciler().reconcile(
            "something light", {"diet": ["vegetarian"], "allergies": []},
        )),
        ("meal_judge", lambda: agents.MealJudge().choose(
            "a rich dinner with something sharp beside it",
            [{
                "meal": "day 1 dinner",
                "options": [
                    "main: Slow pork belly (pork, apple) | side: Buttered peas (peas)",
                    "main: Slow pork belly (pork, apple) | side: Pickled slaw (cabbage, vinegar)",
                ],
            }],
        )),
        ("plan_strategist", lambda: agents.PlanStrategist().propose(
            "something light after the gym", _brief(), VOCAB,
        )),
        ("response_writer", lambda: agents.ResponseWriter().write(
            {"plan": "a vegetarian day", "note": "swapped the dinner"},
            "make it vegetarian",
            "Here is your plan.",
        )),
        ("session_title", lambda: agents.SessionTitler().title(
            "plan me a week of vegetarian dinners",
        )),
        # The plan scorer's three: a pasted plan read, typical servings
        # written, and the one judge call over a scored day.
        ("plan_text_parser", lambda: agents.PlanTextParser().parse(
            "breakfast: porridge with banana\nlunch: lentil soup (lentils, carrot)",
        )),
        ("dish_ingredient_estimator", lambda: agents.DishIngredientEstimator().estimate(
            [{"title": "pasta with zucchini"}],
        )),
        ("plan_judge", lambda: agents.PlanJudge().judge(
            weekly=False, plan_text=_plan_text(), plan_shape="1 day, 3 meals",
            hard_constraints="Diet: vegetarian", conflicts="None found.",
            preferences="(none)", aim="(none)", guidelines="", facts="(none)",
        )),
    ]


def _brief():
    from models.plan_brief import PlanBrief

    return PlanBrief.build({"diet": ["vegetarian"], "allergies": []})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", default=[],
                        help="run just these agents (repeatable)")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set — this script only does live calls.")
        return 2

    from backend.groq import GROQ_DEFAULT_MODEL

    print(f"model (reasoning tier): {GROQ_DEFAULT_MODEL}")
    print(f"fast tier:              {os.getenv('FOODCHAT_FAST_MODEL', '(same)')}")
    print()

    checks = [(n, f) for n, f in _checks() if not args.only or n in args.only]
    if not checks:
        print(f"no agent matches {args.only}")
        return 2

    width = max(len(n) for n, _ in checks)
    empty: list[str] = []
    for name, run in checks:
        started = time.monotonic()
        try:
            answer = run()
        except Exception:                                    # noqa: BLE001
            print(f"{name:<{width}}  RAISED")
            traceback.print_exc()
            empty.append(name)
            continue
        elapsed = time.monotonic() - started
        # Empty is the whole point. Every agent here swallows its own failures
        # and returns a neutral value, so "nothing came back" is what a 400, a
        # bad schema and a stale prompt all look like from outside.
        verdict = "EMPTY" if not answer else "ok"
        if not answer:
            empty.append(name)
        print(f"{name:<{width}}  {verdict:<5} {elapsed:5.1f}s  {str(answer)[:110]}")

    print()
    if empty:
        print(f"{len(empty)} agent(s) returned nothing: {', '.join(empty)}")
        print("That is what a swallowed 400 looks like. Check the logs above.")
        return 1
    print(f"all {len(checks)} agents answered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
