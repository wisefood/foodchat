"""
Plan scorer, step 4 — grade a pasted plan with FoodChat's own metrics.

    PastedPlanScorer(...).score(plan_type, grounded, built, profile, context) → ScoreResult

What runs, and where each number comes from:

    hard constraints (code)  every profile row re-measured against the dishes:
                             allergy, diet, dislike → violated | satisfied;
                             goals and non-checkable diets → unchecked
    daily    fvs                  plan_scoring.food_variety_score over every dish
             daily_nutrition      explainability.nutrition_metrics, target for one day
    weekly   weekly_variety       explainability.variety_metrics over main meals
             weekly_guidelines    explainability.guideline_checklist
             weekly_nutrition     explainability.nutrition_metrics, targets for the days pasted
    both     diversity            ONE PlanJudge call returns all three, with the
             guideline_adherence  daily or weekly system prompt, the guideline text
             fit                  and the measured facts; fit is capped in code

Every metric has the same shape — ``{key, label, score, kind, reasoning,
detail}`` — so the card renders rows without knowing metric names.

Why the weekly functions are called one by one instead of through
``build_weekly_explainability``: that entry point assumes seven days (it
divides by 7 and compares against a seven-day calorie budget) and counts every
entry as a meal. A pasted three-day plan would be reported far short of its
budget, and an apple would count toward "most meals plant-based". The same
measured functions are called here with targets scaled to the days pasted and
snacks kept out of the meal counts; the planner's own call is untouched.

A dish with no ingredient list is not taken as meat-free for want of a meat
word in its name. Its typical serving (a small model's guess) decides its food
group and is checked against the member's allergies and diet — any clash is a
POSSIBLE one: the row says "may be in" or "possibly broken by", it is
``unchecked`` rather than ``violated``, the judge is told it is a possibility,
and it caps nothing. A dish with no guess either is ``unknown``: it counts
towards no food-frequency rule, and a rule it could tip either way is
reported as not decided. The meat count is read from the same food groups,
and a vegetarian or vegan diet's "no meat" is reported by the diet row alone.

Caps, applied in code after the model answers, because a hard constraint is
not a matter of judgment: an allergen in any dish caps the fit score at 1
(and sets it to 1 even if the fit judge failed); a dish that breaks a
checkable hard diet caps it at 2. The reasoning says which dish did it.

One judge call, retried once. Three separate calls sent the same plan and
profile three times and reasoned over them three times — on the Groq
on-demand tier that is most of a minute's token allowance for one paste. When
the call still fails, every judged metric yields ``score: None`` with a
sentence saying so — never 0, which is off the 1–5 scale and would read as a
verdict. The measured metrics never depend on it.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from models.pasted_plan import (
    APPROXIMATE,
    AS_WRITTEN,
    CLOSEST_RECIPE,
    FROM_RECIPE,
    MATCHED,
    NUTRITION_MODEL_ESTIMATE,
    NUTRITION_TYPICAL,
    TYPICAL_SERVING,
    UNKNOWN,
    GroundedMeal,
    is_possible,
)
from services.candidates_client import DIET_FILTER, allergen_conflict, diet_tag_status
from services.plan_scoring import food_variety_score, guidelines_text, ingredient_names
from services.transparency import constraints_ledger
from services.weekly_planner.day_summary import UNKNOWN_CATEGORY, classify_meal
from services.weekly_planner.explainability import (
    MEAT_LIMIT_KEY,
    attach_match_reasons,
    attach_repeat_reasons,
    guideline_checklist,
    nutrition_metrics,
    repeat_facts,
    variety_metrics,
    weekly_constraints_ledger,
)
from services.weekly_planner.state_tracking import WeeklyNutritionalTracker

from .building import (
    MEAL_INDEX,
    DailyScoringInput,
    ScoringInput,
    WeeklyScoringInput,
    category_dict,
    entry_dicts,
)
from .parsing import PLANT_DAIRY

logger = logging.getLogger(__name__)

LIKERT = "likert5"
COUNT = "count"
PERCENT = "percent"
CHECKLIST = "checklist"

DAILY_METRIC_KEYS = ("fvs", "daily_nutrition", "diversity", "guideline_adherence", "fit")
WEEKLY_METRIC_KEYS = (
    "weekly_variety", "weekly_guidelines", "weekly_nutrition",
    "diversity", "guideline_adherence", "fit",
)
LABELS = {
    "fvs": "Food variety",
    "daily_nutrition": "Calories for the day",
    "weekly_variety": "Variety across days",
    "weekly_guidelines": "Food-frequency rules",
    "weekly_nutrition": "Calories across the plan",
    "diversity": "Nutritional diversity",
    "guideline_adherence": "Dietary guidelines",
    "fit": "Fit to your profile",
}

ALLERGEN_CAP = 1
DIET_CAP = 2
UNCHECKED = "unchecked"
NOT_GRADED = "This part could not be graded just now."
DEFAULT_DAILY_KCAL = 2000

# Diets a single dish can be checked against. Low-carb, high-protein and the
# like are about a whole day's balance, so they go to the fit judge instead.
CHECKABLE_DIETS = frozenset({
    "vegetarian", "vegan", "pescatarian", "pescatarian_safe",
    "gluten_free", "dairy_free", "nut_free",
})
_ANIMAL = frozenset({"red meat", "poultry", "fish"})
_MEAT = frozenset({"red meat", "poultry"})
# "gluten-free pasta" names a food that is not a gluten food: the qualifier and
# the word it qualifies are removed together before the check.
_GLUTEN_FREE = re.compile(r"\bgluten[\s-]*free(?:\s+[a-z]+)?\b", re.IGNORECASE)
# The shared allergen synonyms name grains, not the foods made from them. They
# are left alone — the planner's seed and edit gates read them — and the scorer
# adds the obvious wheat foods for its own diet check.
_GLUTEN_WORDS = (
    "bread", "pasta", "spaghetti", "macaroni", "lasagna", "lasagne", "pizza",
    "toast", "bagel", "croissant", "cracker", "biscuit", "cake", "pastry",
    "pancake", "waffle", "muffin",
)
_PLANT_DAIRY = PLANT_DAIRY
_HONEY = re.compile(r"\bhoney\b", re.IGNORECASE)


@dataclass(frozen=True)
class VarietyCourse:
    """Just what ``food_variety_score`` reads."""

    ingredients: str


def with_guesses(variety: dict, built: WeeklyScoringInput) -> dict:
    """Recount a week's unique ingredients with guessed servings included.

    ``variety_metrics`` is the planner's and reads only recipe ingredients,
    which a pasted dish without a match does not have. Only the ingredient
    count changes: distinct recipes and meal categories stay what the member's
    words and the matched recipes say.
    """
    mains = [meal for meal in built.meals if meal.slot in MEAL_INDEX]
    if not any(meal.guessed_ingredients for meal in mains):
        return variety
    items: set = set()
    for meal in mains:
        items.update(ingredient_names(meal.variety_ingredients))
    before = variety["unique_ingredients"]
    variety["unique_ingredients"] = len(items)
    variety["dishes_with_guessed_ingredients"] = sum(1 for m in mains if m.guessed_ingredients)
    variety["reasoning"] = variety["reasoning"].replace(
        f"; {before} unique ingredients;", f"; {len(items)} unique ingredients;", 1,
    ).rstrip(".") + ". " + guessed_ingredients_sentence(mains) + "."
    return variety


@dataclass
class ScoreResult:
    metrics: list = field(default_factory=list)
    constraints: list = field(default_factory=list)
    # {"allergens": [{allergen, dishes}], "diet": [{diet, dishes}], "dislikes": [{dislike, dishes}]}
    # plus the possibilities, which cap nothing: "possible_allergens" and
    # "possible_diet", each [{allergen | diet, dishes}].
    violations: dict = field(default_factory=lambda: empty_violations())
    # Daily only: the dish entries with their reason chips, for the plan card.
    plan_entries: list = field(default_factory=list)


def metric(key: str, score, kind: str, reasoning: str, detail: Optional[dict] = None) -> dict:
    return {
        "key": key,
        "label": LABELS[key],
        "score": score,
        "kind": kind,
        "reasoning": reasoning,
        "detail": detail or {},
    }


# --------------------------------------------------------------------- #
# Hard constraints, measured in code                                      #
# --------------------------------------------------------------------- #

def empty_violations() -> dict:
    return {
        "allergens": [], "diet": [], "dislikes": [],
        "possible_allergens": [], "possible_diet": [],
    }


def _unique(items):
    return list(dict.fromkeys(items))


def _text(meal: GroundedMeal) -> str:
    return f"{meal.title_given} {meal.ingredients}"


def diet_conflict(meal: GroundedMeal, diet_tag: str, guessed: bool = False) -> Optional[str]:
    """What in the dish breaks ``diet_tag`` (a RecipeWrangler diet tag), or None.

    ``guessed`` reads the dish's typical serving instead of its (empty)
    ingredient list — a possibility, which the caller reports as one.
    """
    dish = category_dict(meal, guessed=guessed)
    category = classify_meal(dish)
    text = f"{dish['title']} {dish['ingredients']}"
    if diet_tag == "vegetarian":
        return category if category in _ANIMAL else None
    if diet_tag == "vegan":
        if category in _ANIMAL:
            return category
        if category == "vegan":
            return None
        plain = _PLANT_DAIRY.sub(" ", text)
        if allergen_conflict(plain, ["dairy"]):
            return "dairy"
        if allergen_conflict(plain, ["eggs"]):
            return "eggs"
        if _HONEY.search(plain):
            return "honey"
        return None
    if diet_tag in ("pescatarian", "pescatarian_safe"):
        return category if category in _MEAT else None
    if diet_tag == "gluten_free":
        plain = _GLUTEN_FREE.sub(" ", text)
        if allergen_conflict(plain, ["gluten"]) or any(
            re.search(rf"\b{word}s?\b", plain, re.IGNORECASE) for word in _GLUTEN_WORDS
        ):
            return "gluten"
        return None
    if diet_tag == "dairy_free":
        return "dairy" if allergen_conflict(_PLANT_DAIRY.sub(" ", text), ["dairy"]) else None
    if diet_tag == "nut_free":
        return "nuts" if allergen_conflict(text, ["nuts"]) else None
    return None


def _clear_detail(name_only: int, lead: str, guessed: int = 0) -> str:
    """``lead``, plus what the check could read for dishes with no ingredient
    list: ``guessed`` of them had a typical serving, ``name_only`` nothing."""
    if guessed:
        lead += (
            f"; {guessed} dish(es) have no ingredient list, so their names and a "
            "typical serving's ingredients were checked"
        )
    if name_only:
        lead += (
            f"; {name_only} dish(es) have no ingredient list, so only "
            "their names were checked"
        )
    return lead


_POSSIBLE_WHY = {
    CLOSEST_RECIPE: "the closest catalogue recipe has it, though you didn't list it",
    TYPICAL_SERVING: "a typical serving has it, though you didn't list it",
}


def _possible_detail(by_evidence: dict) -> str:
    """"may be in “X” — the closest catalogue recipe has it …; may be in “Y” — …"."""
    return "; ".join(
        "may be in " + ", ".join(f"“{t}”" for t in titles) + " — " + _POSSIBLE_WHY[evidence]
        for evidence, titles in by_evidence.items() if titles
    )


def constraint_rows(grounded: list[GroundedMeal], profile: dict) -> tuple[list[dict], dict]:
    """The profile's constraint rows, each re-measured against the pasted dishes.

    Starts from ``transparency.constraints_ledger`` so the rows, their wording
    and their household attribution are the ones a generated plan shows. Only
    status and detail change: a generated plan's rows say "satisfied" because
    the planner filtered at fetch time; a pasted plan was never filtered, so
    each row is checked dish by dish or marked ``unchecked``.
    """
    violations = empty_violations()
    no_list = sum(1 for meal in grounded if meal.ingredients_source == UNKNOWN)
    guessed = sum(1 for meal in grounded if meal.guessed_ingredients)
    # Dishes the allergy and diet checks read by name alone; the dislike check
    # reads no guesses, so for it every dish without a list is name-only.
    name_only = sum(
        1 for meal in grounded
        if meal.ingredients_source == UNKNOWN and not meal.guessed_ingredients
    )
    rows: list[dict] = []
    for base in constraints_ledger(profile):
        row = dict(base)
        source = row.get("source")
        text = str(row.get("constraint") or "")

        if source == "allergy":
            allergen = text[3:] if text.startswith("no ") else text
            key = allergen.strip().lower()
            hits = [
                (meal.title_given, conflict.get("evidence"))
                for meal in grounded for conflict in meal.allergen_conflicts
                if conflict.get("allergen") == key
            ]
            definite = [(t, ev) for t, ev in hits if not is_possible(ev)]
            # A dish is named once, under the strongest evidence it has.
            by_evidence = {
                evidence: _unique(t for t, ev in hits if ev == evidence)
                for evidence in (CLOSEST_RECIPE, TYPICAL_SERVING)
            }
            by_evidence[TYPICAL_SERVING] = [
                t for t in by_evidence[TYPICAL_SERVING] if t not in by_evidence[CLOSEST_RECIPE]
            ]
            possible = by_evidence[CLOSEST_RECIPE] + by_evidence[TYPICAL_SERVING]
            if definite:
                named = []
                for title in _unique(t for t, _ in definite):
                    only_recipe = all(ev == FROM_RECIPE for t, ev in definite if t == title)
                    named.append(f"“{title}”" + (" (in the matched recipe)" if only_recipe else ""))
                row.update(status="violated", detail="found in " + ", ".join(named))
                violations["allergens"].append({
                    "allergen": allergen, "dishes": _unique(t for t, _ in definite),
                })
            elif possible:
                # Only a recipe that merely resembles the dish has it, or a
                # typical serving of it does. A warning the member should see,
                # not a verdict: it caps nothing. The judge is told too, so its
                # reasoning cannot say "no allergens".
                row.update(status=UNCHECKED, detail=_possible_detail(by_evidence))
                violations["possible_allergens"].append({"allergen": allergen, "dishes": possible})
            else:
                row.update(status="satisfied", detail=_clear_detail(
                    name_only, "no listed dish contains it", guessed,
                ))

        elif source == "dietary group":
            status, tag = diet_tag_status(text)
            if status == DIET_FILTER and tag in CHECKABLE_DIETS:
                hits, maybe = [], []
                for meal in grounded:
                    reason = diet_conflict(meal, tag)
                    if reason:
                        hits.append((meal.title_given, reason))
                    elif meal.guessed_ingredients:
                        # Live: a "full english" with nothing listed passed a
                        # vegetarian check, while its typical serving — bacon,
                        # sausage, black pudding — sat on the same dish.
                        guess = diet_conflict(meal, tag, guessed=True)
                        if guess:
                            maybe.append((meal.title_given, guess))
                broken_by = ", ".join(f"“{t}” ({r})" for t, r in _unique(hits))
                possibly = ", ".join(f"“{t}” ({r})" for t, r in _unique(maybe))
                if maybe:
                    violations["possible_diet"].append(
                        {"diet": text, "dishes": _unique(t for t, _ in maybe)},
                    )
                if hits:
                    detail = "broken by " + broken_by
                    if maybe:
                        detail += f"; possibly also by {possibly}, judging from a typical serving"
                    row.update(status="violated", detail=detail)
                    violations["diet"].append({"diet": text, "dishes": _unique(t for t, _ in hits)})
                elif maybe:
                    row.update(status=UNCHECKED, detail=(
                        f"possibly broken by {possibly} — a typical serving has it, "
                        "though you didn't list the ingredients"
                    ))
                else:
                    row.update(status="satisfied", detail=_clear_detail(
                        name_only, "no listed dish breaks it", guessed,
                    ))
            else:
                row.update(
                    status=UNCHECKED,
                    detail="not something a single dish can be checked against, so the fit score weighs it",
                )

        elif source == "preferences" and text.startswith("avoiding "):
            dislike = text[len("avoiding "):]
            dishes = _unique(
                meal.title_given for meal in grounded if allergen_conflict(_text(meal), [dislike])
            )
            if dishes:
                row.update(status="violated", detail="in " + ", ".join(f"“{t}”" for t in dishes))
                violations["dislikes"].append({"dislike": dislike, "dishes": dishes})
            else:
                row.update(status="satisfied", detail=_clear_detail(no_list, "no listed dish has it"))

        else:
            row.update(status=UNCHECKED, detail="the fit score weighs this; it is not checked dish by dish")
        rows.append(row)
    return rows, violations


# --------------------------------------------------------------------- #
# Measured metrics                                                        #
# --------------------------------------------------------------------- #

def explicit_calorie_target(profile: dict) -> bool:
    return any("calories target" in str(p).lower() for p in profile.get("preferences") or [])


def scaled_targets(profile: dict, days: int):
    """The planner's weekly targets, scaled to the days actually pasted."""
    tracker = WeeklyNutritionalTracker(profile)
    targets = dict(tracker.targets)
    factor = days / 7.0
    for key in ("calories", "protein", "carbs", "fat"):
        targets[key] = float(targets.get(key) or 0.0) * factor
    limit = int(targets.get("meat_limit") or 0)
    targets["meat_limit"] = math.ceil(limit * factor) if limit else 0
    return targets, tracker


def with_daily_average(nutrition: dict, days: int) -> dict:
    """``nutrition_metrics`` divides by seven; a pasted plan has its own day count."""
    covered = int((nutrition.get("coverage") or {}).get("meals_with_data") or 0)
    total = float((nutrition.get("weekly_totals") or {}).get("kcal") or 0.0)
    nutrition["daily_average_kcal"] = round(total / days, 1) if covered else None
    return nutrition


def estimated_dishes(meals: list) -> dict:
    """How many dishes carry estimated calories, by kind of estimate."""
    counts = {NUTRITION_TYPICAL: 0, NUTRITION_MODEL_ESTIMATE: 0}
    for meal in meals:
        source = getattr(meal, "nutrition_source", "")
        if source in counts:
            counts[source] += 1
    return counts


def nutrition_metric(key: str, nutrition: dict, targets: dict, days: int, explicit: bool) -> dict:
    coverage = nutrition.get("coverage") or {}
    covered = int(coverage.get("meals_with_data") or 0)
    total_dishes = int(coverage.get("total_meals") or 0)
    per_day_target = float(targets.get("calories") or 0.0) / days
    detail = {**nutrition, "days": days, "target_is_default": not explicit}
    if not covered:
        return metric(
            key, None, PERCENT,
            "None of the dishes have nutrition data, so calories were not measured.", detail,
        )

    average = float(nutrition.get("daily_average_kcal") or 0.0)
    pct = nutrition.get("budget_used_pct")
    span = "for the day" if days == 1 else "a day on average"
    whose = "your target" if explicit else "a default target"
    sentence = f"About {average:,.0f} kcal {span} against {whose} of {per_day_target:,.0f} kcal"
    if pct is not None:
        sentence += f" ({pct}%)"
    if covered < total_dishes:
        sentence += f", counting only the {covered} of {total_dishes} dishes with nutrition data"
    status = nutrition.get("budget_status")
    if status == "under":
        sentence += "; short of the target"
    elif status == "over":
        sentence += "; over the target"
    elif status == "on_track":
        sentence += "; on track"
    sentence += "."
    if not explicit:
        sentence += (
            f" Your profile has no calorie target, so {DEFAULT_DAILY_KCAL:,} kcal a day was used."
        )
    estimated = detail.get("estimated_dishes") or {}
    typical = int(estimated.get(NUTRITION_TYPICAL) or 0)
    guessed = int(estimated.get(NUTRITION_MODEL_ESTIMATE) or 0)
    if typical:
        sentence += (
            f" {typical} dish(es) had no recipe, so their calories are estimated from typical"
            " ingredients."
        )
    if guessed:
        sentence += (
            f" {guessed} dish(es) use a rough calorie guess, because their typical ingredients"
            " could not be profiled."
        )
    return metric(key, pct, PERCENT, sentence, detail)


def scale_checklist(rows: list[dict], days: int) -> list[dict]:
    """A weekly minimum cannot be missed by a plan shorter than a week.

    "Eat fish 1–2 times a week" with no fish in three days is not a failure
    yet, so that row becomes not applicable (``met: None``). Ceilings still
    apply to a shorter span, and the plant-based rule already scales by meals.
    """
    if days >= 7:
        return rows
    scaled = []
    for row in rows:
        row = dict(row)
        if str(row.get("rule", "")).startswith("eat fish") and int(row.get("actual") or 0) < 1:
            row["met"] = None
            row["note"] = f"a weekly rule, and this plan covers {days} day(s)"
        scaled.append(row)
    return scaled


_AT_LEAST = re.compile(r"\bat least (\d+)", re.IGNORECASE)
_AT_MOST = re.compile(r"\bat most (\d+)", re.IGNORECASE)
_RANGE = re.compile(r"\b(\d+)\s*[–-]\s*(\d+)\b")


def _bounds(target: str) -> tuple[Optional[int], Optional[int]]:
    """``(floor, ceiling)`` of a checklist target: "at least 5 of 9 meals",
    "at most 3 meals", "1–2 meals". ``(None, None)`` when it says neither."""
    low = _AT_LEAST.search(target)
    high = _AT_MOST.search(target)
    if low or high:
        return (int(low.group(1)) if low else None), (int(high.group(1)) if high else None)
    both = _RANGE.search(target)
    if both:
        return int(both.group(1)), int(both.group(2))
    return None, None


def allow_for_unknown_meals(rows: list[dict], unknown: int) -> list[dict]:
    """A rule the unclassified meals could still tip either way is undecided.

    Two meals nobody could place in a food group might be plant-based or red
    meat. "Most meals plant-based: met" was reported over a full English and a
    shrimp pad thai counted as vegetarian; with those two unknown, "3 of 9,
    target 5" can be neither met nor missed, and saying either is a guess.
    A rule already settled whichever way they fall keeps its verdict.
    """
    if not unknown:
        return rows
    out = []
    for row in rows:
        row = dict(row)
        floor, ceiling = _bounds(str(row.get("target") or ""))
        if row.get("met") is not None and (floor is not None or ceiling is not None):
            actual = int(row.get("actual") or 0)
            could_meet = (floor is None or actual + unknown >= floor) and (
                ceiling is None or actual <= ceiling
            )
            could_miss = (floor is not None and actual < floor) or (
                ceiling is not None and actual + unknown > ceiling
            )
            if could_meet and could_miss:
                row["met"] = None
                row["undecided"] = True
                row["note"] = f"{unknown} meal(s) could not be placed in a food group"
        out.append(row)
    return out


def category_basis(entries: list[dict]) -> dict:
    """Which meals' food group is a guess, and which have none.

    ``{"guessed": [titles], "unknown": [titles]}``, one title per meal.
    """
    guessed, unknown = [], []
    for entry in entries:
        recipe = entry["recipe"]
        title = str(recipe.get("title") or "")
        if classify_meal(recipe) == UNKNOWN_CATEGORY:
            unknown.append(title)
        elif recipe.get("meal_category_basis") == TYPICAL_SERVING:
            guessed.append(title)
    return {"guessed": guessed, "unknown": unknown}


def _some(titles: list[str]) -> str:
    distinct = _unique(titles)
    return _quoted(distinct[:4]) + (", …" if len(distinct) > 4 else "")


def category_sentence(basis: dict) -> str:
    """What the food-group counts rest on, when it is less than an ingredient list."""
    parts = []
    if basis["guessed"]:
        parts.append(
            f"The food group of {len(basis['guessed'])} meal(s) ({_some(basis['guessed'])}) is "
            "judged from a typical serving, since no ingredients were listed"
        )
    if basis["unknown"]:
        parts.append(
            f"{len(basis['unknown'])} meal(s) ({_some(basis['unknown'])}) could not be placed in "
            "a food group, so they count towards no food-frequency rule"
        )
    return ". ".join(parts) + "." if parts else ""


def with_category_basis(variety: dict, basis: dict) -> dict:
    variety["meals_classified_from_typical_serving"] = len(basis["guessed"])
    variety["meals_unclassified"] = len(basis["unknown"])
    sentence = category_sentence(basis)
    if sentence:
        variety["reasoning"] = variety["reasoning"].rstrip(".") + ". " + sentence
    return variety


def _rule_line(row: dict) -> str:
    if row.get("met") is None:
        word = "not decided" if row.get("undecided") else "not applicable"
        verdict = f"{word} ({row.get('note', '')})"
    else:
        verdict = "met" if row["met"] else "not met"
    return f"{row['rule']}: {row['actual']} (target {row['target']}), {verdict}"


def checklist_metric(rows: list[dict], days: int) -> dict:
    applicable = [r for r in rows if r.get("met") is not None]
    met = sum(1 for r in applicable if r["met"])
    reasoning = (
        f"Meets {met} of {len(applicable)} food-frequency rules. "
        + "; ".join(_rule_line(r) for r in rows) + "."
    )
    return metric(
        "weekly_guidelines", met, CHECKLIST, reasoning,
        {"rules": rows, "applicable": len(applicable), "days": days},
    )


def checklist_facts(rows: list[dict], days: int, basis: Optional[dict] = None) -> str:
    facts = f"The plan covers {days} day(s).\n" + "\n".join("- " + _rule_line(r) for r in rows)
    sentence = category_sentence(basis) if basis else ""
    return f"{facts}\n{sentence}" if sentence else facts


# --------------------------------------------------------------------- #
# What the judges read                                                    #
# --------------------------------------------------------------------- #

def _ingredient_line(meal: GroundedMeal) -> str:
    text = (meal.ingredients or "")[:400]
    if meal.ingredients_source == FROM_RECIPE and meal.state == MATCHED:
        return f"Ingredients from the catalogue recipe “{meal.title_matched}”: {text}"
    if meal.ingredients_source == FROM_RECIPE:
        return (
            "Not found exactly. Ingredients of the closest catalogue recipe, "
            f"“{meal.title_matched}”: {text}"
        )
    closest = (
        f" (closest catalogue recipe: “{meal.title_matched}”, which may be a different dish)"
        if meal.state == APPROXIMATE and meal.title_matched else ""
    )
    if meal.ingredients_source == AS_WRITTEN:
        return f"Ingredients as the user wrote them: {text}{closest}"
    if meal.guessed_ingredients:
        return (
            f"Ingredients not known; a typical serving might contain (a guess, not the "
            f"user's words): {meal.variety_ingredients[:400]}.{closest}"
        )
    return f"Ingredients not known.{closest}"


def guessed_ingredients_sentence(meals: list) -> str:
    """The remark a variety count carries when it counted guessed servings."""
    titles = list(dict.fromkeys(m.title_given for m in meals if m.guessed_ingredients))
    if not titles:
        return ""
    return (
        f"{len(titles)} dish(es) ({_quoted(titles[:4])}{', …' if len(titles) > 4 else ''}) are "
        "counted with a typical serving's ingredients — a guess, not what you wrote or a recipe"
    )


def _nutrition_line(meal: GroundedMeal) -> str:
    nutrition = meal.nutrition or {}
    try:
        kcal = float(nutrition.get("kcal") or 0)
    except (TypeError, ValueError):
        kcal = 0.0
    if kcal <= 0:
        return ""
    line = f"About {kcal:.0f} kcal per serving"
    protein = nutrition.get("protein_g")
    if isinstance(protein, (int, float)):
        line += f", {protein:.0f} g protein"
    if meal.nutrition_source == NUTRITION_TYPICAL:
        line += " (estimated from typical ingredients, not a recipe)"
    elif meal.nutrition_source == NUTRITION_MODEL_ESTIMATE:
        line += " (a rough guess, not a recipe)"
    elif meal.state == APPROXIMATE:
        line += " (figures of the closest recipe)"
    return line


def plan_text_for_judges(grounded: list[GroundedMeal], weekly: bool) -> str:
    """The plan as every judge reads it: grouped by day, each dish saying where
    its ingredients came from. A catalogue list is never presented as the
    member's own words."""
    lines: list[str] = []
    current_day = object()
    indent = "  " if weekly else ""
    for meal in grounded:
        if weekly and meal.day != current_day:
            current_day = meal.day
            lines.append(f"Day {meal.day}:")
        quantity = f" ({meal.quantity_note})" if meal.quantity_note else ""
        lines.append(f"{indent}{meal.slot}: {meal.title_given}{quantity}")
        lines.append(f"{indent}  {_ingredient_line(meal)}")
        nutrition = _nutrition_line(meal)
        if nutrition:
            lines.append(f"{indent}  {nutrition}")
    return "\n".join(lines)


def _quoted(titles) -> str:
    return ", ".join(f"“{t}”" for t in titles)


def judge_inputs(
    profile: dict, violations: dict, judge_text: str, weekly: bool, days: int,
    aim: Optional[str], guidelines: str, facts: str,
) -> dict:
    """Everything the single judge call reads, as prompt variables."""
    hard = []
    allergies = [str(a) for a in profile.get("allergies") or [] if str(a).strip()]
    if allergies:
        hard.append("Allergies (never acceptable): " + ", ".join(allergies))
    diets = [str(d) for d in profile.get("diet") or [] if str(d).strip()]
    if diets:
        hard.append("Diet: " + ", ".join(diets))

    conflicts = [
        f"- {_quoted(v['dishes'])} contains {v['allergen']}, an allergy"
        for v in violations["allergens"]
    ] + [
        f"- {_quoted(v['dishes'])} breaks the {v['diet']} diet" for v in violations["diet"]
    ] + [
        f"- {_quoted(v['dishes'])} contains {v['dislike']}, which the user dislikes"
        for v in violations["dislikes"]
    ] + [
        f"- POSSIBLE ONLY: {_quoted(v['dishes'])} might contain {v['allergen']}, an allergy — a "
        "catalogue recipe with a similar name or a typical serving of the dish does, but the user "
        "did not list it and their dish may well not. Not a broken hard constraint: do not score fit as if it were; you may "
        "mention the risk."
        for v in violations.get("possible_allergens", [])
    ] + [
        f"- POSSIBLE ONLY: {_quoted(v['dishes'])} might break the {v['diet']} diet — a typical "
        "serving of it does, but the user listed no ingredients and their dish may not. "
        "Not a broken hard constraint: weigh the likelihood, and say it is a possibility."
        for v in violations.get("possible_diet", [])
    ]

    preferences = []
    if profile.get("food_likes"):
        preferences.append("Likes: " + ", ".join(map(str, profile["food_likes"])))
    if profile.get("food_dislikes"):
        preferences.append("Dislikes: " + ", ".join(map(str, profile["food_dislikes"])))
    if profile.get("preferences"):
        preferences.append("Stated preferences and targets: " + "; ".join(map(str, profile["preferences"])))
    if profile.get("dietary_goals"):
        preferences.append(
            "Goals: " + ", ".join(str(g).replace("_", " ") for g in profile["dietary_goals"])
        )
    nutrition_profile = profile.get("nutrition_profile")
    if isinstance(nutrition_profile, dict) and nutrition_profile:
        preferences.append(
            "Per-serving nutrition targets: "
            + ", ".join(f"{k} {v}" for k, v in nutrition_profile.items())
        )
    if not explicit_calorie_target(profile):
        preferences.append("No calorie target on file.")

    return {
        "weekly": weekly,
        "hard_constraints": "\n".join(hard) or "None on file.",
        "conflicts": "\n".join(conflicts) or "None found.",
        "preferences": "\n".join(preferences),
        "aim": (aim or "").strip() or "None given. Judge against the goals and preferences above.",
        "guidelines": guidelines,
        "facts": facts,
        "plan_shape": f"{days} days" if weekly else "one day",
        "plan_text": judge_text,
    }


# --------------------------------------------------------------------- #
# Judges                                                                  #
# --------------------------------------------------------------------- #

def likert(result) -> tuple[Optional[int], str]:
    """A judge's ``{score, reasoning}`` as (1–5 or None, reasoning)."""
    if not isinstance(result, dict):
        return None, NOT_GRADED
    try:
        score = int(result.get("score"))
    except (TypeError, ValueError):
        score = None
    if score is None or not 1 <= score <= 5:
        return None, NOT_GRADED
    return score, str(result.get("reasoning") or "").strip()


def fit_metric(section: tuple, violations: dict) -> dict:
    """The fit score after the code caps. ``section`` is (score, reasoning)."""
    score, reasoning = section
    model_score = score
    cap = why = None
    if violations["allergens"]:
        found = violations["allergens"][0]
        cap = ALLERGEN_CAP
        why = f"“{found['dishes'][0]}” contains {found['allergen']}, which is on your allergy list"
    elif violations["diet"]:
        found = violations["diet"][0]
        cap = DIET_CAP
        why = f"“{found['dishes'][0]}” breaks your {found['diet']} diet"

    applied = False
    if cap is not None and score is not None and score > cap:
        score, applied = cap, True
        reasoning = f"{reasoning} Capped at {cap}: {why}.".strip()
    elif cap == ALLERGEN_CAP and score is None:
        # The floor of the scale needs no judge: an allergen settles it.
        score, applied = cap, True
        reasoning = f"Scored {cap} without the fit judge: {why}."
    return metric("fit", score, LIKERT, reasoning, {
        "cap": cap, "cap_applied": applied, "model_score": model_score,
    })


JUDGE_ATTEMPTS = 2
JUDGED_METRIC_KEYS = ("diversity", "guideline_adherence", "fit")


def judgement_sections(payload) -> dict[str, tuple]:
    """The three (score, reasoning) pairs of one judge payload.

    A payload missing a section, or carrying a score off the scale, leaves that
    metric ungraded rather than failing the others: one bad section is not a
    reason to drop two good ones.
    """
    sections = {}
    for key in JUDGED_METRIC_KEYS:
        part = payload.get(key) if isinstance(payload, dict) else None
        sections[key] = likert(part)
    return sections


def retrying(job: Callable[[], dict], attempts: int = JUDGE_ATTEMPTS) -> Callable[[], dict]:
    """One more try for a judge call that raised or returned nothing usable."""
    def run():
        result = None
        for attempt in range(1, attempts + 1):
            try:
                result = job()
            except Exception as exc:  # noqa: BLE001
                if attempt == attempts:
                    raise
                logger.warning(
                    "Plan judge failed (attempt %d): %s: %s", attempt, type(exc).__name__, exc,
                )
                continue
            if any(score is not None for score, _ in judgement_sections(result).values()):
                return result
            logger.warning("Plan judge returned no usable score (attempt %d)", attempt)
        return result
    return run


class PastedPlanScorer:
    """Step 4 for one pasted plan. Stateless; the judges are injectable."""

    def __init__(
        self,
        judge=None,
        guidelines: Callable[[str, dict], str] = guidelines_text,
    ):
        if judge is None:
            from agents import PlanJudge

            judge = PlanJudge()
        self.judge = judge
        self.guidelines = guidelines

    def score(
        self,
        plan_type: str,
        grounded: list[GroundedMeal],
        built: ScoringInput,
        profile: dict,
        context: Optional[str] = None,
    ) -> ScoreResult:
        profile = profile or {}
        constraints, violations = constraint_rows(grounded, profile)
        weekly = isinstance(built, WeeklyScoringInput)
        judge_text = plan_text_for_judges(grounded, weekly=weekly)
        entries: list = []

        if weekly:
            days = max(int(built.days or 0), 1)
            measured, rows, checklist, basis = self._weekly_measured(built, profile)
            constraints = constraints + rows
            facts = checklist_facts(checklist, days, basis)
        else:
            days = 1
            measured, entries = self._daily_measured(built, profile)
            facts = ""

        inputs = judge_inputs(
            profile, violations, judge_text, weekly, days, context,
            self.guidelines("weekly" if weekly else "daily", profile), facts,
        )
        try:
            payload = retrying(lambda: self.judge.judge(**inputs))()
        except Exception as exc:  # noqa: BLE001 — the measured metrics still stand
            logger.warning("Plan judge failed: %s: %s", type(exc).__name__, exc)
            payload = None
        sections = judgement_sections(payload)

        metrics = list(measured)
        for key in ("diversity", "guideline_adherence"):
            score, reasoning = sections[key]
            metrics.append(metric(key, score, LIKERT, reasoning))
        metrics.append(fit_metric(sections["fit"], violations))
        return ScoreResult(
            metrics=metrics, constraints=constraints, violations=violations, plan_entries=entries,
        )

    @staticmethod
    def _daily_measured(built: DailyScoringInput, profile: dict):
        entries = entry_dicts(built.meals)
        attach_match_reasons(entries, profile)

        guessed = [meal for meal in built.meals if meal.guessed_ingredients]
        if guessed:
            count, reasoning = food_variety_score(
                [VarietyCourse(meal.variety_ingredients) for meal in built.meals]
            )
            reasoning += ". " + guessed_ingredients_sentence(built.meals)
        else:
            count, reasoning = food_variety_score(built.all_courses)
        name_only = sum(1 for meal in built.meals if not meal.variety_ingredients)
        if name_only:
            reasoning += (
                f". {name_only} dish(es) have no ingredient list, so they add "
                "nothing to this count"
            )
        targets, _tracker = scaled_targets(profile, 1)
        nutrition = with_daily_average(nutrition_metrics(entries, targets), 1)
        nutrition["estimated_dishes"] = estimated_dishes(built.meals)
        explicit = explicit_calorie_target(profile)
        metrics = [
            metric("fvs", count, COUNT, reasoning, {
                "unique_items": count, "dishes_without_ingredients": name_only,
                "dishes_with_guessed_ingredients": len(guessed),
            }),
            nutrition_metric("daily_nutrition", nutrition, targets, 1, explicit),
        ]
        return metrics, entries

    @staticmethod
    def _weekly_measured(built: WeeklyScoringInput, profile: dict):
        days = max(int(built.days or 0), 1)
        targets, tracker = scaled_targets(profile, days)
        everything = built.entries + built.extras
        attach_match_reasons(everything, profile)
        repeats = repeat_facts(built.entries)
        attach_repeat_reasons(built.entries, repeats)

        variety = variety_metrics(built.entries, repeats)
        with_guesses(variety, built)
        basis = category_basis(built.entries)
        with_category_basis(variety, basis)
        checklist = allow_for_unknown_meals(
            scale_checklist(
                guideline_checklist(variety["category_distribution"], variety["total_meals"]), days,
            ),
            len(basis["unknown"]),
        )
        nutrition = with_daily_average(nutrition_metrics(everything, targets), days)
        nutrition["estimated_dishes"] = estimated_dishes(built.meals)
        # Counted from the same food groups the variety line and the checklist
        # show, so the three cannot disagree. Live, the meat row said 3 while
        # the variety line said 2 poultry and 2 red meat: one read the
        # member's "caesar salad", the other the closest recipe's "Chicken
        # Caesar salad".
        meat_groups = {"red meat", "poultry"} | ({"fish"} if tracker.counts_fish_as_meat else set())
        meat_entries = [e for e in everything if classify_meal(e["recipe"]) in meat_groups]
        meat_count = len(meat_entries)
        guessed_meat = sum(
            1 for e in meat_entries if e["recipe"].get("meal_category_basis") == TYPICAL_SERVING
        )
        ledger = weekly_constraints_ledger(profile, meat_count, targets, [], 0, nutrition, repeats=repeats)
        profile_rows = constraints_ledger(profile)
        explicit = explicit_calorie_target(profile)
        diet_says_no_meat = bool(targets.get("meat_limit_from_diet")) and not targets.get(
            "meat_limit_explicit"
        )
        measured_rows = []
        for row in ledger:
            if row in profile_rows:
                continue  # re-measured by constraint_rows
            if row.get("key") == MEAT_LIMIT_KEY and diet_says_no_meat:
                # A vegetarian or vegan diet's "no meat": the diet row above
                # already names every dish that breaks it, and which of them
                # only possibly do. A second row counting the same meals — by
                # food group, guesses included — would be the same fact twice,
                # with a different number.
                continue
            row = dict(row)
            if row.get("key") == MEAT_LIMIT_KEY:
                if guessed_meat:
                    row["detail"] += (
                        f"; {guessed_meat} of the meat meals judged from a typical serving"
                    )
                if basis["unknown"]:
                    row["detail"] += (
                        f"; {len(basis['unknown'])} meal(s) could not be placed in a food group"
                    )
            constraint = str(row.get("constraint") or "")
            if days < 7 and constraint.endswith(" this week"):
                row["constraint"] = constraint[: -len(" this week")] + f" over these {days} days"
            # The row used to arrive claiming a default was the member's, and
            # this rewrote it here — one module correcting another's
            # attribution after the fact. `weekly_constraints_ledger` now says
            # whose number it is when it builds the row, and names the
            # reference it used, so the correction has nothing left to correct.
            measured_rows.append(row)

        metrics = [
            metric("weekly_variety", variety["distinct_recipes"], COUNT, variety["reasoning"], variety),
            checklist_metric(checklist, days),
            nutrition_metric("weekly_nutrition", nutrition, targets, days, explicit),
        ]
        return metrics, measured_rows, checklist, basis
