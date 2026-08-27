import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from .environment import WeeklyMealPlanEnv
from .reward_logic import apply_hard_constraints, constraint_score

logger = logging.getLogger(__name__)

TOTAL_SLOTS = 21  # 7 days x 3 meals


class PlanGenerationError(RuntimeError):
    """A slot could not be filled and the plan cannot be completed.

    Carries which slot failed so the caller can *say* it. This surfaced to
    members as an HTTP 500 with a stack trace — a member asking for a week of
    meals got an internal error page because day 1's breakfast came back
    empty. An unfillable slot is a conversational answer ("here is what I
    couldn't do, and why"), not a server fault.
    """

    def __init__(self, meal_type: str, day: int):
        self.meal_type = str(meal_type)
        self.day = int(day)
        super().__init__(f"No candidate recipes found for {meal_type} on day {day}")


# Ingredients that sit in every kitchen and never go to waste. Reuse scoring
# ignores them: two meals sharing "olive oil" have not saved anyone a wilting
# vegetable, and counting staples would reward exactly nothing while drowning
# the signal from the ingredients that do spoil.
_PANTRY_STAPLES: frozenset = frozenset({
    "salt", "pepper", "water", "sugar", "flour", "oil", "olive", "butter",
    "vinegar", "garlic", "onion", "onions", "stock", "sauce", "soy", "honey",
    "mustard", "cumin", "paprika", "oregano", "thyme", "cinnamon", "vanilla",
    "baking", "powder", "cornflour", "cornstarch", "rice", "pasta", "milk",
    "eggs", "cheese", "lemon", "juice", "cloves", "seeds", "chilli", "chili",
    "ginger", "parsley", "coriander", "basil", "extract", "yeast", "breadcrumbs",
})

# Measurement noise that survives naive tokenising of an ingredients blob.
_UNITS: frozenset = frozenset({
    "cup", "cups", "tablespoon", "tablespoons", "teaspoon", "teaspoons",
    "tbsp", "tsp", "gram", "grams", "kg", "ml", "litre", "liter", "large",
    "small", "medium", "fresh", "dried", "chopped", "sliced", "diced",
    "grated", "finely", "roughly", "optional", "taste", "extra", "plus",
})


def _singular(word: str) -> str:
    """The pantry matcher's stem, so both modules agree on what a word is."""
    from services.pantry_service import singular  # local import; avoids a cycle

    return singular(word)


def perishable_tokens(ingredients_text: Any) -> set:
    """The words in an ingredients blob worth counting as shared.

    In practice a staples-and-noise filter, not a shelf-life judgement: it
    drops what every kitchen already has (`_PANTRY_STAPLES`) and what naive
    tokenising drags in (`_UNITS`), and keeps the rest. Nothing here knows
    or claims when an ingredient expires.
    """
    words = str(ingredients_text or "").lower().replace(",", " ").replace(";", " ").split()
    return {
        w.strip("().:-")
        for w in words
        if len(w) > 3
        and w.strip("().:-").isalpha()
        and w.strip("().:-") not in _PANTRY_STAPLES
        and w.strip("().:-") not in _UNITS
    }


# --- Reuse spacing (M8) -----------------------------------------------------
#
# Reusing an ingredient is worth something; eating it two days running is
# not. That is the whole model. We do not know when anything expires — no
# purchase dates, no shelf lives, nothing in the state records it — so the
# spacing below is about the week being worth eating, never about freshness.

# The gap, in days, at which a repeat stops reading as repetition. Below it
# the ingredient is penalised however much it saves on the shopping list.
_MIN_REUSE_GAP_DAYS = 2

# Reuse across the week, not a theme. Rewarding an ingredient indefinitely
# would buy a smaller shopping list with a duller week, however well spaced.
_MAX_REWARDED_USES = 2

# Monotony weight per shared ingredient. Same day is the worst — cabbage at
# lunch and again at dinner — the day after is milder, and an ingredient
# already used its allowance is repetition wherever it lands.
_SAME_DAY_MONOTONY = 1.0
_ADJACENT_DAY_MONOTONY = 0.5
_OVERUSED_MONOTONY = 0.5

# Both axes are capped, and deliberately capped separately. Without a cap on
# the penalty the scorer would quietly prefer recipes with short ingredient
# lists — they have less to collide with — which is a bias about writing
# style, not about food. The reuse cap is the one that predates this and is
# left where it was.
_REUSE_CAP = 4
_MONOTONY_CAP = 3.0


class IngredientBasket:
    """What the plan has already put in the basket, and the days it lands on.

    The food-waste axis needs more than a flat set of tokens. Reusing
    Monday's cabbage on Wednesday is reuse; reusing it at Monday's dinner is
    just cabbage twice in one day, and on Tuesday it is cabbage two days
    running. Only a basket that remembers *when* can tell those apart, which
    is the whole reason this class exists rather than the `set` it replaces.

    Days are the planner's 1-based day index, so "gap" below is in days, not
    in slots: breakfast-to-dinner on one day is a gap of 0.
    """

    __slots__ = ("_days", "_raw", "_texts", "_item_cache")

    def __init__(self) -> None:
        self._days: Dict[str, List[int]] = {}
        # Unstemmed, purely so `tokens()` hands a pre-M8 scorer exactly the
        # set it used to build for itself. Stemming that would quietly change
        # what those scorers match on.
        self._raw: set = set()
        # Each committed meal's ingredient text, kept so a *pantry item* —
        # which may be a phrase like "olive oil", and is matched with the
        # pantry module's own stemming rules — can be located in the plan.
        # Single tokens cannot answer that question on their own.
        self._texts: List[Tuple[int, str]] = []
        self._item_cache: Dict[Tuple[str, int], List[int]] = {}

    def add(self, ingredients_text: Any, day: int) -> None:
        """Record one committed meal's countable ingredients against its day."""
        for token in perishable_tokens(ingredients_text):
            # Keyed by singular stem, so "tomatoes" on Monday and "tomato" on
            # Tuesday are the same ingredient. Storing them apart let a member
            # be served tomatoes two days running with no penalty at all —
            # the same asymmetry the pantry matcher was fixed for.
            self._days.setdefault(_singular(token), []).append(int(day))
            self._raw.add(token)
        self._texts.append((int(day), str(ingredients_text or "")))

    def days_used(self, token: str) -> List[int]:
        """Every day this ingredient has been eaten on, in commit order."""
        return self._days.get(_singular(token), [])

    def days_matching(self, item: str) -> List[int]:
        """Every day a *pantry item* appears on, by the pantry matcher's rules."""
        key = (item, len(self._texts))
        if key not in self._item_cache:
            from services.pantry_service import matched_items

            self._item_cache[key] = [
                day for day, text in self._texts if matched_items(text, (item,))
            ]
        return self._item_cache[key]

    def tokens(self) -> frozenset:
        """The flat basket, for scorers that predate the day-aware one."""
        return frozenset(self._raw)

    def __bool__(self) -> bool:
        return bool(self._days)


def _reuse_and_monotony(
    ingredients: Any, basket: "IngredientBasket", current_day: int
) -> Tuple[int, float]:
    """Score one candidate's overlap with the basket as (reuse, monotony).

    Two numbers rather than one, because they answer to different masters:
    the reuse half is a food-waste preference the member sets on the ribbon
    and can switch off, while monotony is a property of a week worth eating
    and applies at every setting. Collapsing them would make "off" mean
    "repeat freely", which no member has ever asked for.
    """
    reuse_hits = 0
    monotony = 0.0
    for token in perishable_tokens(ingredients):
        days = basket.days_used(token)
        if not days:
            continue
        gap = int(current_day) - max(days)
        if gap <= 0:
            monotony += _SAME_DAY_MONOTONY
        elif gap < _MIN_REUSE_GAP_DAYS:
            monotony += _ADJACENT_DAY_MONOTONY
        elif len(days) >= _MAX_REWARDED_USES:
            # Well spaced, but this ingredient has had its run of the week.
            monotony += _OVERUSED_MONOTONY
        else:
            reuse_hits += 1
    return reuse_hits, monotony


def _pantry_item_wanted(
    item: str, basket: "IngredientBasket", current_day: int
) -> bool:
    """Whether boosting this on-hand item again still serves the member.

    Same two rules the reuse bonus follows, for the same reason: an item the
    plan has already used twice has been used up as far as this plan can tell,
    and one eaten yesterday is repetition however much of it is in the fridge.
    """
    days = basket.days_matching(item)
    if not days:
        return True
    if len(days) >= _MAX_REWARDED_USES:
        return False
    return int(current_day) - max(days) >= _MIN_REUSE_GAP_DAYS


def build_preference_scorer(
    user_profile: Dict[str, Any], pantry: tuple = (),
) -> Callable:
    """Heuristic candidate scorer (M4): preference-aware selection at zero
    LLM cost. Favorites dominate, liked ingredients boost, similarity to
    meals already in the plan penalizes (variety), and — when the member has
    switched it on — sharing perishable ingredients with meals already chosen
    scores positively (food waste).

    scorer(candidate, chosen_titles, basket=frozenset(), current_day=0) -> float

    The waste weights sit deliberately between the variety penalty (−2 per
    shared title token) and the favourites bonus (+5): `reuse` nudges, it does
    not override a favourite or flatten variety; `strict` is allowed to beat
    the variety penalty, because that is what the member asked for — a
    smaller shopping list at some cost to sameness. This scorer is the only
    place the weekly planner can honour the setting at all: it selects
    without an LLM, so a preference that never becomes a number here is a
    preference that does not exist.

    ``pantry`` — on-hand ingredients the member asked to use up. +3 per
    matched item, capped at two: an explicit this-turn request outranks a
    standing favourite (+5 < 6) but cannot ride a long ingredient list into
    flattening variety. Always on when the member stated a pantry —
    unlike the perishable-reuse axis it needs no slider, because the member
    asked for it in words.

    **Reuse spacing (M8).** Pass an `IngredientBasket` as the third argument
    and the day being planned as the fourth, and the overlap is judged by
    *when* the ingredient was last eaten rather than merely whether it was:

    - same day, or the day after — monotony, −1.0 / −0.5 per ingredient;
    - two or more days later — reuse, rewarded at the slider's weight;
    - after two meals — monotony again, wherever it lands.

    No shelf life is modelled anywhere in this: nothing in the state records
    when an ingredient was bought or when it spoils, so the gap is about the
    week being worth eating and nothing else. A five-day gap is reuse for
    the same reason a two-day gap is.

    The monotony half is not gated on the waste slider. `off` has always
    meant "sharing ingredients earns nothing", and it still does; it never
    meant "serve the same vegetable three days running", and the flat basket
    had no way to notice that it had. Capped at −3.0, below the favourites
    bonus on purpose: a monotonous week is worse than a varied one, but a
    dish the member explicitly favourited still wins its slot.

    Passing a plain set as the third argument (the pre-M8 contract) keeps the
    old flat behaviour exactly — reward on overlap, no spacing, no penalty.
    """
    favorites = {str(f) for f in (user_profile.get("favorite_recipe_ids") or [])}
    likes = [str(l).lower() for l in (user_profile.get("food_likes") or [])]

    from services import plan_parameters  # local import; avoids a cycle at module load
    from services.pantry_service import matched_items, normalize_items
    waste = plan_parameters.waste_mode(user_profile.get("plan_parameters") or {})
    waste_weight = {"off": 0.0, "reuse": 0.8, "strict": 1.6}[waste]
    pantry_items = list(normalize_items(pantry))

    def scorer(
        candidate: Dict[str, Any],
        chosen_titles: List[str],
        basket: Any = frozenset(),
        current_day: int = 0,
    ) -> float:
        score = 0.0
        if str(candidate.get("recipe_id", "")) in favorites:
            score += 5.0
        ingredients = str(candidate.get("recipe_ingredients", "")).lower()
        score += sum(1.0 for like in likes if like and like in ingredients)

        # Pantry (food waste): boost recipes that use what the member has.
        if pantry_items:
            hits = matched_items(
                f"{candidate.get('recipe_title', '')} {ingredients}", pantry_items
            )
            if isinstance(basket, IngredientBasket):
                # Spaced like any other reuse. "I have tomatoes" is a request
                # to use them up, not to eat them every day — and at +3 each
                # this boost outranks the monotony cap on its own, so without
                # this the member gets the item in nearly every slot. Observed:
                # a stated pantry of tomatoes put them in 15 of 21 meals.
                hits = [h for h in hits if _pantry_item_wanted(h, basket, current_day)]
            score += 3.0 * min(len(hits), 2)

        # Variety: penalize title-token overlap with meals already planned.
        title_tokens = {
            t for t in str(candidate.get("recipe_title", "")).lower().split() if len(t) > 3
        }
        for chosen in chosen_titles:
            overlap = title_tokens & {t for t in chosen.lower().split() if len(t) > 3}
            if overlap:
                score -= 2.0 * len(overlap)

        # Food waste + monotony: what this candidate shares with the plan so
        # far, judged by when it was last eaten. Both halves capped so one
        # candidate cannot ride a busy ingredient list in either direction.
        if isinstance(basket, IngredientBasket):
            reuse_hits, monotony = _reuse_and_monotony(
                ingredients, basket, current_day
            )
            score += waste_weight * min(reuse_hits, _REUSE_CAP)
            score -= min(monotony, _MONOTONY_CAP)
        elif waste_weight and basket:
            # Pre-M8 flat basket: no days to reason about, so reward overlap
            # the way it was rewarded before the window existed.
            shared = perishable_tokens(ingredients) & set(basket)
            score += waste_weight * min(len(shared), _REUSE_CAP)
        return score

    return scorer


class WeeklyPlanner:
    """
    Orchestration pipeline to generate a full 7-day meal plan.

    Supports pinned anchor slots (seeded planning, M2): a pinned (day,
    meal_idx) slot always receives its anchor recipe; only free slots are
    filled from the action space.
    """

    def __init__(self, env: WeeklyMealPlanEnv):
        self.env = env

    def generate_full_plan(
        self,
        user_query: str = None,
        pinned: Optional[Dict[Tuple[int, int], Dict[str, Any]]] = None,
        scorer: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run the 21-step (7 days × 3 meals) planning loop.

        Args:
            user_query: Optional user query to guide reward evaluation.
            pinned: {(day, meal_idx): recipe-action dict} anchor slots.
            scorer: optional preference scorer (build_preference_scorer) —
                    argmax selection with random tiebreak among equals.
                    Without it, selection is uniformly random.

        Returns:
            The 21 generated entries with day, meal type, recipe, and reward.
        """
        pinned = pinned or {}
        # Scorers written before the food-waste axis take (candidate, titles);
        # the pre-M8 one took a third flat-basket argument; ours takes a
        # fourth day. Measured once here, not guessed per call with a
        # try/except that would also swallow a scorer's own TypeErrors — and
        # each arity is called with exactly what it accepts, so adding the
        # day did not silently break a 3-argument scorer.
        import inspect
        try:
            arity = len(inspect.signature(scorer).parameters) if scorer else 0
        except (TypeError, ValueError):  # builtins, C callables, odd wrappers
            arity = 2
        state = self.env.reset(user_query=user_query)
        done = False
        chosen_titles: List[str] = []
        # Perishables already committed to the plan, with the day each landed
        # on — the food-waste scorer's working set. Days are what let it tell
        # thrift (Monday's cabbage again on Wednesday) from monotony (again
        # at Monday's dinner); see build_preference_scorer.
        basket = IngredientBasket()

        def score_candidate(candidate: Dict[str, Any]) -> float:
            if scorer is None:
                return 0.0
            if arity >= 4:
                return scorer(candidate, chosen_titles, basket, state["day"])
            if arity == 3:
                return scorer(candidate, chosen_titles, basket.tokens())
            return scorer(candidate, chosen_titles)

        while not done:
            slot_key = (state["day"], state["meal_idx"])
            if slot_key in pinned:
                # User-requested anchor — bypass candidate selection entirely.
                chosen_recipe = {**pinned[slot_key], "pinned": True}
                logger.info(
                    "Day %d %s pinned to %r",
                    state["day"], state["meal_type"], chosen_recipe.get("recipe_title"),
                )
            else:
                candidates = self.env.action_space.get_candidate_actions(
                    state["meal_type"], state
                )
                if not candidates:
                    # No recipes for this course — fail loudly rather than
                    # produce a plan with holes. Typed so the service can turn
                    # it into a sentence instead of a 500. (Constraint
                    # relaxation is a planned improvement.)
                    raise PlanGenerationError(state["meal_type"], state["day"])
                # Hard constraints prune the pool BEFORE the pick (M6):
                # e.g. meat candidates leave once the weekly limit is spent.
                # Prunes/relaxations are recorded on env.selection_events at
                # decision time (M7 explainability).
                candidates = apply_hard_constraints(
                    candidates, self.env.tracker,
                    events=self.env.selection_events,
                    slot={"day": state["day"], "meal_type": state["meal_type"]},
                )

                # Preference score + soft constraint score (calorie budget)
                # rank the pool; random tiebreak among equals keeps variety.
                # Without a scorer and without nutrition data every score is
                # 0.0 and selection stays uniformly random, as before.
                slots_remaining = TOTAL_SLOTS - len(self.env.plan)
                scored = [
                    (
                        score_candidate(c)
                        + constraint_score(c, self.env.tracker, slots_remaining),
                        c,
                    )
                    for c in candidates
                ]
                best_score = max(s for s, _ in scored)
                top = [c for s, c in scored if s == best_score]
                chosen_recipe = random.choice(top)

            chosen_titles.append(str(chosen_recipe.get("recipe_title", "")))
            # Recorded against the day it is eaten, before the environment
            # advances the clock — pinned slots included, since a member's
            # anchor puts food in the basket like any other meal.
            basket.add(chosen_recipe.get("recipe_ingredients", ""), state["day"])
            # Advance the environment (updates tracker, computes reward).
            state, reward, done, info = self.env.step(chosen_recipe)

        return self.env.plan
