"""The 21-slot weekly planning loop, and the scorer that fills it.

``WeeklyPlanner.generate_full_plan`` walks 7 days x 3 meals, asking the action
space for candidates, pruning them against hard constraints, and picking the
best-scoring one. Fully LLM-free, which is the constraint everything here
answers to: a preference that never becomes a number in
``build_preference_scorer`` is a preference this planner cannot honour at all,
because there is no grader downstream to compensate.

Three things live here beyond the loop itself:

- ``IngredientBasket`` — what the week has already bought, and the day each
  item landed on. Days are what let the scorer tell reuse (Monday's cabbage
  again on Wednesday) from monotony (again at Monday's dinner). No shelf life
  is modelled anywhere: nothing in the state records when an ingredient was
  bought or when it spoils, so every gap here is about the week being worth
  eating, never about freshness.
- ``nameable_phrases`` — how an ingredients blob becomes ingredient *names*.
  Shared with ``explainability`` (which words the chips) and, through the
  basket, with ``action_adapter`` (which searches for them), so that "an
  ingredient" means one thing across the three.
- ``build_preference_scorer`` — favourites, likes, the member's stated pantry,
  variety, the food-waste axis, and the cost of a repeat.
"""

import logging
import random
import re
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


# --- Naming an ingredient ---------------------------------------------------
#
# Naming is held to a stricter standard than ranking, and the unit of a
# *name* is the ingredient phrase, not the token.
#
# `perishable_tokens` splits on whitespace, which is right for scoring — two
# meals sharing "green" really are a little more alike, and averaging absorbs
# the noise. It is wrong anywhere a member reads the result, and equally wrong
# as a search term. Tokenising "self raising flour" drops the staple and leaves
# the modifiers, so the first version of the reuse chip told a member she was
# reusing *"Thursday's self"* and *"Thursday's raising"*, from one bag of
# flour. Others seen on real plans: "Monday's green" (green beans), "Wednesday's
# brown" (brown rice), "Monday's leaf" (bay leaf).
#
# So a share is *found* by token — that part works — and then *named* by the
# phrase the token came from. Several tokens out of one phrase collapse into
# one item, which is what makes "self" and "raising" a single "self raising
# flour" rather than two ingredients.
#
# Anything the phrase rules cannot name is not used. Under-reporting is the
# safe direction, the same posture the pantry matcher takes: a missed match
# says less than it could, it never invents a saving. Both callers depend on
# that — `explainability` will not show a member a chip it cannot word, and
# `action_adapter` will not spend an HTTP round-trip searching for "self".

# A token must anchor on one of these to be worth naming at all — a phrase
# of pure adjectives ("whole, raw") names nothing.
_UNNAMEABLE: frozenset = frozenset({
    # colours
    "green", "brown", "white", "black", "yellow", "purple", "golden", "dark",
    "light", "wholemeal", "wholegrain",
    # generic categories
    "vegetable", "vegetables", "veggie", "veggies", "fruit", "fruits",
    "leaf", "leaves", "dressing", "seasoning", "spice", "spices", "herb",
    "herbs", "mix", "mixed", "blend", "topping", "filling", "garnish",
    "flakes", "powder", "paste", "puree", "concentrate", "granules",
    # preparation / provenance adjectives
    "raw", "whole", "cooked", "frozen", "canned", "tinned", "smoked",
    "roasted", "toasted", "boneless", "skinless", "seedless", "unsalted",
    "salted", "standard", "plain", "instant", "ready", "free", "style",
    "organic", "natural", "baby", "wild", "sweet", "sour", "spicy",
    # condiment modifiers whose noun is a staple and gets dropped
    "balsamic", "wine", "cider", "malt", "sesame", "sunflower", "rapeseed",
})

# Quantity words to drop from a displayed phrase. `_UNITS` already covers
# measurements; these are the counting words that survive it ("half a
# cabbage" should read "cabbage").
_QUANTITY_WORDS: frozenset = frozenset({
    "half", "quarter", "third", "few", "some", "handful", "pinch", "dash",
    "splash", "piece", "pieces", "packet", "packets", "pouch", "pouches",
    "tin", "tins", "can", "cans", "jar", "jars", "bunch", "sprig", "sprigs",
    "stick", "sticks", "tub", "block", "each", "about", "approx",
})

# Longer than this and the "phrase" is a run-together blob rather than an
# ingredient name ("brown sugar light brown cane sugar"). Not named.
_MAX_PHRASE_WORDS = 4


def _unnameable_stems() -> frozenset:
    return frozenset(_singular(w) for w in _UNNAMEABLE)


def nameable_phrases(ingredients_text: Any, exclude_stems: Any = ()) -> List[Tuple[str, set]]:
    """``[(display_name, {stems})]`` for the phrases in a blob worth naming.

    A phrase is dropped when it names a staple — "self raising flour" *is*
    flour, "brown sugar" *is* sugar, "macadamia nut oil" *is* oil — because
    sharing a staple has never been a saving, and it is exactly those phrases
    whose modifiers survive tokenising and end up impersonating ingredients.

    ``exclude_stems`` drops a phrase **whole** rather than filtering it down
    to its other words. Callers pass the member's own pantry stems: "eggplant
    aubergine" anchored on "aubergine" would otherwise be shown to a member
    who told us about their eggplants, crediting the plan for their fridge —
    and would send a second search for something already being searched for.
    """
    exclude = set(exclude_stems or ())
    unnameable = _unnameable_stems()
    out: List[Tuple[str, set]] = []
    for part in re.split(r"[\n,;•]+", str(ingredients_text or "")):
        words = re.sub(r"[^a-z\s]", " ", part.lower()).split()
        if not words or len(words) > _MAX_PHRASE_WORDS:
            continue
        if any(w in _PANTRY_STAPLES for w in words):
            continue
        # Measurements and counting words are dropped BEFORE the stems are
        # taken, not only from the displayed name. "half" clears the length
        # filter in `perishable_tokens`, so leaving it in made "half a
        # cabbage" and "half a pumpkin" share an ingredient called "half" —
        # and the later meal was then credited with reusing the cabbage.
        content = [w for w in words if w not in _UNITS and w not in _QUANTITY_WORDS]
        all_stems = {_singular(t) for t in perishable_tokens(" ".join(content))}
        if all_stems & exclude:
            continue
        stems = {s for s in all_stems if s not in unnameable}
        if not stems:
            continue
        display = " ".join(w for w in content if len(w) > 2)
        if display:
            out.append((display, stems))
    return out


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

# What a second serving of the same recipe costs against an equally good new
# dish (`action_adapter`'s repeat policy decides whether one is legal at all;
# this decides whether it is chosen).
#
# **Zero, deliberately.** It shipped at 1.0, justified as "enough that a repeat
# does not win a coin flip". That reasoning assumed candidates are spread over
# a range of scores. They are not: a member with no favourites and no liked
# *ingredients* (two liked cuisines are filtered out of `food_likes` by
# `split_cuisines` before they reach here) leaves almost every candidate
# scoring exactly 0.0 — so −1.0 was not a tiebreak, it was a veto. A legal
# repeat lost to any fresh candidate that existed, and on a real week with a
# healthy pool the cooldown could never fire at all. Observed on a live plan:
# seven distinct breakfasts, `planned_repeats: 0`.
#
# At zero a legal repeat joins the tie pool and takes a proportional share of
# the slots — which is the whole ask, since nobody eats seven different
# breakfasts. It is not unguarded: the cooldown decides *whether* a repeat is
# legal and the cap decides *how often*, and those two were always the real
# controls. This constant stays as the dial to turn if repeats become too
# frequent, and stays applied INSTEAD of the ingredient axis rather than on
# top of it — see below.
_REPEAT_PENALTY = 0.0


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

    __slots__ = ("_days", "_raw", "_texts", "_item_cache", "_named")

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
        # stem -> (earliest day it appeared, how that day named it). Scoring
        # works fine on tokens; sourcing and chips need a name a human wrote,
        # so the basket keeps both (see `nameable_phrases`).
        self._named: Dict[str, Tuple[int, str]] = {}

    def add(self, ingredients_text: Any, day: int) -> None:
        """Record one committed meal's countable ingredients against its day."""
        for token in perishable_tokens(ingredients_text):
            # Keyed by singular stem, so "tomatoes" on Monday and "tomato" on
            # Tuesday are the same ingredient. Storing them apart let a member
            # be served tomatoes two days running with no penalty at all —
            # the same asymmetry the pantry matcher was fixed for.
            self._days.setdefault(_singular(token), []).append(int(day))
            self._raw.add(token)
        for display, stems in nameable_phrases(ingredients_text):
            for stem in stems:
                seen = self._named.get(stem)
                if seen is None or int(day) < seen[0]:
                    self._named[stem] = (int(day), display)
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

    def reusable_items(
        self, current_day: int, limit: int = 3, exclude_stems: Any = ()
    ) -> List[str]:
        """Ingredient names the week could still get one more meal out of.

        The same two rules the reuse bonus scores by — a gap of at least
        `_MIN_REUSE_GAP_DAYS`, and at most `_MAX_REWARDED_USES` meals — so a
        caller that turns these into recipe searches asks for exactly what the
        scorer would then reward. Sourcing something the scorer goes on to
        penalise would buy latency and a worse week.

        Ordered by fewest meals so far, then by the most recent eligible day:
        an ingredient used once has the most headroom left, and clustering
        reuse beats scattering single hits across the week. Nothing here is a
        freshness judgement — no purchase date or shelf life is recorded
        anywhere, so "still worth another meal" means the shopping list, not
        the fridge.
        """
        exclude = set(exclude_stems or ())
        ranked: List[Tuple[int, int, str]] = []
        for stem, (_first_day, display) in self._named.items():
            if stem in exclude:
                continue
            days = self._days.get(stem) or []
            if not days or len(days) >= _MAX_REWARDED_USES:
                continue
            last = max(days)
            if int(current_day) - last < _MIN_REUSE_GAP_DAYS:
                continue
            ranked.append((len(days), -last, display))
        ranked.sort()
        out: List[str] = []
        for _uses, _recency, display in ranked:
            if len(out) >= max(int(limit), 0):
                break
            if display not in out:
                out.append(display)
        return out

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

    **Repeats (M9).** ``action_adapter`` decides whether a recipe may come back
    at all — breakfast only, spaced, capped — and marks a candidate it allows
    back with ``repeat_of_day``. This scorer decides whether one is actually
    chosen, and treats it as a case of its own rather than as an ordinary
    candidate that happens to look familiar:

    - it is not charged the variety penalty for its OWN earlier title (that
      penalty scales with title length and would always beat the cooldown),
      but pays it in full for resembling any other dish;
    - it sits out the ingredient axis, because it shares every ingredient with
      its own earlier serving, and pays the flat ``_REPEAT_PENALTY`` instead.

    With that penalty at zero, a legal repeat competes on equal terms with a
    new dish and takes a proportional share of the tie pool. Whether it may
    come back at all, and how often, are the cooldown's and the cap's job —
    not this scorer's. A favourite (+5) still wins its slot outright, and at
    `strict` a fresh candidate that genuinely reuses an ingredient outscores a
    repeat, because only the fresh one collects the reuse bonus.
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
        #
        # A sanctioned repeat (`action_adapter` marks one with `repeat_of_day`
        # once its slot cooldown has elapsed) is exempt from its OWN earlier
        # title, and from nothing else. The cooldown has already decided
        # whether a second serving is allowed and how far apart; charging the
        # candidate again for resembling itself would leave the two rules
        # fighting, and the exact-match penalty is large enough that the
        # cooldown would always lose. An overlap with any *other* dish still
        # counts in full, so a repeat that also looks like a third meal is
        # still penalised for that.
        own_title = str(candidate.get("recipe_title", "")).lower()
        sanctioned_repeat = bool(candidate.get("repeat_of_day"))
        title_tokens = {t for t in own_title.split() if len(t) > 3}
        for chosen in chosen_titles:
            if sanctioned_repeat and chosen.lower() == own_title:
                continue
            overlap = title_tokens & {t for t in chosen.lower().split() if len(t) > 3}
            if overlap:
                score -= 2.0 * len(overlap)

        # Food waste + monotony: what this candidate shares with the plan so
        # far, judged by when it was last eaten. Both halves capped so one
        # candidate cannot ride a busy ingredient list in either direction.
        #
        # A sanctioned repeat sits out this axis entirely and pays the flat
        # repeat penalty instead: it shares its ingredients with itself, so
        # neither the reward nor the penalty would be saying anything about
        # the week. Its spacing is the cooldown's job, and was settled before
        # the candidate ever reached the scorer.
        #
        # This branch still matters with `_REPEAT_PENALTY` at zero — sitting
        # out the axis is the load-bearing half. Without it, a repeat collects
        # the full reuse bonus for overlapping with its own earlier serving,
        # and at `strict` repeating becomes the cheapest possible way to
        # score.
        if sanctioned_repeat:
            score -= _REPEAT_PENALTY
        elif isinstance(basket, IngredientBasket):
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
        # The day whose pool has already been offered a derived pantry, so the
        # offer is made once per day rather than once per slot.
        sourced_day: Optional[int] = None

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
                # Sourcing half of the cross-day reuse axis. The scorer can
                # only reorder the pool it is handed, and a day's pool is
                # fetched knowing nothing about what the week has already
                # bought — so before M9 reuse happened only when a shared
                # ingredient turned up by luck. Offer the action space the
                # ingredients this basket says are still worth another meal;
                # it decides whether the member's food-waste setting justifies
                # the extra round-trips, and records the decision either way.
                #
                # Duck-typed: the fakes in the tests, and any action space that
                # predates this, simply never receive the offer.
                if state["day"] != sourced_day:
                    sourced_day = state["day"]
                    offer = getattr(
                        self.env.action_space, "offer_derived_pantry", None
                    )
                    if callable(offer):
                        offer(basket.reusable_items(state["day"]), state["day"])
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
