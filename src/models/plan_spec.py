"""PlanSpec — how many days, which meals, and how many plates each.

This is the shape `DYNAMIC_MEALS_PLAN.md` §4.1 specifies, in the vocabulary the
rest of FoodChat already uses:

    PlanSpec = {
      "num_days": 1,
      "meals": ["breakfast", "lunch", "dinner"],
      "plates": {"lunch": ["main", "side"]},
    }

The core model for this landed in Phase 1 (`Meal.plates`, `DayPlan`,
`MealPlan.days`, `MealCourse.role`). What blocked Phase 2 was named in the plan
itself: *"side/dessert/drink plates pull from role-scoped RW queries … this
needs a `role` hint on the RW candidate fetch — check whether
`candidates_client` can filter by dish_type/course; if not, this is the one
RW-side dependency."*

That dependency is now met. RecipeWrangler's `/api/v2/tools/plan_meals` takes a
`course_types` override **per slot**, so a role maps directly onto a query the
service can answer. `ROLE_COURSE_TYPES` below is that mapping, and it is the
whole reason this module can exist.

**Roles, not course types.** FoodChat reasons in roles — a plate is a `main`, a
`side`, a `dessert`, a `drink` — because that is what the nutrition budget is
split by and what `MealCourse.role` records. RecipeWrangler reasons in course
types, because that is what the corpus is annotated with. Translating at the
boundary keeps each side speaking its own language; a `PlanSpec` carrying
`"main-dish"` would put RecipeWrangler's taxonomy into FoodChat's persisted
plans forever.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# Roles a plate can have. `MealCourse.role` stores exactly these.
#
# `salad` and `soup` are separate from the generic `side` because users ask for
# them by name — "a main and a salad" is a different request from "a main and a
# side", and answering the first with a pastry (tagged `side` in the corpus) is
# answering a question nobody asked.
ROLES: tuple[str, ...] = ("main", "side", "salad", "soup", "dessert", "drink")

# Role → the RecipeWrangler course types that can satisfy it.
#
# Several course types per role on purpose: a side can be a salad, a soup or a
# side, and asking for only one of those would empty the plate on a corpus that
# annotated it as another. RecipeWrangler ORs the list, which is the right
# semantics here — unlike a *multi-course meal*, where each course needs its own
# request because the courses must all be present.
ROLE_COURSE_TYPES: dict[str, tuple[str, ...]] = {
    "main": ("main-dish",),
    # Broad on purpose: a generic side can be any of these, and asking for only
    # one would empty the plate on a corpus that annotated it as another.
    "side": ("side", "salad", "soup"),
    # Narrow on purpose: someone who said "salad" meant salad.
    "salad": ("salad",),
    "soup": ("soup",),
    "dessert": ("desserts",),
    "drink": ("beverages",),
}

# How a meal's calorie budget divides across its plates — the honest-nutrition
# rule from §4.2: "main + side ≈ one meal, not two".
#
# A side that carried a full meal's allowance would let a two-plate lunch quietly
# double the day's target, which is the failure the rule exists to prevent.
ROLE_KCAL_WEIGHT: dict[str, float] = {
    "main": 0.70,
    "side": 0.30,
    "salad": 0.25,
    "soup": 0.35,
    "dessert": 0.25,
    "drink": 0.10,
}

# Slots that are not a meal on their own. A day made only of these is not a
# day's eating, however sensible each entry is by itself.
_NON_MEAL_SLOTS: frozenset[str] = frozenset({"snack", "dessert", "side", "drink"})

# Above this many dessert plates in one day, the plan stops being a meal plan.
# Not a nutrition standard — a threshold at which an assistant should say
# something rather than silently comply.
_MAX_DESSERTS_PER_DAY = 1

# Slots RecipeWrangler can fill. Mirrored so a spec can be validated without a
# network call; `PlanClient.planning_options()` reconciles against the live list.
KNOWN_SLOTS: tuple[str, ...] = (
    "breakfast", "brunch", "lunch", "dinner", "snack", "dessert", "side", "drink",
)

DEFAULT_MEALS: tuple[str, ...] = ("breakfast", "lunch", "dinner")

# Product-level bounds. RecipeWrangler enforces its own (count 1-20, days 1-14);
# these are tighter because a plan is rendered as cards and read aloud, and a
# forty-plate day is neither.
MAX_MEALS_PER_DAY = 6
MAX_PLATES_PER_MEAL = 3
MAX_DAYS = 7


@dataclass(frozen=True)
class PlanSpec:
    """What to generate. Defaults to exactly today's plan shape."""

    num_days: int = 1
    meals: tuple[str, ...] = DEFAULT_MEALS
    # slot -> ordered roles. A slot absent from this map is a single `main`.
    plates: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def is_default(self) -> bool:
        """The classic one-day, three-meal, single-plate shape.

        Used to keep the existing LLM-graded path, which composes one recipe per
        slot and cannot express a second plate.
        """
        return (
            self.num_days == 1
            and self.meals == DEFAULT_MEALS
            and not any(len(r) > 1 or r not in ((), ("main",)) for r in self.plates.values())
        )

    def roles_for(self, slot: str) -> tuple[str, ...]:
        """The plates a slot is made of. Every meal has at least a main."""
        return self.plates.get(slot) or ("main",)

    @property
    def total_plates(self) -> int:
        return sum(len(self.roles_for(slot)) for slot in self.meals) * self.num_days

    def describe(self) -> str:
        """One sentence a chat agent can say back, to confirm the shape."""
        parts = []
        for slot in self.meals:
            roles = self.roles_for(slot)
            if roles == ("main",):
                parts.append(slot)
            else:
                parts.append(f"{slot}: {' + '.join(roles)}")
        day_word = "1 day" if self.num_days == 1 else f"{self.num_days} days"
        return f"{day_word} — " + "; ".join(parts)

    def kcal_split(self, slot: str) -> dict[str, float]:
        """A meal's calorie budget divided across its plates.

        Normalised so the weights sum to 1 whatever combination of roles a meal
        has: a main-plus-dessert meal must still add up to one meal, not to
        0.95 of one or 1.4 of one.
        """
        roles = self.roles_for(slot)
        weights = {role: ROLE_KCAL_WEIGHT.get(role, 0.5) for role in roles}
        total = sum(weights.values()) or 1.0
        return {role: weight / total for role, weight in weights.items()}

    def concerns(self) -> list[str]:
        """Reasons an assistant should push back before building this plan.

        Returned rather than raised. The member asked for something and is
        entitled to an answer; what they are not entitled to is a nutrition
        assistant that builds three desserts and calls it a day's eating
        without a word. So the shape is still buildable, and the caller is
        given the sentences to say first.

        Deliberately narrow. These are not dietary guidelines — the corpus's
        Nutri-Scores and the member's own targets do that work, per plate. This
        catches the handful of *shapes* that are wrong before a single recipe is
        chosen, which is the only thing a spec can know.

        Empty means nothing structurally alarming, not "this plan is healthy".
        """
        notes: list[str] = []

        dessert_plates = sum(
            1
            for slot in self.meals
            for role in self.roles_for(slot)
            if role == "dessert"
        ) + sum(1 for slot in self.meals if slot == "dessert")

        if dessert_plates > _MAX_DESSERTS_PER_DAY:
            notes.append(
                f"This plan is {dessert_plates} desserts in a day. I can build "
                "it, but it is not a balanced day's eating — shall I put them "
                "alongside proper meals instead?"
            )

        # A day made only of snacks, sides, desserts and drinks is not a day's
        # eating. Checked on the *slots*, not the roles: `from_spec` anchors
        # every meal with a main, so a role-based check here could never fire —
        # the gap is someone asking for "just snacks and a dessert", where each
        # meal is fine and the day is not.
        if self.meals and all(slot in _NON_MEAL_SLOTS for slot in self.meals):
            notes.append(
                "That is only " + ", ".join(self.meals) + " — no actual meal in "
                "the day. I can plan it, but shall I add a lunch or a dinner?"
            )

        if len(self.meals) == 1 and self.num_days > 1:
            notes.append(
                f"This plans only {self.meals[0]} for {self.num_days} days — "
                "the other meals are left to you. That is fine if it is what "
                "you meant."
            )

        return notes

    def to_request_slots(self) -> list[dict[str, Any]]:
        """The `slots` payload `/api/v2/tools/plan_meals` expects.

        **One entry per plate**, not one per meal. RecipeWrangler treats a
        slot's `course_types` as a `terms` filter — an OR — so a single entry
        listing main-dish and salad returns one recipe that is either, and a
        two-plate lunch would come back with one plate missing and nothing to
        say so. Asking for a main *and* a side means asking twice.

        Entries keep the same slot name, which the service echoes back, so the
        plates regroup into one `Meal` on the way home.
        """
        out: list[dict[str, Any]] = []
        for slot in self.meals:
            for role in self.roles_for(slot):
                out.append(
                    {
                        "slot": slot,
                        "count": 1,
                        "course_types": list(ROLE_COURSE_TYPES.get(role, ())),
                    }
                )
        return out

    def role_sequence(self) -> list[tuple[str, str]]:
        """`(slot, role)` in the same order as `to_request_slots`.

        The response carries the slot but not the role — roles are FoodChat's
        vocabulary, not RecipeWrangler's — so the two are zipped back together
        by position. Both are generated from `self.meals`, so they cannot drift.
        """
        return [(slot, role) for slot in self.meals for role in self.roles_for(slot)]

    def to_dict(self) -> dict[str, Any]:
        """The JSON-safe form `from_spec` reads back.

        A `PlanSpec` travels inside the session profile snapshot (so a plan
        shape survives a clarification round-trip), and that snapshot is
        `json.dumps`-ed onto the session row. Carrying the dataclass itself
        raised "Object of type PlanSpec is not JSON serializable" from
        whichever turn happened to ask a clarifying question — so the
        serializable form lives here, next to the parser, rather than being
        open-coded by each caller that needs to persist one.
        """
        return {
            "num_days": self.num_days,
            "meals": list(self.meals),
            "plates": {slot: list(roles) for slot, roles in self.plates.items()},
        }

    @classmethod
    def coerce(cls, raw: Any) -> "PlanSpec":
        """A `PlanSpec` from either a stored dict or a live instance.

        Readers accept both: a spec that rode through persistence arrives as
        a dict, while one taken straight from in-memory state is already an
        object (and, after a failed write, an in-process session can still
        hold one).
        """
        if isinstance(raw, cls):
            return raw
        return cls.from_spec(raw)

    @classmethod
    def default(cls) -> "PlanSpec":
        return cls()

    @classmethod
    def from_spec(
        cls, raw: Any, *, known_slots: Iterable[str] = KNOWN_SLOTS
    ) -> "PlanSpec":
        """Build from loose input — an LLM extraction, or a stored request.

        Everything is treated as untrusted. An unknown slot is dropped rather
        than forwarded: RecipeWrangler would accept `elevenses`, match nothing,
        and return an empty meal, which reads to the user as a broken planner
        rather than as a word nobody understood.

        Falls back to the default shape when nothing usable survives. A plan
        with the wrong number of meals can be corrected in the next turn; no
        plan at all cannot.
        """
        if not isinstance(raw, dict):
            return cls()

        slot_vocab = {s.lower() for s in known_slots}

        try:
            num_days = max(1, min(int(raw.get("num_days") or 1), MAX_DAYS))
        except (TypeError, ValueError):
            num_days = 1

        meals: list[str] = []
        for value in raw.get("meals") or ():
            slot = str(value or "").strip().lower()
            if slot in slot_vocab and slot not in meals:
                meals.append(slot)
            if len(meals) >= MAX_MEALS_PER_DAY:
                break

        plates: dict[str, tuple[str, ...]] = {}
        for slot, roles in (raw.get("plates") or {}).items():
            key = str(slot or "").strip().lower()
            if key not in slot_vocab:
                continue
            cleaned: list[str] = []
            for role in roles or ():
                name = str(role or "").strip().lower()
                if name in ROLES and name not in cleaned:
                    cleaned.append(name)
                if len(cleaned) >= MAX_PLATES_PER_MEAL:
                    break
            # A meal is always anchored by a main; a spec asking for only a side
            # describes a side dish, not a meal.
            if cleaned and "main" not in cleaned:
                cleaned.insert(0, "main")
            if cleaned and cleaned != ["main"]:
                plates[key] = tuple(cleaned)

        if not meals:
            # Plates may name a slot the meals list forgot — "dinner should have
            # a side" implies dinner is in the plan.
            meals = [s for s in DEFAULT_MEALS] if not plates else sorted(plates)

        return cls(num_days=num_days, meals=tuple(meals), plates=plates)
