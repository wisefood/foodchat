"""
What this plan is trying to do, decided before a single recipe is fetched.

The pipeline has always gone straight from a message to a search: reconcile the
query, splat some filters, grade what comes back. Nothing wrote down what the
plan was *for*, which is why nothing could check whether it succeeded — the
ledger reported the request back to the member and called it a result.

A brief is that missing artefact. It says, in one place:

    what must hold          allergens, a stated diet
    what should hold        facets, claim tags, a time budget, a score floor
    what the numbers are    a calorie target, split across the day's plates
    what may be given up    the relaxation order, worst signal first
    what to check after     the names the verifier will report on

Two properties make it worth having rather than passing the same values around
as loose keyword arguments:

**It is built deterministically.** `PlanBrief.build()` reads the profile, the
standing state and the plan spec, and nothing else. No model call, no guessing.
A reasoning step can then adjust it (see `PlanStrategist`), but only within
`with_strategy()`, which validates every value it is handed — so a strategist
that invents a mood cannot empty the search, and a strategist that fails leaves
the deterministic brief exactly as it was.

**It defines the contract with the verifier.** `to_requested()` produces the
dict `plan_verifier.verify` consumes, so "what we asked for" and "what we check"
cannot drift apart into two hand-maintained lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

# The order preferences are given up when a slot cannot be filled.
#
# Mirrors RecipeWrangler's own ladder, because a client that expects a
# different order than the server applies is a client whose explanation of what
# happened is wrong. Weakest signal first: a mood is a mood, a cuisine is a
# stated taste, a time limit is something a person will actually notice.
DEFAULT_RELAXATION_ORDER: tuple[str, ...] = (
    "tags",
    "moods",
    "flavor_profiles",
    "food_groups",
    "cuisines",
    "max_minutes",
)

# Never given up, at any point, for any reason. An allergen exclusion dropped to
# fill a slot is a safety failure, not a degraded result.
NEVER_RELAXED: frozenset[str] = frozenset({
    "allergens", "diet", "exclude_ingredients", "exclude_recipe_ids",
    "include_ingredients", "min_nutri_score", "sources", "course_types",
})

# Fallback when a member has a goal but no measured energy requirement.
#
# Stated as a default and reported as one: the weekly tracker's habit of
# rendering this number with `source: "calorie target"` for someone who never
# set one is exactly the confusion this comment exists to prevent. FoodChat has
# no age, sex, weight or activity data, so anything more specific would be
# arithmetic performed on nothing.
DEFAULT_DAILY_KCAL = 2000.0


@dataclass(frozen=True)
class PlanBrief:
    """The plan for the plan."""

    # ── hard: never relaxed, always verified ────────────────────────────
    allergens: tuple[str, ...] = ()
    diet: tuple[str, ...] = ()

    # ── soft: shape the search, may be relaxed, verified where possible ─
    cuisines: tuple[str, ...] = ()
    moods: tuple[str, ...] = ()
    flavor_profiles: tuple[str, ...] = ()
    food_groups: tuple[str, ...] = ()
    claim_tags: tuple[str, ...] = ()

    # ── numeric ─────────────────────────────────────────────────────────
    kcal_target: Optional[float] = None
    # A figure the day may be REPORTED against when the member set no target
    # of their own — and never one the plan may be ranked, filtered or
    # apportioned by. Separate from `kcal_target` precisely so that cannot
    # happen by accident: `kcal_by_slot` and the plate critic read the one
    # above, which stays None until somebody chooses a number.
    kcal_reference: Optional[float] = None
    # Where `kcal_reference` came from, in words the member can read. Empty
    # when there is no reference. A number on a plan with no stated origin is
    # how the weekly meat limit came to apologise for a rule nobody set.
    kcal_reference_basis: str = ""
    # The day's budget divided across a meal's plates, per slot. Resurrects
    # `PlanSpec.kcal_split`, which has been correct and unused since it was
    # written: a main-plus-dessert dinner is one meal's calories split two
    # ways, not two meals' worth.
    kcal_by_slot: dict[str, dict[str, float]] = field(default_factory=dict)
    max_minutes: Optional[int] = None
    min_nutri_score: Optional[str] = None

    # ── context the search uses and the verifier checks ─────────────────
    pantry: tuple[str, ...] = ()
    anchors: dict[str, str] = field(default_factory=dict)
    exclude_recipe_ids: tuple[str, ...] = ()

    # ── policy ──────────────────────────────────────────────────────────
    relaxation_order: tuple[str, ...] = DEFAULT_RELAXATION_ORDER
    # Why this brief looks the way it does. Written by the strategist when one
    # ran, empty when the deterministic build stands on its own — and empty is
    # honest rather than a placeholder sentence nobody wrote.
    rationale: str = ""

    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def build(cls, profile: dict, state=None, spec=None) -> "PlanBrief":
        """The brief the request implies, with no model in the loop.

        Everything here already existed and was assembled ad hoc at each fetch
        site, which is how three of them ended up passing different subsets.
        """
        from services.candidates_client import (
            effective_diet,
            screening_allergens,
        )
        from services import reference_intake
        from services.intent_facets import claim_tags_for, effective_facets

        profile = profile or {}
        params = profile.get("plan_parameters") or {}

        facets = effective_facets(profile, state)
        claims = claim_tags_for(
            params.get("goal"),
            list(profile.get("_claim_tags") or [])
            + list(getattr(state, "claim_tags", ()) or ()),
            params.get("difficulty"),
        )

        kcal_target = _kcal_target(profile, params)
        reference = reference_intake.reference_for(profile)
        return cls(
            allergens=tuple(screening_allergens(profile)),
            diet=tuple(effective_diet(profile)),
            cuisines=tuple(facets.get("cuisines") or ()),
            moods=tuple(facets.get("moods") or ()),
            flavor_profiles=tuple(facets.get("flavor_profiles") or ()),
            food_groups=tuple(facets.get("food_groups") or ()),
            claim_tags=tuple(claims),
            kcal_target=kcal_target,
            # Reporting only — never `kcal_by_slot`, which apportions a budget
            # and must therefore be the member's own number or nothing.
            kcal_reference=reference.kcal if reference else None,
            kcal_reference_basis=reference.basis if reference else "",
            kcal_by_slot=_kcal_by_slot(spec, kcal_target),
            max_minutes=_as_int(params.get("cooking_time")),
            min_nutri_score=_nutri_floor(profile, params),
            pantry=tuple(profile.get("_pantry") or getattr(state, "pantry", ()) or ()),
            anchors=dict(getattr(state, "anchors", {}) or {}),
            exclude_recipe_ids=tuple(profile.get("_excluded_recipe_ids") or ()),
        )

    def with_strategy(self, proposal: dict, vocabularies: Optional[dict] = None) -> "PlanBrief":
        """Apply a strategist's proposal, keeping only what is real.

        Every value is checked against the LIVE vocabulary before it lands.
        This is not defensive tidiness: RecipeWrangler ANDs facet values and
        never relaxes an unlisted one to nothing, so an invented mood does not
        soften the search — it empties it, and the member is told no meals
        exist. A strategist is allowed to be wrong; it is not allowed to be
        wrong in a way that produces an empty plan and no explanation.

        Hard constraints are not adjustable here. A reasoning step may decide
        how to search; it may not decide to drop an allergen.
        """
        vocab = vocabularies or {}
        updates: dict = {}

        for family in ("cuisines", "moods", "flavor_profiles", "food_groups"):
            proposed = proposal.get(family)
            if not proposed:
                continue
            allowed = {str(v).strip().lower() for v in (vocab.get(family) or ())}
            kept = [
                str(v).strip().lower() for v in proposed
                if str(v).strip().lower() in allowed
            ] if allowed else []
            merged = list(getattr(self, family))
            for value in kept:
                if value not in merged:
                    merged.append(value)
            updates[family] = tuple(merged)

        claims = proposal.get("claim_tags")
        if claims:
            from services.intent_facets import CLAIM_TAG_ALIASES, GOAL_CLAIM_TAGS

            # The corpus's claim vocabulary, not an open field: a tag nothing
            # carries is the same empty-search problem as an invented mood.
            known = set(CLAIM_TAG_ALIASES.values()) | {
                tag for tags in GOAL_CLAIM_TAGS.values() for tag in tags
            }
            merged = list(self.claim_tags)
            for claim in claims:
                slug = CLAIM_TAG_ALIASES.get(
                    str(claim).strip().lower(), str(claim).strip().lower()
                )
                if slug in known and slug not in merged:
                    merged.append(slug)
            updates["claim_tags"] = tuple(merged)

        order = proposal.get("relaxation_order")
        if order:
            # A strategist may reorder what gets given up first, and may not
            # add anything to the list — least of all something on NEVER_RELAXED.
            allowed = set(DEFAULT_RELAXATION_ORDER)
            reordered = [
                str(step).strip().lower() for step in order
                if str(step).strip().lower() in allowed
            ]
            for step in DEFAULT_RELAXATION_ORDER:
                if step not in reordered:
                    reordered.append(step)
            updates["relaxation_order"] = tuple(reordered)

        kcal = proposal.get("kcal_target")
        if kcal:
            try:
                value = float(kcal)
            except (TypeError, ValueError):
                value = 0.0
            # A plausible day. A strategist proposing 400 or 9000 has made an
            # arithmetic error, and a nutrition assistant should not build a
            # plan around one.
            if 1200 <= value <= 4000:
                updates["kcal_target"] = value

        rationale = str(proposal.get("rationale") or "").strip()
        if rationale:
            updates["rationale"] = rationale[:500]

        return replace(self, **updates) if updates else self

    # ------------------------------------------------------------------ #
    # Contracts                                                           #
    # ------------------------------------------------------------------ #

    def to_requested(self) -> dict:
        """The dict `plan_verifier.verify` checks against.

        One definition, so what was asked for and what gets checked cannot
        drift into two hand-maintained lists.
        """
        return {
            "allergens": list(self.allergens),
            "diet": list(self.diet),
            "cuisines": list(self.cuisines),
            "moods": list(self.moods),
            "flavor_profiles": list(self.flavor_profiles),
            "food_groups": list(self.food_groups),
            "tags": list(self.claim_tags),
            "kcal_target": self.kcal_target,
            "kcal_reference": self.kcal_reference,
            "kcal_reference_basis": self.kcal_reference_basis,
            "max_minutes": self.max_minutes,
            "min_nutri_score": self.min_nutri_score,
            "pantry": list(self.pantry),
            "anchors": dict(self.anchors),
        }

    def facet_kwargs(self) -> dict:
        """The facet arguments a `plan_meals` call takes, ready to splat."""
        return {
            "cuisines": list(self.cuisines),
            "moods": list(self.moods),
            "flavor_profiles": list(self.flavor_profiles),
            "food_groups": list(self.food_groups),
            "tags": list(self.claim_tags),
        }

    def describe(self) -> str:
        """One line, for the log and for a grounded reply."""
        parts = []
        if self.diet:
            parts.append("diet: " + ", ".join(self.diet))
        if self.allergens:
            parts.append("avoiding " + ", ".join(self.allergens))
        for family in ("cuisines", "moods", "flavor_profiles", "food_groups"):
            values = getattr(self, family)
            if values:
                parts.append(f"{family.replace('_', ' ')}: " + ", ".join(values))
        if self.claim_tags:
            parts.append("asked for " + ", ".join(t.replace("_", " ") for t in self.claim_tags))
        if self.kcal_target:
            parts.append(f"{int(self.kcal_target)} kcal/day")
        if self.max_minutes:
            parts.append(f"under {self.max_minutes} min")
        if self.pantry:
            parts.append("using " + ", ".join(self.pantry))
        return " · ".join(parts) or "no standing constraints"


# ── deterministic derivations ─────────────────────────────────────────────

def _as_int(value) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _kcal_target(profile: dict, params: dict) -> Optional[float]:
    """The day's calorie budget, or None when nobody set one.

    None rather than the default is deliberate at this level: a target the
    member never set must not be reported as theirs, and must not rank their
    plates either. The default only applies when a GOAL is set, because a goal
    is the member asking to be planned against something.

    The stated target now comes from `reference_intake.stated_target`, which
    reads BOTH the structured key and the prose string the gateway mapping
    writes into `preferences`. Reading only the key — which nothing sets — is
    why this returned None for every member who had set a target.
    """
    from services import reference_intake

    stated = reference_intake.stated_target(profile)
    if stated:
        return stated
    goal = str(params.get("goal") or "").strip().lower()
    return DEFAULT_DAILY_KCAL if goal in {"weight_loss", "balanced", "high_protein", "energy"} else None


def _kcal_by_slot(spec, target: Optional[float]) -> dict[str, dict[str, float]]:
    """Each meal's share of the day, divided again across its plates.

    `PlanSpec.kcal_split` has been correct and unused since it was written.
    Without it a two-plate dinner is budgeted as if each plate were a whole
    meal, which is how a "light" plan comes back at 3,000 calories.
    """
    if spec is None or not target:
        return {}
    slots = list(getattr(spec, "meals", ()) or ())
    if not slots:
        return {}
    per_meal = float(target) / len(slots)
    out: dict[str, dict[str, float]] = {}
    for slot in slots:
        try:
            split = spec.kcal_split(slot)
        except Exception:  # noqa: BLE001 - a spec we cannot split is not fatal
            split = {"main": 1.0}
        out[slot] = {role: round(per_meal * weight, 1) for role, weight in split.items()}
    return out


def _nutri_floor(profile: dict, params: dict) -> Optional[str]:
    """The Nutri-Score floor a goal implies, or whatever the profile states."""
    stated = str(profile.get("min_nutri_score") or "").strip().lower()
    if stated in {"a", "b", "c", "d", "e"}:
        return stated
    goal = str(params.get("goal") or "").strip().lower()
    # Only the goals that are actually about nutrition quality. `energy` and
    # `high_protein` are about composition, and a score floor would quietly
    # exclude perfectly good high-protein dishes for an unrelated reason.
    return "c" if goal in {"weight_loss", "balanced"} else None
