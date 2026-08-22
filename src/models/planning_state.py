"""What the member has told us about their plan, kept across turns.

The chat loop rewrote the whole request into a fresh query every turn and
regenerated from scratch. Nothing accumulated, so a conversation looked like
this:

    "no" (to favourites)        -> a favourite appeared in the plan
    "apple pie for breakfast"   -> scrambled eggs
    "add salads as side dishes" -> a different breakfast, no salads
    "that has eggs, I'm vegan"  -> poached eggs

Those are not four bugs. Each turn, what the member had already said stopped
existing: it was words in a sentence that got rewritten, not state anyone kept.

`PlanningState` is that state. It is **durable** — persisted on the session —
and each turn produces a *delta* that merges into it rather than replacing it.
Saying "no favourites" once means no favourites until the member says
otherwise; adding salads on the side does not lose the anchor set two turns
ago.

Two rules make merging predictable:

**Silence is not a retraction.** A delta only carries what this turn mentioned.
A turn that says nothing about favourites leaves the favourites decision alone;
the alternative is that every unmentioned preference quietly resets, which is
the bug this exists to fix.

**An explicit reset is possible.** "Start over" clears state, because a member
who wants a fresh plan should not have to argue with three turns of accumulated
constraints. That is `reset()`, and only an explicit request triggers it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

from models.plan_spec import PlanSpec


@dataclass(frozen=True)
class PlanningState:
    """Everything the member has said that should outlive a single turn."""

    # The shape of plan asked for. Defaults to today's three meals.
    spec: PlanSpec = field(default_factory=PlanSpec.default)

    # slot -> recipe_id the member named for it ("apple pie for breakfast").
    # Kept as ids, not resolved recipes: the recipe can be re-fetched, and
    # storing a snapshot would serve a stale title after the member adapts it.
    anchors: dict[str, str] = field(default_factory=dict)

    # Recipes the member has rejected — "not that one", a downvote, or a swap
    # away from. They must not come back on the next regeneration.
    excluded_recipe_ids: tuple[str, ...] = ()

    # Set when the member declines the favourites offer. Tri-state on purpose:
    # None means never asked, False means asked and declined. A plain bool
    # cannot tell "they said no" from "we never offered", and the first must
    # suppress the offer while the second must not.
    use_favorites: Optional[bool] = None

    # Free-text constraints the member stated that no filter captures
    # ("nothing too heavy in the evening"). Carried into the grader's prompt.
    notes: tuple[str, ...] = ()

    # On-hand ingredients the member wants used up ("I have zucchini and
    # spinach") — the food-waste pantry. Normalized lowercase names, insertion
    # order kept. Session-scoped planning state like everything here, NOT a
    # durable profile field; see services.pantry_service.
    pantry: tuple[str, ...] = ()

    # Diet the member stated IN CHAT ("I need something vegetarian"), as
    # RecipeWrangler diet tags. Standing for the session and merged with the
    # profile's own diet at fetch time — an explicit request outranks a stored
    # setting, and `omnivore` is not a restriction to be protected.
    #
    # This existed nowhere before: the daily path read only `profile["diet"]`,
    # so a stated diet reached the grader as prose over a pool that had never
    # been filtered for it. Durable only if the member accepts the memory
    # nudge (kind "diet"); until then it dies with the session.
    diet_tags: tuple[str, ...] = ()

    # Nutrition claims the member asked for — "high protein", "low carb".
    # These are NOT diets: no recipe carries them as a diet tag, so sending one
    # as a diet filter empties every slot. They are RecipeWrangler claim tags,
    # which is a different request field.
    #
    # They used to be written to `notes`, which turned out to be write-only
    # (read solely by `describe()`, which is only logged) — so a claim was
    # correctly stopped from becoming an empty filter and then dropped on the
    # floor instead of reaching anything.
    claim_tags: tuple[str, ...] = ()

    # Facet preferences stated in chat — "something comforting", "light and
    # fresh", "more vegetables", "Thai tonight". One tuple per RecipeWrangler
    # facet family, mirroring `diet_tags`: standing for the session, additive,
    # and never cleared by silence.
    #
    # These are the families FoodChat's own client declared and never passed,
    # so every one of those requests reached the grader as prose over a pool
    # that had never been shaped by it. Cuisine is here too because nothing
    # extracted a cuisine from a message at all — the only `cuisines` filter
    # ever sent came from the stored profile.
    cuisines: tuple[str, ...] = ()
    moods: tuple[str, ...] = ()
    flavor_profiles: tuple[str, ...] = ()
    food_groups: tuple[str, ...] = ()

    # A cooking-time ceiling the member stated in words, in minutes.
    #
    # The same constraint the cooking-time slider expresses, and it reaches the
    # same place: `profile["plan_parameters"]["cooking_time"]`, which all seven
    # fetch sites already read through `plan_parameters.max_duration_minutes`.
    # Held here rather than written straight to the profile because it is a
    # standing statement like every other — "keep it under 20 minutes" still
    # means that next turn.
    #
    # The persona has promised for a long time that a member can steer by
    # cooking time. Only the slider ever could.
    max_minutes: Optional[int] = None

    #: The facet families carried above, in the order RecipeWrangler relaxes
    #: them last-to-first. Iterating this beats four copies of everything.
    FACET_FIELDS = ("cuisines", "moods", "flavor_profiles", "food_groups")

    def facets(self) -> dict[str, list[str]]:
        """The stated facets as a fetch-ready mapping (empty families omitted)."""
        return {
            family: list(getattr(self, family))
            for family in self.FACET_FIELDS
            if getattr(self, family)
        }

    def merge(self, delta: "PlanningStateDelta") -> "PlanningState":
        """Apply one turn's changes. Absent fields leave state untouched."""
        if delta.reset:
            return PlanningState()

        spec = delta.spec if delta.spec is not None else self.spec

        anchors = dict(self.anchors)
        for slot, recipe_id in (delta.anchors or {}).items():
            if recipe_id:
                anchors[slot] = recipe_id
            else:
                # An explicit empty value clears that slot's anchor — "actually
                # never mind the apple pie".
                anchors.pop(slot, None)

        excluded = list(self.excluded_recipe_ids)
        for recipe_id in delta.excluded_recipe_ids or ():
            if recipe_id and recipe_id not in excluded:
                excluded.append(recipe_id)

        notes = list(self.notes)
        for note in delta.notes or ():
            if note and note not in notes:
                notes.append(note)

        # Pantry: additive, with explicit removal ("I used up the zucchini").
        # Silence leaves it alone, like everything else here.
        pantry = list(self.pantry)
        removed = {str(r).strip().lower() for r in (delta.pantry_remove or ()) if r}
        if removed:
            pantry = [item for item in pantry if item not in removed]
        for item in delta.pantry_add or ():
            value = str(item).strip().lower()
            if value and value not in pantry:
                pantry.append(value)

        # Diet: additive, with an explicit clear for "actually, follow my
        # profile". `diet_clear` is a flag rather than a remove-list because a
        # member retracting a stated diet retracts all of it, not one tag.
        diet_tags = [] if delta.diet_clear else list(self.diet_tags)
        for tag in delta.diet_tags or ():
            value = str(tag).strip().lower()
            if value and value not in diet_tags:
                diet_tags.append(value)

        # Claims ride the same retraction as the diet they arrive with: a
        # member taking back "vegetarian" is taking back that whole statement.
        claim_tags = [] if delta.diet_clear else list(self.claim_tags)
        for tag in delta.claim_tags or ():
            value = str(tag).strip().lower()
            if value and value not in claim_tags:
                claim_tags.append(value)

        # Facets: additive per family, with an explicit removal list so a
        # member can take one back ("actually not spicy") — and so the UI's
        # removable chips have something to call.
        removed = {str(r).strip().lower() for r in (delta.facets_remove or ()) if r}
        facet_values: dict[str, tuple[str, ...]] = {}
        for family in self.FACET_FIELDS:
            current = [v for v in getattr(self, family) if v not in removed]
            for value in getattr(delta, family, ()) or ():
                slug = str(value).strip().lower()
                if slug and slug not in current and slug not in removed:
                    current.append(slug)
            facet_values[family] = tuple(current)

        max_minutes = (
            None if delta.max_minutes_clear
            else (self.max_minutes if delta.max_minutes is None else delta.max_minutes)
        )

        return replace(
            self,
            spec=spec,
            max_minutes=max_minutes,
            **facet_values,
            anchors=anchors,
            excluded_recipe_ids=tuple(excluded),
            use_favorites=(
                self.use_favorites if delta.use_favorites is None else delta.use_favorites
            ),
            notes=tuple(notes),
            pantry=tuple(pantry),
            diet_tags=tuple(diet_tags),
            claim_tags=tuple(claim_tags),
        )

    def describe(self) -> str:
        """What is currently in force, for the agent to say or confirm.

        The member should be able to ask "what are you working with?" and get an
        answer. A system that silently accumulates constraints is as confusing
        as one that silently forgets them.
        """
        parts = [self.spec.describe()]
        if self.anchors:
            parts.append(
                "anchored: "
                + ", ".join(f"{slot}={rid}" for slot, rid in sorted(self.anchors.items()))
            )
        if self.use_favorites is False:
            parts.append("favourites declined")
        if self.excluded_recipe_ids:
            parts.append(f"{len(self.excluded_recipe_ids)} recipe(s) ruled out")
        if self.pantry:
            parts.append("pantry to use up: " + ", ".join(self.pantry))
        if self.diet_tags:
            parts.append("diet stated in chat: " + ", ".join(self.diet_tags))
        if self.claim_tags:
            parts.append("asked for: " + ", ".join(self.claim_tags))
        if self.max_minutes:
            parts.append(f"under {self.max_minutes} min per meal")
        for family, values in self.facets().items():
            parts.append(f"{family.replace('_', ' ')}: " + ", ".join(values))
        if self.notes:
            parts.append("; ".join(self.notes))
        return " · ".join(parts)

    def as_query(self) -> str:
        """A short natural request equivalent to what is standing.

        Used when the plan is regenerated with no new member message — a facet
        chip removed, a pantry item ticked off. The wording matters: the query
        drives the semantic search and the grader, so it has to describe what
        the member still wants. Describing the EDIT instead ("without the
        spicy flavour") would search for the thing being removed.
        """
        bits: list[str] = []
        bits += [t.replace("_", " ") for t in self.diet_tags]
        bits += [t.replace("_", " ") for t in self.claim_tags]
        for values in self.facets().values():
            bits += [v.replace("_", " ") for v in values]

        seen: list[str] = []
        for bit in bits:
            if bit and bit not in seen:
                seen.append(bit)

        text = f"a {', '.join(seen)} meal plan" if seen else "a meal plan"
        if self.pantry:
            text += " using up " + ", ".join(self.pantry)
        return text

    # -- persistence ---------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "anchors": dict(self.anchors),
            "excluded_recipe_ids": list(self.excluded_recipe_ids),
            "use_favorites": self.use_favorites,
            "notes": list(self.notes),
            "pantry": list(self.pantry),
            "diet_tags": list(self.diet_tags),
            "claim_tags": list(self.claim_tags),
            "max_minutes": self.max_minutes,
            **{family: list(getattr(self, family)) for family in self.FACET_FIELDS},
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "PlanningState":
        """Rebuild from a stored payload, tolerating anything.

        A session whose stored state cannot be parsed gets a fresh one rather
        than an exception: losing the accumulated constraints is a bad turn,
        losing the ability to plan at all is a broken product.
        """
        if not isinstance(raw, dict):
            return cls()
        spec_raw = raw.get("spec")
        spec = PlanSpec.from_spec(spec_raw) if isinstance(spec_raw, dict) else PlanSpec.default()
        anchors = {
            str(k): str(v)
            for k, v in (raw.get("anchors") or {}).items()
            if k and v
        }
        favourites = raw.get("use_favorites")
        return cls(
            spec=spec,
            anchors=anchors,
            excluded_recipe_ids=tuple(
                str(r) for r in (raw.get("excluded_recipe_ids") or []) if r
            ),
            use_favorites=favourites if isinstance(favourites, bool) else None,
            notes=tuple(str(n) for n in (raw.get("notes") or []) if n),
            pantry=tuple(
                str(p).strip().lower() for p in (raw.get("pantry") or []) if p
            ),
            diet_tags=tuple(
                str(d).strip().lower() for d in (raw.get("diet_tags") or []) if d
            ),
            claim_tags=tuple(
                str(c).strip().lower() for c in (raw.get("claim_tags") or []) if c
            ),
            max_minutes=_as_minutes(raw.get("max_minutes")),
            **{
                family: tuple(
                    str(v).strip().lower() for v in (raw.get(family) or []) if v
                )
                for family in PlanningState.FACET_FIELDS
            },
        )


@dataclass(frozen=True)
class PlanningStateDelta:
    """One turn's changes. Every field optional — absent means "not mentioned".

    Separate from `PlanningState` so the difference between "set this to empty"
    and "did not mention it" is expressible. A single mutable dict cannot say
    both, which is exactly how "no favourites" got lost.
    """

    spec: Optional[PlanSpec] = None
    anchors: Optional[dict[str, str]] = None
    excluded_recipe_ids: tuple[str, ...] = ()
    use_favorites: Optional[bool] = None
    notes: tuple[str, ...] = ()
    # Pantry items this turn added ("I have …") / declared spent ("used up
    # the …"). Separate tuples so adding and removing in one turn both land.
    pantry_add: tuple[str, ...] = ()
    pantry_remove: tuple[str, ...] = ()
    # Diet tags stated this turn, and the retraction flag for "never mind, use
    # my profile" (answered NO to the dietary-conflict question).
    diet_tags: tuple[str, ...] = ()
    # Nutrition claims stated this turn — they reach RecipeWrangler's `tags`
    # field, not the diet filter.
    claim_tags: tuple[str, ...] = ()
    diet_clear: bool = False
    # A cooking-time ceiling stated in words, and its retraction ("take as long
    # as you need"). A flag rather than a sentinel value because 0 minutes is
    # not a retraction, it is a nonsense constraint.
    max_minutes: Optional[int] = None
    max_minutes_clear: bool = False
    # Facets stated this turn, plus values to take back (the UI's removable
    # chips, and "actually not spicy").
    cuisines: tuple[str, ...] = ()
    moods: tuple[str, ...] = ()
    flavor_profiles: tuple[str, ...] = ()
    food_groups: tuple[str, ...] = ()
    facets_remove: tuple[str, ...] = ()
    reset: bool = False

    @property
    def is_empty(self) -> bool:
        return not (
            self.spec
            or self.anchors
            or self.excluded_recipe_ids
            or self.use_favorites is not None
            or self.notes
            or self.pantry_add
            or self.pantry_remove
            or self.diet_tags
            or self.claim_tags
            or self.diet_clear
            or self.max_minutes is not None
            or self.max_minutes_clear
            or self.cuisines
            or self.moods
            or self.flavor_profiles
            or self.food_groups
            or self.facets_remove
            or self.reset
        )


def _as_minutes(value: Any) -> Optional[int]:
    """A stored minute count, or None. Tolerates anything a session row holds."""
    if value is None:
        return None
    try:
        minutes = int(value)
    except (TypeError, ValueError):
        return None
    return minutes if minutes > 0 else None
