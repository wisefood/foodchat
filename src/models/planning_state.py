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

        return replace(
            self,
            spec=spec,
            anchors=anchors,
            excluded_recipe_ids=tuple(excluded),
            use_favorites=(
                self.use_favorites if delta.use_favorites is None else delta.use_favorites
            ),
            notes=tuple(notes),
            pantry=tuple(pantry),
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
        if self.notes:
            parts.append("; ".join(self.notes))
        return " · ".join(parts)

    # -- persistence ---------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "anchors": dict(self.anchors),
            "excluded_recipe_ids": list(self.excluded_recipe_ids),
            "use_favorites": self.use_favorites,
            "notes": list(self.notes),
            "pantry": list(self.pantry),
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
            or self.reset
        )
