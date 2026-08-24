"""
Recipe qualities stated in chat → standing planning state → a real filter.

"Energy boost meal plan for today" was the request that exposed this. It matched
nothing, for three reasons stacked on top of each other:

* `plan_client.plan_meals` **declares** `moods`, `flavor_profiles` and
  `food_groups`; RecipeWrangler accepts all three and puts them first in its
  relaxation ladder. **No caller ever passed any of them.** Only `cuisines` was
  ever sent, and only from the stored profile.
* There is no cuisine extractor anywhere either — "something Thai tonight" never
  became a `cuisines` filter on any path.
* "energy" is not in any vocabulary: not a mood, not a flavour, not a food
  group. It is a `plan_parameters.goal` value that only ever became prose.

So the member's words reached the grader as text, over a pool that had never
been shaped by them, and the plan came back indistinguishable from one with no
request at all.

Contracts, mirroring `diet_intent` and `pantry_service`:

    extract_facet_delta(message)        → PlanningStateDelta (never raises)
    facets_for_goal(goal)               → the facets a slider goal implies
    claim_tags_for(goal, facets)        → RecipeWrangler claim tags (Phase 1b)

Two rules the whole module exists to hold:

**Never send a value the corpus does not carry.** RecipeWrangler ANDs facet
values and never relaxes an unlisted one to nothing — it just matches no recipe.
So an invented mood does not soften the search, it empties it, and the member is
told no meals exist. Every value here is validated against the LIVE vocabulary
twice: once in the extractor's prompt, once after it answers.

**Silence is not a retraction.** A facet stated three turns ago still holds;
only an explicit removal takes it back.
"""

from __future__ import annotations

import logging
from typing import Optional

from models.planning_state import PlanningState, PlanningStateDelta

logger = logging.getLogger(__name__)

# What each plan-parameter goal actually means, in vocabulary that exists.
#
# The slider offers weight_loss / balanced / high_protein / energy, and all four
# were prose to the grader and nothing else. They are mapped here onto facets
# RecipeWrangler annotates and claim tags its corpus carries, because a goal
# that cannot become a filter is a goal the member set for nothing.
#
# `energy` is the interesting one: there is no "energising" mood in the corpus,
# so it is read as sustaining food — protein and fibre — rather than invented as
# a facet. That is a judgement, and it is written down here rather than hidden
# in a prompt.
GOAL_FACETS: dict[str, dict[str, list[str]]] = {
    "weight_loss": {"moods": ["light"]},
    "balanced": {},
    "high_protein": {},
    "energy": {"moods": ["hearty"]},
}

# Claim tags live on RecipeWrangler's `tags` field, not `diet_tags`. Counts are
# from the corpus census, so a mapping to a tag nothing carries is visible here
# rather than at runtime:
#   high_protein 1676 · low_fat 1081 · high_fibre 457 · low_calorie 184
#   30_minutes_or_less 2809 · healthy_and_nutritious 2535 · 5_ingredients_or_less 563
GOAL_CLAIM_TAGS: dict[str, list[str]] = {
    "weight_loss": ["low_calorie"],
    "balanced": ["healthy_and_nutritious"],
    "high_protein": ["high_protein"],
    "energy": ["high_protein", "high_fibre"],
}

# Effort — what "difficulty" turned into once it had to be true.
#
# RecipeWrangler has NO difficulty field: `grep -ri difficulty` across its
# source returns nothing. There is no tag, no vocabulary, nothing to ask for.
# So Easy/Medium/Elaborate was a control where two of the three options changed
# the plan in no way at all, which is the question a member eventually asks out
# loud: why is this here?
#
# What the corpus does carry is `5_ingredients_or_less` (563 recipes), which is
# a real, checkable form of "keep it simple". So the control is now that, and
# "Elaborate" is gone rather than kept as decoration.
DIFFICULTY_CLAIM_TAGS: dict[str, list[str]] = {
    # `5_ingredients_or_less` only, not `30_minutes_or_less` as well: cooking
    # time is its own control, and two controls writing the same filter is how
    # they end up disagreeing about what the member asked for.
    "easy": ["5_ingredients_or_less"],
    # "Any" — the default, and an honest answer rather than a dead option.
    "medium": [],
}

# The nutrition claims a member states directly, as opposed to via the slider.
# `diet_intent.split_diet_intent` already separates these from real diets
# (they are on zero recipes as diet tags); this is where they become searchable.
CLAIM_TAG_ALIASES: dict[str, str] = {
    "high-protein": "high_protein",
    "high_protein": "high_protein",
    "low-fat": "low_fat",
    "low_fat": "low_fat",
    "low-carb": "low_calorie",   # no low_carb tag in the corpus; calorie is the closest honest proxy
    "low_carb": "low_calorie",
    "high-fibre": "high_fibre",
    "high_fibre": "high_fibre",
    "high-fiber": "high_fibre",
    "low-calorie": "low_calorie",
    "quick": "30_minutes_or_less",
}


def extract_facet_delta(message: str, *, extractor=None, vocabularies=None):
    """What this turn asks for, as facets. Never raises.

    Returns an empty delta when the vocabulary is unavailable — with no live
    list there is no value that is safe to send, and sending nothing is exactly
    the behaviour that existed before facets were wired at all.
    """
    text = (message or "").strip()
    if not text:
        return PlanningStateDelta()

    try:
        from services.candidates_client import CANDIDATES

        vocab = vocabularies if vocabularies is not None else CANDIDATES.vocabularies()
        if not vocab:
            return PlanningStateDelta()
        if extractor is None:
            from agents import PlanIntentExtractor

            extractor = PlanIntentExtractor()
        found = extractor.extract(text, vocab)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Facet extraction failed: %s", exc)
        return PlanningStateDelta()

    delta = PlanningStateDelta(
        cuisines=tuple(found.get("cuisines") or ()),
        moods=tuple(found.get("moods") or ()),
        flavor_profiles=tuple(found.get("flavor_profiles") or ()),
        food_groups=tuple(found.get("food_groups") or ()),
    )
    if not delta.is_empty:
        logger.info(
            "Facet intent: cuisines=%s moods=%s flavours=%s food_groups=%s",
            delta.cuisines, delta.moods, delta.flavor_profiles, delta.food_groups,
        )
    return delta


# Taking a facet back.
#
# The UI could always do this — a chip has an × and it calls
# `DELETE /facets/{value}`. Chat could not: the extractor answers with what a
# member ASKED for, and nothing read "actually not spicy" as a retraction. So
# the removable-chip affordance existed on one channel only, and a member who
# said it out loud watched the chip stay.
#
# Deterministic, and deliberately narrow: it can only remove a value that is
# ALREADY standing. That is what makes a regex safe here — there is nothing to
# invent, only a known list to match against, so the failure mode is missing a
# retraction rather than inventing one.
_NEGATION = (
    r"(?:not|no|no more|not more|without|drop|forget|skip|lose|remove|"
    r"stop|less|nothing)"
)
# Words that can sit between the negation and the thing: "not too spicy",
# "no more of the Thai food", "drop that comfort food".
_FILLER = r"(?:the|a|an|any|more|that|too|so|of|it|them)"


def extract_facet_removals(message: str, state) -> PlanningStateDelta:
    """Facets this turn takes back. Never raises.

    Matched against what is standing, so "not spicy" removes the spicy flavour
    only if the member ever asked for it. A negation naming something they never
    said is not a retraction — it is a new request, and the facet extractor
    reads that.
    """
    import re

    text = (message or "").strip()
    if not text:
        return PlanningStateDelta()

    standing: list[str] = []
    for family in PlanningState.FACET_FIELDS:
        standing.extend(getattr(state, family, ()) or ())
    if not standing:
        return PlanningStateDelta()

    removals: list[str] = []
    for value in standing:
        if value in removals:
            continue
        # Slugs are stored with underscores; a member says them with spaces or
        # hyphens ("comfort food", "comfort-food").
        parts = [re.escape(part) for part in str(value).split("_") if part]
        if not parts:
            continue
        spelled = r"[\s_-]+".join(parts)
        pattern = rf"\b{_NEGATION}\s+(?:{_FILLER}\s+)*{spelled}\b"
        if re.search(pattern, text, re.IGNORECASE):
            removals.append(value)

    if removals:
        logger.info("Facets taken back: %s", ", ".join(removals))
    return PlanningStateDelta(facets_remove=tuple(removals))


def facets_for_goal(goal: Optional[str]) -> dict[str, list[str]]:
    """The facets a plan-parameter goal implies, or `{}`."""
    return dict(GOAL_FACETS.get(str(goal or "").strip().lower(), {}))


def claim_tags_for(
    goal: Optional[str] = None,
    claims: Optional[list[str]] = None,
    difficulty: Optional[str] = None,
) -> list[str]:
    """RecipeWrangler claim tags implied by a goal and any stated claims.

    Kept separate from facets because they ride a different request field, and
    that field does not exist upstream yet — so this returns the right answer
    before anything can send it, and the fetch sites start using it the day the
    parameter lands.
    """
    tags: list[str] = []
    for tag in GOAL_CLAIM_TAGS.get(str(goal or "").strip().lower(), []):
        if tag not in tags:
            tags.append(tag)
    for tag in DIFFICULTY_CLAIM_TAGS.get(str(difficulty or "").strip().lower(), []):
        if tag not in tags:
            tags.append(tag)
    for claim in claims or []:
        tag = CLAIM_TAG_ALIASES.get(str(claim).strip().lower())
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def facet_kwargs(profile: dict, cuisines: Optional[list[str]] = None) -> dict:
    """The facet arguments for a `plan_meals` call, ready to splat.

    Every fetch site already computed `cuisines` from the profile and passed it
    alone. This merges that with the three families nobody sent, so one `**`
    replaces one `cuisines=` and no site has to learn about the rest.

    Always returns all four keys — an empty list is what `plan_meals` expects
    for "no preference", and omitting a key would leave the request shape
    varying between call sites for no reason.
    """
    merged = effective_facets(profile)
    # The slider goal implies facets as well as tags — "weight loss" means the
    # `light` mood, and that was prose to a grader two of three paths never run.
    goal_facets = facets_for_goal((profile.get("plan_parameters") or {}).get("goal"))
    for family, values in goal_facets.items():
        existing = merged.setdefault(family, [])
        for value in values:
            if value not in existing:
                existing.append(value)
    if cuisines:
        existing = merged.setdefault("cuisines", [])
        for value in cuisines:
            slug = str(value).strip().lower()
            if slug and slug not in existing:
                existing.append(slug)
    # Claim tags ride the same splat so a fetch site does not need to know
    # they exist. They come from two places: the slider goal, and claims the
    # member stated in words ("high protein"). `plan_client` decides whether
    # the live RecipeWrangler can actually receive them.
    params = profile.get("plan_parameters") or {}
    tags = claim_tags_for(
        params.get("goal"),
        list(profile.get("_claim_tags") or []),
        params.get("difficulty"),
    )

    return {
        "cuisines": merged.get("cuisines") or [],
        "moods": merged.get("moods") or [],
        "flavor_profiles": merged.get("flavor_profiles") or [],
        "food_groups": merged.get("food_groups") or [],
        "tags": tags,
    }


def effective_facets(profile: dict, state: Optional[PlanningState] = None) -> dict:
    """Facets to send: stated in chat, plus the profile's own preference words.

    `profile["_facets"]` is the transient stash chat_service fills from standing
    state (same underscore convention as `_pantry` and `_diet_tags`). Profile
    words come through `split_preferences`, which sorts `food_likes` into the
    family each word belongs to — so a stored "greek" drives the cuisine filter
    it was always meant to, and a stored "comfort" now drives a mood instead of
    being searched for as an ingredient.
    """
    from services.candidates_client import CANDIDATES

    stated = dict(profile.get("_facets") or {})
    if state is not None:
        for family, values in state.facets().items():
            merged = list(stated.get(family) or [])
            for value in values:
                if value not in merged:
                    merged.append(value)
            stated[family] = merged

    likes = list(profile.get("food_likes") or [])
    from_profile = CANDIDATES.split_preferences(likes)

    out: dict[str, list[str]] = {}
    for family in CANDIDATES.FACET_FAMILIES:
        merged = list(stated.get(family) or [])
        for value in from_profile.get(family) or []:
            if value not in merged:
                merged.append(value)
        if merged:
            out[family] = merged
    return out
