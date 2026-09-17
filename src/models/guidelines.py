"""
Dietary guidelines from the WiseFood Data API — which rules, and the rows.

    GuidelineScope   which rules apply: region, life stage, plan type, and an
                     optional pin to whole guides or to named rule ids. The
                     ONE place catalog filter strings (`fq`) are built.
    Guideline        one rule as the catalog returns it, the fields FoodChat
                     reads and nothing else required

Today every member gets the deployment default (Ireland, adults): the profile
carries no region or age group yet. `services.guidelines_service.resolve_scope`
already reads them when they appear, and a caller that wants a different
country or a subset of rules passes its own scope — nothing below changes.

The filter strings are the catalog's Solr-style syntax, verified live
(2026-09-17). Two of them are not the obvious spelling:

- "tagged X, or not tagged at all" is ``field:(X) OR (*:* -field:*)``. The
  shorter ``(field:(X) OR -field:*)`` is accepted and silently matches nothing.
- URNs contain colons, so they are quoted.

Only `active` rules are asked for. `review_status: verified` is the same 526
rows today, and a draft is someone's work in progress.

Layering rule: like every module in ``models``, no imports from ``agents``
or ``services``.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The catalog's own vocabularies (GuidelineLifeStage / GuidelineTargetPopulation).
LIFE_STAGES = (
    "infancy", "early_childhood", "school_age", "adolescence",
    "adulthood", "older_adulthood", "pregnancy", "lactation",
)

# `life_stage` and `target_populations` are both enriched, and a rule often
# carries one without the other, so a stage filter reads both.
POPULATION_FOR_STAGE: dict[str, str] = {
    "infancy": "infants",
    "early_childhood": "under_5_years",
    "school_age": "ages_5_to_18",
    "adolescence": "ages_5_to_18",
    "adulthood": "adults",
    "older_adulthood": "elderly",
    "pregnancy": "pregnant_people",
    "lactation": "lactating_people",
}

# Rules about exercise, not food. `other` is 4 rows of nothing in particular.
EXCLUDED_TYPES = ("activity", "other")

# A day cannot show a weekly or monthly rule being kept or broken.
NOT_JUDGEABLE_IN_A_DAY = ("weekly", "monthly")

PlanType = Literal["daily", "weekly"]


def _quoted(values) -> str:
    return " OR ".join(f'"{v}"' for v in values)


class GuidelineScope(BaseModel):
    """Which guidelines a plan is judged against."""

    model_config = ConfigDict(frozen=True)

    regions: tuple[str, ...] = ()
    """ISO 3166-1 alpha-2, upper case, as the catalog stores `guide_region`."""
    life_stage: Optional[str] = None
    """One of LIFE_STAGES. Rules tagged for it, plus rules tagged for nobody."""
    plan_type: Optional[PlanType] = None
    guide_urns: tuple[str, ...] = ()
    """Pin whole guides."""
    rule_ids: tuple[str, ...] = ()
    """An explicit subset. When set, every other facet is ignored."""
    limit: int = Field(default=300, ge=1, le=1000)

    def fq(self) -> list[str]:
        if self.rule_ids:
            return [f"id:({_quoted(self.rule_ids)})", "status:active"]

        fq = ["status:active"]
        if self.regions:
            fq.append(f"guide_region:({' OR '.join(self.regions)})")
        if self.guide_urns:
            fq.append(f"guide_urn:({_quoted(self.guide_urns)})")
        if self.life_stage:
            fq.append(f"life_stage:({self.life_stage}) OR (*:* -life_stage:*)")
            populations = ["general_population"]
            if self.life_stage in POPULATION_FOR_STAGE:
                populations.append(POPULATION_FOR_STAGE[self.life_stage])
            fq.append(
                f"target_populations:({' OR '.join(populations)})"
                " OR (*:* -target_populations:*)"
            )
        fq.append(f"-guideline_type:({' OR '.join(EXCLUDED_TYPES)})")
        if self.plan_type == "daily":
            fq.append(f"-frequency:({' OR '.join(NOT_JUDGEABLE_IN_A_DAY)})")
        return fq

    def cache_key(self) -> str:
        return f"guidelines::{self.limit}::" + "|".join(self.fq())


class Guideline(BaseModel):
    """One catalog rule. Unknown fields are ignored; missing lists are empty."""

    model_config = ConfigDict(extra="ignore")

    id: str = ""
    guide_urn: str = ""
    guide_region: Optional[str] = None
    rule_text: str = ""
    title: Optional[str] = None
    sequence_no: Optional[int] = None
    guideline_type: Optional[str] = None
    frequency: Optional[str] = None
    quantity: Optional[dict] = None
    food_groups: list[str] = Field(default_factory=list)
    topic: list[str] = Field(default_factory=list)
    life_stage: list[str] = Field(default_factory=list)
    target_populations: list[str] = Field(default_factory=list)
    status: Optional[str] = None
    review_status: Optional[str] = None

    @field_validator(
        "food_groups", "topic", "life_stage", "target_populations", mode="before",
    )
    @classmethod
    def _null_is_empty(cls, value):
        return [] if value is None else value

    @field_validator("rule_text", mode="before")
    @classmethod
    def _text_or_empty(cls, value):
        return "" if value is None else value
