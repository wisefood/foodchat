"""Who is at the table — the household behind a session, one member at a time.

Role
----
A session plans for one or more household members ("diners"). Until this
module existed the only record of them was the MERGED profile dict that
`ProfileService.merge_profiles` produced: one flat set of allergies, diets,
likes and dislikes. That flattening is lossy by construction — once it has
happened nothing downstream can ask "how good is this dish for Anna?", and a
group strategy other than satisfy-all cannot be expressed at all.

`HouseholdContext` keeps every diner's own profile (`DinerProfile`) beside the
strategy used to reconcile them. The merged dict is now a DERIVED VIEW
(`merged_view`), so every existing consumer — planner, weekly MDP, scorer,
ledger — keeps reading exactly the keys it read before.

Contracts
---------
* `DinerProfile.from_profile` reads the FoodChat profile dict that
  `ProfileService._map_profile` + `_attach_context` build. It copies only the
  planning-relevant fields; durable-memory bookkeeping (`memory_log`,
  `memory_optouts`), favorites and adapted recipes stay on the session
  member's profile and are NOT copied per diner.
* `diners[0]` is always the primary (session-owning) member.
* `merged_view(primary_profile)` returns a new dict; it never mutates its
  argument. Under `satisfy_all` it is byte-for-byte what `merge_profiles`
  returned before this module, plus the `household` snapshot key.
* The context is persisted INSIDE the session's profile snapshot under
  `HOUSEHOLD_KEY` (plain JSON via `model_dump`), so it is restart- and
  replica-safe with no schema change. `HouseholdContext.from_profile` reads it
  back and returns None — never raises — for sessions created before this
  existed or for a snapshot it cannot parse.
* `strategy` is a closed vocabulary (`STRATEGIES`). Only `satisfy_all` is
  implemented; the field exists so later strategies (least-misery, weekly
  fairness, split slots) are a new value and a new view, not a new model.

Introduced by the household-planning milestone, phase 0.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# The key the context is stored under inside `session.user_profile`.
HOUSEHOLD_KEY = "household"

SATISFY_ALL = "satisfy_all"
STRATEGIES = (SATISFY_ALL,)

# Dietary-goal slug -> soft preference string. Imported lazily in
# `merged_view` to keep this model importable without the service layer.


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(v) for v in value if v is not None and str(v).strip()]


class DinerProfile(BaseModel):
    """One household member's own planning profile, unmerged."""

    member_id: str = ""
    name: str = ""
    primary: bool = False

    # Identity / safety
    diet: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    # Taste
    food_likes: list[str] = Field(default_factory=list)
    food_dislikes: list[str] = Field(default_factory=list)
    # Goals and the soft strings derived from them
    dietary_goals: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    # Who they are — what an energy reference or a guideline scope needs
    calorie_target: Optional[float] = None
    sex: Optional[str] = None
    age_group: Optional[str] = None

    @field_validator(
        "diet", "allergies", "food_likes", "food_dislikes",
        "dietary_goals", "preferences", mode="before",
    )
    @classmethod
    def _listify(cls, value: Any) -> list[str]:
        return _as_list(value)

    @field_validator("calorie_target", mode="before")
    @classmethod
    def _numeric_or_none(cls, value: Any) -> Optional[float]:
        # The gateway's nutritional_preferences blob is free-form: a target
        # can arrive as 2000, "2000" or "about 2000". Unparseable is unknown.
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    @classmethod
    def from_profile(
        cls, profile: dict, name: str = "", primary: bool = False,
        member_id: str = "",
    ) -> "DinerProfile":
        profile = profile or {}
        return cls(
            member_id=str(member_id or profile.get("member_id") or ""),
            name=str(name or ""),
            primary=primary,
            diet=profile.get("diet"),
            allergies=profile.get("allergies"),
            food_likes=profile.get("food_likes"),
            food_dislikes=profile.get("food_dislikes"),
            dietary_goals=profile.get("dietary_goals"),
            preferences=profile.get("preferences"),
            calorie_target=profile.get("calorie_target"),
            sex=profile.get("sex") or None,
            age_group=profile.get("age_group") or None,
        )


class HouseholdContext(BaseModel):
    """Every diner of a session, plus how their needs are reconciled."""

    strategy: str = SATISFY_ALL
    diners: list[DinerProfile] = Field(default_factory=list)

    @field_validator("strategy")
    @classmethod
    def _known_strategy(cls, value: str) -> str:
        if value not in STRATEGIES:
            raise ValueError(
                f"Unknown household strategy {value!r}; expected one of {STRATEGIES}"
            )
        return value

    # ------------------------------------------------------------------ #
    # Construction / persistence                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def build(
        cls,
        primary: dict,
        others: Optional[list[dict]] = None,
        names: Optional[list[str]] = None,
        member_ids: Optional[list[str]] = None,
        strategy: str = SATISFY_ALL,
    ) -> "HouseholdContext":
        """Build from profile dicts. `names` / `member_ids` are
        ``[primary, *others]``; a short list yields empty values."""
        names = [str(n) for n in (names or [])]
        ids = [str(i) for i in (member_ids or [])]
        profiles = [primary, *(others or [])]
        diners = [
            DinerProfile.from_profile(
                profile,
                name=names[i] if i < len(names) else "",
                primary=(i == 0),
                member_id=ids[i] if i < len(ids) else "",
            )
            for i, profile in enumerate(profiles)
        ]
        return cls(strategy=strategy, diners=diners)

    @classmethod
    def from_profile(cls, profile: Optional[dict]) -> Optional["HouseholdContext"]:
        """The context stored on a session profile, or None when there is
        none (pre-existing sessions) or it cannot be read."""
        raw = (profile or {}).get(HOUSEHOLD_KEY)
        if not isinstance(raw, dict):
            return None
        try:
            return cls.model_validate(raw)
        except Exception as exc:  # noqa: BLE001 — a bad snapshot is "no context"
            logger.warning("Unreadable household snapshot ignored: %s", exc)
            return None

    def snapshot(self) -> dict:
        """Plain-JSON form for the session profile snapshot."""
        return self.model_dump()

    # ------------------------------------------------------------------ #
    # Reading                                                             #
    # ------------------------------------------------------------------ #

    @property
    def primary(self) -> Optional[DinerProfile]:
        return self.diners[0] if self.diners else None

    @property
    def others(self) -> list[DinerProfile]:
        return self.diners[1:]

    @property
    def names(self) -> list[str]:
        return [d.name for d in self.diners]

    @property
    def member_ids(self) -> list[str]:
        return [d.member_id for d in self.diners]

    @property
    def is_group(self) -> bool:
        """True when more than one person is eating."""
        return len(self.diners) > 1

    def diner(self, key: str) -> Optional[DinerProfile]:
        """Look a diner up by member id or (case-insensitive) name."""
        needle = str(key or "").strip().lower()
        if not needle:
            return None
        for d in self.diners:
            if d.member_id.lower() == needle or d.name.lower() == needle:
                return d
        return None

    # ------------------------------------------------------------------ #
    # Mid-session changes                                                 #
    # ------------------------------------------------------------------ #

    def remember_for_primary(self, kind: str, value: str) -> bool:
        """Mirror a consented memory onto the primary diner's own profile.

        `MemoryService` writes an accepted memory to the durable profile and
        to the merged session profile. Without this the per-member record
        would go stale the moment a member said "I'm vegetarian" — the merged
        view would know and the member it is about would not. Returns whether
        anything changed. Kinds with no per-diner field are ignored.
        """
        primary = self.primary
        value = str(value or "").strip().lower()
        if primary is None or not value:
            return False

        def add(items: list[str]) -> bool:
            if value in [i.lower() for i in items]:
                return False
            items.append(value)
            return True

        def drop(items: list[str]) -> bool:
            kept = [i for i in items if i.lower() != value]
            changed = len(kept) != len(items)
            items[:] = kept
            return changed

        if kind in ("like", "cuisine"):
            # Both run: a like is added AND leaves the dislikes
            return any([add(primary.food_likes), drop(primary.food_dislikes)])
        if kind == "dislike":
            return any([add(primary.food_dislikes), drop(primary.food_likes)])
        if kind == "allergy_hint":
            return add(primary.allergies)
        if kind == "diet":
            # "omnivore" is the absence of a restriction, not one to keep
            # beside a real one — same rule as the durable write.
            dropped = False
            if value != "omnivore":
                before = len(primary.diet)
                primary.diet[:] = [d for d in primary.diet if d.lower() != "omnivore"]
                dropped = len(primary.diet) != before
            return any([add(primary.diet), dropped])
        if kind == "dietary_goal":
            return add(primary.dietary_goals)
        return False

    # ------------------------------------------------------------------ #
    # Derived views                                                       #
    # ------------------------------------------------------------------ #

    def merged_view(self, primary_profile: dict) -> dict:
        """The single plan-generation profile every existing consumer reads.

        Dispatches on `strategy`; `satisfy_all` is the only one today.
        """
        merged = self._satisfy_all(primary_profile)
        merged[HOUSEHOLD_KEY] = self.snapshot()
        return merged

    def _satisfy_all(self, primary_profile: dict) -> dict:
        """One plan everybody can eat.

        Hard constraints (safety/identity) are the UNION across all diners:
        allergies, dietary groups, dislikes-as-exclusions. Soft preferences
        keep the primary member's weighting (their likes lead, other diners'
        likes follow). Everything not listed here — favorites, calorie
        target, numeric goal targets — stays the primary's.

        Goals are the delicate part. The primary member's goals keep their
        numeric targets (``nutrition_profile``, ``min_nutri_score``), because
        stacking several members' targets can empty the candidate set. Every
        other diner's goal is demoted to the soft ``preferences`` channel
        instead of being dropped, and the whole picture — whose goal, applied
        as a target or only as a signal — is recorded in
        ``goal_reconciliation`` so the plan can say so out loud.

        ``constraint_origins`` keeps the member each constraint came from, so
        the ledger can attribute "no peanuts" to the diner who is allergic
        rather than to everyone at the table.
        """
        from services.profile_service import goal_preference_strings

        merged = dict(primary_profile)
        origins: dict[str, dict[str, list[str]]] = {}

        def union(key: str) -> list:
            seen: dict[str, str] = {}
            key_origins: dict[str, list[str]] = {}
            for d in self.diners:
                for item in getattr(d, key):
                    low = str(item).lower()
                    seen.setdefault(low, item)
                    who = key_origins.setdefault(low, [])
                    if d.name and d.name not in who:
                        who.append(d.name)
            # Re-key origins by the value as it will be displayed
            origins[key] = {seen[low]: who for low, who in key_origins.items()}
            return list(seen.values())

        merged["allergies"] = union("allergies")
        merged["diet"] = union("diet")
        merged["food_dislikes"] = union("food_dislikes")
        merged["constraint_origins"] = origins

        likes: list[str] = []
        for d in self.diners:
            for like in d.food_likes:
                if str(like).lower() not in [str(seen).lower() for seen in likes]:
                    likes.append(like)
        merged["food_likes"] = likes

        # ── Goal reconciliation ──
        primary = self.primary
        primary_name = primary.name if primary else ""
        primary_goals = [s.lower() for s in (primary.dietary_goals if primary else [])]
        reconciliation: list[dict] = [
            {"slug": slug, "member": primary_name, "applied": "target"}
            for slug in primary_goals
        ]

        demoted: list[str] = []
        for d in self.others:
            for slug in d.dietary_goals:
                slug = str(slug).lower()
                if slug in primary_goals:
                    # Two diners wanting the same thing is agreement, not conflict
                    reconciliation.append({"slug": slug, "member": d.name, "applied": "target"})
                    continue
                if slug not in demoted:
                    demoted.append(slug)
                reconciliation.append({"slug": slug, "member": d.name, "applied": "soft"})

        if demoted:
            preferences = list(merged.get("preferences") or [])
            for text in goal_preference_strings(demoted):
                if text not in preferences:
                    preferences.append(text)
            merged["preferences"] = preferences

        merged["goal_reconciliation"] = reconciliation
        merged["cooking_for_names"] = self.names
        return merged
