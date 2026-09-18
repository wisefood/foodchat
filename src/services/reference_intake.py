"""
Whose calorie number is this, and where did it come from.

    stated_target(profile)   -> float | None     the member's OWN target
    reference_for(profile)   -> EnergyReference  a number plus its provenance

Three separate readers already parsed a calorie target, and the daily planner
was not one of them:

    weekly_planner.state_tracking   "2000 calories target" in `preferences`  ✓
    plan_scorer.scoring             the same string                          ✓
    models.plan_brief._kcal_target  `profile["calorie_target"]`              ✗

`profile["calorie_target"]` is never set. `ProfileService._map_profile` turns
the gateway's `nutritional_preferences.calories` into the PROSE string
`"2000 calories target"` and puts it in `preferences`, so the daily path's
target has always resolved to None. Every consequence follows from that: the
plate critic's calorie fit ranked on nothing, `kcal_by_slot` was empty, and
`plan_verifier._check_kcal` returned before it measured. A day could come back
900 kcal over the member's own target without a word.

**Two different numbers, and the distinction is the whole module.**

`stated_target` is the member's. It may shape the plan — rank plates, split a
meal's budget, fail a check — because they asked for it.

`reference_for` may return a number NOBODY set, and that number must never
shape a plan. It exists so a day can be *reported* against something ("2,470
kcal against a 2,000 kcal reference"), and it carries `chosen=False` and a
`basis` sentence naming where it came from, so no surface can render it as the
member's own. This is the lesson of the weekly meat limit, which was FoodChat's
own default rendered as `source: "dietary preference"` with an apology
addressed to a member who never set one.

**On sex, and on what this can honestly say.** Energy requirements track body
size, composition and activity. FoodChat has none of those. The only signal
available is a free-text `gender` field on the gateway profile, which is not
the same thing as sex and was never collected for this purpose. So it is used
as a coarse proxy for picking a POPULATION REFERENCE, never as a requirement
calculated for this person; anything the map does not recognise — unset,
non-binary, declined — falls to the ungendered figure rather than a guess. And
for anyone who is not an adult the answer is no number at all, because the
range across childhood and adolescence is far too wide for one to mean
anything.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# EU Regulation 1169/2011 Annex XIII — the reference intake for an average
# adult, the figure every nutrition label in the EU is expressed against. It is
# deliberately not split by sex, which is why it is the fallback here.
EU_REFERENCE_INTAKE = 2000.0

# Commonly published average daily energy requirements for adults (EFSA's
# dietary reference values for energy at a moderate activity level; the same
# rounded pair the NHS publishes). A population average at an assumed activity
# level, which is why nothing here may rank or filter on it.
ADULT_REFERENCE = {"male": 2500.0, "female": 2000.0}

# Age groups that are adults. Anything else gets no number: energy needs across
# childhood and adolescence vary by several hundred kcal a year, and a single
# figure spanning them would be worse than saying nothing.
_ADULT_AGE_GROUPS = frozenset({"adult", "adults", "senior", "seniors", "older_adult", ""})

# What the gender field may say, mapped onto the two rows the reference table
# has. Everything else — non-binary, other, prefer-not-to-say, unset — lands on
# the ungendered EU figure, which is the honest answer rather than a coin toss.
_SEX_PROXY = {
    "male": "male", "m": "male", "man": "male", "men": "male",
    "female": "female", "f": "female", "woman": "female", "women": "female",
}

# "2000 calories target" — the string `_build_preferences` writes. The number
# may carry a separator ("2,000") and need not lead the line.
_TARGET_IN_PREFERENCE = re.compile(
    r"(\d[\d,.]*)\s*(?:kcal|calorie|calories)\b[^.;]*\btarget\b"
    r"|\btarget\b[^.;]*?(\d[\d,.]*)\s*(?:kcal|calorie|calories)\b",
    re.IGNORECASE,
)

# A day's calories outside a plausible band is a data problem, not a target.
# RecipeWrangler has served 6 kcal pancakes and 14,000 kcal weeks.
MIN_PLAUSIBLE_KCAL = 800.0
MAX_PLAUSIBLE_KCAL = 6000.0


@dataclass(frozen=True)
class EnergyReference:
    """A daily calorie figure and an honest account of where it came from."""

    kcal: float
    # True only when the MEMBER set it. Everything that ranks, filters or
    # apportions must check this before using the number.
    chosen: bool
    # A sentence naming the number's origin, for the member to read. Never
    # empty, because a number on a plan with no stated origin is the failure
    # this class exists to prevent.
    basis: str

    @property
    def source(self) -> str:
        """The `source` a ledger row should carry."""
        return "your calorie target" if self.chosen else "a population reference"


def _number(text: str, *, whose: str = "") -> Optional[float]:
    """A plausible daily calorie figure, or None — and a log line either way.

    The band rejects data errors, and it also rejects a real number somebody
    typed. Dropping one of those silently is how a member ends up planned
    against something they did not choose with nothing anywhere saying so, so
    the rejection is logged with the value that caused it.
    """
    try:
        value = float(str(text).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    if MIN_PLAUSIBLE_KCAL <= value <= MAX_PLAUSIBLE_KCAL:
        return value
    logger.warning(
        "Ignoring %s daily calorie figure of %g — outside the %g–%g band a "
        "day's eating can plausibly fall in, so it is treated as a data error "
        "rather than planned against",
        whose or "a", value, MIN_PLAUSIBLE_KCAL, MAX_PLAUSIBLE_KCAL,
    )
    return None


def stated_target(profile: dict) -> Optional[float]:
    """The daily calorie target the MEMBER set, or None.

    Reads both the structured keys and the prose string the gateway mapping
    produces, because which of the two a profile carries depends on how it was
    built — and a reader that knows only one of them is how the daily path
    ended up measuring against nothing.
    """
    profile = profile or {}
    for key in ("calorie_target", "daily_kcal_target"):
        raw = profile.get(key)
        if raw in (None, ""):
            continue
        value = _number(raw, whose=f"the profile's {key}")
        if value:
            return value
    for preference in profile.get("preferences") or ():
        match = _TARGET_IN_PREFERENCE.search(str(preference))
        if not match:
            continue
        value = _number(match.group(1) or match.group(2), whose="the stated")
        if value:
            return value
    return None


def sex_proxy(profile: dict) -> Optional[str]:
    """`"male"` / `"female"` from the profile, or None when it cannot say.

    None is a real answer here and the common one. See the module docstring:
    the field is a gender field, it is free text, and it is being read as a
    coarse proxy for a population row — so anything outside the two rows that
    table has falls through to the figure that needs no such split.
    """
    prefs = (profile or {}).get("nutritional_preferences") or {}
    for key in ("sex", "gender", "biological_sex"):
        raw = str((profile or {}).get(key) or prefs.get(key) or "").strip().lower()
        if raw in _SEX_PROXY:
            return _SEX_PROXY[raw]
    return None


def reference_for(profile: dict) -> Optional[EnergyReference]:
    """The number a day's calories may be REPORTED against, with its provenance.

    `None` when nothing honest can be said — which is the answer for anyone who
    is not an adult, and it is a better one than a figure invented to fill the
    slot.
    """
    profile = profile or {}

    stated = stated_target(profile)
    if stated:
        return EnergyReference(
            kcal=stated, chosen=True,
            basis="the daily calorie target on your profile",
        )

    age_group = str(profile.get("age_group") or "").strip().lower()
    if age_group not in _ADULT_AGE_GROUPS:
        logger.debug(
            "No energy reference for age group %r — the range across childhood "
            "is too wide for one figure", age_group,
        )
        return None

    sex = sex_proxy(profile)
    if sex in ADULT_REFERENCE:
        return EnergyReference(
            kcal=ADULT_REFERENCE[sex], chosen=False,
            basis=(
                f"a published average daily requirement for an adult "
                f"{'man' if sex == 'male' else 'woman'} at moderate activity — "
                "a population figure, not a calculation for you"
            ),
        )
    return EnergyReference(
        kcal=EU_REFERENCE_INTAKE, chosen=False,
        basis=(
            "the EU reference intake for an average adult (the figure on "
            "nutrition labels) — a population figure, not a calculation for you"
        ),
    )
