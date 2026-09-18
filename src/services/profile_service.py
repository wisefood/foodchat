import datetime as _dt
import logging
import time
from typing import Any

from backend.platform import WISEFOOD, WiseFoodPool

logger = logging.getLogger(__name__)


# Dietary-goal slugs (written by FoodScholar into properties.dietary_goals)
# that map onto NUMERIC per-serving targets for RecipeWrangler's
# nutrition_profile candidate filter. The corpus carries NO low-fat/low-carb/
# high-protein tags (verified against recipes_v2: the diet-tag vocabulary is
# vegan/vegetarian/pescatarian/dairy_free/nut_free/gluten_free), so mapping
# goals to hard diet tags returned zero candidates for every goal-carrying
# member. Numeric targets are the mechanism RecipeWrangler actually supports —
# and it applies them leniently (recipes with missing nutrition pass through).
GOAL_TO_NUTRITION_PROFILE: dict[str, dict[str, float]] = {
    "reduce_fat": {"max_fat_g": 20},
    "reduce_carbs": {"max_carbs_g": 45},
    "reduce_calories": {"max_calories": 650},
    "lose_weight": {"max_calories": 650},
    "increase_protein": {"min_protein_g": 20},
    "gain_muscle": {"min_protein_g": 20},
}


# Goals that imply a Nutri-Score floor as well as a macro target.
#
# A Nutri-Score is a whole-recipe judgement — composition, not one nutrient —
# so it expresses "generally healthier" in a way a macro bound cannot. Someone
# losing weight is served better by a B-or-better 600 kcal meal than by any
# 600 kcal meal, and the score is already on every recipe.
#
# Deliberately lenient: "B" admits A and B, which is roughly half the corpus.
# A floor of "A" would be a stricter filter than any of these goals implies,
# and RecipeWrangler relaxes nothing here — an over-tight floor returns an
# empty slot rather than a slightly worse meal.
# The dietary_groups the gateway will accept from us: the real Postgres enum,
# not the five values the UI's picker happens to offer.
#
# Restricting this to the picker's five was a mistake with a precise cost: the
# enum also holds `gluten_free`, `dairy_free` and `nut_free`, which are exactly
# the three diets FoodChat CAN filter on. So "remember I'm gluten-free" was
# offered, accepted by the member, refused here, and returned applied=false —
# the three it could act on were the three it would not persist.
#
# `diabetic_friendly` is included because the gateway's own enum copies have
# been reconciled; anything outside this set is still refused rather than
# risking a 422 on a value the column does not know.
GATEWAY_DIET_GROUPS = {
    # dietary patterns
    "omnivore", "vegetarian", "lacto_vegetarian", "ovo_vegetarian",
    "lacto_ovo_vegetarian", "pescatarian", "vegan", "raw_vegan", "plant_based",
    "flexitarian",
    # religious / cultural
    "halal", "kosher", "jain", "buddhist_vegetarian",
    # free-from
    # No `lactose_free` — it is in FREE_FROM_TO_ALLERGEN for backstop purposes
    # but NOT in the gateway enum, so writing it would 422.
    "gluten_free", "dairy_free", "nut_free", "peanut_free",
    "egg_free", "soy_free", "shellfish_free", "fish_free", "sesame_free",
    # nutrition-flavoured
    "low_carb", "low_fat", "low_sodium", "sugar_free", "no_added_sugar",
    "high_protein", "high_fiber", "low_cholesterol", "low_calorie",
    "keto", "paleo", "whole30", "mediterranean", "diabetic_friendly",
}

GOAL_TO_MIN_NUTRI_SCORE: dict[str, str] = {
    "reduce_fat": "C",
    "reduce_carbs": "C",
    "reduce_calories": "B",
    "lose_weight": "B",
}

# Worst to best, for picking the strictest floor across several goals.
_NUTRI_RANK = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}


def goals_min_nutri_score(goal_slugs: list[str]) -> str | None:
    """The strictest Nutri-Score floor implied by a member's goals.

    None when no goal implies one, which leaves the filter off entirely rather
    than defaulting to a floor nobody asked for.
    """
    floors = [
        GOAL_TO_MIN_NUTRI_SCORE[slug]
        for slug in goal_slugs
        if slug in GOAL_TO_MIN_NUTRI_SCORE
    ]
    if not floors:
        return None
    return min(floors, key=lambda label: _NUTRI_RANK[label])


def goals_nutrition_profile(goal_slugs: list[str]) -> dict[str, float]:
    """Merge the numeric targets of all mapped goals (strictest bound wins)."""
    merged: dict[str, float] = {}
    for slug in goal_slugs:
        for key, value in GOAL_TO_NUTRITION_PROFILE.get(slug, {}).items():
            if key.startswith("max_"):
                merged[key] = min(merged[key], value) if key in merged else value
            else:
                merged[key] = max(merged[key], value) if key in merged else value
    return merged

# Human-readable soft-signal strings for every goal slug, fed into the planner's
# `preferences` list so the LLM grader steers even when there's no hard tag.
GOAL_PREFERENCE_STRINGS = {
    "reduce_fat": "prefers lower-fat meals",
    "reduce_sugar": "prefers lower-sugar meals",
    "reduce_sodium": "prefers lower-sodium meals",
    "reduce_calories": "prefers lower-calorie meals",
    "reduce_carbs": "prefers lower-carb meals",
    "increase_protein": "prefers higher-protein meals",
    "increase_fiber": "prefers higher-fiber meals",
    "increase_hydration": "prefers hydrating meals",
    "lose_weight": "goal: weight loss (favor lighter, filling meals)",
    "gain_weight": "goal: weight gain (favor calorie-dense meals)",
    "gain_muscle": "goal: muscle gain (favor high-protein meals)",
    "maintain_weight": "goal: weight maintenance",
}


def goal_preference_strings(goal_slugs: list[str]) -> list[str]:
    """Map dietary-goal slugs to soft preference strings for the LLM grader."""
    return [GOAL_PREFERENCE_STRINGS[s] for s in goal_slugs if s in GOAL_PREFERENCE_STRINGS]


class ProfileService:
    """Service for fetching and mapping WiseFood member profiles."""

    def __init__(self, client_pool: WiseFoodPool = None):
        self.client_pool = WISEFOOD

    def get_member_profile(self, member_id: str) -> dict:
        logger.info("Fetching profile for member %s from WiseFood.", member_id)
        try:
            with self.client_pool.client() as client:
                member = client.members.get(member_id)
                profile = self._map_profile(member.profile)
                # WHO this member is, not only what they eat.
                #
                # `age_group` is a first-class field on the member and
                # `region` belongs to their household; neither reached
                # FoodChat, so `guidelines_service.resolve_scope` — which
                # reads both — fell back to the deployment default for
                # everybody. Regional guidelines were applied, and an Irish
                # household and a Hungarian one were judged by the same rules.
                self._attach_context(client, member, profile)
            logger.info(
                "Profile fetched for member %s — diet=%s allergies=%s food_likes=%d food_dislikes=%d.",
                member_id,
                profile.get("diet"),
                profile.get("allergies"),
                len(profile.get("food_likes", [])),
                len(profile.get("food_dislikes", [])),
            )
            return profile
        except Exception as e:
            logger.error("Failed to fetch profile for member %s: %s", member_id, e, exc_info=True)
            raise

    def get_member_favorites(self, member_id: str) -> list[str]:
        """Fetch the member's favorited recipe ids from the gateway (M2).

        Best-effort: favorites are a soft personalization signal, so any
        failure degrades to an empty list rather than blocking the session.
        """
        try:
            with self.client_pool.client() as client:
                response = client.get(f"members/{member_id}/favorites")
                payload = response.json()
            rows = payload.get("result", payload) or []
            favorites = [r["recipe_id"] for r in rows if isinstance(r, dict) and r.get("recipe_id")]
            logger.info("Fetched %d favorites for member %s.", len(favorites), member_id)
            return favorites
        except Exception as e:
            logger.warning("Could not fetch favorites for member %s: %s", member_id, e)
            return []

    def get_member_adapted_recipes(self, member_id: str) -> dict[str, dict]:
        """Fetch the member's saved adapted recipes from the gateway.

        Returns {original_recipe_id: {"title": ..., "payload": {...}}}.
        Adapted recipes are strictly owner-scoped at the gateway; this call
        reads only the authorized member's own adaptations. Best-effort:
        failures degrade to an empty map rather than blocking the session.
        """
        try:
            with self.client_pool.client() as client:
                response = client.get(f"members/{member_id}/adapted-recipes")
                payload = response.json()
            rows = payload.get("result", payload) or []
            adapted = {
                r["recipe_id"]: {"title": r.get("title"), "payload": r.get("payload") or {}}
                for r in rows
                if isinstance(r, dict) and r.get("recipe_id")
            }
            logger.info("Fetched %d adapted recipes for member %s.", len(adapted), member_id)
            return adapted
        except Exception as e:
            logger.warning("Could not fetch adapted recipes for member %s: %s", member_id, e)
            return {}

    # ------------------------------------------------------------------ #
    # Consented memory write-back (M3)                                     #
    # ------------------------------------------------------------------ #
    # The SDK profile object auto-PATCHes the gateway on attribute
    # assignment (sync=True), so each field write below persists
    # immediately. Every durable write carries provenance in
    # properties.memory_log — personalization must stay auditable.

    @staticmethod
    def _ensure_profile_row(client, member_id: str, profile):
        """Create the member's profile row, then hand back a writable handle.

        Only PATCH /members/{id}/profile is lossless on the gateway, so the row
        has to exist before anything is written into it. Best-effort: a failure
        here means the caller reports "not applied" rather than claiming a write
        that did not happen.
        """
        try:
            client.post(
                f"members/{member_id}/profile",
                json={
                    "nutritional_preferences": {},
                    "dietary_groups": [],
                    "allergies": [],
                    "properties": {},
                },
            )
            logger.info("Created missing profile row for member %s", member_id)
            return client.members.get(member_id).profile
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not create a profile row for member %s: %s", member_id, exc
            )
            return None

    def apply_memory(
        self,
        member_id: str,
        kind: str,
        value: str,
        session_id: str,
        evidence: str = "",
    ) -> bool:
        """Persist a user-CONSENTED memory to the durable member profile.

        kind → destination:
          like/cuisine   → nutritional_preferences.food_likes
          dislike        → nutritional_preferences.food_dislikes
          allergy_hint   → allergies (safety field — only ever via consent)
          standing_seed  → properties.standing_seeds [{name}]
          constraint     → properties.feedback_history (appended line)
          dietary_goal   → properties.dietary_goals [{slug, label}] — the same
                           field FoodScholar's own consent flow writes, so both
                           apps converge on one goal store the planner reads
          diet           → dietary_groups (the gateway's enum column; replaces
                           a non-restrictive "omnivore" rather than sitting
                           beside it)
        """
        try:
            with self.client_pool.client() as client:
                member = client.members.get(member_id)
                profile = member.profile
                # A guest household is created with a member and no profile
                # ROW, so the gateway 404s and the SDK swallows it into an
                # empty object. Every field write below then patched nothing
                # and every guest memory acceptance was a silent no-op — the
                # member said yes and we agreed and stored nothing. Create the
                # row first so consent means something.
                if profile is None or getattr(profile, "id", None) is None:
                    profile = self._ensure_profile_row(client, member_id, profile)
                    if profile is None:
                        return False
                prefs = dict(profile.nutritional_preferences or {})
                props = dict(profile.properties or {})

                value_norm = value.strip().lower()
                # Every branch sets `changed` to whether it made a durable edit.
                # Re-accepting a memory already on the profile must be a no-op:
                # no second memory_log entry, no needless write. (Applies to all
                # kinds — the original code only guarded the value lists, then
                # logged unconditionally, producing duplicate log rows.)
                changed = False
                if kind in ("like", "cuisine"):
                    likes = list(prefs.get("food_likes") or [])
                    if value_norm not in [v.lower() for v in likes]:
                        likes.append(value_norm)
                        changed = True
                    prefs["food_likes"] = likes
                    # Contradiction resolution: liking something removes it from
                    # dislikes — itself a durable change worth recording.
                    dislikes_before = prefs.get("food_dislikes") or []
                    dislikes_after = [v for v in dislikes_before if v.lower() != value_norm]
                    if len(dislikes_after) != len(dislikes_before):
                        changed = True
                    prefs["food_dislikes"] = dislikes_after
                    profile.nutritional_preferences = prefs
                elif kind == "dislike":
                    dislikes = list(prefs.get("food_dislikes") or [])
                    if value_norm not in [v.lower() for v in dislikes]:
                        dislikes.append(value_norm)
                        changed = True
                    prefs["food_dislikes"] = dislikes
                    likes_before = prefs.get("food_likes") or []
                    likes_after = [v for v in likes_before if v.lower() != value_norm]
                    if len(likes_after) != len(likes_before):
                        changed = True
                    prefs["food_likes"] = likes_after
                    profile.nutritional_preferences = prefs
                elif kind == "allergy_hint":
                    allergies = list(profile.allergies or [])
                    if value_norm not in [a.lower() for a in allergies]:
                        allergies.append(value_norm)
                        changed = True
                    profile.allergies = allergies
                elif kind == "standing_seed":
                    seeds = list(props.get("standing_seeds") or [])
                    if value_norm not in [s.get("name", "").lower() for s in seeds]:
                        seeds.append({"name": value_norm})
                        changed = True
                    props["standing_seeds"] = seeds
                elif kind == "diet":
                    # The gateway column is a Postgres ENUM ARRAY, so only its
                    # own vocabulary can be written — an off-list value is
                    # rejected at the API boundary. The UI writes one of five.
                    if value_norm not in GATEWAY_DIET_GROUPS:
                        logger.warning(
                            "Diet %r is not a gateway dietary_group — not applied",
                            value_norm,
                        )
                        return False
                    groups = [str(d).strip().lower() for d in (profile.dietary_groups or [])]
                    if value_norm not in groups:
                        # "omnivore" is the absence of a restriction, not a
                        # restriction to keep beside a real one — leaving it in
                        # place is how a stated vegetarian kept planning as an
                        # omnivore.
                        groups = [g for g in groups if g != "omnivore"]
                        groups.append(value_norm)
                        profile.dietary_groups = groups
                        changed = True
                elif kind == "dietary_goal":
                    if value_norm not in GOAL_PREFERENCE_STRINGS:
                        logger.warning("Unknown dietary goal %r — not applied", value_norm)
                        return False
                    goals = list(props.get("dietary_goals") or [])
                    existing_slugs = [
                        str(g.get("slug", "")).lower() for g in goals if isinstance(g, dict)
                    ]
                    if value_norm not in existing_slugs:
                        goals.append({
                            "slug": value_norm,
                            "label": value_norm.replace("_", " ").capitalize(),
                        })
                        changed = True
                    props["dietary_goals"] = goals
                elif kind == "constraint":
                    # Free-text feedback is always new information.
                    history = props.get("feedback_history", "") or ""
                    props["feedback_history"] = (history + "\n" if history else "") + value
                    changed = True
                else:
                    logger.warning("Unknown memory kind %r — not applied", kind)
                    return False

                if not changed:
                    logger.info(
                        "Memory %s=%r already set for member %s — no-op",
                        kind, value_norm, member_id,
                    )
                    return False

                # Provenance: who learned what, where, when. Reached only when a
                # durable change was actually made above.
                log = list(props.get("memory_log") or [])
                entry = {
                    "kind": kind, "value": value_norm,
                    "source": "foodchat", "session_id": session_id,
                    "recorded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                }
                # What the member said that led here — the memory panel shows
                # it under "Why am I seeing this?". Omitted rather than stored
                # empty, so old entries and new ones read the same way.
                if evidence:
                    entry["evidence"] = evidence
                log.append(entry)
                props["memory_log"] = log
                profile.properties = props

            logger.info("Memory applied for member %s: %s=%r", member_id, kind, value_norm)
            return True
        except Exception as e:
            logger.error("Failed to apply memory for member %s: %s", member_id, e, exc_info=True)
            return False

    def record_memory_optout(self, member_id: str, value: str) -> bool:
        """Remember a declined suggestion so we never nag about it again."""
        try:
            with self.client_pool.client() as client:
                member = client.members.get(member_id)
                profile = member.profile
                props = dict(profile.properties or {})
                optouts = list(props.get("memory_optouts") or [])
                value_norm = value.strip().lower()
                if value_norm not in optouts:
                    optouts.append(value_norm)
                    props["memory_optouts"] = optouts
                    profile.properties = props
            return True
        except Exception as e:
            logger.error("Failed to record opt-out for member %s: %s", member_id, e, exc_info=True)
            return False

    # ------------------------------------------------------------------ #
    # Household diners (M3 — "who are we cooking for?")                    #
    # ------------------------------------------------------------------ #

    def get_member_name(self, member_id: str) -> str:
        try:
            with self.client_pool.client() as client:
                return client.members.get(member_id).name or member_id
        except Exception:
            return member_id

    # Session-scoped profile state that is NOT derived from member records:
    # rebuilding the profile (e.g. when diners change) must carry it over or
    # the session silently forgets what the member did in it.
    SESSION_SCOPED_KEYS = ("manual_picks", "plan_parameters")

    @classmethod
    def carry_session_state(cls, old_profile: dict, new_profile: dict) -> dict:
        """Re-apply session-scoped state onto a freshly rebuilt profile.

        ``history`` is special: the rebuilt profile starts from the member's
        durable feedback history, while the session may have appended facts
        collected during clarification or slider applies. Keep both.
        """
        for key in cls.SESSION_SCOPED_KEYS:
            if old_profile.get(key):
                new_profile[key] = old_profile[key]

        old_history = (old_profile.get("history") or "").strip()
        new_history = (new_profile.get("history") or "").strip()
        if old_history and old_history != new_history:
            extra = [
                line for line in old_history.splitlines()
                if line.strip() and line not in new_history
            ]
            if extra:
                new_profile["history"] = "\n".join(
                    part for part in [new_history, *extra] if part
                )
        return new_profile

    @staticmethod
    def merge_profiles(primary: dict, others: list[dict], diner_names: list[str]) -> dict:
        """Merge diner profiles into one plan-generation profile.

        Hard constraints (safety/identity) are the UNION across all diners:
        allergies, dietary groups, dislikes-as-exclusions. Soft preferences
        keep the primary member's weighting (their likes lead, other diners'
        likes follow). The primary member's favorites keep their boost.

        Goals are the delicate part. The primary member's goals keep their
        numeric targets (``nutrition_profile``, ``min_nutri_score``), because
        stacking several members' targets can empty the candidate set. Every
        other diner's goal is demoted to the soft ``preferences`` channel
        instead of being dropped, and the whole picture — whose goal, applied
        as a target or only as a signal — is recorded in
        ``goal_reconciliation`` so the plan can say so out loud. A goal that
        vanished silently is what made a household plan look like one member's
        plan.

        ``constraint_origins`` keeps the member each constraint came from, so
        the ledger can attribute "no peanuts" to the diner who is allergic
        rather than to everyone at the table.
        """
        merged = dict(primary)
        names = [str(n) for n in (diner_names or [])]
        primary_name = names[0] if names else ""
        # names is [primary, *others]; a short list just yields empty names
        other_names = names[1:]
        attributed = [(primary_name, primary)] + [
            (other_names[i] if i < len(other_names) else "", profile)
            for i, profile in enumerate(others)
        ]

        origins: dict[str, dict[str, list[str]]] = {}

        def union(key: str) -> list:
            seen: dict[str, str] = {}
            key_origins: dict[str, list[str]] = {}
            for name, profile in attributed:
                for item in profile.get(key) or []:
                    low = str(item).lower()
                    seen.setdefault(low, item)
                    who = key_origins.setdefault(low, [])
                    if name and name not in who:
                        who.append(name)
            # Re-key origins by the value as it will be displayed
            origins[key] = {seen[low]: who for low, who in key_origins.items()}
            return list(seen.values())

        merged["allergies"] = union("allergies")
        merged["diet"] = union("diet")
        merged["food_dislikes"] = union("food_dislikes")
        merged["constraint_origins"] = origins

        likes = list(primary.get("food_likes") or [])
        for other in others:
            for like in other.get("food_likes") or []:
                if str(like).lower() not in [str(seen).lower() for seen in likes]:
                    likes.append(like)
        merged["food_likes"] = likes

        # ── Goal reconciliation ──
        primary_goals = [str(s).lower() for s in (primary.get("dietary_goals") or [])]
        reconciliation: list[dict] = [
            {"slug": slug, "member": primary_name, "applied": "target"}
            for slug in primary_goals
        ]

        demoted: list[str] = []
        for name, profile in attributed[1:]:
            for slug in profile.get("dietary_goals") or []:
                slug = str(slug).lower()
                if slug in primary_goals:
                    # Two diners wanting the same thing is agreement, not conflict
                    reconciliation.append({"slug": slug, "member": name, "applied": "target"})
                    continue
                if slug not in demoted:
                    demoted.append(slug)
                reconciliation.append({"slug": slug, "member": name, "applied": "soft"})

        if demoted:
            preferences = list(merged.get("preferences") or [])
            for text in goal_preference_strings(demoted):
                if text not in preferences:
                    preferences.append(text)
            merged["preferences"] = preferences

        merged["goal_reconciliation"] = reconciliation
        merged["cooking_for_names"] = diner_names
        return merged

    # household_id -> (region, fetched_at). A household's country does not
    # change between turns, and this lookup sits on the profile fetch, which is
    # on the path of every plan.
    _region_cache: dict = {}
    _REGION_TTL_SECONDS = 3600

    def _attach_context(self, client, member, profile: dict) -> None:
        """Add the member's age group and their household's region.

        Best-effort by construction, like everything else the catalog feeds: a
        household lookup that fails costs the plan its regional guidelines and
        nothing else — `resolve_scope` falls back to the deployment default,
        which is exactly what happened for every member before this.
        """
        # Who this profile belongs to. Read by `plate_critic` as the stable
        # key for its tiebreak, so two members do not walk the same corpus
        # ranking top-down and see the same breakfasts forever.
        member_id = str(getattr(member, "id", "") or getattr(member, "member_id", "") or "")
        if member_id:
            profile["member_id"] = member_id

        age_group = str(getattr(member, "age_group", "") or "").strip().lower()
        if age_group:
            profile["age_group"] = age_group

        household_id = str(getattr(member, "household_id", "") or "").strip()
        if not household_id:
            return
        profile["household_id"] = household_id

        cached = self._region_cache.get(household_id)
        if cached and (time.time() - cached[1]) < self._REGION_TTL_SECONDS:
            if cached[0]:
                profile["region"] = cached[0]
            return
        try:
            household = client.households.get(household_id)
            region = str(getattr(household, "region", "") or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not read the household region for %s: %s", household_id, exc,
            )
            return
        self._region_cache[household_id] = (region, time.time())
        if region:
            profile["region"] = region

    def _map_profile(self, wisefood_profile: Any) -> dict:
        """Map WiseFood profile structure to FoodChat format.

        WiseFood structure:
        - dietary_groups: list (vegetarian, gluten_free, etc.)
        - allergies: list (shellfish, sesame, etc.)
        - nutritional_preferences: dict (calories, protein, fat, carbs,
          food_likes, food_dislikes)
        - properties: dict (age_group, feedback_history, liked_recipes)

        FoodChat structure:
        - diet: list[str]
        - allergies: list[str]
        - preferences: list[str]
        - history: str
        - food_likes: list[str]
        - food_dislikes: list[str]
        """
        # Handle both object attributes and dict-like access
        if hasattr(wisefood_profile, "dietary_groups"):
            dietary_groups = wisefood_profile.dietary_groups or []
            allergies = wisefood_profile.allergies or []
            nutritional_preferences = wisefood_profile.nutritional_preferences or {}
            properties = wisefood_profile.properties or {}
        else:
            dietary_groups = wisefood_profile.get("dietary_groups", []) or []
            allergies = wisefood_profile.get("allergies", []) or []
            nutritional_preferences = (
                wisefood_profile.get("nutritional_preferences", {}) or {}
            )
            properties = wisefood_profile.get("properties", {}) or {}

        # Dietary goals expressed elsewhere (e.g. FoodScholar Q&A) live under
        # properties.dietary_goals as [{"slug", "label"}]. Surface the slugs so
        # the planner can act on them (structural diet tags + soft preferences).
        dietary_goals = [
            str(g.get("slug", "")).strip().lower()
            for g in (properties.get("dietary_goals") or [])
            if isinstance(g, dict) and g.get("slug")
        ]

        # Goals become NUMERIC per-serving targets (nutrition_profile) — the
        # planner pipeline already forwards profile["nutrition_profile"] to
        # RecipeWrangler. Hard diet tags stay reserved for real diets from
        # dietary_groups (vegan, vegetarian, ...), which exist in the corpus.
        # Goals with no numeric mapping ride the soft `preferences` channel.
        goal_nutrition = goals_nutrition_profile(dietary_goals)

        return {
            "diet": list(dict.fromkeys(dietary_groups)),
            "nutrition_profile": goal_nutrition or None,
            # Whole-recipe quality floor, alongside the per-nutrient targets.
            # Only reaches the v2 planning surface; the v1 candidate endpoint
            # has no equivalent, so it is simply unused there.
            "min_nutri_score": goals_min_nutri_score(dietary_goals),
            "allergies": list(allergies),
            "preferences": self._build_preferences(nutritional_preferences, properties)
            + goal_preference_strings(dietary_goals),
            "history": properties.get("feedback_history", "") or "",
            # The member's own daily calorie target as a NUMBER, beside the
            # prose string `_build_preferences` writes.
            #
            # The prose was the only form, and `PlanBrief._kcal_target` reads
            # `calorie_target` — a key nothing ever set. So the weekly tracker
            # and the plan scorer both saw the member's target (they parse the
            # string) and the daily planner saw None: no calorie ranking, no
            # per-meal budget, and `plan_verifier._check_kcal` returning before
            # it measured anything. A day could land 900 kcal over the member's
            # own target without a word.
            "calorie_target": nutritional_preferences.get("calories") or None,
            # Only what an energy reference can use, and only from what the
            # member already put on their profile. `nutritional_preferences` is
            # a free-form blob; carrying it whole would put arbitrary gateway
            # keys into every prompt and log that renders a profile.
            "sex": nutritional_preferences.get("sex")
            or nutritional_preferences.get("gender")
            or None,
            "food_likes": list(nutritional_preferences.get("food_likes", []) or []),
            "food_dislikes": list(nutritional_preferences.get("food_dislikes", []) or []),
            "dietary_goals": dietary_goals,
            # M3 memory fields (written only through the consent flow)
            "standing_seeds": list(properties.get("standing_seeds", []) or []),
            "memory_optouts": list(properties.get("memory_optouts", []) or []),
            "memory_log": list(properties.get("memory_log", []) or []),
        }

    def _build_preferences(
        self, nutritional_prefs: dict, properties: dict
    ) -> list[str]:
        """Convert nutritional preferences and properties to preference strings."""
        prefs = []

        # Nutritional preferences
        if nutritional_prefs.get("calories"):
            prefs.append(f"{nutritional_prefs['calories']} calories target")

        if nutritional_prefs.get("protein"):
            prefs.append(f"high protein ({nutritional_prefs['protein']}g)")

        if nutritional_prefs.get("fiber"):
            prefs.append(f"high fiber ({nutritional_prefs['fiber']}g)")

        if nutritional_prefs.get("carbs"):
            prefs.append(f"{nutritional_prefs['carbs']}g carbs")

        if nutritional_prefs.get("fat"):
            prefs.append(f"{nutritional_prefs['fat']}g fat")

        # Additional preferences from properties.notes if structured
        notes = properties.get("notes", "")
        if notes and "prefers" in notes.lower():
            # Extract cuisine preferences from notes if present
            prefs.append(notes)

        return prefs

    def format_profile_for_display(self, profile: dict) -> str:
        """Format a profile dict for display/logging."""
        lines = []
        if profile.get("diet"):
            lines.append(f"Diet: {', '.join(profile['diet'])}")
        if profile.get("allergies"):
            lines.append(f"Allergies: {', '.join(profile['allergies'])}")
        if profile.get("preferences"):
            lines.append(f"Preferences: {', '.join(profile['preferences'])}")
        if profile.get("history"):
            lines.append(f"Notes: {profile['history']}")
        return "\n".join(lines) if lines else "No profile information"
