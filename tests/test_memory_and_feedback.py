"""M3 — consented memory (nudge policy + decisions), feedback signals, diner merge."""

import json
import uuid

import pytest

from services.feedback_service import FeedbackService
from services.memory_service import MemoryService
from services.profile_service import ProfileService


class FakeExtractor:
    def __init__(self, memories):
        self._memories = memories

    def extract(self, message):
        return self._memories


class FakeProfileService:
    """Records apply/optout calls instead of PATCHing the gateway."""

    def __init__(self):
        self.applied = []
        self.optouts = []

    def apply_memory(self, member_id, kind, value, session_id):
        self.applied.append((member_id, kind, value))
        return True

    def record_memory_optout(self, member_id, value):
        self.optouts.append((member_id, value))
        return True


def _mk_memory_service(session_service, memories, profile_service=None):
    return MemoryService(
        session_service,
        profile_service or FakeProfileService(),
        extractor=FakeExtractor(memories),
    )


def _session(session_service, profile):
    return session_service.create_session(f"member-{uuid.uuid4()}", profile)


CAND = {
    "kind": "dislike", "value": "blueberries",
    "statement": "It seems you don't like blueberries — remember this?",
    "evidence": "user said 'I hate blueberries'", "confidence": "high",
}


class TestNudgePolicy:
    def test_high_confidence_candidate_is_suggested(self, session_service, sample_profile):
        svc = _mk_memory_service(session_service, [CAND])
        session = _session(session_service, sample_profile)
        suggestions = svc.suggest(session, "I hate blueberries")
        assert len(suggestions) == 1
        assert suggestions[0]["kind"] == "dislike"
        assert suggestions[0]["value"] == "blueberries"
        assert suggestions[0]["id"]

    def test_low_confidence_is_filtered_except_allergy(self, session_service, sample_profile):
        low = {**CAND, "confidence": "low"}
        allergy = {**CAND, "kind": "allergy_hint", "value": "shrimp", "confidence": "low"}
        svc = _mk_memory_service(session_service, [low, allergy])
        session = _session(session_service, sample_profile)
        suggestions = svc.suggest(session, "…")
        assert [s["kind"] for s in suggestions] == ["allergy_hint"]

    def test_known_values_not_resuggested(self, session_service, sample_profile):
        # sample_profile already dislikes olives and is allergic to peanuts
        svc = _mk_memory_service(session_service, [
            {**CAND, "value": "olives"},
            {**CAND, "kind": "allergy_hint", "value": "peanuts"},
        ])
        session = _session(session_service, sample_profile)
        assert svc.suggest(session, "…") == []

    def test_optouts_never_resuggested(self, session_service, sample_profile):
        profile = {**sample_profile, "memory_optouts": ["blueberries"]}
        svc = _mk_memory_service(session_service, [CAND])
        session = _session(session_service, profile)
        assert svc.suggest(session, "…") == []

    def test_suggestions_capped_per_turn(self, session_service, sample_profile):
        many = [{**CAND, "value": f"item-{i}"} for i in range(5)]
        svc = _mk_memory_service(session_service, many)
        session = _session(session_service, sample_profile)
        assert len(svc.suggest(session, "…")) == 2


class TestMemoryDecisions:
    def test_accept_applies_and_updates_session(self, session_service, sample_profile):
        fake_profiles = FakeProfileService()
        svc = _mk_memory_service(session_service, [], fake_profiles)
        session = _session(session_service, sample_profile)

        applied = svc.decide(session, CAND, "accept")
        assert applied
        assert fake_profiles.applied == [(session.member_id, "dislike", "blueberries")]
        # Session profile updated immediately — the next plan honors it
        assert "blueberries" in session.user_profile["food_dislikes"]

    def test_accept_persists_across_restart(self, session_service, sample_profile):
        from services.session_service import SessionService

        svc = _mk_memory_service(session_service, [], FakeProfileService())
        session = _session(session_service, sample_profile)
        svc.decide(session, {**CAND, "kind": "standing_seed", "value": "pastitsio"}, "accept")

        restored = SessionService().get_session(session.session_id)
        assert {"name": "pastitsio"} in restored.user_profile["standing_seeds"]

    def test_decline_records_optout(self, session_service, sample_profile):
        fake_profiles = FakeProfileService()
        svc = _mk_memory_service(session_service, [], fake_profiles)
        session = _session(session_service, sample_profile)

        applied = svc.decide(session, CAND, "decline")
        assert not applied
        assert fake_profiles.optouts == [(session.member_id, "blueberries")]
        assert "blueberries" in session.user_profile["memory_optouts"]

    def test_invalid_payload_rejected(self, session_service, sample_profile):
        svc = _mk_memory_service(session_service, [], FakeProfileService())
        session = _session(session_service, sample_profile)
        with pytest.raises(ValueError):
            svc.decide(session, {"kind": "hack", "value": "x"}, "accept")


class TestFeedbackSignals:
    def _plan_with_feedback(self, session_service, sample_profile, rating, comment=None):
        """Create session → plan → assistant message tagged with plan → feedback."""
        from conftest import make_candidates
        from db import SessionLocal, db_upsert_feedback

        session = _session(session_service, sample_profile)
        plan = session_service.add_meal_plan(session.session_id, make_candidates("fb"), "r", {})
        session_service.add_message(
            session.session_id, "assistant", "here is your plan", plan_id=plan.id
        )
        # The in-memory Message has no DB id — fetch it like the router does.
        message_id = session_service.get_messages_page(session.session_id, limit=5)[-1]["id"]
        db = SessionLocal()
        try:
            db_upsert_feedback(db, message_id=message_id,
                               session_id=session.session_id,
                               member_id=session.member_id, rating=rating, comment=comment)
        finally:
            db.close()
        return session, plan

    def test_downvote_excludes_plan_recipes(self, session_service, sample_profile):
        from db import SessionLocal, db_get_messages

        session, plan = self._plan_with_feedback(session_service, sample_profile, "down", "too heavy")
        # feedback row above needs the real message id — fetch it
        db = SessionLocal()
        try:
            rows = db_get_messages(db, session.session_id, limit=5)
        finally:
            db.close()
        assert rows, "message should be persisted"

        signals = FeedbackService().get_signals(session.member_id)
        assert set(signals.downvoted_recipe_ids) == {"fb-b", "fb-l", "fb-d"}
        assert "Disliked" in signals.history_text
        assert "too heavy" in signals.history_text

    def test_upvote_keeps_recipes(self, session_service, sample_profile):
        session, _ = self._plan_with_feedback(session_service, sample_profile, "up")
        signals = FeedbackService().get_signals(session.member_id)
        assert signals.downvoted_recipe_ids == []
        assert "Liked" in signals.history_text

    def test_no_feedback_no_signals(self, session_service, sample_profile):
        session = _session(session_service, sample_profile)
        signals = FeedbackService().get_signals(session.member_id)
        assert signals.downvoted_recipe_ids == []
        assert signals.history_text == ""


class TestDinerMerge:
    def test_hard_constraints_union_soft_weighted(self):
        primary = {
            "diet": ["omnivore"], "allergies": ["peanuts"],
            "food_likes": ["salmon"], "food_dislikes": ["olives"],
            "preferences": ["2000 calories target"],
            "favorite_recipe_ids": ["r-1"],
        }
        anna = {
            "diet": ["vegetarian"], "allergies": [],
            "food_likes": ["halloumi", "salmon"], "food_dislikes": ["mushrooms"],
            "preferences": ["1600 calories target"],
        }
        tom = {
            "diet": [], "allergies": ["shellfish"],
            "food_likes": [], "food_dislikes": [],
            "preferences": [],
        }
        merged = ProfileService.merge_profiles(primary, [anna, tom], ["Me", "Anna", "Tom"])

        # Hard constraints: union across all diners
        assert set(merged["allergies"]) == {"peanuts", "shellfish"}
        assert set(merged["diet"]) == {"omnivore", "vegetarian"}
        assert set(merged["food_dislikes"]) == {"olives", "mushrooms"}
        # Soft: primary's likes lead, others follow without duplicates
        assert merged["food_likes"] == ["salmon", "halloumi"]
        # Primary keeps macros and favorites
        assert merged["preferences"] == ["2000 calories target"]
        assert merged["favorite_recipe_ids"] == ["r-1"]
        assert merged["cooking_for_names"] == ["Me", "Anna", "Tom"]
