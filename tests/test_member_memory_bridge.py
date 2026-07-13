"""dietary_goal memory kind — worries/objectives said in chat steer planning.

The FoodScholar app has its own consent flow writing properties.dietary_goals;
this covers the FoodChat side of the same loop (including nutrition questions
bridged through chat). No LLM calls — extractor and profile service are fakes.
"""

import uuid

from services.memory_service import MemoryService


class FakeExtractor:
    def __init__(self, memories):
        self._memories = memories

    def extract(self, message):
        return self._memories


class FakeProfileService:
    """Records writes instead of PATCHing the gateway."""

    def __init__(self):
        self.applied = []
        self.optouts = []

    def apply_memory(self, member_id, kind, value, session_id):
        self.applied.append((member_id, kind, value))
        return True

    def record_memory_optout(self, member_id, value):
        self.optouts.append((member_id, value))
        return True


def _svc(session_service, memories, profile_service=None):
    return MemoryService(
        session_service,
        profile_service or FakeProfileService(),
        extractor=FakeExtractor(memories),
    )


GOAL_CAND = {
    "kind": "dietary_goal", "value": "reduce_fat",
    "statement": "You mentioned watching your cholesterol — aim for lower-fat plans?",
    "evidence": "user said 'my cholesterol is high'", "confidence": "high",
}


class TestDietaryGoalKind:
    def test_goal_candidate_is_suggested(self, session_service, sample_profile):
        svc = _svc(session_service, [GOAL_CAND])
        session = session_service.create_session(f"member-{uuid.uuid4()}", dict(sample_profile))
        suggestions = svc.suggest(session, "my cholesterol is high")
        assert [s["kind"] for s in suggestions] == ["dietary_goal"]
        assert suggestions[0]["value"] == "reduce_fat"

    def test_off_list_goal_slug_is_dropped(self, session_service, sample_profile):
        """Only canonical planner slugs may be written — anything else would
        sit in the profile and never be acted on."""
        svc = _svc(session_service, [{**GOAL_CAND, "value": "become_immortal"}])
        session = session_service.create_session(f"member-{uuid.uuid4()}", dict(sample_profile))
        assert svc.suggest(session, "…") == []

    def test_existing_goal_not_resuggested(self, session_service, sample_profile):
        profile = {**sample_profile, "dietary_goals": ["reduce_fat"]}
        svc = _svc(session_service, [GOAL_CAND])
        session = session_service.create_session(f"member-{uuid.uuid4()}", profile)
        assert svc.suggest(session, "…") == []

    def test_accept_writes_durably_and_syncs_live_session(self, session_service, sample_profile):
        """Accepted goal must shape the very next plan: slug + soft preference
        string + hard diet tag (increase_protein → high-protein), exactly as a
        fresh profile fetch would map it."""
        fake_profiles = FakeProfileService()
        svc = _svc(session_service, [], profile_service=fake_profiles)
        session = session_service.create_session(f"member-{uuid.uuid4()}", dict(sample_profile))

        applied = svc.decide(
            session, {**GOAL_CAND, "value": "increase_protein"}, "accept"
        )
        assert applied
        assert fake_profiles.applied == [
            (session.member_id, "dietary_goal", "increase_protein"),
        ]
        assert "increase_protein" in session.user_profile["dietary_goals"]
        assert "prefers higher-protein meals" in session.user_profile["preferences"]
        assert "high-protein" in session.user_profile["diet"]

    def test_decline_records_optout(self, session_service, sample_profile):
        fake_profiles = FakeProfileService()
        svc = _svc(session_service, [], fake_profiles)
        session = session_service.create_session(f"member-{uuid.uuid4()}", dict(sample_profile))
        applied = svc.decide(session, GOAL_CAND, "decline")
        assert applied is False
        assert fake_profiles.optouts == [(session.member_id, "reduce_fat")]
        assert fake_profiles.applied == []
