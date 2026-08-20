"""
The constraints ledger must not claim more than it measured.

Two failure modes, both of which reached the member as prose:

* A ``relaxed`` or ``violated`` row handed to the response writer under
  ``constraints_honored`` — the writer is told to mention "an honored
  request", so it announced a goal as met while the ledger rendered beside it
  said the opposite.
* A memory accepted mid-session bypassed ``merge_profiles``, so its row had no
  attribution (``members: []``) next to properly attributed siblings, and an
  accepted goal had no row at all.
"""

import uuid

from services.memory_service import MemoryService
from services.transparency import constraints_ledger, split_ledger


class FakeExtractor:
    def __init__(self, memories):
        self._memories = memories

    def extract(self, message):
        return self._memories


class FakeProfileService:
    def apply_memory(self, member_id, kind, value, session_id, evidence=""):
        return True

    def record_memory_optout(self, member_id, value):
        return True


def _svc(session_service):
    return MemoryService(
        session_service, FakeProfileService(), extractor=FakeExtractor([])
    )


def _household_profile(**over) -> dict:
    profile = {
        "cooking_for_names": ["Ana", "Bruno"],
        "allergies": ["peanuts"],
        "food_dislikes": [],
        "diet": [],
        "constraint_origins": {"allergies": {"peanuts": ["Bruno"]}},
        "dietary_goals": ["reduce_fat"],
        "goal_reconciliation": [
            {"slug": "reduce_fat", "member": "Bruno", "applied": "target"},
        ],
    }
    profile.update(over)
    return profile


class TestSplitLedger:
    def test_relaxed_is_never_honored(self):
        honored, not_honored = split_ledger([
            {"constraint": "vegetarian", "status": "satisfied"},
            {"constraint": "increase protein", "status": "relaxed"},
        ])
        assert honored == ["vegetarian"]
        assert not_honored == ["increase protein"]

    def test_violated_is_never_honored(self):
        # The weekly meat limit is the row this ledger exists to police.
        honored, not_honored = split_ledger([
            {"constraint": "at most 3 meat meal(s) this week", "status": "violated"},
        ])
        assert honored == []
        assert not_honored == ["at most 3 meat meal(s) this week"]

    def test_an_unknown_status_is_claimed_neither_way(self):
        # Plans stored before the status field existed. Calling them honored is
        # a false positive; apologising for them is a false negative.
        honored, not_honored = split_ledger([{"constraint": "legacy row"}])
        assert honored == [] and not_honored == []

    def test_slicing_no_longer_smuggles_a_relaxed_row_in(self):
        # The original bug: [:4] over a mixed ledger put a relaxed goal fourth.
        ledger = [
            {"constraint": "vegetarian", "status": "satisfied"},
            {"constraint": "avoiding fish", "status": "satisfied"},
            {"constraint": "reduce fat", "status": "satisfied"},
            {"constraint": "increase protein", "status": "relaxed"},
        ]
        assert [c["constraint"] for c in ledger[:4]][-1] == "increase protein"
        honored, _ = split_ledger(ledger)
        assert "increase protein" not in honored

    def test_limit_applies_per_half(self):
        honored, not_honored = split_ledger(
            [{"constraint": f"h{i}", "status": "satisfied"} for i in range(6)]
            + [{"constraint": f"r{i}", "status": "relaxed"} for i in range(6)],
            limit=2,
        )
        assert honored == ["h0", "h1"] and not_honored == ["r0", "r1"]

    def test_empty_and_untitled_rows_are_ignored(self):
        assert split_ledger([]) == ([], [])
        assert split_ledger(None) == ([], [])
        assert split_ledger([{"status": "satisfied"}]) == ([], [])


class TestMidSessionAcceptIsAttributed:
    def test_an_accepted_allergy_names_the_member_who_accepted_it(
        self, session_service
    ):
        session = session_service.create_session(
            f"member-{uuid.uuid4()}", _household_profile()
        )
        _svc(session_service).decide(
            session,
            {"kind": "allergy_hint", "value": "shellfish",
             "statement": "?", "confidence": "high"},
            "accept",
        )
        rows = {r["constraint"]: r["members"] for r in
                constraints_ledger(session.user_profile)}
        assert rows["no shellfish"] == ["Ana"], "accepted allergy left unattributed"
        assert rows["no peanuts"] == ["Bruno"], "existing attribution disturbed"

    def test_an_accepted_dislike_is_attributed_too(self, session_service):
        session = session_service.create_session(
            f"member-{uuid.uuid4()}", _household_profile()
        )
        _svc(session_service).decide(
            session,
            {"kind": "dislike", "value": "olives",
             "statement": "?", "confidence": "high"},
            "accept",
        )
        rows = {r["constraint"]: r["members"] for r in
                constraints_ledger(session.user_profile)}
        assert rows["avoiding olives"] == ["Ana"]

    def test_an_accepted_goal_gets_a_ledger_row(self, session_service):
        session = session_service.create_session(
            f"member-{uuid.uuid4()}", _household_profile()
        )
        _svc(session_service).decide(
            session,
            {"kind": "dietary_goal", "value": "increase_protein",
             "statement": "?", "confidence": "high"},
            "accept",
        )
        rows = {r["constraint"] for r in constraints_ledger(session.user_profile)}
        assert "increase protein" in rows, "accepted goal had no row at all"

    def test_a_solo_session_gains_no_fake_household_record(self, session_service):
        # transparency attributes only when several people eat, so writing
        # origins here would invent a household of one.
        profile = _household_profile(
            cooking_for_names=["Ana"], constraint_origins={}, goal_reconciliation=[],
        )
        session = session_service.create_session(f"member-{uuid.uuid4()}", profile)
        _svc(session_service).decide(
            session,
            {"kind": "allergy_hint", "value": "shellfish",
             "statement": "?", "confidence": "high"},
            "accept",
        )
        assert not session.user_profile.get("constraint_origins")
        assert not session.user_profile.get("goal_reconciliation")
        # …but the constraint itself still applies and still renders.
        rows = {r["constraint"] for r in constraints_ledger(session.user_profile)}
        assert "no shellfish" in rows

    def test_accepting_a_goal_twice_does_not_duplicate_the_record(
        self, session_service
    ):
        session = session_service.create_session(
            f"member-{uuid.uuid4()}", _household_profile()
        )
        svc = _svc(session_service)
        cand = {"kind": "dietary_goal", "value": "reduce_fat",
                "statement": "?", "confidence": "high"}
        svc.decide(session, cand, "accept")
        slugs = [r["slug"] for r in session.user_profile["goal_reconciliation"]]
        assert slugs.count("reduce_fat") == 1
