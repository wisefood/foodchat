"""Session-scoped profile state must survive profile rebuilds, and the
plan-analyst context must name weekday correctly.

Both are things the member notices immediately when they break: hand-picked
dishes vanishing after adding a diner, and the assistant answering about the
wrong day. No LLM calls.
"""

import uuid

from conftest import make_candidates
from services.orchestrator_service import OrchestratorService
from services.profile_service import ProfileService


class TestCarrySessionState:
    def test_manual_picks_and_slider_values_survive_a_rebuild(self):
        old = {
            "diet": ["vegetarian"],
            "manual_picks": {"daily": [{"recipe_id": "rw-1", "meal_type": "lunch"}]},
            "plan_parameters": {"cooking_time": 20},
            "history": "durable line\nQ: cravings?\nA: something greek",
        }
        rebuilt = {"diet": ["vegetarian", "low-fat"], "history": "durable line"}

        merged = ProfileService.carry_session_state(old, rebuilt)

        assert merged["manual_picks"] == old["manual_picks"]
        assert merged["plan_parameters"] == {"cooking_time": 20}
        # The rebuilt durable history is kept AND the session-collected facts
        # are re-appended (not duplicated)
        assert merged["history"].splitlines() == [
            "durable line", "Q: cravings?", "A: something greek",
        ]
        # The rebuild's own updates are not clobbered
        assert merged["diet"] == ["vegetarian", "low-fat"]

    def test_nothing_invented_when_session_had_no_extras(self):
        merged = ProfileService.carry_session_state({}, {"diet": []})
        assert "manual_picks" not in merged
        assert "plan_parameters" not in merged

    def test_identical_history_is_not_duplicated(self):
        merged = ProfileService.carry_session_state(
            {"history": "same"}, {"history": "same"},
        )
        assert merged["history"] == "same"


class TestAnalystDayLabels:
    def test_weekly_days_are_labeled_1_based(self, session_service, sample_profile):
        """Entries are 1-based (1=Monday); labeling day 1 'Tuesday' made the
        analyst answer about the wrong day."""
        session = session_service.create_session(
            f"member-{uuid.uuid4()}", dict(sample_profile),
        )
        session_service.add_weekly_meal_plan(
            session.session_id,
            [
                {"day": 1, "meal_idx": 0, "meal_type": "breakfast",
                 "recipe": {"recipe_id": "r1", "title": "Monday oats"}, "reward": 0.0},
                {"day": 7, "meal_idx": 2, "meal_type": "dinner",
                 "recipe": {"recipe_id": "r7", "title": "Sunday roast"}, "reward": 0.0},
            ],
        )
        orch = OrchestratorService.__new__(OrchestratorService)
        orch.session_service = session_service

        summary = orch._summarize_active_plan(session)

        assert "Monday:" in summary
        assert "Sunday:" in summary
        assert "Day 8" not in summary
        monday_block = summary.split("Monday:")[1]
        assert "Monday oats" in monday_block.split("Sunday:")[0]

    def test_daily_plan_summary_still_works(self, session_service, sample_profile):
        session = session_service.create_session(
            f"member-{uuid.uuid4()}", dict(sample_profile),
        )
        session_service.add_meal_plan(session.session_id, make_candidates("d"), "r", {})
        orch = OrchestratorService.__new__(OrchestratorService)
        orch.session_service = session_service

        summary = orch._summarize_active_plan(session)
        assert "Oatmeal" in summary
