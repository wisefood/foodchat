"""What the console learns about meal plans.

The session page compares plans generated against plans kept — the one number
that says whether planning is working at all — and the tile read zero, because
`activity.report_event` had no call sites. Six methods produce a plan (daily,
weekly, structured, and a refinement of each) and exactly one act keeps it, so
these tests cover that all seven are wired and that none of them leaks anything
the member typed.
"""

import uuid

import pytest

from conftest import make_candidates


def _new_session(session_service, sample_profile):
    member_id = f"member-{uuid.uuid4()}"
    return session_service.create_session(member_id, sample_profile), member_id


@pytest.fixture()
def events(monkeypatch):
    """Every event the service reports, in order."""
    import activity

    seen = []
    monkeypatch.setattr(
        activity.wf_telemetry.TELEMETRY,
        "event",
        lambda event_type, **kw: seen.append((event_type, kw)),
    )
    return seen


def _of_type(events, event_type):
    return [props for name, props in events if name == event_type]


class TestPlanGeneration:
    def test_a_daily_plan_is_reported(self, session_service, sample_profile, events):
        session, _ = _new_session(session_service, sample_profile)
        plan = session_service.add_meal_plan(
            session.session_id, make_candidates("a"), "because", {}
        )
        (row,) = _of_type(events, "chat.plan_generated")
        assert row["props"] == {
            "session_id": session.session_id,
            "plan_id": plan.id,
            "plan_type": "daily",
            "version": 1,
            "refinement": False,
            "item_count": 3,
        }
        assert row["app"] == "foodchat"

    def test_a_refinement_is_reported_as_one(
        self, session_service, sample_profile, events
    ):
        session, _ = _new_session(session_service, sample_profile)
        session_service.add_meal_plan(
            session.session_id, make_candidates("a"), "first", {}
        )
        v2 = session_service.refine_meal_plan(
            session.session_id, make_candidates("b"), "second", {}
        )
        rows = _of_type(events, "chat.plan_generated")
        assert len(rows) == 2
        assert rows[1]["props"]["plan_id"] == v2.id
        assert rows[1]["props"]["version"] == 2
        assert rows[1]["props"]["refinement"] is True

    def test_a_refinement_without_a_canvas_reports_once_not_twice(
        self, session_service, sample_profile, events
    ):
        """`refine_meal_plan` delegates to `add_meal_plan` when there is nothing
        to refine from. Reporting in both would double every first plan."""
        session, _ = _new_session(session_service, sample_profile)
        session_service.refine_meal_plan(
            session.session_id, make_candidates("a"), "first", {}
        )
        assert len(_of_type(events, "chat.plan_generated")) == 1

    def test_a_weekly_plan_reports_its_own_type_and_size(
        self, session_service, sample_profile, events
    ):
        session, _ = _new_session(session_service, sample_profile)
        entries = [{"day": day, "meal": "lunch"} for day in range(1, 8)]
        plan = session_service.add_weekly_meal_plan(session.session_id, entries)
        (row,) = _of_type(events, "chat.plan_generated")
        assert row["props"]["plan_type"] == "weekly"
        assert row["props"]["plan_id"] == plan.id
        assert row["props"]["item_count"] == 7

    def test_nothing_the_member_typed_is_reported(
        self, session_service, sample_profile, events
    ):
        """The reasoning is written about the member's own dietary situation."""
        session, _ = _new_session(session_service, sample_profile)
        session_service.add_meal_plan(
            session.session_id,
            make_candidates("a"),
            "chosen because you avoid peanuts",
            {},
        )
        (row,) = _of_type(events, "chat.plan_generated")
        assert "peanut" not in repr(row).lower()
        assert set(row["props"]) == {
            "session_id",
            "plan_id",
            "plan_type",
            "version",
            "refinement",
            "item_count",
        }


class TestPlanSaving:
    def test_keeping_a_plan_is_reported(
        self, session_service, sample_profile, events
    ):
        session, member_id = _new_session(session_service, sample_profile)
        plan = session_service.add_meal_plan(
            session.session_id, make_candidates("a"), "because", {}
        )
        assert session_service.set_plan_saved(
            session.session_id, member_id, plan.id, True, "Tuesday"
        )
        (row,) = _of_type(events, "chat.plan_saved")
        assert row["props"] == {
            "session_id": session.session_id,
            "plan_id": plan.id,
        }
        # The title is the member's own words and is not a prop.
        assert "Tuesday" not in repr(row)

    def test_unsaving_is_not_a_save(self, session_service, sample_profile, events):
        """Counting an unsave would make a member who changed their mind look
        twice as satisfied."""
        session, member_id = _new_session(session_service, sample_profile)
        plan = session_service.add_meal_plan(
            session.session_id, make_candidates("a"), "because", {}
        )
        session_service.set_plan_saved(
            session.session_id, member_id, plan.id, False
        )
        assert _of_type(events, "chat.plan_saved") == []

    def test_saving_a_plan_that_is_not_there_reports_nothing(
        self, session_service, sample_profile, events
    ):
        session, member_id = _new_session(session_service, sample_profile)
        assert not session_service.set_plan_saved(
            session.session_id, member_id, "no-such-plan", True
        )
        assert _of_type(events, "chat.plan_saved") == []

    def test_another_members_save_is_refused_and_unreported(
        self, session_service, sample_profile, events
    ):
        session, _ = _new_session(session_service, sample_profile)
        plan = session_service.add_meal_plan(
            session.session_id, make_candidates("a"), "because", {}
        )
        assert not session_service.set_plan_saved(
            session.session_id, "intruder", plan.id, True
        )
        assert _of_type(events, "chat.plan_saved") == []


class TestReportingNeverCostsAPlan:
    def test_a_broken_reporter_does_not_lose_the_plan(
        self, session_service, sample_profile, monkeypatch
    ):
        import activity

        def explode(*args, **kwargs):
            raise RuntimeError("telemetry is broken")

        monkeypatch.setattr(activity, "report_event", explode)
        session, member_id = _new_session(session_service, sample_profile)
        plan = session_service.add_meal_plan(
            session.session_id, make_candidates("a"), "because", {}
        )
        assert plan is not None
        assert session_service.set_plan_saved(
            session.session_id, member_id, plan.id, True
        )
