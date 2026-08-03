"""PlanningPipeline + DocumentGrader plumbing (grader LLM faked)."""

import json

import pytest

from models.recipe import CandidateRecipe, ScoredPlan
from services import planning_pipeline as pp
from services.planning_pipeline import PlanningPipeline


def _slot(prefix, n):
    return [
        CandidateRecipe(f"{prefix}{i}", f"{prefix.title()} {i}", f"ing-{i}", f"dir-{i}")
        for i in range(n)
    ]


class FakeGrader:
    """DocumentGrader stand-in: deterministic scores keyed by breakfast id."""

    def grade_daily_plans(self, query, candidates, profile, feedback_history=""):
        plans = []
        for i, b in enumerate(candidates["breakfast"]):
            plans.append(ScoredPlan(
                breakfast=b, lunch=candidates["lunch"][0], dinner=candidates["dinner"][0],
                score=5 - i, reasoning=f"combo {i}",
            ))
        return sorted(plans, key=lambda p: p.score, reverse=True)[:3]


class FakeCandidatesClient:
    def __init__(self, result):
        self.result = result
        self.called_with = None

    def split_cuisines(self, likes):
        """No vocabulary in tests, so nothing is a cuisine.

        Mirrors the real client's behaviour when the vocabulary fetch fails —
        everything stays an ingredient — which is the safe degradation these
        tests should be asserting against anyway.
        """
        return [], list(likes or [])

    def pool(self, **kwargs):
        self.called_with = kwargs
        return self.result


@pytest.fixture
def pipeline(monkeypatch):
    p = PlanningPipeline.__new__(PlanningPipeline)
    p.grader = FakeGrader()
    return p


class TestGenerate:
    def test_returns_ranked_plans(self, pipeline, monkeypatch, sample_profile):
        fake = FakeCandidatesClient({
            "breakfast": _slot("b", 2), "lunch": _slot("l", 1), "dinner": _slot("d", 1),
        })
        monkeypatch.setattr(pp, "CANDIDATES", fake)
        monkeypatch.setattr(pp, "_fetch_candidate_pool", lambda **kw: fake.pool(**kw))

        plans = pipeline.generate("vegan day", sample_profile)
        assert [p.score for p in plans] == [5, 4]
        # Profile constraints reach the pool. They travel inside `profile` now
        # rather than as flat kwargs: the pool builder forwards allergens, diet
        # and dislikes to `plan_meals`, which applies them as hard filters.
        sent = fake.called_with
        assert sent["profile"]["allergies"] == ["peanuts"]
        assert sent["profile"]["food_dislikes"] == ["olives"]
        # Liked ingredients stay a *boost*: `plan_meals` treats
        # `include_ingredients` as a requirement, so forwarding them would
        # demand chickpeas in every breakfast and empty the slot.
        assert "chickpeas" in sent["liked_ingredients"]

    def test_empty_slot_yields_no_plans(self, pipeline, monkeypatch, sample_profile):
        fake = FakeCandidatesClient({
            "breakfast": [], "lunch": _slot("l", 1), "dinner": _slot("d", 1),
        })
        monkeypatch.setattr(pp, "CANDIDATES", fake)
        monkeypatch.setattr(pp, "_fetch_candidate_pool", lambda **kw: fake.pool(**kw))
        assert pipeline.generate("anything", sample_profile) == []


class TestDocumentGraderParsing:
    def test_grader_builds_scored_plans_from_llm_json(self):
        """Real DocumentGrader with a faked LLM client — checks parsing/sorting."""
        from agents import DocumentGrader

        grader = DocumentGrader.__new__(DocumentGrader)
        grader.max_plans_to_score = 10

        class FakeLLM:
            """One batch call in, one grades array out — the new contract."""

            def __init__(self):
                self.calls = 0

            def invoke(self, messages, config=None):
                class R: pass
                r = R()
                self.calls += 1
                r.content = json.dumps({"grades": [
                    {"plan_index": 0, "score": 3, "reasoning": "r0"},
                    {"plan_index": 1, "score": 5, "reasoning": "r1"},
                    {"plan_index": 99, "score": 4, "reasoning": "junk index dropped"},
                ]})
                return r

        fake = FakeLLM()
        grader.grader = fake
        candidates = {"breakfast": _slot("b", 2), "lunch": _slot("l", 1), "dinner": _slot("d", 1)}
        plans = grader.grade_daily_plans("q", candidates, {"preferences": []})
        assert fake.calls == 1, "the whole batch grades in ONE call"
        assert len(plans) == 2, "junk indices are dropped, real ones kept"
        assert plans[0].score == 5 and plans[1].score == 3, "sorted best-first"
        assert all(isinstance(p, ScoredPlan) for p in plans)

    def test_grader_empty_candidates(self):
        from agents import DocumentGrader

        grader = DocumentGrader.__new__(DocumentGrader)
        grader.max_plans_to_score = 10
        grader.grader = None  # must not be touched
        assert grader.grade_daily_plans("q", {"breakfast": [], "lunch": [], "dinner": []}, {}) == []
