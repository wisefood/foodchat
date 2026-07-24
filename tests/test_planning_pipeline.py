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

    def fetch_candidates(self, **kwargs):
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

        plans = pipeline.generate("vegan day", sample_profile)
        assert [p.score for p in plans] == [5, 4]
        # profile constraints forwarded as hard filters
        assert fake.called_with["allergens"] == ["peanuts"]
        assert fake.called_with["exclude_ingredients"] == ["olives"]
        assert "chickpeas" in fake.called_with["include_ingredients"]

    def test_empty_slot_yields_no_plans(self, pipeline, monkeypatch, sample_profile):
        fake = FakeCandidatesClient({
            "breakfast": [], "lunch": _slot("l", 1), "dinner": _slot("d", 1),
        })
        monkeypatch.setattr(pp, "CANDIDATES", fake)
        assert pipeline.generate("anything", sample_profile) == []


class TestDocumentGraderParsing:
    def test_grader_builds_scored_plans_from_llm_json(self):
        """Real DocumentGrader with a faked LLM client — checks parsing/sorting."""
        from agents import DocumentGrader

        grader = DocumentGrader.__new__(DocumentGrader)
        grader.max_plans_to_score = 10

        class FakeLLM:
            def __init__(self):
                self.n = 0

            def invoke(self, messages, config=None):
                class R: pass
                r = R()
                self.n += 1
                r.content = json.dumps({"score": self.n, "reasoning": f"r{self.n}"})
                return r

        grader.grader = FakeLLM()
        candidates = {"breakfast": _slot("b", 2), "lunch": _slot("l", 1), "dinner": _slot("d", 1)}
        plans = grader.grade_daily_plans("q", candidates, {"preferences": []})
        assert len(plans) == 2
        assert plans[0].score >= plans[1].score
        assert all(isinstance(p, ScoredPlan) for p in plans)

    def test_grader_empty_candidates(self):
        from agents import DocumentGrader

        grader = DocumentGrader.__new__(DocumentGrader)
        grader.max_plans_to_score = 10
        grader.grader = None  # must not be touched
        assert grader.grade_daily_plans("q", {"breakfast": [], "lunch": [], "dinner": []}, {}) == []
