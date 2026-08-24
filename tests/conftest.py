"""
Test environment bootstrap.

Must run before any src import: db.py binds its engine and backend/groq.py
reads its key at import time, so the environment is prepared here first.
Unit tests never call an LLM — agents are constructed against a dummy key and
their clients are replaced with fakes per test — unit tests never call an LLM.
"""

import os
import sys
import tempfile
from pathlib import Path

# --- environment BEFORE src imports -------------------------------------- #
_TMP_DIR = tempfile.mkdtemp(prefix="foodchat-tests-")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DIR}/test.db"
os.environ.setdefault("GROQ_API_KEY", "gsk_test_dummy_key")
os.environ.setdefault("RECIPEWRANGLER_API_URL", "http://recipewrangler.test:8001")

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import pytest  # noqa: E402

from db import init_db  # noqa: E402
from models.recipe import CandidateRecipe  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _database():
    """Create the schema once for the whole test session."""
    init_db()


@pytest.fixture
def session_service():
    """A fresh SessionService (empty in-memory cache; shared test DB file)."""
    from services.session_service import SessionService
    return SessionService()


@pytest.fixture
def sample_profile() -> dict:
    return {
        "diet": ["vegetarian"],
        "allergies": ["peanuts"],
        "preferences": ["2000 calories target"],
        "history": "",
        "food_likes": ["chickpeas"],
        "food_dislikes": ["olives"],
    }


def make_candidates(prefix: str = "r") -> list[CandidateRecipe]:
    """Three distinct candidates usable as [breakfast, lunch, dinner]."""
    # WITH nutrition, because every candidate from `plan_meals` carries it.
    #
    # These were built bare, and that is how a crash on every real plan went
    # unnoticed through 1500 tests: `CandidateRecipe` is a frozen dataclass and
    # therefore looks hashable, but its `nutrition` dict makes `hash()` raise —
    # so the grader threw on production pools and never on these.
    #
    # A fixture that is easier to construct than the real thing tests a
    # different program.
    return [
        CandidateRecipe(f"{prefix}-b", "Oatmeal", "oats, milk", "Cook the oats.",
                        nutrition={"kcal": 320.0, "protein_g": 12.0,
                                   "carbs_g": 40.0, "fat_g": 9.0}),
        CandidateRecipe(f"{prefix}-l", "Lentil soup", "lentils, carrot", "Simmer 30 min.",
                        nutrition={"kcal": 520.0, "protein_g": 24.0,
                                   "carbs_g": 60.0, "fat_g": 14.0}),
        CandidateRecipe(f"{prefix}-d", "Veggie stew", "potato, beans", "Stew it all.",
                        nutrition={"kcal": 610.0, "protein_g": 26.0,
                                   "carbs_g": 70.0, "fat_g": 18.0}),
    ]


@pytest.fixture(autouse=True)
def _isolate_class_level_agents():
    """Undo any test's override of the orchestrator's lazy singletons.

    `OrchestratorService._tool_selector` is held on the CLASS on purpose — an
    instance built with `__new__`, which this suite does constantly, would
    otherwise have none. The cost is that a test which swaps in a stub swaps it
    in for everything that runs afterwards, and a stub that always chooses
    `save_plan` turns some later test's plan question into a save. That is a
    silent, order-dependent failure, so it is undone here rather than in each
    test that remembers.
    """
    from services.orchestrator_service import OrchestratorService

    before = OrchestratorService._tool_selector
    yield
    OrchestratorService._tool_selector = before


@pytest.fixture(autouse=True)
def _fresh_turn_intake():
    """The intake memo is per turn; no test inherits another's."""
    from services import turn_intake

    turn_intake.forget()
    yield
    turn_intake.forget()
