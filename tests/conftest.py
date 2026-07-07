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
    return [
        CandidateRecipe(f"{prefix}-b", "Oatmeal", "oats, milk", "Cook the oats."),
        CandidateRecipe(f"{prefix}-l", "Lentil soup", "lentils, carrot", "Simmer 30 min."),
        CandidateRecipe(f"{prefix}-d", "Veggie stew", "potato, beans", "Stew it all."),
    ]
