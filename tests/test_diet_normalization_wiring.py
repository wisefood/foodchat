"""Every path to RecipeWrangler must normalise diet tags first.

RecipeWrangler ANDs diet tags and never relaxes them, so one value it does not
recognise empties every slot. Measured against the live service:

    diet=["vegan"]                  -> breakfast 8, lunch 8, dinner 8
    diet=["vegan", "balanced"]      -> breakfast 0, lunch 0, dinner 0
    diet=["omnivore"]               -> breakfast 0, lunch 0, dinner 0

The member sees "I'm sorry, I couldn't find enough recipes to build a complete
meal plan" and has no way to guess that the word on their profile is the cause.

`normalize_diet_tags` drops those values, and its docstring has always said
why. The candidate source moved off `fetch_candidates` and the new call sites
passed `profile.get("diet")` straight through — the guard was not removed on
purpose, it was simply not carried across. A grep-based test rather than a
behavioural one, because the failure mode is a *missing* call: any new path
that forgets it fails here instead of in production.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"


def _calls_passing_diet(path: pathlib.Path) -> list[tuple[int, str]]:
    """Every `diet=<expr>` keyword argument in a file, with its source line."""
    tree = ast.parse(path.read_text())
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == "diet":
                out.append((kw.value.lineno, ast.unparse(kw.value)))
    return out


def _normalised(expr: str) -> bool:
    """Whether a `diet=` argument is guarded.

    A literal empty list is fine — it forwards no tags at all. So is a value
    already normalised upstream and named as such.
    """
    return (
        "normalize_diet_tags" in expr
        or expr in ("[]", "()", "None")
        or "normalized" in expr
    )


@pytest.mark.parametrize(
    "relative",
    [
        "services/planning_pipeline.py",
        "services/candidates_client.py",
        "services/seed_service.py",
        "services/weekly_planner/action_adapter.py",
    ],
)
def test_no_raw_diet_reaches_recipewrangler(relative):
    path = SRC / relative
    unguarded = [
        (line, expr) for line, expr in _calls_passing_diet(path) if not _normalised(expr)
    ]

    assert not unguarded, (
        f"{relative} passes an unnormalised diet to RecipeWrangler at "
        + ", ".join(f"line {line}: diet={expr}" for line, expr in unguarded)
        + " — an unrecognised tag ANDs every slot down to zero and the member "
        "is told no recipes exist"
    )


class TestTheGuardItself:
    def test_an_unknown_tag_is_dropped_not_forwarded(self):
        from services.candidates_client import normalize_diet_tags

        assert normalize_diet_tags(["vegan", "balanced"]) == ["vegan"]
        assert normalize_diet_tags(["omnivore"]) == []
        assert normalize_diet_tags(["mediterranean"]) == []

    def test_a_real_restriction_survives(self):
        from services.candidates_client import normalize_diet_tags

        assert "vegan" in normalize_diet_tags(["vegan"])

    @pytest.mark.parametrize("junk", [None, "", [], "vegan"])
    def test_loose_input_does_not_raise(self, junk):
        from services.candidates_client import normalize_diet_tags

        assert isinstance(normalize_diet_tags(junk), list)
