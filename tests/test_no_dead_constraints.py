"""
Every standing constraint has something that writes it.

This exact bug has now shipped three times in the same file:

* nutrition claims were filed under `notes`, which is read only by
  `describe()`, which is only logged — heard, stored, dropped;
* `excluded_recipe_ids` had the daily fetch, the structured fetch and the
  swap all reading it, and nothing anywhere writing it;
* a cooking-time ceiling could be set by the slider and by nothing else,
  while the persona promised a member could just say it.

Each one looks fine from either end. The field exists, the consumers exist,
and the gap is only visible if you ask "who writes this?" — which nobody does
while adding a consumer. So the question is asked here, mechanically, over the
whole source tree: for every field of `PlanningStateDelta`, some module has to
construct it.

A new field with no producer fails this test until it has one, or until it is
deleted. Both are better outcomes than a constraint the member can state and
the planner cannot see.
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib
import sys

sys.path.insert(0, "src")

from models.planning_state import PlanningState, PlanningStateDelta  # noqa: E402

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _producers() -> dict[str, set[str]]:
    """field name -> the modules that construct a delta setting it.

    An AST walk rather than a grep: `PlanningStateDelta(diet_tags=...)` spread
    over three lines is the normal case in this codebase, and a line-based
    search reports the field as unwritten.
    """
    found: dict[str, set[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:                                  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name != "PlanningStateDelta":
                continue
            for keyword in node.keywords:
                if keyword.arg:
                    found.setdefault(keyword.arg, set()).add(
                        str(path.relative_to(SRC)),
                    )
    return found


class TestNoConstraintIsWriteOnly:
    def test_every_delta_field_has_a_producer(self):
        producers = _producers()
        fields = [f.name for f in dataclasses.fields(PlanningStateDelta)]
        missing = sorted(f for f in fields if f not in producers)
        assert not missing, (
            "no code constructs a delta setting: " + ", ".join(missing) +
            " — a member can state it and nothing will record it"
        )

    def test_every_state_field_reaches_a_fetch_or_is_reported(self):
        """The other half of the same question.

        A field can have a producer and still go nowhere — that was `notes`,
        written by the weekly path and read only by a log line. So each field
        has to be named somewhere OTHER than the model and its own producer:
        a fetch site, the transparency ledger, the brief, or the API response.
        """
        readers: dict[str, set[str]] = {}
        for path in sorted(SRC.rglob("*.py")):
            if path.name in {"planning_state.py"}:
                continue
            text = path.read_text()
            for field in (f.name for f in dataclasses.fields(PlanningState)):
                if f"state.{field}" in text or f'"{field}"' in text or f"_{field}" in text:
                    readers.setdefault(field, set()).add(str(path.relative_to(SRC)))

        unread = sorted(
            f.name for f in dataclasses.fields(PlanningState)
            if not readers.get(f.name)
        )
        assert not unread, (
            "nothing outside the model reads: " + ", ".join(unread)
        )

    def test_the_producers_are_where_they_should_be(self):
        """A spot check with real names, so the mechanical test above cannot
        pass on an accident — a delta built inside a test helper, or one module
        writing every field."""
        producers = _producers()
        assert any("pantry_service" in m for m in producers["pantry_add"])
        assert any("diet_intent" in m for m in producers["diet_tags"])
        assert any("intent_facets" in m for m in producers["cuisines"])
        assert any("intent_facets" in m for m in producers["facets_remove"])
        assert any("plan_parameters" in m for m in producers["max_minutes"])
        assert any("planning_delta" in m for m in producers["reset"])
        assert any("edit_service" in m for m in producers["excluded_recipe_ids"])
        assert any("edit_service" in m for m in producers["anchors"])
