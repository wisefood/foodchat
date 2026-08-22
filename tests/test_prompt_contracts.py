"""
The failures a prompt has that only show up against the real API.

Nothing in this suite calls Groq, by design — which leaves one class of bug
with no coverage at all, and this codebase has shipped it twice:

* **The "json" incident.** Groq 400s any `json_object` request whose messages
  do not contain the word "json". The in-code prompt said it; a stale managed
  copy in Langfuse did not, and the agent's own `except` swallowed the 400 into
  "no shape extracted". Multi-plate planning was disabled in production while
  every local test passed. The fix was to make the requirement structural, and
  this file is what keeps it structural.
* **A placeholder that nobody fills.** `compile(**vars)` catches a bad managed
  template and falls back to the in-code one — but if the IN-CODE text has a
  placeholder the call site does not pass, that fallback raises `KeyError`, the
  agent's outer `except` catches it, and the feature silently does nothing.

Both are checkable offline, and both are checked here: the prompt text and the
call site are read out of the source and compared. That is not a substitute for
a live smoke run — see `scripts/smoke_agents.py`, which is the one command to
run when a key exists — but it is the part that can be automated, and it is the
part that has actually broken.
"""

from __future__ import annotations

import ast
import pathlib
import string
import sys

import pytest

sys.path.insert(0, "src")

import prompts as prompts_module                                    # noqa: E402

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _placeholders(text: str) -> set[str]:
    """The `{name}` fields in a template, ignoring `{{` escapes."""
    return {
        field for _, field, _, _ in string.Formatter().parse(text)
        if field
    }


def _compile_calls() -> list[tuple[str, str, set[str], int]]:
    """(module, PROMPT_NAME, kwargs passed, line) for every `X.compile(...)`."""
    out = []
    for path in sorted(SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:                                  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr != "compile":
                continue
            if not isinstance(func.value, ast.Name):
                continue
            names = {kw.arg for kw in node.keywords if kw.arg}
            out.append((
                str(path.relative_to(SRC)), func.value.id, names, node.lineno,
            ))
    return out


class TestEveryPlaceholderIsFilled:
    """A `{var}` the call site does not pass makes `compile` raise `KeyError`
    on the in-code fallback — which the agent's own `except` swallows, so the
    feature does nothing and says nothing."""

    def test_the_scan_finds_the_call_sites(self):
        """A guard on the guard: an AST walk that silently matches nothing
        would make every assertion below vacuous."""
        calls = _compile_calls()
        assert len(calls) > 15, f"only found {len(calls)} compile() calls"

    def test_no_call_site_is_missing_a_variable(self):
        """Only prompts that are compiled WITH variables.

        `compile()` with no arguments returns the template verbatim and never
        calls `.format`, which is exactly why the system prompts can carry raw
        JSON examples — `{"mentioned": true}` is a brace pair the formatter
        would read as a field name and nothing ever asks it to.
        """
        missing = []
        for module, name, passed, line in _compile_calls():
            prompt = getattr(prompts_module, name, None)
            if prompt is None or not hasattr(prompt, "fallback") or not passed:
                continue
            gap = _placeholders(prompt.fallback) - passed
            if gap:
                missing.append(f"{module}:{line} {name} never fills {sorted(gap)}")
        assert not missing, "\n".join(missing)

    def test_no_call_site_passes_a_variable_the_prompt_ignores(self):
        """Harmless at runtime, and a reliable sign the two have drifted — the
        prompt was edited and the call site was not."""
        extra = []
        for module, name, passed, line in _compile_calls():
            prompt = getattr(prompts_module, name, None)
            if prompt is None or not hasattr(prompt, "fallback"):
                continue
            gap = passed - _placeholders(prompt.fallback)
            if gap:
                extra.append(f"{module}:{line} {name} passes unused {sorted(gap)}")
        assert not extra, "\n".join(extra)

    def test_every_templated_prompt_survives_being_formatted(self):
        """Catches an unbalanced brace — a JSON example written with a single
        `{` in a prompt that IS formatted raises before the model is reached."""
        broken = []
        for module, name, passed, line in _compile_calls():
            prompt = getattr(prompts_module, name, None)
            if prompt is None or not hasattr(prompt, "fallback") or not passed:
                continue
            try:
                prompt.fallback.format(**{key: "x" for key in passed})
            except Exception as exc:  # noqa: BLE001
                broken.append(f"{module}:{line} {name}: {exc}")
        assert not broken, "\n".join(broken)


class TestTheJsonWordIsStructural:
    """Groq 400s any `json_object` request whose messages do not contain the
    word "json". A prompt served from Langfuse can lose it without a deploy, so
    the guarantee cannot live in the prompt text — it has to be in the code
    that builds the call."""

    @staticmethod
    def _schema_agents() -> list[str]:
        """Agent classes that request a structured response."""
        tree = ast.parse((SRC / "agents.py").read_text())
        out = []
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            source = ast.unparse(node)
            if "format=" in source and "model_json_schema" in source:
                out.append(node.name)
        return out

    def test_the_scan_finds_them(self):
        assert len(self._schema_agents()) >= 8

    def test_every_schema_agent_routes_its_messages_through_the_guard(self):
        """One helper, applied at the last point before the call. A new agent
        that forgets fails here rather than in production, silently."""
        tree = ast.parse((SRC / "agents.py").read_text())
        by_name = {
            node.name: ast.unparse(node)
            for node in tree.body if isinstance(node, ast.ClassDef)
        }
        unguarded = [
            name for name in self._schema_agents()
            if "as_json_messages" not in by_name[name]
        ]
        assert not unguarded, (
            "these ask for JSON and would 400 on a managed prompt that lost "
            f"the word: {unguarded}"
        )

    def test_the_guard_adds_the_word_when_it_is_missing(self):
        from agents import as_json_messages
        from langchain_core.messages import HumanMessage, SystemMessage

        out = as_json_messages([
            SystemMessage(content="Rank these plans."),
            HumanMessage(content="two plans"),
        ])
        assert "json" in out[0].content.lower()
        assert out[1].content == "two plans", "only the system message changes"

    def test_it_leaves_a_prompt_that_already_says_it_alone(self):
        from agents import as_json_messages
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content="Return a JSON object."),
            HumanMessage(content="two plans"),
        ]
        assert as_json_messages(messages) is messages

    def test_the_word_counts_wherever_it_appears(self):
        """Groq inspects the whole request, so a user message carrying it is
        enough — adding a second instruction would be noise."""
        from agents import as_json_messages
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content="Rank these plans."),
            HumanMessage(content="Answer as json."),
        ]
        assert as_json_messages(messages) is messages


class TestNoPromptShipsDead:
    """`sync_prompts` creates only MISSING prompts and never overwrites, so
    editing text under an existing name ships it dead. The convention is a new
    name; what can be checked here is that no two prompts share one."""

    def test_registered_names_are_unique(self):
        names = [p.name for p in prompts_module.ALL_PROMPTS]
        duplicates = {n for n in names if names.count(n) > 1}
        assert not duplicates, duplicates

    def test_every_registered_prompt_has_text(self):
        empty = [p.name for p in prompts_module.ALL_PROMPTS if not p.fallback.strip()]
        assert not empty, empty


class TestTheSmokeScriptItselfWorks:
    """A smoke script that has never run is a smoke script that is broken.

    `scripts/smoke_agents.py` is the one command for the thing this suite
    cannot do — a live call per agent. It cannot be run here (no key, by
    design), so what is checked is everything about it that does not need one:
    every agent it names exists, and every call it makes has the right shape.

    The client is stubbed, so each check executes its real argument handling
    and its real response parsing against a canned reply. A signature that
    drifted fails here rather than the first time someone tries to use the
    script in an incident.
    """

    @staticmethod
    def _script():
        import importlib.util

        path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "smoke_agents.py"
        spec = importlib.util.spec_from_file_location("smoke_agents", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_it_covers_every_agent_that_asks_for_json(self):
        """A schema agent missing from the script is one nobody would find out
        about until it failed in front of a member."""
        covered = {name for name, _ in self._script()._checks()}
        # Script names are snake_case of the class; compare on a loose key so
        # renaming one does not silently drop it from the sweep.
        def key(text: str) -> str:
            return text.replace("_", "").lower()

        tree = ast.parse((SRC / "agents.py").read_text())
        schema_agents = [
            node.name for node in tree.body
            if isinstance(node, ast.ClassDef)
            and "format=" in ast.unparse(node)
            and "model_json_schema" in ast.unparse(node)
        ]
        keys = {key(c) for c in covered}
        missing = [
            cls for cls in schema_agents
            if not any(k in key(cls) or key(cls).startswith(k) for k in keys)
        ]
        assert not missing, f"the smoke script never calls: {missing}"

    @pytest.mark.parametrize("name", [
        "orchestrator", "plan_spec", "dietary_intent", "pantry", "plan_intent",
        "seed", "preference", "edit_command", "tool_selector", "grader",
        "diversity", "guidelines", "query_reconciler", "meal_judge",
        "plan_strategist", "response_writer", "session_title",
    ])
    def test_each_check_calls_its_agent_correctly(self, name, monkeypatch):
        """Real arguments, real parsing, canned reply — so a drifted signature
        or a renamed method fails here."""
        import backend.groq as groq_module

        canned = (
            '{"intent": "weekly_plan", "reasoning": "r", "mentioned": true,'
            ' "num_days": 3, "meals": ["dinner"], "plates": {},'
            ' "dietary_tags": ["vegetarian"], "have": ["spinach"], "used_up": [],'
            ' "cuisines": ["thai"], "moods": ["comforting"], "flavor_profiles": [],'
            ' "food_groups": [], "claim_tags": [], "relaxation_order": [],'
            ' "rationale": "r", "seeds": [{"name": "apple pie"}],'
            ' "memories": [{"kind": "dislike", "value": "mushrooms"}],'
            ' "meal_type": "dinner", "day": 2, "directive": "lighter",'
            ' "needs_slot_clarification": false, "question": null,'
            ' "tool": "plan_totals", "plan_type": "weekly",'
            ' "score": 4, "scores": [{"index": 0, "score": 4, "reasoning": "r"}],'
            ' "plans": [{"index": 0, "score": 4, "reasoning": "r"}],'
            ' "choices": [{"meal": "day 1 dinner", "pick": 1, "reason": "r"}],'
            ' "query": "a light vegetarian day", "kcal_target": null}'
        )

        class _Reply:
            content = canned

        class _Client:
            def invoke(self, messages, config=None):
                assert messages, "no messages were built"
                assert "json" in " ".join(
                    str(getattr(m, "content", "")) for m in messages
                ).lower(), "the json guard did not run"
                return _Reply()

        monkeypatch.setattr(
            groq_module.GroqConnectionPool, "get_client",
            lambda self, *a, **k: _Client(),
        )

        run = dict(self._script()._checks())[name]
        run()   # a raise is the failure; the answer's content is not this test's business
