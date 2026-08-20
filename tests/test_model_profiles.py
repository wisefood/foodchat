"""
Per-model capability profiles and how the Groq pool applies them.

The bug these guard against is silent: a reasoning model left at the provider
default returns its deliberation inside ``content``, every ``json.loads`` in
agents.py raises, and each agent's ``except`` swallows it into a fallback. The
plan still renders — it has just lost its grader ranking, its extracted tags
or its written prose. Nothing in a log says why, so the contract is asserted
here instead.
"""

import pytest

from backend.model_profiles import RETIRED, apply_profile, profile_for


class TestReasoningFamilies:
    """gpt-oss / qwen3 / deepseek-r1 must never answer into `content`."""

    @pytest.mark.parametrize("model", [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.6-27b",
        "deepseek-r1-distill-llama-70b",
    ])
    def test_reasoning_is_hidden(self, model):
        resolved = apply_profile(model, {"max_tokens": 4096})
        assert resolved["reasoning_format"] == "hidden", (
            f"{model} would return reasoning in content and break json.loads"
        )

    def test_token_floor_is_raised_never_lowered(self):
        # Below the floor: raised.
        assert apply_profile("openai/gpt-oss-120b", {"max_tokens": 512})["max_tokens"] == 2048
        # Above the floor: the caller's larger budget survives.
        assert apply_profile("openai/gpt-oss-120b", {"max_tokens": 8192})["max_tokens"] == 8192

    def test_missing_or_bad_max_tokens_gets_the_floor(self):
        assert apply_profile("openai/gpt-oss-20b", {})["max_tokens"] == 2048
        assert apply_profile("openai/gpt-oss-20b", {"max_tokens": None})["max_tokens"] == 2048

    def test_caller_preference_wins_over_the_family_default(self):
        resolved = apply_profile(
            "openai/gpt-oss-120b",
            {"max_tokens": 4096, "reasoning_effort": "high"},
        )
        assert resolved["reasoning_effort"] == "high"

    def test_qwen3_gets_no_invented_effort_level(self):
        # The family accepts the parameter but its scale differs from gpt-oss's;
        # guessing a level would be a measurement we have not made.
        assert "reasoning_effort" not in apply_profile("qwen/qwen3.6-27b", {})


class TestNonReasoningFamilies:
    """Llama-family models 400 on the reasoning params, so they are dropped."""

    def test_reasoning_params_are_dropped(self):
        resolved = apply_profile(
            "llama-3.3-70b-versatile",
            {"reasoning_format": "hidden", "reasoning_effort": "low", "max_tokens": 4096},
        )
        assert "reasoning_format" not in resolved
        assert "reasoning_effort" not in resolved

    def test_no_token_floor_is_imposed(self):
        assert apply_profile("llama-3.3-70b-versatile", {"max_tokens": 512})["max_tokens"] == 512

    def test_nothing_is_injected(self):
        assert apply_profile("gemma2-9b-it", {"max_tokens": 1024}) == {"max_tokens": 1024}


class TestUnknownModels:
    def test_an_unknown_id_forwards_explicit_caller_knobs(self):
        # Deliberate beats inferred: we do not strip what the caller asked for.
        resolved = apply_profile("some/未知-model-v9", {"reasoning_format": "hidden"})
        assert resolved["reasoning_format"] == "hidden"

    def test_an_unknown_id_gets_nothing_injected(self):
        assert apply_profile("some/other-unknown", {"max_tokens": 777}) == {"max_tokens": 777}

    def test_profile_never_raises_on_junk(self):
        for junk in (None, "", "   ", "!!!"):
            assert profile_for(junk).family == "unknown"


class TestRetiredModels:
    def test_both_shutdown_llamas_are_recorded(self):
        assert "llama-3.3-70b-versatile" in RETIRED
        assert "llama-3.1-8b-instant" in RETIRED

    def test_a_retired_id_warns_with_date_and_replacement(self, caplog):
        # profile_for warns once per id per process; clear the latch so the
        # assertion does not depend on test ordering.
        import backend.model_profiles as mp
        mp._warned_retired.discard("llama-3.3-70b-versatile")
        with caplog.at_level("WARNING"):
            profile_for("llama-3.3-70b-versatile")
        assert "2026-08-16" in caplog.text
        assert "gpt-oss-120b" in caplog.text

    def test_a_retired_id_still_resolves_its_family(self):
        # It fails at the provider, not here — the row must still be found so
        # the reasoning params stay correctly dropped.
        assert profile_for("llama-3.3-70b-versatile").family == "llama"


class TestPoolApplication:
    """The pool must apply the profile, not just own the module."""

    def test_the_shipped_default_is_built_with_reasoning_hidden(self):
        from backend.groq import GROQ_CHAT

        client = GROQ_CHAT.get_client(model="openai/gpt-oss-120b", temperature=0.0)
        assert client.reasoning_format == "hidden"
        assert client.reasoning_effort == "low"

    def test_a_llama_client_carries_no_reasoning_params(self):
        from backend.groq import GROQ_CHAT

        client = GROQ_CHAT.get_client(model="llama-3.3-70b-versatile", temperature=0.0)
        assert client.reasoning_format is None

    def test_profiled_configs_do_not_share_one_cached_client(self):
        # The floor rewrites max_tokens, so two callers whose requests differ
        # only below the floor must still not collide — and a reasoning model
        # must never be handed a client built for a non-reasoning one.
        from backend.groq import GROQ_CHAT

        oss = GROQ_CHAT.get_client(model="openai/gpt-oss-20b", temperature=0.0)
        llama = GROQ_CHAT.get_client(model="llama-3.3-70b-versatile", temperature=0.0)
        assert oss is not llama
        assert oss.reasoning_format == "hidden" and llama.reasoning_format is None
