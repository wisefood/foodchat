"""
Langfuse integration — the observability layer must be INVISIBLE to tests.

Two contracts (see langfuse-integration-guide.md §6):
  1. Default-off: with no keys set, every helper no-ops and nothing touches the
     network. The whole rest of the suite relies on this.
  2. sync_prompts never overwrites: the Langfuse UI is the source of truth, so
     an existing prompt is skipped, not recreated.

Plus the build_trace_config contract: omits None, coerces to str, carries only
opaque IDs + tags (no PII), and picks up request-scoped session/user from
trace_context.

langfuse is not installed in the test env, so the "keys set" paths still
resolve to disabled here — exactly the production degradation when the package
is missing.
"""

import json
import os
import unittest
from unittest.mock import MagicMock

from backend import observability
from backend.observability import (
    build_trace_config,
    flush_langfuse,
    get_callback_handler,
    get_langfuse_client,
    langchain_callbacks,
    langfuse_enabled,
    trace_context,
)
from prompts import (
    ALL_PROMPTS,
    GRADER_SYSTEM,
    GRADER_SYSTEM_INSTRUCTIONS,
    GRADER_USER,
    GRADER_USER_INSTRUCTIONS,
    _Prompt,
    sync_prompts,
)


def _clear_caches():
    get_callback_handler.cache_clear()
    get_langfuse_client.cache_clear()


class DisabledPathTests(unittest.TestCase):
    """With no keys (the test default), tracing is fully off and silent."""

    def setUp(self):
        _clear_caches()
        for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
            os.environ.pop(key, None)

    def tearDown(self):
        _clear_caches()

    def test_disabled_without_keys(self):
        self.assertFalse(langfuse_enabled())
        self.assertIsNone(get_callback_handler())
        self.assertIsNone(get_langfuse_client())
        self.assertEqual(langchain_callbacks(), [])

    def test_one_key_alone_is_disabled(self):
        os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-test"
        os.environ.pop("LANGFUSE_SECRET_KEY", None)
        _clear_caches()
        self.assertFalse(langfuse_enabled())

    def test_both_keys_but_package_missing_is_disabled(self):
        # langfuse isn't installed in the test env, so even with both keys the
        # import guard keeps it off — the production "package missing" path.
        os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-test"
        os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-test"
        _clear_caches()
        try:
            self.assertFalse(langfuse_enabled())
            self.assertIsNone(get_callback_handler())
        finally:
            os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
            os.environ.pop("LANGFUSE_SECRET_KEY", None)

    def test_flush_is_noop_when_disabled(self):
        # Must not raise even with no client.
        flush_langfuse()

    def test_sync_prompts_noop_when_disabled(self):
        result = sync_prompts()
        self.assertEqual(result, {"created": 0, "skipped": 0, "failed": 0})


class BuildTraceConfigTests(unittest.TestCase):
    def test_run_name_only(self):
        cfg = build_trace_config(run_name="qa_answer")
        self.assertEqual(cfg["run_name"], "qa_answer")
        self.assertNotIn("metadata", cfg)  # no ids/tags → no metadata block

    def test_omits_none_and_coerces_to_str(self):
        cfg = build_trace_config(run_name="x", session_id=123, user_id=None, tags=["qa"])
        meta = cfg["metadata"]
        self.assertEqual(meta["langfuse_session_id"], "123")  # coerced to str
        self.assertNotIn("langfuse_user_id", meta)            # None omitted
        self.assertEqual(meta["langfuse_tags"], ["qa"])
        # Whole config must be JSON-serializable (strings only).
        json.dumps(cfg)

    def test_reads_trace_context(self):
        with trace_context(session_id="sess-1", user_id="member-9"):
            cfg = build_trace_config(run_name="orchestrate", tags=["router"])
        meta = cfg["metadata"]
        self.assertEqual(meta["langfuse_session_id"], "sess-1")
        self.assertEqual(meta["langfuse_user_id"], "member-9")

    def test_explicit_args_override_context(self):
        with trace_context(session_id="sess-1", user_id="member-9"):
            cfg = build_trace_config(run_name="x", session_id="override")
        self.assertEqual(cfg["metadata"]["langfuse_session_id"], "override")
        self.assertEqual(cfg["metadata"]["langfuse_user_id"], "member-9")

    def test_context_resets_after_block(self):
        with trace_context(session_id="sess-1"):
            pass
        cfg = build_trace_config(run_name="x")
        self.assertNotIn("metadata", cfg)  # context cleared → no leakage

    def test_no_pii_only_declared_keys(self):
        cfg = build_trace_config(
            run_name="x", session_id="s", user_id="u", tags=["qa"],
            extra_metadata={"feature": "weekly"},
        )
        # Only the sanctioned keys are ever present — no free-form PII leaks in.
        self.assertEqual(
            set(cfg["metadata"]),
            {"langfuse_session_id", "langfuse_user_id", "langfuse_tags", "feature"},
        )


class SyncPromptsContractTests(unittest.TestCase):
    """The never-overwrite contract, verified against a mock client."""

    def test_skips_existing_prompts(self):
        client = MagicMock()
        client.get_prompt.return_value = object()  # exists in Langfuse
        p = _Prompt("t", "hello {{who}}")
        result = sync_prompts(client=client, registry=[p])
        client.create_prompt.assert_not_called()   # UI wins — never overwrite
        self.assertEqual(result, {"created": 0, "skipped": 1, "failed": 0})

    def test_creates_missing_prompts(self):
        client = MagicMock()
        client.get_prompt.side_effect = Exception("not found")  # missing
        p = _Prompt("t", "hello world")
        result = sync_prompts(client=client, registry=[p])
        client.create_prompt.assert_called_once()
        _, kwargs = client.create_prompt.call_args
        self.assertEqual(kwargs["name"], "foodchat/t")
        self.assertEqual(kwargs["type"], "text")
        self.assertEqual(kwargs["prompt"], "hello world")
        self.assertEqual(kwargs["labels"], ["production"])
        self.assertEqual(result, {"created": 1, "skipped": 0, "failed": 0})

    def test_existence_check_uses_no_fallback_and_no_cache(self):
        client = MagicMock()
        client.get_prompt.return_value = object()
        p = _Prompt("t", "x")
        sync_prompts(client=client, registry=[p])
        _, kwargs = client.get_prompt.call_args
        # Must NOT pass fallback (else missing prompts never raise → never
        # created) and must bypass the cache (fresh existence check).
        self.assertNotIn("fallback", kwargs)
        self.assertEqual(kwargs["cache_ttl_seconds"], 0)


class ManagedPromptFallbackTests(unittest.TestCase):
    """With Langfuse off, compile() must reproduce the old .format() behavior."""

    def setUp(self):
        _clear_caches()
        for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
            os.environ.pop(key, None)

    def test_system_prompt_verbatim_even_with_literal_braces(self):
        # GRADER_SYSTEM contains a literal JSON example ({...}); compile() with
        # no vars must return it untouched — never run .format() on it.
        self.assertEqual(GRADER_SYSTEM.compile(), GRADER_SYSTEM_INSTRUCTIONS)

    def test_user_prompt_formats_like_before(self):
        expected = GRADER_USER_INSTRUCTIONS.format(
            query="q", daily_plan="d", preferences="p", feedback_history="f",
        )
        got = GRADER_USER.compile(
            query="q", daily_plan="d", preferences="p", feedback_history="f",
        )
        self.assertEqual(got, expected)

    def test_registry_is_complete(self):
        # Every registered prompt is namespaced and has a non-empty fallback.
        self.assertTrue(ALL_PROMPTS)
        for p in ALL_PROMPTS:
            self.assertTrue(p.name.startswith("foodchat/"))
            self.assertTrue(p.fallback.strip())


if __name__ == "__main__":
    unittest.main()
