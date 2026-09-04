"""Every turn is counted, including the ones that did not go well.

The reporting used to live in `process()`, which is one of four turn entry
points, and ran only on the success path. Three entry points and every refused,
capped or failed turn were missing from the record — so "turns per member" was
quietly an undercount, and the failures, the interesting part, were invisible.
"""
import sys
import time
import types
import unittest
from unittest import mock

sys.path.insert(0, "src")


class TurnReportingTests(unittest.TestCase):
    def setUp(self):
        import importlib

        import activity

        # `services/__init__.py` binds a module-level singleton named
        # `orchestrator_service`, which shadows the submodule of the same name
        # for attribute-style imports.
        orchestrator = importlib.import_module("services.orchestrator_service")

        self.orchestrator = orchestrator
        self.reported = []
        self._patch = mock.patch.object(
            activity, "report_turn", side_effect=lambda **kw: self.reported.append(kw)
        )
        self._patch.start()
        orchestrator._turn_detail.set(None)

    def tearDown(self):
        self._patch.stop()
        self.orchestrator._turn_detail.set(None)

    def _service(self):
        service = object.__new__(self.orchestrator.OrchestratorService)
        service._turn_lock = __import__("threading").Lock()
        service._turns_in_flight = {}
        return service

    def test_a_normal_turn_is_reported_with_what_the_handler_learned(self):
        service = self._service()
        with service._one_turn_at_a_time("s1") as claimed:
            self.assertTrue(claimed)
            self.orchestrator._turn_detail.set(
                {"intent": "plan", "plan_id": "p1", "opening_turn": True,
                 "has_attribution": False}
            )
        self.assertEqual(len(self.reported), 1)
        event = self.reported[0]
        self.assertEqual(event["session_id"], "s1")
        self.assertEqual(event["intent"], "plan")
        self.assertEqual(event["plan_id"], "p1")
        self.assertEqual(event["extra"]["outcome"], "ok")
        self.assertTrue(event["extra"]["opening_turn"])

    def test_a_refused_turn_is_still_reported(self):
        """A member sending twice is a real signal; it used to be invisible."""
        service = self._service()
        with service._one_turn_at_a_time("s2") as first:
            self.assertTrue(first)
            with service._one_turn_at_a_time("s2") as second:
                self.assertFalse(second)
        outcomes = [e["extra"]["outcome"] for e in self.reported]
        self.assertEqual(sorted(outcomes), ["busy", "ok"])

    def test_a_turn_that_raises_is_reported_as_an_error(self):
        service = self._service()
        with self.assertRaises(RuntimeError):
            with service._one_turn_at_a_time("s3"):
                raise RuntimeError("planner exploded")
        self.assertEqual(len(self.reported), 1)
        self.assertEqual(self.reported[0]["extra"]["outcome"], "error")

    def test_a_turn_that_learned_nothing_is_still_counted(self):
        """The message-cap path returns before any handler detail exists."""
        service = self._service()
        with service._one_turn_at_a_time("s4") as claimed:
            self.assertTrue(claimed)
        self.assertEqual(len(self.reported), 1)
        self.assertIsNone(self.reported[0]["intent"])
        self.assertEqual(self.reported[0]["extra"]["outcome"], "ok")

    def test_detail_does_not_leak_into_the_next_turn(self):
        service = self._service()
        with service._one_turn_at_a_time("s5"):
            self.orchestrator._turn_detail.set({"intent": "plan", "plan_id": "p9"})
        with service._one_turn_at_a_time("s6"):
            pass
        self.assertEqual(self.reported[0]["plan_id"], "p9")
        self.assertIsNone(self.reported[1]["plan_id"])

    def test_reporting_failure_never_breaks_a_turn(self):
        import activity

        service = self._service()
        with mock.patch.object(activity, "report_turn", side_effect=RuntimeError("down")):
            with service._one_turn_at_a_time("s7") as claimed:
                self.assertTrue(claimed)

    def test_latency_is_measured_across_the_whole_turn(self):
        service = self._service()
        with service._one_turn_at_a_time("s8"):
            time.sleep(0.05)
        self.assertGreaterEqual(self.reported[0]["latency_ms"], 40)


if __name__ == "__main__":
    unittest.main()
