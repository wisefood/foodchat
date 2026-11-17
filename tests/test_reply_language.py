"""The reader's language reaches every text FoodChat writes for them.

Before: no language reached FoodChat and no prompt named one, so a Slovenian
reader got English replies and a turn with no typed text got English in every
language.
"""

import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient

import reply_language


def _app():
    seen = {}
    app = FastAPI()

    @app.get("/probe")
    async def probe():
        seen["language"] = reply_language.current()

        async def later():
            return reply_language.current()

        # Work a request starts in the background keeps its language.
        seen["task"] = await asyncio.create_task(later())
        return {}

    app.add_middleware(reply_language.ReplyLanguageMiddleware)
    return app, seen


class TestNormalize:
    def test_supported_codes(self):
        for code in ("en", "hu", "sl", "el"):
            assert reply_language.normalize(code) == code

    def test_regions_and_lists(self):
        assert reply_language.normalize("hu-HU") == "hu"
        assert reply_language.normalize("sl-SI,sl;q=0.9,en;q=0.8") == "sl"
        assert reply_language.normalize("EL") == "el"

    def test_unsupported_or_missing_is_english(self):
        for raw in (None, "", "de", "sk", "fr-FR", "x\r\ny", ";q=1"):
            assert reply_language.normalize(raw) == "en"


class TestMiddleware:
    def test_x_locale_binds_the_language(self):
        app, seen = _app()
        with TestClient(app) as client:
            client.get("/probe", headers={"X-Locale": "sl"})
        assert seen == {"language": "sl", "task": "sl"}

    def test_x_locale_wins_over_the_device_language(self):
        app, seen = _app()
        with TestClient(app) as client:
            client.get("/probe", headers={"X-Locale": "hu", "Accept-Language": "en-GB,en"})
        assert seen["language"] == "hu"

    def test_accept_language_is_the_fallback(self):
        app, seen = _app()
        with TestClient(app) as client:
            client.get("/probe", headers={"Accept-Language": "el-GR,el;q=0.9"})
        assert seen["language"] == "el"

    def test_no_header_is_english_and_nothing_leaks(self):
        app, seen = _app()
        with TestClient(app) as client:
            client.get("/probe", headers={"X-Locale": "sl"})
            client.get("/probe")
        assert seen["language"] == "en"


class TestInstruction:
    def test_english_prompts_are_unchanged(self):
        assert reply_language.instruction("en") == ""

    def test_names_the_language_unmistakably(self):
        text = reply_language.instruction("sl")
        assert "Slovenian (slovenščina)" in text
        assert "recipe titles exactly" in text

    def test_follows_the_request(self):
        token = reply_language.use("hu")
        try:
            assert "Hungarian" in reply_language.instruction()
        finally:
            reply_language.reset(token)
        assert reply_language.instruction() == ""


class TestFixedSentences:
    """Sentences FoodChat writes without a model follow the reader's language,
    and the English ones stay exactly as they were."""

    def _changed(self, **over):
        changed = {
            "meal_type": "dinner", "day": None,
            "old": {"title": "Quinoa pilaf", "kcal": 480.0},
            "new": {"title": "Chicory with bean", "kcal": 71.0},
            "directive": "vegetarian", "verified": True,
        }
        changed.update(over)
        return changed

    def _predicate(self, **over):
        from types import SimpleNamespace
        base = dict(kind="tag", verifiable=True, tag=None, directive="vegetarian")
        base.update(over)
        return SimpleNamespace(**base)

    def _in(self, lang, fn):
        token = reply_language.use(lang)
        try:
            return fn()
        finally:
            reply_language.reset(token)

    def test_a_swap_in_slovenian(self):
        from services.edit_service import EditService
        text = self._in("sl", lambda: EditService._success_text(self._changed(), self._predicate()))
        assert text == ("Končano — zamenjano: večerja, »Quinoa pilaf« → »Chicory with bean«. "
                        "Preverjeno: nova izbira ustreza zahtevi »vegetarijansko«.")

    def test_a_lighter_swap_on_a_weekday_in_hungarian(self):
        from services.edit_service import EditService
        text = self._in("hu", lambda: EditService._success_text(
            self._changed(day=3), self._predicate(kind="lighter"), weekdays=True))
        assert text.startswith("Kész — kicseréltem: vacsora (szerda), „Quinoa pilaf” → „Chicory with bean”.")
        assert "480 kcal helyett 71 kcal" in text

    def test_a_multi_day_plan_has_no_weekdays(self):
        from services.edit_service import EditService
        text = self._in("el", lambda: EditService._success_text(
            self._changed(day=2), self._predicate(), weekdays=False))
        assert "βραδινό (ημέρα 2)" in text

    def test_the_english_swap_is_unchanged(self):
        from services.edit_service import EditService
        text = EditService._success_text(self._changed(), self._predicate())
        assert text == ("Done — I swapped the dinner: “Quinoa pilaf” → “Chicory with bean”. "
                        "Verified: the new pick satisfies “vegetarian”.")

    def test_a_failed_swap_names_the_nearest(self):
        from services.edit_service import EditService
        text = self._in("sl", lambda: EditService._failure_text(
            "lunch", self._predicate(), {"nearest_miss": {"title": "Soup", "kcal": 310.0}}))
        assert text.startswith("Zamenjave (kosilo)") and "»Soup« s 310 kcal" in text

    def test_the_diet_offer(self):
        from services.diet_intent import suggest_diet_memory
        offer = self._in("hu", lambda: suggest_diet_memory(["vegetarian"], "csak zöldséget", {}))
        assert offer["value"] == "vegetarian"            # the code is still English
        assert offer["statement"].startswith("Ezt kérted: „vegetáriánus”.")
        english = suggest_diet_memory(["vegetarian"], "veggie please", {})
        assert english["statement"].startswith("You asked for vegetarian meals")

    def test_every_sentence_exists_in_every_language(self):
        import texts
        for key, by_lang in texts._TEXT.items():
            assert set(by_lang) == {"hu", "sl", "el"}, key

    def test_a_meal_the_plan_does_not_have(self):
        from types import SimpleNamespace
        from services.edit_service import EditService
        days = [SimpleNamespace(day=1, meals=[SimpleNamespace(meal_type="dinner")])]
        text = self._in("sl", lambda: EditService._structured_miss_text(days, 1, "lunch"))
        assert text.startswith("V tem načrtu ni obroka »kosilo« — ima: večerja.")
        english = EditService._structured_miss_text(days, 1, "lunch")
        assert english.startswith("This plan has no lunch — it has dinner.")

    def test_a_day_the_plan_does_not_have(self):
        from types import SimpleNamespace
        from services.edit_service import EditService
        days = [SimpleNamespace(day=1, meals=[]), SimpleNamespace(day=2, meals=[])]
        text = self._in("hu", lambda: EditService._structured_miss_text(days, 3, "lunch"))
        assert text.startswith("Ez a terv ezeket a napokat fedi le: 1. nap, 2. nap — 3. nap nincs benne.")

    def test_names_join_with_the_languages_own_and(self):
        import texts
        assert texts.names(["Anna", "Bence", "Csilla"], "hu") == "Anna, Bence és Csilla"
        assert texts.names(["Ana", "Luka"], "sl") == "Ana in Luka"
        assert texts.names(["Anna"], "el") == "Anna"

    def test_a_pantry_chip(self):
        from services.pantry_service import _badge_reasons
        chip = self._in("sl", lambda: _badge_reasons(["tomatoes", "spinach"]))
        assert chip == [{"kind": "pantry", "label": "porabi: tomatoes, spinach — manj zavržene hrane"}]
        assert _badge_reasons(["tomatoes"])[0]["label"].startswith("uses your tomatoes")
