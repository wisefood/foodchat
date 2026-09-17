"""The deterministic PII layer: what it catches, and what it must leave alone.

The second list is the one that matters in a food chat. A redactor that turns
"1500 kcal" or an order reference into [PHONE] is wrong in a way the person
can see, and a person who has seen it once stops trusting the answers.
"""
import importlib

import pytest


@pytest.fixture
def pii(monkeypatch):
    monkeypatch.setenv("PII_REDACTION", "true")
    import pii as module

    return importlib.reload(module)


class TestCatches:
    @pytest.mark.parametrize("text", [
        "call me on 041 234 567",
        "+386 41 234 567",
        "+30 6912345678",
        "06912345678",
    ])
    def test_phone_numbers(self, pii, text):
        redacted, found = pii.redact(text)
        assert found == {"PHONE": 1} and "[PHONE]" in redacted

    def test_a_luhn_valid_card(self, pii):
        redacted, found = pii.redact("my card is 4111 1111 1111 1111")
        assert found == {"CARD": 1} and "4111" not in redacted

    def test_a_checksummed_iban(self, pii):
        _, found = pii.redact("SI56 1910 0000 0123 438")
        assert found == {"IBAN": 1}

    def test_an_email(self, pii):
        _, found = pii.redact("write to ana.k@example.org please")
        assert found == {"EMAIL": 1}


class TestLeavesAlone:
    @pytest.mark.parametrize("text", [
        "1234567890123456",              # a bare run: card that failed Luhn, an id
        "the 2024 report has 12 pages",
        "I ate 1500 kcal and 250 g of rice",
        "bake at 180 for 25 minutes",
        "serves 4, 350 kcal each",
    ])
    def test_quantities_years_and_bare_ids(self, pii, text):
        redacted, found = pii.redact(text)
        assert found == {} and redacted == text


class TestNeverRaises:
    def test_empty_and_none(self, pii):
        assert pii.redact(None) == (None, {})
        assert pii.redact("") == ("", {})

    def test_too_long_is_reported_not_scanned(self, pii):
        text = "x" * (pii.MAX_SCAN_CHARS + 1)
        assert pii.redact(text) == (text, {"unscanned": 1})
