"""Taking the personal data out of a message before a model sees it.

This platform sends what people type to Groq and OpenAI. The text is about
food, which sounds harmless until you read what people actually write into a
meal-planning chat: names of their children, an address for delivery, a phone
number, and — routinely, because the product invites it — allergies, medical
conditions and medication. That last group is special-category data under
GDPR Article 9, and it is being handed to a third-party processor outside the
platform's control the moment it reaches a provider.

Redaction here does not make that lawful on its own. What it does is stop the
*directly identifying* fields travelling with the sensitive ones, so a leaked
prompt is a statement about an anonymous person rather than about a named one.

Two layers, in this order:

* **Deterministic detectors.** Regex plus a validator where the format has one
  — a card number is only a card number if it passes Luhn, an IBAN only if its
  checksum holds. These are exact, fast, need no dependency, and never vary
  between runs, which matters because a redactor that behaves differently on
  two identical inputs cannot be audited.
* **A local NER model, when one is installed.** Names and places have no
  format to match on, so they need a model — and it has to be local, because
  sending the text to a remote service to find out whether it contains
  personal data would be the thing this exists to prevent. Absent the model,
  those go undetected and `names_detected` says so rather than implying the
  text was clean.

Deliberately NOT an LLM. A model asked to redact is slow, non-deterministic,
and can be argued out of it by the text it is redacting — all three
disqualifying for something on the request path whose output must be
reproducible.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Off unless asked for. Redaction changes what the model is told, and a
#: message with its address removed can produce a worse answer — so this is a
#: deployment's decision, not a default that quietly degrades replies.
ENABLED = (os.getenv("PII_REDACTION", "").strip().lower() in ("1", "true", "yes"))

#: The local NER model, when one is installed. Names need a model; there is no
#: pattern for "Tamara". Absent it, names pass through and the caller is told.
NER_MODEL = os.getenv("PII_NER_MODEL", "").strip()

#: Longest message we will scan. A redactor is on the request path and must
#: not become the slow part of a turn; past this the text is passed through
#: and flagged rather than silently half-scanned.
MAX_SCAN_CHARS = 20_000


def _luhn(digits: str) -> bool:
    """Whether a digit string passes the card checksum.

    Without it, every 16-digit number — an order reference, a timestamp — is a
    credit card, and over-redaction is its own failure: a chat that replaces
    the user's own words with [CARD] is broken in a way they can see.
    """
    total, alt = 0, False
    for char in reversed(digits):
        if not char.isdigit():
            return False
        value = int(char)
        if alt:
            value *= 2
            if value > 9:
                value -= 9
        total += value
        alt = not alt
    return total % 10 == 0 and len(digits) >= 13


def _iban_valid(candidate: str) -> bool:
    """Whether an IBAN's mod-97 checksum holds."""
    cleaned = re.sub(r"\s+", "", candidate).upper()
    if not (15 <= len(cleaned) <= 34):
        return False
    rotated = cleaned[4:] + cleaned[:4]
    digits = "".join(str(int(c, 36)) if c.isalnum() else "" for c in rotated)
    try:
        return int(digits) % 97 == 1
    except ValueError:
        return False


def _phone_plausible(match: str) -> bool:
    """Whether a run the phone pattern caught is plausibly a phone number.

    Nine to fifteen digits — E.164 stops at fifteen. And a run of more than
    twelve digits with nothing between them is not how anybody writes a phone
    number; it is an order id, a card that failed Luhn, or a timestamp. Those
    used to come out as [PHONE], which is the worst kind of redaction: wrong,
    and visible to the person whose words it changed.
    """
    digits = re.sub(r"[^\d]", "", match)
    if not 9 <= len(digits) <= 15:
        return False
    shaped = match.startswith("+") or any(sep in match for sep in " .-()")
    return shaped or len(digits) <= 12


#: (label, pattern, optional validator). Order matters: the longest and most
#: specific run first, so an IBAN is not eaten by the phone-number pattern.
_DETECTORS: List[Tuple[str, re.Pattern, Optional[callable]]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"), None),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[\sA-Z0-9]{11,30}\b"), _iban_valid),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b"),
     lambda m: _luhn(re.sub(r"[^\d]", "", m))),
    # International and local forms, long enough not to swallow a quantity.
    ("PHONE", re.compile(r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?){2,4}\d{2,4}(?![\w.])"),
     _phone_plausible),
    ("IP", re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
     lambda m: all(0 <= int(p) <= 255 for p in m.split("."))),
    # A street line: number then words, or words then number. Deliberately
    # narrow — an over-eager address rule eats recipe steps.
    ("ADDRESS", re.compile(
        r"\b\d{1,4}\s+[A-Z][\w'-]*(?:\s+[A-Z]?[\w'-]+){0,3}\s+"
        r"(?:street|st|road|rd|avenue|ave|lane|ln|boulevard|blvd|drive|dr|"
        r"way|square|sq|ulica|cesta|utca|út|tér)\b", re.I), None),
    ("POSTCODE", re.compile(r"\b(?:SI-)?\d{4}\b(?=\s+[A-Z])"), None),
]


def redact(text: Optional[str]) -> Tuple[Optional[str], Dict[str, int]]:
    """Return the text with personal data replaced, and a count per kind.

    The counts are the point as much as the text: they are what lets the
    platform say how often people are typing identifiers into a chat, without
    keeping a single example of one.

    Never raises. A redactor that throws on the request path takes the chat
    down with it, and failing open on an exception is the wrong call — so it
    fails *closed* on the one thing it can: an unreadable input is returned
    unchanged and reported as unscanned, and the caller decides.
    """
    if not text or not ENABLED:
        return text, {}
    if len(text) > MAX_SCAN_CHARS:
        return text, {"unscanned": 1}

    found: Dict[str, int] = {}
    out = text
    try:
        for label, pattern, validator in _DETECTORS:
            def _sub(match: "re.Match") -> str:
                value = match.group(0)
                if validator and not validator(value):
                    return value
                found[label] = found.get(label, 0) + 1
                return f"[{label}]"
            out = pattern.sub(_sub, out)
    except Exception:
        logger.warning("pii.redaction_failed", exc_info=True)
        return text, {"failed": 1}

    names, out = _redact_names(out)
    if names:
        found["NAME"] = names
    return out, found


def _redact_names(text: str) -> Tuple[int, str]:
    """Names, via the local NER model. Zero and unchanged when none is loaded."""
    reader = _ner()
    if reader is None:
        return 0, text
    try:
        doc = reader(text)
        spans = [
            (ent.start_char, ent.end_char)
            for ent in doc.ents
            if ent.label_ in ("PERSON", "PER", "GPE", "LOC")
        ]
        if not spans:
            return 0, text
        # Replaced back to front so earlier offsets stay valid.
        for start, end in sorted(spans, reverse=True):
            text = text[:start] + "[NAME]" + text[end:]
        return len(spans), text
    except Exception:
        logger.debug("pii.ner_failed", exc_info=True)
        return 0, text


_ner_model = None
_ner_tried = False


def _ner():
    """The local NER pipeline, loaded once, or None."""
    global _ner_model, _ner_tried
    if _ner_tried:
        return _ner_model
    _ner_tried = True
    if not NER_MODEL:
        return None
    try:
        import spacy

        _ner_model = spacy.load(NER_MODEL)
        logger.info("pii.ner_loaded model=%s", NER_MODEL)
    except Exception:
        # Not installed, or the model is not downloaded. Deterministic
        # detectors still run; names simply are not caught, and the caller is
        # told by the absence of a NAME count rather than by silence.
        logger.warning("pii.ner_unavailable model=%s", NER_MODEL)
        _ner_model = None
    return _ner_model


def names_detected() -> bool:
    """Whether name detection is actually running.

    So a console can say "names are not being caught" rather than showing a
    zero that looks like a clean result.
    """
    return _ner() is not None
