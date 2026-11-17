"""The language the reader uses, for every text FoodChat writes for them.

The interface states its language on every call (``X-Locale``) and the gateway
forwards it, already cleaned to a language tag. Until FoodChat read it, nothing
in a prompt said which language to write in, so the model guessed from the
message: a Hungarian request usually got Hungarian, a Slovenian one got
English, and a turn with no typed text at all (compose, plan parameters, a
tool) got English in every language.

The language lives in a ContextVar set by :class:`ReplyLanguageMiddleware`,
so any code that builds a reader-facing prompt can ask :func:`current` without
threading an argument through every call. ``asyncio.create_task`` copies the
context, so work a request starts in the background keeps the request's
language.

Only prompts whose output the reader sees take :func:`instruction`. Prompts
that parse intent, rewrite queries or extract JSON stay language-agnostic:
their output is read by code, and the vocabularies they emit are English codes.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Optional

LOCALE_HEADER = "X-Locale"
DEFAULT = "en"

#: How the interface addresses the reader in each language: informally, as
#: "you". A reply that switched to the formal form, or spoke as the reader
#: ("I like to cook…" in a memory offer), read as a different product.
_REGISTER = {
    "hu": "informally (tegezés: te)",
    "sl": "informally (tikanje: ti)",
    "el": "informally (εσύ)",
}

#: The interface languages (wisefood-ui app/i18n/locales). The value is the
#: name a prompt uses, with the language's own name so the model cannot
#: mistake one for a neighbour (Slovenian for Slovak, above all).
SUPPORTED = {
    "en": "English",
    "hu": "Hungarian (magyar)",
    "sl": "Slovenian (slovenščina)",
    "el": "Greek (ελληνικά)",
}

_LANGUAGE: ContextVar[str] = ContextVar("fc_reply_language", default=DEFAULT)


def normalize(raw: Optional[str]) -> str:
    """A supported language code from a header value, or the default.

    Takes the first entry of an Accept-Language-style list and its primary
    subtag: ``hu-HU,hu;q=0.9`` is Hungarian. Anything unsupported is English,
    the language every text in FoodChat is written in to begin with.
    """
    if not raw:
        return DEFAULT
    first = str(raw).split(",", 1)[0].split(";", 1)[0].strip()
    primary = first.split("-", 1)[0].split("_", 1)[0].lower()
    return primary if primary in SUPPORTED else DEFAULT


def current() -> str:
    """The reader's language for the request in flight."""
    return _LANGUAGE.get()


def use(code: Optional[str]) -> Token:
    """Set the language outside a request (a job, a test). Returns the reset token."""
    return _LANGUAGE.set(normalize(code))


def reset(token: Token) -> None:
    _LANGUAGE.reset(token)


def name(code: Optional[str] = None) -> str:
    return SUPPORTED[normalize(code) if code else current()]


def instruction(code: Optional[str] = None, *, fields: Optional[str] = None) -> str:
    """The sentence a reader-facing prompt ends with; empty for English.

    Empty rather than "Write in English" so the English prompts, which every
    evaluation so far was run against, stay byte for byte as they were.

    ``fields`` names the part of a JSON answer the reader sees (a memory
    offer's ``statement``), for agents whose other fields are read by code
    and must stay in English: a translated diet value would be written to
    the member's profile.

    Appended at call time rather than written into the prompts: the prompt
    text is served from Langfuse, and a registered prompt is never
    overwritten there, so an edit to its text would not reach production.
    """
    lang = normalize(code) if code else current()
    if lang == DEFAULT:
        return ""
    language = SUPPORTED[lang]
    titles = (
        "Keep recipe titles exactly as they are given (they must match the "
        "meal cards the reader sees)"
    )
    register = (
        f"Address the reader {_REGISTER[lang]}, in the second person, as the "
        "app does; never write as if you were the reader."
    )
    if fields:
        return (
            f"\n\nLANGUAGE: the reader uses the app in {language}. Write {fields} "
            f"in {language}. {register} {titles}. Every other field, and every "
            "key, code and enum value, stays in English exactly as specified above."
        )
    return (
        f"\n\nLANGUAGE: the reader uses the app in {language}. Write every "
        f"sentence the reader will see in {language}, whatever language the "
        f"conversation, the profile or the data below is in. {register} "
        f"{titles}, and keep JSON keys, codes and enum values in English."
    )


def request_line(code: Optional[str] = None) -> str:
    """A one-line reminder for the user turn of a small model's prompt; empty for English.

    The session titler runs on the fast model, which followed the system
    prompt's examples (English, Title Case) over its closing language line:
    a Slovenian conversation was still named "Mediterranean High Protein
    Light Dinner". Said again in the request itself, it is followed.
    """
    lang = normalize(code) if code else current()
    if lang == DEFAULT:
        return ""
    return f"\n\n(Answer in {SUPPORTED[lang]}.)"


class ReplyLanguageMiddleware:
    """Bind the reader's language for one request. Never rejects, never raises.

    Pure ASGI, like obs_context's middleware: ``BaseHTTPMiddleware`` runs the
    app in a child task, which would hide the ContextVar from the handlers.
    ``X-Locale`` wins (the language the interface is in); Accept-Language is
    only what the device prefers, so it is the fallback.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        locale = accept = None
        for raw_key, raw_value in scope.get("headers") or []:
            key = raw_key.decode("latin-1").lower()
            if key == LOCALE_HEADER.lower():
                locale = raw_value.decode("latin-1")
            elif key == "accept-language":
                accept = raw_value.decode("latin-1")
        token = _LANGUAGE.set(normalize(locale or accept))
        try:
            await self.app(scope, receive, send)
        finally:
            _LANGUAGE.reset(token)
