"""
Per-model capability profiles for the Groq pool.

A model id is not a drop-in replacement for another one, and two differences
bite this service in particular:

* **Reasoning families spend the completion budget on hidden reasoning.** Left
  at the provider default, ``openai/gpt-oss-*`` and ``qwen3*`` return that
  deliberation inside ``content``. Every agent in ``agents.py`` does
  ``json.loads(result.content)``, so the payload arrives either preceded by
  paragraphs of prose or truncated before it closes — and because each agent
  catches the parse failure and returns a fallback, the symptom is not an
  error but a plan that quietly loses its grader ranking, its extracted diet
  tags, or its written response.
* **Reasoning knobs are family-specific.** ``reasoning_format`` and
  ``reasoning_effort`` are accepted by the Groq reasoning models and rejected
  by the Llama ones. ``ChatGroq`` is configured ``extra="ignore"``, so a
  parameter the client does not model is dropped silently rather than raising
  — which means a wrong knob fails at the provider, one layer further away.

Rather than teach thirteen agents which family they are talking to, the quirks
are declared once here and applied by ``backend.groq.GroqConnectionPool``.
Adding a model to the deployment means adding a row here, not editing agents.

Deliberately narrower than FoodScholar's equivalent module: FoodChat talks to
Groq only, so there are no OpenAI rows and no ``supports_temperature`` axis
(every Groq family accepts a temperature). Keep it that way — a profile field
with no consumer is a field that goes stale.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelProfile:
    """What one model family needs, and what it will refuse."""

    family: str
    # A reasoning model: needs its deliberation kept out of `content`, and a
    # completion budget large enough for reasoning *plus* the JSON payload.
    reasoning: bool = False
    # Groq-specific; 'hidden' keeps reasoning out of the response entirely.
    reasoning_format: Optional[str] = None
    reasoning_effort: Optional[str] = None
    # A floor, never an override: a caller asking for more keeps its value.
    min_max_tokens: Optional[int] = None
    # Whether the family accepts the Groq reasoning_* params at all. Sending
    # them to a family that does not is a provider-side 400, so they are
    # dropped here instead.
    supports_reasoning_params: bool = False
    # False for families with no row; drives the one-time warning.
    known: bool = True


# Matched as a substring against the lowercased model id, first hit wins, so
# more specific markers must precede more general ones.
_FAMILIES: Tuple[Tuple[str, ModelProfile], ...] = (
    (
        "gpt-oss",
        ModelProfile(
            family="gpt-oss",
            reasoning=True,
            reasoning_format="hidden",
            reasoning_effort="low",
            # Reasoning is charged against the same budget as the payload, and
            # the batch grader's response scales with FOODCHAT_MAX_PLANS_TO_SCORE.
            min_max_tokens=2048,
            supports_reasoning_params=True,
        ),
    ),
    (
        "qwen3",
        ModelProfile(
            family="qwen3",
            reasoning=True,
            reasoning_format="hidden",
            # No reasoning_effort default: the family accepts the parameter but
            # its levels are not the same scale as gpt-oss's, so the provider
            # default is the honest choice until it has been measured.
            min_max_tokens=2048,
            supports_reasoning_params=True,
        ),
    ),
    (
        "deepseek-r1",
        ModelProfile(
            family="deepseek-r1",
            reasoning=True,
            reasoning_format="hidden",
            min_max_tokens=2048,
            supports_reasoning_params=True,
        ),
    ),
    ("kimi-k2", ModelProfile(family="kimi-k2")),
    ("llama", ModelProfile(family="llama")),
    ("mixtral", ModelProfile(family="mixtral")),
    ("gemma", ModelProfile(family="gemma")),
)

# Ids the provider has shut down, with the replacement it named. A retired id
# still matches its family row, so nothing else here would notice: the request
# simply fails at the provider. Warning when the client is constructed puts the
# shutdown date and the replacement in the logs of the process that is about to
# fail, which is where an operator will be looking.
RETIRED = {
    "llama-3.1-8b-instant": ("2026-08-16", "openai/gpt-oss-20b"),
    "llama-3.3-70b-versatile": (
        "2026-08-16",
        "openai/gpt-oss-120b or qwen/qwen3.6-27b",
    ),
}

_UNKNOWN = ModelProfile(
    family="unknown",
    # An unregistered id gets nothing injected, but a knob the caller passed
    # explicitly is still forwarded: that request was deliberate, and silently
    # stripping it would change behaviour without saying so.
    supports_reasoning_params=True,
    known=False,
)

_warned_unknown: set = set()
_warned_retired: set = set()


def profile_for(model: str) -> ModelProfile:
    """The capability profile for ``model`` (never raises)."""
    name = (model or "").lower()

    if name in RETIRED and name not in _warned_retired:
        _warned_retired.add(name)
        shutdown, replacement = RETIRED[name]
        logger.warning(
            "Model '%s' was shut down by the provider on %s; calls to it will "
            "fail. Recommended replacement: %s.",
            model, shutdown, replacement,
        )

    for marker, profile in _FAMILIES:
        if marker in name:
            return profile

    if model not in _warned_unknown:
        _warned_unknown.add(model)
        logger.warning(
            "Model '%s' has no capability profile; calling it with caller "
            "defaults only. Add a row to backend.model_profiles if it is a "
            "reasoning model.",
            model,
        )
    return _UNKNOWN


def apply_profile(model: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Reconcile caller kwargs with what ``model`` actually supports.

    Injects the family's defaults where the caller expressed no preference,
    raises ``max_tokens`` to the family floor, and drops parameters the family
    would reject. Returns a new dict; ``kwargs`` is left alone.
    """
    profile = profile_for(model)
    resolved = dict(kwargs)

    if not profile.supports_reasoning_params:
        for key in ("reasoning_format", "reasoning_effort"):
            if key in resolved:
                logger.debug(
                    "Dropping %s for %s: unsupported by the %s family",
                    key, model, profile.family,
                )
                resolved.pop(key)
    elif profile.reasoning:
        # 'hidden' is the whole point: it keeps deliberation out of `content`,
        # where json.loads would choke on it.
        if profile.reasoning_format and "reasoning_format" not in resolved:
            resolved["reasoning_format"] = profile.reasoning_format
        if profile.reasoning_effort and "reasoning_effort" not in resolved:
            resolved["reasoning_effort"] = profile.reasoning_effort

    if profile.min_max_tokens:
        current = resolved.get("max_tokens")
        if not isinstance(current, int) or current < profile.min_max_tokens:
            resolved["max_tokens"] = profile.min_max_tokens

    return resolved
