"""
Attribution domain models — provenance of externally-answered chat turns.

When FoodChat delegates a question to another WiseFood application (M1:
FoodScholar), the answer carries an ``Attribution`` so the UI can label the
source, render citations, and offer a deep link ("Learn more in FoodScholar").

Layering rule: like models.recipe, this module imports nothing from agents
or services.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass(frozen=True)
class Citation:
    """One cited source backing a delegated answer."""

    title: str
    source_type: Literal["article", "guideline"]
    url: Optional[str] = None
    label: Optional[str] = None      # short inline label, e.g. "G1"


@dataclass(frozen=True)
class Attribution:
    """Provenance of an answer produced by another WiseFood application.

    ``learn_more_url`` is a UI-relative path (e.g. ``/foodscholar?q=...``) —
    FoodChat does not know the frontend origin; the UI resolves it.
    """

    source: Literal["foodscholar"]
    confidence: Optional[Literal["high", "medium", "low"]] = None
    citations: list[Citation] = field(default_factory=list)
    learn_more_url: Optional[str] = None
