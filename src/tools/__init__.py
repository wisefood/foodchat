"""
FoodChat's local tool surface.

The planning agent was a fixed chain: one classification per turn picked one
handler, and everything that handler could not do was unreachable. There was no
way to ask "summarise my week" or "redo Thursday" — a member asking for either
got a whole new plan, which is how a verified slot edit gets silently thrown
away by a refinement.

This is the same protocol FoodChat already CONSUMES from RecipeWrangler
(`GET /api/v2/tools` for a manifest, `POST /api/v2/tools/<name>` to invoke —
see `services.plan_client`). Rather than invent a second shape, FoodChat now
speaks it too: a declarative registry of typed tools, discoverable at runtime,
callable by name, MCP-shaped so a model can be handed the manifest directly.

    from tools import MANIFEST, describe_tools, invoke

    MANIFEST                      → [{name, description, parameters, ...}]
    describe_tools()              → prose for a prompt
    invoke("replace_day", {...})  → the tool's result dict

Rules every tool here follows:

* **Deterministic unless it says otherwise.** A tool that reads a plan does no
  model call at all; the numbers it reports are summed, not estimated. Tools
  that regenerate content declare ``uses_model: True`` so a caller can budget.
* **Session-scoped and ownership-checked by the caller.** A tool takes a
  session id and trusts that the router already proved the member owns it —
  same contract as every service in this codebase.
* **Honest about missing data.** Nutrition is absent on some recipes; a total
  says how many meals it could actually see rather than implying a complete
  figure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """A tool could not do what was asked, for a reason worth telling the member.

    Distinct from a bug: the message is member-facing prose, and the router
    turns it into a 4xx rather than a 500.
    """


@dataclass(frozen=True)
class Tool:
    """One callable capability, declared the way a model can read it."""

    name: str
    summary: str
    description: str
    parameters: dict
    handler: Callable[..., dict]
    # True when the tool spends model calls, so a caller can decide whether it
    # can afford this one inside a turn that has already done grading.
    uses_model: bool = False
    # What the tool changes, if anything. A reader is always safe to retry.
    mutates: bool = False
    examples: tuple[str, ...] = field(default_factory=tuple)

    def as_manifest(self) -> dict:
        return {
            "name": self.name,
            "summary": self.summary,
            "description": self.description,
            "parameters": self.parameters,
            "uses_model": self.uses_model,
            "mutates": self.mutates,
            "examples": list(self.examples),
        }


_REGISTRY: dict[str, Tool] = {}


def register(tool: Tool) -> Tool:
    """Add a tool to the registry. Names are unique; a clash is a bug."""
    if tool.name in _REGISTRY:
        raise ValueError(f"Duplicate tool name: {tool.name}")
    _REGISTRY[tool.name] = tool
    return tool


def tool(
    name: str,
    *,
    summary: str,
    description: str,
    parameters: dict,
    uses_model: bool = False,
    mutates: bool = False,
    examples: tuple[str, ...] = (),
):
    """Decorator form: declare a function as a tool."""

    def wrap(fn: Callable[..., dict]) -> Callable[..., dict]:
        register(Tool(
            name=name, summary=summary, description=description,
            parameters=parameters, handler=fn,
            uses_model=uses_model, mutates=mutates, examples=examples,
        ))
        return fn

    return wrap


def _ensure_loaded() -> None:
    """Import the tool modules so their registrations run.

    Deferred rather than imported at module load: the tools reach into services
    that import this package, and a top-level import would be a cycle.
    """
    if _REGISTRY:
        return
    from tools import plan_tools  # noqa: F401


def all_tools() -> list[Tool]:
    _ensure_loaded()
    return [_REGISTRY[n] for n in sorted(_REGISTRY)]


def get(name: str) -> Optional[Tool]:
    _ensure_loaded()
    return _REGISTRY.get(name)


def manifest() -> list[dict]:
    """The machine-readable tool list, same shape RecipeWrangler serves."""
    return [t.as_manifest() for t in all_tools()]


# Kept as a property-like callable rather than a module constant so importing
# this package never triggers the service imports the tools need.
MANIFEST = manifest


def describe_tools() -> str:
    """The manifest as prose, for a prompt.

    Mirrors `plan_client.describe_options`: a model reasons better about a
    paragraph naming what it can do than about a JSON schema dump.
    """
    lines = []
    for t in all_tools():
        args = ", ".join(sorted((t.parameters.get("properties") or {}).keys()))
        cost = " (spends a model call)" if t.uses_model else ""
        change = " (changes the plan)" if t.mutates else ""
        lines.append(f"- {t.name}({args}): {t.summary}{cost}{change}")
    return "\n".join(lines)


def _validate(tool_: Tool, arguments: dict) -> dict:
    """Check arguments against the declared schema. Deliberately small.

    Only what the registry actually promises: required keys present, unknown
    keys rejected, and integers coerced from the strings an HTTP caller sends.
    A wrong argument should fail here with a readable message, not deep inside
    a planner.
    """
    props = tool_.parameters.get("properties") or {}
    required = tool_.parameters.get("required") or []
    args = dict(arguments or {})

    unknown = sorted(set(args) - set(props))
    if unknown:
        raise ToolError(
            f"{tool_.name} does not take {', '.join(unknown)}. "
            f"It takes: {', '.join(sorted(props)) or 'no arguments'}."
        )
    missing = [k for k in required if args.get(k) is None]
    if missing:
        raise ToolError(f"{tool_.name} needs {', '.join(missing)}.")

    for key, spec in props.items():
        if key not in args or args[key] is None:
            continue
        want = spec.get("type")
        if want == "integer":
            try:
                args[key] = int(args[key])
            except (TypeError, ValueError):
                raise ToolError(f"{key} must be a whole number.") from None
            lo, hi = spec.get("minimum"), spec.get("maximum")
            if lo is not None and args[key] < lo:
                raise ToolError(f"{key} must be {lo} or more.")
            if hi is not None and args[key] > hi:
                raise ToolError(f"{key} must be {hi} or less.")
        elif want == "string":
            args[key] = str(args[key])
            allowed = spec.get("enum")
            if allowed and args[key] not in allowed:
                raise ToolError(
                    f"{key} must be one of: {', '.join(allowed)}."
                )
    return args


def invoke(name: str, arguments: Optional[dict] = None) -> dict:
    """Run a tool by name. Raises ToolError for anything the caller can fix."""
    _ensure_loaded()
    tool_ = _REGISTRY.get(name)
    if tool_ is None:
        raise ToolError(
            f"There is no tool called {name!r}. "
            f"Available: {', '.join(sorted(_REGISTRY))}."
        )
    args = _validate(tool_, arguments or {})
    logger.info("tool %s(%s)", name, ", ".join(f"{k}={v!r}" for k, v in args.items()))
    return tool_.handler(**args)
