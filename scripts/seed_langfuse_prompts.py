#!/usr/bin/env python3
"""
Seed FoodChat's prompt registry into Langfuse — outside app boot.

Idempotent: creates ONLY the prompts that are missing (the Langfuse UI is the
source of truth; existing prompts are never overwritten). Useful for a fresh
Langfuse project or from CI. The app also does this automatically in a daemon
thread at startup (see src/main.py lifespan), so running this by hand is
optional.

Requires the three Langfuse env vars to be set (LANGFUSE_PUBLIC_KEY,
LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL/LANGFUSE_HOST for self-hosted);
without them the registry has nothing to talk to and this prints a notice and
exits cleanly.

Usage:
    LANGFUSE_PUBLIC_KEY=pk-lf-... LANGFUSE_SECRET_KEY=sk-lf-... \\
    LANGFUSE_BASE_URL=http://langfuse-web:3000 \\
    python3 scripts/seed_langfuse_prompts.py
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing anything that reads env at import time.
load_dotenv()

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from backend.observability import langfuse_enabled  # noqa: E402
from prompts import ALL_PROMPTS, sync_prompts  # noqa: E402


def main() -> int:
    if not langfuse_enabled():
        print(
            "Langfuse is not enabled (set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY "
            "and install langfuse>=3.0). Nothing to seed."
        )
        return 0

    result = sync_prompts()
    print(f"Done ({len(ALL_PROMPTS)} registry prompts): {result}")
    # Non-zero only if a create genuinely failed, so CI can gate on it.
    return 1 if result.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
