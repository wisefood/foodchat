# RecSys '26 Demo Script — WiseFood / FoodChat

The live walkthrough, beat by beat, with the feature each beat demonstrates
and what to say if something misbehaves. Rehearse against `demo.wisefood-project.eu`.

## Preflight (day before + hour before)

```bash
# 1. Seed / verify the demo household (idempotent; fails loudly if a Greek
#    anchor dish is missing from RecipeWrangler)
WISEFOOD_API_URL=https://demo.wisefood-project.eu/rest \
DEMO_USERNAME=... DEMO_PASSWORD=... python3 scripts/seed_demo.py

# 2. Health checks through the gateway
curl -s $BASE/api/v1/foodchat/status
curl -s $BASE/api/v1/foodscholar/status
curl -s $BASE/api/v1/recipewrangler/status
```

- Log in once in a fresh browser profile → the **consent bar** must appear
  (it's part of the demo — don't pre-accept it).
- Open Langfuse (console → operations) in a second tab for the observability beat.
- Have a **guest-mode** tab ready as fallback identity if the demo account misbehaves.

## The walkthrough (~8 minutes)

| # | Say / do | Feature on display |
|---|---|---|
| 1 | Log in as the demo user → point at the **consent bar**, click Accept | Purpose-limited consent, recorded with timestamp + IP (M3) |
| 2 | Pick **Dimitris** → dashboard → FoodChat | Hub-and-spoke platform, shared member identity |
| 3 | In the diner picker, add **Anna** and **Tom** → "Cooking for: Dimitris, Anna, Tom" banner | Household constraint satisfaction (M3) |
| 4 | Type: *"Plan my week — and I like eating pastitsio and fakes, can you work them in?"* | Seeded planning: both dishes resolved, allergy-checked, pinned + spread (M2). The favorites offer may fire first — accepting it is also a fine beat. |
| 5 | When the plan lands, point at: **pinned chips** ("requested by you"), the **constraint ledger** ("vegetarian — Anna", "no peanuts — Tom"), **kcal chips + images**, the **personalization line**, and open the **quality panel** | Transparency pack (M4a) + multi-diner hard constraints |
| 6 | Type: *"Is keto safe for teenagers?"* | FoodScholar bridge (M1): cited answer, confidence, "Learn more in FoodScholar →" — click it to show the handoff, come back |
| 7 | Type: *"I don't like the meal on Tuesday, swap it for something lighter"* → answer the follow-up ("the dinner") | Verified slot editing (M4b): the reply carries the **before → after kcal proof**, the diff chip renders, one slot changed, pins untouched |
| 8 | Type: *"I really don't like mushrooms"* → the 🧠 nudge appears → click **Remember** | Consented memory (M3): then open my-profile → **"What WiseFood remembers"** panel, show provenance + forget button |
| 9 | Thumbs-down one meal, ask for another plan → the disliked recipe is gone; the grader history now cites it | Feedback loop (M3) |
| 10 | **Apply** the plan → dashboard shows today's meals; heart a recipe from a plan card | Cross-app journey: gateway meal-plan store + server-side favorites (M2) |
| 11 | (Optional) Langfuse tab: the full trace of the turn you just ran | Observability (M5) |

## Failure modes & recovery lines

- **FoodScholar down** → FoodChat apologizes in-chat and keeps planning
  ("graceful degradation is a feature — the plan flow never blocks on Q&A").
- **A directive has no verified match** → the honest-failure reply with the
  nearest miss IS the demo beat — don't apologize for it, highlight it.
- **Slow LLM turn** → narrate the transparency of the last plan while waiting.
- **Demo account broken** → guest mode: one click, ephemeral household,
  the whole journey works (budgets apply).

## Talking points (research framing)

- Consent-gated, **scrutable user modeling**: nothing durable is learned
  silently; every memory has provenance and a forget button.
- **Multi-stakeholder constraint satisfaction**: hard unions (safety) vs
  weighted soft preferences, attributed in the ledger.
- **Verified natural-language editing**: directives become measurable
  predicates checked against a nutrition store — fail-closed, honest misses.
- **Cross-application grounding**: one profile personalizes chat, Q&A, and
  recipes; answers cite literature and national guidelines.
