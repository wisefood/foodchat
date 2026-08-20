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
- **Langfuse prompt check**: `foodscholar/qa-memory-extractor` must carry the
  worry→goal mapping ("heart health → reduce_fat"). Existing Langfuse prompts
  are never overwritten on deploy — push the updated fallback from
  `foodscholar/src/backend/prompts.py` as a new production-labeled version,
  or the heart-health beat (row 11) won't nudge.
- **Irish guidance check**: ask the red-meat question once with the demo
  member's country — confirm G-labeled citations come back from the
  guideline corpus, not only articles.
- **Nudge check**: `python3 scripts/check_memory_nudge.py` (same env vars as
  seed_demo.py) — runs the heart-health beat end-to-end and prints a verdict
  naming the broken link (stale image / Langfuse prompt / suppressed by an
  earlier accept) when the chip won't fire.
- Nudge phrasings that work: a stated family worry (*"we're worried about our
  heart health — is red meat harmful?"*) or a harm question about a specific
  food (*"is red meat harmful?"*). Diet/regimen questions (*"is keto safe?"*)
  and positive questions (*"is salmon good for me?"*) deliberately never nudge.
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
| 10 | Type: *"Give me a plan for tomorrow"* → a **slider card** rides under the fresh daily plan (cooking time / difficulty / goal) → drag **cooking time to 20 min**, hit **Apply** → the plan refines, the card shows the new setting as current | Interactive plan parameters: optional knobs replace interrogation — no question round, deterministic apply, values persist as known facts |
| 11 | Switch to the **FoodScholar app** (same member) and ask: *"We're worried about our family's heart health — is eating red meat often harmful?"* → cited answer (articles + national guidance, G-labels), confidence, follow-ups → the 🧠 chip appears: *"…aim for less saturated fat?"* → click **Remember** | Cross-app goal capture: worry → consented `reduce_fat` goal on the member profile (provenance `source: foodscholar`) |
| 12 | Back in FoodChat: ask for a fresh household plan → it leans lower-fat (the goal is now a hard low-fat candidate filter + soft signal, merged across diners) → my-profile memory panel shows the goal with its FoodScholar provenance | One goal store, two apps: Q&A concerns shape family meal planning |
| 13 | **Apply** the plan → dashboard shows today's meals; heart a recipe from a plan card | Cross-app journey: gateway meal-plan store + server-side favorites (M2) |
| 14 | (Optional) Langfuse tab: the full trace of the turn you just ran | Observability (M5) |

## Failure modes & recovery lines

- **FoodScholar down** → FoodChat apologizes in-chat and keeps planning
  ("graceful degradation is a feature — the plan flow never blocks on Q&A").
- **A directive has no verified match** → the honest-failure reply with the
  nearest miss IS the demo beat — don't apologize for it, highlight it.
- **Slow LLM turn** → narrate the transparency of the last plan while waiting.
- **Demo account broken** → guest mode: one click, ephemeral household,
  the whole journey works (budgets apply).

## The reconciliation beat (C3 / R1-D10)

Run this inside row 12, once the household plan is on the canvas. It is the
answer to "does one member's inferred goal quietly win?"

1. Point at the ledger: **"no peanuts — Tom"**, **"vegetarian — Anna"**. Each row
   names the diner it protects, not the whole table. Safety is a union: nobody
   is out-voted.
2. Point at the goal rows. Dimitris's **less saturated fat** (the goal
   FoodScholar captured in row 11) shows as applied — it sets the plan's numeric
   targets. Anna's goal shows amber, as **relaxed**, and the tooltip says why:
   applied as a preference, not a target, because another diner's goal sets the
   targets.
3. The line to say: *"We could have stacked every diner's targets and produced
   nothing edible, or picked one and hidden it. We pick one and say so, and the
   others still steer the ranking."*
4. If asked what happens when two diners want the same thing: it is recorded as
   agreement, both names on one applied row — not a conflict.

## Free exploration (R2-W1)

Attendees are **not restricted to this script**, and it is worth saying so
unprompted. In a guest session a visitor can: create their own household and add
members, set arbitrary allergens, diets and goals, ask free-text questions in
either app, edit or delete anything in the memory panel, and watch a change in
one app land in the other. The script is a guided path through the same surface,
not a rail.

## Booth data handling (attendee input) — R1-D8, D11

Attendees will type real preferences, real allergies, sometimes a real health
worry, into a shared laptop. The procedure, not the intention, is what protects
them.

**Rule: attendees never touch the demo account.** Hand them a **guest session**
(one click on the login screen). A guest is a real but short-lived Keycloak user
with its own household and member, so nothing they enter can reach the demo
household or another attendee's data. Guest sessions live in `sessionStorage`,
so closing the tab already isolates the next visitor.

**Between attendees, every time:**

1. Have them click **"Erase my data now"** in the amber guest banner — or click
   it for them before handing the laptop on. It calls
   `DELETE /api/v1/system/guest`, which deletes the guest's household, its
   members, its FoodChat sessions and the Keycloak user itself, then drops the
   local session and returns to the login screen. A toast confirms the erase; if
   it says the erase could not be confirmed, fall back to step 3.
2. Open a **new** guest session for the next attendee. Never continue in one
   that has someone else's history.
3. Manual fallback if the button ever fails (offline, gateway hiccup): close the
   tab to drop the session, and the expiry reaper deletes the account and all
   its data within `GUEST_TTL_SECONDS` regardless. Nothing survives the day.

**Say it out loud when someone asks.** "You're in a sandbox that gets deleted —
either when you click erase, or automatically when it expires. Nothing you type
here is used to train anything." That sentence is also the ethics-block clause
in the paper.

**What we log while they use it:** the same LLM traces every turn produces
(Langfuse) and application logs, both tied to the ephemeral guest id, which is
destroyed with the account. If an attendee asks to see the observability tab,
show it on the demo account rather than theirs.

## Talking points (research framing)

- Consent-gated, **scrutable user modeling**: nothing durable is learned
  silently; every memory has provenance and a forget button.
- **Multi-stakeholder constraint satisfaction**: hard unions (safety) vs
  weighted soft preferences, attributed in the ledger.
- **Verified natural-language editing**: directives become measurable
  predicates checked against a nutrition store — fail-closed, honest misses.
- **Cross-application grounding**: one profile personalizes chat, Q&A, and
  recipes; answers cite literature and national guidelines.
