"""
Plan scorer, step 1 — free text into a ``PastedPlan``.

Deterministic first, the LLM only for what the scanner could not place:

    scan(text)                        line scanner: day headings ("Monday",
                                      "Day 2", "Tue:"), ``slot:`` prefixes,
                                      bullets under a slot heading, "oats for
                                      breakfast" phrases, trailing
                                      "(ingredients)". No LLM.
    parse_plan_text(text, parser)     the scanner's reading when it placed
                                      every line from structure alone;
                                      otherwise one PlanTextParser call with
                                      the scanner's reading as a hint, then
                                      normalised and checked against the text.
    looks_like_plan_listing(text)     cheap structural test the orchestrator
                                      uses for explicit "rate this: …" turns.
    needs_shape_question(plan)        one unlabelled block that is plainly
                                      several days?
    days_from_answer(text)            "3 days" / "the whole week" / "just today"
    split_into_days(plan, n)          regroup one block into days

Rules enforced in code as well as in the prompt, because each one is a way to
misreport the member's plan, and a prompt served from Langfuse can drift from
the copy in this repository:

- ingredients are kept only when the member wrote them — every listed item
  must appear in the pasted text, otherwise the whole list is dropped and a
  warning says so;
- an ``unparsed`` line must appear verbatim in the text;
- a dish title the text does not support is dropped, not scored;
- unmentioned slots stay absent. A partial day is not an error.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from models.pasted_plan import (
    MAIN_SLOTS,
    MAX_DAYS,
    PLAN_TYPES,
    SLOTS,
    PastedDay,
    PastedMeal,
    PastedPlan,
)

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

_DAY_WORDS = {
    "monday": 1, "mon": 1,
    "tuesday": 2, "tues": 2, "tue": 2,
    "wednesday": 3, "wed": 3,
    "thursday": 4, "thurs": 4, "thur": 4, "thu": 4,
    "friday": 5, "fri": 5,
    "saturday": 6, "sat": 6,
    "sunday": 7, "sun": 7,
}

_SLOT_WORDS = {
    "breakfast": "breakfast", "brekkie": "breakfast",
    "lunch": "lunch",
    "dinner": "dinner", "supper": "dinner",
    "snack": "snack", "snacks": "snack", "dessert": "snack",
    "brunch": "other",
}
_SLOT_ALT = "breakfast|brekkie|brunch|lunch|dinner|supper|snacks|snack|dessert"

# A day heading must END where a heading ends — a delimiter, the end of the
# line, or a slot word. Without the lookahead "Sun-dried tomato pasta" is
# Sunday's pasta and "Sat down with a coffee" starts Saturday.
_DAY_HEAD = re.compile(
    r"^(?:(?P<name>monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|tues|thurs|thur|mon|tue|wed|thu|fri|sat|sun)\.?"
    r"|day\s*(?P<num>\d{1,2}))"
    rf"(?=\s*(?:$|[:\-–—.),](?:\s|$))|\s+(?:{_SLOT_ALT})\b)"
    r"\s*[:\-–—.),]?\s*",
    re.IGNORECASE,
)
# "breakfast: oats", "Lunch - soup", "dinner = salmon". A bare hyphen needs a
# space after it, so "dinner-party lasagna" is not a dinner.
_SLOT_MARK = re.compile(
    rf"\b(?P<slot>{_SLOT_ALT})\b\s*(?:[:=–—]|-(?=\s))\s*", re.IGNORECASE,
)
# A slot heading with nothing after it: the dishes follow as bullets.
_SLOT_ONLY = re.compile(rf"^(?P<slot>{_SLOT_ALT})\s*[:\-–—]?\s*$", re.IGNORECASE)
# "oats for breakfast, lentil soup for lunch". Prose, so a reading built from
# it is only a hint for the parser, never used on its own.
_FOR_SLOT = re.compile(
    rf"(?P<dish>[^,;:.!?]+?)\s+(?:for|at)\s+(?P<slot>{_SLOT_ALT})\b", re.IGNORECASE,
)
# Each lead word must be followed by a space: "haddock for dinner" is not "had
# dock", and "eaten mess" is a dish, not "eat en mess".
_DISH_LEAD = re.compile(
    r"^(?:(?:and|then|plus|also|&)\s+)*(?:i\s+)?(?:usually\s+)?"
    r"(?:(?:had|have|ate|eat|eating|get)\s+)?",
    re.IGNORECASE,
)
# The clause that introduces a listing belongs to the request, not to the first
# dish: "Rate for me a daily plan consisting of fried eggs" is fried eggs. Only
# these connectors count — a bare "of" would eat "bowl of porridge", which the
# measure words already handle.
_PROSE_INTRO = re.compile(
    r"^.*?\b(?:consisting of|comprising|made up of|made of|composed of|containing"
    r"|which (?:is|was|includes)|that (?:is|was|includes))\s+",
    re.IGNORECASE,
)
# "peanut butter" is not butter, "oat milk" is not milk, "vegan cheese" and
# "dairy-free yoghurt" are not dairy. Masked before a dairy or lactose check.
PLANT_DAIRY = re.compile(
    r"\b(?:peanut|almond|cashew|hazelnut|nut|seed|sunflower|apple|coconut|oat|soy|soya|rice"
    r"|plant|vegan|dairy[\s-]*free|lactose[\s-]*free)"
    r"[\s-]+(?:butter|milk|cream|yogh?urt|cheese)\b",
    re.IGNORECASE,
)
# Words joining the parts of one dish: "roast chicken WITH potatoes",
# "houmous AND pitta", "beans ON toast".
_DISH_PARTS = re.compile(r"\s+(?:served with|with|and|&|plus|on|in|over)\s+|\s*[,+/]\s*", re.IGNORECASE)
# The last word of a part that names a kind of dish rather than a food: the
# head of "tuna nicoise salad" is nicoise, of "lentil soup" lentil.
GENERIC_DISH_WORDS = frozenset({
    "salad", "soup", "stew", "bowl", "plate", "dish", "meal", "curry", "bake",
    "casserole", "platter", "traybake",
})


def dish_heads(title: str) -> list[str]:
    """The food each part of a dish title names — its last content word.

    "roast chicken with potatoes" -> ["chicken", "potato"]; "vegetable
    lasagna" -> ["lasagna"]; "tuna nicoise salad" -> ["nicoise"]. A recipe
    whose name lacks one of them leaves that part of the member's dish out.
    """
    heads = []
    for part in _DISH_PARTS.split(title or ""):
        tokens = content_tokens(part)
        if not tokens:
            continue
        head = tokens[-1]
        if head in GENERIC_DISH_WORDS and len(tokens) > 1:
            head = tokens[-2]
        if head not in heads:
            heads.append(head)
    return heads


_TRAILING_PARENS = re.compile(r"^(?P<title>.*?)\s*\((?P<inner>[^()]*)\)\s*$")
_BULLET = re.compile(r"^\s*(?:[-*•·>#]+|\d+[.)])\s*")
_EDGE_PUNCT = " \t,;.|-–—:\"'"

# Words that say nothing about which dish it is. Used for "does the text
# support this title" and, in grounding, for title similarity.
STOPWORDS = frozenset({
    "a", "an", "and", "the", "of", "with", "w", "in", "on", "at", "to", "for",
    "my", "our", "some", "style", "homemade", "home", "made", "easy", "quick",
    "simple", "fresh", "or", "plus", "side",
})
# Amounts and servings. Dropped from what the MEMBER wrote — "a bowl of
# porridge" is porridge — but kept in a catalogue title, where they name the
# dish: "Honey banana cups" is not porridge with banana and honey. Dropping
# them on both sides matched exactly that pair in the first live run.
MEASURE_WORDS = frozenset({
    "bowl", "plate", "cup", "cups", "piece", "pieces", "slice", "slices",
    "g", "ml", "tbsp", "tsp",
})
_STOPWORDS_AND_MEASURES = STOPWORDS | MEASURE_WORDS

_NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}

# Spellings of the same food. A member writing "vegetable lasagne" and a
# catalogue holding "Roasted vegetable lasagna" mean one dish, and a literal
# word match calls them two. Transliterations (hummus, falafel) vary the same
# way. Keys are alternative spellings, values the form both sides are compared
# in; ``spelling_variants`` searches the catalogue for the other forms too.
SPELLING_VARIANTS = {
    "lasagne": "lasagna",
    "yoghurt": "yogurt", "yogourt": "yogurt", "yoghourt": "yogurt",
    "omelette": "omelet",
    "doughnut": "donut",
    "chilli": "chili", "chile": "chili",
    "bolognaise": "bolognese",
    "houmous": "hummus", "hoummos": "hummus", "humous": "hummus",
    "felafel": "falafel",
    "tabouli": "tabbouleh", "tabouleh": "tabbouleh",
    "pitta": "pita",
    "filo": "phyllo",
    "kabob": "kebab", "kebap": "kebab", "kebob": "kebab",
    "savoury": "savory",
    "flavour": "flavor",
    "wholemeal": "wholewheat", "wholegrain": "wholewheat",
    "porage": "porridge",
    "gnocci": "gnocchi",
    "mayonaise": "mayonnaise",
    "ceasar": "caesar",
}
_SPELLING_GROUPS: dict[str, list[str]] = {}
for _variant, _canonical in SPELLING_VARIANTS.items():
    _SPELLING_GROUPS.setdefault(_canonical, []).append(_variant)
# How many extra spellings of one title the catalogue is searched for.
MAX_SPELLING_QUERIES = 2


def singular(word: str) -> str:
    """A symmetric singular form: "berries" and "berry" both become "berry".

    Not ``pantry_service.singular``: that stem is built to be re-inflected into
    a regex ("berries" → "berri", matched with an optional suffix), so it leaves
    "berry" and "berries" as two different words. Comparing two titles word by
    word needs both spellings to land on the same string.
    """
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("oes"):
        return word[:-2]
    if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def fold_accents(text: str) -> str:
    """"crème fraîche" and "creme fraiche" are the same words."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def content_tokens(text: str, keep_measures: bool = False) -> list[str]:
    """Lower-cased, singular, stop-word-free words of a dish title or text.

    Accents are folded and alternative spellings normalised, so "lasagne" and
    "lasagna" are one word. ``keep_measures`` keeps amount words ("cups",
    "slices") — for catalogue titles, where they are part of the dish's name.
    """
    dropped = STOPWORDS if keep_measures else _STOPWORDS_AND_MEASURES
    words = re.findall(r"[a-z]+", fold_accents(text or "").lower())
    tokens = []
    for word in words:
        if len(word) < 2 or word in dropped:
            continue
        stem = singular(word)
        tokens.append(SPELLING_VARIANTS.get(stem, SPELLING_VARIANTS.get(word, stem)))
    return tokens


def spelling_variants(title: str, limit: int = MAX_SPELLING_QUERIES) -> list[str]:
    """The same title spelled the other ways a catalogue might hold it.

    One word is swapped at a time, so "vegetable lasagna" also searches for
    "vegetable lasagne". Returns [] when no word has a known variant.
    """
    words = re.findall(r"[^\W\d_]+|\W+", fold_accents(title or ""), re.UNICODE)
    out: list[str] = []
    for index, word in enumerate(words):
        lowered = word.lower()
        if not lowered.isalpha():
            continue
        canonical = SPELLING_VARIANTS.get(lowered, lowered)
        others = [canonical] + _SPELLING_GROUPS.get(canonical, [])
        for other in others:
            if other == lowered or len(out) >= limit:
                continue
            swapped = list(words)
            swapped[index] = other
            candidate = "".join(swapped)
            if candidate not in out:
                out.append(candidate)
    return out[:limit]


def _clean_line(raw: str) -> str:
    line = raw.replace("**", "").replace("__", "")
    line = _BULLET.sub("", line)
    return line.strip()


def _clean_title(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip(_EDGE_PUNCT)).strip()


def _meal_from_segment(slot: str, segment: str) -> Optional[PastedMeal]:
    """One dish from the text after a slot marker; ``(…)`` becomes ingredients."""
    text = _clean_title(segment)
    if not text:
        return None
    ingredients = quantity = None
    parens = _TRAILING_PARENS.match(text)
    if parens and parens.group("title").strip():
        inner = parens.group("inner").strip()
        if re.match(r"^\d", inner) and "," not in inner:
            quantity = inner                 # "(2 servings)", "(300 g)"
        elif re.search(r"[a-zA-Z]", inner):
            # A list, with or without amounts: "(150 g yogurt, 20 g walnuts)"
            ingredients = inner
        text = _clean_title(parens.group("title"))
    return PastedMeal(slot=slot, title=text, ingredients=ingredients, quantity_note=quantity)


@dataclass
class ScanResult:
    plan: PastedPlan
    # True when any dish came from "X for dinner" prose rather than structure.
    used_prose: bool = False
    # Days numbered past MAX_DAYS, reported rather than silently scored.
    days_over_limit: list = field(default_factory=list)
    # Dishes read from structure (``slot:`` markers, bullets under a slot
    # heading) and their slots — not from "X for dinner" prose.
    structured_meals: int = 0
    structured_slots: set = field(default_factory=set)


def scan(text: str) -> ScanResult:
    """Read the structure a pasted plan already has. Never calls a model.

    Lines before the first heading or dish are preamble ("here is my week,
    how does it look?") and are not reported. After that, a line that is
    neither a heading, a dish, nor a question is kept in ``unparsed`` —
    verbatim, so the member can see exactly what was not read.
    """
    days: list[PastedDay] = []
    unparsed: list[str] = []
    current: Optional[PastedDay] = None
    open_slot: Optional[str] = None
    started = False
    used_prose = False
    over_limit: list[str] = []
    notes: list[str] = []
    structured = 0
    structured_slots: set[str] = set()

    def day_block() -> PastedDay:
        nonlocal current
        if current is None:
            current = PastedDay()
            days.append(current)
        return current

    for raw in (text or "").splitlines():
        line = _clean_line(raw)
        if not line:
            open_slot = None
            continue

        head = _DAY_HEAD.match(line)
        if head:
            if head.group("name"):
                number = _DAY_WORDS[head.group("name").lower().rstrip(".")]
                label = DAY_NAMES[number - 1]
            else:
                number = int(head.group("num"))
                label = f"Day {number}"
            current = PastedDay(day=number, label=label)
            days.append(current)
            open_slot = None
            started = True
            line = line[head.end():].strip()
            if not line:
                continue

        marks = list(_SLOT_MARK.finditer(line))
        if marks:
            lead = _clean_title(line[: marks[0].start()])
            if lead and started:
                unparsed.append(lead)
            elif lead:
                notes.append(lead)
            for index, mark in enumerate(marks):
                end = marks[index + 1].start() if index + 1 < len(marks) else len(line)
                slot = _SLOT_WORDS[mark.group("slot").lower()]
                meal = _meal_from_segment(slot, line[mark.end():end])
                if meal is not None:
                    day_block().meals.append(meal)
                    structured += 1
                    structured_slots.add(slot)
                    open_slot = None
                else:
                    open_slot = slot
            started = True
            continue

        only = _SLOT_ONLY.match(line)
        if only:
            open_slot = _SLOT_WORDS[only.group("slot").lower()]
            day_block()
            started = True
            continue

        phrases = list(_FOR_SLOT.finditer(line))
        if phrases:
            for phrase in phrases:
                dish = _PROSE_INTRO.sub("", phrase.group("dish").strip())
                dish = _DISH_LEAD.sub("", dish.strip())
                meal = _meal_from_segment(_SLOT_WORDS[phrase.group("slot").lower()], dish)
                if meal is not None:
                    day_block().meals.append(meal)
            used_prose = True
            started = True
            continue

        if open_slot is not None:
            meal = _meal_from_segment(open_slot, line)
            if meal is not None:
                day_block().meals.append(meal)
                structured += 1
                structured_slots.add(open_slot)
            continue

        if started and not line.endswith("?"):
            unparsed.append(raw.strip())
        else:
            notes.append(line)

    plan = _finalize(days, unparsed, [])
    plan.notes = notes
    over_limit = [
        d.label or f"Day {d.day}" for d in days if isinstance(d.day, int) and d.day > MAX_DAYS
    ]
    if over_limit:
        plan.warnings.append(_over_limit_warning(over_limit))
    return ScanResult(
        plan=plan, used_prose=used_prose, days_over_limit=over_limit,
        structured_meals=structured, structured_slots=structured_slots,
    )


def prepass(text: str) -> PastedPlan:
    return scan(text).plan


def _over_limit_warning(labels: list[str]) -> str:
    return (
        f"Only {MAX_DAYS} days are scored, so " + ", ".join(labels)
        + " was left out."
    )


def _finalize(days: list[PastedDay], unparsed: list[str], warnings: list[str]) -> PastedPlan:
    """Merge repeated day headings, number unlabelled days, set the plan type.

    A day heading listed twice ("Monday" at the top and again further down)
    is one day. Days past MAX_DAYS are dropped here and named by the caller.
    Days that ended up with no dishes are not days.
    """
    by_number: dict[int, PastedDay] = {}
    ordered: list[PastedDay] = []
    for day in days:
        if not day.meals:
            continue
        if isinstance(day.day, int):
            if day.day < 1 or day.day > MAX_DAYS:
                continue
            if day.day in by_number:
                by_number[day.day].meals.extend(day.meals)
                continue
            by_number[day.day] = day
        ordered.append(day)

    if len(ordered) > 1:
        used = {d.day for d in ordered if isinstance(d.day, int)}
        free = (n for n in range(1, MAX_DAYS + 1) if n not in used)
        for day in ordered:
            if day.day is None:
                number = next(free, None)
                if number is None:
                    break
                day.day = number
        ordered = [d for d in ordered if isinstance(d.day, int)]
        ordered.sort(key=lambda d: d.day)

    return PastedPlan(
        plan_type="weekly" if len(ordered) > 1 else "daily",
        days=ordered,
        unparsed=list(unparsed),
        warnings=list(warnings),
    )


def looks_like_plan_listing(text: str, structured_only: bool = True) -> bool:
    """At least two dishes in at least two different meal slots.

    ``structured_only`` counts only dishes the scanner read from structure —
    ``slot:`` prefixes and bullets under a slot heading. With it False the
    prose form counts too ("fried eggs for breakfast, pasta for lunch"), which
    is how people write a plan into a chat box.

    Prose alone is not enough to skip intent classification: "what do you think
    of adding salmon for dinner and oats for breakfast?" reads the same and is
    a request. The caller pairs it with a scoring word and the absence of a
    request verb (``OrchestratorService.looks_like_a_pasted_plan``).
    """
    result = scan(text)
    if structured_only:
        slots = {slot for slot in result.structured_slots if slot in MAIN_SLOTS}
        return result.structured_meals >= 2 and len(slots) >= 2
    meals = result.plan.meals
    slots = {meal.slot for meal in meals if meal.slot in MAIN_SLOTS}
    return len(meals) >= 2 and len(slots) >= 2


def describe_structure(plan: PastedPlan) -> str:
    """The scanner's reading, as the hint handed to the parser."""
    if plan.is_empty:
        return "(the line scanner found no meal structure)"
    lines = []
    for day in plan.days:
        heading = day.label or (f"Day {day.day}" if day.day else "One block, no day heading")
        lines.append(f"{heading}:")
        for meal in day.meals:
            extra = f" (ingredients: {meal.ingredients})" if meal.ingredients else ""
            lines.append(f"  {meal.slot}: {meal.title}{extra}")
    if plan.unparsed:
        lines.append("Lines the scanner could not place: " + " | ".join(plan.unparsed))
    return "\n".join(lines)


# --------------------------------------------------------------------- #
# Parser output → PastedPlan                                              #
# --------------------------------------------------------------------- #

def _text_tokens(text: str) -> set[str]:
    return set(content_tokens(text))


def _supported_by_text(phrase: str, text_tokens: set[str]) -> bool:
    """At least half of a phrase's meaningful words appear in the text."""
    tokens = content_tokens(phrase)
    if not tokens:
        return False
    present = sum(1 for t in tokens if t in text_tokens)
    return present * 2 >= len(tokens)


def _ingredients_written(ingredients: str, text_tokens: set[str]) -> bool:
    """Every listed item has a word the member actually wrote."""
    items = [i for i in re.split(r",|;|\band\b", ingredients or "") if i.strip()]
    if not items:
        return False
    return all(any(t in text_tokens for t in content_tokens(item)) for item in items)


def _normalize_slot(value) -> str:
    key = str(value or "").strip().lower()
    if key in SLOTS:
        return key
    return _SLOT_WORDS.get(key, "other")


def _day_from_label(label) -> Optional[int]:
    if not label:
        return None
    head = _DAY_HEAD.match(str(label).strip())
    if not head:
        return None
    if head.group("name"):
        return _DAY_WORDS[head.group("name").lower().rstrip(".")]
    return int(head.group("num"))


def plan_from_parser_payload(payload: dict, text: str) -> PastedPlan:
    """Normalise a ``ParsedPlanSchema`` payload and check it against the text."""
    text_tokens = _text_tokens(text)
    lowered = (text or "").lower()
    days: list[PastedDay] = []
    dropped_titles = 0
    dropped_ingredients = 0
    over_limit: list[str] = []

    for raw_day in payload.get("days") or []:
        if not isinstance(raw_day, dict):
            continue
        number = raw_day.get("day")
        number = int(number) if isinstance(number, int) else _day_from_label(raw_day.get("label"))
        label = raw_day.get("label") or None
        if isinstance(number, int) and number > MAX_DAYS:
            over_limit.append(str(label or f"Day {number}"))
            continue
        day = PastedDay(day=number, label=label)
        for raw_meal in raw_day.get("meals") or []:
            if not isinstance(raw_meal, dict):
                continue
            title = _clean_title(str(raw_meal.get("title") or ""))
            if not title:
                continue
            if not _supported_by_text(title, text_tokens):
                dropped_titles += 1
                continue
            ingredients = raw_meal.get("ingredients")
            ingredients = _clean_title(str(ingredients)) if ingredients else None
            if ingredients and not _ingredients_written(ingredients, text_tokens):
                dropped_ingredients += 1
                ingredients = None
            quantity = raw_meal.get("quantity_note") or None
            day.meals.append(PastedMeal(
                slot=_normalize_slot(raw_meal.get("slot")),
                title=title,
                ingredients=ingredients,
                quantity_note=str(quantity) if quantity else None,
            ))
        days.append(day)

    unparsed = [
        str(line).strip() for line in payload.get("unparsed") or []
        if str(line).strip() and str(line).strip().lower() in lowered
    ]
    warnings: list[str] = []
    if dropped_titles:
        warnings.append(
            f"{dropped_titles} dish(es) the reader suggested are not in your text, "
            "so they were left out."
        )
    if dropped_ingredients:
        warnings.append(
            f"Ingredient lists for {dropped_ingredients} dish(es) were not in your "
            "text, so those dishes are scored without them."
        )
    plan = _finalize(days, unparsed, warnings)
    if over_limit:
        plan.warnings.append(_over_limit_warning(over_limit))
    return plan


def parse_plan_text(text: str, parser=None, plan_type: str = "auto") -> PastedPlan:
    """Step 1: the member's text as a ``PastedPlan``.

    The scanner's reading is used on its own only when it read the text from
    structure alone and placed every line — most pasted plans are shaped like
    that, and the common case then costs no model call. Otherwise the parser
    reads the text with the scanner's reading as a hint. If the parser fails,
    or finds nothing where the scanner found dishes, the scanner's reading
    stands and a warning says the text was only partly understood.

    ``plan_type`` is ``"auto"`` on the chat path. An explicit ``"daily"`` or
    ``"weekly"`` settles a shape question instead of asking it.
    """
    result = scan(text)
    hint = result.plan
    plan = hint
    if hint.is_empty or hint.unparsed or result.used_prose:
        payload = parser.parse(text, describe_structure(hint)) if parser is not None else None
        parsed = plan_from_parser_payload(payload, text) if payload else None
        if parsed is not None and not parsed.is_empty:
            plan = parsed
        elif not hint.is_empty and (hint.unparsed or result.used_prose):
            plan.warnings.append(
                "Part of your text could only be read line by line, so some "
                "dishes may be missing."
            )

    if plan is not hint:
        plan.notes = list(hint.notes)
    if plan_type in PLAN_TYPES:
        plan = _apply_explicit_type(plan, plan_type)
    return plan


def _apply_explicit_type(plan: PastedPlan, plan_type: str) -> PastedPlan:
    if plan_type == "weekly" and len(plan.days) == 1 and needs_shape_question(plan):
        return split_into_days(plan, 0)
    if plan_type == "daily" and len(plan.days) > 1:
        plan.warnings.append(
            f"Your text lists {len(plan.days)} days, so it was read as "
            f"{len(plan.days)} days rather than one."
        )
    return plan


# --------------------------------------------------------------------- #
# Shape clarification                                                     #
# --------------------------------------------------------------------- #

def needs_shape_question(plan: PastedPlan) -> bool:
    """One block with no day heading, in which two or more meals repeat.

    "dinner: pasta / dinner: salad" is one dinner served as two plates. Two
    breakfasts AND two lunches with no heading is a question only the member
    can answer — one long day, or several short ones.
    """
    if len(plan.days) != 1:
        return False
    day = plan.days[0]
    if day.day is not None or day.label:
        return False
    counts = {slot: 0 for slot in MAIN_SLOTS}
    for meal in day.meals:
        if meal.slot in counts:
            counts[meal.slot] += 1
    return sum(1 for n in counts.values() if n >= 2) >= 2


def days_from_answer(text: str) -> Optional[int]:
    """How many days the member says the text covers.

    ``1`` for one day, ``n`` for a stated count, ``0`` for "several days"
    without a count (split where a meal repeats), ``None`` when the reply
    does not answer the question at all.
    """
    t = (text or "").lower()
    if re.search(r"\b(?:whole|full|entire|this|a|one)\s+week\b|\bweekly\b|\b(?:7|seven)\s+days?\b", t):
        return MAX_DAYS
    count = re.search(r"\b(\d{1,2})\s*days?\b", t)
    if count:
        return max(1, min(int(count.group(1)), MAX_DAYS))
    for word, number in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\s+days?\b", t):
            return number
    if re.search(r"\b(?:one|1|single|a)\s+day\b|\btoday\b|\bjust\s+one\b|\bdaily\b", t):
        return 1
    if re.search(r"\b(?:several|multiple|different|separate|many)\s+days?\b|\bmore than one day\b", t):
        return 0
    return None


def split_into_days(plan: PastedPlan, days: int) -> PastedPlan:
    """Regroup a single block into days, starting a new day where a meal repeats.

    ``days == 1`` keeps one day. Otherwise the text's own order decides: the
    second "breakfast" starts the second day. When that gives a different
    count than the member stated, the split stands and a warning names both.
    """
    meals = plan.meals
    warnings = list(plan.warnings)
    if days == 1:
        return PastedPlan(
            plan_type="daily",
            days=[PastedDay(day=None, label=None, meals=list(meals))],
            unparsed=list(plan.unparsed),
            warnings=warnings,
            notes=list(plan.notes),
        )

    groups: list[list[PastedMeal]] = [[]]
    for meal in meals:
        seen = {m.slot for m in groups[-1] if m.slot in MAIN_SLOTS}
        if meal.slot in MAIN_SLOTS and meal.slot in seen:
            groups.append([])
        groups[-1].append(meal)
    groups = [g for g in groups if g]
    if len(groups) > MAX_DAYS:
        warnings.append(_over_limit_warning([f"Day {n}" for n in range(MAX_DAYS + 1, len(groups) + 1)]))
        groups = groups[:MAX_DAYS]
    if days and days != len(groups):
        warnings.append(
            f"You said {days} day(s); the meals split into {len(groups)} "
            "where a meal repeats, so that is how they were read."
        )
    return PastedPlan(
        plan_type="weekly" if len(groups) > 1 else "daily",
        days=[
            PastedDay(day=index + 1, label=f"Day {index + 1}", meals=group)
            for index, group in enumerate(groups)
        ],
        unparsed=list(plan.unparsed),
        warnings=warnings,
        notes=list(plan.notes),
    )
