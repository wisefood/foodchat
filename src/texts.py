"""The fixed sentences FoodChat says to the reader, in Hungarian, Slovenian and Greek.

FoodChat's own code writes some replies without a model: the result of a
swap, a memory offer, a busy notice. Those were English in every language.
The English stays where it was written, in each service, byte for byte (the
tests read it, and every evaluation ran against it); a service asks this
module only when :func:`reply_language.current` is not English.

Meal and day words are the interface's own (wisefood-ui app/i18n/locales,
``foodChatHome.meals``, ``dashboard.schedule.days``, ``foodChatHome.ledger.terms``),
so a reply names a meal the way the card beside it does. Recipe titles, and
ingredient names from the recipes, are catalogue data and stay as stored.

A slot is named "meal (day)" rather than "Wednesday's dinner": the three
languages decline nouns, and a name in brackets stays in the nominative in
every sentence it is dropped into.

Like the interface's Hungarian and Slovenian text, these need a native
speaker's read.
"""

from __future__ import annotations

from typing import Optional

import reply_language

_QUOTES = {"en": ("“", "”"), "hu": ("„", "”"), "sl": ("»", "«"), "el": ("«", "»")}

_MEALS = {
    "hu": {"breakfast": "reggeli", "lunch": "ebéd", "dinner": "vacsora", "brunch": "brunch",
           "snack": "uzsonna", "dessert": "desszert", "side": "köret", "drink": "ital"},
    "sl": {"breakfast": "zajtrk", "lunch": "kosilo", "dinner": "večerja", "brunch": "brunch",
           "snack": "prigrizek", "dessert": "sladica", "side": "priloga", "drink": "pijača"},
    "el": {"breakfast": "πρωινό", "lunch": "μεσημεριανό", "dinner": "βραδινό", "brunch": "brunch",
           "snack": "σνακ", "dessert": "επιδόρπιο", "side": "συνοδευτικό", "drink": "ρόφημα"},
}

_DAYS = {
    "hu": ["hétfő", "kedd", "szerda", "csütörtök", "péntek", "szombat", "vasárnap"],
    "sl": ["ponedeljek", "torek", "sreda", "četrtek", "petek", "sobota", "nedelja"],
    "el": ["Δευτέρα", "Τρίτη", "Τετάρτη", "Πέμπτη", "Παρασκευή", "Σάββατο", "Κυριακή"],
}

_DAY_N = {"hu": "{n}. nap", "sl": "{n}. dan", "el": "ημέρα {n}"}

_TERMS = {
    "hu": {"omnivore": "mindenevő", "vegetarian": "vegetáriánus", "vegan": "vegán",
           "pescatarian": "pescetáriánus", "flexitarian": "flexitáriánus",
           "mediterranean": "mediterrán", "gluten_free": "gluténmentes",
           "lactose_free": "laktózmentes", "dairy_free": "tejmentes", "high_protein": "fehérjedús",
           "low_fat": "zsírszegény", "high_fibre": "rostban gazdag", "low_calorie": "alacsony kalóriatartalmú",
           "low_sugar": "cukorszegény", "low_salt": "sószegény", "keto": "keto", "halal": "halal",
           "kosher": "kóser", "lighter": "könnyebb"},
    "sl": {"omnivore": "vsejed", "vegetarian": "vegetarijansko", "vegan": "vegansko",
           "pescatarian": "peskatarijansko", "flexitarian": "fleksitarijansko",
           "mediterranean": "sredozemsko", "gluten_free": "brez glutena",
           "lactose_free": "brez laktoze", "dairy_free": "brez mlečnih izdelkov",
           "high_protein": "veliko beljakovin", "low_fat": "malo maščob", "high_fibre": "veliko vlaknin",
           "low_calorie": "malo kalorij", "low_sugar": "malo sladkorja", "low_salt": "malo soli",
           "keto": "keto", "halal": "halal", "kosher": "košer", "lighter": "lažje"},
    "el": {"omnivore": "παμφάγο", "vegetarian": "χορτοφαγικό", "vegan": "vegan",
           "pescatarian": "πεσκεταριανό", "flexitarian": "ευέλικτο χορτοφαγικό",
           "mediterranean": "μεσογειακό", "gluten_free": "χωρίς γλουτένη",
           "lactose_free": "χωρίς λακτόζη", "dairy_free": "χωρίς γαλακτοκομικά",
           "high_protein": "πλούσιο σε πρωτεΐνη", "low_fat": "χαμηλά λιπαρά",
           "high_fibre": "πλούσιο σε φυτικές ίνες", "low_calorie": "λίγες θερμίδες",
           "low_sugar": "λίγη ζάχαρη", "low_salt": "λίγο αλάτι", "keto": "κέτο",
           "halal": "χαλάλ", "kosher": "κοσέρ", "lighter": "πιο ελαφρύ"},
}

# The sentences, by key. `{q:...}` is never used: titles are quoted by `_q`
# before formatting, so a title with braces in it cannot break a template.
_TEXT = {
    "edit.done": {
        "hu": "Kész — kicseréltem: {where}, {old} → {new}.",
        "sl": "Končano — zamenjano: {where}, {old} → {new}.",
        "el": "Έγινε — άλλαξα: {where}, {old} → {new}.",
    },
    "edit.lighter": {
        "hu": " Így adagonként {old} kcal helyett {new} kcal.",
        "sl": " Tako se z {old} spusti na {new} kcal na porcijo.",
        "el": " Έτσι πέφτει από {old} σε {new} kcal ανά μερίδα.",
    },
    "edit.uses": {
        "hu": " Ellenőrizve: felhasználja, ami otthon van ({tag}) — kevesebb élelmiszer-pazarlás.",
        "sl": " Preverjeno: porabi, kar imaš doma ({tag}) — manj zavržene hrane.",
        "el": " Επαληθεύτηκε: χρησιμοποιεί ό,τι έχεις ήδη ({tag}) — λιγότερη σπατάλη φαγητού.",
    },
    "edit.verified": {
        "hu": " Ellenőrizve: az új választás megfelel ennek: {directive}.",
        "sl": " Preverjeno: nova izbira ustreza zahtevi {directive}.",
        "el": " Επαληθεύτηκε: η νέα επιλογή πληροί το {directive}.",
    },
    "edit.named_miss": {
        "hu": " Nem találtam ilyen receptet: {dish}, ezért egy illő alternatívát választottam — "
              "ha más ételre gondoltál, nevezd meg.",
        "sl": " Recepta {dish} ni v naši zbirki, zato je izbrana ustrezna alternativa — "
              "če misliš na drugo jed, jo poimenuj.",
        "el": " Δεν βρήκα το {dish} στις συνταγές μας, οπότε διάλεξα μια κατάλληλη εναλλακτική — "
              "πες μου αν είχες άλλο πιάτο στο μυαλό σου.",
    },
    "edit.best_match": {
        "hu": " A legjobb egyezést választottam erre: {directive} — szólj, ha nem egészen az.",
        "sl": " Izbrano je najboljše ujemanje za {directive} — povej, če ni čisto pravo.",
        "el": " Διάλεξα ό,τι ταιριάζει καλύτερα στο {directive} — πες μου αν δεν είναι ακριβώς αυτό.",
    },
    "edit.failed": {
        "hu": "Kerestem cserét ({where}), ami biztosan {directive}, de egyik illő recept sem felel meg",
        "sl": "Zamenjave ({where}), ki bi zanesljivo ustrezala zahtevi {directive}, ni: "
              "noben ustrezen recept ne zadosti merilu",
        "el": "Έψαξα αντικατάσταση ({where}) που να είναι σίγουρα {directive}, αλλά καμία από "
              "τις συνταγές που ταιριάζουν δεν περνά τον πήχη",
    },
    "edit.failed_nearest": {
        "hu": " — a legközelebbi {title}, adagonként {kcal} kcal. Ezt kéred, vagy lazítsak valami máson?",
        "sl": " — najbližji je {title} s {kcal} kcal na porcijo. Želiš tega ali naj omilim kaj drugega?",
        "el": " — η πιο κοντινή είναι η {title} με {kcal} kcal ανά μερίδα. Τη θέλεις ή να χαλαρώσω κάτι άλλο;",
    },
    "edit.failed_relax": {
        "hu": ". Lazítsak ezen a feltételen vagy egy másik megkötésen?",
        "sl": ". Naj omilim zahtevo ali kakšno drugo omejitev?",
        "el": ". Να χαλαρώσω την απαίτηση ή κάποιον άλλο περιορισμό;",
    },
    "edit.no_canvas": {
        "hu": "Még nincs terv — előbb kérj napi vagy heti tervet, utána tudok benne étkezéseket cserélni.",
        "sl": "Načrta še ni — najprej prosi za dnevni ali tedenski načrt, potem lahko v njem zamenjam obroke.",
        "el": "Δεν υπάρχει ακόμα πλάνο — ζήτησέ μου πρώτα ημερήσιο ή εβδομαδιαίο πλάνο και μετά μπορώ να αλλάξω γεύματα σε αυτό.",
    },
    "edit.which_meal": {
        "hu": "Melyik étkezést cseréljem — a reggelit, az ebédet vagy a vacsorát?",
        "sl": "Kateri obrok naj zamenjam — zajtrk, kosilo ali večerjo?",
        "el": "Ποιο γεύμα να αλλάξω — πρωινό, μεσημεριανό ή βραδινό;",
    },
    "edit.which_meal_day": {
        "hu": "Melyik étkezést cseréljem (reggeli, ebéd vagy vacsora), és melyik napon?",
        "sl": "Kateri obrok naj zamenjam (zajtrk, kosilo ali večerja) in na kateri dan?",
        "el": "Ποιο γεύμα να αλλάξω (πρωινό, μεσημεριανό ή βραδινό) και ποια μέρα;",
    },
    "edit.no_daily": {
        "hu": "Nem találom a mostani napi tervet, amit módosíthatnék.",
        "sl": "Trenutnega dnevnega načrta za urejanje ni mogoče najti.",
        "el": "Δεν βρίσκω το τρέχον ημερήσιο πλάνο για να το αλλάξω.",
    },
    "edit.no_weekly": {
        "hu": "Nem találom a mostani heti tervet, amit módosíthatnék.",
        "sl": "Trenutnega tedenskega načrta za urejanje ni mogoče najti.",
        "el": "Δεν βρίσκω το τρέχον εβδομαδιαίο πλάνο για να το αλλάξω.",
    },
    "edit.no_day": {
        "hu": "Ez a terv ezeket a napokat fedi le: {span} — {day} nincs benne. Mondd meg, melyikre gondoltál, és kicserélem.",
        "sl": "Ta načrt zajema: {span}; {day} ni v njem. Povej, katerega misliš, in ga zamenjam.",
        "el": "Αυτό το πλάνο καλύπτει: {span} — η {day} δεν υπάρχει. Πες μου ποια εννοείς και θα την αλλάξω.",
    },
    "edit.no_meal": {
        "hu": "Ebben a tervben nincs {meal} — ez van benne: {have}. Kérd, hogy tervezzek bele egyet, "
              "vagy mondd meg, melyiket változtassam meg.",
        "sl": "V tem načrtu ni obroka {meal} — ima: {have}. Prosi, naj ga dodam, "
              "ali povej, katerega od teh naj spremenim.",
        "el": "Αυτό το πλάνο δεν έχει {meal} — έχει: {have}. Ζήτησέ μου να το προσθέσω "
              "ή πες μου ποιο από αυτά να αλλάξω.",
    },
    "edit.no_meals": {"hu": "semmit", "sl": "nič", "el": "τίποτα"},
    "edit.no_slot": {
        "hu": "Nem találom ezt az étkezést a mostani tervben: {slot}.",
        "sl": "V trenutnem načrtu ni tega obroka: {slot}.",
        "el": "Δεν βρίσκω αυτό το γεύμα στο τρέχον πλάνο: {slot}.",
    },
    "memory.matters": {
        "hu": "Úgy tűnik, ez fontos neked: {value} — megjegyezzem?",
        "sl": "Zdi se, da ti je to pomembno: {value} — si to zapomnim?",
        "el": "Φαίνεται ότι αυτό έχει σημασία για σένα: {value} — να το θυμάμαι;",
    },
    "memory.gone_off": {
        "hu": "{value} most a kedvenceid között van, de úgy hangzik, már nem szereted — frissítsem a profilodat?",
        "sl": "{value} je zdaj med tvojimi priljubljenimi, a zveni, kot da ti ni več všeč — posodobim profil?",
        "el": "Το {value} είναι στα αγαπημένα σου, αλλά ακούγεται ότι δεν σου αρέσει πια — να ενημερώσω το προφίλ σου;",
    },
    "memory.come_round": {
        "hu": "{value} most azok között van, amiket nem szeretsz, de úgy hangzik, most már ízlik — frissítsem a profilodat?",
        "sl": "{value} je zdaj med tem, česar ne maraš, a zveni, kot da ti je zdaj všeč — posodobim profil?",
        "el": "Το {value} είναι σε όσα δεν σου αρέσουν, αλλά ακούγεται ότι τώρα σου αρέσει — να ενημερώσω το προφίλ σου;",
    },
    "memory.diet": {
        "hu": "Ezt kérted: {diet}. Megjegyezzem, hogy minden terv ebből induljon ki?",
        "sl": "Prosiš za: {diet}. Si to zapomnim, da se vsak načrt začne s tem?",
        "el": "Ζήτησες: {diet}. Να το θυμάμαι, ώστε κάθε πλάνο να ξεκινά από εκεί;",
    },
    "turn.limit": {
        "hu": "Ez a beszélgetés elérte a(z) {limit} üzenetes korlátot. A folytatáshoz kezdj új beszélgetést.",
        "sl": "Ta pogovor je dosegel omejitev {limit} sporočil. Za nadaljevanje začni nov pogovor.",
        "el": "Αυτή η συζήτηση έφτασε το όριο των {limit} μηνυμάτων. Ξεκίνα νέα συζήτηση για να συνεχίσεις.",
    },
    "turn.busy": {
        "hu": "Még az előző üzeneteden dolgozom — várj egy kicsit, aztán próbáld újra.",
        "sl": "Še delam na tvojem prejšnjem sporočilu — počakaj trenutek in poskusi znova.",
        "el": "Δουλεύω ακόμα το προηγούμενο μήνυμά σου — περίμενε λίγο και δοκίμασε ξανά.",
    },
    "favorites.offer": {
        "hu": "Mielőtt tervezek: láttam, hogy ezeket kedvencnek jelölted: {named}. "
              "Beletegyem őket ebbe a tervbe? (igen / nem)",
        "sl": "Preden začnem: med tvojimi priljubljenimi so {named}. "
              "Naj jih vključim v ta načrt? (da / ne)",
        "el": "Πριν φτιάξω το πλάνο: είδα ότι έχεις στα αγαπημένα {named}. "
              "Να τα βάλω σε αυτό το πλάνο; (ναι / όχι)",
    },
    "analyst.failed": {
        "hu": "Most nem sikerült elemeznem a tervet — kérdezz újra, vagy mondd meg, mit változtassak, és onnan folytatom.",
        "sl": "Načrta trenutno ni bilo mogoče analizirati — vprašaj znova ali povej, kaj naj spremenim, in nadaljujem od tam.",
        "el": "Δεν μπόρεσα να αναλύσω το πλάνο αυτή τη στιγμή — ρώτα ξανά ή πες μου τι να αλλάξω και συνεχίζω από εκεί.",
    },
    "pantry.noted": {
        "hu": "Rendben — a terveket erre építem: {items}, hogy kevesebb étel menjen kárba. "
              "Kérj napi vagy heti tervet, és beledolgozom őket.",
        "sl": "Zabeleženo — načrti bodo zgrajeni okoli tega: {items}, da bo manj zavržene hrane. "
              "Prosi za dnevni ali tedenski načrt in jih vključim.",
        "el": "Σημειώθηκε — θα σχεδιάσω γύρω από αυτά: {items}, για λιγότερη σπατάλη φαγητού. "
              "Ζήτησέ μου ημερήσιο ή εβδομαδιαίο πλάνο και θα τα βάλω μέσα.",
    },
    "pantry.cleared": {
        "hu": "Rendben — ezeket levettem a felhasználandók listájáról.",
        "sl": "Zabeleženo — to je odstranjeno s seznama za porabo.",
        "el": "Σημειώθηκε — τα αφαίρεσα από τη λίστα με όσα θέλεις να χρησιμοποιήσεις.",
    },
    "switch.weekly": {
        "hu": "Rendben! Félreteszem a napi tervet, és új heti tervet kezdek neked.",
        "sl": "Seveda! Dnevni načrt dam na stran in začnem nov tedenski načrt.",
        "el": "Βεβαίως! Αφήνω στην άκρη το ημερήσιο πλάνο και ξεκινώ ένα νέο εβδομαδιαίο.",
    },
    "switch.daily": {
        "hu": "Rendben! Félreteszem a heti tervet, és új napi tervet kezdek neked.",
        "sl": "Seveda! Tedenski načrt dam na stran in začnem nov dnevni načrt.",
        "el": "Βεβαίως! Αφήνω στην άκρη το εβδομαδιαίο πλάνο και ξεκινώ ένα νέο ημερήσιο.",
    },
}


def active() -> bool:
    """Whether the reader's language is one this module speaks (not English)."""
    return reply_language.current() in _QUOTES and reply_language.current() != reply_language.DEFAULT


def _lang(lang: Optional[str]) -> str:
    return reply_language.normalize(lang) if lang else reply_language.current()


def quote(text: object, lang: Optional[str] = None) -> str:
    left, right = _QUOTES[_lang(lang)]
    return f"{left}{text}{right}"


def meal(meal_type: object, lang: Optional[str] = None) -> str:
    key = str(meal_type or "").strip().lower()
    return _MEALS.get(_lang(lang), {}).get(key, key or "")


def day(n: object, *, weekdays: bool = True, lang: Optional[str] = None) -> str:
    language = _lang(lang)
    try:
        d = int(n)
    except (TypeError, ValueError):
        return str(n)
    if weekdays and 1 <= d <= 7 and language in _DAYS:
        return _DAYS[language][d - 1]
    return _DAY_N.get(language, "day {n}").format(n=d)


def term(value: object, lang: Optional[str] = None) -> str:
    """A diet or goal code in words: `gluten_free` → "gluténmentes"."""
    key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _TERMS.get(_lang(lang), {}).get(key, str(value or "").replace("_", " "))


def slot(meal_type: object, n: object = None, *, weekdays: bool = True,
         lang: Optional[str] = None) -> str:
    """A slot in the nominative: "vacsora", "vacsora (szerda)", "večerja (2. dan)"."""
    name = meal(meal_type, lang)
    return f"{name} ({day(n, weekdays=weekdays, lang=lang)})" if n else name


def say(key: str, lang: Optional[str] = None, **params) -> str:
    """The sentence `key` in the reader's language. Raises KeyError for English:
    the English is the service's own and is never looked up here."""
    return _TEXT[key][_lang(lang)].format(**params)
