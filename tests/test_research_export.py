"""What leaves FoodChat when conversations are used to test WiseFood.

Only people who agreed are exported — the gateway decides that. These pin what
their conversations look like once they leave: names the profile knows are
gone, identifiers with a shape are gone, ids are codes that mean nothing
outside one export, times are the month and seconds since the start, and a
message that could not be checked is dropped rather than passed through.
"""

from datetime import datetime, timedelta, timezone

import pii
import research_export as rx

T0 = datetime(2026, 10, 3, 18, 30, tzinfo=timezone.utc)
SALT = b"s" * 32

PROFILE = {
    "name": "Maria Papadopoulou",
    "cooking_for_names": ["Nikos", "Eleni"],
    "diet": "Mediterranean",
    "allergies": [{"id": "peanuts", "label": "Peanuts"}, "Gluten"],
    "dietary_goals": ["lower salt"],
    "age_group": "adult",
    "food_likes": ["feta", "Feta", "okra"],
    "memory_log": [{"said": "I am Maria and I hate okra"}],
    "notes": "call me on 6912345678",
}


def message(i, role, text, seconds, **extra):
    return {"id": i, "role": role, "content": text, "timestamp": T0 + timedelta(seconds=seconds), **extra}


def record(**overrides):
    args = dict(
        conversation_id="sess-123",
        member_id="member-9",
        profile=PROFILE,
        created_at=T0,
        messages=[
            message(1, "user", "Hi, I'm Maria. A dinner for me and Nikos please, no peanuts.", 0),
            message(2, "assistant", "Here is a plan for Maria and Nikos.", 9,
                    intent="daily_plan", plan_id="p1"),
            message(3, "user", "Swap the fish. Email me at maria@example.com", 40),
            message(4, "assistant", "Done.", 47, intent="refine_plan", plan_id="p2"),
            message(5, "system", "internal note", 48),
        ],
        plans={"p1": {"plan_type": "daily", "version": 1, "saved": 0},
               "p2": {"plan_type": "daily", "version": 2, "saved": 1}},
        ratings={4: "up"},
        salt=SALT,
    )
    args.update(overrides)
    return rx.conversation_record(**args)


class TestNames:
    def test_the_profile_knows_whole_names_and_their_parts(self):
        names = rx.known_names(PROFILE)
        assert names[0] == "Maria Papadopoulou"
        assert {"Maria", "Papadopoulou", "Nikos", "Eleni"} <= set(names)

    def test_names_are_replaced_in_both_directions_of_the_conversation(self):
        texts = [turn["text"] for turn in record()["turns"]]
        assert all("Maria" not in t and "Nikos" not in t for t in texts)
        assert texts[0] == "Hi, I'm [NAME]. A dinner for me and [NAME] please, no peanuts."
        assert texts[1] == "Here is a plan for [NAME] and [NAME]."

    def test_a_name_inside_a_word_is_left_alone(self):
        text, _ = rx.scrub("Elenis and Eleni", ["Eleni"])
        assert text == "Elenis and [NAME]"


class TestIdentifiersWithAShape:
    def test_they_are_removed_even_where_redaction_is_off_for_chat(self, monkeypatch):
        """The deployment switch governs what a model is told mid-turn; text
        leaving for a test set is checked whatever it says."""
        monkeypatch.setattr(pii, "ENABLED", False)
        turns = record()["turns"]
        assert turns[2]["text"] == "Swap the fish. Email me at [EMAIL]"

    def test_text_that_could_not_be_checked_does_not_leave(self, monkeypatch):
        monkeypatch.setattr(pii, "MAX_SCAN_CHARS", 10)
        text, counts = rx.scrub("a long message with a phone 6912345678", [])
        assert text == rx.UNCHECKED
        assert counts.get("unscanned") == 1


class TestIds:
    def test_ids_become_codes_for_one_export(self):
        first = record()
        again = record()
        other = record(salt=b"t" * 32)
        assert first["conversation"] == again["conversation"]
        assert first["conversation"] != other["conversation"]
        assert "sess-123" not in first["conversation"]
        assert "member-9" not in first["member"]

    def test_no_raw_id_or_timestamp_survives_anywhere(self):
        flat = repr(record())
        for raw in ("sess-123", "member-9", "p1", "p2", "18:30", "2026-10-03"):
            assert raw not in flat, raw


class TestShape:
    def test_turns_keep_order_role_intent_plan_and_rating(self):
        turns = record()["turns"]
        assert [t["role"] for t in turns] == ["user", "assistant", "user", "assistant"]
        assert [t["seconds"] for t in turns] == [0, 9, 40, 47]
        assert turns[1]["intent"] == "daily_plan"
        assert turns[3]["plan"] == {"type": "daily", "version": 2, "saved": True}
        assert turns[3]["rating"] == "up"
        assert "rating" not in turns[1]

    def test_times_are_the_month(self):
        assert record()["month"] == "2026-10"

    def test_replies_can_be_left_out(self):
        turns = record(with_replies=False)["turns"]
        assert [t["role"] for t in turns] == ["user", "user"]
        assert [t["turn"] for t in turns] == [0, 1]

    def test_redactions_are_counted_without_keeping_what_was_removed(self):
        counts = record()["redactions"]
        assert counts["NAME"] >= 4
        assert counts["EMAIL"] == 1


class TestProfile:
    def test_only_categories_are_kept(self):
        profile = record()["profile"]
        assert profile == {
            "diet": "mediterranean",
            "allergies": ["gluten", "peanuts"],
            "dietary_goals": ["lower salt"],
            "age_group": "adult",
            "food_likes": ["feta", "okra"],
        }

    def test_a_profile_stored_as_text_is_read_too(self):
        import json

        assert record(profile=json.dumps(PROFILE))["profile"]["diet"] == "mediterranean"


class TestWorthKeeping:
    def test_a_conversation_needs_the_members_own_words(self):
        assert rx.worth_keeping(record())
        silent = record(messages=[message(1, "assistant", "Hello!", 0)])
        assert not rx.worth_keeping(silent)
        assert not rx.worth_keeping(record(), min_user_turns=3)


class TestOrder:
    def test_an_email_made_of_a_name_goes_whole(self):
        """Names first left "[NAME]@example.com" — and the domain with it."""
        text, counts = rx.scrub("write to maria@example.com", ["Maria"])
        assert text == "write to [EMAIL]"
        assert counts == {"EMAIL": 1}
