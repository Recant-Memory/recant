"""The words-only check (`recant.notes.parts` and `holds`), 0.4.0: single letters and digits count, as whole words;
only a lowercase "a" or "i" as written is left out. The memory eval's scorer and the notes evidence rule both use it, so
it can only get stricter: every value the new rule finds in a text, the old rule found too."""
import itertools
import re

import pytest

from recant import eval_memory as em
from recant import notes


def old_parts(value):                                    # the rule before 0.4.0, kept here to compare against
    out = [notes._plain(m) for m in re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-/%@:]*", str(value or ""))]
    return [x for x in out if len(x) >= 2] or ([notes._plain(str(value))] if notes._plain(str(value)) else [])


def old_holds(text, value):
    t = notes._plain(text)
    ps = old_parts(value)
    return bool(ps) and all(x in t for x in ps)


@pytest.mark.parametrize("key,reply,right", [
    ("A minus", "ANSWER: A minus", True),
    ("A minus", "ANSWER: B minus", False),             # was scored right before 0.4.0
    ("B minus", "ANSWER: A minus", False),
    ("plan B", "ANSWER: plan B", True),
    ("plan B", "ANSWER: plan C, but not plan A", False),   # "b" is not found inside "but"
    ("5", "ANSWER: 5", True),
    ("5", "ANSWER: 3", False),
    ("5", "ANSWER: 15", False),                        # nor "5" inside "15"
    ("a dog", "ANSWER: the dog", True),                # a lowercase "a" is an article: left out
    ("Room 4", "ANSWER: room 4", True),
    ("Room 4", "ANSWER: room 14", False),
])
def test_single_letters_and_digits_count_as_whole_words(key, reply, right):
    assert em.correct({"kind": "recall", "answer": key}, reply, strict=True) is right


def test_an_uppercase_single_letter_always_counts():
    assert notes.parts("A minus") == ["a", "minus"] and notes.parts("I") == ["i"]
    assert notes.parts("a dog") == ["dog"] and notes.parts("i think") == ["think"]


VALUES = ["A minus", "B minus", "plan B", "5", "3", "15", "a dog", "Room 4", "EX0002, 2000.00", "I", "x", "Q3 2026",
          "the Hague", "v2", "Dr Sample", "3,300 a month", "a 5", "X Y", "A I", "Y X", "a", "i", "B", "5 3"]
TEXTS = ["My grade is A minus.", "plan B, but not plan A", "room 14 and room 4", "we pay 3,300 a month", "a dog barked",
         "Q3 2026 sales", "the Hague in 2026", "EX0002 costs 2000.00", "I said 5, not 15", "nothing here", "v2 shipped",
         "Dr Sample, Thursdays", "5", "Y X", "I A", "X Y", "3 5", "5 3", "a", "B"]          # the single-letter cases a
                                                                                         # review found looser before


def test_the_rule_is_only_stricter_than_before_so_the_notes_evidence_check_cannot_get_looser():
    looser = [(v, t) for v, t in itertools.product(VALUES, TEXTS) if notes.holds(t, v) and not old_holds(t, v)]
    assert looser == []
    stricter = [(v, t) for v, t in itertools.product(VALUES, TEXTS) if old_holds(t, v) and not notes.holds(t, v)]
    assert ("B minus", "My grade is A minus.") in stricter                  # old accepted it; it no longer does
