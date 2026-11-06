"""`recant eval memory` offline. Two scopes' sessions; questions drafted, the evidence rule dropping a bad one;
the questions hashed; four arms built and scored mechanically; each arm against carried history; the notes arm's notes
checked by the library's evidence rule; memory v2 extracted, accepted and served; costs with the write path."""
import json
import re
from types import SimpleNamespace

import pytest

from recant import eval_memory as em
from recant.provider import Completion, Usage, text

SESSIONS = [
    {"scope": "user:ana", "ts": "2026-10-01T09:00:00+00:00", "messages": [
        {"role": "user", "content": "Our accountant is Alice Example. The shop rent is 3,300 a month."},
        {"role": "assistant", "content": "Noted."}]},
    {"scope": "user:ana", "ts": "2026-10-08T09:00:00+00:00", "messages": [
        {"role": "user", "content": "The landlord raised it: the shop rent is 3,800 a month from November."},
        {"role": "assistant", "content": [{"type": "text", "text": "Understood."}]}]},
    {"scope": "user:bo", "ts": "2026-10-02T09:00:00+00:00", "messages": [
        {"role": "user", "content": "My dentist is Dr Sample, on Thursdays."}, {"role": "assistant", "content": "Got it."}]},
    {"scope": "user:bo", "ts": "2026-10-09T09:00:00+00:00", "messages": [{"role": "user", "content": "Just checking in."}]},
]
FACTS = [("Example", "Alice Example", "accountant"), ("3,800", "3,800", "rent"), ("Sample", "Dr Sample", "dentist")]


def msg(content, usage=(300, 40)):
    return SimpleNamespace(content=content, usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1], cache_read_input_tokens=0,
                                                                 cache_creation_input_tokens=0), stop_reason="end_turn", model="claude-sonnet-5-5")


def t(s):
    return SimpleNamespace(type="text", text=s)


class FakeSDK:
    """Answers by what the system prompt asks for: questions, the note keeper's tool calls, answers from the context."""

    def __init__(self):
        self.messages, self.calls = self, []

    def create(self, *, system, messages, **kw):
        self.calls.append(system[:20])
        user = messages[0]["content"] if isinstance(messages[0]["content"], str) else ""
        if system.startswith(em.QUESTION_SYSTEM[:30]):
            if "Example" in user:
                qs = [{"kind": "recall", "question": "Who is our accountant?", "answer": "Alice Example",
                       "quote": "Our accountant is Alice Example.", "session": 1},
                      {"kind": "update", "question": "What is the shop rent now?", "answer": "3,800",
                       "quote": "the shop rent is 3,800 a month from November", "session": 2},
                      {"kind": "absent", "question": "What is my dog's name?", "answer": "I don't know", "quote": "", "session": 0},
                      {"kind": "recall", "question": "Who is my lawyer?", "answer": "Mr Stone", "quote": "My lawyer is Mr Stone.", "session": 1}]
            else:
                qs = [{"kind": "recall", "question": "Who is my dentist?", "answer": "Dr Sample", "quote": "My dentist is Dr Sample", "session": 1},
                      {"kind": "absent", "question": "Where do I bank?", "answer": "I don't know", "quote": "", "session": 0}]
            return msg([t(json.dumps({"questions": qs}))])
        if system.startswith(em.KEEPER_SYSTEM[:30]):
            if len(messages) > 1:                                    # the tool result came back: done
                return msg([t("Done.")])
            entries = []
            for key, value, attr in FACTS:
                if key in user and value in user.split("The session of", 1)[1]:
                    entries.append({"scope": "household", "attribute": attr, "value": value, "source": "principal"})
            entries.append({"scope": "household", "attribute": "lawyer", "value": "Mr Stone", "source": "principal"})   # refused: never said
            return msg([SimpleNamespace(type="tool_use", id="tu1", name="note_fact", input={"entries": entries})])
        ans = "I don't know"
        q = messages[-1]["content"]
        for key, value, attr in FACTS:
            if attr in q.lower() and key in system:
                ans = value
        if "rent" in q and "3,800" not in system and "3,300" in system:
            ans = "3,300"
        return msg([t(f"From what I have: ANSWER: {ans}")])


class Extractor:
    """A bare ModelClient for memory v2's extraction: nominates every principal line, and the grouped writer keeps each
    fact in it as a keyed memory (scope, attribute), so a later value supersedes an earlier one."""

    def complete(self, req):
        user = req.messages[0].content[0].text
        lines = dict((int(n), line) for n, line in re.findall(r"^(\d+): (.*)$", user, flags=re.M))
        if req.job == "nominate":
            spans = [{"marker": "rule_or_decision", "start_line": n, "end_line": n, "quote": line} for n, line in lines.items() if "] User:" in line]
            return Completion(content=(text(json.dumps({"spans": spans})),), stop_reason="end_turn", usage=Usage(input=200, output=30))
        mems = []
        for k, (n, line) in enumerate(sorted((n, l) for n, l in lines.items() if "] User:" in l)):
            said = re.sub(r"^.*?\+00:00\] [^:]+: ", "", line)
            for attr, word in (("accountant", "accountant"), ("rent", "rent"), ("dentist", "dentist")):
                if word in said:
                    mems.append({"moment": f"m{k + 1}", "claim": said, "kind": "stated", "actor_class": "principal", "actor_id": "x",
                                 "quote": said, "quote_start_line": n, "quote_end_line": n, "timestamp": "", "scope": "household",
                                 "entity": "", "attribute": attr, "value_status": "current", "supersession_condition": "Until it changes.",
                                 "notes": ""})
        return Completion(content=(text(json.dumps({"memories": mems, "declined": []})),), stop_reason="end_turn",
                          usage=Usage(input=400, output=60))


@pytest.fixture
def sess(tmp_path):
    p = tmp_path / "sessions.jsonl"
    p.write_text("".join(json.dumps(s) + "\n" for s in SESSIONS))
    return em.load_sessions(str(p))


def test_questions_are_drafted_and_one_without_the_principals_words_is_dropped(sess, tmp_path):
    out = em.make_questions(sess, raw=FakeSDK(), model="claude-sonnet-5-5", out_path=str(tmp_path / "q.jsonl"), root=tmp_path)
    assert out["kept"] == 5 and out["by_kind"] == {"recall": 2, "update": 1, "absent": 2}
    assert [d["reason"] for d in out["dropped"]] == ["the quote is not the person's own words in that session"]
    rows = [json.loads(x) for x in (tmp_path / "q.jsonl").read_text().splitlines()]
    assert {r["scope"] for r in rows} == {"user:ana", "user:bo"}


def test_four_arms_scored_against_history_with_the_write_path_counted(sess, tmp_path):
    qp = tmp_path / "q.jsonl"
    em.make_questions(sess, raw=FakeSDK(), model="claude-sonnet-5-5", out_path=str(qp), root=tmp_path / "e")
    said = []
    rep = em.evaluate(sess, str(qp), raw=FakeSDK(), model="claude-sonnet-5-5", out=str(tmp_path / "e"), extract_client=Extractor(),
                      say=said.append)
    a = rep["arms"]
    assert said[0].startswith("questions frozen") and rep["header"]["questions_sha256"] in said[0]
    assert a["none"]["correct"] == 2 and a["none"]["by_kind"]["absent"]["accuracy"] == 1.0          # only the absent ones
    assert a["history"]["correct"] == 5
    assert a["notes"]["correct"] == 5 and a["notes"]["notes"]["refused"] >= 1                        # the lawyer, never said
    assert "the principal" in json.dumps(a["notes"]["notes"]["refused_reasons"]) or a["notes"]["notes"]["refused_reasons"]
    assert a["notes"]["cost"]["write_path_usd"] > 0 and a["none"]["cost"]["write_path_usd"] == pytest.approx(0, abs=1e-12)
    assert a["memory_v2"]["scopes"]["user:ana"]["accepted"] >= 2 and a["memory_v2"]["correct"] >= 4
    assert a["notes"]["against_history"]["held"] and not a["none"]["against_history"]["held"]
    v = rep["verdict"]
    assert v[0].startswith("none: ") and "words only" in v[0].split(",")[0]                        # leads with the strict score
    assert any(x.startswith("cheapest arm holding quality against carried history, on value match:") for x in v)
    assert rep["total_spend_usd"] == pytest.approx(sum(x["cost"]["answers_usd"] + x["cost"]["write_path_usd"] for x in a.values()))
    assert any(x.startswith("total spend: $") for x in v)
    md = (tmp_path / "e" / "report.md").read_text()
    assert md.startswith("# recant eval memory") and "Recant's own time per call p50" in md and ", p95 " in md
    assert a["history"]["latency"]["recant_overhead"]["n"] > 0


def test_scoring_is_mechanical():
    q = {"kind": "recall", "answer": "Alice Example"}
    assert em.correct(q, "ANSWER: Alice Example") and not em.correct(q, "ANSWER: I don't know") and not em.correct(q, "Alice Example")
    assert em.correct({"kind": "absent", "answer": "I don't know"}, "ANSWER: I don't know")
    assert not em.correct({"kind": "absent", "answer": "I don't know"}, "ANSWER: Rex")


def test_a_stopped_eval_resumes_without_rebuilding_or_reanswering(sess, tmp_path):
    """A run stopped part-way (by a time limit, say) resumes where it stopped and loses nothing."""
    qp = tmp_path / "q.jsonl"
    em.make_questions(sess, raw=FakeSDK(), model="claude-sonnet-5-5", out_path=str(qp), root=tmp_path / "e")

    class Dies(FakeSDK):
        def __init__(self):
            super().__init__()
            self.answers = 0

        def create(self, *, system, messages, **kw):
            if system.startswith(em.ANSWER_SYSTEM[:30]) and "notes (" in system:
                self.answers += 1
                if self.answers == 3:
                    raise KeyboardInterrupt                             # the process stopped mid-arm
            return super().create(system=system, messages=messages, **kw)
    with pytest.raises(KeyboardInterrupt):
        em.evaluate(sess, str(qp), raw=Dies(), model="claude-sonnet-5-5", out=str(tmp_path / "e"), extract_client=Extractor(),
                    say=lambda *_: None)
    marks = {p: p.stat().st_mtime_ns for p in (tmp_path / "e" / "notes").glob("*/built.json")}
    assert marks                                                                                 # a scope's notes were built
    again = FakeSDK()
    rep = em.evaluate(sess, str(qp), raw=again, model="claude-sonnet-5-5", out=str(tmp_path / "e"), extract_client=Extractor(),
                      say=lambda *_: None)
    assert all(p.stat().st_mtime_ns == t for p, t in marks.items())                             # ... and not built again
    rows = [json.loads(x) for x in (tmp_path / "e" / "answers.jsonl").read_text().splitlines()]
    assert len(rows) == len({(r["arm"], r["id"]) for r in rows}) == 5 * 4                        # every answer once
    assert rep["arms"]["notes"]["correct"] == 5 and rep["arms"]["history"]["correct"] == 5
    with pytest.raises(SystemExit, match="other questions"):
        qp.write_text(qp.read_text() + json.dumps({"id": "x", "scope": "user:ana", "kind": "absent", "question": "?",
                                                   "answer": "I don't know"}) + "\n")
        em.evaluate(sess, str(qp), raw=FakeSDK(), model="claude-sonnet-5-5", out=str(tmp_path / "e"), extract_client=Extractor(),
                    say=lambda *_: None)


def test_scopes_in_parallel_score_the_same(sess, tmp_path):
    qp = tmp_path / "q.jsonl"
    em.make_questions(sess, raw=FakeSDK(), model="claude-sonnet-5-5", out_path=str(qp), root=tmp_path / "s")
    a = em.evaluate(sess, str(qp), raw=FakeSDK(), model="claude-sonnet-5-5", out=str(tmp_path / "s"), extract_client=Extractor(),
                    say=lambda *_: None)
    b = em.evaluate(sess, str(qp), raw=FakeSDK(), model="claude-sonnet-5-5", out=str(tmp_path / "p"), extract_client=Extractor(),
                    say=lambda *_: None, parallel=4)
    assert {k: v["correct"] for k, v in a["arms"].items()} == {k: v["correct"] for k, v in b["arms"].items()}


def test_a_right_answer_in_the_principals_own_format_is_right():
    """the words-only check scored these wrong."""
    assert em.matches("25/03/2027", "2027-03-25") and em.matches("13 May 2027", "2027-05-13")
    assert em.matches("26,000 (EUR)", "26000") and em.matches("€ 2,000.00", "2000.00")
    assert not em.matches("26,001", "26000") and not em.matches("24/03/2027", "2027-03-25")
    assert em.matches("ACC-EXAMPLE1", "ACC-EXAMPLE1") and not em.matches("ACC-EXAMPLE2", "ACC-EXAMPLE1")
    q = {"kind": "recall", "answer": "2027-03-25"}
    assert em.correct(q, "ANSWER: 25/03/2027") and not em.correct(q, "ANSWER: 25/03/2027", strict=True)


@pytest.mark.parametrize("question,drafted,key", [
    ("Which project do I manage?", "Project Juniper", "Juniper"),
    ("What project do I run, and for which cooperative?", "Project Alder, Brook Cooperative", "Alder, Brook"),
    ("What is the deadline now?", "the 14 March deadline", "14 March"),
    ("Who is our accountant?", "Alice Example", "Alice Example"),
    ("How should prices be quoted to me?", "including German VAT", "including German VAT"),   # not the drafter's to judge:
    ("What is my name?", "name", "name"),                                                    # kept, listed to check
    ("What is my New York office address?", "10 New York Avenue", "10 New York Avenue"),     # a word inside is the value's
])
def test_a_suggested_key_drops_the_question_s_words_from_its_ends(question, drafted, key):
    """The suggestion only: the 0.3.2 drafter, told to write minimal keys, still wrote 'Project <name>' (an outside
    developer trimmed 6 of 26 by hand). It is never applied by itself: see the next test."""
    assert em.trim_key(question, drafted) == key


def test_no_key_is_changed_by_its_suggestion():
    """'New York Avenue' for 'my New York office address' would be suggested as 'Avenue', which scores 'Boston Avenue'
    right: so a suggestion is listed, never applied."""
    assert em.trim_key("What is my New York office address?", "New York Avenue") == "Avenue"
    q = {"kind": "recall", "answer": "New York Avenue"}
    assert not em.correct(q, "ANSWER: Boston Avenue")


def test_drafting_reports_trimmed_keys_and_lists_long_ones(sess, tmp_path):
    out = em.make_questions(sess, raw=FakeSDK(), model="claude-sonnet-5-5", out_path=str(tmp_path / "q.jsonl"), root=tmp_path)
    assert out["keys_with_a_suggestion"] == 0 and out["keys_to_check"] == []


def test_a_leading_letter_that_reads_like_an_article_is_kept(tmp_path):
    """'A minus' is a grade, not an article and a word: the key is kept as drafted. (The words-only check reads only
    words of two characters or more, so it cannot tell 'A minus' from 'B minus' either: a known limit, in the README.)"""
    class Grade(FakeSDK):
        def create(self, *, system, messages, **kw):
            if system.startswith(em.QUESTION_SYSTEM[:30]):
                qs = [{"kind": "recall", "question": "What is my grade?", "answer": "A minus", "quote": "My grade is A minus.",
                       "session": 1}]
                return msg([t(json.dumps({"questions": qs}))])
            return super().create(system=system, messages=messages, **kw)
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps({"scope": "user:g", "ts": "2026-01-05T09:00:00+00:00",
                             "messages": [{"role": "user", "content": "My grade is A minus."}]}) + "\n")
    em.make_questions(em.load_sessions(str(p)), raw=Grade(), model="claude-sonnet-5-5", out_path=str(tmp_path / "q.jsonl"), root=tmp_path)
    row, = [json.loads(x) for x in (tmp_path / "q.jsonl").read_text().splitlines()]
    assert row["answer"] == "A minus"


def test_each_arm_is_told_who_is_asking(sess, tmp_path):
    """Without the asker's name, a memory arm can hold a person's facts and still not know they are the one asking."""
    assert {s["name"] for s in sess} == {"ana", "bo"}                       # the scope after its colon, by default
    p = tmp_path / "named.jsonl"
    p.write_text(json.dumps({"scope": "user:7", "ts": "2026-01-05T09:00:00+00:00", "name": "Ana Example",
                             "messages": [{"role": "user", "content": "hi"}]}) + "\n")
    assert em.load_sessions(str(p))[0]["name"] == "Ana Example"

    class Seeing(FakeSDK):
        def __init__(self):
            super().__init__()
            self.systems = []

        def create(self, *, system, messages, **kw):
            self.systems.append(system)
            return super().create(system=system, messages=messages, **kw)
    qp = tmp_path / "q.jsonl"
    em.make_questions(sess, raw=FakeSDK(), model="claude-sonnet-5-5", out_path=str(qp), root=tmp_path / "e")
    raw = Seeing()
    em.evaluate(sess, str(qp), raw=raw, model="claude-sonnet-5-5", out=str(tmp_path / "e"), extract_client=Extractor(),
                arms=("none", "history"), say=lambda *_: None)
    answers = [x for x in raw.systems if x.startswith(em.ANSWER_SYSTEM[:30])]
    assert answers and all("The person asking is ana." in x or "The person asking is bo." in x for x in answers)


def test_a_few_scopes_are_said_to_be_too_few_for_the_intervals(sess, tmp_path):
    qp = tmp_path / "q.jsonl"
    em.make_questions(sess, raw=FakeSDK(), model="claude-sonnet-5-5", out_path=str(qp), root=tmp_path / "e")
    rep = em.evaluate(sess, str(qp), raw=FakeSDK(), model="claude-sonnet-5-5", out=str(tmp_path / "e"), extract_client=Extractor(),
                      arms=("none", "history"), say=lambda *_: None)
    v = rep["verdict"]
    assert v[0].startswith("none: ")                                          # still led by the strict score
    assert any(x.startswith("2 scopes: the intervals resample whole scopes") for x in v)
    assert any("(words only and value match)" in x for x in v)              # one figure where the two agree


def test_an_estimate_runs_the_first_scope_only_and_scales_it(sess, tmp_path):
    qp = tmp_path / "q.jsonl"
    em.make_questions(sess, raw=FakeSDK(), model="claude-sonnet-5-5", out_path=str(qp), root=tmp_path / "e")
    est = em.estimate(sess, str(qp), raw=FakeSDK(), model="claude-sonnet-5-5", out=str(tmp_path / "est"), extract_client=Extractor())
    assert est["scopes"] == 2 and est["questions"] == 5 and est["questions_run"] < est["questions"]
    assert est["estimated_usd"] > est["spent_usd"] > 0
    assert not (tmp_path / "e" / "report.json").exists()                     # the real run did not start


def test_a_question_the_person_asked_is_not_a_fact_they_stated():
    ss = [{"scope": "user:a", "ts": "2026-01-05T09:00:00+00:00", "name": "a",
           "turns": [{"role": "user", "text": "What were the Q3 2026 sales numbers?"}]}]
    q = {"kind": "recall", "question": "Which quarter did I ask about?", "answer": "Q3 2026",
         "quote": "What were the Q3 2026 sales numbers?", "session": 1}
    assert em.check_question(q, ss) == "the quote is a question the person asked, not a fact they stated"


def test_absent_questions_are_held_to_a_third_of_a_scope_s(sess, tmp_path):
    """An outside developer's drafter wrote 11 absent questions of 26, near-duplicates among them."""
    class Absentee(FakeSDK):
        def create(self, *, system, messages, **kw):
            if system.startswith(em.QUESTION_SYSTEM[:30]):
                qs = [{"kind": "recall", "question": "Who is our accountant?", "answer": "Alice Example",
                       "quote": "Our accountant is Alice Example.", "session": 1}] + \
                     [{"kind": "absent", "question": f"How many people are on my team ({i})?", "answer": "I don't know",
                       "quote": "", "session": 0} for i in range(5)]
                return msg([t(json.dumps({"questions": qs}))])
            return super().create(system=system, messages=messages, **kw)
    out = em.make_questions([s for s in sess if s["scope"] == "user:ana"], raw=Absentee(), model="claude-sonnet-5-5",
                            out_path=str(tmp_path / "q.jsonl"), root=tmp_path)
    assert out["by_kind"]["absent"] == 1 and out["by_kind"]["recall"] == 1        # a third of 2, rounded up (it kept 2 of 5)
    assert sum("would be absent" in d["reason"] for d in out["dropped"]) == 4


def test_drafting_keeps_the_drafter_s_key_and_lists_the_suggestion(tmp_path):
    class Address(FakeSDK):
        def create(self, *, system, messages, **kw):
            if system.startswith(em.QUESTION_SYSTEM[:30]):
                qs = [{"kind": "recall", "question": "What is my New York office address?", "answer": "New York Avenue",
                       "quote": "Our New York office is on New York Avenue.", "session": 1}]
                return msg([t(json.dumps({"questions": qs}))])
            return super().create(system=system, messages=messages, **kw)
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps({"scope": "user:ny", "ts": "2026-01-05T09:00:00+00:00",
                             "messages": [{"role": "user", "content": "Our New York office is on New York Avenue."}]}) + "\n")
    out = em.make_questions(em.load_sessions(str(p)), raw=Address(), model="claude-sonnet-5-5", out_path=str(tmp_path / "q.jsonl"),
                            root=tmp_path)
    row, = [json.loads(x) for x in (tmp_path / "q.jsonl").read_text().splitlines()]
    assert row["answer"] == "New York Avenue" and row["answer_suggested"] == "Avenue"
    assert out["keys_to_check"] == [{"id": row["id"], "answer": "New York Avenue", "suggested": "Avenue"}]
