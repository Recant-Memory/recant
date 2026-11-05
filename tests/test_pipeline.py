"""Nominate, draft, review, sample, audit, found, calibrate: the whole path,
against a scripted model that misbehaves in known places."""
import pytest

from recant.eval import NOTHING, load
from recant.eval.audit import audit_report
from recant.eval.blind import AuditError, draw_blind_sample
from recant.eval.founding import FoundingError, found
from recant.eval.ledger import Ledger
from recant.eval.review import BlindFirst, ReviewError, decide, record_blind_answer
from recant.harness.draft import Drafter, provenance, unmarked_windows
from recant.harness.judge import calibrate
from recant.harness.nominate import Span, check_span, dedupe, nominate, spans_from_log
from recant.harness.review import model_review
from recant.log import LogWriter, read_run
from recant.machinery import Machinery
from recant.provider.fake import FakeProvider
from recant.store import ingest
from pipeline_fixtures import build_corpus, scripted
from recant_fixtures import ALLOWLIST, KERNEL_YAML, TENANT

HUMAN = {"kind": "human", "id": "r1"}


def run(layout, machinery, job):
    return LogWriter(layout, TENANT, job, artifact_hash=machinery.artifact_hash, kernel_hash=machinery.kernel_hash)


@pytest.fixture
def world(layout, machinery, tmp_path):
    ingest(layout, TENANT, build_corpus(tmp_path / "corpus"), ALLOWLIST)
    return layout, machinery


@pytest.fixture
def drafted(world):
    layout, machinery = world
    with run(layout, machinery, "nominate") as log:
        spans = nominate(layout, TENANT, machinery, FakeProvider(machinery, log, scripted), log)
    with run(layout, machinery, "draft") as dlog:
        d = Drafter(layout, TENANT, machinery, FakeProvider(machinery, dlog, scripted), dlog, seed=3)
        paths = sorted({s.path for s in spans})
        made = {"marked": d.marked(spans), "unmarked": d.unmarked(spans, paths), "noise": d.noise(spans),
                "seeded": d.seeded([{"question": "Which plan was chosen on the first day?", "paths": ["notes/2026-01-01.md"]},
                                    {"question": "An unanswerable one: which ruleset version?", "paths": ["notes/2026-01-01.md"]}])}
    return layout, machinery, spans, log, dlog, made


def finished(log):
    return [r for r in read_run(log.path, typed=False) if r["kind"] == "run_finished"][0]


# --- nomination ----------------------------------------------------------------------

def test_nomination_keeps_only_what_survives_the_mechanical_check(drafted):
    layout, machinery, spans, log, *_ = drafted
    end = finished(log)
    assert end["examined"] == 37 and end["stopped_by"] == "done"            # 36 dailies and one transcript
    assert end["counts"]["episodes_failed"] == 2                            # one not JSON, one provider error
    assert end["counts"]["dropped: quote is not verbatim in the cited lines"] == 1
    assert end["counts"]["dropped: unknown marker"] == 1 and end["counts"]["dropped: lines out of range"] == 1
    assert len(spans) == 34 * 4 == end["counts"]["spans_after_dedupe"]
    assert {s.marker for s in spans} == {"rule_or_decision", "root_cause", "reversal"}
    assert all(s.path != "notes/2026-01-02.md" for s in spans)
    records = list(read_run(log.path, typed=False))
    assert len([r for r in records if r["kind"] == "nomination"]) == 37       # a failed episode still leaves a record
    bad = next(r for r in records if r["kind"] == "nomination" and r["episode"].startswith("notes/2026-01-04.md"))
    assert sorted(d["reason"] for d in bad["dropped"]) == ["lines out of range", "quote is not verbatim in the cited lines", "unknown marker"]
    assert spans_from_log(records) == spans                                  # the pass need not be paid for twice


def test_check_span_and_dedupe():
    lines = ["alpha", "beta gamma", "delta"]
    ok, why = check_span({"marker": "m", "start_line": 2, "end_line": 3, "quote": "gamma\ndelta"}, "p.md", lines, ["m"])
    assert ok.episode == "p.md#L2-L3" and why is None
    for raw, reason in [({"marker": "m", "start_line": 2, "end_line": 2, "quote": "Gamma"}, "verbatim"),
                        ({"marker": "m", "start_line": 0, "end_line": 1, "quote": "alpha"}, "range"),
                        ({"marker": "m", "start_line": 3, "end_line": 2, "quote": "x"}, "range"),
                        ({"marker": "z", "start_line": 1, "end_line": 1, "quote": "alpha"}, "unknown marker"),
                        ({"marker": "m", "start_line": 1, "end_line": 1, "quote": "  "}, "verbatim"),
                        ({"marker": "m"}, "malformed")]:
        got, why = check_span(raw, "p.md", lines, ["m"])
        assert got is None and reason in why
    joined, _ = check_span({"marker": "m", "start_line": 2, "end_line": 3, "quote": "beta  gamma delta"}, "p.md", lines, ["m"])
    assert joined is not None                                   # a line break returned as a space is still the source's words
    curly, _ = check_span({"marker": "m", "start_line": 1, "end_line": 1, "quote": "i can't - really"}, "p.md",
                          ["i can\u2019t \u2014 really"], ["m"])
    assert curly is not None                                    # punctuation encoding is not paraphrase
    a = Span("m", "p.md#L1-L1", "p.md", 1, 1, "same text")
    b = Span("m", "q.md#L9-L9", "q.md", 9, 9, "same text")      # a re-paste elsewhere
    c = Span("n", "q.md#L9-L9", "q.md", 9, 9, "same text")      # another marker is another moment
    assert dedupe([b, a, c]) == [a, c]


# --- drafting ------------------------------------------------------------------------

def test_every_draft_is_a_valid_item_with_its_provenance_in_the_log(drafted):
    layout, machinery, spans, _, dlog, made = drafted
    drafts = load(layout, TENANT, "drafts", "drafting")
    assert len(drafts) == sum(made.values())
    assert made["marked"] == len(spans) - 5                # five drafts cited lines they were never shown
    logged = {r["item"]: r for r in read_run(dlog.path, typed=False) if r["kind"] == "item_drafted"}
    assert set(logged) == {d.id for d in drafts}
    fp = machinery.drafting_fingerprint()
    for d in drafts:
        rec = logged[d.id]
        assert rec["fingerprint"] == fp and rec["question"] == d.question and rec["candidate_answer"] == d.answer
        assert rec["prompt_version"] in ("d1", "x1", "s1") and rec["model_id"] == "model-medium"
    rejected = [r for r in read_run(dlog.path, typed=False) if r["kind"] == "draft_rejected"]
    assert any("fall outside the lines shown" in r["reason"] for r in rejected)
    assert sum("nothing to ask" in r["reason"] for r in rejected) > 0


def test_item_types_come_out_as_the_spec_describes(drafted):
    layout, *_ , made = drafted
    drafts = load(layout, TENANT, "drafts", "drafting")
    by = lambda f: [d for d in drafts if f(d)]
    supers = by(lambda d: d.tests.type == "supersession")
    assert len(supers) == 34 and all(d.tests.marker == "reversal" for d in supers)
    noise = by(lambda d: d.tests.type == "noise")
    assert len(noise) == made["noise"] + 1 and all(d.episode is None and d.answer == NOTHING for d in noise)
    unmarked = by(lambda d: d.tests.type == "yield" and d.tests.marker is None and not d.tests.hard)
    assert len(unmarked) == made["unmarked"] > 0
    seeded = by(lambda d: d.tests.hard)
    assert {d.tests.type for d in seeded} == {"yield", "noise"} and len(seeded) == 2
    assert {d.tests.stratum for d in drafts} == {"2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"}


def test_a_noise_draft_records_the_search_that_would_contradict_it(drafted):
    layout, _, _, _, dlog, _ = drafted
    noise = [r for r in read_run(dlog.path, typed=False) if r["kind"] == "item_drafted" and r["origin"] == "noise"]
    assert noise and all(r["search_terms"] == ["dashboard", "colour"] and r["prescreen_hits"] == 0 for r in noise)
    assert all(r["source_span"] for r in noise)             # the neighbourhood it was drafted from


def test_the_pool_asks_each_question_once(drafted):
    layout, machinery, spans, _, dlog, _ = drafted
    before = len(load(layout, TENANT, "drafts", "drafting"))
    with run(layout, machinery, "again") as log:
        d = Drafter(layout, TENANT, machinery, FakeProvider(machinery, log, scripted), log, seed=3)
        assert d.marked(spans[:10]) == 0                       # the same spans ask the same questions
    assert len(load(layout, TENANT, "drafts", "drafting")) == before
    dupes = [r for r in read_run(log.path, typed=False) if r["kind"] == "draft_rejected"]
    assert len(dupes) == 10 and all("already in the pool" in r["reason"] for r in dupes)
    questions = [d.question for d in load(layout, TENANT, "drafts", "drafting")]
    assert len(questions) == len(set(questions))


def test_unmarked_windows_touch_no_marked_span():
    texts = {"a.md": [f"l{i}" for i in range(1, 41)]}
    spans = [Span("m", "a.md#L5-L6", "a.md", 5, 6, "q"), Span("m", "a.md#L20-L20", "a.md", 20, 20, "q")]
    got = unmarked_windows(texts, spans, window=4, count=99, seed=1)
    assert got and all(not (a <= 6 and b >= 5) and not (a <= 20 <= b) for _, a, b in got)
    assert got == unmarked_windows(texts, spans, window=4, count=99, seed=1)
    assert len(unmarked_windows(texts, spans, window=4, count=2, seed=1)) == 2


def test_provenance_separates_the_sources_words_from_the_models():
    source = "on 2026-01-01 the outage was caused by cache key 1"
    answer = "It says that the outage was caused by cache key 1 and nothing else"
    spans = provenance(answer, source)
    assert [(answer[s["start"]:s["end"]], s["origin"]) for s in spans] == [
        ("It says that", "generated"), ("the outage was caused by cache key 1", "source"), ("and nothing else", "generated")]
    assert provenance("entirely new words here now", source)[0]["origin"] == "generated"


# --- model review --------------------------------------------------------------------

@pytest.fixture
def reviewed(drafted):
    layout, machinery, *_ = drafted
    with run(layout, machinery, "model_review") as log:
        tally = model_review(layout, TENANT, machinery, FakeProvider(machinery, log, scripted), log)
    return layout, machinery, log, tally


def test_the_model_reviewer_leaves_the_same_record_a_person_would(reviewed):
    layout, machinery, log, tally = reviewed
    drafts = {d.id: d for d in load(layout, TENANT, "drafts", "audit")}
    decisions = [r for r in read_run(log.path, typed=False) if r["kind"] == "review_decision"]
    assert all(r["reviewer"] == {"kind": "model", "id": "model-large"} for r in decisions)
    assert tally["fix"] == 5 and tally["discard"] == 6 and tally["failed"] == 12   # six bad reasons, six refusals
    fixes = [r for r in decisions if r["outcome"] == "fix"]
    assert all(r["key_hash_before"] != r["key_hash_after"] for r in fixes)
    assert all(drafts[r["subject"]["id"]].answer == "The timeout became 20 seconds." for r in fixes)
    assert all(r["key_hash_after"] == drafts[r["subject"]["id"]].key_hash() for r in fixes)
    hard = [d for d in drafts.values() if d.tests.hard]
    assert len(hard) == 2 + 6                                 # the two seeded, and six the reviewer tagged
    ledger = Ledger(layout, TENANT)
    unreviewed = [i for i in drafts if not ledger.state(i).decisions]
    assert len(unreviewed) == 12                              # the failures stay unreviewed, visibly


def test_review_rules_are_enforced(reviewed):
    layout, machinery, log, _ = reviewed
    reasons = machinery.artifact.review_reasons
    drafts = load(layout, TENANT, "drafts", "audit")
    noise = next(d for d in drafts if d.tests.type == "noise")
    plain = next(d for d in drafts if d.tests.type == "yield")
    with run(layout, machinery, "x") as w:
        kw = dict(reviewer=HUMAN, reasons=reasons)
        with pytest.raises(ReviewError, match="reason"):
            decide(layout, TENANT, w, plain.id, outcome="accept", reason="vibes", **kw)
        with pytest.raises(ReviewError, match="outcome"):
            decide(layout, TENANT, w, plain.id, outcome="maybe", reason="staleness", **kw)
        with pytest.raises(ReviewError, match="corrected answer"):
            decide(layout, TENANT, w, plain.id, outcome="fix", reason="staleness", **kw)
        with pytest.raises(ReviewError, match="noise"):
            decide(layout, TENANT, w, noise.id, outcome="fix", reason="staleness", fixed_answer="It was blue.", **kw)
        with pytest.raises(ReviewError, match="noise"):
            decide(layout, TENANT, w, plain.id, outcome="fix", reason="staleness", fixed_answer="nothing relevant", **kw)
        with pytest.raises(KeyError):
            decide(layout, TENANT, w, "01J00000000000000000000000", outcome="accept", reason="staleness", **kw)


# --- the blind-first sample, the audit, founding ------------------------------------------

def settle_failures(layout, machinery):
    """A person clears what the model reviewer could not review, so the pool is finished."""
    ledger = Ledger(layout, TENANT)
    with run(layout, machinery, "cleanup") as w:
        for d in load(layout, TENANT, "drafts", "audit"):
            if not ledger.state(d.id).decisions:
                decide(layout, TENANT, w, d.id, outcome="discard", reason="bad_question" if "bad_question" in
                       machinery.artifact.review_reasons else machinery.artifact.review_reasons[0],
                       reviewer={"kind": "model", "id": "model-large"}, reasons=machinery.artifact.review_reasons)


def test_the_sample_waits_for_a_finished_pool_and_is_drawn_once(reviewed):
    layout, machinery, _, _ = reviewed
    with run(layout, machinery, "blind") as w:
        with pytest.raises(AuditError, match="no review yet"):
            draw_blind_sample(layout, TENANT, machinery, w, seed=11)
        settle_failures(layout, machinery)
        out = draw_blind_sample(layout, TENANT, machinery, w, seed=11)
        assert len(out["items"]) == 6 and out["unfilled"] == []
        with pytest.raises(AuditError, match="already exists"):
            draw_blind_sample(layout, TENANT, machinery, w, seed=12)
    ledger = Ledger(layout, TENANT)
    assert all(ledger.state(i).accepted() for i in out["items"])


def audit_all(layout, machinery, sample, wrong=()):
    drafts = {d.id: d for d in load(layout, TENANT, "drafts", "audit")}
    with run(layout, machinery, "human") as w:
        for n, item_id in enumerate(sample):
            d = drafts[item_id]
            record_blind_answer(layout, TENANT, w, item_id, HUMAN, d.answer if n not in wrong else "I read it differently")
            w.emit("blind_revealed", item=item_id, reviewer=HUMAN)
            if n in wrong and d.tests.type != "noise":
                decide(layout, TENANT, w, item_id, outcome="fix", reason="wrong_answer", reviewer=HUMAN,
                       reasons=machinery.artifact.review_reasons, fixed_answer="The corrected key.")
            elif n in wrong:
                decide(layout, TENANT, w, item_id, outcome="discard", reason="wrong_answer", reviewer=HUMAN,
                       reasons=machinery.artifact.review_reasons)
            else:
                decide(layout, TENANT, w, item_id, outcome="accept", reason="confirmed", reviewer=HUMAN,
                       reasons=machinery.artifact.review_reasons)


def test_blind_first_is_a_state_machine_not_a_courtesy(reviewed):
    layout, machinery, _, _ = reviewed
    settle_failures(layout, machinery)
    with run(layout, machinery, "blind") as w:
        sample = draw_blind_sample(layout, TENANT, machinery, w, seed=11)["items"]
        reasons = machinery.artifact.review_reasons
        with pytest.raises(BlindFirst, match="write the blind answer"):
            decide(layout, TENANT, w, sample[0], outcome="accept", reason="confirmed", reviewer=HUMAN, reasons=reasons)
        with pytest.raises(ReviewError, match="person"):
            record_blind_answer(layout, TENANT, w, sample[0], {"kind": "model", "id": "m"}, "x")
        with pytest.raises(ReviewError, match="text"):
            record_blind_answer(layout, TENANT, w, sample[0], HUMAN, "  ")
        outside = next(d.id for d in load(layout, TENANT, "drafts", "audit") if d.id not in sample)
        with pytest.raises(ReviewError, match="not in the blind-first sample"):
            record_blind_answer(layout, TENANT, w, outside, HUMAN, "x")
        record_blind_answer(layout, TENANT, w, sample[0], HUMAN, "what I wrote before looking")
        with pytest.raises(ReviewError, match="once"):
            record_blind_answer(layout, TENANT, w, sample[0], HUMAN, "a second thought")
        with pytest.raises(BlindFirst, match="reveal"):
            decide(layout, TENANT, w, sample[0], outcome="accept", reason="confirmed", reviewer=HUMAN, reasons=reasons)
        from recant.eval.view import item_view
        assert "answer" not in item_view(layout, TENANT, machinery, w, sample[0], HUMAN)
        assert "answer" not in item_view(layout, TENANT, machinery, w, sample[0], {"kind": "model", "id": "m"}, reveal=True)
        assert item_view(layout, TENANT, machinery, w, sample[0], HUMAN, reveal=True)["answer"]
        rec = decide(layout, TENANT, w, sample[0], outcome="accept", reason="confirmed", reviewer=HUMAN, reasons=reasons)
        assert rec["blind_answer"] == "what I wrote before looking"
        decide(layout, TENANT, w, outside, outcome="accept", reason="confirmed", reviewer=HUMAN, reasons=reasons)  # not blind


def test_the_audit_reports_rates_with_their_n_and_interval_and_never_acts(reviewed):
    layout, machinery, _, _ = reviewed
    settle_failures(layout, machinery)
    assert audit_report(layout, TENANT, machinery) == {"sample": None}
    with run(layout, machinery, "blind") as w:
        sample = draw_blind_sample(layout, TENANT, machinery, w, seed=11)["items"]
    audit_all(layout, machinery, sample[:4], wrong={1})
    rep = audit_report(layout, TENANT, machinery)
    assert rep["sample"] == {"size": 6, "decided": 4, "seed": 11, "unfilled": [], "complete": False, "stale": False, "founded_under": None}
    assert rep["key_disagreement"]["k"] == 1 and rep["key_disagreement"]["n"] == 4
    assert rep["key_disagreement"]["interval_95"][1] > 0.8          # four items say very little, and it shows
    assert rep["anchoring"]["human_accept"]["k"] == 3 and rep["anchoring"]["model_accept_on_same_items"]["n"] == 4
    assert rep["judge_agreement"]["rate"] is None and rep["judge_agreement"]["below_floor"] is None

    with run(layout, machinery, "calibrate") as c:
        done = calibrate(layout, TENANT, machinery, FakeProvider(machinery, c, scripted), c)
    assert done == 4
    rep = audit_report(layout, TENANT, machinery)
    j = rep["judge_agreement"]
    assert j["n"] == 4 and j["floor"] == 0.9 and j["k"] == 4 and j["below_floor"] is False
    with run(layout, machinery, "calibrate") as c:                    # already calibrated: nothing is paid for twice
        assert calibrate(layout, TENANT, machinery, FakeProvider(machinery, c, scripted), c) == 0


def test_founding_once_beside_the_suite_and_frozen(reviewed, tmp_path):
    layout, machinery, _, _ = reviewed
    settle_failures(layout, machinery)
    with run(layout, machinery, "found") as w:
        with pytest.raises(FoundingError, match="no blind-first sample"):
            found(layout, TENANT, machinery, w, seed=5)
        sample = draw_blind_sample(layout, TENANT, machinery, w, seed=11)["items"]
        audit_all(layout, machinery, sample[:5])
        with pytest.raises(FoundingError, match="unfinished: 1"):
            found(layout, TENANT, machinery, w, seed=5)
        audit_all(layout, machinery, sample[5:])
        out = found(layout, TENANT, machinery, w, seed=5)
    c = out["counts"]
    assert (c["founding"]["total"], c["heldout"]["total"], c["reserve"]["total"]) == (20, 4, 4)
    assert 0.25 <= c["founding"]["noise"] / 20 <= 0.30
    yields = c["founding"]["yield"]
    assert c["founding"]["unmarked"] == round(0.1 * yields)          # one in ten yield items, from unmarked windows
    ids = [{i.id for i in load(layout, TENANT, p, "founding")} for p in ("founding", "heldout", "reserve")]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])   # beside, not from
    rec = [r for r in read_run(w.path, typed=False) if r["kind"] == "founding"][0]
    assert rec["seed"] == 5 and set(rec["key_hashes"]["founding"]) == ids[0]
    with run(layout, machinery, "again") as w2:
        with pytest.raises(FoundingError, match="never extended"):
            found(layout, TENANT, machinery, w2, seed=6)

        founded = load(layout, TENANT, "founding", "human_review")[0]
        reasons = machinery.artifact.review_reasons
        if founded.tests.type != "noise":
            with pytest.raises(ReviewError, match="No model|no model"):
                decide(layout, TENANT, w2, founded.id, outcome="fix", reason="wrong_answer", fixed_answer="x y z",
                       reviewer={"kind": "model", "id": "m"}, reasons=reasons)
            with pytest.raises(ReviewError, match="membership"):
                decide(layout, TENANT, w2, founded.id, outcome="discard", reason="wrong_answer", reviewer=HUMAN, reasons=reasons)
            fixed = decide(layout, TENANT, w2, founded.id, outcome="fix", reason="wrong_answer",
                           fixed_answer="A person's correction.", reviewer=HUMAN, reasons=reasons)
            assert fixed["key_hash_before"] != fixed["key_hash_after"]
            assert (layout.eval_dir(TENANT) / "history" / founded.id / "000001.json").exists()
            after = load(layout, TENANT, "founding", "scoring")
            assert len(after) == 20 and next(i for i in after if i.id == founded.id).answer == "A person's correction."


def test_founding_refuses_when_the_prompts_moved_after_the_audit(reviewed, tmp_path):
    layout, machinery, _, _ = reviewed
    settle_failures(layout, machinery)
    with run(layout, machinery, "blind") as w:
        sample = draw_blind_sample(layout, TENANT, machinery, w, seed=11)["items"]
    audit_all(layout, machinery, sample)
    (layout.machinery_dir / "kernel.yaml").write_text(KERNEL_YAML.replace("Draft a question and", "Draft one question and"))
    moved = Machinery.load(layout)
    assert audit_report(layout, TENANT, moved)["sample"]["stale"] is True
    with run(layout, moved, "found") as w:
        with pytest.raises(FoundingError, match="changed after the blind-first audit"):
            found(layout, TENANT, moved, w, seed=5)
        with pytest.raises(AuditError, match="different prompts"):
            draw_blind_sample(layout, TENANT, moved, w, seed=12)


def test_founding_refuses_a_short_pool_and_names_what_is_short(reviewed):
    layout, machinery, _, _ = reviewed
    settle_failures(layout, machinery)
    (layout.machinery_dir / "kernel.yaml").write_text(KERNEL_YAML.replace("suite_size: 20", "suite_size: 300"))
    big = Machinery.load(layout)
    with run(layout, big, "blind") as w:
        sample = draw_blind_sample(layout, TENANT, big, w, seed=11)["items"]
        audit_all(layout, big, sample)
        with pytest.raises(FoundingError, match="cannot fill founding") as caught:
            found(layout, TENANT, big, w, seed=5)
    assert "need" in str(caught.value) and "have" in str(caught.value)
    assert load(layout, TENANT, "founding", "founding") == []


# --- semantic dedupe, and noise held until a person clears it ------------------------------------

import json as _json

from recant.eval import release_held
from recant.eval.search import grep
from recant.harness.draft import overlap, spans_touch
from recant.provider import Usage, text
from recant.provider.base import Raw


def test_question_overlap_and_touching_spans():
    a = "Which form should be used for the example project, the old form-v1 or the new form-v2?"
    b = "For the example project, which form (old form-v1 or new form-v2) should be used, and what is the status of the old form?"
    assert overlap(a, b) >= 0.5
    assert overlap(a, "What made the example client time out when it called the example server?") < 0.2
    assert spans_touch("n/a.md#L78-L78", "n/a.md#L78-L79") and spans_touch("n/a.md#L78-L78", "n/a.md#L79-L80")
    assert not spans_touch("n/a.md#L78-L78", "n/a.md#L81-L82") and not spans_touch("n/a.md#L78-L78", "n/b.md#L78-L78")
    assert not spans_touch(None, "n/a.md#L1-L1")


def test_of_two_drafts_asking_much_the_same_of_touching_lines_the_wider_is_kept(world):
    layout, machinery = world

    def similar(model, req):
        user = req.messages[0].content[0].text
        start, end = map(int, re.search(r"The moment is lines (\d+)-(\d+)", user).groups())
        q = ("What caused the February outage?" if "2026-02-01" in user else
             "Which plan was chosen for the scanner, plan one or plan two?" if end == start else
             "For the scanner, which plan (plan one or plan two) was chosen, and what became of the other?")
        return Raw((text(_json.dumps({"skip": False, "item_type": "yield", "question": q, "candidate_answer": "Plan one.",
                                      "answer_start_line": start, "answer_end_line": end, "superseded_answer": "",
                                      "search_terms": ["plan"], "form": "decision"})),), "end_turn", Usage(input=10, output=10))

    import re
    with run(layout, machinery, "draft") as log:
        d = Drafter(layout, TENANT, machinery, FakeProvider(machinery, log, similar), log)
        made = d.from_spans([("notes/2026-01-01.md", 3, 3, "rule_or_decision"), ("notes/2026-01-01.md", 3, 4, "rule_or_decision"),
                             ("notes/2026-02-01.md", 3, 3, "rule_or_decision")], "marked")
    drafts = load(layout, TENANT, "drafts", "drafting")
    assert made == 2 and sorted(i.episode for i in drafts) == ["notes/2026-01-01.md#L3-L4", "notes/2026-02-01.md#L3-L3"]
    records = list(read_run(log.path, typed=False))
    merged = [r for r in records if r["kind"] == "draft_rejected"]
    assert len(merged) == 1 and merged[0]["source_span"] == "notes/2026-01-01.md#L3-L3" and "much the same question" in merged[0]["reason"]
    assert {r["form"] for r in records if r["kind"] == "item_drafted"} == {"decision"}


def test_a_noise_draft_with_any_prescreen_hit_is_held_until_a_person_clears_it(world):
    layout, machinery = world

    def noisy(model, req):
        day = re.search(r"Episode: notes/(\S+)\.md", req.messages[0].content[0].text).group(1)
        terms = ["cache key", "outage"] if day.endswith("-01") else ["dashboard", "colour"]
        return Raw((text(_json.dumps({"skip": False, "question": f"What did the team decide about the {day} rollout?",
                                      "search_terms": terms, "form": "decision"})),), "end_turn", Usage(input=10, output=10))

    import re
    spans = [Span("root_cause", f"notes/2026-0{m}-0{d}.md#L6-L6", f"notes/2026-0{m}-0{d}.md", 6, 6, "q") for m in (1, 2) for d in (1, 2)]
    kernel = (layout.machinery_dir / "kernel.yaml")
    kernel.write_text(kernel.read_text().replace("noise_draft_ratio: 0.5", "noise_draft_ratio: 1.0"))
    all_in = Machinery.load(layout)
    with run(layout, all_in, "draft") as log:
        made = Drafter(layout, TENANT, all_in, FakeProvider(all_in, log, noisy), log).noise(spans)
    assert made == 2                                                  # only the two the search could not contradict
    assert {i.question[-19:] for i in load(layout, TENANT, "drafts", "drafting")} == {"2026-01-02 rollout?", "2026-02-02 rollout?"}
    held = load(layout, TENANT, "held", "human_review")
    assert len(held) == 2 and all(i.tests.type == "noise" for i in held)
    logged = [r for r in read_run(log.path, typed=False) if r["kind"] == "item_drafted"]
    assert sum(bool(r["held_for"]) for r in logged) == 2 and all(r["prescreen_hits"] > 0 for r in logged if r["held_for"])
    assert [r for r in read_run(log.path, typed=False) if r["kind"] == "run_finished"][0]["counts"]["held"] == 2

    released = release_held(layout, TENANT, held[0].id, into_pool=True)
    release_held(layout, TENANT, held[1].id, into_pool=False)
    assert released.id in {i.id for i in load(layout, TENANT, "drafts", "drafting")}
    assert load(layout, TENANT, "held", "human_review") == [] and len(load(layout, TENANT, "drafts", "drafting")) == 3
    with pytest.raises(KeyError):
        release_held(layout, TENANT, held[0].id, into_pool=True)


def test_the_prescreen_needs_only_two_of_the_terms_and_a_persons_search_needs_all(world):
    layout, _ = world
    terms = ["outage", "cache key", "a phrase that is nowhere"]
    assert grep(layout, TENANT, terms)["total"] == 0
    assert grep(layout, TENANT, terms, min_match=2)["total"] == 36       # one CAUSE line in each of 36 dailies
    assert grep(layout, TENANT, ["outage"], min_match=2)["total"] == 36   # never asks for more terms than there are


def test_a_reply_cut_off_at_the_token_cap_says_so(world):
    layout, machinery = world
    cut = lambda model, req: Raw((), "max_tokens", Usage(input=10, output=4000))
    with run(layout, machinery, "draft") as log:
        d = Drafter(layout, TENANT, machinery, FakeProvider(machinery, log, cut), log)
        assert d.from_spans([("notes/2026-01-01.md", 3, 3, "rule_or_decision")], "marked") == 0
    reason = [r for r in read_run(log.path, typed=False) if r["kind"] == "draft_rejected"][0]["reason"]
    assert "ran out of output tokens" in reason
