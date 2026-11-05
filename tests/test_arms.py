"""The three arms end to end against a scripted model, and the arithmetic they report."""
import json
import re

import pytest
from ulid import ULID

from recant.eval import load, validate_item
from recant.eval.pools import write_pool
from recant.harness.arms import CacheMiss, dailies_prefix, run_arm, verdicts_of
from recant.harness.reader import declared_nothing
from recant.harness.score import compare, mcnemar_one_sided, report_rates
from recant.log import LogWriter, read_run
from recant.provider import ContentBlock, Usage, text
from recant.provider.base import Raw
from recant.provider.fake import FakeProvider
from recant.store import ingest
from pipeline_fixtures import build_corpus
from recant_fixtures import ALLOWLIST, TENANT


def run(layout, machinery, job):
    return LogWriter(layout, TENANT, job, artifact_hash=machinery.artifact_hash, kernel_hash=machinery.kernel_hash)


def item(q, a, episode, kind="yield", marker="root_cause", hard=False, month="2026-01"):
    return validate_item({"id": str(ULID()), "question": q, "answer": a, "episode": episode, "exposure": 0,
                          "tests": {"type": kind, "marker": None if kind == "noise" else marker, "hard": hard, "stratum": month}})


@pytest.fixture
def suite(layout, machinery, tmp_path):
    ingest(layout, TENANT, build_corpus(tmp_path / "corpus"), ALLOWLIST)
    items = [item("What caused the outage on the first?", "Cache key 1.", "notes/2026-01-01.md#L6-L6"),
             item("What caused the outage on the second?", "Cache key 2.", "notes/2026-01-02.md#L6-L6", hard=True),
             item("Which plan was chosen on the third?", "Plan 3.", "notes/2026-01-03.md#L3-L3"),
             item("Was a dashboard colour ever chosen?", "nothing relevant", None, kind="noise"),
             item("What did we stop believing on the fourth, and what replaced it?", "Vendor claim 4 was dropped.",
                  "notes/2026-01-04.md#L15-L15", kind="supersession", marker="reversal")]
    write_pool(layout, TENANT, "founding", items)
    write_pool(layout, TENANT, "heldout", [item("Held out?", "Never seen.", "notes/2026-01-05.md#L6-L6")])
    return layout, machinery, items


def scripted(model, req):
    """A reader that greps, then answers from what it saw; a judge that compares last words."""
    user_texts = [b.text or "" for m in req.messages for b in m.content if b.kind in ("text", "tool_result")]
    if req.job == "judge":
        u = user_texts[0]
        ref = re.search(r"Reference answer: (.*)", u).group(1); ans = re.search(r"Answer to score: (.*)", u).group(1)
        kind = re.search(r"Item type: (\w+)", u).group(1)
        ok = declared_nothing(ans) if kind == "noise" else (ref.split()[-1].strip(".") in ans and not declared_nothing(ans))
        return Raw((text(json.dumps({"correct": ok, "rationale": "compared"})),), "end_turn", Usage(input=50, output=10))
    if "single word: ready" in user_texts[0]:
        return Raw((text("ready"),), "end_turn", Usage(input=0, output=1, cache_write=300))
    q = re.search(r"Question: (.*)", user_texts[0]).group(1)
    if req.job == "eval_grep" and req.tools and not any(b.kind == "tool_result" for m in req.messages for b in m.content):
        words = [w for w in re.findall(r"[a-z]+", q.lower()) if w in ("outage", "plan", "dashboard", "believing")]
        return Raw((ContentBlock("tool_use", tool_name="grep", tool_input={"terms": words or ["zzz"]}, tool_use_id="t1"),), "tool_use", Usage(input=100, output=20))
    material = "\n".join(user_texts) + "\n" + "\n".join(b.text or "" for b in req.system)
    if "dashboard" in q.lower() or "colour" in material and "dashboard" in q.lower():
        return Raw((text("NOTHING RELEVANT: no colour is recorded."),), "end_turn", Usage(input=200, output=10, cache_read=300 if req.job == "eval_context" else 0))
    m = re.search(r"CAUSE: on (2026-01-0\d) the outage was caused by (cache key \d)", material)
    day = re.search(r"on the (\w+)", q).group(1)
    n = {"first": "1", "second": "2", "third": "3", "fourth": "4"}[day]
    if "plan" in q.lower():
        m2 = re.search(rf"on 2026-01-0{n} we chose (plan \d)", material)
        ans = f"It was {m2.group(1).capitalize()}." if m2 else "NOTHING RELEVANT"
    elif "believing" in q.lower():
        m3 = re.search(rf"on 2026-01-0{n} we stopped believing the vendor claim number (\d)", material)
        ans = f"Vendor claim {m3.group(1)} was dropped." if m3 else "NOTHING RELEVANT"
    else:
        m4 = re.search(rf"on 2026-01-0{n} the outage was caused by (cache key \d)", material)
        ans = f"It was {m4.group(1)}." if m4 else "NOTHING RELEVANT"
    usage = Usage(input=200, output=12, cache_read=300 if req.job == "eval_context" else 0)
    if "Plan 3" in ans and req.job == "eval_oracle":
        return Raw((text("It was Plan 3 and furthermore"),), "max_tokens", usage)     # a cut-off reply
    return Raw((text(ans),), "end_turn", usage)


@pytest.mark.parametrize("arm", ["grep", "context", "oracle"])
def test_each_arm_answers_every_item_it_should_and_logs_a_verdict_per_item(suite, arm):
    layout, machinery, items = suite
    with run(layout, machinery, f"eval_{arm}") as log:
        rep = run_arm(layout, TENANT, machinery, FakeProvider(machinery, log, scripted), log, arm, batch=False)
    recs = list(read_run(log.path, typed=False))
    verdicts = [r for r in recs if r["kind"] == "eval_verdict"]
    expected = 4 if arm == "oracle" else 5                       # the oracle skips the noise item
    assert rep["items"] == expected == len(verdicts) and rep["arm"] == arm
    assert all(r["arm"] == arm and r["item_key_hash"].startswith("sha256:") and r["reader_prompt_version"] == "q1" for r in verdicts)
    assert rep["rates"]["yield"]["n"] == 3 and rep["rates"]["supersession"]["n"] == 1
    if arm == "oracle":
        assert "noise" not in rep["rates"] and rep["cut_off"] == 1 and rep["rates"]["yield"]["k"] == 2   # the cut-off reply is wrong
        cut = next(r for r in verdicts if r["cut_off"]); assert cut["verdict"] is False and "cap" in cut["rationale"]
    else:
        assert rep["rates"]["noise"] == {"k": 1, "n": 1, "rate": 1.0, "interval_95": [0.025, 1.0]}
        assert rep["rates"]["yield"]["k"] == 3
    assert rep["by_hard"]["yield"]["hard"]["n"] == 1 and rep["by_hard"]["yield"]["plain"]["n"] == 2
    assert [r for r in recs if r["kind"] == "eval_report"][0]["arm"] == arm
    assert all(i.exposure == 0 for i in load(layout, TENANT, "founding", "scoring"))          # baseline runs do not expose
    assert not any("Never seen" in json.dumps(r) for r in recs)                                # the held-out slice was never read


def test_the_grep_arm_traces_its_searches_and_reports_whether_it_saw_the_cited_span(suite):
    layout, machinery, items = suite
    with run(layout, machinery, "eval_grep") as log:
        rep = run_arm(layout, TENANT, machinery, FakeProvider(machinery, log, scripted), log, "grep", batch=False)
    traces = [r for r in read_run(log.path, typed=False) if r["kind"] == "retrieval_trace"]
    assert len(traces) == 5 and all(t["calls"] and t["calls"][0]["terms"] for t in traces if t["item"] != items[3].id)
    assert rep["retrieval"]["cited_span_seen"]["n"] == 4 and rep["retrieval"]["cited_span_seen"]["k"] >= 3
    # the loop is capped: a reader that never stops searching is made to answer
    def endless(model, req):
        if req.tools:
            return Raw((ContentBlock("tool_use", tool_name="grep", tool_input={"terms": ["outage"]}, tool_use_id="t"),), "tool_use", Usage(input=10, output=5))
        return scripted(model, req)
    with run(layout, machinery, "eval_grep") as log:
        run_arm(layout, TENANT, machinery, FakeProvider(machinery, log, endless), log, "grep", batch=False, items=items[:1])
    t = [r for r in read_run(log.path, typed=False) if r["kind"] == "retrieval_trace"][0]
    assert len(t["calls"]) == machinery.kernel.eval.arms.grep.calls


def test_the_context_arm_stops_on_a_cache_miss_and_its_prefix_is_the_dailies_only(suite):
    layout, machinery, items = suite
    prefix = dailies_prefix(layout, TENANT, machinery.kernel.corpus.full_context)
    assert prefix.count("===== notes/") == 36 and "talk" not in prefix
    def cold(model, req):
        r = scripted(model, req)
        return Raw(r.content, r.stop_reason, Usage(input=r.usage.input, output=r.usage.output, cache_read=0))
    with pytest.raises(CacheMiss):
        with run(layout, machinery, "eval_context") as log:
            run_arm(layout, TENANT, machinery, FakeProvider(machinery, log, cold), log, "context")
    assert [r for r in read_run(log.path, typed=False) if r["kind"] == "run_finished"][0]["stopped_by"] == "error"
    with run(layout, machinery, "eval_context") as log:
        rep = run_arm(layout, TENANT, machinery, FakeProvider(machinery, log, scripted), log, "context")
    assert rep["cache"] == {"calls": 5, "hits": 5, "prefix_tokens": 300}


def test_rates_comparison_and_mcnemar_match_worked_examples():
    assert mcnemar_one_sided(0, 0) == 1.0 and mcnemar_one_sided(5, 0) == pytest.approx(1 / 32)
    assert mcnemar_one_sided(3, 1) == pytest.approx(5 / 16) and mcnemar_one_sided(0, 4) == 1.0
    a = {f"i{n}": n < 70 for n in range(100)}; b = {f"i{n}": n < 60 for n in range(100)}
    c = compare(a, b, margin_pp=4.5)
    assert c == {"items": 100, "a_right_b_wrong": 10, "a_wrong_b_right": 0, "a_rate": 0.7, "b_rate": 0.6, "difference_pp": 10.0,
                 "p_one_sided": round(1 / 1024, 6), "margin_pp": 4.5, "beats_margin": True}
    tie = compare(a, a, margin_pp=4.5); assert tie["beats_margin"] is False and tie["p_one_sided"] == 1.0
    rows = [{"type": "yield", "hard": h, "verdict": v, "unscorable": u, "cut_off": c_, "retrieved": None}
            for h, v, u, c_ in [(True, True, False, False), (True, False, False, False), (False, True, False, False), (False, False, True, False), (False, False, False, True)]]
    rep = report_rates(rows)
    assert rep["rates"]["yield"]["n"] == 4 and rep["by_hard"]["yield"]["hard"]["k"] == 1 and rep["cut_off"] == 1 and rep["judge_failed"] == 1


def test_verdicts_of_reads_one_arm_and_skips_unscorable(suite):
    layout, machinery, items = suite
    with run(layout, machinery, "eval_oracle") as log:
        run_arm(layout, TENANT, machinery, FakeProvider(machinery, log, scripted), log, "oracle", batch=False)
    v = verdicts_of(read_run(log.path, typed=False), "oracle")
    assert len(v) == 4 and verdicts_of(read_run(log.path, typed=False), "grep") == {}


def test_answers_are_logged_as_they_arrive_and_a_run_can_be_scored_from_them(suite):
    layout, machinery, items = suite
    from recant.harness.arms import answers_from_log, score_answers
    with run(layout, machinery, "eval_oracle") as log:
        run_arm(layout, TENANT, machinery, FakeProvider(machinery, log, scripted), log, "oracle", batch=False)
    recs = list(read_run(log.path, typed=False))
    logged = [r for r in recs if r["kind"] == "reader_answer"]
    assert len(logged) == 4 and sum(r["cut_off"] for r in logged) == 1
    assert [r["kind"] for r in recs].index("reader_answer") < [r["kind"] for r in recs].index("eval_verdict")
    rebuilt = answers_from_log(recs, "oracle", items)
    with run(layout, machinery, "eval_oracle") as again:
        rep = score_answers(layout, TENANT, machinery, FakeProvider(machinery, again, scripted), again, "oracle", rebuilt, batch=False)
    assert rep["items"] == 4 and rep["rates"]["yield"]["k"] == 2 and rep["cut_off"] == 1


def test_a_reader_refusal_is_scored_wrong_and_counted_not_unscorable(suite):
    """A refusal is the arm producing no answer. It stays inside n, unlike a
    provider failure, and the report counts it apart from cut-offs and judge failures."""
    layout, machinery, items = suite

    def refusing(model, req):
        if req.job == "eval_oracle" and "second" in "".join(b.text or "" for m in req.messages for b in m.content):
            return Raw((), "refusal", Usage(input=10, output=0))
        return scripted(model, req)

    with run(layout, machinery, "eval_oracle") as log:
        rep = run_arm(layout, TENANT, machinery, FakeProvider(machinery, log, refusing), log, "oracle", batch=False)
    verdicts = [r for r in read_run(log.path, typed=False) if r["kind"] == "eval_verdict"]
    refused = [r for r in verdicts if "refused" in r["rationale"]]
    assert len(refused) == 1 and refused[0]["verdict"] is False and refused[0]["unscorable"] is False and refused[0]["hard"] is True
    assert rep["refused"] == 1 and rep["judge_failed"] == 0 and rep["cut_off"] == 1
    assert rep["rates"]["yield"]["n"] == 3 and rep["rates"]["yield"]["k"] == 1        # the refusal and the cut-off both count wrong
    assert [r for r in read_run(log.path, typed=False) if r["kind"] == "eval_report"][0]["refused"] == 1


def test_report_rates_keeps_a_provider_failure_outside_n_but_a_refusal_inside():
    rows = [{"type": "yield", "hard": False, "verdict": True, "unscorable": False, "cut_off": False, "retrieved": None},
            {"type": "yield", "hard": False, "verdict": False, "unscorable": False, "cut_off": False, "refused": True, "retrieved": None},
            {"type": "yield", "hard": False, "verdict": False, "unscorable": True, "cut_off": False, "retrieved": None}]
    rep = report_rates(rows)
    assert rep["rates"]["yield"]["n"] == 2 and rep["rates"]["yield"]["k"] == 1 and rep["refused"] == 1 and rep["judge_failed"] == 1
