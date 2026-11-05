"""The service: search over the warm tier with status travelling, the manifest, coverage, and the two
service arms end to end against a scripted reader, with exposure counted."""
import json
import re

import pytest
from ulid import ULID

from recant.eval import load, validate_item
from recant.eval.pools import write_pool
from recant.harness.arms import CacheMiss, run_arm
from recant.harness.score import coverage_split
from recant.log import LogWriter, read_run
from recant.provider import ContentBlock, Usage, text
from recant.provider.base import Raw
from recant.provider.fake import FakeProvider
from recant.store import MemoryStore, ingest
from recant.store.review import memory_queue
from recant.store.search import coverage, load_entries, manifest, search_memory
from pipeline_fixtures import build_corpus
from recant_fixtures import ALLOWLIST, TENANT, make_record


def run(layout, machinery, job):
    return LogWriter(layout, TENANT, job, artifact_hash=machinery.artifact_hash, kernel_hash=machinery.kernel_hash)


def item(q, a, episode, kind="yield", marker="root_cause", hard=False, month="2026-01"):
    return validate_item({"id": str(ULID()), "question": q, "answer": a, "episode": episode, "exposure": 0,
                          "tests": {"type": kind, "marker": None if kind == "noise" else marker, "hard": hard, "stratum": month}})


def entry(store, log, claim, episode, *, state="proposed", kind="concluded", cls="agent", scope="scanner", day="2026-01-01",
          derived=(), notes="", superseded_by=None, reason=None):
    rec = make_record(claim=claim, kind=kind, scope=scope, derived_from=list(derived),
                      actor={"class": cls, "id": "owner" if cls == "principal" else "agent"},
                      anchor={"quote": "q", "episode": episode, "channel": None, "message_id": None, "timestamp": day},
                      status={"state": "proposed", "reason": None, "failure_class": None, "superseded_by": None})
    store.put(TENANT, rec, writer="consolidation", log=log)
    path = ["proposed"]
    for nxt in {"probation": ["probation"], "live": ["probation", "live"], "contested": ["probation", "live", "contested"],
                "retired": ["retired"], "superseded": ["probation", "live", "superseded"]}.get(state, []):
        rec["status"] = {"state": nxt, "reason": reason or "r", "failure_class": "contradiction" if nxt == "superseded" else None,
                         "superseded_by": superseded_by if nxt == "superseded" else None}
        store.put(TENANT, rec, writer="test", log=log)
    return rec["id"]


@pytest.fixture
def world(layout, machinery, tmp_path):
    ingest(layout, TENANT, build_corpus(tmp_path / "corpus"), ALLOWLIST)
    store = MemoryStore(layout, ALLOWLIST)
    with run(layout, machinery, "seed") as log:
        ids = {}
        ids["k1"] = entry(store, log, "Cache key 1 caused the outage on the first.", "notes/2026-01-01.md#L6-L6", state="live")
        ids["k2"] = entry(store, log, "Cache key 2 caused the outage on the second.", "notes/2026-01-02.md#L6-L6", state="probation", day="2026-01-02")
        ids["p3"] = entry(store, log, "Plan 3 was chosen for the scanner on the third.", "notes/2026-01-03.md#L3-L3", state="proposed",
                          kind="stated", cls="principal", day="2026-01-03")
        ids["old"] = entry(store, log, "Vendor claim 4 was believed on the fourth.", "notes/2026-01-04.md#L2-L2", day="2026-01-04")
        ids["new"] = entry(store, log, "Vendor claim 4 was dropped on the fourth.", "notes/2026-01-04.md#L15-L15", state="live",
                           kind="stated", day="2026-01-04")
        ids["ret"] = entry(store, log, "A retired claim about the outage on the first.", "notes/2026-01-01.md#L2-L2", state="retired")
        ids["con"] = entry(store, log, "Cache key 1 was maybe not the outage cause on the first.", "notes/2026-01-01.md#L7-L7",
                           state="contested", reason="a later run contradicted it")
        # supersede `old` by `new`
        rec, notes = store.get(TENANT, ids["old"])
        d = rec.to_dict()
        for nxt in ("probation", "live"):
            d["status"] = {"state": nxt, "reason": "r", "failure_class": None, "superseded_by": None}; store.put(TENANT, d, writer="test", log=log)
        d["status"] = {"state": "superseded", "reason": "corrected", "failure_class": "contradiction", "superseded_by": ids["new"]}
        store.put(TENANT, d, writer="test", log=log)
    items = [item("What caused the outage on the first?", "Cache key 1.", "notes/2026-01-01.md#L6-L6"),
             item("What caused the outage on the second?", "Cache key 2.", "notes/2026-01-02.md#L6-L6", hard=True),
             item("Which plan was chosen on the third?", "Plan 3.", "notes/2026-01-03.md#L3-L3"),
             item("Which timeout was set on the fifth?", "50 seconds.", "notes/2026-01-05.md#L8-L8"),          # never extracted
             item("Was a dashboard colour ever chosen?", "nothing relevant", None, kind="noise"),
             item("What did we stop believing on the fourth, and what replaced it?", "Vendor claim 4 was dropped.",
                  "notes/2026-01-04.md#L15-L15", kind="supersession", marker="reversal")]
    write_pool(layout, TENANT, "founding", items)
    write_pool(layout, TENANT, "heldout", [item("Held out?", "Never seen.", "notes/2026-01-05.md#L6-L6")])
    return layout, machinery, store, ids, items


def test_search_carries_status_and_never_returns_retired_or_superseded_alone(world):
    layout, machinery, store, ids, _ = world
    ex = load_entries(store, TENANT, "as_extracted", machinery.artifact.trust_rank)
    rv = load_entries(store, TENANT, "as_reviewed", machinery.artifact.trust_rank)
    assert {e.id for e in ex} == {ids["k1"], ids["k2"], ids["p3"], ids["new"], ids["con"], ids["old"]}
    assert {e.id for e in rv} == {ids["k1"], ids["k2"], ids["new"], ids["con"], ids["old"]}   # proposed is not in the reviewed view
    body, got = search_memory(ex, ["vendor claim"], max_entries=5, max_chars=4000)
    assert got == [ids["new"]] and "replaced an earlier entry that said: Vendor claim 4 was believed" in body   # superseded not returned to a current question
    body, got = search_memory(ex, ["vendor claim", "originally"], max_entries=5, max_chars=4000)
    assert got == [] or ids["old"] not in got                                          # "originally" is historical, but every term must match
    body, got = search_memory(ex, ["2026-01-04", "vendor"], max_entries=5, max_chars=4000)
    assert set(got) == {ids["new"], ids["old"]} and got[0] == ids["new"]                # a dated question sees both, current first
    assert "SUPERSEDED on 2026-01-04 by: Vendor claim 4 was dropped" in body and "not current" in body
    body, got = search_memory(ex, ["outage", "first"], max_entries=5, max_chars=4000)
    assert got == [ids["k1"], ids["con"]] and "(contested: a later run contradicted it)" in body and ids["ret"] not in body
    body, got = search_memory(ex, ["plan 3"], max_entries=5, max_chars=4000)
    assert got == [ids["p3"]] and "(proposed, unreviewed)" in body
    assert search_memory(rv, ["plan 3"], max_entries=5, max_chars=4000)[1] == []
    body, got = search_memory(ex, ["outage"], max_entries=1, max_chars=4000)
    assert len(got) == 1 and "showing 1" in body
    assert search_memory(ex, ["zzz"], max_entries=5, max_chars=4000) == ("No entry carries every term.", [])
    m = manifest(ex, "as_extracted")
    assert "6 entries" in m and "proposed 1" in m and "superseded 1" in m and "scanner (6)" in m


def test_ranking_is_matches_then_trust_then_recency_with_contested_pushed_down(world):
    layout, machinery, store, ids, _ = world
    ex = load_entries(store, TENANT, "as_extracted", machinery.artifact.trust_rank)
    _, got = search_memory(ex, ["cache key"], max_entries=5, max_chars=4000)
    # k1 and k2 tie on matches and rank; k2 is later; the contested entry is last
    assert got == [ids["k2"], ids["k1"], ids["con"]]


def test_coverage_says_whether_any_entry_holds_the_cited_span(world):
    layout, machinery, store, ids, items = world
    ex = load_entries(store, TENANT, "as_extracted", machinery.artifact.trust_rank)
    assert coverage(ex, items[0].episode) is True and coverage(ex, items[3].episode) is False and coverage(ex, None) is None
    rv = load_entries(store, TENANT, "as_reviewed", machinery.artifact.trust_rank)
    assert coverage(rv, items[2].episode) is False                                   # the plan entry is only proposed


def scripted(model, req):
    user_texts = [b.text or "" for m in req.messages for b in m.content if b.kind in ("text", "tool_result")]
    if req.job == "judge":
        u = user_texts[0]
        ref = re.search(r"Reference answer: (.*)", u).group(1); ans = re.search(r"Answer to score: (.*)", u).group(1)
        kind = re.search(r"Item type: (\w+)", u).group(1)
        ok = ans.startswith("NOTHING") if kind == "noise" else (ref.split()[-1].strip(".") in ans and not ans.startswith("NOTHING"))
        return Raw((text(json.dumps({"correct": ok, "rationale": "compared"})),), "end_turn", Usage(input=50, output=10))
    if "single word: ready" in user_texts[0]:
        return Raw((text("ready"),), "end_turn", Usage(input=0, output=1, cache_write=900))
    q = re.search(r"Question: (.*)", user_texts[0]).group(1)
    day = {"first": "first", "second": "second", "third": "third", "fourth": "fourth", "fifth": "fifth"}
    which = next((d for d in day if d in q), None)
    if req.job == "eval_service" and req.tools and not any(b.kind == "tool_result" for m in req.messages for b in m.content):
        terms = ["dashboard"] if "dashboard" in q else ["vendor claim"] if "believing" in q else ["plan"] if "plan" in q else ["outage", which]
        return Raw((ContentBlock("tool_use", tool_name="search_memory", tool_input={"terms": terms}, tool_use_id="t1"),), "tool_use", Usage(input=100, output=20))
    material = "\n".join(user_texts) + "\n" + "\n".join(b.text or "" for b in req.system)
    usage = Usage(input=200, output=12, cache_read=900 if req.job in ("eval_service_context", "eval_service_budget", "eval_service_retrieval") else 0)
    if "dashboard" in q:
        return Raw((text("NOTHING RELEVANT: no colour is recorded."),), "end_turn", usage)
    if "believing" in q:
        return Raw((text("Vendor claim 4 was dropped." if "dropped" in material else "NOTHING RELEVANT"),), "end_turn", usage)
    if "plan" in q:
        m = re.search(r"Plan (\d) was chosen", material)
        return Raw((text(f"Plan {m.group(1)}." if m else "NOTHING RELEVANT"),), "end_turn", usage)
    if "timeout" in q:
        return Raw((text("NOTHING RELEVANT"),), "end_turn", usage)
    m = re.search(rf"Cache key (\d) caused the outage on the {which}", material)
    return Raw((text(f"It was cache key {m.group(1)}." if m else "It was cache key 9."),), "end_turn", usage)


@pytest.mark.parametrize("arm", ["service", "service_context", "service_budget", "service_retrieval"])
def test_each_service_arm_scores_the_suite_with_coverage_and_counts_exposure(world, arm):
    layout, machinery, store, ids, items = world
    with run(layout, machinery, f"eval_{arm}") as log:
        rep = run_arm(layout, TENANT, machinery, FakeProvider(machinery, log, scripted), log, arm,
                      batch=False, store=store, view="as_extracted")
    recs = list(read_run(log.path, typed=False))
    verdicts = {r["item"]: r for r in recs if r["kind"] == "eval_verdict"}
    assert rep["items"] == 6 and rep["store_view"] == "as_extracted" and len(verdicts) == 6 and rep["store"]["entries"] == 6
    assert rep["rates"]["yield"] == {"k": 3, "n": 4, "rate": 0.75, "interval_95": [0.1941, 0.9937]}
    assert rep["rates"]["noise"]["k"] == 1 and rep["rates"]["supersession"]["k"] == 1
    cov = rep["coverage"]
    assert cov["yield"] == {"n": 4, "right": 3, "right_though_not_extracted": 0, "not_extracted": 1, "extracted_not_retrieved": 0, "retrieved_wrong": 0}
    assert cov["hard"]["right"] == 1 and cov["noise"] == {"n": 1, "right": 1, "misses": 0, "misses_with_entries_returned": 0, "misses_with_none_returned": 0}
    missing = next(v for v in verdicts.values() if v["extracted"] is False)
    assert missing["verdict"] is False and missing["store_view"] == "as_extracted"
    assert missing["entries_seen"] == (0 if arm == "service" else 5 if arm == "service_retrieval" else 6)   # the retrieval arm never serves the superseded entry
    if arm == "service_budget":   # the fixture's budget (2,000 tokens at 4 chars each) holds the whole six-entry tier, so every family sees all of it
        fams = rep["budget"]["families"]
        assert rep["budget"]["memory_budget_tokens"] == 2000 and set(fams) and all(f["entries"] == 6 and f["cut"] == 0 for f in fams.values())
        assert sum(f["items"] for f in fams.values()) == 6 and rep["cache"]["calls"] == 6 and rep["cache"]["hits"] == 6
        assert all(r["material"].startswith("budget:") for r in recs if r["kind"] == "retrieval_trace")
    assert rep["store"]["yield_per_100_entries"] == 1250.0
    # a service run is a candidate run: exposure moved once on every founded item, never on the held-out slice
    assert all(i.exposure == 1 for i in load(layout, TENANT, "founding", "scoring"))
    assert all(i.exposure == 0 for i in load(layout, TENANT, "heldout", "founding"))
    assert sum(r["kind"] == "exposure" and r["count_after"] == 1 and r["candidate"] == f"{arm}:as_extracted" for r in recs) == 6
    if arm == "service":
        traces = [r for r in recs if r["kind"] == "retrieval_trace"]
        assert len(traces) == 6 and all(t["calls"][0]["terms"] for t in traces)
        assert [r for r in recs if r["kind"] == "eval_report"][0]["coverage"]["yield"]["right"] == 3
    elif arm == "service_context":
        assert rep["cache"] == {"calls": 6, "hits": 6, "prefix_tokens": 900} and rep["store"]["warm_tokens"] == 900 and rep["store"]["entries"] == 6
    elif arm == "service_retrieval":
        b = rep["budget"]
        assert b["core_tokens"] == 1200 and b["slice_tokens"] == 800 and set(b["held"]) == {"core", "slice", "neither"}
        assert sum(b["held"].values()) == 5 and b["slice_tokens_total"] == 6 * 200 and rep["cache"]["hits"] == 6
        traces = [r for r in recs if r["kind"] == "retrieval_trace"]
        assert len(traces) == 6 and all(t["material"].startswith("retrieval:") for t in traces)
        assert all("scores" in t["calls"][0] and len(t["calls"][0]["scores"]) == len(t["calls"][0]["returned"]) for t in traces)


def test_the_core_and_the_slice_split_the_served_prefix(world):
    from recant.store.budget import bm25_rank, core_prefix, family_of_entry, retrieved_slice, tokens
    layout, machinery, store, ids, items = world
    serve = machinery.kernel.serve
    entries = load_entries(store, TENANT, "as_reviewed", machinery.artifact.trust_rank)
    assert tokens("Which cache-key caused the outage on 2026-01-02?") == ["cache-key", "caused", "outage", "2026-01-02"]
    core = core_prefix(entries, "ops", serve, "as_reviewed")
    states = [e.record.status.state for e in core["entries"]]
    assert "superseded" not in states and core["estimated_tokens"] <= serve.core_tokens
    # live before probation inside the family; a superseded entry never enters the core
    assert states.index("live") < states.index("probation") if "live" in states and "probation" in states else True
    ranked = bm25_rank("Which cache key caused the outage on the second?", entries)
    assert ranked and ranked[0][0] >= ranked[-1][0] and "second" in ranked[0][1].record.claim.lower()
    assert bm25_rank("", entries) == [] and bm25_rank("zzzz", entries) == []
    sl = retrieved_slice("Which cache key caused the outage on the second?", entries, {e.id for e in core["entries"]}, "ops", serve)
    assert all(e.id not in {c.id for c in core["entries"]} for e in sl["entries"]) and sl["estimated_tokens"] <= serve.slice_tokens
    # the slice is tier-wide: an entry of another family reaches it when the question names it
    other = next(e for e in entries if family_of_entry(e, serve.scopes) == "product" and not hasattr(e.record, "opened")
                 and e.record.status.state != "superseded")
    wide = retrieved_slice(other.record.claim, entries, set(), "ops", serve)
    assert other.id in {e.id for e in wide["entries"]}
    assert all(e.record.status.state != "superseded" for e in sl["entries"] if not hasattr(e.record, "opened"))
    tiny = serve.model_copy(update={"core_tokens": 0, "slice_tokens": 400, "memory_budget_tokens": 2000})
    empty_core = core_prefix(entries, "ops", tiny, "as_reviewed")
    assert empty_core["entries"] == []
    sl2 = retrieved_slice("Which cache key caused the outage on the second?", entries, set(), "ops", tiny)
    assert 0 < len(sl2["entries"]) and sl2["estimated_tokens"] <= 400 and sl2["scores"] == sorted(sl2["scores"], reverse=True)


def test_the_budget_fitter_ranks_scope_then_tier_then_recency_and_cuts(world):
    from recant.store.budget import families_of_items, family_of, family_of_channel, family_of_entry, fit_prefix
    layout, machinery, store, ids, items = world
    serve = machinery.kernel.serve
    entries = load_entries(store, TENANT, "as_extracted", machinery.artifact.trust_rank)
    assert family_of("Which cache key caused the outage?", serve.scopes) == "ops" and family_of("nothing here", serve.scopes) is None
    assert family_of_channel("#OPS", serve.scopes) == "ops" and family_of_channel("#other", serve.scopes) is None
    fams = {e.id: family_of_entry(e, serve.scopes) for e in entries}
    assert set(fams.values()) >= {"ops", "product"}
    full = fit_prefix(entries, "ops", serve, "as_extracted")
    assert full["cut"] == 0 and len(full["entries"]) == len(entries) and full["estimated_tokens"] <= serve.memory_budget_tokens
    # ops entries come first for an ops session, product entries first for a product session; the rest follow
    ops_first = [fams[e.id] for e in full["entries"]]
    assert ops_first[:ops_first.count("ops")] == ["ops"] * ops_first.count("ops")
    prod = fit_prefix(entries, "product", serve, "as_extracted")["entries"]
    assert fams[prod[0].id] == "product"
    # inside a family and a tier: a later day first
    from recant.store.budget import tier_rank
    from itertools import groupby
    ops = [e for e in full["entries"] if fams[e.id] == "ops"]
    assert [tier_rank(e) for e in ops] == sorted(tier_rank(e) for e in ops)
    for _, grp in groupby(ops, key=tier_rank):
        days = [e.day for e in grp]
        assert days == sorted(days, reverse=True)
    # a tight budget cuts, in rank order, and the cut is counted
    tight = serve.model_copy(update={"memory_budget_tokens": 1000 // 3})
    cut = fit_prefix(entries, "ops", tight, "as_extracted")
    assert 0 < len(cut["entries"]) < len(entries) and cut["cut"] == len(entries) - len(cut["entries"])
    assert cut["estimated_tokens"] <= 1000 // 3 and [e.id for e in cut["entries"]] == [e.id for e in full["entries"]][:len(cut["entries"])]
    groups = families_of_items(items, serve.scopes)
    assert sum(len(v) for v in groups.values()) == len(items) and "ops" in groups
    from recant.store.budget import family_of_question
    assert family_of("What is the manner rule?", serve.scopes) == "manner" and family_of_question("What is the manner rule?", serve.scopes) is None


def test_the_budgeted_arm_counts_what_the_budget_cut_as_extracted_but_not_retrieved(world):
    layout, machinery, store, ids, items = world
    tight = machinery.kernel.model_copy(update={"serve": machinery.kernel.serve.model_copy(update={"memory_budget_tokens": 1000 // 3})})
    from dataclasses import replace
    m = replace(machinery, kernel=tight)
    with run(layout, m, "eval_service_budget") as log:
        rep = run_arm(layout, TENANT, m, FakeProvider(m, log, scripted), log, "service_budget",
                      batch=False, store=store, view="as_extracted")
    fams = rep["budget"]["families"]
    assert rep["budget"]["memory_budget_tokens"] == 333 and any(f["cut"] > 0 for f in fams.values())
    cov = rep["coverage"]["yield"]
    assert cov["n"] == 4 and cov["extracted_not_retrieved"] + cov["right"] + cov["not_extracted"] + cov["retrieved_wrong"] + 0 >= 4
    assert cov["not_extracted"] == 1                                     # the tier's own gap is unchanged by the budget


def test_the_reviewed_view_hides_the_proposed_entry_and_the_reader_misses_it(world):
    layout, machinery, store, ids, items = world
    with run(layout, machinery, "eval_service") as log:
        rep = run_arm(layout, TENANT, machinery, FakeProvider(machinery, log, scripted), log, "service",
                      batch=False, store=store, view="as_reviewed")
    assert rep["coverage"]["yield"]["not_extracted"] == 2 and rep["rates"]["yield"]["k"] == 2 and rep["store"]["entries"] == 5


def test_service_context_aborts_on_a_cache_miss(world):
    layout, machinery, store, ids, items = world

    def cold(model, req):
        r = scripted(model, req)
        return Raw(r.content, r.stop_reason, Usage(input=900, output=5, cache_read=0)) if req.job == "eval_service_context" else r
    with pytest.raises(CacheMiss):
        with run(layout, machinery, "eval_service_context") as log:
            run_arm(layout, TENANT, machinery, FakeProvider(machinery, log, cold), log, "service_context",
                    batch=False, store=store)


def test_a_service_arm_needs_the_store_and_the_baselines_never_touch_exposure(world):
    layout, machinery, store, ids, items = world
    with run(layout, machinery, "eval_service") as log:
        with pytest.raises(ValueError, match="needs the memory store"):
            run_arm(layout, TENANT, machinery, FakeProvider(machinery, log, scripted), log, "service", batch=False)
    with run(layout, machinery, "eval_oracle") as log:
        run_arm(layout, TENANT, machinery, FakeProvider(machinery, log, scripted), log, "oracle", batch=False)
    assert all(i.exposure == 0 for i in load(layout, TENANT, "founding", "scoring"))


def test_coverage_split_arithmetic():
    rows = [{"type": "yield", "hard": False, "verdict": True, "unscorable": False, "extracted": True, "retrieved": True, "entries_seen": 2},
            {"type": "yield", "hard": True, "verdict": False, "unscorable": False, "extracted": True, "retrieved": False, "entries_seen": 1},
            {"type": "yield", "hard": False, "verdict": False, "unscorable": False, "extracted": False, "retrieved": False, "entries_seen": 0},
            {"type": "yield", "hard": False, "verdict": False, "unscorable": False, "extracted": True, "retrieved": True, "entries_seen": 3},
            {"type": "noise", "hard": False, "verdict": False, "unscorable": False, "extracted": None, "retrieved": None, "entries_seen": 2},
            {"type": "noise", "hard": True, "verdict": False, "unscorable": False, "extracted": None, "retrieved": None, "entries_seen": 0},
            {"type": "noise", "hard": False, "verdict": True, "unscorable": True, "extracted": None, "retrieved": None, "entries_seen": 0}]
    c = coverage_split(rows)
    assert c["yield"] == {"n": 4, "right": 1, "right_though_not_extracted": 0, "not_extracted": 1, "extracted_not_retrieved": 1, "retrieved_wrong": 1}
    assert c["hard"]["extracted_not_retrieved"] == 1
    assert c["noise"] == {"n": 2, "right": 0, "misses": 2, "misses_with_entries_returned": 1, "misses_with_none_returned": 1}


def test_the_memory_queue_puts_contested_entries_first(world):
    layout, machinery, store, ids, items = world
    with run(layout, machinery, "seed") as log:
        a = entry(store, log, "Owner asked for the scanner to retry twice.", "notes/2026-01-05.md#L3-L3", kind="directive", day="2026-01-05")
        b = entry(store, log, "The scanner retries twice.", "notes/2026-01-05.md#L8-L8", kind="directive", day="2026-01-05", derived=[a])
        rec = make_record(claim="Reported without a quote.", actor={"class": "agent", "id": "agent (reporting owner)"},
                          anchor={"quote": "q", "episode": "notes/2026-01-06.md#L3-L3", "channel": None, "message_id": None, "timestamp": "2026-01-06"})
        store.put(TENANT, rec, writer="consolidation", log=log)
    q = memory_queue(store, TENANT, principal_id="owner")
    groups = [r["group"] for r in q]
    assert groups[:2] == ["paired", "paired"] and {q[0]["id"], q[1]["id"]} == {a, b}
    assert groups[2] == "class_overridden" and q[2]["id"] == rec["id"]
    assert groups[3:] == ["rest"] * (len(q) - 3)
    states = [r["state"] for r in q[3:]]
    assert states == sorted(states, key=lambda st: 0 if st == "proposed" else 1)              # undecided first
    for st in ("proposed", "probation"):
        days = [r["day"] for r in q[3:] if r["state"] == st]
        assert days == sorted(days)                                                            # then by day
    # a rank-3 directive whose claim names the principal as deciding sits in the third group
    with run(layout, machinery, "seed") as log:
        c = entry(store, log, "Owner decided the scanner must not retry.", "notes/2026-01-06.md#L8-L8", kind="directive", day="2026-01-06")
    q = memory_queue(store, TENANT, principal_id="owner")
    assert next(r for r in q if r["id"] == c)["group"] == "names_principal" and [r["group"] for r in q][3] == "names_principal"


def test_a_reviewer_may_supersede_and_a_duplicate_folds_into_its_survivor(world):
    from recant.records import RecordError
    from recant.store.review import decide_memory, fold_duplicate, successors
    layout, machinery, store, ids, _ = world
    with run(layout, machinery, "review") as log:
        R = {"kind": "model", "id": "reviewer"}
        cands = successors(store, TENANT, ids["k1"])
        assert cands and all(c["day"] >= "2026-01-01" for c in cands) and cands[0]["same_scope"]
        with pytest.raises(RecordError, match="names the successor"):
            decide_memory(store, TENANT, log, ids["k2"], to_state="superseded", reason="staleness", reviewer=R, reasons=["staleness"])
        rec = decide_memory(store, TENANT, log, ids["k2"], to_state="superseded", reason="staleness", reviewer=R, reasons=["staleness"],
                            superseded_by=ids["k1"], note="k1 restates it")
        assert rec["outcome"] == "supersede" and store.get(TENANT, ids["k2"])[0].status.superseded_by == ids["k1"]
        # the reclassification of a retired entry is the store's own move, logged as such; retired stays terminal otherwise
        with pytest.raises(RecordError, match="only a retired entry"):
            store.reclassify_retired_as_superseded(TENANT, ids["k1"], superseded_by=ids["new"], reason="r", failure_class="staleness", writer="reclassification", log=log)
        store.reclassify_retired_as_superseded(TENANT, ids["ret"], superseded_by=ids["k1"], reason="the note named it", failure_class="staleness",
                                               writer="reclassification", log=log)
        assert store.get(TENANT, ids["ret"])[0].status.state == "superseded"
        # a duplicate's anchor joins the survivor; the duplicate stays retired
        dup = entry(store, log, "Cache key 1 caused the first outage.", "notes/2026-01-01.md#L9-L9", state="retired")
        fold_duplicate(store, TENANT, log, dup, ids["k1"], reviewer=R)
        k1 = store.get(TENANT, ids["k1"])[0]
        assert "notes/2026-01-01.md#L9-L9" in k1.derived_from and store.get(TENANT, dup)[0].status.state == "retired"
        with pytest.raises(RecordError, match="not retired"):
            fold_duplicate(store, TENANT, log, ids["k1"], ids["new"], reviewer=R)
        # a retired entry that should still stand is reinstated to probation: the store's second logged move
        with pytest.raises(RecordError, match="only a retired or superseded entry"):
            store.reinstate_retired(TENANT, ids["k1"], reason="r", writer="reclassification", log=log)
        with pytest.raises(RecordError, match="carries a reason"):
            store.reinstate_retired(TENANT, dup, reason=" ", writer="reclassification", log=log)
        with pytest.raises(RecordError, match="probation or live"):
            store.reinstate(TENANT, dup, to_state="contested", reason="r", writer="reclassification", log=log)
        back = store.reinstate_retired(TENANT, dup, reason="Reinstated: reviewer error.", writer="reclassification", log=log)
        assert back.status.state == "probation" and back.status.failure_class is None and back.status.superseded_by is None
        assert store.get(TENANT, dup)[0].status.reason == "Reinstated: reviewer error."
        with pytest.raises(RecordError, match="not retired"):                      # it no longer folds as a duplicate
            fold_duplicate(store, TENANT, log, dup, ids["k1"], reviewer=R)
    moves = [r for r in read_run(log.path, typed=False) if r["kind"] == "status_transition"]
    assert any(m["from"] == "retired" and m["to"] == "superseded" and m["writer"] == "reclassification" for m in moves)
    assert any(m["from"] == "retired" and m["to"] == "probation" and m["writer"] == "reclassification" for m in moves)
    # a superseded entry folds its anchor into its successor, and one whose successor is gone may come back as live
    with run(layout, machinery, "review") as log:
        R = {"kind": "model", "id": "reviewer"}
        fold_duplicate(store, TENANT, log, ids["k2"], ids["k1"], reviewer=R, from_states=("superseded",))
        assert store.get(TENANT, ids["k2"])[0].anchor.episode in store.get(TENANT, ids["k1"])[0].derived_from
        with pytest.raises(RecordError, match="not retired"):
            fold_duplicate(store, TENANT, log, ids["k2"], ids["k1"], reviewer=R)
        live = store.reinstate(TENANT, ids["k2"], to_state="live", reason="no successor stands", writer="reclassification", log=log)
        assert live.status.state == "live" and live.status.superseded_by is None
    ex = load_entries(store, TENANT, "as_reviewed", machinery.artifact.trust_rank)
    assert coverage(ex, "notes/2026-01-01.md#L9-L9") is True                          # the folded anchor now covers the item


def test_the_held_out_slice_opens_as_an_overfitting_check_and_counts_its_use(world):
    layout, machinery, store, ids, items = world
    with run(layout, machinery, "eval_service") as log:
        rep = run_arm(layout, TENANT, machinery, FakeProvider(machinery, log, scripted), log, "service",
                      batch=False, store=store, pool="heldout", purpose="overfitting_check")
    assert rep["pool"] == "heldout" and rep["items"] == 1
    assert [i.exposure for i in load(layout, TENANT, "heldout", "founding")] == [1]           # once, on the slice
    assert all(i.exposure == 0 for i in load(layout, TENANT, "founding", "scoring"))         # not on the suite
    assert [r for r in read_run(log.path, typed=False) if r["kind"] == "eval_report"][0]["pool"] == "heldout"
    with run(layout, machinery, "eval_grep") as log:
        run_arm(layout, TENANT, machinery, FakeProvider(machinery, log, scripted), log, "grep", batch=False,
                pool="heldout", purpose="overfitting_check")
    with pytest.raises(PermissionError):                                                    # the guard is still the guard
        run_arm(layout, TENANT, machinery, FakeProvider(machinery, log, scripted), log, "grep", batch=False, pool="heldout")
    assert [i.exposure for i in load(layout, TENANT, "heldout", "founding")] == [1]           # a baseline never counts


def test_a_decision_whose_log_record_does_not_validate_changes_nothing(world):
    """A reviewer kind the log refuses made decide_memory write the entry and then fail to log it.
    The record is now checked first: nothing is written, nothing is logged, no sequence number is spent."""
    from pydantic import ValidationError
    from recant.log import read_run
    from recant.store.review import decide_memory, decide_task
    layout, machinery, store, ids, _ = world
    before, before_notes = store.get(TENANT, ids["k2"])
    with run(layout, machinery, "review") as log:
        with pytest.raises(ValidationError):
            decide_memory(store, TENANT, log, ids["k2"], to_state="retired", reason="staleness",
                          reviewer={"kind": "ai", "id": "review-bot"}, reasons=["staleness"])
        after, after_notes = store.get(TENANT, ids["k2"])
        assert after.to_dict() == before.to_dict() and after_notes == before_notes
        ok = decide_memory(store, TENANT, log, ids["k2"], to_state="retired", reason="staleness",
                           reviewer={"kind": "model", "id": "review-bot"}, reasons=["staleness"])
    recs = [r for r in read_run(log.path, typed=False) if r["kind"] == "review_decision"]
    assert len(recs) == 1 and recs[0]["reviewer"]["kind"] == "model" and ok["seq"] == recs[0]["seq"]
    seqs = [r["seq"] for r in read_run(log.path, typed=False)]
    assert seqs == list(range(1, len(seqs) + 1))                                              # no gap where the bad record was
