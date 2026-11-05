"""The task ledger: its store and lifecycle, the extractor that routes task moments to it
and closes what a later line completes, its place in the warm-tier search, and coverage over both stores."""
import json
import re

import pytest
from ulid import ULID

from recant.harness.extract import TaskExtractor, task_spans_from_log
from recant.harness.nominate import Span
from recant.log import LogWriter, read_run
from recant.provider import Usage, text
from recant.provider.base import Raw
from recant.provider.fake import FakeProvider
from recant.records import RecordError, parse_task_file, render_task_file, validate_task
from recant.store import MemoryStore, RejectedWrite, TaskStore, ingest
from recant.store.review import decide_task, task_queue
from recant.store.search import coverage, load_entries, load_tasks, manifest, search_memory
from pipeline_fixtures import build_corpus
from recant_fixtures import ALLOWLIST, TENANT

PRINCIPAL = "owner"


def run(layout, machinery, job, **params):
    return LogWriter(layout, TENANT, job, artifact_hash=machinery.artifact_hash, kernel_hash=machinery.kernel_hash, params=params)


def task(**over):
    t = {"id": str(ULID()), "claim": "Move the three billing reports to the new client.", "scope": "dashboard",
         "actor": {"class": "principal", "id": PRINCIPAL},
         "anchor": {"quote": "port them", "episode": "notes/2026-01-02.md#L3-L3", "channel": None, "message_id": None, "timestamp": "2026-01-02"},
         "opened": "2026-01-02", "status": "open", "closed": None, "closed_by": None, "superseded_by": None, "derived_from": []}
    t.update(over)
    return t


def test_the_task_record_is_closed_and_its_lifecycle_is_not_memorys():
    t = validate_task(task())
    assert t.status == "open" and parse_task_file(render_task_file(t, "n"))[0] == t
    with pytest.raises(RecordError, match="date it closed"):
        validate_task(task(status="done"))
    with pytest.raises(RecordError, match="open task has no"):
        validate_task(task(closed="2026-01-03"))
    with pytest.raises(RecordError, match="on or after"):
        validate_task(task(status="done", closed="2026-01-01"))
    with pytest.raises(RecordError, match="superseded_by"):
        validate_task(task(status="superseded", closed="2026-01-03"))
    with pytest.raises(RecordError):
        validate_task(task(retired=True))                                    # no field the schema does not name


def test_the_task_store_writes_versions_and_refuses_bad_transitions(layout, machinery, log):
    store = TaskStore(layout, ALLOWLIST)
    t = task()
    store.put(TENANT, t, writer="consolidation", log=log)
    assert store.list(TENANT) == [{"id": t["id"], "status": "open", "opened": "2026-01-02"}]
    done = {**t, "status": "done", "closed": "2026-01-04", "closed_by": "notes/2026-01-04.md#L3-L3"}
    store.put(TENANT, done, writer="review", log=log)
    assert store.get(TENANT, t["id"])[0].status == "done" and len(list(layout.versions_dir(TENANT, t["id"]).glob("*.md"))) == 1
    with pytest.raises(RejectedWrite, match="no task transition done -> open"):
        store.put(TENANT, t, writer="review", log=log)
    with pytest.raises(RejectedWrite, match="enters as open"):
        store.put(TENANT, task(status="dropped", closed="2026-01-02"), writer="consolidation", log=log)
    with pytest.raises(RejectedWrite, match="outside the allowlist"):
        store.put(TENANT, task(anchor={**t["anchor"], "episode": "secrets/x.md#L1-L1"}), writer="consolidation", log=log)
    recs = list(read_run(log.path, typed=False)) if False else None
    log.finish()
    recs = list(read_run(log.path, typed=False))
    moves = [r for r in recs if r["kind"] == "task_transition"]
    assert [(m["from"], m["to"]) for m in moves] == [(None, "open"), ("open", "done")] and moves[1]["closed_by"] == done["closed_by"]
    assert sum(r["kind"] == "write_rejected" for r in recs) == 3


def test_tasks_search_beside_memory_open_first_closed_with_dates_stale_when_old(layout, machinery, log):
    store, tasks = MemoryStore(layout, ALLOWLIST), TaskStore(layout, ALLOWLIST)
    a = task(claim="Move the three billing reports to the new client.")
    b = task(id=str(ULID()), claim="Port the login page to the new client.", opened="2026-03-01",
             anchor={**a["anchor"], "episode": "notes/2026-03-01.md#L3-L3", "timestamp": "2026-03-01"})
    c = task(id=str(ULID()), claim="Port the admin panel to the new client.", status="done", closed="2026-01-05", opened="2026-01-03",
             anchor={**a["anchor"], "episode": "notes/2026-01-03.md#L3-L3", "timestamp": "2026-01-03"})
    for t in (a, b, c):
        tasks.put(TENANT, t, writer="consolidation", log=log)
    entries = load_tasks(tasks, TENANT, machinery.artifact.trust_rank, stale_after_days=30)
    by = {e.id: e for e in entries}
    assert by[a["id"]].stale is True and by[b["id"]].stale is False and by[c["id"]].stale is False   # today is the ledger's last day, 03-01
    body, ids = search_memory(entries, ["new client"], max_entries=5, max_chars=4000)
    assert ids[:2] == [b["id"], a["id"]] and ids[2] == c["id"]                       # open first (later first), then closed
    assert "open, unconfirmed since 2026-01-02: no closing line" in body and "task, done 2026-01-05" in body
    mem = load_entries(store, TENANT, "as_extracted", machinery.artifact.trust_rank)
    m = manifest(mem + entries, "as_extracted")
    assert "task ledger: 3 tasks (2 open, 1 stale)" in m
    assert coverage(mem, "notes/2026-01-03.md#L3-L3") is False and coverage(mem + entries, "notes/2026-01-03.md#L3-L3") is True
    q = task_queue(tasks, TENANT)
    assert [r["status"] for r in q] == ["open", "open", "done"] and q[0]["id"] == a["id"]
    decide_task(tasks, TENANT, log, a["id"], to_state="done", reviewer={"kind": "human", "id": "r"})
    assert tasks.get(TENANT, a["id"])[0].closed == "2026-01-02"                          # today unless the entry carries a date: it opened then
    with pytest.raises(RecordError):
        decide_task(tasks, TENANT, log, c["id"], to_state="dropped", reviewer={"kind": "human", "id": "r"})


def _raw(payload):
    return Raw((text(json.dumps(payload)),), "end_turn", Usage(input=300, output=80))


def _task(**over):
    base = {"verdict": "task", "not_a_task_reason": "none", "claim": "", "scope": "scanner", "actor_class": "agent",
            "actor_id": "agent", "quote": "", "quote_start_line": 0, "quote_end_line": 0, "status": "open", "closed_line": 0,
            "channel": "", "timestamp": "", "notes": ""}
    base.update(over)
    return _raw(base)


def tasking(model, req):
    user = req.messages[0].content[0].text
    start = int(re.search(r"The moment is lines (\d+)-", user).group(1))
    line = re.search(rf"^{start}: (.*)$", user, flags=re.M).group(1)
    day = re.search(r"Day: (\S+)", user).group(1)
    if "plan" in line:
        n = re.search(r"plan (\d)", line).group(1)
        return _task(claim=f"Roll out plan {n} for the scanner.", quote=line, quote_start_line=start, quote_end_line=start,
                     actor_class="principal", actor_id=PRINCIPAL)
    if "timeout" in line:
        if day.endswith("-03"):
            return _task(verdict="not_a_task", not_a_task_reason="is_memory")
        # the timeout change is the completion of the plan roll-out on the same day: closes it
        return _task(claim=f"Roll out plan {day[-1]} for the scanner: done, timeout set.", quote=line, quote_start_line=start,
                     quote_end_line=start, status="done", closed_line=start)
    if "CAUSE" in line:
        return _task(claim="Investigate the outage.", quote="not the line", quote_start_line=start, quote_end_line=start)
    return _task(verdict="not_a_task", not_a_task_reason="protocol_exhaust")


@pytest.fixture
def world(layout, machinery, tmp_path):
    ingest(layout, TENANT, build_corpus(tmp_path / "corpus"), ALLOWLIST)
    spans = []
    for d in ("2026-01-01", "2026-01-02", "2026-01-03"):
        p = f"notes/{d}.md"
        spans += [Span("rule_or_decision", f"{p}#L3-L3", p, 3, 3, ""), Span("root_cause", f"{p}#L6-L6", p, 6, 6, ""),
                  Span("rule_or_decision", f"{p}#L8-L8", p, 8, 8, ""), Span("reversal", f"{p}#L15-L15", p, 15, 15, "")]
    return layout, machinery, TaskStore(layout, ALLOWLIST), spans


def test_the_task_extractor_writes_open_tasks_and_closes_them_from_a_later_line(world):
    layout, machinery, tasks, spans = world
    with run(layout, machinery, "extract_tasks", extractor_fingerprint=machinery.extractor_fingerprint("tasks")) as log:
        ex = TaskExtractor(layout, TENANT, machinery, tasks, FakeProvider(machinery, log, tasking), log, principal_id=PRINCIPAL)
        out = ex.run(spans)
    recs = list(read_run(log.path, typed=False))
    ext = [r for r in recs if r["kind"] == "extraction"]
    assert all(r["store"] == "tasks" for r in ext) and out["calls"] == 12
    # three plan tasks written open; two of them closed by the same-day timeout line; day 3's timeout was a memory
    assert out["written"] == 3 and out["closed"] == 2
    ledger = {t["id"]: tasks.get(TENANT, t["id"])[0] for t in tasks.list(TENANT)}
    assert sorted(t.status for t in ledger.values()) == ["done", "done", "open"]
    done = next(t for t in ledger.values() if t.status == "done" and t.opened == "2026-01-01")
    assert done.closed == "2026-01-01" and done.closed_by == "notes/2026-01-01.md#L8-L8" and done.actor.cls == "agent"   # a daily does not quote
    reasons = [r["rejection_reason"] for r in ext if r["verdict"] == "rejected"]
    assert reasons.count("quote_not_verbatim") == 3 and reasons.count("not_a_task: protocol_exhaust") == 3 and reasons.count("not_a_task: is_memory") == 1
    end = [r for r in recs if r["kind"] == "run_finished"][0]
    assert end["counts"]["tasks_closed"] == 2 and end["counts"]["written: open"] == 3 and end["counts"]["actor_class_overridden"] == 3
    # the same pass again pays for nothing
    with run(layout, machinery, "extract_tasks", extractor_fingerprint=machinery.extractor_fingerprint("tasks")) as log2:
        again = TaskExtractor(layout, TENANT, machinery, tasks, FakeProvider(machinery, log2, tasking), log2, principal_id=PRINCIPAL).run(spans)
    assert again["calls"] == 0


def test_task_spans_are_rebuilt_from_the_first_pass_log(world):
    layout, machinery, tasks, spans = world
    fp = machinery.extractor_fingerprint("first_pass")
    with run(layout, machinery, "nominate") as nlog:
        nlog.emit("nomination", episode="notes/2026-01-01.md#L1-L19", model_id="m", prompt_version="n1",
                  markers=[{"marker": "directive_to_act", "span": "notes/2026-01-01.md#L3-L3", "quote": "q"}])
    with run(layout, machinery, "extract_first_pass", extractor_fingerprint=fp) as elog:
        elog.emit("extraction", episode="notes/2026-01-01.md#L3-L3", candidate={}, spans=[], model_id="m", prompt_version="e1",
                  verdict="rejected", rejection_reason="not_a_memory: task")
        elog.emit("extraction", episode="notes/2026-01-01.md#L6-L6", candidate={}, spans=[], model_id="m", prompt_version="e1",
                  verdict="rejected", rejection_reason="not_a_memory: restates")
    got = task_spans_from_log(layout, TENANT, fp)
    assert [(s.episode, s.marker) for s in got] == [("notes/2026-01-01.md#L3-L3", "directive_to_act")]
    assert task_spans_from_log(layout, TENANT, "sha256:other") == []
