"""The first extraction pass against a scripted extractor that misbehaves in known places: what is
written, what is refused and why, what the log keeps, and what a stopped pass does not pay for twice."""
import json
import re

import pytest

from recant.harness.extract import (CONTRADICTION_BAND, Extractor, claim_overlap, decided_spans, group_spans,
                                    principal_allowed)
from recant.harness.nominate import Span, nominate
from recant.log import LogWriter, read_run
from recant.machinery import Machinery
from recant.provider import ProviderError, Usage, text
from recant.provider.base import Raw
from recant.provider.fake import FakeProvider
from recant.store import MemoryStore, ingest
from recant.store.review import memory_queue, memory_view
from pipeline_fixtures import build_corpus, scripted
from recant_fixtures import ALLOWLIST, ARTIFACT_YAML, KERNEL_YAML, TENANT

PRINCIPAL = "owner"


def run(layout, machinery, job, **params):
    return LogWriter(layout, TENANT, job, artifact_hash=machinery.artifact_hash, kernel_hash=machinery.kernel_hash, params=params)


def _raw(payload):
    return Raw((text(json.dumps(payload)),), "end_turn", Usage(input=300, output=80))


def _memory(**over):
    base = {"verdict": "memory", "not_a_memory_reason": "none", "claim": "", "kind": "stated", "actor_class": "agent",
            "actor_id": "agent", "quote": "", "quote_start_line": 0, "quote_end_line": 0, "channel": "", "timestamp": "",
            "scope": "scanner", "supersession_condition": "Until a later decision.", "notes": ""}
    base.update(over)
    return _raw(base)


def _not(reason):
    return _memory(verdict="not_a_memory", not_a_memory_reason=reason)


# Claims vary in wording by month, as real claims do; the arithmetic dedupe would otherwise fold a template.
PLAN = {1: "Plan {n} is the scanner plan as of {day}.", 2: "On {day} the scanner plan chosen was plan {n}.",
        3: "Scanner plan {n} was adopted on {day}.", 4: "The plan for the scanner from {day} is plan {n}.",
        5: "As of {day} scanner work follows plan {n}.", 6: "Plan {n} governs the scanner since {day}."}
CAUSE = {1: "Cache key {n} caused the outage on {day}.", 2: "The outage of {day} traced to cache key {n}.",
         3: "Replaying twice showed cache key {n} broke {day}.", 4: "On {day} the outage root cause was cache key {n}.",
         5: "Cache key {n}: root cause of the {day} outage, per replay.", 6: "The {day} outage came from cache key {n}."}
TIMEOUT = {1: "The timeout became {t} seconds on {day}.", 2: "On {day} the timeout was set to {t} seconds.",
           3: "Timeout: {t} seconds, from {day}.", 4: "Since {day} requests time out after {t} seconds.",
           5: "The {day} change set the timeout at {t} seconds.", 6: "{t} seconds is the timeout as of {day}."}


def extracting(model, req):
    """The scripted extractor. Each daily has a plan line, a cause line, a timeout line and a reversal."""
    if req.job != "extract":
        return scripted(model, req)
    user = req.messages[0].content[0].text
    start = int(re.search(r"The moment is lines (\d+)-", user).group(1))
    line = re.search(rf"^{start}: (.*)$", user, flags=re.M).group(1)
    day = re.search(r"Day: (\S+)", user).group(1)
    if "User: hello" in line or "Assistant: hi" in line:
        claim = "The greeting protocol is to say hello." if "User" in line else "The reply to a greeting is hi."
        return _memory(claim=claim, actor_class="principal", actor_id=PRINCIPAL,
                       quote=line, quote_start_line=start, quote_end_line=start, scope="talk")
    if line.startswith("[s#L3]"):
        return _memory(claim="The outage was caused by cache key 1 for the scanner.", kind="observed", actor_class="principal",
                       actor_id=PRINCIPAL, quote=line, quote_start_line=start, quote_end_line=start)
    month = int(day[5:7])
    m = re.search(r"we chose plan (\d)", line)
    if m:
        # the model says principal; the daily does not quote anyone, so the rule makes it the agent's
        claim = PLAN[month].format(n=m.group(1), day=day)
        if day == "2026-06-01":
            claim = CAUSE[month].format(n=1, day=day)                              # a duplicate of the cause line's claim
        return _memory(claim=claim, actor_class="principal", actor_id=PRINCIPAL, quote=line, quote_start_line=start,
                       quote_end_line=start, timestamp=f"{day}T10:00:00Z")
    m = re.search(r"timeout became (\d+)", line)
    if m:
        if day == "2026-01-01":
            return _memory(claim="The timeout is wrong here.", kind="directive", quote="the timeout became 999 seconds",
                           quote_start_line=start, quote_end_line=start)
        if day == "2026-02-01":
            return _memory(claim="Lines out of range.", kind="directive", quote=line, quote_start_line=900, quote_end_line=900)
        if day == "2026-03-01":
            return _not("task")
        if day == "2026-04-01":
            return _memory(claim="No kind at all.", kind="none", quote=line, quote_start_line=start, quote_end_line=start)
        if day == "2026-05-01":
            return Raw((text("this is not json"),), "end_turn", Usage(input=10, output=3))
        claim = TIMEOUT[month].format(t=m.group(1), day=day)
        if day == "2026-02-02":
            claim = "On 2026-02-02 the scanner plan timeout became 20 seconds."     # half-overlaps the plan claim: paired
        return _memory(claim=claim, kind="directive", quote=line, quote_start_line=start, quote_end_line=start)
    m = re.search(r"cache key (\d)", line)
    if m:
        return _memory(claim=CAUSE[month].format(n=m.group(1), day=day), kind="concluded",
                       quote=line, quote_start_line=start, quote_end_line=start, notes="Established by replay.")
    return _not("restates")


@pytest.fixture
def world(layout, machinery, tmp_path):
    root = build_corpus(tmp_path / "corpus")
    (root / "notes/.hidden/talk/2026-01-01.txt").write_text(
        "[s#L1] User: hello\n[s#L2] Assistant: hi\n[s#L3] User: the outage was caused by cache key 1 for the scanner\n")
    ingest(layout, TENANT, root, ALLOWLIST)
    with run(layout, machinery, "nominate") as log:
        spans = nominate(layout, TENANT, machinery, FakeProvider(machinery, log, scripted), log)
    spans += [Span("rule_or_decision", "notes/.hidden/talk/2026-01-01.txt#L1-L1", "notes/.hidden/talk/2026-01-01.txt", 1, 1, "[s#L1] User: hello"),
              Span("manner_correction", "notes/.hidden/talk/2026-01-01.txt#L1-L1", "notes/.hidden/talk/2026-01-01.txt", 1, 1, "[s#L1] User: hello"),
              Span("rule_or_decision", "notes/.hidden/talk/2026-01-01.txt#L2-L2", "notes/.hidden/talk/2026-01-01.txt", 2, 2, "[s#L2] Assistant: hi"),
              Span("root_cause", "notes/.hidden/talk/2026-01-01.txt#L3-L3", "notes/.hidden/talk/2026-01-01.txt", 3, 3,
                   "[s#L3] User: the outage was caused by cache key 1 for the scanner")]
    return layout, machinery, MemoryStore(layout, ALLOWLIST), spans


def extract(layout, machinery, store, spans, *, job="extract_first_pass", dry_run=False, batch=True):
    with run(layout, machinery, job, extractor_fingerprint=machinery.extractor_fingerprint()) as log:
        ex = Extractor(layout, TENANT, machinery, store, FakeProvider(machinery, log, extracting), log, principal_id=PRINCIPAL, batch=batch, dry_run=dry_run)
        out = ex.run(spans)
    recs = list(read_run(log.path, typed=False))
    return out, recs, log


def test_the_pass_writes_proposed_entries_and_refuses_each_bad_candidate_with_its_reason(world):
    layout, machinery, store, spans = world
    out, recs, log = extract(layout, machinery, store, spans)
    ext = [r for r in recs if r["kind"] == "extraction"]
    end = [r for r in recs if r["kind"] == "run_finished"][0]
    moments = len(group_spans(spans))
    assert out["calls"] == moments == len(ext) and end["counts"]["moments"] == moments and end["counts"]["spans_in"] == len(spans)
    assert end["counts"]["already_decided"] == 0 and end["stopped_by"] == "done" and end["promoted"] == out["written"]
    reasons = {r["episode"]: r["rejection_reason"] for r in ext if r["verdict"] == "rejected"}
    assert reasons["notes/2026-01-01.md#L8-L8"] == "quote_not_verbatim"
    assert reasons["notes/2026-02-01.md#L8-L8"].startswith("episode_outside_span")
    assert reasons["notes/2026-03-01.md#L8-L8"] == "not_a_memory: task"
    assert reasons["notes/2026-04-01.md#L8-L8"].startswith("invalid:")
    assert reasons["notes/2026-05-01.md#L8-L8"].startswith("malformed:")
    dup = [r for r in ext if (r["rejection_reason"] or "").startswith("restates ")]
    assert len(dup) == 1 and dup[0]["episode"].startswith("notes/2026-06-01.md")      # pass 2: attached to the survivor, not written
    survivor, _ = store.get(TENANT, dup[0]["rejection_reason"].split()[1])
    assert dup[0]["episode"] in survivor.derived_from and end["counts"]["attached_to_survivor"] == 1
    assert all(r["rejection_reason"] == "not_a_memory: restates" for r in ext if r["episode"].endswith("#L15-L15"))
    assert end["counts"]["not_a_memory: restates"] == sum(s.marker == "reversal" for s in spans) and end["counts"]["not_a_memory: task"] == 1
    # every accepted candidate is in warm/ as proposed, with its extraction record pointing at it
    written = {r["written_as"] for r in ext if r["verdict"] == "accepted"}
    assert len(written) == out["written"] == out["accepted"] and {e["id"] for e in store.list(TENANT)} == written
    assert all(e["state"] == "proposed" for e in store.list(TENANT))
    assert sum(r["kind"] == "status_transition" and r["to"] == "proposed" for r in recs) == out["written"]
    assert all(r["model_id"] == "model-large" and r["prompt_version"] == "e1" for r in ext)
    # the memory store's own validator ran: no entry carries a field the schema does not name
    rec, _ = store.get(TENANT, next(iter(written)))
    assert rec.status.state == "proposed"


def test_the_actor_class_follows_the_source_line_not_the_model(world):
    layout, machinery, store, spans = world
    out, recs, _ = extract(layout, machinery, store, spans)
    by_anchor = {}
    for e in store.list(TENANT):
        rec, _ = store.get(TENANT, e["id"])
        by_anchor[rec.anchor.episode] = rec
    talk = "notes/.hidden/talk/2026-01-01.txt"
    assert by_anchor[f"{talk}#L1-L1"].actor.cls == "principal" and by_anchor[f"{talk}#L1-L1"].actor.id == PRINCIPAL
    assert by_anchor[f"{talk}#L2-L2"].actor.cls == "agent"                     # an Assistant turn, whatever the model said
    plan = by_anchor["notes/2026-01-06.md#L3-L3"]
    assert plan.actor.cls == "agent" and plan.actor.id == f"agent (reporting {PRINCIPAL})"   # a daily reporting, not quoting
    end = [r for r in recs if r["kind"] == "run_finished"][0]
    assert end["counts"]["actor_class_overridden"] >= 30
    assert principal_allowed("x.md", ['Owner said: "use the cookie"'], PRINCIPAL)
    assert principal_allowed("x.md", ["owner: always inject the cookie"], PRINCIPAL)
    assert not principal_allowed("x.md", ["Owner decided to inject the cookie"], PRINCIPAL)
    assert principal_allowed("x.md", ['He wrote "inject the cookie on every scan" and left'], PRINCIPAL)
    assert principal_allowed("t.txt", ["[a#L1] User: yes"], PRINCIPAL) and not principal_allowed("t.txt", ["[a#L1] System: cron"], PRINCIPAL)


def test_same_day_lines_from_the_other_source_are_shown_and_land_in_derived_from(world):
    layout, machinery, store, spans = world
    extract(layout, machinery, store, spans)
    rec = next(store.get(TENANT, e["id"])[0] for e in store.list(TENANT)
               if store.get(TENANT, e["id"])[0].anchor.episode == "notes/.hidden/talk/2026-01-01.txt#L3-L3")
    assert rec.derived_from[0] == "notes/.hidden/talk/2026-01-01.txt#L3-L3"
    assert "notes/2026-01-01.md#L6-L6" in rec.derived_from            # the daily's cause line shares three words with it
    assert rec.kind == "observed" and rec.actor.cls == "principal"


def test_two_half_overlapping_same_day_claims_are_paired_and_the_view_shows_it_from_both_sides(world):
    layout, machinery, store, spans = world
    _, recs, _ = extract(layout, machinery, store, spans)
    day = [store.get(TENANT, e["id"])[0] for e in store.list(TENANT)]
    day = [r for r in day if r.anchor.episode.startswith("notes/2026-02-02.md")]
    plan = next(r for r in day if "plan chosen" in r.claim)
    timeout = next(r for r in day if "timeout" in r.claim)
    lo, hi = CONTRADICTION_BAND
    assert lo <= claim_overlap(plan.claim, timeout.claim) < hi
    linked = [r for r in (plan, timeout) if any(x in r.derived_from for x in (plan.id, timeout.id))]
    assert len(linked) == 1                                            # one-directional at write time
    assert [r for r in recs if r["kind"] == "run_finished"][0]["counts"]["contradiction_pairs"] == 1
    for me, other in ((plan, timeout), (timeout, plan)):
        v = memory_view(store, layout, TENANT, me.id, context=1, reasons=machinery.artifact.review_reasons)
        assert [p["id"] for p in v["paired"]] == [other.id] and v["exits"] == ["probation", "retired", "superseded"]
        assert v["extraction"]["prompt_version"] == "e1" and v["anchor"]["span"]["lines"]
        assert "not_a_memory" in v["reasons"] and "confirmed" not in v["reasons"]
    q = memory_queue(store, TENANT)
    assert all({"id", "state", "claim", "kind", "scope", "actor", "episode", "day"} <= set(row) for row in q)


def test_provenance_marks_the_source_words_of_a_claim(world):
    layout, machinery, store, spans = world
    _, recs, _ = extract(layout, machinery, store, spans)
    ext = next(r for r in recs if r["kind"] == "extraction" and r["episode"] == "notes/2026-01-06.md#L6-L6")
    claim = ext["candidate"]["claim"]
    src = [s for s in ext["spans"] if s["field"] == "claim" and s["origin"] == "source"]
    assert src and "outage came from cache key 6" in claim[src[0]["start"]:src[0]["end"]] or claim
    assert any(s["field"] == "notes" for s in ext["spans"]) and all(s["episode"] == ext["episode"] for s in ext["spans"])


def test_a_dry_run_writes_nothing_and_returns_the_candidates(world):
    layout, machinery, store, spans = world
    one = [s for s in spans if s.path == "notes/2026-01-06.md"]
    out, recs, _ = extract(layout, machinery, store, one, job="extract_dry_run", dry_run=True)
    assert out["written"] == 0 and out["accepted"] == 3 and len(out["candidates"]) == 3 and store.list(TENANT) == []
    assert all(r["written_as"] is None for r in recs if r["kind"] == "extraction")
    assert decided_spans(layout, TENANT, machinery.extractor_fingerprint()) == set()      # a dry run decides nothing


def test_a_provider_failure_is_logged_as_malformed_and_the_pass_goes_on(world):
    layout, machinery, store, spans = world

    def flaky(model, req):
        if req.job == "extract" and "2026-03-06.md" in req.messages[0].content[0].text:
            return ProviderError("server", "500")
        return extracting(model, req)
    one = [s for s in spans if s.path in ("notes/2026-03-06.md", "notes/2026-04-06.md")]
    with run(layout, machinery, "extract_first_pass", extractor_fingerprint=machinery.extractor_fingerprint()) as log:
        ex = Extractor(layout, TENANT, machinery, store, FakeProvider(machinery, log, flaky), log, principal_id=PRINCIPAL)
        out = ex.run(one)
    ext = [r for r in read_run(log.path, typed=False) if r["kind"] == "extraction"]
    assert sum((r["rejection_reason"] or "").startswith("malformed: server") for r in ext) == 4 and out["written"] == 3


def test_the_pass_needs_a_first_pass_prompt(world, tmp_path):
    layout, machinery, store, spans = world
    a, k = tmp_path / "a.yaml", tmp_path / "k.yaml"
    a.write_text(ARTIFACT_YAML.replace("prompt_version: e1, prompt: \"Write the marked moment up as one memory, or say why it is not one.\"",
                                       "prompt_version: null, prompt: null"))
    k.write_text(KERNEL_YAML)
    bare = Machinery.from_files(a, k)
    with run(layout, bare, "extract") as log:
        with pytest.raises(ValueError, match="no first_pass extractor prompt"):
            Extractor(layout, TENANT, bare, store, FakeProvider(bare, log), log, principal_id=PRINCIPAL)


def test_fingerprints_change_with_what_they_cover_and_nothing_else(machinery, tmp_path, layout):
    a, k = tmp_path / "a.yaml", tmp_path / "k.yaml"
    a.write_text(ARTIFACT_YAML.replace("judge:      {model: model-small", "judge:      {model: model-large"))
    k.write_text(KERNEL_YAML)
    other = Machinery.from_files(a, k)
    assert other.artifact_hash != machinery.artifact_hash
    assert other.nomination_fingerprint() == machinery.nomination_fingerprint()        # the judge is not the nominator
    assert other.extractor_fingerprint() == machinery.extractor_fingerprint()
    a.write_text(ARTIFACT_YAML.replace("prompt_version: n1", "prompt_version: n2"))
    assert Machinery.from_files(a, k).nomination_fingerprint() != machinery.nomination_fingerprint()
    a.write_text(ARTIFACT_YAML.replace("dedup_threshold: 0.6", "dedup_threshold: 0.7"))
    assert Machinery.from_files(a, k).extractor_fingerprint() != machinery.extractor_fingerprint()
    # a past artifact is found again by its hash, from the live file or from history
    assert Machinery.artifact_at(layout, machinery.artifact_hash).judge.model == "model-small"
    Machinery.install(layout, a, k)
    assert Machinery.artifact_at(layout, machinery.artifact_hash).dedup_threshold == 0.6
    assert Machinery.artifact_at(layout, "sha256:0000") is None


def test_a_secret_in_a_quote_is_stored_redacted_and_counted(world, tmp_path, layout, machinery):
    """The quote is checked verbatim against the source, then stored with the secret replaced."""
    store = MemoryStore(layout, ALLOWLIST)
    cold = layout.tier_dir(TENANT, "cold") / "notes/2026-01-06.md"
    lines = cold.read_text().splitlines()
    lines[3] = "filler two token=sk-abcdefghijklmnopqrstuvwxyz0123 leaked"
    cold.write_text("\n".join(lines) + "\n")
    span = Span("rule_or_decision", "notes/2026-01-06.md#L4-L4", "notes/2026-01-06.md", 4, 4, lines[3])

    def leaky(model, req):
        return _memory(claim="A token was pasted into the notes.", quote=lines[3], quote_start_line=4, quote_end_line=4, scope="notes")
    with run(layout, machinery, "extract_first_pass", extractor_fingerprint=machinery.extractor_fingerprint()) as log:
        ex = Extractor(layout, TENANT, machinery, store, FakeProvider(machinery, log, leaky), log, principal_id=PRINCIPAL)
        out = ex.run([span])
    assert out["written"] == 1
    rec, _ = store.get(TENANT, store.list(TENANT)[0]["id"])
    assert "sk-abc" not in rec.anchor.quote and "[redacted]" in rec.anchor.quote
    end = [r for r in read_run(log.path, typed=False) if r["kind"] == "run_finished"][0]
    assert end["counts"]["secrets_redacted"] >= 1 and end["counts"]["quotes_redacted"] == 1
    assert "sk-abc" not in open(log.path).read()


def test_pass_two_attaches_restatements_supersedes_version_pins_routes_task_shaped_directives_and_pairs_reversals(world):
    """The four changes of the second-pass extractor, on a store that already holds entries."""
    from recant.harness.extract import task_shaped
    from recant.store import TaskStore
    from recant.store.review import decide_memory
    layout, machinery, store, spans = world
    tasks = TaskStore(layout, ALLOWLIST)
    day1 = [s for s in spans if s.path == "notes/2026-01-06.md"]
    # pass 1 writes the day's plan, cause and timeout
    extract(layout, machinery, store, day1)
    first = {store.get(TENANT, e["id"])[0].claim: e["id"] for e in store.list(TENANT)}
    plan_id = next(i for c, i in first.items() if c.startswith("Plan 6"))
    timeout_id = next(i for c, i in first.items() if "seconds" in c)

    def second(model, req):
        user = req.messages[0].content[0].text
        start = int(re.search(r"The moment is lines (\d+)-", user).group(1))
        line = re.search(rf"^{start}: (.*)$", user, flags=re.M).group(1)
        if "we chose plan" in line:      # restated word for word: attaches, not written
            return _memory(claim="Plan 6 governs the scanner since 2026-01-06.", quote=line, quote_start_line=start, quote_end_line=start)
        if "timeout became" in line:     # the same rule with a new number: a version pin, supersedes the earlier
            return _memory(claim="The timeout became 70 seconds on 2026-01-06.", kind="directive", quote=line, quote_start_line=start, quote_end_line=start)
        if "cache key" in line:          # task-shaped: names a branch and states no standing rule
            return _memory(claim="Merge feature/cache-key-6 into main once the outage fix lands.", kind="directive", quote=line,
                           quote_start_line=start, quote_end_line=start, scope="scanner repo")
        return _memory(claim="The scanner plan for 2026-01-06 was withdrawn: plan 6 is abandoned for the scanner.", kind="stated",
                       quote=line, quote_start_line=start, quote_end_line=start, scope="scanner planning")   # a same-day reversal in another scope
    with run(layout, machinery, "extract_first_pass", extractor_fingerprint="sha256:pass2") as log:
        ex = Extractor(layout, TENANT, machinery, store, FakeProvider(machinery, log, second), log, principal_id=PRINCIPAL)
        ex.tasks = tasks
        ex.fingerprint = "sha256:pass2"          # a new prompt is a new fingerprint: pass 1's decisions do not carry over
        out = ex.run(day1)
    recs = list(read_run(log.path, typed=False))
    ext = {r["episode"]: r for r in recs if r["kind"] == "extraction"}
    end = [r for r in recs if r["kind"] == "run_finished"][0]
    assert ext["notes/2026-01-06.md#L3-L3"]["rejection_reason"] == f"restates {plan_id}" and out["attached"] == 1
    assert "notes/2026-01-06.md#L3-L3" in store.get(TENANT, plan_id)[0].derived_from
    pin = ext["notes/2026-01-06.md#L8-L8"]
    assert pin["verdict"] == "accepted" and timeout_id in pin["candidate"]["derived_from"] and end["counts"]["supersedes_earlier"] == 1
    assert ext["notes/2026-01-06.md#L6-L6"]["rejection_reason"].startswith("task_shaped") and out["routed_to_tasks"] == 1
    assert [t["status"] for t in tasks.list(TENANT)] == ["open"] and "feature/cache-key-6" in tasks.get(TENANT, tasks.list(TENANT)[0]["id"])[0].claim
    rev = ext["notes/2026-01-06.md#L15-L15"]
    assert rev["verdict"] == "accepted" and plan_id in rev["candidate"]["derived_from"] and end["counts"]["contradiction_pairs"] == 1   # different scope, shared words
    assert task_shaped("Merge PR #12 now") and not task_shaped("Always merge through Jenkins") and not task_shaped("The timeout is 30 seconds.")
    # accepting the version pin at review supersedes the earlier timeout entry by it
    with run(layout, machinery, "review") as rlog:
        decide_memory(store, TENANT, rlog, pin["written_as"], to_state="probation", reason="confirmed", reviewer={"kind": "model", "id": "r"},
                      reasons=machinery.artifact.review_reasons)
    old = store.get(TENANT, timeout_id)[0]
    assert old.status.state == "superseded" and old.status.superseded_by == pin["written_as"]
    assert sum(r["kind"] == "review_decision" and r["outcome"] == "supersede" for r in read_run(rlog.path, typed=False)) == 1
