"""The mechanical write rules, and the already-decided skip keyed on the run's own job: the mechanical rules on the
extractor's candidates, each the shape of a case a live run produced."""
import re

from recant.harness.extract import Extractor, decided_spans
from recant.harness.nominate import Span
from recant.log import LogWriter, read_run
from recant.provider import Usage, text
from recant.provider.base import Raw
from recant.provider.fake import FakeProvider
from recant.store import MemoryStore, ingest
from pipeline_fixtures import build_corpus
from recant_fixtures import ALLOWLIST, TENANT

PRINCIPAL = "owner"
TALK = "notes/.hidden/talk/2026-01-01.txt"


def live_machinery(machinery):
    """The fixture's steady-state extractor has no prompt; the rules are the live pass's, so give it one."""
    from dataclasses import replace
    ex = machinery.artifact.extractor
    art = machinery.artifact.model_copy(update={"extractor": ex.model_copy(update={
        "steady_state": ex.steady_state.model_copy(update={"prompt": "Write the moment up as one memory.", "prompt_version": "s1"})})})
    return replace(machinery, artifact=art)


def run(layout, machinery, job, **params):
    return LogWriter(layout, TENANT, job, artifact_hash=machinery.artifact_hash, kernel_hash=machinery.kernel_hash, params=params)


def answer(claim, kind, actor, line, n):
    import json
    return Raw((text(json.dumps({"verdict": "memory", "not_a_memory_reason": "none", "claim": claim, "kind": kind, "actor_class": actor,
                                 "actor_id": actor, "quote": line, "quote_start_line": n, "quote_end_line": n, "channel": "",
                                 "timestamp": "", "scope": "scanner", "supersession_condition": "Until a later decision.", "notes": ""})),),
               "end_turn", Usage(input=100, output=40))


def responder(claims):
    """claims: {line number: (claim, kind, actor_class)}"""
    def f(model, req):
        user = req.messages[0].content[0].text
        n = int(re.search(r"The moment is lines (\d+)-", user).group(1))
        line = re.search(rf"^{n}: (.*)$", user, flags=re.M).group(1)
        claim, kind, actor = claims[n]
        return answer(claim, kind, actor, line, n)
    return f


def world(layout, tmp_path):
    root = build_corpus(tmp_path / "corpus")
    (root / TALK).write_text("[s#L1] User: the outage was caused by cache key 1 for the scanner\n"
                             "[s#L2] Assistant: I can only search memory and the task ledger; I cannot edit entries.\n"
                             "[s#L3] User: again: the outage was caused by cache key 1\n")
    ingest(layout, TENANT, root, ALLOWLIST)
    spans = [Span("root_cause", f"{TALK}#L1-L1", TALK, 1, 1, "cache key 1"),
             Span("rule_or_decision", f"{TALK}#L2-L2", TALK, 2, 2, "I can only search"),
             Span("root_cause", f"{TALK}#L3-L3", TALK, 3, 3, "cache key 1")]
    return MemoryStore(layout, ALLOWLIST), spans


def extract(layout, machinery, store, spans, claims, *, job="extract_live"):
    machinery = live_machinery(machinery)
    with run(layout, machinery, job, extractor_fingerprint=machinery.extractor_fingerprint("steady_state")) as log:
        ex = Extractor(layout, TENANT, machinery, store, FakeProvider(machinery, log, responder(claims)), log, principal_id=PRINCIPAL, batch=False, job="steady_state")
        out = ex.run(spans)
    recs = [r for r in read_run(log.path, typed=False) if r["kind"] == "extraction"]
    return out, {r["episode"].rsplit("#", 1)[1]: r for r in recs}, log


def test_an_agent_line_is_never_stated_and_a_claim_repeating_one_written_this_run_folds(layout, machinery, tmp_path):
    store, spans = world(layout, tmp_path)
    claims = {1: ("Cache key 1 caused the scanner outage.", "concluded", "agent"),
              2: ("The agent can only search memory and the task ledger and cannot edit entries.", "stated", "agent"),
              3: ("Cache key 1 caused the scanner outage yesterday.", "concluded", "agent")}
    out, by_line, log = extract(layout, machinery, store, spans, claims)
    assert out["written"] == 1 and out["rejected"] == 2
    assert by_line["L1-L1"]["verdict"] == "accepted"
    assert by_line["L2-L2"]["rejection_reason"].startswith("agent_self_description")        # the identity file owns it
    written_id = by_line["L1-L1"]["written_as"]
    assert by_line["L3-L3"]["rejection_reason"] == f"duplicate_in_run {written_id}"            # overlap ≥ 0.7, numbers alike or not
    assert log.counts["rejected: agent_self_description"] == 1 and log.counts["rejected: duplicate_in_run"] == 1
    assert log.counts["attached_to_survivor"] == 1
    # an agent turn may still conclude: the concluded candidate was written, by an agent
    rec, _ = store.get(TENANT, written_id)
    assert rec.kind == "concluded" and rec.actor.cls == "agent"


def test_the_same_anchor_line_as_a_stored_entry_is_a_duplicate_in_any_state_and_a_live_rerun_skips_decided_spans(layout, machinery, tmp_path):
    store, spans = world(layout, tmp_path)
    first = {1: ("Cache key 1 caused the scanner outage.", "concluded", "agent"),
             2: ("Nothing here.", "concluded", "agent"),
             3: ("Cache key 1 broke the scanner, said twice.", "concluded", "agent")}
    out1, by1, log1 = extract(layout, machinery, store, spans, first, job="extract_live")
    assert out1["written"] >= 1
    kept = by1["L1-L1"]["written_as"]
    # a second run for the same day under the same job and fingerprint decides nothing again (a nightly job can run twice)
    fp = live_machinery(machinery).extractor_fingerprint("steady_state")
    assert decided_spans(layout, TENANT, fp, job="extract_live") == {s.episode for s in spans}
    assert decided_spans(layout, TENANT, fp, job="extract_first_pass") == set()
    out2, by2, log2 = extract(layout, machinery, store, spans, first, job="extract_live")
    assert out2["calls"] == 0 and log2.counts["already_decided"] == 3
    # a run that does reach the line again — a different nomination, the same anchor — folds to the entry on that line
    fresh = {1: ("A new wording: the outage traces to the first cache key.", "concluded", "agent"),
             2: ("Nothing here either.", "concluded", "agent"), 3: ("And a third wording of the same.", "concluded", "agent")}
    out3, by3, _ = extract(layout, machinery, store, spans, fresh, job="extract_live_rerun")      # a different job name: nothing pre-decided
    assert by3["L1-L1"]["rejection_reason"] == f"duplicate_anchor {kept} (proposed)"
    assert out3["written"] == 0
    # a retired twin still blocks the line, without attaching to it
    rec, notes = store.get(TENANT, kept)
    data = rec.to_dict(); data["status"] = {"state": "retired", "reason": "noise", "failure_class": None, "superseded_by": None}
    with run(layout, machinery, "review") as log:
        store.put(TENANT, data, writer="review", log=log, notes=notes)
    store2 = MemoryStore(layout, ALLOWLIST)
    out4, by4, log4 = extract(layout, machinery, store2, spans, fresh, job="extract_dry_run")
    assert by4["L1-L1"]["rejection_reason"] == f"duplicate_anchor {kept} (retired)"
    assert log4.counts["attached_to_survivor"] == 2                   # L2 and L3 attached to their standing twins; the retired one took nothing
    assert all(by4[k]["rejection_reason"].endswith("(proposed)") for k in ("L2-L2", "L3-L3"))


def test_a_claim_the_store_already_holds_is_not_written_again_but_a_changed_value_is(layout, machinery, tmp_path):
    """Write-time dedupe: the placeholder's billing port was in the store three times,
    reworded each day. A claim sharing a number or path with a standing entry, overlapping it at
    store_dedupe_overlap and adding no number, attaches to it; one carrying a new number is written."""
    from recant.harness.extract import held_already
    from test_service import entry
    store, spans = world(layout, tmp_path)
    with run(layout, machinery, "seed") as log:
        held = entry(store, log, "The billing service listens on port 8431 in staging, set in config/staging/billing.yaml.",
                     "notes/2026-01-01.md#L6-L6", state="probation", kind="observed", scope="billing")
    from dataclasses import replace as _r
    spans = [_r(sp, marker="rule_or_decision") for sp in spans]       # plain markers: fixes, reversals and corrections are exempt
    claims = {1: ("Staging's billing port is 8431; config/staging/billing.yaml sets it for that environment.", "observed", "agent"),
              2: ("The billing service listens on port 8500 in staging, set in config/staging/billing.yaml.", "observed", "agent"),
              3: ("Scanner deploys wait for a queue check before anyone edits config/staging/billing.yaml.", "observed", "agent")}
    out, by_line, log = extract(layout, machinery, store, spans, claims)
    assert by_line["L1-L1"]["rejection_reason"] == f"held_already {held}"                    # reworded, nothing new: not written
    assert by_line["L2-L2"]["verdict"] == "accepted"                                         # a new number may be a change
    assert by_line["L3-L3"]["verdict"] == "accepted"                                         # same path, a different fact
    rec, _ = store.get(TENANT, held)
    assert f"{TALK}#L1-L1" in rec.derived_from                                               # the second sighting is evidence
    assert log.counts["rejected: held_already"] == 1
    # the rule itself, on the placeholder's own rewordings and on a near-miss
    a = "The billing service listens on port 8431 in staging, set in config/staging/billing.yaml via the line `port: 8431`."
    assert held_already("In the staging environment, the billing service listens on port 8431, as configured in config/staging/billing.yaml.", a, 0.35)
    assert not held_already("The billing service's staging port is set to 8431; production uses 9431.", a, 0.35)   # 9431 is new
    assert not held_already("The billing service listens on a port set in its staging config.", a, 0.35)          # no shared token
    # only plain restatements fold: a fix is not folded into the bug it fixed, whatever they share
    bug = "The example app's report job was reading the OLDEST 20 rows (sorted ascending, first page only) in jobs/report.py."
    fix = "The example app's report job now reads the NEWEST 20 rows in jobs/report.py, shown in order."
    assert claim_overlap_ok(fix, bug) and not held_already(fix, bug, 0.35)
    off = live_machinery(machinery)
    assert off.kernel.extraction.store_dedupe_overlap == 0.35


def claim_overlap_ok(a, b):
    from recant.harness.extract import claim_overlap, signature
    return claim_overlap(a, b) >= 0.35 and signature(a) <= signature(b)



def test_a_moment_marked_as_a_fix_reversal_or_correction_is_never_folded(layout, machinery, tmp_path):
    """the markers that carry a fix, a reversal or a correction keep a claim out of write-time dedupe."""
    from recant.harness.extract import FOLD_EXEMPT_MARKERS
    from test_service import entry
    store, spans = world(layout, tmp_path)
    with run(layout, machinery, "seed") as log:
        entry(store, log, "The billing service listens on port 8431 in staging, set in config/staging/billing.yaml.",
              "notes/2026-01-01.md#L6-L6", state="probation", kind="observed", scope="billing")
    assert {"reversal", "root_cause", "failure_with_reason", "manner_correction"} <= FOLD_EXEMPT_MARKERS
    claims = {1: ("Staging's billing port is 8431; config/staging/billing.yaml sets it for that environment.", "observed", "agent"),
              2: ("Nothing new.", "observed", "agent"), 3: ("Nothing new either.", "observed", "agent")}
    # world()'s span on line 1 is marked root_cause: the same restatement that folds under a plain marker is written here
    out, by_line, log = extract(layout, machinery, store, spans, claims)
    assert spans[0].marker == "root_cause" and by_line["L1-L1"]["verdict"] == "accepted"
