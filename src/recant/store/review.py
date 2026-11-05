"""The memory lifecycle's manual exits, taken by a reviewer. The same interaction
as reviewing an eval item: a proposal, its provenance, a decision."""
from __future__ import annotations

from typing import Optional

from ..records import RecordError
from .memory import MemoryStore

# (from, to) -> the review outcome it is logged as
MANUAL_EXITS = {("proposed", "probation"): "accept", ("proposed", "retired"): "discard",
                ("probation", "live"): "to_live", ("probation", "retired"): "retire",
                ("live", "retired"): "retire", ("contested", "live"): "confirm",
                # four outcomes — accept, fix, supersede, discard
                ("proposed", "superseded"): "supersede", ("probation", "superseded"): "supersede",
                ("contested", "superseded"): "supersede"}
AWAITING = ("proposed", "probation", "contested")


_NAMES_PRINCIPAL = r"\b{who}\b[^.]{{0,40}}\b(decid|direct|instruct|told|asked|ask|want|approv|said|says|ruled|prefer|require|order|insist|confirm)"
GROUPS = {"paired": 1, "class_overridden": 2, "names_principal": 3, "rest": 4}


def memory_queue(store: MemoryStore, tenant: str, principal_id: Optional[str] = None) -> list[dict]:
    """The entries awaiting a decision, contested ones first: the contradiction pairs, the
    entries whose actor class the rule overrode, then rank-3 entries whose claim names the principal as
    deciding or directing (a claim should not), then the rest by day."""
    import re
    from ..records import is_ulid
    rows, paired = [], set()
    records = {}
    for entry in store.list(tenant):
        if entry["state"] in AWAITING:
            record, _ = store.get(tenant, entry["id"])
            records[record.id] = record
            for ref in record.derived_from:
                if is_ulid(ref):
                    paired.update((record.id, ref))
    names = re.compile(_NAMES_PRINCIPAL.format(who=re.escape(principal_id)), re.I) if principal_id else None
    for record in records.values():
        if record.id in paired:
            group = "paired"
        elif "(reporting" in record.actor.id:
            group = "class_overridden"
        elif names and record.actor.cls == "agent" and record.kind in ("stated", "directive") and names.search(record.claim):
            group = "names_principal"
        else:
            group = "rest"
        rows.append({"id": record.id, "state": record.status.state, "claim": record.claim,
                     "kind": record.kind, "scope": record.scope, "actor": record.actor.cls,
                     "episode": record.anchor.episode, "day": record.anchor.timestamp, "group": group})
    # Undecided first: a proposed entry is the reviewer's work, a probation one is already theirs.
    rows.sort(key=lambda r: (0 if r["state"] == "proposed" else 1, GROUPS[r["group"]], r["day"] or "", r["id"]))
    return rows


# Reasons a reviewer may give on a memory's manual exits; each is a failure class or a review reason.
MEMORY_DISCARD_REASONS = ("not_a_memory", "wrong_answer", "contradiction", "staleness", "premise_collapse")


def memory_view(store: MemoryStore, layout, tenant: str, entry_id: str, *, context: int, reasons: list[str]) -> dict:
    """Everything a decision on one entry needs, from the store and the log: the record, the anchor with
    its lines, every derived span with a line of context, the provenance colouring from the extraction that
    wrote it, the entry it was paired with, and the decisions so far."""
    from ..eval.view import span_view
    from ..log import read_run
    from ..records import is_ulid
    record, notes = store.get(tenant, entry_id)
    anchor = span_view(layout, tenant, record.anchor.episode, context)
    derived, paired = [], []
    for ref in record.derived_from:
        if is_ulid(ref):
            try:
                other, _ = store.get(tenant, ref)
                paired.append({"id": other.id, "claim": other.claim, "state": other.status.state})
            except KeyError:
                paired.append({"id": ref, "claim": None, "state": None})
        elif ref != record.anchor.episode:
            derived.append(span_view(layout, tenant, ref, 1))
    for entry in store.list(tenant):      # the pairing is one-directional at write time; show it from both sides
        if entry["id"] != entry_id and entry["state"] not in ("retired", "superseded"):
            other, _ = store.get(tenant, entry["id"])
            if entry_id in other.derived_from and all(p["id"] != other.id for p in paired):
                paired.append({"id": other.id, "claim": other.claim, "state": other.status.state})
    extraction, decisions = None, []
    for path in sorted(layout.log_dir(tenant).glob("*.jsonl")):
        for r in read_run(path, typed=False):
            if r["kind"] == "extraction" and r.get("written_as") == entry_id:
                extraction = {"model_id": r["model_id"], "prompt_version": r["prompt_version"], "spans": r["spans"],
                              "episode": r["episode"], "run": path.stem}
            elif r["kind"] == "review_decision" and r.get("subject", {}).get("id") == entry_id:
                decisions.append({"ts": r["ts"], "outcome": r["outcome"], "reason": r["reason"], "note": r.get("note"),
                                  "reviewer": r["reviewer"], "fixed_answer": r.get("fixed_answer")})
    return {"id": record.id, "state": record.status.state, "claim": record.claim, "kind": record.kind,
            "actor": {"class": record.actor.cls, "id": record.actor.id}, "scope": record.scope,
            "supersession_condition": record.supersession_condition, "status": record.status.model_dump(mode="json"),
            "anchor": {"quote": record.anchor.quote, "episode": record.anchor.episode, "channel": record.anchor.channel,
                       "timestamp": record.anchor.timestamp, "span": anchor},
            "derived": derived, "paired": paired, "notes": notes, "extraction": extraction, "decisions": decisions,
            "exits": sorted(to for (frm, to) in MANUAL_EXITS if frm == record.status.state),
            "reasons": [r for r in MEMORY_DISCARD_REASONS if r in reasons]}


def decide_memory(store: MemoryStore, tenant: str, log, entry_id: str, *, to_state: str, reason: str,
                  reviewer: dict, reasons: list[str], note: Optional[str] = None,
                  failure_class: Optional[str] = None, claim: Optional[str] = None,
                  superseded_by: Optional[str] = None) -> dict:
    from ..records import is_ulid
    record, notes = store.get(tenant, entry_id)
    move = (record.status.state, to_state)
    if move not in MANUAL_EXITS:
        raise RecordError(f"{move[0]} -> {move[1]} is not an exit a reviewer takes")
    if reason not in reasons:
        raise RecordError(f"reason is one of {reasons}")
    if claim is not None and move != ("proposed", "probation"):
        raise RecordError("the claim is corrected only on the way from proposed to probation")
    if to_state == "superseded":
        if not (superseded_by and is_ulid(superseded_by)) or superseded_by == entry_id:
            raise RecordError("superseding names the successor entry's id")
        failure_class = failure_class or "staleness"
    data = record.to_dict()
    outcome = MANUAL_EXITS[move]
    if claim is not None:
        data["claim"], outcome = claim, "fix"   # the original text is kept as a version
    data["status"] = {"state": to_state, "reason": note or reason, "failure_class": failure_class,
                      "superseded_by": superseded_by if to_state == "superseded" else None}
    decision = dict(subject={"type": "memory", "id": entry_id}, outcome=outcome, reason=reason, note=note, reviewer=reviewer,
                    fixed_answer=claim if claim is not None else superseded_by if to_state == "superseded" else None)
    log.validate("review_decision", **decision)      # the record is checked before the entry is written
    store.put(tenant, data, writer="review", log=log, notes=notes)
    rec = log.emit("review_decision", **decision)
    # Pass 2: an accepted entry whose notes say "Supersedes <id>" moves that earlier entry to
    # superseded by it — the write the extractor could not make, because only review may end an entry's standing.
    if outcome in ("accept", "fix"):
        import re as _re
        for earlier in _re.findall(r"Supersedes ([0-9A-HJKMNP-TV-Z]{26})", notes or ""):
            try:
                old, old_notes = store.get(tenant, earlier)
            except KeyError:
                continue
            if old.status.state in ("proposed", "probation", "live", "contested"):
                d = old.to_dict()
                d["status"] = {"state": "superseded", "reason": f"superseded by {entry_id} on accept", "failure_class": "staleness", "superseded_by": entry_id}
                store.put(tenant, d, writer="review", log=log, notes=old_notes)
                log.emit("review_decision", subject={"type": "memory", "id": earlier}, outcome="supersede", reason="staleness",
                         note=f"superseded by {entry_id} on its acceptance (pass 2 version pin)", reviewer=reviewer, fixed_answer=entry_id)
    return rec


def successors(store: MemoryStore, tenant: str, entry_id: str, limit: int = 8) -> list[dict]:
    """A short list to supersede into: entries in the same scope with a later date, nearest first, then any
    later entry sharing words with the claim."""
    from ..harness.extract import claim_overlap
    record, _ = store.get(tenant, entry_id)
    day = (record.anchor.timestamp or "")[:10]
    scope = record.scope.strip().lower()
    out = []
    for e in store.list(tenant):
        if e["id"] == entry_id or e["state"] in ("retired", "superseded"):
            continue
        other, _ = store.get(tenant, e["id"])
        oday = (other.anchor.timestamp or "")[:10]
        if oday < day:
            continue
        same_scope = other.scope.strip().lower() == scope
        overlap = claim_overlap(record.claim, other.claim)
        if same_scope or overlap >= 0.2:
            out.append({"id": other.id, "claim": other.claim, "state": other.status.state, "day": oday,
                        "same_scope": same_scope, "overlap": round(overlap, 2)})
    out.sort(key=lambda r: (not r["same_scope"], -r["overlap"], r["day"]))
    return out[:limit]


def fold_duplicate(store: MemoryStore, tenant: str, log, retired_id: str, survivor_id: str, *, reviewer: dict,
                   note: Optional[str] = None, from_states: tuple = ("retired",)) -> dict:
    """A retired duplicate's anchor joins the survivor's `derived_from`; the survivor keeps its own anchor as
    primary and the retired entry stays retired. Coverage then finds the item. With
    `from_states` including "superseded", a superseded entry's anchor folds into its successor the same way
    so the successor covers what the history it carries is cited at."""
    dup, _ = store.get(tenant, retired_id)
    if dup.status.state not in from_states:
        raise RecordError(f"{retired_id} is {dup.status.state}, not {' or '.join(from_states)}")
    survivor, notes = store.get(tenant, survivor_id)
    if survivor.status.state in ("retired", "superseded"):
        raise RecordError(f"the survivor {survivor_id} is {survivor.status.state}")
    data = survivor.to_dict()
    added = [ref for ref in [dup.anchor.episode, *dup.derived_from] if ref not in data["derived_from"] and ref != retired_id]
    decision = dict(subject={"type": "memory", "id": survivor_id}, outcome="fix", reason="confirmed",
                    note=note or f"{dup.status.state} {retired_id} folded: {len(added)} anchor(s) added to derived_from", reviewer=reviewer)
    log.validate("review_decision", **decision)      # checked before the entry is written
    if added:
        data["derived_from"] = data["derived_from"] + added
        store.put(tenant, data, writer="review", log=log, notes=notes)
    return log.emit("review_decision", **decision)


# --- the task ledger's review: done / dropped / still open ------------------------------

TASK_EXITS = {("open", "done"): "done", ("open", "dropped"): "dropped", ("open", "open"): "still_open"}


def task_queue(tasks, tenant: str) -> list[dict]:
    """Open first, then closed with their dates."""
    rows = []
    for t in tasks.list(tenant):
        rec, _ = tasks.get(tenant, t["id"])
        rows.append({"id": rec.id, "status": rec.status, "claim": rec.claim, "scope": rec.scope, "actor": rec.actor.cls,
                     "opened": rec.opened, "closed": rec.closed, "closed_by": rec.closed_by, "episode": rec.anchor.episode})
    rows.sort(key=lambda r: (0 if r["status"] == "open" else 1, r["opened"], r["id"]))
    return rows


def decide_task(tasks, tenant: str, log, task_id: str, *, to_state: str, reviewer: dict, closed: Optional[str] = None,
                note: Optional[str] = None) -> dict:
    rec, notes = tasks.get(tenant, task_id)
    move = (rec.status, to_state)
    if move not in TASK_EXITS:
        raise RecordError(f"{move[0]} -> {move[1]} is not an exit a reviewer takes on a task")
    data = rec.to_dict()
    decision = dict(subject={"type": "memory", "id": task_id}, outcome=TASK_EXITS[move], reason="confirmed", note=note, reviewer=reviewer)
    log.validate("review_decision", **decision)      # checked before the entry is written
    if to_state != "open":
        data.update(status=to_state, closed=closed or rec.closed or rec.opened)
        tasks.put(tenant, data, writer="review", log=log, notes=notes)
    return log.emit("review_decision", **decision)
