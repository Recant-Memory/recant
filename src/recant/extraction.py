"""`recant.extract`: one day's sessions into proposals. The consumer schedules it and brings the model
client; Recant derives the day, nominates, writes `proposed` memories into the review queue, hands the moments it
routed as task to the ledger in the same run, and reports what the run used. No limit of any kind: a consumer that
wants a ceiling wraps the provider it passes."""
from __future__ import annotations

from typing import Any, Callable, Optional

from .harness.extract import Extractor, TaskExtractor
from .harness.grouped import GroupedExtractor
from .harness.nominate import Span, nominate
from .log import LogWriter, read_run
from .machinery import Machinery
from .provider.base import BaseProvider
from .records import parse_episode_id
from .store.hot import derive_daily, session_paths
from .store.layout import StoreLayout


def _usage(log_path) -> dict[str, dict[str, int]]:
    """Tokens per model, summed from the run's own model_call records."""
    out: dict[str, dict[str, int]] = {}
    for r in read_run(log_path, typed=False):
        if r["kind"] == "model_call":
            row = out.setdefault(r["model_id"], {"calls": 0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0})
            row["calls"] += 1
            for k in ("input", "output", "cache_read", "cache_write"):
                row[k] += (r.get("usage") or {}).get(k, 0)
    return out


def extract(layout: StoreLayout, tenant: str, day: str, *, machinery: Machinery, client: Any, store, ledger=None,
            principal_id: str, model: Optional[str] = None, batch: bool = False, wait: Optional[Callable[[], None]] = None,
            dry_run: bool = False, log: Optional[LogWriter] = None, job: str = "extract",
            params: Optional[dict] = None, prices: Optional[dict] = None, sessions: Optional[list[str]] = None,
            lines: Optional[tuple[int, int]] = None) -> dict:
    """Extract one day's hot-tier sessions. `client` is a `ModelClient`, or a `harness.Provider` the consumer has
    already built (to meter or cap it its own way). `log` is the consumer's open run, or None to open one under
    `job`. `model` overrides the artifact's nomination and extraction models. Returns the run id, what was
    proposed and routed, tokens per model, dollars when prices were given (`spent_usd`, else None) and counts.
    `sessions` limits the run to those of the day's session files (episode paths), for a consumer that extracts each
    session as it closes rather than the day at once. `lines` (the overflow trigger) limits nomination to that
    range of lines: what has just left the carried window."""
    paths = session_paths(layout, tenant, day)
    if sessions is not None:
        unknown = set(sessions) - set(paths)
        if unknown:
            raise ValueError(f"not session files of {day}: {sorted(unknown)}")
        paths = [p for p in paths if p in set(sessions)]
    daily = derive_daily(layout, tenant, day)
    kw = dict(machinery=machinery, client=client, store=store, ledger=ledger, principal_id=principal_id, model=model,
              batch=batch, wait=wait or (lambda: None), dry_run=dry_run, prices=prices, lines=lines)
    if log is not None:
        return _run(layout, tenant, day, paths, log, **kw)
    p = {"day": day, "dry_run": dry_run, "batch": batch, "sessions": len(paths),
         "daily": str(daily.relative_to(layout.tenant_dir(tenant))),
         "extractor_fingerprint": machinery.extractor_fingerprint("steady_state"), "model": model, **(params or {})}
    with LogWriter(layout, tenant, job, artifact_hash=machinery.artifact_hash, kernel_hash=machinery.kernel_hash,
                   params=p) as own:
        return _run(layout, tenant, day, paths, own, **kw)


def _run(layout, tenant, day, paths, log, *, machinery, client, store, ledger, principal_id, model, batch, wait, dry_run,
         prices, lines=None) -> dict:
    out: dict[str, Any] = {"run": log.run, "day": day, "sessions": len(paths), "spans": 0, "memory": {}, "tasks": {}}
    if not paths:
        log.count("no_sessions")
        return {**out, "usage": {}, "spent_usd": 0.0, "counts": log.counts, "note": "no session files for the day"}
    provider = client if isinstance(client, BaseProvider) else BaseProvider(machinery, log, client, prices=prices)
    spans = nominate(layout, tenant, machinery, provider, log, paths=paths, batch=batch, wait=wait, model=model, line_range=lines)
    out["spans"] = len(spans)
    rules = machinery.kernel.extraction
    if rules.grouped:                    # memory v2: moments grouped; memories and tasks from one call
        ex = GroupedExtractor(layout, tenant, machinery, store, provider, log, principal_id=principal_id,
                              job="steady_state", batch=batch, wait=wait, dry_run=dry_run, model=model, ledger=ledger)
    else:
        ex = Extractor(layout, tenant, machinery, store, provider, log, principal_id=principal_id,
                       job="steady_state", batch=batch, wait=wait, dry_run=dry_run, model=model)
    ex.tasks = ledger
    mem = ex.run(spans)
    if "tasks" in mem:                   # one pass: the ledger's items came back with the memories
        tasks = mem.pop("tasks")
        for d in (mem, tasks):
            if not dry_run:
                d.pop("candidates", None)
        log.promoted = mem.get("written", 0)
        out.update(memory=mem, tasks=tasks)
        spent = log.reported_spend_usd
        return {**out, "usage": _usage(log.path), "spent_usd": round(spent, 4) if spent is not None else None,
                "counts": log.counts}
    # the moments this run routed as task go to the ledger's extractor in the same run, with their markers
    markers = {s.episode: (s.marker, s.quote) for s in spans}
    routed = []
    for r in read_run(log.path, typed=False):
        if r["kind"] == "extraction" and (r.get("rejection_reason") or "") == "not_a_memory: task" and r["episode"] in markers:
            ref = parse_episode_id(r["episode"]); m, q = markers[r["episode"]]
            routed.append(Span(m, r["episode"], ref.path, ref.start, ref.end, q))
    tasks: dict[str, Any] = {"routed": len(routed)}
    if routed and ledger is not None and machinery.artifact.extractor.tasks is not None:
        tx = TaskExtractor(layout, tenant, machinery, ledger, provider, log, principal_id=principal_id,
                           batch=batch, wait=wait, dry_run=dry_run, model=model)
        tasks.update(tx.run(routed))
    for d in (mem, tasks):
        if not dry_run:
            d.pop("candidates", None)
    log.promoted = mem.get("written", 0)
    out.update(memory=mem, tasks=tasks)
    spent = log.reported_spend_usd
    return {**out, "usage": _usage(log.path), "spent_usd": round(spent, 4) if spent is not None else None,
            "counts": log.counts}
