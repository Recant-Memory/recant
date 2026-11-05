"""The three baseline arms and the two service arms. Each produces one answer per item
from what it retrieves, through the same reader; the judge then scores every answer the same way.
Nothing here reads the held-out slice. Only a service arm touches `exposure`, because only a service
arm is a candidate run."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

from ..eval.items import EvalItem
from ..eval.ledger import Ledger
from ..eval.pools import increment_exposure, load
from ..eval.search import grep
from ..machinery import Machinery
from ..provider import ContentBlock, Message, ModelResponse, ProviderError, text
from ..records import parse_episode_id
from ..store.episodes import read_manifest
from ..store.layout import StoreLayout
from ..store.search import coverage, load_entries, load_tasks, manifest, search_memory
from .common import run_all
from .judge import judge_request, read_verdict
from .reader import GREP_TOOL, MEMORY_TOOL, answer_of, declared_nothing, request, system_blocks
from .score import coverage_split, report_rates

ARMS = ("grep", "context", "oracle", "service", "service_context", "service_budget", "service_retrieval")
SERVICE_ARMS = ("service", "service_context", "service_budget", "service_retrieval")


class CacheMiss(RuntimeError):
    stopped_by = "error"


@dataclass
class Answer:
    item: EvalItem
    text: str = ""
    cut_off: bool = False
    failed: str | None = None
    retrieved: bool | None = None
    trace: list[dict] = field(default_factory=list)
    tokens_seen: int = 0
    material: str | None = None
    responses: int = 0
    extracted: bool | None = None      # service arms: some visible entry covers the cited span
    entries_seen: int = 0              # service arms: entries shown to the reader


def _lines(layout, tenant, path) -> list[str]:
    return layout.episode_file(tenant, path).read_text(encoding="utf-8", errors="replace").splitlines()


def _span_seen(item: EvalItem, seen: set[tuple[str, int]]) -> bool | None:
    if not item.episode:
        return None
    r = parse_episode_id(item.episode)
    return any((r.path, n) in seen for n in range(r.start, r.end + 1))


# --- arm 1: grep in a bounded loop ------------------------------------------------------------------

def _grep_result(layout, tenant, terms, cfg) -> tuple[str, dict, set[tuple[str, int]]]:
    found = grep(layout, tenant, terms, limit=cfg.lines_per_call)
    seen: set[tuple[str, int]] = set()
    out = []
    for h in found["hits"]:
        r = parse_episode_id(h["episode"])
        lines = _lines(layout, tenant, r.path)
        a, b = max(1, r.start - cfg.context_lines), min(len(lines), r.end + cfg.context_lines)
        seen.update((r.path, n) for n in range(a, b + 1))
        block = "\n".join(f"{'>' if n == r.start else ' '} {n}: {lines[n - 1][:400]}" for n in range(a, b + 1))
        out.append(f"[{r.path}]\n{block}")
    head = (f"{found['total']} matching lines; showing {len(found['hits'])}." if found["total"] else "No line contains every term.")
    return head + "\n\n" + "\n\n".join(out), {"terms": terms, "total": found["total"],
                                            "returned": [h["episode"] for h in found["hits"]]}, seen


def run_grep(layout, tenant, machinery, provider, log, items, *, batch, wait) -> list[Answer]:
    cfg = machinery.kernel.eval.arms.grep
    note = (f"You have a `grep` tool. Search the record with it, up to {cfg.calls} times, then answer. "
            f"Search first; do not answer from memory.")
    system = system_blocks(machinery, None, tool_note=note)
    answers = {it.id: Answer(it) for it in items}
    convo = {it.id: [Message("user", (text(f"Question: {it.question}"),))] for it in items}
    seen: dict[str, set] = {it.id: set() for it in items}
    calls: dict[str, int] = {it.id: 0 for it in items}
    pending = [it.id for it in items]
    for rnd in range(cfg.calls + 1):
        if not pending:
            break
        last = rnd == cfg.calls
        reqs = []
        for i in pending:
            msgs = list(convo[i])
            if last:
                msgs.append(Message("user", (text("No more searches are available. Answer now from what you have seen, "
                                                   f"or say {'NOTHING RELEVANT'}."),)))
            reqs.append(request(machinery, system=system, messages=tuple(msgs), job="eval_grep",
                                custom_id=i, tools=() if last else (GREP_TOOL,)))
        still = []
        for cid, resp in run_all(provider, reqs, batch=batch, wait=wait):
            a = answers[cid]
            a.responses += 1
            if isinstance(resp, ProviderError):
                a.failed = str(resp)
                continue
            a.tokens_seen = max(a.tokens_seen, resp.usage.input + resp.usage.cache_read + resp.usage.cache_write)
            uses = [b for b in resp.content if b.kind == "tool_use"]
            if uses and not last:
                convo[cid].append(Message("assistant", resp.content))
                results = []
                for u in uses:
                    terms = [t for t in (u.tool_input or {}).get("terms", []) if isinstance(t, str)]
                    body, trace, hit = _grep_result(layout, tenant, terms, cfg)
                    a.trace.append(trace)
                    seen[cid] |= hit
                    calls[cid] += 1
                    results.append(ContentBlock("tool_result", text=body, tool_use_id=u.tool_use_id))
                convo[cid].append(Message("user", tuple(results)))
                still.append(cid)
            else:
                a.text, a.cut_off = answer_of(resp)
        pending = still
    for i, a in answers.items():
        a.retrieved = _span_seen(a.item, seen[i])
        log.emit("retrieval_trace", item=i, arm="grep", calls=a.trace, tokens_seen=a.tokens_seen)
    return list(answers.values())


# --- arm 2: the dailies in full context, one cached prefix ----------------------------------------------

def dailies_prefix(layout, tenant, patterns) -> str:
    """The full-context arm's prefix: every archived file the kernel's `full_context` patterns admit, in path
    order, each headed by its path. Byte-identical from one item to the next by construction."""
    from ..records import path_allowed
    parts = []
    for e in read_manifest(layout, tenant):
        if path_allowed(e["path"], patterns):
            parts.append(f"===== {e['path']} =====\n" + (layout.tier_dir(tenant, "cold") / e["path"]).read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join(parts)


def run_context(layout, tenant, machinery, provider, log, items) -> list[Answer]:
    """Sequential by design: the point of the arm is that every call after the first reads the cache."""
    ttl = machinery.kernel.eval.arms.context_cache_ttl
    prefix = dailies_prefix(layout, tenant, machinery.kernel.corpus.full_context)
    system = system_blocks(machinery, prefix, cache_material=True, ttl=ttl)
    warm = provider.call(request(machinery, system=system, messages=(Message("user", (text("Reply with the single word: ready"),)),),
                                 job="eval_context", custom_id="warm"))
    stats = {"prefix_tokens": warm.usage.input + warm.usage.cache_write + warm.usage.cache_read,
             "cache_write": warm.usage.cache_write, "calls": 0, "cache_hits": 0}
    answers = []
    for n, it in enumerate(items):
        a = Answer(it, material="prefix", retrieved=True if it.episode else None)
        try:
            resp = provider.call(request(machinery, system=system, messages=(Message("user", (text(f"Question: {it.question}"),)),),
                                         job="eval_context", custom_id=it.id))
        except ProviderError as exc:
            a.failed = str(exc)
            answers.append(a)
            continue
        stats["calls"] += 1
        hit = resp.usage.cache_read > 0
        stats["cache_hits"] += hit
        if n == 0 and not hit:
            log.emit("retrieval_trace", item=it.id, arm="context", material="prefix", tokens_seen=resp.usage.input)
            raise CacheMiss("the first item after the warming call read nothing from the cache; the arm would cost "
                            "the uncached price. Stopped.")
        a.text, a.cut_off = answer_of(resp)
        a.tokens_seen = resp.usage.input + resp.usage.cache_read + resp.usage.cache_write
        a.responses = 1
        answers.append(a)
    log.count("context_calls", stats["calls"])
    log.count("context_cache_hits", stats["cache_hits"])
    log.count("context_prefix_tokens", stats["prefix_tokens"])
    return answers


# --- arm 3: the named episode alone ---------------------------------------------------------------------

def run_oracle(layout, tenant, machinery, provider, log, items, *, batch, wait) -> list[Answer]:
    answers, reqs = {}, []
    for it in items:
        if not it.episode:
            continue   # a noise item has no episode: the oracle arm skips it
        path = parse_episode_id(it.episode).path
        material = f"===== {path} =====\n" + "\n".join(_lines(layout, tenant, path))
        answers[it.id] = Answer(it, material=path, retrieved=True)
        reqs.append(request(machinery, system=system_blocks(machinery, material),
                            messages=(Message("user", (text(f"Question: {it.question}"),)),), job="eval_oracle", custom_id=it.id))
    for cid, resp in run_all(provider, reqs, batch=batch, wait=wait):
        a = answers[cid]
        a.responses = 1
        if isinstance(resp, ProviderError):
            a.failed = str(resp)
        else:
            a.text, a.cut_off = answer_of(resp)
            a.tokens_seen = resp.usage.input + resp.usage.cache_read + resp.usage.cache_write
        log.emit("retrieval_trace", item=cid, arm="oracle", material=a.material, tokens_seen=a.tokens_seen)
    return list(answers.values())


# --- the service arms: retrieval over the warm tier --------------------------------------------

def service_note(view: str) -> str:
    return ("You have a `search_memory` tool over a store of memories written from the record. Each entry carries its "
            "status: a proposed entry is unreviewed; a probation entry is in use but not yet confirmed; a contested entry "
            "has a challenge attached; an entry that replaced an earlier one shows what it replaced. Search first, more "
            "than once with different terms if the first search finds nothing; do not answer from memory. "
            f"The store is the {view.replace('_', ' ')} view.")


def _entries(layout, tenant, machinery, store, view, tasks=None):
    """Memory in the view asked for, and every ledger task beside it (searched alongside)."""
    entries = load_entries(store, tenant, view, machinery.artifact.trust_rank)
    if tasks is not None:
        entries = entries + load_tasks(tasks, tenant, machinery.artifact.trust_rank,
                                       stale_after_days=machinery.kernel.tasks.stale_after_days)
    return entries, manifest(entries, view)


def run_service(layout, tenant, machinery, store, provider, log, items, *, view, batch, wait, tasks=None) -> list[Answer]:
    cfg = machinery.kernel.eval.arms.service
    entries, man = _entries(layout, tenant, machinery, store, view, tasks)
    system = system_blocks(machinery, man, tool_note=service_note(view) + f" You may search up to {cfg.rounds} times.")
    answers = {it.id: Answer(it, extracted=coverage(entries, it.episode)) for it in items}
    convo = {it.id: [Message("user", (text(f"Question: {it.question}"),))] for it in items}
    seen: dict[str, set[str]] = {it.id: set() for it in items}
    pending = [it.id for it in items]
    for rnd in range(cfg.rounds + 1):
        if not pending:
            break
        last = rnd == cfg.rounds
        reqs = []
        for i in pending:
            msgs = list(convo[i])
            if last:
                msgs.append(Message("user", (text("No more searches are available. Answer now from what you have seen, "
                                                   f"or say {'NOTHING RELEVANT'}."),)))
            reqs.append(request(machinery, system=system, messages=tuple(msgs), job="eval_service",
                                custom_id=i, tools=() if last else (MEMORY_TOOL,)))
        still = []
        for cid, resp in run_all(provider, reqs, batch=batch, wait=wait):
            a = answers[cid]
            a.responses += 1
            if isinstance(resp, ProviderError):
                a.failed = str(resp)
                continue
            a.tokens_seen = max(a.tokens_seen, resp.usage.input + resp.usage.cache_read + resp.usage.cache_write)
            uses = [b for b in resp.content if b.kind == "tool_use"]
            if uses and not last:
                convo[cid].append(Message("assistant", resp.content))
                results = []
                for u in uses:
                    terms = [t for t in (u.tool_input or {}).get("terms", []) if isinstance(t, str)]
                    body, ids = search_memory(entries, terms, max_entries=cfg.entries_per_search, max_chars=cfg.chars_per_search)
                    a.trace.append({"terms": terms, "returned": ids})
                    seen[cid].update(ids)
                    results.append(ContentBlock("tool_result", text=body, tool_use_id=u.tool_use_id))
                convo[cid].append(Message("user", tuple(results)))
                still.append(cid)
            else:
                a.text, a.cut_off = answer_of(resp)
        pending = still
    by_id = {e.id: e for e in entries}
    for i, a in answers.items():
        a.entries_seen = len(seen[i])
        from ..store.search import covers
        a.retrieved = None if not a.item.episode else any(covers(by_id[x], a.item.episode) for x in seen[i] if x in by_id)
        log.emit("retrieval_trace", item=i, arm="service", calls=a.trace, tokens_seen=a.tokens_seen)
    log.count("service_entries", len(entries))
    return list(answers.values())


def warm_prefix(entries, man: str) -> str:
    """The whole visible warm tier as one block, byte-identical across items."""
    return man + "\n\n===== entries =====\n\n" + "\n\n".join(e.render() for e in entries)


def run_service_context(layout, tenant, machinery, store, provider, log, items, *, view, tasks=None) -> list[Answer]:
    """Sequential like the dailies arm: the warm tier is small enough to be one cached prefix."""
    ttl = machinery.kernel.eval.arms.context_cache_ttl
    entries, man = _entries(layout, tenant, machinery, store, view, tasks)
    prefix = warm_prefix(entries, man)
    system = system_blocks(machinery, prefix, cache_material=True, ttl=ttl, tool_note=service_note(view).split(" Search first")[0])
    warm = provider.call(request(machinery, system=system, messages=(Message("user", (text("Reply with the single word: ready"),)),),
                                 job="eval_service_context", custom_id="warm"))
    stats = {"prefix_tokens": warm.usage.input + warm.usage.cache_write + warm.usage.cache_read, "calls": 0, "cache_hits": 0}
    answers = []
    for n, it in enumerate(items):
        a = Answer(it, material="warm_prefix", extracted=coverage(entries, it.episode), entries_seen=len(entries))
        a.retrieved = a.extracted if it.episode else None       # every visible entry is in front of the reader
        try:
            resp = provider.call(request(machinery, system=system, messages=(Message("user", (text(f"Question: {it.question}"),)),),
                                         job="eval_service_context", custom_id=it.id))
        except ProviderError as exc:
            a.failed = str(exc)
            answers.append(a)
            continue
        stats["calls"] += 1
        hit = resp.usage.cache_read > 0
        stats["cache_hits"] += hit
        if n == 0 and not hit:
            raise CacheMiss("the first item after the warming call read nothing from the cache. Stopped.")
        a.text, a.cut_off = answer_of(resp)
        a.tokens_seen = resp.usage.input + resp.usage.cache_read + resp.usage.cache_write
        a.responses = 1
        answers.append(a)
    log.count("context_calls", stats["calls"])
    log.count("context_cache_hits", stats["cache_hits"])
    log.count("context_prefix_tokens", stats["prefix_tokens"])
    log.count("service_entries", len(entries))
    return answers


def run_service_budget(layout, tenant, machinery, store, provider, log, items, *, view, tasks=None) -> tuple[list[Answer], dict]:
    """The served path: each question gets the prefix its scope family would get in production —
    the warm tier ranked scope-first, then tier, then recency, cut at the kernel's memory budget — as one cached
    prefix per family. `extracted` is coverage over the whole tier, `retrieved` over the served prefix, so the
    partition's "extracted but not retrieved" counts what the budget cut."""
    from ..store.budget import GENERAL, families_of_items, fit_prefix
    ttl = machinery.kernel.eval.arms.context_cache_ttl
    serve = machinery.kernel.serve
    entries, _ = _entries(layout, tenant, machinery, store, view, tasks)
    groups = families_of_items(items, serve.scopes)
    stats = {"calls": 0, "cache_hits": 0, "prefix_tokens": 0, "families": {}}
    answers = []
    for family in sorted(groups, key=lambda f: (f == GENERAL, f)):
        fit = fit_prefix(entries, None if family == GENERAL else family, serve, view)
        served = fit["entries"]
        prefix = warm_prefix(served, fit["manifest"])
        system = system_blocks(machinery, prefix, cache_material=True, ttl=ttl, tool_note=service_note(view).split(" Search first")[0])
        warm = provider.call(request(machinery, system=system, messages=(Message("user", (text("Reply with the single word: ready"),)),),
                                     job="eval_service_budget", custom_id=f"warm-{family}"))
        tokens = warm.usage.input + warm.usage.cache_write + warm.usage.cache_read
        stats["families"][family] = {"items": len(groups[family]), "entries": len(served), "matched": fit["matched"], "cut": fit["cut"],
                                     "estimated_tokens": fit["estimated_tokens"], "prefix_tokens": tokens}
        stats["prefix_tokens"] = max(stats["prefix_tokens"], tokens)
        for n, it in enumerate(groups[family]):
            a = Answer(it, material=f"budget:{family}", extracted=coverage(entries, it.episode), entries_seen=len(served))
            a.retrieved = coverage(served, it.episode) if it.episode else None
            try:
                resp = provider.call(request(machinery, system=system, messages=(Message("user", (text(f"Question: {it.question}"),)),),
                                             job="eval_service_budget", custom_id=it.id))
            except ProviderError as exc:
                a.failed = str(exc)
                answers.append(a)
                continue
            stats["calls"] += 1
            hit = resp.usage.cache_read > 0
            stats["cache_hits"] += hit
            if n == 0 and not hit:
                raise CacheMiss(f"the first item of family {family!r} after its warming call read nothing from the cache. Stopped.")
            a.text, a.cut_off = answer_of(resp)
            a.tokens_seen = resp.usage.input + resp.usage.cache_read + resp.usage.cache_write
            a.responses = 1
            log.emit("retrieval_trace", item=it.id, arm="service_budget", material=a.material, tokens_seen=a.tokens_seen)
            answers.append(a)
    log.count("context_calls", stats["calls"])
    log.count("context_cache_hits", stats["cache_hits"])
    log.count("context_prefix_tokens", stats["prefix_tokens"])          # the largest served prefix
    log.count("service_entries", len(entries))
    log.count("budget_families", len(groups))
    return answers, stats


def run_service_retrieval(layout, tenant, machinery, store, provider, log, items, *, view, tasks=None) -> tuple[list[Answer], dict]:
    """The served path: a cached core per scope family (globals, then live, then probation, by
    recency, inside `core_tokens`) and, per call, a retrieved slice — one BM25 shot of the question over the family's
    entries the core left out plus the ledger's open tasks, inside `slice_tokens`. The core caches; the slice is
    uncached. `retrieved` is coverage over core plus slice; the trace says which part held the cited span."""
    from ..store.budget import GENERAL, core_prefix, families_of_items, retrieved_slice
    ttl = machinery.kernel.eval.arms.context_cache_ttl
    serve = machinery.kernel.serve
    entries, _ = _entries(layout, tenant, machinery, store, view, tasks)
    groups = families_of_items(items, serve.scopes)
    stats = {"calls": 0, "cache_hits": 0, "prefix_tokens": 0, "families": {}, "slice_tokens_total": 0, "held": {"core": 0, "slice": 0, "neither": 0}}
    answers = []
    for family in sorted(groups, key=lambda f: (f == GENERAL, f)):
        fam = None if family == GENERAL else family
        core = core_prefix(entries, fam, serve, view)
        core_ids = {e.id for e in core["entries"]}
        prefix = warm_prefix(core["entries"], core["manifest"])
        note = service_note(view).split(" Search first")[0] + " A retrieved slice for this question follows the core."
        system = system_blocks(machinery, prefix, cache_material=True, ttl=ttl, tool_note=note)
        warm = provider.call(request(machinery, system=system, messages=(Message("user", (text("Reply with the single word: ready"),)),),
                                     job="eval_service_retrieval", custom_id=f"warm-{family}"))
        core_tokens = warm.usage.input + warm.usage.cache_write + warm.usage.cache_read
        fstats = {"items": len(groups[family]), "core_entries": len(core["entries"]), "core_estimated_tokens": core["estimated_tokens"],
                  "core_tokens": core_tokens, "slice_entries": 0, "slice_tokens": 0}
        stats["prefix_tokens"] = max(stats["prefix_tokens"], core_tokens)
        for n, it in enumerate(groups[family]):
            sl = retrieved_slice(it.question, entries, core_ids, fam, serve)
            tail = "===== retrieved for this question =====\n\n" + "\n\n".join(e.render() for e in sl["entries"]) if sl["entries"] else None
            served = core["entries"] + sl["entries"]
            a = Answer(it, material=f"retrieval:{family}", extracted=coverage(entries, it.episode), entries_seen=len(served))
            a.retrieved = coverage(served, it.episode) if it.episode else None
            where = None if not it.episode else "core" if coverage(core["entries"], it.episode) else "slice" if coverage(sl["entries"], it.episode) else "neither"
            if where:
                stats["held"][where] += 1
            fstats["slice_entries"] += len(sl["entries"])
            try:
                resp = provider.call(request(machinery, system=system_blocks(machinery, prefix, cache_material=True, ttl=ttl, tool_note=note, uncached_tail=tail),
                                             messages=(Message("user", (text(f"Question: {it.question}"),)),),
                                             job="eval_service_retrieval", custom_id=it.id))
            except ProviderError as exc:
                a.failed = str(exc)
                answers.append(a)
                continue
            stats["calls"] += 1
            hit = resp.usage.cache_read > 0
            stats["cache_hits"] += hit
            if n == 0 and not hit:
                raise CacheMiss(f"the first item of family {family!r} after its warming call read nothing from the cache. Stopped.")
            a.text, a.cut_off = answer_of(resp)
            a.tokens_seen = resp.usage.input + resp.usage.cache_read + resp.usage.cache_write
            fstats["slice_tokens"] += resp.usage.input                  # the uncached part of the call: the slice and the question
            stats["slice_tokens_total"] += resp.usage.input
            a.responses = 1
            log.emit("retrieval_trace", item=it.id, arm="service_retrieval", material=f"{a.material}:{where or 'noise'}", tokens_seen=a.tokens_seen,
                     calls=[{"terms": [], "total": sl["candidates"], "returned": [e.id for e in sl["entries"]], "scores": sl["scores"]}])
            answers.append(a)
        stats["families"][family] = fstats
    log.count("context_calls", stats["calls"])
    log.count("context_cache_hits", stats["cache_hits"])
    log.count("context_prefix_tokens", stats["prefix_tokens"])          # the largest cached core
    log.count("service_entries", len(entries))
    return answers, stats


# --- scoring an arm's answers -------------------------------------------------------------------------

def score(layout, tenant, machinery, provider, log, arm: str, answers: list[Answer], *, batch, wait,
          view: str | None = None) -> dict:
    ledger = Ledger(layout, tenant)
    j, r = machinery.artifact.judge, machinery.kernel.eval.reader
    rows, reqs, by_id = [], [], {}
    service = arm in SERVICE_ARMS
    for a in answers:
        it = a.item
        base = dict(item=it.id, item_key_hash=it.key_hash(), arm=arm, answer=a.text, judge_model_id=j.model,
                    judge_prompt_version=j.prompt_version, reader_model_id=machinery.artifact.reader.model,
                    reader_prompt_version=r.prompt_version, retrieved_cited_span=a.retrieved,
                    declared_nothing=declared_nothing(a.text) if a.text else None, hard=it.tests.hard,
                    extracted=a.extracted if service else None, entries_seen=a.entries_seen if service else None,
                    store_view=view if service else None)
        extra = {"extracted": a.extracted, "entries_seen": a.entries_seen} if service else {}
        scores = {d: None for d in ("yield", "noise", "supersession")}
        if a.failed or a.cut_off:
            # A refusal is the arm producing no answer, like a cut-off: scored wrong and counted, never judged
            # (as a cut-off answer is). Any other provider failure is unscorable and outside every n.
            refused = bool(a.failed) and a.failed.startswith("refusal")
            unscorable = bool(a.failed) and not refused
            scores[it.tests.type] = 0.0
            rationale = ("the reader refused: scored wrong, not judged" if refused else
                         a.failed or "cut off at the reader's output cap: scored wrong, not judged")
            log.emit("eval_verdict", verdict=False, scores=scores, unscorable=unscorable, cut_off=a.cut_off,
                     rationale=rationale, **base)
            rows.append({"type": it.tests.type, "hard": it.tests.hard, "verdict": False, "unscorable": unscorable,
                         "cut_off": a.cut_off, "refused": refused, "retrieved": a.retrieved, **extra})
            continue
        by_id[it.id] = (a, base)
        reqs.append(judge_request(machinery, question=it.question, reference=it.answer, answer=a.text, item_type=it.tests.type,
                                  superseded_answer=(ledger.state(it.id).drafted or {}).get("superseded_answer"), custom_id=it.id))
    for cid, resp in run_all(provider, reqs, batch=batch, wait=wait):
        a, base = by_id[cid]
        verdict, note = read_verdict(resp)
        scores = {d: None for d in ("yield", "noise", "supersession")}
        extra = {"extracted": a.extracted, "entries_seen": a.entries_seen} if service else {}
        if verdict is None:
            log.emit("eval_verdict", verdict=False, scores=scores, unscorable=True, rationale=note, **base)
            rows.append({"type": a.item.tests.type, "hard": a.item.tests.hard, "verdict": False, "unscorable": True,
                         "cut_off": False, "retrieved": a.retrieved, **extra})
            continue
        scores[a.item.tests.type] = 1.0 if verdict else 0.0
        log.emit("eval_verdict", verdict=verdict, scores=scores, rationale=note, **base)
        rows.append({"type": a.item.tests.type, "hard": a.item.tests.hard, "verdict": verdict, "unscorable": False,
                     "cut_off": False, "retrieved": a.retrieved, **extra})
    rep = report_rates(rows)
    if service:
        rep["coverage"] = coverage_split(rows)
    return rep


def _log_answers(machinery, log, arm: str, answers: list[Answer]) -> None:
    for a in answers:
        log.emit("reader_answer", item=a.item.id, arm=arm, text=a.text, cut_off=a.cut_off, failed=a.failed,
                 retrieved=a.retrieved, tokens_seen=a.tokens_seen, reader_model_id=machinery.artifact.reader.model,
                 reader_prompt_version=machinery.kernel.eval.reader.prompt_version, entries_seen=a.entries_seen or None)


def answers_from_log(records, arm: str, items: list[EvalItem], entries=None) -> list[Answer]:
    """Rebuild an arm's answers from a run's reader_answer records, so they can be scored again or scored
    for the first time by a run that outlived the one that paid for them. For a service arm, `extracted` is
    not on the answer record: it is recomputed over `entries` (the store as it stands now, which is said in
    the rescoring run's params); for the prefix arm it equals `retrieved` by construction."""
    by_id = {it.id: it for it in items}
    out = []
    for r in records:
        if r["kind"] == "reader_answer" and r["arm"] == arm and r["item"] in by_id:
            a = Answer(by_id[r["item"]], text=r["text"], cut_off=r["cut_off"], failed=r["failed"],
                       retrieved=r["retrieved"], tokens_seen=r["tokens_seen"], entries_seen=r.get("entries_seen") or 0)
            if arm == "service_context":
                a.extracted = a.retrieved
            elif arm == "service" and entries is not None:
                a.extracted = coverage(entries, a.item.episode)
            out.append(a)
    return out


def store_figures(rep: dict, entries: int, warm_tokens: int | None, memory_entries: int | None = None, task_entries: int | None = None) -> dict:
    """Yield density: yield per thousand warm-tier tokens and per hundred entries."""
    y = rep["rates"].get("yield", {}).get("rate")
    return {"entries": entries, "memory_entries": memory_entries, "task_entries": task_entries, "warm_tokens": warm_tokens,
            "yield_per_1k_tokens": round(y * 100 / (warm_tokens / 1000), 3) if y is not None and warm_tokens else None,
            "yield_per_100_entries": round(y * 100 / (entries / 100), 3) if y is not None and entries else None}


def score_answers(layout, tenant, machinery, provider, log, arm: str, answers: list[Answer], *,
                  batch: bool = True, wait=lambda: None, cache=None, view: str | None = None,
                  store_counts: tuple | None = None, pool: str = "founding", budget_stats: dict | None = None) -> dict:
    log.examined = len(answers)
    rep = score(layout, tenant, machinery, provider, log, arm, answers, batch=batch, wait=wait, view=view)
    if store_counts is not None:
        rep["store"] = store_figures(rep, *store_counts)
    budget_ = None
    if budget_stats is not None:
        budget_ = {"memory_budget_tokens": machinery.kernel.serve.memory_budget_tokens, "families": budget_stats["families"],
                   **{k: v for k, v in budget_stats.items() if k in ("held", "slice_tokens_total")}}
        if arm == "service_retrieval":
            budget_.update(core_tokens=machinery.kernel.serve.core_tokens, slice_tokens=machinery.kernel.serve.slice_tokens)
    log.emit("eval_report", arm=arm, items=len(answers), rates=rep["rates"], by_hard=rep["by_hard"], retrieval=rep["retrieval"],
             cut_off=rep["cut_off"], refused=rep["refused"], judge_failed=rep["judge_failed"], spent_usd=round(log.spent_usd, 4) if log.reported_spend_usd is not None else None, cache=cache,
             reader_model_id=machinery.artifact.reader.model, judge_model_id=machinery.artifact.judge.model,
             store_view=view, coverage=rep.get("coverage"), store=rep.get("store"), pool=pool, budget=budget_)
    return {**rep, "arm": arm, "items": len(answers), "cache": cache, "store_view": view, "pool": pool, "budget": budget_}


def run_arm(layout: StoreLayout, tenant: str, machinery: Machinery, provider, log, arm: str, *,
            batch: bool = True, wait=lambda: None, items: list[EvalItem] | None = None,
            store=None, view: str = "as_extracted", pool: str = "founding", purpose: str = "scoring", tasks=None) -> dict:
    """`pool` and `purpose` are the caller's: the harness names no guarded pool and no guarded purpose, so the
    pool guard stays with whoever opens the slice, and that is logged on the run."""
    if arm not in ARMS:
        raise ValueError(f"arm is one of {ARMS}")
    items = items if items is not None else load(layout, tenant, pool, purpose)
    figures = None
    if arm == "grep":
        answers = run_grep(layout, tenant, machinery, provider, log, items, batch=batch, wait=wait)
    elif arm == "context":
        answers = run_context(layout, tenant, machinery, provider, log, items)
    elif arm == "oracle":
        answers = run_oracle(layout, tenant, machinery, provider, log, items, batch=batch, wait=wait)
    else:
        if store is None:
            raise ValueError("a service arm needs the memory store")
        budget_stats = None
        if arm == "service":
            answers = run_service(layout, tenant, machinery, store, provider, log, items, view=view, batch=batch, wait=wait, tasks=tasks)
        elif arm == "service_budget":
            answers, budget_stats = run_service_budget(layout, tenant, machinery, store, provider, log, items, view=view, tasks=tasks)
        elif arm == "service_retrieval":
            answers, budget_stats = run_service_retrieval(layout, tenant, machinery, store, provider, log, items, view=view, tasks=tasks)
        else:
            answers = run_service_context(layout, tenant, machinery, store, provider, log, items, view=view, tasks=tasks)
        entries, man = _entries(layout, tenant, machinery, store, view, tasks)
        warm_tokens = log.counts.get("context_prefix_tokens") or None
        if warm_tokens is None:
            try:
                warm_tokens = provider.count_tokens(request(machinery, system=system_blocks(machinery, warm_prefix(entries, man)),
                                                            messages=(Message("user", (text("x"),)),), job=f"eval_{arm}", custom_id="count")) or None
            except Exception:   # a provider without a counter: the figure is simply absent
                warm_tokens = None
        figures = (entries, warm_tokens)
    _log_answers(machinery, log, arm, answers)
    cache = None
    if arm in ("context", "service_context", "service_budget", "service_retrieval"):
        cache = {"calls": log.counts.get("context_calls", 0), "hits": log.counts.get("context_cache_hits", 0),
                 "prefix_tokens": log.counts.get("context_prefix_tokens", 0)}
    if arm in SERVICE_ARMS:
        # A service run is a candidate run: every item in the pool has now been seen once more.
        increment_exposure(layout, tenant, pool, log, candidate=f"{arm}:{view}")
        from ..store.search import TaskEntry
        n_tasks = sum(isinstance(e, TaskEntry) for e in figures[0])
        return score_answers(layout, tenant, machinery, provider, log, arm, answers, batch=batch, wait=wait, cache=cache,
                             view=view, store_counts=(len(figures[0]), figures[1], len(figures[0]) - n_tasks, n_tasks), pool=pool,
                             budget_stats=budget_stats)
    return score_answers(layout, tenant, machinery, provider, log, arm, answers, batch=batch, wait=wait, cache=cache, pool=pool)


def verdicts_of(records: Iterator[dict], arm: str) -> dict[str, bool]:
    """Item -> verdict for one arm from a run's records, scorable items only."""
    return {r["item"]: r["verdict"] for r in records if r["kind"] == "eval_verdict" and r["arm"] == arm and not r["unscorable"]}
