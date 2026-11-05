"""The served path's prefix under a budget.

A session is served the warm tier filtered to its scope family and cut at the kernel's memory budget: scope
match first, then tier (live before probation, an open task with probation, superseded and closed tasks last),
then recency, cut where the next entry would not fit. Every step is a lookup or arithmetic; the chars-per-token
ratio is the kernel's measured number, so nothing here calls a model.
"""
from __future__ import annotations

import re
from math import ceil
from typing import Optional

from .search import Entry, TaskEntry, manifest

GENERAL = "general"     # a session no family claims: the whole tier competes on tier and recency alone

_TIER = {"live": 0, "contested": 1, "probation": 2, "proposed": 2, "superseded": 3}


def family_of(text: str, scopes) -> Optional[str]:
    """The family whose longest `match` string occurs in the text, case-insensitively ("how the agent works with
    its principal" beats "agent"); ties go to the family listed first. None when nothing matches."""
    low = (text or "").lower()
    best = None
    for i, fam in enumerate(scopes):
        for m in fam.match:
            if m.lower() in low and (best is None or (len(m), -i) > best[0]):
                best = ((len(m), -i), fam.name)
    return best[1] if best else None


def family_of_channel(channel: str, scopes) -> Optional[str]:
    low = (channel or "").lower()
    for fam in scopes:
        if any(c.lower() == low for c in fam.channels):
            return fam.name
    return None


def family_of_entry(entry, scopes) -> Optional[str]:
    """An entry's family: by its scope, then by its claim."""
    r = entry.record
    return family_of(r.scope, scopes) or family_of(r.claim, scopes)


def tier_rank(entry) -> int:
    if isinstance(entry, TaskEntry):
        return 2 if entry.record.status == "open" else 3
    return _TIER.get(entry.record.status.state, 3)


def estimate_tokens(text: str, chars_per_token: float) -> int:
    return ceil(len(text) / chars_per_token)


def rank_key(entry, family: Optional[str], scopes):
    fam = family_of_entry(entry, scopes)
    globals_ = {f.name for f in scopes if f.global_}
    matched = fam is not None and (fam == family or fam in globals_)
    return (0 if matched else 1, tier_rank(entry), entry.day or "", entry.id)


def fit_prefix(entries: list, family: Optional[str], serve, view: str) -> dict:
    """The entries a session of `family` is served, in rank order, cut at the budget. Returns the kept entries,
    the manifest built over them, the estimated token count, and how many were cut. The manifest's own
    estimate (over the whole tier, its upper bound) is reserved first."""
    ordered = sorted(entries, key=lambda e: (rank_key(e, family, serve.scopes)[:2], _later_first(e), e.id))
    reserve = estimate_tokens(manifest(entries, view), serve.chars_per_token) + 8   # the block headers
    budget = serve.memory_budget_tokens - reserve
    kept, used = [], 0
    for e in ordered:
        n = estimate_tokens(e.render(pointer=_pointer(serve)), serve.chars_per_token) + 1
        if used + n > budget:
            break
        kept.append(e); used += n
    man = manifest(kept, view)
    return {"entries": kept, "manifest": man, "estimated_tokens": used + estimate_tokens(man, serve.chars_per_token),
            "cut": len(entries) - len(kept), "matched": sum(rank_key(e, family, serve.scopes)[0] == 0 for e in kept)}


def _later_first(entry) -> str:
    """Sort helper: a later day sorts first (descending) inside the same rank and tier."""
    day = entry.day or ""
    return "".join(chr(0x10FFFF - ord(c)) for c in day) if day else "￿"


def family_of_question(question: str, scopes) -> Optional[str]:
    """A question's family by its words — never a global family: naming the principal says nothing about which
    project the question is about, so such a question is served the general prefix."""
    fam = family_of(question, scopes)
    if fam is not None and any(f.name == fam and f.global_ for f in scopes):
        return None
    return fam


def families_of_items(items, scopes) -> dict[str, list]:
    """Group eval items by the family their question names; the rest are `general`."""
    groups: dict[str, list] = {}
    for it in items:
        groups.setdefault(family_of_question(it.question, scopes) or GENERAL, []).append(it)
    return groups


# --- the served path as a cached core plus a retrieved slice ---------------------------------

_STOP = frozenset("""a an and are as at be but by for from has have how in is it its of on or that the this to was were what
when where which who why will with did do does not no yes we our you your i me my""".split())
_TOKEN = re.compile(r"[a-z0-9][a-z0-9_.\-/]*")


def tokens(text_: str) -> list[str]:
    """Lower-cased word tokens without stopwords; version strings, paths and hyphenated names stay whole."""
    return [w.strip(".-/_") for w in _TOKEN.findall((text_ or "").lower()) if w not in _STOP and len(w.strip(".-/_")) > 1]


def core_prefix(entries: list, family: Optional[str], serve, view: str, ledger: Optional[dict] = None) -> dict:
    """The cached core: the global families' memory entries, then the session family's `live`
    entries by recency, then its `probation` entries by recency, cut at `core_tokens`. Memory only; no superseded
    entry; the same for every call in the family, so it caches. `retired` never reaches a view."""
    globals_ = {f.name for f in serve.scopes if f.global_}
    fams = {e.id: family_of_entry(e, serve.scopes) for e in entries}
    memory = [e for e in entries if isinstance(e, Entry) and e.record.status.state != "superseded"]
    if getattr(serve, "mode", "normal") == "hybrid":      # standing rules, and the current value of what has changed
        memory = [e for e in memory if e.record.kind == "directive" or e.replaced]

    def band(e):
        fam, state = fams[e.id], e.record.status.state
        if fam in globals_:
            return 0
        if family is not None and fam != family:
            return None                                  # another family's entry never enters this family's core
        return 1 if state == "live" else 2               # contested and probation sit together after live
    ordered = sorted((e for e in memory if band(e) is not None), key=lambda e: (band(e), _later_first(e), e.id))
    reserve = estimate_tokens(manifest(entries, view, ledger=ledger), serve.chars_per_token) + 8
    budget = serve.core_tokens - reserve
    kept, used = [], 0
    for e in ordered:
        n = estimate_tokens(e.render(pointer=_pointer(serve)), serve.chars_per_token) + 1
        if used + n > budget:
            break
        kept.append(e); used += n
    man = manifest(kept, view, ledger=ledger)      # the core holds memory only; the ledger's counts come in by hand
    return {"entries": kept, "manifest": man, "estimated_tokens": used + estimate_tokens(man, serve.chars_per_token),
            "holds_all": len(kept) == len(ordered)}   # every memory entry the core could serve is in it


def bm25_rank(query: str, candidates: list, *, k1: float = 1.5, b: float = 0.75) -> list[tuple[float, object]]:
    """BM25 of the question over each candidate's claim, scope and anchor text (and notes), highest first; zero-score
    candidates are left out. Arithmetic over the candidate set, no index kept."""
    from collections import Counter
    from math import log
    q = tokens(query)
    if not q or not candidates:
        return []
    docs = []
    for e in candidates:
        r = e.record
        body = " ".join([r.claim, r.scope, r.anchor.quote or "", getattr(e, "notes", "") or "", *getattr(e, "replaced", [])])
        docs.append(Counter(tokens(body)))
    n = len(docs)
    avg = sum(sum(d.values()) for d in docs) / n
    df = Counter()
    for d in docs:
        for w in set(q):
            if w in d:
                df[w] += 1
    out = []
    for e, d in zip(candidates, docs):
        dl = sum(d.values()) or 1
        s = 0.0
        for w in q:
            if w not in d:
                continue
            idf = log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            tf = d[w]
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg))
        if s > 0:
            out.append((s, e))
    out.sort(key=lambda t: (-t[0], t[1].id))
    return out


def entry_version(e, serve=None) -> str:
    """A served entry's version: a digest of what it renders to, so a changed value, state or correction is
    a new version and is served again."""
    import hashlib
    return hashlib.sha256(e.render(pointer=_pointer(serve)).encode()).hexdigest()[:12]


def in_window(e, window: Optional[tuple]) -> bool:
    """the entry's source is still inside the carried window — the carried session's file, at or after the line
    the window starts on."""
    if not window:
        return False
    path, first = window
    ep = getattr(getattr(e.record, "anchor", None), "episode", "") or ""
    if "#L" not in ep or ep.split("#L", 1)[0] != path:
        return False
    try:
        return int(ep.rsplit("#L", 1)[1].split("-L")[0]) >= first
    except ValueError:
        return False


def retrieved_slice(question: str, entries: list, core_ids: set, family: Optional[str], serve,
                    visible: Optional[dict] = None, window: Optional[tuple] = None) -> dict:
    """The retrieved slice: one lexical shot of the question over every memory entry the core left out, across all
    families (family is the core's rule only), superseded excluded, plus the ledger's open tasks;
    the top scores fill `slice_tokens`. `visible` maps entry ids to the version the conversation already
    holds: those are left out, a changed one is not."""
    def eligible(e):
        if e.id in core_ids:
            return False
        if visible and visible.get(e.id) == entry_version(e, serve):
            return False
        if in_window(e, window):                         # the conversation still carries its source
            return False
        if isinstance(e, TaskEntry):
            return e.record.status == "open"
        return e.record.status.state != "superseded"
    ranked = bm25_rank(question, [e for e in entries if eligible(e)])
    kept, used = [], 0
    for score, e in ranked:
        n = estimate_tokens(e.render(pointer=_pointer(serve)), serve.chars_per_token) + 1
        if used + n > serve.slice_tokens:
            continue                                     # a long entry is skipped, a shorter one below it may still fit
        kept.append((score, e)); used += n
    return {"entries": [e for _, e in kept], "scores": [round(s, 3) for s, _ in kept], "estimated_tokens": used,
            "candidates": len(ranked)}


CORE_HEADER = "===== memory core ====="
SLICE_HEADER = "===== retrieved for this message ====="


def fit(entries: list, *, family: Optional[str], serve, view: str, question: Optional[str] = None,
        ledger: Optional[dict] = None) -> dict:
    """The served prefix as two text blocks (the public contract). `core_text` is the family's cached
    core — the manifest, then the entries in their stable order; `slice_text` is the one lexical shot of `question`
    over what the core left out, or "" with no question or no hit. Both carry their estimated token counts. The
    consumer places the cache breakpoints. Deterministic: identical store state, family, view and question give
    byte-identical blocks, which is what makes the core cacheable across calls."""
    core = core_prefix(entries, family, serve, view, ledger=ledger)
    core_ids = {e.id for e in core["entries"]}
    sl = retrieved_slice(question, entries, core_ids, family, serve) if question else {"entries": [], "scores": [], "estimated_tokens": 0, "candidates": 0}
    core_text = core["manifest"] + "\n\n" + CORE_HEADER + "\n\n" + "\n\n".join(e.render(pointer=_pointer(serve)) for e in core["entries"])
    slice_text = _slice_text(sl["entries"], serve)
    return {"core_text": core_text, "core_entries": core["entries"], "core_tokens": core["estimated_tokens"], "core_holds_all": core["holds_all"],
            "slice_text": slice_text, "slice_entries": sl["entries"], "slice_tokens": sl["estimated_tokens"],
            "slice_scores": sl["scores"], "candidates": sl["candidates"]}


def _pointer(serve) -> bool:
    """Memory v2's served form: a pointer to the source in place of the quote."""
    return getattr(serve, "provenance", "quote") == "pointer"


def _slice_text(entries: list, serve=None) -> str:
    return (SLICE_HEADER + "\n\n" + "\n\n".join(e.render(pointer=_pointer(serve)) for e in entries)) if entries else ""


def slice_for(question: str, entries: list, core_ids: set, *, family: Optional[str], serve,
              visible: Optional[dict] = None, window: Optional[tuple] = None) -> dict:
    """The slice alone, against a core the consumer already holds: a consumer that keeps one core
    for a whole session, so its cached prefix never moves, passes that core's entry ids and gets this message's slice,
    rendered as `fit` renders it, to place after the prefix. With `serve.serve_once` the consumer also passes
    the versions the conversation already holds (`visible`), and gets back the versions this slice adds."""
    seen = visible if getattr(serve, "serve_once", False) else None
    win = window if getattr(serve, "mode", "normal") == "hybrid" else None
    sl = retrieved_slice(question, entries, core_ids, family, serve, visible=seen, window=win) if question else {"entries": [], "scores": [], "estimated_tokens": 0}
    return {"slice_text": _slice_text(sl["entries"], serve), "slice_entries": sl["entries"], "slice_tokens": sl["estimated_tokens"],
            "slice_scores": sl["scores"], "slice_versions": {e.id: entry_version(e, serve) for e in sl["entries"]}}


def corrections_for(core_entries: list, entries: list, done: set, serve=None) -> tuple[str, list]:
    """the core is fitted once a session, so a fact corrected mid-session stays wrong there. For each core entry
    now superseded, its successor is appended to the conversation once ("Correction to the memory core"), instead of
    the core being rewritten. Returns (text, successors added); `done` holds core ids already corrected."""
    by_id = {e.id: e for e in entries}
    lines, added = [], []
    for c in core_entries:
        if c.id in done or isinstance(c, TaskEntry):
            continue
        cur = by_id.get(c.id)
        nxt = getattr(getattr(getattr(cur, "record", None), "status", None), "superseded_by", None) if cur else None
        if cur is None or cur.record.status.state != "superseded" or nxt not in by_id:
            continue
        done.add(c.id)
        added.append(by_id[nxt])
        lines.append("Correction to the memory core: " + by_id[nxt].render(pointer=_pointer(serve)))
    return "\n\n".join(lines), added
