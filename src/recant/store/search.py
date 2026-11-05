"""Search over the warm tier: what the memory service's own retrieval loop calls.

Lexical and mechanical. Every term must be on the entry (claim, anchor quote, scope, notes), the way the
grep arm's tool works, so the service arm differs from grep in what is searched and nothing else. Status
travels with the entry: probation flagged, contested down-ranked with its challenge, superseded never
returned without what replaced it, retired never returned, proposed only in the as-extracted view."""
from __future__ import annotations

import re

from dataclasses import dataclass, field
from typing import Optional

from ..eval.search import plain
from ..records import MemoryRecord, TaskRecord, is_ulid, parse_episode_id
from .memory import MemoryStore
from .tasks import TaskStore

# `superseded` is in every view: returned only to a historical question, or when
# nothing current answers, and always marked "superseded on <date> by <claim>". Never as current.
VIEWS = {"as_extracted": ("proposed", "probation", "live", "contested", "superseded"),
         "as_reviewed": ("probation", "live", "contested", "superseded")}
FLAGS = {"proposed": "proposed, unreviewed", "probation": "accepted, in probation", "live": "live", "contested": "contested"}   # probation is accepted
_HISTORICAL = ("originally", "at the time", "before", "previously", "used to", "earlier", "first", "initially", "was the",
               "what did", "until", "prior", "old", "former")


def historical(terms: list[str]) -> bool:
    """Does the query ask about a past state: a date, or one of the words that reach back?"""
    import re
    joined = " ".join(t.lower() for t in terms if isinstance(t, str))
    return bool(re.search(r"\b20\d\d-\d\d(-\d\d)?\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", joined)) \
        or any(w in joined for w in _HISTORICAL)


@dataclass
class Entry:
    record: MemoryRecord
    notes: str
    trust_rank: int
    replaced: list[str] = field(default_factory=list)   # claims of entries this one superseded
    successor: Optional["MemoryRecord"] = None          # for a superseded entry: what replaced it
    unresolved: list[str] = field(default_factory=list)  # claims this one may replace or conflict with, unreviewed

    @property
    def id(self) -> str:
        return self.record.id

    @property
    def day(self) -> str:
        return (self.record.anchor.timestamp or "")[:10]

    def haystack(self) -> str:
        r = self.record
        return plain(" ".join([r.claim, r.anchor.quote, r.scope, self.notes, r.anchor.timestamp or "", *self.replaced]))

    def render(self, pointer: bool = False) -> str:
        """The entry as served. `pointer` (memory v2): the claim, its date and a pointer to its source in
        place of the quote, which the memory tool fetches when it is needed; a changed fact shows what it replaced."""
        r = self.record
        if pointer:
            flag = FLAGS.get(r.status.state, r.status.state)
            if r.status.state == "superseded":
                s = self.successor
                flag = (f"SUPERSEDED on {(s.anchor.timestamp or '')[:10] or 'a later date'} by: {s.claim}" if s else "SUPERSEDED") + " — not current"
            since = f" (since {self.day}; before that: {'; '.join(self.replaced)})" if self.replaced and self.day else \
                (f" (before that: {'; '.join(self.replaced)})" if self.replaced else "")
            if self.unresolved:                      # an uncertain key match is served as unresolved, both sides
                flag += " — UNRESOLVED against: " + "; ".join(self.unresolved) + " (check before relying on either)"
            return (f"[{r.id}] ({flag}) {r.claim}{since}\n"
                    f"  {r.kind} · {r.actor.cls} · {r.scope} · {self.day or 'undated'} · source {r.anchor.episode}")
        flag = FLAGS.get(r.status.state, r.status.state)
        if r.status.state == "contested" and r.status.reason:
            flag += f": {r.status.reason}"
        if r.status.state == "superseded":
            s = self.successor
            flag = (f"SUPERSEDED on {(s.anchor.timestamp or '')[:10] or 'a later date'} by: {s.claim}" if s
                    else "SUPERSEDED by an entry no longer in the store") + " — not current"
        lines = [f"[{r.id}] ({flag}) {r.kind} · {r.actor.cls} · scope: {r.scope}",
                 f"  {r.claim}",
                 f"  anchor {r.anchor.episode}{' · ' + r.anchor.timestamp if r.anchor.timestamp else ''}: \"{r.anchor.quote}\"",
                 f"  until: {r.supersession_condition}"]
        for old in self.replaced:
            lines.append(f"  replaced an earlier entry that said: {old}")
        return "\n".join(lines)


@dataclass
class TaskEntry:
    """A ledger task, searched alongside memory. True while open; shown closed with its
    date, never as if open; stale when open too long with no closing line."""
    record: TaskRecord
    notes: str
    trust_rank: int
    stale: bool = False
    presumed_dropped: bool = False      # the live ledger's second threshold: kept as history, never as open
    replaced: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.record.id

    @property
    def day(self) -> str:
        return self.record.opened

    def haystack(self) -> str:
        r = self.record
        return plain(" ".join([r.claim, r.anchor.quote, r.scope, self.notes, r.opened, r.closed or ""]))

    @property
    def waiting(self) -> bool:
        return self.record.status == "open" and self.record.waiting_on == "principal"

    def render(self, pointer: bool = False) -> str:          # a task renders in full either way (pointer serving is for memories)
        r = self.record
        if r.status == "open":
            flag = (f"task, presumed dropped: opened {r.opened}, no closing line" if self.presumed_dropped else
                    f"question, waiting on the principal since {r.opened}" if self.waiting else
                    f"task, open, unconfirmed since {r.opened}: no closing line" if self.stale else f"task, open since {r.opened}")
        else:
            flag = f"task, {r.status} {r.closed}" + (f" by {r.closed_by}" if r.closed_by else "")
        return "\n".join([f"[{r.id}] ({flag}) {r.actor.cls} · scope: {r.scope}", f"  {r.claim}",
                          f"  anchor {r.anchor.episode}: \"{r.anchor.quote}\""]
                         + [f"  note {n.day} ({n.by}): {n.text}" for n in r.notes])


def load_tasks(tasks: TaskStore, tenant: str, trust_rank, *, stale_after_days: int, today: Optional[str] = None,
               presumed_dropped_days: Optional[int] = None) -> list[TaskEntry]:
    """Every task, open first. `today` defaults to the latest day the ledger knows: the corpus is historical,
    so staleness is measured against its own end, not the wall clock. The live runtime passes the wall clock
    and its two thresholds: stale after `stale_after_days`, presumed dropped after
    `presumed_dropped_days`."""
    from datetime import date
    out = []
    for t in tasks.list(tenant):
        rec, notes = tasks.get(tenant, t["id"])
        out.append(TaskEntry(rec, notes, trust_rank("directive", rec.actor.cls)))
    days = [e.record.closed or e.record.opened for e in out]
    today = today or (max(days) if days else None)
    if today:
        end = date.fromisoformat(today)
        for e in out:
            if e.record.status != "open":
                continue
            age = (end - date.fromisoformat(e.record.opened)).days
            if presumed_dropped_days is not None and age > presumed_dropped_days:
                e.presumed_dropped = True
            if age > stale_after_days:
                e.stale = True
    out.sort(key=lambda e: (e.record.status != "open", e.presumed_dropped, e.stale))   # open first, presumed-dropped last among them
    return out


def load_entries(store: MemoryStore, tenant: str, view: str, trust_rank) -> list[Entry]:
    """The entries a view can see, each with its trust rank and the claims it superseded attached."""
    if view not in VIEWS:
        raise ValueError(f"view is one of {sorted(VIEWS)}")
    visible, all_recs = {}, {}
    for e in store.list(tenant):
        rec, notes = store.get(tenant, e["id"])
        all_recs[rec.id] = rec
        if rec.status.state in VIEWS[view]:
            visible[rec.id] = Entry(rec, notes, trust_rank(rec.kind, rec.actor.cls))
    for ent in list(visible.values()):
        old = ent.record
        if old.status.state == "superseded":
            ent.successor = all_recs.get(old.status.superseded_by)
            if old.status.superseded_by in visible:      # the successor also carries the old text
                visible[old.status.superseded_by].replaced.append(old.claim)
    for ent in list(visible.values()):                   # "Unresolved with <id>" in the notes marks both sides
        m = re.search(r"Unresolved with ([0-9A-HJKMNP-TV-Z]{26})", ent.notes or "")
        other = visible.get(m.group(1)) if m else None
        if other is not None and other.record.status.state != "superseded":
            ent.unresolved.append(other.record.claim)
            other.unresolved.append(ent.record.claim)
    return list(visible.values())


def search_memory(entries: list[Entry], terms: list[str], *, max_entries: int, max_chars: int) -> tuple[str, list[str]]:
    """(text for the reader, ids returned). Every term must match; ranked by matches, then trust rank
    (contested pushed down), then recency; cut at the caps."""
    needles = [plain(t) for t in terms if isinstance(t, str) and plain(t)]
    if not needles:
        return "No terms given.", []
    scored, past = [], []
    for e in entries:
        hay = e.haystack()
        if all(n in hay for n in needles):
            matches = sum(hay.count(n) for n in needles)
            if isinstance(e, TaskEntry):
                rank = e.trust_rank + (0 if e.record.status == "open" else 5)   # open tasks first, then closed with their dates
            elif e.record.status.state == "superseded":
                past.append((-matches, e.trust_rank + 20, e))                   # only for a historical question, or last resort
                continue
            else:
                rank = e.trust_rank + (10 if e.record.status.state == "contested" else 0)
            scored.append((-matches, rank, e))
    if past and (historical(terms) or not scored):
        scored.extend(past)
    scored.sort(key=lambda s: s[2].day, reverse=True)      # recency last: a later day first among equals (stable sorts)
    scored.sort(key=lambda s: (s[0], s[1]))                 # then matches, then trust rank with contested pushed down
    out, ids, used = [], [], 0
    for _, _, e in scored:
        block = e.render()
        if len(ids) >= max_entries or used + len(block) > max_chars:
            break
        out.append(block)
        ids.append(e.id)
        used += len(block)
    total = len(scored)
    head = (f"{total} matching entr{'y' if total == 1 else 'ies'}; showing {len(ids)}." if total else "No entry carries every term.")
    return head + ("\n\n" + "\n\n".join(out) if out else ""), ids


def ledger_counts(tasks: list) -> dict:
    """The live ledger's counts: open, open but unconfirmed (stale), presumed dropped, closed.
    Over every task, including the presumed-dropped ones the served entries leave out."""
    t = [e for e in tasks if isinstance(e, TaskEntry)]
    return {"total": len(t),
            "open": sum(e.record.status == "open" and not e.stale and not e.presumed_dropped and not e.waiting for e in t),
            "unconfirmed": sum(e.record.status == "open" and e.stale and not e.presumed_dropped and not e.waiting for e in t),
            "waiting_on_principal": sum(e.waiting and not e.presumed_dropped for e in t),   # listed apart from the agent's own work
            "presumed_dropped": sum(e.presumed_dropped for e in t),
            "closed": sum(e.record.status != "open" for e in t)}


def _ledger_clause(tasks: list, ledger: Optional[dict]) -> str:
    if ledger is not None:
        if not ledger["total"]:
            return "; task ledger: empty"
        return (f"; task ledger: {ledger['total']} tasks ({ledger['open']} open, {ledger['unconfirmed']} open but unconfirmed, "
                f"{ledger.get('waiting_on_principal', 0)} waiting on the principal, "
                f"{ledger['presumed_dropped']} presumed dropped — history, not open, {ledger['closed']} closed)")
    return (f"; task ledger: {len(tasks)} tasks ({sum(t.record.status == 'open' for t in tasks)} open, "
            f"{sum(t.stale for t in tasks)} stale)" if tasks else "; task ledger: empty")


def manifest(entries: list, view: str, top_scopes: int = 12, ledger: Optional[dict] = None) -> str:
    """What the stores hold, in a few lines, so the reader knows what it can ask for. Memory and the
    task ledger are named separately; a search returns hits from both. `ledger` is the live ledger's counts
    (`ledger_counts`) when the caller serves a core that holds no task entries: the line then
    says, for example, "100 tasks, all presumed dropped, none open" rather than "empty"."""
    from collections import Counter
    mem = [e for e in entries if isinstance(e, Entry)]
    tasks = [e for e in entries if isinstance(e, TaskEntry)]
    states = Counter(e.record.status.state for e in mem)
    kinds = Counter(e.record.kind for e in mem)
    scopes = Counter(e.record.scope.strip() for e in entries)
    days = sorted(e.day for e in entries if e.day)
    lines = [f"Memory store, view {view}: {len(mem)} entries" + _ledger_clause(tasks, ledger)
             + (f"; anchored {days[0]} to {days[-1]}" if days else "") + ".",
             "Memory by state: " + (", ".join(f"{k} {v}" for k, v in states.most_common()) or "none") + ".",
             "Memory by kind: " + (", ".join(f"{k} {v}" for k, v in kinds.most_common()) or "none") + ".",
             "Most common scopes: " + "; ".join(f"{s} ({n})" for s, n in scopes.most_common(top_scopes)) + "."]
    return "\n".join(lines)


def covers(entry, episode: Optional[str]) -> bool:
    """Does this entry anchor in, or derive from, the cited span (same file, overlapping lines)?"""
    if not episode:
        return False
    want = parse_episode_id(episode)
    for ref in [entry.record.anchor.episode, *entry.record.derived_from]:
        if is_ulid(ref):
            continue
        r = parse_episode_id(ref)
        if r.path == want.path and r.start <= want.end and want.start <= r.end:
            return True
    return False


def coverage(entries: list, episode: Optional[str]) -> bool | None:
    """None for an item with no cited span (noise); else whether any visible entry covers it."""
    if not episode:
        return None
    return any(covers(e, episode) for e in entries)
