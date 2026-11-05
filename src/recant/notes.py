"""Notes mode: a structured notes file the agent keeps itself while that is cheaper than memory,
the priced switch to memory, and the migration that turns the notes into memory items with no model call.

Entries. A note is one entry: a key (scope, entity, attribute), a value, the date it holds from, and its source
— the principal's own words (a quote that must appear in one of the principal's turns of the session) or a tool result (a
call that must have been made in the session). A new value for a key keeps the earlier one as history. The evidence
rule is checked here, in code, before anything is written: an entry with no source, an absence, a quote that is not
in the principal's turns or a call that was not made is refused, counted and kept apart; it is never loaded.

The switch. Notes cost a session T × p × (w + r·(n − 1)): a block always loaded, written to the cache once and read
on the session's n − 1 later calls. Memory costs extraction a session,
plus the same for min(T, core) — what memory would serve of the same facts, never the whole store — plus the slice budget
uncached on every call. The store switches when notes cost more for `runs` sessions
running; until the host has `min_sessions` of its own data, the default rule is the notes rendering larger than the
configured core plus slice. One way only: a store that switched stays in memory mode.

Migration: each current entry becomes a memory record (format 1, as `recant.extract` writes one) and each earlier
value a record superseded by it; no model call. An entry that cannot be anchored is returned apart, for extraction.

Blocks. Managed notes are memory's default form; extraction is the fallback. One notes block per scope
(user, team, org): a conversation loads its user's block and the shared blocks it may read, each rendered byte-identically
across conversations so the cache is shared; writes go to one block, the user's by default, a shared one only with the
host's permission. A call may carry many entries, each checked on its own. Two writers to one key in one window,
with different values, are both kept and served as unresolved until a later single write settles it. A tool note
whose cited line carries the tool's output must find its value there. The switch's default is a measured crossover,
95k counted tokens (see the CHANGELOG, 0.1.0).

The library knows nothing about the host: the consumer passes the session's lines, the model's price profile and the
host's call shape, and places the rendered block in its prompt. Nothing here is a cap.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator
from ulid import ULID

NOTES_FORMAT_VERSION = 1
HEADER = "===== notes ====="
INSTRUCTION = (
    "You keep structured notes with the note_fact tool; they are shown to you at the start of every session under "
    f"'{HEADER}', one line per scope and entity, as attribute=value (date; source). Whenever the principal states a fact "
    "about their world, a standing rule, or a change to either, and whenever a tool result gives you a durable fact, "
    "record it with note_fact before you reply — several entries in one call — each with its scope, entity and attribute, "
    "the value as the principal or the tool gave it, the date it holds from, and its source: the principal (the value "
    "must appear in what they said) or the tool call it came from. A new value for a key replaces the earlier one, which "
    "stays as history. Answer from your notes; read_notes gives an entry's full source when you need the words.")
EXTRACTION_JOBS = ("nominate", "extract", "extract_tasks")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class NoteSource(_Strict):
    kind: Literal["principal", "tool"]
    quote: Optional[str] = None          # principal: the words, exactly as said
    call: Optional[str] = None           # tool: the call, as the session's tool line records it
    episode: Optional[str] = None        # the session line the quote or call was found on, as an episode id


class NoteEntry(_Strict):
    """One note (notes format 1). `key` is `<scope>::<entity>::<attribute>`, normalized as memory v2's keys are."""
    id: str
    key: str
    scope: str
    entity: Optional[str] = None
    attribute: str
    value: str
    as_of: str
    source: NoteSource
    written: str                        # the day it was written, so the order of values for a key is the order written
    agent: Optional[str] = None         # who wrote it, when more than one agent writes a block
    written_ts: Optional[str] = None    # when, to the second, for the concurrent-writer window

    @field_validator("id")
    @classmethod
    def _ulid(cls, v: str) -> str:
        from .records import is_ulid
        if not is_ulid(v):
            raise ValueError("id must be a ULID")
        return v

    @field_validator("scope", "attribute", "value", "as_of")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v


# ---------------------------------------------------------------- the evidence rule, in code, applied to notes
def _plain(s: str) -> str:
    """Words only: quotes, backticks and spacing differ between what the agent writes and what the line holds."""
    return re.sub(r"\s+", " ", re.sub(r"[`\"'’“”]", "", s or "")).strip().lower().strip(" .")


def parts(value) -> list[str]:
    """The checkable parts of a value: its numbers, codes and words of two characters or more, each in
    plain form. "EX0002, 2000.00" is ["ex0002", "2000.00"]: a composite value passes when every part is in the source."""
    out = [_plain(m) for m in re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-/%@:]*", str(value or ""))]
    return [x for x in out if len(x) >= 2] or ([_plain(str(value))] if _plain(str(value)) else [])


def holds(text_: str, value) -> bool:
    """Every part of `value` appears in `text_` (plain forms)."""
    t = _plain(text_)
    ps = parts(value)
    return bool(ps) and all(x in t for x in ps)


def check(entry: dict, lines: Iterable, episode_path: str) -> tuple[Optional[str], Optional[str]]:
    """(the reason this entry is refused, or None; the episode id of the line its source was found on). `lines` are the
    session's lines (`recant.store.hot.Line`: role, n, end, text), `episode_path` the session file's episode path.
    A principal note with a quote is anchored where the quote appears in a principal's turn; one without a quote (the
    agent need not copy the sentence) is anchored on the principal's turn that holds every part of the
    value, and that line's text becomes its stored quote (`entry["source"]["quote"]` is filled in). A tool note is
    anchored on the call's line, and where the line carries the call's output every part of the value must be in it."""
    from .harness.extract import ABSENCE
    src = entry.get("source") or {}
    kind = src.get("kind")
    if kind not in ("principal", "tool"):
        return "no source", None
    for field_ in ("value", "attribute"):
        if ABSENCE.search(str(entry.get(field_) or "")):
            return "absence", None                            # an absence is never evidence
    if kind == "principal":
        q = _plain(src.get("quote") or "")
        if q and len(q) < 3:
            return "no quote", None
        for ln in lines:
            if ln.role != "user":
                continue
            if q and q in _plain(ln.text):
                return None, f"{episode_path}#L{ln.n}-L{ln.end}"
            if not q and holds(ln.text, entry.get("value")):
                src["quote"] = ln.text.split(": ", 1)[-1][:300]          # the principal's words, kept in the store, not written by the agent
                return None, f"{episode_path}#L{ln.n}-L{ln.end}"
        return ("quote not in the principal's turns" if q else "value not in the principal's turns"), None
    c = _plain(src.get("call") or "")
    if not c:
        return "no call", None
    for ln in lines:
        if ln.role == "tool" and not getattr(ln, "failed", False) and c in _plain(ln.text):
            if OUTPUT_MARK in ln.text and not holds(ln.text.split(OUTPUT_MARK, 1)[1], entry.get("value")):
                return "value not in the tool's output", None    # the record carries the output: every part of the value must be in it
            return None, f"{episode_path}#L{ln.n}-L{ln.end}"
    return "call not made in this session", None


OUTPUT_MARK = "\n⟶ "         # what a host puts between a tool line's summary and the bounded output it keeps (`recant.notes.record_output`)


def record_output(summary: str, output: str, chars: int) -> str:
    """The tool line a host writes when it keeps a bounded part of the output (opt-in): the summary, the mark,
    then the first `chars` characters of the output; the summary alone when `chars` is 0."""
    if chars <= 0 or not output:
        return summary
    cut = output[:chars]
    return summary + OUTPUT_MARK + cut + ("" if len(output) <= chars else f" … (+{len(output) - chars} chars)")


def check_many(entries: list, lines: Iterable, episode_path: str) -> list[tuple[Optional[str], Optional[str]]]:
    """one call, many entries, each checked on its own: a list of (reason, anchor) in the entries' order."""
    lines = list(lines)
    return [check(e, lines, episode_path) for e in entries]


# ---------------------------------------------------------------- the store: append-only, one directory per block
SHARED_KINDS = ("team", "org")
UNRESOLVED_WINDOW_MINUTES = 120        # two writers to one key this close in time, with different values, are both kept


def block_dir_name(block: Optional[str]) -> str:
    """A block id ("user:owner", "team:ops", "org") as a directory name; None is the legacy single block, the root."""
    return "" if not block else re.sub(r"[^A-Za-z0-9_.-]+", "_", block)


def is_shared(block: Optional[str]) -> bool:
    return bool(block) and block.split(":", 1)[0] in SHARED_KINDS


class NotesStore:
    """One block's notes (a block per scope — the user's, a team's, the org's; None is the legacy single block):
    `entries.jsonl` (one line per entry; the latest for a key is current, earlier ones history), `refused.jsonl` (what the
    checks refused, with the reason, for the report), `mode.json` (notes or memory, and the switch's state, per block).
    Files are canonical; nothing here is an index."""

    def __init__(self, layout, tenant: str, block: Optional[str] = None):
        self.block = block
        self.dir: Path = layout.notes_dir(tenant) / block_dir_name(block) if block else layout.notes_dir(tenant)
        self.entries_path, self.refused_path, self.mode_path = self.dir / "entries.jsonl", self.dir / "refused.jsonl", self.dir / "mode.json"

    def _append(self, path: Path, row: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def add(self, entry: dict, *, today: str, agent: Optional[str] = None, ts: Optional[str] = None) -> NoteEntry:
        from .harness.extract import key_of
        e = dict(entry)
        e.setdefault("id", str(ULID()))
        e["written"] = today
        e.setdefault("as_of", today)
        if agent is not None:
            e["agent"] = agent
        if ts is not None:
            e["written_ts"] = ts
        e["key"] = key_of(e.get("scope", ""), str(e.get("attribute") or ""), (e.get("entity") or None)) or ""
        if not e["key"]:
            raise ValueError("a note needs a scope and an attribute")
        rec = NoteEntry.model_validate(e)
        self._append(self.entries_path, rec.model_dump(mode="json"))
        return rec

    def refuse(self, entry: dict, reason: str, *, today: str) -> None:
        self._append(self.refused_path, {"entry": entry, "reason": reason, "day": today})

    def entries(self) -> list[NoteEntry]:
        if not self.entries_path.exists():
            return []
        return [NoteEntry.model_validate(json.loads(l)) for l in self.entries_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def refused(self) -> list[dict]:
        if not self.refused_path.exists():
            return []
        return [json.loads(l) for l in self.refused_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def current(self, *, window_minutes: int = UNRESOLVED_WINDOW_MINUTES) -> list[tuple[NoteEntry, list[NoteEntry]]]:
        """(the current entry, its earlier values oldest first) per key, keys in order of first appearance. When
        the last two values for a key came from different agents within `window_minutes` of each other and differ,
        neither is current: the key is unresolved, and `unresolved(key)` gives both. A later single write resolves it."""
        by: dict[str, list[NoteEntry]] = {}
        for e in self.entries():
            by.setdefault(e.key, []).append(e)
        return [(vs[-1], vs[:-1]) for vs in by.values()]

    def unresolved(self, *, window_minutes: int = UNRESOLVED_WINDOW_MINUTES) -> dict[str, tuple[NoteEntry, NoteEntry]]:
        """key -> (earlier, later) for every key whose last two values are two agents' different values within the
        window; a single later write (one agent, outside the window, or the same agent) resolves it."""
        out = {}
        by: dict[str, list[NoteEntry]] = {}
        for e in self.entries():
            by.setdefault(e.key, []).append(e)
        for k, vs in by.items():
            if len(vs) < 2:
                continue
            a, b = vs[-2], vs[-1]
            if a.agent and b.agent and a.agent != b.agent and a.value != b.value and a.written_ts and b.written_ts \
                    and abs(_minutes(a.written_ts, b.written_ts)) <= window_minutes:
                out[k] = (a, b)
        return out

    def mode(self) -> dict:
        try:
            return json.loads(self.mode_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"mode": "notes", "sessions": 0, "over_run": 0}

    def save_mode(self, state: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mode_path.write_text(json.dumps(state, indent=1) + "\n", encoding="utf-8")


def _minutes(a: str, b: str) -> float:
    from datetime import datetime
    return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds() / 60.0


def source_ref(e: NoteEntry) -> str:
    """A short source reference: the session's day and line, with "tool" for a tool result; the full
    quote stays in the store and is read with the notes tool when needed."""
    ep = e.source.episode or ""
    m = re.search(r"/(\d{4}-(\d{2}-\d{2}))/[^#]*#L(\d+)", ep)
    ref = f"{m.group(2)} L{m.group(3)}" if m else (e.as_of[5:] if len(e.as_of) >= 10 else e.as_of)
    return ref + (" tool" if e.source.kind == "tool" else "")


def render(current: list[tuple[NoteEntry, list[NoteEntry]]], *, block: Optional[str] = None,
           unresolved: Optional[dict] = None) -> str:
    """The notes as served (terse): one line per scope and entity, its attributes as `attribute=value
    (date; source)`, earlier values after "was", sorted by scope, entity and attribute so the block is byte-stable across
    sessions and conversations with the same notes (it is the cached prefix, shared when the block is). An
    unresolved key shows both values, each with its writer and time. The full quotes stay in the store."""
    lines = [HEADER if not block else f"===== notes ({block}) ====="]
    unresolved = unresolved or {}
    groups: dict[tuple, list] = {}
    for cur, hist in sorted(current, key=lambda ch: (ch[0].scope, ch[0].entity or "", ch[0].attribute)):
        if cur.key in unresolved:
            a, b = unresolved[cur.key]
            val = f"unresolved {a.value} ({a.agent} {a.written_ts}) | {b.value} ({b.agent} {b.written_ts})"
            hist = hist[:-1]
        else:
            val = f"{cur.value} ({cur.as_of}; {source_ref(cur)})"
        if hist:
            val += "; was " + ", ".join(f"{h.value} ({h.as_of})" for h in reversed(hist))
        groups.setdefault((cur.scope, cur.entity or "-"), []).append(f"{cur.attribute}={val}")
    for (scope, entity), attrs in groups.items():
        lines.append(f"{scope}/{entity}: " + "; ".join(attrs))
    if len(lines) == 1:
        lines.append("(no notes yet)")
    return "\n".join(lines)


def render_blocks(layout, tenant: str, blocks: list) -> tuple[str, int]:
    """the blocks a conversation loads, each rendered on its own and joined; (text, entries). A block that has
    switched to memory renders nothing here (memory serves it); one in "both" renders its notes beside memory."""
    parts, n = [], 0
    for b in blocks:
        ns = NotesStore(layout, tenant, b)
        if ns.mode().get("mode", "notes") not in ("notes", "both"):
            continue
        cur = ns.current()
        parts.append(render(cur, block=b, unresolved=ns.unresolved()))
        n += len(cur)
    return "\n\n".join(parts), n


# ---------------------------------------------------------------- the switch, priced per host
@dataclass(frozen=True)
class HostFigures:
    """What the switch needs to know about the host, from its own metered calls: how many sessions it has seen, the
    model calls a session makes, and what extraction costs a session (the consumer's prior until this host has run it)."""
    sessions: int
    calls_per_session: float
    extraction_usd_per_session: float
    source: str = ""


def figures_from_log(layout, tenant: str, *, extraction_usd_default: float) -> HostFigures:
    """The host's figures from the tenant's log: sessions are the distinct runs that made `turn` calls; extraction is
    the metered cost of its extraction jobs per session, else the consumer's default."""
    runs, calls, extraction, ext_runs = set(), 0, 0.0, set()
    for p in sorted(layout.log_dir(tenant).glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if '"model_call"' not in line:
                continue
            r = json.loads(line)
            if r.get("kind") != "model_call":
                continue
            if r.get("job") == "turn":
                runs.add(r.get("run")); calls += 1
            elif r.get("job") in EXTRACTION_JOBS:
                extraction += r.get("cost_usd") or 0.0; ext_runs.add(r.get("run"))
    n = len(runs)
    per = (extraction / n) if (ext_runs and n) else extraction_usd_default
    return HostFigures(sessions=n, calls_per_session=(calls / n) if n else 0.0, extraction_usd_per_session=per,
                       source=("metered" if ext_runs else "consumer's default") + f" over {n} sessions")


def notes_usd(tokens: int, profile, calls: float) -> float:
    """The price of a block that is always loaded: written to the cache on the session's first call, read on the rest."""
    return tokens * profile.p * (profile.w + profile.r * max(calls - 1, 0))


def memory_usd(tokens: int, profile, calls: float, *, core_tokens: int, slice_tokens: int, extraction_usd: float) -> float:
    """Extraction a session; the core (what fits of the same facts, at most the core ceiling) served as the notes would
    be; and the slice, which every call retrieves afresh and leaves in the conversation, so a session of n
    calls writes n slices to the cache and reads n(n-1)/2 of them back (pricing one slice a session under-states it; see
    the CHANGELOG, 0.1.0)."""
    n = max(calls, 1.0)
    core = min(tokens, core_tokens) * profile.p * (profile.w + profile.r * max(n - 1, 0))
    slices = slice_tokens * profile.p * (profile.w * n + profile.r * n * (n - 1) / 2)
    return extraction_usd + core + slices


def decide(state: dict, *, tokens: int, profile, figures: HostFigures, serve) -> dict:
    """One session start's decision. `state` is the store's (`NotesStore.mode()`); `serve` the kernel's serve block
    (core_tokens, slice_tokens, notes_min_sessions, notes_switch_runs). Returns the decision with both prices, the rule
    that applied and the state to save."""
    n = figures.calls_per_session or 1.0
    n_usd = notes_usd(tokens, profile, n)
    m_usd = memory_usd(tokens, profile, n, core_tokens=serve.core_tokens, slice_tokens=serve.slice_tokens,
                       extraction_usd=figures.extraction_usd_per_session)
    priced = figures.sessions >= serve.notes_min_sessions
    default = getattr(serve, "notes_default_tokens", 95_000)   # the measured crossover (CHANGELOG 0.1.0), until the host has measured both prices
    over = (n_usd > m_usd) if priced else (tokens > default)
    rule = "priced: notes cost against memory cost" if priced else f"default until {serve.notes_min_sessions} sessions: notes over {default} tokens"
    over_run = (state.get("over_run", 0) + 1) if over else 0
    return {"switch": over_run >= serve.notes_switch_runs, "rule": rule, "over": over, "over_run": over_run,
            "sessions": state.get("sessions", 0) + 1, "notes_tokens": tokens, "notes_usd": round(n_usd, 6), "memory_usd": round(m_usd, 6),
            "calls_per_session": round(n, 2), "extraction_usd_per_session": round(figures.extraction_usd_per_session, 6),
            "figures": figures.source}


# ---------------------------------------------------------------- migration, no model call
def migrate(current: list[tuple[NoteEntry, list[NoteEntry]]], *, principal_id: str, agent_id: str = "agent") -> dict:
    """{"records": [...], "unmapped": [...]}. Each record is (memory record dict in state proposed, the id it is
    superseded by or None), current value first, then its history oldest first; a note whose source was not anchored to
    a session line cannot become a record and is returned under `unmapped` for extraction."""
    from .harness.extract import key_of
    out, unmapped = [], []
    for cur, hist in current:
        ids = {}
        for e in [cur, *hist]:
            if not e.source.episode:
                unmapped.append(e)
                continue
            ids[e.id] = str(ULID())
        if cur.id not in ids:
            continue
        for e in [cur, *hist]:
            if e.id not in ids:
                continue
            where = f"{e.scope}'s {e.attribute.replace('_', ' ')}" + (f" for {e.entity}" if e.entity else "")
            rec = {"id": ids[e.id], "claim": f"The {where} is {e.value} (as of {e.as_of}).",
                   "kind": "stated" if e.source.kind == "principal" else "observed",
                   "actor": {"class": "principal", "id": principal_id} if e.source.kind == "principal" else {"class": "agent", "id": f"{agent_id} (tool result)"},
                   "anchor": {"quote": (e.source.quote or e.source.call or e.value)[:500], "episode": e.source.episode, "channel": None,
                              "message_id": None, "timestamp": e.as_of},
                   "scope": e.scope, "supersession_condition": "a later value stated for the same key",
                   "derived_from": [], "status": {"state": "proposed", "reason": None, "failure_class": None, "superseded_by": None},
                   "key": key_of(e.scope, e.attribute, e.entity), "value_status": "current" if e is cur else "historical"}
            out.append((rec, None if e is cur else ids[cur.id]))
    return {"records": out, "unmapped": unmapped}


def apply_migration(store, tenant: str, log, migration: dict, *, writer: str = "notes_migration") -> dict:
    """Write the migrated records through the store's lifecycle: each enters as proposed and is accepted into probation
    (the review step a host's reviewer would take, taken here because the notes were the agent's own checked writes);
    an earlier value is then superseded by its current one. Returns counts."""
    n_cur = n_hist = 0
    for rec, sup in migration["records"]:
        store.put(tenant, rec, writer=writer, log=log)
        acc = {**rec, "status": {"state": "probation", "reason": "migrated from notes", "failure_class": None, "superseded_by": None}}
        store.put(tenant, acc, writer=writer, log=log)
        if sup:
            old = {**acc, "status": {"state": "superseded", "reason": "an earlier value in the notes", "failure_class": "staleness", "superseded_by": sup}}
            store.put(tenant, old, writer=writer, log=log)
            n_hist += 1
        else:
            n_cur += 1
    return {"current": n_cur, "historical": n_hist, "unmapped": len(migration["unmapped"])}


# ---------------------------------------------------------------- an existing memory store, as notes
def from_memory(entries, *, principal_id: Optional[str] = None) -> list[dict]:
    """The store's current memories as note entries (so the switch to notes never hides a host's existing store
    behind an empty block): one entry per memory entry that is not superseded — its v2 key where it has one, else the record's scope and
    a short attribute from the claim; the claim as the value; the anchor's day, words and episode as the source."""
    from .harness.extract import key_of
    out = []
    for e in entries:
        r = getattr(e, "record", None)
        if r is None or r.status.state in ("superseded", "retired") or not hasattr(r, "claim"):
            continue
        if getattr(r, "key", None):
            scope, entity, attribute = (r.key.split("::") + [None, None])[:3]
            entity = None if entity in (None, "-", "") else entity
        else:
            words = re.findall(r"[A-Za-z0-9]+", r.claim)[:5]
            scope, entity, attribute = r.scope, None, "_".join(w.lower() for w in words) or "claim"
        day = (r.anchor.timestamp or "")[:10] or "undated"
        out.append({"id": str(ULID()), "key": key_of(scope, attribute, entity) or f"{scope}::-::{attribute}", "scope": scope, "entity": entity,
                    "attribute": attribute, "value": r.claim, "as_of": day, "written": day,
                    "source": {"kind": "principal" if r.actor.cls == "principal" else "tool", "quote": (r.anchor.quote or r.claim)[:300],
                               "call": None, "episode": r.anchor.episode}})
    return out


# ---------------------------------------------------------------- the host's one call at a session start
def profile_for(model: Optional[str], prices: Optional[dict] = None):
    """A price profile for the switch's arithmetic: the consumer's row for the model when given, else the
    library's dated list price, else zero prices — the switch then runs on the default threshold alone, and the logged
    dollars read 0."""
    from .costshape import LIST_PRICES, Profile, profile_from_prices
    if prices and model in prices:
        return profile_from_prices(prices[model])
    if model in LIST_PRICES:
        return LIST_PRICES[model]
    return Profile(p=0.0, po=0.0, w=1.0, r=1.0)


def counter_for(client, model: Optional[str]):
    """`count(text) -> (tokens, measured)` from a client that counts: a Recant model client with `count_tokens` (the
    reference client has one), or the raw Anthropic SDK (and `recant.wrap` of it), whose `messages.count_tokens` is
    used. None when the client cannot count, so the estimate from characters applies."""
    if client is None or model is None:
        return None
    sdk = getattr(getattr(client, "messages", None), "count_tokens", None)
    if hasattr(client, "count_tokens"):
        from .metering import _count
        one = lambda t: _count(client, model, t)                                                      # noqa: E731
    elif callable(sdk):
        one = lambda t: int(sdk(model=model, messages=[{"role": "user", "content": t}]).input_tokens)   # noqa: E731
    else:
        return None
    base: dict = {}

    def count(t: str) -> tuple[int, bool]:
        if "probe" not in base:
            base["probe"] = one("x")
        return max(0, one(t) - base["probe"]), True
    return count


def session_start(layout, tenant: str, store, log=None, *, principal_id: str, profile=None, serve, chars_per_token: float = 4.0,
                  extraction_usd_default: float = 0.0, session_id: Optional[str] = None, today: str,
                  block: Optional[str] = None, shared: Optional[list] = None, count=None, entries=None,
                  model: Optional[str] = None, client=None, prices: Optional[dict] = None) -> Optional[dict]:
    """While the block is in notes mode: the rendered notes for this session's prefix, as {"text", "tokens", "entries",
    "shared"}, after this session's switch check (logged as `notes_switch`, with both prices); on the switch, the notes
    migrate and None is returned — the block is in memory mode from here, and stays there. None also when already in
    memory mode. `block` is the conversation's own (the user's; None is the legacy single block); `shared` the blocks it
    may read, rendered after it in `text` and never switched here (each shared block switches on its own terms).
    `count(text) -> (tokens, measured)` is the host's token counter (the provider's: the switch and its log use
    measured tokens); without one the tokens are estimated from characters at `chars_per_token` and logged as such.
    `entries` are the memory store's current entries (`load_entries`), read on a scope's first session: a store
    already over the threshold keeps memory and takes new notes alongside it (mode "both": the block is returned with
    `memory` True and the host serves its core and slice too); a store under it is rendered into the block, as entries,
    so nothing the store held is hidden."""
    if profile is None:                                                 # a stranger's host passes a model, not a profile
        profile = profile_for(model, prices)
    if count is None and client is not None:
        count = counter_for(client, model)
    own_log = None
    if log is None:                                                     # ...and no log: the library opens one for the switch records
        from .log import LogWriter
        own_log = log = LogWriter(layout, tenant, "notes", artifact_hash="host", kernel_hash="host", params={"block": block})
    try:
        return _session_start(layout, tenant, store, log, principal_id=principal_id, profile=profile, serve=serve, chars_per_token=chars_per_token,
                              extraction_usd_default=extraction_usd_default, session_id=session_id, today=today, block=block, shared=shared,
                              count=count, entries=entries)
    finally:
        if own_log is not None:
            own_log.finish()


def _session_start(layout, tenant, store, log, *, principal_id, profile, serve, chars_per_token, extraction_usd_default, session_id, today,
                   block, shared, count, entries) -> Optional[dict]:
    def size(t: str) -> tuple[int, bool]:
        if count is not None:
            try:
                n, ok = count(t)
                if ok:
                    return int(n), True
            except Exception:                                  # noqa: BLE001 — a counter that fails leaves the estimate
                pass
        return int(len(t) / chars_per_token) + 1, False
    ns = NotesStore(layout, tenant, block)
    state = ns.mode()
    if not ns.mode_path.exists() and entries is not None:          # the scope's first session: what does the store hold already?
        imported = [NoteEntry.model_validate(e) for e in from_memory(entries, principal_id=principal_id)]
        store_tokens, measured0 = size(render([(e, []) for e in imported], block=block)) if imported else (0, True)
        over = store_tokens > getattr(serve, "notes_default_tokens", 95_000)
        state = {"mode": "both" if over else "notes", "sessions": 0, "over_run": 0, "store_entries": len(imported),
                 "store_tokens": store_tokens, "store_measured": measured0, "store_read": today}
        if not over:
            for e in imported:                                    # the store's memories become the block's first entries
                ns._append(ns.entries_path, e.model_dump(mode="json"))
        ns.save_mode(state)
        log.count("notes_scope_opened: " + state["mode"])
    shared_text, shared_n = render_blocks(layout, tenant, list(shared or []))
    if state.get("mode") == "both":                               # memory serves the store; the block carries the new notes beside it
        cur = ns.current()
        own = render(cur, block=block, unresolved=ns.unresolved())
        full = own + ("\n\n" + shared_text if shared_text else "")
        return {"text": full, "tokens": size(full)[0], "entries": len(cur) + shared_n, "shared": bool(shared_text), "memory": True, "both": True}
    if state.get("mode") != "notes":
        return None if not shared_text else {"text": shared_text, "tokens": size(shared_text)[0], "entries": shared_n,
                                             "shared": True, "memory": True}
    cur = ns.current()
    text_ = render(cur, block=block, unresolved=ns.unresolved())
    tokens, measured = size(text_)
    fig = figures_from_log(layout, tenant, extraction_usd_default=extraction_usd_default)
    d = decide(state, tokens=tokens, profile=profile, figures=fig, serve=serve)
    d["measured"] = measured
    d["chars"] = len(text_)
    log.emit("notes_switch", session=session_id, **{k: v for k, v in d.items() if k != "figures"}, figures=d["figures"])
    ns.save_mode({**state, "mode": "notes", "sessions": d["sessions"], "over_run": d["over_run"], "last": d})
    if d["switch"]:
        counts = apply_migration(store, tenant, log, migrate(cur, principal_id=principal_id))
        ns.save_mode({**state, "mode": "memory", "switched": today, "sessions": d["sessions"], "decision": d, "migrated": counts})
        log.count("notes_switched")
        return None if not shared_text else {"text": shared_text, "tokens": size(shared_text)[0], "entries": shared_n,
                                             "shared": True, "memory": True}
    full = text_ + ("\n\n" + shared_text if shared_text else "")
    return {"text": full, "tokens": size(full)[0] if shared_text else tokens, "entries": len(cur) + shared_n, "shared": bool(shared_text),
            "memory": False, "measured": measured}


# ---------------------------------------------------------------- the agent's tools, for any host
_ENTRY_SCHEMA = {"type": "object",
                 "properties": {"scope": {"type": "string", "minLength": 1}, "entity": {"type": "string"}, "attribute": {"type": "string", "minLength": 1},
                                "value": {"type": "string", "minLength": 1}, "as_of": {"type": "string"},
                                "source": {"type": "string", "enum": ["principal", "tool"]}, "quote": {"type": "string"}, "call": {"type": "string"}},
                 "required": ["scope", "attribute", "value", "source"], "additionalProperties": False}

TOOL_SPECS = [
    {"name": "note_fact",
     "description": ("Record facts in your notes, which you are shown at the start of every session; one call can hold everything a "
                     "message says. Each entry: `scope` is what it is about; `entity` the thing within it, when there is one; `attribute` "
                     "the property; `value` the current value, written as the principal or the tool gave it; `as_of` the date it holds "
                     "from (today when unsaid). `source` is `principal` — the value must appear in what the principal said in this "
                     "session; `quote` is optional — or `tool` with `call`, the tool call the value came from. An entry whose value is "
                     "not in the principal's words, whose call was not made, or whose value is not in that call's output is refused; "
                     "the others are kept. A new value for the same scope, entity and attribute replaces the old one, which stays as "
                     "history. `block` is `user` (yours, the default) or a shared block (`team`, `org`) when the host lets you write it."),
     "input_schema": {"type": "object", "properties": {"entries": {"type": "array", "items": _ENTRY_SCHEMA, "minItems": 1},
                                                       "block": {"type": "string", "enum": ["user", "team", "org"]}},
                      "required": ["entries"], "additionalProperties": False}},
    {"name": "read_notes",
     "description": ("Read notes in full — the value, its date, the principal's words or the tool call it came from, and where — by words "
                     "from the attribute, entity or scope (every term must match). The notes block you are shown carries only a short "
                     "source reference; use this when you need the exact words."),
     "input_schema": {"type": "object", "properties": {"terms": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1}},
                      "required": ["terms"], "additionalProperties": False}},
]
TOOL_NAMES = tuple(t["name"] for t in TOOL_SPECS)
REFUSAL_NOTE = ("A principal note's value must be in the principal's own words from this session; a tool note names a tool call made in "
                "this session and a value found in its output.")


def handle(tool_name: str, tool_input: dict, *, layout, tenant: str, lines, episode_path: str, today: str, principal_id: str,
           agent: Optional[str] = None, ts: Optional[str] = None, shared_read=(), shared_write: bool = False, log=None) -> str:
    """The two tools' handler for any host: `lines` are the session's lines so far (`store.hot.read_session`),
    `episode_path` the session file's path. note_fact: every entry checked on its own (`check_many`), the kept ones written
    to the block, the refused ones kept apart with their reason; the result names each. read_notes: entries in full from
    the user's block and the shared blocks it may read. Returns the tool result text."""
    user_block = f"user:{principal_id}"
    if tool_name == "read_notes":
        terms = [str(t).strip().lower() for t in (tool_input.get("terms") or []) if str(t).strip()]
        if not terms:
            return "read_notes needs `terms`: words from the attribute, entity or scope."
        out = []
        for b in [user_block, *shared_read]:
            for cur, hist in NotesStore(layout, tenant, b).current():
                hay = f"{cur.scope} {cur.entity or ''} {cur.attribute}".lower()
                if all(t in hay for t in terms):
                    src = cur.source
                    out.append(f"- [{b}] {cur.scope}/{cur.entity or '-'}/{cur.attribute} = {cur.value} (as of {cur.as_of}); source {src.kind}: "
                               + (f'"{src.quote}"' if src.quote else f"call {src.call}") + f" at {src.episode or 'unanchored'}"
                               + ("; earlier: " + ", ".join(f"{h.value} ({h.as_of})" for h in reversed(hist)) if hist else ""))
                if len(out) >= 20:
                    break
        return "\n".join(out) if out else "No note matches all of: " + ", ".join(terms)
    if tool_name != "note_fact":
        return f"{tool_name} is not a notes tool."
    raw = tool_input.get("entries")
    if not isinstance(raw, list) or not raw:
        return "note_fact needs `entries`: a list of one or more entries."
    kind = (tool_input.get("block") or "user").strip()
    if kind == "user":
        block = user_block
    else:
        found = [b for b in shared_read if b == kind or b.startswith(kind + ":")]
        if not found or not shared_write:
            return f"Not noted: writing the {kind} block needs the host's permission; your notes go to your own block."
        block = found[0]
    entries = []
    for e in raw:
        if isinstance(e, dict):
            entries.append({"scope": str(e.get("scope") or "").strip(), "entity": str(e.get("entity") or "").strip() or None,
                            "attribute": str(e.get("attribute") or "").strip(), "value": str(e.get("value") or "").strip(),
                            "as_of": str(e.get("as_of") or today).strip(),
                            "source": {"kind": e.get("source"), "quote": str(e.get("quote") or "").strip() or None, "call": str(e.get("call") or "").strip() or None}})
    store = NotesStore(layout, tenant, block)
    results = check_many(entries, lines, episode_path)
    before = {c.key: c for c, _ in store.current()}
    out, refused = [], 0
    for e, (reason, anchor) in zip(entries, results):
        name = f"{e['scope']} / {e['entity'] or '-'} / {e['attribute']}"
        if not (e["scope"] and e["attribute"] and e["value"]):
            reason = "scope, attribute and value are needed"
        if reason:
            store.refuse(e, reason, today=today)
            if log is not None:
                log.count(f"note_refused: {reason}")
            out.append(f"- {name}: not noted ({reason})")
            refused += 1
            continue
        e["source"]["episode"] = anchor
        rec = store.add(e, today=today, agent=agent, ts=ts)
        old = before.get(rec.key)
        out.append(f"- {name} = {rec.value} (as of {rec.as_of})" + (f"; replaces {old.value} (as of {old.as_of})" if old else ""))
    head = f"Noted {len(entries) - refused} of {len(entries)} in the {kind} block" + (f"; {refused} refused. {REFUSAL_NOTE}" if refused else ".")
    return head + "\n" + "\n".join(out)
