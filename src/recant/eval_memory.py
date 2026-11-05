"""`recant eval memory`: memory on the developer's own past sessions, four arms side by side.

    recant eval memory --sessions sessions.jsonl --make-questions [--model M]      writes questions.jsonl, for you to read
    recant eval memory --sessions sessions.jsonl --questions questions.jsonl [--arms none,history,notes,memory_v2]

Sessions: one per line, `{"scope": "user:ana", "ts": "2026-10-01T09:00:00+00:00", "messages": [...]}`, the messages
Anthropic-shaped (a string or blocks; tool results count as tool output); or `--load module:function` returning the same.

Questions, written first and frozen (the file is hashed when scoring starts; the hash is in the report):
- recall: a fact the principal stated in an earlier session, its answer quoted from the principal's own turn;
- update: a fact that changed later, where only the newest value is right;
- absent: a question no session answers, where the right answer is "I don't know" — these catch invented answers.
A drafted question whose quote is not in the principal's own words in the session it names is dropped, with the reason.

The arms, each answering every question of a scope after its sessions:
- none: the question alone;
- history: the scope's sessions in full, newest first until the window is full (the ceiling);
- notes: the managed notes block Recant built from the sessions, each session read once by a note keeper whose entries
  pass the library's evidence check (`recant.notes.handle`);
- memory_v2: extraction over the sessions (`recant.extract`, memory v2's machinery), every proposal accepted, and the
  served core and slice (`recant.budget.fit`) for each question.

Scoring is mechanical: the reply's last `ANSWER:` line holds the answer's words (`recant.notes.holds`), or, for an absent
question, says it does not know. The report: accuracy per arm and kind with 95% intervals clustered by scope; each arm
against carried history on Recant's one standard; cost per question and per correct answer, the write path included;
tokens by class; latency; notes size per scope against the 95k switch point; entries the evidence check refused."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from . import costshape as cs, notes, quality, switch
from .log import read_run
from .machinery import Machinery
from .machinery.loader import DEFAULTS
from .store import MemoryStore, StoreLayout
from .store import episodes
from .store.budget import fit
from .store.review import decide_memory, memory_queue
from .store.search import load_entries
from .version import describe, info as version_info
from .wrap import wrap

TENANT = "00000000-0000-0000-0000-000000000001"
ARMS = ("none", "history", "notes", "memory_v2")
KINDS = ("recall", "update", "absent")
SWITCH_TOKENS = 95_000                     # memory v2 takes over a scope above 95k counted tokens of notes
CHARS_PER_TOKEN = 4.0
REVIEWER = {"kind": "model", "id": "eval-memory:accept-all"}
SPEAKER = {"user": "Person", "assistant": "Assistant"}
DONT_KNOW = re.compile(r"\b(i\s+)?(don'?t|do not|cannot|can'?t)\s+know\b|\bnot\s+(stated|mentioned|recorded|in (my|the) (notes|memory|records))\b|"
                       r"\bno (record|information|note)s?\b|\bunknown\b", re.I)

ANSWER_SYSTEM = ("You are the assistant this person works with. Answer the question from what you know of them below, "
                 "if anything. Be brief. End with one line: `ANSWER: <the answer in a few words>`, or `ANSWER: I don't know` "
                 "when nothing below says.")
KEEPER_SYSTEM = ("You keep notes for an assistant about the person in this session. Read the session and record, with the "
                 "note_fact tool, each durable fact the person stated (who, what, when, how much, preferences, decisions), "
                 "quoting the person's own words. Record a changed fact again with its new value. Skip small talk.")
QUESTION_SYSTEM = (
    "You write a memory test from one person's past sessions with an assistant. Return JSON only: "
    '{"questions": [{"kind": "recall"|"update"|"absent", "question": "...", "answer": "...", "quote": "...", "session": N}]}.\n'
    "- recall: a fact the person stated in an earlier session about themselves, their work or their plans (never what they "
    "asked the assistant); `quote` is their exact words from session N stating it, and `answer` is the fewest words that "
    "identify the value, taken from the quote: the name, number, date or choice itself, without the words around it "
    "(\"Juniper\", not \"the Juniper account\"; \"40\", not \"40 boxes\"; \"aisle seats\", not \"always aisle seats\").\n"
    "- update: a fact the person changed later; `answer` is the newest value, in the fewest words that identify it, `quote` "
    "their exact words from the session N where they last stated it.\n"
    "- absent: a natural question about them that no session answers; `answer` is \"I don't know\", `quote` empty, `session` 0.\n"
    "Ask each question as the assistant would be asked it later, without hinting at the answer.")


# ---------------------------------------------------------------- sessions
def _text(content) -> list[tuple[str, str]]:
    """(role, text) pieces of one message's content: text blocks as said, tool results as tool output, tool calls named."""
    if isinstance(content, str):
        return [("", content)]
    out = []
    for b in content or []:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            out.append(("", b.get("text") or ""))
        elif t == "tool_result":
            c = b.get("content")
            out.append(("tool", c if isinstance(c, str) else " ".join(x.get("text", "") for x in (c or []) if isinstance(x, dict))))
        elif t == "tool_use":
            out.append(("tool", f"[called {b.get('name')}]"))
    return out


def load_sessions(path: Optional[str] = None, hook: Optional[Callable] = None) -> list[dict]:
    rows = hook() if hook else [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    out = []
    for r in rows:
        if not r.get("scope") or not r.get("ts") or not isinstance(r.get("messages"), list):
            raise SystemExit("a session needs scope, ts and messages")
        turns = []
        for m in r["messages"]:
            for role, t in _text(m.get("content")):
                if t.strip():
                    turns.append({"role": role or m.get("role", "user"), "text": t.strip()})
        out.append({"scope": str(r["scope"]), "ts": str(r["ts"]), "turns": turns})
    return sorted(out, key=lambda s: (s["scope"], s["ts"]))


def by_scope(sessions: list[dict]) -> dict[str, list[dict]]:
    d: dict = {}
    for s in sessions:
        d.setdefault(s["scope"], []).append(s)
    return d


def slug(scope: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", scope.lower()).strip("-")[:40] + "-" + hashlib.sha256(scope.encode()).hexdigest()[:6]


def transcript(ss: list[dict], *, numbered: bool = True) -> str:
    parts = []
    for k, s in enumerate(ss, 1):
        head = f"Session {k} ({s['ts'][:10]}):" if numbered else f"Session of {s['ts'][:10]}:"
        parts.append(head + "\n" + "\n".join(f"{SPEAKER.get(t['role'], 'Tool')}: {t['text']}" for t in s["turns"]))
    return "\n\n".join(parts)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


# ---------------------------------------------------------------- the metered client
def _meter_client(raw, root: Path):
    layout = StoreLayout(root)
    switch.set_state(layout, "meter")                               # every call metered; nothing on the request changed
    return wrap(raw, store=layout, tenant=TENANT, job="eval")


def _reply_text(r) -> str:
    return "".join(getattr(b, "text", "") or "" for b in getattr(r, "content", []) if getattr(b, "type", "") == "text").strip()


# ---------------------------------------------------------------- questions
def make_questions(sessions: list[dict], *, raw, model: str, out_path: str, root: Path, per_scope: int = 12,
                   max_chars: int = 400_000) -> dict:
    client = _meter_client(raw, root / "questions-store")
    kept, dropped = [], []
    for scope, ss in by_scope(sessions).items():
        text_ = transcript(ss)[-max_chars:]
        with client.run("make_questions", session=f"q:{slug(scope)}"):
            r = client.messages.create(model=model, max_tokens=8000, system=QUESTION_SYSTEM,
                                       messages=[{"role": "user", "content": f"Write up to {per_scope} questions, about a third "
                                                  f"of each kind.\n\n{text_}"}])
        raw_text = _reply_text(r)
        m = re.search(r"\{.*\}", raw_text, re.S)
        try:
            qs = json.loads(m.group(0))["questions"] if m else []
        except (ValueError, KeyError):
            qs = []
            dropped.append({"scope": scope, "reason": "the model's reply was not the JSON asked for"})
        for n, q in enumerate(qs):
            why = check_question(q, ss)
            row = {"id": f"q-{slug(scope)}-{n}", "scope": scope, "kind": q.get("kind"), "question": q.get("question"),
                   "answer": q.get("answer"), "quote": q.get("quote") or "", "session": q.get("session") or 0}
            (dropped if why else kept).append({**row, "reason": why} if why else row)
    Path(out_path).write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in kept))
    return {"kept": len(kept), "dropped": dropped, "path": out_path,
            "by_kind": {k: sum(q["kind"] == k for q in kept) for k in KINDS}}


def check_question(q: dict, ss: list[dict]) -> Optional[str]:
    """Why a drafted question is dropped, or None. The evidence rule: a recall or update answer is in the principal's
    own words, quoted from the session the question names."""
    kind = q.get("kind")
    if kind not in KINDS or not (q.get("question") or "").strip() or not (q.get("answer") or "").strip():
        return "kind, question and answer are needed"
    if kind == "absent":
        return None
    n = q.get("session")
    if not isinstance(n, int) or not 1 <= n <= len(ss):
        return f"session {n} is not one of this scope's {len(ss)} sessions"
    quote = _norm(q.get("quote") or "")
    said = [_norm(t["text"]) for t in ss[n - 1]["turns"] if t["role"] == "user"]
    if not quote or not any(quote in s for s in said):
        return "the quote is not the person's own words in that session"
    if not notes.holds(q["quote"], q["answer"]):
        return "the answer is not in the quote"
    return None


# ---------------------------------------------------------------- building each arm's memory
def _write_sessions(layout, ss, principal):
    paths = []
    for s in ss:
        day = s["ts"][:10]
        w = episodes.open_session(layout, TENANT, day)
        for t in s["turns"]:
            role = t["role"] if t["role"] in ("user", "assistant") else "tool"
            w.append(role, t["text"], who=principal if role == "user" else None, when=s["ts"] if role == "user" else None)
        paths.append((day, w.episode_path, s))
    return paths


def build_notes(scope: str, ss: list[dict], *, raw, model: str, root: Path, rounds: int = 4) -> dict:
    layout = StoreLayout(root / slug(scope))
    layout.ensure(TENANT)
    principal = slug(scope)
    client = _meter_client(raw, root / "write-store")
    t0 = time.monotonic()
    for day, ep, s in _write_sessions(layout, ss, principal):
        block, _ = notes.render_blocks(layout, TENANT, [f"user:{principal}"])
        lines = episodes.read_session(layout, TENANT, ep)
        body = "\n".join(f"{SPEAKER.get(t['role'], 'Tool')}: {t['text']}" for t in s["turns"])
        msgs = [{"role": "user", "content": f"The notes so far:\n{block}\n\nThe session of {day}:\n{body}"}]
        with client.run("notes_write", session=f"notes:{principal}:{day}"):
            for _ in range(rounds):
                r = client.messages.create(model=model, max_tokens=2048, system=KEEPER_SYSTEM + "\n\n" + notes.INSTRUCTION,
                                           tools=notes.TOOL_SPECS, messages=msgs)
                calls = [b for b in r.content if getattr(b, "type", "") == "tool_use"]
                if not calls:
                    break
                msgs.append({"role": "assistant", "content": [b.model_dump() if hasattr(b, "model_dump") else vars(b) for b in r.content]})
                results = [{"type": "tool_result", "tool_use_id": c.id,
                            "content": notes.handle(c.name, c.input, layout=layout, tenant=TENANT, lines=lines, episode_path=ep,
                                                    today=day, principal_id=principal)} for c in calls]
                msgs.append({"role": "user", "content": results})
    text_, n = notes.render_blocks(layout, TENANT, [f"user:{principal}"])
    ns = notes.NotesStore(layout, TENANT, f"user:{principal}")
    refused = ns.refused()
    return {"text": text_, "entries": n, "tokens_est": int(len(text_) / CHARS_PER_TOKEN), "seconds": time.monotonic() - t0,
            "refused": len(refused), "refused_reasons": _count([r["reason"] for r in refused]), "layout": layout}


def build_memory_v2(scope: str, ss: list[dict], *, extract_client, root: Path) -> dict:
    from .extraction import extract
    layout = StoreLayout(root / slug(scope))
    layout.ensure(TENANT)
    Machinery.seed(layout, DEFAULTS / "v2" / "artifact.yaml", DEFAULTS / "v2" / "kernel.yaml")
    m = Machinery.load(layout)
    store = MemoryStore(layout, list(m.kernel.corpus.allowlist) + list(episodes.HOT_PATTERNS))
    principal = slug(scope)
    t0 = time.monotonic()
    days = sorted({d for d, _, _ in _write_sessions(layout, ss, principal)})
    from .log import LogWriter
    accepted = 0
    for day in days:
        extract(layout, TENANT, day, machinery=m, client=extract_client, store=store, principal_id=principal)
        with LogWriter(layout, TENANT, "review", artifact_hash=m.artifact_hash, kernel_hash=m.kernel_hash, params={"day": day}) as log:
            for row in memory_queue(store, TENANT):
                if row["state"] == "proposed" and store.get(TENANT, row["id"])[0].status.state == "proposed":
                    decide_memory(store, TENANT, log, row["id"], to_state="probation", reason="confirmed", reviewer=REVIEWER,
                                  reasons=m.artifact.review_reasons)
                    accepted += 1
    entries = load_entries(store, TENANT, "as_reviewed", m.artifact.trust_rank)
    return {"entries": entries, "serve": m.kernel.serve, "accepted": accepted, "seconds": time.monotonic() - t0, "layout": layout}


# ---------------------------------------------------------------- scoring
def answer_of(reply: str) -> str:
    found = re.findall(r"ANSWER:\s*(.+)", reply or "")
    return found[-1].strip() if found else ""


_MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}


def _numbers(s: str) -> list[float]:
    """Numbers as written, with thousands separators and currency marks left out: "26,000 (EUR)" -> [26000.0]."""
    out = []
    for m in re.finditer(r"-?\d[\d,\u202f ]*(?:\.\d+)?", s or ""):
        t = re.sub(r"[,\u202f ]", "", m.group(0))
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out


def _dates(s: str) -> set[tuple[int, int, int]]:
    """(year, month, day) for the dates in a string: ISO, D/M/Y and M/D/Y (both kept when ambiguous), D.M.Y, and
    with the month named ("13 May 2027", "May 13, 2027")."""
    out = set()
    for y, mo, d in re.findall(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", s or ""):
        out.add((int(y), int(mo), int(d)))
    for a, b_, y in re.findall(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b", s or ""):
        out.add((int(y), int(b_), int(a)))
        out.add((int(y), int(a), int(b_)))
    for d, mo, y in re.findall(r"\b(\d{1,2})\s+([A-Za-z]{3})[a-z]*\.?,?\s+(\d{4})\b", s or ""):
        if mo.lower() in _MONTHS:
            out.add((int(y), _MONTHS[mo.lower()], int(d)))
    for mo, d, y in re.findall(r"\b([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})\b", s or ""):
        if mo.lower() in _MONTHS:
            out.add((int(y), _MONTHS[mo.lower()], int(d)))
    return out


def matches(answer: str, key: str) -> bool:
    """The answer holds the key: its words (`notes.holds`), or the same date in another format, or the same number
    written another way (a principal's own format rules may turn 2027-03-25 into 25/03/2027 and 26000 into
    "26,000 (EUR)", and a words-only check scores such right answers wrong)."""
    if notes.holds(answer, key):
        return True
    kd = _dates(key)
    if kd and kd & _dates(answer):
        return True
    kn = _numbers(key)
    if kn and not kd and len(kn) == 1 and re.fullmatch(r"[\s\d,.\-]+", key.strip() or "x"):
        return any(abs(x - kn[0]) < 1e-6 for x in _numbers(answer))
    return False


def correct(q: dict, reply: str, *, strict: bool = False) -> bool:
    """`strict`: the key's words only, the check before 0.2.2, kept beside the score in the report."""
    a = answer_of(reply)
    if q["kind"] == "absent":
        return bool(DONT_KNOW.search(a or reply or ""))
    if not a or DONT_KNOW.search(a):
        return False
    return notes.holds(a, q["answer"]) if strict else matches(a, q["answer"])


def _count(xs):
    d: dict = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d


def _spent(root: Path) -> dict:
    """Every model call under `root`, priced at list: dollars, tokens by class, calls, latency."""
    usd, tok, calls, lat, own = 0.0, {k: 0 for k in ("input", "cache_write", "cache_read", "output", "reasoning")}, 0, [], []
    for p in root.glob("**/log/**/*.jsonl"):
        for r in read_run(p, typed=False):
            if r["kind"] != "model_call":
                continue
            u = r.get("usage") or {}
            pf = cs.LIST_PRICES.get(r.get("model_id"))
            if pf is not None:
                usd += pf.bill(uncached=u.get("input", 0), write=u.get("cache_write", 0), read=u.get("cache_read", 0),
                               output=u.get("output", 0), write_long=u.get("cache_write_long", 0))
            for k in tok:
                tok[k] += u.get(k, 0) or 0
            calls += 1
            if r.get("latency_ms") is not None:
                lat.append(r["latency_ms"] / 1000)
            if r.get("overhead_ms") is not None:
                own.append(r["overhead_ms"])
    return {"usd": usd, "tokens": tok, "calls": calls, "latency": lat, "overhead_ms": own}


def _coverage(root: Path) -> dict:
    """The Coverage line's figures for every store under `root`: the calls Recant metered against what the
    watch counted going past it."""
    from . import coverage
    seen, unseen, watched = [], [], False
    for p in root.glob("**/log/**/*.jsonl"):
        for r in read_run(p, typed=False):
            if r["kind"] == "model_call":
                seen.append(r)
            elif r["kind"] == "coverage":
                watched = True
                if r.get("requests"):
                    unseen.append(r)

    def price(model, u):
        pf = cs.LIST_PRICES.get(model)
        return None if pf is None else pf.bill(uncached=u.get("input", 0), write=u.get("cache_write", 0),
                                               read=u.get("cache_read", 0), output=u.get("output", 0))
    return dict(coverage.summarize(seen, unseen, price=price), watched=watched)


def _memory_entries(layout) -> tuple[list, Any]:
    m = Machinery.load(layout)
    store = MemoryStore(layout, list(m.kernel.corpus.allowlist) + list(episodes.HOT_PATTERNS))
    return load_entries(store, TENANT, "as_reviewed", m.artifact.trust_rank), m.kernel.serve


def _built(arm_root: Path, scope: str, build: Callable) -> dict:
    """A scope's notes or memory, built once (a run must survive being stopped): a store finished
    before is reused from its marker; one a stopped run left half-built is cleared and built again."""
    d = arm_root / slug(scope)
    mark = d / "built.json"
    if mark.exists():
        return json.loads(mark.read_text())
    if d.exists():
        shutil.rmtree(d)
    info = build()
    mark.write_text(json.dumps(info, default=str))
    return info


def evaluate(sessions: list[dict], questions_path: str, *, raw, model: str, out: str, arms=ARMS, window_tokens: int = 150_000,
             extract_client=None, say: Callable = print, parallel: int = 1) -> dict:
    """Every answer is written as it lands and every built store marked, so a stopped run resumes where it stopped,
    against the same questions only. `parallel` scopes at a time; a scope's sessions are always built in order."""
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    qraw = Path(questions_path).read_bytes()
    qhash = hashlib.sha256(qraw).hexdigest()
    qs = [json.loads(x) for x in qraw.decode().splitlines() if x.strip()]
    man = root / "manifest.json"
    if man.exists() and json.loads(man.read_text()).get("questions_sha256") != qhash:
        raise SystemExit("this eval directory was started on other questions; use a new --out")
    man.write_text(json.dumps({"questions_sha256": qhash, "model": model, "arms": list(arms), "recant": version_info()}) + "\n")
    say(f"questions frozen: {questions_path} sha256 {qhash} ({len(qs)} questions)")
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"an arm is one of {ARMS}, not {arm!r}")
    ans_p = root / "answers.jsonl"
    rows = [json.loads(x) for x in ans_p.read_text().splitlines() if x.strip()] if ans_p.exists() else []
    done = {(r["arm"], r["id"]) for r in rows}
    lock = threading.Lock()
    scopes = by_scope(sessions)
    built: dict = {arm: {} for arm in arms}
    extractor = extract_client or _default_extract_client(raw)

    def one_scope(arm: str, scope: str, ss: list, answers) -> None:
        sq = [q for q in qs if q["scope"] == scope]
        if not sq:
            return
        arm_root = root / arm
        ctx, mv = "", None
        if arm in ("notes", "memory_v2") and not (arm_root / slug(scope)).exists():
            say(f"arm {arm}: writing the {'notes' if arm == 'notes' else 'memory'} for a scope ({len(ss)} sessions; the write path "
                f"is the slow part)")
        if arm == "history":
            kept, total = [], 0
            for s in reversed(ss):                                     # newest first, until the window is full
                t = transcript([s], numbered=False)
                if total + len(t) / CHARS_PER_TOKEN > window_tokens:
                    break
                kept.insert(0, s)
                total += len(t) / CHARS_PER_TOKEN
            ctx = "Your earlier sessions with them:\n\n" + transcript(kept, numbered=False)
            info = {"sessions_carried": len(kept), "sessions": len(ss), "tokens_est": int(total)}
        elif arm == "notes":
            def build():
                b = build_notes(scope, ss, raw=raw, model=model, root=arm_root)
                i = {k: v for k, v in b.items() if k not in ("layout", "text")}
                i["over_switch_point"] = b["tokens_est"] > SWITCH_TOKENS
                return i
            info = _built(arm_root, scope, build)
            ctx, _ = notes.render_blocks(StoreLayout(arm_root / slug(scope)), TENANT, [f"user:{slug(scope)}"])
        elif arm == "memory_v2":
            def build():
                b = build_memory_v2(scope, ss, extract_client=extractor, root=arm_root)
                return {"entries": len(b["entries"]), "accepted": b["accepted"], "seconds": b["seconds"]}
            info = _built(arm_root, scope, build)
            mv = _memory_entries(StoreLayout(arm_root / slug(scope)))
        else:
            info = {}
        with lock:
            built[arm][scope] = info
        for q in sq:
            if (arm, q["id"]) in done:
                continue
            c = ctx
            if mv is not None:
                f = fit(mv[0], family=None, serve=mv[1], view="as_reviewed", question=q["question"])
                c = f["core_text"] + ("\n\n" + f["slice_text"] if f["slice_text"] else "")
            system = ANSWER_SYSTEM + ("\n\n" + c if c else "")
            with answers.run("answer", session=f"{arm}:{q['id']}"):
                try:
                    r = answers.messages.create(model=model, max_tokens=1024, system=system,
                                                messages=[{"role": "user", "content": q["question"]}])
                    reply, err = _reply_text(r), None
                except Exception as exc:                              # noqa: BLE001 — a failed answer is a wrong answer
                    reply, err = "", type(exc).__name__
            row = {"arm": arm, "id": q["id"], "scope": scope, "kind": q["kind"], "correct": correct(q, reply) and err is None,
                   "error": err, "reply": reply}
            with lock:
                with ans_p.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
                done.add((arm, q["id"]))
        with lock:
            n = sum(1 for (a, _) in done if a in arms)
            say(f"arm {arm}: a scope answered; {n} of {len(qs) * len(arms)} answers in")

    for arm in arms:
        answers = _meter_client(raw, root / arm / "answer-store")
        jobs = list(scopes.items())
        if parallel <= 1:
            for scope, ss in jobs:
                one_scope(arm, scope, ss, answers)
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(parallel) as ex:
                for f in [ex.submit(one_scope, arm, scope, ss, answers) for scope, ss in jobs]:
                    f.result()
        say(f"arm {arm}: {sum(r['correct'] for r in rows if r['arm'] == arm)} of {sum(r['arm'] == arm for r in rows)} correct")
    by_id = {q["id"]: q for q in qs}
    for r in rows:                                     # scored at report time from the kept reply, so a better check
        q = by_id.get(r["id"])                         # rescores a finished run without a model call
        if q is not None:
            r["correct"] = correct(q, r.get("reply") or "") and r.get("error") is None
            r["strict_correct"] = correct(q, r.get("reply") or "", strict=True) and r.get("error") is None
    return report(root, rows, built, arms, qhash, model, len(qs))


def _default_extract_client(raw):
    from .clients.anthropic import AnthropicClient
    return AnthropicClient(raw)


def report(root: Path, rows, built, arms, qhash, model, n_questions) -> dict:
    out = {"header": {"recant": version_info(), "questions_sha256": qhash, "model": model, "questions": n_questions,
                      "scopes": len({r["scope"] for r in rows}), "arms": list(arms), "scoring": "mechanical: the ANSWER line"},
           "arms": {}}
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        acc = quality.paired_bootstrap([(r["scope"], 0, int(r["correct"])) for r in rs])
        strict = quality.paired_bootstrap([(r["scope"], 0, int(r.get("strict_correct", r["correct"]))) for r in rs])
        kinds = {}
        for k in KINDS:
            kr = [r for r in rs if r["kind"] == k]
            if kr:
                b = quality.paired_bootstrap([(r["scope"], 0, int(r["correct"])) for r in kr])
                kinds[k] = {"accuracy": b["delta"], "interval": b["interval"], "n": len(kr)}
        ans, write = _spent(root / arm / "answer-store"), _spent(root / arm)
        write_usd = write["usd"] - ans["usd"]
        n_correct = sum(r["correct"] for r in rs)
        info = built.get(arm, {})
        out["arms"][arm] = {
            "accuracy": acc["delta"], "interval": acc["interval"], "n": len(rs), "correct": n_correct, "by_kind": kinds,
            "correct_words_only": sum(1 for r in rs if r.get("strict_correct", r["correct"])),
            "accuracy_words_only": strict["delta"], "interval_words_only": strict["interval"],
            "cost": {"answers_usd": ans["usd"], "write_path_usd": write_usd, "per_question_usd": write["usd"] / len(rs),
                     "per_correct_usd": write["usd"] / n_correct if n_correct else None},
            "tokens": write["tokens"], "calls": write["calls"],
            "latency": {"answer_call": {"p50": cs._pct(ans["latency"], 0.5), "p95": cs._pct(ans["latency"], 0.95)},
                        "recant_overhead": {"p50": cs._pct(write["overhead_ms"], 0.5), "p95": cs._pct(write["overhead_ms"], 0.95),
                                            "n": len(write["overhead_ms"])},
                        "write_path_s": sum(v.get("seconds", 0) for v in info.values()), "write_on_request_path": False},
            "failed_answers": sum(1 for r in rs if r["error"]),
            "scopes": info}
        if arm == "notes":
            out["arms"][arm]["notes"] = {"tokens_per_scope": {s: v["tokens_est"] for s, v in info.items()},
                                         "over_switch_point": [s for s, v in info.items() if v.get("over_switch_point")],
                                         "refused": sum(v["refused"] for v in info.values()),
                                         "refused_reasons": _merge([v["refused_reasons"] for v in info.values()])}
    if "history" in out["arms"]:
        h = {r["id"]: r for r in rows if r["arm"] == "history"}
        for arm in out["arms"]:
            if arm == "history":
                continue
            pairs = [(r["scope"], int(h[r["id"]]["correct"]), int(r["correct"])) for r in rows if r["arm"] == arm and r["id"] in h]
            b = quality.paired_bootstrap(pairs)
            out["arms"][arm]["against_history"] = {"difference": b["delta"], "interval": b["interval"],
                                                   "held": quality.held(b["delta"], b["interval"][0]), "verdict": quality.verdict(b)}
    out["coverage"] = _coverage(root)
    out["total_spend_usd"] = sum(v["cost"]["answers_usd"] + v["cost"]["write_path_usd"] for v in out["arms"].values())
    out["verdict"] = verdict(out)
    from . import friction
    friction.report("eval_memory")
    out["setup"] = friction.summary()
    (root / "report.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    (root / "report.md").write_text(render(out) + "\n")
    return out


def _merge(ds):
    out: dict = {}
    for d in ds:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def verdict(r: dict) -> list[str]:
    a = r["arms"]
    lines = []
    if "history" not in a:
        return ["no carried-history arm: nothing to hold quality against"]
    ok = [("history", a["history"]["cost"]["per_question_usd"])] + \
         [(k, v["cost"]["per_question_usd"]) for k, v in a.items() if k != "history" and v.get("against_history", {}).get("held")]
    best = min(ok, key=lambda x: x[1])
    for k, v in a.items():                             # the words-only score first, the value match beside it
        ah = v.get("against_history")
        lines.append(f"{k}: {v['accuracy_words_only']:.1%} words only (95% {v['interval_words_only'][0]:.1%} to "
                     f"{v['interval_words_only'][1]:.1%}), {v['accuracy']:.1%} value match (95% {v['interval'][0]:.1%} to "
                     f"{v['interval'][1]:.1%}), ${v['cost']['per_question_usd']:.4f} a question"
                     + (f"; against history, value match, {ah['difference'] * 100:+.1f} points ({ah['verdict']})" if ah else ""))
    lines.append(f"cheapest arm holding quality against carried history, on value match: {best[0]} (${best[1]:.4f} a question)")
    if r.get("total_spend_usd") is not None:
        lines.append(f"total spend: ${r['total_spend_usd']:.2f} at list prices (answers and write paths, every arm)")
    if r.get("coverage") is not None:
        from . import coverage
        lines.append(coverage.line(r["coverage"], watched=r["coverage"].get("watched", False)))
    return lines


def render(r: dict) -> str:
    h = r["header"]
    L = [f"# recant eval memory", "", f"- Recant {describe(h['recant'])}",
         f"- questions sha256 {h['questions_sha256']}: {h['questions']} questions over {h['scopes']} scopes; model {h['model']}",
         f"- scoring: {h['scoring']}; words only (the answer holds the key's words) first, value match (the same date or "
         "amount in another format) beside it", "", "## Verdict", ""] + [f"- {x}" for x in r["verdict"]] + ["", "## Arms", "",
         "| arm | correct, words only | correct, value match | recall | update | absent | $/question | $/correct | write path $ | against history |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in r["arms"].items():
        kk = lambda n: f"{v['by_kind'][n]['accuracy']:.0%} ({v['by_kind'][n]['n']})" if n in v["by_kind"] else "—"   # noqa: E731
        ah = v.get("against_history")
        vs = "%+.1f pts, %s" % (ah["difference"] * 100, ah["verdict"]) if ah else "—"
        L.append(f"| {k} | {v['accuracy_words_only']:.1%} ({v['interval_words_only'][0]:.0%}–{v['interval_words_only'][1]:.0%}) | "
                 f"{v['accuracy']:.1%} ({v['interval'][0]:.0%}–{v['interval'][1]:.0%}) | {kk('recall')} | {kk('update')} | "
                 f"{kk('absent')} | {v['cost']['per_question_usd']:.4f} | {v['cost']['per_correct_usd'] or 0:.4f} | "
                 f"{v['cost']['write_path_usd']:.4f} | {vs} |")
    if "notes" in r["arms"]:
        n = r["arms"]["notes"]["notes"]
        L += ["", "## Notes", "", f"- size per scope (tokens, estimated): {n['tokens_per_scope']}",
              f"- over the {SWITCH_TOKENS:,}-token switch point: {n['over_switch_point'] or 'none'}",
              f"- refused by the evidence check: {n['refused']} {n['refused_reasons'] or ''}"]
    L += ["", "## Tokens and latency", ""]
    for k, v in r["arms"].items():
        lat, own = v["latency"]["answer_call"], v["latency"].get("recant_overhead") or {}
        L.append(f"- {k}: {v['calls']} calls, tokens {v['tokens']}; answer call p50 {lat['p50']} s, p95 {lat['p95']} s; "
                 f"Recant's own time per call p50 {own.get('p50')} ms, p95 {own.get('p95')} ms ({own.get('n', 0)} calls); "
                 f"write path {v['latency']['write_path_s']:.1f} s, off the request's path")
    return "\n".join(L)
