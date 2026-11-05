"""The first extraction pass: a marked span becomes a proposed memory, or a logged reason why not.

One frontier-tier call per marked span (spans on the same lines share a call), then function: the quote
is verbatim, the anchor lies inside what was shown, the actor class follows the source line, the record
validates against the store's own schema, a near-duplicate claim is not written twice, and two same-day
claims in one scope that half-overlap are paired for review. Every candidate is logged in the trainable
shape whether written or not; a span decided under this fingerprint is never paid for again."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ulid import ULID

from ..machinery import Machinery
from ..records import KINDS, RecordError, is_ulid as is_ulid_, parse_episode_id, validate_record, validate_task
from ..redact import redact_secrets
from ..store.episodes import read_manifest
from ..store.layout import StoreLayout
from ..store.memory import MemoryStore, RejectedWrite
from ..store.tasks import TaskStore
from .common import numbered, obj, parse_json, request, run_all
from .draft import provenance
from .nominate import Span, _PLAIN

NOT_A_MEMORY = ("gauge_reading", "transient_metric", "dead_pointer", "protocol_exhaust", "task", "restates", "no_proposition",
                "own_tool_failure")    # the agent's own failed or repeated tool calls: served back, they prime the failure
MAX_TOKENS = 4000               # the reader's cap, for the reader's reason: thinking counts against it
CONTRADICTION_BAND = (0.3, 0.6)  # same day, same scope, this much claim overlap: paired for review, both written
REVERSAL_SHARED_WORDS = 2        # pass 2: same day, no scope match needed, this many distinctive words shared: paired
_TASK_SHAPED = re.compile(r"\b(pr\s*#?\d+|feature/[\w./-]+|fix/[\w./-]+|rebuild/[\w./-]+|branch\s+`?[\w./-]+|phase\s+\d|lane\b|the brief\b|item\s+[a-z]?\d|"
                          r"one-time|one-off|this round|this sprint|this session|for now\b|green[- ]light|go-ahead|merge\s+now|today only)", re.I)
_STANDING = re.compile(r"\b(always|never|standing|from now on|whenever|every time|must not|going forward|rule|default)\b", re.I)


def task_shaped(claim: str) -> bool:
    """Pass 2: a directive whose scope is one job — it names a branch,
    PR, phase, lane, brief or item, or a one-time gate — and states no standing rule, is task state. A
    function over the claim, so the model's `directive` cannot override it."""
    return bool(_TASK_SHAPED.search(claim)) and not _STANDING.search(claim)
CROSS_REF_LINES = 12            # same-day lines from the other source shown beside the span, at most
CROSS_REF_MIN_WORDS = 3         # a same-day line must share this many distinctive words with the quote
_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_ROLE = re.compile(r"^\[[^\]]*\]\s*(\w+):")        # `[main/sessions/x#L5] User: text`
_QUOTED = re.compile(r"[\"“]([^\"”]{12,})[\"”]")
_STOP = frozenset("the and for was were are not has had its his her this that with from into about after again "
                  "against all also because been before being between could did does doing down during each further "
                  "having here just more most much once only other over same should some such than their them then "
                  "there these they those through under until very what when where which while will would your".split())

EXTRACT_SCHEMA = obj({
    "verdict": {"type": "string", "enum": ["memory", "not_a_memory"]},
    "not_a_memory_reason": {"type": "string", "enum": [*NOT_A_MEMORY, "none"]},
    "claim": {"type": "string"},
    "kind": {"type": "string", "enum": [*KINDS, "none"]},
    "actor_class": {"type": "string", "enum": ["principal", "agent", "none"]},
    "actor_id": {"type": "string"},
    "quote": {"type": "string"},
    "quote_start_line": {"type": "integer"},
    "quote_end_line": {"type": "integer"},
    "channel": {"type": "string"},
    "timestamp": {"type": "string"},
    "scope": {"type": "string"},
    "supersession_condition": {"type": "string"},
    "notes": {"type": "string"}})


def extract_schema(split: bool = False, keys: bool = False) -> dict:
    """The extractor's answer. Memory v2, each part only when the kernel turns it on: `more_facts`,
    one further memory per further fact the moment states, each with its own quote and lines, the actor the moment's
   ; `attribute`, the key of what a claim is about within its scope."""
    if not split and not keys:
        return EXTRACT_SCHEMA
    props = dict(EXTRACT_SCHEMA["properties"])
    attr = {"attribute": {"type": "string"}} if keys else {}
    props.update(attr)
    if split:
        props["more_facts"] = {"type": "array", "items": obj({
            "claim": {"type": "string"}, "kind": {"type": "string", "enum": [*KINDS, "none"]}, "quote": {"type": "string"},
            "quote_start_line": {"type": "integer"}, "quote_end_line": {"type": "integer"}, "scope": {"type": "string"},
            "supersession_condition": {"type": "string"}, "notes": {"type": "string"}, **attr})}
    return obj(props)


def key_of(scope: str, attribute: str, entity: str | None = None) -> str | None:
    """A memory's key: its scope, entity and attribute, normalized — lower case, words joined by
    underscores, the last word singular — so the same attribute stated in other words lands on the same key. The entity
    keeps apart the people, accounts, sites or periods a scope has several of; "-" (or none) when it has one."""
    def norm(x: str) -> str:
        w = [t for t in re.split(r"[^a-z0-9]+", (x or "").lower()) if t]
        if w:
            last = w[-1]
            if last.endswith("ies") and len(last) > 4:
                w[-1] = last[:-3] + "y"
            elif last.endswith("s") and not last.endswith(("ss", "us", "is")) and len(last) > 3:
                w[-1] = last[:-1]
        return "_".join(w)
    a = norm(attribute)
    sc = re.sub(r"^(the|a|an|our|my|their|his|her)_", "", norm(scope))   # "the company" and "company" are one scope
    en = re.sub(r"^(the|a|an|our|my|their|his|her)_", "", norm(entity or "")) or "-"
    return f"{sc}::{en}::{a}" if a and sc else None


def near_key(k: str, keys) -> str | None:
    """An existing key the new one may mean: same scope and entity, an attribute sharing at least half its
    words or containing the other; or same scope and attribute under another entity (an entity that may be the value
    itself: "site_lyon" against "site_cork"). A supersession that turns on such a match is not applied: it goes to
    review. Values true at once are `concurrent` and never reach this check."""
    try:
        sc, en, at = k.split("::")
    except ValueError:
        return None
    words = set(at.split("_"))
    for o in keys:
        try:
            osc, oen, oat = o.split("::")
        except ValueError:
            continue
        if o == k or osc != sc:
            continue
        if oat == at and oen != en:
            return o
        if oen != en:
            continue
        ow = set(oat.split("_"))
        if at in oat or oat in at or len(words & ow) / max(1, len(words | ow)) >= 0.5:
            return o
    return None


# what memory may not hold on the agent's own say-so
ABSENCE = re.compile(r"\b(never (been )?(provided|told|given|stated|shared|mentioned|saved)|(has|have|had) not (yet )?(provided|given|stated|shared|told)|(has|have|had|was|were) not (yet )?been "
                     r"(provided|told|given|stated|shared)|not (on file|on record|in (the )?(store|memory|records?))|no record (of|on file)|"
                     r"(is|are|remains?) unknown)\b", re.I)
SELF = re.compile(r"\b(the (agent|assistant)|ai assistant|this assistant)\b.{0,60}\b(cannot|can't|can not|can only|has no|have no|"
                  r"is unable|lacks|does not have|doesn't have|no (access|ability|capability|mechanism|way))\b", re.I)
_YOU_LIMIT = re.compile(r"\byou\b.{0,40}\b(cannot|can't|can not|can only|have no|don't have|do not have|no access|aren't able)\b", re.I)


def cited_roles(lines: list[str], qs: int, qe: int) -> list[tuple[str | None, str]]:
    """(role, text) of each cited line in the session format: user, assistant, tool, tool_failed; continuation lines
    take the turn's role. Empty when the file is not in the session format."""
    from ..store.hot import FAILED_MARK, LINE
    out, role = [], None
    for i, raw in enumerate(lines, 1):
        m = LINE.match(raw)
        if m and m.group("role") != "  ":
            role = {"User": "user", "Assistant": "assistant", "Tool": "tool_failed" if m.group("text").startswith(FAILED_MARK) else "tool"}[m.group("role")]
        if qs <= i <= qe:
            out.append((role if m else None, m.group("text") if m else raw))
    return out if any(r for r, _ in out) else []


def evidence_ok(lines: list[str], qs: int, qe: int) -> bool | None:
    """True when the cited lines include a turn of someone other than the agent or a tool's result that did not
    fail; False when they are only the agent's own words or its failed calls (its echo, narration, claims, or account of
    its own failures). None when the file is not in the session format (the corpus): the rule does not apply."""
    from ..store.hot import FAILED_MARK, LINE
    roles, role, parsed = [], None, False
    for raw in lines:
        m = LINE.match(raw)
        if not m:
            roles.append(None)
            continue
        parsed = True
        lab = m.group("role")
        if lab != "  ":
            role = {"User": "user", "Assistant": "assistant", "Tool": "tool_failed" if m.group("text").startswith(FAILED_MARK) else "tool"}[lab]
        roles.append(role)
    if not parsed:
        return None
    return any(r in ("user", "tool") for r in roles[qs - 1:qe])


def _same_quote(a, b) -> bool:
    """Two anchors on overlapping lines are one fact only when their quoted text overlaps (several facts may share a
    line); an anchor whose quote is not known matches on its lines alone, as before."""
    if not a or not b:
        return True
    x, y = squash(a), squash(b)
    return bool(x) and bool(y) and (x in y or y in x)


@dataclass
class Target:
    """One call: the lines of a marked moment, every marker that fired on them, and what is shown."""
    path: str
    start: int
    end: int
    markers: list[str]
    quote: str
    a: int = 0
    b: int = 0
    shown: str = ""
    cross: list[tuple[str, int, str]] = field(default_factory=list)   # (path, line number, text)

    @property
    def episode(self) -> str:
        return f"{self.path}#L{self.start}-L{self.end}"


def squash(t: str) -> str:
    return " ".join(t.translate(_PLAIN).split())


def claim_words(text_: str) -> frozenset[str]:
    """Content words of a claim; a number counts however short, because it is often the whole difference."""
    words = (w.rstrip(".,;:!?'\"") for w in re.findall(r"[a-z0-9][a-z0-9_.'-]*", text_.lower()))
    return frozenset(w for w in words if w and (len(w) > 2 or any(c.isdigit() for c in w)) and w not in _STOP)


def claim_overlap(a: str, b: str) -> float:
    x, y = claim_words(a), claim_words(b)
    return len(x & y) / len(x | y) if x | y else 0.0


def numbers_of(text_: str) -> frozenset[str]:
    """Every token carrying a digit: a count, a version, a date. Two claims that differ in one of these
    state different facts however alike their words are, so they are never folded as duplicates."""
    return frozenset(w.rstrip(".,;:") for w in re.findall(r"[a-z0-9][a-z0-9_.'-]*", text_.lower()) if any(c.isdigit() for c in w))


def signature(text_: str) -> frozenset[str]:
    """The tokens that pin a claim to one fact: a path, a file name, or a number of three characters or more (a
    port, a version, an id). A bare small number ("cache key 1") or a date places a claim but does not identify it:
    two different facts of one day share the date."""
    out = set()
    for w in re.findall(r"[a-z0-9][a-z0-9_./'-]*", text_.lower()):
        w = w.rstrip(".,;:)")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", w):
            continue
        if "/" in w or re.search(r"\.[a-z]{2,4}$", w) or (any(c.isdigit() for c in w) and len(w) >= 3):
            out.add(w)
    return frozenset(out)


_STATUS = re.compile(r"\b(now|no longer|fixed|fixes|broken|was|were|previously|formerly|changed|instead|switched|replaced|"
                     r"reverted|moved|used to|resolved|unresolved|open|closed|done|failing|fails|passing|deprecated|removed)\b")
FOLD_EXEMPT_MARKERS = frozenset({"reversal", "root_cause", "failure_with_reason", "manner_correction"})


def status_words(text_: str) -> frozenset[str]:
    """The words that say what state a fact is in, or that it changed: a claim saying "now" and one saying "was"
    describe different moments of the same thing, and one must not be folded into the other."""
    return frozenset(_STATUS.findall(text_.lower()))


def held_already(claim: str, known_claim: str, at: float) -> bool:
    """Write-time dedupe: the store already holds this fact when every signature token of the candidate is in the
    known claim (a claim naming its own path or id is a different fact), it brings no number the known claim lacks
    (a new number may be a changed value, which is written so pairing and supersession see it), and the two overlap
    at `at`."""
    sig = signature(claim)
    return bool(sig) and sig <= signature(known_claim) and numbers_of(claim) <= numbers_of(known_claim) \
        and status_words(claim) == status_words(known_claim) and claim_overlap(claim, known_claim) >= at


def same_fact(a: str, b: str, threshold: float) -> bool:
    return numbers_of(a) == numbers_of(b) and claim_overlap(a, b) >= threshold


def distinctive(quote: str) -> list[str]:
    return sorted({w for w in re.findall(r"[a-z0-9][a-z0-9_.'-]*", quote.lower()) if len(w) >= 5 and w not in _STOP})


def day_of(path: str) -> str | None:
    """The day an episode is from: the last date anywhere in its path. A daily or a corpus transcript names it in the
    file (`memory/2026-09-03.md`); a live session file in its folder (`hot/sessions/2026-09-29/<id>.md`), so reading the
    file name alone would date that session's tasks 1970-01-01."""
    found = _DATE.findall(path)
    return found[-1] if found else None


def task_day(anchor_ts: str | None, path: str) -> str | None:
    """A task's opened day: its anchor's timestamp, else its episode's day; never a made-up one."""
    ts = (anchor_ts or "")[:10]
    if _DATE.fullmatch(ts) and ts >= "2000-01-01":
        return ts
    return day_of(path)


def line_role(line: str) -> str | None:
    """Who a transcript line is from, when it says: `User`, `Assistant`, `System`..."""
    m = _ROLE.match(line)
    return m.group(1) if m else None


def own_tool_failure(lines: list[str], qs: int, qe: int, claim: str) -> bool:
    """Structural: True when every cited line (qs..qe, 1-based) is the agent's own failed
    tool call, a retry of one, or the agent's own account of one — whatever the tool or the wording. A tool line marked
    failed (session format v2) is a failure; an agent line is an account of one when it names a tool that failed in
    the session, or when the claim names one. A user line or a tool that ran is other evidence: an environment fact
    found that way is kept. Files without the session format (the corpus) never match."""
    from ..store.hot import FAILED_MARK, LINE
    roles, failed_names = [], set()
    role = None
    for raw in lines:
        m = LINE.match(raw)
        if not m:
            roles.append((None, raw))
            continue
        r = {"User": "user", "Assistant": "assistant", "Tool": "tool"}.get(m.group("role"), role)    # "  " continues
        role = r
        text_ = m.group("text")
        is_failed = m.group("role") == "Tool" and text_.startswith(FAILED_MARK)
        if is_failed:
            failed_names.add(text_[len(FAILED_MARK):].split(":", 1)[0].strip())
        roles.append((("tool_failed" if is_failed else r) if m.group("role") != "  " else
                      (roles[-1][0] if roles else r), text_))
    if not failed_names:
        return False
    names = {n for n in failed_names if n}
    pat = re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + r")\b") if names else None
    claim_names_one = bool(pat and pat.search(claim or ""))
    cited = roles[qs - 1:qe]
    if not cited:
        return False
    for r, text_ in cited:
        if r == "tool_failed":
            continue
        if r == "assistant" and (claim_names_one or (pat and pat.search(text_))):
            continue
        return False
    return True


def principal_allowed(path: str, lines: list[str], principal_id: str) -> bool:
    """The actor-class rule: in a transcript the anchored line must be the
    principal's own turn; in a daily it must quote the principal, by name or in quotation marks. A line that
    reports the principal without quoting them is the agent's."""
    if any(line_role(l) == "User" for l in lines):
        return True
    if any(line_role(l) is not None for l in lines):
        return False
    who = re.escape(principal_id)
    text_ = "\n".join(lines)
    return bool(re.search(rf"(?i)\b{who}\b\s*(:|said|says|wrote|asked|told|replied|ruled)", text_)
                or _QUOTED.search(text_))


def group_spans(spans: list[Span]) -> list[Target]:
    """Spans on the same lines are one moment carrying several markers: one call, not one per marker."""
    by_lines: dict[tuple[str, int, int], Target] = {}
    for s in sorted(spans, key=lambda s: (s.path, s.start, s.end, s.marker)):
        key = (s.path, s.start, s.end)
        if key not in by_lines:
            by_lines[key] = Target(s.path, s.start, s.end, [], s.quote)
        if s.marker not in by_lines[key].markers:
            by_lines[key].markers.append(s.marker)
    return list(by_lines.values())


def decided_spans(layout: StoreLayout, tenant: str, fingerprint: str, job: str = "extract_first_pass", store: str | None = None) -> set[str]:
    """Span episode ids an earlier run of `job` already decided under this fingerprint. A dry run
    (job `extract_dry_run`) decides nothing. `store` narrows to the memory pass's records (None) or the ledger's
    ("tasks") when both passes share one run, as a live run does (keyed on the harness's job name alone,
    a second run for the same day would re-decide every span)."""
    from ..log import read_run
    out: set[str] = set()
    for path in sorted(layout.log_dir(tenant).glob("*.jsonl")):
        recs = list(read_run(path, typed=False))
        if not recs or recs[0].get("job") != job or recs[0].get("params", {}).get("extractor_fingerprint") != fingerprint:
            continue
        out.update(r["episode"] for r in recs if r["kind"] == "extraction" and r.get("store") == store)
    return out


class Extractor:
    def __init__(self, layout: StoreLayout, tenant: str, machinery: Machinery, store: MemoryStore, provider,
                 log, *, principal_id: str, job: str = "first_pass", batch: bool = True,
                 wait=lambda: None, dry_run: bool = False, agent_id: str = "agent", model: str | None = None):
        ex = getattr(machinery.artifact.extractor, job)
        if ex.prompt is None:
            raise ValueError(f"the artifact has no {job} extractor prompt")
        self.model = model or ex.model                      # the live path passes the tier table's model
        self.layout, self.tenant, self.m, self.store, self.provider = layout, tenant, machinery, store, provider
        self.log, self.batch, self.wait, self.dry_run = log, batch, wait, dry_run
        self.job, self.ex, self.principal_id, self.agent_id = job, ex, principal_id, agent_id
        self.fingerprint = machinery.extractor_fingerprint(job)
        self.threshold = machinery.artifact.dedup_threshold if machinery.artifact.dedup_threshold is not None else 0.6
        self.context = machinery.kernel.eval.context_lines
        self.cold = layout.tier_dir(tenant, "cold")
        self._texts: dict[str, list[str]] = {}
        self.manifest = [e["path"] for e in read_manifest(layout, tenant)]
        # What is already in the store, for the duplicate check: claim, scope, id, day. Pass 2 keeps every
        # non-retired entry's state too, so a restatement attaches to a probation/live survivor rather than
        # being written again, and a version pin can name the entry it supersedes.
        self.known: list[tuple[str, str, str, str | None]] = []
        self.known_state: dict[str, str] = {}
        # every entry's anchor, any state: a candidate on a line the store already holds an
        # entry for is a duplicate of it — attached if that entry stands, dropped if a reviewer retired it
        self.known_anchors: list[tuple[str, int, int, str, str, str | None]] = []
        self.run_written: list[tuple[str, str]] = []            # (claim, id) written by this run, for the within-run overlap rule
        self.rules = getattr(machinery.kernel, "extraction", None)
        self.keyed: dict[str, tuple[str, str]] = {}               # standing entry per key -> (id, claim)
        for entry in store.list(tenant):
            rec, _ = store.get(tenant, entry["id"])
            try:
                ref = parse_episode_id(rec.anchor.episode)
                self.known_anchors.append((ref.path, ref.start, ref.end, rec.id, entry["state"], getattr(rec.anchor, "quote", None)))
            except Exception:                                   # a corpus-era anchor shape the parser rejects: not a twin of anything
                pass
            if entry["state"] in ("superseded", "retired"):
                continue
            if getattr(rec, "key", None) and getattr(rec, "value_status", None) in (None, "current"):
                self.keyed[rec.key] = (rec.id, rec.claim)
            self.known.append((rec.claim, rec.scope.strip().lower(), rec.id, day_of(rec.anchor.episode)))
            self.known_state[rec.id] = entry["state"]
        self.tasks = None                 # a TaskStore when task-shaped directives are to be routed (pass 2)

    # --- material ------------------------------------------------------------------------------------
    def lines(self, path: str) -> list[str]:
        if path not in self._texts:
            self._texts[path] = self.layout.episode_file(self.tenant, path).read_text(encoding="utf-8", errors="replace").splitlines()
        return self._texts[path]

    def cross_refs(self, t: Target) -> list[tuple[str, int, str]]:
        """Same-day lines from the other source files that share distinctive words with the quote: the daily
        is the full-text source for a transcript moment, and the transcript audits a daily's claim."""
        day, words = day_of(t.path), distinctive(t.quote)
        need = min(CROSS_REF_MIN_WORDS, len(words))
        if not day or not need:
            return []
        out = []
        for other in self.manifest:
            if other == t.path or day_of(other) != day:
                continue
            for n, line in enumerate(self.lines(other), 1):
                low = line.lower()
                if sum(w in low for w in words) >= need:
                    out.append((other, n, line))
                    if len(out) >= CROSS_REF_LINES:
                        return out
        return out

    def prepare(self, t: Target) -> Target:
        n = len(self.lines(t.path))
        t.a, t.b = max(1, t.start - self.context), min(n, t.end + self.context)
        t.shown = numbered(self.lines(t.path)[t.a - 1:t.b], t.a)
        t.cross = self.cross_refs(t)
        return t

    def user_text(self, t: Target) -> str:
        descs = {m.name: m.description for m in self.m.artifact.markers}
        head = [f"Episode: {t.path}", f"Day: {day_of(t.path) or 'unknown'}",
                "Markers that fired on this moment:", *(f"- {m}: {descs.get(m, '')}" for m in t.markers),
                f"The moment is lines {t.start}-{t.end}. Lines shown: {t.a}-{t.b}.", "", t.shown]
        if t.cross:
            head += ["", "Same-day lines from the other source that share words with the moment "
                         "(for context and derived_from; the anchor must stay in the episode above):"]
            head += [f"{p}#L{n}: {line}" for p, n, line in t.cross]
        if getattr(self, "rules", None) is not None and self.rules.keys:   # reuse a key when the attribute is the same
            keys = sorted(self.keyed)[:200]
            head += ["", "Keys already in the store (scope::entity::attribute — current claim):" if keys else "Keys already in the store: none yet."]
            head += [f"- {k} — {self.keyed[k][1][:120]}" for k in keys]
        return "\n".join(head)

    def request(self, t: Target, cid: str):
        return request("extract", model=self.model, system=self.ex.prompt, user=self.user_text(t),
                       schema=extract_schema(bool(self.rules and self.rules.split_facts), bool(self.rules and self.rules.keys)),
                       max_tokens=MAX_TOKENS, custom_id=cid, reasoning=self.ex.reasoning)

    # --- the checks ------------------------------------------------------------------------------------
    def _log(self, t: Target, candidate, *, verdict: str, reason: str | None = None, spans=None, written_as=None):
        self.log.emit("extraction", episode=t.episode, candidate=candidate, spans=spans or [], model_id=self.model,
                      prompt_version=self.ex.prompt_version, verdict=verdict, rejection_reason=reason, written_as=written_as)
        if verdict == "rejected":
            self.log.count(f"rejected: {reason.split(' ', 1)[0].rstrip(':')}" if reason else "rejected")

    def build(self, t: Target, d: dict) -> tuple[dict | None, str | None, dict]:
        """(record, rejection reason, extras). The checks run in a fixed order; the first that fails names the reason."""
        extras: dict = {}
        if d.get("verdict") == "not_a_memory":
            why = d.get("not_a_memory_reason") if d.get("not_a_memory_reason") in NOT_A_MEMORY else "no_proposition"
            self.log.count(f"not_a_memory: {why}")
            return None, f"not_a_memory: {why}", extras
        try:
            qs, qe = int(d["quote_start_line"]), int(d["quote_end_line"])
        except (KeyError, TypeError, ValueError):
            return None, "malformed: quote lines", extras
        quote = str(d.get("quote", ""))
        if getattr(self, "rules", None) is not None and self.rules.evidence_rule and "\\u" in quote:
            quote = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), quote)   # "\u0026" written for "&"
        if not (t.a <= qs <= qe <= t.b):
            return None, f"episode_outside_span: quote lines {qs}-{qe} are outside the lines shown, {t.a}-{t.b}", extras
        cited = self.lines(t.path)[qs - 1:qe]
        if not quote.strip() or squash(quote) not in squash("\n".join(cited)):
            return None, "quote_not_verbatim", extras
        if getattr(self, "rules", None) is not None and self.rules.evidence_rule and evidence_ok(self.lines(t.path), qs, qe) is False:
            self.log.count("not_a_memory: agent_words_only")                # the agent's own words are never the only evidence
            return None, "not_a_memory: agent_words_only (structural)", extras
        if getattr(self, "rules", None) is not None and self.rules.evidence_rule:
            claim_ = str(d.get("claim") or "")
            if ABSENCE.search(claim_):                                   # an absence is never evidence
                self.log.count("not_a_memory: absence")
                return None, "not_a_memory: absence (structural)", extras
            if SELF.search(claim_):                                      # what the agent can't do, on its own say-so
                roles = cited_roles(self.lines(t.path), qs, qe)
                if roles and not any(r == "tool" or (r == "user" and _YOU_LIMIT.search(x)) for r, x in roles):
                    self.log.count("not_a_memory: agent_self_claim")
                    return None, "not_a_memory: agent_self_claim (structural)", extras
        if own_tool_failure(self.lines(t.path), qs, qe, str(d.get("claim") or "")):    # by structure
            self.log.count("not_a_memory: own_tool_failure")
            return None, "not_a_memory: own_tool_failure (structural)", extras
        # The actor class is the rule's, not the model's (function beats the model): principal only where the
        # anchored line is the principal's own turn or quotes them; otherwise the agent's, at trust rank 3.
        wanted = "principal" if principal_allowed(t.path, cited, self.principal_id) and d.get("actor_class") == "principal" else "agent"
        if d.get("actor_class") == "principal" and wanted == "agent":
            self.log.count("actor_class_overridden")
            extras["actor_class_overridden"] = True
        # an agent's line is never `stated` — what the agent says of its own tools, limits or
        # manner is the identity file's, not a memory; an agent turn may still yield `concluded`
        live = self.job == "steady_state"                # the live pass's extractor; the harness's measured first pass is unchanged
        if live and wanted == "agent" and d.get("kind") == "stated" and (self.rules is None or self.rules.agent_lines_never_stated):
            return None, "agent_self_description: an agent line is never written as stated", extras
        # the same anchor line as an entry the store already holds, in any state
        twin = next(((id_, state) for p, s_, e, id_, state, q in self.known_anchors
                     if p == t.path and s_ <= qe and qs <= e and _same_quote(q, quote)), None) if live else None
        if twin:
            if twin[1] not in ("retired", "superseded"):
                extras["attach_to"] = twin[0]
            return None, f"duplicate_anchor {twin[0]} ({twin[1]})", extras
        actor_id = self.principal_id if wanted == "principal" else (str(d.get("actor_id") or "").strip() or self.agent_id)
        if actor_id.strip().lower() == self.principal_id.lower() and wanted == "agent":
            actor_id = f"{self.agent_id} (reporting {self.principal_id})"
        # Secret-shaped strings never land in the store, not even inside a verbatim quote: the quote was
        # checked against the source above, and is stored with the secret replaced ("redaction preserves the trail").
        claim, n1 = redact_secrets(str(d.get("claim", "")))
        notes, n2 = redact_secrets(str(d.get("notes", "")))
        quote, n3 = redact_secrets(quote)
        scope, n4 = redact_secrets(str(d.get("scope", "")))
        cond, n5 = redact_secrets(str(d.get("supersession_condition", "")))
        if n1 + n2 + n3 + n4 + n5:
            self.log.count("secrets_redacted", n1 + n2 + n3 + n4 + n5)
            if n3:
                self.log.count("quotes_redacted")
        derived = [t.episode] + [f"{p}#L{n}-L{n}" for p, n, _ in t.cross]
        record = {"id": str(ULID()), "claim": claim, "kind": d.get("kind"),
                  "actor": {"class": wanted, "id": actor_id},
                  "anchor": {"quote": quote, "episode": f"{t.path}#L{qs}-L{qe}", "channel": str(d.get("channel") or "").strip() or None,
                             "message_id": None, "timestamp": str(d.get("timestamp") or "").strip() or day_of(t.path)},
                  "scope": scope, "supersession_condition": cond,
                  "derived_from": derived,
                  "status": {"state": "proposed", "reason": None, "failure_class": None, "superseded_by": None}}
        keyed = self.rules is not None and self.rules.keys
        vstatus = d.get("value_status") if d.get("value_status") in ("current", "historical", "concurrent") else None
        if keyed:
            k = key_of(scope, str(d.get("attribute") or ""), str(d.get("entity") or "") or None)
            if k:
                record["key"] = k
                record["value_status"] = vstatus or "current"
        try:
            validate_record(record)
        except RecordError as exc:
            return None, f"invalid: {exc}", extras
        extras["notes"] = notes
        scope = record["scope"].strip().lower()
        if keyed and record.get("key") in self.keyed:                # by key first, similarity second
            known_id, known_claim = self.keyed[record["key"]]
            if same_fact(claim, known_claim, self.threshold) and numbers_of(claim) == numbers_of(known_claim) \
                    and not claim_words(claim) - claim_words(known_claim):   # a word added under the same key is a new value
                extras["attach_to"] = known_id
                return None, f"restates {known_id} (same key)", extras
            if record.get("value_status") == "current":              # only a current value supersedes
                record["derived_from"].append(known_id)
                extras["supersedes"] = known_id
                extras["notes"] = (notes + f" Supersedes {known_id}.").strip()
                self.log.count("superseded_by_key")
            else:
                self.log.count(f"same_key_{record.get('value_status')}")
            return record, None, extras
        if keyed and record.get("key") and record.get("value_status") == "current":
            near = near_key(record["key"], self.keyed)
            if near:                                                 # an uncertain key match is never applied
                known_id, _ = self.keyed[near]
                extras["notes"] = (notes + f" Unresolved with {known_id}: uncertain key match ({near} against {record['key']}).").strip()
                extras["unresolved_with"] = known_id
                self.log.count("uncertain_key_match")
                return record, None, extras
        # within this run, a claim overlapping one already written folds to it, numbers or not
        # (the same correction posted twice, in two sessions, is one memory)
        overlap_at = self.rules.run_dedupe_overlap if self.rules is not None else 0.7
        for written_claim, written_id in (self.run_written if live else []):
            if claim_overlap(claim, written_claim) >= overlap_at:
                extras["attach_to"] = written_id
                return None, f"duplicate_in_run {written_id}", extras
        # Pass 2: a restatement of any non-retired entry anywhere in the store — same numbers, overlap at the
        # threshold, scope ignored — attaches its span to that entry instead of being written.
        # A moment the nominator marked as a reversal is never a restatement, however many words it shares
        # with what it reverses: it goes to pairing below.
        if "reversal" not in t.markers:
            for known_claim, known_scope, known_id, _ in self.known:
                if keyed and record.get("key") and claim_words(claim) - claim_words(known_claim):
                    continue                     # a claim adding a word (a new value) is never a restatement
                if same_fact(claim, known_claim, self.threshold):
                    extras["attach_to"] = known_id
                    return None, f"restates {known_id}", extras
            # only plain restatements fold: never a fix, a reversal or a correction, never
            # into an entry that is not standing, never across a change of status (held_already compares them)
            at = self.rules.store_dedupe_overlap if self.rules is not None else None
            if at is not None and not (set(t.markers) & FOLD_EXEMPT_MARKERS):
                for known_claim, known_scope, known_id, _ in self.known:
                    if self.known_state.get(known_id, "proposed") not in ("proposed", "probation", "live"):
                        continue
                    if held_already(claim, known_claim, at):
                        extras["attach_to"] = known_id
                        return None, f"held_already {known_id}", extras
        # A version pin: the same words with different numbers as an existing entry in the same scope. Written,
        # deriving from the earlier entry and noting that it supersedes it; review's accept moves the earlier one.
        for known_claim, known_scope, known_id, _ in self.known:
            if known_scope == scope and claim_overlap(claim, known_claim) >= self.threshold and numbers_of(claim) != numbers_of(known_claim) \
                    and numbers_of(claim) and numbers_of(known_claim):
                record["derived_from"].append(known_id)
                extras["supersedes"] = known_id
                extras["notes"] = (notes + f" Supersedes {known_id}.").strip()
                break
        # A task-shaped directive is the ledger's, decided by a function.
        if record["kind"] == "directive" and task_shaped(claim):
            extras["route_to_tasks"] = True
        day = day_of(t.path)
        lo, hi = CONTRADICTION_BAND
        cw = claim_words(claim)
        for known_claim, known_scope, known_id, known_day in self.known:
            if known_day != day or known_id in record["derived_from"]:
                continue
            ov = claim_overlap(claim, known_claim)
            shared = len([w for w in cw & claim_words(known_claim) if len(w) >= 5])
            if (known_scope == scope and lo <= ov < hi) or shared >= REVERSAL_SHARED_WORDS:
                record["derived_from"].append(known_id)
                extras["paired_with"] = known_id
                break
        return record, None, extras

    def _attach(self, survivor_id: str, t: Target) -> None:
        """A second sighting of a fact is evidence for the entry that holds it: the span joins its derived_from."""
        rec, notes = self.store.get(self.tenant, survivor_id)
        data = rec.to_dict()
        if t.episode not in data["derived_from"]:
            data["derived_from"] = data["derived_from"] + [t.episode]
            self.store.put(self.tenant, data, writer="consolidation", log=self.log, notes=notes)

    def _route_to_tasks(self, record: dict, t: Target, notes: str) -> None:
        """A task-shaped directive becomes an open task in the ledger, from the same fields; no second call."""
        opened = task_day((record.get("anchor") or {}).get("timestamp"), t.path)
        if opened is None:
            self.log.count("task_without_a_date")
            return
        task = {"id": str(ULID()), "claim": record["claim"], "scope": record["scope"], "actor": record["actor"],
                "anchor": record["anchor"], "opened": opened, "status": "open", "closed": None,
                "closed_by": None, "superseded_by": None, "derived_from": [d for d in record["derived_from"] if not is_ulid_(d)]}
        try:
            self.tasks.put(self.tenant, task, writer="consolidation", log=self.log, notes=notes)
        except RejectedWrite:
            self.log.count("write_rejected")

    def provenance_of(self, record: dict, notes: str, t: Target) -> list[dict]:
        source = "\n".join([*self.lines(t.path)[t.a - 1:t.b], *(line for _, _, line in t.cross)])
        spans = []
        for fld in ("claim", "scope", "supersession_condition"):
            spans += [{**sp, "episode": t.episode} for sp in provenance(record[fld], source, field=fld)]
        if notes:
            spans += [{**sp, "episode": t.episode} for sp in provenance(notes, source, field="notes")]
        return spans

    # --- the pass --------------------------------------------------------------------------------------
    def run(self, spans: list[Span]) -> dict:
        targets = [self.prepare(t) for t in group_spans(spans)]
        skip = set() if self.dry_run else decided_spans(self.layout, self.tenant, self.fingerprint, job=self.log.job)
        todo = [t for t in targets if t.episode not in skip]
        self.log.count("spans_in", len(spans))
        self.log.count("moments", len(targets))
        self.log.count("already_decided", len(targets) - len(todo))
        reqs, meta = [], {}
        for i, t in enumerate(todo):
            meta[f"e{i}"] = t
            reqs.append(self.request(t, f"e{i}"))
        out = {"calls": len(reqs), "written": 0, "accepted": 0, "rejected": 0, "candidates": []}
        for cid, resp in run_all(self.provider, reqs, batch=self.batch, wait=self.wait):
            t = meta[cid]
            self.log.examined += 1
            d, err = parse_json(resp)
            if err or not isinstance(d, dict):
                self._log(t, {"raw": getattr(resp, "text", None) if not err else None}, verdict="rejected", reason=f"malformed: {err or 'not an object'}")
                out["rejected"] += 1
                continue
            decisions = [d] + ([{**d, **x, "verdict": "memory", "not_a_memory_reason": "none", "more_facts": []}
                                for x in (d.get("more_facts") or []) if isinstance(x, dict)] if d.get("verdict") == "memory" else [])
            if len(decisions) > 1:
                self.log.count("more_facts", len(decisions) - 1)
            for d in decisions:
                self._decide(t, d, out)
        return out

    def _decide(self, t: Target, d: dict, out: dict) -> None:
        """One fact's decision: built, checked, routed or written (each further fact of a moment comes here too)."""
        record, why, extras = self.build(t, d)
        if record is None:
            if extras.get("attach_to") and not self.dry_run:
                self._attach(extras["attach_to"], t)
                self.log.count("attached_to_survivor")
                out["attached"] = out.get("attached", 0) + 1
            self._log(t, d, verdict="rejected", reason=why)
            out["rejected"] += 1
            return
        spans_ = self.provenance_of(record, extras.get("notes", ""), t)
        if extras.get("paired_with"):
            self.log.count("contradiction_pairs")
        if extras.get("supersedes"):
            self.log.count("supersedes_earlier")
        if extras.get("route_to_tasks") and self.tasks is not None:
            self.log.count("routed_to_tasks")
            out["routed_to_tasks"] = out.get("routed_to_tasks", 0) + 1
            if not self.dry_run:
                self._route_to_tasks(record, t, extras.get("notes", ""))
            self._log(t, record, verdict="rejected", reason="task_shaped: routed to the ledger", spans=spans_)
            out["rejected"] += 1
            return
        out["accepted"] += 1
        out["candidates"].append({"episode": t.episode, "markers": t.markers, "record": record, "notes": extras.get("notes", ""),
                                  "paired_with": extras.get("paired_with"), "actor_class_overridden": extras.get("actor_class_overridden", False)})
        if self.dry_run:
            self._log(t, record, verdict="accepted", spans=spans_)
            return
        try:
            self.store.put(self.tenant, record, writer="consolidation", log=self.log, notes=extras.get("notes", ""))
        except RejectedWrite as exc:      # already dead-lettered by the store
            self.log.count("write_rejected")
            self._log(t, record, verdict="rejected", reason=f"write_rejected: {exc}", spans=spans_)
            out["rejected"] += 1
            return
        self.known.append((record["claim"], record["scope"].strip().lower(), record["id"], day_of(t.path)))
        self.run_written.append((record["claim"], record["id"]))
        if record.get("key") and record.get("value_status") in (None, "current"):
            self.keyed[record["key"]] = (record["id"], record["claim"])
        ep = record["anchor"]["episode"]
        self.known_anchors.append((t.path, int(ep.rsplit("#L", 1)[1].split("-L")[0]), int(ep.rsplit("-L", 1)[1]), record["id"], "proposed",
                                   (record.get("anchor") or {}).get("quote")))
        self.log.promoted += 1
        out["written"] += 1
        self._log(t, record, verdict="accepted", spans=spans_, written_as=record["id"])


# --- the task ledger's extractor: the `task` class is a route, not a refusal --------------

NOT_A_TASK = ("is_memory", "protocol_exhaust", "gauge_reading", "no_proposition")
TASK_SCHEMA = obj({
    "verdict": {"type": "string", "enum": ["task", "not_a_task"]},
    "not_a_task_reason": {"type": "string", "enum": [*NOT_A_TASK, "none"]},
    "claim": {"type": "string"},
    "scope": {"type": "string"},
    "actor_class": {"type": "string", "enum": ["principal", "agent", "none"]},
    "actor_id": {"type": "string"},
    "quote": {"type": "string"},
    "quote_start_line": {"type": "integer"},
    "quote_end_line": {"type": "integer"},
    "status": {"type": "string", "enum": ["open", "done", "dropped", "none"]},
    "closed_line": {"type": "integer"},
    "channel": {"type": "string"},
    "timestamp": {"type": "string"},
    "notes": {"type": "string"}})
CLOSE_OVERLAP = 0.5   # a done/dropped task claim overlapping an open one in the same scope this much closes it


def task_spans_from_log(layout: StoreLayout, tenant: str, fingerprint: str) -> list[Span]:
    """The spans the first pass routed as `task` under the memory extractor's fingerprint, rebuilt from the
    log with their markers from the nomination run, so the ledger pays for no nomination."""
    from ..log import read_run
    spans, markers = [], {}
    for path in sorted(layout.log_dir(tenant).glob("*.jsonl")):
        recs = list(read_run(path, typed=False))
        if not recs:
            continue
        if recs[0].get("job") == "nominate":
            for r in recs:
                if r["kind"] == "nomination":
                    for m in r["markers"]:
                        markers.setdefault(m["span"], (m["marker"], m["quote"]))
        if recs[0].get("job") == "extract_first_pass" and recs[0].get("params", {}).get("extractor_fingerprint") == fingerprint:
            for r in recs:
                if r["kind"] == "extraction" and (r.get("rejection_reason") or "") == "not_a_memory: task":
                    ref = parse_episode_id(r["episode"])
                    marker, quote = markers.get(r["episode"], ("rule_or_decision", ""))
                    spans.append(Span(marker, r["episode"], ref.path, ref.start, ref.end, quote))
    return spans


class TaskExtractor(Extractor):
    """Same material, same checks, a different record and store. Reuses the memory extractor's windowing,
    cross-references, actor rule and redaction; writes `open` or `done` tasks and closes an open task when a
    later span in the same scope carries its completion."""

    def __init__(self, layout, tenant, machinery, tasks: TaskStore, provider, log, *, principal_id: str,
                 batch: bool = True, wait=lambda: None, dry_run: bool = False, agent_id: str = "agent", model: str | None = None):
        ex = machinery.artifact.extractor.tasks
        if ex is None or ex.prompt is None:
            raise ValueError("the artifact has no tasks extractor prompt")
        self.model = model or ex.model
        self.layout, self.tenant, self.m, self.tasks, self.provider = layout, tenant, machinery, tasks, provider
        self.log, self.batch, self.wait, self.dry_run = log, batch, wait, dry_run
        self.job, self.ex, self.principal_id, self.agent_id = "tasks", ex, principal_id, agent_id
        self.fingerprint = machinery.extractor_fingerprint("tasks")
        self.context = machinery.kernel.eval.context_lines
        self.cold = layout.tier_dir(tenant, "cold")
        self._texts = {}
        self.manifest = [e["path"] for e in read_manifest(layout, tenant)]
        self.open: list[tuple[str, str, str]] = []      # (claim, scope, id) of open tasks in the ledger
        for t in tasks.list(tenant, status="open"):
            rec, _ = tasks.get(tenant, t["id"])
            self.open.append((rec.claim, rec.scope.strip().lower(), rec.id))

    def request(self, t: Target, cid: str):
        return request("extract_tasks", model=self.model, system=self.ex.prompt, user=self.user_text(t),
                       schema=TASK_SCHEMA, max_tokens=MAX_TOKENS, custom_id=cid, reasoning=self.ex.reasoning)

    def _log(self, t: Target, candidate, *, verdict: str, reason: str | None = None, spans=None, written_as=None):
        self.log.emit("extraction", episode=t.episode, candidate=candidate, spans=spans or [], model_id=self.model,
                      prompt_version=self.ex.prompt_version, verdict=verdict, rejection_reason=reason, written_as=written_as,
                      store="tasks")
        if verdict == "rejected":
            self.log.count(f"rejected: {reason.split(' ', 1)[0].rstrip(':')}" if reason else "rejected")

    def build(self, t: Target, d: dict) -> tuple[dict | None, str | None, dict]:
        extras: dict = {}
        if d.get("verdict") == "not_a_task":
            why = d.get("not_a_task_reason") if d.get("not_a_task_reason") in NOT_A_TASK else "no_proposition"
            self.log.count(f"not_a_task: {why}")
            return None, f"not_a_task: {why}", extras
        try:
            qs, qe = int(d["quote_start_line"]), int(d["quote_end_line"])
        except (KeyError, TypeError, ValueError):
            return None, "malformed: quote lines", extras
        quote = str(d.get("quote", ""))
        if not (t.a <= qs <= qe <= t.b):
            return None, f"episode_outside_span: quote lines {qs}-{qe} are outside the lines shown, {t.a}-{t.b}", extras
        cited = self.lines(t.path)[qs - 1:qe]
        if not quote.strip() or squash(quote) not in squash("\n".join(cited)):
            return None, "quote_not_verbatim", extras
        wanted = "principal" if principal_allowed(t.path, cited, self.principal_id) and d.get("actor_class") == "principal" else "agent"
        if d.get("actor_class") == "principal" and wanted == "agent":
            self.log.count("actor_class_overridden")
            extras["actor_class_overridden"] = True
        actor_id = self.principal_id if wanted == "principal" else (str(d.get("actor_id") or "").strip() or self.agent_id)
        if actor_id.strip().lower() == self.principal_id.lower() and wanted == "agent":
            actor_id = f"{self.agent_id} (reporting {self.principal_id})"
        claim, n1 = redact_secrets(str(d.get("claim", "")))
        notes, n2 = redact_secrets(str(d.get("notes", "")))
        quote, n3 = redact_secrets(quote)
        scope, n4 = redact_secrets(str(d.get("scope", "")))
        if n1 + n2 + n3 + n4:
            self.log.count("secrets_redacted", n1 + n2 + n3 + n4)
        day = task_day(str(d.get("timestamp") or ""), t.path)
        if day is None:
            return None, "no_date", extras                 # never the epoch: no date is not 1970-01-01
        status = d.get("status") if d.get("status") in ("open", "done", "dropped") else "open"
        closed_line = int(d.get("closed_line") or 0)
        closed_by = f"{t.path}#L{closed_line}-L{closed_line}" if status != "open" and t.a <= closed_line <= t.b else None
        record = {"id": str(ULID()), "claim": claim, "scope": scope, "actor": {"class": wanted, "id": actor_id},
                  "anchor": {"quote": quote, "episode": f"{t.path}#L{qs}-L{qe}", "channel": str(d.get("channel") or "").strip() or None,
                             "message_id": None, "timestamp": str(d.get("timestamp") or "").strip() or day},
                  "opened": day, "status": status, "closed": day if status != "open" else None, "closed_by": closed_by,
                  "superseded_by": None, "derived_from": [t.episode] + [f"{p}#L{n}-L{n}" for p, n, _ in t.cross]}
        try:
            validate_task(record)
        except RecordError as exc:
            return None, f"invalid: {exc}", extras
        extras["notes"] = notes
        # A completion closes the open task it completes (open/close pairs), rather than writing a sibling.
        if status != "open":
            for open_claim, open_scope, open_id in self.open:
                if open_scope == scope.strip().lower() and claim_overlap(claim, open_claim) >= CLOSE_OVERLAP:
                    extras["closes"] = open_id
                    break
        elif any(open_scope == scope.strip().lower() and same_fact(claim, open_claim, self.threshold) for open_claim, open_scope, _ in self.open):
            return None, "duplicate_of open task", extras
        return record, None, extras

    threshold = 0.6

    def run(self, spans: list[Span]) -> dict:
        targets = [self.prepare(t) for t in group_spans(spans)]
        skip = set() if self.dry_run else decided_spans(self.layout, self.tenant, self.fingerprint, job=self.log.job, store="tasks")
        todo = [t for t in targets if t.episode not in skip]
        self.log.count("spans_in", len(spans)); self.log.count("moments", len(targets)); self.log.count("already_decided", len(targets) - len(todo))
        reqs, meta = [], {}
        for i, t in enumerate(todo):
            meta[f"t{i}"] = t
            reqs.append(self.request(t, f"t{i}"))
        out = {"calls": len(reqs), "written": 0, "closed": 0, "accepted": 0, "rejected": 0, "candidates": []}
        # Results come back in any order; a closing line must see the task it closes already in the ledger,
        # so they are applied in the record's own order: by file, then line.
        results = sorted(run_all(self.provider, reqs, batch=self.batch, wait=self.wait),
                         key=lambda cr: (meta[cr[0]].path, meta[cr[0]].start))
        for cid, resp in results:
            t = meta[cid]
            self.log.examined += 1
            d, err = parse_json(resp)
            if err or not isinstance(d, dict):
                self._log(t, {"raw": None}, verdict="rejected", reason=f"malformed: {err or 'not an object'}")
                out["rejected"] += 1
                continue
            self.apply(t, d, out)
        return out

    def apply(self, t: Target, d: dict, out: dict) -> None:
        """One task decision: built, checked, then the close of the open task it completes, or a new task."""
        record, why, extras = self.build(t, d)
        if record is None:
            self._log(t, d, verdict="rejected", reason=why)
            out["rejected"] += 1
            return
        spans_ = [{**sp, "episode": t.episode} for fld in ("claim", "scope") for sp in
                  provenance(record[fld], "\n".join([*self.lines(t.path)[t.a - 1:t.b], *(l for _, _, l in t.cross)]), field=fld)]
        out["accepted"] += 1
        out["candidates"].append({"episode": t.episode, "markers": t.markers, "record": record, "closes": extras.get("closes")})
        if self.dry_run:
            self._log(t, record, verdict="accepted", spans=spans_)
            return
        try:
            if extras.get("closes"):
                old, old_notes = self.tasks.get(self.tenant, extras["closes"])
                data = old.to_dict()
                data.update(status=record["status"], closed=record["closed"], closed_by=record["anchor"]["episode"])
                self.tasks.put(self.tenant, data, writer="consolidation", log=self.log, notes=old_notes)
                self.open = [o for o in self.open if o[2] != old.id]
                self.log.count("tasks_closed")
                out["closed"] += 1
                self._log(t, record, verdict="accepted", spans=spans_, written_as=old.id)
                return
            self.tasks.put(self.tenant, record, writer="consolidation", log=self.log, notes=extras.get("notes", ""))
        except RejectedWrite as exc:
            self.log.count("write_rejected")
            self._log(t, record, verdict="rejected", reason=f"write_rejected: {exc}", spans=spans_)
            out["rejected"] += 1
            return
        if record["status"] == "open":
            self.open.append((record["claim"], record["scope"].strip().lower(), record["id"]))
        self.log.promoted += 1
        out["written"] += 1
        self.log.count(f"written: {record['status']}")
        self._log(t, record, verdict="accepted", spans=spans_, written_as=record["id"])
