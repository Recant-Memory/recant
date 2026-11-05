"""Drafting: marked moments, windows no marker fired on, noise, and questions a
person supplied. Every draft is logged with where it came from; every refusal to
draft is logged with why."""
from __future__ import annotations

import math
import random
import re

from ulid import ULID

from ..eval.items import NOTHING, ItemError, validate_item
from ..eval.pools import append_draft
from ..eval.search import grep
from ..machinery import Machinery
from ..store.layout import StoreLayout
from .common import numbered, obj, parse_json, request, run_all
from .nominate import Span

_MONTH = re.compile(r"(\d{4}-\d{2})-\d{2}")

DRAFT_SCHEMA = obj({
    "skip": {"type": "boolean"}, "item_type": {"type": "string", "enum": ["yield", "supersession"]},
    "question": {"type": "string"}, "candidate_answer": {"type": "string"},
    "answer_start_line": {"type": "integer"}, "answer_end_line": {"type": "integer"},
    "superseded_answer": {"type": "string"}, "search_terms": {"type": "array", "items": {"type": "string"}},
    "form": {"type": "string"}})
NOISE_SCHEMA = obj({"skip": {"type": "boolean"}, "question": {"type": "string"},
                    "search_terms": {"type": "array", "items": {"type": "string"}}, "form": {"type": "string"}})
# Room for the model to think and then answer. A cap that only fits the answer returns nothing at all
# when the model thinks first, which a tier-2 model does by default.
MAX_TOKENS = 4000
PRESCREEN_MIN_MATCH = 2   # a line carrying two of a noise draft's search terms is reason enough to hold it

_STOP = frozenset("a an and are as at be been but by did do does for from had has have how i if in is it its of on or "
                  "that the their them they this to was were what when where which who why will with would about".split())


from ..redact import REDACTED, redact_secrets  # noqa: E402,F401  (re-exported for callers)


def opener(question: str) -> str:
    """The word a question opens with, lower-cased: the cheapest tell there is."""
    m = re.match(r"[^A-Za-z]*([A-Za-z']+)", question)
    return m.group(1).lower() if m else ""


def source_share(candidate: str, source: str) -> float:
    """How much of a key is the source's own words, by characters."""
    spans = provenance(candidate, source)
    total = sum(sp["end"] - sp["start"] for sp in spans) or 1
    return sum(sp["end"] - sp["start"] for sp in spans if sp["origin"] == "source") / total


def key_problem(candidate: str, source: str, path: str) -> str | None:
    """What is wrong with a drafted key, if anything, that needs no model to see."""
    k = candidate.rstrip()
    if k.endswith(("...", "\u2026")) or k.endswith(("...'", '..."', "\u2026'", '\u2026"')):
        return "the key ends mid-sentence"
    if path.endswith(".txt") and source_share(candidate, source) > 0.8:
        return "the key is a raw quote from a transcript, not a drafted answer"
    return None


def question_words(question: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"[a-z0-9][a-z0-9_.-]*", question.lower()) if w not in _STOP)


def overlap(a: str, b: str) -> float:
    """Jaccard overlap of two questions' content words."""
    x, y = question_words(a), question_words(b)
    return len(x & y) / len(x | y) if x | y else 0.0


def spans_touch(a: str | None, b: str | None) -> bool:
    """Same episode, and the line ranges overlap or sit next to each other."""
    if not a or not b:
        return False
    from ..records import parse_episode_id
    p, q = parse_episode_id(a), parse_episode_id(b)
    return p.path == q.path and p.start <= q.end + 1 and q.start <= p.end + 1
SEEDED_SCHEMA = obj({"answerable": {"type": "boolean"}, "candidate_answer": {"type": "string"},
                     "episode_path": {"type": "string"}, "answer_start_line": {"type": "integer"},
                     "answer_end_line": {"type": "integer"},
                     "search_terms": {"type": "array", "items": {"type": "string"}}})


def stratum(path: str) -> str:
    m = _MONTH.search(path)
    return m.group(1) if m else "unknown"


def provenance(candidate: str, source: str, field: str = "candidate_answer", min_words: int = 4) -> list[dict]:
    """Which stretches of a drafted answer are the source's own words. A run of
    `min_words` or more words found verbatim in the source is source text; the
    rest the model wrote."""
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", candidate)]
    spans, i = [], 0
    while i < len(words):
        j = i + min_words
        if j <= len(words) and candidate[words[i][0]:words[j - 1][1]] in source:
            while j < len(words) and candidate[words[i][0]:words[j][1]] in source:
                j += 1
            spans.append({"field": field, "start": words[i][0], "end": words[j - 1][1], "origin": "source"})
            i = j
        else:
            if spans and spans[-1]["origin"] == "generated":
                spans[-1]["end"] = words[i][1]
            else:
                spans.append({"field": field, "start": words[i][0], "end": words[i][1], "origin": "generated"})
            i += 1
    return spans


def extend_over_continuation(lines: list[str], start: int, end: int) -> int:
    """A bullet's content often lives in the indented lines beneath it. The span runs on over them: every
    following line that is indented and not blank belongs to the same bullet. Returns the new end."""
    if not (1 <= end <= len(lines)) or not lines[start - 1].lstrip().startswith(("-", "*", "+")) \
            and not re.match(r"\s*\d+[.)]\s", lines[start - 1]):
        return end
    base = len(lines[start - 1]) - len(lines[start - 1].lstrip())
    while end < len(lines):
        nxt = lines[end]
        if not nxt.strip() or (len(nxt) - len(nxt.lstrip())) <= base:
            break
        end += 1
    return end


def block_end(lines: list[str], start: int, end: int) -> int:
    """A span never runs over a blank line or a heading: the answer is one place in the file, not a
    section. Returns `end` cut to the first blank or heading line after `start`, when there is one."""
    for n in range(start + 1, min(end, len(lines)) + 1):
        line = lines[n - 1]
        if not line.strip() or line.lstrip().startswith("#"):
            return n - 1
    return end


def key_outside_span(candidate: str, cited: str, shown: str, min_words: int = 6) -> str | None:
    """A run of the key's words found verbatim in what the model was shown but not in the lines it
    cites is a key drawing on lines outside its citation."""
    for sp in provenance(candidate, shown, min_words=min_words):
        if sp["origin"] == "source":
            run_ = candidate[sp["start"]:sp["end"]]
            if run_ not in cited and " ".join(run_.split()) not in " ".join(cited.split()):
                return f"the key uses '{run_[:60]}', which is outside the lines it cites"
    return None


def unmarked_windows(texts: dict[str, list[str]], spans: list[Span], window: int, count: int, seed: int
                     ) -> list[tuple[str, int, int]]:
    """Fixed windows that touch no marked span, sampled with a seed."""
    marked: dict[str, list[tuple[int, int]]] = {}
    for s in spans:
        marked.setdefault(s.path, []).append((s.start, s.end))
    candidates = []
    for path in sorted(texts):
        n = len(texts[path])
        for start in range(1, n + 1, window):
            end = min(start + window - 1, n)
            if end - start + 1 >= max(3, window // 3) and \
                    not any(a <= end and b >= start for a, b in marked.get(path, [])):
                candidates.append((path, start, end))
    random.Random(seed).shuffle(candidates)
    return sorted(candidates[:count])


class Drafter:
    def __init__(self, layout: StoreLayout, tenant: str, machinery: Machinery, provider, log,
                 *, batch: bool = True, wait=lambda: None, seed: int = 0):
        self.layout, self.tenant, self.m, self.provider = layout, tenant, machinery, provider
        self.log, self.batch, self.wait, self.seed = log, batch, wait, seed
        self.cfg = machinery.kernel.eval
        self.fingerprint = machinery.drafting_fingerprint()
        self.cold = layout.tier_dir(tenant, "cold")
        self._texts: dict[str, list[str]] = {}
        from ..eval.pools import load
        existing = load(layout, tenant, "drafts", "drafting") + load(layout, tenant, "held", "drafting")
        self._asked = {self._norm(d.question) for d in existing}
        self._known = [(d.question, d.episode) for d in existing]

    @staticmethod
    def _norm(question: str) -> str:
        return re.sub(r"\W+", " ", question).strip().lower()

    def lines(self, path: str) -> list[str]:
        if path not in self._texts:
            self._texts[path] = (self.cold / path).read_text(encoding="utf-8", errors="replace").splitlines()
        return self._texts[path]

    def _window(self, path: str, start: int, end: int) -> tuple[int, int, str]:
        ctx, n = self.cfg.context_lines, len(self.lines(path))
        a, b = max(1, start - ctx), min(n, end + ctx)
        return a, b, numbered(self.lines(path)[a - 1:b], a)

    def _reject(self, origin, span, prompt, reason, raw=None):
        self.log.count(f"rejected: {reason}")
        self.log.emit("draft_rejected", source_span=span, origin=origin, model_id=self._model(prompt),
                      prompt_version=prompt.prompt_version, reason=reason, raw=raw)

    def _model(self, prompt) -> str:
        return self.m.artifact.routing[prompt.tier]

    def _emit(self, item, *, origin, span, marker, prompt, candidate, superseded, terms, hits, source_text,
              form=None, held_for=None) -> bool:
        """False when the question is already asked: the same text, or much the same words about
        lines that touch. The suite asks each question once."""
        key = self._norm(item.question)
        if key in self._asked:
            self._reject(origin, span, prompt, "the same question is already in the pool")
            return False
        where = item.episode or span
        for question, episode in self._known:
            if spans_touch(where, episode) and overlap(item.question, question) >= self.cfg.dedupe_question_overlap:
                self._reject(origin, span, prompt, f"much the same question is already asked of these lines: {question!r}")
                return False
        self._asked.add(key)
        self._known.append((item.question, where))
        append_draft(self.layout, self.tenant, item, pool="held" if held_for else "drafts")
        if held_for:
            self.log.count("held")
        self.log.promoted += 1
        self.log.emit("item_drafted", item=item.id, item_type=item.tests.type, origin=origin,
                      source_span=span, marker=marker, model_id=self._model(prompt),
                      prompt_version=prompt.prompt_version, fingerprint=self.fingerprint,
                      question=item.question, candidate_answer=candidate, superseded_answer=superseded or None,
                      search_terms=terms, prescreen_hits=hits, form=form, held_for=held_for,
                      spans=provenance(candidate, source_text) if candidate != NOTHING else [])
        return not held_for

    # --- marked moments and unmarked windows -----------------------------------------
    def from_spans(self, targets: list[tuple[str, int, int, str | None]], origin: str) -> int:
        """targets: (path, start, end, marker-or-None)."""
        prompt, reqs, meta = self.cfg.drafting, [], {}
        for i, (path, start, end, marker) in enumerate(targets):
            end = extend_over_continuation(self.lines(path), start, end)
            a, b, shown = self._window(path, start, end)
            head = (f"Marker: {marker}" if marker else
                    "No marker fired on this window. Skip unless something here would change what an agent does later.")
            user = f"Episode: {path}\n{head}\nThe moment is lines {start}-{end}. Lines shown: {a}-{b}.\n\n{shown}"
            cid = f"{origin[0]}{i}"
            meta[cid] = (path, start, end, marker, a, b)
            reqs.append(request(f"draft_{origin}", tier=prompt.tier, system=prompt.prompt, user=user,
                                schema=DRAFT_SCHEMA, max_tokens=MAX_TOKENS, custom_id=cid))
        made = 0

        def width(pair):  # the wider answer first: of two drafts asking the same thing, it holds more of the answer
            d, _ = parse_json(pair[1])
            try:
                return -(int(d["answer_end_line"]) - int(d["answer_start_line"]))
            except (TypeError, KeyError, ValueError):
                return 0

        results = sorted(run_all(self.provider, reqs, batch=self.batch, wait=self.wait), key=width)
        for cid, resp in results:
            path, start, end, marker, a, b = meta[cid]
            span = f"{path}#L{start}-L{end}"
            self.log.examined += 1
            d, err = parse_json(resp)
            if err or not isinstance(d, dict):
                self._reject(origin, span, prompt, err or "output is not an object")
                continue
            if d.get("skip"):
                self._reject(origin, span, prompt, "the drafter found nothing to ask")
                continue
            try:
                s, e = int(d["answer_start_line"]), int(d["answer_end_line"])
                if 1 <= s <= e <= len(self.lines(path)):
                    e = min(b, extend_over_continuation(self.lines(path), s, e))
                    e = block_end(self.lines(path), s, e)
                if not (a <= s <= e <= b):
                    raise ItemError(f"answer lines {s}-{e} fall outside the lines shown, {a}-{b}")
                outside = key_outside_span(str(d["candidate_answer"]), "\n".join(self.lines(path)[s - 1:e]),
                                           "\n".join(self.lines(path)[a - 1:b]))
                if outside:
                    raise ItemError(outside)
                kind = "supersession" if d.get("item_type") == "supersession" and marker == "reversal" else "yield"
                source = "\n".join(self.lines(path)[a - 1:b])
                problem = key_problem(str(d["candidate_answer"]), source, path)
                if problem:
                    raise ItemError(problem)
                d["candidate_answer"], n1 = redact_secrets(str(d["candidate_answer"]))
                d["question"], n2 = redact_secrets(str(d["question"]))
                if n1 + n2:
                    self.log.count("secrets_redacted", n1 + n2)
                item = validate_item({"id": str(ULID()), "question": d["question"], "answer": d["candidate_answer"],
                                      "episode": f"{path}#L{s}-L{e}", "exposure": 0,
                                      "tests": {"type": kind, "marker": marker, "hard": False, "stratum": stratum(path)}})
            except (ItemError, KeyError, TypeError, ValueError) as exc:
                self._reject(origin, span, prompt, f"draft failed its checks: {exc}", raw=d)
                continue
            terms = [t for t in d.get("search_terms", []) if isinstance(t, str)][:6]
            made += self._emit(item, origin=origin, span=span, marker=marker, prompt=prompt, candidate=item.answer,
                               superseded=d.get("superseded_answer") if kind == "supersession" else None, terms=terms,
                               hits=None, source_text="\n".join(self.lines(path)[a - 1:b]), form=d.get("form") or None)
        return made

    def marked(self, spans: list[Span]) -> int:
        return self.from_spans([(s.path, s.start, s.end, s.marker) for s in spans], "marked")

    def unmarked(self, spans: list[Span], paths: list[str]) -> int:
        u = self.cfg.unmarked
        if u.share <= 0 or not spans:
            return 0
        count = math.ceil(len(spans) * u.share / (1 - u.share) * u.oversample)
        windows = unmarked_windows({p: self.lines(p) for p in paths}, spans, u.window_lines, count, self.seed)
        return self.from_spans([(p, a, b, None) for p, a, b in windows], "unmarked")

    # --- noise ------------------------------------------------------------------------
    def noise(self, spans: list[Span]) -> int:
        prompt = self.cfg.noise
        count = math.ceil(len(spans) * self.cfg.noise_draft_ratio)
        chosen = sorted(random.Random(self.seed + 1).sample(spans, min(count, len(spans))),
                        key=lambda s: (s.path, s.start))
        # A noise question must not be recognisable by how it opens. Each draft is told which word to
        # open with, drawn with a seed from the way the answerable questions in the pool open.
        from ..eval.pools import load
        openers = [opener(i.question) for i in load(self.layout, self.tenant, "drafts", "drafting") if i.tests.type != "noise"]
        openers = [o for o in openers if o] or ["what"]
        rng = random.Random(self.seed + 2)
        reqs, meta, want = [], {}, {}
        for i, s in enumerate(chosen):
            a, b, shown = self._window(s.path, s.start, s.end)
            meta[f"x{i}"] = s
            want[f"x{i}"] = rng.choice(openers)
            reqs.append(request("draft_noise", tier=prompt.tier, system=prompt.prompt, schema=NOISE_SCHEMA,
                                user=f"Episode: {s.path}\nThe marked moment is lines {s.start}-{s.end}. "
                                     f"Neighbourhood shown: lines {a}-{b}.\n"
                                     f"Your question must open with the word: {want[f'x{i}'].capitalize()}\n\n{shown}",
                                max_tokens=MAX_TOKENS, custom_id=f"x{i}"))
        made = 0
        for cid, resp in run_all(self.provider, reqs, batch=self.batch, wait=self.wait):
            s = meta[cid]
            self.log.examined += 1
            d, err = parse_json(resp)
            if err or not isinstance(d, dict) or d.get("skip") or not str(d.get("question", "")).strip():
                self._reject("noise", s.episode, prompt, err or "no noise question offered")
                continue
            terms = [t for t in d.get("search_terms", []) if isinstance(t, str) and t.strip()][:6]
            if not terms:
                self._reject("noise", s.episode, prompt, "a noise draft names the terms its absence is checked by", raw=d)
                continue
            if opener(str(d["question"])) != want[cid]:
                self._reject("noise", s.episode, prompt, f"asked to open with {want[cid]!r} and did not", raw=d)
                continue
            d["question"], _ = redact_secrets(str(d["question"]))
            hits = grep(self.layout, self.tenant, terms, limit=0, min_match=PRESCREEN_MIN_MATCH)["total"]
            item = validate_item({"id": str(ULID()), "question": d["question"], "answer": NOTHING, "episode": None,
                                  "exposure": 0, "tests": {"type": "noise", "marker": None, "hard": False,
                                                           "stratum": stratum(s.path)}})
            # Absence is certified by search before a noise item enters the pool. Any hit holds it for a person.
            held = f"{hits} lines in the corpus carry two or more of its search terms" if hits else None
            made += self._emit(item, origin="noise", span=s.episode, marker=None, prompt=prompt, candidate=NOTHING,
                               superseded=None, terms=terms, hits=hits, source_text="", form=d.get("form") or None,
                               held_for=held)
        return made

    # --- items a person wrote -------------------------------------------------------------------
    def authored(self, payload: dict, *, author: str, basis: str, cites: list[str], superseded: str | None = None,
                 form: str | None = None) -> bool:
        """An item written by a person, key and all, on the strength of lines they cite. Logged as seeded,
        with who wrote it where a model id would be."""
        item = validate_item({"id": str(ULID()), "exposure": 0, **payload})
        for ref in [item.episode, *cites]:
            if ref:
                from ..records import parse_episode_id
                r = parse_episode_id(ref)
                if r.end > len(self.lines(r.path)):
                    raise ItemError(f"{ref} cites lines the episode does not have")

        class _By:  # stands where an eval prompt would: who wrote it, and on what authority
            tier, prompt_version = None, basis
        self._model = lambda _p, _a=author: _a
        try:
            return self._emit(item, origin="seeded", span="; ".join(cites) if cites else item.episode, marker=item.tests.marker,
                              prompt=_By, candidate=item.answer, superseded=superseded, terms=[], hits=None,
                              source_text="", form=form)
        finally:
            del self._model

    # --- questions a person supplied ---------------------------------------------------
    def seeded(self, questions: list[dict]) -> int:
        """Each: {"question": str, "paths": [episode paths to read]}. All are hard-case candidates."""
        prompt, reqs, meta = self.cfg.seeded, [], {}
        for i, q in enumerate(questions):
            body = "\n\n".join(f"=== {p}\n{numbered(self.lines(p))}" for p in q["paths"])
            meta[f"s{i}"] = q
            reqs.append(request("draft_seeded", tier=prompt.tier, system=prompt.prompt, schema=SEEDED_SCHEMA,
                                user=f"Question: {q['question']}\n\n{body}", max_tokens=MAX_TOKENS, custom_id=f"s{i}"))
        made = 0
        for cid, resp in run_all(self.provider, reqs, batch=self.batch, wait=self.wait):
            q = meta[cid]
            self.log.examined += 1
            d, err = parse_json(resp)
            if err or not isinstance(d, dict):
                self._reject("seeded", None, prompt, err or "output is not an object", raw={"question": q["question"]})
                continue
            terms = [t for t in d.get("search_terms", []) if isinstance(t, str) and t.strip()][:6]
            try:
                if d.get("answerable"):
                    path, s, e = d["episode_path"], int(d["answer_start_line"]), int(d["answer_end_line"])
                    if path not in q["paths"] or not (1 <= s <= e <= len(self.lines(path))):
                        raise ItemError("the cited lines are not in the episodes that were read")
                    e = block_end(self.lines(path), s, e)
                    shown_all = "\n".join("\n".join(self.lines(p)) for p in q["paths"])
                    outside = key_outside_span(str(d["candidate_answer"]), "\n".join(self.lines(path)[s - 1:e]), shown_all)
                    if outside:
                        raise ItemError(outside)
                    item = validate_item({"id": str(ULID()), "question": q["question"], "answer": d["candidate_answer"],
                                          "episode": f"{path}#L{s}-L{e}", "exposure": 0,
                                          "tests": {"type": "yield", "marker": None, "hard": True, "stratum": stratum(path)}})
                    span, hits, src = item.episode, None, "\n".join(self.lines(path)[s - 1:e])
                else:
                    item = validate_item({"id": str(ULID()), "question": q["question"], "answer": NOTHING,
                                          "episode": None, "exposure": 0,
                                          "tests": {"type": "noise", "marker": None, "hard": True,
                                                    "stratum": stratum(q["paths"][0])}})
                    span, hits, src = None, grep(self.layout, self.tenant, terms, limit=0)["total"], ""
            except (ItemError, KeyError, TypeError, ValueError) as exc:
                self._reject("seeded", None, prompt, f"draft failed its checks: {exc}", raw=d)
                continue
            made += self._emit(item, origin="seeded", span=span, marker=None, prompt=prompt, candidate=item.answer,
                               superseded=None, terms=terms, hits=hits, source_text=src)
        return made
