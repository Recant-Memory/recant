"""The nomination pass: a cheap model reads each episode and says where the
marked moments are. What it returns is then checked without a model: the lines
must exist, the marker must be one of ours, and the quote must be there verbatim."""
from __future__ import annotations

import re

from dataclasses import dataclass

from ..machinery import Machinery
from ..store.episodes import read_manifest
from ..store.layout import StoreLayout
from ..provider import ProviderError
from .common import numbered, obj, parse_json, request, run_all


@dataclass(frozen=True)
class Span:
    marker: str
    episode: str      # <path>#L<start>-L<end>
    path: str
    start: int
    end: int
    quote: str


_PLAIN = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"})


def _schema(markers: list[str]) -> dict:
    return obj({"spans": {"type": "array", "items": obj({
        "marker": {"type": "string", "enum": markers}, "start_line": {"type": "integer"},
        "end_line": {"type": "integer"}, "quote": {"type": "string"}})}})


def _system(machinery: Machinery) -> str:
    a = machinery.artifact
    listing = "\n".join(f"- {m.name}: {m.description}" for m in a.markers)
    return a.nomination.prompt.replace("{markers}", listing)


def _repair(quote: str, cited: str, squash) -> str:
    """Memory v2: a quote whose \\uXXXX escapes are decoded, and, when it fails only on trailing characters (a
    model's quote of the whole line ending in "...team.of"), its longest verbatim prefix if that keeps at least 90% of
    it and 20 characters. Anything else is returned as it was, to fail the check."""
    q = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), quote)
    if squash(q) in cited:
        return q
    for n in range(len(q) - 1, max(19, int(len(q) * 0.9)) - 1, -1):
        if squash(q[:n]) and squash(q[:n]) in cited:
            return q[:n]
    return quote


def check_span(raw: dict, path: str, lines: list[str], markers: list[str], *, repair: bool = False) -> tuple[Span | None, str | None]:
    try:
        marker, start, end, quote = raw["marker"], int(raw["start_line"]), int(raw["end_line"]), raw["quote"]
    except (KeyError, TypeError, ValueError):
        return None, "malformed span"
    if marker not in markers:
        return None, "unknown marker"
    if not (1 <= start <= end <= len(lines)):
        return None, "lines out of range"
    # Verbatim in its words. A model returns a line break as a space and a typographic apostrophe,
    # quote or dash as the plain one, so those are folded on both sides. Nothing else is forgiven.
    squash = lambda t: " ".join(t.translate(_PLAIN).split())
    if repair and isinstance(quote, str):
        quote = _repair(quote, squash("\n".join(lines[start - 1:end])), squash)
    if not isinstance(quote, str) or not quote.strip() or squash(quote) not in squash("\n".join(lines[start - 1:end])):
        return None, "quote is not verbatim in the cited lines"
    return Span(marker, f"{path}#L{start}-L{end}", path, start, end, quote), None


def dedupe(spans: list[Span]) -> list[Span]:
    """Byte-identical quotes under the same marker are one moment, re-pasted."""
    seen, out = set(), []
    for s in sorted(spans, key=lambda s: (s.path, s.start, s.end, s.marker)):
        key = (s.marker, s.quote)
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def chunks(lines: list[str], size: int, overlap: int) -> list[tuple[int, int]]:
    """(start, end) line ranges, 1-based and inclusive, of at most `size` characters each (a longer single line is
    its own chunk). Consecutive chunks share `overlap` lines so a moment on a boundary is seen whole at least once;
    a moment nominated twice is the same quote under the same marker, which dedupe() folds."""
    n = len(lines)
    if n == 0:
        return [(1, 1)]
    out, s = [], 1
    while True:
        e, used = s, len(lines[s - 1]) + 1
        while e < n and used + len(lines[e]) + 1 <= size:
            used += len(lines[e]) + 1
            e += 1
        out.append((s, e))
        if e == n:
            return out
        s = max(e + 1 - overlap, s + 1)


def _halves(s: int, e: int, overlap: int) -> list[tuple[int, int]]:
    mid, overlap = (s + e) // 2, min(overlap, (e - s) // 4)
    return [(s, mid), (mid + 1 - overlap, e)]


def nominate(layout: StoreLayout, tenant: str, machinery: Machinery, provider, log, *,
             paths: list[str] | None = None, batch: bool = True, wait=lambda: None,
             max_tokens: int = 8000, model: str | None = None,
             chunk_chars: int = 12000, overlap_lines: int = 5, max_splits: int = 3,
             line_range: tuple[int, int] | None = None) -> list[Span]:
    """`model` overrides the artifact's nomination model: the live path passes the kernel tier table's extraction
    model. The harness's own passes leave it None and use the artifact's.

    A long episode is read in chunks of `chunk_chars`, numbered with its own line numbers. A chunk whose answer
    stops at `max_tokens` is split in half and asked again, at most `max_splits` times; then it fails, and the
    nomination record says why."""
    a = machinery.artifact
    model = model or a.nomination.model
    max_tokens = a.nomination.max_tokens or max_tokens          # the machinery's cap wins (a guard, not a saving)
    if a.nomination.prompt is None:
        raise ValueError("the artifact has no nomination prompt")
    markers, system = a.marker_names(), _system(machinery)
    paths = paths if paths is not None else [e["path"] for e in read_manifest(layout, tenant)]
    texts = {p: layout.episode_file(tenant, p).read_text(encoding="utf-8", errors="replace").splitlines() for p in paths}
    log.examined += len(paths)
    def ranges(p):                    # only the lines that left the carried window, when a range is given
        if line_range is None:
            return chunks(texts[p], chunk_chars, overlap_lines)
        a, b = max(1, line_range[0]), min(len(texts[p]), line_range[1])
        return [] if b < a else [(s + a - 1, e + a - 1) for s, e in chunks(texts[p][a - 1:b], chunk_chars, overlap_lines)]
    pending = [(p, s, e, 0) for p in paths for s, e in ranges(p)]
    found: list[Span] = []
    sent = 0
    while pending:
        reqs, by_id = [], {}
        for p, s, e, depth in pending:
            cid, sent = f"n{sent}", sent + 1
            lines, total = texts[p], max(len(texts[p]), 1)
            part = "" if (s, e) == (1, total) else f"Lines {s}-{e} of {total}.\n\n"
            reqs.append(request("nominate", model=model, system=system, schema=_schema(markers),
                                user=f"Episode: {p}\n\n{part}{numbered(lines[s - 1:e], s)}",
                                max_tokens=max_tokens, custom_id=cid, reasoning=a.nomination.reasoning))
            by_id[cid] = (p, s, e, depth)
        log.count("chunks", len(reqs))
        pending = []
        for cid, resp in run_all(provider, reqs, batch=batch, wait=wait):
            path, s, e, depth = by_id[cid]
            episode = f"{path}#L{s}-L{e}"
            if not isinstance(resp, ProviderError) and resp.stop_reason == "max_tokens" \
                    and e > s and depth < max_splits:
                log.count("chunks_split")
                pending.extend((path, hs, he, depth + 1) for hs, he in _halves(s, e, overlap_lines))
                continue
            data, err = parse_json(resp)
            if err is None and not (isinstance(data, dict) and isinstance(data.get("spans"), list)):
                err = "output has no spans list"
            if err:
                log.count("episodes_failed")
                log.emit("nomination", episode=episode, markers=[], model_id=model,
                         prompt_version=a.nomination.prompt_version, error=err)
                continue
            kept, dropped = [], []
            for raw in data["spans"]:
                span, why = check_span(raw, path, texts[path], markers, repair=bool(getattr(machinery.kernel.extraction, "evidence_rule", False)))
                if span is not None and not (s <= span.start and span.end <= e):
                    span, why = None, "lines outside the chunk"
                if span is None:
                    log.count(f"dropped: {why}")
                    dropped.append({"reason": why, "returned": raw})
                else:
                    kept.append(span)
            found.extend(kept)
            log.emit("nomination", episode=episode,
                     markers=[{"marker": s.marker, "span": s.episode, "quote": s.quote} for s in kept],
                     model_id=model, prompt_version=a.nomination.prompt_version, dropped=dropped)
    unique = dedupe(found)
    log.count("spans_found", len(found))
    log.count("spans_after_dedupe", len(unique))
    return unique


def spans_from_log(records) -> list[Span]:
    """Rebuild the pass's output from its log, so a later job need not pay for it again."""
    from ..records import parse_episode_id
    out = []
    for r in records:
        if r["kind"] == "nomination":
            for m in r["markers"]:
                ref = parse_episode_id(m["span"])
                out.append(Span(m["marker"], m["span"], ref.path, ref.start, ref.end, m["quote"]))
    return dedupe(out)
