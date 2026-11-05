"""A literal search of the cold tier. Mechanical: it is the pre-screen for a
noise item, the reviewer's search, and the search box a person uses to certify
that something is absent."""
from __future__ import annotations

import re

from ..redact import redact_secrets
from ..store.episodes import read_manifest
from ..store.layout import StoreLayout


_MARKUP = re.compile(r"[*`]+")


def plain(text: str) -> str:
    """Lower-cased, with markdown emphasis and code ticks removed and whitespace collapsed. A term taken
    from rendered text ("<a href> tags") must still find the source line that wrote it as **`<a href>` tags**."""
    return " ".join(_MARKUP.sub("", text).lower().split())


def grep(layout: StoreLayout, tenant: str, terms: list[str], *, limit: int = 50,
         min_match: int | None = None) -> dict:
    """Lines containing the terms, case-insensitively. By default every term must be on the
    line. `min_match` lowers that to "at least this many of them", which is what a pre-screen
    wants: it is looking for a reason to doubt an absence, so it should err towards finding one."""
    needles = [plain(t) for t in terms if isinstance(t, str) and plain(t)]
    if not needles:
        return {"total": 0, "hits": []}
    need = len(needles) if min_match is None else max(1, min(min_match, len(needles)))
    cold, total, hits = layout.tier_dir(tenant, "cold"), 0, []
    for entry in read_manifest(layout, tenant):
        text = (cold / entry["path"]).read_text(encoding="utf-8", errors="replace")
        for n, line in enumerate(text.splitlines(), 1):
            low = plain(line)
            if sum(t in low for t in needles) >= need:
                total += 1
                if len(hits) < limit:
                    hits.append({"episode": f"{entry['path']}#L{n}-L{n}", "line": redact_secrets(line[:400])[0]})
    return {"total": total, "hits": hits}
