"""What a reviewer is shown. Blind-first is decided here, on the serving side:
an item in the blind sample is served without its key, its type, its model
review or its search results until the person's own answer is on record. A page
that hides a field it has already received is not blind."""
from __future__ import annotations

from typing import Any, Optional

from ..machinery import Machinery
from ..records import parse_episode_id
from ..redact import redact_secrets
from ..store.layout import StoreLayout
from .ledger import ItemState, Ledger
from .pools import find_founded, load
from .search import grep


def span_view(layout: StoreLayout, tenant: str, span: str, context: int) -> dict[str, Any]:
    ref = parse_episode_id(span)
    path = layout.tier_dir(tenant, "cold") / ref.path
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines() if path.is_file() else []
    a, b = max(1, ref.start - context), min(len(lines), ref.end + context)
    return {"path": ref.path, "start": ref.start, "end": ref.end, "first": a, "last": b,
            "total_lines": len(lines),
            "lines": [{"n": n, "text": redact_secrets(lines[n - 1])[0]} for n in range(a, b + 1)]}


def _status(st: ItemState) -> str:
    human, model = st.latest("human"), st.latest("model")
    if human:
        return f"human_{human['outcome']}"
    return f"model_{model['outcome']}" if model else "unreviewed"


def _blind_hidden(ledger: Ledger, st: ItemState, item_id: str, reveal: bool = False) -> bool:
    """Hidden until the person's answer is on record, and then until they ask to see the key."""
    if not ledger.in_blind_sample(item_id) or st.latest("human") or st.revealed_at:
        return False
    return not (st.blind_answer and reveal)


def queue(layout: StoreLayout, tenant: str, *, pool: str = "drafts", item_type: Optional[str] = None,
          stratum: Optional[str] = None, reviewed_by: Optional[str] = None, blind_only: bool = False
          ) -> list[dict[str, Any]]:
    ledger = Ledger(layout, tenant)
    out = []
    for it in load(layout, tenant, pool, "human_review"):
        st = ledger.state(it.id)
        in_blind = ledger.in_blind_sample(it.id)
        hidden = _blind_hidden(ledger, st, it.id)
        by = "human" if st.latest("human") else "model" if st.latest("model") else "none"
        if blind_only and not in_blind:
            continue
        if stratum and it.tests.stratum != stratum:
            continue
        if reviewed_by and by != reviewed_by:
            continue
        if item_type and (hidden or it.tests.type != item_type):
            continue
        out.append({"id": it.id, "question": it.question, "stratum": it.tests.stratum, "blind": in_blind,
                    "type": None if hidden else it.tests.type, "hard": None if hidden else it.tests.hard,
                    "status": "awaiting_blind_answer" if hidden else _status(st), "reviewed_by": by})
    # Undecided first: what still needs a blind answer, then what is answered but not decided, then the
    # rest a person has not seen, and what a person has decided last.
    def rank(r):
        if r["status"] == "awaiting_blind_answer":
            return 0
        if r["blind"] and r["reviewed_by"] != "human":
            return 1
        return 2 if r["reviewed_by"] != "human" else 3
    return sorted(out, key=lambda r: (rank(r), r["id"]))


def item_view(layout: StoreLayout, tenant: str, machinery: Machinery, log, item_id: str,
              reviewer: dict, reveal: bool = False) -> dict[str, Any]:
    pool = find_founded(layout, tenant, item_id) or "drafts"
    item = next((i for i in load(layout, tenant, pool, "human_review") if i.id == item_id), None)
    if item is None:
        raise KeyError(item_id)
    ledger = Ledger(layout, tenant)
    st = ledger.state(item_id)
    drafted = st.drafted or {}
    in_blind = ledger.in_blind_sample(item_id)
    hidden = _blind_hidden(ledger, st, item_id, reveal and reviewer.get("kind") == "human")
    # A noise item has no episode. While it is hidden it is shown with no excerpt at all, rather than a
    # decoy the reviewer reads in vain; once revealed, the neighbourhood it was drafted near is shown.
    source = item.episode or (None if hidden else drafted.get("source_span"))
    view: dict[str, Any] = {
        "id": item.id, "pool": pool, "question": item.question, "stratum": item.tests.stratum,
        "span": span_view(layout, tenant, source, machinery.kernel.eval.context_lines) if source else None,
        "blind": {"in_sample": in_blind, "answered": bool(st.blind_answer), "hidden": hidden,
                  "answer_text": st.blind_answer["text"] if st.blind_answer else None,
                  "choice": st.blind_answer.get("choice") if st.blind_answer else None,
                  "revealed_at": st.revealed_at},
        "reasons": machinery.artifact.review_reasons,
    }
    if hidden:
        return view
    if in_blind and st.blind_answer and not st.revealed_at and reviewer.get("kind") == "human":
        rec = log.emit("blind_revealed", item=item_id, reviewer=reviewer)
        view["blind"]["revealed_at"] = rec["ts"]
    terms = drafted.get("search_terms", [])
    view.update({
        "answer": item.answer, "type": item.tests.type, "marker": item.tests.marker, "hard": item.tests.hard,
        "episode": item.episode, "key_hash": item.key_hash(), "status": _status(st),
        "provenance": {k: drafted.get(k) for k in ("origin", "marker", "model_id", "prompt_version",
                                                    "source_span", "candidate_answer", "superseded_answer", "spans")},
        "search": {"terms": terms, **grep(layout, tenant, terms, limit=20)} if terms else None,
        "decisions": [{k: d.get(k) for k in ("ts", "outcome", "reason", "note", "reviewer", "fixed_answer", "hard")}
                      for d in st.decisions],
    })
    return view
