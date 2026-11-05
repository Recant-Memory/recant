"""The model reviewer: a second reader over the whole drafting pool. Its accepts
are drafting, not ground truth; the record says a model made them."""
from __future__ import annotations

from ..eval.ledger import Ledger
from ..eval.pools import load
from ..eval.review import ReviewError, decide
from ..eval.search import grep
from ..machinery import Machinery
from ..records import parse_episode_id
from ..store.layout import StoreLayout
from .common import numbered, obj, parse_json, request, run_all


def _schema(reasons: list[str]) -> dict:
    return obj({"outcome": {"type": "string", "enum": ["accept", "fix", "discard"]},
                "reason": {"type": "string", "enum": reasons}, "fixed_answer": {"type": "string"},
                "hard": {"type": "boolean"}, "note": {"type": "string"}})


def review_requests(layout: StoreLayout, tenant: str, machinery: Machinery, *, only_unreviewed: bool = True):
    """The requests a review of the drafting pool would send, and which item each is for."""
    cfg, reasons = machinery.kernel.eval, machinery.artifact.review_reasons
    prompt = cfg.review
    ledger, cold = Ledger(layout, tenant), layout.tier_dir(tenant, "cold")
    reqs, ids = [], {}
    for n, item in enumerate(load(layout, tenant, "drafts", "model_review")):
        st = ledger.state(item.id)
        if only_unreviewed and st.decisions:
            continue
        drafted = st.drafted or {}
        span = item.episode or drafted.get("source_span")
        shown = "(no source span)"
        if span:
            ref = parse_episode_id(span)
            lines = (cold / ref.path).read_text(encoding="utf-8", errors="replace").splitlines()
            a, b = max(1, ref.start - cfg.context_lines), min(len(lines), ref.end + cfg.context_lines)
            shown = f"{ref.path}, lines {a}-{b} (the drafted span is {ref.start}-{ref.end}):\n{numbered(lines[a - 1:b], a)}"
        found = grep(layout, tenant, drafted.get("search_terms", []), limit=15)
        search = "\n".join(f"{h['episode']}: {h['line']}" for h in found["hits"]) or "(no lines contain every term)"
        user = (f"Item type: {item.tests.type}\nQuestion: {item.question}\nCandidate answer: {item.answer}\n\n"
                f"Source:\n{shown}\n\nA literal search of the whole corpus for {drafted.get('search_terms', [])} "
                f"found {found['total']} lines:\n{search}")
        ids[f"v{n}"] = item.id
        reqs.append(request("model_review", tier=prompt.tier, system=prompt.prompt, user=user,
                            schema=_schema(reasons), max_tokens=4000, custom_id=f"v{n}"))
    return reqs, ids


def model_review(layout: StoreLayout, tenant: str, machinery: Machinery, provider, log, *,
                 batch: bool = True, wait=lambda: None, only_unreviewed: bool = True) -> dict[str, int]:
    cfg, reasons = machinery.kernel.eval, machinery.artifact.review_reasons
    prompt, model = cfg.review, machinery.artifact.routing[cfg.review.tier]
    reviewer = {"kind": "model", "id": model}
    reqs, ids = review_requests(layout, tenant, machinery, only_unreviewed=only_unreviewed)
    tally: dict[str, int] = {}
    for cid, resp in run_all(provider, reqs, batch=batch, wait=wait):
        log.examined += 1
        d, err = parse_json(resp)
        try:
            if err or not isinstance(d, dict):
                raise ReviewError(err or "output is not an object")
            decide(layout, tenant, log, ids[cid], outcome=d["outcome"], reason=d["reason"], reviewer=reviewer,
                   reasons=reasons, note=d.get("note") or None, fixed_answer=d.get("fixed_answer") or None,
                   hard=True if d.get("hard") else None)  # a model adds the tag; it never removes one
            tally[d["outcome"]] = tally.get(d["outcome"], 0) + 1
        except (ReviewError, KeyError) as exc:
            log.count("review_failed")
            tally["failed"] = tally.get("failed", 0) + 1
            log.emit("draft_rejected", source_span=None, origin="model_review", model_id=model,
                     prompt_version=prompt.prompt_version, reason=f"review of {ids[cid]} unusable: {exc}", raw=d)
    for k, v in tally.items():
        log.count(f"review: {k}", v)
    return tally
