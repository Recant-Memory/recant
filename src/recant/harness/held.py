"""Clearing held noise drafts by model. A noise draft is held when a search found lines that might
answer it. A second reader sees the question and every such line and rules: answered, or not. Not
answered, it enters the pool as noise certified by a model, and the record says a model certified it.
Answered, it is dropped, citing the line. What the model cites is checked: it must be one of the
lines it was shown."""
from __future__ import annotations

from ..eval.ledger import Ledger
from ..eval.pools import load, release_held
from ..eval.review import decide
from ..eval.search import grep
from ..machinery import Machinery
from ..store.layout import StoreLayout
from .common import obj, parse_json, request, run_all
from .draft import PRESCREEN_MIN_MATCH

SCHEMA = obj({"answered": {"type": "boolean"}, "answering_line": {"type": "string"}, "note": {"type": "string"}})
MAX_HITS_SHOWN = 300


def clear_held(layout: StoreLayout, tenant: str, machinery: Machinery, provider, log, *,
               batch: bool = True, wait=lambda: None) -> dict[str, int]:
    cfg, prompt = machinery.kernel.eval, machinery.kernel.eval.held_review
    model = machinery.artifact.routing[prompt.tier]
    reviewer = {"kind": "model", "id": model}
    ledger = Ledger(layout, tenant)
    reqs, shown = [], {}
    for n, item in enumerate(load(layout, tenant, "held", "model_review")):
        terms = (ledger.state(item.id).drafted or {}).get("search_terms", [])
        found = grep(layout, tenant, terms, limit=MAX_HITS_SHOWN, min_match=PRESCREEN_MIN_MATCH)
        lines = "\n".join(f"[{h['episode']}] {h['line']}" for h in found["hits"])
        more = f"\n({found['total'] - len(found['hits'])} further lines matched and are not shown.)" if found["total"] > len(found["hits"]) else ""
        shown[f"h{n}"] = (item.id, {h["episode"] for h in found["hits"]}, found["total"])
        reqs.append(request("clear_held", tier=prompt.tier, system=prompt.prompt, schema=SCHEMA, max_tokens=4000,
                            custom_id=f"h{n}", user=f"Question: {item.question}\n\nSearch terms: {terms}\n"
                            f"{found['total']} lines in the record carry two or more of them:\n{lines}{more}"))
    tally = {"released": 0, "answered": 0, "left_held": 0}
    for cid, resp in run_all(provider, reqs, batch=batch, wait=wait):
        item_id, hits, total = shown[cid]
        log.examined += 1
        d, err = parse_json(resp)
        if err or not isinstance(d, dict) or not isinstance(d.get("answered"), bool):
            tally["left_held"] += 1
            log.count("unusable_ruling")
            continue
        if d["answered"]:
            cited = str(d.get("answering_line", "")).strip().strip("[]")
            if cited not in hits:            # it must point at a line it was shown, or the ruling is not usable
                tally["left_held"] += 1
                log.count("answered_without_a_valid_citation")
                continue
            release_held(layout, tenant, item_id, into_pool=False)
            log.emit("held_cleared", item=item_id, reviewer=reviewer, released=False, note=(d.get("note") or None),
                     answered_by=cited, hits_shown=len(hits))
            tally["answered"] += 1
        else:
            release_held(layout, tenant, item_id, into_pool=True)
            log.emit("held_cleared", item=item_id, reviewer=reviewer, released=True, note=(d.get("note") or None),
                     hits_shown=len(hits))
            decide(layout, tenant, log, item_id, outcome="accept", reason="confirmed", reviewer=reviewer,
                   reasons=machinery.artifact.review_reasons,
                   note=f"noise certified by model: none of {len(hits)} of {total} matching lines answers it")
            tally["released"] += 1
            log.promoted += 1
    for k, v in tally.items():
        log.count(k, v)
    return tally
