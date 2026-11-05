"""One question per fact per day. Two passes over the working pools.

First arithmetic: drafts from the same day with the same answer span, or whose questions or keys
share most of their words, are one item. Then, because the same fact is often written in different
words in different places (a daily and its transcript, a decision and its recap), a model reads each
day's drafts together and groups the ones that ask for the same fact. What it returns is checked:
ids must be that day's, and a group's survivor must be one of its members."""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from ..eval.items import EvalItem, validate_item
from ..eval.ledger import Ledger
from ..eval.pools import load, move_between, save_draft
from ..machinery import Machinery
from ..store.layout import StoreLayout
from .common import obj, parse_json, request, run_all
from .draft import overlap

_DAY = re.compile(r"(\d{4}-\d{2}-\d{2})")
SCHEMA = obj({"groups": {"type": "array", "items": obj({
    "keep": {"type": "string"}, "drop": {"type": "array", "items": {"type": "string"}}, "reason": {"type": "string"}})}})


def day_of(item: EvalItem, drafted: dict | None) -> str:
    m = _DAY.search(item.episode or (drafted or {}).get("source_span") or "")
    return m.group(1) if m else "unknown"


class _Pool:
    """The drafts and held pools seen as one, with the bookkeeping a merge needs."""

    def __init__(self, layout, tenant, log):
        self.layout, self.tenant, self.log = layout, tenant, log
        self.ledger = Ledger(layout, tenant)
        self.where = {i.id: "drafts" for i in load(layout, tenant, "drafts", "drafting")}
        self.where.update({i.id: "held" for i in load(layout, tenant, "held", "drafting")})
        self.items = {i.id: i for p in ("drafts", "held") for i in load(layout, tenant, p, "drafting")}
        self.merged = 0

    def drafted(self, item_id):
        return self.ledger.state(item_id).drafted or {}

    def marker_counts(self) -> Counter:
        return Counter(i.tests.marker for i in self.items.values() if i.tests.marker)

    def merge(self, keep: str, drop: list[str], reason: str, by: str, version: str | None = None) -> None:
        group = [self.items[keep], *[self.items[d] for d in drop]]
        markers = sorted({g.tests.marker for g in group if g.tests.marker})
        survivor = self.items[keep]
        if survivor.tests.type != "noise" and markers:
            # A moment may carry two markers; an item carries one. It keeps the rarer, so a rare
            # stratum is not emptied by a merge, and the log keeps them all.
            counts = self.marker_counts()
            rarest = min(markers, key=lambda m: (counts[m], m))
            if rarest != survivor.tests.marker:
                survivor = validate_item({**survivor.to_dict(), "tests": {**survivor.to_dict()["tests"], "marker": rarest}})
                save_draft(self.layout, self.tenant, survivor, pool=self.where[keep])
                self.items[keep] = survivor
        # A noise draft whose twin was held is held too: one search found something the other's words missed.
        if survivor.tests.type == "noise" and self.where[keep] == "drafts" and any(self.where[d] == "held" for d in drop):
            move_between(self.layout, self.tenant, [keep], "drafts", "held")
            self.where[keep] = "held"
        for d in drop:
            move_between(self.layout, self.tenant, [d], self.where[d], None)
            self.log.emit("draft_merged", item=d, into=keep, reason=reason, by=by, prompt_version=version, markers=markers)
            del self.items[d], self.where[d]
            self.merged += 1

    def drop_answered_noise(self, noise_id: str, answered_by: str, by: str, version: str | None) -> None:
        move_between(self.layout, self.tenant, [noise_id], self.where[noise_id], None)
        self.log.emit("draft_merged", item=noise_id, into=None, by=by, prompt_version=version,
                      reason=f"a noise draft, but {answered_by} answers the same question from the record")
        del self.items[noise_id], self.where[noise_id]
        self.merged += 1


def _width(item: EvalItem) -> int:
    if not item.episode:
        return 0
    a, b = map(int, re.search(r"#L(\d+)-L(\d+)$", item.episode).groups())
    return b - a


def dedupe_by_function(pool: _Pool, threshold: float) -> None:
    by_day: dict[str, list[str]] = defaultdict(list)
    for i in pool.items.values():
        by_day[day_of(i, pool.drafted(i.id))].append(i.id)
    for day, ids in by_day.items():
        if day == "unknown":
            continue
        kept: list[str] = []
        for cand in sorted(ids, key=lambda x: (-_width(pool.items[x]), x)):     # the wider answer stands
            c = pool.items[cand]
            twin = next((k for k in kept if pool.items[k].tests.type == c.tests.type and (
                (c.episode and c.episode == pool.items[k].episode)
                or overlap(c.question, pool.items[k].question) >= threshold
                or (c.tests.type != "noise" and overlap(c.answer, pool.items[k].answer) >= threshold))), None)
            if twin:
                pool.merge(twin, [cand], "the same span, or most of the same words, on the same day", "function")
            else:
                kept.append(cand)


def dedupe(layout: StoreLayout, tenant: str, machinery: Machinery, provider, log, *,
           batch: bool = True, wait=lambda: None, days: list[str] | None = None, effort: str | None = None
           ) -> dict[str, int]:
    """`days` limits the model pass to those days, for retrying a day whose reply was cut off; `effort`
    lowers how long the model deliberates, which is what a very large day needs to finish at all."""
    cfg = machinery.kernel.eval
    pool = _Pool(layout, tenant, log)
    before = len(pool.items)
    dedupe_by_function(pool, cfg.dedupe_question_overlap)
    by_function = pool.merged

    prompt, by_day = cfg.dedupe, defaultdict(list)
    for i in pool.items.values():
        by_day[day_of(i, pool.drafted(i.id))].append(i)
    reqs, days_by_id, day_name = [], {}, {}
    for n, (day, items) in enumerate(sorted(by_day.items())):
        if day == "unknown" or len(items) < 2 or (days is not None and day not in days):
            continue
        listing = "\n".join(f"- id {i.id} | {i.tests.type} | Q: {i.question} | Key: {i.answer}"
                            for i in sorted(items, key=lambda i: i.id))
        days_by_id[f"g{n}"] = {i.id for i in items}
        day_name[f"g{n}"] = day
        reqs.append(request("dedupe", tier=prompt.tier, system=prompt.prompt, schema=SCHEMA, max_tokens=16000,
                            user=f"Day {day}. {len(items)} drafted items:\n{listing}", custom_id=f"g{n}", effort=effort))
    for cid, resp in run_all(provider, reqs, batch=batch, wait=wait):
        log.examined += 1
        data, err = parse_json(resp)
        if err or not isinstance(data, dict) or not isinstance(data.get("groups"), list):
            log.count("days_failed")
            log.count(f"failed: {day_name.get(cid, cid)}")
            continue
        for g in data["groups"]:
            keep, drop = g.get("keep"), [d for d in g.get("drop", []) if d != g.get("keep")]
            members = [keep, *drop]
            if not drop or any(m not in days_by_id[cid] or m not in pool.items for m in members):
                log.count("groups_refused")               # an id from another day, or one already merged away
                continue
            types = {pool.items[m].tests.type for m in members}
            if "noise" in types and len(types) > 1:
                # The record answers it, so it is not noise. The answerable items stay as they are.
                answered = next(m for m in members if pool.items[m].tests.type != "noise")
                for m in members:
                    if pool.items[m].tests.type == "noise":
                        pool.drop_answered_noise(m, answered, machinery.artifact.routing[prompt.tier], prompt.prompt_version)
                continue
            if pool.items[keep].tests.type == "supersession" or not any(pool.items[d].tests.type == "supersession" for d in drop):
                pass
            else:                                           # never merge a supersession item away into a plain one
                keep, drop = next(d for d in drop if pool.items[d].tests.type == "supersession"), \
                    [m for m in members if pool.items[m].tests.type != "supersession"]
            pool.merge(keep, drop, str(g.get("reason", ""))[:300], machinery.artifact.routing[prompt.tier],
                       prompt.prompt_version)
    out = {"before": before, "merged_by_function": by_function, "merged_by_model": pool.merged - by_function,
           "after": len(pool.items)}
    for k, v in out.items():
        log.count(k, v)
    return out
