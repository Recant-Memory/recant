"""Recant's one quality standard, for cost levers and memory alike: quality is held when the observed
difference shows no loss and the 95% interval's lower end is no worse than −3 points. Intervals are a paired bootstrap
by cluster — the principal, or the task in a run of repeated trials — because the items inside one are not
independent: a cluster is resampled whole, with every item and trial in it.
"""
from __future__ import annotations

import math
import random
from typing import Iterable, Optional

FLOOR = -0.03            # the interval's lower end may be no worse than −3 points
LEVEL = 0.95


def paired_bootstrap(items: Iterable[tuple], *, n: int = 10_000, seed: int = 0, level: float = LEVEL) -> dict:
    """`items` are (cluster, a, b): the same item scored under the current setting (a) and the candidate (b), as 1/0
    or any number. The difference is b − a, pooled over items; its interval comes from resampling clusters with
    replacement, `n` times. Returns delta, interval, items, clusters."""
    by: dict = {}
    for c, a, b in items:
        s = by.setdefault(c, [0.0, 0])
        s[0] += float(b) - float(a)
        s[1] += 1
    keys = sorted(by, key=str)
    k, total = len(keys), sum(by[c][1] for c in keys)
    if not total:
        return {"delta": 0.0, "interval": (0.0, 0.0), "items": 0, "clusters": 0}
    delta = sum(by[c][0] for c in keys) / total
    rng, ds = random.Random(seed), []
    for _ in range(n):
        d = m = 0.0
        for _ in range(k):
            s = by[keys[rng.randrange(k)]]
            d += s[0]
            m += s[1]
        ds.append(d / m)
    ds.sort()
    lo, hi = ds[int((1 - level) / 2 * n)], ds[min(n - 1, int((1 + level) / 2 * n))]
    return {"delta": delta, "interval": (lo, hi), "items": total, "clusters": k}


def held(delta: float, lo: float, floor: float = FLOOR) -> bool:
    """No observed loss, and the interval's lower end no worse than the floor."""
    return delta >= 0 and lo >= floor


def verdict(res: dict, floor: float = FLOOR) -> str:
    """'held', 'loss observed' (the difference itself is below zero: no sample confirms it), or 'interval too wide'
    (no loss observed, but the lower end is below the floor: a larger sample may confirm it)."""
    lo = res["interval"][0]
    if res["delta"] < 0:
        return "loss observed"
    return "held" if lo >= floor else "interval too wide"


def clusters_needed(res: dict, floor: float = FLOOR) -> Optional[int]:
    """Clusters needed for the lower end to clear the floor if the observed difference held and the spread per
    cluster stayed the same (the interval's half-width scales with 1/√clusters). None when a loss was observed —
    then more of the same sample cannot pass it. A projection, not a promise."""
    d, (lo, _), k = res["delta"], res["interval"], res["clusters"]
    if d < 0 or not k:
        return None
    if lo >= floor:
        return k
    half, room = d - lo, d - floor
    return math.ceil(k * (half / room) ** 2)


# Levers that leave what the model sees unchanged are exempt from the quality gate. In its place,
# an equivalence check: the model's input is byte-identical apart from cache markers. Their observed quality is still
# reported. Folding changes what the model sees and is gated like any other content lever.
EXEMPT = ("breakpoint.", "ttl.", "batch.")


def exempt(lever_id: str) -> bool:
    return lever_id.startswith(EXEMPT)


def strip_cache_markers(obj):
    """A request with its cache markers removed: every `cache_control` key, and a content list that is then one bare
    text block written back as the string it wraps (the provider reads both the same; a breakpoint can only be set
    on a block, so placing one turns a string into a block)."""
    if isinstance(obj, dict):
        out = {k: strip_cache_markers(v) for k, v in obj.items() if k != "cache_control"}
        for key in ("content", "system"):           # a message's content, or the request's system (the levers mark both)
            c = out.get(key)
            if isinstance(c, list) and len(c) == 1 and isinstance(c[0], dict) and set(c[0]) == {"type", "text"} and c[0]["type"] == "text":
                out[key] = c[0]["text"]
        return out
    if isinstance(obj, list):
        return [strip_cache_markers(x) for x in obj]
    return obj


def equivalent(built, sent) -> bool:
    """The equivalence check: what the host built and what was sent serialize to the same bytes once cache markers
    are stripped from both."""
    import json
    enc = lambda x: json.dumps(strip_cache_markers(x), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()  # noqa: E731
    return enc(built) == enc(sent)
