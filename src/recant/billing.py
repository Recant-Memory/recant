"""Reconcile what Recant metered against what the provider billed. Recant sees only the calls that
pass through it; the provider's own usage report sees every call on a key. Per day, key and model: billed but not
metered is traffic outside Recant (a fix-the-cause item); metered but not billed points to a pricing or logging bug on
Recant's side. Days are compared once settled: the provider's report lags, so a day counts when it is old enough and its
figures stopped changing between two fetches. Provider-neutral: a `BillingSource` supplies the billed side.

Without a billing source, Recant says so ("calls outside Recant are not visible") and never guesses."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Optional, Protocol

TOKEN_CLASSES = ("uncached", "cache_write_5m", "cache_write_1h", "cache_read", "output")
NOT_VISIBLE = ("billing: not reconciled (no billing access), so calls made on your key outside this process are not checked "
               "against the provider's bill; the coverage line above counts only what this process sent")


@dataclass(frozen=True)
class BilledUsage:
    """One day of one key's use of one model, as the provider billed it, in tokens."""
    day: str
    key_id: str
    model: str
    uncached: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0
    cache_read: int = 0
    output: int = 0


class BillingSource(Protocol):
    def usage(self, start_day: str, end_day: str) -> list[BilledUsage]:
        """Billed usage per day, key and model, for [start_day, end_day)."""
        ...

    def key_ids(self, secrets: dict[str, str]) -> dict[str, str]:
        """The provider's id for each of the consumer's keys ({label: secret} -> {label: id}), without exposing them."""
        ...


def model_family(model: Optional[str]) -> Optional[str]:
    """A model id without the date a provider may append (`claude-haiku-4-5-20251001` is `claude-haiku-4-5`)."""
    import re
    return re.sub(r"-\d{8}$", "", model) if model else model


def metered_tokens(call: dict) -> dict[str, int]:
    """A model_call record's usage in the billed token classes."""
    u = call.get("usage") or {}
    long_ = min(u.get("cache_write_long", 0), u.get("cache_write", 0))
    return {"uncached": u.get("input", 0), "cache_write_5m": u.get("cache_write", 0) - long_, "cache_write_1h": long_,
            "cache_read": u.get("cache_read", 0), "output": u.get("output", 0)}


def _usd(tokens: dict, price) -> Optional[float]:
    """Dollars for token classes at a `recant.provider.Prices` row (USD per million tokens)."""
    if price is None:
        return None
    write = price.cache_write if price.cache_write is not None else price.input
    write_long = price.cache_write_long if price.cache_write_long is not None else write
    read = price.cache_read if price.cache_read is not None else price.input
    return (tokens["uncached"] * price.input + tokens["cache_write_5m"] * write + tokens["cache_write_1h"] * write_long
            + tokens["cache_read"] * read + tokens["output"] * price.output) / 1_000_000


def reconcile(calls: list[dict], billed: list[BilledUsage], *, key_ids: dict[str, str], days: set[str],
              prices: Optional[dict] = None, label_of: Callable[[dict], Optional[str]] = lambda c: c.get("key"),
              tolerance: float = 0.01) -> list[dict]:
    """Per (day, key label, model) over `days`: what was billed, what was metered, and the difference both ways, in
    tokens and (where a price applies) dollars. Batch calls are billed at half; `prices` rows carry batch_factor.
    `label_of(call)` names the key a call spent; billed rows on a key the consumer did not name are labelled by id.
    A difference under `tolerance` of the billed tokens (rounding) is not reported."""
    by_id = {v: k for k, v in key_ids.items()}
    met: dict = {}
    for c in calls:
        if c.get("kind", "model_call") != "model_call":
            continue                                    # a failed call with usage (a refusal) is billed: it counts
        day = (c.get("ts") or "")[:10]
        if day not in days:
            continue
        key = (day, label_of(c) or "unlabelled", model_family(c.get("model_id")))
        t = met.setdefault(key, {k: 0 for k in TOKEN_CLASSES} | {"calls": 0, "unsplit": False})
        m = metered_tokens(c)
        for k in TOKEN_CLASSES:
            t[k] += m[k]
        t["calls"] += 1
        if "cache_write_long" not in (c.get("usage") or {}):
            t["unsplit"] = True                         # an older record: one number for both cache lifetimes
    bil: dict = {}
    for b in billed:
        if b.day not in days:
            continue
        key = (b.day, by_id.get(b.key_id, f"unnamed key {b.key_id}"), model_family(b.model))
        t = bil.setdefault(key, {k: 0 for k in TOKEN_CLASSES})
        for k in TOKEN_CLASSES:
            t[k] += getattr(b, k)
    out = []
    for key in sorted(set(met) | set(bil), key=lambda k: (k[0], k[1], k[2] or "")):
        day, label, model = key
        m = dict(met.get(key, {k: 0 for k in TOKEN_CLASSES}))
        b = bil.get(key, {k: 0 for k in TOKEN_CLASSES})
        if m.get("unsplit"):                            # compare cache writes as a total where the log did not split them
            writes = m["cache_write_5m"] + m["cache_write_1h"]
            m["cache_write_1h"] = min(writes, b["cache_write_1h"])
            m["cache_write_5m"] = writes - m["cache_write_1h"]
        over = {k: max(0, b[k] - m[k]) for k in TOKEN_CLASSES}     # billed, not metered
        under = {k: max(0, m[k] - b[k]) for k in TOKEN_CLASSES}    # metered, not billed
        billed_total = sum(b.values())
        if sum(over.values()) <= tolerance * billed_total and sum(under.values()) <= tolerance * max(billed_total, 1):
            status = "matches"
        elif sum(over.values()) > sum(under.values()):
            status = "unmetered"
        else:
            status = "metered_not_billed"
        price = (prices or {}).get(model)
        out.append({"day": day, "key": label, "model": model, "status": status, "calls_metered": m.get("calls", 0),
                    "billed_tokens": b, "metered_tokens": {k: m[k] for k in TOKEN_CLASSES},
                    "unmetered_tokens": sum(over.values()), "unmetered_usd": _usd(over, price),
                    "metered_not_billed_tokens": sum(under.values()), "metered_not_billed_usd": _usd(under, price),
                    "billed_usd_list": _usd(b, price)})
    return out


def settled_days(layout, tenant: str, billed: list[BilledUsage], *, today: Optional[date] = None, lag_days: int = 2,
                 final_after_days: int = 7) -> set[str]:
    """Days whose billed figures are final enough to compare: at least `lag_days` old and unchanged since the last
    fetch, or older than `final_after_days`. Each fetch is kept as a snapshot under the store's costshape directory."""
    today = today or datetime.now(timezone.utc).date()
    d = layout.costshape_dir(tenant) / "billing"
    d.mkdir(parents=True, exist_ok=True)
    by_day: dict = {}
    for b in billed:
        by_day.setdefault(b.day, []).append(asdict(b))
    out = set()
    for day, rows in by_day.items():
        age = (today - date.fromisoformat(day)).days
        snap = d / f"{day}.json"
        now = json.dumps(sorted(rows, key=lambda r: (r["key_id"], r["model"])), sort_keys=True)
        before = snap.read_text() if snap.exists() else None
        snap.write_text(now)
        if age >= final_after_days or (age >= lag_days and before == now):
            out.add(day)
    return out


def summary(rows: list[dict]) -> dict:
    """Per key: unmetered and metered-not-billed tokens and dollars, and the days each was seen."""
    out: dict = {}
    for r in rows:
        s = out.setdefault(r["key"], {"unmetered_tokens": 0, "unmetered_usd": 0.0, "metered_not_billed_tokens": 0,
                                      "metered_not_billed_usd": 0.0, "days_unmetered": [], "days_compared": set()})
        s["days_compared"].add(r["day"])
        if r["status"] == "unmetered":
            s["unmetered_tokens"] += r["unmetered_tokens"]
            s["unmetered_usd"] += r["unmetered_usd"] or 0.0
            s["days_unmetered"].append(r["day"])
        elif r["status"] == "metered_not_billed":
            s["metered_not_billed_tokens"] += r["metered_not_billed_tokens"]
            s["metered_not_billed_usd"] += r["metered_not_billed_usd"] or 0.0
    for s in out.values():
        s["days_compared"] = len(s["days_compared"])
        s["days_unmetered"] = sorted(set(s["days_unmetered"]))
    return out


def last_window(days: int = 14, today: Optional[date] = None) -> tuple[str, str]:
    today = today or datetime.now(timezone.utc).date()
    return (today - timedelta(days=days)).isoformat(), today.isoformat()
