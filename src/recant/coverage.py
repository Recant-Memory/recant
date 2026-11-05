"""What Recant can't see. Recant must never present a result as covering a host's traffic when part
of that traffic went past it.

- `watch()`: in the host's own process, counts the requests to known model-API hosts (`recant.known_hosts`) that did
  not go through Recant, by hooking the HTTP library the SDKs share (httpx, or its successor httpx2, whichever is
  installed). On by default inside `recant.wrap` and `recant bench`; it counts nothing while the store's switch is off.
  Counts only: host, provider id, the model the body names, a request count, and the usage numbers a non-streamed
  response reports. No content is kept. A request to an unknown host is counted only when its body looks like a model
  call (`model` with `messages`, `prompt`, `input` or `contents`), as `other:<hash of the domain>`; the domain is never
  stored. It cannot see subprocesses (Claude Code itself, say), nor clients on another HTTP library (boto3's Bedrock
  client, `requests`); `reconcile()` covers those from the host's own totals.
- `summarize()` and `line()`: the Coverage line every report carries.
- `reconcile()`: the host's own per-run totals (`ts`, `model`, `tokens_in`, `tokens_out`, `cache_read`, `cache_write`,
  optional `billed`) matched against what Recant metered over the same days, through the same per-day comparison
  `recant.billing.reconcile` makes against a provider's bill, so both paths are one mechanism."""
from __future__ import annotations

import atexit
import hashlib
import json
import math
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional

from . import known_hosts

TRANSPORTS = ("httpx2", "httpx")          # the HTTP libraries the installed SDKs send through (anthropic 1.8+: httpx2)
QUALIFY_BELOW = 0.90                      # a bench verdict is qualified below this share of calls or of cost
NO_OVERALL_BELOW = 0.50                   # below this, no overall saving, only the covered part's
_SKIP_PATH = re.compile(r"count_tokens|tokenize", re.I)
_MODEL_IN_PATH = re.compile(r"/models?/([^/:?]+)")
_LLM_KEYS = ("messages", "prompt", "input", "contents")

_through: ContextVar[bool] = ContextVar("recant_through", default=False)
_lock = threading.Lock()
_counts: dict = {}
_since = time.time()
_sinks: list = []                         # (layout, tenant) the watch was started for: the switch is read from these
_hooked: list = []
_flusher: Optional[threading.Thread] = None


# ---------------------------------------------------------------- marking Recant's own requests
@contextmanager
def through():
    """Requests sent inside this block went through Recant (the wrapper, Recant's own client): not counted."""
    tok = _through.set(True)
    try:
        yield
    finally:
        _through.reset(tok)


def is_through() -> bool:
    return _through.get()


# ---------------------------------------------------------------- the watch
def _hash(domain: str) -> str:
    return "other:" + hashlib.sha256(domain.lower().encode()).hexdigest()[:12]


def classify(method: str, host: str, path: str, body: Optional[bytes]) -> Optional[tuple]:
    """(provider, host, model) for a request Recant should count, else None. Exposed for tests; reads the body only for
    the `model` field and the shape's keys, and keeps neither."""
    if (method or "").upper() != "POST" or _SKIP_PATH.search(path or ""):
        return None
    provider = known_hosts.provider_of_host(host)
    data: Any = None
    if body:
        try:
            data = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            data = None
    model = data.get("model") if isinstance(data, dict) and isinstance(data.get("model"), str) else None
    if model is None:
        m = _MODEL_IN_PATH.search(path or "")
        model = m.group(1) if m else None
    if provider is not None:
        return provider, host.lower(), model
    if isinstance(data, dict) and model and any(k in data for k in _LLM_KEYS):
        return "other", _hash(host), model
    return None


def _usage_of(data: Any) -> Optional[dict]:
    """The usage numbers a response body reports, in Recant's names, for the providers' common shapes; else None."""
    if not isinstance(data, dict):
        return None
    u = data.get("usage")
    if isinstance(u, dict) and "input_tokens" in u:                                    # Anthropic
        return {"input": int(u.get("input_tokens") or 0), "output": int(u.get("output_tokens") or 0),
                "cache_read": int(u.get("cache_read_input_tokens") or 0),
                "cache_write": int(u.get("cache_creation_input_tokens") or 0)}
    if isinstance(u, dict) and "prompt_tokens" in u:                                   # OpenAI's shape (and the routers')
        cached = int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
        return {"input": int(u.get("prompt_tokens") or 0) - cached, "output": int(u.get("completion_tokens") or 0),
                "cache_read": cached, "cache_write": 0}
    g = data.get("usageMetadata")
    if isinstance(g, dict):                                                            # Gemini
        cached = int(g.get("cachedContentTokenCount") or 0)
        return {"input": int(g.get("promptTokenCount") or 0) - cached, "output": int(g.get("candidatesTokenCount") or 0),
                "cache_read": cached, "cache_write": 0}
    return None


def _enabled() -> bool:
    from . import switch
    if not _sinks:
        return True
    return any(switch.state(layout) != "off" for layout, _ in _sinks)


def _note(key: tuple, response: Any, streamed: bool) -> None:
    try:
        _count(key, response, streamed)
    except Exception:                                                                  # noqa: BLE001 — counting never fails a call
        pass


def _count(key: tuple, response: Any, streamed: bool) -> None:
    usage = None
    if response is not None and not streamed:
        try:
            usage = _usage_of(json.loads(response.content))
        except Exception:                                                              # noqa: BLE001 — counting never fails a call
            usage = None
    with _lock:
        c = _counts.setdefault(key, {"requests": 0, "with_usage": 0, "usage": {}})
        c["requests"] += 1
        if usage:
            c["with_usage"] += 1
            for k, v in usage.items():
                c["usage"][k] = c["usage"].get(k, 0) + v


def _request_key(request: Any) -> Optional[tuple]:
    try:                                                                               # inside the host's own HTTP
        if _through.get() or not _enabled():                                           # call: nothing here may raise
            return None
        try:
            body = request.content
        except Exception:                                                              # noqa: BLE001 — a streamed body
            body = None
        return classify(request.method, request.url.host, request.url.path, body)
    except Exception:                                                                  # noqa: BLE001
        return None


def _hook(mod) -> bool:
    client, aclient = getattr(mod, "Client", None), getattr(mod, "AsyncClient", None)
    if client is None or getattr(client.send, "_recant_watch", False):
        return client is not None
    sync_send, async_send = client.send, aclient.send

    def send(self, request, *a, **kw):
        key = _request_key(request)
        resp = None
        try:
            resp = sync_send(self, request, *a, **kw)
            return resp
        finally:
            if key:
                _note(key, resp, bool(kw.get("stream", a[0] if a else False)))

    async def asend(self, request, *a, **kw):
        key = _request_key(request)
        resp = None
        try:
            resp = await async_send(self, request, *a, **kw)
            return resp
        finally:
            if key:
                _note(key, resp, bool(kw.get("stream", a[0] if a else False)))

    send._recant_watch = asend._recant_watch = True                                    # type: ignore[attr-defined]
    send._recant_original, asend._recant_original = sync_send, async_send             # type: ignore[attr-defined]
    client.send, aclient.send = send, asend
    return True


def watch(store=None, tenant: Optional[str] = None, *, flush_every_s: Optional[float] = None) -> list[str]:
    """Start counting (idempotent). With a store and tenant, leftover counts are written there when the process exits,
    and every `flush_every_s` seconds if given (a long-running host); a wrapped client's runs write theirs as they
    close. Returns the HTTP libraries hooked; an empty list means the watch cannot see anything in this process."""
    import importlib
    global _flusher
    for name in TRANSPORTS:
        if name in _hooked:
            continue
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        if _hook(mod):
            _hooked.append(name)
    if store is not None and tenant:
        from .store import StoreLayout
        layout = store if isinstance(store, StoreLayout) else StoreLayout(store)
        with _lock:
            first = not _sinks
            if all(str(l.root) != str(layout.root) or t != tenant for l, t in _sinks):
                _sinks.append((layout, tenant))
        if first:
            atexit.register(_flush_at_exit)
        if flush_every_s and _flusher is None:
            def loop():
                while True:
                    time.sleep(flush_every_s)
                    _flush_at_exit()
            _flusher = threading.Thread(target=loop, name="recant-coverage-flush", daemon=True)
            _flusher.start()
    return list(_hooked)


def watching() -> bool:
    return bool(_hooked)


def snapshot() -> dict:
    """The counts so far (not drained), {(provider, host, model): {requests, with_usage, usage}}."""
    with _lock:
        return {k: {"requests": v["requests"], "with_usage": v["with_usage"], "usage": dict(v["usage"])} for k, v in _counts.items()}


def delta(before: dict, after: Optional[dict] = None) -> dict:
    """What was counted between two snapshots."""
    after = snapshot() if after is None else after
    out = {}
    for k, v in after.items():
        b = before.get(k, {"requests": 0, "with_usage": 0, "usage": {}})
        n = v["requests"] - b["requests"]
        if n > 0:
            out[k] = {"requests": n, "with_usage": v["with_usage"] - b["with_usage"],
                      "usage": {u: v["usage"].get(u, 0) - b["usage"].get(u, 0) for u in v["usage"]}}
    return out


def drain() -> tuple[dict, float, float]:
    """The counts since the last drain, and their window; the counter starts again."""
    global _since
    with _lock:
        out, t0, t1 = dict(_counts), _since, time.time()
        _counts.clear()
        _since = t1
    return out, t0, t1


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


def records(counts: dict, t0: float, t1: float) -> list[dict]:
    """The `coverage` records for a window's counts; a watched window with nothing unseen is one record of none."""
    if not counts:
        return [dict(provider="none", host="none", model=None, requests=0, since=_iso(t0), until=_iso(t1))]
    return [dict(provider=p, host=h, model=m, requests=c["requests"], since=_iso(t0), until=_iso(t1),
                 usage=c["usage"] or None, with_usage=c["with_usage"]) for (p, h, m), c in sorted(counts.items(), key=str)]


def flush_into(log) -> int:
    """Write the counts since the last drain into an open log run (a wrapped client's run, as it closes)."""
    if not _hooked:
        return 0
    counts, t0, t1 = drain()
    recs = records(counts, t0, t1)
    for r in recs:
        log.emit("coverage", **r)
    return len(recs)


def _flush_at_exit() -> None:
    if not _hooked or not _sinks:
        return
    counts, t0, t1 = drain()
    if not counts:
        return
    try:
        from .log import LogWriter
        layout, tenant = _sinks[0]
        log = LogWriter(layout, tenant, "coverage", artifact_hash="host", kernel_hash="host", params={})
        for r in records(counts, t0, t1):
            log.emit("coverage", **r)
        log.finish()
    except Exception:                                                                  # noqa: BLE001 — never fail the host's exit
        pass


# ---------------------------------------------------------------- the Coverage line
def summarize(seen: Iterable[dict], unseen: Iterable[dict], *, price: Optional[Callable[[str, dict], Optional[float]]] = None) -> dict:
    """Coverage from what Recant metered (`seen`: costshape calls, or model_call records) and the watch's `coverage`
    records (`unseen`). Shares of calls, and of list-priced cost where the unseen usage was read and the model priced."""
    seen, unseen = list(seen), [u for u in unseen if (u.get("requests") or 0) > 0]
    n_seen = sum(int(c.get("calls") or 1) for c in seen)
    usd_seen = sum((price(c.get("model") or c.get("model_id"), _usage_rec(c)) or 0.0) for c in seen) if price else None
    by: dict = {}
    for u in unseen:
        k = (u["provider"], u["host"], u.get("model"))
        b = by.setdefault(k, {"requests": 0, "with_usage": 0, "usage": {}})
        b["requests"] += u["requests"]
        b["with_usage"] += u.get("with_usage") or 0
        for t, v in (u.get("usage") or {}).items():
            b["usage"][t] = b["usage"].get(t, 0) + v
    n_unseen = sum(b["requests"] for b in by.values())
    usd_unseen, unpriced = 0.0, 0
    for (p, h, m), b in by.items():
        usd = price(m, b["usage"]) if (price and b["with_usage"]) else None
        b["usd_list"] = usd
        if usd is None or b["with_usage"] < b["requests"]:
            unpriced += b["requests"] - (b["with_usage"] if usd is not None else 0)
        usd_unseen += usd or 0.0
    total = n_seen + n_unseen
    share_usd = None
    if price and unpriced == 0 and (usd_seen or 0) + usd_unseen > 0:
        share_usd = usd_seen / (usd_seen + usd_unseen)
    return {"calls_seen": n_seen, "calls_unseen": n_unseen, "share_calls": (n_seen / total) if total else None,
            "usd_seen": usd_seen, "usd_unseen": usd_unseen if price else None, "share_usd": share_usd,
            "unseen_unpriced": unpriced,
            "unseen": [{"provider": p, "host": h, "model": m, **b} for (p, h, m), b in sorted(by.items(), key=lambda kv: -kv[1]["requests"])]}


def _usage_rec(c: dict) -> dict:
    if "usage" in c and isinstance(c["usage"], dict):
        u = c["usage"]
        return {"input": u.get("input", 0), "output": u.get("output", 0), "cache_read": u.get("cache_read", 0),
                "cache_write": u.get("cache_write", 0)}
    return {"input": c.get("uncached", 0), "output": c.get("output", 0), "cache_read": c.get("cache_read", 0),
            "cache_write": c.get("cache_write", 0)}


def share(s: dict) -> Optional[float]:
    """The share a verdict is judged on: the lower of calls and list-priced cost, where both are known."""
    xs = [x for x in (s.get("share_calls"), s.get("share_usd")) if x is not None]
    return min(xs) if xs else None


def _where(u: dict) -> str:
    host = "an unlisted host (domain hashed)" if u["host"].startswith("other:") else u["host"]
    return f"{u['requests']} calls to {host}" + (f", model {u['model']}" if u.get("model") else "")


def line(s: Optional[dict], *, watched: bool = True) -> str:
    """"Recant saw 96% of model calls (unseen: 41 calls to openrouter.ai, model glm-latest)"."""
    if s is None or (not s["calls_seen"] and not s["calls_unseen"]):
        return "coverage: no model calls in this window"
    if not s["calls_unseen"]:
        return (f"coverage: Recant saw all {s['calls_seen']:,} model calls the watch could see in this process"
                if watched else "coverage: not measured (the watch was not on); calls outside Recant are not visible")
    seen = pct(s["share_calls"])
    cost = f", {pct(s['share_usd'])} of list-priced cost" if s.get("share_usd") is not None else ""
    shown = "; ".join(_where(u) for u in s["unseen"][:3]) + ("; …" if len(s["unseen"]) > 3 else "")
    return f"coverage: Recant saw {seen} of model calls{cost} (unseen: {shown})"


def verdict_qualifier(s: Optional[dict]) -> tuple[Optional[str], bool]:
    """(the line qualifying a verdict, whether an overall saving may be given). Below 90% of calls or of list-priced
    cost the verdict is qualified; below 50% it gives no overall saving, only the covered part's."""
    sh = share(s) if s else None
    if sh is None or sh >= QUALIFY_BELOW:
        return None, True
    to = ", ".join(sorted({(u["host"] if not u["host"].startswith("other:") else "an unlisted host") for u in s["unseen"]}))
    msg = f"this result covers only {sh:.0%} of your traffic; the rest went to {to}, which Recant did not see"
    if sh < NO_OVERALL_BELOW:
        return msg + ". No overall saving is given: the figures below are for the covered part only", False
    return msg, True


# ---------------------------------------------------------------- reconciling against the host's own totals
USAGE_FIELDS = ("ts", "model", "tokens_in", "tokens_out", "cache_read", "cache_write")


def read_usage(path) -> list[dict]:
    """A per-run usage file, JSON lines in the minimal shape (`ts`, `model`, `tokens_in`, `tokens_out`, `cache_read`,
    `cache_write`, optional `billed`); extra fields are ignored and never stored."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, ln in enumerate(f, 1):
            if not ln.strip():
                continue
            r = json.loads(ln)
            missing = [k for k in ("ts", "model") if not r.get(k)]
            if missing:
                raise ValueError(f"{path}:{i}: missing {', '.join(missing)}")
            rows.append({"ts": r["ts"], "model": r["model"], "tokens_in": int(r.get("tokens_in") or 0),
                         "tokens_out": int(r.get("tokens_out") or 0), "cache_read": int(r.get("cache_read") or 0),
                         "cache_write": int(r.get("cache_write") or 0), "billed": r.get("billed")})
    return rows


def reconcile(usage: list[dict], calls: list[dict], *, price: Optional[Callable[[str, dict], Optional[float]]] = None) -> dict:
    """The host's own totals (`usage`, as `read_usage` gives them) against the model_call records Recant metered
    (`calls`), per day and model as `recant.billing.reconcile` compares a bill, then summed by model and provider: tokens
    and list-priced cost reported, the part Recant saw, and the runs reported against the calls metered. Runs are not
    matched one by one: a usage file has no request ids, so "seen" is the metered share of the same days' tokens."""
    from .billing import BilledUsage, model_family
    from .billing import reconcile as by_day
    days = {r["ts"][:10] for r in usage}
    billed = [BilledUsage(day=r["ts"][:10], key_id="host", model=r["model"], uncached=r["tokens_in"],
                          cache_write_5m=r["cache_write"], cache_read=r["cache_read"], output=r["tokens_out"]) for r in usage]
    rows = by_day([dict(c, key="host") for c in calls], billed, key_ids={"host": "host"}, days=days)
    by: dict = {}
    runs: dict = {}
    for r in usage:
        m = model_family(r["model"])
        x = runs.setdefault(m, {"runs": 0, "subscription": 0})
        x["runs"] += 1
        x["subscription"] += r.get("billed") == "subscription"
    for row in rows:
        m = row["model"]
        b, me = row["billed_tokens"], row["metered_tokens"]
        seen = {k: min(b[k], me[k]) for k in b}
        t = by.setdefault(m, {"tokens": 0, "tokens_seen": 0, "usd_list": 0.0, "usd_seen": 0.0, "priced": True,
                              "calls_metered": 0})
        t["tokens"] += sum(b.values())
        t["tokens_seen"] += sum(seen.values())
        t["calls_metered"] += row["calls_metered"]
        as_usage = lambda d: {"input": d["uncached"], "output": d["output"], "cache_read": d["cache_read"],   # noqa: E731
                              "cache_write": d["cache_write_5m"] + d["cache_write_1h"]}
        usd_b = price(m, as_usage(b)) if price else None
        usd_s = price(m, as_usage(seen)) if price else None
        if usd_b is None:
            t["priced"] = False
        else:
            t["usd_list"] += usd_b
            t["usd_seen"] += usd_s or 0.0
    models = []
    for m, t in sorted(by.items(), key=lambda kv: -kv[1]["tokens"]):
        if not t["tokens"] and m not in runs:
            continue                                        # metered, and not in the host's file: not the file's traffic
        rr = runs.get(m, {"runs": 0, "subscription": 0})
        models.append({"model": m, "provider": known_hosts.provider_of_model(m), "route": known_hosts.route_of_model(m),
                       "runs": rr["runs"], "runs_subscription": rr["subscription"], "calls_metered": t["calls_metered"],
                       "tokens": t["tokens"], "tokens_seen": t["tokens_seen"],
                       "share_tokens": (t["tokens_seen"] / t["tokens"]) if t["tokens"] else None,
                       "usd_list": t["usd_list"] if t["priced"] else None, "usd_seen": t["usd_seen"] if t["priced"] else None})
    tok = sum(m["tokens"] for m in models)
    tok_seen = sum(m["tokens_seen"] for m in models)
    priced = [m for m in models if m["usd_list"] is not None]
    usd, usd_seen = sum(m["usd_list"] for m in priced), sum(m["usd_seen"] for m in priced)
    prov: dict = {}
    for m in models:
        p = prov.setdefault(m["provider"], {"tokens": 0, "tokens_seen": 0, "runs": 0})
        p["tokens"] += m["tokens"]
        p["tokens_seen"] += m["tokens_seen"]
        p["runs"] += m["runs"]
    return {"window": [min(days), max(days)] if days else None, "runs": len(usage),
            "runs_subscription": sum(r.get("billed") == "subscription" for r in usage),
            "tokens": tok, "tokens_seen": tok_seen, "share_tokens": (tok_seen / tok) if tok else None,
            "usd_list_priced": usd, "usd_seen": usd_seen, "share_usd": (usd_seen / usd) if usd else None,
            "unpriced_models": [m["model"] for m in models if m["usd_list"] is None],
            "by_model": models, "by_provider": prov}


def pct(x: float) -> str:
    """A share as a percentage that never reads 100% while some of it is missing: 0.998 is 99.8%, not 100%."""
    if x < 1 and round(x * 100) >= 100:
        return f"{math.floor(x * 1000) / 10:.1f}%"
    return f"{x:.0%}"


def render_reconcile(r: dict) -> str:
    out = [f"coverage against your own totals, {r['window'][0]} to {r['window'][1]}: {r['runs']} runs reported"
           if r.get("window") else "coverage against your own totals: no runs in the file"]
    if r.get("share_tokens") is not None:
        by = {m["model"]: m for m in r["by_model"]}
        left = [f"{m} ({by[m]['tokens']:,} tokens, {pct(by[m]['share_tokens'])} seen)" if m in by and by[m].get("share_tokens") is not None
                else m for m in r["unpriced_models"]]
        cost = (f", {pct(r['share_usd'])} of the list-priced cost (${r['usd_seen']:,.2f} of ${r['usd_list_priced']:,.2f}"
                + (f"; unpriced, so left out of the cost share: {', '.join(left)}" if left else "") + ")") \
            if r.get("share_usd") is not None else ""
        out.append(f"Recant saw {pct(r['share_tokens'])} of the tokens ({r['tokens_seen']:,} of {r['tokens']:,}){cost}")
    out.append("by model (runs reported; calls Recant metered on those days; tokens; share seen; list-priced):")
    for m in r["by_model"]:
        via = f" via {m['route']}" if m["route"] else ""
        usd = f"${m['usd_list']:,.2f}" if m["usd_list"] is not None else "unpriced"
        sub = f", {m['runs_subscription']} on a subscription" if m["runs_subscription"] else ""
        out.append(f"  {m['model']} ({m['provider']}{via}): {m['runs']} runs{sub}; {m['calls_metered']} calls metered; "
                   f"{m['tokens']:,} tokens; {pct(m['share_tokens'])} seen; {usd}" if m["share_tokens"] is not None else
                   f"  {m['model']} ({m['provider']}{via}): {m['runs']} runs{sub}; no tokens reported")
    if r["runs_subscription"]:
        out.append(f"subscription: {r['runs_subscription']} runs were billed to a subscription (cost 0, real tokens): most "
                   "likely Claude Code or the Claude Agent SDK, which recant.wrap does not see. Meter them with "
                   "recant.metering.from_claude_code or from_agent_sdk; on a subscription the point is tokens against "
                   "the plan's limits, not dollars")
    other = {p: v for p, v in r["by_provider"].items() if p != "anthropic" and v["tokens"]}
    for p, v in sorted(other.items(), key=lambda kv: -kv[1]["tokens"]):
        out.append(f"provider {p}: {v['runs']} runs, {v['tokens']:,} tokens ({v['tokens'] / max(1, r['tokens']):.0%} of the "
                   "file's), with no Recant wrapper for it: this is the evidence a second provider wrapper would rest on")
    return "\n".join(out)
