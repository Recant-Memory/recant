"""costshape: a host's context-cost shape, learned from the calls Recant meters, and Recant's levers sized on it. Standard library
only; nothing about the host's internals,
its prompts or its provider is read — only the numbers each call and run already carries.

Per call (Recant's own names): run, seq, session, start (unix seconds), latency, model, job, uncached, cache_write,
cache_write_long (the part at the long lifetime), cache_read, output, reasoning, and `cost`, the host's billed
figure when it has one. Per run, optionally: memory, the tokens Recant served at its start.

Prices, in order: the consumer's table; the library's dated list prices (`LIST_PRICES`, warned when stale); prices
learned by regression from billed cost. A model with none of these is reported in tokens only (by design:
levers are always sized in tokens, and in dollars where a table applies). What is safe to apply without a quality
test is what leaves the model's input unchanged: cache lifetime, breakpoints, batching background work, and folding
less often."""
from __future__ import annotations

import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from statistics import median
from typing import Iterable, Optional

from .toolloop import TOOL_LOOP_MIN, args_hash   # one definition with the runtime guard

M = 1e-6
LIST_PRICES_AS_OF = date(2026, 11, 5)
STALE_AFTER_DAYS = 90


@dataclass
class Profile:
    """Prices per token; w and r are the cache-write and cache-read multipliers on input, w_long the long-lifetime
    write's; batch multiplies a batch call's whole cost."""
    p: float
    po: float
    w: float = 1.0
    r: float = 1.0
    w_long: Optional[float] = None
    short_ttl: float = 300.0
    long_ttl: float = 3600.0
    batch: float = 1.0
    above: Optional[int] = None          # a model priced by prompt length: a request whose prompt is over this many
    over: Optional["Profile"] = None     # tokens (uncached, cache reads and cache writes together) pays `over`'s prices

    @property
    def wl(self) -> float:
        """The long-lifetime write multiplier: `w_long`, else the short write's (a table's documented default)."""
        return self.w_long if self.w_long is not None else self.w

    def at(self, prompt) -> "Profile":
        """The rate card a request with a prompt of this many tokens pays: each request is priced on its own."""
        return self.over if (self.above is not None and self.over is not None and prompt > self.above) else self

    def bill(self, uncached=0, write=0, read=0, output=0, write_long=0, *, prompt=None):
        """One request's cost. Its card is chosen by its prompt, `uncached + write + read` unless `prompt` is given
        (as when only part of a request is billed)."""
        pf = self.at(uncached + write + read if prompt is None else prompt)
        long_ = min(write_long, write)
        wl = pf.w_long if pf.w_long is not None else pf.w
        return pf.p * (uncached + pf.w * (write - long_) + wl * long_ + pf.r * read) + pf.po * output


def _prompt(c) -> int:
    """A metered call's prompt in tokens: what decides its rate card on a model priced by prompt length."""
    return (c.get("uncached") or 0) + (c.get("cache_read") or 0) + (c.get("cache_write") or 0)


# The library's default: list prices by model ID, dated. Vendor data, not any host's; a
# consumer's table overrides it row by row. Rows from the vendor's price page (platform.claude.com, pricing), read
# 2026-11-05: cache reads are 0.05x input on Opus 5.5 and Sonnet 5.5, 0.025x on Fable 5.1, 0.1x elsewhere; 5-minute
# writes 1.25x, 1-hour writes 2x; batch half price. Haiku 5.5 has two cards, split at a 100,000-token prompt.
# The shortest prefix the API caches, by model: a shorter one is sent with its marker and silently not cached (no write,
# no read). From the vendor's prompt-caching documentation, read 2026-11-06, which says to check the values before
# relying on them. A model not listed is taken at 4,096, the longest minimum listed, so a saving is never sized on a
# prefix that may not cache.
CACHE_MIN_TOKENS_AS_OF = date(2026, 11, 6)
CACHE_MIN_TOKENS: dict[str, int] = {
    **{m: 512 for m in ("claude-opus-5-5", "claude-opus-5", "claude-fable-5", "claude-mythos-5", "claude-fable-5-1",
                        "claude-mythos-5-1", "claude-sonnet-5-5", "claude-haiku-5-5")},
    **{m: 1024 for m in ("claude-opus-4-8", "claude-sonnet-5", "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-opus-4-1",
                         "claude-opus-4", "claude-sonnet-4")},
    **{m: 2048 for m in ("claude-opus-4-7", "claude-3-5-haiku")},
    **{m: 4096 for m in ("claude-opus-4-6", "claude-opus-4-5", "claude-haiku-4-5")},
}


def cache_min(model: str) -> int:
    """The shortest cacheable prefix for `model` (a dated snapshot or alias suffix allowed), else 4,096."""
    for m in sorted(CACHE_MIN_TOKENS, key=len, reverse=True):
        if model == m or model.startswith(m + "-") or model.startswith(m + "@"):
            return CACHE_MIN_TOKENS[m]
    return max(CACHE_MIN_TOKENS.values())


LIST_PRICES: dict[str, Profile] = {
    "claude-opus-5-5": Profile(p=4 * M, po=20 * M, w=1.25, r=0.05, w_long=2.0, batch=0.5),
    "claude-fable-5-1": Profile(p=10 * M, po=50 * M, w=1.25, r=0.025, w_long=2.0, batch=0.5),
    "claude-mythos-5-1": Profile(p=10 * M, po=50 * M, w=1.25, r=0.025, w_long=2.0, batch=0.5),
    "claude-fable-5": Profile(p=10 * M, po=50 * M, w=1.25, r=0.10, w_long=2.0, batch=0.5),
    "claude-sonnet-5": Profile(p=2 * M, po=10 * M, w=1.25, r=0.10, w_long=2.0, batch=0.5),
    "claude-sonnet-5-5": Profile(p=2 * M, po=10 * M, w=1.25, r=0.05, w_long=2.0, batch=0.5),   # $2/$10, cache reads $0.10 (0.05x; the row read $0.20 before 0.3.2)
    "claude-opus-5": Profile(p=5 * M, po=25 * M, w=1.25, r=0.10, w_long=2.0, batch=0.5),
    "claude-opus-4-8": Profile(p=5 * M, po=25 * M, w=1.25, r=0.10, w_long=2.0, batch=0.5),
    "claude-opus-4-7": Profile(p=5 * M, po=25 * M, w=1.25, r=0.10, w_long=2.0, batch=0.5),
    "claude-opus-4-6": Profile(p=5 * M, po=25 * M, w=1.25, r=0.10, w_long=2.0, batch=0.5),
    "claude-sonnet-4-6": Profile(p=3 * M, po=15 * M, w=1.25, r=0.10, w_long=2.0, batch=0.5),
    "claude-haiku-4-5": Profile(p=1 * M, po=5 * M, w=1.25, r=0.10, w_long=2.0, batch=0.5),
    "claude-haiku-5-5": Profile(p=0.10 * M, po=0.50 * M, w=1.25, r=0.10, w_long=2.0, batch=0.5, above=100_000,
                                over=Profile(p=0.50 * M, po=2.50 * M, w=1.25, r=0.10, w_long=2.0, batch=0.5)),
}


def list_prices_stale(today: Optional[date] = None) -> Optional[str]:
    """A warning when the list prices are older than STALE_AFTER_DAYS, else None."""
    today = today or datetime.now(timezone.utc).date()
    age = (today - LIST_PRICES_AS_OF).days
    if age > STALE_AFTER_DAYS:
        return f"the library's list prices are {age} days old (as of {LIST_PRICES_AS_OF}); pass your own table"
    return None


def profile_from_prices(row) -> Profile:
    """A consumer's price row (anything with input, output, cache_read, cache_write, cache_write_long, batch_factor in
    USD per million tokens, such as `recant.provider.Prices`) as a Profile."""
    p = row.input * M
    w = (row.cache_write if row.cache_write is not None else row.input) / row.input if row.input else 1.0
    r = (row.cache_read if row.cache_read is not None else row.input) / row.input if row.input else 1.0
    cwl = getattr(row, "cache_write_long", None)
    over = getattr(row, "over", None)
    return Profile(p=p, po=row.output * M, w=w, r=r, w_long=(cwl / row.input) if (cwl is not None and row.input) else None,
                   batch=getattr(row, "batch_factor", 1.0), above=getattr(row, "above", None) if over is not None else None,
                   over=profile_from_prices(over) if over is not None else None)


def price_row(pf: Profile) -> dict:
    """A Profile as a row of the `--prices` table (USD per million tokens), the format `recant.provider.Prices` reads."""
    row = {"input": round(pf.p / M, 6), "output": round(pf.po / M, 6), "cache_read": round(pf.p * pf.r / M, 6),
           "cache_write": round(pf.p * pf.w / M, 6), "batch_factor": pf.batch}
    if pf.w_long is not None:
        row["cache_write_long"] = round(pf.p * pf.w_long / M, 6)
    if pf.above is not None and pf.over is not None:
        row["above"], row["over"] = pf.above, price_row(pf.over)
    return row


@dataclass
class Run:
    run: str
    session: Optional[str]
    model: str
    job: str
    calls: list = field(default_factory=list)
    memory: Optional[float] = None
    memory_measured: Optional[bool] = None

    @property
    def n(self):
        return len(self.calls)

    def X(self, i):
        c = self.calls[i]
        return c["uncached"] + c["cache_read"] + c["cache_write"]


def group(calls, run_meta=None):
    runs = {}
    for c in sorted(calls, key=lambda c: (c["run"], c["seq"])):
        r = runs.setdefault(c["run"], Run(c["run"], c.get("session"), c["model"], c.get("job", "")))
        r.calls.append(c)
    for rid, meta in (run_meta or {}).items():
        if rid in runs:
            runs[rid].memory = meta.get("memory")
            runs[rid].memory_measured = meta.get("measured")
    return list(runs.values())


# ---------------------------------------------------------------- prices
def learn_prices(calls, model):
    """Least-squares fit of cost = a*uncached + b*write + c*read + e*output over a model's calls that carry a billed
    cost. Returns (Profile, the largest relative residual), or None when there is too little data."""
    rows = [c for c in calls if c.get("model") == model and c.get("cost")]
    if len(rows) < 6:
        return None
    X = [[c["uncached"], c["cache_write"], c["cache_read"], c["output"]] for c in rows]
    y = [c["cost"] for c in rows]
    cols = [j for j in range(4) if any(x[j] for x in X)]           # drop classes this model never used
    A = [[sum(x[i] * x[j] for x in X) for j in cols] for i in cols]
    b = [sum(x[i] * t for x, t in zip(X, y)) for i in cols]
    sol = _solve(A, b)
    if sol is None:
        return None
    coef = dict(zip(cols, sol))
    a, wr, rd, out = (coef.get(j, 0.0) for j in range(4))
    if a <= 0:                                                       # no uncached tokens: anchor on writes at w = 1.25
        a = wr / 1.25 if wr else rd / 0.1
    worst = max(abs(sum(coef.get(j, 0) * x[j] for j in range(4)) - t) / t for x, t in zip(X, y))
    return Profile(p=a, po=out, w=wr / a if a else 1.0, r=rd / a if a else 1.0), worst


def _solve(A, b):
    n = len(A)
    M_ = [row[:] + [v] for row, v in zip(A, b)]
    for i in range(n):
        piv = max(range(i, n), key=lambda k: abs(M_[k][i]))
        if abs(M_[piv][i]) < 1e-30:
            return None
        M_[i], M_[piv] = M_[piv], M_[i]
        for k in range(n):
            if k != i:
                f = M_[k][i] / M_[i][i]
                M_[k] = [x - f * y for x, y in zip(M_[k], M_[i])]
    return [M_[i][n] / M_[i][i] for i in range(n)]


def profiles_for(calls, prices: Optional[dict] = None, *, today: Optional[date] = None,
                 use_list: bool = True) -> tuple[dict[str, Profile], dict[str, str], list[str]]:
    """A Profile per model the calls name: the consumer's row, else the list price, else learned from billed cost.
    Returns (profiles, the source of each, warnings). A model with no source has no profile: tokens only."""
    out, src, notes = {}, {}, []
    used_list = False
    for model in sorted({c["model"] for c in calls}):
        if prices and model in prices:
            out[model], src[model] = profile_from_prices(prices[model]), "consumer's table"
        elif use_list and model in LIST_PRICES:
            out[model], src[model] = LIST_PRICES[model], f"list prices as of {LIST_PRICES_AS_OF}"
            used_list = True
        else:
            got = learn_prices(calls, model)
            if got:
                pf, worst = got
                out[model] = pf
                src[model] = f"learned from {sum(1 for c in calls if c['model'] == model and c.get('cost'))} billed calls (worst residual {worst:.1e})"
            else:
                src[model] = "none: tokens only"
    stale = list_prices_stale(today) if used_list else None
    if stale:
        warnings.warn(stale)
        notes.append(stale)
    return out, src, notes


# ---------------------------------------------------------------- shape
def run_shape(r: Run, pf: Profile):
    """What one run's money went on, from its calls alone."""
    out = defaultdict(float)
    X1 = r.X(0)
    prev_cached = None
    base = pf
    for i, c in enumerate(r.calls):
        pf = base.at(_prompt(c))                                     # each call on its own card
        wl = c.get("cache_write_long", 0)
        cost = pf.bill(c["uncached"], c["cache_write"], c["cache_read"], c["output"], wl)
        write_cost = pf.bill(write=c["cache_write"], write_long=wl, prompt=_prompt(c))
        out["cost"] += cost
        out["output"] += pf.po * c["output"]
        out["reasoning"] += pf.po * (c.get("reasoning") or 0)
        out["uncached"] += pf.p * c["uncached"]
        if i == 0:
            out["start_write"] += write_cost                            # what the run had to (re)write to begin
            out["start_read"] += pf.p * pf.r * c["cache_read"]
        else:
            lost = max(0, prev_cached - c["cache_read"])              # cached before, not read now: a break
            out["break_tokens"] += lost
            out["break_cost"] += pf.p * (pf.w - pf.r) * min(lost, c["cache_write"])
            carried_read = min(c["cache_read"], X1)
            out["rent_start"] += pf.p * pf.r * carried_read            # re-reading what the run started with
            out["rent_added"] += pf.p * pf.r * (c["cache_read"] - carried_read)
            out["write_added"] += write_cost
        prev_cached = c["cache_read"] + c["cache_write"]
    out["n"] = r.n
    out["X1"] = X1
    out["d"] = (r.X(r.n - 1) - X1) / (r.n - 1) if r.n > 1 else 0.0
    out["o"] = sum(c["output"] for c in r.calls) / r.n
    out["warm"] = r.calls[0]["cache_read"] / X1 if X1 else 0.0
    return out


def closed_form(pf: Profile, n, P, d, o, warm=0.0):
    """Steady-growth run cost: the start context P (a fraction `warm` of it already cached), d added per call. On a
    model priced by prompt length, at the card of the run's mean context (a closed form cannot switch cards mid-run)."""
    n = max(1.0, n)
    pf = pf.at(P + d * (n - 1) / 2)
    start = pf.p * P * (warm * pf.r + (1 - warm) * pf.w)
    return (start + pf.p * pf.r * P * (n - 1) + pf.p * d * (pf.w * (n - 1) + pf.r * (n - 1) * (n - 2) / 2)
            + pf.po * n * o)


def price_tags(runs, profiles, memory_known=True):
    """What a 10% change in each input does to the bill, run by run on its own fitted shape. Runs whose model has no
    profile are left out (they are sized in tokens only)."""
    base, tags = 0.0, defaultdict(float)
    for r in runs:
        pf = profiles.get(r.model)
        if pf is None:
            continue
        s = run_shape(r, pf)
        n, P, d, o, wm = s["n"], s["X1"], max(0.0, s["d"]), s["o"], s["warm"]
        b = closed_form(pf, n, P, d, o, wm)
        base += b
        tags["10% fewer calls"] += closed_form(pf, 0.9 * n, P, d, o, wm) - b
        tags["10% smaller start context"] += closed_form(pf, n, 0.9 * P, d, o, wm) - b
        tags["10% less added per call"] += closed_form(pf, n, P, 0.9 * d, o, wm) - b
        tags["10% less output"] += closed_form(pf, n, P, max(0.0, d - 0.1 * o), 0.9 * o, wm) - b
        if memory_known and r.memory:
            tags["1k fewer memory tokens served"] += closed_form(pf, n, max(0.0, P - 1000), d, o, wm) - b
    return ({k: v / base for k, v in tags.items()} if base else {}), base


def job_shapes(runs) -> dict[str, dict]:
    """The shape per job, in tokens: runs, calls per run, start context, growth per call, output per call, the share
    of the start already cached, and served memory where it is known."""
    by = defaultdict(list)
    for r in runs:
        by[r.job].append(r)
    out = {}
    for job, rs in sorted(by.items()):
        ns = [r.n for r in rs]
        X1 = [r.X(0) for r in rs]
        d = [(r.X(r.n - 1) - r.X(0)) / (r.n - 1) for r in rs if r.n > 1]
        o = [sum(c["output"] for c in r.calls) / r.n for r in rs]
        reas = sum(c.get("reasoning") or 0 for r in rs for c in r.calls)
        outp = sum(c["output"] for r in rs for c in r.calls)
        mem = [r.memory for r in rs if r.memory]
        out[job] = {"runs": len(rs), "calls_median": median(ns), "calls_max": max(ns), "start_context_median": median(X1),
                    "growth_per_call_median": median(d) if d else 0.0, "output_per_call_median": median(o),
                    "reasoning_share_of_output": (reas / outp) if outp else 0.0,
                    "warm_start_share": sum(1 for r in rs if r.calls[0]["cache_read"] > 0) / len(rs),
                    "memory_served_median": median(mem) if mem else None,
                    "memory_measured": all(r.memory_measured for r in rs if r.memory) if mem else None}
    return out


# ---------------------------------------------------------------- safe levers
def session_gaps(runs):
    """Gap before each run: its first call's start minus the start of the previous run's last request in the same
    session. A cache entry's lifetime runs from the request that wrote or read it, not from its reply."""
    by = defaultdict(list)
    for r in runs:
        if r.session:
            by[r.session].append(r)
    gaps = []
    for rs in by.values():
        rs.sort(key=lambda r: r.calls[0]["start"])
        for a, b in zip(rs, rs[1:]):
            gaps.append((b.calls[0]["start"] - a.calls[-1]["start"], a, b))
    return gaps


def cache_lifetime(runs, profiles):
    """Would a long-lived cache entry on each run's final context pay, given the measured gaps? Exact under the
    profile: the next run reads the carried context instead of rewriting it; the previous run's last write pays the
    long-lifetime price."""
    save = cost = 0.0
    k = 0
    for gap, a, b in session_gaps(runs):
        pf = profiles.get(b.model)
        if pf is None or pf.w_long is None or not (pf.short_ttl < gap < pf.long_ttl) or a.model != b.model:
            continue
        carried = min(b.calls[0]["cache_write"], a.X(a.n - 1) + a.calls[-1]["output"])
        if carried <= 0:
            continue
        pb, pa = pf.at(_prompt(b.calls[0])), pf.at(_prompt(a.calls[-1]))   # each request on its own card
        k += 1
        save += pb.p * (pb.w - pb.r) * carried
        cost += pa.p * (pa.wl - pa.w) * a.calls[-1]["cache_write"]
    return dict(runs=k, save=save, cost=cost, net=save - cost)


def remaining_calls(runs):
    """E[calls left | the run has reached call k], from the measured run lengths: what a fold must beat."""
    ns = sorted(r.n for r in runs)

    def expect(k):
        tail = [n - k for n in ns if n > k]
        return sum(tail) / len(tail) if tail else 0.0
    return expect


def fold_pays(pf: Profile, removed, rewritten, summary_cost, remaining):
    """A fold pays when the rent it saves over the remaining calls beats its rewrite premium and its summary.
    Returns (pays, m*), m* the remaining calls at which it breaks even."""
    m_star = (pf.p * (pf.w - pf.r) * rewritten + summary_cost) / (pf.p * pf.r * removed) if removed else math.inf
    return remaining >= m_star, m_star


# ---------------------------------------------------------------- from Recant's log
def _ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _call(c: dict, rid: str, seq: int, log_run: str) -> dict:
    u = c.get("usage") or {}
    lat = (c.get("latency_ms") or 0) / 1000
    return dict(run=rid, seq=seq, session=c.get("session") or log_run, start=_ts(c["ts"]) - lat, latency=lat,
                model=c["model_id"], job=c["job"], uncached=u.get("input", 0), cache_write=u.get("cache_write", 0),
                cache_write_long=u.get("cache_write_long", 0), cache_read=u.get("cache_read", 0),
                output=u.get("output", 0), reasoning=u.get("reasoning", 0), cost=c.get("billed_usd"),
                batch=c.get("batch_id") is not None, segments=c.get("segments"), stop_reason=c.get("stop_reason"),
                setting_fp=c.get("setting_fp") or _old_fp(c), extends_previous=c.get("extends_previous"),
                setting_rest=c.get("setting_fp_rest"), sys_prefix=c.get("system_prefix"), sys_suffix=c.get("system_suffix"),
                error=c.get("error"), ttft=None if c.get("ttft_ms") is None else c["ttft_ms"] / 1000,
                overhead=None if c.get("overhead_ms") is None else c["overhead_ms"] / 1000, bench_task=c.get("bench_task"))


def calls_from_records(records: Iterable[dict], *, split_on: Optional[str] = None, jobs: Optional[set] = None,
                       label: str = "", failed: Optional[list] = None, managed: Optional[list] = None) -> tuple[list[dict], dict]:
    """costshape's calls and run metadata from one log run's records (dicts as read). With `split_on`, a record of
    that kind closes a host run (a consumer whose log run holds several tasks names its own end-of-task kind); the
    `served` record in a host run gives its memory. `jobs` keeps only calls of those jobs. A call that failed is never
    one of the calls the shape is read from; with `failed` given, it goes there, with whatever usage it reported (a
    stream dropped mid-body is billed for what it streamed). A tool's own totals (a record with `via`: Claude Code,
    the Agent SDK) are never part of the shape either, since no lever can change that traffic;
    with `managed` given, they go there."""
    calls, meta = [], {}
    k, cur, errs, served = 0, [], [], None
    recs = sorted(records, key=lambda r: r.get("seq", 0))
    log_run = recs[0]["run"] if recs else label

    def close():
        nonlocal k, cur, errs, served
        rid = f"{log_run}:{k}" if split_on else log_run
        if failed is not None:
            failed.extend(_call(c, rid, -1, log_run) for c in errs)
        if cur:
            calls.extend(_call(c, rid, i, log_run) for i, c in enumerate(cur))
            if served:
                meta[rid] = {"memory": (served.get("core_tokens") or 0) + (served.get("slice_tokens") or 0) or None,
                             "measured": served.get("measured"), "core": served.get("core_tokens"), "slice": served.get("slice_tokens")}
        k += 1                                    # the k-th host run keeps index k, calls or none
        cur, errs, served = [], [], None
    for r in recs:
        if r["kind"] == "model_call" and r.get("via"):
            if managed is not None:
                managed.append(dict(_call(r, log_run, -1, log_run), via=r["via"], billing=r.get("billing"),
                                    calls=r.get("calls"), tool_cost=r.get("cost_usd")))
        elif r["kind"] == "model_call" and (jobs is None or r["job"] in jobs):
            (errs if r.get("error") else cur).append(r)
        elif r["kind"] == "served":
            served = r
        elif split_on and r["kind"] == split_on:
            close()
    close()
    return calls, meta


# ---------------------------------------------------------------- the whole picture
def shape(calls: list[dict], *, prices: Optional[dict] = None, run_meta: Optional[dict] = None,
          background_jobs: Iterable[str] = (), today: Optional[date] = None, use_list: bool = True) -> dict:
    """Everything costshape derives from metered calls, as data: the bill split, the shape per job, price tags,
    cache lifetime from the measured gaps, the fold rule's inputs, the batch lever, the prices used and their
    sources. Tokens throughout; dollars where a model has a profile."""
    runs = [r for r in group(calls, run_meta) if r.n]
    profiles, sources, notes = profiles_for(calls, prices, today=today, use_list=use_list)
    priced = [r for r in runs if r.model in profiles]
    tot = defaultdict(float)
    for r in priced:
        for k, v in run_shape(r, profiles[r.model]).items():
            if k not in ("n", "X1", "d", "o", "warm"):
                tot[k] += v
    C = tot["cost"]
    unpriced = [r for r in runs if r.model not in profiles]
    tags, _ = price_tags(priced, profiles, memory_known=any(r.memory for r in priced))
    gaps = [g for g, _, _ in session_gaps(runs)]
    expect = remaining_calls(runs) if runs else (lambda k: 0.0)
    bg = sum(run_shape(r, profiles[r.model])["cost"] for r in priced if r.job in set(background_jobs))
    batch_factor = min((profiles[r.model].batch for r in priced if r.job in set(background_jobs)), default=1.0)
    already_batched = sum(1 for r in priced for c in r.calls if r.job in set(background_jobs) and c.get("batch"))
    return {
        "runs": len(runs), "calls": sum(r.n for r in runs),
        "bill_usd": C,
        "split": {k: (tot[k] / C if C else 0.0) for k in ("output", "reasoning", "start_write", "start_read", "rent_start",
                                                       "write_added", "rent_added", "uncached", "break_cost")},
        "break_tokens": tot["break_tokens"],
        "jobs": job_shapes(runs),
        "price_tags": tags,
        "gaps": {"n": len(gaps), "under_5_min": sum(g < 300 for g in gaps), "5_to_60_min": sum(300 <= g < 3600 for g in gaps),
                 "over_60_min": sum(g >= 3600 for g in gaps)},
        "cache_lifetime": {**cache_lifetime(priced, profiles),                    # change_to_bill: negative is a saving
                           "change_to_bill": (-cache_lifetime(priced, profiles)["net"] / C) if C else 0.0},
        "remaining_calls_at": {k: expect(k) for k in (5, 10, 25, 50, 100)},
        "batch": {"background_jobs": sorted(background_jobs), "cost_usd": bg, "already_batched_calls": already_batched,
                  "change_to_bill_at_most": (-bg * (1 - batch_factor) / C) if C else 0.0},
        "prices": {m: {"source": sources[m], **({"input_per_mtok": profiles[m].p / M, "output_per_mtok": profiles[m].po / M,
                                                  "write_x": profiles[m].w, "read_x": profiles[m].r,
                                                  "row": price_row(profiles[m])} if m in profiles else {})}
                   for m in sources},
        "unpriced": {"runs": len(unpriced), "tokens": sum(r.X(i) + r.calls[i]["output"] for r in unpriced for i in range(r.n))},
        "warnings": notes,
    }


def render(s: dict, title: str = "host") -> str:
    """The shape as text."""
    sp = s["split"]
    lines = [f"# {title}: {s['runs']} runs, {s['calls']} calls, ${s['bill_usd']:.2f}"]
    for w in s["warnings"]:
        lines.append(f"warning: {w}")
    if not s["runs"]:
        return "\n".join(lines + ["no metered calls in this window"])
    lines.append(f"bill: output {sp['output']:.0%} (reasoning {sp['reasoning']:.0%} of the bill), start writes {sp['start_write']:.0%}, "
                 f"rent on the start context {sp['rent_start'] + sp['start_read']:.0%}, writes of added content {sp['write_added']:.0%}, "
                 f"rent on added content {sp['rent_added']:.0%}, uncached {sp['uncached']:.0%}")
    lines.append(f"cache breaks: {s['break_tokens']:,.0f} tokens rewritten ({sp['break_cost']:.1%} of the bill)")
    for job, j in s["jobs"].items():
        mem = f", memory served {j['memory_served_median']:,.0f} ({'counted' if j['memory_measured'] else 'estimated'})" \
            if j["memory_served_median"] else ""
        lines.append(f"job {job}: {j['runs']} runs, calls median {j['calls_median']:.0f} (max {j['calls_max']}), start context "
                     f"{j['start_context_median']:,.0f}, growth {j['growth_per_call_median']:,.0f}/call, output "
                     f"{j['output_per_call_median']:,.0f}/call ({j['reasoning_share_of_output']:.0%} of it reasoning){mem}")
    if s["price_tags"]:
        lines.append("price tags: " + ", ".join(f"{k} {v:+.1%}" for k, v in s["price_tags"].items()))
    g = s["gaps"]
    if g["n"]:
        cl = s["cache_lifetime"]
        lines.append(f"gaps between runs in a session: {g['n']}; <5 min {g['under_5_min']}, 5-60 min {g['5_to_60_min']}, "
                     f">60 min {g['over_60_min']}")
        lines.append(f"long cache lifetime on each run's final context: {cl['runs']} runs, saves ${cl['save']:.2f}, costs "
                     f"${cl['cost']:.2f}, net {cl['change_to_bill']:+.1%} of the bill")
    if s["batch"]["background_jobs"]:
        b = s["batch"]
        lines.append(f"batch pricing on {b['background_jobs']}: {b['change_to_bill_at_most']:+.1%} of the bill at most "
                     f"({b['already_batched_calls']} calls already batched)")
    lines.append("prices: " + "; ".join(f"{m} — {p['source']}" for m, p in s["prices"].items()))
    if s["unpriced"]["runs"]:
        lines.append(f"unpriced (tokens only): {s['unpriced']['runs']} runs, {s['unpriced']['tokens']:,} tokens")
    return "\n".join(lines)


# ======================================================================= the advisor
# One loop over Recant's own log: measure, diagnose, advise, apply the safe levers, check each against its
# prediction, revert what loses money or trips a guard. Three kinds of advice:
#   applied    — on the safe line, input unchanged (cache lifetime, breakpoints, batching background work, folding only when it pays):
#                Recant applies them itself, through one overlay, and checks them;
#   your_call  — they change what the model sees or what memory stores (budgets, reasoning, a hand-off): off until the
#                developer turns them on, each sized and shown with its evidence; quality is never Recant's estimate;
#   fix_cause  — a defect Recant can see but not fix, with its cost and the calls that show it.
# No lever moves a step to a cheaper model: none is generated.
import json as _json
from pathlib import Path as _Path

WEEK = 7 * 86400
FLOOR_SHARE = 0.01                 # a safe lever is applied only when it saves at least 1% of its job's bill
MIN_RUNS = 20                      # ... and the job has at least 20 runs in the window
CHECK_RUNS, CHECK_SECONDS = 50, WEEK
MIN_JUDGE_CALLS = 50                # under this many affected calls a lever is "too little traffic to judge"
TTL_SECONDS = {"5m": 300.0, "1h": 3600.0}


@dataclass
class HostShape:
    """How a consumer's log divides into runs and what its settings are now. Recant knows only its own record kinds;
    the consumer names the one that ends a host run, which log runs hold the agent's calls and which hold background
    work, the lifetime its stable prefix is cached at, and its served-memory budgets."""
    split_on: Optional[str] = None
    agent_jobs: tuple = ()
    agent_run_jobs: tuple = ()
    background_jobs: tuple = ()
    background_run_jobs: tuple = ()
    prefix_ttl: str = "5m"
    memory_budgets: dict = field(default_factory=dict)       # {"core": tokens, "slice": tokens}
    label: str = "host"
    carries_out: tuple = ()          # the safe levers this host reads from the overlay; the advisor applies only these
    lever_runs: dict = field(default_factory=dict)   # lever id -> {param: value} marking the background runs the lever changed
                                                     # (say, a night started as a batch only because the lever is on)
    turn_served_keys: tuple = ()     # (core key, slice key) in the end-of-run record's `served`, for runs logged before
                                     # the `served` record existed: estimates, and marked so
    fixed_reasoning: tuple = ()      # models on which reasoning cannot be turned off (effort is their control):
                                     # the reasoning lever is not offered for their calls
    known_unmetered_keys: tuple = ()  # keys whose spend outside this host's Meter is known (a bench key):
                                      # reported as known, off the health line


@dataclass
class Lever:
    id: str
    kind: str                       # applied | your_call | fix_cause
    job: str
    saves_tokens: float             # per week, always
    saves_usd: Optional[float]      # per week, at the host's prices; None when unpriced
    share: Optional[float]          # of the job's bill in the window
    safe: bool
    state: str = "available"        # applied | available | needs_evidence | reverted | finding | host_advice
    evidence: dict = field(default_factory=dict)
    action: dict = field(default_factory=dict)
    note: str = ""

    def line(self, agent: str = "agent", scale: Optional[float] = None) -> str:
        """`scale`: the factor the lever's weekly figures were scaled by; given, the line shows the dollars measured
        instead (the report does this while the traffic spans under a day)."""
        unit, k = ("", scale) if scale else ("/wk", 1.0)
        usd = f"{_usd(self.saves_usd / k)}{unit}" if self.saves_usd is not None else "unpriced"
        only = (self.action or {}).get("jobs") if isinstance(self.action, dict) else None
        whose = ("job " if len(only) == 1 else "jobs ") + " + ".join(only) if only else \
            (agent if self.job == "agent" else f"job {self.job}")
        sh = f" ({-self.share:+.1%} of {whose}'s bill)" if self.share is not None else ""
        split = self.evidence.get("by_job") if isinstance(self.evidence, dict) else None
        parts = (" [" + ", ".join(f"{j} {_usd(v / k)}{unit}" for j, v in split.items()) + "]") if split and len(split) > 1 else ""
        return f"{self.state:15} {self.id:28} {usd:>12}{sh}{parts}  {self.note}"


def traffic_span(calls: list) -> float:
    """Seconds from the first request to the last reply: the one span the report's header, its table and its lead use."""
    if not calls:
        return 0.0
    return max(c["start"] + (c.get("latency") or 0) for c in calls) - min(c["start"] for c in calls)


def _usd(v: float) -> str:
    """Dollars to the cent, with no negative zero: "$-0.00" reads as a loss that is not there."""
    return "$0.00" if abs(v) < 0.005 else f"${v:,.2f}"


@dataclass
class Report:
    host: HostShape
    since: float
    until: float
    calls: list
    meta: dict
    tools: list
    background: list
    folds: list
    shape: dict
    profiles: dict
    prices_used: dict
    changes: list = field(default_factory=list)       # [{at, jobs, what}] — setting changes, oldest first
    failed: list = field(default_factory=list)        # calls that failed, with the usage they reported
    managed: list = field(default_factory=list)       # a tool's own totals (Claude Code, the Agent SDK): metered, no levers
    unseen: list = field(default_factory=list)        # the coverage watch's counts in the window
    watched: bool = False                              # whether the watch wrote anything in the window

    def price(self, model: Optional[str], usage: dict) -> Optional[float]:
        """List-priced dollars for a usage dict, at the report's prices, else the library's list prices."""
        from .billing import model_family
        pf = self.profiles.get(model) or LIST_PRICES.get(model_family(model) or "")
        if pf is None:
            return None
        return pf.bill(usage.get("input", 0), usage.get("cache_write", 0), usage.get("cache_read", 0), usage.get("output", 0))

    def coverage(self) -> dict:
        from . import coverage as cov
        seen = list(self.calls) + list(self.failed) + [dict(c, calls=c.get("calls") or 1) for c in self.managed]
        return cov.summarize(seen, self.unseen, price=self.price)

    def window_start(self, job: str) -> tuple[float, str]:
        """Where a lever is priced from: the latest setting change to any of the jobs it covers inside the window, else
        the window's start. "agent" covers the host's agent jobs; a background run kind covers its background jobs."""
        if job == "agent":
            jobs = set(self.host.agent_jobs) | {"agent"}
        elif job in self.host.background_run_jobs:
            jobs = set(self.host.background_jobs) | {job}
        else:
            jobs = {job}
        hits = [c for c in self.changes if self.since < c["at"] <= self.until and ("*" in c["jobs"] or jobs & set(c["jobs"]))]
        return (hits[-1]["at"], hits[-1]["what"]) if hits else (self.since, "")

    def restrict(self, job: str) -> "Report":
        """This report from the latest setting change to `job` on: what that job's levers are priced on."""
        start, _ = self.window_start(job)
        if start <= self.since:
            return self
        return Report(self.host, start, self.until, [c for c in self.calls if c["start"] >= start], self.meta,
                      [t for t in self.tools if t["ts"] >= start], [b for b in self.background if b["start"] >= start],
                      self.folds, self.shape, self.profiles, self.prices_used, self.changes,
                      [c for c in self.failed if c["start"] >= start], [c for c in self.managed if c["start"] >= start],
                      [u for u in self.unseen if _ts(u["until"]) >= start], self.watched)

    @property
    def per_week(self) -> float:
        return WEEK / max(86400.0, self.until - self.since)

    def runs(self, jobs: Optional[set] = None) -> list:
        return [r for r in group([c for c in self.calls if jobs is None or c["job"] in jobs], self.meta) if r.n]

    def job_bill(self, jobs: set) -> float:
        return sum(self.cost(c) for c in self.calls if c["job"] in jobs)

    def cost(self, c) -> float:
        pf = self.profiles.get(c["model"])
        if pf is None:
            return 0.0
        return pf.bill(c["uncached"], c["cache_write"], c["cache_read"], c["output"], c.get("cache_write_long", 0)) * \
            (pf.batch if c.get("batch") else 1.0)

    def health(self) -> dict:
        """What went wrong in the window, from the records alone: model calls that failed, refusals, failed attempts
        a later call of the same run and job recovered (retries), and tool calls that failed, by tool."""
        failed = list(self.failed) + [c for c in self.calls if c.get("error")]   # failed calls carry their billed usage too
        ok_after = {(c["run"], c["job"]) for c in self.calls if not c.get("error")}
        tool_err: dict = {}
        for t in self.tools:
            if t.get("error"):
                tool_err[t["name"]] = tool_err.get(t["name"], 0) + 1
        return {"calls": len(self.calls) + len(self.failed), "failed_calls": len(failed),
                "refusals": sum(1 for c in self.calls if c.get("stop_reason") == "refusal"),
                "retried": sum(1 for c in failed if (c["run"], c["job"]) in ok_after),
                "tool_calls": len(self.tools), "failed_tool_calls": sum(tool_err.values()),
                "failed_tools": dict(sorted(tool_err.items(), key=lambda kv: -kv[1])[:5])}

    def text(self, levers: Optional[list] = None, ledger: Optional[list] = None) -> str:
        """Three sections: cost (leaks, levers, predicted against realized), quality (what memory adds or
        costs, measured), health (loops, failures, retries, unmetered spend, unexpected traffic). Each item is
        labelled automatic, opt-in or pointed out; a section with nothing to say says so."""
        days = (self.until - self.since) / 86400
        basis = self.calls or list(self.failed)             # the lead's span: the failed calls when none succeeded
        starts = [c["start"] for c in basis]
        ends = [c["start"] + (c.get("latency") or 0) for c in basis]           # to the last reply, not the last request
        span = traffic_span(basis)
        short = bool(starts) and span < 86400              # under a day of traffic: dollars as measured, no weekly rate
        out = [f"recant cost: {self.host.label}, the window {datetime.fromtimestamp(self.since, timezone.utc):%m-%d} to "
               f"{datetime.fromtimestamp(self.until, timezone.utc):%m-%d} UTC ({days:.1f} days)"
               + (f"; traffic in it from {datetime.fromtimestamp(min(starts), timezone.utc):%m-%d %H:%M} to "
                  f"{datetime.fromtimestamp(max(ends), timezone.utc):%H:%M} UTC ({_period(span)})" if starts else "")]
        from . import coverage as cov
        levers = levers or []
        led = ledger or []
        ready = {lv.id for lv in levers if eligible(self, lv, ledger_=led)[0]}
        out[1:1] = lead(self, levers, ready)             # the three lines first, before the detail
        out.append(cov.line(self.coverage(), watched=self.watched))
        by = {sec: [lv for lv in levers if section_of(lv) == sec] for sec in ("cost", "quality", "health")}

        has_default = any(c["job"] == "agent" for c in self.calls)          # the default job named when it has calls
        jobs = sorted(set(self.host.agent_jobs) - ({"agent"} if not has_default else set())) or ["agent"]
        agent = "job " + jobs[0] if len(jobs) == 1 else "jobs " + " + ".join(jobs)

        def items(ls):
            return ["  " + f"{label_of(lv):12} " + lv.line(agent, scale=self.restrict(lv.job).per_week if short else None)
                    for lv in ls]
        out.append("\n== cost ==")
        out.append(render(self.shape, title=self.host.label))
        leaks = [lv for lv in by["cost"] if lv.kind == "fix_cause"]
        out.append("leaks found:" if leaks else "leaks found: none")
        out += items(leaks)
        rest = [lv for lv in by["cost"] if lv.kind != "fix_cause"]
        out.append(("levers ($: what each lever would have saved on the calls it is priced on, as measured; a weekly "
                    "rate once the traffic spans a day):" if short else
                    f"levers ($/wk: each lever's saving on the calls it is priced on, scaled to a week; over this whole "
                    f"{days:.1f}-day window that is \u00d7{self.per_week:.2f}, and a lever priced from a later settings change is "
                    f"scaled from that shorter span, a span under a day counted as a day):") if rest else
                   "levers: none sized in this window")
        out += items(sorted(rest, key=lambda lv: ({"automatic": 0, "opt-in": 1}.get(label_of(lv), 2), -(lv.saves_usd or 0))))
        out += [f"note: {n}" for n in setting_notes(self)]
        checks = [e for e in (ledger or []) if e.get("event") == "checked"]
        if checks:
            last: dict = {}
            for e in checks:
                last[e["id"]] = e
            out.append("predicted against realized (each lever's latest check, per affected call; the same calls priced both ways):")
            out += ["  " + delivered_line(e) + (" (checked now, not recorded: `recant cost --nightly` records it, and "
                                                 "reverts a safe lever whose guard trips or that loses money)"
                                                 if e.get("live") else "")
                    for e in last.values()]
            acc = accuracy(ledger)
            if acc:
                out.append("  advisor accuracy (realized / predicted): " + ", ".join(f"{k} {v:.0%}" for k, v in acc.items()))
        else:
            out.append("predicted against realized: no applied lever has been checked yet")
        if self.managed:
            out += render_managed(self.managed, self.price)
        out.append("\n== quality ==")
        served = [m.get("memory") for m in self.meta.values() if m.get("memory")]
        out.append(f"memory served: median {int(median(served)):,} tokens a run over {len(served)} runs" if served
                   else "memory served: none in this window")
        evidence = [e for e in (ledger or []) if e.get("event") == "evidence"]
        out += items(by["quality"])
        if evidence:
            out.append("measured (evidence runs):")
            for e in evidence[-5:]:
                ev = e.get("evidence") or {}
                pw = ev.get("pairwise") or {}
                q = (f"better {pw.get('better')}, same {pw.get('same')}, worse {pw.get('worse')}, delta {pw.get('delta', 0):+.2f} "
                     f"({pw.get('interval', ('?', '?'))[0]:+.2f} to {pw.get('interval', ('?', '?'))[1]:+.2f})") if pw else \
                    json_dumps_short(ev.get("by_type"))
                out.append(f"  {e['id']:28} {q}; spent ${ev.get('spent_usd')}")
        else:
            out.append("measured: no evidence run in the ledger; what memory adds or costs is not measured here")
        out.append("\n== latency ==")
        out += render_latency(latency([c for c in self.calls if not c.get("batch")], self.background))
        out.append("\n== health ==")
        h = self.health()
        out.append(f"model calls {h['calls']}: failed {h['failed_calls']} (retried and recovered {h['retried']}), refusals "
                   f"{h['refusals']}; "
                   + (f"tool calls {h['tool_calls']}: failed {h['failed_tool_calls']}" if h["tool_calls"] else
                      "tool calls: none metered (recant.wrap sees model calls only; record tools with Meter.tool)")
                   + (" (" + ", ".join(f"{k} {v}" for k, v in h["failed_tools"].items()) + ")" if h["failed_tools"] else ""))
        out += items(by["health"]) if by["health"] else ["  no loops, capped outputs, unmetered spend or unexpected traffic found"]
        known = [lv for lv in levers if section_of(lv) == "known"]
        if known:
            out.append("known, off the health line:")
            out += ["  " + lv.note for lv in known]
        return "\n".join(out)


def render_managed(managed: list, price) -> list[str]:
    """A tool's own traffic (Claude Code, the Agent SDK), in its own rows: billed and priced at list. No lever
    touches it: the tool manages its own requests and caching, so this is metering and advice only."""
    by: dict = {}
    for c in managed:
        k = (c["via"], c["model"], c.get("billing") or "unknown")
        b = by.setdefault(k, {"runs": 0, "calls": 0, "tokens": 0, "list": 0.0, "priced": True, "billed": 0.0, "billed_known": True})
        u = {"input": c["uncached"], "output": c["output"], "cache_read": c["cache_read"], "cache_write": c["cache_write"]}
        usd = price(c["model"], u)
        if usd is None:
            usd = c.get("tool_cost")
        b["runs"] += 1
        b["calls"] += c.get("calls") or 0
        b["tokens"] += sum(u.values())
        if usd is None:
            b["priced"] = False
        else:
            b["list"] += usd
        if k[2] == "subscription":
            pass                                        # billed 0: the plan pays
        elif c.get("cost") is not None:
            b["billed"] += c["cost"]
        else:
            b["billed_known"] = False
    out = ["traffic a tool made itself (metered, not changed; no lever applies):"]
    for (via, model, billing), b in sorted(by.items()):
        billed = "$0.00 (subscription)" if billing == "subscription" else (f"${b['billed']:.2f}" if b["billed_known"] else
                                                                          "not reported (an API key: about the list price)")
        lst = f"${b['list']:.2f}" if b["priced"] else "unpriced"
        out.append(f"  {via} on {model}: {b['runs']} runs, {b['calls'] or '?'} calls, {b['tokens']:,} tokens; billed {billed}; "
                   f"at list {lst}")
    if any(k[2] == "subscription" for k in by):
        out.append("  on a subscription the point is tokens against the plan's limits, not dollars")
    return out


def _pct(xs: list, q: float) -> Optional[float]:
    """The q-th quantile, nearest rank; None for no values."""
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))] if xs else None


SLOW_CALL_S = 30.0                       # a call this long is named on the latency line, however few there are


def latency(calls: list, background: Optional[list] = None) -> dict:
    """Seconds: end-to-end per call and per task (a run's first request to its last reply), p50 and p95; time to
    first token on streamed calls; the metering wrapper's own time per call (network excluded); the write path —
    background runs, off the request's path by construction."""
    tasks: dict = {}
    for c in calls:
        end = c["start"] + (c.get("latency") or 0)
        t0, t1 = tasks.get((c["run"], c.get("session")), (c["start"], end))
        tasks[(c["run"], c.get("session"))] = (min(t0, c["start"]), max(t1, end))
    q = lambda xs: {"p50": _pct(xs, 0.5), "p95": _pct(xs, 0.95), "n": sum(x is not None for x in xs)}   # noqa: E731
    bg = [b for b in (background or []) if b.get("end") is not None]
    slow = [c for c in calls if (c.get("latency") or 0) > SLOW_CALL_S]
    call = q([c.get("latency") for c in calls])
    call.update(slow=len(slow), longest=max((c["latency"] for c in slow), default=None),   # a tail p95 can hide: a stall,
                slow_ttft=max((c.get("ttft") or 0 for c in slow), default=None))           # or the SDK retrying inside
    return {"call": call, "task": q([t1 - t0 for t0, t1 in tasks.values()]),
            "ttft": q([c.get("ttft") for c in calls]), "overhead": q([c.get("overhead") for c in calls]),
            "write_path": {"on_request_path": False, **q([b["end"] - b["start"] for b in bg])}}


def render_latency(lat: dict) -> list[str]:
    def f(d, scale=1.0, unit="s"):
        if not d["n"]:
            return "not measured"
        return f"p50 {d['p50'] * scale:,.2f} {unit}, p95 {d['p95'] * scale:,.2f} {unit} ({d['n']})"
    wp = lat["write_path"]
    c = lat["call"]
    slow = (f"; {c['slow']} calls over {SLOW_CALL_S:.0f} s, the longest {c['longest']:,.0f} s"
            + (f" with its first token at {c['slow_ttft']:,.0f} s (a stall, or a retry the SDK made inside the call)"
               if c.get("slow_ttft") and c["slow_ttft"] > SLOW_CALL_S / 2 else "") if c.get("slow") else "")
    return [f"per call: {f(lat['call'])}{slow}; per run (one run(...) block, or one conversation outside any): {f(lat['task'])}",
            f"time to first token (streamed calls): {f(lat['ttft'])}",
            f"Recant's own time per call (network excluded): {f(lat['overhead'], 1000, 'ms')}",
            "write path: " + (f"{f(wp)}, off the request's path" if wp["n"] else "none in this window")]


HEALTH = ("fix.tool_loop", "fix.output_cap.", "fix.unmetered.", "fix.metered_not_billed.", "fix.egress.", "fix.unwrapped.")


def section_of(lv) -> str:
    """Which section of the report an item belongs in: cost, quality (memory) or health."""
    if lv.id.startswith("memory."):
        return "quality"
    if getattr(lv, "state", None) == "known":
        return "known"
    if lv.id.startswith(HEALTH):
        return "health"
    return "cost"


LEAD_MIN_CALLS = 20        # below this a first report says it is too early, and sizes nothing


def _period(seconds: float) -> str:
    if seconds < 90:
        k = max(1, round(seconds))
        return f"{k} second{'s' if k != 1 else ''}"
    if seconds < 5400:
        return f"{round(seconds / 60)} minutes"
    if seconds < 2 * 86400:
        return f"{seconds / 3600:.1f} hours"
    return f"{seconds / 86400:.1f} days"


CHANGES = {                # what each safe lever does to a request, for `recant apply --dry-run`
    "breakpoint.add": ("adds a cache breakpoint (cache_control) on the system prompt's last block and on the last "
                       "message's last block; a plain-string system prompt or message becomes a one-block list holding the "
                       "same text. Nothing else in the request changes: the model is sent the same input"),
    "ttl.prefix": "sets the system prompt's cache breakpoint to the 1-hour lifetime (ttl 1h); nothing else changes",
}


def lead(rep: "Report", levers: list, ready: Optional[set] = None) -> list[str]:
    """The report's first three lines: what was spent, what the safe levers would save of it with no change to what the
    model sees, and what that rests on. In dollars on the calls measured; a weekly rate only once the traffic spans a
    day. Below LEAD_MIN_CALLS it says it is too early and sizes nothing."""
    calls = rep.calls
    n = len(calls)
    bad = list(getattr(rep, "failed", []) or [])         # failed calls are billed for the usage they reported
    if not n and not bad:
        return ["No calls metered in this window: wrap your client with recant.wrap(client) and run your app."]
    span = traffic_span(calls) if calls else traffic_span(bad)   # the header's and the table's span: the calls that
                                                                    # succeeded (all of them, failed, when none did)
    period = _period(span)
    priced = [c for c in calls if c["model"] in rep.profiles]
    bill = sum(rep.cost(c) for c in priced)
    bad_bill = sum(rep.cost(c) for c in bad if c["model"] in rep.profiles)
    bad_tokens = sum(c["uncached"] + c["cache_read"] + c["cache_write"] + c["output"] for c in bad)
    unpriced = n - len(priced)
    if not n:
        return [f"You spent ${bad_bill:,.2f} on {len(bad)} calls over {period}, and every one of them failed: see the "
                "health line below.",
                "Recant sizes savings on calls that succeeded; there are none in this window yet.",
                f"Measured on {len(bad)} failed calls; nothing is estimated."]
    out = [f"You spent ${bill + bad_bill:,.2f} on {n + len(bad)} calls over {period}"
           + (f" ({unpriced} more on models with no price, counted in tokens only)" if unpriced else "")
           + ((f", {'under $0.01' if bad_bill < 0.005 else f'${bad_bill:,.2f}'} of it on {len(bad)} that failed (on the "
               "health line)" if bad_bill > 0 else
               f", {len(bad)} of them failed, with {bad_tokens:,} tokens of usage on models with no price (on the "
               "health line)" if bad_tokens else
               f", {len(bad)} of them failed with no usage reported (on the health line)") if bad else "") + "."]
    if n < LEAD_MIN_CALLS:
        out.append(f"Too little traffic to size a saving yet: {n} calls metered, and Recant needs about {LEAD_MIN_CALLS}. "
                   "Run your app as usual and ask again.")
        out.append(f"Measured on {n} calls; nothing is estimated until there are more.")
        return out
    safe = [lv for lv in levers if label_of(lv) == "automatic" and lv.state == "available" and (lv.share or 0) > 0]   # a share
                                                                       # is the part of its job's bill the lever saves
    ready = {lv.id for lv in safe} if ready is None else ready
    on = [lv for lv in levers if label_of(lv) == "automatic" and lv.state == "applied"]
    if safe:
        save = 0.0
        for lv in safe:                       # a share is of its jobs' bill since the jobs' last setting change
            r = rep.restrict(lv.job)
            save += lv.share * r.job_bill(lever_jobs(r, lv.job, lv.action))
        weekly = sum(lv.saves_usd or 0.0 for lv in safe)   # the levers table's own weekly figures
        runs = min(evidence(rep, lever_jobs(rep, lv.job, lv.action)) for lv in safe)
        then = ("Run `recant apply`." if any(lv.id in ready for lv in safe) else
                f"`recant apply` turns it on once the job has {MIN_RUNS} runs ({runs} so far).")
        line = (f"Recant can save about ${save:,.2f} of that ({save / (bill + bad_bill):.0%}) with no change to what the model sees: "
                f"{', '.join(lv.id for lv in safe)}. {then}" if bill else
                f"Recant has safe savings to apply ({', '.join(lv.id for lv in safe)}). {then}")
        if span >= 86400 and weekly and bill:
            line += f" At this rate that is about ${weekly:,.2f} a week."
        out.append(line)
    elif on:
        out.append(f"The safe savings are on ({', '.join(lv.id for lv in on)}): `recant cost` checks each against its "
                   "prediction below, once it has 50 runs or 7 days.")
    else:
        out.append("Recant finds no safe saving to apply on this traffic; the levers below that change what the model sees "
                   "need evidence first (`recant bench`).")
    out.append(f"Measured on {n} calls over {period}{' (the ones that succeeded)' if bad else ''}; the estimate firms up "
               "with more traffic."
               + (" A saving is sized from each job's last setting change on." if any(rep.restrict(lv.job) is not rep for lv in safe) else ""))
    return out


def label_of(lv) -> str:
    """automatic: a safe lever Recant applies itself; opt-in: a your-call lever, applied only on the host's word after
    evidence; pointed out: a cause for the host to fix, advice, or a lever not worth building."""
    if lv.kind == "applied" and lv.state not in ("not_worth_building", "no_slack"):
        return "automatic"
    if lv.kind == "your_call" and lv.state != "host_advice":
        return "opt-in"
    return "pointed out"


def json_dumps_short(x) -> str:
    return _json.dumps(x, default=str)[:160] if x is not None else "no paired result"


def _since(since, now: float) -> float:
    if since is None:
        return 0.0
    if isinstance(since, (int, float)):
        return float(since)
    s = str(since)
    if s.endswith("d") and s[:-1].isdigit():
        return now - int(s[:-1]) * 86400
    return _ts(s if "T" in s else s + "T00:00:00+00:00")


def tools_from_records(records, *, split_on: Optional[str] = None) -> list[dict]:
    """Tool calls by host run, aligned with calls_from_records' run ids: name, targets, sizes, and whether the result
    was cut on its way to the model."""
    recs = sorted(records, key=lambda r: r.get("seq", 0))
    log_run = recs[0]["run"] if recs else ""
    k, out = 0, []
    for r in recs:
        if split_on and r["kind"] == split_on:
            k += 1
        elif r["kind"] == "tool_call":
            fetched, sent = r.get("bytes_fetched"), r.get("bytes_to_model")
            out.append({"run": f"{log_run}:{k}" if split_on else log_run, "ts": _ts(r["ts"]), "name": r["name"],
                        "targets": r.get("targets"), "bytes_fetched": fetched, "bytes_to_model": sent,
                        "cut": bool(fetched and sent and fetched > sent), "error": r.get("error"),
                        "args": args_hash(r.get("input"))})                # a hash, no text (the guard's own identity)
    return out


def load(layout, tenant: str, host: HostShape, *, since=None, until=None) -> tuple:
    """Calls, run metadata, tool calls, background runs and folds from Recant's log, for one window."""
    from .log import read_run
    now = datetime.now(timezone.utc).timestamp()
    t0, t1 = _since(since, now), (_since(until, now) if until else now)
    calls, meta, tools, background, folds, changes, failed = [], {}, [], [], [], [], []
    managed, unseen, watched = [], [], False
    for path in sorted(layout.log_dir(tenant).glob("*.jsonl")):
        recs = list(read_run(path, typed=False))
        for r in recs:                                   # setting changes, wherever they were logged
            if r["kind"] == "coverage":                 # the watch's counts, whatever run they were written into
                if t0 <= _ts(r["until"]) < t1 + 1:
                    watched = True
                    if r.get("requests"):
                        unseen.append(r)
            elif r["kind"] == "model_call" and r.get("via") and t0 <= _ts(r["ts"]) < t1:
                managed.append(dict(_call(r, r["run"], -1, r["run"]), via=r["via"], billing=r.get("billing"),
                                    calls=r.get("calls"), tool_cost=r.get("cost_usd")))
            elif r["kind"] == "setting_change":         # a change that names no job restarts every job
                changes.append({"at": _ts(r["ts"]), "jobs": list(r.get("jobs") or ["*"]), "what": r.get("what", "setting change")})
            elif r["kind"] == "advice" and r.get("event") in ("applied", "opted_in", "reverted"):
                lv = r.get("lever", "")
                jobs = list(host.background_jobs) if lv.startswith("batch.") else list(host.agent_jobs or ["agent"])
                changes.append({"at": _ts(r["ts"]), "jobs": jobs, "what": f"{r['event']} {lv}"})
        start = next((r for r in recs if r["kind"] == "run_started"), None)
        if start is None or not (t0 <= _ts(start["ts"]) < t1):
            continue
        job = start.get("job")
        if job in host.background_run_jobs:
            c, m = calls_from_records(recs, jobs=set(host.background_jobs) or None, failed=failed)
            seqs: dict = {}
            for x in c:                                       # one background run per job; background work has no session
                x["run"], x["session"] = f"{x['run']}:{x['job']}", None
                x["seq"] = seqs[x["run"]] = seqs.get(x["run"], -1) + 1
            fin = next((r for r in recs if r["kind"] == "run_finished"), {})
            background.append({"run": start["run"], "job": job, "params": start.get("params") or {}, "start": _ts(start["ts"]),
                               "end": _ts(fin["ts"]) if fin else None, "batch": any(x.get("batch") for x in c),
                               "promoted": fin.get("promoted"), "spent_usd": fin.get("spent_usd")})
        elif not host.agent_run_jobs or job in host.agent_run_jobs:
            c, m = calls_from_records(recs, split_on=host.split_on, jobs=set(host.agent_jobs) or None, failed=failed)
            tools += tools_from_records(recs, split_on=host.split_on)
            if host.turn_served_keys and host.split_on:
                ck, sk = host.turn_served_keys
                ends = [r for r in sorted(recs, key=lambda r: r.get("seq", 0)) if r["kind"] == host.split_on]
                for k, t in enumerate(ends):
                    rid, sv = f"{start['run']}:{k}", t.get("served") or {}
                    if rid not in m and (sv.get(ck) or sv.get(sk)):
                        m[rid] = {"memory": (sv.get(ck) or 0) + (sv.get(sk) or 0), "measured": False,
                                  "core": sv.get(ck), "slice": sv.get(sk)}
            for r in recs:
                for f in (r.get("compactions") or []) if r["kind"] == "turn" else []:
                    folds.append({**f, "log_run": start["run"], "remaining": max(0, (r.get("rounds") or 0) - (f.get("at_round") or 0))})
        else:
            continue
        if job == "host":                                 # recant.wrap's run for calls outside any client.run
            for x in c:
                x["host_run"] = True
        calls += c
        meta.update(m)
    changes += _fingerprint_changes(calls)
    return calls, meta, tools, background, folds, t0, t1, sorted(changes, key=lambda c: c["at"]), failed, managed, unseen, watched


def _old_fp(c: dict) -> Optional[str]:
    """For calls logged before the setting fingerprint: the model and the first cached system segment, else nothing."""
    segs = [s for s in (c.get("segments") or []) if s["at"] == "system.0"]
    return f"{c['model_id']}:{segs[0]['hash']}" if segs else None


def _common(a: Optional[list], b: Optional[list]) -> int:
    """The longest length, in characters, at which two calls' system-prompt hashes agree (0: none)."""
    if not a or not b:
        return 0
    hb = {k: h for k, h in b}
    return max((k for k, h in a if hb.get(k) == h), default=0)


def _lcp(members: list) -> int:
    """The leading part every one of these prompts shares, in characters (prefixes nest: the least of each one's
    share with the first)."""
    first = members[0]["sys_prefix"]
    return min([_common(first, m["sys_prefix"]) for m in members[1:]] or [first[-1][0]])


def _conversations(calls: list) -> dict:
    """The conversation each call belongs to (by id), for counting where a prompt opens. A session the host gave is
    one, namespaced by its run when recant.wrap made it up (auto:, one per wrapper, which may repeat in another); with
    no session given (it is then the run's id), each fresh request starts a conversation and its later turns, which
    extend it, belong to the same one."""
    out, chain = {}, {}
    for c in sorted(calls, key=lambda c: c["start"]):
        run = c["run"].split(":")[0]
        s = c.get("session")
        if s and s != run:
            out[id(c)] = (run, s) if str(s).startswith("auto:") else s
            continue
        key = (c["job"], run)
        if not c.get("extends_previous") or key not in chain:
            chain[key] = ("request", run, c["seq"], c["start"])
        out[id(c)] = chain[key]
    return out


def stable_starts(calls: list) -> dict[str, dict]:
    """Per job: which part of each system prompt is a setting, judged the way the cache reads it (0.4.1). The cache
    keeps a prefix per model, at least the model's minimum long, so prompts are compared per (job, model): two distinct
    prompts are one fixed part when they share at least that minimum from the start (a family). A prompt's head, what
    a setting change is judged on, is:

    - in a family: what the whole family shares from the start (a per-request name or brief after a fixed opening
      stays out of it);
    - a prompt of its own that opens in two or more conversations (sent fresh, or switched to, in each): all of it (a
      fixed prompt);
    - a prompt of its own that opens in one conversation only: nothing (content: a per-request part, at the top or
      after a fixed part too short to cache, or a one-off), never a change by itself (0.4.2).

    In a family whose prompts each open two or more requests (no per-request member), each whole prompt is also a
    variant, and a new variant is a change: instructions added to or cut from a fixed prompt. A family with a
    per-request member is a template, and its variants do not count. Only the window's calls are read: the same calls
    give the same heads. `heads` maps each call (by id) to (head length, head hash, variant or ""); `content` lists the
    calls whose prompt is content only. Jobs whose calls carry no hashes are left out."""
    by: dict = {}
    old_opens: dict = {}                                     # calls logged before 0.4.1: their openings, by whole fingerprint
    convs = _conversations(calls)
    first_use, last_in = set(), {}                          # a call that opens its prompt: a fresh request, or one whose
    for c in sorted(calls, key=lambda c: c["start"]):       # conversation switched to this prompt (from another, or from
        conv = (c["job"], convs[id(c)])                     # none): a growing history does not mean the system prompt
        key = c["sys_prefix"][-1][1] if c.get("sys_prefix") else None   # stayed the same
        if not c.get("extends_previous") or last_in.get(conv, "unset") != key:
            first_use.add(id(c))
        last_in[conv] = key
    for c in sorted(calls, key=lambda c: c["start"]):
        if c.get("sys_prefix") and c.get("setting_rest"):
            by.setdefault((c["job"], c["model"]), []).append(c)
        elif c.get("setting_fp") and not c.get("extends_previous"):
            old_opens.setdefault((c["job"], c["setting_fp"]), set()).add(convs[id(c)])
    out: dict = {}
    for (job, model), cs in by.items():
        floor = cache_min(model) * CHARS_PER_TOKEN
        at_floor = -(-int(floor) // 256) * 256              # the first hashed boundary at or past the minimum
        prompts: dict = {}                                    # a prompt's full hash -> its calls, in order
        for c in cs:
            prompts.setdefault(c["sys_prefix"][-1][1], []).append(c)
        opens = {k: len({convs[id(c)] for c in pcs if id(c) in first_use}                # the conversations it opens in,
                        | set().union(*[old_opens.get((job, fp), set()) for fp in {c.get("setting_fp") for c in pcs}]))
                 for k, pcs in prompts.items()}                                            # its calls before 0.4.1 too
        family: dict = {}                                     # the hash at the minimum -> the prompts that share it
        for key, pcs in prompts.items():
            h = dict((k, v) for k, v in pcs[0]["sys_prefix"]).get(at_floor)
            if h is not None:
                family.setdefault(h, []).append(key)
        fam_of = {k: f for f, ks in family.items() for k in ks if len(ks) >= 2}
        fam_head = {f: _lcp([prompts[k][0] for k in ks]) for f, ks in family.items() if len(ks) >= 2}
        fixed_family = {f for f, ks in family.items() if len(ks) >= 2 and all(opens[k] >= 2 for k in ks)}
        templated = any(len(ks) >= 2 and f not in fixed_family for f, ks in family.items())   # a family with a
                                                                                               # per-request member
        st = out.setdefault(job, {"heads": {}, "content": [], "calls": 0, "prompts": 0, "floor_chars": {}, "opening": set(),
                                  "templated": False})
        st["templated"] = st["templated"] or templated
        st["opening"] |= {id(c) for c in cs if id(c) in first_use}
        st.setdefault("conv", {}).update({id(c): convs[id(c)] for c in cs})
        for key, pcs in prompts.items():
            pre = dict((k, v) for k, v in pcs[0]["sys_prefix"])
            full = pcs[0]["sys_prefix"][-1][0]
            if key in fam_of:
                head = fam_head[fam_of[key]]                  # the family's shared start
                variant = key if fam_of[key] in fixed_family else ""
            elif opens[key] >= 2:
                head, variant = full, ""                      # a fixed prompt of its own
            else:
                head, variant = 0, ""
            for c in pcs:
                st["heads"][id(c)] = (head, pre.get(head, "") if head else "", variant)
                if not head:
                    st["content"].append(c)
        st["calls"] += len(cs)
        st["prompts"] += len(prompts)
        st["floor_chars"][model] = floor
    for job, st in out.items():
        latest = max((c for c in calls if c["job"] == job), key=lambda c: c["start"])
        st["latest"] = latest
        st["start"] = st["heads"].get(id(latest), (0,))[0]
        st["model"] = latest["model"]
        st["full"] = latest["sys_prefix"][-1][0] if latest.get("sys_prefix") else 0
    return out


def setting_key(c: dict, starts: dict) -> Optional[str]:
    """What a setting change is judged on: the settings apart from the system prompt, and the system prompt's head
    (`stable_starts`; 0.4.1). A call logged without the hashes keeps its whole fingerprint; a call whose prompt is
    content only has no key, and is never a change."""
    st = starts.get(c["job"])
    if st is None or not c.get("setting_rest"):
        return c.get("setting_fp")
    if not c.get("sys_prefix"):                           # this call sent no system prompt
        return f"{c['setting_rest']}:none"
    head, h, _ = st["heads"].get(id(c), (None, None, None))
    if head is None:
        return c.get("setting_fp")
    return f"{c['setting_rest']}:{head}:{h}" if head else f"{c['setting_rest']}:content"


def _fingerprint_changes(calls: list) -> list[dict]:
    """A job sending a request with settings it has never sent before (model, effort, output cap, tools, the system
    prompt's stable start) is a setting change for that job alone. Going back to settings it already uses is not: a
    chat that sometimes answers on another model does not restart its window each time. A per-request name or brief in
    the system prompt, after its fixed opening, is content, not a setting: the prompt is judged by the part that stays
    the same (`stable_starts`)."""
    out, seen, new_format, variants, rests, heads = [], {}, {}, {}, {}, {}
    starts = stable_starts(calls)
    for c in sorted(calls, key=lambda c: c["start"]):
        fp = setting_key(c, starts)
        if not fp:
            continue
        job = c["job"]
        st = starts.get(job)
        fmt = bool(c.get("setting_rest")) and fp != c.get("setting_fp")   # keyed by parts (0.4.1), or whole
        head = fp.split(":", 1)[1] if fmt else None                       # the system prompt's part of the key
        variant = st["heads"].get(id(c), (0, "", ""))[2] if st else ""
        if fmt and new_format.get(job) is False:          # the first call logged with 0.4.1's hashes, after calls
            rests.setdefault(job, set()).add(c["setting_rest"])   # without: a new way of keying, not a new setting
            heads.setdefault(job, set()).add(head)        # content too: a kind, as below
            if variant:
                variants.setdefault((job, head), set()).add(variant)
            new_format[job] = True
            continue
        new_format[job] = fmt
        if not fmt:                                       # a call logged before 0.4.1: its whole fingerprint, as before
            known = seen.setdefault(job, set())
            if known and fp not in known:
                out.append({"at": c["start"], "jobs": [job], "what": f"{job} sends new settings"})
            known.add(fp)
            continue
        changed = False                                   # each part judged on its own: the other settings, the system
        rs = rests.setdefault(job, set())                 # prompt's fixed part, its fixed variant
        changed = bool(rs) and c["setting_rest"] not in rs
        rs.add(c["setting_rest"])
        hs = heads.setdefault(job, set())
        if head == "content":                             # a prompt that is content only is never a change by itself,
            hs.add(head)                                  # but a different kind after it (none, a fixed part) is
        else:
            changed = changed or (bool(hs) and head not in hs)
            hs.add(head)
            if variant:
                vs = variants.setdefault((job, head), set())
                changed = changed or (bool(vs) and variant not in vs)
                vs.add(variant)
        if changed:
            out.append({"at": c["start"], "jobs": [job], "what": f"{job} sends new settings"})
    return out


def report(layout, tenant: str, host: Optional[HostShape] = None, *, since="7d", until=None, prices: Optional[dict] = None,
           today=None, use_list: bool = True) -> Report:
    """Read Recant's log for one window and derive the cost shape (bill split, shapes per job, price tags, breaks)."""
    host = host or HostShape()
    calls, meta, tools, background, folds, t0, t1, changes, failed, managed, unseen, watched = load(layout, tenant, host, since=since, until=until)
    s = shape(calls, prices=prices, run_meta=meta, background_jobs=host.background_jobs, today=today, use_list=use_list) \
        if calls else {"runs": 0, "calls": 0, "bill_usd": 0.0, "split": {}, "jobs": {}, "price_tags": {}, "warnings": [],
                       "prices": {}, "gaps": {"n": 0}, "unpriced": {"runs": 0}}
    profiles, _, _ = profiles_for(calls, prices, today=today, use_list=use_list) if calls else ({}, {}, [])
    return Report(host, t0, t1, calls, meta, tools, background, folds, s, profiles, s.get("prices", {}), changes, failed,
                  managed, unseen, watched)


# ---------------------------------------------------------------- the levers
def _agent(rep: Report) -> set:
    return set(rep.host.agent_jobs) or {c["job"] for c in rep.calls if c.get("session")}


def _prefix_key(c) -> str:
    segs = [s for s in (c.get("segments") or []) if s["at"].startswith("system.")]
    return f"{c['model']}:{segs[-1]['hash']}" if segs else f"{c['model']}:session:{c.get('session')}"


def _prefix_starts(rep: Report) -> list[dict]:
    """Each agent run's start: the gap since its prefix was last used (by segment hash; by session where there are
    no hashes yet), and the prefix's size, from the smallest warm read of that prefix (the shared head when none)."""
    calls = sorted([c for c in rep.calls if c["job"] in _agent(rep)], key=lambda c: c["start"])
    last, warm, starts, seen = {}, {}, [], set()
    for c in calls:
        key = _prefix_key(c)
        if c["cache_read"] > 0:
            warm[key] = min(warm.get(key, 10 ** 12), c["cache_read"])
    head = min([c["cache_read"] for c in calls if c["cache_read"] > 0] or [0])
    for c in calls:
        key = _prefix_key(c)
        if c["run"] not in seen:
            seen.add(c["run"])
            m = rep.meta.get(c["run"]) or {}
            # the prefix is at least the shared head plus the core served; a smaller warm read means the core missed
            size = max(warm.get(key) or 0, head + (m.get("core") or 0))
            starts.append({"run": c["run"], "model": c["model"], "gap": c["start"] - last[key] if key in last else math.inf,
                           "prefix": size, "session": c.get("session"), "X1": c["uncached"] + c["cache_read"] + c["cache_write"],
                           "slice": m.get("slice") or 0, "start": c["start"]})
        last[key] = c["start"]                  # the lifetime runs from the request's start
    return starts


def _ttl_cost(rep: Report, starts: list, ttl: str) -> float:
    out = 0.0
    for s in starts:
        pf = rep.profiles.get(s["model"])
        if pf is None:
            continue
        pf = pf.at(s["X1"])
        w = pf.w_long if (ttl == "1h" and pf.w_long is not None) else pf.w
        out += pf.p * s["prefix"] * (pf.r if s["gap"] <= TTL_SECONDS[ttl] else w)
    return out


def lever_ttl_prefix(rep: Report) -> Optional[Lever]:
    starts = _prefix_starts(rep)
    if not starts:
        return None
    cur = rep.host.prefix_ttl
    alt = "5m" if cur == "1h" else "1h"
    c_cur, c_alt = _ttl_cost(rep, starts, cur), _ttl_cost(rep, starts, alt)
    bill = rep.job_bill(_agent(rep))
    gaps = [s["gap"] for s in starts]
    saves = (c_cur - c_alt) * rep.per_week
    tokens = sum(s["prefix"] for s in starts if (s["gap"] <= TTL_SECONDS[alt]) != (s["gap"] <= TTL_SECONDS[cur])) * rep.per_week
    return Lever("ttl.prefix", "applied", "agent", tokens, saves, (c_cur - c_alt) / bill if bill else None, True,
                 action={"ttl": alt, "was": cur},
                 evidence={"starts": len(starts), "within_5m": sum(g <= 300 for g in gaps), "5_to_60m": sum(300 < g <= 3600 for g in gaps),
                           "over_60m_or_new": sum(g > 3600 for g in gaps), "cost_now": round(c_cur, 4), "cost_alt": round(c_alt, 4)},
                 note=f"prefix lifetime {cur} -> {alt} on {len(starts)} run starts")


def _history(rep: Report, s: dict) -> float:
    """Carried history at a run's first call: its context less the stable prefix, the slice and the
    message (the message is not metered apart; it is left in)."""
    return max(0.0, s["X1"] - s["prefix"] - s["slice"])


def _resumed(rep: Report, starts: list) -> list[tuple[dict, dict, float]]:
    """(previous run start, this run start, gap since the previous run's last request started) for runs that follow
    one in a session (the cache lifetime runs from the request, not its reply)."""
    by_run = {}
    for c in rep.calls:
        by_run.setdefault(c["run"], []).append(c)
    ends = {r: max(x["start"] for x in cs) for r, cs in by_run.items()}
    out, prev = [], {}
    for s in sorted(starts, key=lambda s: s["start"]):
        if s["session"] and s["session"] in prev:
            p = prev[s["session"]]
            out.append((p, s, s["start"] - ends[p["run"]]))
        if s["session"]:
            prev[s["session"]] = s
    return out


def _shared_head(calls: list[dict], firsts: list[dict], ttl: float) -> dict[int, float]:
    """The system block a run's first call would read from the cache that other runs wrote: by id of
    each first call, the tokens shared by every first call on the same model and setting (the smallest such context —
    the system block plus the shortest opening message, so an upper bound by that message), when a call on that
    setting has finished before this one starts and the block was last used within its lifetime. Runs started
    together find it cold: none of them has written it yet."""
    head: dict = {}
    for c in firsts:
        k = (c["model"], c.get("setting_fp"))
        head[k] = min(head.get(k, math.inf), c["uncached"] + c["cache_read"] + c["cache_write"])
    by_key: dict = {}
    for c in calls:
        by_key.setdefault((c["model"], c.get("setting_fp")), []).append(c)
    out = {}
    for c in firsts:
        k = (c["model"], c.get("setting_fp"))
        if k[1] is None:
            continue
        order = (c["start"], c["run"], c["seq"])         # the log's millisecond ties broken by the log itself: run ids are
        others = [d for d in by_key[k] if d is not c and (d["start"], d["run"], d["seq"]) < order]   # ULIDs, seq orders a run
        done = any(d["start"] + (d.get("latency") or 0) <= c["start"] for d in others)
        if done and c["start"] - max(d["start"] for d in others) <= ttl:
            out[id(c)] = head[k]
    return out


def lever_breakpoint_add(rep: Report) -> Optional[Lever]:
    """Cache breakpoints where a host sets none: calls with no cache activity whose request only appends to
    the previous one in its session would read the previous context at the cache-read rate instead of full price, and
    pay the write premium on what is new. A run's first call reads the cached system block other runs on the same
    setting wrote, when one has finished within the block's lifetime (`_shared_head`; leaving it out under-predicted
    two measured runs by 23–24%; see the CHANGELOG, 0.2.0). Changes nothing the model sees. Sized only where
    `extends_previous` is metered; a later call that does not extend its predecessor is priced as written in full."""
    by_run: dict = {}                   # per run and session: one log run may carry concurrent sessions
    for c in rep.calls:
        if c["job"] in _agent(rep):
            by_run.setdefault((c["run"], c.get("session")), []).append(c)
    groups = []
    for cs_ in by_run.values():
        cs_.sort(key=lambda c: c["seq"])
        if not any(c["cache_read"] or c["cache_write"] for c in cs_):
            groups.append(cs_)
    ttl = TTL_SECONDS.get(rep.host.prefix_ttl, TTL_SECONDS["5m"])
    head = _shared_head([c for g in groups for c in g], [g[0] for g in groups], ttl)
    save = cost = tokens = 0.0
    seen = 0
    net_by_job: dict = {}                    # the net saving per job, so a host with two agents sees each one's part
    for cs_ in groups:
        prev_x = 0                                   # the previous call's prefix, as written to the cache (0: not)
        for i, c in enumerate(cs_):
            pf = rep.profiles.get(c["model"])
            if pf is None:
                continue
            x = c["uncached"] + c["cache_read"] + c["cache_write"]
            pf = pf.at(x)
            small = x < cache_min(c["model"])        # under the model's minimum: the marker caches nothing
            reused = 0 if small else (prev_x if c.get("extends_previous") else
                                      (min(x, head.get(id(c), 0)) if i == 0 else 0))
            if reused and reused < cache_min(c["model"]):
                reused = 0
            gain, premium = pf.p * (1 - pf.r) * reused, 0 if small else pf.p * (pf.w - 1) * max(0, x - reused)
            save += gain
            cost += premium
            net_by_job[c["job"]] = net_by_job.get(c["job"], 0.0) + gain - premium
            tokens += reused
            seen += 1
            prev_x = 0 if small else x
    if not seen:
        return None
    bill = rep.job_bill(_agent(rep))
    pays = sorted(j for j, v in net_by_job.items() if v > 0)
    net = save - cost
    action = {"cache": "moving breakpoint on the last message, the system block cached"}
    if pays and len(pays) < len(net_by_job):          # a job where the write premium is more than the reads: left out
        net = sum(net_by_job[j] for j in pays)
        action["jobs"] = pays
        bill = rep.job_bill(set(pays))                # its share is of the bill of the jobs it is applied to
    return Lever("breakpoint.add", "applied", "agent", tokens * rep.per_week, net * rep.per_week, net / bill if bill else None, True,
                 action=action,
                 evidence={"calls": seen, "reused_tokens": tokens, "first_calls_warm": len(head), "save": round(save, 4),
                           "write_premium": round(cost, 4),
                           "by_job": {j: round(v * rep.per_week, 4) for j, v in sorted(net_by_job.items())}},
                 note=f"no cache breakpoints on {seen} calls whose requests only append")


def lever_postreply_write(rep: Report) -> Optional[Lever]:
    """A cache-only write after each reply, holding the prefix and history the next turn
    re-sends, at the long lifetime. Saves (w - r) on that history for every run that comes back within the hour;
    costs the long write of each run's history and a read of the prefix, on every run, whether one comes back or not."""
    starts = _prefix_starts(rep)
    res = _resumed(rep, starts)
    if not res:
        return None
    save = cost = tokens = 0.0
    nxt = {p["run"]: (s, g) for p, s, g in res}
    for s in starts:
        pf = rep.profiles.get(s["model"])
        if pf is None or pf.w_long is None:
            continue
        pf = pf.at(s["X1"])
        follow = nxt.get(s["run"])
        h_after = _history(rep, follow[0]) if follow else _history(rep, s)
        cost += pf.p * (pf.wl * h_after + pf.r * s["prefix"])
        if follow and follow[1] <= 3600:
            h = _history(rep, follow[0])
            save += pf.p * pf.w * h                     # the next run no longer writes it ...
            cost += pf.p * pf.r * h                     # ... it reads it
            tokens += h
    bill = rep.job_bill(_agent(rep))
    net = (save - cost) * rep.per_week
    lv = Lever("postreply.write", "applied", "agent", tokens * rep.per_week, net, (save - cost) / bill if bill else None, True,
               state="available" if bill and (save - cost) / bill >= 0.02 else "not_worth_building",
               evidence={"runs": len(starts), "resumed": len(res), "resumed_within_1h": sum(1 for _, _, g in res if g <= 3600),
                         "save": round(save, 4), "cost": round(cost, 4)},
               note="a cache-only write after each reply; built only above 2% of the bill")
    return lv


BATCH_LATENCY_PRIOR_S = 4 * 3600   # with no batch run of its own in the log, a batch is assumed to take up to this long
                                   # (gaps vary: minutes apart at night, hours apart by day)


def lever_batch(rep: Report) -> list[Lever]:
    """Batch a background job only where it has the slack: the time from a run's start to the next agent
    run, which needs its output. A run is worth batching at C·(β − P_late), β the discount and P_late the share of the
    host's own batch latencies longer than that slack (with none, the prior: late unless the slack is past it). A job
    whose runs have no slack (the next session starts as the run ends) is pointed out as such and never applied."""
    out = []
    agent_starts = sorted({c["start"] for c in rep.calls if c["job"] in _agent(rep) and c["seq"] == 0})
    for job in sorted({b["job"] for b in rep.background}):
        bruns = [b for b in rep.background if b["job"] == job]
        lat = sorted(b["end"] - b["start"] for b in bruns if b.get("batch") and b.get("end"))
        run_ids = {b["run"] for b in bruns}
        direct = [c for c in rep.calls if c["run"].split(":")[0] in run_ids and not c.get("batch")]
        if not direct:
            continue
        factor = min((rep.profiles[c["model"]].batch for c in direct if c["model"] in rep.profiles), default=1.0)
        beta = 1 - factor
        bill = sum(rep.cost(c) for c in rep.calls if c["run"].split(":")[0] in run_ids)
        saves = gross = 0.0
        slacks, late_runs = [], 0
        by_run: dict = {}
        for c in direct:
            by_run.setdefault(c["run"].split(":")[0], []).append(c)
        for b in bruns:
            cs_ = by_run.get(b["run"])
            if not cs_:
                continue
            nxt = next((t for t in agent_starts if t > b["start"]), None)
            slack = (nxt - b["start"]) if nxt is not None else None
            if slack is None:                            # nothing has needed its output yet: the slack is unknown
                continue
            slacks.append(slack)
            p_late = (sum(1 for x in lat if x > slack) / len(lat)) if lat else (0.0 if slack > BATCH_LATENCY_PRIOR_S else 1.0)
            late_runs += p_late >= beta
            cost = sum(rep.cost(c) for c in cs_)
            gross += cost * beta
            saves += cost * (beta - p_late)
        if not slacks:
            continue
        n_direct = len(by_run)
        med = sorted(slacks)[len(slacks) // 2]
        evidence = {"runs": len(bruns), "direct_runs": n_direct, "median_slack_s": round(med), "batch_latencies": len(lat),
                    "runs_late": late_runs, "gross_usd": round(gross, 4)}
        if saves <= 0:
            out.append(Lever(f"batch.{job}", "applied", job, 0.0, saves * rep.per_week, saves / bill if bill else None, False,
                             state="no_slack", evidence=evidence,
                             note=f"not worth batching: the next agent run starts a median {med:,.0f}s after it, "
                                  f"less than a batch takes, so its output would be late"))
            continue
        out.append(Lever(f"batch.{job}", "applied", job, 0.0, saves * rep.per_week, saves / bill if bill else None, True,
                         action={"batch": True, "deadline_fallback": "direct"}, evidence=evidence,
                         note=f"{n_direct} of {len(bruns)} runs went direct; median slack {med / 3600:.1f}h, "
                              f"{late_runs} would have been late"))
    return out

def lever_folds(rep: Report) -> Optional[Lever]:
    if not rep.folds:
        return None
    bad, loss = 0, 0.0
    pf = next(iter(rep.profiles.values()), None)
    for f in rep.folds:
        removed = max(0, (f.get("tokens_before") or 0) - (f.get("tokens_after") or 0))
        pays, m_star = fold_pays(pf.at(f.get("tokens_before") or 0), removed, f.get("tokens_after") or 0, f.get("cost_usd") or 0.0, f.get("remaining", 0)) if pf else (True, 0)
        if not pays:
            bad += 1
            fp = pf.at(f.get("tokens_after") or 0)
            loss += fp.p * (fp.w - fp.r) * (f.get("tokens_after") or 0) + (f.get("cost_usd") or 0.0)
    # a your-call lever: folding changes what the model sees, so it is gated on quality, never applied by
    # Recant; its size is shown with the evidence it needs
    return Lever("fold.window_only", "your_call", "agent", 0.0, loss * rep.per_week, None, False, state="needs_evidence",
                 action={"fold": "window_safety_or_m_star"}, evidence={"folds": len(rep.folds), "not_paying": bad},
                 note="fold only at the window safety or when the remaining calls beat m*; it changes what the model "
                      "sees, so it needs an evidence run")


def lever_memory(rep: Report) -> list[Lever]:
    out, budgets = [], rep.host.memory_budgets or {}
    runs = [r for r in rep.runs(_agent(rep)) if r.model in rep.profiles]
    bill = rep.job_bill(_agent(rep))
    for part in ("core", "slice"):
        cur = budgets.get(part)
        if not cur or not runs:
            continue
        new = cur // 2
        delta = tok = 0.0
        for r in runs:
            served = (rep.meta.get(r.run) or {}).get(part)
            if not served:
                continue
            cut = max(0.0, served - new)
            pf = rep.profiles[r.model]
            s = run_shape(r, pf)
            n, P, d, o, w = s["n"], s["X1"], max(0.0, s["d"]), s["o"], s["warm"]
            delta += closed_form(pf, n, P, d, o, w) - closed_form(pf, n, max(0.0, P - cut), d, o, w)
            tok += cut * n
        out.append(Lever(f"memory.{part}_budget", "your_call", "agent", tok * rep.per_week, delta * rep.per_week,
                         delta / bill if bill else None, False, state="needs_evidence", action={f"{part}_tokens": new, "was": cur},
                         note=f"{part} {cur:,} -> {new:,} tokens served"))
    return out


def _cannot_turn_off() -> frozenset:
    from .levers import CANNOT_TURN_REASONING_OFF        # vendor data: models whose reasoning cannot be turned off
    return CANNOT_TURN_REASONING_OFF


def lever_reasoning(rep: Report, runs: Optional[set] = None) -> list[Lever]:
    """`runs`: only the calls of these log runs (a background job's own runs, priced on its own window)."""
    out = []
    calls = [c for c in rep.calls if runs is None or c["run"].split(":")[0] in runs]
    for job in sorted({c["job"] for c in calls}):
        cs = [c for c in calls if c["job"] == job and (c.get("reasoning") or 0) > 0 and c["model"] in rep.profiles
              and c["model"] not in rep.host.fixed_reasoning and c["model"] not in _cannot_turn_off()]
        if not cs:
            continue
        usd = sum(rep.profiles[c["model"]].at(_prompt(c)).po * c["reasoning"] for c in cs)
        bill = rep.job_bill({job})
        out.append(Lever(f"reasoning.{job}", "your_call", job, sum(c["reasoning"] for c in cs) * rep.per_week, usd * rep.per_week,
                         usd / bill if bill else None, False, state="needs_evidence", action={"reasoning": "off"},
                         note="the most turning reasoning off could save; the share it really cuts needs an on/off pair"))
    return out


def lever_effort(rep: Report) -> list[Lever]:
    """Output and effort settings: a lower effort setting on an agent job writes less. How much less, and at what
    quality, only an evidence run can say, so the lever is sized per 10% less output on the job's own runs (the price
    tag, growth included) and offered as a your-call lever that needs evidence. Not offered below the 1% floor."""
    out = []
    for job in sorted(_agent(rep)):
        runs = rep.runs({job})
        tags, base = price_tags(runs, rep.profiles, memory_known=False)
        share = -tags.get("10% less output", 0.0)
        if share < FLOOR_SHARE:
            continue
        out_tok = sum(c["output"] for c in rep.calls if c["job"] == job)
        out.append(Lever(f"effort.{job}", "your_call", job, 0.1 * out_tok * rep.per_week, share * base * rep.per_week, share, False,
                         state="needs_evidence", action={"effort": "one step lower"},
                         note=f"priced per 10% less output ({share:.1%} of the job); an evidence run measures the real cut and the quality"))
    return out


def lever_handoff(rep: Report, h: int = 300) -> Optional[Lever]:
    """A hand-off of h tokens instead of the carried history, where the history is larger than break-even."""
    starts = _prefix_starts(rep)
    res = _resumed(rep, starts)
    if not res:
        return None
    by_run = {r.run: r for r in rep.runs(_agent(rep))}
    save = tok = 0.0
    hs, above = [], 0
    for _, s, _g in res:
        pf, r = rep.profiles.get(s["model"]), by_run.get(s["run"])
        if pf is None or r is None:
            continue
        pf = pf.at(s["X1"])
        H = _history(rep, s)
        hs.append(H)
        n = r.n
        first = r.calls[0]
        s0 = pf.r if first["cache_read"] >= s["prefix"] + 0.9 * H else pf.w
        carry = pf.p * H * (s0 + pf.r * (n - 1))
        hand = pf.po * h + pf.p * h * (pf.w + pf.r * (n - 1))
        if carry > hand:
            above += 1
            save += carry - hand
            tok += (H - h) * n
    bill = rep.job_bill(_agent(rep))
    hs.sort()
    return Lever("handoff.session", "your_call", "agent", tok * rep.per_week, save * rep.per_week, save / bill if bill else None,
                 False, state="needs_evidence", action={"handoff_tokens": h},
                 evidence={"resumed": len(hs), "history_median": hs[len(hs) // 2] if hs else 0, "history_max": hs[-1] if hs else 0,
                           "above_break_even": above},
                 note=f"a {h}-token hand-off beats carrying in {above} of {len(hs)} resumed runs")


# ---------------------------------------------------------------- fix the cause
def detect_segment_changes(rep: Report) -> Optional[Lever]:
    """A cached segment whose bytes changed within a session while its entry was still alive: by segment hash where
    the calls carry them; before that, a run start that read only the shared head though its session's previous call
    was within the hour (a heuristic). Read on the agent's window since its requests last
    changed: a fix to the cause changes what the job sends, and a break from before it is not a finding."""
    rep = rep.restrict("agent")
    calls = sorted([c for c in rep.calls if c["job"] in _agent(rep) and c.get("session")], key=lambda c: c["start"])
    prev, found, usd, toks = {}, [], 0.0, 0.0
    for c in calls:
        k = (c["session"], c["model"])
        p = prev.get(k)
        if p is not None and c["start"] - p["start"] < 3600:
            a = {s["at"]: s for s in (p.get("segments") or []) if s["at"].startswith("system.")}
            b = {s["at"]: s for s in (c.get("segments") or []) if s["at"].startswith("system.")}
            moved = [at for at in sorted(set(a) & set(b)) if a[at]["hash"] != b[at]["hash"]]
            ctx = p["uncached"] + p["cache_read"] + p["cache_write"]
            by_heuristic = not a and c["cache_read"] < 5000 and ctx > 10000 and c["seq"] == 0
            if moved or by_heuristic:
                pf = rep.profiles.get(c["model"])
                cp = pf.at(_prompt(c)) if pf else None
                cost = cp.p * (cp.w - cp.r) * c["cache_write"] if cp else 0.0
                usd += cost
                toks += c["cache_write"]
                found.append({"session": c["session"], "at": moved[0] if moved else "system (no hashes)", "seconds": round(c["start"] - p["start"]),
                              "when": datetime.fromtimestamp(c["start"], timezone.utc).isoformat(timespec="seconds"),
                              "rewritten": c["cache_write"], "usd": round(cost, 4)})
        prev[k] = c
    if not found:
        return None
    bill = rep.job_bill(_agent(rep))
    return Lever("fix.segment_changed", "fix_cause", "agent", toks * rep.per_week, usd * rep.per_week, usd / bill if bill else None,
                 False, state="finding", evidence={"cases": found},
                 note=f"{len(found)} cached segments rewritten within a session, {found[0]['seconds']}s after the last use and on")


def detect_output_cap(rep: Report) -> list[Lever]:
    out = []
    for job in sorted({c["job"] for c in rep.calls}):
        r = rep.restrict(job)                       # since the job's settings last changed: a raised cap ends the finding
        cs = [c for c in r.calls if c["job"] == job]
        hit = [c for c in cs if c.get("stop_reason") == "max_tokens"]
        outs = sorted(c["output"] for c in cs)
        p999 = outs[min(len(outs) - 1, int(0.999 * len(outs)))] if outs else 0
        if hit:
            usd = sum(r.cost(c) for c in hit)
            out.append(Lever(f"fix.output_cap.{job}", "fix_cause", job, sum(c["output"] for c in hit) * r.per_week, usd * r.per_week,
                             None, False, state="finding", evidence={"stopped": len(hit), "calls": len(cs), "p999_output": p999},
                             note=(f"{len(hit)} of {len(cs)} {'call' if len(cs) == 1 else 'calls'} stopped at the output "
                                   f"cap: {'its' if len(hit) == 1 else 'their'} output was cut short, so the cap is below what "
                                   "the answers need: raise it"
                                   + (f"; {think:.0%} of the job's output was thinking, which counts against the cap (some "
                                      "models think unless the request turns it off: README, \"Thinking you didn't ask "
                                      "for\")" if (think := sum(c.get("reasoning") or 0 for c in cs) /
                                                   max(1, sum(c["output"] for c in cs))) >= 0.2 else ""))))
    return out


CHARS_PER_TOKEN = 4.0                    # a stable start is measured in characters; its tokens are estimated at this


def setting_notes(rep: Report) -> list[str]:
    """What the setting-change rule cannot see on this traffic, said where it applies (0.4.2): a job whose latest
    prompts are content (each opens in one conversation, and they share no cacheable start) has no change counted for
    them; a job whose system prompts are a template (a fixed part, then a per-request part) is judged by the fixed part
    all its prompts share, so an instruction added at the end of that part looks like more of the per-request part."""
    out = []
    for job, st in sorted(stable_starts(rep.calls).items()):
        latest = st["latest"]
        if latest.get("sys_prefix") and not st["heads"].get(id(latest), (1,))[0]:
            out.append(f"job {job}'s latest system prompts each open in one conversation only and share no cacheable "
                       "start, so they are read as content: a change to them is not counted as a setting change, so its "
                       "levers may be priced on calls from before it; the realized-against-predicted check (shown by "
                       "recant cost after 50 runs or 7 days) is the guard")
        if st.get("templated"):
            out.append(f"job {job}'s system prompts carry a per-request part after a fixed part: a change at the end of "
                       "that fixed part may not be detected as a setting change, so its levers may be priced on calls from "
                       "before it; the realized-against-predicted check (shown by recant cost after 50 runs or 7 days) is "
                       "the guard")
    return out


def detect_system_order(rep: Report) -> list[Lever]:
    """A job whose latest system prompts are content (each opens in one conversation only, and no two share the
    model's cacheable minimum from the start): no conversation can reuse another's prompt from the cache (within one,
    its later requests resend the same prompt and can). One line per job, while its latest call is such a prompt and
    there are two or more of them. When all of those prompts share an ending long enough to cache, it is sized as if
    put first, over each conversation's first use of its prompt: a read at the cache-read rate when an entry on the
    same model and settings is ready (its writing request has finished) and within the cache's lifetime of its last
    use (every request, priced or not, refreshes it), that lifetime's write premium when not; each priced on its own
    whole prompt's card, at the batch rate when batched."""
    out = []
    starts = stable_starts(rep.calls)
    ttl = TTL_SECONDS.get(rep.host.prefix_ttl, TTL_SECONDS["5m"])
    for job, st in sorted(starts.items()):
        latest = st["latest"]                              # the job's latest call: is it content from the top?
        if not latest.get("sys_prefix") or st["heads"].get(id(latest), (1,))[0]:
            continue
        cs = sorted((c for c in st["content"] if c["model"] == latest["model"]), key=lambda c: c["start"])
        firsts, opening = set(), []                          # each conversation's first use of each prompt: its later
        for c in cs:                                         # turns and requests resend the same prompt and can already
            k = (st["conv"].get(id(c)), c["sys_prefix"][-1][1])   # read it; across conversations, not
            if id(c) in st["opening"] and k not in firsts:
                firsts.add(k)
                opening.append(c)
        if len(cs) < 2 or latest["sys_prefix"][-1][0] < st["floor_chars"][latest["model"]]:
            continue
        floor = cache_min(st["model"])
        distinct = list({c["sys_prefix"][-1][1]: c for c in cs}.values())
        start = _lcp(distinct) / CHARS_PER_TOKEN
        if len(distinct) < 2 or start >= floor:              # one prompt, or prompts sharing a cacheable start: nothing
            continue                                         # to reorder
        tail = min(_common(distinct[0]["sys_suffix"], d["sys_suffix"]) for d in distinct[1:]) / CHARS_PER_TOKEN \
            if len(distinct) > 1 else 0.0
        usd = share = None
        tokens = 0.0
        if tail >= floor:
            gain, prev = 0.0, {}                           # the cache is per model and setting (tools come first)
            premium = "wl" if rep.host.prefix_ttl == "1h" else "w"
            priced = {id(c) for c in opening}
            for c in cs:                                      # every request keeps the entry warm; only each
                k = (c["model"], c.get("setting_rest"))       # conversation's first use of its prompt is priced
                st_k = prev.get(k)                            # (ready: when a write finishes; last: its latest use)
                warm = bool(st_k) and st_k["ready"] <= c["start"] and c["start"] - st_k["last"] <= ttl
                pf = rep.profiles.get(c["model"])
                if id(c) in priced and pf is not None:
                    p = pf.at(c["uncached"] + c["cache_read"] + c["cache_write"])   # the card the whole prompt pays
                    scale = p.batch if c.get("batch") else 1.0
                    if warm:
                        gain += tail * p.p * (1 - p.r) * scale
                        tokens += tail
                    else:
                        gain -= tail * p.p * (getattr(p, premium) - 1) * scale
                if warm:
                    st_k["last"] = c["start"]
                else:                                         # it writes: readable once it has finished (one already
                    ready = c["start"] + (c.get("latency") or 0)   # in flight, within the lifetime, may finish first)
                    if st_k and st_k["ready"] > c["start"] and c["start"] - st_k["last"] <= ttl:
                        ready = min(ready, st_k["ready"])
                    prev[k] = {"ready": ready, "last": c["start"]}
            if gain > 0:
                usd = gain
                bill = rep.job_bill({job})
                share = gain / bill if bill else None
        where = ("share nothing from the start" if start < 1 else f"share only their first ~{start:,.0f} tokens")
        advice = ("put the per-request part after the fixed instructions" if start < 1 else
                  "a fixed part that short is not cached across conversations; if more fixed instructions follow the "
                  "per-request part, put the per-request part after them")
        note = (f"job {job}'s system prompts {where}, under {st['model']}'s {floor:,}-token cache minimum, so no "
                f"conversation can reuse another's: {advice}"
                + (f"; the last ~{tail:,.0f} tokens are the same in all of them, and put first they would have saved "
                   f"about {_usd(usd)} over these {len(opening)} conversations" if usd else "")
                + ". Changes to these prompts are not counted as setting changes: the realized-against-predicted check "
                  "is the guard")
        out.append(Lever(f"fix.system_order.{job}", "fix_cause", job, tokens * rep.per_week,
                         None if usd is None else usd * rep.per_week, share, False, state="finding",
                         evidence={"stable_start_tokens": round(start), "shared_end_tokens": round(tail), "calls": len(cs),
                                   "conversations": len(opening), "cache_min": floor}, note=note))
    return out


def detect_duplicate_background(rep: Report) -> Optional[Lever]:
    seen, dup, reruns = {}, [], []
    for b in sorted(rep.background, key=lambda b: b["start"]):
        day = b["params"].get("day") or datetime.fromtimestamp(b["start"], timezone.utc).date().isoformat()
        key = (b["job"], day)
        if key in seen:                                     # a duplicate once an earlier run for that night wrote something;
            (dup if seen[key] else reruns).append(b)       # after runs that spent and wrote nothing, a rerun
        done = b.get("end") is not None and not ((b.get("spent_usd") or 0) > 0 and not b.get("promoted"))
        seen[key] = seen.get(key, False) or done
    if not dup:
        return None
    usd = sum(rep.cost(c) for c in rep.calls if c["run"].split(":")[0] in {b["run"] for b in dup})
    return Lever("fix.duplicate_background", "fix_cause", "background", 0.0, usd * rep.per_week, None, False, state="finding",
                 evidence={"runs": [{"job": b["job"], "day": b["params"].get("day"), "run": b["run"],
                                     "started_by": b["params"].get("started_by")} for b in dup], "reruns_after_a_failed_run": len(reruns)},
                 note=f"{len(dup)} background runs repeated a job and day already run")




def detect_tool_loop(rep: Report, n: int = TOOL_LOOP_MIN) -> Optional[Lever]:
    """The same tool call, the same arguments, `n` or more times in a run, or after its own error: the
    agent did not take the result in. Priced by the model calls that issued the repeats, each by its share of the tool
    calls it made. Fix-the-cause: the tool's result, error or description is what to change, not a setting."""
    by_run: dict = {}
    for t in rep.tools:
        if t.get("args"):
            by_run.setdefault(t["run"], []).append(t)
    calls_by_run: dict = {}
    for c in rep.calls:
        calls_by_run.setdefault(c["run"], []).append(c)
    loops, usd = [], 0.0
    for run, ts in by_run.items():
        ts.sort(key=lambda t: t["ts"])
        groups: dict = {}
        for t in ts:
            groups.setdefault((t["name"], t["args"]), []).append(t)
        cs_ = sorted(calls_by_run.get(run, []), key=lambda c: c["start"] + c["latency"])
        def issuer(t):                                  # the last model call to end before the tool ran
            prior = [c for c in cs_ if c["start"] + c["latency"] <= t["ts"] + 1e-3]
            return prior[-1] if prior else None
        issued: dict = {}
        for t in ts:
            c = issuer(t)
            if c is not None:
                issued[id(c)] = issued.get(id(c), 0) + 1
        for (name, _), g in groups.items():
            after_error = any(g[i].get("error") for i in range(len(g) - 1))
            if len(g) < n and not (after_error and len(g) >= 2):
                continue
            cost = 0.0
            for t in g[1:]:                             # the first call was the agent's choice; the repeats are the loop
                c = issuer(t)
                if c is not None:
                    cost += rep.cost(c) / issued[id(c)]
            usd += cost
            loops.append({"run": run, "tool": name, "calls": len(g), "after_error": after_error, "usd": round(cost, 4)})
    if not loops:
        return None
    loops.sort(key=lambda x: -x["usd"])
    repeats = sum(x["calls"] - 1 for x in loops)
    bill = rep.job_bill(_agent(rep))
    return Lever("fix.tool_loop", "fix_cause", "agent", 0.0, usd * rep.per_week, usd / bill if bill else None, False,
                 state="finding", evidence={"loops": loops[:20], "runs": len({x["run"] for x in loops}), "repeats": repeats},
                 note=f"{repeats} repeated tool calls in {len(loops)} loops ({loops[0]['tool']} {loops[0]['calls']}x in one run): "
                      "the agent did not take the result in")


def detect_refetch(rep: Report) -> Optional[Lever]:
    """A trimmed tool result followed, in the same run, by another call on one of its targets."""
    by_run = {}
    for t in rep.tools:
        by_run.setdefault(t["run"], []).append(t)
    cut = refetched = plain = plain_refetched = 0
    for ts in by_run.values():
        ts.sort(key=lambda t: t["ts"])
        for i, t in enumerate(ts):
            if not t.get("targets"):
                continue
            later = any(set(t["targets"]) & set(u.get("targets") or []) for u in ts[i + 1:])
            if t["cut"]:
                cut += 1
                refetched += later
            else:
                plain += 1
                plain_refetched += later
    if not cut:
        return None
    return Lever("advice.trim_tool_output", "your_call", "agent", 0.0, None, None, False, state="host_advice",
                 evidence={"cut": cut, "refetched_after_cut": refetched, "refetch_rate_cut": refetched / cut,
                           "refetch_rate_plain": plain_refetched / plain if plain else None},
                 note=f"after a cut result the agent went back to the same target {refetched / cut:.0%} of the time "
                      f"({plain_refetched / plain if plain else 0:.0%} otherwise): cut harder early in long runs, not late in short ones")


def advise(rep: Report, *, overlay: Optional[dict] = None, ledger: Optional[list] = None) -> list[Lever]:
    """Every lever and finding for this window, ranked by dollars a week. Applied levers and reverted ones carry their
    state from the overlay and the ledger."""
    overlay, ledger = overlay or {}, ledger or []
    levers: list = []
    agent = rep.restrict("agent")                      # each lever priced only since its job's last setting change
    for fn in (lever_breakpoint_add, lever_ttl_prefix, lever_postreply_write, lever_folds, lever_handoff, lever_memory):
        out = fn(agent)
        levers += [out] if isinstance(out, Lever) else (out or [])
    for job in sorted({b["job"] for b in rep.background}):
        levers += lever_batch(rep.restrict(job))
    for fn in (detect_segment_changes, detect_duplicate_background, detect_refetch, detect_tool_loop):   # findings: the whole window
        lv = fn(rep)
        if lv is not None:
            levers.append(lv)
    agent_runs = {c["run"].split(":")[0] for c in agent.calls if c["job"] in _agent(rep)}
    levers += lever_reasoning(agent, runs=agent_runs)
    levers += lever_effort(agent)
    for job in sorted({b["job"] for b in rep.background}):
        r = rep.restrict(job)
        levers += [lv for lv in lever_reasoning(r, runs={b["run"] for b in r.background if b["job"] == job})
                   if not any(x.id == lv.id for x in levers)]
    levers += detect_output_cap(rep)
    levers += detect_system_order(rep)
    bg_of = {c["job"]: b["job"] for b in rep.background for c in rep.calls if c["run"].split(":")[0] == b["run"]}
    for lv in levers:
        if lv.kind != "fix_cause" and lv.id != "advice.trim_tool_output":
            job = lv.job if lv.job == "agent" or not lv.id.startswith("reasoning.") else bg_of.get(lv.job, "agent")
            start, what = rep.window_start(job)
            lv.evidence["priced_since"] = datetime.fromtimestamp(start, timezone.utc).isoformat(timespec="minutes")
            if what:
                lv.evidence["since_change"] = what
    reverted = {e["id"] for e in ledger if e.get("event") == "reverted"}
    for lv in levers:
        if lv.id in overlay:
            lv.state = "applied" if overlay[lv.id].get("kind") == "applied" else "opted_in"
        elif lv.id in reverted and lv.state == "available":
            lv.state = "reverted"
    return sorted(levers, key=lambda lv: -(lv.saves_usd or 0.0))


# ---------------------------------------------------------------- overlay, ledger, apply, check, revert
def _dir(layout, tenant) -> _Path:
    d = layout.costshape_dir(tenant)
    d.mkdir(parents=True, exist_ok=True)
    return d


def overlay(layout, tenant: str) -> dict:
    """What Recant has changed: {lever id: entry}. A consumer reads it at the start of a run."""
    p = layout.costshape_dir(tenant) / "overlay.json"
    return _json.loads(p.read_text()) if p.exists() else {}


def ledger(layout, tenant: str) -> list[dict]:
    p = layout.costshape_dir(tenant) / "ledger.jsonl"
    return [_json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _write_overlay(layout, tenant, ov: dict) -> None:
    (_dir(layout, tenant) / "overlay.json").write_text(_json.dumps(ov, indent=1, sort_keys=True) + "\n")


def _record(layout, tenant, log, entry: dict) -> None:
    with (_dir(layout, tenant) / "ledger.jsonl").open("a") as f:
        f.write(_json.dumps(entry) + "\n")
    if log is not None:
        log.emit("advice", lever=entry["id"], lever_kind=entry["kind"], event=entry["event"], action=entry.get("action") or {},
                 predicted_usd=entry.get("predicted_usd"), predicted_tokens=entry.get("predicted_tokens"),
                 realized_usd=entry.get("realized_usd"), window_runs=entry.get("window_runs"), note=entry.get("note", ""))


def _shape_sig(rep: Report, job: str) -> str:
    rs = rep.runs(_agent(rep)) if job == "agent" else [r for r in rep.runs() if r.job == job]
    if not rs:
        return ""
    n = sorted(r.n for r in rs)[len(rs) // 2]
    x = sorted(r.X(0) for r in rs)[len(rs) // 2]
    return f"n{n.bit_length()}:x{int(x).bit_length()}"              # coarse: a shape change moves a bucket


def lever_jobs(rep: "Report", job: str, action: Optional[dict] = None) -> set:
    """The jobs a lever is sized, applied, predicted and checked on: its action's `jobs` when it is applied to some
    only (a breakpoint left off a job where it would cost), else the agent jobs for "agent", else its own job."""
    only = (action or {}).get("jobs")
    if only:
        return set(only)
    return _agent(rep) if job == "agent" else {job}


def evidence(rep: Report, jobs: set) -> int:
    """How much traffic a lever of the agent jobs rests on: each run the host opened, plus each call made outside any run
    (recant.wrap with no client.run keeps one run per client open for the life of the process, so its calls, not that
    one run, are the units of evidence)."""
    return len([r for r in rep.runs(jobs) if not r.calls[0].get("host_run")]) + \
        sum(1 for c in rep.calls if c["job"] in jobs and c.get("host_run"))


def eligible(rep: Report, lv: Lever, *, ledger_: Optional[list] = None, min_runs: int = MIN_RUNS) -> tuple[bool, str]:
    if not lv.safe or lv.kind != "applied" or lv.state != "available":
        return False, "not a safe lever that is available"
    if lv.id not in rep.host.carries_out:
        return False, "the host does not carry this lever out yet"
    if lv.share is None or lv.share < FLOOR_SHARE:
        return False, f"saves under {FLOOR_SHARE:.0%} of its job's bill"
    if lv.job == "agent":
        runs, unit = evidence(rep, lever_jobs(rep, lv.job, lv.action)), "runs"   # a call outside any client.run counts as one
    else:                                   # a nightly job runs once a night: 20 runs never fit a 7-day window, so its calls count
        bruns = {b["run"] for b in rep.background if b["job"] == lv.job}
        runs, unit = sum(1 for c in rep.calls if c["run"].split(":")[0] in bruns), "calls"
    if runs < min_runs:
        return False, f"{runs} {unit} of the job, under {min_runs}"
    for e in reversed(ledger_ or []):
        if e["id"] == lv.id and e["event"] == "reverted" and e.get("shape_sig") == _shape_sig(rep, lv.job):
            return False, "reverted before, and the job's shape has not changed"
    return True, ""


def apply(layout, tenant: str, rep: Report, lv: Lever, *, log=None, at: Optional[float] = None, min_runs: int = MIN_RUNS) -> dict:
    """`min_runs`: the nightly's guard against acting on little evidence; `recant bench` lowers it to its calibration,
    whose prediction is frozen and then judged on the paired run."""
    ok, why = eligible(rep, lv, ledger_=ledger(layout, tenant), min_runs=min_runs)
    if not ok:
        raise ValueError(f"{lv.id}: {why}")
    at = at or datetime.now(timezone.utc).timestamp()
    ov = overlay(layout, tenant)
    jobs = lever_jobs(rep, lv.job, lv.action)
    ov[lv.id] = {"kind": lv.kind, "action": lv.action, "applied_at": at, "predicted_usd": lv.saves_usd,
                 "predicted_tokens": lv.saves_tokens, "predicted_share": lv.share, "job": lv.job,
                 "assumed_calls": round(sum(1 for c in rep.calls if c["job"] in jobs) * rep.per_week)}   # the volume the prediction assumed
    _write_overlay(layout, tenant, ov)
    entry = {"id": lv.id, "kind": lv.kind, "event": "applied", "at": at, "action": lv.action, "predicted_usd": lv.saves_usd,
             "predicted_tokens": lv.saves_tokens, "note": f"{lv.note}: priced {-lv.share:+.1%} of {lv.job}; check after {CHECK_RUNS} runs or 7 days"}
    _record(layout, tenant, log, entry)
    return entry


def opt_in(layout, tenant: str, lever_id: str, *, action: dict, evidence: str, predicted_usd: Optional[float] = None,
           at: Optional[float] = None, log=None) -> dict:
    """The developer turned on a your-call lever; Recant records it, with its evidence, and checks its cost."""
    at = at or datetime.now(timezone.utc).timestamp()
    ov = overlay(layout, tenant)
    ov[lever_id] = {"kind": "your_call", "action": action, "applied_at": at, "evidence": evidence, "predicted_usd": predicted_usd}
    _write_overlay(layout, tenant, ov)
    entry = {"id": lever_id, "kind": "your_call", "event": "opted_in", "at": at, "action": action, "predicted_usd": predicted_usd,
             "note": f"opted in on: {evidence}"}
    _record(layout, tenant, log, entry)
    return entry


def revert(layout, tenant: str, lever_id: str, *, reason: str, rep: Optional[Report] = None, log=None) -> dict:
    ov = overlay(layout, tenant)
    was = ov.pop(lever_id, None)
    if was is None:
        raise ValueError(f"{lever_id} is not in the overlay")
    _write_overlay(layout, tenant, ov)
    entry = {"id": lever_id, "kind": was["kind"], "event": "reverted", "at": datetime.now(timezone.utc).timestamp(),
             "action": was["action"], "note": reason, "shape_sig": _shape_sig(rep, "agent") if rep else ""}
    _record(layout, tenant, log, entry)
    return entry


def check(lever_id: str, rep: Report, entry: dict) -> dict:
    """Did an applied lever deliver? The same calls, since it was applied, priced as measured and as they would have
    cost under the old setting (the measured gap, the direct price, the tokens served before), never before against
    after. Returns the realized saving over the window, the prediction scaled to the same window, and whether the
    window is complete."""
    since = entry["applied_at"]
    after = Report(rep.host, since, rep.until, [c for c in rep.calls if c["start"] >= since], rep.meta,
                   [t for t in rep.tools if t["ts"] >= since], [b for b in rep.background if b["start"] >= since],
                   rep.folds, rep.shape, rep.profiles, rep.prices_used)
    realized, runs, guard, calls = None, 0, None, 0
    if lever_id == "breakpoint.add":                     # each call as measured, and as it would have cost uncached
        only = set((entry.get("action") or {}).get("jobs") or [])        # the jobs it was applied to, when not all
        cs_ = [c for c in after.calls if c["job"] in _agent(after) and c["model"] in after.profiles
               and (not only or c["job"] in only)]
        runs, calls = len({(c["run"], c.get("session")) for c in cs_}), len(cs_)
        realized = sum(after.profiles[c["model"]].at(_prompt(c)).p * _prompt(c)            # the card the whole prompt
                       + after.profiles[c["model"]].at(_prompt(c)).po * c["output"] - after.cost(c) for c in cs_)   # pays
    elif lever_id == "ttl.prefix":
        starts = _prefix_starts(after)
        runs, calls = len(starts), sum(1 for c in after.calls if c["job"] in _agent(after))
        realized = _ttl_cost(after, starts, entry["action"]["was"]) - _ttl_cost(after, starts, entry["action"]["ttl"])
    elif lever_id.startswith("batch."):
        bruns = {b["run"] for b in after.background if b["job"] == lever_id.split(".", 1)[1]}
        mark = rep.host.lever_runs.get(lever_id)
        changed = {b["run"] for b in after.background if b["run"] in bruns and mark
                   and all(b["params"].get(k) == v for k, v in mark.items())} if mark else bruns
        cs = [c for c in after.calls if c["run"].split(":")[0] in changed and c.get("batch")]   # only what the lever changed
        runs, calls = len(bruns), len(cs)
        realized = sum(after.cost(c) / after.profiles[c["model"]].batch - after.cost(c) for c in cs if c["model"] in after.profiles)
        nights: dict = {}
        for b in sorted((b for b in after.background if b["run"] in bruns), key=lambda b: b["start"]):
            nights.setdefault(b["params"].get("day"), []).append(b)
        missed = sum(1 for bs in nights.values() if any(x["batch"] and x.get("end") is None for x in bs[:-1]))
        realized -= sum(after.cost(c) for c in after.calls if c["run"].split(":")[0] in bruns and not c.get("batch"))
        guard = f"{missed} batch deadlines missed" if missed else None   # a night whose batch never finished and was rerun
    elif lever_id.startswith("memory.") and entry.get("action"):
        part = lever_id.split(".")[1].split("_")[0]
        typical = entry.get("typical_before")              # the served size before the apply, saved once; else read
        if not typical:                                    # from this window's runs before it (their set built once)
            prior = {c["run"] for c in rep.calls if c["start"] < since}
            before = [m.get(part) for r, m in rep.meta.items() if m.get(part) and r in prior]
            typical = sorted(before)[len(before) // 2] if before else None   # the
        rs = [r for r in after.runs(_agent(after)) if r.model in after.profiles]   # served size before the apply, not this window
        runs, realized = len(rs), 0.0
        for r in rs:
            served = (after.meta.get(r.run) or {}).get(part)
            if served is None or typical is None:
                continue
            calls += len(r.calls)
            pf, s = after.profiles[r.model], run_shape(r, after.profiles[r.model])
            n, P, d, o, w = s["n"], s["X1"], max(0.0, s["d"]), s["o"], s["warm"]
            realized += closed_form(pf, n, P + max(0.0, typical - served), d, o, w) - closed_form(pf, n, P, d, o, w)
    span = max(1.0, after.until - since)
    predicted = (entry.get("predicted_usd") or 0.0) * span / WEEK
    job = entry.get("job")
    jobs = lever_jobs(after, job, entry.get("action")) if job == "agent" else \
        ({job} if job in after.host.background_jobs else None)
    if entry.get("predicted_share") is not None and jobs and realized is not None:
        # by the work done since, not the clock: the predicted share of what the job's calls in the window would have
        # cost under the old setting (measured plus realized). A week's prediction scaled by elapsed time misreads any
        # window whose traffic differs from the week the lever was priced on.
        predicted = entry["predicted_share"] * (after.job_bill(jobs) + realized)
    return {"id": lever_id, "runs": runs, "calls": calls, "realized_usd": None if realized is None else round(realized, 4),
            "predicted_usd": round(predicted, 4), "complete": runs >= CHECK_RUNS or span >= CHECK_SECONDS, "guard": guard}


def judge(res: dict, entry: dict, min_calls: int = MIN_JUDGE_CALLS) -> dict:
    """Like with like. A lever is judged on its saving per affected call against the prediction per call
    (the weekly prediction over the calls it assumed), never on this window's dollars against a week's; below
    `min_calls` affected calls there is too little traffic to judge. Weekly dollars are context, not the test."""
    n, assumed, week = res.get("calls") or 0, entry.get("assumed_calls"), entry.get("predicted_usd")
    out = {"calls": n, "assumed_calls": assumed, "realized_per_call": None, "predicted_per_call": None, "ratio": None}
    if res.get("realized_usd") is None:
        return {**out, "verdict": "not_measured"}
    if n < min_calls:
        return {**out, "verdict": "too_little_traffic"}
    out["realized_per_call"] = res["realized_usd"] / n
    if assumed and week is not None:
        out["predicted_per_call"] = week / assumed
        out["ratio"] = out["realized_per_call"] / out["predicted_per_call"] if out["predicted_per_call"] else None
    if out["ratio"] is None:
        return {**out, "verdict": "no_prediction_per_call"}
    return {**out, "verdict": "met" if abs(out["ratio"] - 1) <= 0.25 else "missed"}


def delivered_line(e: dict) -> str:
    """One checked lever in words, with the volume it covers."""
    n, assumed = e.get("window_calls"), e.get("assumed_calls")
    vol = f"{n if n is not None else '?'} calls this window" + (f"; the prediction assumed about {assumed:,} a week" if assumed else "")
    v = e.get("verdict")
    if v == "too_little_traffic":
        return f"{e['id']}: too little traffic to judge ({vol})"
    if v in ("met", "missed"):
        return (f"{e['id']}: {'met' if v == 'met' else 'MISSED'} — ${e['realized_per_call']:.5f} a call against "
                f"${e['predicted_per_call']:.5f} predicted ({e['ratio']:.0%}; {vol}); ${e.get('realized_usd')} this window")
    return f"{e['id']}: realized ${e.get('realized_usd')} against ${e.get('predicted_window_usd')} predicted ({vol})"


def baseline(layout, tenant: str, host: HostShape, lid: str, entry: dict, *, prices: Optional[dict] = None) -> dict:
    """What a lever's prediction assumed, from the week before it was applied: that week's calls of its job, and for a
    memory lever the median size served then. Recorded once in the overlay entry."""
    at = entry.get("applied_at")
    if at is None:
        return {}
    pre = report(layout, tenant, host, since=at - WEEK, until=at, prices=prices)
    jobs = lever_jobs(pre, entry.get("job") or "agent", entry.get("action"))
    out = {"assumed_calls": sum(1 for c in pre.calls if c["job"] in jobs)}
    if lid.startswith("memory."):
        part = lid.split(".")[1].split("_")[0]
        vals = sorted(m.get(part) for m in pre.meta.values() if m.get(part))
        if vals:
            out["typical_before"] = vals[len(vals) // 2]
    return out


def accuracy(ledger_: list) -> dict:
    """Realized against predicted, per lever type, over every completed check."""
    acc: dict = {}
    for e in ledger_:
        if e.get("event") == "checked" and e.get("realized_usd") is not None and e.get("predicted_window_usd") \
                and e.get("verdict") != "too_little_traffic":
            k = e["id"].split(".")[0]
            r, p = acc.get(k, (0.0, 0.0))
            acc[k] = (r + e["realized_usd"], p + e["predicted_window_usd"])
    return {k: r / p for k, (r, p) in acc.items() if p}


def worth_listing(lv: "Lever") -> bool:
    """An automatic lever still available that saves under half a cent in the window is noise on a small store;
    the report leaves it out unless asked for everything (`recant cost --all`)."""
    return lv.kind != "applied" or lv.state != "available" or (lv.saves_usd or 0) > 0.005


def live_checks(layout, tenant: str, host: HostShape, rep: "Report", *, prices: Optional[dict] = None) -> list[dict]:
    """Each applied lever's check as of now, for a plain `recant cost`: the same check and judgement as the nightly
    pass, once the lever's window is complete (50 runs or 7 days), computed in memory and written nowhere. Only the
    nightly pass records a check, and reverts a safe lever whose guard trips or that loses money."""
    out = []
    now = rep.until
    for lid, entry in overlay(layout, tenant).items():
        entry = dict(entry)
        if "assumed_calls" not in entry or (lid.startswith("memory.") and "typical_before" not in entry):
            entry.update(baseline(layout, tenant, host, lid, entry, prices=prices))
        res = check(lid, rep, entry)
        if not res["complete"]:
            continue
        j = judge(res, entry)
        e = {"id": lid, "kind": entry["kind"], "event": "checked", "at": now, "action": entry["action"], "live": True,
             "realized_usd": res["realized_usd"], "predicted_window_usd": res["predicted_usd"], "window_runs": res["runs"],
             "window_calls": j["calls"], "assumed_calls": j["assumed_calls"], "verdict": j["verdict"],
             "realized_per_call": j["realized_per_call"], "predicted_per_call": j["predicted_per_call"], "ratio": j["ratio"]}
        e["note"] = delivered_line(e)
        out.append(e)
    return out


def nightly(layout, tenant: str, host: HostShape, *, prices: Optional[dict] = None, log=None, on_advice=None,
            now: Optional[float] = None, apply_safe: bool = True, show_all: bool = True) -> dict:
    """The nightly pass: the last 7 days; the price table's age; check each applied lever, revert what loses money
    (realized below noise) or tripped a guard (a lever that saves less than predicted stays on, its check saying so);
    apply what is newly eligible; write the report to the store."""
    now = now or datetime.now(timezone.utc).timestamp()
    rep = report(layout, tenant, host, since=now - WEEK, until=now, prices=prices)
    events = []
    ov, led = overlay(layout, tenant), ledger(layout, tenant)
    for lid, entry in list(ov.items()):
        if "assumed_calls" not in entry or (lid.startswith("memory.") and "typical_before" not in entry):
            entry.update(baseline(layout, tenant, host, lid, entry, prices=prices))     # once: what the prediction assumed
            ov[lid] = entry
            _write_overlay(layout, tenant, ov)
        res = check(lid, rep, entry)
        if not res["complete"]:
            continue
        j = judge(res, entry)
        bill = rep.job_bill(_agent(rep)) * (now - entry["applied_at"]) / WEEK
        noise = 0.005 * bill
        e = {"id": lid, "kind": entry["kind"], "event": "checked", "at": now, "action": entry["action"],
             "realized_usd": res["realized_usd"], "predicted_window_usd": res["predicted_usd"], "window_runs": res["runs"],
             "window_calls": j["calls"], "assumed_calls": j["assumed_calls"], "verdict": j["verdict"],
             "realized_per_call": j["realized_per_call"], "predicted_per_call": j["predicted_per_call"], "ratio": j["ratio"]}
        e["note"] = delivered_line(e)
        _record(layout, tenant, log, e)
        events.append(e)
        if j["verdict"] == "too_little_traffic":
            continue                                       # never reverted, or judged, on too little traffic
        if entry["kind"] == "applied" and res.get("guard"):
            events.append(revert(layout, tenant, lid, reason=f"guard tripped: {res['guard']}", rep=rep, log=log))
        elif entry["kind"] == "applied" and res["realized_usd"] is not None and res["realized_usd"] < -noise:
            events.append(revert(layout, tenant, lid, reason=f"realized ${res['realized_usd']} over {res['runs']} runs, below noise", rep=rep, log=log))
    levers = advise(rep, overlay=overlay(layout, tenant), ledger=ledger(layout, tenant))
    if apply_safe:
        for lv in levers:
            if eligible(rep, lv, ledger_=ledger(layout, tenant))[0]:
                events.append(apply(layout, tenant, rep, lv, log=log, at=now))
                lv.state = "applied"
    bill = billing_findings(layout, tenant, since_day=datetime.fromtimestamp(now - WEEK, timezone.utc).date().isoformat(),
                            known_keys=host.known_unmetered_keys)   # None without billing access; the week only
    levers += bill or []
    levers += coverage_findings(rep)                          # traffic past Recant, on the same health line
    stale = list_prices_stale(datetime.fromtimestamp(now, timezone.utc).date())
    from .billing import NOT_VISIBLE
    shown = levers if show_all else [lv for lv in levers if worth_listing(lv)]    # the stored JSON keeps every lever
    text_ = rep.text(shown, ledger(layout, tenant)) + (f"\nwarning: {stale}" if stale else "") + ("" if bill is not None else f"\n{NOT_VISIBLE}")
    d = _dir(layout, tenant) / "reports"
    d.mkdir(exist_ok=True)
    stamp = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d")
    (d / f"{stamp}.txt").write_text(text_ + "\n")
    (d / f"{stamp}.json").write_text(_json.dumps({"shape": rep.shape, "levers": [vars(lv) for lv in levers], "events": events},
                                                  indent=1, default=str) + "\n")
    out = {"report": text_, "levers": levers, "events": events, "path": str(d / f"{stamp}.txt"), "coverage": rep.coverage(),
           "watched": rep.watched}
    if on_advice is not None:
        on_advice(out)
    return out


def qac(cost_usd: float, correct: int, asked: int) -> dict:
    """Dollars per correct answer, and per question."""
    return {"usd_per_correct": cost_usd / correct if correct else math.inf, "usd_per_question": cost_usd / asked if asked else math.inf,
            "correct": correct, "asked": asked}


# ---------------------------------------------------------------- evidence runs
def _sign_p(b: int, w: int) -> float:
    n, k = b + w, min(b, w)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def paired(a: dict, b: dict) -> dict:
    """Item-by-item comparison of two runs of the same items, per item type: {type: n, a_correct, b_correct, better,
    worse, delta (b - a, a share), its 95% interval, the sign test's p}. `a` and `b` map item id to (type, correct)."""
    out = {}
    for t in sorted({v[0] for v in a.values()}):
        ids = [i for i in a if a[i][0] == t and i in b]
        n = len(ids)
        if not n:
            continue
        better = sum(1 for i in ids if b[i][1] and not a[i][1])
        worse = sum(1 for i in ids if a[i][1] and not b[i][1])
        d = (better - worse) / n
        se = math.sqrt(max(0.0, (better + worse) - (better - worse) ** 2 / n)) / n
        out[t] = {"n": n, "a_correct": sum(1 for i in ids if a[i][1]), "b_correct": sum(1 for i in ids if b[i][1]),
                  "better": better, "worse": worse, "delta": d, "interval": (d - 1.96 * se, d + 1.96 * se), "p": _sign_p(better, worse)}
    return out


def evaluate(lever_id: str, *, current: dict, candidate: dict, ceiling_usd: float, estimate_usd: float,
             harness=None, baseline: Optional[dict] = None, replay=None, judge=None, runs: Optional[list] = None,
             sample: Optional[int] = None, saving: Optional[Lever] = None, say=print, layout=None, tenant=None, log=None) -> dict:
    """An evidence run for a your-call lever: quality at the lever's setting against the current one, priced,
    with an interval — never Recant's estimate. The estimate is said before anything runs; a run whose estimate is over
    the ceiling does not start, and one that reaches it stops. Two ways to get the quality:
      harness(setting) -> ({item: (type, correct)}, cost_usd): the host's own harness, run at both settings (or at the
          candidate only, given `baseline`); items paired one by one;
      replay(run, setting) -> (answer, cost_usd) and judge(answer_a, answer_b) -> ("a"|"b"|"same", cost_usd): a sample
          of the host's runs replayed both ways and judged blind against each other — never against what was sent.
          Runs that used tools cannot be replayed faithfully: they are left out and counted, with their share.
    Report-only: nothing it learns changes a setting. The result goes in the ledger as the lever's evidence."""
    say(f"evidence run for {lever_id}: {candidate} against {current}; estimated ${estimate_usd:.2f}, ceiling ${ceiling_usd:.2f}")
    if estimate_usd > ceiling_usd:
        return {"id": lever_id, "started": False, "why": f"estimated ${estimate_usd:.2f} is over the ${ceiling_usd:.2f} ceiling"}
    spent, out = 0.0, {"id": lever_id, "started": True, "current": current, "candidate": candidate}
    if harness is not None:
        if baseline is None:
            baseline, c = harness(current)
            spent += c
        if spent >= ceiling_usd:
            out.update(stopped=f"the ceiling was reached at ${spent:.2f}, before the candidate ran")
        else:
            cand, c = harness(candidate)
            spent += c
            out["by_type"] = paired(baseline, cand)
    elif replay is not None and judge is not None:
        pool = list(runs or [])
        with_tools = [r for r in pool if r.get("used_tools")]
        pool = [r for r in pool if not r.get("used_tools")][: sample or None]
        tally = {"better": 0, "same": 0, "worse": 0}
        import random as _random
        rng = _random.Random(7)
        for r in pool:
            if spent >= ceiling_usd:
                out["stopped"] = f"the ceiling was reached at ${spent:.2f} after {sum(tally.values())} runs"
                break
            a, ca = replay(r, current)
            b, cb = replay(r, candidate)
            flip = rng.random() < 0.5                    # blind: the judge never knows which setting wrote which
            verdict, cj = judge(b, a) if flip else judge(a, b)
            spent += ca + cb + cj
            if verdict == "same":
                tally["same"] += 1
            else:
                cand_won = (verdict == "a") if flip else (verdict == "b")
                tally["better" if cand_won else "worse"] += 1
        n = sum(tally.values())
        d = (tally["better"] - tally["worse"]) / n if n else 0.0
        se = math.sqrt(max(0.0, tally["better"] + tally["worse"] - (tally["better"] - tally["worse"]) ** 2 / n)) / n if n else 0.0
        out["pairwise"] = {**tally, "n": n, "delta": d, "interval": (d - 1.96 * se, d + 1.96 * se),
                           "p": _sign_p(tally["better"], tally["worse"]), "left_out_with_tools": len(with_tools),
                           "share_with_tools": len(with_tools) / (len(with_tools) + len(pool)) if (with_tools or pool) else 0.0}
    else:
        raise ValueError("an evidence run needs a harness, or a replay and a judge")
    out["spent_usd"] = round(spent, 4)
    if saving is not None:
        out["saving"] = {"saves_usd_week": saving.saves_usd, "saves_tokens_week": saving.saves_tokens, "share": saving.share}
    if layout is not None and tenant is not None:
        _record(layout, tenant, log, {"id": lever_id, "kind": "your_call", "event": "evidence", "at": datetime.now(timezone.utc).timestamp(),
                                      "action": {"current": current, "candidate": candidate}, "evidence": out,
                                      "note": "evidence run (report-only)"})
    return out


# ---------------------------------------------------------------- the bill
def billing_findings(layout, tenant: str, *, since_day: Optional[str] = None, known_keys: tuple = ()) -> Optional[list]:
    """Fix-the-cause items from the latest reconciliation against the provider's bill (`reconcile.json`, written by the
    consumer's nightly pass): traffic billed on a key but not metered, and the reverse, each said in words with its days,
    models and size. `since_day` limits them to the days of a window (a weekly digest's week), so a finding
    from weeks ago does not read as this week's. A key in `known_keys` (a bench key, whose runs meter into stores this
    host does not search) is reported as known, off the health line. None when there is no billing access."""
    p = layout.costshape_dir(tenant) / "reconcile.json"
    if not p.exists():
        return None
    rec = _json.loads(p.read_text())
    if not rec.get("rows"):                                       # an older file: its per-key summary, all days
        return _billing_findings_summary(rec)
    rows = [r for r in rec["rows"] if since_day is None or r.get("day", "") >= since_day]
    days_seen = len({r["day"] for r in rows}) or 1
    by: dict = {}
    for r in rows:
        k = by.setdefault(r["key"], {"u_usd": 0.0, "u_tok": 0, "u_days": set(), "u_models": set(),
                                     "m_usd": 0.0, "m_tok": 0, "m_calls": 0, "m_days": set(), "m_models": set()})
        if r["status"] == "unmetered" and (r.get("unmetered_tokens") or 0):
            k["u_usd"] += r.get("unmetered_usd") or 0.0
            k["u_tok"] += r.get("unmetered_tokens") or 0
            k["u_days"].add(r["day"])
            k["u_models"].add(r.get("model"))
        elif r["status"] == "metered_not_billed":
            k["m_usd"] += r.get("metered_not_billed_usd") or 0.0
            k["m_tok"] += r.get("metered_not_billed_tokens") or 0
            k["m_calls"] += r.get("calls_metered") or 0
            k["m_days"].add(r["day"])
            k["m_models"].add(r.get("model"))
    out = []
    span = lambda d: min(d)[5:] + (f" to {max(d)[5:]}" if len(d) > 1 else "")      # noqa: E731
    models = lambda ms: ", ".join(sorted(m for m in ms if m))                       # noqa: E731
    for key, k in sorted(by.items()):
        if k["u_tok"]:
            known = key in known_keys
            note = (f"billed on the {key} key but in no store this host meters: ${k['u_usd']:.2f} on {span(k['u_days'])} "
                    f"({models(k['u_models'])}) — "
                    + ("known: runs on this key meter into stores outside this host's search, or call outside the Meter"
                       if known else "a script that called the API outside the Meter, or a run against a copy of the store"))
            out.append(Lever(f"fix.unmetered.{key}", "fix_cause", key, k["u_tok"] * 7 / days_seen, k["u_usd"] * 7 / days_seen,
                             None, False, state="known" if known else "finding",
                             evidence={"days": sorted(k["u_days"]), "usd": k["u_usd"], "tokens": k["u_tok"]}, note=note))
        if k["m_tok"]:
            out.append(Lever(f"fix.metered_not_billed.{key}", "fix_cause", key, 0.0, None, None, False, state="finding",
                             evidence={"days": sorted(k["m_days"]), "tokens": k["m_tok"], "calls": k["m_calls"], "usd": k["m_usd"]},
                             note=(f"metered as the {key} key, but the provider billed none of it on that key: {k['m_calls']} calls, "
                                   f"{k['m_tok']:,} tokens on {span(k['m_days'])} ({models(k['m_models'])}) — usually calls "
                                   "metered without their key label that really ran on another key")))
    return out


def coverage_findings(rep: "Report") -> list:
    """`fix.unwrapped.<provider>`: model calls the coverage watch saw go past Recant to a known provider in the
    window, per week, priced at list where the response reported its usage. An unknown host (`other`) is counted on the
    Coverage line, not here: there is no provider to name."""
    s = rep.coverage()
    per_week = rep.per_week
    by: dict = {}
    for u in s["unseen"]:
        if u["provider"] in ("other", "none"):
            continue
        b = by.setdefault(u["provider"], {"calls": 0, "usd": 0.0, "priced": True, "hosts": set(), "models": set()})
        b["calls"] += u["requests"]
        b["hosts"].add(u["host"])
        b["models"].add(u.get("model") or "model not named")
        if u.get("usd_list") is None or u.get("with_usage", 0) < u["requests"]:
            b["priced"] = False
        b["usd"] += u.get("usd_list") or 0.0
    out = []
    for p, b in sorted(by.items(), key=lambda kv: -kv[1]["calls"]):
        usd = b["usd"] * per_week if b["priced"] else None
        cost = f", about ${usd:.2f} at list" if usd is not None else (f", at least ${b['usd'] * per_week:.2f} at list" if b["usd"] else "")
        note = (f"{b['calls'] * per_week:.0f} calls/wk to {', '.join(sorted(b['hosts']))} ({', '.join(sorted(b['models']))}) went "
                f"past Recant{cost}: a client the wrapper does not hold, or a provider it has no wrapper for")
        out.append(Lever(f"fix.unwrapped.{p}", "fix_cause", p, 0.0, usd, None, False, state="finding",
                         evidence={"calls": b["calls"], "usd_list": b["usd"] if b["priced"] else None}, note=note))
    return out


def _billing_findings_summary(rec: dict) -> list:
    days = max(1, rec.get("days_compared", 1))
    out = []
    for key, s in sorted(rec.get("keys", {}).items()):
        if s.get("unmetered_tokens"):
            usd = s.get("unmetered_usd")
            out.append(Lever(f"fix.unmetered.{key}", "fix_cause", key, s["unmetered_tokens"] * 7 / days,
                             (usd * 7 / days) if usd is not None else None, None, False, state="finding",
                             evidence={"days": s.get("days_unmetered"), "usd": usd, "tokens": s["unmetered_tokens"]},
                             note=f"billed on the {key} key but not metered: ${usd or 0:.2f} over {len(s.get('days_unmetered') or [])} days"))
        if s.get("metered_not_billed_tokens"):
            out.append(Lever(f"fix.metered_not_billed.{key}", "fix_cause", key, 0.0, None, None, False, state="finding",
                             evidence={"tokens": s["metered_not_billed_tokens"], "usd": s.get("metered_not_billed_usd")},
                             note="metered, but the provider billed none of it on that key"))
    return out
