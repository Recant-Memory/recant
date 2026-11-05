"""The cost check: does serving memory cut what a task costs?. The consumer runs the same tasks with
memory served (arm `on`) and without (arm `off`); this module turns each task run's model calls into one record,
summarizes each arm with intervals, applies the kernel's gate, and writes the rows in a flat export shape. It runs
no task and restricts nothing: the runner, the switch and the tasks are the consumer's.

Why calls and carried history, not input size: on a long agent run the cost is history re-read on every call, and a
cut that breaks the prompt cache raises cost. So the gate asks for a lower cost per successful task *and* fewer
calls, with success not lower and the cache-hit ratio not more than `cache_hit_drop_max_pp` lower. A run's cost
includes its share of the write path (`write_path_usd`, the extraction of its day); the cost without it is shown beside.

Definitions. A task run's tokens are summed over its `model_call` records. `cache_hit_ratio` = cache_read /
(input + cache_read + cache_write), the share of prompt tokens served from cache. `cost_per_success` = the arm's
total cost / its successful runs. Intervals are percentile bootstraps over task runs, resampled within each arm,
seeded, so the same runs give the same report."""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

TOKEN_KEYS = ("input", "cache_read", "cache_write", "output")


@dataclass(frozen=True)
class TaskRun:
    task: str
    arm: str
    rep: int
    run_id: str
    started: str                      # ISO timestamp
    model_id: str
    calls: int
    turns: int                        # messages the task sent the agent
    input: int                        # uncached input tokens
    cache_read: int
    cache_write: int
    output: int
    cost_usd: Optional[float]         # None when a call had no price
    cost_split: dict = field(default_factory=dict)    # usd by token kind, when priced
    duration_ms: int = 0
    outcome: bool = False             # the task's own check
    stale_uses: int = 0               # served entries superseded or contested at serve time
    error: Optional[str] = None
    finished: str = ""                # ISO timestamp
    chat_id: str = ""                 # the conversation the run was: a session id
    write_path_usd: float = 0.0       # its share of the extraction that turned its day into memory; counted in its cost

    @property
    def total_usd(self) -> Optional[float]:
        return None if self.cost_usd is None else self.cost_usd + self.write_path_usd

    @property
    def prompt_tokens(self) -> int:
        return self.input + self.cache_read + self.cache_write


def task_run(model_calls: Iterable[dict], *, task: str, arm: str, rep: int, run_id: str, started: str, turns: int,
             duration_ms: int, outcome: bool, stale_uses: int = 0, prices: Optional[dict] = None,
             error: Optional[str] = None, finished: str = "", chat_id: str = "") -> TaskRun:
    """One task run from its `model_call` log records (dicts as read from the log). `prices` ({model_id: Prices})
    splits the cost by token kind; without it the records' own `cost_usd` is summed and the split is empty."""
    calls = [c for c in model_calls if c.get("kind", "model_call") == "model_call"]
    tok = {k: sum((c.get("usage") or {}).get(k, 0) for c in calls) for k in TOKEN_KEYS}
    split: dict[str, float] = {}
    if prices:
        for c in calls:
            p, u = prices.get(c["model_id"]), c.get("usage") or {}
            if p is None:
                split = {}
                break
            rate = {"input": p.input, "output": p.output,
                    "cache_read": p.cache_read if p.cache_read is not None else p.input,
                    "cache_write": p.cache_write if p.cache_write is not None else p.input}
            long_ = min(u.get("cache_write_long", 0), u.get("cache_write", 0))      # long-lifetime writes at their own rate
            long_rate = p.cache_write_long if p.cache_write_long is not None else rate["cache_write"]
            for k in TOKEN_KEYS:
                n = u.get(k, 0) - (long_ if k == "cache_write" else 0)
                split[k] = split.get(k, 0.0) + n * rate[k] / 1_000_000
            split["cache_write"] = split.get("cache_write", 0.0) + long_ * long_rate / 1_000_000
    costs = [c.get("cost_usd") for c in calls]
    cost = round(sum(split.values()), 6) if split else (None if any(x is None for x in costs) else round(sum(costs), 6))
    models = sorted({c["model_id"] for c in calls})
    return TaskRun(task=task, arm=arm, rep=rep, run_id=run_id, started=started, model_id=",".join(models) or "none",
                   calls=len(calls), turns=turns, cost_usd=cost, cost_split={k: round(v, 6) for k, v in split.items()},
                   duration_ms=duration_ms, outcome=outcome, stale_uses=stale_uses, error=error, finished=finished,
                   chat_id=chat_id, **tok)


# --- per arm -----------------------------------------------------------------------------------------------------
def _point(runs: list[TaskRun]) -> dict[str, Optional[float]]:
    n = len(runs)
    ok = sum(r.outcome for r in runs)
    prompt = sum(r.prompt_tokens for r in runs)
    priced = all(r.cost_usd is not None for r in runs)
    total = sum(r.total_usd for r in runs) if priced else None
    bare = sum(r.cost_usd for r in runs) if priced else None
    return {"success_rate": ok / n if n else None,
            "calls_per_task": sum(r.calls for r in runs) / n if n else None,
            "cost_per_success": (total / ok) if (total is not None and ok) else None,
            "cost_per_success_without_extraction": (bare / ok) if (bare is not None and ok) else None,
            "cache_hit_ratio": sum(r.cache_read for r in runs) / prompt if prompt else None}


def _boot(runs: list[TaskRun], b: int, rng: random.Random) -> list[dict]:
    return [_point([runs[rng.randrange(len(runs))] for _ in runs]) for _ in range(b)] if runs else []


def _pct(xs: list[float], level: float) -> Optional[list[float]]:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return [round(xs[int(lo * (len(xs) - 1))], 6), round(xs[int(hi * (len(xs) - 1))], 6)]


def summarize(runs: list[TaskRun], *, level: float = 0.95, resamples: int = 2000, seed: int = 0) -> dict[str, dict]:
    """Per arm: n, successes, the four point numbers with intervals, tokens, the cost split, stale uses."""
    out = {}
    for arm in sorted({r.arm for r in runs}):
        rs = [r for r in runs if r.arm == arm]
        point = _point(rs)
        boot = _boot(rs, resamples, random.Random(f"{seed}:{arm}"))
        split: dict[str, float] = {}
        for r in rs:
            for k, v in r.cost_split.items():
                split[k] = round(split.get(k, 0.0) + v, 6)
        out[arm] = {"n": len(rs), "successes": sum(r.outcome for r in rs),
                    **{k: (round(v, 6) if v is not None else None) for k, v in point.items()},
                    "intervals": {k: _pct([p[k] for p in boot], level) for k in point},
                    "tokens": {k: sum(getattr(r, k) for r in rs) for k in TOKEN_KEYS},
                    "cost_usd": round(sum(r.total_usd for r in rs), 6) if all(r.cost_usd is not None for r in rs) else None,
                    "write_path_usd": round(sum(r.write_path_usd for r in rs), 6),
                    "cost_split": split, "stale_uses": sum(r.stale_uses for r in rs),
                    "duration_ms_mean": round(sum(r.duration_ms for r in rs) / len(rs)) if rs else None}
    return out


# --- the gate ----------------------------------------------------------------------------------------------------
def _criterion(boot_values: list[float], ok, point: Optional[float], level: float) -> dict:
    """met: the whole interval passes; missed: the whole interval fails; unclear: it straddles (run more)."""
    iv = _pct(boot_values, level)
    if point is None or iv is None:
        return {"point": point, "interval": iv, "verdict": "unclear"}
    lo, hi = iv
    verdict = "met" if ok(lo) and ok(hi) else "missed" if not ok(lo) and not ok(hi) else "unclear"
    return {"point": round(point, 6), "interval": iv, "verdict": verdict}


def gate(runs: list[TaskRun], g, *, on: str = "on", off: str = "off") -> dict:
    """The kernel's cost-check gate over both arms. `g` is the kernel's `cost_check` block. Each criterion is judged
    on its interval at this N, not its point: pass if every criterion is met, fail if any is missed, otherwise
    run more."""
    a, b = [r for r in runs if r.arm == on], [r for r in runs if r.arm == off]
    rng_a, rng_b = random.Random(f"{g.seed}:gate:{on}"), random.Random(f"{g.seed}:gate:{off}")
    ba, bb = _boot(a, g.resamples, rng_a), _boot(b, g.resamples, rng_b)
    pa, pb = _point(a), _point(b)

    def rel_cut(x, y):
        return None if x is None or y in (None, 0) else 1 - x / y

    def diff(x, y):
        return None if x is None or y is None else x - y
    pairs = list(zip(ba, bb))
    crit = {
        "cost_per_success_cut": _criterion([rel_cut(x["cost_per_success"], y["cost_per_success"]) for x, y in pairs],
                                           lambda v: v >= g.gate.cost_cut_min, rel_cut(pa["cost_per_success"], pb["cost_per_success"]), g.level),
        "calls_per_task_diff": _criterion([diff(x["calls_per_task"], y["calls_per_task"]) for x, y in pairs],
                                          lambda v: v < 0, diff(pa["calls_per_task"], pb["calls_per_task"]), g.level),
        "success_rate_diff": _criterion([diff(x["success_rate"], y["success_rate"]) for x, y in pairs],
                                        lambda v: v >= 0, diff(pa["success_rate"], pb["success_rate"]), g.level),
        "cache_hit_diff_pp": _criterion([None if diff(x["cache_hit_ratio"], y["cache_hit_ratio"]) is None
                                         else 100 * diff(x["cache_hit_ratio"], y["cache_hit_ratio"]) for x, y in pairs],
                                        lambda v: v >= -g.gate.cache_hit_drop_max_pp,
                                        None if diff(pa["cache_hit_ratio"], pb["cache_hit_ratio"]) is None
                                        else 100 * diff(pa["cache_hit_ratio"], pb["cache_hit_ratio"]), g.level),
    }
    beside = _criterion([rel_cut(x["cost_per_success_without_extraction"], y["cost_per_success_without_extraction"]) for x, y in pairs],
                        lambda v: v >= g.gate.cost_cut_min,
                        rel_cut(pa["cost_per_success_without_extraction"], pb["cost_per_success_without_extraction"]), g.level)
    verdicts = [c["verdict"] for c in crit.values()]
    overall = "pass" if all(v == "met" for v in verdicts) else "fail" if "missed" in verdicts else "run more"
    return {"verdict": overall, "criteria": crit, "n_per_arm": {on: len(a), off: len(b)}, "level": g.level,
            "beside": {"cost_per_success_cut_without_extraction": beside},     # shown, not gated on
            "thresholds": {"cost_cut_min": g.gate.cost_cut_min, "cache_hit_drop_max_pp": g.gate.cache_hit_drop_max_pp}}


# --- the flat export ---------------------------------------------------------------------------------------------
EXPORT_FIELDS = ("ts", "run_id", "chat_id", "role", "model", "tokens_in", "tokens_out", "cache_read", "cache_write",
                 "cost_usd", "turns", "duration_ms", "outcome", "arm")


def export_row(r: TaskRun, *, role: str = "orchestrator") -> dict[str, Any]:
    """One task run in the flat per-run export shape: `ts` when it finished, `chat_id` the conversation (the task
    run's session), `tokens_in` uncached input, `turns` the internal model calls, `outcome` `done` when the task's
    own check passed and `failed` otherwise; `cost_usd` includes the run's share of extraction; `arm` added."""
    return {"ts": r.finished or r.started, "run_id": r.run_id, "chat_id": r.chat_id or r.task, "role": role, "model": r.model_id,
            "tokens_in": r.input, "tokens_out": r.output, "cache_read": r.cache_read, "cache_write": r.cache_write,
            "cost_usd": r.total_usd, "turns": r.calls, "duration_ms": r.duration_ms,
            "outcome": "done" if r.outcome else "failed", "arm": r.arm}


def write_outputs(out_dir: Path, runs: list[TaskRun], g, *, title: str, notes: Iterable[str] = (), on: str = "on",
                  off: str = "off") -> dict[str, Path]:
    """`runs.jsonl` (the export rows), `task_runs.jsonl` (the full records) and `report.md` under `out_dir`. `on` and
    `off` name the arm the gate tests and the arm it tests against."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs.jsonl").write_text("".join(json.dumps(export_row(r)) + "\n" for r in runs))
    (out_dir / "task_runs.jsonl").write_text("".join(json.dumps(asdict(r)) + "\n" for r in runs))
    summ = summarize(runs, level=g.level, resamples=g.resamples, seed=g.seed)
    verdict = gate(runs, g, on=on, off=off)
    (out_dir / "report.md").write_text(render(title, summ, verdict, notes))
    (out_dir / "summary.json").write_text(json.dumps({"summary": summ, "gate": verdict}, indent=1) + "\n")
    return {"runs": out_dir / "runs.jsonl", "task_runs": out_dir / "task_runs.jsonl", "report": out_dir / "report.md",
            "summary": out_dir / "summary.json"}


def _fmt(v, iv, pct=False, usd=False):
    if v is None:
        return "—"
    f = (lambda x: f"{100 * x:.1f}%") if pct else (lambda x: f"${x:.4f}") if usd else (lambda x: f"{x:.2f}")
    return f"{f(v)} [{f(iv[0])}, {f(iv[1])}]" if iv else f(v)


def render(title: str, summ: dict, verdict: dict, notes: Iterable[str] = ()) -> str:
    lines = [f"# {title}", "", f"Gate: **{verdict['verdict']}** (n per arm {verdict['n_per_arm']}; intervals at "
             f"{int(100 * verdict['level'])}%, bootstrap over task runs).", "",
             "| arm | n | success rate | calls per task | cost per successful task | without extraction | cache-hit ratio | stale uses |",
             "|---|---|---|---|---|---|---|---|"]
    for arm, s in summ.items():
        iv = s["intervals"]
        lines.append(f"| {arm} | {s['n']} | {_fmt(s['success_rate'], iv['success_rate'], pct=True)} | "
                     f"{_fmt(s['calls_per_task'], iv['calls_per_task'])} | {_fmt(s['cost_per_success'], iv['cost_per_success'], usd=True)} | "
                     f"{_fmt(s['cost_per_success_without_extraction'], iv['cost_per_success_without_extraction'], usd=True)} | "
                     f"{_fmt(s['cache_hit_ratio'], iv['cache_hit_ratio'], pct=True)} | {s['stale_uses']} |")
    lines += ["", "| arm | uncached input | cache read | cache write | output | cost: input / read / write / output |",
              "|---|---|---|---|---|---|"]
    for arm, s in summ.items():
        t, c = s["tokens"], s["cost_split"]
        split = " / ".join(f"${c.get(k, 0):.4f}" for k in TOKEN_KEYS) if c else "—"
        lines.append(f"| {arm} | {t['input']:,} | {t['cache_read']:,} | {t['cache_write']:,} | {t['output']:,} | {split} |")
    lines += ["", "| criterion | point | interval | verdict |", "|---|---|---|---|"]
    for name, c in verdict["criteria"].items():
        lines.append(f"| {name} | {c['point']} | {c['interval']} | {c['verdict']} |")
    for name, c in verdict.get("beside", {}).items():
        lines.append(f"| {name} (shown, not gated) | {c['point']} | {c['interval']} | {c['verdict']} |")
    notes = list(notes)
    if notes:
        lines += ["", *[f"- {n}" for n in notes]]
    return "\n".join(lines) + "\n"
