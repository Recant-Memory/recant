"""`recant bench`: Recant against no Recant, on the developer's own tasks, fair by construction.

    recant bench --tasks tasks.jsonl --run mypkg.agent:run_task [--grade mypkg.agent:grade] [--n 40] [--calibrate 10]
                 [--mode safe] [--arm-b off|safe|safe+reasoning.agent] [--out bench/<run-id>] [--yes]

The developer supplies `run_task(task, client) -> result` (sync or async; it must make every model call through
`client`) and optionally `grade(task, result) -> bool | float`. Then:

1. Price-out: the first task runs (metered), its cost is scaled to the whole run, printed, and confirmed (`--yes`).
2. Calibration: the first `--calibrate` tasks run without Recant, metered only, and are left out of the comparison.
3. Predictions: from the calibration, the levers each arm will carry out and the cost per task they predict are
   written to `predictions.json`, hashed, the hash printed. The arms do not start until it exists; a changed file fails
   the report.
4. Paired runs: every remaining task runs in both arms, the order alternating task by task (A-B, B-A, …). Arm A is off
   by default: the raw client, metered from the response's usage only, nothing from Recant on the request.
5. Cache isolation: each arm (and the calibration) has its own cache salt — a line of the same length at the very start
   of the system prompt — so no arm reads another's provider cache; every arm starts cold. The salt is the only thing
   the kit adds to an off arm's request, and the report lists it.
6. Store isolation: each arm has a fresh store in the bench directory; the developer's own store is never read
   (`--store` starts the Recant arm from a copy, and the report says so). Memory carries task to task inside an arm,
   never across arms or bench runs.

A task's pass or fail is its grader's; a task whose run raised (or whose grader raised) is failed in its arm and stays in
the comparison. A failed model call (a dropped stream included) counts its billed usage and is reported in its arm, but a
host that retries it and finishes the task is not marked failed for it. The report (`report.md`, `report.json`) carries cost, tokens by class, quality on Recant's one standard,
latency, predicted against realized lever by lever, failures per arm, setup friction, and plain verdict lines."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib
import inspect
import json
import math
import os
import random
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from . import costshape as cs, quality, switch
from .log import read_run
from .store import StoreLayout
from .version import describe, info as version_info
from .wrap import wrap

TENANT = "bench"
ARM_SPECS = ("off", "safe")                       # plus "safe+<lever>" for an opted-in your-call lever
QUARTER = 0.25                                    # a prediction is met when realized is within a quarter of it


# ---------------------------------------------------------------- the cache salt
def salt_line(run_id: str, arm: str) -> str:
    """The same length for every arm of a run: only the arm's letter differs."""
    return f"[recant-bench {run_id} arm {arm}]"


def salted(kwargs: dict, salt: str) -> dict:
    """The request with the salt ahead of everything a cached prefix can hold: the provider's prefix
    runs tools, then system, then messages, so the salt opens the first tool's description when there are tools, and
    the system prompt always. A list of system blocks keeps its block count (the salt joins the first text block), so
    cache markers stay where they were. A first tool with no description field (a server tool) gets a salt-only custom
    tool ahead of it, the same in every arm."""
    out = dict(kwargs)
    tools = out.get("tools")
    if tools:
        tools = copy.deepcopy(list(tools))
        first = tools[0] if isinstance(tools[0], dict) else None
        if first is not None and ("input_schema" in first or first.get("type") in (None, "custom")):
            first["description"] = salt + "\n" + (first.get("description") or "")
        else:
            tools.insert(0, {"name": "bench_salt", "description": salt, "input_schema": {"type": "object", "properties": {}}})
        out["tools"] = tools
    sys_ = out.get("system")
    if sys_ is None or sys_ == "":
        out["system"] = salt
    elif isinstance(sys_, str):
        out["system"] = salt + "\n" + sys_
    else:
        blocks = copy.deepcopy(list(sys_))
        if blocks and isinstance(blocks[0], dict) and blocks[0].get("type") == "text":
            blocks[0]["text"] = salt + "\n" + blocks[0]["text"]
        else:
            blocks.insert(0, {"type": "text", "text": salt})
        out["system"] = blocks
    return out


class _SaltedMessages:
    def __init__(self, inner, salt):
        self._inner, self._salt = inner, salt

    def create(self, **kw):
        return self._inner.messages.create(**salted(kw, self._salt))

    def stream(self, **kw):
        return self._inner.messages.stream(**salted(kw, self._salt))

    def count_tokens(self, **kw):
        return self._inner.messages.count_tokens(**salted(kw, self._salt))


class _AsyncSaltedMessages(_SaltedMessages):
    async def create(self, **kw):
        return await self._inner.messages.create(**salted(kw, self._salt))

    async def count_tokens(self, **kw):
        return await self._inner.messages.count_tokens(**salted(kw, self._salt))


class Salted:
    """The raw client with the arm's salt put on every request: the innermost layer, next to the network, so the arms
    differ by Recant and by the salt's one letter, nothing else."""

    def __init__(self, inner, salt: str):
        self._inner, self.salt = inner, salt
        create = getattr(getattr(inner, "messages", None), "create", None)
        aio = create is not None and inspect.iscoroutinefunction(inspect.unwrap(create))
        self.messages = (_AsyncSaltedMessages if aio else _SaltedMessages)(inner, salt)

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ---------------------------------------------------------------- arms
@dataclass
class Arm:
    name: str                       # A, B; C for the calibration, P for a price-out probe
    spec: str                       # off | safe | safe+<lever id>
    layout: StoreLayout
    client: Any
    salt: str


def check_spec(spec: str) -> str:
    if spec == "notes":
        raise SystemExit("--mode notes is not built yet: the wrapper does not carry managed notes, so there is no notes arm "
                         "to run. Use --mode safe for now.")
    if spec in ARM_SPECS or (spec.startswith("safe+") and spec[5:]):
        return spec
    raise SystemExit(f"an arm is off, safe or safe+<lever id> (an opted-in your-call lever), not {spec!r}")


def make_arm(name: str, spec: str, root: Path, raw, run_id: str) -> Arm:
    layout = StoreLayout(root)
    switch.set_state(layout, "meter" if spec == "off" else "on")   # off: metered from the response only, request untouched
    salt = salt_line(run_id, name)
    return Arm(name, spec, layout, wrap(Salted(raw, salt), store=layout, tenant=TENANT), salt)


def _host(layout):
    p = layout.costshape_dir(TENANT) / "host.json"
    if not p.exists():
        return cs.HostShape(agent_jobs=("agent",), agent_run_jobs=("agent",), label="bench arm")
    d = json.loads(p.read_text())
    return cs.HostShape(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})


def tag_of(i: int, attempt: str) -> str:
    return f"task-{i}#{attempt}"


def _report(layout, prices=None, *, declared: bool = False):
    """The arm's log. For the advisor (`declared`), the host shape the wrapper declared; for the bench's own counting,
    every run and every job, since a host's agents may each open their own."""
    host = _host(layout) if declared else cs.HostShape(label="bench arm")
    return cs.report(layout, TENANT, host, since=0, until=time.time() + 60, prices=prices)


def _cost(rep, c) -> Optional[float]:
    """A call's cost at the price table; None for a model with no price (counted apart, never as $0)."""
    if c["model"] not in rep.profiles:
        return None
    try:
        return rep.cost(c)
    except Exception:                                               # noqa: BLE001
        return None


def accounted(layout, rows: list, prices=None) -> dict:
    """The calls of each task's counted attempt (the one its results row names), failed calls included with the usage
    they reported, and the calls of every other attempt, discarded."""
    rep = _report(layout, prices)
    want = {tag_of(r["task"], r["attempt"]): r["task"] for r in rows if r.get("attempt")}
    by: dict = {}
    discarded = []
    for c in rep.calls + rep.failed:
        t = c.get("bench_task")
        if t in want:
            by.setdefault(want[t], []).append(c)
        elif t:
            discarded.append(c)
    return {"rep": rep, "by": by, "discarded": discarded}


def _priced(rep, calls) -> tuple[float, int]:
    """(dollars, calls with no price) for a list of calls."""
    usd, unpriced = 0.0, 0
    for c in calls:
        v = _cost(rep, c)
        if v is None:
            unpriced += 1
        else:
            usd += v
    return usd, unpriced


# ---------------------------------------------------------------- running tasks
def load_hook(spec: Optional[str]) -> Optional[Callable]:
    if not spec:
        return None
    mod, _, fn = spec.partition(":")
    if not fn:
        raise SystemExit(f"a hook is module:function, not {spec!r}")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    return getattr(importlib.import_module(mod), fn)


def default_client(is_async: bool):
    from .clients.anthropic import raw_sdk                          # the reference adapter's; --client names another
    return raw_sdk(is_async)


class Runner:
    def __init__(self, run_task: Callable, grade: Optional[Callable]):
        self.run_task, self.grade = run_task, grade
        self.is_async = inspect.iscoroutinefunction(run_task)
        self.loop = asyncio.new_event_loop() if self.is_async else None   # one loop for the bench: clients live on it

    def one(self, arm: Arm, i: int, task) -> dict:
        """One attempt at a task in an arm. Every call the client makes meanwhile carries the attempt's tag, whatever
        session, run, job, thread or asyncio task the host puts it in; only the attempt that writes a
        results row counts, so an attempt a dead process left behind is discarded on resume."""
        t0, err, result = time.time(), None, None
        attempt = os.urandom(4).hex()
        arm.client.task_tag = tag_of(i, attempt)
        try:
            if self.is_async:
                async def go():
                    async with arm.client.run("agent", session=f"task-{i}", task=i, arm=arm.name, attempt=attempt):
                        return await self.run_task(task, arm.client)
                result = self.loop.run_until_complete(go())
            else:
                with arm.client.run("agent", session=f"task-{i}", task=i, arm=arm.name, attempt=attempt):
                    result = self.run_task(task, arm.client)
        except Exception as exc:                                    # noqa: BLE001 — a failed task stays in its arm
            err = type(exc).__name__
        finally:
            arm.client.task_tag = None
        score = None
        if self.grade is not None and err is None:
            try:
                score = float(self.grade(task, result))
            except Exception as exc:                                # noqa: BLE001
                err = f"grade:{type(exc).__name__}"
        return {"task": i, "arm": arm.name, "attempt": attempt, "started": t0, "ended": time.time(), "error": err, "score": score}

    def close(self):
        if self.loop is not None:
            self.loop.close()


# ---------------------------------------------------------------- predictions
def _calibration_numbers(layout, rows: list, prices=None) -> dict:
    """Cost and calls per task over the counted attempts (every job, failed calls included), and the declared report the
    advisor sizes levers on."""
    acc = accounted(layout, rows, prices)
    calls = [c for cs_ in acc["by"].values() for c in cs_]
    n = max(1, len(acc["by"]))
    usd, unpriced = _priced(acc["rep"], calls)
    return {"tasks": len(acc["by"]), "cost_per_task_usd": usd / n, "calls_per_task": len(calls) / n, "unpriced_calls": unpriced,
            "calls": len(calls), "unpriced_models": sorted({c["model"] for c in calls if c["model"] not in acc["rep"].profiles}),
            "rep": _report(layout, prices, declared=True)}


def arm_levers(spec: str, calib_rep) -> tuple[list, list]:
    """(safe levers to apply, opted-in your-call levers) for an arm, sized on the calibration."""
    if spec == "off":
        return [], []
    levers = cs.advise(calib_rep, overlay={}, ledger=[])
    safe = [lv for lv in levers if cs.eligible(calib_rep, lv, ledger_=[], min_runs=1)[0]]
    opted = []
    if spec.startswith("safe+"):
        want = spec[5:]
        found = [lv for lv in levers if lv.id == want]
        if not found:
            raise SystemExit(f"{want}: the advisor did not size this lever on the calibration, so it cannot be opted in "
                             f"(sized: {', '.join(lv.id for lv in levers) or 'none'})")
        opted = found
    return safe, opted


def write_predictions(out: Path, arms: dict, calib: Optional[dict], run_id: str = "") -> tuple[Path, str]:
    body = {"written": datetime.now(timezone.utc).isoformat(timespec="seconds"), "recant": version_info(),
            "calibration": None if calib is None else {k: v for k, v in calib.items() if k != "rep"}, "arms": {}}
    for name, (spec, safe, opted) in arms.items():
        base = calib["cost_per_task_usd"] if calib else None
        rows = [{"id": lv.id, "kind": "safe" if lv in safe else "your_call", "share": lv.share,
                 "predicted_saving_per_task_usd": None if base is None or lv.share is None else base * lv.share,
                 "note": lv.note} for lv in safe + opted]
        total = sum(r["share"] or 0 for r in rows)
        body["arms"][name] = {"spec": spec, "levers": rows,
                              "predicted_cost_per_task_usd": None if base is None else base * (1 - total),
                              "predicted_calls_per_task_change": 0.0}
    p = out / "predictions.json"
    raw = (json.dumps(body, indent=1, sort_keys=True, default=str) + "\n").encode()
    p.write_bytes(raw)
    h = hashlib.sha256(raw).hexdigest()
    from . import friction                                           # the hash outside the run directory too
    friction.write("predictions", run_id=run_id, sha256=h)
    return p, h


def anchored_hashes(run_id: str) -> list[str]:
    """The predictions hashes the friction log holds for this run (outside the run directory)."""
    from . import friction
    p = friction.path()
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("event") == "predictions" and r.get("run_id") == run_id:
            out.append(r.get("sha256"))
    return out


# ---------------------------------------------------------------- the run
def balanced_orders(n: int, seed: int) -> list[str]:
    """Which arm runs first, per paired task: half "AB" and half "BA" (one more "AB" when odd), shuffled with a fixed
    seed — balanced, and tied to nothing in the task file's order."""
    orders = ["AB"] * ((n + 1) // 2) + ["BA"] * (n // 2)
    random.Random(seed).shuffle(orders)
    return orders


STORE_PARTS = ("tenants", "versions", "machinery")                 # memory, notes and the machinery; never log/, costshape/, the switch


def copy_store(src: Path, dst: Path) -> list[str]:
    copied = []
    for part in STORE_PARTS:
        if (src / part).exists():
            shutil.copytree(src / part, dst / part)
            copied.append(part)
    if (src / "FORMAT.json").exists():
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / "FORMAT.json", dst / "FORMAT.json")
    return copied


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def run(*, tasks_path: str, run_spec: str, grade_spec: Optional[str] = None, n: Optional[int] = None, calibrate: int = 10,
        arm_a: str = "off", arm_b: str = "safe", out: Optional[str] = None, yes: bool = False, estimate_only: bool = False,
        store: Optional[str] = None, client_spec: Optional[str] = None, allow_mixed: bool = False, prices=None,
        sdk_factory: Optional[Callable] = None, ask: Callable = input, say: Callable = print, parallel: int = 1) -> dict:
    arm_a, arm_b = check_spec(arm_a), check_spec(arm_b)
    os.environ.pop(switch.ENV, None)                                 # the bench sets each arm's switch itself
    all_tasks = [json.loads(line) for line in Path(tasks_path).read_text().splitlines() if line.strip()]
    tasks = all_tasks[:n] if n else all_tasks
    if len(tasks) <= calibrate:
        raise SystemExit(f"{len(tasks)} tasks leave none to compare after {calibrate} calibration tasks")
    runner = Runner(load_hook(run_spec), load_hook(grade_spec))
    make_raw = sdk_factory or load_hook(client_spec) or default_client
    raw = lambda: make_raw(runner.is_async)                          # noqa: E731 — one raw client per arm
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + hashlib.sha256(os.urandom(8)).hexdigest()[:4]
    root = Path(out) if out else Path("bench") / run_id
    root.mkdir(parents=True, exist_ok=True)
    cfg = {"tasks": hashlib.sha256(Path(tasks_path).read_bytes()).hexdigest()[:16], "n": len(tasks), "calibrate": calibrate,
           "arm_a": arm_a, "arm_b": arm_b, "run": run_spec, "grade": grade_spec, "client": client_spec, "store": bool(store)}
    man_p = root / "manifest.json"
    vi = version_info()
    if man_p.exists():                                               # a resumed bench: the same version, or say so
        man = json.loads(man_p.read_text())
        run_id = man["run_id"]
        if (man["recant"]["version"], man["recant"]["commit"]) != (vi["version"], vi["commit"]) and not allow_mixed:
            raise SystemExit(f"this bench was started on {man['recant']} and this is {vi}: runs are not compared across "
                             "versions (pass --allow-mixed to do it anyway; the report header will say so)")
        if man["config_hash"] != config_hash(cfg):
            raise SystemExit("this bench directory was started with another configuration; use a new --out")
        if (man["recant"]["version"], man["recant"]["commit"]) != (vi["version"], vi["commit"]):
            man.setdefault("mixed_versions", []).append(vi)
    else:
        man = {"run_id": run_id, "recant": vi, "config": cfg, "config_hash": config_hash(cfg), "created": time.time(),
               "client_mode": "async" if runner.is_async else "sync",
               "salts": {}, "store_from_copy": bool(store), "estimate": None, "predictions_hash": None}
    save = lambda: man_p.write_text(json.dumps(man, indent=1, default=str) + "\n")   # noqa: E731
    save()
    results_p = root / "results.jsonl"
    rows = [json.loads(x) for x in results_p.read_text().splitlines() if x.strip()] if results_p.exists() else []
    done = {(r["task"], r["arm"]) for r in rows}

    lock = threading.Lock()

    took: list = []

    def record(res):
        with lock:
            with results_p.open("a") as f:
                f.write(json.dumps(res) + "\n")
            rows.append(res)
            done.add((res["task"], res["arm"]))
            if man.get("confirmed") and man.get("estimate"):          # progress, once the spend is agreed
                took.append(res["ended"] - res["started"])
                total = man["estimate"]["tasks_run"]
                n = len({(r["task"], r["arm"]) for r in rows if r["arm"] != "P"})
                left = max(0, total - n) * (sum(took) / len(took)) / max(1, parallel)
                what = res["error"] or ("passed" if res["score"] else "failed" if res["score"] is not None else "done")
                say(f"[{n}/{total}] task {res['task']} arm {res['arm']}: {what}, {res['ended'] - res['started']:.1f} s; "
                    f"about {left / 60:.0f} min left")

    if parallel > 1 and runner.is_async:
        raise SystemExit("--parallel runs sync agents on threads; an async run_task already runs its own work concurrently")
    local = threading.local()

    def own(name, spec, sub):
        """This worker's own client for an arm (same store, same salt): a client's task tag is one task at a time."""
        mine = getattr(local, "arms", None)
        if mine is None:
            mine = local.arms = {}
        if name not in mine:
            mine[name] = make_arm(name, spec, root / sub, raw(), run_id)
        return mine[name]

    def pool(jobs, fn):
        if parallel <= 1:
            for j in jobs:
                fn(j)
            return
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(parallel) as ex:
            for f in [ex.submit(fn, j) for j in jobs]:
                f.result()

    def arm_rows(name):
        return [r for r in rows if r["arm"] == name]

    # 1-2. price-out and calibration (the first calibration task is the price-out; with none, a probe task outside the run)
    calib_arm = make_arm("C", "off", root / "calibration", raw(), run_id) if calibrate else None
    probe = calib_arm or make_arm("P", "off", root / "probe", raw(), run_id)
    man["salts"].update({probe.name: probe.salt})
    if man["estimate"] is None:
        if (0, probe.name) not in done:
            record(runner.one(probe, 0, tasks[0]))
        nums = _calibration_numbers(probe.layout, arm_rows(probe.name), prices)
        first = nums["cost_per_task_usd"]
        runs_left = calibrate + 2 * (len(tasks) - calibrate)
        man["estimate"] = {"first_task_usd": first, "tasks_run": runs_left + (0 if calibrate else 1),
                           "estimated_usd": first * runs_left + (0 if calibrate else first),
                           "unpriced_calls": nums["unpriced_calls"], "calls": nums["calls"], "unpriced_models": nums["unpriced_models"]}
        save()
    est = man["estimate"]
    say(f"recant bench {run_id}: estimated spend ${est['estimated_usd']:.2f} ({est['tasks_run']} task runs at "
        f"${est['first_task_usd']:.4f}, the first task's cost)")
    if est.get("unpriced_models"):
        say(f"warning: {est['unpriced_calls']} of the first task's {est['calls']} calls are on models with no price "
            f"({', '.join(est['unpriced_models'])}). The estimate leaves them out, and the verdict will read \"cost not "
            f"measured\". Pass --prices with a row for each (README, \"Your own prices\").")
    if estimate_only:
        runner.close()
        return {"root": str(root), "estimate": est}
    if not yes and not man.get("confirmed"):
        if not sys.stdin.isatty() and ask is input:
            raise SystemExit("not a terminal: pass --yes to confirm the estimated spend")
        if ask("continue? [y/N] ").strip().lower() not in ("y", "yes"):
            raise SystemExit("stopped before spending more")
    man["confirmed"] = True
    save()
    if calib_arm is not None:
        def calib_one(i):
            arm = calib_arm if parallel <= 1 else own("C", "off", "calibration")
            record(runner.one(arm, i, tasks[i]))
        pool([i for i in range(calibrate) if (i, "C") not in done], calib_one)
    calib = _calibration_numbers(calib_arm.layout, arm_rows("C"), prices) if calib_arm else None

    # 3. the arms, their levers, and the predictions — frozen before an arm runs
    arms: dict[str, Arm] = {}
    plan = {}
    for name, spec in (("A", arm_a), ("B", arm_b)):
        arm_root = root / f"arm-{name}"
        fresh = not arm_root.exists()
        if fresh and store and spec != "off":                     # a copy of a real store's memory and notes, never the store:
            man["store_copied"] = copy_store(Path(store), arm_root)   # its log, levers and switch stay behind
        arms[name] = make_arm(name, spec, arm_root, raw(), run_id)
        man["salts"][name] = arms[name].salt
        safe, opted = arm_levers(spec, calib["rep"]) if (calib and spec != "off") else ([], [])
        if spec != "off" and not calib:
            raise SystemExit(f"arm {name} ({spec}) needs a calibration to size its levers: --calibrate 1 or more")
        plan[name] = (spec, safe, opted)
        if fresh or not cs.overlay(arms[name].layout, TENANT):
            for lv in safe:
                cs.apply(arms[name].layout, TENANT, calib["rep"], lv, min_runs=1)
            for lv in opted:
                cs.opt_in(arms[name].layout, TENANT, lv.id, action=lv.action or {}, evidence=f"recant bench {run_id} arm {name}",
                          predicted_usd=lv.saves_usd)
    if man["predictions_hash"] is None:
        if any(r["arm"] in ("A", "B") for r in rows):
            raise SystemExit("the predictions are missing and paired tasks have run: writing them now would put them after the "
                             "first paired task, so the bench refuses")
        _, h = write_predictions(root, plan, calib, man["run_id"])
        man["predictions_hash"] = h
        save()
    say(f"predictions written: {root / 'predictions.json'} sha256 {man['predictions_hash']}")

    # 4. paired runs: each task in both arms, which arm goes first drawn per task, balanced and seeded
    if "orders" not in man:
        man["order_seed"] = int(hashlib.sha256(man["run_id"].encode()).hexdigest()[:8], 16)
        man["orders"] = balanced_orders(len(tasks) - calibrate, man["order_seed"])
        save()
    def pair(k):                                                      # one task in both arms, in its drawn order
        i = calibrate + k
        for name in (("A", "B") if man["orders"][k] == "AB" else ("B", "A")):
            if (i, name) not in done:
                arm = arms[name] if parallel <= 1 else own(name, arms[name].spec, f"arm-{name}")
                record(dict(runner.one(arm, i, tasks[i]), order=man["orders"][k], predictions_sha256=man["predictions_hash"]))
    pool(list(range(len(tasks) - calibrate)), pair)
    runner.close()
    rep = build_report(root, prices=prices)
    say(render(rep))
    return rep


# ---------------------------------------------------------------- the report
def _boot_ratio(rows, f, *, n=10_000, seed=0) -> tuple[float, tuple[float, float]]:
    """A statistic of resampled tasks (paired: a task is resampled with both its arms), its 95% interval."""
    val = f(rows)
    rng, out = random.Random(seed), []
    for _ in range(n):
        s = [rows[rng.randrange(len(rows))] for _ in rows]
        v = f(s)
        if v is not None and not math.isnan(v):
            out.append(v)
    out.sort()
    if not out:
        return val, (float("nan"), float("nan"))
    return val, (out[int(0.025 * len(out))], out[min(len(out) - 1, int(0.975 * len(out)))])


def _q(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {"p50": None, "p95": None, "n": 0}
    return {"p50": cs._pct(xs, 0.5), "p95": cs._pct(xs, 0.95), "n": len(xs)}


CLASSES = ("uncached", "cache_write", "cache_read", "output", "reasoning")


def _counters(layout) -> dict:
    out: dict = {}
    d = layout.log_dir(TENANT)
    for p in sorted(d.glob("*.jsonl")) if d.exists() else []:
        for r in read_run(p, typed=False):
            if r["kind"] == "run_finished":
                for k, v in (r.get("counts") or {}).items():
                    out[k] = out.get(k, 0) + v
    return out


def check_frozen(root: Path, man: dict, pred_raw: bytes, results: list) -> str:
    """The freeze. The file's hash must equal the manifest's, the friction log's entry for this run
    (outside the run directory) and every paired results row's; it must have been written before the first paired task;
    and each arm must carry out exactly the levers it froze. Returns where the hash was anchored."""
    h = hashlib.sha256(pred_raw).hexdigest()
    if h != man["predictions_hash"]:
        raise SystemExit("predictions.json changed after it was written: the report fails")
    outside = anchored_hashes(man["run_id"])
    if outside and h not in outside:
        raise SystemExit("predictions.json and the manifest agree, but not with the hash recorded outside the run directory "
                         "when the predictions were written: the report fails")
    paired = [r for r in results if r["arm"] in ("A", "B")]
    if any(r.get("predictions_sha256") not in (None, h) for r in paired):
        raise SystemExit("a paired task ran against other predictions than these: the report fails")
    pred = json.loads(pred_raw)
    written = datetime.fromisoformat(pred["written"]).timestamp()
    if paired and written > min(r["started"] for r in paired) + 1:
        raise SystemExit("predictions.json was written after the first paired task started: the report fails")
    for name in ("A", "B"):
        ov = cs.overlay(StoreLayout(root / f"arm-{name}"), TENANT)
        frozen = {r["id"]: r["share"] for r in pred["arms"][name]["levers"]}
        carried = {k: v.get("predicted_share") for k, v in ov.items() if v.get("kind") in ("applied", "your_call")}
        if set(carried) != set(frozen) or any(carried[k] is not None and frozen[k] is not None and abs(carried[k] - frozen[k]) > 1e-9
                                              for k in frozen):
            raise SystemExit(f"arm {name}'s levers differ from the frozen predictions ({sorted(carried)} against {sorted(frozen)}): "
                             "the report fails")
    return "the manifest and the friction log" if outside else "the manifest only: this machine's friction log has no entry for the run"


def no_saving_from_failure(a: dict, b: dict) -> tuple[float, float]:
    """A pair's costs as compared: an arm whose task failed is counted at no less than the other arm's cost,
    so a failure that cut a task short is never a saving; what a failure cost on top still counts."""
    ca, cb = a["cost"], b["cost"]
    if a["failed"] and not b["failed"]:
        ca = max(ca, cb)
    elif b["failed"] and not a["failed"]:
        cb = max(cb, ca)
    return ca, cb


def build_report(root, prices=None) -> dict:
    root = Path(root)
    man = json.loads((root / "manifest.json").read_text())
    pred_raw = (root / "predictions.json").read_bytes()
    results = [json.loads(x) for x in (root / "results.jsonl").read_text().splitlines() if x.strip()]
    anchor = check_frozen(root, man, pred_raw, results)
    pred = json.loads(pred_raw)
    paired_ids = sorted({r["task"] for r in results if r["arm"] == "A"} & {r["task"] for r in results if r["arm"] == "B"})
    res = {(r["task"], r["arm"]): r for r in results}
    arms, reps = {}, {}
    models, graded = set(), any(r.get("score") is not None for r in results if r["arm"] in ("A", "B"))
    for name in ("A", "B"):
        layout = StoreLayout(root / f"arm-{name}")
        acc = accounted(layout, [r for r in results if r["arm"] == name], prices)
        rep = reps[name] = acc["rep"]
        tasks, unpriced = [], 0
        for i in paired_ids:
            calls = acc["by"].get(i, [])
            models |= {c["model"] for c in calls}
            failed_calls = sum(1 for c in calls if c.get("error"))
            r = res[(i, name)]
            failed = bool(r["error"])                                 # the run raised; a failed call alone is not a failed task
            usd, up = _priced(rep, calls)
            unpriced += up
            fusd, _ = _priced(rep, [c for c in calls if c.get("error")])
            tasks.append({"task": i, "cost": usd, "failed_calls_usd": fusd, "calls": len(calls),
                          "tokens": {k: sum(c.get(k) or 0 for c in calls) for k in CLASSES}, "failed": failed, "error": r["error"],
                          "failed_calls": failed_calls, "score": (0.0 if failed else r["score"]) if graded else None,
                          "duration": r["ended"] - r["started"]})
        counted = [c for i in paired_ids for c in acc["by"].get(i, [])]
        dusd, _ = _priced(rep, acc["discarded"])
        lv_checks = []
        for row in pred["arms"][name]["levers"]:                      # the frozen predictions, and nothing else
            chk = cs.check(row["id"], rep, {"applied_at": 0, "predicted_share": row["share"], "job": "agent", "action": {}}) \
                if row["kind"] == "safe" else None
            lv_checks.append({"id": row["id"], "kind": row["kind"], "share_predicted": row["share"],
                              "realized_usd": None if chk is None else chk["realized_usd"],
                              "predicted_usd": None if chk is None else chk["predicted_usd"],
                              "ratio": None if not chk or not chk["predicted_usd"] else chk["realized_usd"] / chk["predicted_usd"]})
        ctr = _counters(layout)
        arms[name] = {"spec": pred["arms"][name]["spec"], "tasks": tasks, "latency": cs.latency(counted), "levers": lv_checks,
                      "unpriced_calls": unpriced, "equivalent": ctr.get("levers_equivalent", 0),
                      "not_equivalent": ctr.get("levers_not_equivalent", 0),
                      "opted_in_calls": {k: v for k, v in ctr.items() if k.startswith("opted_in:")},
                      "discarded": {"attempts": len({c["bench_task"] for c in acc["discarded"]}), "calls": len(acc["discarded"]), "usd": dusd}}
    A, B = arms["A"]["tasks"], arms["B"]["tasks"]
    pairs = list(zip(A, B))
    out: dict = {"header": {"recant": man["recant"], "mixed_versions": man.get("mixed_versions"), "config_hash": man["config_hash"],
                            "predictions_sha256": man["predictions_hash"], "predictions_anchored_in": anchor, "models": sorted(models),
                            "prices": "your table" if prices else f"the library's list prices as of {cs.LIST_PRICES_AS_OF}",
                            "tasks_paired": len(pairs), "calibration_tasks": man["config"]["calibrate"], "grader": graded,
                            "arms": {k: v["spec"] for k, v in arms.items()}, "salts": man["salts"],
                            "store_from_copy": man.get("store_from_copy"), "store_copied": man.get("store_copied", []),
                            "run_id": man["run_id"], "client_mode": man.get("client_mode"), "order_seed": man.get("order_seed")},
                 "estimate": man["estimate"], "discarded": {k: v["discarded"] for k, v in arms.items()}}
    from . import coverage as cov                                   # what the host sent past Recant meanwhile
    out["coverage"] = cov.summarize([c for rp in reps.values() for c in rp.calls + rp.failed],
                                    [u for rp in reps.values() for u in rp.unseen], price=reps["A"].price)
    out["coverage"]["watched"] = any(rp.watched for rp in reps.values())
    if not pairs:
        out["verdict"] = ["no task has run in both arms yet"]
        return _finish(root, out)
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")       # noqa: E731
    compared = [no_saving_from_failure(a, b) for a, b in pairs]
    base = mean([ca for ca, _ in compared])
    priced = base > 0 and not (arms["A"]["unpriced_calls"] or arms["B"]["unpriced_calls"])
    pb = quality.paired_bootstrap([(a["task"], ca, cb) for (a, _), (ca, cb) in zip(pairs, compared)])
    raw_pb = quality.paired_bootstrap([(a["task"], a["cost"], b["cost"]) for a, b in pairs])
    ok = [(a, b) for a, b in pairs if not a["failed"] and not b["failed"]]
    ok_pb = quality.paired_bootstrap([(a["task"], a["cost"], b["cost"]) for a, b in ok]) if ok else None

    def per_pass(rows):
        sa = sum(ca for (_, _), (ca, _) in rows)
        sb = sum(cb for (_, _), (_, cb) in rows)
        pa, pbb = sum(a["score"] or 0 for (a, _), _ in rows), sum(b["score"] or 0 for (_, b), _ in rows)
        if not pa or not pbb or not sa:
            return None
        return (sb / pbb) / (sa / pa) - 1
    pp = _boot_ratio(list(zip(pairs, compared)), per_pass) if (graded and priced) else (None, (None, None))
    spend = 0.0
    for sub in ("probe", "calibration", "arm-A", "arm-B"):            # every call the bench made, failed and discarded ones too
        if (root / sub).exists():
            rp = _report(StoreLayout(root / sub), prices)
            spend += _priced(rp, rp.calls + rp.failed)[0]
    rel = lambda d: d / base if priced else None                        # noqa: E731
    out["cost"] = {
        "priced": priced, "unpriced_calls": {k: v["unpriced_calls"] for k, v in arms.items()},
        "per_task": {"A": base, "B": mean([cb for _, cb in compared])},
        "paired_difference_per_task": {"delta": pb["delta"], "interval": pb["interval"], "relative": rel(pb["delta"]),
                                       "relative_interval": [rel(x) for x in pb["interval"]] if priced else None},
        "as_measured": {"delta": raw_pb["delta"], "interval": raw_pb["interval"]},
        "both_succeeded": None if ok_pb is None else {"pairs": len(ok), "delta": ok_pb["delta"], "interval": ok_pb["interval"]},
        "pairs_with_a_failure": len(pairs) - len(ok),
        "per_passed_task_relative": {"delta": pp[0], "interval": pp[1]} if graded else None,
        "without_first_task": {"A": mean([ca for ca, _ in compared[1:]]), "B": mean([cb for _, cb in compared[1:]])} if len(pairs) > 1 else None,
        "failed_calls_usd": {k: sum(t["failed_calls_usd"] for t in v["tasks"]) for k, v in arms.items()},
        "discarded_usd": {k: v["discarded"]["usd"] for k, v in arms.items()},
        "total_spend_usd": spend}
    out["tokens"] = {name: {"per_task": {k: mean([t["tokens"][k] for t in arms[name]["tasks"]]) for k in CLASSES},
                            "calls_per_task": mean([t["calls"] for t in arms[name]["tasks"]])} for name in arms}
    if graded:
        q = quality.paired_bootstrap([(a["task"], a["score"], b["score"]) for a, b in pairs])
        out["quality"] = {"pass_rate": {"A": mean([a["score"] for a in A]), "B": mean([b["score"] for b in B])},
                          "difference": q["delta"], "interval": q["interval"], "held": quality.held(q["delta"], q["interval"][0]),
                          "verdict": quality.verdict(q)}
    else:
        out["quality"] = None
    lat_pb = quality.paired_bootstrap([(a["task"], a["duration"], b["duration"]) for a, b in pairs])
    out["latency"] = {name: {"per_call": arms[name]["latency"]["call"], "per_task": _q([t["duration"] for t in arms[name]["tasks"]]),
                             "ttft": arms[name]["latency"]["ttft"], "recant_overhead": arms[name]["latency"]["overhead"]}
                      for name in arms}
    out["latency"]["paired_task_difference_s"] = {"delta": lat_pb["delta"], "interval": lat_pb["interval"]}
    out["failures"] = {name: {"failed_tasks": sum(t["failed"] for t in arms[name]["tasks"]),
                              "failed_calls": sum(t["failed_calls"] for t in arms[name]["tasks"]),
                              "errors": _count([t["error"] for t in arms[name]["tasks"] if t["error"]])} for name in arms}
    out["equivalence"] = {name: {"equivalent": arms[name]["equivalent"], "not_equivalent": arms[name]["not_equivalent"]} for name in arms}
    out["opted_in"] = {name: arms[name]["opted_in_calls"] for name in arms}
    preds = []
    # What is frozen is each lever's share of the bill. The dollars are at the report's prices: a lever's from its arm's
    # calls, cost per task from the calibration's measured cost, repriced here, so predicted and realized always share
    # one price table (a table passed to --report after the run moved one side and not the other before 0.3.2).
    base_now = None
    if (root / "calibration").exists() and pred.get("calibration"):
        base_now = _calibration_numbers(StoreLayout(root / "calibration"), [r for r in results if r["arm"] == "C"],
                                        prices)["cost_per_task_usd"]
    for name in arms:
        p = pred["arms"][name]
        for lv in arms[name]["levers"]:
            met = None if lv["ratio"] is None else abs(lv["ratio"] - 1) <= QUARTER
            preds.append({"arm": name, **lv, "met": met})
        if p.get("predicted_cost_per_task_usd") is not None:
            frozen = p["predicted_cost_per_task_usd"]
            if base_now is not None:
                shares = sum(lv["share"] or 0 for lv in p["levers"])
                predicted = base_now * (1 - shares)
            else:
                predicted = frozen
            realized_cost = mean([t["cost"] for t in arms[name]["tasks"]])
            preds.append({"arm": name, "id": "cost per task", "predicted_usd": predicted, "predicted_usd_at_freeze": frozen,
                          "realized_usd": realized_cost, "ratio": realized_cost / predicted if predicted else None})
    out["predictions"] = preds
    out["verdict"] = verdict_lines(out)
    from . import friction
    friction.report("bench")
    out["setup"] = _friction()
    return _finish(root, out)


def _count(xs):
    d: dict = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d


def _friction():
    try:
        from .friction import summary
        return summary()
    except Exception:                                                # noqa: BLE001 — no friction log yet
        return None


def verdict_lines(r: dict) -> list[str]:
    from . import coverage as cov
    lines = []
    qual, overall = cov.verdict_qualifier(r.get("coverage"))
    if qual:                                                         # never a result for traffic Recant did not see
        lines.append(qual)
    covered = "" if overall else "covered part only: "
    cost = r["cost"]
    c = cost["paired_difference_per_task"]
    arms = f"arm B ({r['header']['arms']['B']}) against arm A ({r['header']['arms']['A']})"
    if not cost["priced"]:
        up = cost["unpriced_calls"]
        lines.append(f"cost not measured: calls on models not priced in the table (arm A {up['A']}, arm B {up['B']}); "
                     "pass --prices with a row for each model")
    elif r["header"]["arms"]["A"] == r["header"]["arms"]["B"]:
        lo, hi = c["relative_interval"]
        zero = c["interval"][0] <= 0 <= c["interval"][1]
        lz = r["latency"]["paired_task_difference_s"]["interval"]
        lines.append(f"A/A: cost difference {c['relative']:+.1%} per task (95% {lo:+.1%} to {hi:+.1%}): interval "
                     f"{'contains' if zero else 'does NOT contain'} zero; latency difference interval "
                     f"{'contains' if lz[0] <= 0 <= lz[1] else 'does NOT contain'} zero")
    else:
        lo, hi = c["relative_interval"]
        pp = cost.get("per_passed_task_relative")
        if pp and pp["delta"] is not None:
            lines.append(f"{covered}{'saved' if pp['delta'] < 0 else 'cost'} {abs(pp['delta']):.1%} {'' if pp['delta'] < 0 else 'more '}per "
                         f"passed task (95% {pp['interval'][0]:+.1%} to {pp['interval'][1]:+.1%}), {arms}")
        lines.append(f"{covered}cost per task {c['relative']:+.1%} (95% {lo:+.1%} to {hi:+.1%}), {arms}")
    if cost["pairs_with_a_failure"]:
        bs = cost["both_succeeded"]
        lines.append(f"{cost['pairs_with_a_failure']} pairs with a failure: a failed arm's cost is counted at no less than the "
                     "other arm's, so a failure is never a saving"
                     + (f"; on the {bs['pairs']} pairs where both succeeded the difference is ${bs['delta']:+.4f} a task"
                        if bs else "; no pair succeeded in both arms"))
    q = r.get("quality")
    if q is None:
        lines.append("quality not measured: no grader")
    else:
        lines.append(f"quality {'held' if q['held'] else 'NOT held (' + q['verdict'] + ')'}: pass rate {q['pass_rate']['A']:.1%} "
                     f"-> {q['pass_rate']['B']:.1%}, difference {q['difference'] * 100:+.1f} points "
                     f"(95% {q['interval'][0] * 100:+.1f} to {q['interval'][1] * 100:+.1f})")
    for p in r["predictions"]:
        if p.get("ratio") is None or p["id"] == "cost per task":
            continue
        lines.append(f"prediction {'met' if p['met'] else 'MISSED'}: {p['id']} (arm {p['arm']}) realized ${p['realized_usd']:.4f} "
                     f"against ${p['predicted_usd']:.4f} predicted ({p['ratio']:.0%}; met within a quarter)")
    fa, fb = r["failures"]["A"]["failed_tasks"], r["failures"]["B"]["failed_tasks"]
    if fa != fb:
        lines.append(f"failures differ: arm A {fa} failed tasks, arm B {fb}; failed tasks count as failed, none dropped")
    elif fa:
        lines.append(f"failures: {fa} failed tasks in each arm, counted as failed, none dropped")
    for name, d in r["discarded"].items():
        if d["calls"]:
            lines.append(f"arm {name}: {d['attempts']} interrupted attempts discarded ({d['calls']} calls, ${d['usd']:.4f}, "
                         "in the total spend, not in the comparison)")
    for name, e in r["equivalence"].items():
        if e["not_equivalent"]:
            lines.append(f"equivalence NOT held on {e['not_equivalent']} calls in arm {name} (the host's own request was sent)")
    if r.get("coverage") is not None:
        lines.append(cov.line(r["coverage"], watched=r["coverage"].get("watched", False)))
    return lines


def _finish(root: Path, out: dict) -> dict:
    (root / "report.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    (root / "report.md").write_text(render(out) + "\n")
    return out


def _f(x, fmt="{:.4f}"):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else fmt.format(x)


def render(r: dict) -> str:
    h = r["header"]
    L = [f"# recant bench {h['run_id']}", "",
         f"- Recant {describe(h['recant'])}"
         + (f"; MIXED VERSIONS: {h['mixed_versions']}" if h.get("mixed_versions") else ""),
         f"- config {h['config_hash']}; predictions sha256 {h['predictions_sha256']}",
         f"- models {', '.join(h['models']) or '—'}; prices: {h['prices']}",
         f"- {h['tasks_paired']} paired tasks after {h['calibration_tasks']} calibration tasks; grader: {'yes' if h['grader'] else 'no'}",
         f"- arms: A = {h['arms']['A']}, B = {h['arms']['B']}; order alternating task by task",
         f"- cache salts (the only thing the kit adds to an off arm's request): {h['salts']}",
         f"- the Recant arm started from a copy of a store: " + (f"yes ({', '.join(h.get('store_copied') or [])})" if h.get('store_from_copy') else "no (fresh)"),
         f"- predictions hash anchored in {h.get('predictions_anchored_in')}; arm order drawn per task (seed {h.get('order_seed')})",
         f"- estimated spend ${r['estimate']['estimated_usd']:.2f} before the run" if r.get("estimate") else "", ""]
    L += ["## Verdict", ""] + [f"- {x}" for x in r["verdict"]] + [""]
    if "cost" not in r:
        return "\n".join(L)
    c = r["cost"]
    L += ["## Cost", "", "| | arm A | arm B |", "|---|---|---|",
          f"| per task | ${_f(c['per_task']['A'])} | ${_f(c['per_task']['B'])} |"]
    if c.get("without_first_task"):
        L.append(f"| per task, each arm's first task left out (cold start) | ${_f(c['without_first_task']['A'])} | ${_f(c['without_first_task']['B'])} |")
    d = c["paired_difference_per_task"]
    L += ["", f"Paired difference per task (B − A): ${_f(d['delta'])} (95% ${_f(d['interval'][0])} to ${_f(d['interval'][1])}),"
          f" {_f(d['relative'], '{:+.1%}')} of arm A." + ("" if c["priced"] else " Not priced: see the verdict."),
          f"Per passed task: {_f((c.get('per_passed_task_relative') or {}).get('delta'), '{:+.1%}')}"
          + (f" (95% {_f(c['per_passed_task_relative']['interval'][0], '{:+.1%}')} to {_f(c['per_passed_task_relative']['interval'][1], '{:+.1%}')})"
             if c.get("per_passed_task_relative") else " (no grader)"),
          f"Pairs with a failure: {c['pairs_with_a_failure']} (a failed arm counted at no less than the other: never a saving); "
          f"as measured, the difference is ${_f(c['as_measured']['delta'])} a task",
          f"Failed calls' billed usage: arm A ${_f(c['failed_calls_usd']['A'])}, arm B ${_f(c['failed_calls_usd']['B'])}; "
          f"discarded attempts: arm A ${_f(c['discarded_usd']['A'])}, arm B ${_f(c['discarded_usd']['B'])}",
          f"Total spend: ${_f(c['total_spend_usd'], '{:.2f}')}", ""]
    L += ["## Tokens by class, per task", "", "| | " + " | ".join(CLASSES) + " | calls |", "|---" * (len(CLASSES) + 2) + "|"]
    for name, t in r["tokens"].items():
        L.append(f"| arm {name} | " + " | ".join(f"{t['per_task'][k]:,.0f}" for k in CLASSES) + f" | {t['calls_per_task']:.1f} |")
    L.append("")
    if r.get("quality"):
        q = r["quality"]
        L += ["## Quality", "", f"Pass rate: A {q['pass_rate']['A']:.1%}, B {q['pass_rate']['B']:.1%}; paired difference "
              f"{q['difference'] * 100:+.1f} points (95% {q['interval'][0] * 100:+.1f} to {q['interval'][1] * 100:+.1f}). "
              "Held means no observed loss and the interval no worse than −3 points: "
              f"**{'held' if q['held'] else q['verdict']}**.", ""]
    else:
        L += ["## Quality", "", "Not measured: no grader was given.", ""]
    L += ["## Latency (seconds; Recant's own time in ms)", "", "| | per call p50 / p95 | per task p50 / p95 | first token p50 / p95 | Recant's own time p50 / p95 |",
          "|---|---|---|---|---|"]
    for name in ("A", "B"):
        x = r["latency"][name]
        g = lambda d, s=1.0: f"{_f(d['p50'] and d['p50'] * s, '{:.2f}')} / {_f(d['p95'] and d['p95'] * s, '{:.2f}')}"   # noqa: E731
        L.append(f"| arm {name} | {g(x['per_call'])} | {g(x['per_task'])} | {g(x['ttft'])} | {g(x['recant_overhead'], 1000)} |")
    lp = r["latency"]["paired_task_difference_s"]
    L += ["", f"Paired task time difference (B − A): {lp['delta']:+.2f} s (95% {lp['interval'][0]:+.2f} to {lp['interval'][1]:+.2f}).", ""]
    L += ["## Predicted against realized", ""]
    for p in r["predictions"]:
        L.append(f"- arm {p['arm']} {p['id']}: realized ${_f(p.get('realized_usd'))} against ${_f(p.get('predicted_usd'))} predicted"
                 + (f" ({p['ratio']:.0%})" if p.get("ratio") else ""))
    L += ["", "## Failures and equivalence", ""]
    for name in ("A", "B"):
        f_ = r["failures"][name]
        e = r["equivalence"][name]
        L.append(f"- arm {name}: {f_['failed_tasks']} failed tasks, {f_['failed_calls']} failed calls {f_['errors'] or ''}; "
                 f"equivalence {e['equivalent']} of {e['equivalent'] + e['not_equivalent']} calls")
    L += ["", "## Setup", "", json.dumps(r.get("setup")) if r.get("setup") else "no friction log", ""]
    return "\n".join(L)
