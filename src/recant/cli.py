"""`recant cost`: the cost advisor on a store's log. Costs nothing: it reads the log, prices it,
and — only with --nightly — applies, checks and reverts the safe levers through the overlay.

    recant cost --store DIR --tenant ID [--since 7d] [--until T] [--host host.json] [--prices prices.json] [--json]
    recant cost ... --nightly                      the nightly pass: check, revert, apply, write the report
    recant cost ... --revert ID --reason TEXT      undo one change
    recant cost ... --opt-in ID --action JSON --evidence TEXT   record a your-call lever the developer turned on
    recant off --store DIR [--meter]               the wrapper a pure pass-through from the next call (--meter: metering only)
    recant on --store DIR                          back on from the next call
    recant status --store DIR                      the switch in force, and where it comes from
    recant bench --tasks T --run mod:run_task [--grade mod:grade] ...   CALLS THE MODEL: Recant against none (bench.py)
    recant eval memory --sessions S --make-questions | --questions Q ...  CALLS THE MODEL: memory on your sessions (eval_memory.py)

host.json holds a HostShape's fields; prices.json maps a model id to {input, output, cache_read, cache_write,
cache_write_long, batch_factor} in USD per million tokens. Without a price table the library's dated list applies,
then prices learned from billed cost, then tokens only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import costshape as cs
from .provider import Prices
from .store import StoreLayout


def _prices(path):
    if not path:
        return None
    return {m: Prices(**row) for m, row in json.loads(Path(path).read_text()).items()}


def _host(path, layout=None, tenant=None):
    """The host shape from --host, else the one a wrapped host wrote into the store (`recant.wrap`), else the default."""
    if not path and layout is not None and tenant is not None and (layout.costshape_dir(tenant) / "host.json").exists():
        path = layout.costshape_dir(tenant) / "host.json"
    if not path:
        return cs.HostShape()
    d = json.loads(Path(path).read_text())
    return cs.HostShape(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})


def run_cost(layout, tenant, host, *, prices=None, since="7d", until=None, as_json=False, nightly=False, revert=None, reason="",
             opt_in=None, action=None, evidence="", log=None, on_advice=None, show_all: bool = False) -> str:
    if revert:
        return json.dumps(cs.revert(layout, tenant, revert, reason=reason or "by hand", log=log), indent=1)
    if opt_in:
        return json.dumps(cs.opt_in(layout, tenant, opt_in, action=action or {}, evidence=evidence, log=log), indent=1)
    if nightly:
        out = cs.nightly(layout, tenant, host, prices=prices, log=log, on_advice=on_advice, show_all=show_all)
        return json.dumps({"events": out["events"], "path": out["path"]}, indent=1, default=str) if as_json else out["report"]
    rep = cs.report(layout, tenant, host, since=since, until=until, prices=prices)
    levers = cs.advise(rep, overlay=cs.overlay(layout, tenant), ledger=cs.ledger(layout, tenant))
    if not show_all:                                    # a lever that saves nothing is noise on a small store; --all lists it
        levers = [lv for lv in levers if cs.worth_listing(lv)]
    bill = cs.billing_findings(layout, tenant)          # None without billing access
    levers += bill or []
    if as_json:
        return json.dumps({"shape": rep.shape, "levers": [vars(lv) for lv in levers], "overlay": cs.overlay(layout, tenant),
                           "billing": "reconciled" if bill is not None else "not visible"}, indent=1, default=str)
    from .billing import NOT_VISIBLE
    return rep.text(levers, cs.ledger(layout, tenant)) + ("" if bill is not None else f"\n{NOT_VISIBLE}")


def run_coverage(usage_path, store=None, tenant=None, *, prices=None, as_json=False) -> str:
    """`recant coverage --usage`: the host's own totals against every model_call Recant metered on the same
    days, a tool's own totals (Claude Code, the Agent SDK) included."""
    from . import costshape as cs_
    from . import coverage
    from .log import read_run
    rows = coverage.read_usage(usage_path)
    days = {r["ts"][:10] for r in rows}
    calls = []
    if store:
        if not tenant:
            raise SystemExit("--store needs --tenant")
        for p in sorted(StoreLayout(store).log_dir(tenant).glob("*.jsonl")):
            calls += [r for r in read_run(p, typed=False) if r["kind"] == "model_call" and r["ts"][:10] in days]
    table = {m: cs_.profile_from_prices(v) for m, v in (prices or {}).items()}

    def price(model, u):
        from .billing import model_family
        pf = table.get(model) or cs_.LIST_PRICES.get(model_family(model) or "")
        return None if pf is None else pf.bill(u.get("input", 0), u.get("cache_write", 0), u.get("cache_read", 0), u.get("output", 0))
    r = coverage.reconcile(rows, calls, price=price)
    from . import friction
    friction.report("coverage")
    return json.dumps(r, indent=1, default=str) if as_json else coverage.render_reconcile(r)


def main(argv=None) -> int:
    """Every command appends to the friction log: when, which, and how it failed."""
    from . import friction
    argv = list(sys.argv[1:] if argv is None else argv)
    friction.started(argv)
    try:
        code = _main(argv)
    except SystemExit as exc:
        if exc.code not in (0, None):
            friction.failed(argv, exc)
        else:
            friction.finished(argv, 0)
        raise
    except BaseException as exc:
        friction.failed(argv, exc)
        raise
    friction.finished(argv, code)
    return code


def _main(argv) -> int:
    from .version import line as version_line
    ap = argparse.ArgumentParser(prog="recant")
    ap.add_argument("--version", action="version", version=version_line())
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cost", help="the cost advisor on a store's log (costs nothing)")
    c.add_argument("--store", required=True)
    c.add_argument("--tenant", required=True)
    c.add_argument("--since", default="7d")
    c.add_argument("--until")
    c.add_argument("--host")
    c.add_argument("--prices")
    c.add_argument("--json", action="store_true")
    c.add_argument("--nightly", action="store_true")
    c.add_argument("--revert")
    c.add_argument("--reason", default="")
    c.add_argument("--opt-in", dest="opt_in")
    c.add_argument("--action", type=json.loads)
    c.add_argument("--evidence", default="")
    c.add_argument("--all", dest="show_all", action="store_true", help="list levers that save nothing in this window too")
    for name, what in (("off", "the wrapper a pure pass-through from the next call; --meter keeps metering only"),
                       ("on", "the wrapper on again from the next call"), ("status", "the switch in force")):
        sw = sub.add_parser(name, help=what)
        sw.add_argument("--store", required=True)
        if name == "off":
            sw.add_argument("--meter", action="store_true", help="meter every call, change nothing on the request")
    b = sub.add_parser("bench", help="CALLS THE MODEL: Recant against no Recant on your own tasks, paired and fair")
    b.add_argument("--tasks", help="tasks.jsonl, one task per line, any JSON your run_task understands")
    b.add_argument("--run", help="module:function, run_task(task, client) -> result (sync or async)")
    b.add_argument("--grade", help="module:function, grade(task, result) -> bool | float; without it quality is not measured")
    b.add_argument("--n", type=int, help="use the first N tasks (calibration included); default all")
    b.add_argument("--calibrate", type=int, default=10, help="tasks run without Recant first, to size the levers (default 10)")
    b.add_argument("--mode", default="safe", help="arm B: safe (the safe levers, the default); notes is not built yet")
    b.add_argument("--arm-a", default="off", help="arm A: off (the default), safe, or safe+<lever id>")
    b.add_argument("--arm-b", help="arm B, overriding --mode: off (an A/A run), safe, or safe+<lever id> (an opted-in lever)")
    b.add_argument("--out", help="the bench directory; default bench/<run id>; an existing one resumes")
    b.add_argument("--store", help="start the Recant arm from a copy of this store (the report says so)")
    b.add_argument("--client", help="module:function returning your raw client (sync or async); default anthropic's")
    b.add_argument("--prices", help="a price table, as for recant cost")
    b.add_argument("--yes", action="store_true", help="do not ask before spending the estimate")
    b.add_argument("--parallel", type=int, default=1, help="tasks in flight at once (sync agents; each task still runs its two arms in its drawn order)")
    b.add_argument("--estimate", action="store_true", help="run the first task, print the estimated spend, stop")
    b.add_argument("--allow-mixed", action="store_true", help="resume a bench started on another Recant version")
    b.add_argument("--report", action="store_true", help="only rebuild the report of the bench in --out")
    ev = sub.add_parser("eval", help="CALLS THE MODEL: measure memory on your own sessions").add_subparsers(dest="what", required=True)
    em = ev.add_parser("memory", help="four arms over your past sessions: none, carried history, managed notes, memory v2")
    em.add_argument("--sessions", help="sessions.jsonl: {scope, ts, messages} per line")
    em.add_argument("--load", help="module:function returning the same list of sessions")
    em.add_argument("--model", default="claude-sonnet-5-5")
    em.add_argument("--make-questions", action="store_true", help="draft questions.jsonl from the sessions, then stop for you to read it")
    em.add_argument("--questions", default="questions.jsonl")
    em.add_argument("--per-scope", type=int, default=12)
    em.add_argument("--arms", default=",".join(("none", "history", "notes", "memory_v2")))
    em.add_argument("--window", type=int, default=150_000, help="carried history's window, in estimated tokens")
    em.add_argument("--out", help="the eval directory; default eval/<UTC stamp>; an existing one resumes")
    em.add_argument("--parallel", type=int, default=1, help="scopes built and answered at once")
    cv = sub.add_parser("coverage", help="what Recant saw of your traffic, against your own per-run totals (costs nothing)")
    cv.add_argument("--usage", required=True, help="your per-run log, JSON lines: ts, model, tokens_in, tokens_out, cache_read, "
                                                   "cache_write, optional billed (api_key | subscription)")
    cv.add_argument("--store", help="the store Recant metered into (none: compare against an empty meter)")
    cv.add_argument("--tenant")
    cv.add_argument("--prices", help="a price table, as for recant cost; default the library's list prices")
    cv.add_argument("--json", action="store_true")
    fb = sub.add_parser("feedback", help="write a numbers-only feedback bundle (and a short form); sends nothing")
    fb.add_argument("--bench", nargs="*", help="bench directories (default: every bench/*/ with a report)")
    fb.add_argument("--eval", nargs="*", dest="evals", help="eval directories (default: every eval/*/ with a report)")
    fb.add_argument("--store", help="a store whose recant cost summary goes in too")
    fb.add_argument("--tenant")
    fb.add_argument("--no-form", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "coverage":
        print(run_coverage(a.usage, a.store, a.tenant, prices=_prices(a.prices), as_json=a.json))
        return 0
    if a.cmd == "feedback":
        from . import feedback as fb_
        benches = [Path(x) for x in a.bench] if a.bench is not None else sorted(p.parent for p in Path("bench").glob("*/report.json"))
        evals = [Path(x) for x in a.evals] if a.evals is not None else sorted(p.parent for p in Path("eval").glob("*/report.json"))
        cost = None
        if a.store and a.tenant:
            layout = StoreLayout(a.store)
            cost = json.loads(run_cost(layout, a.tenant, _host(None, layout, a.tenant), as_json=True))
            cost = {"shape": cost.get("shape"), "levers": [{"id": lv.get("id"), "kind": lv.get("kind"), "saves_usd": lv.get("saves_usd"),
                                                          "share": lv.get("share")} for lv in cost.get("levers", [])]}
        fb_.run(bench_dirs=benches, eval_dirs=evals, cost_json=cost, form=not a.no_form)
        return 0
    if a.cmd == "eval":
        from datetime import datetime, timezone
        from . import eval_memory as em_
        from .bench import load_hook
        from .clients.anthropic import raw_sdk
        if not (a.sessions or a.load):
            ap.error("recant eval memory needs --sessions or --load")
        sessions = em_.load_sessions(a.sessions, load_hook(a.load))
        out = Path(a.out or Path("eval") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        out.mkdir(parents=True, exist_ok=True)
        if a.make_questions:
            r = em_.make_questions(sessions, raw=raw_sdk(False), model=a.model, out_path=a.questions, root=out, per_scope=a.per_scope)
            print(json.dumps(r, indent=1, default=str))
            print(f"read and edit {a.questions}; it is hashed when scoring starts")
            return 0
        em_.evaluate(sessions, a.questions, raw=raw_sdk(False), model=a.model, out=str(out), arms=tuple(a.arms.split(",")),
                     window_tokens=a.window, parallel=a.parallel)
        print((out / "report.md").read_text())
        return 0
    if a.cmd == "bench":
        from . import bench as bench_
        if a.report:
            if not a.out:
                ap.error("--report needs --out, the bench directory")
            print(bench_.render(bench_.build_report(a.out, prices=_prices(a.prices))))
            return 0
        if not (a.tasks and a.run):
            ap.error("recant bench needs --tasks and --run")
        bench_.run(tasks_path=a.tasks, run_spec=a.run, grade_spec=a.grade, n=a.n, calibrate=a.calibrate, arm_a=a.arm_a,
                   arm_b=a.arm_b or a.mode, out=a.out, yes=a.yes, estimate_only=a.estimate, store=a.store, client_spec=a.client,
                   allow_mixed=a.allow_mixed, prices=_prices(a.prices), parallel=a.parallel)
        return 0
    if a.cmd in ("off", "on", "status"):
        from . import switch
        if a.cmd != "status":
            switch.set_state(a.store, "meter" if getattr(a, "meter", False) else a.cmd)
        print(json.dumps(switch.describe(a.store), indent=1))
        return 0
    layout = StoreLayout(a.store)
    if not (a.revert or a.opt_in):
        from . import friction
        friction.report("cost")
    print(run_cost(layout, a.tenant, _host(a.host, layout, a.tenant), prices=_prices(a.prices), since=a.since, until=a.until,
                   as_json=a.json, nightly=a.nightly, revert=a.revert, reason=a.reason, opt_in=a.opt_in, action=a.action,
                   evidence=a.evidence, show_all=a.show_all))
    return 0


if __name__ == "__main__":
    sys.exit(main())
