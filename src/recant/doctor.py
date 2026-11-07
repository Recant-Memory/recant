"""`recant doctor`: the setup checked in one go, one line per check, and for each problem one line saying what to do.
Reads only: it calls no model, makes no directory, and writes nothing to the friction log; on an existing store the
writable check creates one new temporary file there and removes it."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from .store import StoreLayout


def _check(name: str, ok: bool, detail: str, fix: Optional[str] = None) -> dict:
    return {"check": name, "ok": ok, "detail": detail, **({"fix": fix} if not ok and fix else {})}


def run(store, tenant: str) -> list[dict]:
    from . import costshape as cs, switch
    from .log import read_run
    from .version import line
    out = [_check("version", True, line())]
    import importlib.metadata as md                  # its version, not an import: only the adapter imports the SDK
    try:
        out.append(_check("anthropic SDK", True, f"anthropic {md.version('anthropic')} installed"))
    except md.PackageNotFoundError:
        out.append(_check("anthropic SDK", False, "the Anthropic SDK is not installed",
                          'pip install "recant[anthropic]" (recant.wrap wraps its clients)'))
    root = Path(store)
    if not root.exists():                                # nothing is made here: can recant.wrap make it?
        parent = next((p for p in root.parents if p.exists()), None)
        can = parent is not None and os.access(parent, os.W_OK | os.X_OK)
        out.append(_check("store", can, f"{root} does not exist yet; recant.wrap makes it" if can else
                          f"{root} cannot be made ({parent} is not writable)",
                          "set RECANT_HOME to a writable directory, or pass store= to recant.wrap"))
        out.append(_check("metering", False, f"nothing metered yet in {root}",
                          "wrap your client: client = recant.wrap(anthropic.Anthropic()), then run your app"))
        return out
    try:                                                 # a new file of its own (exclusive, random name), then removed:
        fd, probe = tempfile.mkstemp(prefix=".doctor-", dir=root)    # nothing already in the store is touched
        os.close(fd)
        os.unlink(probe)
        out.append(_check("store", True, f"{root} is writable"))
    except OSError as exc:
        out.append(_check("store", False, f"{root} is not writable ({type(exc).__name__})",
                          "set RECANT_HOME to a writable directory, or pass store= to recant.wrap"))
        return out
    layout = StoreLayout(root)
    state = switch.state(layout)
    out.append(_check("switch", state != "off", f"the switch is {state}",
                      "recant on (recant off makes the wrapper a pure pass-through: nothing is metered)"))
    logs = layout.log_dir(tenant)
    calls = [r for p in sorted(logs.glob("*.jsonl")) for r in read_run(p, typed=False) if r["kind"] == "model_call"] \
        if logs.is_dir() else []
    wrapped = (layout.costshape_dir(tenant) / "host.json").exists()
    if not calls:
        out.append(_check("metering", False, f"no model call metered for tenant {tenant!r} in {root}",
                          "wrap your client: client = recant.wrap(anthropic.Anthropic()), then run your app"
                          if not wrapped else "the client is wrapped but no call has gone through it: run your app"))
        return out
    last = max(cs._ts(r["ts"]) for r in calls if r.get("ts"))
    ago = time.time() - last
    out.append(_check("metering", True, f"{len(calls)} model calls metered, the last {cs._period(ago)} ago"
                      + ("" if wrapped else " (through a Meter, not recant.wrap)")))
    rep = cs.report(layout, tenant, cs.HostShape(), since=0)
    unpriced = sorted({c["model"] for c in rep.calls if c["model"] not in rep.profiles})
    out.append(_check("prices", not unpriced, "every model Recant metered has a price (calls it did not see are on the "
                      "coverage line)" if not unpriced else
                      f"no price for {', '.join(unpriced)}: counted in tokens only",
                      "pass --prices FILE to recant cost with a row for each (README, \"Your own prices\")"))
    cov = rep.coverage()
    unseen = cov.get("calls_unseen") or 0
    if not rep.watched:
        out.append(_check("coverage", True, "the coverage watch has recorded nothing yet (it writes as a run closes)"))
    else:
        un = cov.get("unseen") or []
        where = lambda us: "; ".join(f"{u['requests']} calls to {u['host']}" + (f", model {u['model']}" if u.get("model") else "")  # noqa: E731
                                     for u in us[:3])
        ours = [u for u in un if u.get("provider") == "anthropic"]          # a client recant.wrap could have wrapped
        other = [u for u in un if u.get("provider") != "anthropic"]
        if not unseen:
            out.append(_check("coverage", True, f"Recant saw all {cov.get('calls_seen', 0)} model calls the watch could see"))
        if ours:
            out.append(_check("coverage", False, f"{sum(u['requests'] for u in ours)} Anthropic calls went past Recant: "
                              f"{where(ours)}", "wrap that client too: recant.wrap(anthropic.Anthropic()) or AsyncAnthropic"))
        if other:                       # the wrapper is Anthropic's only: named, not a fault to fix
            out.append(_check("coverage", True, f"{sum(u['requests'] for u in other)} calls to another provider went past "
                              f"Recant: {where(other)}. The wrapper is Anthropic's only, so recant cost names them on its "
                              "Coverage line; to count them too, record each with recant.metering.Meter (input = the "
                              "provider's uncached prompt tokens, output = its completion tokens)"))
    return out


def render(checks: list[dict]) -> str:
    lines = []
    for c in checks:
        lines.append(f"{'ok ' if c['ok'] else 'FIX'} {c['check']:13} {c['detail']}" + (f"\n    -> {c['fix']}" if c.get("fix") else ""))
    bad = sum(1 for c in checks if not c["ok"])
    lines.append("all good" if not bad else f"{bad} to fix")
    return "\n".join(lines)
