"""The safe levers carried out on an Anthropic-shaped request: a host that keeps its own client
reads Recant's overlay and asks `apply` to put the cache markers where the advisor said — nothing else changes. The
request differs from what the host built only in `cache_control` keys (and a string turned into the one text block a
marker can sit on); `recant.quality.equivalent` checks exactly that.

    params = apply(params, costshape.overlay(layout, tenant))

Levers carried out: `breakpoint.add` (a marker on the last system block and on the last block of the last message, so
the stable prefix and the appended history are read from the cache) and `ttl.prefix` (the system block's marker at the
lifetime the advisor chose). `CARRIES_OUT` names them; `declare_host` tells the advisor this host carries them out. Other providers' request shapes
wait until a host needs them."""
from __future__ import annotations

import copy
import json
import os
import threading
from typing import Any, Optional

CARRIES_OUT = ("breakpoint.add", "ttl.prefix")

# How a request turns a model's reasoning off (vendor data, probed 2026-11-03): most take {"type": "disabled"};
# claude-sonnet-5-5 refuses that and takes {"type": "between_tools"} (no thinking before responding; the short updates it
# writes between tool calls come back as thinking blocks); claude-opus-5-5 and claude-fable-5-1 refuse both — their
# reasoning cannot be turned off, only steered with an effort setting. A model not listed takes "disabled".
THINKING_OFF: dict[str, Optional[dict]] = {"claude-sonnet-5-5": {"type": "between_tools"},
                                           "claude-opus-5-5": None, "claude-fable-5-1": None}
CANNOT_TURN_REASONING_OFF = frozenset(m for m, v in THINKING_OFF.items() if v is None)


def thinking_off(model: Optional[str]) -> Optional[dict]:
    """The `thinking` value that turns this model's reasoning off, or None when it cannot be turned off."""
    return THINKING_OFF.get(model or "", {"type": "disabled"})
_DECLARE_LOCK = threading.Lock()


def declare_host(layout, tenant: str, job: str = "agent", run_job: Optional[str] = None) -> dict:
    """Tell `recant cost` this host carries out the safe levers with `apply` (and which log runs are its agent's): the
    host shape in the store, where the advisor finds it. Until a host declares it, the levers are reported and never
    applied. `recant.wrap` declares it itself, once for its job and again for each run job it meters — a multi-agent
    host opens one run per agent, and every one of them is the agent's work; the lists only grow."""
    d = layout.costshape_dir(tenant)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "host.json"
    with _DECLARE_LOCK:                                   # a host's threads may declare at once: read, merge, write as one
        old = json.loads(p.read_text()) if p.exists() else {}
        jobs = list(dict.fromkeys([*(old.get("agent_jobs") or []), job]))
        runs = list(dict.fromkeys([*(old.get("agent_run_jobs") or []), run_job or job]))
        host = {"agent_jobs": jobs, "agent_run_jobs": runs, "carries_out": list(CARRIES_OUT), "label": "wrapped host"}
        if old != host:                                   # written whole and renamed: no reader sees half a file
            tmp = p.with_name(f".host.json.{os.getpid()}.{threading.get_ident()}")
            tmp.write_text(json.dumps(host, indent=1) + "\n")
            os.replace(tmp, p)
    return host


def apply_opted_in(params: dict, overlay: dict, job: str) -> tuple[dict, list[str]]:
    """The your-call levers the developer opted in for this job (`recant cost --opt-in`), carried out on the request:
    `reasoning.<job>` with action {"reasoning": "off"} sends thinking disabled; `effort.<job>` with {"effort": <level>}
    sends that effort. These change what the model does, so they are judged by the quality standard, never by the
    equivalence check; the wrapper applies them after it. Returns the request and the levers carried out."""
    out, done = copy.deepcopy(params), []
    for lid, entry in (overlay or {}).items():
        if not isinstance(entry, dict) or entry.get("kind") != "your_call":
            continue
        action = entry.get("action") or {}
        if lid == f"reasoning.{job}" and action.get("reasoning") == "off":
            off = thinking_off(out.get("model"))
            if off is not None:                            # a model that cannot turn it off is sent as the host built it
                out["thinking"] = dict(off)
                done.append(lid)
        elif lid == f"effort.{job}" and action.get("effort"):
            out["output_config"] = {**(out.get("output_config") or {}), "effort": action["effort"]}
            done.append(lid)
    return out, done


def _blocks(content) -> list:
    """A string or a list of blocks as a list of blocks (a marker sits on a block)."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content or [])


def _mark(block: dict, ttl: Optional[str]) -> dict:
    out = dict(block)
    out["cache_control"] = {"type": "ephemeral", **({"ttl": "1h"} if ttl == "1h" else {})}
    return out


def apply(params: dict, overlay: dict, *, default_ttl: str = "5m") -> dict:
    """The request with the overlay's safe levers carried out; the input is left untouched. With no applied lever in the
    overlay the request comes back as a deep copy of itself."""
    out = copy.deepcopy(params)
    applied = {k for k, v in (overlay or {}).items() if isinstance(v, dict) and v.get("kind") == "applied"}
    if "breakpoint.add" not in applied:
        return out
    ttl = default_ttl
    if "ttl.prefix" in applied:
        ttl = str(((overlay.get("ttl.prefix") or {}).get("action") or {}).get("ttl") or default_ttl)
    if out.get("system"):
        sysb = _blocks(out["system"])
        sysb[-1] = _mark(sysb[-1], ttl)
        out["system"] = sysb
    msgs = out.get("messages") or []
    if msgs:
        last = dict(msgs[-1])
        blocks = _blocks(last.get("content"))
        if blocks:
            blocks[-1] = _mark(blocks[-1], None)              # the moving breakpoint: the history up to here, at the short lifetime
            last["content"] = blocks
            msgs[-1] = last
        out["messages"] = msgs
    return out
