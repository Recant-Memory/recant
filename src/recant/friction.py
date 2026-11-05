"""The friction log: every `recant` command appends one line to a local file — when it ran, which
command, whether it failed and with which error class from which Recant function, `--help` calls, switch and lever
changes, and when a complete report was first written. Setup friction is a result: every report's Setup section and
the feedback bundle carry `summary()`. The file stays on this machine (`~/.recant/friction.jsonl`, or `$RECANT_HOME`);
it holds no arguments' values, no paths and no text from any session."""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Optional

CONFIG_CMDS = {"off", "on"}
CONFIG_FLAGS = {"--opt-in", "--revert", "--prices", "--host"}


def path() -> Path:
    return Path(os.environ.get("RECANT_HOME") or Path.home() / ".recant") / "friction.jsonl"


def _cmd(argv: list) -> str:
    words = [a for a in argv if not a.startswith("-")][:2]
    if words[:1] == ["eval"]:
        return " ".join(words)
    return words[0] if words else ""


def write(event: str, argv: Optional[list] = None, **fields) -> None:
    rec = {"ts": time.time(), "event": event, **({"cmd": _cmd(argv)} if argv is not None else {}), **fields}
    try:
        p = path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass                                                        # a read-only home never breaks a command


def started(argv: list) -> None:
    flags = sorted({a.split("=")[0] for a in argv if a.startswith("--")} & (CONFIG_FLAGS | {"--help"}) | ({"-h"} & set(argv)))
    write("command", argv, flags=flags, config_change=_cmd(argv) in CONFIG_CMDS or bool(set(flags) & CONFIG_FLAGS))


def failed(argv: list, exc: BaseException) -> None:
    where = None
    for fr in reversed(traceback.extract_tb(exc.__traceback__)):
        parts = Path(fr.filename).parts
        if "recant" in parts:
            mod = ".".join(parts[len(parts) - 1 - parts[::-1].index("recant"):])[:-3]
            where = f"{mod}.{fr.name}"
            break
    write("failed", argv, error=type(exc).__name__, where=where)


def finished(argv: list, code: int = 0) -> None:
    write("finished", argv, code=code)


def report(kind: str) -> None:
    write("report", kind=kind)


def summary() -> Optional[dict]:
    p = path()
    if not p.exists():
        return None
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows:
        return None
    cmds = [r for r in rows if r["event"] == "command"]
    fails = [r for r in rows if r["event"] == "failed"]
    reports = [r for r in rows if r["event"] == "report"]
    first = rows[0]["ts"]
    retries, last_failed = 0, set()
    for r in rows:                                                  # a command run again after it failed
        if r["event"] == "failed":
            last_failed.add(r["cmd"])
        elif r["event"] == "command" and r["cmd"] in last_failed:
            retries += 1
            last_failed.discard(r["cmd"])
    by: dict = {}
    for f in fails:
        k = f"{f['error']} in {f.get('where') or 'unknown'}"
        by[k] = by.get(k, 0) + 1
    return {"commands": len(cmds), "first_command_ts": first, "first_report_ts": reports[0]["ts"] if reports else None,
            "minutes_to_first_report": round((reports[0]["ts"] - first) / 60, 1) if reports else None,
            "failed_commands": len(fails), "errors": by, "retries": retries,
            "config_changes": sum(1 for c in cmds if c.get("config_change")),
            "help_calls": sum(1 for c in cmds if "--help" in c.get("flags", []) or "-h" in c.get("flags", []))}
