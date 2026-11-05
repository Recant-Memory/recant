"""`recant off`, `recant on`, `recant status`: one switch, read by the wrapper on every call, so it
takes effect on the next call with no code change and no restart.

- `on` (the default): the wrapper meters every call and carries out the applied safe levers.
- `off`: a pure pass-through. The request is the host's own, byte for byte (Recant's own `session=` argument is the one
  thing taken out, since the SDK does not know it); no cache markers, no notes block, no tools added; nothing is
  written to the store or the log.
- `meter` (`recant off --meter`): metering only. The request is still the host's own, byte for byte.

The setting lives in the store (`recant.switch.json` at its root). `RECANT=off`, `RECANT=meter` or `RECANT=on` in the
environment wins over the file."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

STATES = ("on", "off", "meter")
FILE = "recant.switch.json"
ENV = "RECANT"


def _path(root) -> Path:
    return Path(getattr(root, "root", root)) / FILE


def state(root) -> str:
    """The switch for the store at `root` (a path or a StoreLayout): the environment's, else the file's, else on."""
    env = (os.environ.get(ENV) or "").strip().lower()
    if env in STATES:
        return env
    try:
        s = json.loads(_path(root).read_text(encoding="utf-8")).get("state")
    except (OSError, ValueError):
        return "on"
    return s if s in STATES else "on"


def set_state(root, new: str) -> dict:
    if new not in STATES:
        raise ValueError(f"the switch is one of {STATES}, not {new!r}")
    p = _path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"state": new, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    tmp = p.with_name(f".{p.name}.{os.getpid()}.{threading.get_ident()}")      # concurrent writers never share a temp file
    tmp.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    tmp.replace(p)
    return rec


def describe(root) -> dict:
    """What `recant status` prints: the state in force, and where it comes from."""
    env = (os.environ.get(ENV) or "").strip().lower()
    try:
        file_rec: Optional[dict] = json.loads(_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        file_rec = None
    return {"state": state(root), "from": f"environment {ENV}={env}" if env in STATES else (str(_path(root)) if file_rec else "default"),
            "file": file_rec}
