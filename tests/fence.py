"""Everything live is fenced off for the whole test session, not call by call: a test must never reach a real store,
a real agent or a real credential. Installed once by each suite's conftest:

- no credential in the environment: any variable that names a key, token, secret or password is removed, so a client
  that reaches for one fails at once, and so does any process a test starts, and no process is handed a removed value;
  the one exception is the live tests' own target (RECANT_LIVE_TESTS=1), which keeps the model key and nothing else;
- no live path: a store or file under a path the consumer names is refused when it is opened;
- no service manager: a process named launchctl or systemctl is refused before it starts.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

CREDENTIAL = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)", re.I)
REFUSED_PROGRAMS = ("launchctl", "systemctl")


class LiveAccess(AssertionError):
    """A test reached for something live."""


LIVE_OPT_IN = "RECANT_LIVE_TESTS"     # set to 1 by the one target that runs the live tests on purpose; it keeps that key only


def install(mp, *, live_paths: tuple = (), extra_env: tuple = (), store_env: dict | None = None, live_key: str = "ANTHROPIC_API_KEY") -> None:
    keep = {live_key} if os.environ.get(LIVE_OPT_IN) == "1" else set()
    removed = {os.environ[n] for n in os.environ if (CREDENTIAL.search(n) or n in extra_env) and n not in keep and os.environ[n]}
    for name in list(os.environ):
        if (CREDENTIAL.search(name) or name in extra_env) and name not in keep:
            mp.delenv(name, raising=False)
    for k, v in (store_env or {}).items():
        mp.setenv(k, str(v))
    live = tuple(Path(p).resolve() for p in live_paths)

    def is_live(path) -> bool:
        try:
            p = Path(path).resolve()
        except (TypeError, OSError):
            return False
        return any(p == root or root in p.parents for root in live)

    from recant.store import layout as _layout
    real_init = _layout.StoreLayout.__init__

    def guarded_init(self, root, *a, **k):
        if is_live(root):
            raise LiveAccess(f"a test opened a live store: {root}")
        return real_init(self, root, *a, **k)
    mp.setattr(_layout.StoreLayout, "__init__", guarded_init)

    real_popen_init = subprocess.Popen.__init__

    def guarded_popen(self, args, *a, **k):
        argv = [args] if isinstance(args, (str, bytes)) else list(args)
        text = " ".join(str(x) for x in argv)
        first = Path(str(argv[0]).split()[0]).name if argv else ""
        if first in REFUSED_PROGRAMS or any(f" {p} " in f" {text} " or text.startswith(p) for p in REFUSED_PROGRAMS):
            raise LiveAccess(f"a test tried to run {first or text[:40]}")
        env = k.get("env")
        if env and any(is_live(v) for key, v in env.items() if key.endswith("_STORE")):
            raise LiveAccess("a test started a process pointed at a live store")
        if env and any(v in removed for v in env.values() if v):          # a made-up key in a fake is fine; a real one is not
            raise LiveAccess("a test passed a real credential to a process")
        return real_popen_init(self, args, *a, **k)
    mp.setattr(subprocess.Popen, "__init__", guarded_popen)
