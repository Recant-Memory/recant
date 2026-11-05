"""Which Recant this is: the version and the commit, for `recant --version` and every report and
bundle header. Installed from git (`pip install git+…@<tag>`), pip keeps the commit in the install record; a build that
writes `_commit` into the package carries the commit it was built from; from a source checkout, git is asked; otherwise
the commit is "unknown", and every header says so in words."""
from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path

from . import __version__


@lru_cache(maxsize=1)
def info() -> dict:
    commit, source = None, "unknown"
    try:
        import importlib.metadata as md
        raw = md.distribution("recant").read_text("direct_url.json")
        d = json.loads(raw) if raw else {}
        commit = (d.get("vcs_info") or {}).get("commit_id")
        if commit:
            source = "install record"
    except Exception:                                                 # noqa: BLE001 — not installed as a distribution
        pass
    here = Path(__file__).resolve().parent
    if not commit and (here / "_commit").is_file():
        built = (here / "_commit").read_text().strip()
        if len(built) == 40 and all(ch in "0123456789abcdef" for ch in built):
            commit, source = built, "release build"
    if not commit:
        try:
            p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=5)
            if p.returncode == 0 and p.stdout.strip():
                commit, source = p.stdout.strip(), "source checkout"
                dirty = subprocess.run(["git", "status", "--porcelain", "--", str(here.parent)], cwd=here, capture_output=True,
                                       text=True, timeout=5)
                if dirty.returncode == 0 and dirty.stdout.strip():
                    source = "source checkout, uncommitted changes"
        except (OSError, subprocess.TimeoutExpired):
            pass
    return {"version": __version__, "commit": commit or "unknown", "commit_from": source}


def describe(i: dict) -> str:
    """A version record as one phrase: `0.3.2 (1cdf63996a1e, release build)`."""
    if i.get("commit") in (None, "unknown"):
        return f"{i['version']} (commit not recorded in this install)"
    return f"{i['version']} ({i['commit'][:12]}, {i['commit_from']})"


def line() -> str:
    return "recant " + describe(info())
