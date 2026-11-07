"""Where Recant keeps its files when you name nothing: `recant.wrap(client)` and every `recant` command use the same
store and tenant by default, so the first step needs no configuration.

- The store is `$RECANT_HOME` if set, else `./.recant/` in the working directory. It holds counts, token figures,
  timings and hashes of what was sent, never the text of a prompt, a reply or a tool result.
- The tenant is `default`.
- A store made here gets a `.gitignore` of its own (`*`), so it never lands in a repository by accident."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DEFAULT_TENANT = "default"


def default_store() -> Path:
    """The store when none is named: `$RECANT_HOME`, else `./.recant` (made absolute now, so a later chdir never moves
    it)."""
    env = os.environ.get("RECANT_HOME")
    return (Path(env).expanduser() if env else Path.cwd() / ".recant").absolute()


def store_or_default(store) -> Path | object:
    """The store a caller named, else the default one, made and git-ignored if it is new."""
    if store is not None:
        return store
    p = default_store()
    try:
        fresh = not p.exists()
        p.mkdir(parents=True, exist_ok=True)
        gi = p / ".gitignore"
        if fresh or not gi.exists():
            gi.write_text("# Recant's store: counts and hashes, never text. Not for version control.\n*\n")
    except OSError:
        pass                                         # an unwritable place fails later, where the store is written
    return p


def tenant_or_default(tenant: Optional[str]) -> str:
    return tenant if tenant else DEFAULT_TENANT
