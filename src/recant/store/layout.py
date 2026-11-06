"""Where everything lives on disk. Tenant scoping is a path boundary."""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..records import MEMORY_FORMAT_VERSION, is_ulid

STORE_FORMAT = {"layout": 1, "memory": MEMORY_FORMAT_VERSION, "log": 1}
TIERS = ("hot", "warm", "cold", "index")
EVAL_POOLS = ("founding", "heldout", "reserve", "failures")
DRAFT_POOL = "drafts"
SURPLUS_POOL = "surplus"   # drafts set aside unreviewed; kept, never in the running for founding until reviewed
HELD_POOL = "held"   # drafts that may not enter the pool until a person clears them

_TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


class LayoutError(ValueError):
    pass


def check_tenant(tenant: str) -> str:
    if not isinstance(tenant, str) or not _TENANT_RE.match(tenant):
        raise LayoutError(f"tenant id must be one safe path segment: {tenant!r}")
    return tenant


def check_entry_id(entry_id: str) -> str:
    if not is_ulid(entry_id):
        raise LayoutError(f"entry id must be a ULID: {entry_id!r}")
    return entry_id


class StoreLayout:
    def __init__(self, root: str | Path):
        self.root = Path(root).absolute()        # fixed when made: a later chdir never moves the store

    # tenant-scoped -----------------------------------------------------
    def tenant_dir(self, tenant: str) -> Path:
        return self.root / "tenants" / check_tenant(tenant)

    def tier_dir(self, tenant: str, tier: str) -> Path:
        if tier not in TIERS:
            raise LayoutError(f"unknown tier: {tier!r}")
        return self.tenant_dir(tenant) / tier

    def entry_path(self, tenant: str, entry_id: str) -> Path:
        return self.tier_dir(tenant, "warm") / f"{check_entry_id(entry_id)}.md"

    def hot_session_dir(self, tenant: str, day: str) -> Path:
        """Live sessions of one day: `hot/sessions/<day>/<ulid>.md`."""
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", day):
            raise LayoutError(f"not a day: {day!r}")
        return self.tier_dir(tenant, "hot") / "sessions" / day

    def hot_session_path(self, tenant: str, day: str, session_id: str) -> Path:
        return self.hot_session_dir(tenant, day) / f"{check_entry_id(session_id)}.md"

    def hot_daily_path(self, tenant: str, day: str) -> Path:
        """The day's sessions concatenated by a function at the cut-off: `hot/<day>.md`."""
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", day):
            raise LayoutError(f"not a day: {day!r}")
        return self.tier_dir(tenant, "hot") / f"{day}.md"

    def episode_file(self, tenant: str, path: str) -> Path:
        """Where an episode path lives: a `hot/...` path is the tenant's hot tier, anything else is the cold
        archive. Every reader of episode text goes through here, so the live session files read
        exactly as the corpus does."""
        if path.startswith("hot/"):
            return self.tenant_dir(tenant) / path
        return self.tier_dir(tenant, "cold") / path

    def tasks_dir(self, tenant: str) -> Path:
        """The task ledger: a second store beside the tiers."""
        return self.tenant_dir(tenant) / "tasks"

    def task_path(self, tenant: str, task_id: str) -> Path:
        return self.tasks_dir(tenant) / f"{check_entry_id(task_id)}.md"

    def versions_dir(self, tenant: str, entry_id: str) -> Path:
        return self.root / "versions" / check_tenant(tenant) / check_entry_id(entry_id)

    def notes_dir(self, tenant: str) -> Path:
        """Notes mode's files: the agent's structured notes, what was refused, and the mode."""
        return self.tenant_dir(tenant) / "notes"

    def log_dir(self, tenant: str) -> Path:
        return self.root / "log" / check_tenant(tenant)

    def batches_dir(self, tenant: str) -> Path:
        return self.log_dir(tenant) / "batches"

    def eval_dir(self, tenant: str) -> Path:
        return self.root / "eval" / check_tenant(tenant)

    def costshape_dir(self, tenant: str) -> Path:
        """The cost advisor's files: the overlay Recant reads, the advice ledger, the reports.
        Operational data, like the log: never a second source of truth for memory."""
        return self.root / "costshape" / check_tenant(tenant)

    def lock_path(self, tenant: str) -> Path:
        return self.tenant_dir(tenant) / ".lock"

    # store-wide ----------------------------------------------------------
    @property
    def machinery_dir(self) -> Path:
        return self.root / "machinery"

    @property
    def format_path(self) -> Path:
        return self.root / "FORMAT.json"

    def ensure(self, tenant: str) -> None:
        """Create the directories for one tenant. Idempotent."""
        for tier in TIERS:
            self.tier_dir(tenant, tier).mkdir(parents=True, exist_ok=True)
        self.tasks_dir(tenant).mkdir(parents=True, exist_ok=True)
        (self.root / "versions" / tenant).mkdir(parents=True, exist_ok=True)
        self.batches_dir(tenant).mkdir(parents=True, exist_ok=True)
        self.eval_dir(tenant).mkdir(parents=True, exist_ok=True)
        for pool in EVAL_POOLS:
            (self.eval_dir(tenant) / f"{pool}.jsonl").touch(exist_ok=True)
        self.machinery_dir.mkdir(parents=True, exist_ok=True)
        if not self.format_path.exists():
            self.format_path.write_text(json.dumps(STORE_FORMAT, sort_keys=True) + "\n")
        else:
            found = json.loads(self.format_path.read_text())
            if {k: found.get(k) for k in STORE_FORMAT} != STORE_FORMAT:
                raise LayoutError(f"store format {found} is not the format this library writes {STORE_FORMAT}")
