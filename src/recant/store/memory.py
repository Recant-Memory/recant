"""The warm tier: one markdown file per entry, every earlier state kept."""
from __future__ import annotations

from typing import Any, Optional

from ulid import ULID

from ..records import (MemoryRecord, RecordError, check_transition, parse_memory_file,
                       path_allowed, render_memory_file, validate_record)
from .files import atomic_write, file_lock
from .layout import LayoutError, StoreLayout, check_entry_id

REDACTABLE = ("claim", "scope", "supersession_condition", "actor.id", "anchor.quote",
              "anchor.channel", "anchor.message_id", "anchor.timestamp", "status.reason", "notes")


class RejectedWrite(ValueError):
    """The write was refused. A write_rejected record has been logged."""


class NotFound(KeyError):
    pass


def new_id() -> str:
    return str(ULID())


class MemoryStore:
    def __init__(self, layout: StoreLayout, allowlist: list[str] | tuple[str, ...]):
        self.layout = layout
        self.allowlist = tuple(allowlist)

    # reads ---------------------------------------------------------------
    def get(self, tenant: str, entry_id: str) -> tuple[MemoryRecord, str]:
        path = self.layout.entry_path(tenant, entry_id)
        if not path.exists():
            raise NotFound(entry_id)
        return parse_memory_file(path.read_text(encoding="utf-8"))

    def list(self, tenant: str, state: Optional[str] = None) -> list[dict[str, str]]:
        out = []
        for path in sorted(self.layout.tier_dir(tenant, "warm").glob("*.md")):
            record, _ = parse_memory_file(path.read_text(encoding="utf-8"))
            if state is None or record.status.state == state:
                out.append({"id": record.id, "state": record.status.state})
        return out

    def versions(self, tenant: str, entry_id: str) -> list[dict[str, int]]:
        vdir = self.layout.versions_dir(tenant, entry_id)
        if not vdir.exists():
            return []
        return [{"n": int(p.stem), "bytes": p.stat().st_size} for p in sorted(vdir.glob("*.md"))]

    def read_version(self, tenant: str, entry_id: str, n: int) -> str:
        path = self.layout.versions_dir(tenant, entry_id) / f"{n:06d}.md"
        if not path.exists():
            raise NotFound(f"{entry_id} version {n}")
        return path.read_text(encoding="utf-8")

    # writes --------------------------------------------------------------
    def put(self, tenant: str, payload: Any, *, writer: str, log, notes: str = "") -> MemoryRecord:
        """Validate and write. `writer` names who is making the change; which
        writers may make which transition is the calling surface's rule."""
        raw = payload.to_dict() if isinstance(payload, MemoryRecord) else payload
        try:
            if not isinstance(writer, str) or not writer.strip():
                raise RecordError("a write names its writer")
            if not isinstance(notes, str):
                raise RecordError("notes must be text")
            record = validate_record(raw)
            for path in record.episode_paths():
                if not path_allowed(path, self.allowlist):
                    raise RecordError(f"episode path is outside the allowlist: {path!r}")
        except (RecordError, LayoutError) as exc:
            self._reject(log, raw, str(exc))
        target = self.layout.entry_path(tenant, record.id)
        with file_lock(self.layout.lock_path(tenant)):
            old_state = None
            if target.exists():
                old, _ = parse_memory_file(target.read_text(encoding="utf-8"))
                old_state = old.status.state
            try:
                check_transition(old_state, record.status)
            except RecordError as exc:
                self._reject(log, raw, str(exc))
            if target.exists():
                self._keep_version(tenant, record.id, target.read_bytes())
            atomic_write(target, render_memory_file(record, notes).encode("utf-8"))
        if old_state != record.status.state:
            log.emit("status_transition", entry=record.id, from_state=old_state,
                     to_state=record.status.state, writer=writer, reason=record.status.reason,
                     failure_class=record.status.failure_class,
                     superseded_by=record.status.superseded_by)
        return record

    def reclassify_retired_as_superseded(self, tenant: str, entry_id: str, *, superseded_by: str, reason: str,
                                         failure_class: str, writer: str, log, notes: Optional[str] = None) -> MemoryRecord:
        """The one move the lifecycle table does not have: a `retired` entry that should have been `superseded`
       . Retired stays terminal for every other writer; this is a logged correction
        of a review mistake, never a state the pass or the surface can reach."""
        record, old_notes = self.get(tenant, entry_id)
        if record.status.state != "retired":
            raise RecordError(f"only a retired entry is reclassified; {entry_id} is {record.status.state}")
        data = record.to_dict()
        data["status"] = {"state": "superseded", "reason": reason, "failure_class": failure_class, "superseded_by": superseded_by}
        new = validate_record(data)
        target = self.layout.entry_path(tenant, entry_id)
        with file_lock(self.layout.lock_path(tenant)):
            self._keep_version(tenant, entry_id, target.read_bytes())
            atomic_write(target, render_memory_file(new, notes if notes is not None else old_notes).encode("utf-8"))
        log.emit("status_transition", entry=entry_id, from_state="retired", to_state="superseded", writer=writer,
                 reason=reason, failure_class=failure_class, superseded_by=superseded_by)
        return new

    def reinstate_retired(self, tenant: str, entry_id: str, *, reason: str, writer: str, log,
                          notes: Optional[str] = None) -> MemoryRecord:
        """A `retired` entry that should still stand goes back to `probation`."""
        return self.reinstate(tenant, entry_id, to_state="probation", reason=reason, writer=writer, log=log, notes=notes)

    def reinstate(self, tenant: str, entry_id: str, *, to_state: str, reason: str, writer: str, log,
                  notes: Optional[str] = None) -> MemoryRecord:
        """The second correction of a review mistake, beside the reclassification above: a `retired` entry goes
        back to `probation`, or a `superseded` entry whose successor is gone goes to `probation` or
        `live`. A logged move the store makes on a person's ruling, never a state the pass
        or the surface can reach."""
        record, old_notes = self.get(tenant, entry_id)
        if record.status.state not in ("retired", "superseded"):
            raise RecordError(f"only a retired or superseded entry is reinstated; {entry_id} is {record.status.state}")
        if to_state not in ("probation", "live"):
            raise RecordError(f"an entry is reinstated to probation or live, not {to_state}")
        if not (reason and reason.strip()):
            raise RecordError("a reinstatement carries a reason")
        data = record.to_dict()
        data["status"] = {"state": to_state, "reason": reason, "failure_class": None, "superseded_by": None}
        new = validate_record(data)
        target = self.layout.entry_path(tenant, entry_id)
        with file_lock(self.layout.lock_path(tenant)):
            self._keep_version(tenant, entry_id, target.read_bytes())
            atomic_write(target, render_memory_file(new, notes if notes is not None else old_notes).encode("utf-8"))
        log.emit("status_transition", entry=entry_id, from_state=record.status.state, to_state=to_state, writer=writer, reason=reason)
        return new

    def delete(self, tenant: str, entry_id: str, *, writer: str, log) -> int:
        """Remove from the warm tier. The last state is kept as a version first,
        so an entry written once still survives its deletion."""
        target = self.layout.entry_path(tenant, entry_id)
        with file_lock(self.layout.lock_path(tenant)):
            if not target.exists():
                raise NotFound(entry_id)
            n = self._keep_version(tenant, entry_id, target.read_bytes())
            target.unlink()
        log.emit("entry_deleted", entry=entry_id, writer=writer, kept_as_version=n)
        return n

    def redact(self, tenant: str, entry_id: str, field: str, replacement: str, *, log) -> list[int]:
        """Replace one field in the live file and in every kept version. The
        log records that it happened and where, never what was removed."""
        if field not in REDACTABLE:
            raise RecordError(f"not a redactable field: {field!r}")
        if not isinstance(replacement, str) or not replacement.strip():
            raise RecordError("a redaction leaves a non-blank marker behind")
        check_entry_id(entry_id)
        target = self.layout.entry_path(tenant, entry_id)
        vdir = self.layout.versions_dir(tenant, entry_id)
        touched: list[int] = []
        with file_lock(self.layout.lock_path(tenant)):
            version_paths = sorted(vdir.glob("*.md")) if vdir.exists() else []
            if not target.exists() and not version_paths:
                raise NotFound(entry_id)
            for path in ([target] if target.exists() else []) + version_paths:
                record, notes = parse_memory_file(path.read_text(encoding="utf-8"))
                if field == "notes":
                    notes = replacement
                else:
                    data = record.to_dict()
                    node = data
                    *parents, leaf = field.split(".")
                    for key in parents:
                        node = node[key]
                    node[leaf] = replacement
                    record = validate_record(data)
                atomic_write(path, render_memory_file(record, notes).encode("utf-8"))
                if path is not target:
                    touched.append(int(path.stem))
        log.emit("redaction", entry=entry_id, field=field, versions=touched,
                 live_file=target.exists())
        return touched

    # internals -----------------------------------------------------------
    def _keep_version(self, tenant: str, entry_id: str, data: bytes) -> int:
        vdir = self.layout.versions_dir(tenant, entry_id)
        vdir.mkdir(parents=True, exist_ok=True)
        existing = [int(p.stem) for p in vdir.glob("*.md")]
        n = (max(existing) if existing else 0) + 1
        atomic_write(vdir / f"{n:06d}.md", data)
        return n

    @staticmethod
    def _reject(log, payload: Any, error: str):
        log.emit("write_rejected", payload=payload if isinstance(payload, (dict, list, str)) else repr(payload),
                 error=error)
        raise RejectedWrite(error)
