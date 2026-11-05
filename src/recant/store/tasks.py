"""The task ledger: one markdown file per task, every earlier state kept, beside the memory tiers.
A task is true while open; it is closed with a date, never retired."""
from __future__ import annotations

from typing import Any, Optional

from ..records import (RecordError, TaskRecord, check_task_transition, parse_task_file, path_allowed,
                       render_task_file, validate_task, check_task_dates)
from .files import atomic_write, file_lock
from .layout import LayoutError, StoreLayout
from .memory import NotFound, RejectedWrite


class TaskStore:
    def __init__(self, layout: StoreLayout, allowlist: list[str] | tuple[str, ...]):
        self.layout = layout
        self.allowlist = tuple(allowlist)

    def get(self, tenant: str, task_id: str) -> tuple[TaskRecord, str]:
        path = self.layout.task_path(tenant, task_id)
        if not path.exists():
            raise NotFound(task_id)
        return parse_task_file(path.read_text(encoding="utf-8"))

    def list(self, tenant: str, status: Optional[str] = None) -> list[dict[str, str]]:
        out = []
        d = self.layout.tasks_dir(tenant)
        for path in sorted(d.glob("*.md")) if d.exists() else []:
            record, _ = parse_task_file(path.read_text(encoding="utf-8"))
            if status is None or record.status == status:
                out.append({"id": record.id, "status": record.status, "opened": record.opened})
        return out

    def put(self, tenant: str, payload: Any, *, writer: str, log, notes: str = "") -> TaskRecord:
        raw = payload.to_dict() if isinstance(payload, TaskRecord) else payload
        try:
            if not isinstance(writer, str) or not writer.strip():
                raise RecordError("a write names its writer")
            record = validate_task(raw)
            check_task_dates(record)
            for path in record.episode_paths():
                if not path_allowed(path, self.allowlist):
                    raise RecordError(f"episode path is outside the allowlist: {path!r}")
        except (RecordError, LayoutError) as exc:
            log.emit("write_rejected", payload=raw if isinstance(raw, (dict, list, str)) else repr(raw), error=str(exc))
            raise RejectedWrite(str(exc))
        target = self.layout.task_path(tenant, record.id)
        with file_lock(self.layout.lock_path(tenant)):
            old_state = None
            if target.exists():
                old, _ = parse_task_file(target.read_text(encoding="utf-8"))
                old_state = old.status
            try:
                check_task_transition(old_state, record)
            except RecordError as exc:
                log.emit("write_rejected", payload=raw, error=str(exc))
                raise RejectedWrite(str(exc))
            if target.exists():
                self._keep_version(tenant, record.id, target.read_bytes())
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(target, render_task_file(record, notes).encode("utf-8"))
        if old_state != record.status:
            log.emit("task_transition", task=record.id, from_state=old_state, to_state=record.status, writer=writer,
                     closed=record.closed, closed_by=record.closed_by, superseded_by=record.superseded_by)
        return record

    def _keep_version(self, tenant: str, task_id: str, data: bytes) -> int:
        vdir = self.layout.versions_dir(tenant, task_id)
        vdir.mkdir(parents=True, exist_ok=True)
        existing = [int(p.stem) for p in vdir.glob("*.md")]
        n = (max(existing) if existing else 0) + 1
        atomic_write(vdir / f"{n:06d}.md", data)
        return n
