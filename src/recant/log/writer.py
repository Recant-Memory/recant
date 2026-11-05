"""Append-only, one file per run. A run that ends says so, whatever it did."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from ulid import ULID

from ..store.layout import StoreLayout
from .records import LOG_FORMAT_VERSION, RECORD_KINDS

FSYNC_EVERY = 100


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class LogWriter:
    def __init__(self, layout: StoreLayout, tenant: str, job: str, *, artifact_hash: str,
                 kernel_hash: str, params: Optional[dict[str, Any]] = None,
                 clock: Callable[[], str] = _now):
        self.layout, self.tenant, self.job = layout, tenant, job
        self.artifact_hash, self.kernel_hash = artifact_hash, kernel_hash
        self.run = str(ULID())
        self.examined = 0
        self.promoted = 0
        self.spent_usd = 0.0          # the priced calls' total; reported as None once a call had no price
        self.unpriced_calls = 0
        self.counts: dict[str, int] = {}
        self._clock = clock
        self._seq = 0
        self._lock = threading.Lock()
        self._closed = False
        log_dir = layout.log_dir(tenant)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"{self.run}.jsonl"
        self._fh = open(self.path, "a", encoding="utf-8")
        self.emit("run_started", job=job, params=params or {})

    @property
    def batches_dir(self):
        return self.layout.batches_dir(self.tenant)

    def emit(self, kind: str, **fields: Any) -> dict[str, Any]:
        model = RECORD_KINDS.get(kind)
        if model is None:
            raise ValueError(f"unknown log record kind: {kind!r}")
        with self._lock:
            if self._closed:
                raise RuntimeError("this run has finished; its log is closed")
            record = model(v=LOG_FORMAT_VERSION, ts=self._clock(), tenant=self.tenant, run=self.run,
                           seq=self._seq + 1, artifact=self.artifact_hash, kernel=self.kernel_hash,
                           **fields)                              # a record that does not validate takes no sequence number
            self._seq += 1
            line = record.model_dump(mode="json", by_alias=True)
            self._fh.write(json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n")
            self._fh.flush()
            if self._seq % FSYNC_EVERY == 0:
                os.fsync(self._fh.fileno())
            return line

    def validate(self, kind: str, **fields: Any) -> None:
        """Build the record `emit` would write and raise if it does not validate, writing nothing (a
        caller that writes an entry and then logs it checks the record first, so the entry never lands unlogged)."""
        model = RECORD_KINDS.get(kind)
        if model is None:
            raise ValueError(f"unknown log record kind: {kind!r}")
        if self._closed:
            raise RuntimeError("this run has finished; its log is closed")
        model(v=LOG_FORMAT_VERSION, ts=self._clock(), tenant=self.tenant, run=self.run, seq=self._seq + 1,
              artifact=self.artifact_hash, kernel=self.kernel_hash, **fields)

    def add_cost(self, cost_usd: Optional[float]) -> None:
        """One model call's cost, or None when the consumer gave no price for it."""
        with self._lock:
            if cost_usd is None:
                self.unpriced_calls += 1
            else:
                self.spent_usd += cost_usd

    @property
    def reported_spend_usd(self) -> Optional[float]:
        return None if self.unpriced_calls else round(self.spent_usd, 6)

    def count(self, name: str, n: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + n

    def finish(self, stopped_by: str = "done", error: Optional[str] = None) -> None:
        if self._closed:
            return
        self.emit("run_finished", examined=self.examined, promoted=self.promoted,
                  spent_usd=self.reported_spend_usd, stopped_by=stopped_by, error=error,
                  counts=dict(self.counts))
        with self._lock:
            os.fsync(self._fh.fileno())
            self._fh.close()
            self._closed = True

    def __enter__(self) -> "LogWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            self.finish("done")
        else:
            self.finish(getattr(exc, "stopped_by", "error"), error=f"{exc_type.__name__}: {exc}")
        return False
