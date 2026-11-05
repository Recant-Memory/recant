"""Read a run back. Unknown kinds pass through, so an older reader can read a newer log."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional

from ..store.layout import StoreLayout
from .records import ENVELOPE_FIELDS, RECORD_KINDS


def read_run(path: str | Path, *, kind: Optional[str] = None, typed: bool = True) -> Iterator[Any]:
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            data = json.loads(line)
            missing = [f for f in ENVELOPE_FIELDS if f not in data]
            if missing:
                raise ValueError(f"{path}:{n} lacks envelope fields {missing}")
            if kind is not None and data["kind"] != kind:
                continue
            model = RECORD_KINDS.get(data["kind"]) if typed else None
            yield model.model_validate(data) if model else data


def list_runs(layout: StoreLayout, tenant: str) -> list[dict[str, Any]]:
    out = []
    for path in sorted(layout.log_dir(tenant).glob("*.jsonl")):
        first = last = None
        for rec in read_run(path, typed=False):
            first = first or rec
            last = rec
        if first is None:
            continue
        finished = last["kind"] == "run_finished"
        out.append({"run": path.stem, "job": first.get("job"), "started": first["ts"],
                    "finished": finished, "stopped_by": last.get("stopped_by") if finished else None,
                    "records": last["seq"]})
    return out
