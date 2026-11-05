"""The public formats, one JSON Schema file each, generated from the record
models and pinned in the package. `python -m recant.schemas` rewrites them; a
test fails if a model changes and its file does not."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from pydantic import TypeAdapter

from ..eval.items import EVAL_ITEM_FORMAT_VERSION, EvalItem
from ..log.records import LOG_FORMAT_VERSION, RECORD_KINDS
from ..machinery.loader import MACHINERY_FORMAT_VERSION, Artifact, Kernel
from ..records import MEMORY_FORMAT_VERSION, TASK_FORMAT_VERSION, MemoryRecord, TaskRecord

HERE = Path(__file__).parent


def build() -> dict[str, dict[str, Any]]:
    log_union = TypeAdapter(Union[tuple(RECORD_KINDS.values())])
    specs = {
        "memory-record": (MEMORY_FORMAT_VERSION, MemoryRecord.model_json_schema(by_alias=True)),
        "task-record": (TASK_FORMAT_VERSION, TaskRecord.model_json_schema(by_alias=True)),
        "eval-item": (EVAL_ITEM_FORMAT_VERSION, EvalItem.model_json_schema()),
        "log-record": (LOG_FORMAT_VERSION, log_union.json_schema(by_alias=True)),
        "machinery-artifact": (MACHINERY_FORMAT_VERSION, Artifact.model_json_schema()),
        "machinery-kernel": (MACHINERY_FORMAT_VERSION, Kernel.model_json_schema()),
    }
    return {name: {"x-format": name, "x-format-version": version, **schema}
            for name, (version, schema) in specs.items()}


def render(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def write_all() -> list[Path]:
    out = []
    for name, schema in build().items():
        path = HERE / f"{name}.schema.json"
        path.write_text(render(schema))
        out.append(path)
    return out
