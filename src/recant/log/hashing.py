"""Hashes that make a logged verdict comparable later."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def item_key_hash(question: str, answer: str, episode: str) -> str:
    """One eval item's answer key. Per item, so fixing one key stales only
    that item's verdicts."""
    return sha256_hex(canonical_json({"question": question, "answer": answer, "episode": episode}).encode())


def request_hash(request: Any) -> str:
    return sha256_hex(canonical_json(request).encode())


def file_hash(path: str | Path) -> str:
    return sha256_hex(Path(path).read_bytes())
