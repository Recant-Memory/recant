"""The cold tier: source episodes, copied once through an allowlist, never edited."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..records import RecordError, parse_episode_id, path_allowed
from .files import atomic_write
from .layout import StoreLayout

MANIFEST = "MANIFEST.json"


class IngestConflict(RuntimeError):
    """A file already in the cold tier differs from the corpus. Cold is an archive."""


@dataclass
class IngestResult:
    copied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.copied) + len(self.skipped)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ingest(layout: StoreLayout, tenant: str, corpus_root: str | Path,
           allowlist: list[str] | tuple[str, ...]) -> IngestResult:
    """Copy exactly what the allowlist admits into cold/, and nothing else.

    The allowlist is a list of positive glob patterns relative to the corpus
    root. Only the patterns are expanded; the corpus is never walked."""
    if not allowlist:
        raise RecordError("ingest needs an allowlist; an empty one admits nothing")
    root = Path(corpus_root)
    if not root.is_dir():
        raise FileNotFoundError(f"corpus root is not a directory: {root}")
    cold = layout.tier_dir(tenant, "cold")
    cold.mkdir(parents=True, exist_ok=True)
    manifest_path = cold / MANIFEST
    manifest: dict[str, dict] = {}
    if manifest_path.exists():
        manifest = {e["path"]: e for e in json.loads(manifest_path.read_text())["files"]}

    result = IngestResult()
    sources: dict[str, Path] = {}
    for pattern in allowlist:
        if pattern.startswith("/") or ".." in pattern.split("/"):
            raise RecordError(f"allowlist pattern must be relative: {pattern!r}")
        if "**" in pattern:
            raise RecordError(f"allowlist pattern must not recurse: {pattern!r}")
        for src in root.glob(pattern):
            if src.is_symlink() or not src.is_file():
                continue
            rel = src.relative_to(root).as_posix()
            if path_allowed(rel, allowlist):
                sources[rel] = src
    for rel in sorted(sources):
        data = sources[rel].read_bytes()
        digest = _sha256(data)
        dest = cold / rel
        if dest.exists():
            if _sha256(dest.read_bytes()) != digest:
                raise IngestConflict(f"{rel} is already archived with different content")
            result.skipped.append(rel)
        else:
            atomic_write(dest, data)
            result.copied.append(rel)
        manifest[rel] = {"path": rel, "bytes": len(data), "sha256": digest}
    body = {"files": [manifest[k] for k in sorted(manifest)]}
    atomic_write(manifest_path, (json.dumps(body, indent=1, sort_keys=True) + "\n").encode())
    return result


def read_manifest(layout: StoreLayout, tenant: str) -> list[dict]:
    path = layout.tier_dir(tenant, "cold") / MANIFEST
    return json.loads(path.read_text())["files"] if path.exists() else []


def resolve(layout: StoreLayout, tenant: str, episode_id: str) -> str:
    """The text of a span, read from cold/."""
    ref = parse_episode_id(episode_id)
    path = layout.episode_file(tenant, ref.path)
    if not path.is_file():
        raise FileNotFoundError(f"no such episode in the hot or cold tier: {ref.path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if ref.end > len(lines):
        raise RecordError(f"span ends at line {ref.end}; {ref.path} has {len(lines)}")
    return "\n".join(lines[ref.start - 1:ref.end])


# The hot tier: the consumer's live sessions, written through Recant.
from .hot import (EPISODE_FORMAT_VERSION, HOT_PATTERNS, Line, SessionWriter, derive_daily, open_session, read_lines,  # noqa: E402
                  read_session, render_user, session_paths)
