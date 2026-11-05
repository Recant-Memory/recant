"""The pools on disk, and the one door to them.

The held-out slice and its refill reserve exist to detect overfitting. Nothing
that produces or scores a candidate may read them, so every read names its
purpose and a guarded pool opens for very few. This is a guard in code, not a
security boundary: the files are on disk."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..store.files import atomic_write, file_lock
from ..store.layout import DRAFT_POOL, EVAL_POOLS, HELD_POOL, SURPLUS_POOL, StoreLayout
from .items import EvalItem, validate_item

POOLS = (*EVAL_POOLS, DRAFT_POOL, HELD_POOL, SURPLUS_POOL)
GUARDED = frozenset({"heldout", "reserve"})
PURPOSES = frozenset({"drafting", "model_review", "human_review", "audit", "founding", "scoring",
                      "overfitting_check"})
MAY_OPEN_GUARDED = frozenset({"overfitting_check", "human_review", "founding"})


class GuardError(PermissionError):
    pass


def pool_path(layout: StoreLayout, tenant: str, pool: str) -> Path:
    if pool not in POOLS:
        raise ValueError(f"unknown pool: {pool!r}")
    return layout.eval_dir(tenant) / f"{pool}.jsonl"


def load(layout: StoreLayout, tenant: str, pool: str, purpose: str) -> list[EvalItem]:
    if purpose not in PURPOSES:
        raise GuardError(f"unknown purpose {purpose!r}; a read of the eval set says what it is for")
    if pool in GUARDED and purpose not in MAY_OPEN_GUARDED:
        raise GuardError(f"the {pool} pool does not open for {purpose!r}")
    path = pool_path(layout, tenant, pool)
    if not path.exists():
        return []
    return [validate_item(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _write(path: Path, items: Iterable[EvalItem]) -> None:
    body = "".join(json.dumps(i.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for i in items)
    atomic_write(path, body.encode("utf-8"))


def write_pool(layout: StoreLayout, tenant: str, pool: str, items: list[EvalItem]) -> None:
    with file_lock(layout.lock_path(tenant)):
        _write(pool_path(layout, tenant, pool), items)


def append_draft(layout: StoreLayout, tenant: str, item: EvalItem, pool: str = DRAFT_POOL) -> None:
    if pool not in (DRAFT_POOL, HELD_POOL):
        raise ValueError("drafts are appended to the drafting pool or the held pool")
    with file_lock(layout.lock_path(tenant)):
        path = pool_path(layout, tenant, pool)
        items = [validate_item(json.loads(l)) for l in path.read_text().splitlines() if l.strip()] \
            if path.exists() else []
        if any(i.id == item.id for i in items):
            raise ValueError(f"draft {item.id} already exists")
        _write(path, [*items, item])


def release_held(layout: StoreLayout, tenant: str, item_id: str, *, into_pool: bool) -> EvalItem:
    """Take one item out of the held pool: into the drafting pool if a person cleared it, nowhere if not."""
    with file_lock(layout.lock_path(tenant)):
        hpath = pool_path(layout, tenant, HELD_POOL)
        held = [validate_item(json.loads(l)) for l in hpath.read_text().splitlines() if l.strip()] if hpath.exists() else []
        item = next((i for i in held if i.id == item_id), None)
        if item is None:
            raise KeyError(f"{item_id} is not held")
        _write(hpath, [i for i in held if i.id != item_id])
        if into_pool:
            dpath = pool_path(layout, tenant, DRAFT_POOL)
            drafts = [validate_item(json.loads(l)) for l in dpath.read_text().splitlines() if l.strip()] if dpath.exists() else []
            _write(dpath, [*drafts, item])
    return item


def _read(path: Path) -> list[EvalItem]:
    return [validate_item(json.loads(l)) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def move_between(layout: StoreLayout, tenant: str, item_ids: Iterable[str], src: str, dst: str | None) -> int:
    """Move drafts between the working pools (drafts, held, surplus), or out altogether (dst None)."""
    working = (DRAFT_POOL, HELD_POOL, SURPLUS_POOL)
    if src not in working or (dst is not None and dst not in working):
        raise ValueError("only the working pools are rearranged; a founded pool's membership never changes")
    ids = set(item_ids)
    with file_lock(layout.lock_path(tenant)):
        spath = pool_path(layout, tenant, src)
        items = _read(spath)
        moving = [i for i in items if i.id in ids]
        _write(spath, [i for i in items if i.id not in ids])
        if dst is not None and moving:
            dpath = pool_path(layout, tenant, dst)
            _write(dpath, [*_read(dpath), *moving])
    return len(moving)


def save_draft(layout: StoreLayout, tenant: str, item: EvalItem, pool: str = DRAFT_POOL) -> None:
    """Replace one draft in place (a fix, a hard-case tag, a merged marker)."""
    _replace(layout, tenant, pool, item, keep_history=False)


def save_founded(layout: StoreLayout, tenant: str, pool: str, item: EvalItem) -> int:
    """Correct a founded item's key. Membership never changes; the prior line is kept."""
    return _replace(layout, tenant, pool, item, keep_history=True)


def _replace(layout, tenant, pool, item, *, keep_history: bool) -> int:
    with file_lock(layout.lock_path(tenant)):
        path = pool_path(layout, tenant, pool)
        items = [validate_item(json.loads(l)) for l in path.read_text().splitlines() if l.strip()]
        idx = next((n for n, i in enumerate(items) if i.id == item.id), None)
        if idx is None:
            raise KeyError(f"{item.id} is not in {pool}")
        n = 0
        if keep_history:
            hdir = layout.eval_dir(tenant) / "history" / item.id
            hdir.mkdir(parents=True, exist_ok=True)
            n = len(list(hdir.glob("*.json"))) + 1
            atomic_write(hdir / f"{n:06d}.json",
                         (json.dumps(items[idx].to_dict(), ensure_ascii=False, sort_keys=True) + "\n").encode())
        items[idx] = item
        _write(path, items)
        return n


def find_founded(layout: StoreLayout, tenant: str, item_id: str) -> str | None:
    """Which founded pool holds this item. For the human review routes only."""
    for pool in ("founding", "heldout", "reserve"):
        if any(i.id == item_id for i in load(layout, tenant, pool, "human_review")):
            return pool
    return None


def increment_exposure(layout: StoreLayout, tenant: str, pool: str, log, candidate: str) -> int:
    """A candidate run has seen every item in the pool once more. The count is on the item and in
    the log, per item; baseline arms and judge calibration never call this."""
    with file_lock(layout.lock_path(tenant)):
        path = pool_path(layout, tenant, pool)
        items = [validate_item(json.loads(l)) for l in path.read_text().splitlines() if l.strip()]
        bumped = [validate_item({**i.to_dict(), "exposure": i.exposure + 1}) for i in items]
        _write(path, bumped)
    for i in bumped:
        log.emit("exposure", item=i.id, candidate=candidate, count_after=i.exposure)
    return len(bumped)
