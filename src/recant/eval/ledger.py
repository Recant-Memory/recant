"""What has happened to each item, folded from the log. The log is the record;
this is a read of it, rebuilt whenever a log file grows."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..log.reader import read_run
from ..store.layout import StoreLayout

_KINDS = {"item_drafted", "review_decision", "blind_answer", "blind_revealed", "blind_sample",
          "judge_calibration", "founding"}
_cache: dict[str, tuple[int, list[dict]]] = {}


@dataclass
class ItemState:
    item: str
    drafted: Optional[dict[str, Any]] = None
    decisions: list[dict[str, Any]] = field(default_factory=list)
    blind_answer: Optional[dict[str, Any]] = None
    revealed_at: Optional[str] = None
    calibrations: list[dict[str, Any]] = field(default_factory=list)

    def latest(self, reviewer_kind: str) -> Optional[dict[str, Any]]:
        for d in reversed(self.decisions):
            if d["reviewer"]["kind"] == reviewer_kind:
                return d
        return None

    def effective(self) -> Optional[dict[str, Any]]:
        """A person's decision outranks a model's; within each, the latest stands."""
        return self.latest("human") or self.latest("model")

    def accepted(self) -> bool:
        d = self.effective()
        return d is not None and d["outcome"] in ("accept", "fix")


class Ledger:
    def __init__(self, layout: StoreLayout, tenant: str):
        self.items: dict[str, ItemState] = {}
        self.blind_samples: list[dict[str, Any]] = []
        self.founding: Optional[dict[str, Any]] = None
        for path in sorted(layout.log_dir(tenant).glob("*.jsonl")):
            for rec in self._records(path):
                self._apply(rec)

    @staticmethod
    def _records(path) -> list[dict]:
        size = path.stat().st_size
        hit = _cache.get(str(path))
        if hit and hit[0] == size:
            return hit[1]
        recs = [r for r in read_run(path, typed=False) if r["kind"] in _KINDS]
        _cache[str(path)] = (size, recs)
        return recs

    def state(self, item: str) -> ItemState:
        return self.items.setdefault(item, ItemState(item))

    def _apply(self, r: dict) -> None:
        k = r["kind"]
        if k == "item_drafted":
            self.state(r["item"]).drafted = r
        elif k == "review_decision" and r["subject"]["type"] == "eval_item":
            self.state(r["subject"]["id"]).decisions.append(r)
        elif k == "blind_answer":
            st = self.state(r["item"])
            st.blind_answer = st.blind_answer or r  # the first answer is the blind one
        elif k == "blind_revealed":
            st = self.state(r["item"])
            st.revealed_at = st.revealed_at or r["ts"]
        elif k == "judge_calibration":
            self.state(r["item"]).calibrations.append(r)
        elif k == "blind_sample":
            self.blind_samples.append(r)
        elif k == "founding":
            self.founding = r

    @property
    def blind_sample(self) -> Optional[dict[str, Any]]:
        return self.blind_samples[-1] if self.blind_samples else None

    def in_blind_sample(self, item: str) -> bool:
        return bool(self.blind_sample) and item in self.blind_sample["items"]
