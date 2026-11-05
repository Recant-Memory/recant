"""The eval item: five fields and an id. Everything else about an item (where
it was drafted from, by what, who reviewed it) lives in the log."""
from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ..log.hashing import item_key_hash
from ..records import is_ulid, parse_episode_id

EVAL_ITEM_FORMAT_VERSION = 1
NOTHING = "nothing relevant"
ItemType = Literal["yield", "noise", "supersession"]
_STRATUM = re.compile(r"^(\d{4}-\d{2}|unknown)$")


class ItemError(ValueError):
    pass


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Tests(_Closed):
    type: ItemType
    marker: Optional[str] = None   # None: drafted from a window no marker fired on, or a noise item
    hard: bool = False
    stratum: str

    @field_validator("stratum")
    @classmethod
    def _month(cls, v: str) -> str:
        if not _STRATUM.match(v):
            raise ValueError("stratum is the source month, YYYY-MM, or 'unknown'")
        return v


class EvalItem(_Closed):
    id: str
    question: str
    answer: str
    episode: Optional[str]
    tests: Tests
    exposure: int = Field(default=0, ge=0)

    @field_validator("id")
    @classmethod
    def _ulid(cls, v: str) -> str:
        if not is_ulid(v):
            raise ValueError("id must be a ULID")
        return v

    @field_validator("question", "answer")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v

    @model_validator(mode="after")
    def _coherent(self) -> "EvalItem":
        noise = self.tests.type == "noise"
        if noise != (self.episode is None):
            raise ValueError("episode is null exactly when the item is a noise item")
        if noise != (self.answer == NOTHING):
            raise ValueError(f"a noise item's key is {NOTHING!r}, and only a noise item's")
        if self.episode is not None:
            parse_episode_id(self.episode)
        return self

    def key_hash(self) -> str:
        return item_key_hash(self.question, self.answer, self.episode or "")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def validate_item(payload: Any) -> EvalItem:
    try:
        return EvalItem.model_validate(payload)
    except ValidationError as exc:
        raise ItemError("; ".join(f"{'.'.join(map(str, e['loc'])) or '(item)'}: {e['msg']}"
                                  for e in exc.errors())) from exc
