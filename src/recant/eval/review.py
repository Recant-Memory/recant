"""One decision function for every reviewer. A person and a model go through
the same door and leave the same record; the record says which."""
from __future__ import annotations

from typing import Optional

from ..store.layout import StoreLayout
from .items import NOTHING, EvalItem, ItemError, validate_item
from .ledger import Ledger
from .pools import find_founded, load, save_draft, save_founded

OUTCOMES = ("accept", "fix", "discard")


class ReviewError(ValueError):
    pass


class BlindFirst(RuntimeError):
    """The item is in the blind-first sample and the reviewer has not yet answered blind."""


def _find(layout, tenant, item_id) -> tuple[str, EvalItem]:
    """A founded item is the one in its founded pool; the copy left in the drafting pool is history."""
    pool = find_founded(layout, tenant, item_id)
    if pool:
        return pool, next(i for i in load(layout, tenant, pool, "human_review") if i.id == item_id)
    for it in load(layout, tenant, "drafts", "human_review"):
        if it.id == item_id:
            return "drafts", it
    raise KeyError(item_id)


BLIND_CHOICES = {"answers": "The highlight answers it.", "does_not_answer": "The record does not answer this.",
                 "partially": "The highlight partially answers it."}


def record_blind_answer(layout: StoreLayout, tenant: str, log, item_id: str, reviewer: dict,
                        text: Optional[str] = None, choice: Optional[str] = None) -> None:
    """A blind answer is one key or some words. A key is recorded as its label, verbatim, with the
    choice beside it, so an audit can count the choices without reading prose."""
    if reviewer.get("kind") != "human":
        raise ReviewError("a blind answer is a person's")
    if choice is not None:
        if choice not in BLIND_CHOICES:
            raise ReviewError(f"choice is one of {sorted(BLIND_CHOICES)}")
        if text and text.strip():
            raise ReviewError("a blind answer is a key or words, not both")
        text = BLIND_CHOICES[choice]
    if not isinstance(text, str) or not text.strip():
        raise ReviewError("a blind answer has text")
    ledger = Ledger(layout, tenant)
    if not ledger.in_blind_sample(item_id):
        raise ReviewError("this item is not in the blind-first sample")
    if ledger.state(item_id).blind_answer:
        raise ReviewError("a blind answer is written once")
    log.emit("blind_answer", item=item_id, reviewer=reviewer, text=text, choice=choice)


CITATION = "citation"   # a fix that narrows the cited span or trims the key to it: the key was right, its citation was not


def decide(layout: StoreLayout, tenant: str, log, item_id: str, *, outcome: str, reason: str,
           reviewer: dict, reasons: list[str], note: Optional[str] = None,
           fixed_answer: Optional[str] = None, fixed_episode: Optional[str] = None,
           hard: Optional[bool] = None) -> dict:
    if outcome not in OUTCOMES:
        raise ReviewError(f"outcome is one of {OUTCOMES}")
    if reason not in reasons:
        raise ReviewError(f"reason is one of {reasons}")
    pool, item = _find(layout, tenant, item_id)
    human = reviewer.get("kind") == "human"
    ledger = Ledger(layout, tenant)
    st = ledger.state(item_id)
    blind_text = revealed = None
    if human and ledger.in_blind_sample(item_id) and not st.latest("human"):
        if not st.blind_answer:
            raise BlindFirst("write the blind answer before deciding")
        if not st.revealed_at:
            raise BlindFirst("reveal the candidate before deciding; a decision made without seeing it compares nothing")
        blind_text, revealed = st.blind_answer["text"], st.revealed_at
    if pool != "drafts":
        if not human:
            raise ReviewError("no model may change a founded item")
        if outcome == "discard":
            raise ReviewError("a founded item's membership never changes; its key can be fixed")
    before = after = item.key_hash()
    updated = item
    if outcome == "fix":
        if not (fixed_answer and fixed_answer.strip()) and not fixed_episode:
            raise ReviewError("a fix carries the corrected answer, or the corrected span")
        if item.tests.type == "noise" or (fixed_answer or "").strip() == NOTHING:
            raise ReviewError("a noise key cannot be fixed into an answer, nor an answer into a noise key; "
                              "discard the item for a wrong answer")
        if fixed_episode and not human:
            raise ReviewError("only a person changes what an item cites")
        updated = validate_item({**item.to_dict(), "answer": fixed_answer or item.answer,
                                 "episode": fixed_episode or item.episode})
        after = updated.key_hash()
    if hard is not None and hard != updated.tests.hard:
        updated = validate_item({**updated.to_dict(), "tests": {**updated.to_dict()["tests"], "hard": hard}})
    decision = dict(subject={"type": "eval_item", "id": item_id}, outcome=outcome,
                    reason=reason, note=note, reviewer=reviewer, blind_answer=blind_text,
                    revealed_at=revealed, fixed_answer=(fixed_answer or item.answer) if outcome == "fix" else None,
                    fixed_episode=fixed_episode if outcome == "fix" else None,
                    key_hash_before=before, key_hash_after=after, hard=hard)
    log.validate("review_decision", **decision)      # checked before the item is saved
    if updated != item:
        try:
            save_draft(layout, tenant, updated) if pool == "drafts" else save_founded(layout, tenant, pool, updated)
        except ItemError as exc:
            raise ReviewError(str(exc)) from exc
    return log.emit("review_decision", **decision)
