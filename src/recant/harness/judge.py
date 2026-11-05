"""The judge: binary, cheap, and never the deciding vote. It scores an answer
against a reference. Its agreement with a person is measured before it is trusted."""
from __future__ import annotations

from ..eval.ledger import Ledger
from ..eval.pools import load
from ..machinery import Machinery
from ..provider import ProviderError
from ..store.layout import StoreLayout
from .common import obj, parse_json, request, run_all

SCHEMA = obj({"correct": {"type": "boolean"}, "rationale": {"type": "string"}})


def judge_request(machinery: Machinery, *, question: str, reference: str, answer: str, item_type: str,
                  superseded_answer: str | None = None, custom_id: str = "j"):
    j = machinery.artifact.judge
    if j.prompt is None:
        raise ValueError("the artifact has no judge prompt")
    user = f"Item type: {item_type}\nQuestion: {question}\nReference answer: {reference}\n"
    if superseded_answer:
        user += f"Superseded answer (returning this without its correction is wrong): {superseded_answer}\n"
    user += f"Answer to score: {answer}"
    return request("judge", model=j.model, system=j.prompt, user=user, schema=SCHEMA, max_tokens=1500,
                   custom_id=custom_id)


def read_verdict(resp) -> tuple[bool | None, str | None]:
    d, err = parse_json(resp)
    if err or not isinstance(d, dict) or not isinstance(d.get("correct"), bool):
        return None, err or "the judge returned no verdict"
    return d["correct"], d.get("rationale")


def calibrate(layout: StoreLayout, tenant: str, machinery: Machinery, provider, log, *,
              batch: bool = False, wait=lambda: None) -> int:
    """Score each audited item's drafted candidate against the reviewer's own
    blind answer. Whether the judge agrees with the person is read off later."""
    ledger = Ledger(layout, tenant)
    sample = ledger.blind_sample
    if not sample:
        raise ValueError("no blind-first sample")
    items = {i.id: i for i in load(layout, tenant, "drafts", "audit")}
    j, reqs = machinery.artifact.judge, []
    for item_id in sample["items"]:
        st = ledger.state(item_id)
        # A blind answer given as a key is a label, not an answer; there is nothing for the judge to score against.
        # Re-measured whenever the judge changes: skip only what this judge model has already scored.
        if st.latest("human") and st.blind_answer and not st.blind_answer.get("choice") and st.drafted \
                and item_id in items and not any(c["judge_model_id"] == j.model for c in st.calibrations):
            reqs.append(judge_request(machinery, question=st.drafted["question"], reference=st.blind_answer["text"],
                                      answer=st.drafted["candidate_answer"], item_type=items[item_id].tests.type,
                                      custom_id=item_id))
    done = 0
    for item_id, resp in run_all(provider, reqs, batch=batch, wait=wait):
        verdict, note = read_verdict(resp)
        log.examined += 1
        if verdict is None:
            log.count("judge_failed")
            continue
        log.emit("judge_calibration", item=item_id, item_key_hash=items[item_id].key_hash(),
                 judge_model_id=j.model, judge_prompt_version=j.prompt_version, verdict=verdict, rationale=note)
        done += 1
    return done
