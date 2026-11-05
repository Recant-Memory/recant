"""The numbers the blind-first audit produces. Each comes with its n and an
exact interval; none of them acts on anything."""
from __future__ import annotations

from collections import Counter

from ..machinery import Machinery
from ..store.layout import StoreLayout
from .ledger import Ledger
from .review import CITATION
from .pools import load
from .stats import rate


def audit_report(layout: StoreLayout, tenant: str, machinery: Machinery) -> dict:
    ledger = Ledger(layout, tenant)
    sample = ledger.blind_sample
    if not sample:
        return {"sample": None}
    founded = ledger.founding
    items = {d.id: d for d in load(layout, tenant, "drafts", "audit")}
    fingerprint = machinery.drafting_fingerprint()
    decided, key_wrong, discarded_other = [], 0, 0
    model_accepts = human_accepts = same = both = 0
    judged = judge_agrees = 0
    by_type: dict[str, Counter] = {}
    choices: Counter = Counter()
    choice_vs_outcome: Counter = Counter()
    for item_id in sample["items"]:
        st = ledger.state(item_id)
        human = st.latest("human")
        if not human:
            continue
        decided.append(item_id)
        # A fix that only corrects what the item cites is not a disagreement with the key.
        wrong = (human["outcome"] == "fix" and human["reason"] != CITATION) or \
                (human["outcome"] == "discard" and human["reason"] == "wrong_answer")
        key_wrong += wrong
        discarded_other += human["outcome"] == "discard" and human["reason"] != "wrong_answer"
        kind = items[item_id].tests.type if item_id in items else "unknown"
        by_type.setdefault(kind, Counter()).update(n=1, wrong=int(wrong))
        how = (st.blind_answer or {}).get("choice") or "words"
        choices[how] += 1
        choice_vs_outcome[f"{how} -> {human['outcome']}"] += 1
        # A citation fix leaves the key standing: for anchoring and for the judge it is an accept.
        human_ok = human["outcome"] == "accept" or (human["outcome"] == "fix" and human["reason"] == CITATION)
        human_accepts += human_ok
        model = st.latest("model")
        if model:
            both += 1
            model_ok = model["outcome"] == "accept"
            model_accepts += model_ok
            same += model_ok == human_ok
        if st.calibrations:
            judged += 1
            judge_agrees += st.calibrations[-1]["verdict"] == human_ok
    floor = machinery.kernel.eval.judge_agreement_floor
    judge = rate(judge_agrees, judged)
    return {
        "sample": {"size": len(sample["items"]), "decided": len(decided), "seed": sample["seed"],
                   "unfilled": sample["unfilled"], "complete": len(decided) == len(sample["items"]),
                   "stale": (sample["fingerprint"] != fingerprint) if not founded else False,
                   "founded_under": founded["fingerprint"] if founded else None},
        "key_disagreement": rate(key_wrong, len(decided)),
        "key_disagreement_by_type": {k: rate(c["wrong"], c["n"]) for k, c in sorted(by_type.items())},
        "discarded_for_other_reasons": rate(discarded_other, len(decided)),
        "blind_answers": {"by_choice": dict(choices), "choice_then_outcome": dict(choice_vs_outcome),
                          "note": "the judge's agreement is measurable only on items answered in words"},
        "anchoring": {"human_accept": rate(human_accepts, len(decided)),
                      "model_accept_on_same_items": rate(model_accepts, both),
                      "model_agrees_with_human": rate(same, both)},
        "judge_agreement": {**judge, "floor": floor,
                            "below_floor": None if judge["rate"] is None else judge["rate"] < floor},
    }
