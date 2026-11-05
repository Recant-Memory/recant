"""The blind-first sample: a stratified draw from the finished, model-reviewed
pool, taken only once the prompts that produced the pool are frozen."""
from __future__ import annotations

from ..machinery import Machinery
from ..store.layout import StoreLayout
from .ledger import Ledger
from .pools import load
from .sample import stratified_sample


class AuditError(RuntimeError):
    pass


def expected_strata(machinery: Machinery) -> list[str]:
    """Every stratum the pool ought to have something in."""
    markers = machinery.artifact.marker_names()
    out = [f"yield/{m}/plain" for m in markers if m != "reversal"] + ["yield/none/plain", "noise/none/plain"]
    if "reversal" in markers:  # a reversal is asked about as a supersession item
        out.append("supersession/reversal/plain")
    return out


def draw_blind_sample(layout: StoreLayout, tenant: str, machinery: Machinery, log, seed: int,
                      redraw_because: str | None = None) -> dict:
    ledger = Ledger(layout, tenant)
    fingerprint = machinery.drafting_fingerprint()
    drafts = load(layout, tenant, "drafts", "audit")
    stale = [d.id for d in drafts if (ledger.state(d.id).drafted or {}).get("fingerprint") != fingerprint
             and ledger.state(d.id).accepted()]
    if stale:
        raise AuditError(f"{len(stale)} accepted drafts were produced under different prompts or markers "
                         "than are in force; the pool is not one pool. Redraft or discard them first.")
    current = ledger.blind_sample
    if current and current["fingerprint"] == fingerprint:
        touched = [i for i in current["items"] if ledger.state(i).blind_answer or ledger.state(i).latest("human")]
        if not redraw_because or touched:
            raise AuditError("a blind sample already exists for these prompts; it is redrawn only when they change, "
                             "or, with a stated reason, before a person has answered any of it")
        log.count("redrawn")
        log.emit("pool_trimmed", seed=seed, kept=0, surplus=0, reason=f"blind sample redrawn, untouched: {redraw_because}")
    audited = {i for i, st in ledger.items.items() if st.latest("human")}
    pool = [d for d in drafts if ledger.state(d.id).accepted() and d.id not in audited]
    unreviewed = [d.id for d in drafts if not ledger.state(d.id).decisions]
    if unreviewed:
        raise AuditError(f"{len(unreviewed)} drafts have no review yet; the sample is drawn from a finished pool")
    n = machinery.kernel.eval.blind_first_sample
    # The sample carries the suite's noise share, and within noise and within the answerable items it is
    # stratified by type, marker, the hard tag and question form, so no form is over-represented by accident.
    forms = {i: (st.drafted or {}).get("form") for i, st in ledger.items.items()}
    noise_n = round(sum(machinery.kernel.noise_share) / 2 * n)
    cap = machinery.kernel.eval.blind_hard_cap
    rest, unfilled = stratified_sample([d for d in pool if d.tests.type != "noise"], n - noise_n, seed + 1,
                                       expected_strata(machinery), forms=forms, hard_cap=cap)
    noise, unfilled_noise = stratified_sample([d for d in pool if d.tests.type == "noise"], n - len(rest), seed,
                                              forms=forms, hard_cap=max(0, cap - sum(r.tests.hard for r in rest)))
    chosen = [*rest, *noise]
    unfilled = [u for u in unfilled if not u.startswith("noise/")] + [u for u in unfilled_noise if u.startswith("pool")]
    rec = log.emit("blind_sample", seed=seed, items=[c.id for c in chosen], fingerprint=fingerprint,
                   unfilled=unfilled)
    return {"items": rec["items"], "unfilled": unfilled, "fingerprint": fingerprint}
