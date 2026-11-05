"""Founding: once, seeded, logged. The gating suite, the held-out slice and its
refill reserve are drawn side by side from the accepted pool and frozen."""
from __future__ import annotations

from collections import Counter

from ..machinery import Machinery
from ..store.layout import StoreLayout
from .items import EvalItem
from .ledger import Ledger
from .pools import load, write_pool
from .sample import stratified_sample


class FoundingError(RuntimeError):
    pass


def _quotas(size: int, noise_mid: float, unmarked_share: float, super_share: float) -> dict[str, int]:
    """`super_share` is supersession's share of the answerable (non-noise) pool, so
    reversals appear in each pool as often as the corpus produced them and no more."""
    noise = round(noise_mid * size)
    supersession = int(super_share * (size - noise))
    yields = size - noise - supersession
    unmarked = round(unmarked_share * yields)
    return {"noise": noise, "supersession": supersession, "unmarked": unmarked, "marked": yields - unmarked}


def found(layout: StoreLayout, tenant: str, machinery: Machinery, log, seed: int) -> dict:
    k = machinery.kernel
    if load(layout, tenant, "founding", "founding"):
        raise FoundingError("the founding suite exists and is never extended")
    if k.heldout.reserve is None:
        raise FoundingError("the kernel does not size the refill reserve")
    ledger = Ledger(layout, tenant)
    sample, fingerprint = ledger.blind_sample, machinery.drafting_fingerprint()
    if not sample:
        raise FoundingError("no blind-first sample has been drawn")
    if sample["fingerprint"] != fingerprint:
        raise FoundingError("the prompts or markers changed after the blind-first audit; audit a fresh sample")
    undecided = [i for i in sample["items"] if not ledger.state(i).latest("human")]
    if undecided:
        raise FoundingError(f"the blind-first audit is unfinished: {len(undecided)} items undecided")

    accepted = [d for d in load(layout, tenant, "drafts", "founding") if ledger.state(d.id).accepted()]
    foreign = [d.id for d in accepted if (ledger.state(d.id).drafted or {}).get("fingerprint") != fingerprint]
    if foreign:
        raise FoundingError(f"{len(foreign)} accepted items were drafted under other prompts or markers")

    classes = {"noise": [], "supersession": [], "unmarked": [], "marked": []}
    for it in accepted:
        t = it.tests
        key = ("noise" if t.type == "noise" else "supersession" if t.type == "supersession"
               else "unmarked" if t.marker is None else "marked")
        classes[key].append(it)

    sizes = {"founding": k.gate.suite_size, "heldout": k.heldout.slice, "reserve": k.heldout.reserve}
    total = sum(sizes.values())
    noise_mid = sum(k.noise_share) / 2
    answerable = sum(len(classes[c]) for c in ("supersession", "unmarked", "marked"))
    super_share = len(classes["supersession"]) / answerable if answerable else 0.0
    quotas = {pool: _quotas(size, noise_mid, k.eval.unmarked.share, super_share) for pool, size in sizes.items()}
    need = Counter()
    for q in quotas.values():
        need.update(q)
    short = {c: f"need {need[c]}, have {len(classes[c])}" for c in classes if len(classes[c]) < need[c]}
    if short:
        raise FoundingError(f"the accepted pool cannot fill founding: {short}")

    pools: dict[str, list[EvalItem]] = {p: [] for p in sizes}
    for offset, cls in enumerate(sorted(classes)):
        remaining = list(classes[cls])
        for pool in sizes:  # the gating suite draws first
            chosen, _ = stratified_sample(remaining, quotas[pool][cls], seed + offset)
            taken = {c.id for c in chosen}
            remaining = [r for r in remaining if r.id not in taken]
            pools[pool].extend(chosen)

    noise_share = sum(i.tests.type == "noise" for i in pools["founding"]) / sizes["founding"]
    if not (k.noise_share[0] <= noise_share <= k.noise_share[1]):
        raise FoundingError(f"noise share {noise_share:.3f} is outside {k.noise_share}")
    for pool, items in pools.items():
        write_pool(layout, tenant, pool, sorted(items, key=lambda i: i.id))
    counts = {p: dict(Counter(i.tests.type for i in items)) | {"total": len(items),
              "unmarked": sum(i.tests.type == "yield" and i.tests.marker is None for i in items),
              "hard": sum(i.tests.hard for i in items)} for p, items in pools.items()}
    log.emit("founding", seed=seed, fingerprint=fingerprint, counts=counts,
             key_hashes={p: {i.id: i.key_hash() for i in items} for p, items in pools.items()})
    return {"counts": counts, "seed": seed}
