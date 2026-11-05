"""Stratified sampling with a seed. Strata are item type, marker and the hard
flag; months rotate inside each stratum. Round-robin across strata, so every
stratum that has an item is represented before any stratum is represented twice."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from .items import EvalItem


def stratum_key(item: EvalItem, form: str | None = None) -> str:
    t = item.tests
    base = f"{t.type}/{t.marker or 'none'}/{'hard' if t.hard else 'plain'}"
    return f"{base}/{form}" if form else base


def primary_key(item: EvalItem) -> str:
    t = item.tests
    return "noise" if t.type == "noise" else f"{t.type}/{t.marker or 'none'}"


def _allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """One slot to every group that has an item, then the rest in proportion to size (largest
    remainder), never more than a group holds."""
    groups = [g for g, k in sizes.items() if k > 0]
    quota = {g: 0 for g in groups}
    if not groups or n <= 0:
        return quota
    for g in sorted(groups, key=lambda g: (-sizes[g], g))[:n]:
        quota[g] = 1
    left = n - sum(quota.values())
    while left > 0:
        room = {g: sizes[g] - quota[g] for g in groups if sizes[g] > quota[g]}
        if not room:
            break
        total = sum(sizes[g] for g in room)
        shares = {g: left * sizes[g] / total for g in room}
        give = {g: min(room[g], int(shares[g])) for g in room}
        if sum(give.values()) == 0:                       # hand out the remainder one at a time
            for g in sorted(room, key=lambda g: (-(shares[g] - int(shares[g])), -sizes[g], g))[:left]:
                give[g] = 1
        for g, k in give.items():
            quota[g] += k
        left = n - sum(quota.values())
    return quota


def stratified_sample(items: Iterable[EvalItem], n: int, seed: int, expected_strata: Iterable[str] = (),
                      forms: dict[str, str] | None = None, hard_cap: int | None = None
                      ) -> tuple[list[EvalItem], list[str]]:
    """Returns (chosen, unfilled).

    Slots go first to type and marker: every marker that has an item gets one, and the rest follow the
    pool's proportions, so the largest marker cannot vanish and the rarest cannot be crowded out.
    Inside a marker the picks rotate over question forms, hard cases first, and every pick prefers the
    month the sample has so far drawn least from. Everything random is seeded."""
    rng = random.Random(seed)
    forms = forms or {}
    pool = sorted(items, key=lambda i: i.id)
    groups: dict[str, list[EvalItem]] = defaultdict(list)
    for it in pool:
        groups[primary_key(it)].append(it)
    quota = _allocate({g: len(v) for g, v in groups.items()}, n)
    months: dict[str, int] = defaultdict(int)
    chosen: list[EvalItem] = []
    hard_taken = 0
    for g in sorted(groups):
        by_form: dict[str, list[EvalItem]] = defaultdict(list)
        for it in groups[g]:
            by_form[forms.get(it.id) or "none"].append(it)
        order = sorted(by_form)
        rng.shuffle(order)
        for bucket in by_form.values():
            rng.shuffle(bucket)
        taken = 0
        while taken < quota[g] and any(by_form.values()):
            for f in order:
                if taken >= quota[g] or not by_form[f]:
                    continue
                # Hard cases first, up to the cap; past it a hard case is taken only if nothing else is left.
                capped = hard_cap is not None and hard_taken >= hard_cap
                pick = min(by_form[f], key=lambda it: ((it.tests.hard if capped else not it.tests.hard),
                                                       months[it.tests.stratum]))
                if capped and pick.tests.hard and any(not it.tests.hard for b in by_form.values() for it in b):
                    continue
                hard_taken += pick.tests.hard
                by_form[f].remove(pick)
                months[pick.tests.stratum] += 1
                chosen.append(pick)
                taken += 1
    chosen.sort(key=lambda it: (not it.tests.hard, primary_key(it), it.id))
    have = {stratum_key(it) for it in pool}
    unfilled = sorted(set(expected_strata) - have)
    if len(chosen) < n:
        unfilled.append(f"pool exhausted at {len(chosen)} of {n}")
    return chosen, unfilled
