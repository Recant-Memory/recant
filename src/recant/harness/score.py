"""The numbers: rates with exact intervals, and the paired comparison between two arms."""
from __future__ import annotations

from math import comb

from ..eval.stats import rate

DIMENSIONS = ("yield", "noise", "supersession")


def type_dimension(item_type: str) -> str:
    return item_type  # yield, noise, supersession


def mcnemar_one_sided(a_right_b_wrong: int, a_wrong_b_right: int) -> float:
    """Exact one-sided p that A is better than B, on the discordant pairs: P(X >= b) for X ~ Bin(b+c, 1/2).
    With no discordant pairs there is no evidence either way, and p is 1."""
    b, c = a_right_b_wrong, a_wrong_b_right
    n = b + c
    if n == 0:
        return 1.0
    return sum(comb(n, i) for i in range(b, n + 1)) / 2 ** n


def compare(verdicts_a: dict[str, bool], verdicts_b: dict[str, bool], *, margin_pp: float) -> dict:
    """Both arms' verdicts keyed by item; only items both scored count (paired)."""
    items = sorted(set(verdicts_a) & set(verdicts_b))
    b = sum(verdicts_a[i] and not verdicts_b[i] for i in items)
    c = sum(verdicts_b[i] and not verdicts_a[i] for i in items)
    ra = sum(verdicts_a[i] for i in items) / len(items) if items else 0.0
    rb = sum(verdicts_b[i] for i in items) / len(items) if items else 0.0
    diff = (ra - rb) * 100
    p = mcnemar_one_sided(b, c)
    return {"items": len(items), "a_right_b_wrong": b, "a_wrong_b_right": c, "a_rate": round(ra, 4), "b_rate": round(rb, 4),
            "difference_pp": round(diff, 2), "p_one_sided": round(p, 6), "margin_pp": margin_pp,
            "beats_margin": diff > margin_pp and p < 0.05}


def report_rates(rows: list[dict]) -> dict:
    """rows: {"type", "hard", "verdict", "unscorable", "cut_off", "refused", "retrieved"}. Rates per dimension, and per
    dimension split by the hard tag; unscorable items are outside every n."""
    scorable = [r for r in rows if not r["unscorable"]]
    out = {"rates": {}, "by_hard": {}, "retrieval": None}
    for d in DIMENSIONS:
        rs = [r for r in scorable if r["type"] == d]
        if rs:
            out["rates"][d] = rate(sum(r["verdict"] for r in rs), len(rs))
            out["by_hard"][d] = {"hard": rate(sum(r["verdict"] for r in rs if r["hard"]), sum(r["hard"] for r in rs)),
                                 "plain": rate(sum(r["verdict"] for r in rs if not r["hard"]), sum(not r["hard"] for r in rs))}
    seen = [r for r in rows if r.get("retrieved") is not None]
    if seen:
        out["retrieval"] = {"cited_span_seen": rate(sum(r["retrieved"] for r in seen), len(seen))}
    out["cut_off"] = sum(r["cut_off"] for r in rows)
    out["refused"] = sum(bool(r.get("refused")) for r in rows)
    out["judge_failed"] = sum(r["unscorable"] and not r["cut_off"] for r in rows)
    return out


def coverage_split(rows: list[dict]) -> dict:
    """The service arms' three-way split of a miss: for answerable items, never extracted /
    extracted but not retrieved / retrieved but wrong / right; for noise, misses by whether entries were
    returned. Rows carry `extracted`, `retrieved`, `entries_seen`, `verdict`, `type`, `hard`."""
    def split(rs):
        """Four buckets that partition the items: right; wrong and never extracted; wrong, extracted but not
        retrieved; wrong although retrieved. A right answer from an uncovered item is counted right (the fact
        was held on other lines) and reported beside."""
        wrong = [r for r in rs if not r["verdict"]]
        return {"n": len(rs), "right": len(rs) - len(wrong),
                "right_though_not_extracted": sum(bool(r["verdict"]) and not r.get("extracted") for r in rs),
                "not_extracted": sum(not r.get("extracted") for r in wrong),
                "extracted_not_retrieved": sum(bool(r.get("extracted")) and not r.get("retrieved") for r in wrong),
                "retrieved_wrong": sum(bool(r.get("retrieved")) for r in wrong)}
    scorable = [r for r in rows if not r["unscorable"]]
    ans = [r for r in scorable if r["type"] in ("yield", "supersession")]
    noise = [r for r in scorable if r["type"] == "noise"]
    misses = [r for r in noise if not r["verdict"]]
    return {"answerable": split(ans), "yield": split([r for r in ans if r["type"] == "yield"]),
            "hard": split([r for r in ans if r["hard"]]), "plain": split([r for r in ans if not r["hard"]]),
            "noise": {"n": len(noise), "right": sum(bool(r["verdict"]) for r in noise), "misses": len(misses),
                      "misses_with_entries_returned": sum(bool(r.get("entries_seen")) for r in misses),
                      "misses_with_none_returned": sum(not r.get("entries_seen") for r in misses)}}
