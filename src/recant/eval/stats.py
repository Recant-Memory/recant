"""Small-sample arithmetic. A rate is never shown without its n and its interval."""
from __future__ import annotations

from math import comb


def _cdf(k: int, n: int, p: float) -> float:
    return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(0, k + 1))


def _solve(f, lo=0.0, hi=1.0) -> float:
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def exact_interval(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson. Exact, and wide at small n, which is the point."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n and n > 0")
    a = (1 - conf) / 2
    lower = 0.0 if k == 0 else _solve(lambda p: a - (1 - _cdf(k - 1, n, p)))
    upper = 1.0 if k == n else _solve(lambda p: _cdf(k, n, p) - a)
    return lower, upper


def rate(k: int, n: int) -> dict:
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "interval_95": None}
    lo, hi = exact_interval(k, n)
    return {"k": k, "n": n, "rate": k / n, "interval_95": [round(lo, 4), round(hi, 4)]}
