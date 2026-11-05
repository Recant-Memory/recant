"""The library's list prices, against the vendor's page as read 2026-11-05: Claude Haiku 5.5 has two rate cards, split
at a 100,000-token prompt (uncached, cache reads and cache writes together), each request priced on its own; Sonnet 5.5
cache reads are 0.05x input. Before 0.3.2 Haiku 5.5 had no row (a host using it got "cost not measured") and the
Sonnet 5.5 row read cache hits at 0.10x."""
import pytest

from recant import costshape as cs

M = 1e6
HAIKU = cs.LIST_PRICES["claude-haiku-5-5"]


def test_haiku_5_5_under_the_threshold_pays_the_page_s_first_card():
    # 50k uncached, 10k 5m write, 30k read (90k prompt), 1k output
    want = (50_000 * 0.10 + 10_000 * 0.125 + 30_000 * 0.01 + 1_000 * 0.50) / M
    assert HAIKU.bill(50_000, 10_000, 30_000, 1_000) == pytest.approx(want)


def test_haiku_5_5_over_the_threshold_pays_the_second_card_even_on_its_cache_hits():
    # 20k uncached, 90k read: a 110k prompt, so every token at the second card
    want = (20_000 * 0.50 + 90_000 * 0.05 + 2_000 * 2.50) / M
    assert HAIKU.bill(20_000, 0, 90_000, 2_000) == pytest.approx(want)
    assert HAIKU.bill(100_000, 0, 0, 0) == pytest.approx(100_000 * 0.10 / M)            # exactly 100k is not over


def test_the_one_hour_write_and_batch_on_both_cards():
    assert HAIKU.bill(write=10_000, write_long=10_000) == pytest.approx(10_000 * 0.20 / M)
    assert HAIKU.at(200_000).bill(write=10_000, write_long=10_000) == pytest.approx(10_000 * 1.00 / M)
    assert HAIKU.batch == 0.5 and HAIKU.at(200_000).batch == 0.5


def test_part_of_a_request_is_billed_at_the_card_of_the_whole_request():
    assert HAIKU.bill(write=5_000, prompt=150_000) == pytest.approx(5_000 * 0.625 / M)
    assert HAIKU.bill(write=5_000) == pytest.approx(5_000 * 0.125 / M)


def test_a_run_s_shape_prices_each_call_on_its_own_card():
    calls = [dict(run="r", seq=0, session="s", start=0, latency=1, model="claude-haiku-5-5", job="agent", uncached=60_000,
                  cache_write=0, cache_write_long=0, cache_read=0, output=100, reasoning=0),
             dict(run="r", seq=1, session="s", start=1, latency=1, model="claude-haiku-5-5", job="agent", uncached=1_000,
                  cache_write=0, cache_write_long=0, cache_read=120_000, output=100, reasoning=0)]
    r = cs.Run("r", "s", "claude-haiku-5-5", "agent", calls)
    out = cs.run_shape(r, HAIKU)
    want = (60_000 * 0.10 + 100 * 0.50) / M + (1_000 * 0.50 + 120_000 * 0.05 + 100 * 2.50) / M
    assert out["cost"] == pytest.approx(want)


@pytest.mark.parametrize("model,inp,out,read", [("claude-sonnet-5-5", 2, 10, 0.10), ("claude-opus-5-5", 4, 20, 0.20),
                                                 ("claude-fable-5-1", 10, 50, 0.25), ("claude-haiku-4-5", 1, 5, 0.10)])
def test_the_current_rows_match_the_page(model, inp, out, read):
    pf = cs.LIST_PRICES[model]
    assert (pf.p * M, pf.po * M, pf.p * pf.r * M) == pytest.approx((inp, out, read))
    assert pf.p * pf.w * M == pytest.approx(inp * 1.25) and pf.p * pf.w_long * M == pytest.approx(inp * 2.0)


def _call(run, seq, start, **u):
    return dict(dict(run=run, seq=seq, session="s", start=start, latency=1, model="claude-haiku-5-5", job="agent", uncached=0,
                     cache_write=0, cache_write_long=0, cache_read=0, output=0, reasoning=0, cost=None, batch=False,
                     segments=None, error=None, stop_reason=None), **u)


def _report(calls):
    return cs.Report(cs.HostShape(), 0, max(c["start"] for c in calls) + 1, calls, {}, [], [], [], {},
                     {"claude-haiku-5-5": HAIKU}, {})


def test_the_breakpoint_check_prices_the_uncached_side_at_the_whole_prompt_s_card():
    """10,000 uncached, 95,000 read, 10,000 output. Measured on the second card $0.03475; uncached, the
    same request is a 105k prompt on the second card, $0.07750: the lever saved $0.04275 (it read -$0.01925)."""
    c = _call("r", 0, 10, uncached=10_000, cache_read=95_000, output=10_000)
    res = cs.check("breakpoint.add", _report([c]), {"applied_at": 0, "predicted_share": None, "job": "agent"})
    assert res["realized_usd"] == pytest.approx(0.04275, abs=1e-4)


def test_the_long_lifetime_premium_is_paid_at_the_card_of_the_request_that_wrote_it():
    """a writes 90,000 (a 90k prompt, first card); b, 600 s later, rewrites 110,000 (second card)."""
    a = cs.Run("ra", "s", "claude-haiku-5-5", "agent", [_call("ra", 0, 0, cache_write=90_000, output=20_000)])
    b = cs.Run("rb", "s", "claude-haiku-5-5", "agent", [_call("rb", 0, 600, cache_write=110_000)])
    out = cs.cache_lifetime([a, b], {"claude-haiku-5-5": HAIKU})
    assert out["cost"] == pytest.approx(90_000 * (0.20 - 0.125) / 1e6)                 # a's own card: $0.00675


def test_an_upper_card_with_no_long_write_price_takes_the_short_write_s():
    """A valid table may give the long write on its base card only; the upper card then defaults it, as documented
    (in an earlier 0.3.2 candidate the cache-lifetime sizing raised TypeError on it)."""
    from recant.provider import Prices
    pf = cs.profile_from_prices(Prices.from_row({"input": 1, "output": 1, "cache_write_long": 2, "above": 100,
                                                 "over": {"input": 2, "output": 2}}, model="m"))
    a = cs.Run("ra", "s", "m", "agent", [dict(_call("ra", 0, 0, cache_write=200), model="m")])
    b = cs.Run("rb", "s", "m", "agent", [dict(_call("rb", 0, 600, cache_write=200), model="m")])
    out = cs.cache_lifetime([a, b], {"m": pf})
    assert out["runs"] == 1 and out["cost"] == pytest.approx(0.0)              # the upper card's long write = its short write
