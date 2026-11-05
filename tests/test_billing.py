"""what Recant metered against what the provider billed, per day, key and model, on settled days only; the
reference adapter reads the usage report across pages and finds a key's id by its hint without showing either."""
import json
from datetime import date

from recant.billing import BilledUsage, reconcile, settled_days, summary
from recant.clients.anthropic_billing import AnthropicBilling
from recant.provider import Prices
from recant_fixtures import TENANT

PRICE = {"m": Prices(input=4.0, output=20.0, cache_read=0.2, cache_write=5.0, cache_write_long=8.0)}


def call(day, key, *, inp=100, out=50, read=0, write=0, long_=0, model="m"):
    return {"kind": "model_call", "ts": f"{day}T10:00:00Z", "key": key, "model_id": model,
            "usage": {"input": inp, "output": out, "cache_read": read, "cache_write": write, "cache_write_long": long_}}


def test_billed_but_not_metered_is_traffic_outside_recant_and_the_reverse_is_a_bug_on_our_side():
    calls = [call("2026-10-01", "live"), call("2026-10-01", "live"), call("2026-10-02", "bench", write=1000, long_=400)]
    billed = [BilledUsage("2026-10-01", "id-live", "m", uncached=200 + 900, output=100 + 300),     # 900 + 300 more than metered
              BilledUsage("2026-10-02", "id-bench", "m", uncached=100, output=50, cache_write_5m=600, cache_write_1h=400),
              BilledUsage("2026-10-02", "id-other", "m", uncached=5000)]                         # a key nobody named
    rows = reconcile(calls + [call("2026-10-03", "live", inp=10_000)], billed, key_ids={"live": "id-live", "bench": "id-bench"},
                     days={"2026-10-01", "2026-10-02", "2026-10-03"}, prices=PRICE)
    st = {(r["day"], r["key"]): r for r in rows}
    live = st[("2026-10-01", "live")]
    assert live["status"] == "unmetered" and live["unmetered_tokens"] == 1200
    assert abs(live["unmetered_usd"] - (900 * 4 + 300 * 20) / 1e6) < 1e-12
    assert st[("2026-10-02", "bench")]["status"] == "matches"                                # long writes split the same way
    assert st[("2026-10-02", "unnamed key id-other")]["status"] == "unmetered"
    assert st[("2026-10-03", "live")]["status"] == "metered_not_billed"                      # metered, never billed: our bug
    s = summary(rows)
    assert s["live"]["days_unmetered"] == ["2026-10-01"] and s["live"]["metered_not_billed_tokens"] == 10_050


def test_a_day_is_compared_only_once_settled(layout):
    rows = [BilledUsage("2026-10-09", "k", "m", uncached=10), BilledUsage("2026-10-01", "k", "m", uncached=5)]
    today = date(2026, 10, 11)
    assert settled_days(layout, TENANT, rows, today=today) == {"2026-10-01"}                  # 2026-10-09: first fetch, not yet
    assert settled_days(layout, TENANT, rows, today=today) == {"2026-10-01", "2026-10-09"}    # unchanged on the next fetch
    moved = [BilledUsage("2026-10-09", "k", "m", uncached=12), rows[1]]
    assert "2026-10-09" not in settled_days(layout, TENANT, moved, today=today)               # still moving
    assert settled_days(layout, TENANT, [BilledUsage("2026-10-10", "k", "m")], today=today) == set()   # under the lag


def test_the_anthropic_adapter_pages_the_usage_report_and_matches_keys_by_hint():
    pages = {
        None: {"data": [{"starting_at": "2026-10-01T00:00:00Z", "results": [
            {"api_key_id": "apikey_A", "model": "m", "uncached_input_tokens": 7, "output_tokens": 3, "cache_read_input_tokens": 1,
             "cache_creation": {"ephemeral_5m_input_tokens": 2, "ephemeral_1h_input_tokens": 4}}]}], "has_more": True, "next_page": "p2"},
        "p2": {"data": [{"starting_at": "2026-10-02T00:00:00Z", "results": [{"api_key_id": "apikey_B", "model": "m"}]}], "has_more": False},
    }
    seen = []

    def opener(req):
        seen.append(req)
        if "api_keys" in req.full_url:
            return json.dumps({"data": [{"id": "apikey_A", "partial_key_hint": "test-key-AB...yz"},
                                        {"id": "apikey_B", "partial_key_hint": "test-key-CD...wx"}], "has_more": False})
        page = dict(p.split("=", 1) for p in req.full_url.split("?", 1)[1].split("&")).get("page")
        return json.dumps(pages[page])
    src = AnthropicBilling("test-admin-key-made-up", opener=opener)
    rows = src.usage("2026-10-01", "2026-10-03")
    assert [(r.day, r.key_id) for r in rows] == [("2026-10-01", "apikey_A"), ("2026-10-02", "apikey_B")]
    assert (rows[0].uncached, rows[0].cache_write_5m, rows[0].cache_write_1h, rows[0].cache_read, rows[0].output) == (7, 2, 4, 1, 3)
    assert src.key_ids({"live": "test-key-AB-the-middle-yz", "other": "test-key-ZZ-no-match"}) == {"live": "apikey_A"}
    assert all(r.headers.get("X-api-key") == "test-admin-key-made-up" for r in seen)           # the admin key, in headers only


def test_dated_model_ids_and_unsplit_cache_writes_compare_as_the_same_calls():
    """The bill names a dated model id; calls logged before 2026-10-09 carry cache writes as one number; a refused call
    with usage is billed and counts."""
    c = call("2026-10-01", "live", write=1000)
    del c["usage"]["cache_write_long"]                           # an old record: no lifetime split
    refused = {**call("2026-10-01", "live", inp=10, out=0), "error": "refusal"}
    billed = [BilledUsage("2026-10-01", "id-live", "m-20251001", uncached=110, output=50, cache_write_5m=400, cache_write_1h=600)]
    (row,) = reconcile([c, refused], billed, key_ids={"live": "id-live"}, days={"2026-10-01"}, prices=PRICE)
    assert row["model"] == "m" and row["status"] == "matches" and row["calls_metered"] == 2
