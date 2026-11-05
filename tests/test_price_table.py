"""The `--prices` table: one JSON object keyed by model ID, each row in USD per million tokens, with the same field names
as `recant.provider.Prices`. A bad table gets a one-line message naming the model and the field (before 0.3.2 the CLI
printed `TypeError: Prices.__init__() got an unexpected keyword argument`), and `recant cost --json` prints each model's
row in the same format, so it can be copied into a table as it stands."""
import json

import pytest

from recant import cli, costshape as cs
from recant.provider import Prices, Usage


def write(tmp_path, obj, name="prices.json"):
    p = tmp_path / name
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj))
    return str(p)


def test_a_good_table_loads_with_its_defaults(tmp_path):
    t = cli._prices(write(tmp_path, {"m": {"input": 1.0, "output": 5.0}}))
    assert t["m"].cache_read is None and cs.profile_from_prices(t["m"]).r == 1.0         # no discount assumed


@pytest.mark.parametrize("table,says", [
    ({"m": {"input_per_mtok": 1.0, "output": 5.0}}, "model 'm': unknown field 'input_per_mtok'; the fields are input, output"),
    ({"m": {"output": 5.0}}, "model 'm': 'input' is required"),
    ({"m": {"input": "1", "output": 5.0}}, "'input' is a number of USD per million tokens"),
    ({"m": [1, 5]}, "a row is an object"),
    ([{"m": 1}], "keyed by model ID"),
    ({"m": {"input": 1, "output": 5, "above": 100000}}, "go together"),
    ("{not json", "--prices"),
])
def test_a_bad_table_is_one_line_naming_the_model_and_the_field(tmp_path, table, says):
    with pytest.raises(SystemExit) as e:
        cli._prices(write(tmp_path, table))
    assert says in str(e.value) and "Traceback" not in str(e.value)


def test_a_consumer_s_two_card_row_prices_each_request_on_its_card():
    row = Prices.from_row({"input": 0.1, "output": 0.5, "cache_read": 0.01, "above": 100_000,
                           "over": {"input": 0.5, "output": 2.5, "cache_read": 0.05}}, model="m")
    assert row.cost_usd(Usage(input=50_000, output=1_000)) == pytest.approx((50_000 * 0.1 + 1_000 * 0.5) / 1e6)
    assert row.cost_usd(Usage(input=10_000, cache_read=95_000)) == pytest.approx((10_000 * 0.5 + 95_000 * 0.05) / 1e6)
    pf = cs.profile_from_prices(row)
    assert pf.bill(10_000, 0, 95_000, 0) == pytest.approx(row.cost_usd(Usage(input=10_000, cache_read=95_000)))


@pytest.mark.parametrize("model", sorted(cs.LIST_PRICES))
def test_the_json_row_round_trips_into_the_same_prices(model):
    pf = cs.LIST_PRICES[model]
    back = cs.profile_from_prices(Prices.from_row(json.loads(json.dumps(cs.price_row(pf))), model=model))
    for u in [(50_000, 2_000, 30_000, 1_000, 0), (20_000, 5_000, 120_000, 3_000, 5_000)]:
        assert back.bill(*u) == pytest.approx(pf.bill(*u))


@pytest.mark.parametrize("table,says", [
    ({"m": {"input": None, "output": 5.0}}, "'input' is required"),
    ({"m": {"input": 1.0, "output": 5.0, "batch_factor": None}}, "'batch_factor' is a factor"),
    ("{\"m\": {\"input\": 1e309, \"output\": 5.0}}", "'input' is a number of USD per million tokens, not inf"),
    ({"m": {"input": 1.0, "output": -5.0}}, "'output' is a number"),
])
def test_null_infinite_or_negative_prices_are_refused_by_name(tmp_path, table, says):
    with pytest.raises(SystemExit) as e:
        cli._prices(write(tmp_path, table))
    assert says in str(e.value)


def test_a_null_optional_price_takes_its_default(tmp_path):
    t = cli._prices(write(tmp_path, {"m": {"input": 1.0, "output": 5.0, "cache_read": None}}))
    assert t["m"].cache_read is None
