"""A tool's own end-of-run totals (Claude Code's `--output-format json`, the Agent SDK's
ResultMessage) metered in their own rows, billed and priced at list, with no lever. The result below has the shape
Claude Code 2.1 writes (field names as it wrote them on 2026-11-04); its numbers are synthetic."""
import json

from recant import costshape as cs
from recant.metering import Meter, from_agent_sdk, from_claude_code
from recant.store import StoreLayout

RESULT = {
    "type": "result", "subtype": "success", "is_error": False, "num_turns": 3, "duration_ms": 9000,
    "session_id": "00000000-0000-0000-0000-000000000001", "stop_reason": "end_turn", "total_cost_usd": 0.0277,
    "usage": {"input_tokens": 6, "output_tokens": 148, "cache_read_input_tokens": 51128, "cache_creation_input_tokens": 8440},
    "modelUsage": {"claude-sonnet-5-5": {"inputTokens": 6, "outputTokens": 148, "cacheReadInputTokens": 51128,
                                         "cacheCreationInputTokens": 8440, "costUSD": 0.0277, "thinkingTokens": 0,
                                         "cacheCreationEphemeral5mInputTokens": 8440, "cacheCreationEphemeral1hInputTokens": 0}},
}
STEPS = [{"type": "assistant", "message": {"id": f"msg_{k}", "model": "claude-sonnet-5-5",
                                           "usage": {"input_tokens": 2, "output_tokens": 16}}} for k in (1, 1, 2, 3)]


def meter(tmp_path):
    layout = StoreLayout(tmp_path / "s")
    layout.ensure("t1")
    return layout, Meter(layout, "t1")


def test_claude_code_json_is_metered_from_its_own_totals_with_calls_from_the_message_ids(tmp_path):
    layout, m = meter(tmp_path)
    m.run("agent")
    [rec] = from_claude_code(json.dumps(RESULT), m, billing="subscription", messages=STEPS)
    m.close()
    assert rec["usage"]["input"] == 6 and rec["usage"]["output"] == 148 and rec["usage"]["cache_read"] == 51128
    assert rec["usage"]["cache_write"] == 8440                    # the run's totals, not the steps' placeholder outputs
    assert (rec["via"], rec["billing"], rec["calls"], rec["billed_usd"], rec["cost_usd"]) == ("claude-code", "subscription", 3, 0.0, 0.0277)
    rep = cs.report(layout, "t1", since=0)
    assert rep.calls == [] and len(rep.managed) == 1             # its own rows, never in the shape the levers read
    text = rep.text()
    assert "claude-code on claude-sonnet-5-5: 1 runs, 3 calls, 59,722 tokens; billed $0.00 (subscription)" in text
    assert "tokens against the plan's limits, not dollars" in text
    assert "coverage:" in text


def test_the_agent_sdks_result_message_as_an_object(tmp_path):
    class Result:                                                 # the Python SDK's dataclass, snake_case
        num_turns, duration_ms, session_id, is_error, total_cost_usd = 3, 9000, "s", False, 0.0277
        model_usage = {"claude-sonnet-5-5": {"input_tokens": 6, "output_tokens": 148, "cache_read_input_tokens": 51128,
                                             "cache_creation_input_tokens": 8440}}
    layout, m = meter(tmp_path)
    [rec] = from_agent_sdk(Result(), m, billing="api_key")
    m.close()
    assert rec["via"] == "claude-agent-sdk" and rec["calls"] == 3 and rec.get("billed_usd") is None
    assert "billed not reported (an API key: about the list price)" in cs.report(layout, "t1", since=0).text()
