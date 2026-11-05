"""A reasoning-off request takes a different form per model: {"type": "disabled"} is refused on every call by
claude-sonnet-5-5. Probed 2026-11-03: sonnet-5-5 takes "between_tools";
opus-5-5 and fable-5-1 take neither; the others take "disabled"."""
from recant import costshape as cs, levers
from recant.clients.anthropic import build_params
from recant.provider import ModelRequest, Message, text
from recant.provider.interface import ProviderError

OV = {"reasoning.agent": {"kind": "your_call", "action": {"reasoning": "off"}}}


def test_the_opted_in_lever_sends_each_models_own_off_form():
    for model, want in (("claude-sonnet-5-5", {"type": "between_tools"}), ("claude-sonnet-5", {"type": "disabled"}),
                        ("claude-haiku-4-5", {"type": "disabled"})):
        out, done = levers.apply_opted_in({"model": model, "messages": []}, OV, "agent")
        assert out["thinking"] == want and done == ["reasoning.agent"], model
    out, done = levers.apply_opted_in({"model": "claude-opus-5-5", "messages": []}, OV, "agent")
    assert "thinking" not in out and done == []                              # cannot be turned off: sent as built


def test_the_reference_adapter_uses_the_same_forms():
    req = ModelRequest(job="x", model="claude-sonnet-5-5", messages=(Message("user", (text("hi"),)),), max_tokens=10, reasoning="off")
    assert build_params("claude-sonnet-5-5", req, supports_effort=True)["thinking"] == {"type": "between_tools"}
    try:
        build_params("claude-opus-5-5", req, supports_effort=True)
        assert False, "opus-5-5 cannot turn reasoning off"
    except ProviderError:
        pass


def test_the_advisor_does_not_offer_the_lever_where_it_cannot_be_carried_out():
    assert "claude-opus-5-5" in levers.CANNOT_TURN_REASONING_OFF and "claude-fable-5-1" in levers.CANNOT_TURN_REASONING_OFF
    assert cs._cannot_turn_off() == levers.CANNOT_TURN_REASONING_OFF
