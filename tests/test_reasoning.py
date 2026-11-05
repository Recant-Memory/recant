"""Reasoning, measured and controllable, and a second adapter stub (tests only):
the request carries a neutral reasoning setting with an explicit "off"; the job's default comes from the consumer's
machinery; the reference adapter maps it and counts what the model spent reasoning; the log keeps the count; and a
client shaped like another provider maps its usage report into the same Usage, so the library names no provider."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from recant.clients.anthropic import _raw, build_params
from recant.harness.nominate import nominate
from recant.log import LogWriter, read_run
from recant.provider import Completion, Message, ModelRequest, ProviderError, Usage, text
from recant.provider.base import BaseProvider
from recant.store import ingest
from pipeline_fixtures import build_corpus
from recant_fixtures import ALLOWLIST, TENANT


def req(**kw):
    return ModelRequest(job="nominate", model="m", messages=(Message("user", (text("x"),)),), max_tokens=100, **kw)


def test_the_reference_adapter_maps_reasoning_and_refuses_off_where_the_model_cannot():
    assert "thinking" not in build_params("m", req(), supports_effort=True)                       # unset: the model's default
    assert build_params("m", req(reasoning="off"), supports_effort=True)["thinking"] == {"type": "disabled"}
    assert build_params("m", req(reasoning="on"), supports_effort=True)["thinking"] == {"type": "adaptive"}
    with pytest.raises(ProviderError, match="cannot turn its reasoning off"):
        build_params("m", req(reasoning="off"), supports_effort=True, can_disable_reasoning=False)


def test_the_reference_adapter_counts_reasoning_tokens_and_keeps_zero_where_none_are_reported():
    def message(**usage):
        return SimpleNamespace(content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text="{}")], stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=10, output_tokens=900, cache_read_input_tokens=5,
                                                     cache_creation_input_tokens=0, **usage))
    got = _raw(message(output_tokens_details=SimpleNamespace(thinking_tokens=850)))
    assert got.usage == Usage(input=10, output=900, cache_read=5, cache_write=0, reasoning=850)
    assert _raw(message()).usage.reasoning == 0


class OtherVendorClient:
    """A stub of another provider's client (tests only, not a shipped adapter): its usage report names
    prompt, cached, completion and reasoning tokens its own way, and it maps them into recant's Usage."""

    def __init__(self, answer: str = '{"spans": []}'):
        self.answer, self.seen = answer, []

    def complete(self, request: ModelRequest) -> Completion:
        self.seen.append(request)
        report = {"prompt_tokens": 1200, "prompt_tokens_details": {"cached_tokens": 1000},
                  "completion_tokens": 300, "completion_tokens_details": {"reasoning_tokens": 0 if request.reasoning == "off" else 240}}
        cached = report["prompt_tokens_details"]["cached_tokens"]
        usage = Usage(input=report["prompt_tokens"] - cached, cache_read=cached, output=report["completion_tokens"],
                      reasoning=report["completion_tokens_details"]["reasoning_tokens"])
        return Completion(content=(text(self.answer),), stop_reason="end_turn", usage=usage)


def test_the_jobs_reasoning_default_comes_from_the_machinery_and_the_log_keeps_the_count(layout, machinery, tmp_path):
    ingest(layout, TENANT, build_corpus(tmp_path / "corpus"), ALLOWLIST)
    a = machinery.artifact
    off = replace(machinery, artifact=a.model_copy(update={"nomination": a.nomination.model_copy(update={"reasoning": "off"})}))
    for m, expect in ((machinery, (None, 240)), (off, ("off", 0))):
        client = OtherVendorClient()
        with LogWriter(layout, TENANT, "nominate", artifact_hash=m.artifact_hash, kernel_hash=m.kernel_hash) as log:
            nominate(layout, TENANT, m, BaseProvider(m, log, client), log, paths=["notes/2026-01-01.md"], batch=False)
        assert client.seen[0].reasoning == expect[0]
        (call,) = [r for r in read_run(log.path, typed=False) if r["kind"] == "model_call"]
        assert call["usage"] == {"input": 200, "output": 300, "cache_read": 1000, "cache_write": 0, "reasoning": expect[1], "cache_write_long": 0}


def test_a_log_written_before_the_reasoning_field_still_reads():
    from recant.log.records import ModelCall
    rec = {"v": 1, "ts": "2026-10-08T00:00:00.000Z", "tenant": TENANT, "run": "R", "seq": 1, "artifact": "a", "kernel": "k",
           "kind": "model_call", "model_id": "m", "job": "j", "request_hash": "h",
           "usage": {"input": 1, "output": 2, "cache_read": 0, "cache_write": 0}}
    assert ModelCall.model_validate(rec).usage.reasoning == 0


def test_the_jobs_output_cap_comes_from_the_machinery(layout, machinery, tmp_path):
    """A job's cap is a runaway guard set in the machinery, above its p99.9; unset, the caller's."""
    ingest(layout, TENANT, build_corpus(tmp_path / "corpus"), ALLOWLIST)
    a = machinery.artifact
    capped = replace(machinery, artifact=a.model_copy(update={"nomination": a.nomination.model_copy(update={"max_tokens": 32000})}))
    for m, expect in ((machinery, 8000), (capped, 32000)):
        client = OtherVendorClient()
        with LogWriter(layout, TENANT, "nominate", artifact_hash=m.artifact_hash, kernel_hash=m.kernel_hash) as log:
            nominate(layout, TENANT, m, BaseProvider(m, log, client), log, paths=["notes/2026-01-01.md"], batch=False)
        assert client.seen[0].max_tokens == expect
