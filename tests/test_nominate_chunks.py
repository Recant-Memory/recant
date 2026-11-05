"""Nomination of a long episode: read whole, a long session can stop at
max_tokens with no spans and no reason in the log. A long episode is read in chunks under its own line
numbers; a chunk that still runs out is halved and asked again; what finally fails says why."""
import json
import re

from recant.harness.nominate import chunks, nominate
from recant.log import LogWriter, read_run
from recant.provider import Usage, text
from recant.provider.base import Raw
from recant.provider.fake import FakeProvider
from recant.store import ingest
from recant_fixtures import ALLOWLIST, TENANT

PATH = "notes/2026-01-01.md"


def _episode(tmp_path, n=400):
    root = tmp_path / "corpus"
    (root / "notes").mkdir(parents=True)
    lines = [f"DECISION: rule number {i} holds" if i % 50 == 0 else f"filler line {i} " + "x" * 60 for i in range(1, n + 1)]
    (root / PATH).write_text("\n".join(lines) + "\n")
    return root, lines


def _responder(limit_lines):
    """Answers like a model with a fixed output allowance: past `limit_lines` lines of input it runs out."""
    def respond(model, req):
        user = req.messages[0].content[0].text
        numbered = re.findall(r"^(\d+): (.*)$", user, flags=re.M)
        if len(numbered) > limit_lines:
            return Raw((text('{"spans": [{"marker": "rule_or_d'),), "max_tokens", Usage(input=100, output=8000))
        spans = [{"marker": "rule_or_decision", "start_line": int(n), "end_line": int(n), "quote": line}
                 for n, line in numbered if line.startswith("DECISION:")]
        return Raw((text(json.dumps({"spans": spans})),), "end_turn", Usage(input=100, output=50))
    return respond


def _run(layout, machinery, tmp_path, limit_lines, **kw):
    root, lines = _episode(tmp_path)
    ingest(layout, TENANT, root, ALLOWLIST)
    with LogWriter(layout, TENANT, "nominate", artifact_hash=machinery.artifact_hash,
                   kernel_hash=machinery.kernel_hash) as log:
        provider = FakeProvider(machinery, log, _responder(limit_lines))
        spans = nominate(layout, TENANT, machinery, provider, log, paths=[PATH], batch=False, **kw)
    return spans, list(read_run(log.path, typed=False)), provider, lines


def test_chunks_cover_every_line_within_the_size_and_overlap():
    lines = [f"{i:03d}" + "y" * 96 for i in range(1, 301)]          # 100 characters each, 101 with the newline
    got = chunks(lines, 2020, 5)
    assert got[0] == (1, 20) and got[-1][1] == 300
    assert all(b[0] == a[1] + 1 - 5 for a, b in zip(got, got[1:]))   # five lines shared at each boundary
    assert chunks([], 2020, 5) == [(1, 1)]
    assert chunks(["z" * 5000, "short"], 100, 5) == [(1, 1), (2, 2)]  # an over-long line is its own chunk


def test_read_whole_with_no_split_it_runs_out_of_tokens_and_says_why(layout, machinery, tmp_path):
    spans, recs, _, _ = _run(layout, machinery, tmp_path, limit_lines=120, chunk_chars=10 ** 9, max_splits=0)
    assert spans == [] and recs[-1]["counts"]["episodes_failed"] == 1
    (nom,) = [r for r in recs if r["kind"] == "nomination"]
    assert nom["episode"] == f"{PATH}#L1-L400" and "ran out of output tokens" in nom["error"]


def test_chunked_nomination_finds_every_moment_under_its_real_line_numbers(layout, machinery, tmp_path):
    spans, recs, provider, lines = _run(layout, machinery, tmp_path, limit_lines=120, chunk_chars=6000)
    want = {i for i in range(50, 401, 50)}
    assert {s.start for s in spans} == want
    assert all(lines[s.start - 1] == s.quote for s in spans)        # the line cited is the line quoted
    end = recs[-1]
    assert end["examined"] == 1 and end["counts"].get("episodes_failed", 0) == 0
    assert end["counts"]["chunks"] >= 5 and "chunks_split" not in end["counts"]
    noms = [r for r in recs if r["kind"] == "nomination"]
    assert noms[0]["episode"].startswith(f"{PATH}#L1-L") and noms[-1]["episode"].endswith("-L400")
    assert all("Lines " in req.messages[0].content[0].text for _, req in provider.sent)


def test_a_chunk_that_runs_out_is_halved_and_asked_again(layout, machinery, tmp_path):
    spans, recs, _, _ = _run(layout, machinery, tmp_path, limit_lines=40, chunk_chars=6000)
    assert {s.start for s in spans} == {i for i in range(50, 401, 50)}
    end = recs[-1]
    assert end["counts"]["chunks_split"] >= 1 and end["counts"].get("episodes_failed", 0) == 0
    calls = [r for r in recs if r["kind"] == "model_call"]
    assert any(c["stop_reason"] == "max_tokens" for c in calls)


def test_what_still_runs_out_after_the_last_split_fails_and_says_why(layout, machinery, tmp_path):
    spans, recs, provider, _ = _run(layout, machinery, tmp_path, limit_lines=0, chunk_chars=6000, max_splits=2)
    assert spans == []
    end = recs[-1]["counts"]
    failed = [r for r in recs if r["kind"] == "nomination" and r.get("error")]
    assert all("ran out of output tokens" in r["error"] for r in failed)
    assert end["episodes_failed"] == len(failed) == 4 * end["chunks"] // 7     # each chunk: 1 + 2 + 4 asks, 4 fail
    assert len(provider.sent) == end["chunks"]                                   # the cost is bounded by max_splits
