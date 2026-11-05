import hashlib
import os

import pytest

from recant.records import RecordError
from recant.store import IngestConflict, ingest, resolve
from recant.store.episodes import read_manifest
from recant_fixtures import ALLOWLIST, TENANT


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    (root / "notes/.hidden/talk").mkdir(parents=True)
    (root / "notes/2026-01-02.md").write_text("line one\nline two\nline three\n")
    (root / "notes/2026-01-03.md").write_text("another day\n")
    (root / "notes/credentials.md").write_text("SECRET=hunter2\n")
    (root / "notes/topic.md").write_text("derived summary\n")
    (root / "notes/.hidden/events.jsonl").write_text('{"telemetry": true}\n')
    (root / "notes/.hidden/talk/2026-01-02.txt").write_text("[s#L1] User: hi\n[s#L2] Assistant: hello\n")
    (root / "notes/.hidden/talk/deeper").mkdir()
    (root / "notes/.hidden/talk/deeper/2026-01-02.txt").write_text("too deep\n")
    (root / "DREAMS.md").write_text("exhaust\n")
    os.symlink(root / "notes/credentials.md", root / "notes/2026-01-09.md")  # a link dressed as a daily
    return root


def test_ingest_copies_exactly_what_the_allowlist_admits(layout, corpus):
    result = ingest(layout, TENANT, corpus, ALLOWLIST)
    assert sorted(result.copied) == ["notes/.hidden/talk/2026-01-02.txt", "notes/2026-01-02.md", "notes/2026-01-03.md"]
    cold = layout.tier_dir(TENANT, "cold")
    found = sorted(p.relative_to(cold).as_posix() for p in cold.rglob("*") if p.is_file())
    assert found == ["MANIFEST.json", "notes/.hidden/talk/2026-01-02.txt", "notes/2026-01-02.md", "notes/2026-01-03.md"]
    assert "hunter2" not in "".join(p.read_text() for p in cold.rglob("*") if p.is_file())


def test_manifest_hashes_match_the_source_bytes(layout, corpus):
    ingest(layout, TENANT, corpus, ALLOWLIST)
    for entry in read_manifest(layout, TENANT):
        src = (corpus / entry["path"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(src).hexdigest() and entry["bytes"] == len(src)
        assert (layout.tier_dir(TENANT, "cold") / entry["path"]).read_bytes() == src


def test_ingest_is_idempotent_and_refuses_a_changed_file(layout, corpus):
    ingest(layout, TENANT, corpus, ALLOWLIST)
    again = ingest(layout, TENANT, corpus, ALLOWLIST)
    assert again.copied == [] and len(again.skipped) == 3
    (corpus / "notes/2026-01-03.md").write_text("rewritten history\n")
    with pytest.raises(IngestConflict):
        ingest(layout, TENANT, corpus, ALLOWLIST)
    assert (layout.tier_dir(TENANT, "cold") / "notes/2026-01-03.md").read_text() == "another day\n"


def test_ingest_needs_an_allowlist_and_a_relative_one(layout, corpus):
    with pytest.raises(RecordError):
        ingest(layout, TENANT, corpus, [])
    with pytest.raises(RecordError):
        ingest(layout, TENANT, corpus, ["../*.md"])
    with pytest.raises(RecordError, match="recurse"):  # a recursive glob is a wider glob
        ingest(layout, TENANT, corpus, ["notes/**/*.txt"])
    assert not any(layout.tier_dir(TENANT, "cold").rglob("*.txt"))


def test_resolve_returns_the_span(layout, corpus):
    ingest(layout, TENANT, corpus, ALLOWLIST)
    assert resolve(layout, TENANT, "notes/2026-01-02.md#L2-L3") == "line two\nline three"
    assert resolve(layout, TENANT, "notes/.hidden/talk/2026-01-02.txt#L2-L2") == "[s#L2] Assistant: hello"
    with pytest.raises(RecordError):
        resolve(layout, TENANT, "notes/2026-01-02.md#L2-L9")
    with pytest.raises(FileNotFoundError):
        resolve(layout, TENANT, "notes/credentials.md#L1-L1")
