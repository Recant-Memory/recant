"""`recant --version` and every report header name the commit a release was built from (`_commit`, written by the
publish workflow); with none recorded they say so in words. In 0.3.1 a PyPI install printed `recant 0.3.1 (unknown,
unknown)`, two unknowns the README explained one of."""
from pathlib import Path

from recant import version


def test_a_release_build_names_its_commit(tmp_path, monkeypatch):
    (tmp_path / "_commit").write_text("1cdf63996a1e" + "0" * 28 + "\n")
    monkeypatch.setattr(version, "__file__", str(tmp_path / "version.py"))
    monkeypatch.setattr(version.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no git")))
    import importlib.metadata as md
    monkeypatch.setattr(md, "distribution", lambda name: (_ for _ in ()).throw(md.PackageNotFoundError(name)))
    version.info.cache_clear()
    try:
        i = version.info()
        assert (i["commit"][:12], i["commit_from"]) == ("1cdf63996a1e", "release build")
        assert version.line().endswith("(1cdf63996a1e, release build)")
    finally:
        version.info.cache_clear()


def test_no_commit_is_said_in_words():
    assert version.describe({"version": "0.3.2", "commit": "unknown", "commit_from": "unknown"}) == \
        "0.3.2 (commit not recorded in this install)"


def test_the_workflow_writes_the_commit_and_the_wheel_ships_it():
    root = Path(__file__).resolve().parents[1]                       # the repository, not the installed package
    assert 'echo "$GITHUB_SHA" > src/recant/_commit' in (root / ".github/workflows/publish.yml").read_text()
    assert '"_commit"' in (root / "pyproject.toml").read_text()
