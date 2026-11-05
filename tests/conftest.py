import pytest

from recant.log import LogWriter
from recant.machinery import Machinery
from recant.store import MemoryStore, StoreLayout

from recant_fixtures import ALLOWLIST, ARTIFACT_YAML, KERNEL_YAML, TENANT


@pytest.fixture(autouse=True)
def _friction_log_in_tmp(tmp_path, monkeypatch):
    """the CLI appends to a friction log in RECANT_HOME; tests keep it out of the real home."""
    monkeypatch.setenv("RECANT_HOME", str(tmp_path / "recant-home"))


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """A retryable provider error is tried again after a backoff; no test waits it out."""
    from recant.provider.base import BaseProvider
    monkeypatch.setattr(BaseProvider, "sleep", staticmethod(lambda s: None))


@pytest.fixture(scope="session", autouse=True)
def _fence_everything_live():
    """No credential, no service manager, for the whole session (fence.py)."""
    from fence import install
    mp = pytest.MonkeyPatch()
    install(mp)
    yield
    mp.undo()


@pytest.fixture
def layout(tmp_path):
    lay = StoreLayout(tmp_path / "store")
    lay.ensure(TENANT)
    return lay


@pytest.fixture
def machinery(layout, tmp_path):
    a, k = tmp_path / "artifact.yaml", tmp_path / "kernel.yaml"
    a.write_text(ARTIFACT_YAML)
    k.write_text(KERNEL_YAML)
    Machinery.seed(layout, a, k)
    return Machinery.load(layout)


@pytest.fixture
def log(layout, machinery):
    writer = LogWriter(layout, TENANT, "test", artifact_hash=machinery.artifact_hash,
                       kernel_hash=machinery.kernel_hash)
    yield writer
    writer.finish()


@pytest.fixture
def store(layout):
    return MemoryStore(layout, ALLOWLIST)
