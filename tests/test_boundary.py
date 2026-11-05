"""The dependency points one way: the library imports nothing but the standard library, itself, its declared
dependencies and its own test helpers, so no consuming application can creep in. And no machine path appears in it.
(A consumer checks the library for its own names in its own suite; the library cannot know them.)"""
import ast
import sys
from pathlib import Path

LIB = Path(__file__).resolve().parents[1]
DEPENDENCIES = {"pydantic", "yaml", "ulid"}                   # pyproject's dependencies, by import name
EXTRAS = {"anthropic", "httpx", "httpx2", "pytest", "openai"}   # the [anthropic] and [test] extras, and what the SDKs send through
PATHS = ["/" + "Users" + "/", "/" + "home" + "/"]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "build", "dist"}


def _files(suffixes=None):
    for path in LIB.rglob("*"):
        if path.is_file() and not (set(path.parts) & SKIP_DIRS) and ".egg-info" not in str(path):
            if suffixes is None or path.suffix in suffixes:
                yield path


def _own_modules() -> set:
    """Recant itself, and every module file of the library's own (test helpers, examples)."""
    return {"recant"} | {p.stem for p in _files({".py"})}


def _top_imports(tree) -> list[str]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append(node.module.split(".")[0])
    return out


def _foreign(tree) -> list[str]:
    allowed = set(sys.stdlib_module_names) | DEPENDENCIES | EXTRAS | _own_modules()
    return [m for m in _top_imports(tree) if m not in allowed]


def test_library_imports_only_the_stdlib_itself_and_its_declared_dependencies():
    offenders = [f"{p.relative_to(LIB)}: imports {m}" for p in _files({".py"}) for m in _foreign(ast.parse(p.read_text()))]
    assert not offenders, offenders


def test_the_check_itself_can_fail():
    """A boundary test that cannot fail proves nothing: a consuming application's package is caught."""
    assert _foreign(ast.parse("from api.settings import Settings\nimport someapp.config\n")) == ["api", "someapp"]


def test_no_machine_path_in_the_library():
    offenders = []
    for path in _files():
        try:
            body = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        offenders += [f"{path.relative_to(LIB)}: contains {p!r}" for p in PATHS if p in body]
    assert not offenders, offenders


def test_only_the_adapter_imports_the_vendor_sdk():
    offenders = []
    for path in _files({".py"}):
        if path.name == "anthropic.py" or "tests" in path.parts or "examples" in path.parts:   # an example is a stranger's
            continue                                                                            # agent, on the raw SDK by design
        if "anthropic" in _top_imports(ast.parse(path.read_text())):
            offenders.append(str(path.relative_to(LIB)))
    assert not offenders, offenders
