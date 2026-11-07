"""A mistake in what a command is given is one line on stderr saying what to do, never a traceback; a failure in
Recant's own code says so and how to report it. RECANT_DEBUG=1 shows the traceback."""
import json

import pytest

from recant import cli


@pytest.fixture
def here(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RECANT_HOME", str(tmp_path / "rh"))
    return tmp_path


def one_line(capsys):
    err = capsys.readouterr().err.strip()
    assert "Traceback" not in err and len(err.splitlines()) == 1, err
    return err


def test_a_missing_usage_file_is_one_line(here):
    with pytest.raises(SystemExit) as e:
        cli.main(["coverage", "--usage", str(here / "nope.jsonl")])
    assert str(e.value).startswith("--usage: ") and "No such file" in str(e.value) and "\n" not in str(e.value)


def test_a_bad_hook_module_or_function_is_one_line(here, capsys):
    (here / "tasks.jsonl").write_text("".join(json.dumps({"id": i}) + "\n" for i in range(6)))
    (here / "mymod.py").write_text("def other(task, client):\n    return 1\n")
    for spec, says in (("nomod:run_task", "no module 'nomod' here"), ("mymod:run_task", "has no function 'run_task'")):
        with pytest.raises(SystemExit) as e:
            cli.main(["bench", "--tasks", str(here / "tasks.jsonl"), "--run", spec, "--estimate", "--calibrate", "2",
                      "--out", str(here / "b")])
        assert says in str(e.value)


def test_a_bad_tenant_is_one_line(here, capsys):
    (here / "s").mkdir()
    assert cli.main(["cost", "--store", str(here / "s"), "--tenant", "a/b"]) == 2
    assert "tenant id must be one safe path segment" in one_line(capsys)


def test_a_failure_in_recant_itself_says_it_is_ours(here, capsys, monkeypatch):
    from recant import doctor

    monkeypatch.setattr(doctor, "_check", {}.__getitem__)          # a TypeError raised inside doctor.run, Recant's code
    assert cli.main(["doctor"]) == 3
    line = one_line(capsys)
    assert "an error in Recant itself" in line and "recant feedback" in line and "RECANT_DEBUG=1" in line
    monkeypatch.setenv("RECANT_DEBUG", "1")
    with pytest.raises(TypeError):
        cli.main(["doctor"])
    with pytest.raises(KeyError) as e:                              # a KeyError in Recant's code is Recant's too
        doctor.render([{}])
    assert cli._ours(e.value) is True


def test_an_error_raised_in_the_callers_own_code_is_theirs(tmp_path, monkeypatch):
    (tmp_path / "hookmod.py").write_text("def run_task(task, client):\n    return {}['missing']\n")
    import importlib.util
    spec = importlib.util.spec_from_file_location("a_hook", tmp_path / "hookmod.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(KeyError) as e:
        mod.run_task({}, None)
    assert cli._ours(e.value) is False


@pytest.mark.parametrize("argv", [["bench"], ["cost", "--bogus"], ["nosuchcommand"]])
def test_a_usage_mistake_is_one_line_on_stderr_and_exit_2(argv, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "--help" in err[0], err


def test_the_top_level_help_starts_with_the_recipe(capsys):
    import pytest as _pytest
    with _pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert "client = recant.wrap(anthropic.Anthropic())" in out and "never change prompts, models" in out


def test_a_frame_in_an_installed_package_is_not_taken_for_the_standard_library(tmp_path, monkeypatch):
    import sysconfig
    site = sysconfig.get_paths()["purelib"]
    code = compile("def hook(prompt):\n    return {}['missing']\n", f"{site}/somevendor/hook.py", "exec")
    ns: dict = {}
    exec(code, ns)
    from recant import feedback
    with pytest.raises(KeyError) as e:
        feedback._answers_or_blank(ns["hook"])("q")                       # Recant's frame, then the installed one
    assert cli._ours(e.value) is False                                    # the deepest is the package's: not Recant's
