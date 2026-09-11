"""Tests for ``trainai setup`` -- the command that offers to reinstall PyTorch.

This file exists because coverage said `src/trainai/cli/setup.py` was at **0%**: the
one command in the project that downloads 2.5 GB and replaces a package in the user's
environment had no test at all. The advice it prints was tested (`test_install.py`);
what it *does* with that advice was not.

The two guards are the reason this matters more than the report formatting. `--install`
refuses to touch a Python that is not in a virtual environment, and asks before running
anything. A regression in either one is not a cosmetic bug -- it is TrainAI overwriting
a shared interpreter without asking.

`subprocess.run` is replaced in every test here. Nothing in this file may install
anything, and a test that let a real pip run would take minutes and change the machine
it ran on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from conftest import flat, unwrapped
from trainai.cli.setup import run_setup
from trainai.errors import ExitCode, UsageError
from trainai.hardware.install import PYTORCH_SELECTOR, InstallAdvice

CUDA_COMMAND = "pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu126"


def advice(**overrides: Any) -> InstallAdvice:
    """A plausible "NVIDIA card, CPU-only wheel" diagnosis -- the case this command is for."""
    fields: dict[str, Any] = {
        "situation": "This machine has an NVIDIA GPU, but PyTorch is the CPU-only build.",
        "needs_action": True,
        "command": CUDA_COMMAND,
        "notes": ["Training on the CPU build is 20-100x slower than this card allows."],
        "detected_vendors": ["nvidia"],
        "driver_cuda_version": (12, 6),
        "torch_build": "cpu",
    }
    fields.update(overrides)
    return InstallAdvice(**fields)


class Ran:
    """A stand-in for ``subprocess.run`` that records instead of installing."""

    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.calls: list[list[str]] = []

    def __call__(self, arguments: list[str], **_kwargs: Any) -> Ran:
        self.calls.append(list(arguments))
        return self


@pytest.fixture
def diagnosis(monkeypatch: pytest.MonkeyPatch) -> Ran:
    """Fix the diagnosis, the environment and the installer for one test.

    Every one of these reads the real machine, and this command's whole job is to react
    to what it finds. Left alone, each test would assert on whatever hardware it happened
    to run on -- which is how a quota test on this project ended up secretly asserting
    the core count of a CI runner.
    """
    monkeypatch.setattr("trainai.cli.setup.detect_gpu_vendors", lambda: ["nvidia"])
    monkeypatch.setattr("trainai.cli.setup.advise_install", lambda vendors: advice())
    monkeypatch.setattr("trainai.cli.setup.in_virtualenv", lambda: True)
    ran = Ran()
    monkeypatch.setattr("trainai.cli.setup.subprocess.run", ran)
    return ran


# --------------------------------------------------------------------------- #
# The default: report, and do nothing
# --------------------------------------------------------------------------- #
def test_without_install_it_only_prints_the_command(
    capsys: pytest.CaptureFixture[str], diagnosis: Ran
) -> None:
    result = run_setup()

    assert result.command == CUDA_COMMAND
    output = capsys.readouterr().out
    assert CUDA_COMMAND in output, "the command must be copyable, so it must appear intact"
    assert "--install" in flat(output), "the report has to say how to act on it"
    assert diagnosis.calls == [], "printing advice must not install anything"


def test_the_printed_install_command_is_not_wrapped(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, diagnosis: Ran
) -> None:
    """The pip command is the one in this project most likely to be pasted blind.

    It is long, it contains a URL, and a user who is being told their install is wrong is
    in no position to reconstruct it from two wrapped lines. Printed through
    `console.print` it broke at the console width; asserted here at a width narrower
    than the command so that a revert fails rather than merely looking untidy.
    """
    from rich.console import Console

    narrow = Console(width=50)
    monkeypatch.setattr("trainai.console.console", narrow)
    monkeypatch.setattr("trainai.cli.setup.console", narrow)

    run_setup()

    assert len(CUDA_COMMAND) > 50
    assert CUDA_COMMAND in capsys.readouterr().out


def test_a_correct_install_is_reported_as_correct(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, diagnosis: Ran
) -> None:
    monkeypatch.setattr(
        "trainai.cli.setup.advise_install",
        lambda vendors: advice(
            situation="PyTorch is built for CUDA 12.6 and this machine has an NVIDIA GPU.",
            needs_action=False,
            command=None,
            notes=[],
            torch_build="cuda 12.6",
        ),
    )

    result = run_setup()

    assert result.needs_action is False
    assert "Nothing to add" in flat(capsys.readouterr().out)


def test_a_problem_with_no_install_that_would_help_says_so(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, diagnosis: Ran
) -> None:
    """An AMD card on Windows, for instance. There is no wheel; saying nothing is worse."""
    monkeypatch.setattr(
        "trainai.cli.setup.advise_install",
        lambda vendors: advice(
            situation="This machine has an AMD GPU, and there is no ROCm wheel for Windows.",
            command=None,
            detected_vendors=["amd"],
        ),
    )

    run_setup()

    assert "No install command would help here" in flat(capsys.readouterr().out)


def test_json_is_machine_readable_and_says_where_it_would_install(
    capsys: pytest.CaptureFixture[str], diagnosis: Ran
) -> None:
    run_setup(json_output=True)

    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == CUDA_COMMAND
    assert payload["needs_action"] is True
    assert payload["torch_build"] == "cpu"
    assert payload["driver_cuda_version"] == "12.6"
    assert payload["selector"] == PYTORCH_SELECTOR
    # Both are about *this* interpreter, and a caller deciding whether to run the
    # command needs them: the same advice is safe in a venv and not outside one.
    assert payload["in_virtualenv"] is True
    assert payload["python"] == sys.version.split()[0]


# --------------------------------------------------------------------------- #
# The guards on --install
# --------------------------------------------------------------------------- #
def test_it_refuses_to_install_into_a_system_python(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], diagnosis: Ran
) -> None:
    """The guard that matters most: no venv, no --allow-global, no install."""
    monkeypatch.setattr("trainai.cli.setup.in_virtualenv", lambda: False)

    with pytest.raises(UsageError) as caught:
        run_setup(install=True, assume_yes=True)

    error = caught.value
    assert error.exit_code == ExitCode.USAGE
    assert "system Python" in error.message
    assert "venv" in (error.hint or ""), "a refusal has to say what to do instead"
    assert error.details["python"] == sys.executable
    assert diagnosis.calls == [], "the refusal must happen before pip is started"


def test_allow_global_is_what_lifts_the_refusal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], diagnosis: Ran
) -> None:
    monkeypatch.setattr("trainai.cli.setup.in_virtualenv", lambda: False)

    run_setup(install=True, allow_global=True, assume_yes=True)

    assert len(diagnosis.calls) == 1
    assert "Virtual env" in flat(capsys.readouterr().out)


def test_a_declined_prompt_installs_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], diagnosis: Ran
) -> None:
    from trainai import console as console_module

    monkeypatch.setattr(console_module.console, "input", lambda _prompt="": "no")

    run_setup(install=True)

    assert diagnosis.calls == []
    output = flat(capsys.readouterr().out)
    assert "Nothing was installed" in output
    assert "2.5 GB" in output, "the prompt has to say how large the download is"


@pytest.mark.parametrize("answer", ["y", "Y", "yes", " YES "])
def test_the_prompt_accepts_the_obvious_yeses(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], diagnosis: Ran, answer: str
) -> None:
    from trainai import console as console_module

    monkeypatch.setattr(console_module.console, "input", lambda _prompt="": answer)

    run_setup(install=True)

    assert len(diagnosis.calls) == 1, f"{answer!r} should have been taken as consent"


@pytest.mark.parametrize("answer", ["", "n", "later", "ok", "sure"])
def test_anything_else_is_a_no(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], diagnosis: Ran, answer: str
) -> None:
    """Bare Enter is a no, and so is anything ambiguous.

    "ok" and "sure" read as consent to a human and are refused on purpose: the cost of
    a wrong yes here is a 2.5 GB download and a replaced PyTorch, and the cost of a
    wrong no is retyping the command.
    """
    from trainai import console as console_module

    monkeypatch.setattr(console_module.console, "input", lambda _prompt="": answer)

    run_setup(install=True)

    assert diagnosis.calls == [], f"{answer!r} must not be taken as consent"


# --------------------------------------------------------------------------- #
# Running pip
# --------------------------------------------------------------------------- #
def test_pip_runs_as_a_module_of_this_interpreter(
    capsys: pytest.CaptureFixture[str], diagnosis: Ran
) -> None:
    """`pip install ...` becomes `<this python> -m pip install ...`, and that is the fix.

    A bare `pip` on PATH is whichever one the shell finds, which on a machine with
    several environments is regularly not the one that will `import torch` afterwards.
    Installing into the wrong interpreter is the exact failure this command exists to
    diagnose, so it would be an unusually poor way to fail.
    """
    run_setup(install=True, assume_yes=True)

    assert len(diagnosis.calls) == 1
    argv = diagnosis.calls[0]
    assert argv[:3] == [sys.executable, "-m", "pip"]
    assert argv[3:] == CUDA_COMMAND.split()[1:]
    assert "pip" not in argv[3:4], "the literal `pip` argument must have been replaced"


def test_a_command_that_is_not_pip_is_run_unchanged(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], diagnosis: Ran
) -> None:
    """Only a leading `pip` is rewritten. Nothing else is assumed to be pip-shaped."""
    monkeypatch.setattr(
        "trainai.cli.setup.advise_install",
        lambda vendors: advice(command="uv pip install torch"),
    )

    run_setup(install=True, assume_yes=True)

    assert diagnosis.calls == [["uv", "pip", "install", "torch"]]


def test_a_failed_install_names_the_status_and_says_nothing_else_changed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("trainai.cli.setup.detect_gpu_vendors", lambda: ["nvidia"])
    monkeypatch.setattr("trainai.cli.setup.advise_install", lambda vendors: advice())
    monkeypatch.setattr("trainai.cli.setup.in_virtualenv", lambda: True)
    monkeypatch.setattr("trainai.cli.setup.subprocess.run", Ran(returncode=1))

    with pytest.raises(UsageError) as caught:
        run_setup(install=True, assume_yes=True)

    error = caught.value
    assert "status 1" in error.message
    assert PYTORCH_SELECTOR in (error.hint or ""), "a 404 on the index needs somewhere to look"
    assert error.details["returncode"] == 1
    assert error.details["command"].startswith(sys.executable)


def test_a_successful_install_points_at_doctor_and_warns_about_the_loaded_torch(
    capsys: pytest.CaptureFixture[str], diagnosis: Ran
) -> None:
    """The new build is not in this process, and a user who reruns `setup` will be confused.

    `import torch` has already happened by the time pip finishes, so the process still
    holds the old build. Saying so is the difference between "it worked" and "it said it
    worked and then reported the same problem again".
    """
    run_setup(install=True, assume_yes=True)

    captured = capsys.readouterr()
    assert "trainai doctor" in flat(captured.out)
    assert "next command" in flat(captured.err), "the caveat belongs on stderr, not in the report"


def test_the_advice_object_is_returned_for_the_caller(diagnosis: Ran) -> None:
    """The web interface reuses this, so the return value is part of the contract."""
    assert run_setup(json_output=True).command == CUDA_COMMAND
    assert run_setup().command == CUDA_COMMAND


def test_the_report_names_the_interpreter_it_is_talking_about(
    capsys: pytest.CaptureFixture[str], diagnosis: Ran
) -> None:
    run_setup()

    report = unwrapped(capsys.readouterr().out)
    assert Path(sys.executable).name in report


def test_the_driver_row_appears_only_when_a_driver_was_read(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, diagnosis: Ran
) -> None:
    """No `nvidia-smi`, no row -- rather than a row reading "unknown".

    An empty row would be a claim about the driver, and there is nothing to claim: on an
    AMD or Intel machine, or one where `nvidia-smi` is not on PATH, the number was never
    read. `trainai doctor` is where the absence of a GPU gets explained.
    """
    run_setup()
    assert "NVIDIA driver" in flat(capsys.readouterr().out)

    monkeypatch.setattr(
        "trainai.cli.setup.advise_install",
        lambda vendors: advice(driver_cuda_version=None, detected_vendors=[]),
    )
    run_setup()

    report = flat(capsys.readouterr().out)
    assert "NVIDIA driver" not in report
    assert "none detected" in report, "an empty vendor list still has to be stated"


# --------------------------------------------------------------------------- #
# The command's wiring
# --------------------------------------------------------------------------- #
def test_every_setup_flag_reaches_the_argument_it_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test above calls ``run_setup`` directly; this is the one that types the command.

    Four flags, four keyword arguments, and a body that is nothing but the mapping between
    them -- which is exactly the kind of code a wrong-way-round default survives in
    unnoticed, because ``run_setup`` is tested to the hilt and the command is one
    function call. Recorded rather than run, so the machine is not read and nothing is
    installed.

    One flag per invocation, not all of them at once. The first version of this test
    typed ``--install --allow-global --yes`` together and passed with ``allow_global``
    and ``assume_yes`` swapped: three trues are three trues in any order. Typed alone,
    each flag has to light up its own argument and no other.
    """
    from trainai.cli.main import main

    received: list[dict[str, Any]] = []

    def record(**kwargs: Any) -> None:
        received.append(kwargs)

    monkeypatch.setattr("trainai.cli.setup.run_setup", record)
    nothing = {"install": False, "allow_global": False, "assume_yes": False, "json_output": False}

    for flag, argument in [
        (None, None),
        ("--install", "install"),
        ("--allow-global", "allow_global"),
        ("--yes", "assume_yes"),
        ("-y", "assume_yes"),
        ("--json", "json_output"),
    ]:
        monkeypatch.setattr(sys, "argv", ["trainai", "setup", *([flag] if flag else [])])
        assert main() == ExitCode.OK
        assert received[-1] == {**nothing, **({argument: True} if argument else {})}, flag
    assert len(received) == 6


def test_setup_json_through_the_command_is_the_advice(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], diagnosis: Ran
) -> None:
    """The real ``run_setup`` behind the real command, with the diagnosis fixed.

    The test above proves the flags arrive; this proves the command as typed produces
    the document a script would parse, and nothing else on stdout to spoil the parse.
    """
    from trainai.cli.main import main

    monkeypatch.setattr(sys, "argv", ["trainai", "setup", "--json"])

    assert main() == ExitCode.OK

    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == CUDA_COMMAND
    assert payload["in_virtualenv"] is True
    assert diagnosis.calls == []
