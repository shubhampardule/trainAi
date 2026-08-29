"""The CLI must keep working across Typer's click reshuffle.

Typer 0.27 vendored click into ``typer._click`` and dropped the ``click`` dependency,
which broke this project in two ways at once: ``import click`` became an ImportError
on a fresh install, and where click was present anyway its exception classes were no
longer the ones Typer raises, so every usage error escaped ``main()`` as a traceback
instead of becoming exit code 2.

Nothing here names a version. Each test asks Typer to produce the thing and then
checks that what :mod:`trainai.cli._click` resolved is what came out -- so these keep
holding through the next reshuffle rather than pinning today's layout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import typer
import typer.main

from trainai.cli._click import Abort, ClickException, Exit, UsageError, is_group
from trainai.cli.main import app, main
from trainai.errors import ExitCode

SRC = Path(__file__).resolve().parents[1] / "src" / "trainai"


def _tiny_app() -> typer.Typer:
    """A throwaway app, so the assertions do not depend on trainai's own flags."""
    scratch = typer.Typer()

    @scratch.command()
    def only(name: str = "world") -> None:  # pragma: no cover - never invoked
        pass

    return scratch


def test_a_usage_error_typer_raises_is_the_class_we_catch() -> None:
    """The bug, stated as a test: a bad option must be a ``ClickException`` to us.

    This is the assertion that failed on Typer 0.27 while every version pin still
    looked satisfied -- ``isinstance(exc, click.ClickException)`` was False for an
    exception Typer had just raised.
    """
    with pytest.raises(BaseException) as caught:
        _tiny_app()(args=["--no-such-flag"], standalone_mode=False)

    assert isinstance(caught.value, ClickException), (
        f"Typer raised {type(caught.value).__module__}.{type(caught.value).__qualname__}, "
        "which trainai.cli._click did not resolve. main() will let it escape as a "
        "traceback instead of returning ExitCode.USAGE."
    )
    assert isinstance(caught.value, UsageError)
    assert hasattr(caught.value, "exit_code")
    assert hasattr(caught.value, "show")


def test_abort_and_exit_carry_what_main_reads_off_them() -> None:
    """``main()`` returns ``exc.exit_code``, so the attribute has to be there."""
    assert issubclass(Abort, BaseException)
    assert hasattr(Exit(1), "exit_code")
    assert hasattr(typer.Exit(1), "exit_code")
    assert issubclass(typer.Abort, BaseException)


def test_is_group_tells_a_group_from_a_leaf_on_the_real_cli() -> None:
    """The idiom the test suite navigates the CLI with.

    ``isinstance(command, click.Group)`` used to do this, and on Typer 0.27 it called
    every group a leaf -- which made sixty tests fail with messages about missing
    flags rather than about the class that had moved.
    """
    root = typer.main.get_command(app)
    assert is_group(root), "the trainai root command is a group"
    assert is_group(root.commands["data"]), "trainai data is a group"
    assert not is_group(root.commands["doctor"]), "trainai doctor is a leaf"


def test_a_mistyped_flag_exits_two_rather_than_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The user-visible half of the same bug, end to end through ``main()``."""
    monkeypatch.setattr(sys, "argv", ["trainai", "doctor", "--no-such-flag"])
    assert main() == ExitCode.USAGE


def test_the_cli_package_does_not_import_the_standalone_click() -> None:
    """The guard that keeps this from coming back.

    ``click`` is not a dependency of this project and, since Typer 0.27, not a
    dependency of Typer either. A module that imports it works on the developer's
    machine for exactly as long as something else in the environment happens to
    install it, and fails on a user's first ``pip install trainai``.

    ``_click.py`` is the one exception: answering this question is its whole job.
    """
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "_click.py":
            continue
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith(("import click", "from click ")):
                offenders.append(f"{path.relative_to(SRC.parent.parent)}:{number}: {stripped}")
    assert not offenders, (
        "these modules import the standalone click package:\n  "
        + "\n  ".join(offenders)
        + "\nImport from trainai.cli._click instead; see that module for why."
    )


def test_click_is_not_declared_as_a_dependency() -> None:
    """If it were, the import above would be legal and the shim pointless."""
    pyproject = (SRC.parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    dependency_block = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "click" not in dependency_block, (
        "click is declared a dependency. Either remove it, or delete "
        "trainai/cli/_click.py and import click directly -- but not both."
    )
