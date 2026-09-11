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


def test_the_vendored_click_layout_is_preferred_where_present() -> None:
    """The branch ``_click.py`` falls back to today, and why it must not.

    On a Typer < 0.27 environment there is no ``typer._click`` to import, so the
    module's ``try`` body runs only far enough to raise ``ModuleNotFoundError`` and
    the ``except`` always wins -- ``VENDORED_CLICK`` is ``False`` and the four
    names come from the standalone ``click`` package. That is exactly what this
    machine has, so every other test in this file exercises only the fallback and
    the vendored branch (lines 36-39) is never reached.

    That is not merely a coverage gap: when both layouts are importable the shim
    must prefer the vendored one, because that is the click Typer actually raises
    -- the original bug was that ``isinstance(exc, click.ClickException)`` was
    ``False`` for an exception Typer had just raised. A mutation that dropped the
    vendored import would let this module silently resolve the wrong classes again
    and only fail on a future Typer, after the ''wrong classes'' bug returned.

    So this test fabricates a ``typer._click`` layout, reloads the shim, and
    asserts it stayed on the vendored branch; the ``finally`` restores whatever
    `typer` and ``sys.modules`` really had and reloads once more, so whatever ran
    afterwards sees the real ``Abort``/``Exit``/etc. ``importlib.reload`` mutates
    the one module object every test shares in ``sys.modules``, so leaving a fake
    layout behind would poison the CLI tests.
    """
    import importlib
    import types

    from trainai.cli import _click

    fake_root = types.ModuleType("typer._click")
    fake_root.__package__ = "typer._click"
    fake_root.__path__ = []

    fake_core = types.ModuleType("typer._click.core")
    fake_core.__package__ = "typer._click.core"
    fake_exc = types.ModuleType("typer._click.exceptions")
    fake_exc.__package__ = "typer._click.exceptions"

    # The vendored classes are whatever Typer raises, so they are caught by base
    # class rather than name. The assertions below check identity, not the names.
    fake_core.Abort = type("Abort", (BaseException,), {})
    fake_core.Exit = type("Exit", (BaseException,), {"exit_code": 0})
    fake_exc.ClickException = type("ClickException", (BaseException,), {})
    fake_exc.UsageError = type("UsageError", (fake_exc.ClickException,), {"exit_code": 2})

    fake_root.core = fake_core
    fake_root.exceptions = fake_exc

    real_typer = sys.modules["typer"]
    had_attr = hasattr(real_typer, "_click")
    if had_attr:
        saved_click = real_typer._click
    # Which submodules were already imported, so teardown can put each entry back.
    saved_modules = {
        name: sys.modules.get(name)
        for name in ("typer._click", "typer._click.core", "typer._click.exceptions")
    }

    sys.modules["typer._click"] = fake_root
    sys.modules["typer._click.core"] = fake_core
    sys.modules["typer._click.exceptions"] = fake_exc
    real_typer._click = fake_root

    try:
        importlib.reload(_click)
        assert _click.VENDORED_CLICK is True, "vendored layout must be preferred"
        assert _click.Exit is fake_core.Exit, "the vendored Exit is the one Typer raises"
        assert _click.UsageError is fake_exc.UsageError, (
            "the vendored UsageError is the one Typer raises"
        )
        assert _click.UsageError.__bases__ == (fake_exc.ClickException,)
    finally:
        for name, original in saved_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        if had_attr:
            real_typer._click = saved_click
        else:
            delattr(real_typer, "_click")
        # Reload with the fabrication gone, so the module resolves whatever this
        # environment's *real* Typer provides, and every later import (the CLI
        # tests, conftest) sees the classes Typer here actually raises.
        importlib.reload(_click)
