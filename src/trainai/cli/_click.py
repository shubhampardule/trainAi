"""The click that Typer is actually using.

Typer 0.27 vendored click into ``typer._click`` and dropped ``click`` from its
dependencies. Two consequences, and both of them break a CLI that imports click
directly:

* In an environment where only this project's declared dependencies are present
  there is no ``click`` to import at all, so a module-level ``import click`` ends
  the program before it starts.
* Where click *is* installed alongside -- pulled in by something else in the
  environment -- its classes are no longer the ones Typer raises. A usage error
  arrives as ``typer._click.exceptions.NoSuchOption``, and
  ``isinstance(exc, click.ClickException)`` is ``False``. The handler does not fire,
  the exception escapes ``main()``, and a mistyped flag prints a traceback instead
  of the exit code :class:`~trainai.errors.ExitCode.USAGE` documented for it.

Both spellings are supported here rather than pinning ``typer<0.27`` in
``pyproject.toml``. A ceiling would put this project in conflict with anything else
in the user's environment that wants a newer Typer, and it would age into a bug of
its own; the version question is better answered once, in four lines, than avoided.

The vendored module is tried *first*, and deliberately so. When both are importable
it is the vendored classes Typer raises, so preferring the standalone package would
reproduce the original bug on precisely the versions that have it.

``tests/test_cli_typer_compat.py`` asserts that what is resolved here is what Typer
really raises, by raising it. That test does not name a version, so it keeps holding
when the next reshuffle happens.
"""

from __future__ import annotations

from typing import Any

try:  # Typer >= 0.27, which vendors click and does not depend on the package.
    from typer._click.core import Abort, Exit
    from typer._click.exceptions import ClickException, UsageError

    VENDORED_CLICK = True
except ImportError:  # Typer < 0.27, which uses the standalone click package.
    from click.exceptions import Abort, ClickException, Exit, UsageError

    VENDORED_CLICK = False


def is_group(command: Any) -> bool:
    """True when ``command`` holds subcommands.

    Asked by attribute rather than by class because Typer's vendored click has no
    ``Group`` type at all -- ``TyperGroup`` derives straight from ``Command``, and
    a group is simply a command with a ``commands`` mapping. There is therefore no
    ``isinstance`` test that can be written against both layouts, and the one that
    used to be written here silently reported every group as a leaf.
    """
    return isinstance(getattr(command, "commands", None), dict)


__all__ = [
    "VENDORED_CLICK",
    "Abort",
    "ClickException",
    "Exit",
    "UsageError",
    "is_group",
]
