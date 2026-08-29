"""Command-line interface.

Command modules in this package must not import ``torch`` at module scope --
see the import-time contract in :mod:`trainai`. Register each command's
implementation in :mod:`trainai.cli.main`, importing heavy modules inside the
command function body.
"""

from __future__ import annotations

__all__: list[str] = []
