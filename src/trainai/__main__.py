"""Allow ``python -m trainai`` as an alternative to the ``trainai`` script."""

from __future__ import annotations

from trainai.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
