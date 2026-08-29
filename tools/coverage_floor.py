"""Fail the build when a module loses its tests, not merely when the total slips.

Run after ``pytest --cov-report=xml``::

    python tools/coverage_floor.py coverage.xml

**Why per file.** This project reached 92% line coverage with ``src/trainai/cli/setup.py``
at exactly 0% -- the one command that downloads about 2.5 GB and replaces PyTorch in the
user's environment, including its refusal to touch a system Python and its confirmation
prompt, had no test at all. A global ``--cov-fail-under`` would never have said so: that
module is 66 of 6392 statements, so deleting every one of its tests moves the total by
about one point. The check that catches an untested module has to look at the modules.

**Why the floors are low.** :data:`PER_FILE_FLOOR` has to clear the *lowest* real number on
*every* platform in the matrix, and the spread is large because the code that reads the
machine is full of platform branches: ``hardware/probe.py`` measures 70% on Linux and 80%
on Windows, and neither is a gap -- each job cannot execute the other's code. A floor set
just under the Windows number would fail on Linux for no defect. So this is a floor, not a
target: it catches a module that lost its tests, and it is not a substitute for reading the
``term-missing`` report.

The gate deliberately does *not* enforce branch coverage per file. ``probe.py`` sits at 61%
branches on Linux for the same platform reason, and a floor loose enough to accept that
would catch nothing that the line floor does not already catch. One honest knob beats two
that each need their own exemption list.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

#: Total line coverage, over everything. Measured at 93.2% on Linux and 92% on Windows
#: when this was written; the margin absorbs a platform difference, not a lost test file.
TOTAL_FLOOR = 90.0

#: Per-module line coverage. See the module docstring for why this is not 80.
PER_FILE_FLOOR = 60.0

#: Modules exempted from the per-file floor, each with the reason. This list is the
#: honest half of the gate: an entry here is a documented gap, and adding one is a
#: decision someone has to write down rather than a number quietly drifting down.
EXEMPT: dict[str, str] = {
    "src/trainai/__main__.py": (
        "two lines guarded by `if __name__`. Importing it does not run them, and it is "
        "tested by running `python -m trainai --version` in a subprocess "
        "(test_conventions.py), whose coverage this process does not collect."
    ),
}


def rates(report: Path) -> tuple[float, dict[str, float]]:
    """Total line rate and per-file line rate, as percentages, from a coverage XML."""
    root = ET.parse(report).getroot()
    total = float(root.get("line-rate", "0")) * 100
    per_file: dict[str, float] = {}
    for element in root.iter("class"):
        name = (element.get("filename") or "").replace("\\", "/")
        per_file[name] = float(element.get("line-rate", "0")) * 100
    return total, per_file


def main(argv: list[str]) -> int:
    report = Path(argv[1] if len(argv) > 1 else "coverage.xml")
    if not report.exists():
        print(f"coverage report not found: {report}", file=sys.stderr)
        return 2

    total, per_file = rates(report)
    if not per_file:
        print(f"{report} contains no files; was the run collected?", file=sys.stderr)
        return 2

    failures: list[str] = []
    if total < TOTAL_FLOOR:
        failures.append(f"total line coverage is {total:.2f}%, below the {TOTAL_FLOOR:.0f}% floor")

    for name, rate in sorted(per_file.items()):
        if name in EXEMPT:
            continue
        if rate < PER_FILE_FLOOR:
            failures.append(f"{name} is at {rate:.2f}%, below the {PER_FILE_FLOOR:.0f}% floor")

    stale = sorted(name for name in EXEMPT if name not in per_file)
    if stale:
        failures.append(
            f"these modules are exempted but are not in the report (renamed or deleted?): "
            f"{', '.join(stale)}"
        )

    if failures:
        print(f"Coverage floor not met ({len(failures)} problems):", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        print(
            "\nAdd tests for the module named above. If the gap is deliberate, add it to "
            "EXEMPT in tools/coverage_floor.py with the reason -- a gap someone chose is "
            "fine, a gap nobody noticed is what this exists to prevent.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Coverage floor met: total {total:.2f}% (floor {TOTAL_FLOOR:.0f}%), "
        f"{len(per_file)} modules checked at or above {PER_FILE_FLOOR:.0f}%, "
        f"{len(EXEMPT)} exempted."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
