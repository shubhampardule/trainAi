"""Tests for the coverage floor -- the gate that would have caught a module at 0%.

`tools/coverage_floor.py` exists because this project reached 92% line coverage with
`src/trainai/cli/setup.py` at exactly 0%: a global `--cov-fail-under` cannot see one small
module losing every test it had. A gate nothing tests is worth very little, and this one
runs in CI where a false pass is silent, so the failure paths are what matter here: a
module below the floor has to be *named*, and an exemption for a module that no longer
exists has to be an error rather than a line nobody reads.

The reports are written by hand rather than produced by running coverage, because what is
under test is the reading of the document and the decision, not coverage.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FLOOR_SCRIPT = REPO_ROOT / "tools" / "coverage_floor.py"


def load_floor() -> Any:
    """Import the script by path: `tools/` is a directory of scripts, not a package."""
    spec = importlib.util.spec_from_file_location("coverage_floor", FLOOR_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


floor = load_floor()


def report(path: Path, *, total: float, files: dict[str, float]) -> Path:
    """A Cobertura report with only the two attributes the gate reads."""
    classes = "\n".join(
        f'      <class filename="{name}" line-rate="{rate}" branch-rate="1"/>'
        for name, rate in files.items()
    )
    path.write_text(
        f'<?xml version="1.0" ?>\n'
        f'<coverage line-rate="{total}" branch-rate="1">\n'
        f"  <packages>\n    <package>\n      <classes>\n{classes}\n"
        f"      </classes>\n    </package>\n  </packages>\n</coverage>\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


EXEMPTED = "src/trainai/__main__.py"


def healthy(tmp_path: Path, **overrides: float) -> Path:
    """A report that passes, which every failure case then perturbs by one thing."""
    files = {"src/trainai/cli/plan.py": 1.0, EXEMPTED: 0.0}
    files.update(overrides)
    return report(tmp_path / "coverage.xml", total=0.93, files=files)


def test_a_healthy_report_passes_and_says_what_it_checked(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert floor.main(["_", str(healthy(tmp_path))]) == 0

    printed = capsys.readouterr().out
    assert "floor met" in printed
    assert "exempted" in printed, "a silent pass hides that a module is being skipped"


def test_a_module_below_the_floor_fails_and_is_named(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Naming it is the point. "coverage too low" sends the reader to the whole report."""
    path = healthy(tmp_path, **{"src/trainai/hardware/probe.py": 0.42})

    assert floor.main(["_", str(path)]) == 1

    complaint = capsys.readouterr().err
    assert "hardware/probe.py" in complaint
    assert "42.00%" in complaint
    assert "EXEMPT" in complaint, "the way out has to be stated, or the gate gets deleted"


def test_the_exempted_module_is_allowed_to_be_at_zero(tmp_path: Path) -> None:
    """`__main__.py` is two lines under `if __name__` and is tested by subprocess."""
    assert floor.main(["_", str(healthy(tmp_path, **{EXEMPTED: 0.0}))]) == 0


def test_a_low_total_fails_even_when_every_module_clears_the_floor(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The two floors catch different things: one module gutted, or the whole suite off."""
    path = report(
        tmp_path / "coverage.xml", total=0.7, files={"src/trainai/cli/plan.py": 0.7, EXEMPTED: 0.0}
    )

    assert floor.main(["_", str(path)]) == 1
    assert "total line coverage is 70.00%" in capsys.readouterr().err


def test_an_exemption_for_a_module_that_is_gone_is_an_error(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Otherwise a rename leaves a permanent hole that reads like a considered decision."""
    path = report(tmp_path / "coverage.xml", total=0.93, files={"src/trainai/cli/plan.py": 1.0})

    assert floor.main(["_", str(path)]) == 1

    complaint = capsys.readouterr().err
    assert EXEMPTED in complaint
    assert "renamed or deleted" in complaint


def test_a_missing_report_is_not_a_pass(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """The dangerous failure: `--cov-report=xml` dropped, and the gate waves it through."""
    assert floor.main(["_", str(tmp_path / "nothing.xml")]) == 2
    assert "not found" in capsys.readouterr().err


def test_a_report_with_no_files_is_not_a_pass(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = report(tmp_path / "coverage.xml", total=1.0, files={})

    assert floor.main(["_", str(path)]) == 2
    assert "no files" in capsys.readouterr().err


def test_windows_separators_in_a_report_are_read_as_posix(tmp_path: Path) -> None:
    """coverage.py writes `src\\trainai\\...` on Windows, and the exemptions are POSIX.

    Without the normalisation the exempt list matches nothing on the four Windows jobs, so
    `__main__.py` fails the floor there and passes on Linux -- and the stale-exemption
    check fires at the same time, for a module that is present.
    """
    path = report(
        tmp_path / "coverage.xml",
        total=0.93,
        files={"src\\trainai\\cli\\plan.py": 1.0, "src\\trainai\\__main__.py": 0.0},
    )

    assert floor.main(["_", str(path)]) == 0


def test_every_exemption_names_a_module_that_exists_and_gives_a_reason() -> None:
    """Checked without a coverage run, so a rename fails here rather than only in CI."""
    for name, reason in floor.EXEMPT.items():
        assert (REPO_ROOT / name).exists(), f"{name} is exempted but does not exist"
        assert len(reason) > 40, f"the exemption for {name} does not explain itself"


def test_the_floors_are_the_ones_the_documentation_promises() -> None:
    """The numbers are quoted in CONTRIBUTING.md, and a floor raised in silence is a trap."""
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

    assert f"{floor.TOTAL_FLOOR:.0f}%" in contributing
    assert f"{floor.PER_FILE_FLOOR:.0f}%" in contributing
