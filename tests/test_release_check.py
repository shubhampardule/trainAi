"""Tests for the release check -- the gate on the one artifact that cannot be corrected.

`tools/release_check.py` is what stands between a tag and three mistakes that are each
cheap to make and expensive to undo: a tag naming a version `pyproject.toml` does not
package, a tag without the `v` that `.github/workflows/release.yml` silently ignores, and
a release whose own notes still sit under `## [Unreleased]`.

The failure paths are what matter, for the same reason they matter in
`test_coverage_floor.py`: this runs in CI, where a gate that passes when it should not is
silent. So every refusal here is asserted on its *message* as well as its exit code -- the
whole value of the check is that whoever hits it can tell which of the two numbers is
wrong without reading the script.

The last few tests read `.github/workflows/release.yml` as text. That workflow passes flags
to this script as literal strings in a shell step, so a renamed option would fail at the
one moment it is most expensive; checking the workflow's flags against the real parser is
the cheapest way to keep the two from drifting.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "release_check.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"


def load_script() -> Any:
    """Import the script by path: `tools/` is a directory of scripts, not a package."""
    spec = importlib.util.spec_from_file_location("release_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release = load_script()

#: Every test that parses a real `pyproject.toml` needs :mod:`tomllib`, which arrived in
#: 3.11 while this project supports 3.10. `packaged_version` refuses below that on purpose
#: -- its docstring says why a regex is the worse trade for a script whose whole job is
#: catching a near-miss version -- and `main` turns that refusal into exit 1. Skipping is
#: what makes these honest rather than merely green: two of them assert exit 1, so on 3.10
#: they passed for a reason that had nothing to do with the tag or the notes they exist to
#: check. The boundary is spelled the way the script spells it so the two cannot drift, and
#: the refusal itself is asserted below on whichever interpreter is running.
needs_tomllib = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="tomllib is 3.11+; the release workflow pins 3.12 and CI covers 3.11-3.13",
)

PYPROJECT = '[project]\nname = "trainai"\nversion = "{version}"\n'

CHANGELOG = """# Changelog

## [Unreleased]

## [0.1.0] - 2026-09-03

### Fixed

- Something that was broken.

## [0.0.9] - 2026-08-01

- An older entry.
"""


def root(tmp_path: Path, *, version: str = "0.1.0", changelog: str = CHANGELOG) -> Path:
    """A repository root holding only the two files the check reads."""
    (tmp_path / "pyproject.toml").write_text(
        PYPROJECT.format(version=version), encoding="utf-8", newline="\n"
    )
    (tmp_path / "CHANGELOG.md").write_text(changelog, encoding="utf-8", newline="\n")
    return tmp_path


# --------------------------------------------------------------------------- #
# The tag
# --------------------------------------------------------------------------- #
def test_a_tag_that_matches_the_packaged_version_is_accepted() -> None:
    assert release.check_tag("v0.1.0", "0.1.0") == []


@pytest.mark.parametrize("version", ["0.1.0a1", "0.2.0b3", "1.0.0rc2"])
def test_a_prerelease_tag_is_accepted(version: str) -> None:
    """Alpha, beta and rc are shapes this project may plausibly release."""
    assert release.check_tag(f"v{version}", version) == []


def test_a_tag_without_the_v_is_refused_and_says_why_that_is_the_worse_failure() -> None:
    """The mistake with no symptom: `tags: ["v*"]` does not match, so nothing runs.

    A tag that fails the release is annoying. A tag that produces *no* release looks
    exactly like a tag that worked until somebody goes looking for the artifacts, so the
    message has to say so rather than only naming the expected shape.
    """
    (problem,) = release.check_tag("0.1.0", "0.1.0")
    assert "'v0.1.0'" in problem
    assert "no release at all" in problem


@pytest.mark.parametrize("tag", ["v0.1", "v0.1.0.1", "0.1.0", "release-0.1.0", "v0.1.0-final", ""])
def test_a_tag_shape_this_project_does_not_release_is_refused(tag: str) -> None:
    assert release.check_tag(tag, "0.1.0") != []


def test_a_tag_naming_a_different_version_names_both_numbers() -> None:
    """Whoever hits this has to be able to tell which of the two is wrong."""
    (problem,) = release.check_tag("v0.2.0", "0.1.0")
    assert "0.2.0" in problem and "0.1.0" in problem
    assert "trainai-0.1.0" in problem, "the message should name the files that would be built"


# --------------------------------------------------------------------------- #
# The changelog
# --------------------------------------------------------------------------- #
def test_a_dated_section_with_entries_is_accepted_and_its_body_is_returned() -> None:
    body, problems = release.changelog_section(CHANGELOG, "0.1.0")
    assert problems == []
    assert body is not None
    assert "Something that was broken." in body
    assert "An older entry." not in body, "the body must stop at the next '## ' heading"
    assert "0.0.9" not in body


def test_a_version_that_is_still_unreleased_is_refused_with_the_steps_to_cut_it() -> None:
    """The real mistake: tagging v0.2.0 while the changelog still says Unreleased.

    Nothing about the repository looks wrong when this happens -- the tag is well formed
    and the version matches -- so the message carries the whole fix.
    """
    body, (problem,) = release.changelog_section(CHANGELOG, "0.2.0")
    assert body is None
    assert "## [0.2.0]" in problem
    assert "## [Unreleased]" in problem
    assert "YYYY-MM-DD" in problem


def test_a_changelog_with_no_unreleased_heading_at_all_says_so_too() -> None:
    """Two problems, and reporting only the first would send someone the wrong way."""
    _body, (problem,) = release.changelog_section("# Changelog\n\nnothing here\n", "0.1.0")
    assert "its own bug" in problem


def test_an_undated_release_heading_is_refused() -> None:
    """`## [0.1.0]` with no date. The release notes are dated from that line."""
    _body, (problem,) = release.changelog_section("# C\n\n## [0.1.0]\n\n- A thing.\n", "0.1.0")
    assert "YYYY-MM-DD" in problem
    assert "line 3" in problem


def test_a_released_section_with_no_entries_is_refused() -> None:
    """Renaming the heading and writing nothing reads as though nothing changed."""
    text = "# C\n\n## [0.1.0] - 2026-09-03\n\n## [0.0.9] - 2026-08-01\n\n- Old.\n"
    _body, (problem,) = release.changelog_section(text, "0.1.0")
    assert "no entries" in problem


def test_two_sections_for_one_version_are_refused() -> None:
    """A copy-paste that leaves two headings makes "the body" meaningless."""
    text = CHANGELOG + "\n## [0.1.0] - 2026-09-04\n\n- A duplicate.\n"
    body, (problem,) = release.changelog_section(text, "0.1.0")
    assert body is None
    assert "2 sections" in problem


# --------------------------------------------------------------------------- #
# End to end, including the notes file
# --------------------------------------------------------------------------- #
@needs_tomllib
def test_a_release_that_agrees_passes_and_writes_its_notes(tmp_path: Path) -> None:
    notes = tmp_path / "notes.md"
    code = release.main(
        ["--tag", "v0.1.0", "--root", str(root(tmp_path)), "--notes-out", str(notes)]
    )
    assert code == 0
    assert "Something that was broken." in notes.read_text(encoding="utf-8")


@needs_tomllib
@pytest.mark.parametrize("tag", ["0.1.0", "v0.2.0"])
def test_a_failed_check_writes_no_notes(tmp_path: Path, tag: str) -> None:
    """A failed check must not leave a notes file behind for a later step to publish.

    The workflow's draft job runs the check *and* takes the notes from the same command,
    so if a refusal still wrote the file, a `continue-on-error` or a reordering would
    publish notes for a release that was rejected.
    """
    notes = tmp_path / "notes.md"
    code = release.main(["--tag", tag, "--root", str(root(tmp_path)), "--notes-out", str(notes)])
    assert code == 1
    assert not notes.exists()


@needs_tomllib
def test_no_notes_are_written_for_a_section_that_was_found_but_refused(tmp_path: Path) -> None:
    """The case the two above cannot reach: a body exists *and* the check failed.

    A bad tag stops before the changelog is read, so nothing was ever found to write. An
    empty-but-dated section is different -- the section is there, its body is returned so
    the caller can quote it, and it is still a refusal. That is the only shape in which
    writing the notes before testing `problems` would actually publish something, so
    without this case a reordering of `main` passes every other test here.
    """
    empty = "# C\n\n## [Unreleased]\n\n## [0.1.0] - 2026-09-03\n"
    notes = tmp_path / "notes.md"
    changed = root(tmp_path, changelog=empty)
    assert release.main(["--tag", "v0.1.0", "--root", str(changed), "--notes-out", str(notes)]) == 1
    assert not notes.exists()


@needs_tomllib
def test_without_a_tag_an_uncut_changelog_is_reported_rather_than_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A manual dispatch has no tag and is allowed to build an uncut version.

    This is the form that runs on `workflow_dispatch`, which is how the release workflow
    gets tested at all without creating a tag. It reports and returns 0.
    """
    assert release.main(["--root", str(root(tmp_path, version="0.9.9"))]) == 0
    out = capsys.readouterr().out
    assert "v0.9.9" in out
    assert "not yet cut" in out


@needs_tomllib
def test_a_missing_version_is_a_failure_not_a_crash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "trainai"\n', encoding="utf-8", newline="\n"
    )
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8", newline="\n")
    assert release.main(["--tag", "v0.1.0", "--root", str(tmp_path)]) == 1
    assert "no project.version" in capsys.readouterr().err


@needs_tomllib
def test_the_version_this_repository_packages_could_be_tagged() -> None:
    """The real `pyproject.toml`, so a version shape no tag can express fails here.

    Deliberately not asserting that the changelog section for it exists: it does not, and
    it should not until a release is actually cut. That half of the check is proved above
    against fixture text.
    """
    version = release.packaged_version((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert release.check_tag(f"v{version}", version) == []


def test_the_interpreter_floor_refuses_with_a_way_out_and_says_so_on_every_version() -> None:
    """What `packaged_version` does on the interpreter running it, whichever one that is.

    The CI matrix includes 3.10, where there is no :mod:`tomllib`, so every other test in
    this section skips there -- which would leave two of the six jobs asserting nothing at
    all about the release check. This one says something on all of them. Below 3.11 the
    refusal has to name both the interpreter it found and one that works, because the
    alternative a contributor gets is `ModuleNotFoundError: No module named 'tomllib'` and
    a guess; from 3.11 the same call has to return the version. Written as one test with
    two branches rather than two skipped tests so that neither branch can rot unnoticed on
    the interpreter that does not run it.
    """
    pyproject = PYPROJECT.format(version="1.2.3")
    running = f"{sys.version_info.major}.{sys.version_info.minor}"

    if sys.version_info < (3, 11):
        with pytest.raises(ValueError) as caught:
            release.packaged_version(pyproject)
        message = str(caught.value)
        assert running in message, f"the refusal does not name the interpreter: {message}"
        assert "3.11" in message, f"the refusal does not name a way out: {message}"
    else:
        assert release.packaged_version(pyproject) == "1.2.3"


# --------------------------------------------------------------------------- #
# The workflow that runs it
# --------------------------------------------------------------------------- #
def workflow_text() -> str:
    """The release workflow, or a skip when this is an unpacked sdist.

    `.github/` is deliberately not shipped in the sdist -- see `require_github_directory`
    in `test_conventions.py` -- and `release.yml` is the workflow that runs the suite from
    that sdist, so these checks have to say "not here" there rather than fail.
    """
    if not WORKFLOW.exists():
        pytest.skip(f"{WORKFLOW.name} is absent, so this is an unpacked sdist")
    return WORKFLOW.read_text(encoding="utf-8")


def invocations(text: str) -> list[str]:
    """The argument strings the workflow passes to this script, one per call."""
    return [m.group(1) for m in re.finditer(r"release_check\.py([^\n]*)", text)]


def test_the_workflow_calls_this_script_with_flags_it_actually_has() -> None:
    """`quickstart` shipped broken for exactly this reason, in the CLI rather than here.

    A flag is passed as literal text in a shell step, so a renamed option is not a type
    error and not a lint failure -- it is an exception on a tag push, at the one moment
    when re-running is not free. `_actions` is argparse's own record of what was
    registered; reading it beats re-typing the option names in this test.
    """
    text = workflow_text()
    calls = invocations(text)
    assert len(calls) >= 2, f"expected the check and the notes call, found {calls}"

    known = {
        option for action in release.build_parser()._actions for option in action.option_strings
    }
    for call in calls:
        for flag in re.findall(r"--[\w-]+", call):
            assert flag in known, f"the workflow passes {flag}, which the parser does not accept"

    assert any("--notes-out" in call for call in calls), (
        "nothing takes the release notes from the changelog, so they would be typed twice"
    )


def test_the_workflow_triggers_on_every_tag_shape_the_script_accepts() -> None:
    """A filter narrower than the check is the failure with no symptom.

    If the script accepts `v0.1.0rc1` and the workflow only matched `v?.?.?`, the release
    for a prerelease tag would simply never run -- and the tag would look fine.
    """
    text = workflow_text()
    assert 'tags: ["v*"]' in text, (
        "the tag filter is quoted in the refusal message, so it is pinned"
    )
    for tag in ("v0.1.0", "v0.2.0a1", "v1.0.0rc2", "v10.20.30"):
        assert release.check_tag(tag, tag[1:]) == []
        assert fnmatch.fnmatch(tag, "v*"), f"{tag} passes the check but not the trigger"


def test_the_workflow_stops_at_a_draft_and_publishes_to_no_index() -> None:
    """Nothing here has ever been uploaded to an index, and the workflow must not imply it.

    A `pypi-publish` step that has never once succeeded is worse than no step: it makes the
    first real release a debugging session, and it makes the repository look as though
    `pip install trainai` works.
    """
    text = workflow_text()
    assert "--draft" in text, "the release must be a draft; publishing is a person's decision"
    for forbidden in ("twine upload", "pypa/gh-action-pypi-publish", "PYPI_API_TOKEN", "TWINE_"):
        assert forbidden not in text, f"{forbidden!r} publishes to an index; nothing here should"


def test_the_check_cannot_be_bypassed_by_the_jobs_that_follow_it() -> None:
    """Job order is the only thing making the check a gate rather than a report."""
    text = workflow_text()
    assert "needs: agree" in text, "the build must not start before the tag is checked"
    assert "needs: artifacts" in text, "the draft must not be created before the build passes"


def test_the_workflow_tests_the_artifact_rather_than_the_checkout() -> None:
    """The one thing this workflow does that CI cannot: run the suite from the sdist.

    CI installs with `pip install -e .`, so it never exercises the files anybody would
    download. A module missing from the wheel and a file the tests read missing from the
    sdist are both invisible to an editable install.
    """
    text = workflow_text()
    assert "dist/*.whl" in text, "the wheel has to be installed as a wheel"
    assert "--strict" in text, "twine check --strict, because a rendering warning matters here"
    assert re.search(r"tar -xzf dist/\*\.tar\.gz", text), "the sdist has to be unpacked and run"
