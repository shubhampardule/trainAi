"""Check that a tag, the packaged version and the changelog agree -- before tagging.

Run it before creating the tag, which is the only moment the answer is still cheap::

    python tools/release_check.py --tag v0.1.0

A tag is the one artifact in this project that cannot be quietly corrected. It is what
``pip install git+https://...@v0.1.0`` resolves, so moving one that anybody has already
fetched breaks their checkout instead of fixing it, and deleting one breaks every release
built from it. So the checks run before the tag exists as well as after:
``.github/workflows/release.yml`` runs this same command on the tag push, which means a
tag created without running it locally still cannot produce a release.

The three failures it exists to catch are all cheap to make and expensive to undo:

* ``v0.1.1`` pushed while ``pyproject.toml`` still says ``0.1.0`` -- a release whose
  artifacts are named after a different version than the tag that produced them.
* ``0.1.0`` pushed without the ``v``, which the release workflow's ``tags: ["v*"]``
  filter does not match. Nothing happens at all, and the tag itself looks fine.
* ``v0.1.0`` pushed while ``CHANGELOG.md`` still files every entry under
  ``## [Unreleased]`` -- a release whose own notes say it is unreleased.

Without ``--tag`` it checks only what does not need one: that the packaged version has a
shape a tag could be built from, and reports what the changelog currently says about it.
That is the form the workflow runs on a manual dispatch, where there is no tag to compare.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Tags this project creates. PEP 440 permits a great deal more; nothing here needs it,
#: and a tag shape nobody has thought about is one the release workflow may not match.
TAG = re.compile(r"^v(?P<version>\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?)$")

#: A released section in Keep a Changelog form: ``## [0.1.0] - 2026-09-03``.
RELEASED = "## [{version}]"

#: The heading everything sits under until a release is cut.
UNRELEASED = "## [Unreleased]"


def packaged_version(pyproject: str) -> str:
    """Return ``project.version`` from *pyproject*, or raise ``ValueError``.

    Parsed with ``tomllib`` rather than a regex because a near-miss here names the
    wrong version in a released artifact, which is exactly the failure this script is
    for. That costs Python 3.11, which the release workflow pins anyway.
    """
    if sys.version_info < (3, 11):
        raise ValueError(
            "release_check.py needs Python 3.11 or newer for tomllib, and this is "
            f"{sys.version_info.major}.{sys.version_info.minor}. The release workflow runs "
            "3.12; to run this check locally, use a 3.11+ interpreter."
        )
    import tomllib

    table = tomllib.loads(pyproject)
    try:
        version = table["project"]["version"]
    except KeyError as exc:
        raise ValueError(f"pyproject.toml has no project.version ({exc})") from exc
    if not isinstance(version, str) or not version:
        raise ValueError(f"project.version is not a non-empty string: {version!r}")
    return version


def check_tag(tag: str, version: str) -> list[str]:
    """Return the problems with *tag* as a tag for *version*. Empty means agreement."""
    match = TAG.match(tag)
    if match is None:
        return [
            f"the tag {tag!r} is not a shape this project releases. Tags are the packaged "
            f"version with a leading 'v' and nothing else, so for {version} that is "
            f"'v{version}'. Note that .github/workflows/release.yml only triggers on "
            "'v*', so a tag without the 'v' does not fail the release -- it produces no "
            "release at all, which is harder to notice."
        ]
    tagged = match.group("version")
    if tagged != version:
        return [
            f"the tag says {tagged} and pyproject.toml says {version}. Whichever is wrong, "
            "fix it before the tag is pushed anywhere: the built sdist and wheel are named "
            f"from pyproject.toml, so this tag would publish trainai-{version} files under "
            f"a {tagged} release."
        ]
    return []


def changelog_section(changelog: str, version: str) -> tuple[str | None, list[str]]:
    """Return the body of *version*'s section in *changelog*, and any problems with it.

    A missing section is a problem; so is a section with no entries under it, because
    renaming the heading and forgetting to write anything produces a release whose notes
    are a date. The body is returned so the caller can quote it.
    """
    heading = RELEASED.format(version=version)
    lines = changelog.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(heading)]
    if not starts:
        problem = (
            f"CHANGELOG.md has no {heading} section. Keep a Changelog files work under "
            f"'{UNRELEASED}' until a release is cut, and cutting it means renaming that "
            f"heading to '{heading} - <YYYY-MM-DD>' and opening a fresh empty "
            f"'{UNRELEASED}' above it."
        )
        if UNRELEASED not in changelog:
            problem += f" There is no '{UNRELEASED}' heading either, which is its own bug."
        return None, [problem]
    if len(starts) > 1:
        return None, [f"CHANGELOG.md has {len(starts)} sections headed {heading}"]

    start = starts[0]
    if not re.match(rf"^{re.escape(heading)} - \d{{4}}-\d\d-\d\d\s*$", lines[start]):
        return None, [
            f"CHANGELOG.md line {start + 1} is {lines[start]!r}; a released section is "
            f"'{heading} - <YYYY-MM-DD>', which is what the release notes are dated from."
        ]
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    body = "\n".join(lines[start + 1 : end]).strip()
    if not any(line.startswith("- ") for line in lines[start + 1 : end]):
        return body, [
            f"CHANGELOG.md's {heading} section has no entries. A release whose notes are "
            "a date is worse than no notes, because it reads as though nothing changed."
        ]
    return body, []


def build_parser() -> argparse.ArgumentParser:
    """The command line, built separately so a test can ask what options exist.

    ``.github/workflows/release.yml`` is the only caller that matters and it passes flags
    as literal text in a shell step, where a renamed option fails at the one moment it
    costs the most. `test_release_check.py` reads those steps and checks every flag in
    them against this parser, so the workflow cannot drift away from the script.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tag",
        help="the tag being released, e.g. v0.1.0. Omit it to check only what a tag is "
        "not needed for.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="repository root to read pyproject.toml and CHANGELOG.md from",
    )
    parser.add_argument(
        "--notes-out",
        type=Path,
        help="write the changelog section for this version here, for the release body. "
        "Only written when every check passed, so a failed check cannot publish notes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        version = packaged_version((args.root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"release check FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"pyproject.toml version: {version}")
    changelog = (args.root / "CHANGELOG.md").read_text(encoding="utf-8")

    if args.tag is None:
        # No tag to compare against, so report rather than judge: a manual dispatch is
        # allowed to build a version whose changelog section has not been cut yet.
        print(f"no --tag given; the tag for this version would be v{version}")
        _body, problems = changelog_section(changelog, version)
        state = "released" if not problems else "not yet cut"
        print(f"CHANGELOG.md section for {version}: {state}")
        return 0

    body = None
    problems = check_tag(args.tag, version)
    if not problems:
        print(f"tag {args.tag} agrees with the packaged version")
        body, problems = changelog_section(changelog, version)
        if not problems:
            print(f"CHANGELOG.md has a dated {version} section with entries")

    if problems:
        for problem in problems:
            print(f"release check FAILED: {problem}", file=sys.stderr)
        return 1

    if args.notes_out is not None:
        assert body is not None  # unreachable: no problems means a body was found
        args.notes_out.write_text(body + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {len(body)} characters of release notes to {args.notes_out}")
    print("release check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
