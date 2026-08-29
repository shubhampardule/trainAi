"""Project-wide invariants that are easy to break and expensive to notice.

Nothing here tests behaviour. These are structural rules the codebase relies on:

* Source files are pure ASCII, so no editor, terminal or code page can silently
  alter them. The one exception is :mod:`trainai.console`, whose entire purpose is
  to emit the handful of non-ASCII glyphs it defines.
* Importing the library does not import torch. ``trainai --help`` and the whole
  data pipeline must stay fast and must work on a machine with no GPU, and a
  stray top-level ``import torch`` anywhere under :mod:`trainai.data` would end
  that without any test failing.
* Every text write anywhere in the repository states its ``newline``, so the bytes
  TrainAI writes -- and the bytes its fixtures write -- do not depend on which
  platform wrote them. That covers ``write_text``, the builtin ``open``,
  ``Path.open`` and the codec ``open``s, which use three different signatures.
* ``.gitignore`` still decides correctly about files nobody has created yet. The
  tracked-source check below can only notice a rule that is already eating
  something; the rule check asks git about the paths a future ``src/trainai/lib/``
  would occupy, and about the secrets and local settings files that have to be
  ignored before the first one exists.
* ``.gitattributes`` declares every suffix whose bytes must survive a checkout intact.
  The repository is all text today, so nothing here is broken -- which is exactly why
  the rule has to be asserted rather than noticed: the check-out that mangles the first
  compressed fixture is the one nobody is watching.
* Every declared optional extra is one the code actually uses, so
  ``pip install trainai[x]`` never installs packages and delivers nothing.
* Every flag and command the code names in a message is one the CLI really has, so a
  hint never tells a user to run something that does not exist.
* The suite size the README and the hardware notes quote is a floor the suite really
  clears, and one that has not fallen far behind it.
* Every model shape and parameter count the documents quote is one the code produces,
  including inside the output blocks the README presents as captured terminal output.
  Prose is where numbers rot, because nothing imports it.
* A measurement table that documents a flag has a row for every value that flag accepts,
  and every tolerance it quotes -- in the table and in the prose around it -- is the one
  the code applies. A missing row reads as untested rather than as unmeasured.
* Every top-level key a serialised artifact carries is named as a key in that artifact's
  format specification. A reader treats those documents as the list of what is in the
  file, so a key nobody wrote down is one nobody knows to preserve -- and when it is
  inside ``content_hash``, one nobody knows changes the hash.
* Every refusal to load a user-editable file names that file, in the message and in the
  machine-readable ``details``. Enforced over the whole function rather than per case,
  because the branch that gets this wrong is the one nobody wrote a test for.
"""

from __future__ import annotations

import ast
import functools
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple, get_args, get_type_hints

import pytest

from conftest import cli_command_paths, cli_flag_owners, cli_group_paths

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The only module allowed to contain non-ASCII bytes, because the bullet, em dash
#: and ellipsis it defines *are* its purpose. Everything else spells non-ASCII out
#: as escapes or ``chr()`` calls.
ASCII_EXEMPT = {Path("src/trainai/console.py")}


def python_sources() -> list[Path]:
    roots = (REPO_ROOT / "src", REPO_ROOT / "tests", REPO_ROOT / "examples")
    return sorted(path for root in roots for path in root.rglob("*.py"))


def test_there_are_sources_to_check() -> None:
    """Guards against a glob that silently matches nothing."""
    assert len(python_sources()) > 5


@pytest.mark.parametrize("path", python_sources(), ids=lambda p: p.name)
def test_python_sources_are_pure_ascii(path: Path) -> None:
    relative = path.relative_to(REPO_ROOT)
    raw = path.read_bytes()
    offenders = [offset for offset, byte in enumerate(raw) if byte > 127]

    if relative in ASCII_EXEMPT:
        pytest.skip(f"{relative} is a documented exception")

    if offenders:
        line = raw[: offenders[0]].count(b"\n") + 1
        pytest.fail(
            f"{relative} has {len(offenders)} non-ASCII byte(s), first on line {line}. "
            "Write the character as an escape sequence or chr(0x...) instead: a "
            "literal that an editor re-encodes is a change nobody sees in review."
        )


def test_the_ascii_exemption_is_still_earned() -> None:
    """If console.py becomes pure ASCII, the exception should go, not linger."""
    for relative in ASCII_EXEMPT:
        raw = (REPO_ROOT / relative).read_bytes()
        assert any(byte > 127 for byte in raw), (
            f"{relative} no longer contains non-ASCII bytes; remove it from ASCII_EXEMPT"
        )


def test_every_source_file_is_tracked_by_git() -> None:
    """Regression: `.gitignore` had an unanchored `data/`, which matches at any depth.

    That silently excluded the whole of ``src/trainai/data/`` -- ingest, analysis,
    validation, the tokenizer, binarization and the loader -- from the repository.
    Everything passed locally; a clone would have installed a package that could
    not import. The rule ignores build output, so it is exactly the kind of rule
    whose over-reach nothing else notices.
    """
    try:
        listed = subprocess.run(
            ["git", "ls-files", "src", "tests", "examples"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=REPO_ROOT,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout, or git is unavailable")

    tracked = {(REPO_ROOT / line).resolve() for line in listed.stdout.split("\n") if line.strip()}
    on_disk = {
        path.resolve()
        for directory in ("src", "tests", "examples")
        for path in (REPO_ROOT / directory).rglob("*.py")
        if "__pycache__" not in path.parts
    }

    missing = sorted(str(path.relative_to(REPO_ROOT)) for path in on_disk - tracked)
    assert not missing, (
        "these source files exist but git does not track them: "
        f"{missing}. Check .gitignore for an unanchored pattern -- prefix "
        "output directories with '/' so they only match at the repository root."
    )


#: What ``.gitignore`` must decide about paths that do not exist, as
#: ``(path, must_be_ignored, deciding_pattern)``.
#:
#: The test above can only see a file that is already there, so it catches the
#: ``data/`` incident and nothing like it. These are the packages nobody has written
#: yet: ``lib``, ``build``, ``dist``, ``var`` and ``parts`` are all names a build
#: backend uses at the repository root *and* plausible names for a module, and an
#: unanchored rule for one erases the other. That is the same failure as ``data/``
#: with one difference that makes it worse -- ``src/trainai/data/`` at least broke
#: the import, so something noticed.
#:
#: The third column is the pattern that has to be the one deciding, and it is not
#: decoration. Two mutations survived a version of this test that only asserted the
#: verdict. Deleting the ``.local-agent/`` line changed nothing, because one
#: developer's *global* excludes file still covered it -- so the test was measuring
#: that machine rather than this repository. And anchoring ``__pycache__/`` changed
#: nothing, because ``*.py[cod]`` caught the same file for an unrelated reason.
IGNORE_RULES: list[tuple[str, bool, str]] = [
    # Must stay trackable: a source tree may use any of these names.
    ("src/trainai/lib/helpers.py", False, ""),
    ("src/trainai/build/graph.py", False, ""),
    ("src/trainai/dist/sampler.py", False, ""),
    ("src/trainai/var/state.py", False, ""),
    ("src/trainai/parts/attention.py", False, ""),
    ("src/trainai/data/ingest.py", False, ""),
    ("tests/lib/test_helpers.py", False, ""),
    # Must stay ignored: the same names, where a build backend really writes them.
    ("build/lib/trainai/__init__.py", True, "/build/"),
    ("dist/trainai-0.1.0.tar.gz", True, "/dist/"),
    ("lib/python3.13/site-packages/x.py", True, "/lib/"),
    # Unanchored on purpose, and still matching at depth.
    ("src/trainai/data/__pycache__/ingest.pyc", True, "__pycache__/"),
    ("src/trainai.egg-info/PKG-INFO", True, "*.egg-info/"),
    # Artifacts a run drops outside runs/, when someone points --out elsewhere.
    ("mymodel/step-000100.pt", True, "step-*.pt"),
    ("mymodel/train_000.bin", True, "train_*.bin"),
    ("mymodel/model.safetensors", True, "*.safetensors"),
    # Secrets. Nothing here reads one yet, which is the point: the rule has to be in
    # place before the first one exists, because a key in a public commit is the one
    # accident that is not undone by removing it again.
    (".env", True, ".env"),
    (".env.local", True, ".env.*"),
    ("deploy.pem", True, "*.pem"),
    (".env.example", False, ""),
    # Per-machine agent settings, which were ignored only by one developer's *global*
    # excludes file -- something no clone and no CI runner has.
    (".local-agent/settings.local.json", True, ".local-agent/"),
    # Corpus extensions this project reads. `.log` is the sharp one: it is a text
    # alias in `trainai.data.ingest`, so a tidy-looking `*.log` rule would ignore
    # input the tool is documented as accepting.
    ("corpus/notes.log", False, ""),
    ("corpus/papers.tar", False, ""),
    ("corpus/dump.sqlite", False, ""),
]


def git_ignore_verdict(path: str) -> tuple[bool, str, str]:
    """``(ignored, source_file, pattern)`` for ``path``, which need not exist.

    The verdict comes from ``-q``, and the difference from ``-v`` is a trap rather
    than a detail: with ``-v`` git exits 0 when the path matches *any* pattern, a
    negation included, so ``!.env.example`` reads as "ignored" and an assertion
    written the obvious way fails against a rule that is working correctly. ``-v``
    is run separately, only to find out which rule in which file decided.
    """
    run = functools.partial(
        subprocess.run,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=REPO_ROOT,
    )
    quiet = run(["git", "check-ignore", "-q", "--no-index", "--", path])
    if quiet.returncode not in (0, 1):
        pytest.skip(f"git check-ignore is unavailable: {quiet.stderr}")
    verbose = run(["git", "check-ignore", "-v", "--no-index", "--", path])
    if not verbose.stdout.strip():
        return quiet.returncode == 0, "", ""
    # Split from the right: the source is a path, and on Windows it may itself contain
    # a colon -- a global excludes file reads as ``C:\\Users\\...``, which is exactly
    # the source this test exists to reject.
    source, _, pattern = verbose.stdout.split("\t")[0].rsplit(":", 2)
    return quiet.returncode == 0, source, pattern


@pytest.mark.parametrize(
    ("path", "ignored", "pattern"),
    IGNORE_RULES,
    ids=[f"{'ignore' if ignored else 'keep'} {path}" for path, ignored, _ in IGNORE_RULES],
)
def test_gitignore_decides_correctly_about_files_that_do_not_exist_yet(
    path: str, ignored: bool, pattern: str
) -> None:
    """The ignore rules, asserted as rules rather than against the current tree."""
    verdict, source, matched = git_ignore_verdict(path)
    assert verdict is ignored, (
        f"{path} should {'be' if ignored else 'not be'} ignored. "
        "Anchor a build-output directory with a leading '/' so it only matches at "
        "the repository root, and keep patterns that occur at any depth unanchored."
    )
    if not ignored:
        return
    assert source == ".gitignore", (
        f"{path} is ignored, but by {source!r} rather than this repository's "
        ".gitignore. A rule that lives in a global excludes file or in "
        ".git/info/exclude does not exist for a fresh clone or for CI."
    )
    assert matched == pattern, (
        f"{path} is ignored by {matched!r}, not by the {pattern!r} rule this asserts. "
        "The right file is being ignored for the wrong reason, so the rule under test "
        "could be deleted without this failing."
    )


#: Suffixes that must be declared ``binary`` in ``.gitattributes``, in addition to the
#: ones :mod:`trainai.data.ingest` names for itself.
#:
#: ``.pt`` is a checkpoint, and the rest are the image formats the docs embed.
EXTRA_BINARY_SUFFIXES = (".bin", ".safetensors", ".pt", ".png", ".jpg", ".jpeg")


def binary_suffixes_required() -> list[str]:
    """Every suffix whose bytes must survive checkout unchanged.

    Read out of the ingest module rather than listed here, so a codec or container
    added there and not declared in ``.gitattributes`` fails this instead of waiting
    for a fixture to be mangled by it.
    """
    from trainai.data import ingest

    suffixes = {
        *ingest.COMPRESSION_SUFFIXES,
        *ingest.ARCHIVE_SUFFIXES,
        *ingest.DOCX_SUFFIXES,
        *ingest.SQLITE_SUFFIXES,
        *EXTRA_BINARY_SUFFIXES,
    }
    return sorted(suffixes)


@pytest.mark.parametrize("suffix", binary_suffixes_required())
def test_gitattributes_declares_every_non_text_suffix_binary(suffix: str) -> None:
    """A file of these bytes is checked out byte for byte, on every platform.

    ``* text=auto eol=lf`` decides per file by looking for a NUL byte in the first
    8 KiB, and a small archive or a compressed corpus need not have one -- so without
    an explicit rule git converts the line endings inside it and the file no longer
    opens. This project checksums file contents, so that is not cosmetic.

    Asserted through ``git check-attr`` on a path that does not exist, for the same
    reason as the ignore rules above: the declaration has to be right before the first
    such fixture is committed, not after.
    """
    proc = subprocess.run(
        ["git", "check-attr", "text", "diff", "--", f"corpus{suffix}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        pytest.skip(f"git check-attr is unavailable: {proc.stderr}")
    attributes = {}
    for line in proc.stdout.strip().splitlines():
        _, name, value = line.rsplit(": ", 2)
        attributes[name] = value
    assert attributes.get("text") == "unset", (
        f"corpus{suffix} is not declared binary in .gitattributes (text is "
        f"{attributes.get('text')!r}). Add `*{suffix} binary`: eol=lf does not apply to "
        "a path marked binary -- that is checked -- but text=auto does apply to one that "
        "is not, and it rewrites the bytes."
    )
    assert attributes.get("diff") == "unset", (
        f"corpus{suffix} would be diffed as text. `binary` unsets diff as well; a rule "
        "that sets only -text puts a megabyte of compressed bytes in a diff."
    )


#: The newline values a text write may ask for. ``"\n"`` is the answer for anything
#: whose bytes are recorded or compared; ``""`` is the one the :mod:`csv` module
#: requires, since it writes its own line terminators.
ALLOWED_NEWLINES = {"\n", ""}


def text_writes() -> list[tuple[Path, int, str]]:
    """Every ``write_text`` call and text-mode ``open`` for writing in the repository.

    Found by parsing rather than by grepping, so a call split across lines or one
    whose arguments are reordered is still seen.

    ``tests/`` is covered as well as ``src/``. None of the eight test-suite writes
    that first failed this check were writing bytes anything compared -- seven had
    no newline in their content at all -- but a fixture that builds a multi-line
    corpus and then asserts a ``content_hash`` would pass on one platform and fail
    on the other, and a rule enforced over half a tree invites exactly that.
    """
    found: list[tuple[Path, int, str]] = []
    for path in python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            if name == "write_text":
                found.append((path, node.lineno, "write_text"))
            elif name == "open" and _is_a_text_write(node):
                found.append((path, node.lineno, "open"))
    return found


#: Attribute ``open``s whose signature copies the builtin's -- the file first, the
#: mode second. :meth:`pathlib.Path.open` is the exception, and the reason this set
#: has to exist: its mode comes first, because the path is the receiver rather than
#: an argument. Anything not named here is read as a ``Path``.
FILE_FIRST_OPENERS = {"gzip", "bz2", "lzma", "tarfile", "io", "codecs"}


def _receiver(node: ast.Call) -> str:
    """The dotted name an attribute call is made on, or ``""`` if it is not a name.

    ``gzip.open(...)`` gives ``"gzip"``; ``source.path.open(...)`` gives
    ``"source.path"``; ``Path(x).open(...)`` gives ``""``.
    """
    if not isinstance(node.func, ast.Attribute):
        return ""
    parts: list[str] = []
    value: ast.expr = node.func.value
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if not isinstance(value, ast.Name):
        return ""
    parts.append(value.id)
    return ".".join(reversed(parts))


def _mode_of(node: ast.Call) -> str:
    """The mode an ``open`` call asks for, or ``""`` when it is not a literal.

    Three conventions are in play, and collapsing them loses calls silently:

    * the builtin ``open(file, mode)`` and the codec and archive modules
      (``gzip``, ``bz2``, ``lzma``, ``tarfile``) put the file first;
    * ``Path.open(mode)`` puts the mode first, the path being the receiver;
    * ``ZipFile.open(name, mode)`` looks like the first but writes bytes and takes
      no ``newline`` at all.

    Reading ``args[0]`` for every attribute call -- the first draft of this check --
    got only the ``Path`` case right. It read the mode of ``open(path, "w")`` out of
    the *filename*, so ``open(self.path, "a")`` in ``train/metrics.py`` was skipped;
    and it read the mode of ``gzip.open(path, "wt")`` out of the path too, so every
    gzipped and bz2'd corpus fixture in the suite was skipped as well. Both are text
    writes that translate newlines.

    Returning ``""`` for a non-literal mode -- ``tarfile.open(path, mode)`` in the
    ingest tests -- means a computed mode is not checked. That is a real gap, not a
    covered case, and it is small only because there is one such call today.
    """
    position = 1 if isinstance(node.func, ast.Name) or _receiver(node) in FILE_FIRST_OPENERS else 0
    if len(node.args) > position and isinstance(node.args[position], ast.Constant):
        return str(node.args[position].value)  # type: ignore[attr-defined]
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            return str(keyword.value.value)
    return ""


def _is_a_text_write(node: ast.Call) -> bool:
    """Whether an ``open`` call writes text, and so translates newlines on the way out.

    Binary modes translate nothing. A mode holding ``:`` is a :mod:`tarfile` mode
    (``"r:*"``, ``"w:gz"``) -- a different vocabulary, and those streams are bytes.

    Known limitation, in the safe direction: ``ZipFile.open(name, "w")`` would be
    flagged, and takes no ``newline``. There is no such call today, and the failure
    would be loud rather than silent, which is the direction a guard should be
    wrong in.
    """
    mode = _mode_of(node)
    if not mode or "b" in mode or ":" in mode:
        return False
    return any(character in mode for character in "wax+")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('open(path, "w")', "w"),
        ('open(path, mode="w")', "w"),
        ('open("a-path-with-a-w-in-it.txt")', ""),
        ('path.open("w")', "w"),
        ('path.open(mode="w")', "w"),
        ('gzip.open(handle, "wt")', "wt"),
        ('bz2.open(tmp / "d.txt.bz2", "wt")', "wt"),
        ('tarfile.open(path, "r:*")', "r:*"),
        ("container.open(member)", ""),
        ("path.open()", ""),
    ],
)
def test_the_mode_is_read_from_the_right_argument(source: str, expected: str) -> None:
    """The builtin takes the file first; ``Path.open`` and the codecs take the mode first.

    Reading ``args[0]`` for both -- the first draft of this check -- read the mode of
    ``open(path, "w")`` out of the *filename*. Every call in that form was misjudged:
    silently skipped unless the path happened to contain a "w", "a", "x" or "+", and
    ``newline=`` wrongly demanded of reads whose paths did.

    The repository-wide walk cannot pin this on its own. It happens to contain a
    ``Path.open("w")`` in ``examples/``, whose mode *is* at ``args[0]``, so the wrong
    reading still finds an ``open`` of some kind and a both-kinds assertion over the
    walk stays green -- measured: that mutation survived once ``tests/`` and
    ``examples/`` joined the walk, having been caught while it covered ``src/`` alone.
    """
    node = ast.parse(source).body[0].value  # type: ignore[attr-defined]
    assert isinstance(node, ast.Call)
    assert _mode_of(node) == expected


def test_there_are_text_writes_of_both_kinds_to_check() -> None:
    """Guards against an AST walk that silently matches nothing.

    Both kinds, not just a count: a walk that lost the ``open`` half entirely would
    still leave a count over the ``write_text`` half comfortably green. Which
    *argument* the mode is read from is pinned separately, by
    ``test_the_mode_is_read_from_the_right_argument`` -- this assertion is too coarse
    for it.
    """
    kinds = {call for _, _, call in text_writes()}
    assert kinds == {"write_text", "open"}, f"the walk sees only {kinds}"


def test_every_text_write_states_its_newline() -> None:
    """Regression: three writers did not, so their output differed by platform.

    Without ``newline=``, Python translates every ``\\n`` to ``os.linesep`` on the
    way out, so the same content is 144 bytes on Windows and 133 on Linux --
    measured on one small exported ``config.json``. That is a reproducibility hole
    in a project that records and compares byte counts, and ``tests/conftest.py``
    has stated the rule since the suite began without anything enforcing it.
    """
    offenders: list[str] = []
    for path, line, call in text_writes():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        node = next(
            item
            for item in ast.walk(tree)
            if isinstance(item, ast.Call) and item.lineno == line and _call_name(item) == call
        )
        given = {
            keyword.value.value
            for keyword in node.keywords
            if keyword.arg == "newline" and isinstance(keyword.value, ast.Constant)
        }
        if not given or not given <= ALLOWED_NEWLINES:
            relative = path.relative_to(REPO_ROOT).as_posix()
            offenders.append(f"{relative}:{line} ({call}, newline={given or 'absent'})")

    assert not offenders, (
        "these text writes do not state a newline, so their bytes depend on the "
        f'platform: {offenders}. Pass newline="\\n" (or newline="" for a csv '
        "writer, which emits its own terminators)."
    )


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def text_reads() -> list[tuple[Path, int]]:
    """Every :meth:`pathlib.Path.read_text` call in the repository.

    ``read_text`` only, and that boundary is measured rather than lazy. A text-mode
    ``open`` for reading has the same defect -- it decodes with the platform's
    preferred encoding, which is cp1252 on a stock Windows -- but it cannot be
    identified by name. ``X.open(y)`` with no mode is a file read when ``X`` is a
    path and something else entirely when it is not, and every one of the 27
    mode-less ``open`` calls in this repository is the latter:
    ``InferenceSession.open(...)`` in ``cli/chat.py``, ``cli/eval.py``,
    ``export/bundle.py`` and ``tests/test_infer_session.py``, and
    ``ZipFile.open(member)`` -- which yields *bytes* -- at three sites in
    ``data/ingest.py``. A rule covering them would be 27 false positives and zero
    findings, and the noise would be paid on every future ``.open()`` of anything.
    ``read_text`` is unambiguous: pathlib defines it and nothing else in the tree
    does.
    """
    found: list[tuple[Path, int]] = []
    for path in python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node) == "read_text":
                found.append((path, node.lineno))
    return found


def test_there_are_text_reads_to_check() -> None:
    """Guards against an AST walk that silently matches nothing.

    The rule below is a "no offenders" assertion, which a walk finding zero calls
    passes trivially. This file alone makes several ``read_text`` calls, so the floor
    is not arbitrary.
    """
    assert len(text_reads()) > 20, "the walk sees almost no read_text calls"


def test_every_text_read_states_its_encoding() -> None:
    """Regression: a sysfs read did not, and one bad card lost every later card.

    Without ``encoding=``, Python decodes with :func:`locale.getpreferredencoding`,
    which is cp1252 on a stock Windows and UTF-8 on a stock Linux. Two failures
    follow, and the second is the one that bit:

    * the same bytes decode differently by platform, which is the read-side twin of
      the ``newline`` hole ``test_every_text_write_states_its_newline`` closes;
    * the raised :exc:`UnicodeDecodeError` is a :exc:`ValueError`, *not* an
      :exc:`OSError`, so a handler written for a missing or unreadable file does not
      catch it. In ``hardware/probe.py`` that meant an undecodable
      ``/sys/class/drm/cardN/device/vendor`` escaped an inner ``except OSError:
      continue`` and unwound the whole card loop, so ``trainai doctor`` reported no
      GPU vendor at all rather than skipping one card.
    """
    offenders: list[str] = []
    for path, line in text_reads():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        node = next(
            item
            for item in ast.walk(tree)
            if isinstance(item, ast.Call)
            and item.lineno == line
            and _call_name(item) == "read_text"
        )
        if not _states_encoding(node):
            offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{line}")

    assert not offenders, (
        "these text reads do not state an encoding, so they decode with the "
        f"platform's locale and can raise UnicodeDecodeError: {offenders}. Pass "
        'encoding="utf-8" (or the encoding the file actually uses).'
    )


def _states_encoding(node: ast.Call) -> bool:
    """Whether a ``read_text`` call names an encoding.

    Keyword-only, because that is the only way ``read_text`` takes one: its signature
    is ``read_text(encoding=None, errors=None)``, so a positional first argument *is*
    the encoding -- but no call in this repository passes it that way, and reading a
    bare positional as an encoding would misjudge any future overload. Requiring the
    keyword is also the form that survives a signature change.
    """
    return any(keyword.arg == "encoding" for keyword in node.keywords)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('path.read_text(encoding="utf-8")', True),
        ("path.read_text(encoding=chosen)", True),
        ('path.read_text(encoding="utf-8", errors="replace")', True),
        ("path.read_text()", False),
        ('path.read_text("utf-8")', False),
        ('path.read_text(errors="replace")', False),
    ],
)
def test_an_encoding_is_recognised_only_when_it_is_named(source: str, expected: bool) -> None:
    """Pins the predicate directly, because the rule above cannot.

    ``test_every_text_read_states_its_encoding`` asserts an *empty* offender list, so
    once the tree is clean a predicate weakened to ``return True`` passes it --
    measured: that mutation survived the whole suite. The rule catches regressions in
    the tree; only this test catches a regression in the rule.
    """
    node = ast.parse(source).body[0].value  # type: ignore[attr-defined]
    assert isinstance(node, ast.Call)
    assert _states_encoding(node) is expected


#: Keywords that put :func:`subprocess.run` into text mode. Any one of them is enough,
#: which is why the walk checks for the set rather than for ``text=``.
TEXT_MODE_KEYWORDS = {"text", "universal_newlines"}


def _is_a_subprocess_call(node: ast.Call) -> bool:
    """Whether a call is ``subprocess.<something>(...)``, directly or via ``partial``.

    ``functools.partial(subprocess.run, capture_output=True, text=True)`` binds the
    same keywords and has the same defect, but its call node is ``functools.partial``
    and ``subprocess.run`` is only an argument. Missing it would leave a hole in the
    rule that this repository already contains -- there is one such site in this very
    file -- so the wrapper is matched too.
    """
    if _receiver(node) == "subprocess":
        return True
    if _receiver(node) != "functools" or _call_name(node) != "partial":
        return False
    return any(
        isinstance(argument, ast.Attribute)
        and isinstance(argument.value, ast.Name)
        and argument.value.id == "subprocess"
        for argument in node.args
    )


def _decodes_captured_output(node: ast.Call) -> bool:
    """Whether a call captures a child's output *and* asks for it as ``str``.

    Both halves are required, and each excludes a real call in this repository. A
    call that inherits the parent's stdio decodes nothing -- ``cli/setup.py`` runs pip
    that way on purpose, so the user sees pip's own progress live -- and a call that
    captures bytes decodes nothing either. It is the combination that reaches a codec.
    """
    if not _is_a_subprocess_call(node):
        return False
    keywords = {keyword.arg for keyword in node.keywords}
    captures = "capture_output" in keywords or "stdout" in keywords
    return captures and bool(keywords & TEXT_MODE_KEYWORDS)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("subprocess.run(cmd, capture_output=True, text=True)", True),
        ("subprocess.run(cmd, capture_output=True, universal_newlines=True)", True),
        ("subprocess.run(cmd, stdout=subprocess.PIPE, text=True)", True),
        ("functools.partial(subprocess.run, capture_output=True, text=True)", True),
        # Captures bytes: no codec is reached, so no encoding is owed.
        ("subprocess.run(cmd, capture_output=True)", False),
        # Inherits the parent's stdio, which is how `trainai setup` runs pip.
        ("subprocess.run(cmd, check=False)", False),
        ("subprocess.run(cmd, text=True)", False),
        ("functools.partial(other.run, capture_output=True, text=True)", False),
        ("shutil.which(cmd)", False),
    ],
)
def test_a_decoding_subprocess_needs_both_halves(source: str, expected: bool) -> None:
    """Pins the predicate directly, because the tree walk cannot.

    Both halves were measured as unpinnable by the walk alone: widening the predicate
    to ``captures = True`` changed nothing about which calls it found, since the one
    uncaptured call in this repository is also the one with no ``text=``. The rule
    catches regressions in the tree; only this test catches a regression in the rule.
    """
    node = ast.parse(source).body[0].value  # type: ignore[attr-defined]
    assert isinstance(node, ast.Call)
    assert _decodes_captured_output(node) is expected


def decoding_subprocesses() -> list[tuple[Path, int]]:
    """Every ``subprocess`` call in the repository that decodes captured output."""
    found: list[tuple[Path, int]] = []
    for path in python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _decodes_captured_output(node):
                found.append((path, node.lineno))
    return found


def test_there_are_decoding_subprocesses_to_check() -> None:
    """Guards against an AST walk that silently matches nothing.

    Eight, not "more than zero": the ``functools.partial`` wrapper is the eighth and
    is the one a narrower walk drops, so a floor below it would pass without it.
    """
    assert len(decoding_subprocesses()) >= 8, "the walk sees too few decoding subprocesses"


def test_every_decoding_subprocess_states_its_encoding() -> None:
    """Regression: `nvidia-smi` was decoded with the locale's codec, strictly.

    ``text=True`` alone decodes with :func:`locale.getpreferredencoding` and
    ``errors=None``, which is *strict*. ``nvidia-smi`` draws a box-art table and prints
    the card's marketing name, and ``git ls-files`` prints whatever the user named
    their files, so an undecodable byte is ordinary rather than exotic.

    What happens then was measured, and it differs by platform -- both outcomes worse
    than a caught error, and neither reachable by the
    ``(OSError, subprocess.SubprocessError)`` handler these call sites wrap themselves
    in, because ``UnicodeDecodeError`` is a ``ValueError``:

    * On POSIX, ``Popen._communicate`` decodes on the calling thread at the end, via
      ``_translate_newlines`` -> ``data.decode(encoding, errors)``. The exception
      propagates out of ``subprocess.run``, so ``trainai doctor`` and ``trainai setup``
      die with a traceback -- in a project whose whole error story is hint-carrying
      messages and no tracebacks.
    * On Windows the decode happens inside ``Popen._readerthread``. Nothing catches it
      there, so the threading machinery prints the traceback straight at the user, the
      buffer stays empty, and ``_communicate`` ends with
      ``stdout = stdout[0] if stdout else None`` -- so ``result.stdout`` is **None**. A
      caller that trusts the attribute gets ``AttributeError``; one that writes
      ``result.stdout or ""`` loses the reading silently instead. Measured with a child
      writing ``b'CUDA Version: 12.6 \\x81\\n'``.

    The fix at every site is ``encoding="utf-8", errors="replace"``. ``replace`` rather
    than strict on purpose: these outputs are searched for one specific thing -- a
    version number, a file list -- and discarding the whole reading over one odd byte
    elsewhere in it is the worse failure.
    """
    offenders: list[str] = []
    for path, line in decoding_subprocesses():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        node = next(
            item
            for item in ast.walk(tree)
            if isinstance(item, ast.Call) and item.lineno == line and _decodes_captured_output(item)
        )
        if not _states_encoding(node):
            offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{line}")

    assert not offenders, (
        "these subprocess calls decode captured output with the platform's locale, "
        f"strictly, and a bad byte does not raise where they catch: {offenders}. "
        'Pass encoding="utf-8", errors="replace".'
    )


#: Extras that legitimately import nothing under ``src/``, with the reason. ``dev``
#: is the tooling that runs *on* the source rather than from it.
EXTRAS_NOT_IMPORTED_BY_SRC = {"dev": "contributor tooling: pytest, pytest-cov, ruff"}

#: Distribution names whose import name differs from the name pip installs.
IMPORT_NAMES = {"uvicorn[standard]": "uvicorn", "pytest-cov": "pytest_cov"}


def _top_level_imports() -> set[str]:
    """Every top-level module name imported anywhere under ``src/``."""
    names: set[str] = set()
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names.add(node.module.split(".")[0])
    return names


def test_every_declared_extra_is_one_the_code_uses() -> None:
    """Regression: a `web` extra installed fastapi and uvicorn to deliver nothing.

    It was declared for milestone M5, which the README marks postponed, so
    ``pip install trainai[web]`` pulled an ASGI server and several compiled
    transitive packages and shipped no server to run -- measured by searching the
    tree: zero references to fastapi, uvicorn or starlette outside the declaration
    itself and the dependency-policy doc. The reasoning for its eventual shape lives
    in ``docs/design/dependencies.md`` under "Planned, and deliberately not declared
    yet"; the extra returns in the commit that adds the server.

    Skipped below Python 3.11, which has no :mod:`tomllib`. That leaves three of the
    four versions in the CI matrix covering it, which is enough for a rule about a
    file that only changes by hand.
    """
    tomllib = pytest.importorskip("tomllib", reason="tomllib is 3.11+; CI covers 3.11-3.13")

    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        declared = tomllib.load(handle)["project"].get("optional-dependencies", {})

    imported = _top_level_imports()
    dead: list[str] = []
    for extra, requirements in declared.items():
        if extra in EXTRAS_NOT_IMPORTED_BY_SRC:
            continue
        packages = [
            requirement.split(">=")[0].split("==")[0].strip() for requirement in requirements
        ]
        if not any(IMPORT_NAMES.get(name, name) in imported for name in packages):
            dead.append(f"{extra} ({', '.join(packages)})")

    assert not dead, (
        f"these extras install packages that nothing under src/ imports: {dead}. "
        "Either use them or remove the extra -- an extra that delivers nothing is a "
        "promise the code does not keep. If it is genuinely not imported by src/, "
        "add it to EXTRAS_NOT_IMPORTED_BY_SRC with the reason."
    )


def test_the_extras_exemption_is_still_earned() -> None:
    """An exemption for an extra that no longer exists should go, not linger."""
    tomllib = pytest.importorskip("tomllib", reason="tomllib is 3.11+; CI covers 3.11-3.13")

    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        declared = tomllib.load(handle)["project"].get("optional-dependencies", {})

    stale = sorted(set(EXTRAS_NOT_IMPORTED_BY_SRC) - set(declared))
    assert not stale, f"these extras are exempted but no longer declared: {stale}"


def _project_table() -> dict[str, Any]:
    """``[project]`` from ``pyproject.toml``, or a skip below 3.11.

    Same bargain as :func:`test_every_declared_extra_is_one_the_code_uses`: no
    :mod:`tomllib` before 3.11, and three of the four versions in the CI matrix is
    enough coverage for a file that only changes by hand.
    """
    tomllib = pytest.importorskip("tomllib", reason="tomllib is 3.11+; CI covers 3.11-3.13")
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        table: dict[str, Any] = tomllib.load(handle)["project"]
    return table


def test_the_version_is_written_in_two_places_that_agree() -> None:
    """Regression: ``0.1.0`` was in ``pyproject.toml`` and in ``__init__.py``, unguarded.

    Both have to exist. The build backend reads the first and cannot import the
    second; ``trainai --version`` prints the second and cannot see the first. So the
    duplication is not the bug -- the bug is that nothing compared them, and the
    version is edited by hand at release time, once, in a hurry. A wheel whose
    metadata says one number and whose ``__version__`` says another is a wheel nobody
    can bisect: it is the field every issue report quotes.
    """
    declared = _project_table()["version"]

    from trainai import __version__

    assert __version__ == declared, (
        f"pyproject.toml says version {declared!r} and trainai.__version__ says "
        f"{__version__!r}. Change both, in the same commit."
    )


def test_python_dash_m_trainai_is_a_working_entry_point() -> None:
    """``python -m trainai`` is the fallback when the console script is not on PATH.

    That is not a hypothetical: a ``pip install --user`` on Windows puts ``trainai.exe``
    in a Scripts directory that is regularly absent from PATH, and the first thing a user
    in that position tries is ``python -m trainai``. The module exists for them and is
    two lines long, which is exactly the kind of file that breaks unnoticed -- an
    ``__init__`` that starts importing torch, or a rename of ``main``, and it fails at
    import with a traceback rather than a message.

    Run as a subprocess because that is the only way to execute ``__main__``: importing
    it does not run the ``if __name__`` block, and the coverage of the child process is
    not collected, so this module stays at 0% in the report and is exempted there. The
    test is what covers it.
    """
    result = subprocess.run(
        [sys.executable, "-m", "trainai", "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=REPO_ROOT,
    )

    from trainai import __version__

    assert result.returncode == 0, f"`python -m trainai --version` failed: {result.stderr}"
    assert __version__ in result.stdout, (
        f"`python -m trainai --version` printed {result.stdout!r}, which does not contain "
        f"the version {__version__!r}"
    )


def test_the_typed_classifier_is_backed_by_a_marker_file() -> None:
    """Regression: ``Typing :: Typed`` was declared and ``py.typed`` did not exist.

    The classifier is the promise; PEP 561 says the marker file inside the package is
    what a type checker actually looks for. Without it mypy and pyright silently treat
    every ``trainai`` import as ``Any`` -- silently being the whole problem, since the
    annotations are all there and the classifier says to expect them, so a downstream
    user gets no types and no error explaining why.

    Checked as a file rather than by building a wheel because the wheel target packages
    the directory wholesale: if the marker is in the tree, it ships.
    """
    classifiers = _project_table()["classifiers"]
    marker = REPO_ROOT / "src" / "trainai" / "py.typed"

    assert ("Typing :: Typed" in classifiers) == marker.exists(), (
        "the `Typing :: Typed` classifier and src/trainai/py.typed have to arrive and "
        f"leave together. Classifier declared: {'Typing :: Typed' in classifiers}, "
        f"marker present: {marker.exists()}."
    )


def test_the_sdist_ships_every_directory_its_own_tests_read() -> None:
    """Regression: the sdist shipped ``tests`` but not the ``docs`` those tests read.

    An sdist is the source of every distro package and the only artifact a reviewer can
    audit, so shipping the suite is right. Shipping a suite that cannot pass is worse
    than shipping none: the packager who runs it gets a failure that says nothing about
    TrainAI, spends an afternoon on it, and concludes the project is broken.

    Read from the include list rather than from a built sdist, so the check costs
    nothing and names the missing entry instead of a missing file.
    """
    tomllib = pytest.importorskip("tomllib", reason="tomllib is 3.11+; CI covers 3.11-3.13")
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        include = tomllib.load(handle)["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]

    if "tests" not in include:
        pytest.skip("the sdist does not ship the suite, so it needs nothing the suite reads")

    read_by_tests = {"docs", "examples", "tools", "CONTRIBUTING.md"}
    missing = sorted(read_by_tests - set(include))
    assert not missing, (
        f"the sdist includes `tests` but not {missing}, which the suite reads: "
        "test_conventions.py checks docs/ against the code and lints examples/, and "
        "test_coverage_floor.py imports tools/coverage_floor.py and reads the floors it "
        "documents in CONTRIBUTING.md. Add them to "
        "[tool.hatch.build.targets.sdist].include, or stop shipping tests."
    )


def imports_in_subprocess(statement: str) -> set[str]:
    code = f"import json, sys\n{statement}\nprint(json.dumps(sorted(sys.modules)))"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=REPO_ROOT,
        check=True,
    )
    import json as _json

    return set(_json.loads(result.stdout))


@pytest.mark.parametrize(
    "statement",
    [
        "import trainai",
        "import trainai.data",
        "import trainai.cli.data",
        "import trainai.cli.main",
    ],
)
def test_importing_does_not_pull_in_torch(statement: str) -> None:
    """Torch costs seconds and a CUDA context, so it loads only when a run needs it."""
    modules = imports_in_subprocess(statement)

    assert "torch" not in modules, f"{statement} imported torch"


def test_the_data_pipeline_does_not_import_torch_even_when_used() -> None:
    """Not just at import: preparing a dataset must work with no torch installed."""
    modules = imports_in_subprocess(
        "from trainai.data import train_tokenizer\n"
        "train_tokenizer(['repeated words repeated words ' * 40], vocab_size=300)"
    )

    assert "torch" not in modules


def test_help_does_not_import_torch_or_the_tokenizer() -> None:
    """The fastest path stays fast: `--help` touches neither numpy nor tokenizers."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, json\n"
            "sys.argv = ['trainai', '--help']\n"
            "from trainai.cli.main import main\n"
            "import io, contextlib\n"
            "with contextlib.redirect_stdout(io.StringIO()):\n"
            "    main()\n"
            "print(json.dumps(sorted(m for m in sys.modules if m in "
            "{'torch', 'numpy', 'tokenizers'})))",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=REPO_ROOT,
        check=True,
    )
    import json as _json

    assert _json.loads(result.stdout.strip().splitlines()[-1]) == []


# --------------------------------------------------------------------------- #
# Advice the CLI can actually take
# --------------------------------------------------------------------------- #
#: Long options that belong to pip rather than to TrainAI. The tool prints one pip
#: command -- installing a CUDA build of torch for hardware whose wheel is wrong -- and
#: checking its flags against the TrainAI CLI would report every one as nonexistent.
#:
#: It used to print a second one, ``pip install --upgrade trainai``, and this exemption
#: is what let that survive: the flag check waved ``--upgrade`` through as pip's, and
#: nothing else asked whether the package it named could be installed at all. It cannot
#: -- see :func:`test_no_message_names_an_install_that_does_not_exist`, which is the
#: check that gap earned.
#:
#: Exempted by exact spelling, not by skipping the strings or the modules that hold them.
#: Skipping by string was tried and does not work: ``_install_command`` builds its command
#: from two literals, so the half holding ``--index-url`` never mentions pip. Skipping by
#: module would hide a real TrainAI flag typo anywhere in the same file.
PIP_FLAGS = frozenset({"--upgrade", "--force-reinstall", "--index-url"})

FLAG_IN_TEXT = re.compile(r"--[a-z][a-z0-9-]+")

#: A backticked ``trainai ...`` invocation. Stops at a backtick, and the capture keeps
#: placeholders like ``<corpus>`` out by matching only lowercase words and flags.
COMMAND_IN_TEXT = re.compile(r"`trainai ((?:[a-z][a-z0-9-]*\s*)*)")


def strings_in(path: Path) -> list[tuple[int, str]]:
    """Every string literal in a module, with its line number."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def library_sources() -> list[Path]:
    return sorted((REPO_ROOT / "src").rglob("*.py"))


def test_every_flag_the_code_names_exists_on_some_command() -> None:
    """A message naming a flag the CLI does not have is advice nobody can take.

    Found four times during one bug hunt, three of them in the same function: a
    corpus finding told the user to pass ``--path``, which only ``trainai doctor``
    has; two others attributed ``--allow-tabular`` and ``--vocab-size`` to the wrong
    command. Those were fixed one by one against ``validate.py``'s issue codes. This
    is the cheaper net underneath: no module can introduce a flag that exists nowhere.

    It cannot check *which* command should own the flag -- a static scan does not know
    which command printed a string -- so ``test_data_analyze`` still does that for the
    corpus findings, where the mistake actually happened. Together they mean a typo
    fails here and a misattribution fails there.
    """
    known = set(cli_flag_owners()) | PIP_FLAGS
    unknown: list[str] = []
    for path in library_sources():
        for lineno, text in strings_in(path):
            for named in FLAG_IN_TEXT.findall(text):
                if named not in known:
                    relative = path.relative_to(REPO_ROOT).as_posix()
                    unknown.append(f"{relative}:{lineno}: {named}")

    assert not unknown, (
        f"these flags are named in source strings but no command accepts them: "
        f"{sorted(set(unknown))}. Either the spelling is wrong or the flag was "
        "removed; a message may not name an option the CLI does not have."
    )


def no_such_command(words: list[str], known: set[str], groups: set[str]) -> str | None:
    """Why ``trainai <words>`` would not run, or ``None`` if it would.

    Resolved by descending the command tree one word at a time, because a hint's words
    do not all name commands: ``trainai chat runs/m4`` continues into a path, and
    ``trainai data prepare corpus`` into an argument. The rule that separates those from
    a typo is *where* the unrecognised word appears. After a leaf command, anything
    goes -- it is that command's argument, and this cannot know what a valid one looks
    like. On a group, the next word has to be one of its subcommands, since a group
    accepts nothing else.

    A first version instead accepted the phrase as soon as any prefix of it was a real
    command path, which let ``trainai data prepair`` through: ``data`` matched. That is
    the mutation this shape exists to catch.
    """
    path: list[str] = []
    for word in words:
        if " ".join([*path, word]) in known:
            path.append(word)
            continue
        if not path:
            return f"{word} is not a command"
        if " ".join(path) in groups:
            return f"{word} is not a subcommand of trainai {' '.join(path)}"
        return None  # an argument to the leaf command reached so far
    return None


def test_every_command_the_code_names_can_be_run() -> None:
    """Same rule for the commands hints tell users to run.

    ``trainai data prepare`` appears in a dozen messages. Renaming or moving a command
    without updating them leaves the tool confidently instructing users to run
    something that exits with "No such command", which is worse than saying nothing.
    """
    known = cli_command_paths()
    groups = cli_group_paths()
    unknown: list[str] = []
    for path in library_sources():
        for lineno, text in strings_in(path):
            for phrase in COMMAND_IN_TEXT.findall(text):
                words = phrase.split()
                if not words:
                    continue  # bare `trainai`, which is the group's own help
                reason = no_such_command(words, known, groups)
                if reason is not None:
                    relative = path.relative_to(REPO_ROOT).as_posix()
                    unknown.append(f"{relative}:{lineno}: trainai {phrase.strip()} ({reason})")

    assert not unknown, (
        f"these commands are named in source strings but do not exist: {sorted(set(unknown))}"
    )


@pytest.mark.parametrize(
    ("phrase", "runs"),
    [
        ("data prepare", True),  # a real leaf
        ("data", True),  # a group, which prints its own help
        ("train", True),
        ("data prepare corpus", True),  # a leaf plus its argument
        ("chat runs", True),  # a leaf plus the start of a path
        ("data prepair", False),  # real group, misspelled leaf
        ("data prepare extra word", True),  # still just arguments
        ("prepare", False),  # a real leaf named at the wrong depth
        ("dtaa prepare", False),
        ("doctor extra", True),
    ],
)
def test_the_command_resolver_stops_descending_at_the_first_leaf(phrase: str, runs: bool) -> None:
    """The cases the check above turns on, stated directly rather than via a source scan.

    Half of these do not appear in any hint today, so the check would keep passing if the
    resolver lost the group/leaf distinction. ``data prepair`` is the one that matters:
    it is what a rename leaves behind, and the earlier prefix-matching version said yes.
    """
    reason = no_such_command(phrase.split(), cli_command_paths(), cli_group_paths())
    assert (reason is None) is runs, f"trainai {phrase}: {reason}"


def test_the_advice_checks_are_actually_looking_at_something() -> None:
    """Both checks above pass trivially if the scan finds no strings.

    The counts are lower bounds rather than exact numbers, so adding a hint does not
    fail this -- but a regex that stops matching, or a source glob that goes empty,
    does.
    """
    flags: set[str] = set()
    commands: set[str] = set()
    for path in library_sources():
        for _, text in strings_in(path):
            flags.update(FLAG_IN_TEXT.findall(text))
            commands.update(phrase.strip() for phrase in COMMAND_IN_TEXT.findall(text))

    assert len(flags - PIP_FLAGS) > 25, f"only {len(flags)} distinct flags found in strings"
    assert len(commands) > 5, f"only {len(commands)} distinct commands found: {commands}"


def test_the_pip_flag_exemption_is_still_earned() -> None:
    """Two ways an exemption goes bad, and both make the check above weaker silently.

    A spelling no source string uses any more is exempting nothing, and should go with
    the code that needed it. A spelling TrainAI has since adopted as its own flag is
    worse: the exemption would then wave through a message that names it, which is the
    exact mistake this section exists to catch.
    """
    named: set[str] = set()
    for path in library_sources():
        for _, text in strings_in(path):
            named.update(FLAG_IN_TEXT.findall(text))

    unused = sorted(PIP_FLAGS - named)
    assert not unused, f"these flags are exempted but no source string names them: {unused}"

    taken = sorted(PIP_FLAGS & set(cli_flag_owners()))
    assert not taken, (
        f"TrainAI now has these flags itself: {taken}. Remove them from PIP_FLAGS so "
        "messages naming them are checked like any other."
    )


#: README.md's own statement that there is nothing to ``pip install``. The repository is
#: now a real place a reader can clone from, but nothing has been uploaded to any package
#: index, so ``pip install trainai`` is still not advice anyone can take.
NOT_PUBLISHED_MARKER = "Not on PyPI yet."

#: An install command naming TrainAI as the package. ``pip install -e ".[dev]"`` -- the
#: documented way in, and the only one that works -- does not name it, so it does not
#: match.
INSTALL_TRAINAI = re.compile(r"pip3?\s+install\b[^`\n]*\btrainai\b", re.IGNORECASE)

#: The places allowed to write it anyway, because they state something *about* the
#: command rather than telling anyone to run it. Exempted by exact spelling, like
#: :data:`PIP_FLAGS`, and checked below for still being present.
PUBLICATION_EXEMPT = (
    # Where the "not on PyPI" fact is declared, so the one place that has to be able
    # to name the command in order to say it does not exist.
    "There is no `pip install trainai`, so install from a clone",
    # An extra that was removed, described in the past tense.
    "`pip install trainai[web]` used to install both",
)


def searchable_lines() -> list[tuple[str, str]]:
    """Everything this section reads: source string literals, then document lines.

    One list built here rather than two loops inside the filter, so a floor test can
    check that both halves are still being read. Splitting them out is not tidiness:
    after this commit no source string names the install at all, so a check counting
    matches in ``src`` would have to expect zero, and would pass just as happily if the
    source half of the scan had gone dead.
    """
    lines: list[tuple[str, str]] = []
    for path in library_sources():
        where = path.relative_to(REPO_ROOT).as_posix()
        for lineno, text in strings_in(path):
            lines.append((f"{where}:{lineno}", collapse(text)))
    for path in prose_documents():
        where = path.relative_to(REPO_ROOT).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            lines.append((f"{where}:{lineno}", line.strip()))
    return lines


def install_advice() -> list[tuple[str, str]]:
    """Every ``pip install ... trainai``, as (where it is, the whole line saying it).

    The whole line, not the matched command, because that is what the exemptions read:
    whether a mention is an instruction or a statement about one is visible in the
    sentence around it and nowhere else.
    """
    return [(where, text) for where, text in searchable_lines() if INSTALL_TRAINAI.search(text)]


def unearned_exemptions(exemptions: Sequence[str], said: Sequence[str]) -> list[str]:
    """Which of ``exemptions`` fails to match exactly one of ``said``, and how.

    A function rather than an inline assertion so the rule can be handed a deliberately
    bad list below. The narrowness half is the reason: "matches at least one" has an
    obvious guard -- a stale entry stops matching and the check fires -- but "matches at
    most one" guards nothing that anyone would notice going missing, and it is the half
    holding the list to sentences instead of fragments.
    """
    complaints: list[str] = []
    for allowed in exemptions:
        matched = [text for text in said if allowed in text]
        if len(matched) != 1:
            complaints.append(
                f"{allowed!r} matches {len(matched)} places, not one. Zero means it is "
                f"stale and should go with the text that needed it; more than one means "
                f"it is a fragment wide enough to wave through a real instruction: "
                f"{matched}"
            )
    return complaints


def test_no_message_names_an_install_that_does_not_exist() -> None:
    """Nothing may tell a user to ``pip install trainai`` while there is no such thing.

    Two error hints did. A dataset or checkpoint written by a newer TrainAI was refused
    with "Upgrade TrainAI with ``pip install --upgrade trainai``", four lines away in
    spirit from README.md saying there is no ``pip install trainai``. Neither hint was
    reachable by accident -- both are the *only* advice given for an artifact the build
    cannot read, so following it was the user's whole path forward, and it went nowhere.

    The flag check above cannot catch this. It asks whether a named flag exists on some
    command, and ``--upgrade`` is exempt as pip's; nothing asked whether the *package*
    could be installed. So this is not a duplicate net, it is the hole in that one.
    """
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert NOT_PUBLISHED_MARKER in readme, (
        "README.md no longer says TrainAI is absent from PyPI. If it is now published "
        "there then `pip install --upgrade trainai` is real advice, and this check "
        "should be deleted deliberately along with the hints that route around it -- see "
        "DatasetManifest.load and load_checkpoint. If only the wording changed, update "
        "NOT_PUBLISHED_MARKER."
    )

    offenders = [
        f"{where}: {text}"
        for where, text in install_advice()
        if not any(allowed in text for allowed in PUBLICATION_EXEMPT)
    ]
    assert not offenders, (
        f"these name an install of TrainAI that does not exist: {offenders}. There is "
        "nothing to install from, so a hint saying so sends the user nowhere; say what "
        "they can do with the build they have instead."
    )


def test_the_publication_exemptions_are_still_earned() -> None:
    """An exemption for a sentence nobody writes any more is an exemption for anything.

    The same failure mode as :func:`test_the_pip_flag_exemption_is_still_earned`, and the
    reason both exist: this check is only as narrow as its exemption list, and a stale
    entry widens it silently.
    """
    complaints = unearned_exemptions(PUBLICATION_EXEMPT, [text for _, text in install_advice()])
    assert not complaints, complaints


@pytest.mark.parametrize(
    ("exemptions", "earned"),
    [
        (("names trainai exactly once",), True),
        (("pip install",), False),  # a fragment covering both -- the widening to rule out
        (("a sentence nobody in this repository writes",), False),  # stale
    ],
)
def test_an_exemption_must_match_exactly_one_place(
    exemptions: tuple[str, ...], earned: bool
) -> None:
    """Stated against fixed text, because the rule has no other guard.

    The check above is the only caller, and it would keep passing with the rule relaxed to
    "match zero or more" -- which is to say with no rule at all -- since the two entries in
    :data:`PUBLICATION_EXEMPT` do match one place each today. The fragment case is the one
    that matters: ``"pip install"`` is a substring of every line the section looks at, so
    accepting it would exempt the whole repository while still looking used.
    """
    said = [
        "pip install trainai -- this line names trainai exactly once",
        "`pip install trainai[web]` used to install both",
    ]
    complaints = unearned_exemptions(exemptions, said)
    assert (not complaints) is earned, (
        f"{list(exemptions)} against {len(said)} lines should have been "
        f"{'accepted' if earned else 'rejected'}, and was not. Complaints: {complaints}"
    )


def test_the_publication_check_reads_both_source_and_documents() -> None:
    """No source string names the install any more, which is exactly what makes this needed.

    Every remaining mention is in a document, so the source half of the scan could stop
    running -- an empty glob, a dropped loop, a rename -- and every check in this section
    would still pass. ``src`` is where the two bad hints lived, so it is the half that
    matters; assert it is still being read at all. Floors, not counts, so ordinary churn
    does not fail this.
    """
    read = searchable_lines()
    from_src = [where for where, _ in read if where.startswith("src/")]
    from_docs = [where for where, _ in read if not where.startswith("src/")]
    assert len(from_src) > 2000, f"only {len(from_src)} source strings read"
    assert len(from_docs) > 1000, f"only {len(from_docs)} document lines read"


def test_the_install_pattern_matches_what_it_is_meant_to() -> None:
    """The regex is the whole check, and a regex that stops matching fails silently.

    ``pip install -e ".[dev]"`` has to keep passing: it is the documented way in, and a
    check that flagged it would be asking for the one true install command to be removed.
    """
    matches = [
        "Upgrade TrainAI with `pip install --upgrade trainai`.",
        "runs ``pip install trainai``, gets the CPU-only wheel",
        "pip3 install trainai",
        "`pip install trainai[web]` used to install both",
    ]
    passes = [
        'pip install -e ".[dev]"',
        "pip install --upgrade --force-reinstall torch",
        "pip install --index-url https://download.pytorch.org/whl/cu128 torch",
        "trainai data prepare",
    ]
    for text in matches:
        assert INSTALL_TRAINAI.search(text), text
    for text in passes:
        assert not INSTALL_TRAINAI.search(text), text


# --------------------------------------------------------------------------- #
# The documented size of this suite
# --------------------------------------------------------------------------- #
#: Where the suite's size is quoted to readers. Both must use the same wording, which
#: is what makes one regex enough and keeps the two from drifting into different claims.
SUITE_SIZE_DOCS = ("README.md", "docs/hardware-support.md")

DOCUMENTED_SUITE_SIZE = re.compile(r"more than ([\d,]+) tests")

#: How far the real count may run ahead of the documented floor before the floor stops
#: describing the suite. Wide enough that adding a test is not a documentation change,
#: narrow enough that "more than 1,400" cannot survive into a suite of five thousand.
SUITE_SIZE_SLACK = 250


def test_the_documented_suite_size_is_still_true(request: pytest.FixtureRequest) -> None:
    """The README and the hardware notes both quote how big this suite is.

    They said "917 tests" for long enough to be wrong by five hundred, which is the
    same defect as a hint naming a flag that does not exist: a number the reader has no
    way to doubt and the repository no longer honours. Quoting a floor rather than an
    exact count is what makes it enforceable without turning every new test into a
    documentation edit -- but a floor with nothing watching it rots the same way, so
    this fails from both sides.

    Counted from ``session.items``, the post-deselection list of what this very run
    collected, so it costs nothing and cannot disagree with reality. Skipped unless
    that list is the whole suite: running one file must not fail, and CI runs all of it.
    """
    if request.config.option.file_or_dir or request.config.option.keyword:
        pytest.skip("a partial run cannot say how big the suite is")

    collected = len(request.session.items)
    for name in SUITE_SIZE_DOCS:
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        match = DOCUMENTED_SUITE_SIZE.search(text)
        assert match is not None, (
            f"{name} no longer says how many tests there are, in the wording "
            f"{DOCUMENTED_SUITE_SIZE.pattern!r} that this check looks for"
        )
        stated = int(match[1].replace(",", ""))
        assert collected > stated, (
            f"{name} claims more than {stated:,} tests, but this run collected "
            f"{collected:,}. Either tests were removed or the number was never true."
        )
        assert collected - stated < SUITE_SIZE_SLACK, (
            f"{name} claims more than {stated:,} tests and there are now {collected:,}. "
            f"Raise it to the next round number below {collected:,}."
        )


# --------------------------------------------------------------------------- #
# The model numbers the documents quote
# --------------------------------------------------------------------------- #
#: A shape as ``ModelConfig.describe`` writes it. Every one of these in the prose is a
#: claim about the code, and `ff` is derived from `d_model` rather than stored, so a
#: hand-edited shape line is not internally consistent even before the counts.
DOC_SHAPE = re.compile(r"L(\d+) d(\d+) h(\d+) ff(\d+) ctx(\d+) vocab(\d+)")

#: "12.2M parameters", and the "Parameters  4.26M  (" row, which are the two ways the
#: documents state a total beside a shape.
PARAM_TOTAL = re.compile(r"([\d.]+[KMB])(?: parameters|  \()")

#: The second number in that row, which is a different formula and worth its own check.
PARAM_NON_EMBEDDING = re.compile(r"\(([\d.]+[KMB]) outside the embedding\)")

#: The preset a shape is attributed to, when it is: "tiny (L4 ..." in a measurement
#: line, "tiny  -  L4 ..." in a panel row. A bare "Shape  L4 ..." names no preset.
SHAPE_OWNER = re.compile(r"(\w+)(?:  -  | \()$")

#: The fenced block in the README holding a `--dry-run` plan, found by the line the
#: command ends with rather than by counting fences.
DRY_RUN_BLOCK = re.compile(r"```\n(-+ Plan -+\n.*?Nothing was trained\..*?)```", re.S)

#: The shape line inside it, and the dataset row that has to agree with its vocabulary.
DRY_RUN_SHAPE = re.compile(r"^Shape +L(\d+) d(\d+) h(\d+) ff(\d+) ctx(\d+) vocab(\d+)$", re.M)
DRY_RUN_VOCAB = re.compile(r"^Vocabulary +([\d,]+)$", re.M)

RICH_MARKUP = re.compile(r"\[/?[a-z ]*\]")


class Shape(NamedTuple):
    """A model shape quoted in a document, and enough context to check it."""

    path: Path
    line: int
    text: str
    dimensions: tuple[int, ...]
    owner: str | None
    following: str


def collapse(text: str) -> str:
    """One space between tokens, so a comparison does not depend on terminal width."""
    return " ".join(text.split())


def prose_documents() -> list[Path]:
    """Every document that describes the code as it is now.

    Numbers rot in prose because nothing imports it, so the net is deliberately wide:
    the README, the contributing guide, and everything under ``docs/``.

    ``CHANGELOG.md`` is left out, and that is not an oversight. It records what was true
    when each entry was written. A shape or a count in a released entry is history, and
    the right response to the preset table changing is to add an entry, never to edit an
    old one -- so a check that failed on it would be asking for the wrong fix.
    """
    return sorted(
        path
        for path in [*REPO_ROOT.glob("*.md"), *(REPO_ROOT / "docs").rglob("*.md")]
        if path.name != "CHANGELOG.md"
    )


def documented_shapes() -> list[Shape]:
    """Every quoted model shape, with the text that follows it on the next two lines.

    Two lines because the plan panel wraps: `Limited by` puts the shape at the end of
    one line and "12.2M parameters" at the start of the next.
    """
    found = []
    for path in prose_documents():
        text = path.read_text(encoding="utf-8")
        for match in DOC_SHAPE.finditer(text):
            owner = SHAPE_OWNER.search(text[: match.start()])
            found.append(
                Shape(
                    path=path,
                    line=text.count("\n", 0, match.start()) + 1,
                    text=match[0],
                    dimensions=tuple(int(group) for group in match.groups()),
                    owner=owner[1] if owner else None,
                    following="\n".join(text[match.end() :].split("\n")[:3]),
                )
            )
    return found


def config_for(shape: Shape) -> Any:
    """The config the quoted shape describes, built from the page rather than a preset."""
    from trainai.model.config import ModelConfig

    layers, width, heads, _, context, vocab = shape.dimensions
    return ModelConfig(
        n_layer=layers, d_model=width, n_head=heads, seq_len=context, vocab_size=vocab
    )


def test_every_model_shape_the_docs_state_is_one_the_code_produces() -> None:
    """A shape in the prose has to be a shape ``describe`` would write.

    ``d_ff`` is not stored on the config -- it is derived from ``d_model`` -- so this is
    not a spelling check. A shape edited by hand to `ff1024` beside `d256` describes a
    model the code cannot build, and the parameter count quoted under it will be wrong
    in a way no reader can see. Rebuilding the config from L/d/h/ctx/vocab and asking
    for the string back catches that, and catches a renamed or reordered field too.
    """
    wrong = []
    for shape in documented_shapes():
        written = config_for(shape).describe()
        if written != shape.text:
            relative = shape.path.relative_to(REPO_ROOT).as_posix()
            wrong.append(f"{relative}:{shape.line}: {shape.text} -- the code writes {written}")
    assert not wrong, "model shapes the code does not produce:\n" + "\n".join(wrong)


def test_every_shape_attributed_to_a_preset_is_that_preset() -> None:
    """ "tiny (L4 d256 ...)" is a claim about the preset table, so check it against it.

    The preceding check rebuilds the config from the page, which pins `ff` against
    `d_model` but cannot notice that the page's `d_model` is not the one `tiny` has. This
    one asks the preset table. The vocabulary and context stay inputs -- both come from
    the corpus and the flags, not the preset -- so they are pinned the other way: a
    preset named twice in one document has to be quoted with the same shape both times.
    That is what stops the `measuring tiny (...)` line drifting away from the
    recommendation panel four lines below it, which quotes the same run.
    """
    from trainai.model.config import PRESETS, preset

    known = {entry.name for entry in PRESETS}
    wrong, seen, checked = [], {}, 0
    for shape in documented_shapes():
        # The word before a shape is not always a preset -- "the shape (L4 ...)" would
        # match too -- so an unknown one is skipped rather than reported.
        if shape.owner not in known:
            continue
        relative = shape.path.relative_to(REPO_ROOT).as_posix()
        where = f"{relative}:{shape.line}"
        _, _, _, _, context, vocab = shape.dimensions
        written = preset(shape.owner, vocab_size=vocab, seq_len=context).describe()
        checked += 1
        if written != shape.text:
            wrong.append(
                f"{where}: quoted as {shape.owner} {shape.text}, but that preset is {written}"
            )
        first = seen.setdefault((shape.path, shape.owner), shape)
        if first.text != shape.text:
            wrong.append(
                f"{where}: {shape.owner} is {shape.text} here and {first.text} at line "
                f"{first.line} of the same document"
            )
    assert not wrong, "shapes that contradict the preset they name:\n" + "\n".join(wrong)
    assert checked >= 4, (
        f"only {checked} shapes were attributed to a preset, so SHAPE_OWNER has stopped "
        "matching the way the panels write one"
    )


def test_every_parameter_count_beside_a_shape_is_the_real_one() -> None:
    """A count quoted next to a shape has to be the count that shape has.

    This is the check the `--dry-run` block needed and the two `trainai plan` blocks
    above it never had: they quote `tiny` and `small` with totals of their own, and the
    only thing keeping them honest was that nobody had edited the preset table yet. The
    count is recomputed from the shape on the page, so it does not matter which preset
    produced it or whether that preset still exists under the same name.
    """
    from trainai.console import fmt_count

    wrong, checked = [], 0
    for shape in documented_shapes():
        config = config_for(shape)
        relative = shape.path.relative_to(REPO_ROOT).as_posix()
        for pattern, count, what in (
            (PARAM_TOTAL, config.parameter_count, "total"),
            (PARAM_NON_EMBEDDING, config.non_embedding_parameter_count, "non-embedding"),
        ):
            claim = pattern.search(shape.following)
            if claim is None:
                continue
            checked += 1
            if claim[1] != fmt_count(count):
                wrong.append(
                    f"{relative}:{shape.line}: {shape.text} has a {what} of "
                    f"{fmt_count(count)}, but the text beside it says {claim[1]}"
                )
    assert not wrong, "parameter counts the shapes beside them contradict:\n" + "\n".join(wrong)
    assert checked >= 4, (
        f"only {checked} parameter counts were found beside a shape, so this check has "
        "stopped looking at the README's plan panels. Check PARAM_TOTAL still matches "
        "the way the panel writes a count."
    )


def test_the_dry_run_output_the_readme_quotes_is_what_the_code_prints() -> None:
    """The README quotes a `--dry-run` plan as what the command above it produces.

    It did not. The block said `vocab8192`, `5.31M` parameters and `Steps 450` while
    the command beside it asked for 600 steps against a 4,096-token vocabulary, and one
    line -- "bf16 (supported by this GPU, no scaler needed)" -- was text the code no
    longer prints in any configuration. A reader running the command got a different
    plan, which is the same defect as a hint naming a flag that does not exist.

    Guarded against the renderer itself, ``_model_rows``, rather than against numbers
    copied out of it, so a change to the preset table, the parameter formula or the
    count formatting fails here. The rows that depend on the machine -- device,
    precision, throughput -- cannot be checked on an arbitrary machine and are left to
    the measurement note that introduces them.

    The vocabulary comes from a dataset the repository does not carry, so it is read out
    of the block -- which on its own would let it drift, since every other row is then
    derived from it and moves in step. The `Vocabulary` row of the block's own dataset
    section is the independent pin, and the preset is resolved from the last command
    before the block rather than the first in the file, so the two are the same run.
    """
    from trainai.cli.train import _model_rows
    from trainai.model.config import preset

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    block = DRY_RUN_BLOCK.search(readme)
    assert block is not None, "the README no longer quotes a --dry-run plan"

    shape = DRY_RUN_SHAPE.search(block[1])
    assert shape is not None, f"no Shape line in the quoted plan:\n{block[1]}"
    # The other groups are matched to pin the line's shape; only these two are inputs.
    context, vocab = int(shape[5]), int(shape[6])

    stated = DRY_RUN_VOCAB.search(block[1])
    assert stated is not None, "the quoted plan has no Vocabulary row to pin the shape to"
    assert int(stated[1].replace(",", "")) == vocab, (
        f"the quoted plan says it tokenised to a vocabulary of {stated[1]} but its shape "
        f"line says vocab{vocab}; one of the two was edited without the other"
    )

    names = re.findall(r"--preset (\w+)", readme[: block.start()])
    assert names, "no --preset appears before the quoted plan"
    config = preset(names[-1], vocab_size=vocab, seq_len=context)

    quoted = collapse(block[1])
    for label, value in _model_rows(config):
        rendered = collapse(f"{label} {RICH_MARKUP.sub('', value)}")
        assert rendered in quoted, (
            f"the README's quoted plan does not contain {rendered!r}. Re-run the command "
            "it shows with --dry-run and paste what it prints."
        )


# --------------------------------------------------------------------------- #
# The parity measurements the export format quotes
# --------------------------------------------------------------------------- #
#: A row of the logits-parity table in ``docs/export-format.md``: the dtype it was
#: measured at, the difference measured, and the tolerance it was judged against.
PARITY_ROW = re.compile(r"^\| `(\w+)` \| ([\d.]+e-\d+) \| ([\d.]+e-\d+) \|$", re.M)
#: Every place the document states how many positions the parity check covers --
#: the check-summary table, the measurement table's column header, and the caveat
#: under it. One number, three sites, so they are checked together. Matched against
#: whitespace-flowed text, since a paragraph rewrap must not change what is found.
PARITY_POSITIONS = re.compile(r"over (\d+) positions")
#: The tolerance stated in prose rather than in the table's own column.
PARITY_TOLERANCE = re.compile(r"Tolerance `([\d.e+-]+)`")


def test_the_export_parity_table_covers_every_dtype_at_the_real_tolerance() -> None:
    """The table of measured logit differences has to describe the command it documents.

    It listed `fp32` and `bf16` and not `fp16`, which `--dtype` accepts, so a third of
    the flag's range read as untested. Both numbers it did carry were stale, and the
    model beside them was described as "4.26M parameters, 6 layers, d_model 256" -- a
    contradiction, since 4.26M is the four-layer count and the checkpoint on disk holds
    ``n_layer: 4``. That shape is now written the way ``describe`` writes it, which puts
    it under the shape and parameter-count checks above rather than under nothing.

    What is checkable without the development machine's run is the table's frame: that
    every dtype has a row, that no row names a dtype the command would refuse, that the
    tolerance column is the tolerance the code actually applies, and that every measured
    value is inside it. The measurements themselves cannot be reproduced on an arbitrary
    machine, which is why the document says so and why the check does not try.

    The tolerance and the position count are stated in prose as well as in the table, so
    every site is compared rather than just asserting the phrase appears somewhere. A
    document-wide substring check passes while the number beside the measurements is
    wrong, as long as some other paragraph still carries the right one.
    """
    from trainai.export.bundle import _PARITY_TOKENS, DTYPE_CHOICES, LOGITS_TOLERANCE

    doc = (REPO_ROOT / "docs" / "export-format.md").read_text(encoding="utf-8")
    #: The table rows need the line anchors, so they are matched against the file as
    #: written. The prose statements are matched against the same text with its
    #: whitespace flowed to single spaces, so that re-wrapping a paragraph -- which
    #: changes nothing a reader cares about -- cannot hide "over 16 positions" behind
    #: a line break and make this check ask for the wrong fix.
    flowed = " ".join(doc.split())
    rows = {match[1]: (float(match[2]), float(match[3])) for match in PARITY_ROW.finditer(doc)}
    assert rows, (
        "no parity measurements found in docs/export-format.md. Either the table is "
        "gone or PARITY_ROW no longer matches the way it is written."
    )

    missing = sorted(set(DTYPE_CHOICES) - set(rows))
    assert not missing, (
        f"--dtype accepts {missing} but the parity table has no row for them, so that "
        "part of the flag's range reads as untested. Export at that dtype and add the "
        "number the check reports."
    )
    unknown = sorted(set(rows) - set(DTYPE_CHOICES))
    assert not unknown, (
        f"the parity table has rows for {unknown}, which --dtype would refuse. "
        f"It accepts {sorted(DTYPE_CHOICES)}."
    )

    for name, (measured, tolerance) in sorted(rows.items()):
        assert tolerance == pytest.approx(LOGITS_TOLERANCE), (
            f"the {name} row is judged against {tolerance:g} but the code applies "
            f"{LOGITS_TOLERANCE:g} (LOGITS_TOLERANCE)"
        )
        assert measured < LOGITS_TOLERANCE, (
            f"the {name} row quotes {measured:g}, which is not inside the "
            f"{LOGITS_TOLERANCE:g} tolerance it claims to pass"
        )

    positions = [int(count) for count in PARITY_POSITIONS.findall(flowed)]
    assert len(positions) >= 3, (
        "the document should state the number of positions the parity check covers in "
        "the check-summary table, in the measurement table's column header and in the "
        f"caveat under it, and PARITY_POSITIONS found {len(positions)} such statements"
    )
    disagreeing = sorted({count for count in positions if count != _PARITY_TOKENS})
    assert not disagreeing, (
        f"the document says the parity check covers {disagreeing} positions somewhere, "
        f"but it covers {_PARITY_TOKENS} (_PARITY_TOKENS). A document-wide substring "
        "check would miss this, which is why every site is compared."
    )

    quoted = [float(value) for value in PARITY_TOLERANCE.findall(flowed)]
    assert quoted, (
        "no tolerance is stated in prose in docs/export-format.md; the check-summary "
        "table's `logits parity` row should name it"
    )
    for value in quoted:
        assert value == pytest.approx(LOGITS_TOLERANCE), (
            f"the document states a tolerance of {value:g} in prose but the code "
            f"applies {LOGITS_TOLERANCE:g} (LOGITS_TOLERANCE)"
        )


# --------------------------------------------------------------------------- #
# Every key an artifact carries is named in that artifact's format spec
# --------------------------------------------------------------------------- #
#: An inline code span. The key has to be the *whole* span, not an identifier inside
#: one: ``docs/dataset-format.md`` mentions ``tokenizer.json`` a dozen times, and a
#: rule that accepted that as documenting the ``tokenizer`` key -- which it did not
#: describe at all -- would check nothing while appearing to.
CODE_SPAN = re.compile(r"`([^`\n]+)`")


class FormatSpec(NamedTuple):
    """A serialised artifact, its single writer, and the document that specifies it."""

    document: str
    module: str
    #: The class the writer is a method of, or ``None`` for a module-level function.
    owner: str | None
    function: str
    #: The local the payload is assigned to, or ``None`` when it is returned directly.
    variable: str | None


FORMAT_SPECS = [
    FormatSpec(
        "docs/plan-format.md", "src/trainai/hardware/planner.py", "TrainingPlan", "to_dict", None
    ),
    FormatSpec(
        "docs/dataset-format.md", "src/trainai/data/binarize.py", "DatasetManifest", "to_dict", None
    ),
    FormatSpec(
        "docs/checkpoint-format.md",
        "src/trainai/train/checkpoint.py",
        None,
        "save_checkpoint",
        "payload",
    ),
]


def written_keys(spec: FormatSpec) -> list[str]:
    """The top-level keys the writer emits, read out of the dict literal it builds.

    Parsed rather than produced. Building a real plan measures the GPU, and building a
    real manifest tokenises a corpus, so a test that called the writers would be slow,
    would need hardware, and would only ever see the keys that one run happened to
    populate. The literal is the definition of the format; the artifact is one sample
    of it.

    Only the outermost dict's keys are taken. ``splits`` holds a nested dict per split
    and ``model`` holds a serialised config -- those are sub-keys, and a spec that
    delegates them to another document is doing the right thing, not omitting them.
    """
    tree = ast.parse((REPO_ROOT / spec.module).read_text(encoding="utf-8"))

    scope: list[ast.stmt] = tree.body
    if spec.owner is not None:
        classes = [
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == spec.owner
        ]
        assert classes, f"{spec.module} has no class {spec.owner}"
        scope = classes[0].body

    functions = [
        node for node in scope if isinstance(node, ast.FunctionDef) and node.name == spec.function
    ]
    assert functions, f"{spec.owner or spec.module} has no def {spec.function}"

    for statement in ast.walk(functions[0]):
        payload = None
        if spec.variable is None:
            if isinstance(statement, ast.Return):
                payload = statement.value
        elif isinstance(statement, ast.AnnAssign):
            if isinstance(statement.target, ast.Name) and statement.target.id == spec.variable:
                payload = statement.value
        elif isinstance(statement, ast.Assign):
            targets = [t.id for t in statement.targets if isinstance(t, ast.Name)]
            if spec.variable in targets:
                payload = statement.value
        if isinstance(payload, ast.Dict):
            keys = [k.value for k in payload.keys if isinstance(k, ast.Constant)]
            assert len(keys) == len(payload.keys), (
                f"{spec.module}:{spec.function} builds its payload with a computed or "
                "unpacked key, so this check can no longer read the format off it"
            )
            return keys

    raise AssertionError(
        f"found no dict literal for {spec.variable or 'the return value'} in "
        f"{spec.module}:{spec.function}"
    )


@pytest.mark.parametrize("spec", FORMAT_SPECS, ids=lambda s: Path(s.document).stem)
def test_every_key_an_artifact_carries_is_named_in_its_format_spec(spec: FormatSpec) -> None:
    """A format specification that omits a key is worse than no specification.

    A reader treats these documents as the list of what is in the file. `plan.json`
    carried `dataset` and `preset` and the spec described neither, so the two facts
    saying *what the plan is about* were the two a reader could not look up. The
    manifest was missing four: `tokenizer`, `shard_tokens`, `documents` and `splits` --
    and `splits` is the block anything reading a dataset actually loads from.

    Two of the four had a second cost. `shard_tokens` and `documents` are inside
    `content_hash`, so an undocumented key was silently part of the reproducibility
    guarantee: someone comparing two hashes had no way to learn that `--shard-tokens`
    moves one.

    `docs/checkpoint-format.md` already named all twelve of its keys and needed no
    change, which is what makes it the control here rather than a fourth fix.

    What this does **not** check: that each mention is a *defining* one. `documents`
    also appears inside the `splits` row, so deleting its own row leaves the key still
    named and this check still passing. Requiring a heading or a first table cell would
    close that, and was tried: it cries wolf on `version` and `regime_detail`, both of
    which have a paragraph about them and no row of their own. So the question answered
    here is "does the spec name this key anywhere", which is the question the six
    defects above actually posed -- and a stricter rule that had to be suppressed in
    two places would be the weaker check.
    """
    document = (REPO_ROOT / spec.document).read_text(encoding="utf-8")
    spans = set(CODE_SPAN.findall(document))
    keys = written_keys(spec)

    assert len(keys) >= 10, (
        f"only {len(keys)} keys were read out of {spec.module}:{spec.function}; these "
        "artifacts all carry more than that, so the parse is finding the wrong literal"
    )

    missing = [key for key in keys if key not in spans]
    assert not missing, (
        f"{spec.module}:{spec.function} writes {missing} but {spec.document} never names "
        f"{'them' if len(missing) > 1 else 'it'} as a key. A reader treats that document "
        "as the list of what is in the file, so an unlisted key is one nobody can rely "
        "on or knows to preserve. Name it in a heading or a table row -- mentioning it "
        "inside a longer code span such as `tokenizer.json` does not count."
    )


# --------------------------------------------------------------------------- #
# Refusing a file the user can edit says which file
# --------------------------------------------------------------------------- #
#: Functions whose whole job is to turn a file on disk into objects, and which therefore
#: owe every refusal the file's name. ``docs/plan-format.md`` promises this for plans in
#: so many words, and the promise is only as good as the least-travelled branch.
FILE_LOADERS = [("src/trainai/cli/train.py", "_configs_from_plan", "file")]


def raised_usage_errors(module: str, function: str) -> list[tuple[int, ast.Call]]:
    """Every ``raise UsageError(...)`` in one function, with its line number."""
    tree = ast.parse((REPO_ROOT / module).read_text(encoding="utf-8"))
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == function
    ]
    assert matches, f"{module} has no def {function}"

    raises = []
    for node in ast.walk(matches[0]):
        if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
            continue
        callee = node.exc.func
        if isinstance(callee, ast.Name) and callee.id == "UsageError":
            raises.append((node.lineno, node.exc))
    return raises


@pytest.mark.parametrize("module,function,variable", FILE_LOADERS)
def test_every_refusal_of_a_file_names_the_file(module: str, function: str, variable: str) -> None:
    """A message about a file the user can edit is useless without the file's name.

    ``trainai train --plan`` refuses on five distinct conditions and the vocabulary
    check was the one that did not name the file -- the one whose whole cause is
    pointing at a plan built for a different dataset, which is a mix-up *between files*.
    Its ``details`` had no ``path`` either, so the machine-readable form was missing it
    too.

    Checked structurally rather than by running the five cases, because the point is the
    branch nobody thought to test. `tests/test_cli_train.py` runs thirteen of them; this
    catches the fourteenth, on the day it is added, without anyone having to remember.
    """
    raises = raised_usage_errors(module, function)
    assert len(raises) >= 4, (
        f"only {len(raises)} UsageError raises found in {module}:{function}; the parse "
        "is looking in the wrong place, since this function refuses more ways than that"
    )

    for line, call in raises:
        where = f"{module}:{line}"
        assert call.args, f"{where} raises UsageError with no message"
        interpolated = {node.id for node in ast.walk(call.args[0]) if isinstance(node, ast.Name)}
        assert variable in interpolated, (
            f"{where} refuses a plan without naming the file. A user with several plans "
            f"on disk cannot tell which one was refused; interpolate `{variable}` into "
            "the message the way the other branches do."
        )
        details = [kw.value for kw in call.keywords if kw.arg == "details"]
        assert details and isinstance(details[0], ast.Dict), (
            f"{where} raises UsageError with no details dict, so nothing reading the "
            "error as data can tell which file it was about"
        )
        keys = {k.value for k in details[0].keys if isinstance(k, ast.Constant)}
        assert "path" in keys, (
            f"{where} omits 'path' from details. The rendered message names the file "
            "but the machine-readable form does not, so a script cannot act on it."
        )


# --------------------------------------------------------------------------- #
# Every serialised field is one that from_dict actually checks
# --------------------------------------------------------------------------- #
def _serialised_config(which: str) -> type:
    """One of the dataclasses rebuilt from a file, by name.

    Imported inside the test rather than at module scope, matching the rest of this
    file: these are assertions *about* trainai, so importing it is part of the test.
    """
    if which == "ModelConfig":
        from trainai.model.config import ModelConfig

        return ModelConfig
    from trainai.train.config import TrainConfig

    return TrainConfig


@pytest.mark.parametrize("which", ["ModelConfig", "TrainConfig"])
def test_every_serialised_annotation_is_one_the_checker_can_check(which: str) -> None:
    """A field whose annotation is unhandled is a field loaded without checking.

    :func:`trainai.serialise.checked_fields` derives what it checks from the annotations,
    via a mapping from annotation to acceptable JSON types. That mapping is deliberately
    not exhaustive over Python -- there is no sensible check for an arbitrary type -- so
    an annotation missing from it does not fail, it goes *unchecked*: the field is
    neither required nor validated, and a file that omits it falls through to the
    dataclass default.

    That is the defect this whole area was fixed for, and it is the one gap the design
    still has, so it is asserted here rather than trusted to whoever adds the next field.
    A ``list[int]`` or ``Path`` field would reopen it silently. If this fails, either add
    the annotation to ``_ANNOTATION_TYPES`` with the JSON types that satisfy it, or state
    in that mapping's comment why the field cannot be checked.

    Both dataclasses are covered, because the checker is shared and a gap in it is a gap
    in every file TrainAI reads: a checkpoint's architecture and its hyperparameters.
    """
    from trainai.serialise import _ANNOTATION_TYPES, _TYPE_NAMES, serialised_field_types

    cls = _serialised_config(which)
    hints = get_type_hints(cls)
    checked = serialised_field_types(cls)

    for name in cls.__dataclass_fields__:
        parts = [
            part for part in (get_args(hints[name]) or (hints[name],)) if part is not type(None)
        ]
        # A Literal's arguments are values, not types. Those fields are required but
        # value-checked by __post_init__ instead, which the next test proves.
        if all(not isinstance(part, type) for part in parts):
            assert checked.get(name) is not None and checked[name].types is None, (
                f"{which}.{name} is annotated {hints[name]!r}, which the checker reads "
                "as a Literal, but it did not come back as one"
            )
            continue
        unhandled = [part for part in parts if part not in _ANNOTATION_TYPES]
        assert not unhandled, (
            f"{which}.{name} is annotated {hints[name]!r}, and "
            f"{[getattr(p, '__name__', p) for p in unhandled]} is not in "
            "_ANNOTATION_TYPES. checked_fields would accept any value for it, and a "
            f"file that omitted {name} would fall back to the default rather than be "
            "refused."
        )
        assert name in checked, (
            f"{which}.{name} is absent from serialised_field_types, so from_dict "
            "neither requires it nor checks it -- a file missing it would fall through "
            "to the dataclass default and rebuild something the file does not describe"
        )

    for name, spec in checked.items():
        assert spec.types is None or spec.types in _TYPE_NAMES, (
            f"{which}.{name} resolves to the JSON types {spec.types!r}, which "
            "_TYPE_NAMES cannot name. Refusing that field would raise KeyError "
            "composing its own message, turning a clear refusal into a traceback."
        )


@pytest.mark.parametrize(
    ("which", "field", "valid"),
    [("TrainConfig", "schedule", "cosine"), ("TrainConfig", "precision", "bf16")],
)
def test_a_literal_field_is_still_refused_by_post_init(which: str, field: str, valid: str) -> None:
    """The exemption above is a claim about ``__post_init__``, so check the claim.

    ``checked_fields`` skips the type of a ``Literal`` field on the grounds that
    ``__post_init__`` refuses bad values with a better message -- one that lists what is
    allowed. If that ever stops being true, the field becomes the *only* kind that is
    required but never validated, and this test is what says so.
    """
    from trainai.errors import ConfigError

    cls = _serialised_config(which)
    good = cls(steps=100).to_dict() if which == "TrainConfig" else cls(vocab_size=256).to_dict()
    assert getattr(cls.from_dict({**good, field: valid}), field) == valid

    with pytest.raises(ConfigError) as caught:
        cls.from_dict({**good, field: "not-a-real-choice"})
    assert field in str(caught.value.message), (
        f"{which}.{field} is not type-checked on load because __post_init__ was meant "
        f"to refuse it by name, and this refusal does not name it: {caught.value.message}"
    )
    assert caught.value.hint, f"{which}.{field} is refused without saying what to use"


# --------------------------------------------------------------------------- #
# Commands the user is meant to copy
# --------------------------------------------------------------------------- #
#: Every CLI module that prints a command for the user to run, and how many such
#: commands it prints. A census rather than a total, so a site that loses its
#: `print_command` names the module it went missing from.
COPYABLE_COMMAND_SITES = {
    "data.py": 2,
    "export.py": 1,
    "plan.py": 2,
    "quickstart.py": 2,
    "setup.py": 1,
    "train.py": 2,
}


def _print_command_calls(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print_command"
    )


def test_every_command_the_user_is_meant_to_copy_is_printed_unwrapped() -> None:
    """A command printed through plain `console.print` is wrapped, and cannot be copied.

    Found by running the documented quickstart: `trainai plan` printed its recommended
    150-character `trainai train` command through `console.print`, so Rich broke it
    across three lines. Pasting that runs one complete command and then a line starting
    `--steps`. Four other sites had the same shape, including the `pip install` command
    that `trainai setup` tells the user to run.

    `print_command` is the only correct way to print one, and `test_console.py` proves
    the helper survives a narrow terminal. This is the census that keeps the *call sites*
    using it -- a revert at any one of them fails here, naming the file.

    What this cannot catch is a *new* site added with `console.print`. It is a floor, not
    a ban, and deliberately so: the original bug was `f"  [bold cyan]{plan.train_command()}[/]"`,
    an f-string around a method call, so no scan for command-shaped string literals could
    have found it either. The narrow-width tests on `plan` and on the helper are the part
    that proves behaviour; this only stops a silent deletion.
    """
    cli = REPO_ROOT / "src" / "trainai" / "cli"
    found = {path.name: _print_command_calls(path) for path in sorted(cli.glob("*.py"))}

    for name, expected in COPYABLE_COMMAND_SITES.items():
        assert found.get(name) == expected, (
            f"{name} calls print_command {found.get(name)} time(s), expected {expected}. "
            "If a command is no longer printed, update COPYABLE_COMMAND_SITES; if it is "
            "printed through console.print instead, it will wrap and cannot be copied."
        )


# --------------------------------------------------------------------------- #
# Documents that describe CI
# --------------------------------------------------------------------------- #
#: Phrasings of "CI has never executed", which stopped being true the first time it did.
#: Matched case-insensitively against the prose of every tracked Markdown file.
NEVER_RAN_CLAIMS = (
    "ci has never",
    "ci had never",
    "never actually executed",
    "never having executed",
    "never having run",
    "has never been run by ci",
)


def test_no_document_says_ci_has_never_run() -> None:
    """This claim went stale for months, and then went stale twice in one commit.

    The `## Known limitations` bullet said "CI has never actually executed" long after ten
    jobs were green. Rewriting it was not enough: the roadmap preamble, forty lines from
    the top of the same file, still pointed a reader *at* that bullet for the same dead
    claim. One phrasing being fixed is no evidence about the others, which is the whole
    reason to assert over the documents rather than over one line of one of them.

    Guarded by the existence of the workflow, so a repository that genuinely has no CI is
    free to say so -- the test is about the two facts disagreeing, not about the wording.
    """
    if not (REPO_ROOT / ".github" / "workflows" / "ci.yml").exists():
        pytest.skip("no CI workflow, so a document saying CI has never run would be true")

    try:
        listed = subprocess.run(
            ["git", "ls-files", "*.md"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=REPO_ROOT,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout, or git is unavailable")

    offenders = []
    for relative in sorted(filter(None, listed.stdout.splitlines())):
        path = REPO_ROOT / relative
        if not path.is_file():
            continue
        lowered = path.read_text(encoding="utf-8").lower()
        offenders += [f"{relative}: {claim!r}" for claim in NEVER_RAN_CLAIMS if claim in lowered]

    assert not offenders, (
        "these documents say CI has never run, and .github/workflows/ci.yml exists:\n  "
        + "\n  ".join(offenders)
    )
