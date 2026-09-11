"""The walkthrough is a recipe, so it is run rather than read.

Three checks already read the documents: a flag has to exist, a command has to exist, and
a flag has to be on a command that takes it. A fourth reads the walkthrough as a sequence
and requires that a directory one step reads is one an earlier step wrote. All four are
static, and all four together still pass on a page whose commands do not work -- a value
no option accepts (`--which latest`), a step that needs a flag it does not pass, an
ordering that is fine on paper and wrong in practice.

So this runs it. The page's own commands, in the page's own order, with the page's own
paths, in a temp directory that is the process's working directory for the duration --
`data/shake` and `runs/m4` are created exactly as written, not rewritten to a fixture
path. What changes is stated in :data:`SMALLER` and :data:`ADDED` and is asserted to be
only that: magnitudes shrink and two bounds are added, no flag the page shows is dropped
or renamed, and no step is quietly skipped.

What this does **not** verify is the transcripts. Every number in them was measured on one
machine with a GPU -- step times, losses, a device line reading `cuda - bf16` -- and a
test that compared them would either fail everywhere or force the numbers to be fiction.
The recorded output stays a recording. This checks that the commands around it run.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import os
import re
import shlex
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from conftest import cli_command_paths, cli_flag_owners, cli_group_paths
from test_conventions import (
    arguments_after,
    as_written,
    command_on_line,
    joined_block_lines,
    no_such_command,
)
from trainai.cli.main import main
from trainai.errors import ExitCode

REPO_ROOT = Path(__file__).resolve().parents[1]
WALKTHROUGH = REPO_ROOT / "docs" / "walkthrough.md"
FETCH_SCRIPT = REPO_ROOT / "examples" / "get_tinyshakespeare.py"

#: Commands the page shows that this test does not run, and why. Keyed by the command as
#: written, comments stripped, so a step whose text changes stops matching and has to be
#: looked at again rather than staying excused by line number.
#:
#: Every command in a ``bash`` block is either run or in here. That is the property worth
#: having: a step added to the page and neither run nor named is a step nobody checks, and
#: an untested step in the middle of a recipe is what this file exists to refuse.
NOT_RUN = {
    "python examples/get_tinyshakespeare.py": (
        "downloads 1.1 MB over the network. Replaced by a corpus written to the same "
        "path the script writes, so the step after it is the page's own command"
    ),
    "trainai data prepare reviews.csv --out data/reviews --csv-text-column review_text": (
        "an aside about spreadsheets; reviews.csv is a file the page never creates"
    ),
    "trainai data inspect ./my-corpus": "./my-corpus is a placeholder for the reader's own",
}

#: Numbers that shrink. The flag stays exactly as the page writes it and only its value
#: changes, which is checked rather than trusted
#: (`test_the_overrides_only_change_values_of_flags_the_page_already_passes`).
#:
#: Magnitudes are not what drifts. A renamed flag, a value no option accepts, a step in
#: the wrong order -- those break a reader's paste and are what this run is for. 600 steps
#: at batch 32 and context 256 is minutes of CPU per run and would price the check out of
#: CI, which is the one place it has to run.
SMALLER = {
    "--context": "64",
    "--seq-len": "64",
    "--steps": "4",
    "--batch-size": "2",
    "--grad-accum": "1",
    "--eval-every": "2",
    "--eval-batches": "1",
    "--checkpoint-every": "2",
    "--tokens": "16",
}

#: The only flags added to what the page shows, each because leaving it off makes the run
#: depend on the machine rather than on the code.
#:
#: ``--device cpu`` goes on every command that accepts it: the page passes no ``--device``
#: because TrainAI picks one, and a test that trains on whatever GPU the runner happens to
#: have is a test whose cost and numerics are not the same twice.
#:
#: ``plan`` is a search that is deliberately unbounded -- it climbs the preset ladder until
#: a rung stops fitting, which on a large runner is minutes -- so it is bounded to one rung
#: and the minimum sample per candidate. Measured on this machine: 21.6s unbounded against
#: 11.3s here, and the ladder itself is what `tests/test_planner.py` is for.
ADDED = {
    "plan": ("--max-preset", "tiny", "--measure-steps", "1", "--warmup-steps", "1"),
}

#: How many of the page's steps this test actually executes. A literal, because every check
#: that runs the page is "for each step that ran, ..." and passes on an empty list -- and the
#: cheapest way to empty that list is to excuse a step, which is one line in :data:`NOT_RUN`
#: that nothing else here would object to. Moving this number is the price of that.
EXECUTED_STEPS = 9

#: Files the page promises in prose, each with the sentence that promises it. Paired here so
#: a check has a claim behind it: the quote is asserted against the page as well, with the
#: page's line wrapping collapsed, so deleting the promise fails rather than leaving an
#: assertion nobody can trace back to something the page told a reader.
#:
#: ``plan.json`` earns its place -- ``plan`` writes it to the working directory with no flag
#: saying so, which is the kind of thing a reader only finds out by looking.
PROMISED = {
    "plan.json": "Plan written to plan.json",
    "runs/m4/metrics.jsonl": "Everything measured goes to `<run>/metrics.jsonl`",
    "models/shakespeare-4m/README.md": "A `README.md` goes into the directory",
    "models/shakespeare-4m/config.json": 'AutoModelForCausalLM.from_pretrained("models/',
}

#: A word pool for the stand-in corpus. Prose-shaped and varied enough that the corpus is
#: not refused as a table or as near-duplicate text, which is all the page's first step
#: needs from it: what is being tested is the command, not the poetry.
WORD_POOL = """my lord good sir the king doth speak of honour and of blood what news from
france a bloody day my liege the crown is heavy on this head who dares to say the word
again i pray you gentle friend be still the night grows long and morning will not come
without a sword i swear upon my mother grave that treason shall not sleep in england
"""

WORDS = WORD_POOL.split()


@dataclass(frozen=True)
class Step:
    """One command the page shows, as written and as run.

    ``code`` is ``None`` for a step :data:`NOT_RUN` excuses, which is how such a step stays
    visible to the run-or-excused check instead of being dropped before anything counts it.
    """

    lineno: int
    written: str
    ran: tuple[str, ...]
    code: int | None


@dataclass(frozen=True)
class Executed:
    """The result of running the page: every step, and the directory it ran in."""

    work: Path
    steps: tuple[Step, ...]

    def ran(self) -> tuple[Step, ...]:
        return tuple(step for step in self.steps if step.code is not None)


def fetch_script_destination() -> Path:
    """Where ``examples/get_tinyshakespeare.py`` puts the corpus, asked of the script.

    Imported rather than hard-coded so the stand-in lands where the real step would. A
    corpus written somewhere else would make the page's next command fail for a reason the
    page is not responsible for, and hard-coding the path would hide the day it moves.
    """
    spec = importlib.util.spec_from_file_location("get_tinyshakespeare", FETCH_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return Path(module.DESTINATION)


def write_stand_in_corpus(destination: Path, target_bytes: int = 1_100_000) -> None:
    """Roughly as much text as the corpus the page downloads, without the download.

    The size matters: the default vocabulary is sized for about a megabyte, and a corpus
    small enough to trip the "vocabulary too large" warning would have the first step
    testing the warning instead of the command.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    paragraphs: list[str] = []
    written, index = 0, 0
    while written < target_bytes:
        lines = []
        for _ in range(40):
            index += 1
            picked = [WORDS[(index * 7 + offset * 13) % len(WORDS)] for offset in range(10)]
            lines.append(f"{' '.join(picked)} {index}.")
        paragraph = "\n".join(lines)
        paragraphs.append(paragraph)
        written += len(paragraph) + 2
    destination.write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8", newline="\n")


def walkthrough_steps() -> list[tuple[int, str]]:
    """Every command in a ``bash`` block on the page, wrapped lines folded, in page order.

    The two helpers the static dataflow check uses, imported rather than reimplemented: if
    the two disagreed about what counts as a step, one of them would be checking a page
    that does not exist. ``language="bash"`` is what separates a command a reader types
    from a transcript of what one printed.
    """
    return joined_block_lines(WALKTHROUGH.read_text(encoding="utf-8"), language="bash")


def as_typed(line: str) -> str:
    """One spelling of a step: comment stripped, whitespace normalised, quoting canonical.

    :data:`NOT_RUN` is keyed by this. The page annotates its steps
    (``# ~1.1 MB of text to play with``), and an excuse that carried the comment would
    lapse the moment the comment was reworded -- excusing a step is a claim about the
    command, not about the prose beside it.
    """
    return shlex.join(shlex.split(line, comments=True))


def accepted_flags() -> dict[str, set[str]]:
    """Long flags per command path, inverted from the ``flag -> commands`` map."""
    accepts: dict[str, set[str]] = {}
    for flag, commands in cli_flag_owners().items():
        for command in commands:
            accepts.setdefault(command, set()).add(flag)
    return accepts


def additions_for(command: str, accepts: set[str]) -> tuple[str, ...]:
    """The flags this run adds to the page's own -- see :data:`ADDED` -- and nothing else."""
    extra: list[str] = ["--device", "cpu"] if "--device" in accepts else []
    extra.extend(ADDED.get(command, ()))
    return tuple(extra)


def as_run(words: Sequence[str], command: str, accepts: set[str]) -> tuple[str, ...]:
    """The page's command with :data:`SMALLER` values substituted and :data:`ADDED` appended.

    A substitution rather than a rewrite: the flag keeps the name and the position the page
    gives it, and only the word after it changes.
    """
    out: list[str] = []
    skip = False
    for index, word in enumerate(words):
        if skip:
            skip = False
            continue
        out.append(word)
        if word in SMALLER and index + 1 < len(words):
            out.append(SMALLER[word])
            skip = True
    return (*out, *additions_for(command, accepts))


def magnitudes_masked(words: Sequence[str]) -> tuple[str, ...]:
    """Every value a :data:`SMALLER` flag takes, replaced by one placeholder.

    Two commands that differ only in those values mask to the same tuple, which is how
    "only magnitudes changed" is asserted without restating the overrides: the page's
    command and the command that ran have to mask identically, so a flag dropped, renamed
    or moved -- or a path rewritten to a fixture -- is a mismatch rather than a detail
    nobody looks at.
    """
    masked: list[str] = []
    skip = False
    for index, word in enumerate(words):
        if skip:
            masked.append("<value>")
            skip = False
            continue
        masked.append(word)
        skip = word in SMALLER and index + 1 < len(words)
    return tuple(masked)


@pytest.fixture(scope="module")
def executed_walkthrough(tmp_path_factory: pytest.TempPathFactory) -> Executed:
    """The page, run once: every step in order, in a temp directory that is the cwd.

    The cwd is what makes the paths the page's own. ``--out data/shake`` is passed exactly
    as the page writes it and lands in the temp tree, so the next step's ``--data
    data/shake`` is the page's text and not a fixture path threaded through it. A step that
    only works because an earlier one wrote somewhere else fails here.

    Module-scoped: the whole page is one run of roughly half a minute, and every assertion
    about it reads the same run rather than paying for its own.
    """
    work = tmp_path_factory.mktemp("walkthrough")
    write_stand_in_corpus(work / fetch_script_destination())

    accepts_by_command = accepted_flags()
    known = cli_command_paths()
    steps: list[Step] = []

    here, argv = Path.cwd(), sys.argv
    try:
        os.chdir(work)
        for lineno, line in walkthrough_steps():
            written = as_typed(line)
            command = command_on_line(line, known)
            if written in NOT_RUN or command is None:
                steps.append(Step(lineno, written, (), None))
                continue
            ran = as_run(
                shlex.split(line, comments=True), command, accepts_by_command.get(command, set())
            )
            sys.argv = list(ran)
            try:
                code = main()
            except SystemExit as exit_:  # a usage error exits rather than returning
                code = exit_.code if isinstance(exit_.code, int) else 1
            steps.append(Step(lineno, written, ran, code))
    finally:
        os.chdir(here)
        sys.argv = argv

    return Executed(work=work, steps=tuple(steps))


# ---------------------------------------------------------------------------
# What is checked without running anything
# ---------------------------------------------------------------------------


def test_every_step_the_walkthrough_shows_is_either_run_or_excused() -> None:
    """A step added to the page and neither run nor named is a step nobody checks.

    The failure this refuses is quiet: someone appends a command to the recipe, the suite
    stays green because nothing enumerates the page, and the new step is the only one on
    the page that has never been executed. Excusing it is fine -- three steps are -- but it
    has to be written down, with a reason, in :data:`NOT_RUN`.
    """
    known = cli_command_paths()
    unaccounted = [
        (lineno, as_typed(line))
        for lineno, line in walkthrough_steps()
        if as_typed(line) not in NOT_RUN and command_on_line(line, known) is None
    ]
    assert unaccounted == [], (
        "these steps are neither a trainai command this test runs nor excused in NOT_RUN: "
        f"{unaccounted}"
    )


def test_the_page_still_shows_every_step_the_excuses_name() -> None:
    """An excuse for a step the page no longer shows is an excuse nobody re-reads.

    Keyed by command text rather than line number precisely so this can be asked: a reworded
    or deleted step stops matching, and the entry has to be looked at again instead of
    silently excusing whatever now occupies that line.
    """
    shown = {as_typed(line) for _, line in walkthrough_steps()}
    assert sorted(set(NOT_RUN) - shown) == []


def test_the_walkthrough_is_a_recipe_of_several_steps() -> None:
    """A floor, so an extraction that quietly stops finding steps fails here.

    Every other check in this file is "for each step, ..." and passes vacuously on an empty
    list. The page has twelve; the floor is loose enough that editing the recipe does not
    move it and tight enough that returning nothing is caught.
    """
    steps = walkthrough_steps()
    assert len(steps) > 8, f"only {len(steps)} steps found in docs/walkthrough.md"
    assert len(steps) - len(NOT_RUN) > 5


def test_the_overrides_only_change_the_values_of_flags_the_page_already_passes() -> None:
    """The invariant that makes this run evidence about the page rather than about a fixture.

    Executing a recipe is only worth something if what executes is the recipe. Mask the
    values :data:`SMALLER` overrides in both the page's command and the command that ran,
    and the two must be identical -- so no flag is dropped to make a step pass, no path is
    redirected to somewhere convenient, and no argument is reordered.
    """
    accepts_by_command, known = accepted_flags(), cli_command_paths()
    compared = 0
    for lineno, line in walkthrough_steps():
        command = command_on_line(line, known)
        if as_typed(line) in NOT_RUN or command is None:
            continue
        words = shlex.split(line, comments=True)
        added = additions_for(command, accepts_by_command.get(command, set()))
        ran = as_run(words, command, accepts_by_command.get(command, set()))
        assert ran[len(ran) - len(added) :] == added, f"line {lineno}: unexpected additions"
        assert magnitudes_masked(ran[: len(ran) - len(added)]) == magnitudes_masked(words), (
            f"line {lineno}: the command that runs is not the command the page shows"
        )
        compared += 1
    assert compared > 5


def test_the_only_flags_this_run_adds_are_the_device_and_the_plan_bounds() -> None:
    """Pinned to a literal, so widening :data:`ADDED` is a decision and not a drift.

    The check above compares the page against what runs and would accept any addition at
    all as long as it were appended. This is the other half: which additions are allowed is
    written out here, and a fourth one has to be justified in a diff someone reads.
    """
    accepts_by_command, known = accepted_flags(), cli_command_paths()
    added: set[str] = set()
    for _, line in walkthrough_steps():
        command = command_on_line(line, known)
        if as_typed(line) in NOT_RUN or command is None:
            continue
        extra = additions_for(command, accepts_by_command.get(command, set()))
        added.update(word for word in extra if word.startswith("--"))
    assert added == {"--device", "--max-preset", "--measure-steps", "--warmup-steps"}


@pytest.mark.parametrize("flag", sorted(SMALLER))
def test_every_override_shrinks_a_magnitude_the_page_actually_passes(flag: str) -> None:
    """Both halves of what :data:`SMALLER` claims to be, asked of the page.

    *Passed*: an override for a flag the page does not use is dead weight that reads as
    coverage, and applies silently the day the page starts passing it.

    *Smaller*: "magnitudes shrink" is the whole justification for overriding anything, and
    it is the sentence that stops :data:`SMALLER` from becoming a place to put a value that
    makes a step pass. ``--lr`` is the one to picture -- it takes a number, so it would slot
    in unnoticed, and replacing the page's learning rate would leave the page's own
    arithmetic untested while every other check here stayed green.
    """
    asked_for = [
        value
        for _, line in walkthrough_steps()
        if as_typed(line) not in NOT_RUN
        for word, value in itertools.pairwise(shlex.split(line, comments=True))
        if word == flag
    ]
    assert asked_for, f"{flag} is overridden but no step of the page passes it"
    assert all(float(SMALLER[flag]) < float(value) for value in asked_for), (
        f"{flag} is overridden to {SMALLER[flag]}, which does not shrink {asked_for}"
    )


@pytest.mark.parametrize(("produced", "promised_by"), sorted(PROMISED.items()))
def test_the_page_still_promises_every_file_the_run_looks_for(
    produced: str, promised_by: str
) -> None:
    """The other half of :data:`PROMISED`: the sentence is still on the page.

    Without this, the file-exists checks slowly become assertions about whatever TrainAI
    happens to write, which is a different and much weaker claim than "the page told a
    reader this would be here". Wrapping is collapsed because the page wraps mid-sentence
    and where it wraps is the formatter's business.
    """
    unwrapped = " ".join(WALKTHROUGH.read_text(encoding="utf-8").split())
    assert " ".join(promised_by.split()) in unwrapped, f"the page no longer promises {produced}"


@pytest.mark.parametrize(
    ("words", "masked"),
    [
        (
            ["trainai", "eval", "runs/m4", "--which", "best"],
            ("trainai", "eval", "runs/m4", "--which", "best"),
        ),
        (["--steps", "600"], ("--steps", "<value>")),
        (["--steps", "600", "--seed", "1234"], ("--steps", "<value>", "--seed", "1234")),
        (["--dry-run", "--steps", "600"], ("--dry-run", "--steps", "<value>")),
        (["--steps"], ("--steps",)),
        (["--tokens", "120", "--tokens", "120"], ("--tokens", "<value>", "--tokens", "<value>")),
    ],
)
def test_masking_hides_the_overridden_values_and_nothing_else(
    words: list[str], masked: tuple[str, ...]
) -> None:
    """The override invariant is a comparison of two masked commands, so the mask matters.

    A mask that hid one word too many would make that comparison pass on commands that
    differ, which is the failure mode where the recipe quietly stops being what runs. Spelled
    out against literals for that reason: ``--seed 1234`` keeps its value because ``--seed``
    is not overridden, and a trailing flag with nothing after it is left alone rather than
    reaching past the end of the command.
    """
    assert magnitudes_masked(words) == masked


def commands_in_source(source: str) -> list[str]:
    """Every ``trainai ...`` command a Python file names, from its prose and its output.

    Line by line, taking the tail from ``trainai`` and dropping the quoting around it, which
    catches both forms the fetch script uses: the recipe in its module docstring and the
    ``print("  trainai ...")`` it leaves on the reader's terminal. Neither is reachable by
    anything that reads ``docs/``, and both are instructions a reader follows.

    Case matters: ``TrainAI`` in a sentence is the project, not a command.
    """
    found: list[str] = []
    for line in source.splitlines():
        match = re.search(r"\btrainai\s.*", line)
        if match is not None:
            found.append(match[0].rstrip().rstrip("\"')").rstrip())
    return found


def why_it_would_not_run(command: str, known: set[str], groups: set[str]) -> str | None:
    """Why ``trainai ...`` as this file writes it would not run, or ``None`` if it would.

    Deliberately not ``command_on_line``, which answers a different question: it returns the
    longest recognised prefix, and a group is itself a recognised path, so ``trainai data
    inpect data/corpus`` resolves happily to ``data`` and a misspelt leaf ships. Descending
    the tree is what separates a typo after a group from an argument after a leaf, and
    ``no_such_command`` already does it for the checks that read ``docs/``.
    """
    return no_such_command(shlex.split(command)[1:], known, groups)


def test_the_fetch_script_only_names_commands_that_exist() -> None:
    """A typo in the recipe the script prints is a dead end nothing else would catch.

    The static docs checks read ``docs/``. This recipe is a Python string in ``examples/``,
    so ``trainai data inpect data/corpus`` would ship.
    """
    named = commands_in_source(FETCH_SCRIPT.read_text(encoding="utf-8"))
    assert named, "the script no longer names a trainai command; is it still the first step?"
    known, groups = cli_command_paths(), cli_group_paths()
    broken = [
        f"{command} ({reason})"
        for command in named
        if (reason := why_it_would_not_run(command, known, groups)) is not None
    ]
    assert broken == [], f"{FETCH_SCRIPT.name} names commands that would not run: {broken}"


def test_the_fetch_script_and_the_walkthrough_agree_on_the_paths() -> None:
    """Where the drift this test file was written for actually lived.

    The script writes a corpus and prints the two commands to run on it; the page picks the
    recipe up from there. Both name directories, and nothing made them agree -- the script
    said ``data/shakespeare`` while the page trained from ``data/shake``, and a reader who
    pasted both got a manifest error four commands later.

    Path equality, not command equality: the script is free to word its advice its own way,
    and demanding it quote the page verbatim would be a rule about phrasing. What it is not
    free to do is name a directory the recipe does not use.
    """
    known, groups = cli_command_paths(), cli_group_paths()
    on_the_page = {
        as_written(word)
        for _, line in walkthrough_steps()
        for word in shlex.split(line, comments=True)
        if "/" in word and not word.startswith("-")
    }
    named = {
        as_written(word)
        for command in commands_in_source(FETCH_SCRIPT.read_text(encoding="utf-8"))
        if why_it_would_not_run(command, known, groups) is None
        for word in shlex.split(command)
        if "/" in word and not word.startswith("-")
    }
    assert named, f"{FETCH_SCRIPT.name} names no paths at all"
    assert sorted(named - on_the_page) == [], (
        f"named by {FETCH_SCRIPT.name} but on no step of docs/walkthrough.md"
    )


def test_the_corpus_the_fetch_script_writes_is_the_one_the_page_prepares() -> None:
    """The handover itself: the page's first ``prepare`` reads where the script writes.

    ``DESTINATION`` is a file; the page prepares its parent directory. Asked of the script
    rather than restated, which is also a smoke test that the script imports at all.
    """
    corpus = fetch_script_destination()
    known = cli_command_paths()
    prepared = [
        word
        for _, line in walkthrough_steps()
        if command_on_line(line, known) == "data prepare" and as_typed(line) not in NOT_RUN
        for word in arguments_after(line, "data prepare")[:1]
    ]
    assert corpus.parent.as_posix() in prepared, (
        f"{FETCH_SCRIPT.name} writes {corpus.as_posix()}, but the page prepares {prepared}"
    )


# ---------------------------------------------------------------------------
# What is checked by running it
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_every_step_of_the_walkthrough_succeeds(executed_walkthrough: Executed) -> None:
    """The whole point: the page's commands, in the page's order, all exit ``OK``.

    This is the check the four static ones cannot make. They read; none of them can tell
    that a value no option accepts, a step that needs a flag it does not pass, or an order
    that is fine on paper will fail when a reader types it.
    """
    examined = executed_walkthrough.ran()
    assert len(examined) >= EXECUTED_STEPS, "this passes on an empty list, so it counts first"
    failed = [
        (step.lineno, step.code, shlex.join(step.ran))
        for step in examined
        if step.code != ExitCode.OK
    ]
    assert failed == [], f"{len(failed)} step(s) of docs/walkthrough.md failed: {failed}"


@pytest.mark.slow
def test_the_run_covers_every_step_the_page_does_not_excuse(
    executed_walkthrough: Executed,
) -> None:
    """Nine steps ran, three are excused, twelve are on the page -- and it adds up.

    The arithmetic is the point: it ties the number of steps that ran to the page's own
    length and to the excuses, so a step that stops being found is a mismatch rather than one
    fewer iteration of a loop nobody counts.
    """
    steps = executed_walkthrough.steps
    assert len(steps) == len(walkthrough_steps())
    assert len(executed_walkthrough.ran()) == len(steps) - len(NOT_RUN)
    assert len(executed_walkthrough.ran()) == EXECUTED_STEPS
    assert {step.written for step in steps if step.code is None} == set(NOT_RUN)


@pytest.mark.slow
@pytest.mark.parametrize("produced", sorted(PROMISED))
def test_the_walkthrough_leaves_behind_what_it_says_it_does(
    executed_walkthrough: Executed, produced: str
) -> None:
    """Exit codes are not evidence that a file was written where the page says it is.

    Every path is one the page promises -- see :data:`PROMISED`, which is also where the
    sentence behind each one is kept and checked.
    """
    assert (executed_walkthrough.work / produced).is_file(), f"the page promises {produced}"


@pytest.mark.slow
def test_the_export_is_the_llama_directory_the_page_says_it_is(
    executed_walkthrough: Executed,
) -> None:
    """The page's claim that the export "**is** a Llama structurally", read off the export.

    Checked here rather than left to the export tests because this is the artefact the
    page's own Python snippet loads, and ``from_pretrained`` picks the architecture out of
    ``model_type``. A wrong value there is the page's snippet failing, not an internal
    detail -- and it needs nothing from ``transformers`` to notice.
    """
    config = json.loads(
        (executed_walkthrough.work / "models/shakespeare-4m/config.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["model_type"] == "llama"
    assert config["architectures"] == ["LlamaForCausalLM"]
