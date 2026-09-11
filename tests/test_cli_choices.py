"""One net over every CLI option whose valid values are a closed set.

The bug this exists to prevent, in the form it actually shipped: ``--precision fp64``
was accepted, silently treated as ``auto``, and then reported as though that had been
the ask. Typer validates an ``Enum`` and a ``bool`` and nothing else, so an option
declared ``TEXT`` whose valid values live only in its help text arrives as an unchecked
string, and every consumer that reaches for it with ``==`` treats it as whatever its
final branch does. A misspelt flag *name* is caught by the parser. A misspelt flag
*value* is caught by nothing unless someone remembered to write the check.

Three nets, because the bug had three faces:

* the option is **classified** -- a new free-text option is either a closed set or
  free-form-with-a-reason, and cannot ship unclassified;
* the option's **help text lists every value it accepts** -- two ``--device`` help
  texts had already drifted, omitting ``xpu``, and nothing reads help prose;
* the **command refuses a value outside the set**, end to end, exit code and message.

The choices come from the modules that enforce them, never from literals typed here.
A tuple copied into a test is a second list of precisions free to disagree with the
first, which is the same defect one layer up.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from conftest import cli_command, cli_command_paths, cli_long_flags, flat
from trainai.cli.eval import SPLIT_CHOICES
from trainai.cli.main import main
from trainai.errors import ExitCode
from trainai.export.bundle import DTYPE_CHOICES, FORMAT_CHOICES
from trainai.infer.session import WHICH_CHOICES
from trainai.model.config import PRESETS
from trainai.train.config import PRECISION_CHOICES, SCHEDULE_CHOICES
from trainai.train.loop import DEVICE_CHOICES

PRESET_NAMES: tuple[str, ...] = tuple(spec.name for spec in PRESETS)

#: Every option whose valid values are a closed set, and where that set is defined.
#: Keyed by flag name rather than by ``(command, flag)``: ``--precision`` means the same
#: thing on all six commands that take it, and keying it once means a seventh command is
#: covered by these tests the moment it is added rather than when someone adds a row.
CHOICES: dict[str, tuple[str, ...]] = {
    "--device": DEVICE_CHOICES,
    "--precision": PRECISION_CHOICES,
    "--schedule": SCHEDULE_CHOICES,
    "--preset": PRESET_NAMES,
    "--max-preset": PRESET_NAMES,
    "--which": WHICH_CHOICES,
    "--split": SPLIT_CHOICES,
    "--format": FORMAT_CHOICES,
    "--dtype": tuple(DTYPE_CHOICES),
    "--on-error": ("fail", "skip"),
}

#: The free-text options, with why each one is genuinely open-ended. Written out for
#: the same reason the coverage floor's exemptions are: a new option lands in neither
#: dict and fails :func:`test_every_free_text_option_is_classified`, which is a
#: question ("is this a closed set?") asked at the only moment anyone knows the answer.
FREE_FORM: dict[str, str] = {
    "--csv-text-column": "a column name in the user's own file",
    "--data": "a filesystem path",
    "--db-table": "a table name in the user's own database",
    "--encoding": "any codec Python knows; the list is not TrainAI's to close",
    "--from": "a filesystem path",
    "--jsonl-field": "a field name in the user's own records",
    "--jsonl-messages-field": "a field name in the user's own records",
    "--max-vram": "a size, parsed and range-checked rather than chosen from a list",
    "--name": "a directory name the user chooses",
    "--out": "a filesystem path",
    "--path": "a filesystem path",
    "--plan": "a filesystem path",
    "--prompt": "arbitrary text to generate from",
    "--resume": "a filesystem path",
    "--time": "a duration, parsed and range-checked rather than chosen from a list",
    "--tokenizer": "a filesystem path",
}

#: The free-form options whose value is nonetheless *parsed* rather than taken as given,
#: each with a command, a value it has to refuse, and a form its hint has to name.
#: ``FREE_FORM`` makes the claim "parsed and range-checked" in prose, and prose in a dict
#: is not a check: a quantity accepted and misread is worse than a closed set unchecked,
#: because ``--max-vram lots`` read as zero is a memory limit that rejects everything and
#: ``--time 2 hours`` read as two seconds is a training run that stops immediately. A path
#: is not parsed in this sense -- it is checked by being opened -- so only sizes and
#: durations are here.
PARSED: dict[str, tuple[str, str, str]] = {
    "--max-vram": ("plan", "plenty", "6GB"),
    "--time": ("plan", "soon", "90s"),
}

#: What each command needs before it gets as far as validating a flag value, as a
#: template over the shared trained-run fixture. A command that accepts an enumerated
#: option and is missing from here fails ``test_a_value_outside_the_choices_is_refused``,
#: so a new command cannot escape the end-to-end half by omission.
RUNNABLE: dict[str, tuple[str, ...]] = {
    "plan": ("plan", "--data", "{data}", "--out", "{tmp}/plan.json"),
    "train": ("train", "--data", "{data}", "--out", "{tmp}/run", "--dry-run"),
    "finetune": (
        "finetune",
        "--data",
        "{data}",
        "--from",
        "{run}",
        "--out",
        "{tmp}/run",
        "--dry-run",
    ),
    "eval": ("eval", "{run}"),
    "chat": ("chat", "{run}", "--prompt", "a"),
    "export": ("export", "{run}", "--out", "{tmp}/export"),
    "quickstart": ("quickstart", "{corpus}", "--out", "{tmp}/quick", "--yes"),
    "data prepare": ("data", "prepare", "{corpus}", "--out", "{tmp}/prepared"),
    "data inspect": ("data", "inspect", "{corpus}"),
}


def free_text_options() -> list[tuple[str, str]]:
    """Every ``(command, flag)`` the parser hands through as an unchecked string.

    Click validates an ``int``, a ``float``, a ``bool`` and its own ``Choice``. What is
    left is ``StringParamType``, and that is exactly the surface where a closed set can
    be declared in prose and enforced nowhere.
    """
    found: list[tuple[str, str]] = []
    for path in sorted(cli_command_paths()):
        command = cli_command(*path.split(" "))
        for parameter in command.params:
            longs = [opt for opt in parameter.opts if opt.startswith("--")]
            if not longs:
                continue
            if type(parameter.type).__name__ != "StringParamType":
                continue
            found.append((path, longs[0]))
    return found


def enumerated_options() -> list[tuple[str, str]]:
    """Every ``(command, flag)`` pair where the flag is one of the closed sets above."""
    return [
        (path, flag)
        for path in sorted(cli_command_paths())
        for flag in sorted(cli_long_flags(cli_command(*path.split(" "))))
        if flag in CHOICES
    ]


def option(path: str, flag: str) -> Any:
    """One click parameter, by the command that owns it and the flag that names it."""
    for parameter in cli_command(*path.split(" ")).params:
        if flag in parameter.opts:
            return parameter
    raise AssertionError(f"{path} does not accept {flag}")


def option_help(path: str, flag: str) -> str:
    """The help text one command shows for one flag, click's own answer."""
    return (getattr(option(path, flag), "help", "") or "").replace("\n", " ")


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def test_there_are_options_to_classify() -> None:
    """The guard against every test below passing because it found nothing.

    Both nets are parametrized over a walk of the real command tree, and a walk that
    silently returns an empty list turns "no unclassified options" and "no command
    accepts a bad value" into vacuous truths -- the same failure mode as a coverage
    floor computed over zero files. The floors are well under the real counts (61
    free-text pairs over 24 distinct flags, 24 enumerated pairs over 9 commands) so
    that adding a command does not move them, but far enough above zero that a
    collector returning nothing cannot pass.
    """
    free = free_text_options()
    enumerated = enumerated_options()

    assert len(free) >= 40, f"the free-text walk found only {len(free)} options"
    assert len({flag for _path, flag in free}) >= 20
    assert len(enumerated) >= 15, f"the enumerated walk found only {len(enumerated)} options"
    assert len({path for path, _flag in enumerated}) >= 6

    assert ("chat", "--precision") in enumerated, "the flag the original bug was about"
    assert ("train", "--out") in free, "a path is free-form, and has to be seen as one"


def test_every_free_text_option_is_classified() -> None:
    """A new ``TEXT`` option is a closed set or it is not, and someone has to say which.

    Left to itself the answer defaults to "not", which is how ``--precision`` came to
    accept ``fp64``: the values were written into the help text and nowhere else.
    """
    unclassified = sorted(
        {f"{path} {flag}" for path, flag in free_text_options() if flag not in CHOICES | FREE_FORM}
    )
    assert unclassified == [], (
        "these options are strings the parser does not check; add each to CHOICES "
        f"(with the constant that enforces it) or to FREE_FORM (with why): {unclassified}"
    )


def test_the_registries_do_not_name_options_that_no_longer_exist() -> None:
    """The other direction, which is what keeps the lists honest as flags are renamed.

    An entry for a flag nobody accepts is a check that silently stopped checking -- the
    same rule the coverage floor applies to an exemption naming a deleted module.
    """
    real = {flag for _path, flag in free_text_options()}
    real |= {
        flag
        for path in cli_command_paths()
        for flag in cli_long_flags(cli_command(*path.split(" ")))
    }
    stale = sorted((CHOICES.keys() | FREE_FORM.keys() | PARSED.keys()) - real)
    assert stale == [], f"no command accepts these any more: {stale}"


def test_no_option_is_in_both_registries() -> None:
    overlap = sorted(CHOICES.keys() & FREE_FORM.keys())
    assert overlap == [], f"a flag is a closed set or it is not: {overlap}"


def test_a_parsed_option_is_free_form_rather_than_a_closed_set() -> None:
    """``PARSED`` is a subset of ``FREE_FORM``, not a third classification.

    A flag whose values are enumerable belongs in ``CHOICES``, where the help text and the
    refusal are both checked against the constant that defines them. ``PARSED`` exists only
    to put a command behind the phrase "parsed and range-checked" that ``FREE_FORM`` uses.
    """
    misfiled = sorted(PARSED.keys() - FREE_FORM.keys())
    assert misfiled == [], f"these are in PARSED but not FREE_FORM: {misfiled}"


# --------------------------------------------------------------------------- #
# The help text
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("path", "flag"), enumerated_options(), ids=lambda value: str(value))
def test_an_options_help_text_lists_every_value_it_accepts(path: str, flag: str) -> None:
    """``--help`` is the only place most people will ever read the list.

    This is the check that was missing when two ``--device`` help texts drifted to
    "auto, cuda, cpu or mps" while the code had accepted ``xpu`` for three commits. Both
    lists were hand-written, so both were free to be wrong, and one was.

    Two shapes, because the CLI has two. An option click knows is a ``Choice`` prints its
    values itself, as ``[fail|skip]``, so the help text repeating them would be noise --
    what is worth checking there is that click's list and the list this file imports are
    the same list. Everything else is ``TEXT``, click prints nothing, and the help text is
    the only place the values can come from.
    """
    parameter = option(path, flag)
    declared = getattr(parameter.type, "choices", None)
    if declared is not None:
        assert tuple(declared) == CHOICES[flag], (
            f"the parser accepts {tuple(declared)} for {flag} but this file was told "
            f"{CHOICES[flag]}; one of the two lists is stale"
        )
        return

    text = option_help(path, flag)
    missing = [value for value in CHOICES[flag] if value not in text]
    assert missing == [], (
        f"`trainai {path} --help` does not mention {missing} for {flag}, "
        f"which it accepts. Help text: {text!r}"
    )


# --------------------------------------------------------------------------- #
# The command itself
# --------------------------------------------------------------------------- #
def _bogus(choices: tuple[str, ...]) -> str:
    """A value no closed set here contains, and not a near miss of one either.

    Deliberately not a mutation of a real value: a near miss would also exercise the
    suggestion machinery, and this test is about the refusal.
    """
    value = "zznope"
    assert value not in choices
    return value


def _argv(path: str, flag: str, run: Any, tmp_path: Any) -> list[str]:
    """The shortest argv that gets ``trainai <path>`` as far as validating a flag value."""
    assert path in RUNNABLE, (
        f"trainai {path} accepts {flag}, whose value this file checks end to end, but has "
        "no RUNNABLE entry, so nothing checks that a bad value reaches its validator"
    )
    substitutions = {
        "data": str(run.data),
        "run": str(run.run),
        "corpus": str(run.corpus),
        "tmp": str(tmp_path),
    }
    return [part.format(**substitutions) for part in RUNNABLE[path]]


@pytest.mark.parametrize(("path", "flag"), enumerated_options(), ids=lambda value: str(value))
def test_a_value_outside_the_choices_is_refused_by_the_command_itself(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Any,
    cli_trained_run: Any,
    path: str,
    flag: str,
) -> None:
    """End to end, because the unit-level check being present is not the claim.

    The claim is that the value a user types reaches it. ``--precision`` was validated
    by ``precision_for`` all along -- on the way to choosing a dtype, after the
    checkpoint was read, and with no branch for a value it did not know.
    """
    argv = _argv(path, flag, cli_trained_run, tmp_path)
    choices = CHOICES[flag]

    monkeypatch.setattr(sys, "argv", ["trainai", *argv, flag, _bogus(choices)])
    code = main()

    message = flat(capsys.readouterr().err)
    assert code == ExitCode.USAGE, f"exited {code}, and the error said: {message}"
    missing = [value for value in choices if value not in message]
    assert missing == [], f"the refusal does not name {missing}. It said: {message}"


@pytest.mark.parametrize(("flag", "spec"), sorted(PARSED.items()), ids=lambda value: str(value))
def test_a_quantity_it_cannot_read_is_refused_with_a_hint_showing_the_form(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Any,
    cli_trained_run: Any,
    flag: str,
    spec: tuple[str, str, str],
) -> None:
    """The check behind ``FREE_FORM``'s "parsed and range-checked", at the layer that failed.

    Both parsers have their own unit tables, and both flags have a test that calls
    ``run_plan`` directly and asserts nothing was measured. Neither of those goes through
    the argument parser, which is exactly the layer that turned ``--precision fp64`` into a
    silent ``auto``: a free-text option is only as validated as its consumer, and a value
    misread as a quantity is worse than one dropped, because it is acted on. ``--max-vram
    lots`` read as zero is a memory limit that rejects everything; ``--time 2 hours`` read
    as two seconds is a run that stops before it starts.

    The hint is asserted as well as the exit code, because "Cannot read 'plenty'" tells the
    user what is wrong without telling them what would have been right.
    """
    path, bad, form = spec
    argv = _argv(path, flag, cli_trained_run, tmp_path)

    monkeypatch.setattr(sys, "argv", ["trainai", *argv, flag, bad])
    code = main()

    message = flat(capsys.readouterr().err)
    assert code == ExitCode.USAGE, f"exited {code}, and the error said: {message}"
    assert bad in message, f"the refusal does not quote the value it refused: {message}"
    assert form in message, (
        f"the refusal never shows {form!r}, a value {flag} does accept. It said: {message}"
    )
