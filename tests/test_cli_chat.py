"""CLI tests for ``trainai chat``.

Two halves. The one-shot half runs the real model: a tiny run from the shared
``cli_trained_run`` fixture, on the CPU, generating a handful of tokens. It is slow
enough to be worth keeping short and real enough to catch what a fake would not --
tokenizer mismatches, device placement, an unprintable byte reaching the console.

The interactive half fakes stdin, because the loop reads it. ``console.input`` calls
the builtin ``input``, and :func:`trainai.cli.chat._piped_prompt` decides between the
playground and a pipe by asking ``sys.stdin.isatty()``, so a test of the loop has to
supply both: a stdin that claims to be a terminal, and answers to ``input``. Where a
test is about the loop's own bookkeeping rather than the model -- ``/more``, Ctrl-C --
``InferenceSession.stream_pieces`` is replaced by a recorder, which is the only way to
assert on the exact prompt the second generation received.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from conftest import flat, unwrapped
from trainai.cli.main import app, main
from trainai.errors import ExitCode

#: The context the shared fixture trains at. A prompt longer than this is truncated.
FIXTURE_CONTEXT = 64


def cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    """Drive the real entry point, argv and all, and return its exit code."""
    monkeypatch.setattr(sys, "argv", ["trainai", *args])
    return main()


class _FakeTerminal:
    """A stdin that claims to be a terminal, so chat picks the interactive loop."""

    def isatty(self) -> bool:
        return True

    def read(self) -> str:  # pragma: no cover - reached only if isatty is ignored
        raise AssertionError("the interactive loop must not read stdin directly")


class _FakePipe:
    """A stdin that claims to be a pipe holding ``text``."""

    def __init__(self, text: str) -> None:
        self._text = text

    def isatty(self) -> bool:
        return False

    def read(self) -> str:
        return self._text


def feed(monkeypatch: pytest.MonkeyPatch, *lines: str | type[BaseException]) -> None:
    """Answer ``input()`` with ``lines``, then end the session.

    An item that is an exception class is raised rather than returned, which is how a
    test spells a keystroke: ``KeyboardInterrupt`` is Ctrl-C at the prompt and
    ``EOFError`` is Ctrl-D.

    Exhaustion raises ``EOFError`` rather than ``StopIteration``: that is what a
    closed stdin does, and the loop already handles it. A test that forgets ``/exit``
    then ends cleanly instead of failing with an unrelated exception.
    """
    monkeypatch.setattr(sys, "stdin", _FakeTerminal())
    answers = iter(lines)

    def fake_input(*_args: Any) -> str:
        try:
            answer = next(answers)
        except StopIteration:
            raise EOFError from None
        if isinstance(answer, type) and issubclass(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr("builtins.input", fake_input)


def record_stream(
    monkeypatch: pytest.MonkeyPatch,
    *pieces: str,
    calls: list[dict[str, Any]] | None = None,
    finish: Any = None,
) -> list[str]:
    """Replace generation with ``pieces``, and return the list of prompts it saw.

    ``calls``, when given, collects each call's keyword arguments as well -- for the
    tests that care what generation was *asked* for, not only what it was given.

    ``finish`` adds the trailing piece a real stream ends with, for the tests about what
    the loop says once it knows why generation stopped. Left out by default: a fake that
    always claimed a reason would make every test here exercise one, and the reason a
    reader would then see asserted is whichever one this helper happened to pick.
    """
    from trainai.infer import InferenceSession, StreamPiece

    seen: list[str] = []

    def fake_stream(self: Any, prompt: str, **kwargs: Any) -> Iterator[StreamPiece]:
        seen.append(prompt)
        if calls is not None:
            calls.append(kwargs)
        for index, text in enumerate(pieces, start=1):
            yield StreamPiece(text, index)
        if finish is not None:
            yield StreamPiece("", len(pieces), finish)

    monkeypatch.setattr(InferenceSession, "stream_pieces", fake_stream)
    return seen


# --------------------------------------------------------------------------- #
# One shot, against the real model
# --------------------------------------------------------------------------- #
def test_a_prompt_generates_text_and_reports_what_it_cost(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """The rate is the one number a user cannot look up: it belongs to their machine."""
    run = cli_trained_run.run

    assert (
        cli(
            monkeypatch,
            "chat",
            str(run),
            "--prompt",
            "The rivers",
            "--tokens",
            "8",
            "--device",
            "cpu",
        )
        == ExitCode.OK
    )

    out = capsys.readouterr().out
    assert "tokens/s" in out
    assert "tokens in" in out


def test_json_carries_the_completion_the_sampling_and_the_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """``--ignore-eot`` so the token count is exactly what was asked for.

    Left at the default, an end-of-text token stops generation early -- correctly --
    and the count is then whatever the model happened to do, which is not something to
    assert on.
    """
    run = cli_trained_run.run

    assert (
        cli(
            monkeypatch,
            "chat",
            str(run),
            "--prompt",
            "The rivers",
            "--tokens",
            "8",
            "--ignore-eot",
            "--device",
            "cpu",
            "--json",
        )
        == ExitCode.OK
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["version"] == 1
    assert payload["prompt"] == "The rivers"
    assert payload["tokens"] == 8
    assert payload["tokens_per_second"] > 0
    assert payload["interrupted"] is False
    assert payload["sampling"]["max_new_tokens"] == 8
    assert payload["sampling"]["stop_at_eot"] is False
    assert payload["model"]["context"] == FIXTURE_CONTEXT
    assert payload["model"]["parameters"] > 0
    assert payload["which"] == "best"
    assert any("base language model" in note for note in payload["notes"])
    # The generation ended because --tokens ran out, which is the reason this run pins:
    # nothing else could have stopped it with --ignore-eot and no stop strings.
    assert payload["finish"] == {"reason": "length", "stop": None}


def test_the_token_count_is_tokens_not_printed_fragments(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """A token that completes half a character prints nothing and still counts.

    Byte-level BPE splits multi-byte characters across tokens, so counting what was
    printed would report a tokens/s below the machine's real rate on any non-ASCII
    output. Driven here with a decoder that reports an incomplete character on the
    first token, which is what that looks like from the stream's side.
    """
    from trainai.data.tokenizer import ByteLevelBPE
    from trainai.infer.session import REPLACEMENT_CHAR

    real_decode = ByteLevelBPE.decode

    def flaky_decode(self: Any, ids: Any) -> str:
        ids = list(ids)
        text = real_decode(self, ids)
        return text + REPLACEMENT_CHAR if len(ids) == 1 else text

    monkeypatch.setattr(ByteLevelBPE, "decode", flaky_decode)
    run = cli_trained_run.run

    assert (
        cli(
            monkeypatch,
            "chat",
            str(run),
            "--prompt",
            "The rivers",
            "--tokens",
            "6",
            "--ignore-eot",
            "--device",
            "cpu",
            "--json",
        )
        == ExitCode.OK
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["tokens"] == 6, "the held-back token was still produced"
    assert payload["completion"], "holding a token back must not swallow the text"


def test_a_seed_and_fp32_reproduce_the_sample_exactly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """The promise ``--seed`` makes. fp32 because reduced precision is not bit-stable."""
    run = cli_trained_run.run
    flags = ("--tokens", "8", "--device", "cpu", "--precision", "fp32", "--seed", "3", "--json")

    assert cli(monkeypatch, "chat", str(run), "--prompt", "The rivers", *flags) == ExitCode.OK
    first = json.loads(capsys.readouterr().out)
    assert cli(monkeypatch, "chat", str(run), "--prompt", "The rivers", *flags) == ExitCode.OK
    second = json.loads(capsys.readouterr().out)

    assert first["completion"] == second["completion"]


def test_no_seed_is_not_asserted_to_differ_but_is_recorded_as_unset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """A tiny model can be confident enough to sample the same tokens twice.

    So this checks what is actually guaranteed -- that the report says the seed was
    unset -- rather than asserting two samples differ, which would flake.
    """
    run = cli_trained_run.run

    assert (
        cli(
            monkeypatch,
            "chat",
            str(run),
            "--prompt",
            "The rivers",
            "--tokens",
            "4",
            "--device",
            "cpu",
            "--json",
        )
        == ExitCode.OK
    )

    assert json.loads(capsys.readouterr().out)["sampling"]["seed"] is None


def test_a_piped_prompt_is_used_without_a_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """``echo "Once upon" | trainai chat runs/mine`` -- the scriptable path."""
    run = cli_trained_run.run
    monkeypatch.setattr(sys, "stdin", _FakePipe("Once upon\n"))

    assert (
        cli(monkeypatch, "chat", str(run), "--tokens", "4", "--device", "cpu", "--json")
        == ExitCode.OK
    )

    assert json.loads(capsys.readouterr().out)["prompt"] == "Once upon\n"


def test_a_prompt_longer_than_the_context_says_it_was_truncated(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """The sampler keeps the last ``seq_len`` tokens, silently. This is the sentence."""
    run = cli_trained_run.run
    long_prompt = "The rivers carry what the previous ones did not. " * 20

    assert (
        cli(
            monkeypatch,
            "chat",
            str(run),
            "--prompt",
            long_prompt,
            "--tokens",
            "2",
            "--device",
            "cpu",
        )
        == ExitCode.OK
    )

    out = flat(capsys.readouterr().out)
    assert "this model's context" in out
    assert "Only the last part of it was used" in out


def test_truncation_is_reported_in_json_too(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    run = cli_trained_run.run
    long_prompt = "The rivers carry what the previous ones did not. " * 20

    assert (
        cli(
            monkeypatch,
            "chat",
            str(run),
            "--prompt",
            long_prompt,
            "--tokens",
            "2",
            "--device",
            "cpu",
            "--json",
        )
        == ExitCode.OK
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["prompt_truncated_from"] > FIXTURE_CONTEXT


def test_which_latest_loads_a_different_checkpoint_than_best(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    run = cli_trained_run.run

    assert (
        cli(
            monkeypatch,
            "chat",
            str(run),
            "--prompt",
            "a",
            "--tokens",
            "2",
            "--device",
            "cpu",
            "--which",
            "latest",
            "--json",
        )
        == ExitCode.OK
    )

    assert json.loads(capsys.readouterr().out)["which"] == "latest"


# --------------------------------------------------------------------------- #
# Refusals -- all of them before a checkpoint is loaded
# --------------------------------------------------------------------------- #
def test_json_without_a_prompt_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """The playground prints as it generates, which is not machine-readable."""
    run = cli_trained_run.run
    monkeypatch.setattr(sys, "stdin", _FakeTerminal())

    assert cli(monkeypatch, "chat", str(run), "--json") == ExitCode.USAGE

    assert "--json needs a prompt" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flag", "value", "expected"),
    [
        ("--tokens", "0", "--tokens must be at least 1"),
        ("--temperature", "-1", "--temperature must not be negative"),
        ("--top-p", "2", "--top-p must be above 0 and at most 1"),
        ("--top-p", "0", "--top-p must be above 0 and at most 1"),
        ("--top-k", "0", "--top-k must be at least 1"),
        ("--penalty", "0", "--penalty must be above 0"),
    ],
)
def test_impossible_sampling_is_refused_before_the_model_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_trained_run: Any,
    flag: str,
    value: str,
    expected: str,
) -> None:
    """Knowable from the flags alone, so answered from the flags alone.

    The session is replaced by something that fails loudly: loading a checkpoint onto
    a device before refusing a number would make the user wait to be told about a typo.
    """

    class Exploding:
        @staticmethod
        def open(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("sampling must be checked before the model is loaded")

    monkeypatch.setattr("trainai.cli.chat.InferenceSession", Exploding)
    run = cli_trained_run.run

    assert cli(monkeypatch, "chat", str(run), "--prompt", "a", flag, value) == ExitCode.USAGE

    assert expected in flat(capsys.readouterr().err)


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--precision", "fp64"), ("--device", "gpu")],
)
def test_a_flag_value_outside_its_choices_is_refused_before_the_checkpoint_is_read(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_trained_run: Any,
    flag: str,
    value: str,
) -> None:
    """The end-to-end half of the fix: the CLI declares both of these ``TEXT``.

    Typer validates an ``Enum`` and a ``bool``, so a closed set declared as ``TEXT`` is a
    string the parser waves through, and the check has to be in the library. ``fp64`` used
    to reach ``precision_for``, match no branch, fall through to the ``auto`` arm and get
    reported as whatever ``auto`` chose -- a run asked for a precision that does not exist,
    answering with a plausible one. ``gpu`` used to reach ``torch.device``, which refuses it
    with a bare ``RuntimeError`` naming twenty backends TrainAI cannot train on, and only
    ``TrainAIError`` is rendered without a traceback.

    Loading is what makes this worth an end-to-end test rather than a unit one: both values
    are knowable wrong from the flags alone, so neither should cost a checkpoint read first.
    """

    def exploding(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(f"{flag} must be checked before the checkpoint is read")

    monkeypatch.setattr("trainai.infer.session.load_checkpoint", exploding)
    run = cli_trained_run.run

    assert cli(monkeypatch, "chat", str(run), "--prompt", "a", flag, value) == ExitCode.USAGE

    message = flat(capsys.readouterr().err)
    assert f"Unknown {flag} {value!r}" in message
    assert "Choose one of: auto," in message, "the real values have to be listed"


def test_a_run_that_does_not_exist_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert cli(monkeypatch, "chat", str(tmp_path / "nowhere"), "--prompt", "a") == ExitCode.USAGE

    assert "Nothing at" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# The interactive loop
# --------------------------------------------------------------------------- #
def test_the_banner_says_it_is_a_base_model_before_the_first_prompt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """The command's main hazard: someone types a question and reads it as broken.

    The assertion is deliberately on "continues text" and on the corpus being named as
    what decides the rest. The banner used to assert "nothing in its training data was a
    dialogue", and this test pinned that phrase -- which made the test complicit once
    the repo began shipping 651,448 ``User:``/``Assistant:`` examples in
    ``chat.jsonl``. A checkpoint carries its dataset's ``content_hash``, not its text,
    so the banner cannot know which it got and now says so.
    """
    run = cli_trained_run.run
    feed(monkeypatch, "/exit")

    assert cli(monkeypatch, "chat", str(run), "--device", "cpu") == ExitCode.OK

    out = flat(capsys.readouterr().out)
    assert "base" in out
    assert "continues text" in out
    assert "depends on the corpus" in out
    assert "nothing in its training data was a dialogue" not in out


def test_a_very_small_model_is_said_to_be_small(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """ "It produces fragments" is the size, not a fault -- and the fixture is tiny."""
    from trainai.cli.chat import SMALL_MODEL_PARAMETERS

    run = cli_trained_run.run
    feed(monkeypatch, "/exit")

    assert cli(monkeypatch, "chat", str(run), "--device", "cpu") == ExitCode.OK

    out = flat(capsys.readouterr().out)
    assert "produces fragments" in out, (
        f"the fixture should be under {SMALL_MODEL_PARAMETERS} parameters"
    )


def test_the_banner_says_when_best_fell_back_to_the_last_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_trained_run: Any,
    tmp_path: Path,
) -> None:
    """The one row of the banner that is conditional, and the one about a different model.

    ``--which best`` is the default, and the pointer file is the only record of which
    checkpoint earned it. Without the pointer the session loads the *last* checkpoint --
    a good one, but not the one that was asked for -- and the banner is the only place an
    interactive user could learn that before reading the replies as the best model's.
    ``locate_run`` produces the note; ``tests/test_infer_session.py`` pins its wording.
    This pins that the banner prints it, which it never did under any test.

    The run is copied first: the fixture is session-scoped and shared, and deleting its
    pointer would turn every later test in this file into a test of the fallback.
    """
    copy = tmp_path / "copy"
    shutil.copytree(cli_trained_run.run, copy)
    (copy / "checkpoints" / "checkpoints.json").unlink()
    feed(monkeypatch, "/exit")

    assert cli(monkeypatch, "chat", str(copy), "--device", "cpu") == ExitCode.OK

    out = flat(capsys.readouterr().out)
    assert "Note" in out
    assert "is the last checkpoint, not the one with the lowest validation loss" in out
    assert "does not exist" in out
    # The reason names the pointer by its full path, which is a single token longer than
    # a panel is wide, so the filename is checked with the spaces taken out.
    assert "checkpoints.json" in unwrapped(out)


def test_an_empty_line_neither_generates_nor_exits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    run = cli_trained_run.run
    seen = record_stream(monkeypatch, "text")
    feed(monkeypatch, "", "   ", "/exit")

    assert cli(monkeypatch, "chat", str(run), "--device", "cpu") == ExitCode.OK

    assert seen == [], "a blank line must not reach the model"


def test_a_setting_command_changes_sampling_and_a_typo_only_costs_a_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """A mistyped value in a session must not throw away the loaded model."""
    run = cli_trained_run.run
    record_stream(monkeypatch, "text")
    feed(monkeypatch, "/temp 0.2", "/temp banana", "/nonsense", "/top-k off", "/settings", "/exit")

    assert cli(monkeypatch, "chat", str(run), "--device", "cpu") == ExitCode.OK

    out = flat(capsys.readouterr().out)
    assert "0.2" in out
    assert "needs a number" in out
    assert "Unknown command /nonsense" in out
    assert "off" in out


def test_every_setting_command_reaches_the_field_it_names(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """Seven settings, one dispatch table, and the test above exercised two rows of it.

    The rest are pinned here against the ``/settings`` panel that follows them, one
    value per row, and every value is a number the panel would not otherwise show --
    ``0.35``, ``1.7``, ``41`` -- so a command that silently landed on the wrong field
    could not be covered by a default that happened to match. The ``off`` spelling is
    checked on the two fields that accept it and were not yet checked with it, and
    ``/eot`` has no number at all: it is the one setting whose argument is a switch.

    Sent as one session rather than one per setting, because the panel at the end is a
    single reading of the *accumulated* sampling. That is the property a per-command
    test cannot see: a setting that reset the others to defaults on its way through
    `replace` would still print its own row right.
    """
    run = cli_trained_run.run
    record_stream(monkeypatch, "text")
    feed(
        monkeypatch,
        "/top-p 0.35",
        "/penalty 1.7",
        "/seed 41",
        "/eot off",
        "/settings",
        "/top-p off",
        "/seed off",
        "/eot on",
        "/settings",
        "/exit",
    )

    assert cli(monkeypatch, "chat", str(run), "--device", "cpu") == ExitCode.OK

    out = flat(capsys.readouterr().out)
    # Every setting command prints a "Sampling" panel of its own, so the two readings
    # asked for by name are the fifth and the ninth, not the last two.
    panels = out.split("Sampling")
    assert len(panels) == 10, f"expected nine Sampling panels, got {len(panels) - 1}"
    first, second = panels[5], panels[9]
    assert "Top-p 0.35" in first
    assert "Penalty 1.7" in first
    assert "Seed 41" in first
    assert "End-of-text ignored" in first

    assert "Top-p off" in second
    assert "Seed unset (a new sample each time)" in second
    assert "End-of-text stops generation" in second
    assert "Penalty 1.7" in second, "/top-p off reset a setting it was not about"


def test_a_setting_typed_without_its_value_says_which_one_it_needs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """``/temp`` on its own, which is what someone types to *ask* what the temperature is.

    There is no reading it back that way -- ``/settings`` is the command for that, and the
    line says so -- but the line has to name the command that was typed, because the same
    text answers ``/top-k`` and ``/tokens`` too. The test above covers the wrong *kind* of
    value; this is the missing one, which is a different branch and used to be a different
    sentence with no test on it.
    """
    run = cli_trained_run.run
    record_stream(monkeypatch, "text")
    feed(monkeypatch, "/temp", "/exit")

    assert cli(monkeypatch, "chat", str(run), "--device", "cpu") == ExitCode.OK

    out = flat(capsys.readouterr().out)
    assert "/temp needs a value" in out
    assert "/settings to see the current ones" in out


def test_more_continues_the_last_completion_rather_than_restarting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """The whole reason ``/more`` exists, asserted on the prompt the model received.

    A base model continues text, so continuing its own output is the operation that
    makes sense. Accumulating a transcript instead would imply a dialogue format
    nothing in the training data had.
    """
    run = cli_trained_run.run
    seen = record_stream(monkeypatch, " and then", " the river")
    feed(monkeypatch, "Once upon", "/more", "/exit")

    assert cli(monkeypatch, "chat", str(run), "--device", "cpu") == ExitCode.OK

    assert seen == ["Once upon", "Once upon and then the river"]


def test_more_with_nothing_to_continue_says_so(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    run = cli_trained_run.run
    seen = record_stream(monkeypatch, "text")
    feed(monkeypatch, "/more", "/exit")

    assert cli(monkeypatch, "chat", str(run), "--device", "cpu") == ExitCode.OK

    assert "Nothing to continue yet" in capsys.readouterr().out
    assert seen == []


def test_help_lists_the_commands(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    run = cli_trained_run.run
    feed(monkeypatch, "/help", "/exit")

    assert cli(monkeypatch, "chat", str(run), "--device", "cpu") == ExitCode.OK

    out = capsys.readouterr().out
    for command in ("/more", "/temp", "/top-p", "/seed", "/settings", "/exit"):
        assert command in out


def test_ctrl_c_stops_the_generation_without_leaving(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """Ctrl-C mid-generation is the common case, and must not end the session.

    The tokens produced before it are kept: they were really generated and the timing
    over them is real.
    """
    from trainai.infer import InferenceSession, StreamPiece

    run = cli_trained_run.run

    def interrupting(self: Any, prompt: str, **_kwargs: Any) -> Iterator[StreamPiece]:
        yield StreamPiece("half a ", 1)
        raise KeyboardInterrupt

    monkeypatch.setattr(InferenceSession, "stream_pieces", interrupting)
    feed(monkeypatch, "Once upon", "still here", "/exit")

    assert cli(monkeypatch, "chat", str(run), "--device", "cpu") == ExitCode.OK

    out = flat(capsys.readouterr().out)
    assert out.count("Stopped.") == 2, "both generations were interrupted and both said so"
    assert "1 tokens in" in out, "the token produced before the interrupt is still counted"


def test_ctrl_c_at_the_prompt_does_not_leave(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """It used to leave, contradicting both things chat says about Ctrl-C.

    The test that blessed the old behaviour justified it with "which is what every REPL
    does". That is backwards: ``code.InteractiveConsole.interact`` catches
    ``KeyboardInterrupt``, resets the buffer and continues, and only ``EOFError`` breaks
    its loop. The session is expensive to rebuild, so a single press keeps it.
    """
    run = cli_trained_run.run
    prompts = record_stream(monkeypatch, "text")
    feed(monkeypatch, KeyboardInterrupt, "still here", "/exit")

    assert cli(monkeypatch, "chat", str(run), "--device", "cpu") == ExitCode.OK

    out = flat(capsys.readouterr().out)
    assert "Interrupted." in out
    assert "Bye." not in out, "one Ctrl-C is not an exit"
    assert prompts == ["still here"], "the session survived, and the interrupt generated nothing"


def test_ctrl_c_at_the_prompt_says_what_does_leave(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """People do reach for Ctrl-C to quit, so refusing to quit has to answer them."""
    feed(monkeypatch, KeyboardInterrupt, "/exit")

    assert cli(monkeypatch, "chat", str(cli_trained_run.run), "--device", "cpu") == ExitCode.OK

    out = flat(capsys.readouterr().out)
    assert "Ctrl-C again, Ctrl-D or /exit to leave." in out


def test_ctrl_c_twice_in_a_row_leaves(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """Which is also what stops a terminal that raises on every read from looping.

    The trailing line is what makes this test say something: exhausting ``feed`` raises
    ``EOFError``, which prints "Bye." too, so asserting only on "Bye." would pass even
    if the second press did nothing. Reaching the line at all is the failure.
    """
    prompts = record_stream(monkeypatch, "text")
    feed(monkeypatch, KeyboardInterrupt, KeyboardInterrupt, "must not be reached")

    assert cli(monkeypatch, "chat", str(cli_trained_run.run), "--device", "cpu") == ExitCode.OK

    assert "Bye." in flat(capsys.readouterr().out)
    assert prompts == [], "the second Ctrl-C left, so the line after it was never read"


def test_a_line_between_two_ctrl_cs_does_not_count_as_twice(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """Otherwise every second stray press over a long session ends it."""
    record_stream(monkeypatch, "text")
    feed(monkeypatch, KeyboardInterrupt, "a prompt", KeyboardInterrupt, "/exit")

    assert cli(monkeypatch, "chat", str(cli_trained_run.run), "--device", "cpu") == ExitCode.OK

    out = flat(capsys.readouterr().out)
    assert out.count("Interrupted.") == 2, "both presses were first presses"
    assert "Bye." not in out


def test_ctrl_d_leaves(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """The escape both the banner and /help name first, so it gets its own test."""
    prompts = record_stream(monkeypatch, "text")
    feed(monkeypatch, EOFError, "must not be reached")

    assert cli(monkeypatch, "chat", str(cli_trained_run.run), "--device", "cpu") == ExitCode.OK

    out = flat(capsys.readouterr().out)
    assert "Bye." in out
    assert "Interrupted." not in out, "Ctrl-D leaves on the first press, not the second"
    assert prompts == [], "and it left immediately"


def test_help_describes_both_things_ctrl_c_does(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """The claim that started this: /help promised one half and omitted the other."""
    feed(monkeypatch, "/help", "/exit")

    assert cli(monkeypatch, "chat", str(cli_trained_run.run), "--device", "cpu") == ExitCode.OK

    out = flat(capsys.readouterr().out)
    assert "stop a generation without leaving" in out
    assert "at the prompt, twice to leave" in out
    assert "Ctrl-C stops a generation without leaving." not in out, "the half-truth is gone"


# --------------------------------------------------------------------------- #
# The chat template
#
# Two fixtures carry this section. `cli_chat_run` was prepared with
# --jsonl-messages-field, so its checkpoint records a template; `cli_trained_run` is
# prose, so it records none. Every default in this section is one of those two records
# being read, which is the only way the mode is ever chosen without a flag.
# --------------------------------------------------------------------------- #
def test_a_run_trained_on_conversations_wraps_what_you_type(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_chat_run: Any
) -> None:
    """The prompt the model gets is not the prompt the user gave, and both are reported."""
    assert (
        cli(
            monkeypatch,
            "chat",
            str(cli_chat_run.run),
            "--prompt",
            "Why do the tides turn?",
            "--tokens",
            "4",
            "--device",
            "cpu",
            "--json",
        )
        == ExitCode.OK
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "chat"
    assert payload["prompt"] == "Why do the tides turn?"
    assert payload["model_prompt"] == "User: Why do the tides turn?\nAssistant:"
    assert payload["chat_template"]["version"] == 1
    assert "rendered as conversations" in payload["reason"]
    assert any("model_prompt" in note for note in payload["notes"]), (
        "a machine reader is told what the extra field is"
    )
    assert not any("[bold]" in note for note in payload["notes"]), (
        "the JSON notes are the markup-free ones, not the banner's"
    )


def test_a_run_trained_on_prose_sends_what_you_type_unchanged(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """The negative control. A prose run records no template, so nothing is added.

    Worth asserting rather than assuming: a mode that defaulted the wrong way would put
    ``User:`` in front of every prompt for every model this project has trained until now,
    and the only visible symptom would be slightly worse text.
    """
    assert (
        cli(
            monkeypatch,
            "chat",
            str(cli_trained_run.run),
            "--prompt",
            "The rivers",
            "--tokens",
            "4",
            "--device",
            "cpu",
            "--json",
        )
        == ExitCode.OK
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "raw"
    assert payload["model_prompt"] == "The rivers" == payload["prompt"]
    assert payload["stop"] == []
    assert payload["chat_template"] == {}


def test_raw_switches_the_template_off_for_a_run_that_records_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_chat_run: Any
) -> None:
    """A chat model is still a base model, and probing it raw is a real thing to want."""
    assert (
        cli(
            monkeypatch,
            "chat",
            str(cli_chat_run.run),
            "--prompt",
            "User: Why do the tides turn?",
            "--tokens",
            "4",
            "--device",
            "cpu",
            "--raw",
            "--json",
        )
        == ExitCode.OK
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "raw"
    assert payload["model_prompt"] == "User: Why do the tides turn?"
    assert payload["reason"] == "--raw"
    assert payload["stop"] == []
    # The record is still reported: --raw says what to do, not what is true.
    assert payload["chat_template"]["version"] == 1


def test_chat_wraps_what_you_type_for_a_run_that_records_no_template(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """Why ``--chat`` exists: a corpus flattened by hand trains a chat model silently.

    ``data prepare --jsonl-field text`` over a file of ``User:``/``Assistant:`` text --
    which is exactly what this repo's own ``chat.jsonl`` is if you point at it that way --
    produces a model that learned the layout and a checkpoint that records nothing.
    """
    assert (
        cli(
            monkeypatch,
            "chat",
            str(cli_trained_run.run),
            "--prompt",
            "Why do the tides turn?",
            "--tokens",
            "4",
            "--device",
            "cpu",
            "--chat",
            "--json",
        )
        == ExitCode.OK
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "chat"
    assert payload["model_prompt"] == "User: Why do the tides turn?\nAssistant:"
    assert "records no template" in payload["reason"]


def test_the_stop_strings_reaching_the_sampler_are_the_templates_own_labels(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_chat_run: Any
) -> None:
    """The whole point of 3c-2 arriving in the CLI: they are the template's, not a copy.

    Asserted against :data:`trainai.data.chat.TURN_BOUNDARIES` rather than against a
    literal ``"\\nUser:"``, so that a template whose labels change takes this test with it
    instead of leaving a stale string in a passing suite.
    """
    from trainai.data.chat import TURN_BOUNDARIES

    calls: list[dict[str, Any]] = []
    record_stream(monkeypatch, "a reply", calls=calls)

    assert (
        cli(
            monkeypatch,
            "chat",
            str(cli_chat_run.run),
            "--prompt",
            "Why?",
            "--device",
            "cpu",
            "--json",
        )
        == ExitCode.OK
    )

    assert [call["stop"] for call in calls] == [TURN_BOUNDARIES]
    assert json.loads(capsys.readouterr().out)["stop"] == list(TURN_BOUNDARIES)


def test_no_stop_strings_reach_the_sampler_in_raw_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """Prose has no turns to end, and cutting it at a line starting ``User:`` would be
    a truncation the corpus never asked for."""
    calls: list[dict[str, Any]] = []
    record_stream(monkeypatch, "a continuation", calls=calls)

    assert (
        cli(
            monkeypatch,
            "chat",
            str(cli_trained_run.run),
            "--prompt",
            "The rivers",
            "--device",
            "cpu",
            "--json",
        )
        == ExitCode.OK
    )

    assert [call["stop"] for call in calls] == [()]
    capsys.readouterr()


def test_chat_and_raw_together_are_refused_before_the_model_is_loaded(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """Preferring one silently would hide the one thing the pair exists to make visible."""

    class Exploding:
        @staticmethod
        def open(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("the flags must be checked before the model is loaded")

    monkeypatch.setattr("trainai.cli.chat.InferenceSession", Exploding)

    assert (
        cli(monkeypatch, "chat", str(cli_trained_run.run), "--prompt", "a", "--chat", "--raw")
        == ExitCode.USAGE
    )

    message = flat(capsys.readouterr().err)
    assert "--chat and --raw ask for opposite things" in message
    assert "the checkpoint's own record" in message


def test_an_empty_prompt_in_chat_mode_is_refused_rather_than_asking_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_chat_run: Any
) -> None:
    """A blank message renders as a bare label, and the refusal for it is not a traceback.

    ``ChatFormatError`` is a ``ValueError``, not a ``TrainAIError``, so the conversion at
    this boundary is what stands between the user and a stack trace.
    """
    assert (
        cli(monkeypatch, "chat", str(cli_chat_run.run), "--prompt", "   ", "--device", "cpu")
        == ExitCode.USAGE
    )

    message = flat(capsys.readouterr().err)
    assert "In chat mode the prompt is a message" in message
    assert "--raw" in message


def _with_template(run: Path, tmp_path: Path, template: Any) -> Path:
    """A copy of ``run`` whose every checkpoint records ``template`` as its chat block.

    Copied rather than edited in place: the run fixtures are session-scoped, and a test
    that damaged one would take every later test with it.
    """
    import shutil

    import torch

    copied = tmp_path / "run"
    shutil.copytree(run, copied)
    for path in sorted((copied / "checkpoints").glob("step-*.pt")):
        payload = torch.load(path, weights_only=False)
        payload["dataset"]["chat"] = template
        torch.save(payload, path)
    return copied


def test_a_template_from_another_version_is_refused_rather_than_applied(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_chat_run: Any,
    tmp_path: Path,
) -> None:
    """The mismatch this mechanism exists to catch, and the only one no real run produces.

    A checkpoint recording version 2 was written by a TrainAI that renders something this
    one does not. Wrapping the prompt in version 1 anyway would answer worse and say
    nothing, which is the failure the template module was written to prevent -- so it is
    refused, with both versions named and both ways through it offered.
    """
    run = _with_template(cli_chat_run.run, tmp_path, {"version": 2, "labels": {"user": "Human"}})

    assert cli(monkeypatch, "chat", str(run), "--prompt", "a", "--device", "cpu") == ExitCode.USAGE

    message = flat(capsys.readouterr().err)
    assert "chat template version 2" in message
    assert "renders version 1" in message
    assert "--chat" in message and "--raw" in message


@pytest.mark.parametrize("flag", ["--chat", "--raw"])
def test_either_flag_gets_past_a_version_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_chat_run: Any,
    tmp_path: Path,
    flag: str,
) -> None:
    """A refusal with no way through it is a wall. Both flags are ways through."""
    run = _with_template(cli_chat_run.run, tmp_path, {"version": 2, "labels": {"user": "Human"}})

    assert (
        cli(
            monkeypatch,
            "chat",
            str(run),
            "--prompt",
            "Why?",
            "--tokens",
            "4",
            "--device",
            "cpu",
            flag,
            "--json",
        )
        == ExitCode.OK
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == ("chat" if flag == "--chat" else "raw")
    if flag == "--chat":
        assert payload["model_prompt"] == "User: Why?\nAssistant:"
        assert "overriding the recorded version 2" in payload["reason"]
    else:
        assert payload["model_prompt"] == "Why?"
    # Either way the record itself is reported unchanged, so a script can see the clash.
    assert payload["chat_template"]["version"] == 2


def test_a_dataset_block_that_is_not_an_object_is_treated_as_prose(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_chat_run: Any,
    tmp_path: Path,
) -> None:
    """A hand-edited or third-party checkpoint does not get to crash the command.

    ``InferenceSession.chat_template`` already flattens a non-object block to ``{}``; this
    is the assertion that the CLI's default reads it that way rather than calling ``.get``
    on whatever was in the file.
    """
    run = _with_template(cli_chat_run.run, tmp_path, "version 1")

    assert (
        cli(
            monkeypatch,
            "chat",
            str(run),
            "--prompt",
            "Why?",
            "--tokens",
            "4",
            "--device",
            "cpu",
            "--json",
        )
        == ExitCode.OK
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "raw"
    assert payload["chat_template"] == {}


# --------------------------------------------------------------------------- #
# The chat template, interactively
# --------------------------------------------------------------------------- #
def test_the_banner_names_the_mode_and_says_what_it_does(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_chat_run: Any
) -> None:
    """A default nobody can see is a default nobody can correct.

    The mode is chosen from the checkpoint, without the user having said anything, and it
    changes what the model is asked. So it is on the header next to the device, and the
    base-model warning that would now be advice against this command's own behaviour is
    replaced rather than kept.
    """
    feed(monkeypatch, "/exit")

    assert cli(monkeypatch, "chat", str(cli_chat_run.run), "--device", "cpu") == ExitCode.OK

    out = flat(capsys.readouterr().out)
    assert "Mode" in out
    assert "chat" in out
    assert "rendered as conversations" in out
    assert "sent as a User message" in out
    assert "give it the beginning of something rather than a request" not in out, (
        "that advice contradicts a command that is writing the labels itself"
    )


def test_the_banner_still_warns_about_a_base_model_in_raw_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_chat_run: Any
) -> None:
    """The negative control for the bullet above: ``--raw`` gets the old warning back."""
    feed(monkeypatch, "/exit")

    assert (
        cli(monkeypatch, "chat", str(cli_chat_run.run), "--device", "cpu", "--raw") == ExitCode.OK
    )

    out = flat(capsys.readouterr().out)
    assert "continues text" in out
    assert "depends on the corpus" in out


def test_raw_and_chat_switch_the_mode_mid_session(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_chat_run: Any
) -> None:
    """Asserted on the prompts the model received, which is the only place it shows.

    Switching costs nothing and answers the question a user actually has -- "is it the
    template or is it my model?" -- without unloading the checkpoint to find out.

    The third prompt carries the first exchange, which pins the other half: the
    conversation survives the round trip through raw mode. Clearing it would make probing
    the model raw and coming back a one-way door, and the raw line in the middle adds
    nothing to it -- prose has no turns.
    """
    seen = record_stream(monkeypatch, "a reply")
    feed(monkeypatch, "Why?", "/raw", "Why?", "/chat", "Why?", "/exit")

    assert cli(monkeypatch, "chat", str(cli_chat_run.run), "--device", "cpu") == ExitCode.OK

    assert seen == [
        "User: Why?\nAssistant:",
        "Why?",
        "User: Why?\nAssistant: a reply\n\nUser: Why?\nAssistant:",
    ]


def test_switching_mode_keeps_the_sampling_it_was_switched_with(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_chat_run: Any
) -> None:
    """``/raw`` is not a reset, asserted on the sampler's arguments rather than the console.

    The console shows a temperature the moment ``/temp`` is typed, so reading it back out
    of the output would pass even if the switch dropped it. What the generation after the
    switch was *asked* for is the only place the answer is.
    """
    calls: list[dict[str, Any]] = []
    record_stream(monkeypatch, "a reply", calls=calls)
    feed(monkeypatch, "/temp 0.2", "/raw", "Why?", "/exit")

    assert cli(monkeypatch, "chat", str(cli_chat_run.run), "--device", "cpu") == ExitCode.OK

    assert [call["temperature"] for call in calls] == [0.2]
    assert calls[0]["stop"] == (), "and the switch itself took effect"


def test_settings_shows_the_mode_alongside_the_sampling(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_chat_run: Any
) -> None:
    """Because the mode is a setting, and one nobody typed. ``/settings`` has to answer for
    it in a session where it was never switched, which is why this test does not switch."""
    feed(monkeypatch, "/settings", "/exit")

    assert cli(monkeypatch, "chat", str(cli_chat_run.run), "--device", "cpu") == ExitCode.OK

    out = flat(capsys.readouterr().out)
    assert out.count("wraps what you type") == 2, "once in the banner, once from /settings"


def test_more_continues_the_reply_rather_than_the_turn_after_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_chat_run: Any
) -> None:
    """In chat mode a reply ends where the next label began, and ``/more`` goes back to it.

    Generation stopped at ``\\nUser:`` and the newline before it is kept -- the corpus
    trained on it. Continuing from after that newline would ask the model to write the
    label it was just cut at, which stops immediately and produces nothing at all. So the
    trailing newlines come off before ``/more`` sends the text back.

    The recorded reply starts with a space because a real one does: the prompt stops at
    the colon and the model's first token carries the space, so a fake without it would
    build a continuation that cannot occur.
    """
    seen = record_stream(monkeypatch, " Because the moon pulls.\n")
    feed(monkeypatch, "Why do the tides turn?", "/more", "/exit")

    assert cli(monkeypatch, "chat", str(cli_chat_run.run), "--device", "cpu") == ExitCode.OK

    assert seen == [
        "User: Why do the tides turn?\nAssistant:",
        "User: Why do the tides turn?\nAssistant: Because the moon pulls.",
    ]


def test_more_in_raw_mode_keeps_every_character_the_model_produced(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """The negative control for the trimming above. Prose is continued exactly as it ended.

    A trailing newline in prose is part of the text -- a paragraph break the model chose --
    and trimming it would change what comes next for every prose model in the project.
    """
    seen = record_stream(monkeypatch, " and then the river.\n")
    feed(monkeypatch, "Once upon", "/more", "/exit")

    assert cli(monkeypatch, "chat", str(cli_trained_run.run), "--device", "cpu") == ExitCode.OK

    assert seen == ["Once upon", "Once upon and then the river.\n"]


def test_a_continuation_is_not_wrapped_in_a_second_label(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_chat_run: Any
) -> None:
    """``/more`` sends model text, and model text has been through the template already.

    The bug this pins is the tempting one: render every prompt in one place, including
    this one, and the model is asked to answer its own reply under a fresh ``User:``.
    """
    seen = record_stream(monkeypatch, " Because the moon pulls")
    feed(monkeypatch, "Why?", "/more", "/exit")

    assert cli(monkeypatch, "chat", str(cli_chat_run.run), "--device", "cpu") == ExitCode.OK

    assert seen[1] == "User: Why?\nAssistant: Because the moon pulls"
    assert seen[1].count("User:") == 1


# --------------------------------------------------------------------------- #
# The conversation the loop keeps
# --------------------------------------------------------------------------- #
def test_a_second_question_carries_the_exchange_before_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_chat_run: Any
) -> None:
    """The whole point of chat mode remembering anything, asserted on the second prompt.

    The corpus this template renders is conversations rather than question/answer pairs --
    the typed corpus in ``data/chat-typed`` is 4,000 of 4,000
    ``user``/``assistant``/``user``/``assistant`` -- so a follow-up with the exchange behind
    it is the layout the model was trained on. Asked without it, the model answers a
    question nobody in its training data ever asked in isolation.
    """
    seen = record_stream(monkeypatch, " The moon.")
    feed(monkeypatch, "Why do the tides turn?", "And the wind?", "/exit")

    assert cli(monkeypatch, "chat", str(cli_chat_run.run), "--device", "cpu") == ExitCode.OK

    assert seen[0] == "User: Why do the tides turn?\nAssistant:"
    assert seen[1] == (
        "User: Why do the tides turn?\nAssistant: The moon.\n\nUser: And the wind?\nAssistant:"
    )


def test_the_reply_is_stored_without_the_characters_the_template_wrote(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_chat_run: Any
) -> None:
    """A real reply arrives wrapped in two characters that belong to the layout.

    The prompt stops at the colon, so the model writes the gap itself as part of its first
    token; generation stops at the newline that begins the next label. Store either and the
    renderer writes a second one, which is a prompt with ``Assistant:  `` or a triple
    newline in it -- a layout no corpus contains, invisible in the reply the user reads,
    and wrong only in the *next* prompt. So the fake reply here has both, and the assertion
    is the whole rendered prompt rather than a substring.
    """
    seen = record_stream(monkeypatch, " Because the moon pulls.\n")
    feed(monkeypatch, "Why?", "Really?", "/exit")

    assert cli(monkeypatch, "chat", str(cli_chat_run.run), "--device", "cpu") == ExitCode.OK

    assert seen[1] == (
        "User: Why?\nAssistant: Because the moon pulls.\n\nUser: Really?\nAssistant:"
    )


def test_a_reply_with_nothing_in_it_is_not_remembered(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_chat_run: Any
) -> None:
    """An empty reply is not an empty turn, and the alternative to dropping it is a crash.

    ``trainai.data.chat`` refuses a message with no content -- it renders as a bare label,
    which is a thing to train a model out of rather than into -- so a remembered empty reply
    raises on the *next* prompt, one line away from anything that explains it. The exit code
    is half the assertion here.
    """
    seen = record_stream(monkeypatch, "")
    feed(monkeypatch, "Why?", "And?", "/exit")

    assert cli(monkeypatch, "chat", str(cli_chat_run.run), "--device", "cpu") == ExitCode.OK

    assert seen == ["User: Why?\nAssistant:", "User: And?\nAssistant:"]


def test_new_forgets_the_conversation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_chat_run: Any
) -> None:
    """``/new`` is the way out of a conversation that has gone somewhere useless.

    It says how much it dropped, because a command whose whole effect is invisible is one
    the user cannot tell they typed successfully.
    """
    seen = record_stream(monkeypatch, " a reply")
    feed(monkeypatch, "Why?", "/new", "And?", "/exit")

    assert cli(monkeypatch, "chat", str(cli_chat_run.run), "--device", "cpu") == ExitCode.OK

    assert seen == ["User: Why?\nAssistant:", "User: And?\nAssistant:"]
    out = flat(capsys.readouterr().out)
    assert "2 messages forgotten" in out


def test_new_with_nothing_to_forget_says_so(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_chat_run: Any
) -> None:
    feed(monkeypatch, "/new", "/exit")

    assert cli(monkeypatch, "chat", str(cli_chat_run.run), "--device", "cpu") == ExitCode.OK

    assert "Nothing to forget yet" in flat(capsys.readouterr().out)


def test_more_grows_the_reply_it_continued_rather_than_starting_a_turn(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_chat_run: Any
) -> None:
    """After ``/more`` the conversation holds one longer reply, not two replies in a row.

    Two would render as ``Assistant:`` twice with nothing between them, which is a shape
    the corpus does not have. The continuation keeps its leading space, unlike a fresh
    reply's: this generation started *inside* the content rather than after a label, so the
    space is one the model chose to put between two words.
    """
    seen = record_stream(monkeypatch, " short")
    feed(monkeypatch, "Why?", "/more", "And?", "/exit")

    assert cli(monkeypatch, "chat", str(cli_chat_run.run), "--device", "cpu") == ExitCode.OK

    assert seen[1] == "User: Why?\nAssistant: short"
    assert seen[2] == "User: Why?\nAssistant: short short\n\nUser: And?\nAssistant:"


def scripted_stream(monkeypatch: pytest.MonkeyPatch, *replies: tuple[str, ...]) -> list[str]:
    """Like :func:`record_stream`, but a different script for each generation in turn.

    ``record_stream`` answers every call with the same pieces, which is what most of
    this file needs and exactly what the tests below cannot use: they are about the
    generation *after* a reply, producing nothing where the one before it produced
    something. Past the end of ``replies`` the model produces nothing, which is a real
    thing a model does and the case these tests are about.
    """
    from trainai.infer import InferenceSession, StreamPiece

    seen: list[str] = []
    scripts = iter(replies)

    def fake_stream(self: Any, prompt: str, **_kwargs: Any) -> Iterator[StreamPiece]:
        seen.append(prompt)
        for index, text in enumerate(next(scripts, ()), start=1):
            yield StreamPiece(text, index)

    monkeypatch.setattr(InferenceSession, "stream_pieces", fake_stream)
    return seen


def test_a_reply_the_model_ends_at_once_is_said_to_be_ended_not_failed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_chat_run: Any
) -> None:
    """``/more`` that adds nothing, and why "no tokens produced" cannot be the whole answer.

    Zero tokens in chat mode has one cause worth naming: the first thing the model wrote
    was the next turn's label, or end-of-text. Both mean it considers the reply finished.
    Without the sentence saying so, the only thing on screen is a cost line reporting no
    tokens, which reads like a failure, and the user's next move is to ask why the model
    broke rather than to ask it something else.

    Two more things are pinned on the same session, because both are only observable
    after a generation that produced nothing. The cost line is the zero-token form -- a
    rate over no tokens is not a number worth printing. And the conversation is the one
    that was there before: the prompt the model sees afterwards still ends in the reply
    the continuation was asked to grow, unchanged.

    Two continuations produce nothing here, and they are not the same case. The first
    yields no token, which is what the sentence and the cost line are about. The second
    yields one token that is only whitespace -- and that is the one the conversation's
    guard is for: appending an empty string to a reply changes nothing with or without a
    guard, so a test with only the first would pass against a loop that grew the reply
    by every blank the model ever produced, and the next prompt would carry a trailing
    space no corpus has before its separator.

    The sentence is counted, not just found. The session has two generations that did
    produce tokens, and a sentence that appeared after every chat reply would say the
    model added nothing under a reply it had just printed.
    """
    seen = scripted_stream(monkeypatch, (" a reply",), (), ("  ",), (" more",))
    feed(monkeypatch, "Why?", "/more", "/more", "And?", "/exit")

    assert cli(monkeypatch, "chat", str(cli_chat_run.run), "--device", "cpu") == ExitCode.OK

    out = flat(capsys.readouterr().out)
    assert out.count("ended its turn straight away rather than adding anything") == 1
    assert out.count("no tokens produced") == 1
    assert "0 tokens in" not in out, "the rate line was printed for a generation with no rate"
    assert seen[3] == "User: Why?\nAssistant: a reply\n\nUser: And?\nAssistant:"


def test_a_prose_model_that_produces_nothing_is_not_told_it_ended_a_turn(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """The negative half of the test above: raw mode has no turns to end.

    A prose model that produced nothing hit end-of-text, or a context so full that the
    window slid past everything -- neither is "its turn", and a sentence about turns on a
    model whose banner just said it continues text would contradict the banner. The cost
    line still has to say no tokens were produced, because that part is true in both modes.
    """
    scripted_stream(monkeypatch, ())
    feed(monkeypatch, "Once upon", "/exit")

    assert cli(monkeypatch, "chat", str(cli_trained_run.run), "--device", "cpu") == ExitCode.OK

    out = flat(capsys.readouterr().out)
    assert "no tokens produced" in out
    assert "ended its turn" not in out


def test_more_before_anything_else_says_what_to_do(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_chat_run: Any
) -> None:
    """There is no reply to continue, and in chat mode the way to get one is to ask."""
    seen = record_stream(monkeypatch, " a reply")
    feed(monkeypatch, "/more", "/exit")

    assert cli(monkeypatch, "chat", str(cli_chat_run.run), "--device", "cpu") == ExitCode.OK

    assert seen == [], "nothing was generated"
    out = flat(capsys.readouterr().out)
    assert "Nothing to continue yet" in out
    assert "Ask it something first" in out


def test_raw_mode_remembers_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """The negative control for all of the above, on a run with no chat template.

    Prose has no turns, and a transcript accumulated in front of it would imply a dialogue
    format a prose-trained model has never seen -- so each line is a fresh completion, as
    it always was. A guard accidentally inverted fails here rather than quietly changing
    what every prose model is prompted with.
    """
    seen = record_stream(monkeypatch, " and then")
    feed(monkeypatch, "The rivers", "The mountains", "/exit")

    assert cli(monkeypatch, "chat", str(cli_trained_run.run), "--device", "cpu") == ExitCode.OK

    assert seen == ["The rivers", "The mountains"]


def test_new_in_raw_mode_forgets_what_more_would_continue(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """``/new`` means the same thing in both modes, so it clears the raw text too."""
    record_stream(monkeypatch, " and then")
    feed(monkeypatch, "The rivers", "/new", "/more", "/exit")

    assert cli(monkeypatch, "chat", str(cli_trained_run.run), "--device", "cpu") == ExitCode.OK

    out = flat(capsys.readouterr().out)
    assert "the last completion forgotten" in out
    assert "Nothing to continue yet" in out


def test_help_lists_both_mode_commands(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_chat_run: Any
) -> None:
    feed(monkeypatch, "/help", "/exit")

    assert cli(monkeypatch, "chat", str(cli_chat_run.run), "--device", "cpu") == ExitCode.OK

    out = flat(capsys.readouterr().out)
    assert "/chat" in out
    assert "/raw" in out
    assert "/new" in out


def test_truncation_is_measured_on_the_prompt_the_model_is_given(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_chat_run: Any
) -> None:
    """The labels are part of the prompt, so a line that fits can still not fit.

    Measured on what was typed, a chat prompt is undercounted by however many tokens the
    labels cost -- and the window where that matters is exactly the window where the
    warning is the only thing standing between the user and a silent truncation. The
    prompt here is grown against the run's own tokenizer until it fills the context
    exactly, so the labels are the whole overflow.
    """
    from trainai.data.chat import render_prompt
    from trainai.data.tokenizer import ByteLevelBPE

    tokenizer = ByteLevelBPE.load(cli_chat_run.run / "tokenizer.json")
    text = "rivers"
    while len(tokenizer.encode(f"{text} rivers")) <= FIXTURE_CONTEXT:
        text = f"{text} rivers"

    assert len(tokenizer.encode(text)) <= FIXTURE_CONTEXT
    rendered = len(tokenizer.encode(render_prompt([{"role": "user", "content": text}])))
    assert rendered > FIXTURE_CONTEXT

    record_stream(monkeypatch, " ok")
    feed(monkeypatch, text, "/exit")

    assert cli(monkeypatch, "chat", str(cli_chat_run.run), "--device", "cpu") == ExitCode.OK

    out = flat(capsys.readouterr().out)
    assert "this model's context" in out
    assert "Only the last part of it was used" in out


def test_a_conversation_too_long_for_the_context_loses_whole_messages(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_chat_run: Any
) -> None:
    """The sampler's own truncation would cut by token count, wherever that lands.

    Mid-word, inside a message, leaving a prompt whose first label is half a label -- a
    layout no corpus contains, and one that still produces a reply, so nothing about it
    looks wrong. Dropping whole messages instead keeps every prompt a conversation.

    The question is grown against the run's own tokenizer until one exchange plus a repeat
    of it overflows the context while one exchange alone still fits, and both halves of
    that are asserted: a setup that stopped being true would otherwise pass this test for
    the wrong reason. The expected prompt is what the renderer produces for the second
    question alone, which pins both that the first exchange went and that what is left is
    a well-formed prompt rather than a fragment.
    """
    from trainai.data.chat import REPLY_ROLE, render_prompt
    from trainai.data.tokenizer import ByteLevelBPE

    tokenizer = ByteLevelBPE.load(cli_chat_run.run / "tokenizer.json")
    reply = "the moon"
    question = "why"
    while len(tokenizer.encode(render_prompt([{"role": "user", "content": question}]))) * 2 <= (
        FIXTURE_CONTEXT
    ):
        question = f"{question} tides"

    alone = render_prompt([{"role": "user", "content": question}])
    both = render_prompt(
        [
            {"role": "user", "content": question},
            {"role": REPLY_ROLE, "content": reply},
            {"role": "user", "content": question},
        ]
    )
    assert len(tokenizer.encode(alone)) <= FIXTURE_CONTEXT, "one question has to fit"
    assert len(tokenizer.encode(both)) > FIXTURE_CONTEXT, "two exchanges must not"

    seen = record_stream(monkeypatch, f" {reply}")
    feed(monkeypatch, question, question, "/exit")

    assert cli(monkeypatch, "chat", str(cli_chat_run.run), "--device", "cpu") == ExitCode.OK

    assert seen == [alone, alone]
    out = flat(capsys.readouterr().out)
    assert "no longer fits this model's context" in out
    assert "oldest 2 messages were left out" in out
    assert "Only the last part of it was used" not in out, "trimming replaced truncation"


# --------------------------------------------------------------------------- #
# Why a reply stopped
#
# Three endings look the same on a terminal: the model finished, it started writing
# somebody else's turn, or --tokens ran out mid-sentence. Only the last one is worth
# acting on, so only the last one is said out loud -- and --json carries all three.
# --------------------------------------------------------------------------- #
def scripted_decode(monkeypatch: pytest.MonkeyPatch, *chunks: str) -> None:
    """Make the real session decode ``chunks``, one more of them per token produced.

    Patched at the tokenizer rather than at ``stream_pieces``, so everything this section
    is about stays the shipped code: the stop-string scan, the hold-back and the finish
    all run for real, over text a test can name. ``stream_pieces`` decodes the whole
    continuation after every token and yields what is new, so the fake has to grow with
    the token list rather than answer per token.

    Past the end of ``chunks`` the text stops growing, which is a real thing a token can
    do -- one that completes no character adds nothing -- so nothing here needs the script
    and the token limit to be the same length.
    """
    from trainai.data.tokenizer import ByteLevelBPE

    def fake_decode(self: Any, ids: Any) -> str:
        return "".join(chunks[: len(list(ids))])

    monkeypatch.setattr(ByteLevelBPE, "decode", fake_decode)


def _user_boundary() -> str:
    """The label a chat model writes when it moves on to the next turn.

    Built from the template's own labels and checked against its own stop strings, so a
    rename takes these tests with it instead of leaving a stale ``"\\nUser:"`` behind in a
    suite that still passes.
    """
    from trainai.data.chat import ROLE_LABELS, TURN_BOUNDARIES

    boundary = f"\n{ROLE_LABELS['user']}:"
    assert boundary in TURN_BOUNDARIES, "the label these tests script has to be one chat stops at"
    return boundary


def test_json_names_the_turn_boundary_a_chat_reply_ended_at(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_chat_run: Any
) -> None:
    """The gap the previous commit left open: ``--json`` could not tell a reply that
    finished from one that hit ``--tokens``.

    ``--ignore-eot`` leaves the stop strings as the only thing that can end this generation
    early, and the script puts one at the third token, so the reason, the string that
    matched and a completion that stops cleanly in front of it are all pinned against a
    real session doing its own matching.
    """
    from trainai.data.chat import TURN_BOUNDARIES

    boundary = _user_boundary()
    scripted_decode(monkeypatch, " the", " tides", f"{boundary} and on")

    assert (
        cli(
            monkeypatch,
            "chat",
            str(cli_chat_run.run),
            "--prompt",
            "Why?",
            "--tokens",
            "8",
            "--ignore-eot",
            "--device",
            "cpu",
            "--json",
        )
        == ExitCode.OK
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["finish"] == {"reason": "stop", "stop": boundary}
    assert payload["completion"] == " the tides", "and the label itself was never emitted"
    assert payload["tokens"] == 3, "the token that produced the match still cost what it cost"
    assert payload["stop"] == list(TURN_BOUNDARIES), (
        "the configured list is still the top-level key, which is why the finish is nested"
    )


def test_a_reply_cut_at_the_token_limit_says_so_and_names_the_flag_that_raises_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """No fake anywhere in this one. A prose run has no stop strings, ``--ignore-eot``
    leaves no end-of-text, and four tokens is then four tokens: the limit is the only thing
    that can have ended it.

    The number is in the message because the fix is a bigger one, and the flag is in it
    because a user who has just been told their reply is unfinished should not have to go
    looking for the way to finish it.
    """
    assert (
        cli(
            monkeypatch,
            "chat",
            str(cli_trained_run.run),
            "--prompt",
            "The rivers",
            "--tokens",
            "4",
            "--ignore-eot",
            "--device",
            "cpu",
        )
        == ExitCode.OK
    )

    out = flat(capsys.readouterr().out)
    assert "still writing at the 4-token limit" in out
    assert "this reply is unfinished" in out
    assert "--tokens raises the limit" in out
    assert "/more" not in out, "there is no next line to type in a one-shot run"


def test_a_reply_that_ended_at_a_turn_boundary_says_nothing_about_the_limit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_chat_run: Any
) -> None:
    """The negative control, and the reason the note is conditional at all.

    A model that stopped because it had finished its turn is not unfinished, and a line
    hinting that it might be would then sit under every well-formed reply this project
    trains for -- which is as useless as no line at all. The cost line is asserted too, so
    a run that failed for some unrelated reason cannot pass this by printing nothing.
    """
    scripted_decode(monkeypatch, " the", " tides", f"{_user_boundary()} and on")

    assert (
        cli(
            monkeypatch,
            "chat",
            str(cli_chat_run.run),
            "--prompt",
            "Why?",
            "--tokens",
            "8",
            "--ignore-eot",
            "--device",
            "cpu",
        )
        == ExitCode.OK
    )

    out = flat(capsys.readouterr().out)
    assert "unfinished" not in out
    assert "3 tokens in" in out, "so the absence above is the note's, not the whole run's"


def test_the_interactive_note_offers_more_and_names_the_limit_in_force(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """Interactively the fix is a command rather than a flag, and ``/more`` is the cheaper
    half of it: continuing costs the tokens it adds, re-running at a higher limit costs the
    ones already paid for again.

    Asserted through a *raised* limit on purpose. Reading the default back would pass just
    as well if the note were built from the sampling the session started with instead of
    the sampling it is on, which is a real way to get this wrong and an invisible one.
    """
    from trainai.infer import Finish

    record_stream(monkeypatch, "half a sen", finish=Finish("length"))
    feed(monkeypatch, "/tokens 12", "Once upon", "/exit")

    assert cli(monkeypatch, "chat", str(cli_trained_run.run), "--device", "cpu") == ExitCode.OK

    out = flat(capsys.readouterr().out)
    assert "still writing at the 12-token limit" in out
    assert "/more continues it" in out
    assert "/tokens raises the limit" in out


def test_an_interrupted_reply_is_not_called_unfinished(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """Ctrl-C already said why the text stopped, and "Stopped." is the truer word for it.

    The fake hands over a finish and *then* interrupts, which is the only shape that tests
    the guard rather than shadowing it: an interrupt earlier in the stream never reaches a
    finish at all, so a version of the note that had forgotten about interruption entirely
    would pass a test written the obvious way.
    """
    from trainai.infer import Finish, InferenceSession, StreamPiece

    def interrupted(self: Any, prompt: str, **_kwargs: Any) -> Iterator[StreamPiece]:
        yield StreamPiece("half a ", 1)
        yield StreamPiece("", 1, Finish("length"))
        raise KeyboardInterrupt

    monkeypatch.setattr(InferenceSession, "stream_pieces", interrupted)
    feed(monkeypatch, "Once upon", "/exit")

    assert cli(monkeypatch, "chat", str(cli_trained_run.run), "--device", "cpu") == ExitCode.OK

    out = flat(capsys.readouterr().out)
    assert "Stopped." in out
    assert "unfinished" not in out


def test_json_reports_no_reason_at_all_for_a_generation_nothing_ended(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """``null`` is the honest answer for an abandoned stream, and the pair is the point.

    An interrupted reply and one cut at the limit are both short text, and a script has no
    other way to tell them apart -- ``interrupted`` true with ``finish`` null is one,
    ``interrupted`` false with a ``"length"`` finish is the other. Inventing a reason for
    the first would make the two indistinguishable again in the one place built to
    distinguish them.
    """
    from trainai.infer import InferenceSession, StreamPiece

    def interrupting(self: Any, prompt: str, **_kwargs: Any) -> Iterator[StreamPiece]:
        yield StreamPiece("half a ", 1)
        raise KeyboardInterrupt

    monkeypatch.setattr(InferenceSession, "stream_pieces", interrupting)

    assert (
        cli(
            monkeypatch,
            "chat",
            str(cli_trained_run.run),
            "--prompt",
            "Once upon",
            "--device",
            "cpu",
            "--json",
        )
        == ExitCode.OK
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["interrupted"] is True
    assert payload["finish"] is None
    assert payload["tokens"] == 1, "and what was produced before the interrupt is still counted"


# --------------------------------------------------------------------------- #
# The CLI's literal defaults must not drift from the library's
# --------------------------------------------------------------------------- #
def test_chat_flag_defaults_match_the_library() -> None:
    """``main.py`` spells its defaults out so ``--help`` does not import torch."""
    import typer.main

    from trainai import infer
    from trainai.cli._click import is_group

    expected = {
        "--which": infer.DEFAULT_WHICH,
        "--tokens": infer.DEFAULT_MAX_NEW_TOKENS,
        "--temperature": infer.DEFAULT_TEMPERATURE,
        "--top-k": infer.DEFAULT_TOP_K,
        "--top-p": infer.DEFAULT_TOP_P,
        "--penalty": infer.DEFAULT_REPETITION_PENALTY,
    }

    group = typer.main.get_command(app)
    assert is_group(group)
    command = group.commands["chat"]

    for flag, want in expected.items():
        param = next(p for p in command.params if flag in p.opts)
        found = param.default.value if hasattr(param.default, "value") else param.default
        assert found == want, f"chat {flag}: CLI has {found!r}, library {want!r}"


# --------------------------------------------------------------------------- #
# The window that slides while the reply is being written
# --------------------------------------------------------------------------- #
class _Fits:
    """A session stub with only what `_slides_after` reads: a context and an encoder."""

    class _Config:
        seq_len = 100

    class _Tokenizer:
        def __init__(self, length: int) -> None:
            self.length = length

        def encode(self, text: str) -> list[int]:
            """The length is what matters; every prompt of it slides the same way."""
            return [0] * self.length

    def __init__(self, prompt_tokens: int) -> None:
        self.model_config = self._Config()
        self.tokenizer = self._Tokenizer(prompt_tokens)


@pytest.mark.parametrize(
    ("prompt_tokens", "produced", "expected"),
    [
        (40, 60, None),  # the reply ended exactly as the context filled: nothing was lost
        (40, 61, 60),  # one token past it, so the front went for that last one
        (40, 10, None),  # the ordinary case, and the one that must stay silent
        (99, 200, 1),  # a full prompt: one clean token, then it erodes
        (100, 200, 0),  # a prompt filling the context slides from the first token
        (101, 200, None),  # did not fit at all -- that is _truncation's sentence
    ],
)
def test_when_the_window_starts_dropping_the_front_of_the_prompt(
    prompt_tokens: int, produced: int, expected: int | None
) -> None:
    """The arithmetic, at its boundaries, without loading a model to get at it.

    ``room >= produced`` rather than ``>`` is the one worth pinning: a reply that ends
    exactly as the context fills has never lost a token, so there is nothing to say. The
    off-by-one matters because it is the difference between a note on every generation
    that happens to reach the context and a note only when something was actually dropped.
    """
    from trainai.cli.chat import _slides_after

    found = _slides_after(_Fits(prompt_tokens), "irrelevant", produced)  # type: ignore[arg-type]
    assert found == expected


def _room(run: Any, prompt: str) -> int:
    """Tokens a reply gets beside *prompt* before the window starts dropping its front."""
    from trainai.data.tokenizer import ByteLevelBPE

    tokenizer = ByteLevelBPE.load(run / "tokenizer.json")
    return FIXTURE_CONTEXT - len(tokenizer.encode(prompt))


def _prompt_leaving(run: Any, gap: int) -> tuple[str, int]:
    """A prompt that fits the context with at most *gap* tokens to spare, and that gap.

    Grown against the run's own tokenizer rather than counted in words: the gap is a
    property of the prompt and the tokenizer together, and a test that guessed it would
    pass or fail on the fixture's vocabulary rather than on the code.
    """
    prompt = "rivers"
    while _room(run, f"{prompt} rivers") >= gap:
        prompt = f"{prompt} rivers"
    room = _room(run, prompt)
    assert room > 0, f"the prompt has to fit the context, and it leaves {room}"
    return prompt, room


def test_a_reply_that_outgrew_its_context_says_how_much_of_it_flew_blind(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """``--tokens`` above the context is the default case on a small model, not a corner.

    ``chat --tokens`` defaults to 200 and this fixture's context is 64, so a reply that
    runs to the limit spends its last 136 tokens continuing nothing but itself. That is
    invisible on screen -- the model keeps producing fluent-looking text -- so it is a
    sentence rather than nothing.

    Generation is stubbed to a known token count so the two figures in the sentence are
    checked against arithmetic rather than against whatever this fixture's undertrained
    model happens to emit before EOT. The prompt is measured with the run's own tokenizer
    for the same reason: ``room`` is a property of the pair, not of the word count.
    """
    run = cli_trained_run.run
    prompt, room = _prompt_leaving(run, 6)
    blind = 6

    record_stream(monkeypatch, *[" tide"] * (room + blind))
    assert (
        cli(monkeypatch, "chat", str(run), "--prompt", prompt, "--tokens", "200", "--device", "cpu")
        == ExitCode.OK
    )

    out = flat(capsys.readouterr().out)
    assert f"The window slid after {room} tokens" in out
    assert f"the last {blind} were written without the start of the prompt" in out
    assert f"context is {FIXTURE_CONTEXT} tokens" in out


def test_a_reply_that_stopped_inside_the_context_says_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """The negative control, and the reason the note is printed after the reply.

    ``--tokens 200`` here exceeds the context by 136, so a note predicted from the budget
    would fire -- but the reply stops one token short of the gap and never loses anything.
    A note on every generation is a note nobody reads.
    """
    run = cli_trained_run.run
    prompt, room = _prompt_leaving(run, 6)
    assert room > 1, "there has to be a gap for a reply to stop inside"

    record_stream(monkeypatch, *[" tide"] * (room - 1))
    assert (
        cli(monkeypatch, "chat", str(run), "--prompt", prompt, "--tokens", "200", "--device", "cpu")
        == ExitCode.OK
    )

    out = flat(capsys.readouterr().out)
    assert "The window slid" not in out
    assert "without the start of the prompt" not in out


def test_a_prompt_that_never_fitted_gets_one_sentence_rather_than_two(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """Both conditions hold at once here, and only the older one has anything to add.

    A prompt past the context was cut *before* generation, so its front is already gone;
    saying it will now be dropped during generation describes the same loss twice.
    """
    run = cli_trained_run.run
    long_prompt = "The rivers carry what the previous ones did not. " * 20

    assert (
        cli(
            monkeypatch,
            "chat",
            str(run),
            "--prompt",
            long_prompt,
            "--tokens",
            "80",
            "--device",
            "cpu",
        )
        == ExitCode.OK
    )

    out = flat(capsys.readouterr().out)
    assert "Only the last part of it was used" in out
    assert "The window slid" not in out


def test_the_sliding_window_is_reported_in_json_too(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """A machine reader gets the number, for the same reason it gets the truncation."""
    run = cli_trained_run.run
    prompt, room = _prompt_leaving(run, 6)

    record_stream(monkeypatch, *[" tide"] * (room + 6))
    assert (
        cli(
            monkeypatch,
            "chat",
            str(run),
            "--prompt",
            prompt,
            "--tokens",
            "200",
            "--device",
            "cpu",
            "--json",
        )
        == ExitCode.OK
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["prompt_truncated_from"] is None
    assert payload["window_slides_after"] == room
    assert payload["tokens"] - payload["window_slides_after"] == 6


def test_the_real_sampler_slides_where_the_arithmetic_says_it_will(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """Unstubbed, against the real model: the note is about generation, not about a stub.

    Every other test here replaces the stream, which pins the message and the payload but
    would keep passing if ``generate_stream`` stopped sliding at all. This one asks for more
    tokens than the gap the prompt leaves and checks the figure the real sampler produced,
    so the two halves stay tied together.
    """
    run = cli_trained_run.run
    prompt, room = _prompt_leaving(run, 3)

    assert (
        cli(
            monkeypatch,
            "chat",
            str(run),
            "--prompt",
            prompt,
            "--tokens",
            str(room + 8),
            "--device",
            "cpu",
            "--seed",
            "0",
            "--json",
        )
        == ExitCode.OK
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["tokens"] > room, "the reply has to outgrow the gap for there to be a slide"
    assert payload["window_slides_after"] == room
