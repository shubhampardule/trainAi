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
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from conftest import flat
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


def record_stream(monkeypatch: pytest.MonkeyPatch, *pieces: str) -> list[str]:
    """Replace generation with ``pieces``, and return the list of prompts it saw."""
    from trainai.infer import InferenceSession, StreamPiece

    seen: list[str] = []

    def fake_stream(self: Any, prompt: str, **_kwargs: Any) -> Iterator[StreamPiece]:
        seen.append(prompt)
        for index, text in enumerate(pieces, start=1):
            yield StreamPiece(text, index)

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
    """The command's main hazard: someone types a question and reads it as broken."""
    run = cli_trained_run.run
    feed(monkeypatch, "/exit")

    assert cli(monkeypatch, "chat", str(run), "--device", "cpu") == ExitCode.OK

    out = flat(capsys.readouterr().out)
    assert "base" in out
    assert "continues text; it does not answer questions" in out


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
