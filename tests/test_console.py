"""Tests for unit formatting and console safety.

These look like trivial string helpers, but they are the project's only defence
against reporting memory in the wrong units, and against Windows consoles
mangling output. Both have already bitten during development.
"""

from __future__ import annotations

import io

import pytest

from conftest import flat
from trainai.console import (
    fmt_bytes,
    fmt_count,
    fmt_duration,
    fmt_int,
    fmt_params,
    print_command,
    render_error,
    supports_unicode,
)
from trainai.errors import DatasetEmptyError

GIB = 1024**3
MIB = 1024**2


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0 B"),
        (512, "512 B"),
        (1023, "1023 B"),
        (1024, "1.00 KiB"),
        (int(1.48 * GIB), "1.48 GiB"),
        (4 * GIB, "4.00 GiB"),
        (int(15.4 * GIB), "15.4 GiB"),
        (789 * MIB, "789 MiB"),
    ],
)
def test_fmt_bytes_uses_binary_units(value: int, expected: str) -> None:
    assert fmt_bytes(value) == expected


def test_fmt_bytes_handles_negative_deltas() -> None:
    # Memory *deltas* can be negative; formatting must not crash or lie.
    assert fmt_bytes(-1024) == "-1.00 KiB"


def test_fmt_bytes_never_uses_decimal_units() -> None:
    """1 GiB must not be reported as 1.07 GB. Mixing units would corrupt planning."""
    assert fmt_bytes(GIB) == "1.00 GiB"
    assert "GB" not in fmt_bytes(GIB)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.5, "500ms"),
        (1, "1s"),
        (45, "45s"),
        (90, "1m 30s"),
        (4560, "1h 16m"),
        (374400, "4d 8h"),
    ],
)
def test_fmt_duration(seconds: float, expected: str) -> None:
    assert fmt_duration(seconds) == expected


def test_fmt_duration_handles_non_finite_and_negative() -> None:
    assert fmt_duration(float("nan")) == "unknown"
    assert fmt_duration(float("inf")) == "unknown"
    assert fmt_duration(-5) == "0s"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (13_900_000, "13.9M"),
        (110_500_000, "110.5M"),
        (1_500_000_000, "1.50B"),
        (2048, "2.0K"),
        (42, "42"),
    ],
)
def test_fmt_params(value: int, expected: str) -> None:
    assert fmt_params(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (999, "999"),
        (1500, "1.50K"),
        (280_000_000, "280M"),
        (2_200_000_000, "2.20B"),
    ],
)
def test_fmt_count(value: int, expected: str) -> None:
    assert fmt_count(value) == expected


def test_fmt_int_uses_thousands_separators() -> None:
    assert fmt_int(61595) == "61,595"
    assert fmt_int(1304.4) == "1,304"


class _FakeStream(io.StringIO):
    def __init__(self, encoding: str) -> None:
        super().__init__()
        self._encoding = encoding

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return self._encoding


@pytest.mark.parametrize("encoding", ["utf-8", "UTF-8", "utf8", "utf-16"])
def test_supports_unicode_accepts_utf_encodings(encoding: str) -> None:
    assert supports_unicode(_FakeStream(encoding)) is True


@pytest.mark.parametrize("encoding", ["cp1252", "cp437", "latin-1", "ascii", ""])
def test_supports_unicode_rejects_legacy_codepages(encoding: str) -> None:
    """cp1252 can encode U+2022, but terminals then mis-decode it. Reject anyway."""
    assert supports_unicode(_FakeStream(encoding)) is False


def test_render_error_prints_message_and_hint(capsys: pytest.CaptureFixture[str]) -> None:
    render_error(
        DatasetEmptyError(
            "corpus/ contained no readable text.",
            hint="Check that the folder has .txt or .jsonl files.",
        )
    )
    error = flat(capsys.readouterr().err)
    assert "no readable text" in error
    assert "What to do" in error
    assert ".jsonl" in error
    assert "Traceback" not in error


def test_render_error_hides_details_unless_asked(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exc = DatasetEmptyError("empty", hint="add files", details={"scanned_paths": 17})
    render_error(exc, show_details=False)
    assert "scanned_paths" not in capsys.readouterr().err

    render_error(exc, show_details=True)
    assert "scanned_paths" in capsys.readouterr().err


def test_the_fallback_spinner_frames_are_pure_ascii() -> None:
    """Regression: a Braille spinner crashes a cp1252 Windows console mid-command.

    Rich downgrades its own box-drawing and progress-bar glyphs when it detects a
    legacy Windows console, but not spinner frames. Its default "dots" spinner is
    U+280x, so `trainai data prepare` in a cp1252 console raised
    UnicodeEncodeError -- after the tokenizer had already started training.
    """
    from rich.spinner import Spinner

    frames = Spinner("line").frames
    assert frames
    for frame in frames:
        frame.encode("ascii")  # raises UnicodeEncodeError if this ever changes


def test_the_spinner_is_chosen_from_what_the_stream_can_encode() -> None:
    from trainai import console

    expected = "dots" if console.supports_unicode() else "line"
    assert expected == console.SPINNER


# --------------------------------------------------------------------------- #
# Commands the user is meant to copy
# --------------------------------------------------------------------------- #
_LONG_COMMAND = (
    "trainai train --data data/shake --preset tiny --context 256 --seq-len 256 "
    "--steps 75 --batch-size 64 --lr 0.000424 --eval-every 5 --eval-batches 1 "
    "--checkpoint-every 10"
)


@pytest.mark.parametrize("width", [40, 60, 80, 100, 120, 200])
def test_a_printed_command_stays_on_one_line_at_any_width(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    """A wrapped command is not a command, and this one is 150 characters long.

    Found by running the documented quickstart end to end: `trainai plan` printed the
    `trainai train` command it recommends through plain `console.print`, so at a normal
    terminal width Rich broke it across three lines. Copying that gives the shell one
    complete command and then a line starting `--steps`, which is not a command at all.

    Swept across widths rather than asserted at one, because the failure is invisible
    at whatever width the author happens to use -- the same reason the console-width
    sweep exists for the report tables.
    """
    _narrow_console(monkeypatch, width)

    print_command(_LONG_COMMAND)

    printed = capsys.readouterr().out
    lines = [line for line in printed.splitlines() if line.strip()]
    assert len(lines) == 1, f"command was split across {len(lines)} lines at COLUMNS={width}"
    assert lines[0].strip() == _LONG_COMMAND


def test_a_printed_command_is_not_cropped_to_the_console_width(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Turning wrapping off is not enough: Rich crops instead unless told not to.

    A cropped command is worse than a wrapped one. Wrapping is visibly broken; cropping
    hands back a shorter command that still runs, with the last few flags silently gone.
    """
    _narrow_console(monkeypatch, 40)

    print_command(_LONG_COMMAND)

    assert capsys.readouterr().out.strip().endswith("--checkpoint-every 10")


def _narrow_console(monkeypatch: pytest.MonkeyPatch, width: int) -> None:
    """Swap in a `Console` of a fixed width, rather than reloading the module.

    Rich fixes a console's width when the object is constructed, so setting `COLUMNS`
    after import changes nothing, and reloading `trainai.console` would leave every
    other module holding the old object while later tests picked up one permanently
    stuck at this width. `monkeypatch.setattr` is undone at the end of the test.
    """
    from rich.console import Console

    monkeypatch.setattr("trainai.console.console", Console(width=width))
