"""Terminal output: a shared Rich console, error rendering, and unit formatting.

Formatting lives here rather than being scattered inline, because consistent
units are a correctness concern in this project. TrainAI reports memory in
binary units (GiB) to match what ``nvidia-smi`` and ``torch.cuda`` actually
report, and never silently mixes them with decimal GB.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from trainai.errors import TrainAIError

__all__ = [
    "BULLET",
    "DASH",
    "ELLIPSIS",
    "SPINNER",
    "console",
    "emit_json",
    "err_console",
    "fmt_bytes",
    "fmt_count",
    "fmt_duration",
    "fmt_int",
    "fmt_params",
    "print_bullets",
    "print_command",
    "print_kv",
    "printable",
    "render_error",
    "rule",
    "supports_unicode",
]


def supports_unicode(stream: Any = None) -> bool:
    """Whether it is safe to print non-ASCII decoration to ``stream``.

    Only UTF encodings qualify. Legacy Windows code pages such as cp1252 are
    excluded even though several of them *can* encode characters like U+2022:
    the bytes survive the encode step but are then mis-decoded by terminals and
    pipes that assume UTF-8, which is how you get a screen full of ``?``.

    Rich already downgrades its own box-drawing characters on legacy consoles.
    This covers the literal glyphs *we* write.
    """
    stream = stream if stream is not None else sys.stdout
    encoding = (getattr(stream, "encoding", None) or "").lower().replace("_", "-")
    return encoding.startswith("utf")


_UNICODE_OK = supports_unicode()

#: List bullet. ASCII on legacy consoles.
BULLET = "•" if _UNICODE_OK else "*"
#: Em dash used between a label and its explanation.
DASH = "—" if _UNICODE_OK else "-"
#: Truncation marker.
ELLIPSIS = "…" if _UNICODE_OK else "..."

#: Spinner for Rich progress displays.
#:
#: Rich's default "dots" spinner draws Braille characters (U+280x). Rich downgrades
#: its *own* box-drawing and progress-bar glyphs on a legacy Windows console, but
#: not spinner frames, so on a cp1252 console the animation raises
#: ``UnicodeEncodeError`` part-way through a long command -- after the work has
#: already started. "line" is ``- \ | /``, which any code page can encode.
SPINNER = "dots" if _UNICODE_OK else "line"

#: Normal output. Goes to stdout so it can be piped.
console = Console(
    highlight=False,
    soft_wrap=False,
    # Respect CI and dumb terminals; Rich already honours NO_COLOR / TERM=dumb.
    force_terminal=None if os.environ.get("TRAINAI_FORCE_COLOR") is None else True,
)

#: Diagnostics and errors. Separate console so stdout stays machine-readable.
err_console = Console(stderr=True, highlight=False)


# --------------------------------------------------------------------------- #
# Output that has to survive a legacy console
# --------------------------------------------------------------------------- #
def printable(text: str, stream: Any = None) -> str:
    """``text`` reduced to characters ``stream`` can actually encode.

    For text this project did not write: a model's own output. A byte-level tokenizer
    can emit a byte sequence that never completes a character, which decodes to U+FFFD
    -- and cp1252, the default on a Windows console, cannot encode U+FFFD. Writing it
    raises ``UnicodeEncodeError`` from inside Rich's writer, mid-stream, after tokens
    have already been printed. Replacing the character costs a fidelity nobody can see
    on that terminal anyway; a traceback costs the whole session.

    UTF-8 consoles take the fast path and get the text unchanged.
    """
    encoding = getattr(stream if stream is not None else console.file, "encoding", None)
    if not encoding:  # pragma: no cover - a stream with no encoding takes str as-is
        return text
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    return text


def emit_json(payload: Any) -> None:
    """Print ``payload`` as the only thing on stdout, in pure ASCII.

    ``ensure_ascii`` is the point. Rich re-serialises JSON with it turned off, so a
    non-ASCII path or a model's own output would be written literally and fail to encode
    on a legacy console -- the same failure :func:`printable` exists for, but on a
    stream a machine is reading. Escaped as ``\\uXXXX`` the JSON is identical once
    parsed and safe on every terminal.
    """
    console.print_json(json.dumps(payload), ensure_ascii=True)


# --------------------------------------------------------------------------- #
# Unit formatting
# --------------------------------------------------------------------------- #
_BINARY_UNITS = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")


def fmt_bytes(n: float, *, precision: int | None = None) -> str:
    """Format a byte count using binary units.

    Precision adapts so that small values keep meaningful digits and large
    values stay readable::

        fmt_bytes(1_589_641_216)  -> '1.48 GiB'
        fmt_bytes(4096)           -> '4.0 KiB'
        fmt_bytes(512)            -> '512 B'
    """
    if n < 0:
        return f"-{fmt_bytes(-n, precision=precision)}"
    size = float(n)
    unit_index = 0
    while size >= 1024.0 and unit_index < len(_BINARY_UNITS) - 1:
        size /= 1024.0
        unit_index += 1
    unit = _BINARY_UNITS[unit_index]
    if unit_index == 0:
        return f"{int(size)} {unit}"
    if precision is None:
        # Keep roughly three significant figures: 1.48 GiB, 15.4 GiB, 789 MiB.
        precision = 2 if size < 10 else (1 if size < 100 else 0)
    return f"{size:.{precision}f} {unit}"


def fmt_int(n: float) -> str:
    """Thousands-separated integer: ``61595`` -> ``'61,595'``."""
    return f"{round(n):,}"


def fmt_count(n: float) -> str:
    """Compact magnitude for large counts: ``280_000_000`` -> ``'280M'``.

    Used for token counts and step counts, where the exact digits rarely matter
    but the order of magnitude always does.
    """
    n = float(n)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= threshold:
            value = n / threshold
            digits = 2 if abs(value) < 10 else (1 if abs(value) < 100 else 0)
            return f"{value:.{digits}f}{suffix}"
    return f"{n:.0f}"


def fmt_params(n: int) -> str:
    """Parameter count, always in millions/billions: ``13_900_000`` -> ``'13.9M'``."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def fmt_duration(seconds: float) -> str:
    """Human-readable wall-clock duration.

    Deliberately coarse at large magnitudes -- claiming '6h 18m 42s' for an
    estimate derived from a five-step benchmark would imply precision we do
    not have::

        fmt_duration(45)     -> '45s'
        fmt_duration(4560)   -> '1h 16m'
        fmt_duration(374400) -> '4d 8h'
    """
    if seconds != seconds or seconds in (float("inf"), float("-inf")):  # NaN / inf
        return "unknown"
    seconds = max(0.0, float(seconds))
    if seconds == 0:
        return "0s"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"


# --------------------------------------------------------------------------- #
# Report layout
# --------------------------------------------------------------------------- #
def print_kv(title: str, rows: Sequence[tuple[str, str]], *, key_width: int = 16) -> None:
    """Print a titled two-column table of labels and values.

    Every command that reports measurements uses this, so labels line up the same
    way in ``doctor``, ``data inspect`` and everything added later. Values may
    contain Rich markup and wrap; keys may not and never wrap.
    """
    table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2, 0, 0))
    table.add_column(style="dim", no_wrap=True, min_width=key_width)
    table.add_column(overflow="fold")
    for key, value in rows:
        table.add_row(key, value)
    console.print(f"[bold cyan]{title}[/]")
    console.print(table)
    console.print()


def print_bullets(title: str, items: Sequence[str], *, empty: str | None = None) -> None:
    """Print a titled bullet list, blank line between items.

    Laid out as a table rather than as ``print(f"* {item}")`` so that a wrapped
    item hangs under its own text instead of returning to column zero -- these
    lists carry a finding plus what to do about it, which is usually two lines.
    ``empty`` is printed when there is nothing to list, because an empty section
    with no explanation reads like the check never ran.
    """
    console.print(f"[bold cyan]{title}[/]")
    if not items:
        if empty is not None:
            console.print(f"  {empty}", highlight=False)
        console.print()
        return
    table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 1, 1, 0))
    table.add_column(no_wrap=True, width=3, style="bold")
    table.add_column(overflow="fold")
    for item in items:
        table.add_row(f"  {BULLET}", item)
    console.print(table, highlight=False)
    console.print()


def print_command(command: str, *, indent: str = "  ", style: str = "bold cyan") -> None:
    """Print a command the user is meant to copy, on exactly one line.

    Wrapping is off, and that is the whole point of the function. Rich wraps at the
    console width, and a wrapped command is not a command: what a user copies out of

        trainai train --data d --preset tiny --context 256 --seq-len 256
        --steps 75 --batch-size 64

    is two lines, and pasting it runs the first and then fails on ``--steps``. Narrow
    terminals and long paths are both ordinary, so this is not an edge case; the
    command *this project prints for the user to run* is exactly the string that must
    survive a copy. Letting the terminal soft-wrap instead keeps it one logical line.

    Two arguments do that, and both are load-bearing. ``overflow="ignore"`` is what
    suppresses the wrap. ``crop=False`` is what keeps the tail: without it Rich cuts the
    line at the console width, which is worse than wrapping, because a cropped command
    still runs -- with the last few flags silently gone.

    ``no_wrap=True`` is deliberately *not* passed. It reads like the argument that does
    this job, and it was here at first, but all eight combinations were measured at
    ``width=40`` and it changes nothing that the other two do not already do. A flag no
    behaviour depends on is worse than absent: it draws the eye away from the two that
    matter, and someone deleting ``overflow="ignore"`` would think the important one had
    been kept. If a future Rich makes it necessary again, the width sweep in
    ``test_console.py`` fails and says so.
    """
    console.print(
        f"{indent}[{style}]{command}[/]",
        overflow="ignore",
        crop=False,
        highlight=False,
    )


# --------------------------------------------------------------------------- #
# Error rendering
# --------------------------------------------------------------------------- #
def render_error(exc: TrainAIError, *, show_details: bool = False) -> None:
    """Print a ``TrainAIError`` as a panel with its hint, without a traceback."""
    body = Text()
    body.append(exc.message)
    if exc.hint:
        body.append("\n\n")
        body.append("What to do: ", style="bold")
        body.append(exc.hint)
    if show_details and exc.details:
        body.append("\n")
        for key, value in exc.details.items():
            body.append(f"\n  {key}: ", style="dim")
            body.append(_short_repr(value), style="dim")

    err_console.print(
        Panel(
            body,
            title=f"[bold red]{_humanise(type(exc).__name__)}[/]",
            border_style="red",
            padding=(1, 2),
        )
    )


def _humanise(class_name: str) -> str:
    """``DatasetDecodeError`` -> ``'Dataset decode error'``."""
    out: list[str] = []
    for index, char in enumerate(class_name):
        if char.isupper() and index:
            out.append(" ")
        out.append(char.lower() if index else char)
    return "".join(out)


def _short_repr(value: Any, limit: int = 160) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + ELLIPSIS


def rule(title: str) -> None:
    """A titled horizontal rule, used to separate phases of a long command."""
    console.rule(f"[bold]{title}[/]", style="dim")
