"""Metrics: a JSONL record of what happened, and a live view of it happening.

Two audiences, one source of numbers. The JSONL file is for later -- plotting a
loss curve, comparing two runs, checking what the learning rate actually was at
step 3,000. The live display is for now, and its job is to answer "is this working
and when will it finish" without the user having to guess.

Every number shown is measured. Throughput is tokens divided by elapsed seconds,
not a constant times a batch size; the time remaining comes from observed
throughput over recent steps, not from a formula. Early in a run that estimate is
poor, which is why it is presented as a time remaining that visibly settles rather
than as a confident total.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from trainai.console import SPINNER, console, fmt_count, fmt_duration, fmt_int

__all__ = ["MetricsWriter", "ThroughputMeter", "TrainingDisplay", "perplexity"]

#: Steps of history the throughput estimate averages over. Long enough that one
#: slow step (a checkpoint write, another process waking up) does not swing the
#: estimate, short enough to follow a real change in speed.
_THROUGHPUT_WINDOW = 50

#: Above this, perplexity is reported as "very high" instead of a number. exp(20)
#: is 485 million: printing it implies a precision the loss does not have, and it
#: only happens in the first few steps or when a run has diverged.
_MAX_REPORTED_PERPLEXITY_LOSS = 20.0


def perplexity(loss: float) -> float:
    """``exp(loss)``, or infinity when the loss is not finite.

    Perplexity is the more intuitive number -- "the model is as uncertain as if it
    were choosing uniformly among N tokens" -- but it is exponential in the loss,
    so it is only meaningful once the loss is small enough for the exponential not
    to dominate. :func:`format_perplexity` is what decides whether to show it.
    """
    if not math.isfinite(loss):
        return math.inf
    if loss > _MAX_REPORTED_PERPLEXITY_LOSS:
        return math.inf
    return math.exp(loss)


def format_perplexity(loss: float) -> str:
    value = perplexity(loss)
    if not math.isfinite(value):
        return "very high"
    return fmt_count(value) if value >= 1000 else f"{value:.1f}"


class ThroughputMeter:
    """Measured tokens per second, over a sliding window of real time."""

    def __init__(self, window: int = _THROUGHPUT_WINDOW) -> None:
        self._samples: deque[tuple[float, int]] = deque(maxlen=window)
        self._total_tokens = 0
        self._started = time.perf_counter()

    def record(self, tokens: int, seconds: float) -> None:
        self._samples.append((max(seconds, 1e-9), tokens))
        self._total_tokens += tokens

    @property
    def tokens_per_second(self) -> float:
        """Over the recent window. 0 until something has been recorded."""
        if not self._samples:
            return 0.0
        seconds = sum(s for s, _ in self._samples)
        tokens = sum(t for _, t in self._samples)
        return tokens / seconds if seconds > 0 else 0.0

    @property
    def average_tokens_per_second(self) -> float:
        """Over the whole run, including time spent on checkpoints and validation."""
        elapsed = time.perf_counter() - self._started
        return self._total_tokens / elapsed if elapsed > 0 else 0.0

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._started

    def seconds_remaining(self, steps_left: int, tokens_per_step: int) -> float:
        """Time left at the recently observed rate, or NaN if nothing is measured."""
        rate = self.tokens_per_second
        if rate <= 0 or steps_left <= 0:
            return float("nan") if rate <= 0 else 0.0
        return steps_left * tokens_per_step / rate


class MetricsWriter:
    """Append-only JSONL log of a run.

    One JSON object per line, flushed after every write. Flushing costs a syscall
    per record and buys the property that matters: if the process is killed -- by
    Ctrl-C, by the OS, by a power cut -- everything up to the last record is on
    disk. A buffered writer would lose the most interesting part, which is whatever
    happened immediately before the run stopped.

    Append rather than truncate, so a resumed run adds to the history instead of
    replacing it. Each record carries the wall-clock time and the step, so the
    ordering survives any number of restarts.
    """

    def __init__(self, path: str | Path, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self._enabled = enabled
        self._handle = None
        self.unreadable_lines = 0
        """Lines the last :meth:`read_all` could not use. Zero until it is called.

        A summary computed from fewer records than the file has lines is a different
        summary, and nothing else in the run would say so.
        """
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Line-buffered text mode, explicit newline so records are LF on every
            # platform: this file is read by plotting code, not by a shell.
            self._handle = open(  # noqa: SIM115 - the handle must outlive __init__
                self.path, "a", encoding="utf-8", newline="\n"
            )

    def write(self, event: str, step: int, **fields: Any) -> None:
        """Record one event. Unknown fields are allowed and encouraged."""
        if self._handle is None:
            return
        record = {"event": event, "step": step, "time": time.time(), **fields}
        self._handle.write(json.dumps(record, separators=(",", ":"), default=_jsonable) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None

    def __enter__(self) -> MetricsWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def read_all(self) -> list[dict[str, Any]]:
        """Every record written so far, for plotting or for a run summary. Never raises.

        This runs at the very end of a training run, after the checkpoints are on
        disk, so anything it raises is a traceback over a *log* thrown at someone
        whose model is already saved. Measured on a three-record file, one edit per
        case: of fifteen ways it can be damaged, nine ended the run with a bare
        ``AttributeError``, ``TypeError`` or ``UnicodeDecodeError`` and exit 1.

        Decoded and parsed per line, so damage stays confined to the line it is on.
        Both whole-file alternatives are worse: ``read_text`` raises
        ``UnicodeDecodeError`` on one stray byte -- it is a ``ValueError``, so the
        ``json.JSONDecodeError`` guard below never saw it -- and ``errors="replace"``
        substitutes characters *inside* a value that then still parses, which turns a
        missing line into a wrong number.

        A line that is not a JSON **object** is dropped rather than returned. The
        annotation here promises ``dict`` and :func:`summarise_run` calls ``.get`` on
        every element, so a line holding ``3`` used to end a finished run with
        ``AttributeError: 'int' object has no attribute 'get'``.

        Nothing here refuses, because this file is the most derived thing the run
        writes: it describes what already happened and no model depends on it. That
        is the same reasoning ``checkpoints.json`` gets and the opposite of
        ``manifest.json``, which is the only description of its shards.
        """
        self.unreadable_lines = 0
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for raw in self.path.read_bytes().splitlines():
            if not raw.strip():
                continue
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                # A partial line means the process died mid-write -- and because the
                # writer appends, a resumed run glues its first record onto that
                # fragment, so the damage can sit mid-file rather than last. Every
                # complete record around it is still good.
                self.unreadable_lines += 1
                continue
            if not isinstance(record, dict):
                self.unreadable_lines += 1
                continue
            records.append(record)
        return records


def _jsonable(value: Any) -> Any:
    """Last resort for numpy scalars and anything else that sneaks in."""
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass
    return str(value)


class TrainingDisplay:
    """The live view: one progress bar, and a line for each thing worth reading.

    Disabled cleanly when output is not a terminal or when ``quiet`` is set, so
    the same loop code serves an interactive run, a CI log and a JSON pipeline.

    The column set is chosen to fit inside 80 characters, which is not cosmetic.
    When the line overflows, Rich truncates it with U+2026 -- and on a cp1252
    Windows console that byte is written as 0x85 and read back as a replacement
    character, so an overlong line loses the step count and the time remaining to
    mojibake. Measured before the fix: the line ``step 12/450 ... 0:00:03 left
    00:12`` came out with 0x85 in place of the total, the elapsed seconds and the
    time remaining -- three of the four numbers it exists to show. The gradient norm
    is therefore left out of the display and kept in the metrics file, and the bar
    has a fixed width rather than expanding to fill whatever is left.
    """

    #: Fixed rather than ``None``. An expanding bar takes all the free space and
    #: pushes the trailing columns off the end of the line.
    BAR_WIDTH = 16

    def __init__(
        self,
        total_steps: int,
        *,
        start_step: int = 0,
        quiet: bool = False,
    ) -> None:
        self.total_steps = total_steps
        self.quiet = quiet
        self._progress = Progress(
            SpinnerColumn(SPINNER),
            MofNCompleteColumn(),
            BarColumn(bar_width=self.BAR_WIDTH),
            TextColumn("{task.fields[stats]}"),
            TimeElapsedColumn(),
            TextColumn("[dim]/[/]"),
            TimeRemainingColumn(compact=True),
            console=console,
            disable=quiet,
            transient=False,
        )
        self._task = None
        self._start_step = start_step

    def __enter__(self) -> TrainingDisplay:
        self._progress.start()
        self._task = self._progress.add_task(
            "training",
            total=self.total_steps,
            completed=self._start_step,
            stats="",
        )
        return self

    def __exit__(self, *exc: object) -> None:
        self._progress.stop()

    def update(
        self,
        step: int,
        *,
        loss: float,
        lr: float,
        tokens_per_second: float,
        grad_norm: float | None = None,
    ) -> None:
        """Refresh the bar.

        ``grad_norm`` is accepted and not displayed: it is genuinely useful for
        diagnosing an impending divergence, but it goes in the metrics file rather
        than on a line that has to fit in 80 columns. See the class docstring.
        """
        if self._task is None:  # pragma: no cover - guarded by the context manager
            return
        stats = f"loss [bold]{loss:.4f}[/] lr {lr:.1e} {fmt_count(tokens_per_second)} tok/s"
        self._progress.update(self._task, completed=step, stats=stats)

    def log_eval(self, step: int, *, val_loss: float, train_loss: float | None = None) -> None:
        """Print a validation result above the bar, where it stays on screen."""
        if self.quiet:
            return
        line = (
            f"[cyan]step {fmt_int(step)}[/]  val loss [bold]{val_loss:.4f}[/]  "
            f"perplexity {format_perplexity(val_loss)}"
        )
        if train_loss is not None:
            line += f"  [dim](train {train_loss:.4f})[/]"
        self._progress.console.print(line, highlight=False)

    def log(self, message: str) -> None:
        if not self.quiet:
            self._progress.console.print(message, highlight=False)


def _is_number(value: Any) -> bool:
    """A JSON number, and not a bool.

    ``bool`` is an ``int`` subclass, so ``isinstance(True, int)`` is ``True`` and a
    ``"loss": true`` in the log would be summarised as a loss of 1.0.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_count(value: Any) -> bool:
    """A JSON integer, and not a bool. See :func:`_is_number` for why bool is named."""
    return isinstance(value, int) and not isinstance(value, bool)


def summarise_run(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce a metrics log to the numbers worth reporting at the end.

    Returns an empty summary for an empty log rather than inventing values, so a
    run that died before its first step reports nothing instead of zeros that look
    like measurements.

    A record whose loss is not a number is skipped rather than summarised, for two
    measured reasons. ``min`` over a mixed list raises ``TypeError: '<' not supported
    between instances of 'str' and 'float'``, which is how a ``"val_loss": null`` in
    the log ended a finished run; and a string that got as far as the summary was
    reported as the final loss, which is the quieter half of the same defect.
    """
    train = [r for r in records if r.get("event") == "train" and _is_number(r.get("loss"))]
    evals = [r for r in records if r.get("event") == "eval" and _is_number(r.get("val_loss"))]
    summary: dict[str, Any] = {
        "train_records": len(train),
        "eval_records": len(evals),
    }
    if train:
        summary["first_loss"] = train[0]["loss"]
        summary["final_loss"] = train[-1]["loss"]
        summary["steps"] = train[-1].get("step")
    if evals:
        best = min(evals, key=lambda r: r["val_loss"])
        summary["best_val_loss"] = best["val_loss"]
        summary["best_val_step"] = best.get("step")
        summary["final_val_loss"] = evals[-1]["val_loss"]
        summary["best_val_perplexity"] = perplexity(best["val_loss"])
    tokens = [r["tokens"] for r in train if _is_count(r.get("tokens"))]
    if tokens:
        summary["tokens_seen"] = max(tokens)
    durations = [r["elapsed"] for r in records if _is_number(r.get("elapsed"))]
    if durations:
        summary["elapsed_seconds"] = max(durations)
        summary["elapsed"] = fmt_duration(max(durations))
    return summary
