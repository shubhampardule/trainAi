"""Error taxonomy for TrainAI.

Every *expected* failure raises a :class:`TrainAIError` subclass. The contract is:

``message``
    What went wrong, in plain language, naming the specific thing that failed.
``hint``
    What the user should do about it. This is not decoration. A ``TrainAIError``
    raised without an actionable hint is treated as a bug in TrainAI.
``details``
    Structured context (paths, sizes, measured numbers) for logs, ``plan.json``
    and -- later -- the web UI.

The CLI renders ``TrainAIError`` without a traceback, because a Python traceback
is noise for someone whose dataset merely has the wrong text encoding. Anything
that is *not* a ``TrainAIError`` keeps its traceback: those are our bugs, and we
want them reported.

Exit codes are stable so that shell scripts and CI can branch on them.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Sequence
from typing import Any

__all__ = [
    "CapacityError",
    "CheckpointCorruptError",
    "CheckpointError",
    "CheckpointIncompatibleError",
    "CheckpointNotFoundError",
    "ConfigError",
    "DatasetDecodeError",
    "DatasetEmptyError",
    "DatasetError",
    "DatasetFormatError",
    "DatasetNotFoundError",
    "ExitCode",
    "ExportError",
    "TokenizerError",
    "TrainAIError",
    "TrainingDivergedError",
    "TrainingError",
    "UsageError",
    "check_choice",
    "json_literal",
    "json_type_name",
]


class ExitCode:
    """Stable process exit codes, so scripts can branch on failure *kind*."""

    OK = 0
    GENERIC = 1
    USAGE = 2
    DATASET = 3
    CAPACITY = 4
    CHECKPOINT = 5
    TRAINING = 6
    EXPORT = 7
    INTERRUPTED = 130  # conventional value for SIGINT


class TrainAIError(Exception):
    """Base class for every failure TrainAI raises on purpose.

    Args:
        message: What went wrong. Name the specific file, config key or number.
        hint: The concrete next action. Required in spirit; see module docstring.
        details: Structured context, safe to serialise to JSON.
    """

    exit_code: int = ExitCode.GENERIC

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details: dict[str, Any] = details or {}

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation, used in ``events.jsonl`` and by the API."""
        return {
            "error": type(self).__name__,
            "message": self.message,
            "hint": self.hint,
            "details": self.details,
        }


# --------------------------------------------------------------------------- #
# Usage / configuration
# --------------------------------------------------------------------------- #
class UsageError(TrainAIError):
    """The command was invoked in a way that cannot work (bad flags, missing arg)."""

    exit_code = ExitCode.USAGE


class ConfigError(TrainAIError):
    """A configuration value is missing, malformed, or internally inconsistent."""

    exit_code = ExitCode.USAGE


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class DatasetError(TrainAIError):
    """Something is wrong with the user's dataset."""

    exit_code = ExitCode.DATASET


class DatasetNotFoundError(DatasetError):
    """The given dataset path does not exist, or matched no supported files."""


class DatasetEmptyError(DatasetError):
    """The dataset exists but yielded no usable text."""


class DatasetDecodeError(DatasetError):
    """A file could not be decoded as text. Carries the byte offset when known."""


class DatasetFormatError(DatasetError):
    """A structured file (e.g. JSONL) is malformed or lacks the expected field."""


class TokenizerError(TrainAIError):
    """Tokenizer training, loading, or round-tripping failed."""

    exit_code = ExitCode.DATASET


# --------------------------------------------------------------------------- #
# Capacity
# --------------------------------------------------------------------------- #
class CapacityError(TrainAIError):
    """No viable training configuration exists under the user's constraints.

    Raised by the planner when every candidate on the ladder is rejected. The
    ``details`` dict is expected to carry the rejected candidates and the
    measured evidence for each rejection, so the user learns *why* rather than
    just being told no.

    This is the only error in the family, deliberately. There were three more --
    ``HardwareError`` and its ``NoAcceleratorError`` and ``InsufficientMemoryError``
    subclasses -- and nothing raised any of them. "Not enough VRAM" is this error:
    it is a fact about a configuration measured against a budget, not about the
    card, and splitting it out would have given two classes the same exit code and
    the same meaning. "No accelerator" is not an error at all -- ``doctor`` reports
    it, ``plan`` measures the CPU instead, and ``train`` runs there -- so a class
    for it described a policy this project does not have.
    """

    exit_code = ExitCode.CAPACITY


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
class CheckpointError(TrainAIError):
    """A checkpoint could not be written, found, or read back."""

    exit_code = ExitCode.CHECKPOINT


class CheckpointNotFoundError(CheckpointError):
    """The requested checkpoint (or run directory) does not exist."""


class CheckpointCorruptError(CheckpointError):
    """A checkpoint is present but unreadable or fails its integrity check."""


class CheckpointIncompatibleError(CheckpointError):
    """The checkpoint was written by an incompatible model or format version."""


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
class TrainingError(TrainAIError):
    """Training failed for a reason we recognise and can explain."""

    exit_code = ExitCode.TRAINING


class TrainingDivergedError(TrainingError):
    """Loss became NaN or infinite. Almost always the learning rate."""


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
class ExportError(TrainAIError):
    """Exporting the trained model failed, or the export failed verification."""

    exit_code = ExitCode.EXPORT


# --------------------------------------------------------------------------- #
# Helpers for composing the ``message`` half of the contract
# --------------------------------------------------------------------------- #
def json_type_name(value: Any) -> str:
    """What to call a JSON value in a message about a file the user may have edited.

    Names what the value *is*, so "its `n_layer` is a string" replaces a report that it
    is merely wrong. The names are JSON's, not Python's, because the file being
    described is JSON: ``null`` rather than ``NoneType``, and no article on it because
    it is a literal rather than a kind of thing.

    ``bool`` is checked before ``int`` deliberately: it is an ``int`` subclass, so the
    obvious ordering reports ``true`` as a number.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, dict):
        return "an object"
    if isinstance(value, list):
        return "an array"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, (int, float)):
        return "a number"
    return type(value).__name__


def json_literal(value: Any, *, limit: int = 40) -> str:
    """A value as it would appear in the file, for a message that asks for an edit.

    JSON's spelling rather than Python's -- ``null``, ``true``, ``"four"`` -- because
    the reader is looking at JSON, and ``None`` is not a thing they can search for.

    A value too long to quote falls back to :func:`json_type_name`. Truncating the JSON
    itself would print an unterminated string, which reads as a second defect.
    """
    try:
        text = json.dumps(value)
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= limit else json_type_name(value)


def check_choice(value: str, choices: Sequence[str], flag: str, *, hint: str = "") -> str:
    """``value`` unchanged if ``flag`` accepts it, or raise saying what ``flag`` accepts.

    For the options whose valid values are a closed set that the CLI nevertheless declares
    as ``TEXT``. Typer validates an ``Enum`` and a ``bool`` and nothing else, so a value
    like ``--precision bf6`` arrives as a string, and every consumer that reaches for it
    with ``==`` treats it as whatever its final branch does. That is how ``--precision``
    came to accept ``fp64``: no branch matched, the fall-through was ``auto``, and the run
    reported the precision it chose rather than the one it was asked for. A misspelt flag
    *name* is caught by the parser; a misspelt flag *value* had nothing to catch it.

    A near miss is named when there is one, because these are short values typed from
    memory and the misses look like ``bf6``, ``fp64`` and ``Best``. The cutoff is above
    :mod:`difflib`'s default: at 0.6, ``--device gpu`` is answered with "did you mean
    xpu?", which is a confident wrong answer to someone holding an NVIDIA card. Listing
    the real values and suggesting nothing is the better failure.

    ``hint`` is appended for a flag whose values mean something a list of them does not
    convey.

    Raises:
        UsageError: If ``value`` is not in ``choices``. Exit code 2 rather than a failure
            code for the operation, because nothing was attempted.
    """
    if value in choices:
        return value
    near = difflib.get_close_matches(value, choices, n=1, cutoff=0.7)
    raise UsageError(
        f"Unknown {flag} {value!r}.",
        hint=" ".join(
            part
            for part in (
                f"Did you mean {near[0]}?" if near else "",
                f"Choose one of: {', '.join(choices)}.",
                hint,
            )
            if part
        ),
        details={"flag": flag, "given": value, "choices": list(choices)},
    )
