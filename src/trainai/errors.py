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

import json
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
    "HardwareError",
    "InsufficientMemoryError",
    "NoAcceleratorError",
    "TokenizerError",
    "TrainAIError",
    "TrainingDivergedError",
    "TrainingError",
    "UsageError",
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
# Hardware and capacity
# --------------------------------------------------------------------------- #
class HardwareError(TrainAIError):
    """The machine cannot do what was asked of it."""

    exit_code = ExitCode.CAPACITY


class NoAcceleratorError(HardwareError):
    """No GPU was found and the requested operation needs one to be practical."""


class InsufficientMemoryError(HardwareError):
    """Not enough VRAM or system RAM, as *measured* rather than estimated."""


class CapacityError(TrainAIError):
    """No viable training configuration exists under the user's constraints.

    Raised by the planner when every candidate on the ladder is rejected. The
    ``details`` dict is expected to carry the rejected candidates and the
    measured evidence for each rejection, so the user learns *why* rather than
    just being told no.
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
