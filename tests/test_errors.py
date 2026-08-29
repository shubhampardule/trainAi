"""Tests for the error taxonomy.

The contract worth protecting is that exit codes stay stable and distinct, and
that errors serialise cleanly for logs and the API.
"""

from __future__ import annotations

import json

import pytest

from trainai import errors
from trainai.errors import (
    CapacityError,
    CheckpointCorruptError,
    DatasetDecodeError,
    DatasetError,
    ExitCode,
    TrainAIError,
    TrainingDivergedError,
    UsageError,
)


def test_message_hint_and_details_are_preserved() -> None:
    exc = DatasetDecodeError(
        "data/a.txt is not valid UTF-8 (first bad byte at offset 1041).",
        hint="Re-save the file as UTF-8, or pass --encoding latin-1.",
        details={"path": "data/a.txt", "offset": 1041},
    )
    assert exc.message.startswith("data/a.txt")
    assert exc.hint is not None and "UTF-8" in exc.hint
    assert exc.details["offset"] == 1041
    assert str(exc) == exc.message


def test_to_dict_is_json_serialisable() -> None:
    exc = CapacityError(
        "No configuration fits in 2.74 GiB.",
        hint="Free VRAM by closing other applications, or pass --max-params 6M.",
        details={"rejected": [{"params": 110_000_000, "peak_gib": 6.55}]},
    )
    payload = json.loads(json.dumps(exc.to_dict()))
    assert payload["error"] == "CapacityError"
    assert payload["hint"]
    assert payload["details"]["rejected"][0]["peak_gib"] == 6.55


def test_hint_and_details_default_safely() -> None:
    exc = TrainAIError("something went wrong")
    assert exc.hint is None
    assert exc.details == {}
    assert exc.to_dict()["details"] == {}


@pytest.mark.parametrize(
    ("exc_type", "expected"),
    [
        (UsageError, ExitCode.USAGE),
        (DatasetError, ExitCode.DATASET),
        (DatasetDecodeError, ExitCode.DATASET),
        (CapacityError, ExitCode.CAPACITY),
        (CheckpointCorruptError, ExitCode.CHECKPOINT),
        (TrainingDivergedError, ExitCode.TRAINING),
        (TrainAIError, ExitCode.GENERIC),
    ],
)
def test_exit_codes_are_as_documented(exc_type: type[TrainAIError], expected: int) -> None:
    assert exc_type("x").exit_code == expected


def test_exit_codes_are_distinct_and_nonzero_for_failures() -> None:
    codes = {
        name: value
        for name, value in vars(ExitCode).items()
        if not name.startswith("_") and isinstance(value, int)
    }
    failures = {name: code for name, code in codes.items() if name != "OK"}
    assert 0 not in failures.values(), "a failure exit code must never be 0"
    assert len(set(failures.values())) == len(failures), f"duplicate exit codes: {failures}"


def test_every_public_error_subclasses_the_base() -> None:
    for name in errors.__all__:
        obj = getattr(errors, name)
        if isinstance(obj, type) and issubclass(obj, Exception):
            assert issubclass(obj, TrainAIError), f"{name} must subclass TrainAIError"


def test_all_exports_exist() -> None:
    missing = [name for name in errors.__all__ if not hasattr(errors, name)]
    assert not missing, f"__all__ lists names that do not exist: {missing}"
