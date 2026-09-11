"""Tests for the error taxonomy.

The contract worth protecting is that exit codes stay stable and distinct, and
that errors serialise cleanly for logs and the API.
"""

from __future__ import annotations

import json
from decimal import Decimal

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
    check_choice,
    json_literal,
    json_type_name,
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


# --------------------------------------------------------------------------- #
# json_type_name and json_literal
#
# The two helpers every "your file has the wrong thing here" message is built from --
# see serialise.checked_fields, binarize's manifest checks and cli/train.py's plan
# reader. No test named either of them until now: they were only ever reached through
# those callers, and every one of those callers asks about a value that is *not* an
# object, so the branch that says "an object" had never run. Neither had the two
# fallbacks for a value that did not come out of a JSON file at all.
# --------------------------------------------------------------------------- #
#: Every branch, in JSON's words rather than Python's. Both members of each pair the
#: implementation could confuse: ``true`` before ``8`` because ``bool`` is an ``int``
#: subclass, and empty containers beside full ones because a falsy value must still be
#: named for what it is rather than for being empty.
JSON_NAMES = [
    (None, "null"),
    (True, "a boolean"),
    (False, "a boolean"),
    (8, "a number"),
    (3e-4, "a number"),
    ({"n_layer": 8}, "an object"),
    ({}, "an object"),
    ([256, 512], "an array"),
    ([], "an array"),
    ("cosine", "a string"),
    ("", "a string"),
]


@pytest.mark.parametrize(("value", "expected"), JSON_NAMES)
def test_a_json_value_is_named_in_the_words_of_the_file_it_came_from(
    value: object, expected: str
) -> None:
    """The reader is looking at JSON, so the name has to be JSON's.

    ``NoneType`` and ``dict`` are not things anyone can search a config file for, and
    "its n_layer is a dict" invites the reply that it is not a dict, it is an object.
    ``null`` carries no article because it is a literal rather than a kind of thing.

    ``test_a_true_that_should_be_a_count_is_named_as_a_boolean`` in
    ``tests/test_train_schedule.py`` pins the ``bool``-before-``int`` ordering on the
    real error; this is the same claim on the helper, alongside the six names that
    ordering could not affect.
    """
    assert json_type_name(value) == expected


def test_a_value_that_is_not_json_at_all_is_named_by_its_python_type() -> None:
    """Nothing ``json.loads`` produces reaches this, which is exactly the risk.

    A library caller assembling a config in Python -- ``ModelConfig.from_dict`` and
    ``TrainConfig.from_dict`` are public, and ``docs/plan-format.md`` points at them --
    can pass a ``Decimal`` from a TOML reader or a numpy scalar from a computed default.
    Those are not JSON types, so the six names above do not apply and the honest answer
    is what Python calls it.

    The alternative is worse than an ugly name: without this line the function falls off
    its end and returns ``None``, so a message that was about the user's file becomes
    ``unsupported format string`` from inside the error path.
    """
    assert json_type_name(Decimal("3e-4")) == "Decimal"
    assert json_type_name({1, 2}) == "set"
    assert json_type_name(object()) == "object"


def test_a_value_json_refuses_is_quoted_as_python_rather_than_raising() -> None:
    """The one thing a formatter in an error path may never do is raise.

    ``json.dumps`` refuses a ``Decimal`` with ``TypeError`` and a structure that
    contains itself with ``ValueError``. Either one, uncaught, would replace a sentence
    naming the field the user has to fix with a traceback from inside the reporting of
    an unrelated failure -- and the field name would be nowhere in it.

    So the fallback quotes Python's spelling. It is not JSON, and it is not pretending
    to be: ``Decimal('0.0003')`` is at least the value, and searchable.
    """
    assert json_literal(Decimal("0.0003")) == "Decimal('0.0003')"

    circular: dict[str, object] = {}
    circular["self"] = circular
    assert json_literal(circular) == "{'self': {...}}"


def test_a_value_too_long_to_quote_is_named_instead_of_truncated() -> None:
    """Both halves of the limit, on values ``json.dumps`` accepts and on ones it refuses.

    Truncating the JSON would print an unterminated string, which reads as a second
    defect in the file. So a long value is described rather than shown -- and that
    applies after the ``repr`` fallback too, which is the case that needs saying: a
    twenty-element set has no JSON form *and* no short one, and the answer is still a
    name rather than 290 characters of Python.
    """
    assert json_literal(list(range(40))) == "an array"
    assert json_literal([1, 2, 3]) == "[1, 2, 3]"

    assert json_literal({Decimal(index) for index in range(20)}) == "set"
    assert json_literal(frozenset()) == "frozenset()"

    # The boundary itself, at the length the callers get by default.
    assert json_literal("x" * 38) == f'"{"x" * 38}"', "40 characters of JSON still fits"
    assert json_literal("x" * 39) == "a string"


# --------------------------------------------------------------------------- #
# check_choice
# --------------------------------------------------------------------------- #
def test_a_value_in_the_choices_is_returned_unchanged() -> None:
    """The negative control. A validator that rejects everything passes every test
    below it, and would stop `--precision bf16` from working at all."""
    assert check_choice("bf16", ("auto", "bf16", "fp16", "fp32"), "--precision") == "bf16"


def test_an_unknown_value_names_the_flag_the_value_and_every_choice() -> None:
    with pytest.raises(UsageError) as caught:
        check_choice("fp64", ("auto", "bf16", "fp16", "fp32"), "--precision")

    assert caught.value.exit_code == ExitCode.USAGE
    assert "--precision" in caught.value.message
    assert "fp64" in caught.value.message, "the rejected value has to appear, to be searchable"
    assert "auto, bf16, fp16, fp32" in (caught.value.hint or "")
    assert caught.value.details == {
        "flag": "--precision",
        "given": "fp64",
        "choices": ["auto", "bf16", "fp16", "fp32"],
    }


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("bf6", "bf16"),
        ("fp64", "fp16"),
        ("Best", "best"),
        ("laest", "latest"),
        ("safetensor", "safetensors"),
    ],
)
def test_a_near_miss_is_named(given: str, expected: str) -> None:
    with pytest.raises(UsageError) as caught:
        check_choice(given, ("auto", "bf16", "fp16", "fp32", "best", "latest", "safetensors"), "-x")

    assert f"Did you mean {expected}?" in (caught.value.hint or "")


@pytest.mark.parametrize("given", ["gpu", "", "nvidia", "16"])
def test_a_value_that_resembles_nothing_is_not_given_a_guess(given: str) -> None:
    """``--device gpu`` used to be answered with "did you mean xpu?" -- a confident wrong
    answer to someone holding an NVIDIA card. Listing the real values is the better
    failure, so the cutoff sits above difflib's default."""
    with pytest.raises(UsageError) as caught:
        check_choice(given, ("auto", "cuda", "cpu", "mps", "xpu"), "--device")

    assert "Did you mean" not in (caught.value.hint or "")
    assert "auto, cuda, cpu, mps, xpu" in (caught.value.hint or "")


def test_a_flag_whose_values_mean_something_can_say_what() -> None:
    with pytest.raises(UsageError) as caught:
        check_choice("newest", ("best", "latest"), "--which", hint="best is the lowest val loss.")

    hint = caught.value.hint or ""
    assert "Choose one of: best, latest." in hint
    assert hint.endswith("best is the lowest val loss.")
