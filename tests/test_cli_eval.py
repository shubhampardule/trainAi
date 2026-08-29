"""CLI tests for ``trainai eval``.

The measurement itself is covered in :mod:`tests.test_eval_perplexity`. What these
check is the wiring and the guard rails: that a run directory alone is enough, that
the flags reach the library, that ``--json`` is machine-readable and silent otherwise,
and that the things this command refuses produce the documented exit code rather than
a traceback.

The one worth reading is ``test_a_dataset_that_moved_says_so_instead_of_crashing``.
Evaluation needs the token shards, which a run directory does not contain, so a
deleted or moved dataset is the normal second-day failure here -- and the difference
between a sentence naming the path and a ``FileNotFoundError`` is the whole point.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from conftest import flat
from trainai.cli.main import main
from trainai.errors import ExitCode


def cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    """Drive the real entry point, argv and all, and return its exit code."""
    monkeypatch.setattr(sys, "argv", ["trainai", *args])
    return main()


# --------------------------------------------------------------------------- #
# The ordinary path
# --------------------------------------------------------------------------- #
def test_a_run_directory_alone_is_enough(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """The dataset is not retyped: the checkpoint records which one it was."""
    run = cli_trained_run.run

    assert cli(monkeypatch, "eval", str(run), "--device", "cpu") == ExitCode.OK

    out = flat(capsys.readouterr().out)
    assert "Val split" in out
    assert "Perplexity" in out
    assert "per token of this run" in out


def test_split_both_reports_train_before_val(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """Both splits appear, and train comes first -- which is the point of asking for both.

    The order is asserted rather than assumed. ``--split both`` exists so the gap between
    the two can be read off, and a reader scanning downward subtracts the second from the
    first; reversing it silently inverts the sign of the thing they came for. The order
    also used to be unpinned here while a comment on ``SPLIT_CHOICES`` claimed the wrong
    one of the two orders in this module was the report's, so "fixing" the code to match
    the comment would have passed.
    """
    run = cli_trained_run.run

    assert cli(monkeypatch, "eval", str(run), "--split", "both", "--device", "cpu") == ExitCode.OK

    out = capsys.readouterr().out
    assert "Train split" in out
    assert "Val split" in out
    assert out.index("Train split") < out.index("Val split")


def test_json_is_parseable_and_carries_what_makes_it_comparable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    run = cli_trained_run.run

    assert cli(monkeypatch, "eval", str(run), "--device", "cpu", "--json") == ExitCode.OK

    report = json.loads(capsys.readouterr().out)
    assert report["version"] == 1
    assert report["splits"][0]["split"] == "val"
    assert report["splits"][0]["perplexity"] > 0
    assert report["splits"][0]["batch_size"] >= 1
    assert report["model"]["vocab_size"] > 0
    assert report["dataset_matches_run"] is True
    assert report["dataset_check"] == "match"
    assert "precision" in report


def test_a_damaged_manifest_says_the_check_could_not_be_made(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_trained_run: Any,
    tmp_path: Path,
) -> None:
    """A copy of the run's own dataset with its content hash stripped.

    Nothing else is touched -- same shards, same tokenizer -- so the copy really is the
    dataset this run trained on, and the only thing lost is the ability to prove it. What
    is asserted is that the report says exactly that: it neither claims the match it did
    not verify nor the mismatch it has no evidence for. Measured before the fix, the
    output of this case was indistinguishable from a verified match, in the text report
    and in the JSON.

    Whitespace is collapsed before matching because rich wraps the dataset row on the
    path, and a ``tmp_path`` is long enough to split the phrase across two lines.
    """
    copied = tmp_path / "hash-stripped"
    shutil.copytree(cli_trained_run.data, copied)
    manifest = copied / "manifest.json"
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["content_hash"] = ""
    manifest.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8", newline="\n")

    args = ("eval", str(cli_trained_run.run), "--data", str(copied), "--device", "cpu")
    assert cli(monkeypatch, *args) == ExitCode.OK

    out = " ".join(capsys.readouterr().out.split())
    assert "cannot tell: a content hash is missing" in out
    assert "Could not check whether this is the dataset this run trained on" in out
    assert "not the one this run trained on" not in out

    assert cli(monkeypatch, *args, "--json") == ExitCode.OK
    report = json.loads(capsys.readouterr().out)
    assert report["dataset_check"] == "unknown"
    assert report["dataset_matches_run"] is False


def test_max_batches_reaches_the_library_and_is_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """An explicit ``--batch-size`` so there is a tail for ``--max-batches`` to cut.

    Left at the default, this fixture's small validation split is one batch, and a cap
    of one batch would score all of it -- the assertion would pass for the wrong reason.
    """
    run = cli_trained_run.run

    assert (
        cli(
            monkeypatch,
            "eval",
            str(run),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--max-batches",
            "1",
            "--json",
        )
        == ExitCode.OK
    )

    report = json.loads(capsys.readouterr().out)
    val = report["splits"][0]
    assert val["batch_size"] == 2
    assert val["batches"] == 1
    assert val["stopped_early"] is True
    assert val["coverage"] < 1.0


def test_which_latest_loads_a_different_checkpoint_than_best(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    run = cli_trained_run.run

    assert cli(monkeypatch, "eval", str(run), "--device", "cpu", "--json") == ExitCode.OK
    best = json.loads(capsys.readouterr().out)
    assert (
        cli(monkeypatch, "eval", str(run), "--device", "cpu", "--which", "latest", "--json")
        == ExitCode.OK
    )
    latest = json.loads(capsys.readouterr().out)

    assert best["which"] == "best"
    assert latest["which"] == "latest"


def test_a_checkpoint_file_can_be_named_directly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    run = cli_trained_run.run
    one = sorted((run / "checkpoints").glob("step-*.pt"))[0]

    assert cli(monkeypatch, "eval", str(one), "--device", "cpu", "--json") == ExitCode.OK

    assert json.loads(capsys.readouterr().out)["which"] == "explicit"


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #
def test_an_unknown_split_is_refused_before_the_model_is_loaded(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    run = cli_trained_run.run

    assert cli(monkeypatch, "eval", str(run), "--split", "sideways") == ExitCode.USAGE

    assert "--split must be one of" in capsys.readouterr().err


def test_a_run_that_does_not_exist_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert cli(monkeypatch, "eval", str(tmp_path / "nowhere")) == ExitCode.USAGE

    assert "Nothing at" in capsys.readouterr().err


def test_a_dataset_that_moved_says_so_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_trained_run: Any,
    tmp_path: Path,
) -> None:
    """A run directory holds no token shards, so this is the normal second-day failure.

    The checkpoint's recorded dataset root is pointed at nothing, which is what
    deleting or moving the prepared dataset looks like from here. A
    ``FileNotFoundError`` from inside the loader would name a shard file and tell the
    user nothing about what to pass.
    """
    from dataclasses import replace

    from trainai.train.checkpoint import load_checkpoint as real_load

    run = cli_trained_run.run
    gone = tmp_path / "gone"
    monkeypatch.setattr(
        "trainai.infer.session.load_checkpoint",
        lambda path, **kw: replace(
            real_load(path, **kw), dataset={**real_load(path, **kw).dataset, "root": str(gone)}
        ),
    )

    assert cli(monkeypatch, "eval", str(run), "--device", "cpu") == ExitCode.USAGE

    err = flat(capsys.readouterr().err)
    assert "no longer at" in err
    assert "--data" in err


def test_a_data_path_that_is_not_a_prepared_dataset_is_a_dataset_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_trained_run: Any,
) -> None:
    """Pointing --data at the raw corpus is the mistake; exit 3 says which kind."""
    run = cli_trained_run.run

    assert (
        cli(
            monkeypatch,
            "eval",
            str(run),
            "--device",
            "cpu",
            "--data",
            str(cli_trained_run.corpus),
        )
        == ExitCode.DATASET
    )

    assert "manifest.json" in capsys.readouterr().err


def test_a_context_beyond_what_the_model_was_built_for_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    run = cli_trained_run.run

    assert cli(monkeypatch, "eval", str(run), "--device", "cpu", "--seq-len", "4096") == (
        ExitCode.USAGE
    )

    err = capsys.readouterr().err
    assert "was built for" in err
    # This fixture is a --context 64 --seq-len 32 run, so the largest legal context and
    # the comparable one are different numbers and both have to be offered.
    assert "--seq-len 64" in err
    assert "--seq-len 32" in err


# --------------------------------------------------------------------------- #
# Which context the report says it used
#
# The fixture trains at 32 inside a model built for 64, which is the shape three of the
# four presets produce. Every assertion here failed before the default changed: the
# table said "Context 64 tokens" for a number the training curve had measured at 32,
# and then explained the resulting gap as the trainer's batch sampling.
# --------------------------------------------------------------------------- #
def test_the_context_row_names_the_trained_length_and_the_built_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    run = cli_trained_run.run

    assert cli(monkeypatch, "eval", str(run), "--device", "cpu") == ExitCode.OK

    out = " ".join(capsys.readouterr().out.split())
    assert "Context 32 tokens (what this run trained at; built for 64)" in out


def test_scoring_at_the_built_context_is_marked_as_past_the_trained_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    """And the gap against the recorded loss is blamed on the context, not on sampling.

    The old text -- "the trainer samples --eval-batches of the split, this scored what
    you asked for" -- was the tool's own default causing a discrepancy and then pointing
    the reader at the wrong cause.
    """
    run = cli_trained_run.run

    assert cli(monkeypatch, "eval", str(run), "--device", "cpu", "--seq-len", "64") == ExitCode.OK

    out = " ".join(capsys.readouterr().out.split())
    assert "Context 64 tokens (past the 32 this run trained at; built for 64)" in out
    assert "the trainer measured at 32 tokens of context, this scored at 64" in out
    assert "samples --eval-batches" not in out
    assert "read this as extrapolation" in out


def test_the_json_carries_both_lengths(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_trained_run: Any
) -> None:
    run = cli_trained_run.run

    assert cli(monkeypatch, "eval", str(run), "--device", "cpu", "--json") == ExitCode.OK

    report = json.loads(capsys.readouterr().out)
    assert report["model"]["trained_seq_len"] == 32
    assert report["splits"][0]["seq_len"] == 32
