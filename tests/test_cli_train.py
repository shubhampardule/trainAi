"""CLI tests for ``trainai train``.

The training behaviour itself is covered in :mod:`tests.test_train_loop`. What
these check is that the flags reach the library, that the guard rails hold, and
that failures produce the documented exit codes rather than a traceback.

Every run here uses the smallest model and step count that still exercises the
whole path, because the point is the wiring, not the learning.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from conftest import flat, unwrapped
from trainai.cli.main import app, main
from trainai.errors import ExitCode
from trainai.hardware.planner import PLAN_VERSION
from trainai.model.config import preset
from trainai.train.config import TrainConfig

runner = CliRunner()

#: A model small enough that a few steps cost milliseconds.
TINY = ("--layers", "1", "--heads", "2", "--width", "32", "--context", "64")

#: Training flags shared by the runs below.
FAST = (
    "--batch-size",
    "2",
    "--seq-len",
    "32",
    "--warmup",
    "1",
    "--eval-every",
    "0",
    "--device",
    "cpu",
)


def cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["trainai", *args])
    return main()


@pytest.fixture
def dataset_dir(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    many_document_corpus: Path,
    tmp_path: Path,
) -> Path:
    """A prepared dataset, made through the CLI so the two commands stay compatible."""
    out = tmp_path / "prepared"
    assert (
        cli(
            monkeypatch,
            "data",
            "prepare",
            str(many_document_corpus),
            "--out",
            str(out),
            "--vocab-size",
            "512",
        )
        == ExitCode.OK
    )
    capsys.readouterr()
    return out


# --------------------------------------------------------------------------- #
# Help
# --------------------------------------------------------------------------- #
def test_help_lists_train() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "train" in result.stdout


def test_train_help_groups_its_flags() -> None:
    """Thirty flat options are not something a first-time user can read."""
    result = runner.invoke(app, ["train", "--help"])

    assert result.exit_code == 0
    assert "Model" in result.stdout
    assert "Training" in result.stdout
    assert "Hardware" in result.stdout


def test_train_requires_a_dataset() -> None:
    result = runner.invoke(app, ["train"])

    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# Dry run
# --------------------------------------------------------------------------- #
def test_dry_run_reports_the_plan_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"

    code = cli(
        monkeypatch, "train", "--data", str(dataset_dir), "--out", str(run_dir), "--dry-run", *TINY
    )

    assert code == ExitCode.OK
    output = capsys.readouterr().out
    assert "Parameters" in output
    assert "Epochs" in output
    assert "Nothing was trained" in output
    assert not run_dir.exists()


def test_a_dry_run_and_a_real_run_agree_on_what_they_refuse(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    """The invariant, rather than the one case that revealed it was broken.

    ``--dry-run`` exists to "catch a badly proportioned run before it costs an
    evening", and it used to miss the runs that could not start at all: the two
    configuration checks lived in ``Trainer.__init__``, which a dry run returns before
    reaching. `--context 64 --seq-len 128 --dry-run` printed a whole plan -- "ctx64"
    on the shape line, "Sequence 128 tokens" three rows below it -- and exited 0,
    where the same flags without ``--dry-run`` exited 6. A dry run that approves what
    the next command rejects is worse than none, because it gets trusted, and
    `train --dry-run && train` is the obvious thing to write.
    """
    flags = (
        "train",
        "--data",
        str(dataset_dir),
        "--layers",
        "1",
        "--heads",
        "2",
        "--width",
        "32",
        "--context",
        "64",
        "--seq-len",
        "128",
        "--steps",
        "1",
        "--device",
        "cpu",
    )

    dry = cli(monkeypatch, *flags, "--out", str(tmp_path / "dry"), "--dry-run")
    dry_err = capsys.readouterr().err
    real = cli(monkeypatch, *flags, "--out", str(tmp_path / "real"))
    real_err = capsys.readouterr().err

    assert dry == real == ExitCode.TRAINING
    assert "--seq-len 128 exceeds the model's context length of 64" in flat(dry_err)
    assert dry_err == real_err, "the same mistake has to read the same either way"
    assert not (tmp_path / "dry").exists()


def test_a_refused_dry_run_prints_no_plan(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    """Refusing after printing the numbers would leave the numbers on screen to trust."""
    code = cli(
        monkeypatch,
        "train",
        "--data",
        str(dataset_dir),
        "--out",
        str(tmp_path / "run"),
        "--dry-run",
        "--seq-len",
        "128",
        "--steps",
        "1",
        *TINY,
    )

    assert code == ExitCode.TRAINING
    output = capsys.readouterr().out
    assert "Nothing was trained" not in output
    assert "Epochs" not in output


def test_dry_run_json_is_machine_readable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    code = cli(
        monkeypatch,
        "train",
        "--data",
        str(dataset_dir),
        "--out",
        str(tmp_path / "run"),
        "--dry-run",
        "--json",
        *TINY,
    )

    assert code == ExitCode.OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["model"]["n_layer"] == 1
    assert payload["train"]["steps"] > 0
    assert payload["budget"]["epochs"] > 0
    assert payload["precision"]


def test_the_step_count_is_derived_from_the_corpus(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    """A constant default cannot be right for both a 1 MB and a 1 GB dataset.

    The rule is "about three passes over the training split, but never fewer than
    50 steps". On a corpus small enough that three passes is under 50 steps -- like
    this fixture -- the floor wins and the run makes more passes than three. That is
    deliberate: a five-step run teaches nothing at all. The next test checks that
    the overshoot is reported rather than hidden.
    """
    from trainai.data.binarize import DatasetManifest

    cli(
        monkeypatch,
        "train",
        "--data",
        str(dataset_dir),
        "--out",
        str(tmp_path / "run"),
        "--dry-run",
        "--json",
        *TINY,
    )
    payload = json.loads(capsys.readouterr().out)

    train_tokens = DatasetManifest.load(dataset_dir).tokens("train")
    tokens_per_step = payload["train"]["tokens_per_step"]
    expected = max(50, min(20_000, int(3.0 * train_tokens / tokens_per_step)))
    assert payload["train"]["steps"] == expected


def test_a_run_that_overshoots_three_epochs_says_so(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    """The 50-step floor can push a small corpus past the memorising threshold.

    Measured on a real run: a 32-epoch schedule reached its best validation loss at
    epoch 8 and got monotonically worse for the remaining 24. So when the derived
    step count exceeds the threshold, the plan has to say it.
    """
    cli(
        monkeypatch,
        "train",
        "--data",
        str(dataset_dir),
        "--out",
        str(tmp_path / "run"),
        "--dry-run",
        "--json",
        *TINY,
    )
    payload = json.loads(capsys.readouterr().out)

    if payload["budget"]["epochs"] > 4.0:
        assert payload["budget"]["will_memorise"] is True
        assert any("passes over the training split" in note for note in payload["warnings"])
    else:
        assert payload["budget"]["will_memorise"] is False


def test_the_dry_run_parameter_count_matches_a_real_model(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    """The dry run reports the analytic count without building anything, so the two
    have to agree or the plan is describing a different model than the run."""
    from trainai.model.config import ModelConfig
    from trainai.model.gpt import GPT

    cli(
        monkeypatch,
        "train",
        "--data",
        str(dataset_dir),
        "--out",
        str(tmp_path / "run"),
        "--dry-run",
        "--json",
        *TINY,
    )
    payload = json.loads(capsys.readouterr().out)

    config = ModelConfig.from_dict(payload["model"])
    assert payload["budget"]["parameters"] == sum(p.numel() for p in GPT(config).parameters())


# --------------------------------------------------------------------------- #
# A real run
# --------------------------------------------------------------------------- #
def test_train_writes_a_checkpoint_and_a_metrics_log(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"

    code = cli(
        monkeypatch,
        "train",
        "--data",
        str(dataset_dir),
        "--out",
        str(run_dir),
        "--steps",
        "6",
        *FAST,
        *TINY,
    )

    assert code == ExitCode.OK
    assert (run_dir / "metrics.jsonl").exists()
    assert list((run_dir / "checkpoints").glob("step-*.pt"))
    assert "Measured" in capsys.readouterr().out


def test_the_report_prints_every_path_with_the_same_separator(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    """Found by running the documented quickstart: the Measured table mixed separators.

    The Checkpoint row went through `str()` while every other path in the project goes
    through `as_posix()`, so on Windows the report read

        Checkpoint   runs\\_e2e\\run\\checkpoints\\step-0000075.pt
        Metrics      runs/_e2e/run/metrics.jsonl

    two rows apart in one table. Cosmetic on its own, but it is the kind of detail that
    makes a user wonder which of the two paths the tool actually wrote.

    `unwrapped` because the path is a temporary directory long enough for Rich to break
    it mid-token. The `\\` assertion only *fails* on a platform where the two spellings
    differ, which is the platform the bug appeared on.
    """
    run_dir = tmp_path / "run"

    code = cli(
        monkeypatch,
        "train",
        "--data",
        str(dataset_dir),
        "--out",
        str(run_dir),
        "--steps",
        "4",
        *FAST,
        *TINY,
    )

    assert code == ExitCode.OK
    report = unwrapped(capsys.readouterr().out)
    assert "checkpoints/step-" in report, "the checkpoint path is not in POSIX form"
    assert "checkpoints\\step-" not in report
    assert "metrics.jsonl" in report


def test_train_json_emits_only_the_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    code = cli(
        monkeypatch,
        "train",
        "--data",
        str(dataset_dir),
        "--out",
        str(tmp_path / "run"),
        "--steps",
        "4",
        "--json",
        *FAST,
        *TINY,
    )

    assert code == ExitCode.OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["steps_completed"] == 4
    assert payload["device"] == "cpu"
    assert payload["diverged"] is False
    assert payload["checkpoint"]


def test_a_named_run_lands_under_runs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    code = cli(
        monkeypatch,
        "train",
        "--data",
        str(dataset_dir),
        "--name",
        "named-run",
        "--dry-run",
        "--json",
        *TINY,
    )

    assert code == ExitCode.OK
    assert Path(json.loads(capsys.readouterr().out)["run_dir"]) == Path("runs/named-run")


def test_the_result_panel_says_why_there_is_no_validation_loss(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    """The panel used to guess, and on this run both of its guesses are wrong.

    It printed "no validation split, or --eval-every 0" whatever the cause. Here the
    dataset has a validation split and --eval-every is 1; the real reason is that the
    split is too small for one window at this --seq-len, and the run knows it.
    """
    from trainai.data.binarize import DatasetManifest
    from trainai.data.loader import window_count

    val_tokens = DatasetManifest.load(dataset_dir).tokens("val")
    # A window needs 2 * seq_len + 1 tokens, so a context of val_tokens cannot fit one.
    seq_len = str(val_tokens)

    code = cli(
        monkeypatch,
        "train",
        "--data",
        str(dataset_dir),
        "--out",
        str(tmp_path / "run"),
        "--steps",
        "2",
        "--warmup",
        "1",
        "--batch-size",
        "1",
        "--seq-len",
        seq_len,
        "--eval-every",
        "1",
        "--device",
        "cpu",
        "--layers",
        "1",
        "--heads",
        "2",
        "--width",
        "32",
        "--context",
        seq_len,
    )
    output = " ".join(capsys.readouterr().out.split())

    assert code == ExitCode.OK
    assert "not measured" in output
    assert "--eval-every 0" not in output, "the old guess, and false here"
    # Said while the run starts, not only in the summary: a run that will never print a
    # val loss should say so before the user waits for one.
    assert "No held-out loss" in output
    suggested = int(re.search(r"--seq-len (\d+)", output).group(1))
    assert window_count(val_tokens, suggested) >= 1
    assert window_count(val_tokens, suggested + 1) == 0


def test_the_panel_prints_both_the_reason_and_the_fix(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Pinned on the panel alone, because the run logs the same words at the start.

    Asserting on a whole run's output cannot tell the two apart, so dropping either
    would still leave the other's copy in the captured text and the test would pass.
    """
    from trainai.cli.train import _print_result
    from trainai.train.loop import TrainResult, ValidationSkipped

    class StubTrainer:
        def memory_note(self) -> str:
            return "not measured on cpu"

    result = TrainResult(
        steps_completed=2,
        final_train_loss=5.0,
        best_val_loss=None,
        best_val_step=None,
        final_val_loss=None,
        tokens_seen=64,
        elapsed_seconds=1.0,
        tokens_per_second=64.0,
        peak_vram_bytes=0,
        checkpoint_path=None,
        run_dir=tmp_path,
        precision="fp32",
        device="cpu",
        validation_skipped=ValidationSkipped("SPLIT-TOO-SMALL", "USE-SEQ-LEN-22"),
    )

    _print_result(result, StubTrainer())  # type: ignore[arg-type]
    output = " ".join(capsys.readouterr().out.split())

    assert "SPLIT-TOO-SMALL" in output, "the reason"
    assert "USE-SEQ-LEN-22" in output, "the fix"


# --------------------------------------------------------------------------- #
# Guard rails
# --------------------------------------------------------------------------- #
def test_training_into_an_existing_run_is_refused_with_the_resume_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    """Two runs in one directory mix their metrics into one file, unrecoverably."""
    run_dir = tmp_path / "run"
    flags = (
        "train",
        "--data",
        str(dataset_dir),
        "--out",
        str(run_dir),
        "--steps",
        "4",
        *FAST,
        *TINY,
    )
    assert cli(monkeypatch, *flags) == ExitCode.OK
    capsys.readouterr()

    code = cli(monkeypatch, *flags)

    assert code == ExitCode.USAGE
    error = capsys.readouterr().err
    assert "--resume" in error
    assert "--force" in error


def test_force_overwrites_an_existing_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    flags = (
        "train",
        "--data",
        str(dataset_dir),
        "--out",
        str(tmp_path / "run"),
        "--steps",
        "4",
        *FAST,
        *TINY,
    )
    assert cli(monkeypatch, *flags) == ExitCode.OK
    capsys.readouterr()

    assert cli(monkeypatch, *flags, "--force") == ExitCode.OK


def test_resume_continues_from_the_run_directory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    common = ("--data", str(dataset_dir), "--out", str(run_dir), *FAST, *TINY)
    assert (
        cli(monkeypatch, "train", *common, "--steps", "6", "--checkpoint-every", "3") == ExitCode.OK
    )
    capsys.readouterr()

    code = cli(monkeypatch, "train", *common, "--steps", "6", "--json", "--resume", str(run_dir))

    assert code == ExitCode.OK
    assert json.loads(capsys.readouterr().out)["steps_completed"] == 6


def test_resume_from_a_path_with_no_checkpoint_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    code = cli(
        monkeypatch,
        "train",
        "--data",
        str(dataset_dir),
        "--out",
        str(tmp_path / "run"),
        "--resume",
        str(tmp_path / "nowhere"),
        *TINY,
    )

    assert code == ExitCode.USAGE
    assert "step-*.pt" in capsys.readouterr().err


def test_training_on_a_raw_corpus_says_to_prepare_it_first(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    many_document_corpus: Path,
    tmp_path: Path,
) -> None:
    code = cli(
        monkeypatch,
        "train",
        "--data",
        str(many_document_corpus),
        "--out",
        str(tmp_path / "run"),
        "--dry-run",
    )

    assert code == ExitCode.DATASET
    assert "data prepare" in capsys.readouterr().err


def test_an_impossible_model_shape_is_refused_as_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    code = cli(
        monkeypatch,
        "train",
        "--data",
        str(dataset_dir),
        "--out",
        str(tmp_path / "run"),
        "--dry-run",
        "--layers",
        "1",
        "--heads",
        "7",
        "--width",
        "32",
    )

    assert code == ExitCode.USAGE
    assert "n_head" in capsys.readouterr().err


def test_an_unknown_preset_lists_the_ones_that_exist(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    code = cli(
        monkeypatch,
        "train",
        "--data",
        str(dataset_dir),
        "--out",
        str(tmp_path / "run"),
        "--dry-run",
        "--preset",
        "enormous",
    )

    assert code == ExitCode.USAGE
    assert "tiny" in capsys.readouterr().err


def test_a_sequence_longer_than_the_context_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    code = cli(
        monkeypatch,
        "train",
        "--data",
        str(dataset_dir),
        "--out",
        str(tmp_path / "run"),
        "--context",
        "64",
        "--seq-len",
        "128",
        "--layers",
        "1",
        "--heads",
        "2",
        "--width",
        "32",
    )

    assert code == ExitCode.TRAINING
    assert "--seq-len" in capsys.readouterr().err


def test_the_default_preset_is_the_smallest_one() -> None:
    """It fits on every GPU this project targets. Choosing a larger default from a
    table would be the formula-instead-of-measurement mistake M3 exists to avoid."""
    from trainai.cli.train import DEFAULT_PRESET
    from trainai.model.config import PRESETS

    assert PRESETS[0].name == DEFAULT_PRESET


def test_verify_checks_the_dataset_before_starting(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    shard = next(iter(sorted(dataset_dir.glob("train_*.bin"))))
    payload = bytearray(shard.read_bytes())
    payload[8] ^= 0x01
    shard.write_bytes(bytes(payload))

    code = cli(
        monkeypatch,
        "train",
        "--data",
        str(dataset_dir),
        "--out",
        str(tmp_path / "run"),
        "--dry-run",
        "--verify",
        *TINY,
    )

    assert code == ExitCode.DATASET
    assert "checksum" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# --plan: every way a plan on disk can be unusable
# --------------------------------------------------------------------------- #
def a_loadable_plan(dataset_dir: Path) -> dict[str, Any]:
    """A plan this build accepts, built from the real serialisers.

    Deliberately not produced by ``trainai plan``: that measures the GPU, and what is
    under test here is the *loader*, whose input is a JSON document on disk.
    """
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    model = preset(
        "tiny", vocab_size=manifest["vocab_size"], n_layer=1, n_head=2, d_model=32, seq_len=64
    )
    train = TrainConfig(
        steps=2, batch_size=2, seq_len=32, warmup_steps=1, eval_every=0, device="cpu"
    )
    return {
        "version": PLAN_VERSION,
        "dataset": str(dataset_dir),
        "preset": "tiny",
        "model": model.to_dict(),
        "train": train.to_dict(),
    }


def _without(key: str) -> Callable[[dict[str, Any]], Any]:
    return lambda plan: {name: value for name, value in plan.items() if name != key}


def _with(key: str, value: Any) -> Callable[[dict[str, Any]], Any]:
    return lambda plan: {**plan, key: value}


def _within(block: str, key: str, value: Any) -> Callable[[dict[str, Any]], Any]:
    return lambda plan: {**plan, block: {**plan[block], key: value}}


def _lacking(block: str, key: str) -> Callable[[dict[str, Any]], Any]:
    """Drop one key from inside a block, leaving the rest of the plan intact."""
    return lambda plan: {
        **plan,
        block: {name: value for name, value in plan[block].items() if name != key},
    }


#: ``None`` means "do not write the file at all". A ``str`` is written verbatim, so a
#: case can be text that is not JSON. Anything else is serialised.
PLAN_DEFECTS: list[tuple[str, Callable[[dict[str, Any]], Any] | None]] = [
    ("the file is not there", None),
    ("the file is not valid JSON", lambda plan: '{"version": 1, "model":'),
    # Valid JSON, but nothing with a `version` to read. These four have no `.get`, so
    # reading the version off them raised AttributeError: traceback, exit 1.
    ("the whole file is an array", lambda plan: [1, 2, 3]),
    ("the whole file is a string", lambda plan: '"small"'),
    ("the whole file is a number", lambda plan: "7"),
    ("the whole file is null", lambda plan: "null"),
    ("the version is not this build's", _with("version", 99)),
    ("the model block is absent", _without("model")),
    ("the model block is null", _with("model", None)),
    ("the model block is a boolean", _with("model", True)),
    ("the model block is a string", _with("model", "small")),
    # This one is why an ``isinstance`` check precedes the deserialiser rather than
    # relying on it to complain: ``"vocab_size" not in raw`` is a *substring* test on a
    # string, so this payload passed the guard inside ``from_dict`` and then reached
    # ``raw.items()``, raising AttributeError -- a traceback and exit 1.
    ("the model block is a string containing a field name", _with("model", "vocab_size here")),
    ("the model block is an array", _with("model", [1, 2, 3])),
    ("a model field has the wrong type", _within("model", "n_layer", "four")),
    # A key *missing* from a block is not the same as one this build does not recognise.
    # Both used to fall through to the dataclass default, which rebuilds something the
    # plan does not describe: a different architecture, or a run on a different learning
    # rate. Both now name the key, on the same exit code as the rest of this table.
    ("a model key is missing", _lacking("model", "n_layer")),
    ("the train block is absent", _without("train")),
    ("the train block is a string", _with("train", "fast")),
    ("a train key is missing", _lacking("train", "lr")),
    ("a train field has the wrong type", _within("train", "lr", "fast")),
    ("a train field is out of range", _within("train", "steps", -1)),
    ("the vocabulary is not the dataset's", _within("model", "vocab_size", 8192)),
]


@pytest.mark.parametrize("label,defect", PLAN_DEFECTS, ids=[case[0] for case in PLAN_DEFECTS])
def test_an_unusable_plan_exits_two_and_names_the_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
    label: str,
    defect: Callable[[dict[str, Any]], Any] | None,
) -> None:
    """The contract `docs/plan-format.md` states, over every way the load can fail.

    A plan is a file the tool prints in order to be argued with, so a hand-edited one is
    ordinary input rather than an abuse. Two of these cases used to produce a
    forty-line traceback and exit 1, and three more named neither the file nor the
    command that would replace it -- one of them advising `trainai train`, which is how
    checkpoints are made, not plans. None of the paths had a test.
    """
    plan_file = tmp_path / "candidate.json"
    if defect is not None:
        payload = defect(a_loadable_plan(dataset_dir))
        plan_file.write_text(
            payload if isinstance(payload, str) else json.dumps(payload, indent=2),
            encoding="utf-8",
            newline="\n",
        )

    code = cli(
        monkeypatch,
        "train",
        "--data",
        str(dataset_dir),
        "--out",
        str(tmp_path / "run"),
        "--plan",
        str(plan_file),
        *FAST,
    )
    captured = capsys.readouterr()
    output = flat(captured.out + captured.err)

    assert code == ExitCode.USAGE, f"{label} exited {code}, not {ExitCode.USAGE}"
    assert "Traceback" not in output, f"{label} produced a traceback: {output[:400]}"
    assert plan_file.name in unwrapped(output), (
        f"{label} did not name the plan file. A user with several plans on disk cannot "
        f"tell which one was refused: {output[:400]}"
    )
    assert "trainai plan" in output, (
        f"{label} did not name the command that would replace the file: {output[:400]}"
    )


#: What the refusal should call each JSON type. A hand-editing user is told what the
#: file *is*, not merely that it is wrong, so the name has to be right -- and ``bool``
#: is an ``int`` subclass, so a naive number check reports ``true`` as a number.
JSON_TYPE_NAMES: list[tuple[str, str, str]] = [
    ("true", "a boolean", "the boolean"),
    ("7", "a number", "the number"),
    ("2.5", "a number", "the float"),
    ('"small"', "a string", "the string"),
    ("[1, 2]", "an array", "the array"),
    ("null", "null", "the null"),
]


@pytest.mark.parametrize(
    "literal,expected,label", JSON_TYPE_NAMES, ids=[c[2] for c in JSON_TYPE_NAMES]
)
def test_a_malformed_block_is_named_by_its_json_type(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
    literal: str,
    expected: str,
    label: str,
) -> None:
    """The refusal names what the block actually is.

    Checked as a message rather than against the private helper, because it is the
    message that has to be true. `null` reads as "is null" rather than "is a null" on
    purpose: it is the JSON literal, and a plan is edited as JSON.
    """
    plan = {**a_loadable_plan(dataset_dir), "model": json.loads(literal)}
    plan_file = tmp_path / "candidate.json"
    plan_file.write_text(json.dumps(plan, indent=2), encoding="utf-8", newline="\n")

    code = cli(
        monkeypatch,
        "train",
        "--data",
        str(dataset_dir),
        "--out",
        str(tmp_path / "run"),
        "--plan",
        str(plan_file),
        *FAST,
    )
    output = flat("".join(capsys.readouterr()))

    assert code == ExitCode.USAGE
    assert f"is {expected}, not an object" in output, (
        f"{label} block was not called {expected!r}: {output[:400]}"
    )


def test_a_loadable_plan_is_actually_loadable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    """The control for the test above.

    Without it, every case there would pass just as well if `--plan` refused *all*
    input, which is the failure mode a suite of negative tests cannot see.
    """
    plan_file = tmp_path / "candidate.json"
    plan_file.write_text(
        json.dumps(a_loadable_plan(dataset_dir), indent=2), encoding="utf-8", newline="\n"
    )

    code = cli(
        monkeypatch,
        "train",
        "--data",
        str(dataset_dir),
        "--out",
        str(tmp_path / "run"),
        "--plan",
        str(plan_file),
        "--dry-run",
    )

    assert code == ExitCode.OK, capsys.readouterr().err
    assert "L1 d32 h2" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# A plan file that is not UTF-8
# --------------------------------------------------------------------------- #
#: Written as bytes rather than through ``write_text``, which is the whole point: the
#: table above cannot express a file that is not decodable, so the encoding path it
#: shares with every case there went untested. ``UnicodeDecodeError`` is a
#: ``ValueError``, so neither the ``FileNotFoundError`` nor the ``json.JSONDecodeError``
#: guard saw these, and both ended the command with a bare traceback and exit 1.
PLAN_ENCODINGS: list[tuple[str, Callable[[bytes], bytes]]] = [
    ("a stray non-utf-8 byte", lambda text: text + b"\xe9"),
    # What "Save as -> Unicode" writes in Notepad. The file looks perfectly normal in the
    # editor that produced it, which is what makes a bare traceback the wrong answer.
    ("utf-16 little-endian", lambda text: text.decode().encode("utf-16")),
    ("utf-16 big-endian", lambda text: text.decode().encode("utf-16-be")),
    ("utf-16 big-endian with a BOM", lambda text: b"\xfe\xff" + text.decode().encode("utf-16-be")),
    ("latin-1 text in a value", lambda text: text.replace(b'"version"', b'"caf\xe9rsion"')),
]


@pytest.mark.parametrize("label,mangle", PLAN_ENCODINGS, ids=[case[0] for case in PLAN_ENCODINGS])
def test_a_plan_that_is_not_utf8_is_refused_and_named(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
    label: str,
    mangle: Callable[[bytes], bytes],
) -> None:
    """The same contract as the table above, on the one path it could not reach."""
    plan_file = tmp_path / "candidate.json"
    plan_file.write_bytes(mangle(json.dumps(a_loadable_plan(dataset_dir), indent=2).encode()))

    code = cli(
        monkeypatch,
        "train",
        "--data",
        str(dataset_dir),
        "--out",
        str(tmp_path / "run"),
        "--plan",
        str(plan_file),
        *FAST,
    )
    captured = capsys.readouterr()
    output = flat(captured.out + captured.err)

    assert code == ExitCode.USAGE, f"{label} exited {code}, not {ExitCode.USAGE}"
    assert "Traceback" not in output, f"{label} produced a traceback: {output[:400]}"
    assert plan_file.name in unwrapped(output), (
        f"{label} did not name the plan file: {output[:400]}"
    )
    assert "trainai plan" in output, f"{label} did not name the replacing command"


@pytest.mark.parametrize(
    "encoding",
    ["utf-16", "utf-16-le", "utf-16-be"],
    ids=["with the platform BOM", "little-endian", "big-endian"],
)
def test_a_utf16_plan_is_told_what_encoding_it_is(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
    encoding: str,
) -> None:
    """ "Not UTF-8 text" is a fact. Naming the encoding makes it an instruction.

    Nobody saves UTF-16 on purpose -- it is one entry in a dropdown -- so the refusal
    says which entry, on the platform where that dropdown is most likely to be open.

    Both byte orders, because the BOM check is a two-way membership test and a one-way
    one passes every other test in this file: a big-endian file still gets refused with
    the right exit code and the right file named, just with the hint that says nothing.
    """
    plan_file = tmp_path / "candidate.json"
    body = json.dumps(a_loadable_plan(dataset_dir), indent=2).encode(encoding)
    # `utf-16` writes its own BOM; the explicit byte orders do not, and a file with no BOM
    # is not identifiable, so the writing editor's BOM is prepended for those.
    bom = b"" if encoding == "utf-16" else (b"\xff\xfe" if encoding.endswith("le") else b"\xfe\xff")
    plan_file.write_bytes(bom + body)

    code = cli(
        monkeypatch,
        "train",
        "--data",
        str(dataset_dir),
        "--out",
        str(tmp_path / "run"),
        "--plan",
        str(plan_file),
        *FAST,
    )
    output = flat("".join(capsys.readouterr()))

    assert code == ExitCode.USAGE
    assert "UTF-16" in output, f"the encoding was not named: {output[:400]}"
    # "UTF-8" alone is not enough: the refusal line already says "is not UTF-8 text", so
    # an assertion on it would pass with no hint at all. "Re-save" only appears in the
    # hint, which is the part that says what to do about it.
    assert "Re-save it as UTF-8" in output, f"the fix was not named: {output[:400]}"
    assert "Notepad" in output, f"the editor that produces this was not named: {output[:400]}"


def test_a_plan_with_a_utf8_bom_loads(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    """Accepted, not merely named. This is Notepad's *default* save.

    It was refused as "is not valid JSON", which sends someone hunting for a syntax error
    in a file that has none. This file is documented as one to open and argue with, so
    the editor adding the BOM is being used as intended.
    """
    plan_file = tmp_path / "candidate.json"
    plan_file.write_bytes(
        b"\xef\xbb\xbf" + json.dumps(a_loadable_plan(dataset_dir), indent=2).encode()
    )

    code = cli(
        monkeypatch,
        "train",
        "--data",
        str(dataset_dir),
        "--out",
        str(tmp_path / "run"),
        "--plan",
        str(plan_file),
        "--dry-run",
    )

    assert code == ExitCode.OK, capsys.readouterr().err
    assert "L1 d32 h2" in capsys.readouterr().out


def test_a_plain_utf8_plan_is_unaffected_by_the_encoding_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    """The negative control for the decode step, with a non-ASCII character in a value.

    ``utf-8-sig`` is a no-op on a file with no BOM, and this is what says so: a guard
    that stripped three bytes unconditionally would fail here rather than quietly
    truncating every plan's first key.
    """
    # Built with chr() rather than written literally: this repo's sources are pure ASCII
    # (enforced by tests/test_conventions.py), and what matters is the bytes on disk.
    plan = {**a_loadable_plan(dataset_dir), "note": "caf" + chr(0xE9) + chr(0x2014) + "measured"}
    plan_file = tmp_path / "candidate.json"
    # `ensure_ascii=False`, or `json.dumps` would escape those two characters back to
    # ASCII and the file on disk would be pure ASCII -- not the case being tested.
    plan_file.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    assert not plan_file.read_bytes().isascii(), "the fixture wrote no multi-byte UTF-8"

    code = cli(
        monkeypatch,
        "train",
        "--data",
        str(dataset_dir),
        "--out",
        str(tmp_path / "run"),
        "--plan",
        str(plan_file),
        "--dry-run",
    )

    assert code == ExitCode.OK, capsys.readouterr().err
    assert "L1 d32 h2" in capsys.readouterr().out
