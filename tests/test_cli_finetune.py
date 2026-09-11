"""``trainai finetune`` -- the command that continues one model on another corpus.

The tests here are mostly about what fine-tuning refuses and what it inherits, not
about whether the loss falls. Two of them exist because the failure they describe is
silent: a mismatched tokenizer trains happily and produces nothing usable, and a
model shape taken from a flag rather than from the checkpoint has nothing to load.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import cli_command, cli_long_flags, flat
from trainai.cli.main import app, main
from trainai.errors import ExitCode
from trainai.train.config import TrainConfig

runner = CliRunner()

#: Enough steps to write a checkpoint and a metrics log, few enough to cost
#: milliseconds. The base model is deliberately tiny; nothing here reads its output.
FAST = (
    "--batch-size",
    "2",
    "--seq-len",
    "32",
    "--steps",
    "4",
    "--warmup",
    "1",
    "--eval-every",
    "0",
    "--device",
    "cpu",
)
TINY = ("--layers", "1", "--heads", "2", "--width", "32", "--context", "64")

#: The base run is the one exception to ``--eval-every 0``. One of the panels under
#: test reports the base model's own validation loss, and a run that never evaluated
#: has none to report -- so a base built with ``FAST`` would make that assertion fail
#: for a reason that has nothing to do with the panel.
BASE_FAST = (
    "--batch-size",
    "2",
    "--seq-len",
    "32",
    "--steps",
    "4",
    "--warmup",
    "1",
    "--eval-every",
    "2",
    "--device",
    "cpu",
)


def cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["trainai", *args])
    return main()


@pytest.fixture
def base_data(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    many_document_corpus: Path,
    tmp_path: Path,
) -> Path:
    """The base model's dataset, and therefore the tokenizer everything else reuses."""
    out = tmp_path / "base-data"
    assert cli(monkeypatch, "data", "prepare", str(many_document_corpus), "--out", str(out)) == 0
    capsys.readouterr()
    return out


@pytest.fixture
def base(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    base_data: Path,
    tmp_path: Path,
) -> Path:
    """A trained run, built through the CLI, whose checkpoint can be fine-tuned."""
    run = tmp_path / "base-run"
    assert (
        cli(monkeypatch, "train", "-d", str(base_data), "--out", str(run), *TINY, *BASE_FAST) == 0
    )
    capsys.readouterr()
    return run


@pytest.fixture
def tune_data(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    base_data: Path,
    tmp_corpus: Path,
    tmp_path: Path,
) -> Path:
    """A second dataset, different text, prepared with the base dataset's tokenizer.

    This is the only way a fine-tune dataset can be built, and it is why
    ``data prepare --tokenizer`` had to exist before this command could.
    """
    out = tmp_path / "tune-data"
    assert (
        cli(
            monkeypatch,
            "data",
            "prepare",
            str(tmp_corpus),
            "--out",
            str(out),
            "--tokenizer",
            str(base_data),
            "--val-fraction",
            "0",
        )
        == 0
    )
    capsys.readouterr()
    return out


# --------------------------------------------------------------------------- #
# Surface
# --------------------------------------------------------------------------- #
def test_help_lists_finetune() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "finetune" in flat(result.output)


def test_finetune_has_no_model_flags() -> None:
    """The shape is read, not chosen, so there is nothing for a shape flag to do.

    Asserted as an absence rather than as a refusal message, because the useful
    property is that the flags do not exist: ``Checkpoint.apply_to`` requires an
    exact match, so a ``--layers`` that parsed would only ever produce a load error
    several seconds later, phrased as a shape mismatch the user did not cause.
    """
    flags = cli_long_flags(cli_command("finetune"))
    assert not flags & {"--layers", "--heads", "--width", "--ffn-width", "--context", "--preset"}
    assert {"--from", "--data", "--lr", "--steps"} <= flags


def test_finetune_requires_a_base_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert cli(monkeypatch, "finetune", "-d", str(tmp_path)) == ExitCode.USAGE


# --------------------------------------------------------------------------- #
# What it inherits
# --------------------------------------------------------------------------- #
def test_the_model_shape_comes_from_the_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    base: Path,
    tune_data: Path,
    tmp_path: Path,
) -> None:
    """The tuned run's model is the base model's, not the default preset's."""
    assert (
        cli(
            monkeypatch,
            "finetune",
            "-d",
            str(tune_data),
            "--from",
            str(base),
            "--out",
            str(tmp_path / "tuned"),
            "--dry-run",
            "--json",
            *FAST,
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    base_start = json.loads((base / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert plan["model"] == base_start["model"]["config"]
    assert plan["model"]["n_layer"] == 1  # TINY, not the tiny *preset*
    assert Path(plan["base"]["path"]).name.startswith("step-")


def test_the_default_learning_rate_is_a_tenth_of_the_pretraining_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    base: Path,
    tune_data: Path,
    tmp_path: Path,
) -> None:
    """Both halves of the claim, so neither can drift alone.

    ``FINETUNE_LR`` is documented as a tenth of the from-scratch default and chosen
    from a measurement recorded beside it. If someone changes ``TrainConfig.lr``, the
    ratio in that comment silently stops being true, and this is what notices.
    """
    from trainai.cli.train import FINETUNE_LR

    assert pytest.approx(TrainConfig.lr / 10) == FINETUNE_LR

    assert (
        cli(
            monkeypatch,
            "finetune",
            "-d",
            str(tune_data),
            "--from",
            str(base),
            "--out",
            str(tmp_path / "tuned"),
            "--dry-run",
            "--json",
            *FAST,
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["train"]["lr"] == pytest.approx(FINETUNE_LR)


def test_an_explicit_learning_rate_wins(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    base: Path,
    tune_data: Path,
    tmp_path: Path,
) -> None:
    assert (
        cli(
            monkeypatch,
            "finetune",
            "-d",
            str(tune_data),
            "--from",
            str(base),
            "--out",
            str(tmp_path / "tuned"),
            "--lr",
            "0.0007",
            "--dry-run",
            "--json",
            *FAST,
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["train"]["lr"] == pytest.approx(7e-4)


def test_finetuning_records_the_model_it_started_from(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    base: Path,
    tune_data: Path,
    tmp_path: Path,
) -> None:
    """Lineage in the run record, because a run directory cannot otherwise say it.

    A fine-tuned checkpoint is a function of a base checkpoint living somewhere else.
    Without this, "which model is this" is answerable only from shell history.
    """
    run = tmp_path / "tuned"
    assert (
        cli(
            monkeypatch,
            "finetune",
            "-d",
            str(tune_data),
            "--from",
            str(base),
            "--out",
            str(run),
            *FAST,
        )
        == 0
    )
    capsys.readouterr()

    start = json.loads((run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    base_start = json.loads((base / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert start["resumed_from"] is None
    assert Path(start["finetuned_from"]).name.startswith("step-")
    # The parent block stores the checkpoint's ``ModelConfig``; the run record's own
    # ``model`` is the wider model summary, which nests that same dict under "config".
    assert start["parent"]["model"] == base_start["model"]["config"]
    assert (
        start["parent"]["dataset"]["tokenizer_fingerprint"]
        == start["dataset"]["tokenizer_fingerprint"]
    )
    assert start["parent"]["dataset"]["content_hash"] != start["dataset"]["content_hash"]
    assert list((run / "checkpoints").glob("step-*.pt"))


def test_finetune_json_emits_only_the_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    base: Path,
    tune_data: Path,
    tmp_path: Path,
) -> None:
    """The real run's ``--json``, which every other ``--json`` test in this file skips.

    ``--dry-run --json`` prints the *plan* and returns before a ``Trainer`` exists, and
    the tests above pass both flags -- so the document they pin is the projection, and
    the measured one went unprinted by any test here. They are not the same document:
    this one has the losses and the checkpoint path in it, which is what a script that
    shells out to ``finetune`` reads to find the model it just made.

    The whole of stdout is parsed rather than searched, because "only the result" is
    half the contract: a panel row or a progress line on the same stream makes the
    output unparseable, and this is the sibling of ``test_train_json_emits_only_the_result``
    for the command that has a second panel to suppress -- the base model's.
    """
    run = tmp_path / "tuned"

    code = cli(
        monkeypatch,
        "finetune",
        "-d",
        str(tune_data),
        "--from",
        str(base),
        "--out",
        str(run),
        "--json",
        *FAST,
    )
    captured = capsys.readouterr()

    assert code == ExitCode.OK, captured.err
    payload = json.loads(captured.out)
    assert payload["steps_completed"] == 4
    assert payload["final_train_loss"] > 0
    assert payload["device"] == "cpu"
    assert payload["diverged"] is False
    assert Path(payload["checkpoint"]).exists(), "the JSON named a checkpoint that is not there"


# --------------------------------------------------------------------------- #
# What it refuses
# --------------------------------------------------------------------------- #
def test_a_dataset_with_its_own_tokenizer_is_refused_and_leaves_no_run_directory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    base: Path,
    tmp_corpus: Path,
    tmp_path: Path,
) -> None:
    """The mistake this command is most likely to meet, and the one that hides best.

    Preparing a dataset the normal way trains it a new tokenizer, so its token ids
    address the base model's embedding rows at random. Nothing about the resulting
    loss curve would look wrong, which is why this is a refusal and not a warning.

    Asserted for ``--dry-run`` as well as for a real run, and that is the half with
    teeth: the ``Trainer`` refuses this too, but only a real run reaches the
    ``Trainer``, so a check that lives there alone lets ``--dry-run`` print a plan and
    exit 0 for a pair of things that cannot be trained together.
    """
    stranger = tmp_path / "stranger"
    assert (
        cli(
            monkeypatch,
            "data",
            "prepare",
            str(tmp_corpus),
            "--out",
            str(stranger),
            "--val-fraction",
            "0",
        )
        == 0
    )
    capsys.readouterr()

    for extra in (("--dry-run",), ()):
        run = tmp_path / f"tuned{len(extra)}"
        code = cli(
            monkeypatch,
            "finetune",
            "-d",
            str(stranger),
            "--from",
            str(base),
            "--out",
            str(run),
            *extra,
            *FAST,
        )
        output = flat(capsys.readouterr().err)

        assert code == ExitCode.CHECKPOINT
        assert "different tokenizer" in output
        assert "data prepare --tokenizer" in output
        assert not run.exists()


def test_a_missing_base_checkpoint_names_the_flag_the_user_typed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tune_data: Path,
    tmp_path: Path,
) -> None:
    """``--from`` and ``--resume`` share the resolver; the message must not share a name.

    Read from stderr: error panels are written there, not to stdout, so that ``--json``
    output stays parseable when a command fails.
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    code = cli(monkeypatch, "finetune", "-d", str(tune_data), "--from", str(empty), *FAST)
    output = flat(capsys.readouterr().err)

    assert code == ExitCode.USAGE
    assert "--from" in output
    assert "--resume" not in output


def test_a_sequence_longer_than_the_base_context_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    base: Path,
    tune_data: Path,
    tmp_path: Path,
) -> None:
    """With no --context flag to raise, the base model's context is the ceiling."""
    code = cli(
        monkeypatch,
        "finetune",
        "-d",
        str(tune_data),
        "--from",
        str(base),
        "--out",
        str(tmp_path / "tuned"),
        "--seq-len",
        "128",
        "--batch-size",
        "2",
        "--steps",
        "2",
        "--warmup",
        "1",
        "--device",
        "cpu",
    )
    output = flat(capsys.readouterr().err)

    assert code == ExitCode.TRAINING
    assert "exceeds the model's context length of 64" in output


# --------------------------------------------------------------------------- #
# What it says
# --------------------------------------------------------------------------- #
def test_the_data_budget_drops_the_from_scratch_reference(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    base: Path,
    tune_data: Path,
    tmp_path: Path,
) -> None:
    """A fine-tune has too few tokens per parameter by definition, so the note changes.

    Both code paths, because they build the budget separately: ``--dry-run`` constructs
    a :class:`DataBudget` itself, while a real run reads the one the ``Trainer`` built,
    so a flag set on only one of them still reports the wrong advice half the time.

    The negative control matters more than either: the same dataset trained from
    scratch must still get the from-scratch advice, so a flag wired backwards fails
    here rather than quietly rewriting every run's report.
    """
    assert (
        cli(
            monkeypatch,
            "finetune",
            "-d",
            str(tune_data),
            "--from",
            str(base),
            "--out",
            str(tmp_path / "planned"),
            "--dry-run",
            *FAST,
        )
        == 0
    )
    tuned = flat(capsys.readouterr().out)

    assert "from-scratch training usually wants around 20" not in tuned
    assert "expected when fine-tuning" in tuned

    assert (
        cli(
            monkeypatch,
            "finetune",
            "-d",
            str(tune_data),
            "--from",
            str(base),
            "--out",
            str(tmp_path / "tuned"),
            *FAST,
        )
        == 0
    )
    ran = flat(capsys.readouterr().out)

    assert "from-scratch training usually wants around 20" not in ran
    assert "expected when fine-tuning" in ran

    assert (
        cli(
            monkeypatch,
            "train",
            "-d",
            str(tune_data),
            "--out",
            str(tmp_path / "scratch"),
            "--dry-run",
            *TINY,
            *FAST,
        )
        == 0
    )
    scratch = flat(capsys.readouterr().out)

    assert "from-scratch training usually wants around 20" in scratch
    assert "expected when fine-tuning" not in scratch


def test_the_run_panel_says_the_base_loss_is_not_comparable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    base: Path,
    tune_data: Path,
    tmp_path: Path,
) -> None:
    """Two validation losses on two corpora in one panel invite a wrong comparison."""
    assert (
        cli(
            monkeypatch,
            "finetune",
            "-d",
            str(tune_data),
            "--from",
            str(base),
            "--out",
            str(tmp_path / "tuned"),
            "--eval-every",
            "2",
            "--batch-size",
            "2",
            "--seq-len",
            "32",
            "--steps",
            "4",
            "--warmup",
            "1",
            "--device",
            "cpu",
        )
        == 0
    )
    output = flat(capsys.readouterr().out)

    assert "Base model" in output
    assert "on its own corpus, not this one" in output
    assert "Optimizer state and RNG were not carried over" in output
