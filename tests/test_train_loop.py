"""Tests for :mod:`trainai.train.loop`, :mod:`trainai.train.budget` and metrics.

The one that carries the milestone is ``test_resuming_is_bitwise_identical``. Exact
resume is the difference between a checkpoint being a safety net and being a
suggestion: if a resumed run is only approximately the original, then a run that
was interrupted three times is not the run anyone thinks it is, and no number
reported about it means quite what it says.

It has a CUDA twin, ``test_resuming_is_bitwise_identical_on_cuda_too``, because the
CPU version pins ``device="cpu"`` and CPU is not where runs happen. Note what that
twin does *not* claim: two fresh CUDA runs of a long command drift apart, and the
README says so with the measurement. Resume is the guarantee; rerunning is not.

Getting that test right took one correction worth recording. The first version had
the partial run declare ``steps=12`` and the reference run declare ``steps=24``,
and the weights differed by 5.7e-03. That was not a resume bug: the learning-rate
schedule is a function of the *total* step count, so the first twelve steps ran at
different rates in the two runs. Both runs must declare the same ``steps`` and one
must simply be stopped early -- which is also why ``Trainer.config_changes`` exists
and is tested below.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import trainai.train.loop
from conftest import flat
from trainai.data.loader import open_split, window_count
from trainai.errors import (
    ConfigError,
    DatasetError,
    DatasetFormatError,
    ExitCode,
    TrainingDivergedError,
    TrainingError,
    UsageError,
)
from trainai.model.config import ModelConfig
from trainai.train.budget import (
    EPOCHS_BEFORE_MEMORISING,
    TOKENS_PER_PARAMETER_TARGET,
    DataBudget,
)
from trainai.train.config import TrainConfig
from trainai.train.loop import (
    DEVICE_CHOICES,
    Trainer,
    check_configs_agree,
    resolve_device,
    resolve_precision,
)
from trainai.train.metrics import (
    MetricsWriter,
    ThroughputMeter,
    format_perplexity,
    perplexity,
    summarise_run,
)


def model_config(dataset: Any, **overrides: Any) -> ModelConfig:
    settings: dict[str, Any] = {
        "vocab_size": dataset.vocab_size,
        "n_layer": 2,
        "n_head": 4,
        "d_model": 64,
        "seq_len": 64,
        "dropout": 0.1,
    }
    settings.update(overrides)
    return ModelConfig(**settings)


def train_config(**overrides: Any) -> TrainConfig:
    settings: dict[str, Any] = {
        "steps": 12,
        "batch_size": 4,
        "seq_len": 64,
        "lr": 1e-3,
        "warmup_steps": 3,
        "eval_every": 0,
        "eval_batches": 2,
        "checkpoint_every": 0,
        "log_every": 3,
        "seed": 7,
        "device": "cpu",
    }
    settings.update(overrides)
    return TrainConfig(**settings)


def make_trainer(dataset: Any, run_dir: Path, **overrides: Any) -> Trainer:
    model_overrides = overrides.pop("model", {})
    return Trainer(
        dataset=dataset,
        model_config=model_config(dataset, **model_overrides),
        train_config=train_config(**overrides),
        run_dir=run_dir,
        quiet=True,
    )


# --------------------------------------------------------------------------- #
# Device and precision
# --------------------------------------------------------------------------- #
def test_auto_device_resolves_to_something_real() -> None:
    assert resolve_device("auto").type in {"cuda", "mps", "cpu"}


def test_an_explicit_device_is_never_silently_downgraded() -> None:
    """Falling back from --device cuda turns a 20-minute run into a 12-hour one."""
    if torch.cuda.is_available():
        assert resolve_device("cuda").type == "cuda"
        return

    with pytest.raises(TrainingError) as caught:
        resolve_device("cuda")

    assert "pytorch.org" in (caught.value.hint or "")
    assert "--device cpu" in (caught.value.hint or "")


def test_a_device_name_this_project_does_not_support_is_refused_in_its_own_words() -> None:
    """``torch.device`` would refuse it too, but not in a form anyone can act on.

    It raises a bare ``RuntimeError`` -- so a traceback, since only ``TrainAIError``
    is rendered without one -- listing the twenty-odd backends it was compiled with:
    ``mkldnn``, ``opengl``, ``ideep``, ``ve``, ``fpga``, ``lazy``. TrainAI can train on
    five of those names. The list the user is given has to be the five.
    """
    with pytest.raises(UsageError) as caught:
        resolve_device("gpu")

    assert caught.value.exit_code == ExitCode.USAGE
    hint = caught.value.hint or ""
    assert ", ".join(DEVICE_CHOICES) in hint
    assert "mkldnn" not in hint and "opengl" not in hint


@pytest.mark.parametrize("requested", ["cuda:x", "cuda:", "auto:0", "cpu:1:2"])
def test_a_device_index_that_is_not_a_number_is_refused(requested: str) -> None:
    """``cuda:0`` has to keep working -- it is how a second GPU is picked -- so the
    check splits the index off rather than comparing the whole string. That split is
    what needs its own cases: ``auto:0`` names no device, and a trailing colon is not
    an index at all."""
    with pytest.raises(UsageError):
        resolve_device(requested)


def test_a_device_with_an_index_is_still_accepted() -> None:
    """The negative control for the test above: validation must not have closed the
    door on ``cuda:1``, which is the only way to choose between two cards."""
    assert resolve_device("cpu:0") == torch.device("cpu:0")


# ------------------------------------------------------ machines this one is not
#
# `resolve_device` and `resolve_precision` read the machine directly, so on this
# developer box -- which has a CUDA card and no Metal -- the auto path stops at the
# first branch and four of the outcomes never run. The tests below state the machine
# instead of asking it. Patching `torch.cuda.is_available` is the whole seam: both
# functions call it through the module-level `torch`, and nothing here touches a
# device, only names one.


def no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make this machine look like one without an NVIDIA card."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def metal(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    """Make `torch.backends.mps` answer as an Apple machine would, or would not."""
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: available)


def test_auto_picks_metal_when_there_is_no_cuda_card(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Apple ordering: CUDA first, Metal second, and only then the CPU."""
    no_cuda(monkeypatch)
    metal(monkeypatch, True)

    assert resolve_device("auto") == torch.device("mps")


def test_auto_falls_back_to_the_cpu_when_the_machine_has_no_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The end of the chain, and what CI itself runs on."""
    no_cuda(monkeypatch)
    metal(monkeypatch, False)

    assert resolve_device("auto") == torch.device("cpu")


def test_asking_for_cuda_without_cuda_names_the_likeliest_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same refusal as the test above, made on every machine rather than some.

    ``test_an_explicit_device_is_never_silently_downgraded`` asserts this only where
    CUDA is absent, so on a developer box with a card it checks the opposite branch and
    the error text is never read. The likeliest cause of `--device cuda` failing is a
    CPU-only wheel rather than a missing GPU, which is why the hint leads with the
    install link instead of with "buy a GPU".
    """
    no_cuda(monkeypatch)

    with pytest.raises(TrainingError) as caught:
        resolve_device("cuda")

    hint = caught.value.hint or ""
    assert "pytorch.org" in hint
    assert "--device cpu" in hint
    assert caught.value.details == {"requested": "cuda"}


def test_asking_for_metal_where_there_is_no_metal_is_refused_not_downgraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--device mps` on Linux or Windows. Falling back would be the silent kind of wrong."""
    metal(monkeypatch, False)

    with pytest.raises(TrainingError) as caught:
        resolve_device("mps")

    assert "Metal is not available" in str(caught.value)
    assert "doctor" in (caught.value.hint or "")


def test_metal_gets_fp32_because_this_project_has_never_verified_mps_autocast() -> None:
    """A dtype nobody has run on the backend is the silent-wrong-answer risk, not a win.

    ``resolve_precision`` reaches this without any Apple hardware: the device is named,
    not opened, and the answer is a decision about a backend rather than a measurement
    from one.
    """
    dtype, needs_scaler, note = resolve_precision("auto", torch.device("mps"))

    assert dtype == torch.float32
    assert not needs_scaler
    assert "Apple MPS" in note


def fake_xpu(monkeypatch: pytest.MonkeyPatch, checker: Any) -> None:
    """Replace `torch.xpu.is_bf16_supported`, or remove it, with `checker`.

    ``None`` deletes the attribute, which is how a torch built before XPU support -- or
    one where Intel changed the name again -- presents itself.
    """
    if checker is None:
        monkeypatch.delattr(torch.xpu, "is_bf16_supported", raising=False)
    else:
        monkeypatch.setattr(torch.xpu, "is_bf16_supported", checker)


def test_an_intel_card_that_reports_bf16_is_given_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_xpu(monkeypatch, lambda: True)

    dtype, needs_scaler, note = resolve_precision("auto", torch.device("xpu"))

    assert dtype == torch.bfloat16
    assert not needs_scaler
    assert "xpu" in note


def test_an_intel_card_without_bf16_gets_fp16_and_a_gradient_scaler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fp16 without the scaler underflows to zero gradients and trains nothing."""
    fake_xpu(monkeypatch, lambda: False)

    dtype, needs_scaler, note = resolve_precision("auto", torch.device("xpu"))

    assert dtype == torch.float16
    assert needs_scaler is True
    assert "XPU runtime reports no bf16" in note


@pytest.mark.parametrize(
    ("label", "checker"),
    [
        ("raises", lambda: 1 / 0),
        ("missing", None),
    ],
)
def test_an_xpu_runtime_that_cannot_answer_is_read_as_no_bf16(
    monkeypatch: pytest.MonkeyPatch, label: str, checker: Any
) -> None:
    """Unknown has to resolve to fp16, not to bf16, and not to a traceback.

    Intel's runtime is the youngest of the four and the one this project has no
    hardware for. Reading an error or a missing attribute as "yes" would autocast into
    a format the card may not have -- training that produces numbers rather than a
    crash. fp16 with a scaler is correct everywhere, so it is the safe default.
    """
    fake_xpu(monkeypatch, checker)

    dtype, needs_scaler, _ = resolve_precision("auto", torch.device("xpu"))

    assert (dtype, needs_scaler) == (torch.float16, True), label


def test_the_xpu_branch_survives_a_torch_with_no_xpu_support_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`torch.xpu` is recent. A build without it must still answer for `--device xpu`."""
    monkeypatch.delattr(torch, "xpu", raising=False)

    dtype, needs_scaler, _ = resolve_precision("auto", torch.device("xpu"))

    assert (dtype, needs_scaler) == (torch.float16, True)


def test_cpu_gets_fp32_and_says_why() -> None:
    dtype, needs_scaler, note = resolve_precision("auto", torch.device("cpu"))

    assert dtype == torch.float32
    assert not needs_scaler
    assert "CPU" in note


def test_half_precision_on_cpu_falls_back_and_reports_it() -> None:
    """The resolved choice is reported, not just applied: "mixed precision enabled"
    that quietly meant fp32 is exactly the claim this project avoids."""
    dtype, _, note = resolve_precision("bf16", torch.device("cpu"))

    assert dtype == torch.float32
    assert "CUDA or ROCm" in note


@pytest.mark.gpu
def test_auto_on_a_gpu_pairs_the_dtype_with_the_scaler_decision() -> None:
    dtype, needs_scaler, note = resolve_precision("auto", torch.device("cuda"))

    assert note
    if dtype == torch.bfloat16:
        assert not needs_scaler, "bf16 has the range for gradients; a scaler is pointless"
    else:
        assert dtype == torch.float16
        assert needs_scaler, "fp16 without a scaler underflows small gradients to zero"


# --------------------------------------------------------------------------- #
# A run trains
# --------------------------------------------------------------------------- #
def test_a_run_reduces_the_loss_and_records_it(prepared_dataset: Any, tmp_path: Path) -> None:
    trainer = make_trainer(prepared_dataset, tmp_path / "run", steps=40, eval_every=20)

    result = trainer.run()

    assert result.steps_completed == 40
    assert result.final_train_loss < result.summary["first_loss"]
    assert result.best_val_loss is not None
    assert not result.diverged
    assert result.tokens_seen == 40 * trainer.train_config.tokens_per_step
    assert result.tokens_per_second > 0


def test_the_metrics_log_is_one_json_object_per_line(prepared_dataset: Any, tmp_path: Path) -> None:
    trainer = make_trainer(prepared_dataset, tmp_path / "run", steps=12, eval_every=6)
    trainer.run()

    lines = (tmp_path / "run" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]

    events = [r["event"] for r in records]
    assert events[0] == "start"
    assert events[-1] == "end"
    assert all("step" in r and "time" in r for r in records)
    assert any(r["event"] == "train" for r in records)
    assert any(r["event"] == "eval" for r in records)


def test_the_logged_learning_rate_matches_the_schedule(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    """A run that reports a rate it did not use is worse than one that reports none."""
    trainer = make_trainer(prepared_dataset, tmp_path / "run", steps=12, log_every=1)
    trainer.run()

    records = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for record in (r for r in records if r["event"] == "train"):
        assert record["lr"] == trainer.schedule(record["step"] - 1)


def test_the_start_record_carries_the_model_and_the_budget(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    trainer = make_trainer(prepared_dataset, tmp_path / "run", steps=6)
    trainer.run()

    start = json.loads(
        (tmp_path / "run" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )

    assert start["model"]["parameters"] == trainer.model.parameter_count()
    assert start["budget"]["train_tokens"] == prepared_dataset.tokens("train")
    assert start["precision"] == trainer.precision_note


def test_validation_is_sequential_so_two_measurements_are_comparable(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    """Sampled validation makes the curve noisy for reasons unrelated to the model."""
    trainer = make_trainer(prepared_dataset, tmp_path / "run", steps=6, eval_every=0)
    from trainai.data.loader import open_split

    batcher, stream = open_split(
        prepared_dataset, "val", seq_len=64, batch_size=4, seed=trainer.train_config.seed
    )
    try:
        first = trainer.evaluate(batcher, 3)
        second = trainer.evaluate(batcher, 3)
    finally:
        stream.close()

    assert first == second


def test_validation_loss_is_weighted_by_tokens_not_by_batches(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    """A ragged final batch must not carry a full batch's weight.

    ``sequential_batches`` keeps the tail of the split rather than dropping it, so the
    last batch can hold one row where the others hold sixteen. Averaging per-batch
    losses gives that row sixteen times its share. This was a real bug: on the 1.1 MB
    Shakespeare validation split, six sequences out of thirty-eight carried half of the
    reported number, shifting it by 0.012 nats -- and since the best checkpoint is
    chosen on this value, the skew could pick the wrong one.
    """
    from trainai.data.loader import open_split

    trainer = make_trainer(
        prepared_dataset, tmp_path / "run", steps=4, warmup_steps=1, eval_every=0
    )
    batch_size = 16
    batcher, stream = open_split(
        prepared_dataset, "val", seq_len=64, batch_size=batch_size, seed=trainer.train_config.seed
    )
    try:
        # A ragged tail is what makes the two means differ at all.
        assert batcher.windows % batch_size != 0, (
            f"this fixture no longer has a ragged final batch at batch_size "
            f"{batch_size} ({batcher.windows} windows); pick another size"
        )
        reported = trainer.evaluate(batcher, max_batches=batcher.windows)

        trainer.model.eval()
        per_batch: list[tuple[int, float]] = []
        with torch.no_grad():
            for batch in batcher.sequential_batches():
                inputs = torch.from_numpy(batch.inputs).to(trainer.device).long()
                targets = torch.from_numpy(batch.targets).to(trainer.device).long()
                _, loss, _ = trainer.model(inputs, targets)
                assert loss is not None
                per_batch.append((int(targets.numel()), float(loss)))
    finally:
        stream.close()

    assert len(per_batch) > 1
    weighted = sum(n * loss for n, loss in per_batch) / sum(n for n, _ in per_batch)
    unweighted = sum(loss for _, loss in per_batch) / len(per_batch)

    assert reported == pytest.approx(weighted, abs=1e-9)
    assert weighted != pytest.approx(unweighted, abs=1e-9), (
        "the two means coincide here, so this test would pass even unweighted"
    )


def test_a_run_carries_the_tokenizer_that_can_decode_it(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    """A run without its tokenizer is a directory of numbers.

    The checkpoint stores only the tokenizer's fingerprint -- enough to detect a
    mismatch, not enough to decode anything. Without a copy in the run directory,
    ``trainai chat`` and ``trainai export`` would depend on the dataset still being
    there, and the dataset is the large thing people delete when training finishes.
    """
    from trainai.data.binarize import TOKENIZER_NAME

    run_dir = tmp_path / "run"
    make_trainer(prepared_dataset, run_dir, steps=4, warmup_steps=1, eval_every=0).run()

    copied = run_dir / TOKENIZER_NAME
    assert copied.is_file()

    source = Path(prepared_dataset.root) / TOKENIZER_NAME
    assert copied.read_bytes() == source.read_bytes()


def test_a_missing_tokenizer_is_reported_and_does_not_lose_the_run(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    """Failing to copy the tokenizer must not throw away an otherwise fine run."""
    from trainai.data.binarize import TOKENIZER_NAME

    run_dir = tmp_path / "run"
    trainer = make_trainer(prepared_dataset, run_dir, steps=4, warmup_steps=1, eval_every=0)
    trainer.dataset = replace(trainer.dataset, root=tmp_path / "gone")

    warning = trainer._copy_tokenizer()

    assert warning is not None
    assert TOKENIZER_NAME in warning
    assert not (run_dir / TOKENIZER_NAME).exists()


def test_a_tokenizer_already_in_the_run_directory_is_not_copied_over(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    """A resume must not overwrite the tokenizer the first half of the run was written with.

    The file in the run directory is the one that can decode this run's checkpoints. If
    the dataset were re-prepared between the two halves -- a new tokenizer, the same
    path -- copying again would replace a tokenizer that matches the weights with one
    that does not, and the mismatch would show up as garbled text rather than as an
    error. The fingerprint check catches a mismatched *dataset*; it cannot catch the run
    directory being edited underneath it.

    Asserted on the bytes, with a file that is deliberately not a tokenizer at all: a
    test that wrote a real one could not tell "left alone" from "copied over with an
    identical file".
    """
    from trainai.data.binarize import TOKENIZER_NAME

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / TOKENIZER_NAME).write_bytes(b"the tokenizer this run was trained with")
    trainer = make_trainer(prepared_dataset, run_dir, steps=2, warmup_steps=1, eval_every=0)

    assert trainer._copy_tokenizer() is None, "an existing tokenizer is not a problem to report"
    assert (run_dir / TOKENIZER_NAME).read_bytes() == b"the tokenizer this run was trained with"


def test_a_dataset_that_lives_nowhere_is_not_reported_as_a_missing_tokenizer(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    """An in-memory manifest has no directory to copy from, and that is not a warning.

    ``binarize_documents`` returns a manifest with a ``root``; a caller that built one
    itself, or loaded it and dropped the path, has ``None`` there. The test above sets
    ``root`` to a directory that does not exist, which is a different case and the one
    that *should* warn -- someone moved the dataset. ``None`` means nobody ever claimed
    there was a file, so there is nothing to report and nothing to copy.
    """
    from trainai.data.binarize import TOKENIZER_NAME

    run_dir = tmp_path / "run"
    trainer = make_trainer(prepared_dataset, run_dir, steps=2, warmup_steps=1, eval_every=0)
    trainer.dataset = replace(trainer.dataset, root=None)

    assert trainer._copy_tokenizer() is None
    assert not (run_dir / TOKENIZER_NAME).exists()


def test_the_run_says_out_loud_what_changed_across_a_resume(
    prepared_dataset: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The three lines a non-quiet resume prints, none of which had ever been printed.

    ``config_changes`` is already pinned by
    ``test_changing_a_setting_across_a_resume_is_allowed_and_reported``, but only as a
    dict on the trainer -- the run never said it. That is the half that matters: changing
    ``--steps`` across a resume is a legitimate thing to want and a terrible thing to
    have happen silently, because the remaining steps then follow a schedule the
    completed half never followed. Nothing else in the output would say so.

    The tokenizer warning is on the same session because it takes the same route, and
    both are only reachable with ``quiet=False`` -- every other test in this file runs
    quiet. Rich wraps at the console width, so the assertions go through ``flat``.

    The dataset is copied and its ``tokenizer.json`` deleted, rather than ``root`` being
    repointed the way the tests above do it: ``root`` resolves the shard paths as well,
    so a run -- as opposed to a direct ``_copy_tokenizer`` call -- cannot read a dataset
    that is not there. What is left is a real dataset missing exactly one file, which is
    what the warning is about.
    """
    import shutil

    whole = make_trainer(prepared_dataset, tmp_path / "whole", steps=24, checkpoint_every=12)
    whole.run()
    halfway = sorted((tmp_path / "whole" / "checkpoints").glob("step-*.pt"))[0]
    capsys.readouterr()

    untokenized = tmp_path / "untokenized"
    shutil.copytree(prepared_dataset.root, untokenized)
    (untokenized / "tokenizer.json").unlink()

    longer = Trainer(
        dataset=replace(prepared_dataset, root=untokenized),
        model_config=model_config(prepared_dataset),
        train_config=train_config(steps=48, lr=5e-4),
        run_dir=tmp_path / "longer",
        quiet=False,
    )
    longer.resume(halfway)
    longer.run()
    shown = flat(capsys.readouterr().out)

    assert "Resumed" in shown
    assert f"from step {12}" in shown
    assert "steps changed since the checkpoint: 24 -> 48" in shown
    assert "not a continuation of the original schedule" in shown
    assert "lr changed since the checkpoint" in shown, "every change is named, not just one"
    # The warning from _copy_tokenizer, which the two tests above only ever read as a
    # return value. Reaching the display is the part that tells anybody.
    assert "note:" in shown
    assert "will not carry one" in shown


def test_a_run_with_no_validation_reports_that_rather_than_a_number(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    trainer = make_trainer(prepared_dataset, tmp_path / "run", steps=6, eval_every=0)

    result = trainer.run()

    assert result.best_val_loss is None
    assert result.final_val_loss is None


# --------------------------------------------------------------------------- #
# Why a run has no held-out loss
#
# The absence used to be reported as a bare `None`, so whoever had to explain it
# guessed: the result panel printed "no validation split, or --eval-every 0" for
# every cause. Measured on a dataset with a 45-token validation split and
# --eval-every 2, both named causes were false and the real one -- too small for
# one window at that --seq-len -- went unsaid, though the loader had computed the
# context that would fit. These pin the reason to the cause.
# --------------------------------------------------------------------------- #
#: One window at ``seq_len`` 512 in the shared fixture, fewer than the batch size.
ONE_WINDOW_SEQ_LEN = 512

#: Too short for a single window: the fixture's validation split cannot reach it.
NO_WINDOW_SEQ_LEN = 1024


def test_a_validation_split_of_one_window_is_measured_not_dropped(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    """The regression test. A batch larger than the split is a short batch, not a refusal.

    Validation is a forward pass with no gradient, so ``evaluate_split`` has always
    shrunk the batch to what the split holds. The training loop passed the training
    batch size straight through instead, so the *training* batch size decided whether a
    held-out loss existed: on a 45-token split at --seq-len 16, --batch-size 1 reported
    4.9795 and --batch-size 4 reported nothing at all.
    """
    assert window_count(prepared_dataset.tokens("val"), ONE_WINDOW_SEQ_LEN) == 1
    trainer = make_trainer(
        prepared_dataset,
        tmp_path / "run",
        steps=2,
        warmup_steps=1,
        eval_every=1,
        batch_size=4,
        seq_len=ONE_WINDOW_SEQ_LEN,
        model={"seq_len": ONE_WINDOW_SEQ_LEN},
    )

    result = trainer.run()

    assert result.validation_skipped is None
    assert result.best_val_loss is not None
    assert math.isfinite(result.best_val_loss)


def test_the_training_batch_size_does_not_decide_whether_validation_happens(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    """The same split, the same context, two batch sizes: both measured."""
    losses = []
    for batch_size in (1, 4):
        trainer = make_trainer(
            prepared_dataset,
            tmp_path / f"run{batch_size}",
            steps=2,
            warmup_steps=1,
            eval_every=1,
            batch_size=batch_size,
            seq_len=ONE_WINDOW_SEQ_LEN,
            model={"seq_len": ONE_WINDOW_SEQ_LEN},
        )
        losses.append(trainer.run().best_val_loss)

    assert all(loss is not None for loss in losses)


def test_a_split_too_small_for_one_window_names_a_context_that_works(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    """The reason is stated, and the context it suggests has to yield a window."""
    val_tokens = prepared_dataset.tokens("val")
    assert window_count(val_tokens, NO_WINDOW_SEQ_LEN) == 0
    trainer = make_trainer(
        prepared_dataset,
        tmp_path / "run",
        steps=2,
        warmup_steps=1,
        eval_every=1,
        batch_size=1,
        seq_len=NO_WINDOW_SEQ_LEN,
        model={"seq_len": NO_WINDOW_SEQ_LEN},
    )

    result = trainer.run()

    assert result.best_val_loss is None
    skipped = result.validation_skipped
    assert skipped is not None
    assert str(val_tokens) in skipped.reason or f"{val_tokens:,}" in skipped.reason

    suggested = int(re.search(r"--seq-len (\d+)", skipped.hint or "").group(1))
    assert window_count(val_tokens, suggested) >= 1
    assert window_count(val_tokens, suggested + 1) == 0


def test_the_skipped_reason_is_the_loaders_own_words(prepared_dataset: Any, tmp_path: Path) -> None:
    """Not a second wording of the loader's refusal -- the refusal itself.

    Two places restating the same requirement is what let `trainai eval` claim a split
    needed half the tokens it needs. There is one source for it.
    """
    trainer = make_trainer(
        prepared_dataset,
        tmp_path / "run",
        steps=2,
        warmup_steps=1,
        eval_every=1,
        batch_size=1,
        seq_len=NO_WINDOW_SEQ_LEN,
        model={"seq_len": NO_WINDOW_SEQ_LEN},
    )

    skipped = trainer.run().validation_skipped

    with pytest.raises(DatasetError) as caught:
        open_split(prepared_dataset, "val", seq_len=NO_WINDOW_SEQ_LEN, batch_size=1, seed=7)

    assert skipped is not None
    assert skipped.reason == str(caught.value)
    assert skipped.hint == caught.value.hint


def test_eval_every_zero_says_so_and_does_not_blame_the_dataset(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    """The two causes used to be reported as one string that named both."""
    trainer = make_trainer(
        prepared_dataset, tmp_path / "run", steps=2, warmup_steps=1, eval_every=0
    )

    skipped = trainer.run().validation_skipped

    assert skipped is not None
    assert "--eval-every" in skipped.reason
    assert "no validation split" not in skipped.reason.lower()


def test_a_dataset_with_no_validation_split_says_that_and_how_to_get_one(
    dataset_without_validation: Any, tmp_path: Path
) -> None:
    """The third cause, and the only one where blaming the dataset is correct.

    Prepared with --val-fraction 0, so the split genuinely does not exist. The reason
    has to name that rather than --eval-every, which is 1 here, and it has to say how
    to get a split -- the run cannot make one.
    """
    assert dataset_without_validation.tokens("val") == 0
    trainer = make_trainer(
        dataset_without_validation, tmp_path / "run", steps=2, warmup_steps=1, eval_every=1
    )

    result = trainer.run()

    assert result.best_val_loss is None
    skipped = result.validation_skipped
    assert skipped is not None
    assert "no validation split" in skipped.reason.lower()
    assert "--eval-every" not in skipped.reason, "--eval-every is 1; it is not the cause"
    assert skipped.hint and "--val-fraction" in skipped.hint


def test_the_reason_reaches_the_metrics_file(prepared_dataset: Any, tmp_path: Path) -> None:
    """A run's own log has to answer this, not only the terminal that has scrolled away."""
    run_dir = tmp_path / "run"
    make_trainer(
        dataset=prepared_dataset, run_dir=run_dir, steps=2, warmup_steps=1, eval_every=0
    ).run()

    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    start = next(r for r in records if r["event"] == "start")
    end = next(r for r in records if r["event"] == "end")

    assert start["validation_skipped"]["reason"]
    assert end["validation_skipped"] == start["validation_skipped"]


def test_a_run_that_validates_records_no_skip_reason(prepared_dataset: Any, tmp_path: Path) -> None:
    """The negative control: the field stays empty when nothing was skipped."""
    run_dir = tmp_path / "run"
    result = make_trainer(prepared_dataset, run_dir, steps=2, warmup_steps=1, eval_every=1).run()

    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert result.validation_skipped is None
    assert result.to_dict()["validation_skipped"] is None
    assert all(r.get("validation_skipped") is None for r in records)


def test_a_damaged_validation_shard_stops_the_run(
    prepared_dataset: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken dataset is not "your corpus is small", and must not be swallowed.

    ``open_split`` reads no shard bytes today, so a truncated shard surfaces later, at
    the first read. This drives the clause directly: were the check ever moved earlier,
    a `DatasetFormatError` must still stop the run rather than be reported as a
    validation split too small to use -- which is what a broad ``except Exception``
    did, and what the run would then have trained happily through.
    """

    real = trainai.train.loop.open_split

    def refuse_only_val(dataset: Any, split: str, **kwargs: Any) -> Any:
        if split == "val":
            raise DatasetFormatError("Shard val_00000.bin is 89 bytes; the manifest says 90.")
        return real(dataset, split, **kwargs)

    monkeypatch.setattr("trainai.train.loop.open_split", refuse_only_val)
    trainer = make_trainer(
        prepared_dataset, tmp_path / "run", steps=2, warmup_steps=1, eval_every=1
    )

    with pytest.raises(DatasetFormatError):
        trainer.run()


def test_an_unexpected_error_is_not_reported_as_a_missing_split(
    prepared_dataset: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The handler catches ``DatasetError``, not everything.

    It used to be ``except Exception``, which meant a bug anywhere inside
    ``open_split`` -- a typo, a bad argument, a torch failure -- came out of the run as
    "this dataset has no validation split" and the run finished green. Narrowing the
    clause is only worth anything if something notices when it widens again.
    """

    real = trainai.train.loop.open_split

    def crash_only_val(dataset: Any, split: str, **kwargs: Any) -> Any:
        if split == "val":
            raise RuntimeError("not a dataset problem at all")
        return real(dataset, split, **kwargs)

    monkeypatch.setattr("trainai.train.loop.open_split", crash_only_val)
    trainer = make_trainer(
        prepared_dataset, tmp_path / "run", steps=2, warmup_steps=1, eval_every=1
    )

    with pytest.raises(RuntimeError, match="not a dataset problem"):
        trainer.run()


def test_grad_accum_produces_the_same_gradient_scale_as_a_bigger_batch(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    """Summing instead of averaging would multiply the effective learning rate."""
    norms = {}
    for accum, batch in ((1, 8), (2, 4)):
        trainer = make_trainer(
            prepared_dataset,
            tmp_path / f"accum-{accum}",
            steps=6,
            batch_size=batch,
            grad_accum=accum,
            grad_clip=0.0,
        )
        trainer.run()
        norms[accum] = float(
            torch.cat(
                [p.grad.flatten() for p in trainer.model.parameters() if p.grad is not None]
            ).norm()
        )

    assert 0.2 < norms[2] / norms[1] < 5.0, f"ratio {norms[2] / norms[1]}"


# --------------------------------------------------------------------------- #
# The loss mask
# --------------------------------------------------------------------------- #
# What these pin is a number, not a flag: a masked run and an unmasked run of the same
# checkpoint report losses that differ in the first decimal, and nothing in the output
# distinguishes them except the count the run says it averaged over. Every test here
# uses ``masked_dataset``, whose fixture asserts 0 < trained_tokens < total_tokens, so a
# trainer that ignored the mask entirely could not pass by accident.
def test_a_masked_run_scores_the_targets_and_says_how_many(
    masked_dataset: Any, tmp_path: Path
) -> None:
    """The share is the measurement. Reporting only ``loss_mask: true`` would say the
    mask was read, not that it selected anything."""
    trainer = make_trainer(masked_dataset, tmp_path / "masked", steps=4, seq_len=32)

    result = trainer.run()

    predicted = result.steps_completed * trainer.train_config.tokens_per_step
    assert result.loss_mask is True
    assert 0 < result.scored_tokens < predicted
    assert result.unscored_steps == 0
    assert result.to_dict()["scored_tokens"] == result.scored_tokens


def test_no_loss_mask_scores_every_position_of_the_same_dataset(
    masked_dataset: Any, tmp_path: Path
) -> None:
    """The deliberate opt-out, and the control on the test above.

    Same dataset, same steps: if the mask were being ignored, this run and the masked
    one would report the same count, and the assertion below would be the one to fail.
    """
    trainer = make_trainer(masked_dataset, tmp_path / "plain", steps=4, seq_len=32, loss_mask=False)

    result = trainer.run()

    assert result.loss_mask is False
    assert result.scored_tokens == result.steps_completed * trainer.train_config.tokens_per_step


def test_requiring_a_mask_the_dataset_has_not_got_is_refused(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    """Someone who asked for a mask in writing must not silently get every token."""
    trainer = make_trainer(prepared_dataset, tmp_path / "asked", steps=4, loss_mask=True)

    with pytest.raises(DatasetError) as caught:
        trainer.run()

    assert "--jsonl-messages-field" in (caught.value.hint or "")
    assert "--loss-mask" in (caught.value.hint or "") or "--loss-mask" in str(caught.value)


def _masked_batcher(dataset: Any, config: TrainConfig) -> Any:
    batcher, _stream = open_split(
        dataset,
        "train",
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        seed=config.seed,
        loss_mask=True,
    )
    return batcher


def test_accumulation_weights_a_micro_batch_by_what_it_scored(
    masked_dataset: Any, tmp_path: Path
) -> None:
    """A masked step must equal the single batch that holds all of its rows.

    Two micro-batches of a masked split score different numbers of positions, so
    ``1/grad_accum`` is the wrong weight: it gives a micro-batch that scored three
    tokens the same say as one that scored three hundred. The reference here is
    computed the only way that settles it -- one forward pass over the rows of both
    micro-batches concatenated, which is by definition the mean the step claims to be
    taking. Dropout is off, or the two passes would differ for a reason that is not
    the weighting.
    """
    trainer = make_trainer(
        masked_dataset,
        tmp_path / "weighted",
        steps=4,
        batch_size=2,
        grad_accum=2,
        seq_len=32,
        grad_clip=0.0,
        model={"dropout": 0.0},
    )
    config = trainer.train_config
    micro = [_masked_batcher(masked_dataset, config).batch(index) for index in range(2)]
    counts = [batch.trained_tokens for batch in micro]
    assert counts[0] != counts[1], "the fixture stopped exercising the case"

    inputs = torch.from_numpy(np.concatenate([b.inputs for b in micro])).long()
    targets = torch.from_numpy(np.concatenate([b.targets for b in micro])).long()
    mask = torch.from_numpy(np.concatenate([b.loss_mask for b in micro]))
    trainer.model.train()
    trainer.model.zero_grad(set_to_none=True)
    _, reference_loss, _ = trainer.model(inputs, targets, loss_mask=mask)
    reference_loss.backward()
    reference_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in trainer.model.named_parameters()
        if parameter.grad is not None
    }

    outcome = trainer._optimizer_step(_masked_batcher(masked_dataset, config), config)

    assert outcome.scored_tokens == sum(counts)
    assert outcome.loss == pytest.approx(float(reference_loss.detach()), rel=1e-6)
    for name, parameter in trainer.model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.allclose(parameter.grad, reference_grads[name], atol=1e-6), name


def test_a_step_that_scores_nothing_is_named_rather_than_counted_as_zero(
    masked_dataset: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A loss of 0.0 in the curve would be a minimum the run never reached.

    Arranged by blanking the mask of one step's batches rather than by hunting for a
    window that happens to be all prompt: what is under test is the branch, and a
    fixture that produced such a window by luck would stop doing so the day the
    tokenizer changed.
    """
    from trainai.data.loader import TokenBatcher

    blanked = 2
    original = TokenBatcher.batch

    def batch(self: Any, step: int) -> Any:
        drawn = original(self, step)
        if drawn.loss_mask is not None and step == blanked:
            return replace(drawn, loss_mask=np.zeros_like(drawn.loss_mask))
        return drawn

    monkeypatch.setattr(TokenBatcher, "batch", batch)
    trainer = make_trainer(masked_dataset, tmp_path / "unscored", steps=4, seq_len=32)

    result = trainer.run()

    records = [
        json.loads(line)
        for line in (tmp_path / "unscored" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    unscored = [record for record in records if record["event"] == "unscored"]
    assert result.unscored_steps == len(unscored) == 1
    assert unscored[0]["loss"] is None
    assert unscored[0]["step"] == blanked + 1
    assert "target" in unscored[0]["reason"]
    assert not any(
        record["event"] == "train" and record["step"] == blanked + 1 for record in records
    ), "a step that scored nothing must not appear in the loss curve at all"
    assert result.steps_completed == 4, "the step still counted, it just taught nothing"


def test_a_validation_batch_that_scores_nothing_is_left_out_of_the_mean(
    masked_dataset: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The evaluation half of the test above, and the same failure with worse reach.

    A batch with no target scores a loss of 0, and weighting that into the mean pulls the
    reported validation loss towards zero -- towards a minimum the model never reached.
    Unlike the training case, this number is the one checkpoints are chosen by, so a
    diluted validation loss picks the wrong best checkpoint.

    The mask of every second validation batch is blanked, and the reference is a run over
    only the batches that were left -- not a run over all of them. Blanking changes which
    tokens the mean covers as well as how they are weighted, so the two would differ for
    a legitimate reason; the reference isolates the one under test by dropping the same
    batches entirely. A skipped batch weighted in at 0 makes the mean lower, so the
    difference has a known sign as well as a known size.
    """
    from trainai.data.loader import TokenBatcher

    original = TokenBatcher.sequential_batches

    def blank_odd(self: Any, **kwargs: Any) -> Any:
        for batch in original(self, **kwargs):
            if batch.loss_mask is not None and batch.step % 2 == 1:
                yield replace(batch, loss_mask=np.zeros_like(batch.loss_mask))
            else:
                yield batch

    def drop_odd(self: Any, **kwargs: Any) -> Any:
        for batch in original(self, **kwargs):
            if batch.loss_mask is None or batch.step % 2 == 0:
                yield batch

    settings: dict[str, Any] = {"steps": 4, "seq_len": 32, "eval_every": 4, "eval_batches": 4}

    monkeypatch.setattr(TokenBatcher, "sequential_batches", drop_odd)
    dropped = make_trainer(masked_dataset, tmp_path / "dropped", **settings).run()

    monkeypatch.setattr(TokenBatcher, "sequential_batches", blank_odd)
    blanked = make_trainer(masked_dataset, tmp_path / "blanked", **settings).run()

    assert dropped.best_val_loss is not None and blanked.best_val_loss is not None
    assert blanked.best_val_loss == pytest.approx(dropped.best_val_loss), (
        "a batch with nothing to score changed the mean, so it was weighted in at 0"
    )


@pytest.mark.parametrize("masked", [False, True], ids=["no mask", "masked"])
def test_a_validation_split_with_nothing_to_score_is_refused_by_its_cause(
    prepared_dataset: Any,
    masked_dataset: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    masked: bool,
) -> None:
    """Two ways to score nothing at all, and they need different advice.

    Without a mask it means the split produced no batches, and a larger
    ``--val-fraction`` is the fix. With one it means the held-out documents are all
    context -- real text, really held out, with no assistant reply anywhere in it -- and
    the fix is either more data or training on every token. Telling a chat dataset's
    owner that their split is empty sends them to check a number that is fine.

    Reached by yielding no batch at all in the unmasked case and only blank-masked ones
    in the masked case, which is the difference the two messages are about. The loader's
    own guards refuse a split too small before it gets here, so this is a branch only a
    fixture can reach -- and it is the branch that would otherwise divide by zero.
    """
    from trainai.data.loader import TokenBatcher
    from trainai.errors import TrainingError

    original = TokenBatcher.sequential_batches

    def nothing(self: Any, **kwargs: Any) -> Any:
        for batch in original(self, **kwargs):
            if batch.loss_mask is None:
                return
            yield replace(batch, loss_mask=np.zeros_like(batch.loss_mask))

    monkeypatch.setattr(TokenBatcher, "sequential_batches", nothing)
    dataset = masked_dataset if masked else prepared_dataset
    trainer = make_trainer(
        dataset,
        tmp_path / "empty",
        steps=4,
        seq_len=32,
        warmup_steps=1,
        eval_every=4,
        eval_batches=2,
    )

    with pytest.raises(TrainingError) as caught:
        trainer.run()

    message, hint = str(caught.value), caught.value.hint or ""
    if masked:
        assert "no token of the validation split is a training target" in message.lower()
        assert "all context" in hint
        assert "no assistant" in hint
    else:
        assert "produced no batches" in message
        assert "all context" not in hint, "a maskless dataset has no context to blame"
    assert "--val-fraction" in hint, "both causes are fixable by holding out more"


# --------------------------------------------------------------------------- #
# The chat template in the checkpoint
# --------------------------------------------------------------------------- #
# The dataset's manifest already records the layout its shards were rendered in. The
# checkpoint copies it because a run has to stay usable after its dataset is deleted --
# the same reason the tokenizer is copied into the run directory -- and because the
# thing that needs the layout is inference, which happens later and elsewhere.
def test_a_run_on_a_chat_dataset_records_the_layout_in_its_checkpoint(
    masked_dataset: Any, tmp_path: Path
) -> None:
    from trainai.train.checkpoint import load_checkpoint

    trainer = make_trainer(
        masked_dataset, tmp_path / "chat", steps=4, seq_len=32, checkpoint_every=4
    )
    trainer.run()

    checkpoint = load_checkpoint(sorted((tmp_path / "chat" / "checkpoints").glob("step-*.pt"))[0])

    assert checkpoint.dataset["chat"] == masked_dataset.chat
    assert checkpoint.dataset["chat"]["version"] == 1
    assert checkpoint.dataset["chat"]["trained_roles"] == ["assistant"]


def test_a_run_on_prose_records_an_empty_layout(prepared_dataset: Any, tmp_path: Path) -> None:
    """The negative control. A checkpoint claiming a chat layout for a prose run would
    make ``trainai chat`` wrap plain text in role labels the model never saw."""
    from trainai.train.checkpoint import load_checkpoint

    trainer = make_trainer(prepared_dataset, tmp_path / "prose", steps=4, checkpoint_every=4)
    trainer.run()

    checkpoint = load_checkpoint(sorted((tmp_path / "prose" / "checkpoints").glob("step-*.pt"))[0])

    assert checkpoint.dataset["chat"] == {}


def test_the_layout_is_recorded_even_when_the_run_ignores_the_mask(
    masked_dataset: Any, tmp_path: Path
) -> None:
    """``--no-loss-mask`` changes what is scored, not what the shards say.

    The role labels are in the tokens either way, so the layout a model has to be
    prompted in is the same. Reading it off ``train_config.loss_mask`` instead of off the
    dataset is the mistake this pins -- the same one the manifest's ``typed_documents``
    counter exists to avoid one layer down.
    """
    from trainai.train.checkpoint import load_checkpoint

    trainer = make_trainer(
        masked_dataset,
        tmp_path / "unmasked",
        steps=4,
        seq_len=32,
        checkpoint_every=4,
        loss_mask=False,
    )
    trainer.run()

    checkpoint = load_checkpoint(
        sorted((tmp_path / "unmasked" / "checkpoints").glob("step-*.pt"))[0]
    )

    assert checkpoint.train_config.loss_mask is False
    assert checkpoint.dataset["chat"] == masked_dataset.chat


def test_a_checkpoint_from_before_the_layout_was_recorded_still_resumes(
    masked_dataset: Any, tmp_path: Path
) -> None:
    """Adding a key to the dataset block must not invalidate anyone's checkpoint.

    Built as the real pre-release case rather than the easy one: a run on a *chat*
    dataset, with ``chat`` stripped from the saved block, so the checkpoint says nothing
    about a layout while the dataset it is resumed against describes one in full. Only
    the fingerprint and the content hash are compared, so this resumes -- and a later
    change that compared the whole block would refuse it, which is a refusal to continue
    a long run over a key that says nothing about whether the tokens match.
    """
    whole = make_trainer(
        masked_dataset, tmp_path / "whole", steps=24, seq_len=32, checkpoint_every=12
    )
    whole.run()
    halfway = sorted((tmp_path / "whole" / "checkpoints").glob("step-*.pt"))[0]
    payload = torch.load(halfway, map_location="cpu", weights_only=False)
    assert payload["dataset"].pop("chat")["version"] == 1
    torch.save(payload, halfway)

    resumed = make_trainer(masked_dataset, tmp_path / "resumed", steps=24, seq_len=32)
    resumed.resume(halfway)
    assert resumed.step == 12
    result = resumed.run()

    assert result.steps_completed == 24
    assert masked_dataset.chat, "otherwise the two sides agreed and nothing was tested"


# --------------------------------------------------------------------------- #
# Exact resume
# --------------------------------------------------------------------------- #
def test_resuming_is_bitwise_identical_to_an_uninterrupted_run(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    """The M2 milestone criterion. See the module docstring for the correction."""
    steps = 24
    whole = make_trainer(prepared_dataset, tmp_path / "whole", steps=steps, checkpoint_every=12)
    reference_result = whole.run()
    reference = {k: v.clone() for k, v in whole.model.state_dict().items()}
    halfway = sorted((tmp_path / "whole" / "checkpoints").glob("step-*.pt"))[0]
    assert halfway.name.endswith("0000012.pt")

    # Same declared steps, so the same schedule. Stopped early, then continued.
    resumed = make_trainer(prepared_dataset, tmp_path / "resumed", steps=steps)
    resumed.resume(halfway)
    assert resumed.step == 12
    assert resumed.config_changes == {}
    resumed_result = resumed.run()

    assert resumed_result.steps_completed == steps
    after = resumed.model.state_dict()
    for key, value in reference.items():
        assert torch.equal(value, after[key]), f"{key} differs after resume"
    assert resumed_result.final_train_loss == reference_result.final_train_loss


def test_resuming_restores_the_optimizer_moments_exactly(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    """Weights alone are not enough: Adam's moments steer the next hundred steps."""
    steps = 24
    whole = make_trainer(prepared_dataset, tmp_path / "whole", steps=steps, checkpoint_every=12)
    whole.run()
    halfway = sorted((tmp_path / "whole" / "checkpoints").glob("step-*.pt"))[0]

    resumed = make_trainer(prepared_dataset, tmp_path / "resumed", steps=steps)
    resumed.resume(halfway)
    resumed.run()

    reference = whole.optimizer.state_dict()["state"]
    restored = resumed.optimizer.state_dict()["state"]
    for key in reference:
        for field in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(reference[key][field], restored[key][field])


@pytest.mark.gpu
def test_resuming_is_bitwise_identical_on_cuda_too(prepared_dataset: Any, tmp_path: Path) -> None:
    """The milestone test above pins ``device="cpu"``, which is not where runs happen.

    The claim the README makes -- "a resumed run continues bitwise-identically" -- is
    stated about the project, not about CPU, and the reader most likely to lean on it is
    resuming a GPU run they had to interrupt. CUDA is where it is least obvious: reduced
    precision and non-deterministic reductions are both in play, and two *fresh* runs of
    a long enough command on this hardware do drift apart (measured: three runs of the
    README's 600-step command agreed bit for bit to step 120, first disagreed at step
    130, and ended 1.5e-03 apart). Resume is not that, and it holds -- but nothing
    proved it here, so a regression in the GPU path could not have failed a test.

    This runs at the default precision rather than forcing fp32, because bf16 is what
    the resolver picks on a GPU and therefore what a user actually resumes under.
    """
    steps = 24
    whole = make_trainer(
        prepared_dataset, tmp_path / "whole", steps=steps, checkpoint_every=12, device="cuda"
    )
    reference_result = whole.run()
    assert whole.device.type == "cuda", "otherwise this silently duplicates the CPU test"
    reference = {k: v.clone() for k, v in whole.model.state_dict().items()}
    reference_moments = {
        key: {field: value[field].clone() for field in ("exp_avg", "exp_avg_sq")}
        for key, value in whole.optimizer.state_dict()["state"].items()
    }
    halfway = sorted((tmp_path / "whole" / "checkpoints").glob("step-*.pt"))[0]
    assert halfway.name.endswith("0000012.pt")

    resumed = make_trainer(prepared_dataset, tmp_path / "resumed", steps=steps, device="cuda")
    resumed.resume(halfway)
    assert resumed.step == 12
    resumed_result = resumed.run()

    assert resumed_result.steps_completed == steps
    after = resumed.model.state_dict()
    for key, value in reference.items():
        assert torch.equal(value, after[key]), f"{key} differs after resume on cuda"
    restored = resumed.optimizer.state_dict()["state"]
    for key, fields in reference_moments.items():
        for field, value in fields.items():
            assert torch.equal(value, restored[key][field]), f"{key}.{field} differs on cuda"
    assert resumed_result.final_train_loss == reference_result.final_train_loss


def test_changing_a_setting_across_a_resume_is_allowed_and_reported(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    """Legitimate to want, terrible to have happen silently: the remaining steps
    follow a different schedule than the half already completed."""
    whole = make_trainer(prepared_dataset, tmp_path / "whole", steps=24, checkpoint_every=12)
    whole.run()
    halfway = sorted((tmp_path / "whole" / "checkpoints").glob("step-*.pt"))[0]

    longer = make_trainer(prepared_dataset, tmp_path / "longer", steps=48)
    longer.resume(halfway)

    assert longer.config_changes["steps"] == (24, 48)


def test_resuming_against_a_different_dataset_is_refused(
    prepared_dataset: Any, tmp_path: Path, many_document_corpus: Path
) -> None:
    from trainai.data import Ingestor, binarize_documents, train_tokenizer
    from trainai.errors import CheckpointIncompatibleError

    whole = make_trainer(prepared_dataset, tmp_path / "whole", steps=4, checkpoint_every=4)
    whole.run()
    checkpoint = sorted((tmp_path / "whole" / "checkpoints").glob("step-*.pt"))[0]

    ingestor = Ingestor()
    documents = list(ingestor.documents(ingestor.discover(many_document_corpus)))
    other = binarize_documents(
        documents,
        tmp_path / "other",
        train_tokenizer((d.text for d in documents), vocab_size=prepared_dataset.vocab_size),
        seed=99,
        val_fraction=0.1,
    )
    trainer = Trainer(
        dataset=other,
        model_config=model_config(other),
        train_config=train_config(steps=4),
        run_dir=tmp_path / "wrong",
        quiet=True,
    )

    with pytest.raises(CheckpointIncompatibleError):
        trainer.resume(checkpoint)


# --------------------------------------------------------------------------- #
# Fine-tuning: initialise_from
# --------------------------------------------------------------------------- #
def _same_tokenizer_other_corpus(prepared: Any, corpus: Path, root: Path) -> Any:
    """A second dataset: this corpus, read through ``prepared``'s tokenizer.

    The pair that fine-tuning is *for*. Same ``tokenizer_fingerprint``, because the
    tokenizer object is the same one; different ``content_hash``, because the text is
    different. ``train --resume`` refuses this pair and is right to; ``initialise_from``
    accepts it, and the difference between those two answers is the entire point of
    having a separate method.
    """
    from trainai.data import Ingestor, binarize_documents
    from trainai.data.tokenizer import ByteLevelBPE

    assert prepared.root is not None
    tokenizer = ByteLevelBPE.load(prepared.root / "tokenizer.json")
    ingestor = Ingestor()
    documents = list(ingestor.documents(ingestor.discover(corpus)))
    return binarize_documents(
        documents, root, tokenizer, seed=99, val_fraction=0.1, ingest_options=ingestor.options
    )


def _a_base_checkpoint(prepared: Any, tmp_path: Path, **overrides: Any) -> Path:
    base = make_trainer(prepared, tmp_path / "base", steps=8, checkpoint_every=8, **overrides)
    base.run()
    return sorted((tmp_path / "base" / "checkpoints").glob("step-*.pt"))[-1]


def test_finetuning_across_a_content_hash_change_is_allowed(
    prepared_dataset: Any, tmp_path: Path, many_document_corpus: Path
) -> None:
    """The refusal that `resume` applies correctly is the thing fine-tuning must not inherit.

    ``load_checkpoint`` takes ``expect_dataset`` as an *optional* argument, and that is
    the whole seam: `resume` passes the dataset identity and gets the refusal, `initialise_from`
    passes nothing and gets the weights. Asserted together here, on one checkpoint and one
    dataset, because the two behaviours are only correct as a pair -- if either drifts to
    match the other, this fails.
    """
    from trainai.errors import CheckpointIncompatibleError

    checkpoint = _a_base_checkpoint(prepared_dataset, tmp_path)
    other = _same_tokenizer_other_corpus(prepared_dataset, many_document_corpus, tmp_path / "other")

    assert other.content_hash != prepared_dataset.content_hash
    assert other.tokenizer_fingerprint == prepared_dataset.tokenizer_fingerprint

    resuming = make_trainer(other, tmp_path / "resumed", model={"vocab_size": other.vocab_size})
    with pytest.raises(CheckpointIncompatibleError):
        resuming.resume(checkpoint)

    finetuning = make_trainer(other, tmp_path / "tuned", model={"vocab_size": other.vocab_size})
    loaded = finetuning.initialise_from(checkpoint)

    assert loaded.step == 8


def test_finetuning_starts_the_schedule_over_and_does_not_inherit_the_optimizer(
    prepared_dataset: Any, tmp_path: Path, many_document_corpus: Path
) -> None:
    """Weights come across; the optimizer's moments and the step counter do not.

    AdamW's exp_avg is an average of gradients taken on the *base* corpus. Carrying it
    into a different one preconditions the most fragile steps of the new run with stale
    curvature, and leaving ``step`` at the checkpoint's value would start the new run
    partway down a cosine decay computed for a different total. Both are asserted
    because both are silent when wrong: the loss curve looks ordinary either way.
    """
    checkpoint = _a_base_checkpoint(prepared_dataset, tmp_path)
    other = _same_tokenizer_other_corpus(prepared_dataset, many_document_corpus, tmp_path / "other")

    tuned = make_trainer(other, tmp_path / "tuned", model={"vocab_size": other.vocab_size})
    before = [p.detach().clone() for p in tuned.model.parameters()]
    tuned.initialise_from(checkpoint)

    assert tuned.step == 0
    assert all(not state for state in tuned.optimizer.state.values())
    assert any(
        not torch.equal(was, is_now)
        for was, is_now in zip(before, tuned.model.parameters(), strict=True)
    )


def test_finetuning_against_a_different_tokenizer_is_refused(
    prepared_dataset: Any, tmp_path: Path, many_document_corpus: Path
) -> None:
    """The half of the dataset check that fine-tuning must keep.

    A token id is a row index into the embedding matrix. Read the same shard bytes
    through a different tokenizer and every id selects a vector trained for some other
    text -- the run trains, the loss falls, and the model is worthless.

    The reported ``field`` is the assertion that carries this. A refusal is easy to get
    for the wrong reason: these two tokenizers also end up with different vocabulary
    sizes, because a BPE trainer stops when the corpus runs out of merges rather than
    when it reaches the target, so both fixtures asked for 435 and got 435 and 417. A
    size mismatch is caught anyway by the embedding shape, several layers down. The
    fingerprint is checked *first* precisely so that the equal-size case -- the one
    nothing else in the pipeline can detect -- is the one being refused here.
    """
    from trainai.data import Ingestor, binarize_documents, train_tokenizer
    from trainai.errors import CheckpointIncompatibleError

    checkpoint = _a_base_checkpoint(prepared_dataset, tmp_path)

    ingestor = Ingestor()
    documents = list(ingestor.documents(ingestor.discover(many_document_corpus)))
    stranger = binarize_documents(
        documents,
        tmp_path / "stranger",
        train_tokenizer((d.text for d in documents), vocab_size=prepared_dataset.vocab_size),
        seed=99,
        val_fraction=0.1,
    )

    assert stranger.tokenizer_fingerprint != prepared_dataset.tokenizer_fingerprint

    tuned = make_trainer(stranger, tmp_path / "tuned", model={"vocab_size": stranger.vocab_size})
    with pytest.raises(CheckpointIncompatibleError) as raised:
        tuned.initialise_from(checkpoint)

    assert "tokenizer" in str(raised.value)
    assert raised.value.details["field"] == "tokenizer_fingerprint"


def test_finetuning_records_which_model_it_started_from(
    prepared_dataset: Any, tmp_path: Path, many_document_corpus: Path
) -> None:
    """A fine-tuned run is not reproducible from its own directory alone.

    It is a function of a checkpoint that lives somewhere else, so the run record names
    that parent and its dataset rather than leaving the connection to a file path the
    user is free to move. Read back out of ``metrics.jsonl`` rather than off the
    attribute, because the file is what survives the process.
    """
    checkpoint = _a_base_checkpoint(prepared_dataset, tmp_path)
    other = _same_tokenizer_other_corpus(prepared_dataset, many_document_corpus, tmp_path / "other")

    tuned = make_trainer(other, tmp_path / "tuned", steps=4, model={"vocab_size": other.vocab_size})
    tuned.initialise_from(checkpoint)
    tuned.run()

    start = json.loads(
        (tmp_path / "tuned" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )

    assert start["resumed_from"] is None
    assert start["finetuned_from"] == str(checkpoint)
    assert start["parent"]["step"] == 8
    assert start["parent"]["dataset"]["content_hash"] == prepared_dataset.content_hash
    assert start["dataset"]["content_hash"] == other.content_hash


def test_the_equal_vocabulary_size_collision_is_still_refused(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    """The case the fixtures above cannot construct, checked directly.

    Two different tokenizers of the *same* vocabulary size produce a base model whose
    embedding matrix loads without complaint, and every downstream shape check passes.
    The fingerprint is the only thing that can tell them apart, so it is tested on a
    checkpoint whose dataset record is edited to differ in nothing else. Building this
    from real corpora would mean coaxing a BPE trainer into hitting an exact target,
    which it does not promise to do.
    """
    from trainai.errors import CheckpointIncompatibleError
    from trainai.train.checkpoint import load_checkpoint
    from trainai.train.loop import check_finetune_compatible

    checkpoint = load_checkpoint(_a_base_checkpoint(prepared_dataset, tmp_path))
    checkpoint.dataset["tokenizer_fingerprint"] = "0" * 64
    assert checkpoint.dataset["vocab_size"] == prepared_dataset.vocab_size

    with pytest.raises(CheckpointIncompatibleError) as raised:
        check_finetune_compatible(prepared_dataset, checkpoint)

    assert raised.value.details["field"] == "tokenizer_fingerprint"
    assert "data prepare --tokenizer" in (raised.value.hint or "")


def test_a_plain_run_records_no_parent(prepared_dataset: Any, tmp_path: Path) -> None:
    """The negative control: lineage fields exist and are empty when nothing was inherited.

    Without this, a bug that filled ``parent`` in from the run's own dataset would pass
    every test above.
    """
    trainer = make_trainer(prepared_dataset, tmp_path / "plain", steps=4)
    trainer.run()

    start = json.loads(
        (tmp_path / "plain" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )

    assert start["finetuned_from"] is None
    assert start["parent"] is None


# --------------------------------------------------------------------------- #
# Divergence
# --------------------------------------------------------------------------- #
def test_divergence_stops_the_run_and_leaves_the_weights_usable(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    """The step is not applied, so the model in memory is the last good one."""
    trainer = make_trainer(
        prepared_dataset,
        tmp_path / "diverge",
        steps=40,
        lr=1e3,
        warmup_steps=1,
        grad_clip=0.0,
    )

    with pytest.raises(TrainingDivergedError) as caught:
        trainer.run()

    assert "--lr" in (caught.value.hint or "")
    assert "not applied" in str(caught.value)
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters())


def test_divergence_is_recorded_and_the_run_closes_cleanly(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    trainer = make_trainer(
        prepared_dataset, tmp_path / "diverge", steps=40, lr=1e3, warmup_steps=1, grad_clip=0.0
    )

    with pytest.raises(TrainingDivergedError):
        trainer.run()

    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "diverge" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert "diverged" in events
    assert events[-1] == "end", "the metrics file was not closed properly"


def test_a_zero_grad_clip_is_named_in_the_divergence_hint(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    trainer = make_trainer(
        prepared_dataset, tmp_path / "diverge", steps=40, lr=1e3, warmup_steps=1, grad_clip=0.0
    )

    with pytest.raises(TrainingDivergedError) as caught:
        trainer.run()

    assert "--grad-clip" in (caught.value.hint or "")


# --------------------------------------------------------------------------- #
# The gradient scaler
# --------------------------------------------------------------------------- #
# Only fp16 needs a scaler, and fp16 needs a CUDA or ROCm card, so `resolve_precision`
# never asks for one on a CPU device -- which is every machine CI runs on. The scaled
# path therefore had no coverage at all: the loss scaled before the backward pass, the
# gradients unscaled before the clip, the optimizer stepped through the scaler, and an
# overflowing gradient treated as the scaler's business rather than as a diverging run.
#
# `with_a_scaler` patches the resolver rather than assigning to `trainer.scaler`, so the
# construction stays inside the test and the scaler is a real `torch.amp.GradScaler`
# doing real arithmetic -- it works on a CPU device, which is the whole reason this is
# testable here. The dtype stays fp32, because the scale factor is what is under test
# and autocast is a separate decision, tested further up. That is also the limit of
# these tests: they cover every decision the code makes around fp16 without
# reproducing fp16 itself.
def with_a_scaler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the next `Trainer` build itself a gradient scaler, on this CPU."""
    monkeypatch.setattr(
        trainai.train.loop,
        "resolve_precision",
        lambda requested, device: (torch.float32, True, "fp16 with gradient scaling (test)"),
    )


def overflow_the_gradients(trainer: Trainer) -> None:
    """Make one parameter's gradient arrive as inf, the way an fp16 overflow does.

    Spelled rather than provoked, deliberately. A learning rate high enough to overflow
    a gradient in this model overflows the loss as well, and then the earlier branch --
    a non-finite mean loss, tested above -- is what returns, so the case below would
    never be reached by turning the dial up.
    """
    parameter = next(iter(trainer.model.parameters()))
    parameter.register_hook(lambda grad: torch.full_like(grad, float("inf")))


def unmasked_batcher(dataset: Any, config: TrainConfig) -> Any:
    """`_masked_batcher`'s twin for a corpus with no mask, where every token scores."""
    batcher, _stream = open_split(
        dataset, "train", seq_len=config.seq_len, batch_size=config.batch_size, seed=config.seed
    )
    return batcher


def weights_of(trainer: Trainer) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in trainer.model.named_parameters()}


def test_the_scaler_is_built_for_the_device_the_run_is_on(
    prepared_dataset: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scaler built for a device that is not there disables itself and scales nothing.

    `torch.amp.GradScaler("cuda")` on a machine without CUDA does not raise. It warns
    once and switches itself off, after which `scale()` returns the loss untouched and
    the scale reads 1.0. An fp16 run then trains on the gradients that underflow to zero
    -- the one failure a scaler exists to prevent -- and the only trace of it is a line
    on stderr and a loss curve that does not move.

    The machine is stated rather than asked, and that is what makes this a test. CUDA is
    present here, so a scaler built for the wrong device would work by accident; on the
    machine CI runs on it would not, and a portability decision checked only where the
    hardware happens to agree is not checked at all.
    """
    no_cuda(monkeypatch)
    with_a_scaler(monkeypatch)

    trainer = make_trainer(prepared_dataset, tmp_path / "cpu-only")

    assert trainer.device.type == "cpu"
    assert trainer.scaler is not None
    assert trainer.scaler.is_enabled(), "a disabled scaler is a no-op that reports success"
    assert trainer.scaler.get_scale() == 65536.0, "a scale of 1.0 is a scaler doing nothing"


def test_a_scaled_step_lands_on_exactly_the_weights_an_unscaled_one_would(
    prepared_dataset: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mixed precision is meant to change how fast a step runs, not what it computes.

    Two trainers, the same seed and therefore the same initial weights, the same first
    batch, one step each -- and one of them multiplies the loss by 65536 before the
    backward pass. Every claim in the scaled branch shows up in the comparison, but the
    one worth naming is the order: `unscale_` runs before the clip, and if it did not,
    the norm reported here would come back 65536x too large and `--grad-clip 1.0` would
    fire on every step of every fp16 run, clipping gradients that never needed it.

    The agreement is exact rather than approximate, and that is not luck. 65536 is a
    power of two, so multiplying an fp32 gradient by it and dividing it back changes no
    bit of the mantissa. `pytest.approx` here would pass with the arithmetic subtly
    wrong, which is the failure this test exists to refuse.
    """
    plain = make_trainer(prepared_dataset, tmp_path / "plain", model={"dropout": 0.0})
    config = plain.train_config
    plain_outcome = plain._optimizer_step(unmasked_batcher(prepared_dataset, config), config)
    plain_weights = weights_of(plain)

    with_a_scaler(monkeypatch)
    mixed = make_trainer(prepared_dataset, tmp_path / "mixed", model={"dropout": 0.0})
    mixed_outcome = mixed._optimizer_step(unmasked_batcher(prepared_dataset, config), config)

    # The control. Without this the test would still pass if both runs were unscaled.
    assert plain.scaler is None
    assert mixed.scaler is not None
    assert mixed.scaler.get_scale() == 65536.0
    assert mixed.needs_scaler is True

    assert mixed_outcome.applied is True
    assert mixed_outcome.grad_norm == plain_outcome.grad_norm
    assert mixed_outcome.loss == plain_outcome.loss
    assert mixed_outcome.scored_tokens == plain_outcome.scored_tokens
    for name, parameter in mixed.model.named_parameters():
        assert torch.equal(parameter.detach(), plain_weights[name]), name


def test_an_overflowing_gradient_is_the_scalers_business_and_not_the_runs(
    prepared_dataset: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under fp16 an inf gradient is routine, and aborting on one would abort at step 1.

    A scaler discovers its scale is too high the only way anything can -- by
    overflowing. It then halves the scale and skips the weight update itself, which is
    why the step counts as applied here: as far as the run is concerned this step
    happened and the loss it reports is the real one it measured. The weights have not
    moved, and the divergence check must not fire. This is the mechanism working.
    """
    with_a_scaler(monkeypatch)
    trainer = make_trainer(prepared_dataset, tmp_path / "overflow", model={"dropout": 0.0})
    config = trainer.train_config
    overflow_the_gradients(trainer)
    before = weights_of(trainer)

    outcome = trainer._optimizer_step(unmasked_batcher(prepared_dataset, config), config)

    assert outcome.applied is True
    assert math.isinf(outcome.grad_norm or 0.0)
    assert math.isfinite(outcome.loss), "the loss was finite; only the gradient overflowed"
    assert trainer.scaler.get_scale() == 32768.0, "the scale was not backed off"
    for name, parameter in trainer.model.named_parameters():
        assert torch.equal(parameter.detach(), before[name]), name


def test_the_same_overflow_without_a_scaler_is_a_run_coming_apart(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    """Nothing will halve a scale that does not exist, so the step must be thrown away.

    The same injected overflow as above on a run with no scaler. The gradients are inf
    while the loss is still finite, which means the damage would only become visible on
    the *next* forward pass -- by which point the last good weights are gone. So the
    step is not applied, and the loss is reported as nan rather than as the finite
    number that was measured: it is that loss the run's divergence check reads, and a
    step that was discarded should not appear in the curve as a step that happened.
    """
    trainer = make_trainer(prepared_dataset, tmp_path / "overflow", model={"dropout": 0.0})
    config = trainer.train_config
    overflow_the_gradients(trainer)
    before = weights_of(trainer)

    outcome = trainer._optimizer_step(unmasked_batcher(prepared_dataset, config), config)

    assert trainer.scaler is None
    assert outcome.applied is False
    assert math.isnan(outcome.loss)
    assert math.isinf(outcome.grad_norm or 0.0), "the pre-clip norm is what is reported"
    assert outcome.scored_tokens > 0, "the step read its data; it declined to apply it"
    for name, parameter in trainer.model.named_parameters():
        assert torch.equal(parameter.detach(), before[name]), name


def test_a_run_whose_every_step_overflows_finishes_and_backs_the_scale_off(
    prepared_dataset: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four overflows in a row is the case a scaler is for, not a failure of the run.

    The scale starts high on purpose -- 65536 is torch's default -- because a scale too
    low silently loses the small gradients fp16 cannot represent, and the only cost of
    starting too high is the first few steps. Halving on each of four steps takes 65536
    to 4096 and the run completes. A run that aborted here would abort for every user
    whose card has no bf16: every GTX card, the RTX 20 series, and RDNA2 on AMD.
    """
    with_a_scaler(monkeypatch)
    trainer = make_trainer(prepared_dataset, tmp_path / "run", steps=4, checkpoint_every=4)
    assert trainer.scaler.get_scale() == 65536.0
    overflow_the_gradients(trainer)

    result = trainer.run()

    assert result.diverged is False
    assert result.steps_completed == 4
    assert trainer.scaler.get_scale() == 4096.0, "one halving per overflowing step"
    assert (tmp_path / "run" / "checkpoints").is_dir()


def test_resuming_restores_the_scale_the_first_run_had_reached(
    prepared_dataset: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scale is learned state, exactly like the optimizer's moments.

    It was paid for with the overflows of the first run: four steps that read their
    data, computed their gradients and applied none of it. A resumed run that starts
    back at 65536 pays for the same four steps again, and does it every time the run is
    interrupted. The fresh trainer's own scale is asserted first, so this cannot pass
    by the two numbers happening to agree.
    """
    with_a_scaler(monkeypatch)
    first = make_trainer(prepared_dataset, tmp_path / "first", steps=4, checkpoint_every=4)
    overflow_the_gradients(first)
    first.run()
    checkpoint_path = sorted((tmp_path / "first" / "checkpoints").glob("step-*.pt"))[0]

    second = make_trainer(prepared_dataset, tmp_path / "second", steps=8, checkpoint_every=4)
    assert second.scaler.get_scale() == 65536.0

    checkpoint = second.resume(checkpoint_path)

    assert checkpoint.scaler_state is not None, "the scale was never written"
    assert checkpoint.scaler_state["scale"] == 4096.0
    assert second.scaler.get_scale() == 4096.0


def test_an_fp32_checkpoint_resumed_into_a_scaled_run_keeps_the_default_scale(
    prepared_dataset: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`train --resume --precision fp16` on a run that was started without it.

    An fp32 run has no scaler, so it writes no scale, so there is nothing for the new
    run's scaler to restore and it keeps the default it was built with. What makes that
    work is the second half of an `and`; with only the first half this raises
    `AttributeError` on a `None` state dict and the resume fails on a file that is
    perfectly good.
    """
    plain = make_trainer(prepared_dataset, tmp_path / "fp32", steps=4, checkpoint_every=4)
    assert plain.scaler is None
    plain.run()
    checkpoint_path = sorted((tmp_path / "fp32" / "checkpoints").glob("step-*.pt"))[0]

    with_a_scaler(monkeypatch)
    mixed = make_trainer(prepared_dataset, tmp_path / "fp16", steps=8, checkpoint_every=4)

    checkpoint = mixed.resume(checkpoint_path)

    assert checkpoint.scaler_state is None
    assert mixed.scaler.get_scale() == 65536.0
    assert mixed.step == 4, "the rest of the checkpoint was restored"


# --------------------------------------------------------------------------- #
# Refusals at construction
# --------------------------------------------------------------------------- #
def test_a_sequence_longer_than_the_model_context_is_refused(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    with pytest.raises(TrainingError) as caught:
        make_trainer(prepared_dataset, tmp_path / "run", seq_len=128, model={"seq_len": 64})

    assert caught.value.details["model_seq_len"] == 64


def test_a_vocabulary_smaller_than_the_dataset_is_refused(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    """Some tokens would have no embedding, and the loss would not say so.

    This used to assert ``"--vocab-size" in hint`` and so pinned the wrong advice in
    place: there is no ``--vocab-size`` on ``trainai train``, and following the hint
    exited 2 with "No such option". The flag exists only on ``trainai data prepare``,
    where it is a target the tokenizer can fall short of, so it is not the lever either.
    What the hint has to name is the number and where it comes from.
    """
    with pytest.raises(TrainingError) as caught:
        make_trainer(prepared_dataset, tmp_path / "run", model={"vocab_size": 8})

    hint = caught.value.hint or ""

    assert f"vocab_size={prepared_dataset.vocab_size}" in hint, (
        "the hint has to name the number the caller should build with"
    )
    assert "--vocab-size" not in hint, "no flag sets this, so the hint may not spell it like one"
    assert caught.value.details["dataset_vocab"] == prepared_dataset.vocab_size


#: Both refusals ``check_configs_agree`` can raise, as (kwargs, expected substring) --
#: enough to reach each branch through the public entry point.
AGREEMENT_REFUSALS = [
    ({"seq_len": 128, "model": {"seq_len": 64}}, "exceeds the model's context"),
    ({"model": {"vocab_size": 8}}, "smaller than"),
]


@pytest.mark.parametrize(("kwargs", "expected"), AGREEMENT_REFUSALS)
def test_every_flag_these_refusals_name_is_a_flag_of_trainai_train(
    prepared_dataset: Any, tmp_path: Path, kwargs: dict[str, Any], expected: str
) -> None:
    """A refusal raised by ``trainai train`` may only name flags ``train`` accepts.

    The vocabulary hint named ``--vocab-size``, which is real -- on ``trainai data
    prepare``. Grepping the repo for the flag therefore found it and said nothing was
    wrong; only running the advice showed it. So the check is against the command's own
    parameter list rather than against the repo's text, and it covers every guard in
    ``check_configs_agree`` so the next one added is checked too.

    Scoped to this function deliberately: it is reached from exactly one command, which
    is what makes "the flags must be that command's" a true statement rather than a
    guess. A hint raised from somewhere reachable by several commands cannot be held to
    this without knowing which one the user ran.
    """
    import typer.main

    from trainai.cli._click import is_group
    from trainai.cli.main import app

    command = typer.main.get_command(app)
    assert is_group(command)
    train_flags = {opt for parameter in command.commands["train"].params for opt in parameter.opts}

    with pytest.raises(TrainingError) as caught:
        make_trainer(prepared_dataset, tmp_path / "run", **kwargs)

    assert expected in str(caught.value), "the parametrization reached the wrong branch"
    text = f"{caught.value} {caught.value.hint or ''}"
    named = set(re.findall(r"--[a-z][a-z-]+", text))
    unknown = named - train_flags

    assert not unknown, (
        f"{sorted(unknown)} named by a refusal from `trainai train`, which has no such "
        f"option. Running the advice exits 2. Flags `train` does accept: "
        f"{sorted(f for f in train_flags if f.startswith('--'))}"
    )


def test_the_configuration_checks_need_no_device_and_build_no_model(
    monkeypatch: pytest.MonkeyPatch, prepared_dataset: Any
) -> None:
    """Which is the property that lets ``--dry-run`` apply them.

    Both refusals used to live inline in ``Trainer.__init__``, so the only way to reach
    them was to construct a Trainer -- which resolves a device and allocates a model.
    A dry run does neither and returned before building one, so it printed a full plan
    and exited 0 for `--context 128 --seq-len 512` while the same flags without
    ``--dry-run`` exited 6. Extracting them is only a fix if they stay cheap, and that
    is what this pins: with the device resolver and the model class both booby-trapped,
    the checks still run.
    """

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a configuration check reached the device or the model")

    monkeypatch.setattr(trainai.train.loop, "resolve_device", explode)
    monkeypatch.setattr(trainai.train.loop, "GPT", explode)

    fits = model_config(prepared_dataset, seq_len=64)
    assert check_configs_agree(prepared_dataset, fits, train_config(seq_len=64)) is None

    with pytest.raises(TrainingError) as caught:
        check_configs_agree(prepared_dataset, fits, train_config(seq_len=128))

    assert caught.value.details == {"train_seq_len": 128, "model_seq_len": 64}


def test_a_larger_vocabulary_than_the_dataset_is_allowed(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    """Wasteful but not wrong: the extra rows simply never receive gradient."""
    trainer = make_trainer(
        prepared_dataset,
        tmp_path / "run",
        steps=4,
        model={"vocab_size": prepared_dataset.vocab_size + 64},
    )

    assert trainer.run().steps_completed == 4


# --------------------------------------------------------------------------- #
# Data budget
# --------------------------------------------------------------------------- #
def budget(**overrides: Any) -> DataBudget:
    settings: dict[str, Any] = {
        "train_tokens": 100_000,
        "val_tokens": 5_000,
        "parameters": 1_000_000,
        "non_embedding_parameters": 700_000,
        "tokens_per_step": 1_000,
        "steps": 100,
    }
    settings.update(overrides)
    return DataBudget(**settings)


def test_epochs_is_tokens_processed_over_training_tokens() -> None:
    assert budget(steps=100, tokens_per_step=1000, train_tokens=100_000).epochs == 1.0
    assert budget(steps=400, tokens_per_step=1000, train_tokens=100_000).epochs == 4.0


def test_tokens_per_parameter_is_the_data_starvation_measure() -> None:
    assert budget(train_tokens=20_000_000, parameters=1_000_000).tokens_per_parameter == 20.0


def test_a_run_that_will_memorise_says_so_and_names_a_step() -> None:
    """Measured: a 32-epoch run's best validation loss was at epoch 8, not 32."""
    heavy = budget(steps=3200, tokens_per_step=1000, train_tokens=100_000)

    assert heavy.will_memorise
    notes = heavy.warnings()
    assert any("passes over the training split" in note for note in notes)
    assert any("best checkpoint" in note for note in notes)


def test_a_well_proportioned_run_has_nothing_to_warn_about() -> None:
    """The negative control for every note in :meth:`DataBudget.warnings`.

    The original fixture here processed 100,000 tokens of a 100,000,000-token split and
    called itself well proportioned, because it was written to dodge only the two checks
    that existed: it is not memorising, and it is not data-starved. It was also reading
    0.1% of the corpus, which nothing measured until the unread-tokens note was added.
    Kept at two full passes now, so this stays a run with genuinely nothing wrong with
    it rather than one whose problems happen to be unmeasured.
    """
    fine = budget(
        train_tokens=10_000_000,
        parameters=1_000_000,
        steps=2_000,
        tokens_per_step=10_000,
        val_tokens=1_000,
    )

    assert fine.epochs == 2.0
    assert not fine.will_memorise
    assert not fine.is_data_limited
    assert not fine.leaves_corpus_unread
    assert fine.warnings() == []


def test_a_data_starved_run_quotes_the_ratio_and_the_target() -> None:
    starved = budget(train_tokens=1_000, parameters=10_000_000)

    assert starved.is_data_limited
    note = next(n for n in starved.warnings() if "tokens per parameter" in n)
    assert str(int(TOKENS_PER_PARAMETER_TARGET)) in note
    assert f"{starved.compute_optimal_tokens:,}" in note


def test_no_validation_split_is_flagged_as_a_missing_measurement() -> None:
    blind = budget(val_tokens=0, train_tokens=100_000_000, parameters=1_000_000)

    assert any("no validation split" in note for note in blind.warnings())


def test_the_memorising_threshold_is_the_documented_constant() -> None:
    just_under = budget(
        steps=int(EPOCHS_BEFORE_MEMORISING * 100) - 1, tokens_per_step=1000, train_tokens=100_000
    )
    just_over = budget(
        steps=int(EPOCHS_BEFORE_MEMORISING * 100) + 1, tokens_per_step=1000, train_tokens=100_000
    )

    assert not just_under.will_memorise
    assert just_over.will_memorise


def test_the_budget_does_not_divide_by_zero() -> None:
    empty = budget(train_tokens=0, parameters=0, tokens_per_step=0)

    assert empty.epochs == 0.0
    assert empty.tokens_per_parameter == 0.0
    assert empty.steps_per_epoch == 0.0
    # Not a division, so it carries no guard of its own: ``max`` floors the
    # subtraction. Asserted here so removing that guard stays a safe deletion.
    assert empty.tokens_never_read == 0
    assert budget(train_tokens=0, tokens_per_step=1_000, steps=100).tokens_never_read == 0


def test_a_data_limited_finetune_is_told_the_ratio_means_something_else() -> None:
    """Same arithmetic, different advice -- and the advice is the whole point.

    A fine-tune corpus is small relative to the parameter count almost by
    definition, so ``is_data_limited`` fires on essentially every fine-tune. The
    from-scratch remedy it would otherwise print -- "a smaller model would
    generalise better" -- is not available to someone whose model shape is fixed
    by the checkpoint they are tuning, so following it is impossible and reading
    it is alarming for a run that is behaving normally.

    Only ``test_a_data_starved_run_quotes_the_ratio_and_the_target`` had reached
    this branch of ``warnings()``, and it takes the ``else``. Both budgets here
    are identical apart from ``finetune``, so the assertions below are about the
    branch and not about the numbers.
    """
    settings = {"train_tokens": 1_000_000, "parameters": 1_000_000}
    tuned = budget(**settings, finetune=True)
    scratch = budget(**settings)

    assert tuned.is_data_limited and scratch.is_data_limited
    assert tuned.tokens_per_parameter == scratch.tokens_per_parameter

    tuned_note = next(n for n in tuned.warnings() if "tokens per parameter" in n)
    scratch_note = next(n for n in scratch.warnings() if "tokens per parameter" in n)

    assert "expected when fine-tuning" in tuned_note
    assert "lower --lr" in tuned_note
    assert "smaller model" not in tuned_note, "the from-scratch remedy is not available here"

    # The control: the other branch really does say the thing this one withholds.
    assert "smaller model" in scratch_note
    assert "fine-tuning" not in scratch_note

    # Both quote the same measurement, so only the judgement differs.
    assert f"{tuned.tokens_per_parameter:.2f} training tokens per parameter" in tuned_note
    assert f"{scratch.tokens_per_parameter:.2f} training tokens per parameter" in scratch_note


def test_the_budget_is_json_safe() -> None:
    assert json.loads(json.dumps(budget().to_dict()))["epochs"] == 1.0


def test_the_trainer_computes_its_own_budget(prepared_dataset: Any, tmp_path: Path) -> None:
    trainer = make_trainer(prepared_dataset, tmp_path / "run", steps=10)

    assert trainer.budget.train_tokens == prepared_dataset.tokens("train")
    assert trainer.budget.parameters == trainer.model.parameter_count()
    assert trainer.budget.steps == 10


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #
def test_perplexity_is_the_exponential_of_the_loss() -> None:
    assert perplexity(0.0) == 1.0
    assert perplexity(math.log(50)) == pytest.approx(50.0)


def test_an_absurd_loss_reports_very_high_rather_than_a_number() -> None:
    """exp(20) is 485 million. Printing it implies precision the loss does not have."""
    assert not math.isfinite(perplexity(100.0))
    assert format_perplexity(100.0) == "very high"
    assert format_perplexity(float("nan")) == "very high"


def test_a_small_perplexity_is_formatted_with_a_decimal() -> None:
    assert format_perplexity(math.log(12.3)) == "12.3"


def test_the_throughput_meter_measures_rather_than_computes() -> None:
    meter = ThroughputMeter()

    meter.record(tokens=1000, seconds=0.5)
    meter.record(tokens=1000, seconds=0.5)

    assert meter.tokens_per_second == pytest.approx(2000.0)
    assert meter.total_tokens == 2000


def test_an_empty_throughput_meter_reports_zero_not_a_guess() -> None:
    meter = ThroughputMeter()

    assert meter.tokens_per_second == 0.0
    assert math.isnan(meter.seconds_remaining(100, 1000))


def test_time_remaining_uses_the_observed_rate() -> None:
    meter = ThroughputMeter()
    meter.record(tokens=1000, seconds=1.0)

    assert meter.seconds_remaining(10, 1000) == pytest.approx(10.0)
    assert meter.seconds_remaining(0, 1000) == 0.0


def test_the_metrics_writer_flushes_every_record(tmp_path: Path) -> None:
    """A process killed mid-run must leave everything up to the last record on disk."""
    writer = MetricsWriter(tmp_path / "metrics.jsonl")

    writer.write("train", 1, loss=2.5)

    assert (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").strip()
    writer.close()


def test_the_metrics_writer_appends_across_runs(tmp_path: Path) -> None:
    """A resumed run adds to the history rather than replacing it."""
    path = tmp_path / "metrics.jsonl"
    with MetricsWriter(path) as writer:
        writer.write("train", 1, loss=1.0)
    with MetricsWriter(path) as writer:
        writer.write("train", 2, loss=0.5)

    assert len(MetricsWriter(path, enabled=False).read_all()) == 2


def test_a_disabled_writer_writes_nothing(tmp_path: Path) -> None:
    writer = MetricsWriter(tmp_path / "metrics.jsonl", enabled=False)

    writer.write("train", 1, loss=1.0)
    writer.close()

    assert not (tmp_path / "metrics.jsonl").exists()


def test_a_truncated_final_line_is_skipped_not_fatal(tmp_path: Path) -> None:
    """What a process killed mid-write leaves. Every complete record is still good."""
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"event":"train","step":1}\n{"event":"tr', encoding="utf-8", newline="\n")

    records = MetricsWriter(path, enabled=False).read_all()

    assert len(records) == 1
    assert records[0]["step"] == 1


def test_records_are_written_as_one_line_of_ascii_json(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    with MetricsWriter(path) as writer:
        writer.write("train", 1, loss=1.0, note="plain")

    raw = path.read_bytes()
    assert raw.count(b"\n") == 1
    assert b"\r\n" not in raw


def test_a_numpy_scalar_is_logged_as_a_number_not_as_its_repr(tmp_path: Path) -> None:
    """The reason ``json.dumps`` is given a ``default=`` at all.

    Every loss in this project arrives from torch, and ``tensor.item()`` returns a
    Python float -- but a metric computed with numpy along the way is a
    ``numpy.float32``, which ``json.dumps`` refuses outright. Without the fallback
    the write raises and the run dies over a log line.

    The fallback has two branches and only one of them is correct here, which is what
    this test separates. Unwrapping via ``.item()`` writes ``"loss":0.5`` -- a JSON
    number that ``summarise_run`` and any plotting code can do arithmetic on. Falling
    through to ``str(value)`` writes ``"loss":"0.5"`` instead: still valid JSON, still
    parses, still *looks* right in the file, and then every consumer that compares or
    averages losses is comparing strings. Asserting on the raw bytes is what tells the
    two apart, since both branches contain the characters ``0.5``.
    """
    path = tmp_path / "metrics.jsonl"

    with MetricsWriter(path) as writer:
        writer.write("train", 1, loss=np.float32(0.5))

    raw = path.read_bytes()
    assert b'"loss":0.5' in raw, f"logged as text, not a number: {raw!r}"
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert isinstance(record["loss"], float)
    assert record["loss"] == pytest.approx(0.5)


def test_a_field_json_cannot_encode_is_stringified_rather_than_ending_the_run(
    tmp_path: Path,
) -> None:
    """``write`` accepts unknown fields, so it has to survive an unknown *type*.

    The docstring on :meth:`MetricsWriter.write` says unknown fields are "allowed and
    encouraged", which is an invitation to pass whatever describes the moment -- and a
    ``Path`` is the likeliest thing to turn up that way. It has no ``.item()`` and
    ``json.dumps`` cannot encode it, so it reaches the last line of the fallback.

    Stringifying is the right answer only because of what the alternative costs. An
    exception here propagates out of the training step that logged it, killing a run
    that was otherwise fine, to protect a log. So the test writes a second record
    afterwards and reads both back: the claim is not just that the odd value was
    encoded, but that the run kept going.
    """
    path = tmp_path / "metrics.jsonl"

    with MetricsWriter(path) as writer:
        writer.write("train", 1, out_dir=tmp_path / "run")
        writer.write("train", 2, loss=1.0)

    records = MetricsWriter(path, enabled=False).read_all()
    assert [record["step"] for record in records] == [1, 2]
    assert records[0]["out_dir"] == str(tmp_path / "run")


def test_summarise_reduces_a_log_to_the_numbers_worth_reporting() -> None:
    records = [
        {"event": "start", "step": 0},
        {"event": "train", "step": 1, "loss": 5.0, "tokens": 100, "elapsed": 1.0},
        {"event": "eval", "step": 1, "val_loss": 4.5},
        {"event": "train", "step": 2, "loss": 3.0, "tokens": 200, "elapsed": 2.0},
        {"event": "eval", "step": 2, "val_loss": 4.8},
        {"event": "end", "step": 2},
    ]

    summary = summarise_run(records)

    assert summary["first_loss"] == 5.0
    assert summary["final_loss"] == 3.0
    assert summary["best_val_loss"] == 4.5
    assert summary["best_val_step"] == 1
    assert summary["final_val_loss"] == 4.8
    assert summary["tokens_seen"] == 200
    assert summary["elapsed_seconds"] == 2.0


def test_summarising_an_empty_log_invents_nothing() -> None:
    """Zeros that look like measurements are worse than an absent field."""
    summary = summarise_run([])

    assert summary == {"train_records": 0, "eval_records": 0}


# --------------------------------------------------------------------------- #
# A damaged metrics log
# --------------------------------------------------------------------------- #
#: A log that is entirely valid, so a case below can damage exactly one thing.
GOOD_METRICS: list[dict[str, Any]] = [
    {"event": "train", "step": 1, "time": 1.0, "loss": 4.0, "tokens": 100, "elapsed": 1.0},
    {"event": "train", "step": 2, "time": 2.0, "loss": 3.0, "tokens": 200, "elapsed": 2.0},
    {"event": "eval", "step": 2, "time": 2.5, "val_loss": 3.5},
]


def metrics_bytes(*extra: bytes) -> bytes:
    """``GOOD_METRICS`` as JSONL, with ``extra`` raw lines appended."""
    lines = [json.dumps(record, separators=(",", ":")).encode() for record in GOOD_METRICS]
    return b"\n".join(lines + list(extra)) + b"\n"


#: (what the edit is, the bytes on disk). Every one of these ended a *finished* run
#: with a bare ``AttributeError``, ``TypeError`` or ``UnicodeDecodeError`` and exit 1:
#: ``read_all`` runs after the checkpoints are on disk, so the traceback was thrown at
#: someone whose model was already saved. Numbers as well as strings for the "not an
#: object" cases -- ``"event" not in "text"`` is merely False, so a suite that tries
#: only strings leaves the check that stops a number unpinned.
METRICS_DAMAGE: list[tuple[str, bytes]] = [
    ("a line that is a number", metrics_bytes(b"3")),
    ("a line that is a string", metrics_bytes(b'"train"')),
    ("a line that is an array", metrics_bytes(b"[1,2]")),
    ("a line that is null", metrics_bytes(b"null")),
    ("a line that is true", metrics_bytes(b"true")),
    ("a partial final line", metrics_bytes(b'{"event":"train","step":3,"lo')),
    # The writer appends, so a resumed run glues its first record onto the fragment the
    # killed one left. The damage is mid-file, not last.
    ("two records glued by a resume", metrics_bytes(b'{"event":"train","st{"step":3}')),
    # UnicodeDecodeError is a ValueError, so the JSONDecodeError guard never saw this.
    ("bytes that are not utf-8", metrics_bytes(b'{"event":"train","note":"caf\xe9"}')),
    ("the whole file is one json array", b'[{"event":"train","step":1,"loss":4.0}]\n'),
    ("a single NUL byte", b"\x00\n"),
]


@pytest.mark.parametrize(("case", "raw"), METRICS_DAMAGE, ids=[c for c, _ in METRICS_DAMAGE])
def test_a_damaged_metrics_log_still_summarises(tmp_path: Path, case: str, raw: bytes) -> None:
    """Nothing here refuses. This file describes what already happened.

    It is the most derived thing a run writes -- no model depends on it -- so losing a
    completed run's summary to a traceback over a log is the worse of the two outcomes.
    """
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(raw)
    writer = MetricsWriter(path, enabled=False)

    records = writer.read_all()
    summary = summarise_run(records)

    # Every case damages exactly one line, and the good records around it survive it.
    assert writer.unreadable_lines == 1, f"{case} was dropped without being counted"
    assert records in (GOOD_METRICS, []), case
    assert summary["train_records"] == len(records) - (1 if records else 0)


def test_dropped_metric_lines_are_reported_in_the_run_summary(tmp_path: Path) -> None:
    """Counted, not absorbed. This is the only place anyone would find out."""
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(metrics_bytes(b"3", b'"train"'))
    writer = MetricsWriter(path, enabled=False)

    writer.read_all()

    assert writer.unreadable_lines == 2

    # The count describes the last read, not every read there has ever been.
    path.write_bytes(metrics_bytes())
    writer.read_all()
    assert writer.unreadable_lines == 0


def test_a_blank_line_is_not_counted_as_damage(tmp_path: Path) -> None:
    """A blank line is nothing lost, and saying otherwise is a false alarm.

    Every case in ``METRICS_DAMAGE`` costs a record and is counted for it, because
    ``unreadable_metric_lines`` in the run summary is the only place a user finds out
    that their summary was computed from fewer records than the file has lines. That
    makes a wrong count expensive in the other direction too: a blank line loses
    nothing, and reporting one would send someone looking through a log for corruption
    that is not in it.

    Blank lines are the ordinary result of the writer being killed between the record
    and its newline and then resumed, and of anything that concatenates two logs.
    """
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(metrics_bytes(b"", b"   ", b"\t"))
    writer = MetricsWriter(path, enabled=False)

    records = writer.read_all()

    assert records == GOOD_METRICS, "the good records survive a blank line between them"
    assert writer.unreadable_lines == 0, "a blank line is not a dropped record"
    assert summarise_run(records)["train_records"] == len(GOOD_METRICS) - 1


def test_a_run_that_wrote_no_metrics_summarises_as_empty(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    """``write_metrics=False`` and then a summary read from a file that is not there.

    ``run`` calls ``read_all`` unconditionally after the last checkpoint is on disk, so
    this branch is on the path of every run built with ``write_metrics=False`` -- a
    public constructor argument that nothing in this repository passes, which is why it
    had never run. The planner builds Trainers to measure a few steps and throw away,
    and a measurement harness is exactly the caller that does not want a log.

    Returning ``[]`` rather than raising matters for the same reason the damage cases
    do: this happens after the model is saved, so a ``FileNotFoundError`` here would end
    a finished run with a traceback about a log nobody asked for.
    """
    run_dir = tmp_path / "quiet"
    result = Trainer(
        dataset=prepared_dataset,
        model_config=ModelConfig(
            vocab_size=prepared_dataset.vocab_size,
            n_layer=1,
            n_head=2,
            d_model=32,
            seq_len=32,
        ),
        train_config=TrainConfig(
            steps=2,
            batch_size=2,
            seq_len=32,
            lr=1e-3,
            warmup_steps=1,
            eval_every=0,
            checkpoint_every=2,
            log_every=2,
            seed=7,
            device="cpu",
        ),
        run_dir=run_dir,
        quiet=True,
        write_metrics=False,
    ).run()

    assert result.steps_completed == 2, "the run has to finish, not survive"
    assert not (run_dir / "metrics.jsonl").exists(), "nothing asked for a log"
    assert result.summary == {"train_records": 0, "eval_records": 0}, (
        "an absent log counts nothing; it does not invent a loss it never read"
    )
    assert "unreadable_metric_lines" not in result.summary
    assert "final_loss" not in result.summary and "best_val_loss" not in result.summary

    # And the same thing read directly, which is what the trainer just did.
    writer = MetricsWriter(run_dir / "metrics.jsonl", enabled=False)
    assert writer.read_all() == []
    assert writer.unreadable_lines == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [("loss", "nan"), ("loss", None), ("loss", True), ("val_loss", "3.0"), ("val_loss", None)],
)
def test_a_loss_that_is_not_a_number_is_left_out_of_the_summary(field: str, value: Any) -> None:
    """``min`` over a mixed list raises, and a string that got through was reported.

    ``True`` is in here because ``bool`` is an ``int`` subclass, so the obvious
    ``isinstance(value, int)`` guard would summarise it as a loss of 1.0.
    """
    event = "train" if field == "loss" else "eval"
    records = [*GOOD_METRICS, {"event": event, "step": 3, field: value}]

    summary = summarise_run(records)

    assert summary["final_loss"] == 3.0
    assert summary["best_val_loss"] == 3.5
    assert summary["final_val_loss"] == 3.5


def test_a_bool_is_not_counted_as_a_measurement() -> None:
    """The same bool-is-an-int trap, on the two fields a number is read from.

    Alone in the log, so the assertion can see it: ``max`` over ``[100, 200, True]`` is
    200 either way, which is how a guard like this passes a test that does not isolate
    it. ``tokens`` of ``true`` is otherwise a one-token run and ``elapsed`` of ``true``
    a one-second one.
    """
    records = [{"event": "train", "step": 1, "loss": 2.0, "tokens": True, "elapsed": True}]

    summary = summarise_run(records)

    assert "tokens_seen" not in summary
    assert "elapsed_seconds" not in summary
    assert summary["final_loss"] == 2.0


def test_a_run_reports_metric_lines_it_could_not_read(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    """Pre-seeded, because the writer appends: a bad line an editor or an earlier
    killed run left behind is still there when the summary is computed at the end."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_bytes(b"3\n")
    trainer = make_trainer(prepared_dataset, run_dir, steps=4, eval_every=4)

    result = trainer.run()

    assert result.summary["unreadable_metric_lines"] == 1
    assert result.summary["train_records"] >= 1


def test_a_clean_run_reports_no_unreadable_lines(prepared_dataset: Any, tmp_path: Path) -> None:
    """The negative control at the run level, so an inverted guard fails a test."""
    trainer = make_trainer(prepared_dataset, tmp_path / "run", steps=4, eval_every=4)

    result = trainer.run()

    assert "unreadable_metric_lines" not in result.summary


def test_an_undamaged_metrics_log_is_unaffected_by_any_of_this(tmp_path: Path) -> None:
    """The negative control. A guard accidentally inverted fails here."""
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(metrics_bytes())
    writer = MetricsWriter(path, enabled=False)

    records = writer.read_all()

    assert records == GOOD_METRICS
    assert writer.unreadable_lines == 0
    summary = summarise_run(records)
    assert "unreadable_metric_lines" not in summary
    assert (summary["final_loss"], summary["best_val_loss"], summary["tokens_seen"]) == (
        3.0,
        3.5,
        200,
    )


# --------------------------------------------------------------------------- #
# GPU
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_a_run_on_cuda_reports_a_measured_peak(prepared_dataset: Any, tmp_path: Path) -> None:
    trainer = make_trainer(prepared_dataset, tmp_path / "gpu", steps=8, device="cuda")

    result = trainer.run()

    assert result.device.startswith("cuda")
    assert result.peak_vram_bytes > 0
    assert "peak" in trainer.memory_note()


@pytest.mark.gpu
def test_a_cpu_run_says_there_is_no_vram_to_account_for(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    trainer = make_trainer(prepared_dataset, tmp_path / "cpu", steps=4, device="cpu")

    result = trainer.run()

    assert result.peak_vram_bytes == 0
    assert "no VRAM accounting" in trainer.memory_note()


def test_the_result_is_json_safe(prepared_dataset: Any, tmp_path: Path) -> None:
    result = make_trainer(prepared_dataset, tmp_path / "run", steps=6, eval_every=3).run()

    payload = json.loads(json.dumps(result.to_dict()))

    assert payload["steps_completed"] == 6
    assert payload["summary"]["final_loss"] == pytest.approx(result.final_train_loss)


def test_an_invalid_model_shape_is_refused_before_anything_is_written(
    prepared_dataset: Any, tmp_path: Path
) -> None:
    with pytest.raises(ConfigError):
        make_trainer(prepared_dataset, tmp_path / "run", model={"n_head": 7, "d_model": 64})

    assert not (tmp_path / "run").exists()
