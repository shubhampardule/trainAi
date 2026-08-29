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

import pytest
import torch

import trainai.train.loop
from trainai.data.loader import open_split, window_count
from trainai.errors import (
    ConfigError,
    DatasetError,
    DatasetFormatError,
    TrainingDivergedError,
    TrainingError,
)
from trainai.model.config import ModelConfig
from trainai.train.budget import (
    EPOCHS_BEFORE_MEMORISING,
    TOKENS_PER_PARAMETER_TARGET,
    DataBudget,
)
from trainai.train.config import TrainConfig
from trainai.train.loop import Trainer, check_configs_agree, resolve_device, resolve_precision
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
    fine = budget(
        train_tokens=100_000_000,
        parameters=1_000_000,
        steps=100,
        tokens_per_step=1_000,
        val_tokens=1_000,
    )

    assert not fine.will_memorise
    assert not fine.is_data_limited
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
