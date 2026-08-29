"""Tests for the empirical planner.

The planner decides from measurements, which is exactly what makes it awkward to test:
a test that needs one specific GPU cannot run anywhere else. So the measurement
function is injectable, and these tests pass one that *models* hardware instead of
having it -- memory from :func:`rough_peak_bytes`, throughput from a fixed FLOP rate,
and a twenty-fold slowdown when the peak exceeds total VRAM, which is what WDDM's
silent spill to system RAM actually does.

So what is under test here is the *search*: ladder order, micro-batch halving, which of
the three caps bound the answer, and what evidence ends up in the record. The
measurement itself is tested in ``test_benchmark.py``, on real hardware, behind the
``gpu`` marker. Keeping those two apart is deliberate -- the search has to be provable
on a laptop with no GPU, and the measurement cannot be proved anywhere else.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

import pytest

from test_hardware_portability import ALL_MACHINES, gpu, machine
from trainai.data.binarize import DatasetManifest, ShardInfo
from trainai.data.loader import open_split
from trainai.errors import CapacityError, DatasetError, UsageError
from trainai.hardware.benchmark import (
    VERDICT_FITS,
    VERDICT_OOM,
    VERDICT_OVER_BUDGET,
    BenchmarkResult,
    flops_per_token,
    rough_peak_bytes,
)
from trainai.hardware.planner import (
    EFFICIENCY_COLLAPSE_RATIO,
    MIN_STEPS,
    PLAN_VERSION,
    REGIME_BLOCKED,
    REGIME_COMPUTE_LIMITED,
    REGIME_DATA_LIMITED,
    REGIME_UNCONSTRAINED,
    REGIME_VRAM_LIMITED,
    REGIMES,
    SIZE_HORIZON_SECONDS,
    TARGET_TOKENS_PER_STEP,
    VERDICT_NOT_ENOUGH_DATA,
    VERDICT_THROUGHPUT_COLLAPSE,
    VERDICT_TOO_SLOW,
    Candidate,
    TrainingPlan,
    _cadence,
    _largest_divisor_at_most,
    _micro_batch_ladder,
    _preset_ladder,
    max_windows_for,
    parse_duration,
    plan_training,
    recommended_lr,
)
from trainai.model.config import PRESETS, ModelConfig
from trainai.train.config import TrainConfig

GIB = 1024**3


# --------------------------------------------------------------------------- #
# Fixtures: synthetic datasets and a measurement function that models hardware
# --------------------------------------------------------------------------- #
def dataset(train_tokens: int, val_tokens: int = 0, *, vocab_size: int = 512) -> DatasetManifest:
    """A manifest that reports token counts. Nothing here reads the shard files."""
    shards: dict[str, list[ShardInfo]] = {
        "train": [ShardInfo("train-000.bin", train_tokens, train_tokens * 2, "0" * 64)]
    }
    if val_tokens > 0:
        shards["val"] = [ShardInfo("val-000.bin", val_tokens, val_tokens * 2, "0" * 64)]
    return DatasetManifest(
        dtype="uint16",
        vocab_size=vocab_size,
        eot_id=0,
        tokenizer_fingerprint="synthetic",
        seed=1234,
        val_fraction=0.1,
        shard_tokens=1 << 20,
        shards=shards,
    )


def fake_measure(
    *,
    total_vram: int,
    device_flops: float = 4.0e12,
    memory_measured: bool = True,
    spill_penalty: float = 20.0,
) -> Any:
    """A measurement function that behaves like a GPU without being one.

    ``total_vram`` is the physical card size. Exceeding it does not raise -- throughput
    is divided by ``spill_penalty`` and the run continues, which is the whole reason
    this project measures rather than catching exceptions.
    """

    def measure(
        model: ModelConfig,
        *,
        batch_size: int,
        grad_accum: int,
        seq_len: int,
        device: Any,
        precision: str = "auto",
        budget_bytes: int = 0,
        warmup_steps: int = 2,
        measure_steps: int = 3,
        seed: int = 1234,
    ) -> BenchmarkResult:
        peak = rough_peak_bytes(model, batch_size=batch_size, seq_len=seq_len)
        rate = device_flops / flops_per_token(model, seq_len=seq_len)
        if total_vram > 0 and peak > total_vram:
            rate /= spill_penalty
        tokens_per_step = batch_size * grad_accum * seq_len
        result = BenchmarkResult(
            ok=True,
            verdict=VERDICT_FITS,
            detail="measured",
            peak_bytes=peak if memory_measured else 0,
            reserved_bytes=peak if memory_measured else 0,
            budget_bytes=budget_bytes,
            tokens_per_second=rate,
            step_seconds=tokens_per_step / rate,
            steps_measured=measure_steps,
            tokens_per_step=tokens_per_step,
            precision="bf16" if memory_measured else "fp32 (on CPU)",
            device=str(device),
            memory_measured=memory_measured,
        )
        if memory_measured and budget_bytes > 0 and peak > budget_bytes:
            return replace(
                result,
                ok=False,
                verdict=VERDICT_OVER_BUDGET,
                detail=f"the measured peak of {peak} bytes is over the budget",
            )
        return result

    return measure


def card(total_gib: float, free_gib: float | None = None) -> Any:
    """A synthetic Windows NVIDIA machine, which is the hard case."""
    free = total_gib * 0.8 if free_gib is None else free_gib
    info = gpu("Synthetic", total_gib=total_gib, capability=(8, 6), bf16=True)
    profile = machine(
        f"Synthetic {total_gib:g} GiB",
        gpus=[info],
        backend="cuda",
        os_name="Windows",
        bf16=True,
        vendors=["nvidia"],
    )
    profile.gpus[0] = replace(info, free_vram_bytes=int(free * GIB))
    return profile


def plan_for(
    manifest: DatasetManifest,
    *,
    total_gib: float = 8.0,
    free_gib: float | None = None,
    **kwargs: Any,
) -> TrainingPlan:
    """Plan with a modelled card, on the CPU device so nothing touches a GPU."""
    kwargs.setdefault("measure", fake_measure(total_vram=int(total_gib * GIB)))
    kwargs.setdefault("time_budget_seconds", None)
    return plan_training(
        manifest,
        hardware=card(total_gib, free_gib),
        device="cpu",
        dataset_path="data/synthetic",
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Durations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("30", 1800.0),
        ("0.5", 30.0),
        ("90s", 90.0),
        ("45m", 2700.0),
        ("2h", 7200.0),
        ("1d", 86400.0),
        ("1h30m", 5400.0),
        ("1h 30m", 5400.0),
        ("2H", 7200.0),
    ],
)
def test_parse_duration_reads_what_a_person_would_type(text: str, seconds: float) -> None:
    assert parse_duration(text) == pytest.approx(seconds)


@pytest.mark.parametrize(
    "text",
    ["", "   ", "abc", "2 hours", "0", "0m", "-5", "-5m", "h", "1x", "1h30"],
)
def test_parse_duration_refuses_what_it_cannot_read(text: str) -> None:
    """Refusing beats guessing: reading "2 hours" as 2 seconds would be silent."""
    with pytest.raises(UsageError):
        parse_duration(text)


# --------------------------------------------------------------------------- #
# Window arithmetic: the planner has to agree with the loader exactly
# --------------------------------------------------------------------------- #
def test_max_windows_for_matches_the_loader_on_a_real_dataset(prepared_dataset: Any) -> None:
    """If these two ever disagree, the planner recommends a batch the loader refuses."""
    for seq_len in (32, 64, 128, 256):
        batcher, _ = open_split(prepared_dataset, "train", seq_len=seq_len, batch_size=1, seed=0)
        predicted = max_windows_for(prepared_dataset.tokens("train"), seq_len)
        assert predicted == batcher.windows_per_epoch, seq_len


@pytest.mark.parametrize(
    ("tokens", "seq_len", "expected"),
    [
        (0, 128, 0),
        (128, 128, 0),
        (129, 128, 0),
        (5000, 256, 18),
        (5000, 512, 8),
        (5000, 1024, 3),
    ],
)
def test_max_windows_for_is_zero_rather_than_negative(
    tokens: int, seq_len: int, expected: int
) -> None:
    assert max_windows_for(tokens, seq_len) == expected


# --------------------------------------------------------------------------- #
# The rules of thumb, labelled as such but still expected to behave
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("d_model", "expected"),
    [(256, 4.24e-4), (384, 3.46e-4), (512, 3.0e-4), (768, 2.45e-4), (1024, 2.12e-4)],
)
def test_recommended_lr_scales_as_inverse_root_width(d_model: int, expected: float) -> None:
    assert recommended_lr(d_model) == pytest.approx(expected, rel=1e-2)


def test_recommended_lr_falls_back_rather_than_dividing_by_zero() -> None:
    assert recommended_lr(0) > 0
    assert recommended_lr(-1) > 0


def test_micro_batch_ladder_halves_to_one() -> None:
    assert _micro_batch_ladder(64, 64) == [64, 32, 16, 8, 4, 2, 1]
    assert _micro_batch_ladder(1, 64) == [1]
    assert _micro_batch_ladder(100, 8) == [8, 4, 2, 1]


def test_largest_divisor_keeps_the_effective_batch_exact() -> None:
    """Capping 64 at 38 must give 32, not 38: 38 x 2 is 76 and overshoots the target."""
    assert _largest_divisor_at_most(64, 38) == 32
    assert _largest_divisor_at_most(64, 64) == 64
    assert _largest_divisor_at_most(64, 100) == 64
    assert _largest_divisor_at_most(64, 1) == 1
    assert _largest_divisor_at_most(64, 0) == 1
    for cap in range(1, 65):
        chosen = _largest_divisor_at_most(64, cap)
        assert 64 % chosen == 0
        assert chosen <= max(1, cap)


def test_cadence_has_a_floor_and_still_fires_once() -> None:
    """A short run must not evaluate every three steps, and must leave a checkpoint."""
    assert _cadence(1000, target_count=20, floor=5, ceiling=250) == 50
    assert _cadence(100_000, target_count=20, floor=5, ceiling=250) == 250
    assert _cadence(68, target_count=20, floor=5, ceiling=250) == 5
    assert _cadence(4, target_count=10, floor=10, ceiling=500) == 4
    assert _cadence(1, target_count=10, floor=10, ceiling=500) == 1


def test_preset_ladder_truncates_and_refuses_a_name_it_does_not_know() -> None:
    assert [spec.name for spec in _preset_ladder(None)] == [spec.name for spec in PRESETS]
    assert [spec.name for spec in _preset_ladder("small")] == ["tiny", "small"]
    with pytest.raises(UsageError):
        _preset_ladder("enormous")


# --------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------- #
def test_a_generous_machine_and_corpus_climbs_the_whole_ladder() -> None:
    plan = plan_for(
        dataset(4_000_000_000, 40_000_000),
        total_gib=80,
        time_budget_seconds=10_000 * 3600,
    )

    assert plan.preset_name == PRESETS[-1].name
    assert plan.regime == REGIME_UNCONSTRAINED
    assert plan.measurement.ok


def test_a_small_corpus_is_data_limited_and_says_which_rung_it_refused() -> None:
    plan = plan_for(dataset(5_000_000, 500_000), total_gib=80)

    assert plan.regime == REGIME_DATA_LIMITED
    assert plan.preset_name == "tiny"
    refused = plan.rejected[-1]
    assert refused.result.verdict == VERDICT_NOT_ENOUGH_DATA
    # The refusal has to quote both numbers, or "not enough data" is unactionable.
    assert "5.00M" in refused.result.detail
    assert refused.candidate.name == "small"


def test_the_smallest_rung_is_always_measured_before_the_data_cap_applies() -> None:
    """A CapacityError must never be arithmetic alone -- something has to be measured."""
    plan = plan_for(dataset(200_000, 20_000), total_gib=80)

    first = plan.candidates[0]
    assert first.candidate.name == "tiny"
    assert first.result.steps_measured > 0, "the smallest rung was refused without measuring"
    assert first.result.tokens_per_second > 0


def test_a_small_card_is_vram_limited_and_records_every_micro_batch_it_tried() -> None:
    plan = plan_for(
        dataset(4_000_000_000, 40_000_000),
        total_gib=2,
        free_gib=1.6,
        time_budget_seconds=10_000 * 3600,
    )

    assert plan.regime == REGIME_VRAM_LIMITED
    refusals = [record for record in plan.rejected if record.candidate.name == "large"]
    assert len(refusals) >= 2, "a rejected rung must be retried at smaller micro-batches"
    assert [record.candidate.micro_batch for record in refusals] == sorted(
        (record.candidate.micro_batch for record in refusals), reverse=True
    )
    assert refusals[-1].candidate.micro_batch == 1, "it must try all the way down to 1"
    for record in refusals:
        assert record.result.verdict == VERDICT_OVER_BUDGET
        assert record.result.peak_bytes > record.result.budget_bytes > 0


def test_a_rejection_on_memory_cites_a_measured_peak_not_an_estimate() -> None:
    plan = plan_for(
        dataset(4_000_000_000, 40_000_000),
        total_gib=2,
        free_gib=1.6,
        time_budget_seconds=10_000 * 3600,
    )

    for record in plan.rejected:
        if record.result.verdict == VERDICT_OVER_BUDGET:
            assert record.result.memory_measured
            assert record.result.steps_measured > 0, "rejected without being run"


def test_throughput_collapse_is_caught_when_the_allocator_reports_a_fit() -> None:
    """The WDDM case the counters cannot see: peak looks fine, throughput does not."""

    def liar(model: ModelConfig, *, batch_size, grad_accum, seq_len, device, **kwargs: Any):
        tokens_per_step = batch_size * grad_accum * seq_len
        rate = 4.0e12 / flops_per_token(model, seq_len=seq_len)
        if model.parameter_count > 20_000_000:
            rate /= 25.0  # paging, invisible to the allocator
        return BenchmarkResult(
            ok=True,
            verdict=VERDICT_FITS,
            detail="measured",
            peak_bytes=64 * 1024 * 1024,  # always claims to fit
            budget_bytes=kwargs.get("budget_bytes", 0),
            tokens_per_second=rate,
            step_seconds=tokens_per_step / rate,
            steps_measured=3,
            tokens_per_step=tokens_per_step,
            precision="bf16",
            device=str(device),
            memory_measured=True,
        )

    plan = plan_training(
        dataset(4_000_000_000, 40_000_000),
        hardware=card(80),
        device="cpu",
        measure=liar,
        time_budget_seconds=10_000 * 3600,
    )

    assert plan.regime == REGIME_VRAM_LIMITED
    collapsed = [r for r in plan.rejected if r.result.verdict == VERDICT_THROUGHPUT_COLLAPSE]
    assert collapsed, "a 25x slowdown against the best rate was not caught"
    assert plan.model_config.parameter_count <= 20_000_000


def test_collapse_is_not_triggered_by_a_bigger_model_being_legitimately_slower() -> None:
    """A larger model is slower per token. Comparing FLOP rates is what avoids this."""
    plan = plan_for(
        dataset(4_000_000_000, 40_000_000),
        total_gib=80,
        time_budget_seconds=10_000 * 3600,
    )

    assert not [r for r in plan.rejected if r.result.verdict == VERDICT_THROUGHPUT_COLLAPSE]
    assert plan.preset_name == PRESETS[-1].name


def test_a_slow_machine_is_compute_limited_by_the_default_horizon_and_says_so() -> None:
    plan = plan_for(
        dataset(4_000_000_000, 40_000_000),
        total_gib=80,
        measure=fake_measure(total_vram=80 * GIB, device_flops=2.0e10),
    )

    assert plan.regime == REGIME_COMPUTE_LIMITED
    refused = plan.rejected[-1]
    assert refused.result.verdict == VERDICT_TOO_SLOW
    # A cap the user did not ask for has to name itself.
    assert any("default" in note for note in plan.notes)
    assert plan.estimated_seconds <= SIZE_HORIZON_SECONDS


def test_a_time_budget_cuts_the_step_count_and_warns_that_it_did() -> None:
    generous = plan_for(dataset(400_000_000, 4_000_000), total_gib=80)
    limited = plan_for(dataset(400_000_000, 4_000_000), total_gib=80, time_budget_seconds=120.0)

    assert limited.train_config.steps < generous.train_config.steps
    assert limited.estimated_seconds <= 120.0 * 1.001
    assert any("time budget" in note for note in limited.notes)


def test_a_time_budget_too_small_for_a_real_run_says_that_too() -> None:
    plan = plan_for(dataset(400_000_000, 4_000_000), total_gib=80, time_budget_seconds=60.0)

    if plan.train_config.steps < MIN_STEPS:
        assert any(str(MIN_STEPS) in note for note in plan.notes)


def test_every_candidate_holds_the_target_tokens_per_step() -> None:
    """Holding tokens/step fixed is what makes the lr rule and step counts comparable."""
    plan = plan_for(
        dataset(4_000_000_000, 40_000_000),
        total_gib=80,
        time_budget_seconds=10_000 * 3600,
    )

    for record in plan.candidates:
        if record.candidate.micro_batch <= 0:
            continue  # ruled out before it was sized
        assert record.candidate.tokens_per_step == TARGET_TOKENS_PER_STEP


def test_gradient_accumulation_preserves_the_effective_batch_when_memory_bites() -> None:
    plan = plan_for(
        dataset(4_000_000_000, 40_000_000),
        total_gib=2,
        free_gib=1.6,
        time_budget_seconds=10_000 * 3600,
    )

    config = plan.train_config
    assert config.grad_accum > 1, "a 2 GiB card should have needed accumulation"
    assert config.effective_batch_size * config.seq_len == TARGET_TOKENS_PER_STEP
    assert any("accumulation" in note for note in plan.notes)


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #
def test_an_empty_dataset_is_a_dataset_error_not_a_capacity_error() -> None:
    with pytest.raises(DatasetError):
        plan_for(dataset(0))


def test_a_machine_that_can_train_nothing_raises_with_the_evidence_attached() -> None:
    with pytest.raises(CapacityError) as caught:
        plan_for(dataset(5_000_000, 500_000), total_gib=0.03, free_gib=0.03)

    error = caught.value
    assert error.hint
    candidates = error.details["candidates"]
    assert candidates, "a refusal with no recorded candidate explains nothing"
    assert any(record["result"]["steps_measured"] > 0 for record in candidates), (
        "every candidate was refused without measuring anything"
    )
    assert error.details["vram_budget_bytes"] > 0


def test_a_corpus_too_short_for_one_window_is_refused_with_a_reason() -> None:
    with pytest.raises(CapacityError) as caught:
        plan_for(dataset(100), total_gib=80, seq_len=1024)

    verdicts = {r["result"]["verdict"] for r in caught.value.details["candidates"]}
    assert VERDICT_NOT_ENOUGH_DATA in verdicts


def test_an_unexpected_failure_is_reported_as_blocked_rather_than_as_a_memory_limit() -> None:
    """Guessing "vram-limited" for an unknown fault would be a claim about a measurement."""
    calls: list[int] = []

    def flaky(model: ModelConfig, **kwargs: Any) -> BenchmarkResult:
        calls.append(1)
        if len(calls) == 1:
            return fake_measure(total_vram=80 * GIB)(model, **kwargs)
        return BenchmarkResult(
            ok=False,
            verdict="error",
            detail="RuntimeError: the driver fell over",
            budget_bytes=kwargs.get("budget_bytes", 0),
        )

    plan = plan_for(dataset(4_000_000_000, 40_000_000), total_gib=80, measure=flaky)

    assert plan.regime == REGIME_BLOCKED
    assert "driver fell over" in plan.regime_detail


def test_seq_len_below_the_floor_is_a_usage_error() -> None:
    with pytest.raises(UsageError):
        plan_for(dataset(5_000_000), seq_len=4)


# --------------------------------------------------------------------------- #
# Honesty about what was not measured
# --------------------------------------------------------------------------- #
def test_a_cpu_plan_says_the_memory_fit_was_not_verified() -> None:
    plan = plan_training(
        dataset(4_000_000_000, 40_000_000),
        hardware=ALL_MACHINES["cpu_only"](),
        device="cpu",
        measure=fake_measure(total_vram=0, device_flops=6.0e10, memory_measured=False),
    )

    assert not plan.measurement.memory_measured
    assert any("not measured" in note for note in plan.notes)
    # Throughput is still a real measurement, so the estimate is still meaningful.
    assert plan.measurement.tokens_per_second > 0
    assert plan.estimated_seconds > 0


def test_a_cpu_plan_reports_no_memory_verdict_it_did_not_take() -> None:
    plan = plan_training(
        dataset(4_000_000_000, 40_000_000),
        hardware=ALL_MACHINES["cpu_only"](),
        device="cpu",
        measure=fake_measure(total_vram=0, device_flops=6.0e10, memory_measured=False),
    )

    for record in plan.candidates:
        assert record.result.verdict != VERDICT_OVER_BUDGET
        assert record.result.verdict != VERDICT_OOM


def test_every_plan_states_that_dataset_read_time_is_excluded() -> None:
    plan = plan_for(dataset(5_000_000, 500_000))
    assert any("read time" in note for note in plan.notes)


def test_a_dataset_without_validation_is_flagged() -> None:
    plan = plan_for(dataset(5_000_000, val_tokens=0))
    assert any("no validation split" in note for note in plan.notes)


def test_a_validation_split_too_small_for_one_window_is_flagged_rather_than_ignored() -> None:
    """Flagged on the window count, and the fix it names has to be one that works."""
    plan = plan_for(dataset(400_000_000, 300), total_gib=80, seq_len=256)

    assert max_windows_for(300, plan.train_config.seq_len) == 0
    note = next(note for note in plan.notes if "Validation will not run" in note)

    suggested = int(re.search(r"--seq-len (\d+)", note).group(1))
    assert max_windows_for(300, suggested) >= 1
    assert max_windows_for(300, suggested + 1) == 0


def test_a_validation_split_of_one_window_is_not_called_unrunnable() -> None:
    """The trainer sizes the validation batch to the split, so one window is enough.

    This used to be flagged: the plan compared the micro-batch against the window count
    and announced that validation would not run whenever the batch was larger. The
    trainer no longer works that way, and a plan promising a missing number that the run
    does produce is worse than saying nothing.
    """
    plan = plan_for(dataset(400_000_000, 600), total_gib=80, seq_len=256)

    assert max_windows_for(600, plan.train_config.seq_len) == 1
    assert plan.train_config.batch_size > 1, "the case only bites when the batch is larger"
    assert not any("Validation will not run" in note for note in plan.notes)


def test_the_training_batch_is_not_shrunk_to_protect_validation() -> None:
    """A small validation split must not shape the training batch.

    The plan used to cap the micro-batch at the validation window count whenever that
    count came within a factor of four of the effective batch -- a workaround for a
    trainer that gave up when the batch exceeded the split, since fixed. Measured at
    --seq-len 256, where the effective batch is 64: on a 5,000-token validation split
    (18 windows) the plan recommended --batch-size 16 with --grad-accum 4, against 64
    and 1 for the same dataset with no validation split at all. Four times the launches
    per step, to protect a number the trainer now measures at any batch size.

    The window count matters to the scenario: at one window the old cap did not fire at
    all, so a test built on a 600-token split would pass with the cap back in place.
    """
    small_val = plan_for(dataset(400_000_000, 5_000), total_gib=80, seq_len=256)
    no_val = plan_for(dataset(400_000_000, 0), total_gib=80, seq_len=256)

    assert max_windows_for(5_000, small_val.train_config.seq_len) == 18
    assert small_val.train_config.batch_size == no_val.train_config.batch_size
    assert small_val.train_config.grad_accum == no_val.train_config.grad_accum


# --------------------------------------------------------------------------- #
# The plan is a contract with `trainai train`
# --------------------------------------------------------------------------- #
def test_the_recommended_configuration_can_open_the_dataset(prepared_dataset: Any) -> None:
    """The strongest cross-check available without a GPU: the loader must accept it."""
    plan = plan_training(
        prepared_dataset,
        hardware=card(8),
        device="cpu",
        measure=fake_measure(total_vram=8 * GIB),
    )

    config = plan.train_config
    batcher, _ = open_split(
        prepared_dataset,
        "train",
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    assert batcher.steps_per_epoch >= 1


def test_the_plan_round_trips_through_the_config_loaders() -> None:
    """Rebuilding from plan.json must give the same architecture and the same schedule.

    ``ModelConfig.to_dict`` writes *resolved* values for ``d_ff`` and ``n_kv_head``,
    unlike ``TrainConfig`` and its warmup. That asymmetry is deliberate rather than an
    oversight: the same method serialises checkpoints, where the architecture has to be
    reproduced byte-for-byte or the saved weights will not load, so a derived width must
    not be free to change when the derivation does. The round trip is therefore checked
    for equivalence and idempotence rather than for field-by-field equality.
    """
    plan = plan_for(dataset(400_000_000, 4_000_000))
    payload = plan.to_dict()

    model = ModelConfig.from_dict(payload["model"])
    train = TrainConfig.from_dict(payload["train"])

    assert model.parameter_count == plan.model_config.parameter_count
    assert model.describe() == plan.model_config.describe()
    assert model.ffn_dim == plan.model_config.ffn_dim
    assert model.kv_heads == plan.model_config.kv_heads
    assert model.to_dict() == payload["model"], "a second round trip must be stable"

    # The training configuration *is* exactly equal, warmup included.
    assert train == plan.train_config
    assert train.resolved_warmup_steps == plan.train_config.resolved_warmup_steps


def test_the_plan_json_is_serialisable_and_carries_its_version(tmp_path: Any) -> None:
    import json

    plan = plan_for(dataset(400_000_000, 4_000_000))
    written = plan.write(tmp_path)
    payload = json.loads(written.read_text(encoding="utf-8"))

    assert payload["version"] == PLAN_VERSION
    assert payload["regime"] in REGIMES
    assert payload["train_command"].startswith("trainai train ")


def test_provenance_separates_measured_from_derived_from_guessed() -> None:
    """The three kinds of claim must not be mixed, because they are not equally strong."""
    plan = plan_for(dataset(400_000_000, 4_000_000))
    provenance = plan.provenance()

    assert provenance["measured"]["tokens_per_second"] == plan.measurement.tokens_per_second
    assert provenance["derived_from_data"]["steps"] == plan.train_config.steps
    assert provenance["rules_of_thumb"]["lr"] == plan.train_config.lr
    # A learning rate is never presented as a measurement.
    assert "lr" not in provenance["measured"]
    assert "peak_bytes" not in provenance["rules_of_thumb"]


def test_provenance_reports_none_rather_than_zero_for_what_was_not_measured() -> None:
    plan = plan_training(
        dataset(400_000_000, 4_000_000),
        hardware=ALL_MACHINES["cpu_only"](),
        device="cpu",
        measure=fake_measure(total_vram=0, device_flops=6.0e10, memory_measured=False),
    )

    measured = plan.provenance()["measured"]
    assert measured["peak_bytes"] is None, "zero would read as 'it used no memory'"
    assert measured["memory_measured"] is False


def test_the_printed_command_contains_every_decision_it_made() -> None:
    plan = plan_for(dataset(4_000_000_000, 40_000_000), total_gib=2, free_gib=1.6)
    command = plan.train_command()

    config = plan.train_config
    assert f"--preset {plan.preset_name}" in command
    assert f"--steps {config.steps}" in command
    assert f"--batch-size {config.batch_size}" in command
    assert f"--seq-len {config.seq_len}" in command
    assert f"--context {plan.model_config.seq_len}" in command
    if config.grad_accum > 1:
        assert f"--grad-accum {config.grad_accum}" in command


@pytest.mark.parametrize("width", [50, 60, 80, 120])
def test_the_recommended_command_survives_a_narrow_terminal(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    """The command `trainai plan` recommends must be copyable out of the report.

    This is the site the bug was found on: `_report` printed the command through plain
    `console.print`, so Rich broke a 150-character command across three lines and what
    the user copied ran the first line and then failed on `--steps`. The helper in
    `trainai.console` has its own tests; this one pins the *call site*, because reverting
    this function to `console.print` would otherwise pass every test in the suite.

    Asserted as a substring of the raw output: only an unwrapped, uncropped command
    appears in it verbatim.
    """
    from pathlib import Path

    from rich.console import Console

    from trainai.cli import plan as plan_cli

    plan = plan_for(dataset(4_000_000_000, 40_000_000), total_gib=2, free_gib=1.6)
    narrow = Console(width=width)
    # Both: `plan.py` holds its own reference to the console object, and `print_command`
    # reaches for `trainai.console.console` at call time.
    monkeypatch.setattr("trainai.console.console", narrow)
    monkeypatch.setattr("trainai.cli.plan.console", narrow)

    plan_cli._report(plan, Path("runs/plan.json"))

    printed = capsys.readouterr().out
    command = plan.train_command()
    assert len(command) > width, "the test is vacuous unless the command is wider than the terminal"
    assert command in printed, f"the recommended command did not survive COLUMNS={width}"


def test_max_preset_truncates_the_search_and_says_it_did() -> None:
    plan = plan_for(
        dataset(4_000_000_000, 40_000_000),
        total_gib=80,
        max_preset="small",
        time_budget_seconds=10_000 * 3600,
    )

    assert plan.preset_name == "small"
    assert {record.candidate.name for record in plan.candidates} <= {"tiny", "small"}
    assert any("max-preset" in note for note in plan.notes)


# --------------------------------------------------------------------------- #
# Every machine, real and synthetic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label", sorted(ALL_MACHINES))
def test_every_synthetic_machine_produces_a_coherent_plan(label: str) -> None:
    """Nobody owns all of these. The planner still has to behave on each."""
    profile = ALL_MACHINES[label]()
    has_gpu = profile.primary_gpu is not None
    measure = fake_measure(
        total_vram=profile.primary_gpu.total_vram_bytes if has_gpu else 0,
        device_flops=4.0e12 if has_gpu else 6.0e10,
        memory_measured=has_gpu,
    )

    plan = plan_training(
        dataset(400_000_000, 4_000_000),
        hardware=profile,
        device="cpu",
        measure=measure,
        time_budget_seconds=1_000 * 3600,
    )

    assert plan.regime in REGIMES
    assert plan.regime_detail
    assert plan.train_config.steps >= 1
    assert plan.train_config.batch_size >= 1
    assert plan.model_config.vocab_size == 512
    assert plan.estimated_seconds > 0
    assert plan.measurement.ok
    if has_gpu:
        assert plan.measurement.peak_bytes <= plan.vram_budget_bytes


@pytest.mark.parametrize("label", sorted(ALL_MACHINES))
def test_a_bigger_card_never_recommends_a_smaller_model(label: str) -> None:
    """Monotonicity: more memory must not produce a worse plan."""
    profile = ALL_MACHINES[label]()
    if profile.primary_gpu is None:
        pytest.skip("no GPU to enlarge")

    def sized(total_gib: float) -> TrainingPlan:
        info = replace(
            profile.primary_gpu,
            total_vram_bytes=int(total_gib * GIB),
            free_vram_bytes=int(total_gib * GIB),
        )
        bigger = replace(profile, gpus=[info])
        return plan_training(
            dataset(4_000_000_000, 40_000_000),
            hardware=bigger,
            device="cpu",
            measure=fake_measure(total_vram=int(total_gib * GIB)),
            time_budget_seconds=10_000 * 3600,
        )

    order = [spec.name for spec in PRESETS]
    small, large = sized(2.0), sized(80.0)
    assert order.index(large.preset_name) >= order.index(small.preset_name)


def test_candidate_describe_does_not_print_a_batch_of_zero() -> None:
    """Rungs ruled out before being sized have no batch. "batch 0" reads as a bug."""
    model = ModelConfig(vocab_size=512)
    unsized = Candidate("medium", model, 512, 0, 0)
    sized = Candidate("medium", model, 512, 8, 4)

    assert "batch" not in unsized.describe()
    assert "batch 8 x 4" in sized.describe()


def test_efficiency_collapse_ratio_separates_the_real_measurements() -> None:
    """Calibration check against the numbers measured on the development RTX 2050.

    The spilling configuration ran at 0.18 of the best observed FLOP rate; the largest
    configuration that merely did not fit ran at 0.76. The threshold has to sit between
    them, or it either misses the spill or rejects a model that was simply bigger.
    """
    assert 0.18 < EFFICIENCY_COLLAPSE_RATIO < 0.76
