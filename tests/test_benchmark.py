"""Tests for the measurement itself.

Split from ``test_planner.py`` on purpose. The planner's search is pure enough to prove
against synthetic hardware; the measurement is not provable anywhere except on the
hardware it is measuring. So the arithmetic and the reporting are tested everywhere, and
the parts that need a real allocator are marked ``gpu`` and skipped when there is none.

That split is also why the ``gpu`` tests here matter more than their number suggests:
they are the only thing standing between "we read the allocator's own counters" and a
number that merely looks plausible. One of them exists because that exact failure
happened -- see ``test_the_measured_peak_includes_activations_not_just_parameters``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from trainai.hardware.benchmark import (
    ACTIVATION_SLOTS_PER_LAYER,
    PERSISTENT_BYTES_PER_PARAMETER,
    VERDICT_ERROR,
    VERDICT_FITS,
    VERDICT_OVER_BUDGET,
    VERDICT_SKIPPED,
    BenchmarkResult,
    describe,
    flops_per_token,
    measure_candidate,
    rough_peak_bytes,
    skipped,
)
from trainai.model.config import ModelConfig
from trainai.train.loop import resolve_device

CPU = torch.device("cpu")

#: Small enough that the CPU tests finish quickly, large enough that the activation and
#: logit terms are not rounding error against the parameters.
SMALL = ModelConfig(vocab_size=512, n_layer=2, n_head=2, d_model=64, seq_len=64)

#: Used by the GPU tests: a vocabulary big enough that the logit tensor is real, which
#: is where the peak of a small model actually lives.
GPU_MODEL = ModelConfig(vocab_size=4096, n_layer=4, n_head=4, d_model=256, seq_len=256)


# --------------------------------------------------------------------------- #
# The result object
# --------------------------------------------------------------------------- #
def test_a_default_result_claims_nothing() -> None:
    result = BenchmarkResult(ok=False, verdict=VERDICT_ERROR, detail="boom")

    assert result.peak_bytes == 0
    assert result.memory_measured is False
    assert result.budget_fraction == 0.0
    assert result.steps_measured == 0


def test_budget_fraction_is_zero_when_memory_was_not_measured() -> None:
    """A fraction computed from an unmeasured peak would read as "it used nothing"."""
    unmeasured = BenchmarkResult(
        ok=True, verdict=VERDICT_FITS, detail="", peak_bytes=0, budget_bytes=1 << 30
    )
    measured = BenchmarkResult(
        ok=True,
        verdict=VERDICT_FITS,
        detail="",
        peak_bytes=1 << 29,
        budget_bytes=1 << 30,
        memory_measured=True,
    )

    assert unmeasured.budget_fraction == 0.0
    assert measured.budget_fraction == pytest.approx(0.5)


def test_budget_fraction_is_zero_without_a_budget() -> None:
    result = BenchmarkResult(
        ok=True, verdict=VERDICT_FITS, detail="", peak_bytes=1 << 20, memory_measured=True
    )
    assert result.budget_fraction == 0.0


def test_skipped_is_labelled_so_it_cannot_be_read_as_a_measurement() -> None:
    result = skipped("the estimate was far over budget", budget_bytes=1 << 30)

    assert result.ok is False
    assert result.verdict == VERDICT_SKIPPED
    assert result.steps_measured == 0
    assert result.memory_measured is False
    assert "not measured" in describe(result)


def test_to_dict_carries_every_field_a_report_needs() -> None:
    payload = BenchmarkResult(
        ok=True, verdict=VERDICT_FITS, detail="", tokens_per_second=1.0
    ).to_dict()

    for key in ("ok", "verdict", "detail", "peak_bytes", "tokens_per_second", "memory_measured"):
        assert key in payload


# --------------------------------------------------------------------------- #
# The estimate -- kept only to order the search, never to decide
# --------------------------------------------------------------------------- #
def test_rough_peak_counts_parameters_activations_and_logits() -> None:
    config = ModelConfig(vocab_size=1000, n_layer=4, n_head=4, d_model=128, seq_len=128)
    batch, seq = 4, 128
    tokens = batch * seq

    estimate = rough_peak_bytes(config, batch_size=batch, seq_len=seq, activation_bytes=2)

    expected = (
        config.parameter_count * PERSISTENT_BYTES_PER_PARAMETER
        + tokens * config.d_model * config.n_layer * ACTIVATION_SLOTS_PER_LAYER * 2
        + tokens * config.vocab_size * 4 * 2
    )
    assert estimate == expected


@pytest.mark.parametrize("field", ["batch_size", "seq_len"])
def test_rough_peak_rises_with_the_work(field: str) -> None:
    base = {"batch_size": 4, "seq_len": 128}
    doubled = {**base, field: base[field] * 2}

    assert rough_peak_bytes(SMALL, **doubled) > rough_peak_bytes(SMALL, **base)


def test_flops_per_token_is_six_n_plus_the_attention_term() -> None:
    config = ModelConfig(vocab_size=512, n_layer=4, n_head=4, d_model=128, seq_len=256)

    at_default = flops_per_token(config)
    expected = 6.0 * config.parameter_count + 12.0 * config.n_layer * 256 * config.d_model

    assert at_default == pytest.approx(expected)


def test_flops_per_token_honours_a_sequence_length_override() -> None:
    """Attention's quadratic term differs by 4x across the preset ladder's contexts."""
    config = ModelConfig(vocab_size=512, n_layer=4, n_head=4, d_model=128, seq_len=256)

    assert flops_per_token(config, seq_len=1024) > flops_per_token(config, seq_len=256)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def test_describe_renders_a_pre_measurement_rejection_as_a_reason() -> None:
    result = BenchmarkResult(ok=False, verdict="not-enough-data", detail="only 300 tokens")

    rendered = describe(result)

    assert "not-enough-data" in rendered
    assert "only 300 tokens" in rendered
    assert "tokens/s" not in rendered, "a throughput that was never measured must not appear"


def test_describe_renders_a_measured_result_with_its_verdict() -> None:
    """The planner invents its own verdicts, so describe must not assume a fixed set."""
    result = BenchmarkResult(
        ok=False,
        verdict="too-slow",
        detail="would take three days",
        peak_bytes=1 << 30,
        tokens_per_second=1304.0,
        steps_measured=3,
        memory_measured=True,
    )

    rendered = describe(result)

    assert "too-slow" in rendered
    assert "tokens/s" in rendered
    assert "1.00 GiB" in rendered


def test_describe_says_not_measured_rather_than_zero_for_memory() -> None:
    result = BenchmarkResult(
        ok=True, verdict=VERDICT_FITS, detail="", tokens_per_second=100.0, steps_measured=3
    )

    assert "peak not measured" in describe(result)


# --------------------------------------------------------------------------- #
# Measuring on the CPU: real throughput, and honesty about the rest
# --------------------------------------------------------------------------- #
def test_measure_candidate_runs_a_real_step_on_the_cpu() -> None:
    result = measure_candidate(
        SMALL, batch_size=2, grad_accum=1, seq_len=32, device=CPU, warmup_steps=1, measure_steps=2
    )

    assert result.ok, result.detail
    assert result.verdict == VERDICT_FITS
    assert result.tokens_per_second > 0
    assert result.step_seconds > 0
    assert result.steps_measured == 2
    assert result.tokens_per_step == 2 * 32
    assert result.device == "cpu"


def test_a_cpu_measurement_reports_that_memory_was_not_measured() -> None:
    result = measure_candidate(
        SMALL, batch_size=2, grad_accum=1, seq_len=32, device=CPU, warmup_steps=1, measure_steps=1
    )

    assert result.memory_measured is False
    assert result.peak_bytes == 0
    assert "not measured" in describe(result)


def test_a_budget_is_inert_where_there_are_no_counters_to_check_it_against() -> None:
    """A budget of one byte must not produce an over-budget verdict on the CPU.

    Reporting "over budget" from a peak of zero would be inventing a measurement. The
    planner relies on this: it passes the same budget regardless of device, because the
    alternative is a device-specific branch in the caller that is easy to get wrong.
    """
    result = measure_candidate(
        SMALL,
        batch_size=2,
        grad_accum=1,
        seq_len=32,
        device=CPU,
        budget_bytes=1,
        warmup_steps=1,
        measure_steps=1,
    )

    assert result.ok
    assert result.verdict == VERDICT_FITS
    assert result.memory_measured is False


def test_gradient_accumulation_is_counted_in_tokens_per_step() -> None:
    result = measure_candidate(
        SMALL, batch_size=2, grad_accum=3, seq_len=32, device=CPU, warmup_steps=1, measure_steps=1
    )

    assert result.tokens_per_step == 2 * 3 * 32


def test_an_unexpected_fault_comes_back_as_a_result_not_an_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caller's next move is to try something smaller, not to unwind the stack."""

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the driver fell over")

    monkeypatch.setattr("trainai.hardware.benchmark.GPT", explode)

    result = measure_candidate(SMALL, batch_size=1, grad_accum=1, seq_len=32, device=CPU)

    assert result.ok is False
    assert result.verdict == VERDICT_ERROR
    assert "driver fell over" in result.detail
    assert result.steps_measured == 0


def test_an_out_of_memory_runtime_error_is_recognised_by_its_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ROCm and XPU raise a plain RuntimeError, so the message has to be read."""

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("HIP out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr("trainai.hardware.benchmark.GPT", explode)

    result = measure_candidate(SMALL, batch_size=1, grad_accum=1, seq_len=32, device=CPU)

    assert result.ok is False
    assert result.verdict == "out-of-memory"


def test_measurement_is_repeatable_enough_to_compare_candidates() -> None:
    """Not bitwise -- timings vary. But the same shape must not swing by an order."""
    runs = [
        measure_candidate(
            SMALL,
            batch_size=2,
            grad_accum=1,
            seq_len=32,
            device=CPU,
            warmup_steps=1,
            measure_steps=3,
        ).tokens_per_second
        for _ in range(3)
    ]

    assert min(runs) > 0
    assert max(runs) / min(runs) < 10.0, runs


# --------------------------------------------------------------------------- #
# The allocator counters. Only provable on real hardware.
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_the_measured_peak_includes_activations_not_just_parameters() -> None:
    """Regression test for a bug that made every memory verdict meaningless.

    The peak used to be read *after* the ``try`` block, and the ``finally`` clause calls
    ``_release``, which calls ``reset_peak_memory_stats`` so one candidate's cached
    blocks cannot leak into the next. So the read happened after the reset and returned
    whatever was still live -- on a small model, almost exactly the parameter bytes.

    That is why it went unnoticed: 65 MiB for a 4.26M-parameter model looks like a
    reasonable answer rather than like a broken one. It was wrong by a factor of fifteen
    against the 957 MiB the trainer reported for the same configuration.
    """
    device = resolve_device("cuda")
    result = measure_candidate(
        GPU_MODEL, batch_size=16, grad_accum=1, seq_len=256, device=device, measure_steps=2
    )

    assert result.ok, result.detail
    assert result.memory_measured
    persistent = GPU_MODEL.parameter_count * PERSISTENT_BYTES_PER_PARAMETER
    assert result.peak_bytes > 2 * persistent, (
        f"peak {result.peak_bytes} is barely above the {persistent} bytes of parameters, "
        "gradients and AdamW moments, so activations and logits were not counted"
    )


@pytest.mark.gpu
def test_the_measured_peak_is_the_same_order_as_the_estimate() -> None:
    """The estimate is crude by design, but a 15x disagreement means one is broken."""
    device = resolve_device("cuda")
    estimate = rough_peak_bytes(GPU_MODEL, batch_size=16, seq_len=256)

    result = measure_candidate(
        GPU_MODEL, batch_size=16, grad_accum=1, seq_len=256, device=device, measure_steps=2
    )

    assert result.ok, result.detail
    assert 0.25 * estimate < result.peak_bytes < 4.0 * estimate, (
        f"measured {result.peak_bytes} against an estimate of {estimate}"
    )


@pytest.mark.gpu
def test_the_peak_never_exceeds_what_the_allocator_reserved() -> None:
    device = resolve_device("cuda")

    result = measure_candidate(
        GPU_MODEL, batch_size=8, grad_accum=1, seq_len=256, device=device, measure_steps=2
    )

    assert result.ok, result.detail
    assert result.reserved_bytes >= result.peak_bytes > 0


@pytest.mark.gpu
def test_a_peak_over_budget_is_rejected_after_actually_being_run() -> None:
    """The WDDM rejection: nothing raised, the steps completed, the peak was too high."""
    device = resolve_device("cuda")

    result = measure_candidate(
        GPU_MODEL,
        batch_size=16,
        grad_accum=1,
        seq_len=256,
        device=device,
        budget_bytes=1 << 20,
        measure_steps=2,
    )

    assert result.ok is False
    assert result.verdict == VERDICT_OVER_BUDGET
    assert result.peak_bytes > result.budget_bytes
    assert result.steps_measured > 0, "rejected without being measured"
    assert result.tokens_per_second > 0, "the measurement is kept, only the verdict changes"


@pytest.mark.gpu
def test_a_generous_budget_is_accepted() -> None:
    device = resolve_device("cuda")

    result = measure_candidate(
        GPU_MODEL,
        batch_size=8,
        grad_accum=1,
        seq_len=256,
        device=device,
        budget_bytes=1 << 40,
        measure_steps=2,
    )

    assert result.ok, result.detail
    assert result.verdict == VERDICT_FITS
    assert 0.0 < result.budget_fraction < 1.0


@pytest.mark.gpu
def test_one_candidate_does_not_inflate_the_next_ones_peak() -> None:
    """``_release`` exists for this: without it the second reading includes the first."""
    device = resolve_device("cuda")
    big = measure_candidate(
        GPU_MODEL, batch_size=16, grad_accum=1, seq_len=256, device=device, measure_steps=2
    )
    small = measure_candidate(
        GPU_MODEL, batch_size=2, grad_accum=1, seq_len=256, device=device, measure_steps=2
    )

    assert big.ok and small.ok
    assert small.peak_bytes < big.peak_bytes, (
        f"a smaller batch reported {small.peak_bytes} after a larger one reported "
        f"{big.peak_bytes}: the previous candidate's memory was not released"
    )
