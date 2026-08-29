"""Find out whether a configuration fits by running it and reading the counters.

This module exists because the usual approach does not work on the machine this
project was developed on.

The usual approach is: estimate memory use with a formula, run the configuration,
and treat ``torch.cuda.OutOfMemoryError`` as the signal that it did not fit. Under
the Windows WDDM driver model there is no such signal. The OS lets CUDA
oversubscribe VRAM and pages the excess to system RAM, so a configuration that
needs 6.55 GiB on a 4 GiB card runs to completion, raises nothing, and is 47x
slower. Measured here on an RTX 2050:

    model         batch x ctx   tokens/s   peak allocated
    13.9M params      32 x 256     61,595   1.48 GiB
    33.8M params      16 x 512     29,923   2.87 GiB
    110.5M params      8 x 512      5,854   4.01 GiB   over budget
    110.9M params      8 x 1024     1,304   6.55 GiB   over budget, no exception

So the question "does this fit" is answered here by running real forward,
backward and optimizer steps and reading:

* ``allocated_bytes.all.peak`` -- the high-water mark of tensor memory, which is
  what exceeded the card above while nothing raised.
* ``num_alloc_retries`` -- how often the caching allocator had to release cached
  blocks and ask the driver again. Non-zero means the run is already at the edge.
* wall-clock time over a fixed number of steps, which is where a spill shows up
  even on a platform whose counters we cannot read.

The optimizer step is included on purpose. AdamW's two moment buffers are
allocated lazily on the first ``step()``, and they are 8 bytes per parameter --
two thirds of the persistent footprint of a small model. A benchmark that only
does forward and backward reports a peak the real run will exceed.

Batches are synthetic random token ids rather than the user's shards. Peak memory
and throughput are functions of the *shapes*, not of the token values, and
synthetic batches keep the measurement independent of how large the dataset
happens to be -- a micro-batch of 64 is a legitimate thing to measure even when
the validation split holds only 18 sequences. The cost of that choice is that the
measured time excludes dataset read time; on a warm page cache that is small, and
:mod:`trainai.hardware.planner` states it rather than hiding it.
"""

from __future__ import annotations

import contextlib
import gc
import time
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn

from trainai.console import fmt_bytes, fmt_count
from trainai.model.config import ModelConfig
from trainai.model.gpt import GPT
from trainai.train.config import Precision
from trainai.train.loop import resolve_precision

__all__ = [
    "ACTIVATION_SLOTS_PER_LAYER",
    "PERSISTENT_BYTES_PER_PARAMETER",
    "VERDICT_ALLOCATOR_PRESSURE",
    "VERDICT_ERROR",
    "VERDICT_FITS",
    "VERDICT_OOM",
    "VERDICT_OVER_BUDGET",
    "VERDICT_SKIPPED",
    "BenchmarkResult",
    "describe",
    "flops_per_token",
    "measure_candidate",
    "rough_peak_bytes",
    "skipped",
]

#: Verdicts. ``fits`` is the only one that means yes.
VERDICT_FITS = "fits"
VERDICT_OVER_BUDGET = "over-budget"
VERDICT_OOM = "out-of-memory"
VERDICT_ALLOCATOR_PRESSURE = "allocator-pressure"
VERDICT_ERROR = "error"
VERDICT_SKIPPED = "not-measured"

#: Bytes of persistent state per parameter during training: fp32 master weights,
#: fp32 gradients, and AdamW's two fp32 moment buffers. Mixed precision does not
#: reduce this -- autocast casts activations, the parameters stay fp32.
PERSISTENT_BYTES_PER_PARAMETER = 16

#: Tensors the size of the residual stream that a block keeps alive for the
#: backward pass, per token. Counted off the module graph in
#: :mod:`trainai.model.gpt`: the two norm outputs, q/k/v, the attention output,
#: the two SwiGLU projections and their product, and the down projection. It is an
#: estimate and is used only to order candidates and to skip hopeless ones -- a
#: rung is never *rejected* on it. See :func:`rough_peak_bytes`.
ACTIVATION_SLOTS_PER_LAYER = 12


@dataclass(frozen=True)
class BenchmarkResult:
    """What running a configuration for a few steps actually did.

    Every field other than ``budget_bytes`` is measured on this machine. When
    ``memory_measured`` is False the platform gave us no allocator counters (CPU,
    MPS, and XPU builds without ``memory_stats``), and ``peak_bytes`` is 0 --
    which must be reported as "not measured" and never as "fits comfortably".
    """

    ok: bool
    verdict: str
    detail: str
    peak_bytes: int = 0
    reserved_bytes: int = 0
    budget_bytes: int = 0
    tokens_per_second: float = 0.0
    step_seconds: float = 0.0
    alloc_retries: int = 0
    steps_measured: int = 0
    tokens_per_step: int = 0
    precision: str = ""
    device: str = ""
    memory_measured: bool = False

    @property
    def budget_fraction(self) -> float:
        """Measured peak as a share of the budget. 0 when memory is unmeasurable."""
        if not self.memory_measured or self.budget_bytes <= 0:
            return 0.0
        return self.peak_bytes / self.budget_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "verdict": self.verdict,
            "detail": self.detail,
            "peak_bytes": self.peak_bytes,
            "reserved_bytes": self.reserved_bytes,
            "budget_bytes": self.budget_bytes,
            "budget_fraction": round(self.budget_fraction, 4),
            "tokens_per_second": round(self.tokens_per_second, 1),
            "step_seconds": round(self.step_seconds, 5),
            "alloc_retries": self.alloc_retries,
            "steps_measured": self.steps_measured,
            "tokens_per_step": self.tokens_per_step,
            "precision": self.precision,
            "device": self.device,
            "memory_measured": self.memory_measured,
        }


def skipped(reason: str, *, budget_bytes: int = 0) -> BenchmarkResult:
    """A candidate that was not run, labelled so it cannot be read as a measurement."""
    return BenchmarkResult(
        ok=False,
        verdict=VERDICT_SKIPPED,
        detail=reason,
        budget_bytes=budget_bytes,
    )


# --------------------------------------------------------------------------- #
# Estimates -- for ordering the search only, never for deciding
# --------------------------------------------------------------------------- #
def rough_peak_bytes(
    config: ModelConfig,
    *,
    batch_size: int,
    seq_len: int,
    activation_bytes: int = 2,
) -> int:
    """A cheap upper-ish estimate of training memory, in bytes.

    This is the formula that this module exists to distrust, and it is kept for
    exactly one purpose: deciding the order to measure candidates in, and skipping
    a candidate whose estimate is so far past the budget that measuring it would
    cost minutes of driver paging for an answer already known. A candidate is
    never rejected on this number -- rejection always cites a measurement, and
    anything skipped is reported as ``not-measured``.

    The three terms:

    * parameters: 16 bytes each (fp32 weights, gradients, two AdamW moments).
    * activations: ``ACTIVATION_SLOTS_PER_LAYER`` residual-width tensors per layer
      per token, at ``activation_bytes`` each (2 under autocast, 4 in fp32).
    * logits: ``batch x seq x vocab``, in fp32 because cross-entropy upcasts, and
      doubled for the gradient. On a small model with a large vocabulary this term
      is often the largest of the three, which is why it is not folded into a
      per-parameter constant.
    """
    tokens = batch_size * seq_len
    parameters = config.parameter_count * PERSISTENT_BYTES_PER_PARAMETER
    activations = tokens * config.d_model * config.n_layer * ACTIVATION_SLOTS_PER_LAYER
    activations *= activation_bytes
    logits = tokens * config.vocab_size * 4 * 2
    return parameters + activations + logits


def flops_per_token(config: ModelConfig, *, seq_len: int | None = None) -> float:
    """Estimated training FLOPs per token: ``6N`` plus the attention term.

    ``6N`` is the standard forward-plus-backward estimate (2N forward, 4N
    backward). The second term is attention's quadratic cost, which ``6N`` omits
    and which differs by a factor of two between the shortest and longest context
    on the preset ladder, so leaving it out would make a long-context candidate
    look less efficient than it is.

    Used only as a *ratio* between candidates measured on the same machine, to
    detect the throughput collapse that a spill produces. It is not accurate
    enough to quote as this hardware's FLOP rate, and nothing here does.
    """
    length = seq_len if seq_len is not None else config.seq_len
    return 6.0 * config.parameter_count + 12.0 * config.n_layer * length * config.d_model


# --------------------------------------------------------------------------- #
# The measurement
# --------------------------------------------------------------------------- #
def measure_candidate(
    config: ModelConfig,
    *,
    batch_size: int,
    grad_accum: int,
    seq_len: int,
    device: torch.device,
    precision: Precision = "auto",
    budget_bytes: int = 0,
    warmup_steps: int = 2,
    measure_steps: int = 3,
    seed: int = 1234,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
) -> BenchmarkResult:
    """Run this configuration for a few steps and report what it cost.

    The steps are the real thing: ``grad_accum`` micro-batches of forward and
    backward, gradient clipping, then an AdamW step, under the autocast dtype the
    trainer would resolve for this device. Warmup steps run first and are not
    timed -- the first step pays for kernel autotuning and the lazy allocation of
    AdamW's moment buffers, and including it would make a short benchmark look
    slower than the run it is predicting.

    Never raises for a configuration that does not fit: an out-of-memory error, a
    peak over ``budget_bytes``, or an unexpected exception all come back as a
    :class:`BenchmarkResult` with ``ok`` False and the evidence attached, because
    the caller's next move is to try a smaller candidate rather than to stop.

    Args:
        budget_bytes: Bytes this configuration may peak at. 0 means "no budget
            known", which is reported as unmeasured rather than as unlimited.
    """
    dtype, needs_scaler, precision_note = resolve_precision(precision, device)
    tokens_per_step = batch_size * grad_accum * seq_len

    model: GPT | None = None
    optimizer: torch.optim.Optimizer | None = None
    try:
        torch.manual_seed(seed)
        # net/opt are the ones the step closure uses; model/optimizer exist so the
        # `finally` clause can release whatever was built even if construction of
        # the second one raised.
        net = GPT(config).to(device)
        model = net
        opt = torch.optim.AdamW(
            net.parameter_groups(weight_decay),
            lr=1e-4,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=device.type == "cuda",
        )
        optimizer = opt
        scaler = torch.amp.GradScaler(device.type) if needs_scaler else None
        generator = torch.Generator().manual_seed(seed)

        def one_step() -> None:
            net.train()
            opt.zero_grad(set_to_none=True)
            for _ in range(grad_accum):
                window = torch.randint(
                    0,
                    config.vocab_size,
                    (batch_size, seq_len + 1),
                    generator=generator,
                    dtype=torch.int64,
                )
                inputs = window[:, :-1].to(device, non_blocking=True)
                targets = window[:, 1:].to(device, non_blocking=True)
                with _autocast(device, dtype):
                    _, loss, _ = net(inputs, targets)
                scaled = loss / grad_accum
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
            if scaler is not None:
                scaler.unscale_(opt)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()

        for _ in range(max(1, warmup_steps)):
            one_step()
        _synchronize(device)

        # Reset the peak *after* warmup so the figure is the steady state of a
        # long run. reset_peak_memory_stats sets the peak to what is currently
        # live rather than to zero, so the persistent parameters, gradients and
        # moment buffers are still counted.
        retries_before = _alloc_retries(device)
        if device.type == "cuda":
            with contextlib.suppress(Exception):
                torch.cuda.reset_peak_memory_stats(device)

        started = time.perf_counter()
        for _ in range(max(1, measure_steps)):
            one_step()
        _synchronize(device)
        elapsed = time.perf_counter() - started

        # Read the counters here, not after the try block. The ``finally`` below calls
        # _release, which calls reset_peak_memory_stats to stop this candidate's cached
        # blocks leaking into the next one -- and that would wipe the high-water mark
        # this whole function exists to report. Reading it after the release gave the
        # allocation still live at that moment, which on a small model is just the
        # parameter bytes: a 4.26M-parameter run measured 65 MiB where the real
        # training peak was 957 MiB.
        peak, reserved, memory_measured = _peak_memory(device)
        retries = max(0, _alloc_retries(device) - retries_before)

    except torch.cuda.OutOfMemoryError as exc:
        return _failure(
            VERDICT_OOM,
            "The allocator ran out of memory: " + _first_line(exc),
            budget_bytes=budget_bytes,
            precision=precision_note,
            device=device,
            tokens_per_step=tokens_per_step,
        )
    except RuntimeError as exc:
        # ROCm and XPU report exhaustion as a plain RuntimeError rather than as
        # torch.cuda.OutOfMemoryError, so the text has to be inspected. Anything
        # else is a real fault and is reported as one, still without raising:
        # the caller's job is to try something smaller and say what happened.
        message = str(exc)
        if _looks_like_oom(message):
            return _failure(
                VERDICT_OOM,
                "The device ran out of memory: " + _first_line(exc),
                budget_bytes=budget_bytes,
                precision=precision_note,
                device=device,
                tokens_per_step=tokens_per_step,
            )
        return _failure(
            VERDICT_ERROR,
            f"{exc.__class__.__name__}: {_first_line(exc)}",
            budget_bytes=budget_bytes,
            precision=precision_note,
            device=device,
            tokens_per_step=tokens_per_step,
        )
    except Exception as exc:  # a candidate must not abort the search
        return _failure(
            VERDICT_ERROR,
            f"{exc.__class__.__name__}: {_first_line(exc)}",
            budget_bytes=budget_bytes,
            precision=precision_note,
            device=device,
            tokens_per_step=tokens_per_step,
        )
    finally:
        _release(model, optimizer, device)

    steps = max(1, measure_steps)
    step_seconds = elapsed / steps

    result = BenchmarkResult(
        ok=True,
        verdict=VERDICT_FITS,
        detail="",
        peak_bytes=peak,
        reserved_bytes=reserved,
        budget_bytes=budget_bytes,
        tokens_per_second=(tokens_per_step / step_seconds) if step_seconds > 0 else 0.0,
        step_seconds=step_seconds,
        alloc_retries=retries,
        steps_measured=steps,
        tokens_per_step=tokens_per_step,
        precision=precision_note,
        device=str(device),
        memory_measured=memory_measured,
    )

    if memory_measured and budget_bytes > 0 and peak > budget_bytes:
        # The WDDM case: nothing raised, the steps completed, and the allocator's
        # own high-water mark is past what this GPU has to give. This is the
        # rejection the formula-plus-try/except approach misses.
        return _reject(
            result,
            VERDICT_OVER_BUDGET,
            f"Peaked at {fmt_bytes(peak)}, over the {fmt_bytes(budget_bytes)} budget "
            f"({result.budget_fraction:.1f}x). Nothing raised: on this platform the "
            "driver pages the excess to system RAM instead, so the run would finish "
            "at a fraction of the speed rather than fail.",
        )
    if retries > 0:
        return _reject(
            result,
            VERDICT_ALLOCATOR_PRESSURE,
            f"The allocator had to free cached blocks and retry {retries} time(s) in "
            f"{steps} steps, at a peak of {fmt_bytes(peak)}. It completed, but there is "
            "no headroom left for fragmentation or for anything else using the GPU.",
        )
    return result


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _autocast(device: torch.device, dtype: torch.dtype) -> Any:
    if dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _synchronize(device: torch.device) -> None:
    """Wait for the device, so a timer measures work rather than queue depth."""
    if device.type == "cuda":
        with contextlib.suppress(Exception):
            torch.cuda.synchronize(device)
    elif device.type == "xpu":  # pragma: no cover - no XPU here
        xpu = getattr(torch, "xpu", None)
        if xpu is not None:
            with contextlib.suppress(Exception):
                xpu.synchronize(device)
    elif device.type == "mps":  # pragma: no cover - no Apple hardware here
        mps = getattr(torch, "mps", None)
        if mps is not None:
            with contextlib.suppress(Exception):
                mps.synchronize()


def _peak_memory(device: torch.device) -> tuple[int, int, bool]:
    """Return ``(allocated_peak, reserved_peak, measured)``.

    ``measured`` False means this platform exposes no allocator counters, so the
    caller must say "not measured" rather than infer anything from a zero.
    """
    if device.type != "cuda":
        return 0, 0, False
    try:
        stats = torch.cuda.memory_stats(device)
    except Exception:  # pragma: no cover - driver-level failure
        try:
            return int(torch.cuda.max_memory_allocated(device)), 0, True
        except Exception:
            return 0, 0, False
    allocated = int(stats.get("allocated_bytes.all.peak", 0))
    reserved = int(stats.get("reserved_bytes.all.peak", 0))
    if allocated == 0:  # pragma: no cover - only on a build without these keys
        return int(torch.cuda.max_memory_allocated(device)), reserved, True
    return allocated, reserved, True


def _alloc_retries(device: torch.device) -> int:
    """``num_alloc_retries``: how often the allocator had to ask the driver twice."""
    if device.type != "cuda":
        return 0
    try:
        return int(torch.cuda.memory_stats(device).get("num_alloc_retries", 0))
    except Exception:  # pragma: no cover - driver-level failure
        return 0


def _release(model: Any, optimizer: Any, device: torch.device) -> None:
    """Give every byte back before the next candidate is measured.

    Without the collection and the cache drop, the next candidate inherits this
    one's cached blocks and either measures a peak that includes them or fails for
    a reason that belongs to its predecessor.
    """
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    del optimizer
    del model
    gc.collect()
    if device.type == "cuda":
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
        with contextlib.suppress(Exception):
            torch.cuda.reset_peak_memory_stats(device)


def _failure(
    verdict: str,
    detail: str,
    *,
    budget_bytes: int,
    precision: str,
    device: torch.device,
    tokens_per_step: int,
) -> BenchmarkResult:
    return BenchmarkResult(
        ok=False,
        verdict=verdict,
        detail=detail,
        budget_bytes=budget_bytes,
        precision=precision,
        device=str(device),
        tokens_per_step=tokens_per_step,
    )


def _reject(result: BenchmarkResult, verdict: str, detail: str) -> BenchmarkResult:
    """Keep every measured number, change the verdict to no."""
    return replace(result, ok=False, verdict=verdict, detail=detail)


def _looks_like_oom(message: str) -> bool:
    lowered = message.lower()
    return (
        "out of memory" in lowered
        or "cuda error: out of memory" in lowered
        or "hip out of memory" in lowered
        or "failed to allocate" in lowered
    )


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    return text[0] if text else exc.__class__.__name__


def describe(result: BenchmarkResult) -> str:
    """One line for a log or a table row."""
    if result.verdict == VERDICT_SKIPPED:
        return f"not measured {result.detail}"
    if not result.ok and result.steps_measured == 0:
        return f"{result.verdict}: {result.detail}"
    memory = fmt_bytes(result.peak_bytes) if result.memory_measured else "not measured"
    return f"{fmt_count(result.tokens_per_second)} tokens/s, peak {memory}, {result.verdict}"
