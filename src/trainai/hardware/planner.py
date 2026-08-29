"""``trainai plan`` -- decide what this machine can actually train, by measuring it.

The usual way to answer "will this model fit?" is a formula: count the parameters,
multiply by sixteen, add a guess for the activations, compare against the number on
the box. That formula is wrong often enough to waste an evening, and on Windows it is
wrong in the worst possible way -- exceeding VRAM under WDDM does not raise
:class:`torch.cuda.OutOfMemoryError`. The driver silently pages to system RAM and the
run continues at a fraction of the speed. A user watching a progress bar cannot tell
that apart from "this model is just big".

So this module does not predict. It runs
:func:`trainai.hardware.benchmark.measure_candidate` -- real forward, real backward,
real AdamW step -- at each rung of a ladder of candidate configurations, reads the
allocator's own counters, and keeps the largest rung that actually worked. The
reported time to finish is the measured step time multiplied by the step count, not an
extrapolation from a FLOP rate.

Five things can end the search, and the plan names whichever one did:

``vram-limited``
    The next rung's measured peak was over the safety budget, or the allocator threw,
    or throughput collapsed against the best rate seen on this machine.

``data-limited``
    The next rung has more parameters than this corpus can train -- the target is
    :data:`~trainai.train.budget.TOKENS_PER_PARAMETER_TARGET` tokens per parameter --
    so a bigger model would memorise rather than learn.

``compute-limited``
    The next rung fits and has the data behind it, but could not finish its step count
    inside the time horizon.

``unconstrained``
    Nothing stopped it: the largest preset on the ladder passed. This machine has room
    for a bigger model than the project ships a preset for.

``blocked``
    A candidate failed for a reason that is not a capacity limit at all. Reported as
    itself rather than folded into ``vram-limited``, because guessing at a cause would
    be a claim about something that was not measured.

Two design decisions worth stating, because both were tempting to get wrong.

The measurement function is *injectable* -- the ``measure`` argument of
:func:`plan_training`. That is what lets the search be unit-tested against synthetic
hardware profiles on a machine with no GPU at all, which is the only honest way this
project can claim to support hardware its author does not own.

On a machine with no CUDA device, :func:`measure_candidate` reports
``memory_measured=False`` and its over-budget check does not run. The plan says so in
:attr:`TrainingPlan.notes` rather than reporting a fit it did not verify. Throughput
and the time estimate are still real on CPU; only the memory fit is unknown, and the
data cap is what keeps a CPU user from being handed the largest preset.

What the measurement does not include: dataset read time. The benchmark feeds random
token ids so the number it reports is the model's cost rather than the disk's. On a
warm page cache the difference is small, and this module states that rather than
hiding it.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from trainai.console import fmt_bytes, fmt_count, fmt_duration, fmt_int
from trainai.data.binarize import DatasetManifest
from trainai.data.loader import largest_seq_len
from trainai.errors import CapacityError, DatasetError, UsageError
from trainai.hardware.benchmark import (
    VERDICT_ALLOCATOR_PRESSURE,
    VERDICT_ERROR,
    VERDICT_OOM,
    VERDICT_OVER_BUDGET,
    VERDICT_SKIPPED,
    BenchmarkResult,
    flops_per_token,
    measure_candidate,
    rough_peak_bytes,
    skipped,
)
from trainai.hardware.probe import DEFAULT_VRAM_SAFETY_FRACTION, HardwareProfile
from trainai.model.config import PRESETS, ModelConfig, preset
from trainai.train.budget import EPOCHS_BEFORE_MEMORISING, TOKENS_PER_PARAMETER_TARGET, DataBudget
from trainai.train.config import TrainConfig
from trainai.train.loop import resolve_device

__all__ = [
    "Candidate",
    "PlanCandidate",
    "TrainingPlan",
    "max_windows_for",
    "parse_duration",
    "plan_training",
    "recommended_lr",
]

#: Written into ``plan.json`` so ``trainai train --plan`` can refuse a file it does not
#: understand instead of silently misreading one.
PLAN_VERSION = 1
PLAN_FILENAME = "plan.json"

# --------------------------------------------------------------------------- #
# Search shape
# --------------------------------------------------------------------------- #
#: Tokens per optimiser step that the search aims for, held roughly constant across
#: sequence lengths by trading micro-batch against sequence length. Holding it fixed
#: is what makes the learning-rate rule of thumb and the derived step counts
#: comparable between rungs: a plan at seq 1024 and a plan at seq 256 then differ in
#: model shape rather than in how much text each update sees.
TARGET_TOKENS_PER_STEP = 16_384

#: Never recommend an effective batch smaller than this. Below roughly eight
#: sequences per update the gradient is noisy enough that the loss curve stops being
#: readable, which matters more here than the memory saved -- a user who cannot tell
#: whether training is working has no way to act on the run.
MIN_EFFECTIVE_BATCH = 8

#: A candidate whose rough estimate exceeds the budget by more than this multiple is
#: skipped rather than measured, and reported as ``not-measured``. Measuring a
#: configuration that is four times over budget costs minutes of driver paging for an
#: answer already known. Nothing is ever *rejected* on the estimate.
SKIP_ESTIMATE_MULTIPLE = 4.0

#: Throughput below this fraction of the best FLOP rate seen on this machine is read
#: as a spill rather than as a bigger model being slower. A rule of thumb, and only a
#: secondary net behind the allocator counters: on the development RTX 2050 the
#: configuration that spilled ran at 0.18 of the best observed rate, while the largest
#: configuration that merely did not fit ran at 0.76.
EFFICIENCY_COLLAPSE_RATIO = 0.35

#: Bounds on the derived step count. The floor keeps a tiny corpus from producing a
#: run too short to show a loss curve; the ceiling keeps a large one from producing a
#: plan measured in weeks without the user asking for it.
MIN_STEPS = 50
MAX_STEPS = 100_000

#: The default time horizon used to reject a rung as too slow when the user gave no
#: ``--time``. Six hours is one overnight run: long enough that a real model is
#: reachable, short enough that a plan exceeding it deserves to be flagged rather than
#: silently recommended.
SIZE_HORIZON_SECONDS = 6 * 3600

#: The learning-rate rule of thumb: 3e-4 at width 512, scaled as the inverse square
#: root of the width. Explicitly a rule of thumb and reported as one -- unlike memory
#: and throughput, nothing here measures whether it is the right learning rate, and
#: the only way to find that out is to run and look at the curve.
LR_REFERENCE_D_MODEL = 512
LR_AT_REFERENCE = 3e-4

# --------------------------------------------------------------------------- #
# What bound the plan
# --------------------------------------------------------------------------- #
REGIME_DATA_LIMITED = "data-limited"
REGIME_VRAM_LIMITED = "vram-limited"
REGIME_COMPUTE_LIMITED = "compute-limited"
REGIME_UNCONSTRAINED = "unconstrained"
REGIME_BLOCKED = "blocked"

REGIMES = (
    REGIME_DATA_LIMITED,
    REGIME_VRAM_LIMITED,
    REGIME_COMPUTE_LIMITED,
    REGIME_UNCONSTRAINED,
    REGIME_BLOCKED,
)

#: Verdicts the planner adds to the hardware verdicts in
#: :mod:`trainai.hardware.benchmark`. They live here rather than there because they
#: are not hardware facts: a corpus being too small and a run being too long are
#: judgements about this dataset and this user's patience.
VERDICT_NOT_ENOUGH_DATA = "not-enough-data"
VERDICT_TOO_SLOW = "too-slow"
VERDICT_THROUGHPUT_COLLAPSE = "throughput-collapse"


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Candidate:
    """One rung of the ladder: a model shape plus how it would be trained."""

    name: str
    model: ModelConfig
    seq_len: int
    micro_batch: int
    grad_accum: int

    @property
    def effective_batch(self) -> int:
        """Sequences per optimiser step, after accumulation."""
        return self.micro_batch * self.grad_accum

    @property
    def tokens_per_step(self) -> int:
        return self.effective_batch * self.seq_len

    def describe(self) -> str:
        # A micro-batch of zero means this rung was ruled out before it was ever sized
        # -- by the data cap, for instance. Printing "batch 0" would read as a bug.
        shape = f"{self.name} ({self.model.describe()})"
        if self.micro_batch <= 0:
            return f"{shape}, seq {self.seq_len}"
        batch = str(self.micro_batch)
        if self.grad_accum > 1:
            batch += f" x {self.grad_accum}"
        return f"{shape}, batch {batch}, seq {self.seq_len}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.name,
            "seq_len": self.seq_len,
            "micro_batch": self.micro_batch,
            "grad_accum": self.grad_accum,
            "effective_batch": self.effective_batch,
            "tokens_per_step": self.tokens_per_step,
            "parameters": self.model.parameter_count,
            "model": self.model.to_dict(),
        }


@dataclass(frozen=True)
class PlanCandidate:
    """A candidate together with what happened when it was tried."""

    candidate: Candidate
    result: BenchmarkResult

    @property
    def ok(self) -> bool:
        return self.result.ok

    def to_dict(self) -> dict[str, Any]:
        return {"candidate": self.candidate.to_dict(), "result": self.result.to_dict()}


@dataclass
class TrainingPlan:
    """A recommendation, the measurements behind it, and everything it rejected.

    The rejected candidates are kept rather than discarded because "why not something
    bigger?" is the first question a user asks, and the honest answer is a list of
    configurations with the measurement that ruled each one out.
    """

    model_config: ModelConfig
    train_config: TrainConfig
    budget: DataBudget
    regime: str
    regime_detail: str
    hardware: HardwareProfile
    measurement: BenchmarkResult
    candidates: list[PlanCandidate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    dataset_path: str = ""
    preset_name: str = ""
    vram_budget_bytes: int = 0

    @property
    def estimated_seconds(self) -> float:
        """Measured step time times step count.

        Not an extrapolation from a FLOP rate: :attr:`measurement` came from running
        this exact configuration. It excludes dataset read time, which the benchmark
        deliberately does not measure -- see the module docstring.
        """
        return self.measurement.step_seconds * self.train_config.steps

    @property
    def accepted(self) -> PlanCandidate | None:
        """The candidate that was recommended, if it is in the record."""
        for record in reversed(self.candidates):
            if record.ok:
                return record
        return None

    @property
    def rejected(self) -> list[PlanCandidate]:
        return [record for record in self.candidates if not record.ok]

    def train_command(self, dataset: str | None = None) -> str:
        """The exact ``trainai train`` command this plan describes.

        This is the review step: the user reads a command, changes it if they disagree,
        and runs it. A plan that could only be applied by a flag the user cannot see
        would make the recommendation unauditable.
        """
        data = dataset if dataset is not None else (self.dataset_path or "DATASET")
        config = self.train_config
        parts = [
            "trainai train",
            f"--data {data}",
            f"--preset {self.preset_name}",
            f"--context {self.model_config.seq_len}",
            f"--seq-len {config.seq_len}",
            f"--steps {config.steps}",
            f"--batch-size {config.batch_size}",
        ]
        if config.grad_accum > 1:
            parts.append(f"--grad-accum {config.grad_accum}")
        parts.append(f"--lr {config.lr:g}")
        parts.append(f"--eval-every {config.eval_every}")
        parts.append(f"--eval-batches {config.eval_batches}")
        parts.append(f"--checkpoint-every {config.checkpoint_every}")
        if config.precision != "auto":
            parts.append(f"--precision {config.precision}")
        if config.device != "auto":
            parts.append(f"--device {config.device}")
        return " ".join(parts)

    def provenance(self) -> dict[str, Any]:
        """Split every number in the plan into measured, derived, or rule of thumb.

        The whole point of this project is that these three are not the same kind of
        claim, so the plan refuses to present them in one undifferentiated table. A
        ``None`` under ``measured`` means the platform exposes no counter for it, not
        that the value was zero.
        """
        gpu = self.hardware.primary_gpu
        measured = self.measurement
        return {
            "measured": {
                "peak_bytes": measured.peak_bytes if measured.memory_measured else None,
                "reserved_bytes": measured.reserved_bytes if measured.memory_measured else None,
                "memory_measured": measured.memory_measured,
                "tokens_per_second": measured.tokens_per_second,
                "step_seconds": measured.step_seconds,
                "steps_measured": measured.steps_measured,
                "alloc_retries": measured.alloc_retries,
                "device": measured.device,
                "precision": measured.precision,
                "vram_free_bytes": gpu.free_vram_bytes if gpu is not None else None,
                "vram_budget_bytes": self.vram_budget_bytes or None,
            },
            "derived_from_data": {
                "steps": self.train_config.steps,
                "tokens_per_step": self.train_config.tokens_per_step,
                "epochs": self.budget.epochs,
                "tokens_per_parameter": self.budget.tokens_per_parameter,
            },
            "rules_of_thumb": {
                "lr": self.train_config.lr,
                "schedule": self.train_config.schedule,
                "warmup_steps": self.train_config.resolved_warmup_steps,
                "eval_every": self.train_config.eval_every,
                "eval_batches": self.train_config.eval_batches,
                "checkpoint_every": self.train_config.checkpoint_every,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": PLAN_VERSION,
            "dataset": self.dataset_path,
            "preset": self.preset_name,
            "regime": self.regime,
            "regime_detail": self.regime_detail,
            "model": self.model_config.to_dict(),
            "train": self.train_config.to_dict(),
            "budget": self.budget.to_dict(),
            "measurement": self.measurement.to_dict(),
            "estimated_seconds": self.estimated_seconds,
            "train_command": self.train_command(),
            "hardware": self.hardware.to_dict(),
            "candidates": [record.to_dict() for record in self.candidates],
            "notes": list(self.notes),
            "provenance": self.provenance(),
        }

    def write(self, destination: str | Path) -> Path:
        """Write ``plan.json``, either at a path or inside a directory."""
        path = Path(destination)
        target = path if path.suffix == ".json" else path / PLAN_FILENAME
        if target.parent != Path():
            target.parent.mkdir(parents=True, exist_ok=True)
        # ``newline="\n"`` so a plan written on Windows is byte-identical to one
        # written on Linux, and two plans can be diffed across machines.
        target.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        return target


def _rejected(verdict: str, detail: str, *, budget_bytes: int = 0) -> BenchmarkResult:
    """A rejection decided before any measurement was taken.

    ``steps_measured`` stays zero, which is what
    :func:`trainai.hardware.benchmark.describe` uses to render this as a reason rather
    than as a throughput reading that was never taken.
    """
    return BenchmarkResult(ok=False, verdict=verdict, detail=detail, budget_bytes=budget_bytes)


def _rejected_after(result: BenchmarkResult, verdict: str, detail: str) -> BenchmarkResult:
    """Keep every measured number; change the verdict to no."""
    return replace(result, ok=False, verdict=verdict, detail=detail)


#: Verdicts that mean "this did not fit", from either module. Grouped here so the
#: regime mapping has one place to be wrong rather than five.
_MEMORY_VERDICTS = frozenset(
    {
        VERDICT_OVER_BUDGET,
        VERDICT_OOM,
        VERDICT_ALLOCATOR_PRESSURE,
        VERDICT_THROUGHPUT_COLLAPSE,
        VERDICT_SKIPPED,
    }
)


def _regime_for(stop: PlanCandidate | None) -> tuple[str, str]:
    """Name the constraint that ended the search, and quote the evidence for it.

    ``stop`` is the candidate that stopped the ladder, or ``None`` when nothing did.
    The detail always repeats the actual verdict and the measurement's own words, so a
    user who disagrees with the regime can see what it was inferred from.
    """
    if stop is None:
        return (
            REGIME_UNCONSTRAINED,
            "Every rung fitted, had the data behind it, and finished inside the time "
            "horizon, so the largest preset was accepted.",
        )
    verdict = stop.result.verdict
    label = stop.candidate.describe()
    detail = stop.result.detail
    if verdict in _MEMORY_VERDICTS:
        return REGIME_VRAM_LIMITED, f"{label} was rejected -- {detail}"
    if verdict == VERDICT_NOT_ENOUGH_DATA:
        return REGIME_DATA_LIMITED, f"{label} was rejected -- {detail}"
    if verdict == VERDICT_TOO_SLOW:
        return REGIME_COMPUTE_LIMITED, f"{label} was rejected -- {detail}"
    if verdict == VERDICT_ERROR:
        return REGIME_BLOCKED, f"{label} failed for a reason unrelated to capacity -- {detail}"
    return REGIME_BLOCKED, f"{label} ended the search with verdict {verdict!r} -- {detail}"


# --------------------------------------------------------------------------- #
# Arithmetic the search depends on
# --------------------------------------------------------------------------- #
_DURATION_HINT = "Give minutes as a bare number, or use units: 90s, 45m, 2h, 1h30m, 1d."
_UNIT_SECONDS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([smhd])")


def parse_duration(raw: str) -> float:
    """Read ``90s``, ``45m``, ``2h``, ``1h30m`` or a bare number of minutes, in seconds.

    A bare number means minutes: ``--time 30`` reads as half an hour to anyone who has
    waited for a training run, and treating it as thirty seconds would quietly produce
    a plan of a handful of steps and call it a recommendation.
    """
    text = raw.strip().lower()
    if not text:
        raise UsageError("A time budget cannot be empty.", hint=_DURATION_HINT)

    try:
        minutes = float(text)
    except ValueError:
        pass
    else:
        if minutes <= 0:
            raise UsageError(f"A time budget must be positive, not {raw!r}.", hint=_DURATION_HINT)
        return minutes * 60.0

    total = 0.0
    matched = False
    for amount, unit in _DURATION_RE.findall(text):
        total += float(amount) * _UNIT_SECONDS[unit]
        matched = True
    # Anything the pattern did not consume is a typo, not a unit to ignore. Silently
    # reading "2 hours" as 2 seconds would be worse than refusing it.
    leftover = _DURATION_RE.sub("", text).strip()
    if not matched or leftover:
        raise UsageError(
            f"Cannot read {raw!r} as a duration.",
            hint=_DURATION_HINT,
            details={"value": raw, "not_understood": leftover},
        )
    if total <= 0:
        raise UsageError(f"A time budget must be positive, not {raw!r}.", hint=_DURATION_HINT)
    return total


def max_windows_for(tokens: int, seq_len: int) -> int:
    """How many training windows a split of ``tokens`` tokens yields at ``seq_len``.

    This repeats the arithmetic in :class:`trainai.data.loader.TokenBatcher` on
    purpose. ``TokenBatcher`` raises :class:`~trainai.errors.DatasetError` when the
    batch size exceeds the window count, so the planner has to know the same number to
    avoid recommending a configuration that cannot open its own dataset. The test suite
    checks the two against each other rather than trusting the comment.
    """
    usable = int(tokens) - int(seq_len)
    if usable <= 0:
        return 0
    return max(0, usable // (int(seq_len) + 1))


def _round_significant(value: float, digits: int) -> float:
    if value == 0.0 or not math.isfinite(value):
        return 0.0
    exponent = math.floor(math.log10(abs(value)))
    return round(value, -(exponent - digits + 1))


def recommended_lr(d_model: int) -> float:
    """A starting learning rate for this width: 3e-4 at 512, scaled as 1/sqrt(width).

    A rule of thumb, reported as one. Unlike the memory and throughput numbers in a
    plan, nothing here measures whether this is the right learning rate for this
    corpus -- the only way to find that out is to run and look at the curve. It is
    included because "pick a learning rate" is the question that stops people who have
    never trained a model, and a defensible starting point beats a blank.
    """
    if d_model <= 0:
        return LR_AT_REFERENCE
    return _round_significant(LR_AT_REFERENCE * math.sqrt(LR_REFERENCE_D_MODEL / d_model), 3)


def _data_derived_steps(train_tokens: int, parameters: int, tokens_per_step: int) -> int:
    """How many steps this corpus supports for a model of this size.

    The smaller of two limits: enough tokens to train the parameters
    (:data:`~trainai.train.budget.TOKENS_PER_PARAMETER_TARGET` per parameter) and few
    enough passes over the corpus to stay short of memorising it
    (:data:`~trainai.train.budget.EPOCHS_BEFORE_MEMORISING`). Both numbers and the
    evidence for them live in :mod:`trainai.train.budget`; this is the same arithmetic
    used to choose a shape rather than only to describe one.
    """
    if tokens_per_step <= 0:
        return MIN_STEPS
    by_epochs = EPOCHS_BEFORE_MEMORISING * float(max(0, train_tokens))
    by_size = TOKENS_PER_PARAMETER_TARGET * float(max(0, parameters))
    steps = int(min(by_epochs, by_size) // tokens_per_step)
    return max(MIN_STEPS, min(MAX_STEPS, steps))


def _largest_divisor_at_most(value: int, cap: int) -> int:
    """The largest divisor of ``value`` that is no greater than ``cap``.

    Used whenever something caps the micro-batch. Snapping to a divisor keeps
    ``grad_accum`` exact, so the effective batch stays the number the search aimed for
    instead of overshooting it: capping 64 at 38 gives 32, not 38, and 32 x 2 is 64
    rather than 38 x 2 being 76.
    """
    if cap >= value:
        return value
    for candidate in range(max(1, min(cap, value)), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _cadence(steps: int, *, target_count: int, floor: int, ceiling: int) -> int:
    """How often to do something ``target_count`` times over ``steps`` steps.

    The floor stops a short run from evaluating every three steps, which is noise
    rather than a signal. The final ``min`` with ``steps`` guarantees the cadence fires
    at least once, so a very short run still leaves a checkpoint behind.
    """
    return max(1, min(ceiling, max(floor, steps // max(1, target_count)), max(1, steps)))


def _start_micro_batch(effective: int, train_windows: int) -> int:
    """The largest micro-batch worth trying first, before memory has a say.

    Capped at the training window count because ``TokenBatcher`` refuses a batch it
    cannot fill, then snapped down to a divisor of the effective batch.

    It used to be capped at the *validation* window count too, because the trainer
    opened validation at the micro-batch size and silently gave up when the split held
    fewer windows than that -- so a plan had to shrink the training batch to keep a
    held-out loss. The trainer now sizes the validation batch to the split, as
    ``trainai eval`` always did, and the training batch is no longer shaped by it:
    measured at --seq-len 256 on a 5,000-token validation split, the plan moved from
    --batch-size 16 with --grad-accum 4 to --batch-size 64 with --grad-accum 1, the
    same effective batch in a quarter of the launches.
    """
    micro = max(1, effective)
    if train_windows >= 1:
        micro = min(micro, train_windows)
    return _largest_divisor_at_most(max(1, effective), max(1, micro))


def _micro_batch_ladder(start: int, effective: int) -> list[int]:
    """``start``, then halves down to 1: what to try as memory rejections come back.

    Halving the micro-batch while raising accumulation keeps the effective batch the
    same, so a rejected candidate is retried at the same learning dynamics rather than
    being quietly turned into a different plan.
    """
    ladder: list[int] = []
    micro = max(1, min(start, max(1, effective)))
    while True:
        ladder.append(micro)
        if micro == 1:
            return ladder
        micro //= 2


def _preset_ladder(max_preset: str | None) -> list[Any]:
    """The presets to try, smallest first, truncated at ``max_preset`` if given."""
    if max_preset is None:
        return list(PRESETS)
    names = [spec.name for spec in PRESETS]
    if max_preset not in names:
        raise UsageError(
            f"Unknown preset {max_preset!r}.",
            hint=f"Choose one of: {', '.join(names)}.",
            details={"presets": names},
        )
    return list(PRESETS[: names.index(max_preset) + 1])


# --------------------------------------------------------------------------- #
# Judging one measurement
# --------------------------------------------------------------------------- #
def _record(
    records: list[PlanCandidate], candidate: Candidate, result: BenchmarkResult
) -> PlanCandidate:
    """Append to the audit trail and hand the record back.

    Every candidate the search touches ends up here, accepted or not. A plan that only
    reported its winner could not answer "why not something bigger?".
    """
    record = PlanCandidate(candidate, result)
    records.append(record)
    return record


def _check_collapse(result: BenchmarkResult, rate: float, best_rate: float) -> BenchmarkResult:
    """Reject a measurement whose throughput collapsed against the best seen here.

    This is the WDDM net. A configuration that spills to system RAM keeps running and
    reports no error, so the only signal left is that it became absurdly slow relative
    to what this same machine managed a moment ago. Compared as FLOPs per second rather
    than tokens per second, because a larger model is legitimately slower per token.

    Deliberately a loose threshold: a false positive costs the user a smaller model
    than they could have had, while a false negative costs them a run that appears to
    work and takes twenty times as long. The allocator counters remain the primary
    check; this only catches what they miss.
    """
    if best_rate <= 0 or rate <= 0:
        return result
    ratio = rate / best_rate
    if ratio >= EFFICIENCY_COLLAPSE_RATIO:
        return result
    return _rejected_after(
        result,
        VERDICT_THROUGHPUT_COLLAPSE,
        f"throughput fell to {ratio:.2f} of the best rate measured on this machine "
        f"({fmt_count(result.tokens_per_second)} tokens/s), which is what memory "
        "spilling to system RAM looks like rather than what a bigger model looks like",
    )


def _check_horizon(
    result: BenchmarkResult, candidate: Candidate, train_tokens: int, horizon: float
) -> BenchmarkResult:
    """Reject a rung that cannot finish the step count its size calls for.

    Note what is being compared: not "is this run long?" but "can this model be trained
    properly inside the time available?". Recommending a model that has to be stopped
    at a third of its step count is recommending an undertrained model, which is a
    worse answer than recommending a smaller one that finishes.
    """
    steps = _data_derived_steps(
        train_tokens, candidate.model.parameter_count, candidate.tokens_per_step
    )
    projected = result.step_seconds * steps
    if projected <= horizon:
        return result
    return _rejected_after(
        result,
        VERDICT_TOO_SLOW,
        f"the {fmt_int(steps)} steps this size needs, at the measured "
        f"{result.step_seconds:.3f}s each, would take {fmt_duration(projected)} -- past "
        f"the {fmt_duration(horizon)} budget",
    )


def _capacity_hint(records: list[PlanCandidate], budget_bytes: int) -> str:
    """What to actually do about a machine that could not train anything."""
    if not records:
        return "No candidate was attempted at all, which is a bug worth reporting."
    last = records[-1].result
    if last.verdict == VERDICT_NOT_ENOUGH_DATA:
        return (
            "The corpus is too small even for the smallest preset. Add more text, or "
            "lower --seq-len so the same tokens yield more training windows."
        )
    if last.verdict in _MEMORY_VERDICTS:
        budget = fmt_bytes(budget_bytes) if budget_bytes > 0 else "available"
        return (
            f"The smallest preset at a micro-batch of 1 did not fit the {budget} memory "
            "budget. Close other GPU applications and try again, lower --seq-len, or "
            "pass --device cpu to train slowly rather than not at all."
        )
    if last.verdict == VERDICT_TOO_SLOW:
        return (
            "Nothing fits the time budget. Raise --time, or lower --seq-len to make each "
            "step cheaper."
        )
    return f"The last candidate failed with verdict {last.verdict!r}: {last.detail}"


# --------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------- #
def plan_training(
    dataset: DatasetManifest,
    *,
    hardware: HardwareProfile,
    device: str = "auto",
    dataset_path: str = "",
    time_budget_seconds: float | None = None,
    max_preset: str | None = None,
    seq_len: int | None = None,
    precision: str = "auto",
    safety_fraction: float = DEFAULT_VRAM_SAFETY_FRACTION,
    measure: Callable[..., BenchmarkResult] = measure_candidate,
    warmup_steps: int = 2,
    measure_steps: int = 3,
    seed: int = 1234,
    on_candidate: Callable[[Candidate], None] | None = None,
    on_result: Callable[[Candidate, BenchmarkResult], None] | None = None,
) -> TrainingPlan:
    """Measure this machine and return the largest plan it can actually train.

    The search climbs the preset ladder from ``tiny`` upwards and keeps the last rung
    that passed. Climbing rather than descending means the pathological configurations
    are never run: on the development GPU the 110M-parameter shape at sequence length
    1024 spent minutes paging at 1,300 tokens per second, and a descending search would
    have paid that cost on every planning run.

    Within a rung, a rejection halves the micro-batch and raises gradient accumulation
    to hold the effective batch constant, so a candidate is retried at the same learning
    dynamics rather than quietly becoming a different plan.

    Three caps apply, and :attr:`TrainingPlan.regime` names whichever one bound:

    * **memory** -- the measured peak must fit ``safety_fraction`` of *free* VRAM.
    * **data** -- the model must not have more parameters than the corpus can train.
    * **time** -- the rung must finish its data-derived step count inside the horizon.

    The data and time caps are only applied *after* something smaller has already
    passed. The smallest rung is therefore always measured, which is what guarantees
    that a :class:`~trainai.errors.CapacityError` carries a real measurement rather
    than a refusal derived from arithmetic.

    ``measure`` is injectable so this whole search can be unit-tested against synthetic
    hardware profiles on a machine with no GPU. ``on_candidate`` and ``on_result`` let a
    caller report each rung as it happens, because a real measurement pass takes long
    enough that silence looks like a hang.

    Raises :class:`~trainai.errors.DatasetError` if the dataset has no training tokens,
    and :class:`~trainai.errors.CapacityError` if every candidate was rejected.
    """
    train_tokens = dataset.tokens("train")
    if train_tokens <= 0:
        raise DatasetError(
            "This dataset has no training tokens, so there is nothing to plan for.",
            hint=(
                "Run `trainai data prepare` on your text first, then point `trainai plan` "
                "at the directory it wrote."
            ),
            details={"dataset": dataset_path or "<unknown>"},
        )
    if seq_len is not None and int(seq_len) < 8:
        raise UsageError(
            f"A sequence length of {seq_len} is too short to train on.",
            hint="Use at least 8, and in practice 256 or more.",
        )

    val_tokens = dataset.tokens("val")
    budget_bytes = hardware.vram_budget_bytes(safety_fraction)
    torch_device = resolve_device(device)
    horizon = float(time_budget_seconds) if time_budget_seconds else float(SIZE_HORIZON_SECONDS)
    ladder = _preset_ladder(max_preset)

    records: list[PlanCandidate] = []
    notes: list[str] = []
    accepted: PlanCandidate | None = None
    stop: PlanCandidate | None = None
    best_flop_rate = 0.0

    for spec in ladder:
        seq = int(seq_len) if seq_len else int(spec.seq_len)
        model = preset(spec.name, vocab_size=dataset.vocab_size, seq_len=seq)
        effective = max(MIN_EFFECTIVE_BATCH, TARGET_TOKENS_PER_STEP // seq)
        train_windows = max_windows_for(train_tokens, seq)
        shape = Candidate(spec.name, model, seq, 0, 0)

        # Cap (b): data. A model with more parameters than the corpus can train will
        # memorise it, which looks like success in the training loss and like nothing
        # at all in the validation loss.
        wanted = int(model.parameter_count * TOKENS_PER_PARAMETER_TARGET)
        if accepted is not None and train_tokens < wanted:
            stop = _record(
                records,
                shape,
                _rejected(
                    VERDICT_NOT_ENOUGH_DATA,
                    f"{fmt_count(model.parameter_count)} parameters want about "
                    f"{fmt_count(wanted)} training tokens and this corpus has "
                    f"{fmt_count(train_tokens)}, so it would memorise rather than learn",
                    budget_bytes=budget_bytes,
                ),
            )
            break

        if train_windows < 1:
            stop = _record(
                records,
                shape,
                _rejected(
                    VERDICT_NOT_ENOUGH_DATA,
                    f"a sequence length of {fmt_int(seq)} leaves no complete training "
                    f"window in {fmt_count(train_tokens)} tokens",
                    budget_bytes=budget_bytes,
                ),
            )
            break

        rung_ok: PlanCandidate | None = None
        rung_stop: PlanCandidate | None = None
        start = _start_micro_batch(effective, train_windows)

        for micro in _micro_batch_ladder(start, effective):
            grad_accum = max(1, math.ceil(effective / micro))
            candidate = Candidate(spec.name, model, seq, micro, grad_accum)

            # Skipped, never rejected, on the estimate: measuring something several
            # times over budget costs minutes of driver paging for a known answer.
            estimate = rough_peak_bytes(model, batch_size=micro, seq_len=seq)
            if budget_bytes > 0 and estimate > budget_bytes * SKIP_ESTIMATE_MULTIPLE:
                rung_stop = _record(
                    records,
                    candidate,
                    skipped(
                        f"-- the estimate of {fmt_bytes(estimate)} is over "
                        f"{SKIP_ESTIMATE_MULTIPLE:g}x the {fmt_bytes(budget_bytes)} "
                        "budget, so it was not run",
                        budget_bytes=budget_bytes,
                    ),
                )
                continue

            if on_candidate is not None:
                on_candidate(candidate)
            result = measure(
                model,
                batch_size=micro,
                grad_accum=grad_accum,
                seq_len=seq,
                device=torch_device,
                precision=precision,
                budget_bytes=budget_bytes,
                warmup_steps=warmup_steps,
                measure_steps=measure_steps,
                seed=seed,
            )
            rate = flops_per_token(model, seq_len=seq) * result.tokens_per_second
            if result.ok:
                result = _check_collapse(result, rate, best_flop_rate)
            # Cap (c): time. Like the data cap, only once something has passed, so the
            # first measurement is never thrown away for being slow.
            if result.ok and accepted is not None:
                result = _check_horizon(result, candidate, train_tokens, horizon)
            if on_result is not None:
                on_result(candidate, result)
            record = _record(records, candidate, result)

            if result.ok:
                best_flop_rate = max(best_flop_rate, rate)
                rung_ok = record
                break
            rung_stop = record
            if result.verdict == VERDICT_TOO_SLOW:
                # Halving the micro-batch raises accumulation and makes the step
                # slower, never faster, so nothing below this is worth measuring.
                break

        if rung_ok is None:
            stop = rung_stop
            break
        accepted = rung_ok

    if accepted is None:
        raise CapacityError(
            "Nothing on the candidate ladder could be trained on this machine.",
            hint=_capacity_hint(records, budget_bytes),
            details={
                "dataset": dataset_path or "<unknown>",
                "device": str(torch_device),
                "vram_budget_bytes": budget_bytes,
                "train_tokens": train_tokens,
                "candidates": [record.to_dict() for record in records],
            },
        )

    return _assemble(
        accepted=accepted,
        stop=stop,
        records=records,
        notes=notes,
        dataset_path=dataset_path,
        hardware=hardware,
        budget_bytes=budget_bytes,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        horizon=horizon,
        time_budget_seconds=time_budget_seconds,
        max_preset=max_preset,
        ladder=ladder,
        precision=precision,
        device=device,
        seed=seed,
    )


def _assemble(
    *,
    accepted: PlanCandidate,
    stop: PlanCandidate | None,
    records: list[PlanCandidate],
    notes: list[str],
    dataset_path: str,
    hardware: HardwareProfile,
    budget_bytes: int,
    train_tokens: int,
    val_tokens: int,
    horizon: float,
    time_budget_seconds: float | None,
    max_preset: str | None,
    ladder: list[Any],
    precision: str,
    device: str,
    seed: int,
) -> TrainingPlan:
    """Turn the winning candidate into a full :class:`TrainingPlan`.

    Separated from the search so that the search reads as a search. Everything here is
    arithmetic over decisions already made, except for one real decision: the time
    horizon also *cuts* the step count, rather than only rejecting larger rungs.

    That cut is what keeps the plan self-consistent. The smallest rung is exempt from
    the time cap during the search -- otherwise a slow machine could produce a
    CapacityError with nothing measured -- so without a cut here a slow machine could be
    told "``small`` was rejected because it needs 6h31m" and then handed a ``tiny`` plan
    needing 21 hours. Rejecting one rung for overrunning a horizon while recommending a
    slower one is not a defensible answer, whichever number is right.
    """
    candidate = accepted.candidate
    result = accepted.result
    model_config = candidate.model
    steps = _data_derived_steps(
        train_tokens, model_config.parameter_count, candidate.tokens_per_step
    )
    regime, regime_detail = _regime_for(stop)

    if result.step_seconds > 0:
        affordable = int(horizon // result.step_seconds)
        if affordable < steps:
            if time_budget_seconds is not None:
                notes.append(
                    f"Cut to fit the time budget: {fmt_duration(horizon)} affords "
                    f"{fmt_int(max(1, affordable))} steps at the measured "
                    f"{result.step_seconds:.3f}s each, but this corpus and model size "
                    f"would support {fmt_int(steps)}. The model will be undertrained -- "
                    "raise --time to let it finish."
                )
            else:
                notes.append(
                    f"Cut to fit the default {fmt_duration(horizon)} horizon: that "
                    f"affords {fmt_int(max(1, affordable))} steps at the measured "
                    f"{result.step_seconds:.3f}s each, but this corpus and model size "
                    f"would support {fmt_int(steps)}. Pass --time to train for longer; "
                    "the run can also be stopped early and resumed from a checkpoint."
                )
            if affordable < MIN_STEPS:
                notes.append(
                    f"That is fewer than {MIN_STEPS} steps. A run this short is worth "
                    "doing only as a check that the pipeline works, not as a model."
                )
            steps = max(1, affordable)
            if regime == REGIME_UNCONSTRAINED:
                regime = REGIME_COMPUTE_LIMITED
                regime_detail = (
                    "the preset ladder was not the limit -- the "
                    f"{fmt_duration(horizon)} time horizon was, and it cut the step "
                    "count short of what the data supports"
                )

    # Evaluation and checkpoint cadence have to scale with the step count. The trainer
    # forces an evaluation on the final step, so a 150-step run at the default
    # eval_every of 250 would evaluate exactly once and leave best-checkpoint selection
    # with nothing to choose between.
    val_windows = max_windows_for(val_tokens, candidate.seq_len)
    eval_every = _cadence(steps, target_count=20, floor=5, ceiling=TrainConfig.eval_every)
    checkpoint_every = _cadence(
        steps, target_count=10, floor=10, ceiling=TrainConfig.checkpoint_every
    )
    if val_windows >= 1:
        # The trainer opens validation at ``min(batch_size, val_windows)`` rows. That is
        # the same batch count as dividing by the micro-batch alone -- whichever is
        # smaller does the work -- but it is written the way the trainer works, so the
        # two cannot drift if either changes.
        rows = max(1, min(candidate.micro_batch, val_windows))
        batches = math.ceil(val_windows / rows)
        eval_batches = max(1, min(TrainConfig.eval_batches, batches))
    else:
        eval_batches = TrainConfig.eval_batches

    if val_tokens <= 0:
        notes.append(
            "This dataset has no validation split, so nothing in the run can tell "
            "learning from memorising. Re-prepare it with --val-fraction 0.1."
        )
    elif val_windows < 1:
        # Fires on the window count, not on the micro-batch. It used to say validation
        # would not run whenever the micro-batch exceeded the window count, which was
        # true of the trainer then and is not now -- the batch is sized to the split, so
        # one window is enough. Keeping that wording would have promised a missing
        # number the run does in fact produce.
        usable = largest_seq_len(val_tokens)
        fix = (
            f"Plan with --seq-len {usable} or lower, or re-prepare the dataset with a "
            "larger --val-fraction."
            if usable
            else "Re-prepare the dataset with a larger --val-fraction."
        )
        notes.append(
            f"Validation will not run: the validation split holds {fmt_int(val_tokens)} "
            f"token(s), not enough for one sequence of {fmt_int(candidate.seq_len)}. "
            f"{fix}"
        )
    if not result.memory_measured:
        notes.append(
            f"Memory was not measured: {result.device or 'this device'} exposes no "
            "allocator counters, so the fit was not verified. The throughput and the "
            "time estimate are real measurements; the peak usage is unknown."
        )
    if candidate.grad_accum > 1:
        notes.append(
            f"Gradient accumulation is in use: {candidate.micro_batch} sequence(s) at a "
            f"time, {candidate.grad_accum} times per step, for an effective batch of "
            f"{candidate.effective_batch}. That is what made this shape fit."
        )
    wanted_batch = max(MIN_EFFECTIVE_BATCH, TARGET_TOKENS_PER_STEP // candidate.seq_len)
    if candidate.effective_batch != wanted_batch:
        notes.append(
            f"The effective batch is {candidate.effective_batch} rather than the "
            f"{wanted_batch} aimed for, because {wanted_batch} is not a whole multiple of "
            f"the {candidate.micro_batch} that fitted."
        )
    if max_preset is not None and ladder and ladder[-1].name != PRESETS[-1].name:
        notes.append(
            f"The search stopped at the {max_preset!r} preset because --max-preset said "
            "to. Anything larger was not measured."
        )
    if time_budget_seconds is None and stop is not None and stop.result.verdict == VERDICT_TOO_SLOW:
        # A size cap the user did not ask for has to name itself, or it reads as a
        # hardware limit. Only for the rejection case -- if the *cut* above set the
        # regime, its own note already said so.
        notes.append(
            f"The model size was capped by the default {fmt_duration(horizon)} horizon "
            "rather than by anything you asked for: a larger model fitted the memory "
            "budget but could not finish its step count in that time. Pass --time to "
            "allow a longer run, then plan again."
        )
    notes.append(
        "The measured step time excludes dataset read time -- the benchmark feeds random "
        "token ids so the number is the model's cost rather than the disk's. On a warm "
        "page cache the difference is small."
    )

    train_config = TrainConfig(
        steps=steps,
        batch_size=candidate.micro_batch,
        grad_accum=candidate.grad_accum,
        seq_len=candidate.seq_len,
        lr=recommended_lr(model_config.d_model),
        eval_every=eval_every,
        eval_batches=eval_batches,
        checkpoint_every=checkpoint_every,
        seed=seed,
        precision=precision,
        device=device,
    )
    budget = DataBudget(
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        parameters=model_config.parameter_count,
        non_embedding_parameters=model_config.non_embedding_parameter_count,
        tokens_per_step=train_config.tokens_per_step,
        steps=train_config.steps,
    )
    return TrainingPlan(
        model_config=model_config,
        train_config=train_config,
        budget=budget,
        regime=regime,
        regime_detail=regime_detail,
        hardware=hardware,
        measurement=result,
        candidates=records,
        notes=notes,
        dataset_path=dataset_path,
        preset_name=candidate.name,
        vram_budget_bytes=budget_bytes,
    )
