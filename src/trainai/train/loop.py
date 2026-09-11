"""The training loop.

The loop is short; what surrounds it is where the care is. Specifically:

**Precision is chosen from what the hardware reports, not requested and hoped for.**
``auto`` resolves to bf16 only where the runtime says it is supported, fp16 with a
gradient scaler where it is not, and fp32 on CPU and MPS -- and the resolved choice
is reported, because "mixed precision enabled" that quietly meant fp32 is the sort
of claim this project exists to avoid. The decision lives in :func:`precision_for`,
a pure function of stated facts, so it can be tested against hardware nobody here
owns: on NVIDIA compute capability 8.0 is the bf16 threshold, but on ROCm the same
two numbers carry a gfx architecture where no such threshold exists, and applying
the NVIDIA rule there would tell an RX 6900 XT owner they have bf16 when they do
not.

**Divergence is caught before it is written down.** A NaN loss means the gradients
of that step are useless, so the step is not applied: the weights in memory, and
therefore any checkpoint taken afterwards, stay at the last good value. The run
stops with the step named, rather than filling a metrics file with NaN for six
hours and leaving a checkpoint full of them.

**Exact resume needs no bookkeeping.** Batch order is a pure function of
``(seed, step)`` and the learning rate is a pure function of ``step``, so resuming
restores weights, optimizer moments, the scaler and the RNG streams -- and then
simply continues from the recorded step. There is no iterator position to rewind
and no scheduler counter to advance the right number of times.

**Gradient accumulation divides the loss, not the gradient.** Each micro-batch's
loss is divided by ``grad_accum`` before ``backward()``, so the accumulated
gradient is the mean over the effective batch rather than the sum. Getting this
wrong scales the effective learning rate by ``grad_accum`` and looks like an
unstable model rather than an arithmetic mistake.
"""

from __future__ import annotations

import contextlib
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from trainai.console import fmt_bytes, fmt_count, fmt_int
from trainai.data.binarize import TOKENIZER_NAME, DatasetManifest
from trainai.data.loader import Batch, TokenBatcher, open_split, windows_available
from trainai.errors import (
    CheckpointIncompatibleError,
    DatasetError,
    DatasetFormatError,
    TrainingDivergedError,
    TrainingError,
    UsageError,
    check_choice,
)
from trainai.model.config import ModelConfig
from trainai.model.gpt import GPT
from trainai.train.budget import DataBudget
from trainai.train.checkpoint import (
    Checkpoint,
    capture_rng,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
)
from trainai.train.config import PRECISION_CHOICES, Precision, TrainConfig
from trainai.train.metrics import (
    MetricsWriter,
    ThroughputMeter,
    TrainingDisplay,
    perplexity,
    summarise_run,
)
from trainai.train.schedule import LearningRateSchedule

__all__ = [
    "DEVICE_CHOICES",
    "TrainResult",
    "Trainer",
    "precision_for",
    "resolve_device",
    "resolve_precision",
]

#: The device names this project supports, as ``--device`` accepts them. Not
#: ``torch``'s own list: :func:`torch.device` accepts twenty-odd backends, including
#: ``opengl``, ``fpga`` and ``lazy``, and TrainAI has a training path for five of them.
#: An index may follow any of the real ones -- ``cuda:1`` picks the second GPU.
DEVICE_CHOICES: tuple[str, ...] = ("auto", "cuda", "cpu", "mps", "xpu")

#: Settings whose value changes what the remaining steps of a resumed run do,
#: rather than only how many are left. ``steps`` belongs here because the
#: learning-rate schedule is a function of the total: resuming a 24-step run as a
#: 48-step run does not continue the original schedule, it starts following a new
#: one whose first half already happened at different rates. That is a legitimate
#: thing to want and a terrible thing to have happen without being told.
_BEHAVIOUR_CHANGING_SETTINGS = (
    "steps",
    "lr",
    "min_lr_ratio",
    # The resolved value, not the raw one: an unset warmup is derived from the step
    # count, so resuming with a different total silently moves it. Comparing the raw
    # field would show None against None and report nothing.
    "resolved_warmup_steps",
    "schedule",
    "seed",
    "batch_size",
    "grad_accum",
    "seq_len",
    "weight_decay",
    "beta1",
    "beta2",
    "grad_clip",
)


def synchronize(device: torch.device) -> None:
    """Wait for ``device``, so a timer measures work rather than queue depth.

    CUDA is not the only asynchronous backend: MPS and XPU also queue kernels and
    return immediately, so a ``perf_counter`` pair around a step on an M-series Mac
    times the dispatch and not the arithmetic. Every reported tokens/second and every
    ETA is derived from that pair, so leaving the other two backends unsynchronised
    does not merely lose precision -- it reports a number that was never measured.

    Failures are suppressed on purpose. A synchronise that raises is a driver problem
    that the next real operation will surface with a better message; turning it into a
    crash here would end a training run over a timing call.
    """
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


def resolve_device(requested: str = "auto") -> torch.device:
    """Pick a device, preferring CUDA. Never silently falls back from an explicit ask.

    Raises:
        UsageError: If ``requested`` does not name a device this project supports.
            :func:`torch.device` would answer for itself, but it answers with its own
            twenty-odd backends -- ``mkldnn``, ``opengl``, ``ideep``, ``fpga`` -- as a
            bare ``RuntimeError`` that reaches the user as a traceback. ``--device gpu``
            deserves the five names that work.
        TrainingError: If a specific device was requested and is unavailable.
            Falling back to CPU from an explicit ``--device cuda`` would turn a
            twenty-minute run into a twelve-hour one without saying so.
    """
    if requested in ("auto", ""):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    name, separator, index = requested.partition(":")
    check_choice(name, DEVICE_CHOICES, "--device")
    if separator and (not index.isdigit() or name == "auto"):
        raise UsageError(
            f"--device {requested!r} does not name a device.",
            hint=(
                "auto takes no device number; write cuda:0 to pick a particular GPU."
                if name == "auto"
                else f"Write {name} for the default one, or {name}:0 for the first."
            ),
            details={"given": requested, "index": index},
        )

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TrainingError(
            "--device cuda was requested but no CUDA device is available.",
            hint=(
                "Run `trainai doctor` to see what this machine has. If you have an "
                "NVIDIA GPU, the installed PyTorch is probably the CPU-only build; "
                "reinstall it from https://pytorch.org/get-started/locally/. Use "
                "--device cpu to train on the CPU, which is far slower."
            ),
            details={"requested": requested},
        )
    if device.type == "mps" and not (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    ):
        raise TrainingError(
            "--device mps was requested but Metal is not available.",
            hint="Use --device cpu, or run `trainai doctor` to see what is detected.",
            details={"requested": requested},
        )
    return device


def precision_for(
    requested: Precision,
    *,
    backend: str,
    supports_bf16: bool,
    capability: tuple[int, int] | None = None,
) -> tuple[torch.dtype, bool, str]:
    """Decide the autocast dtype from stated facts, without touching any device.

    Split out from :func:`resolve_precision` so the decision table can be tested
    against hardware nobody here owns. Every branch below corresponds to a real
    machine -- a GTX 1080 with no bf16, an RX 6900 XT whose architecture number
    looks like a CUDA capability and is not one, an Apple M2 -- and the only way to
    cover them without buying them is to make the decision a pure function of the
    facts and feed it synthetic ones.

    Returns ``(dtype, needs_gradient_scaler, explanation)``. The explanation is
    shown to the user and written to the metrics log, because the difference
    between bf16 and fp16 shows up as stability rather than as an error message.

    Raises:
        UsageError: If ``requested`` is not one of :data:`PRECISION_CHOICES`. Checked
            here rather than at the CLI because every caller funnels through this
            function, and because the cost of not checking was invisible: no branch
            below matches an unknown value, so it fell through to ``auto`` and the run
            reported the precision it picked as though that had been the ask.
        TrainingError: If bf16 was requested explicitly and is unavailable.
    """
    check_choice(requested, PRECISION_CHOICES, "--precision")

    if backend in ("cpu", "mps"):
        # MPS autocast exists but this project has never verified it, so fp32 it is:
        # a wrong dtype on an untested backend produces silently bad training, and
        # "slower than it could be" is the better failure.
        where = "CPU" if backend == "cpu" else "Apple MPS"
        if requested in ("bf16", "fp16"):
            return (
                torch.float32,
                False,
                f"fp32 ({requested} needs a CUDA or ROCm device; this is {where})",
            )
        return torch.float32, False, f"fp32 (on {where})"

    if requested == "fp32":
        return torch.float32, False, "fp32 (requested)"

    if requested == "bf16":
        if not supports_bf16:
            raise TrainingError(
                "--precision bf16 was requested but this device does not support it"
                + (
                    f" (compute capability {capability[0]}.{capability[1]})"
                    if capability and backend == "cuda"
                    else ""
                )
                + ".",
                hint=(
                    "Use --precision fp16, which works on this hardware with a "
                    "gradient scaler, or --precision fp32 to trade speed for "
                    "simplicity. On NVIDIA, bf16 needs compute capability 8.0 or "
                    "newer."
                ),
                details={
                    "backend": backend,
                    "capability": list(capability) if capability else None,
                },
            )
        return torch.bfloat16, False, "bf16 (requested)"

    if requested == "fp16":
        return torch.float16, True, "fp16 with gradient scaling (requested)"

    # auto
    if supports_bf16:
        return torch.bfloat16, False, f"bf16 (supported by this {backend} device)"
    reason = "this device does not support bf16"
    if backend == "cuda" and capability is not None:
        reason = f"compute capability {capability[0]}.{capability[1]} predates bf16"
    elif backend == "rocm":
        reason = "the ROCm runtime reports no bf16 support"
    elif backend == "xpu":
        reason = "the XPU runtime reports no bf16 support"
    return torch.float16, True, f"fp16 with gradient scaling ({reason})"


def resolve_precision(requested: Precision, device: torch.device) -> tuple[torch.dtype, bool, str]:
    """Read this machine's facts and hand them to :func:`precision_for`.

    Kept as a thin wrapper so that everything device-specific happens in one place
    and the decision itself stays testable.
    """
    if device.type == "cpu":
        return precision_for(requested, backend="cpu", supports_bf16=False)
    if device.type == "mps":
        return precision_for(requested, backend="mps", supports_bf16=False)

    if device.type == "xpu":
        xpu = getattr(torch, "xpu", None)
        checker = getattr(xpu, "is_bf16_supported", None) if xpu is not None else None
        supports = False
        if checker is not None:
            with contextlib.suppress(Exception):
                supports = bool(checker())
        return precision_for(requested, backend="xpu", supports_bf16=supports)

    # CUDA or ROCm. torch.version.hip is the only reliable way to tell them apart,
    # and it matters: get_device_capability returns a gfx architecture on ROCm,
    # where the 8.0 rule that is correct for NVIDIA means nothing.
    backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
    supports_bf16 = False
    with contextlib.suppress(Exception):
        supports_bf16 = bool(torch.cuda.is_bf16_supported())
    capability: tuple[int, int] | None = None
    with contextlib.suppress(Exception):
        major, minor = torch.cuda.get_device_capability(device)
        capability = (int(major), int(minor))
    return precision_for(
        requested, backend=backend, supports_bf16=supports_bf16, capability=capability
    )


def check_configs_agree(
    dataset: DatasetManifest, model_config: ModelConfig, train_config: TrainConfig
) -> None:
    """Refuse a pair of configurations that cannot train together.

    Neither dataclass can catch these alone: each setting is valid on its own and
    only the combination is impossible. Both are cheap, need no device and read no
    shards, which is what lets ``--dry-run`` apply them -- it used to skip them by
    returning before the :class:`Trainer` that held them, so a run that could not
    start printed a full plan and exited 0. A dry run that green-lights a run the
    next command refuses is worse than no dry run, because it is trusted.

    Raises:
        TrainingError: The training sequence is longer than the model's context, or
            the model has no embedding row for some of the dataset's tokens.
    """
    if train_config.seq_len > model_config.seq_len:
        raise TrainingError(
            f"--seq-len {train_config.seq_len} exceeds the model's context length "
            f"of {model_config.seq_len}.",
            hint=(
                f"Train at {model_config.seq_len} tokens or fewer, or build the "
                "model with a longer context."
            ),
            details={
                "train_seq_len": train_config.seq_len,
                "model_seq_len": model_config.seq_len,
            },
        )
    if model_config.vocab_size < dataset.vocab_size:
        # This said "Build the model with --vocab-size N", and there is no such flag.
        # `trainai train --vocab-size 300` exits 2 with "No such option: --vocab-size
        # Did you mean --batch-size?". Nor should there be one: `_build_model_config`
        # passes `vocab_size=dataset.vocab_size` itself and its docstring says
        # "vocab_size is never a flag" -- so the hint's own second sentence, "the
        # dataset's tokenizer decides this, not you", contradicted its first.
        #
        # No CLI path reaches here: `_load_plan` compares the two with `!=` before the
        # trainer is built and sends the user to `trainai plan`. The audience is a
        # library caller, so the lever to name is the keyword argument.
        raise TrainingError(
            f"The model's vocabulary ({model_config.vocab_size}) is smaller than "
            f"the dataset's ({dataset.vocab_size}), so some tokens have no "
            "embedding.",
            hint=(
                f"Build the model with vocab_size={dataset.vocab_size}, the number in "
                "the dataset manifest -- `trainai.model.config.preset` takes it as a "
                "keyword argument. It is an output of preparing the data, not a setting "
                "to choose: the tokenizer decides it, and `trainai data prepare` takes "
                "only a target it may fall short of."
            ),
            details={
                "model_vocab": model_config.vocab_size,
                "dataset_vocab": dataset.vocab_size,
            },
        )


def check_finetune_compatible(dataset: DatasetManifest, checkpoint: Checkpoint) -> None:
    """Refuse a base model whose vocabulary is not this dataset's vocabulary.

    A different ``content_hash`` is the point of fine-tuning and is deliberately not
    checked here -- :func:`load_checkpoint` is called without ``expect_dataset`` for
    exactly that reason. A different tokenizer is the opposite: token ids are row
    indices into the embedding matrix, so the same id read through a different
    tokenizer selects a vector trained for some other piece of text. Nothing raises,
    the loss curve looks entirely ordinary, and the model learns nothing usable.

    Module level rather than a :class:`Trainer` method so that a caller can apply it
    *before* building anything -- ``trainai finetune`` checks it before creating the
    run directory, because a refused fine-tune should not leave one behind.

    Raises:
        CheckpointIncompatibleError: The tokenizer or the vocabulary size differs.
    """
    for key, label, why in (
        (
            "tokenizer_fingerprint",
            "tokenizer",
            "Every token id in this dataset would index a row of the embedding "
            "matrix that was trained for a different piece of text.",
        ),
        (
            "vocab_size",
            "vocabulary size",
            "The embedding matrix has one row per token, so the shapes cannot "
            "even be made to line up.",
        ),
    ):
        have = checkpoint.dataset.get(key)
        want = {
            "tokenizer_fingerprint": dataset.tokenizer_fingerprint,
            "vocab_size": dataset.vocab_size,
        }.get(key)
        if have is None or want is None or have == want:
            continue
        raise CheckpointIncompatibleError(
            f"The base model was trained with a different {label}.",
            hint=(
                f"{why} Fine-tune against a dataset prepared with the same "
                "tokenizer as the base model -- `trainai data prepare "
                "--tokenizer <the base run's tokenizer.json>` reuses it instead "
                "of training a new one."
            ),
            details={
                "path": str(checkpoint.path) if checkpoint.path else None,
                "field": key,
                "base_model": have,
                "dataset": want,
            },
        )


@dataclass(frozen=True)
class ValidationSkipped:
    """Why a run measured no held-out loss.

    Carried rather than discarded. ``_open_validation`` used to return a bare ``None``,
    so by the time the result panel wanted to explain the absence it had nothing left to
    explain it with and guessed: it printed "no validation split, or --eval-every 0"
    for every cause, including the common one where neither is true. Measured on a
    dataset with a 45-token validation split and ``--eval-every 2``, both named causes
    were false and the real one -- the split is too small for one window at this
    ``--seq-len`` -- was never mentioned, though the loader had computed it exactly.

    ``hint`` is the loader's own hint where the reason came from the loader, not a
    second wording of it. ``trainai eval`` reaches the same condition and answers
    "Use --seq-len 22 or lower"; a run should not answer the same question differently.
    """

    reason: str
    hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"reason": self.reason, "hint": self.hint}


@dataclass(frozen=True)
class StepOutcome:
    """What one optimizer step produced.

    A tuple of ``(loss, grad_norm)`` was enough until the loss mask arrived. With a
    mask, a step can legitimately score *nothing* -- every window it drew lay inside a
    prompt -- and the mean loss over zero positions is not a number. Returning 0.0
    would put a fake minimum in the loss curve and returning NaN would be read as
    divergence, so the count comes back with the loss and the caller decides.
    """

    loss: float
    grad_norm: float | None
    scored_tokens: int
    applied: bool


@dataclass
class TrainResult:
    """What a finished run produced. Every field is measured, not projected."""

    steps_completed: int
    final_train_loss: float
    best_val_loss: float | None
    best_val_step: int | None
    final_val_loss: float | None
    tokens_seen: int
    elapsed_seconds: float
    tokens_per_second: float
    peak_vram_bytes: int
    checkpoint_path: Path | None
    run_dir: Path
    precision: str
    device: str
    diverged: bool = False
    validation_skipped: ValidationSkipped | None = None
    summary: dict[str, Any] = None  # type: ignore[assignment]
    #: Whether the loss was computed on the tokens a masked dataset marks as targets.
    loss_mask: bool = False
    #: Positions the loss actually covered. Equal to ``tokens_seen`` without a mask,
    #: and below it with one -- which is the number that explains a loss curve sitting
    #: where an unmasked run's would not.
    scored_tokens: int = 0
    #: Steps whose every window was context, so nothing was applied. Always 0 without
    #: a mask, and worth seeing with one: a run of them means the corpus has documents
    #: longer than the context with no reply inside a window.
    unscored_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps_completed": self.steps_completed,
            "final_train_loss": self.final_train_loss,
            "best_val_loss": self.best_val_loss,
            "best_val_step": self.best_val_step,
            "final_val_loss": self.final_val_loss,
            "best_val_perplexity": (
                perplexity(self.best_val_loss) if self.best_val_loss is not None else None
            ),
            "tokens_seen": self.tokens_seen,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "tokens_per_second": round(self.tokens_per_second, 1),
            "peak_vram_bytes": self.peak_vram_bytes,
            "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else None,
            "run_dir": str(self.run_dir),
            "precision": self.precision,
            "device": self.device,
            "diverged": self.diverged,
            "loss_mask": self.loss_mask,
            "scored_tokens": self.scored_tokens,
            "unscored_steps": self.unscored_steps,
            "validation_skipped": (
                self.validation_skipped.to_dict() if self.validation_skipped else None
            ),
            "summary": self.summary or {},
        }


class Trainer:
    """Trains one model on one dataset.

    Construction is deliberately cheap and side-effect free apart from allocating
    the model: nothing is written to disk and no data is read until :meth:`run` is
    called, so the planner (M3) can build a Trainer purely to measure a few steps
    and then throw it away.
    """

    def __init__(
        self,
        *,
        dataset: DatasetManifest,
        model_config: ModelConfig,
        train_config: TrainConfig,
        run_dir: str | Path,
        device: torch.device | None = None,
        quiet: bool = False,
        write_metrics: bool = True,
        finetune: bool = False,
    ) -> None:
        self.dataset = dataset
        self.train_config = train_config
        self.run_dir = Path(run_dir)
        self.quiet = quiet
        self._write_metrics = write_metrics

        # Also applied by the CLI before ``--dry-run`` returns, so a plan that cannot
        # train is refused at the same point whether or not the run is real. Cheap and
        # idempotent, so running it twice on the real path costs nothing.
        check_configs_agree(dataset, model_config, train_config)

        self.device = device if device is not None else resolve_device(train_config.device)
        self.dtype, self.needs_scaler, self.precision_note = resolve_precision(
            train_config.precision, self.device
        )

        # Seed before constructing the model, so initialisation is reproducible.
        torch.manual_seed(train_config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(train_config.seed)

        self.model_config = model_config
        self.model = GPT(model_config).to(self.device)
        self.schedule = LearningRateSchedule(train_config)
        self.optimizer = torch.optim.AdamW(
            self.model.parameter_groups(train_config.weight_decay),
            lr=train_config.lr,
            betas=(train_config.beta1, train_config.beta2),
            eps=train_config.eps,
            fused=self.device.type == "cuda",
        )
        self.scaler = torch.amp.GradScaler(self.device.type) if self.needs_scaler else None
        self.step = 0
        self.best_val_loss: float | None = None
        self.best_val_step: int | None = None
        self._resumed_from: Path | None = None
        self._finetuned_from: Path | None = None
        self._parent: dict[str, Any] | None = None
        self.config_changes: dict[str, tuple[Any, Any]] = {}
        self.budget = DataBudget(
            train_tokens=dataset.tokens("train"),
            val_tokens=dataset.tokens("val"),
            parameters=self.model.parameter_count(),
            non_embedding_parameters=model_config.non_embedding_parameter_count,
            tokens_per_step=train_config.tokens_per_step,
            steps=train_config.steps,
            finetune=finetune,
        )

    # -- resume ------------------------------------------------------------- #

    def resume(self, path: str | Path) -> Checkpoint:
        """Restore from a checkpoint so that training continues exactly.

        Sets :attr:`config_changes` to any setting whose value differs from the one
        the checkpoint was written with. Those are not refused -- resuming a run
        with more steps or a lower learning rate is a reasonable thing to do -- but
        they are reported, because the continuation then follows a different
        schedule than the one already half-completed, and nothing else would say so.

        Raises:
            CheckpointIncompatibleError: If the checkpoint was trained on different
                data or for a different model shape.
        """
        checkpoint = load_checkpoint(
            path,
            map_location=self.device,
            expect_dataset=self._dataset_identity(),
        )
        checkpoint.apply_to(self.model)
        self.optimizer.load_state_dict(checkpoint.optimizer_state)
        if self.scaler is not None and checkpoint.scaler_state is not None:
            self.scaler.load_state_dict(checkpoint.scaler_state)
        if checkpoint.rng is not None:
            restore_rng(checkpoint.rng)
        self.step = checkpoint.step
        self.best_val_loss = checkpoint.metrics.get("best_val_loss")
        self.best_val_step = checkpoint.metrics.get("best_val_step")
        self._resumed_from = checkpoint.path
        self.config_changes = self._compare_configs(checkpoint.train_config)
        return checkpoint

    # -- fine-tune ---------------------------------------------------------- #

    def initialise_from(self, source: str | Path | Checkpoint) -> Checkpoint:
        """Take the weights from a checkpoint and start a *new* run from them.

        This is the fine-tuning entry point, and it is deliberately not
        :meth:`resume`. Resuming continues one run: same data, same schedule, the
        optimizer's moments and the RNG streams restored so that step 601 is the step
        601 that would have happened anyway. Fine-tuning is the opposite claim -- new
        data, new schedule -- so only the weights come across:

        * **Optimizer state is not loaded.** AdamW's moments are an average over the
          gradients of the *previous* corpus. Carrying them into a different one
          applies a stale preconditioner to the first steps, which is exactly when
          the run is most fragile. Fresh moments plus this run's own warmup is the
          honest start.
        * **The RNG is not restored.** The data order belongs to this dataset.
        * **:attr:`step` stays at 0**, so the learning-rate schedule warms up from
          the beginning rather than resuming somewhere down a cosine decay that was
          computed for a different number of total steps.

        What it *does* check is the tokenizer. :func:`load_checkpoint` is called
        without ``expect_dataset``, because a different ``content_hash`` is the whole
        point -- but the two halves of that check mean different things and only one
        of them is negotiable. Token ids are row indices into the embedding matrix,
        so the same id read through a different tokenizer selects a vector trained
        for some other piece of text. The loss curve looks entirely ordinary and the
        model learns nothing usable, which is the failure this project refuses to
        let happen quietly.

        The model shape is checked by :meth:`Checkpoint.apply_to`; callers are
        expected to have built ``model_config`` *from* the checkpoint rather than
        from flags, because there is no useful sense in which a fine-tune chooses its
        own layer count.

        Args:
            source: A checkpoint file, a directory to take the latest from, or an
                already-loaded :class:`Checkpoint`. The last is what ``trainai
                finetune`` passes: it has to read the checkpoint before this call to
                get the model shape to build the model *from*, and reading a
                multi-gigabyte file twice to learn the same thing is not free.

        Raises:
            CheckpointIncompatibleError: If the checkpoint's tokenizer or vocabulary
                differs from this dataset's, or its model shape differs from this
                model's.
        """
        checkpoint = (
            source
            if isinstance(source, Checkpoint)
            else load_checkpoint(source, map_location=self.device, expect_dataset=None)
        )
        check_finetune_compatible(self.dataset, checkpoint)
        checkpoint.apply_to(self.model)
        self._finetuned_from = checkpoint.path
        self._parent = self._describe_parent(checkpoint)
        return checkpoint

    def _describe_parent(self, checkpoint: Checkpoint) -> dict[str, Any]:
        """Lineage for the run record: which model this one started from.

        A fine-tuned checkpoint is not reproducible from its own run directory alone
        -- it is a function of a base checkpoint that lives somewhere else -- so the
        identity of that parent is recorded rather than left to the file path, which
        the user is free to move.
        """
        return {
            "path": str(checkpoint.path) if checkpoint.path else None,
            "step": checkpoint.step,
            "created_with": checkpoint.created_with,
            "model": checkpoint.model_config.to_dict(),
            "dataset": {
                key: checkpoint.dataset.get(key)
                for key in ("content_hash", "tokenizer_fingerprint", "vocab_size")
            },
            "val_loss": checkpoint.metrics.get("best_val_loss"),
        }

    def _compare_configs(self, previous: TrainConfig) -> dict[str, tuple[Any, Any]]:
        """Settings that differ between the checkpoint's config and this one."""
        was, now = previous.to_dict(), self.train_config.to_dict()
        return {
            key: (was.get(key), now.get(key))
            for key in _BEHAVIOUR_CHANGING_SETTINGS
            if was.get(key) != now.get(key)
        }

    def _dataset_identity(self) -> dict[str, Any]:
        """What the checkpoint records about its training data."""
        return {
            "tokenizer_fingerprint": self.dataset.tokenizer_fingerprint,
            "content_hash": self.dataset.content_hash,
            "vocab_size": self.dataset.vocab_size,
            "train_tokens": self.dataset.tokens("train"),
            "val_tokens": self.dataset.tokens("val"),
            "root": str(self.dataset.root) if self.dataset.root else None,
            # Copied in rather than looked up later, for the reason the tokenizer is
            # copied into the run directory: a run has to stay usable after its dataset
            # is deleted, and datasets are the large thing people delete once training
            # is done. Empty for a plain corpus. Not part of the resume check -- only
            # the fingerprint and the content hash are -- so a checkpoint from before
            # this release resumes unchanged.
            "chat": dict(self.dataset.chat),
        }

    def _copy_tokenizer(self) -> str | None:
        """Copy the dataset's tokenizer into the run directory.

        A run without its tokenizer is a directory of numbers that cannot be turned
        back into text. The checkpoint records the tokenizer's *fingerprint*, which is
        enough to detect a mismatch and not enough to decode anything -- so
        ``trainai chat`` and ``trainai export`` would depend on the dataset directory
        still existing. Datasets are the large thing people delete once training is
        done; a tokenizer.json is a few hundred kilobytes. Copying it is what makes a
        run self-contained.

        Returns a warning to show, or ``None``. A failure here must not lose a
        training run that is otherwise fine, so it is reported rather than raised; the
        fingerprint check downstream is what catches the consequence.
        """
        if self.dataset.root is None:
            return None
        source = Path(self.dataset.root) / TOKENIZER_NAME
        target = self.run_dir / TOKENIZER_NAME
        if target.is_file():
            return None
        if not source.is_file():
            return (
                f"No {TOKENIZER_NAME} beside the dataset, so this run will not carry one. "
                "Decoding its output will need the tokenizer that produced the shards."
            )
        try:
            shutil.copyfile(source, target)
        except OSError as error:  # pragma: no cover - depends on the filesystem
            return (
                f"Could not copy the tokenizer into the run directory: {error}. "
                f"Decoding this run's output will need {source}."
            )
        return None

    # -- the loop ----------------------------------------------------------- #

    def run(self) -> TrainResult:
        """Train to ``train_config.steps`` and return what happened."""
        config = self.train_config
        self.run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = self.run_dir / "checkpoints"
        tokenizer_warning = self._copy_tokenizer()

        train_batcher, _train_stream = open_split(
            self.dataset,
            "train",
            seq_len=config.seq_len,
            batch_size=config.batch_size,
            seed=config.seed,
            loss_mask=self._use_loss_mask(config),
        )
        val_batcher, _val_stream, val_skipped = self._open_validation(config)

        metrics = MetricsWriter(self.run_dir / "metrics.jsonl", enabled=self._write_metrics)
        meter = ThroughputMeter()
        last_loss = float("nan")
        unscored_steps = 0
        scored_tokens = 0
        diverged = False
        final_val_loss: float | None = None
        latest_checkpoint: Path | None = None

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        metrics.write(
            "start",
            self.step,
            model=self.model.summary(),
            train=config.to_dict(),
            dataset=self._dataset_identity(),
            budget=self.budget.to_dict(),
            device=str(self.device),
            precision=self.precision_note,
            resumed_from=str(self._resumed_from) if self._resumed_from else None,
            finetuned_from=str(self._finetuned_from) if self._finetuned_from else None,
            parent=self._parent,
            config_changes={k: list(v) for k, v in self.config_changes.items()},
            validation_skipped=val_skipped.to_dict() if val_skipped else None,
        )

        try:
            with TrainingDisplay(config.steps, start_step=self.step, quiet=self.quiet) as display:
                if self._resumed_from is not None:
                    display.log(
                        f"[green]Resumed[/] from step {fmt_int(self.step)} "
                        f"({self._resumed_from.name})"
                    )
                    for key, (was, now) in self.config_changes.items():
                        display.log(
                            f"  [yellow]{key}[/] changed since the checkpoint: "
                            f"{was} -> {now}. The remaining steps follow the new value, "
                            "so this is not a continuation of the original schedule."
                        )
                if self._finetuned_from is not None and self._parent is not None:
                    display.log(
                        f"[green]Fine-tuning[/] the weights from step "
                        f"{fmt_int(self._parent['step'])} of {self._finetuned_from.name}. "
                        "Optimizer state and RNG were not carried over, so this run warms "
                        "up and schedules on its own from step 0."
                    )
                display.log(
                    f"[dim]{self.model_config.describe()}  {fmt_count(self.model.parameter_count())}"
                    f" params  {self.precision_note} on {self.device}[/]"
                )
                display.log(f"[dim]{self.budget.describe()}[/]")
                for note in self.budget.warnings():
                    display.log(f"[yellow]note:[/] {note}")
                if tokenizer_warning is not None:
                    display.log(f"[yellow]note:[/] {tokenizer_warning}")
                if val_skipped is not None:
                    # Said at the start, not only in the summary. A run that will never
                    # print a val loss should say so before the user waits for one.
                    display.log(f"[yellow]note:[/] No held-out loss: {val_skipped.reason}")
                    if val_skipped.hint:
                        display.log(f"      [dim]{val_skipped.hint}[/]")

                while self.step < config.steps:
                    started = time.perf_counter()
                    outcome = self._optimizer_step(train_batcher, config)
                    loss_value, grad_norm = outcome.loss, outcome.grad_norm
                    synchronize(self.device)
                    elapsed = time.perf_counter() - started

                    self.step += 1
                    meter.record(config.tokens_per_step, elapsed)
                    unscored = outcome.scored_tokens == 0
                    if unscored:
                        # Every window this step drew fell inside a prompt, so nothing
                        # was applied and there is no loss. metrics.jsonl gets an
                        # "unscored" record with a null loss and no "train" record at
                        # all: a zero there would be a fake minimum in the curve, and
                        # the mean over no positions is not a number. Validation and
                        # checkpointing still happen on schedule -- the step counted,
                        # it just taught nothing.
                        unscored_steps += 1
                        metrics.write(
                            "unscored",
                            self.step,
                            loss=None,
                            lr=self.schedule(self.step - 1),
                            reason="no token of this step's windows is a training target",
                        )
                        if unscored_steps == 1:
                            display.log(
                                f"[yellow]note:[/] Step {fmt_int(self.step)} scored no "
                                "tokens: every sequence it drew is context under the loss "
                                "mask, so the step was not applied."
                            )
                    else:
                        last_loss = loss_value
                        scored_tokens += outcome.scored_tokens

                    if not unscored and not math.isfinite(loss_value):
                        diverged = True
                        gradients_overflowed = grad_norm is not None and not math.isfinite(
                            grad_norm
                        )
                        what = (
                            "The gradients overflowed"
                            if gradients_overflowed
                            else f"The loss became {loss_value}"
                        )
                        metrics.write(
                            "diverged",
                            self.step,
                            loss=loss_value,
                            grad_norm=grad_norm,
                            gradients_overflowed=gradients_overflowed,
                            lr=self.schedule(self.step - 1),
                        )
                        raise TrainingDivergedError(
                            f"{what} at step {self.step}. The step was not applied, so "
                            "the weights are still those of the last good step.",
                            hint=(
                                f"Almost always the learning rate. Try "
                                f"--lr {config.lr / 3:.2g}"
                                + (
                                    ", and set --grad-clip 1.0 (it is currently 0, so "
                                    "nothing bounded the update)."
                                    if config.grad_clip == 0
                                    else "."
                                )
                                + " Any checkpoint in the run directory is from before"
                                " this step and is unaffected."
                            ),
                            details={
                                "step": self.step,
                                "lr": self.schedule(self.step - 1),
                                "grad_clip": config.grad_clip,
                                "grad_norm": grad_norm,
                            },
                        )

                    lr = self.schedule(self.step - 1)
                    display.update(
                        self.step,
                        # The progress line carries the last loss there was on an
                        # unscored step. The record of what each step produced is
                        # metrics.jsonl, which says "unscored" with a null loss.
                        loss=last_loss,
                        lr=lr,
                        tokens_per_second=meter.tokens_per_second,
                        grad_norm=grad_norm,
                    )

                    if not unscored and (
                        self.step % config.log_every == 0 or self.step == config.steps
                    ):
                        metrics.write(
                            "train",
                            self.step,
                            loss=loss_value,
                            lr=lr,
                            grad_norm=grad_norm,
                            tokens=meter.total_tokens,
                            tokens_per_second=round(meter.tokens_per_second, 1),
                            elapsed=round(meter.elapsed, 3),
                            step_seconds=round(elapsed, 4),
                        )

                    due_for_eval = (
                        val_batcher is not None
                        and config.eval_every > 0
                        and (self.step % config.eval_every == 0 or self.step == config.steps)
                    )
                    if due_for_eval:
                        assert val_batcher is not None
                        final_val_loss = self.evaluate(val_batcher, config.eval_batches)
                        improved = self.best_val_loss is None or final_val_loss < self.best_val_loss
                        if improved:
                            self.best_val_loss = final_val_loss
                            self.best_val_step = self.step
                        display.log_eval(self.step, val_loss=final_val_loss, train_loss=last_loss)
                        metrics.write(
                            "eval",
                            self.step,
                            val_loss=final_val_loss,
                            val_perplexity=perplexity(final_val_loss),
                            train_loss=last_loss,
                            improved=improved,
                            batches=config.eval_batches,
                        )

                    due_for_checkpoint = config.checkpoint_every > 0 and (
                        self.step % config.checkpoint_every == 0
                    )
                    if due_for_checkpoint and self.step < config.steps:
                        latest_checkpoint = self._save(
                            checkpoint_dir, metrics, is_best=self.best_val_step == self.step
                        )

                latest_checkpoint = self._save(
                    checkpoint_dir, metrics, is_best=self.best_val_step == self.step
                )
        finally:
            train_batcher.close()
            if val_batcher is not None:
                val_batcher.close()

            peak = self._peak_vram()
            metrics.write(
                "end",
                self.step,
                final_loss=last_loss,
                best_val_loss=self.best_val_loss,
                tokens=meter.total_tokens,
                elapsed=round(meter.elapsed, 3),
                tokens_per_second=round(meter.average_tokens_per_second, 1),
                peak_vram_bytes=peak,
                diverged=diverged,
                validation_skipped=val_skipped.to_dict() if val_skipped else None,
                loss_mask=train_batcher.has_loss_mask,
                scored_tokens=scored_tokens,
                unscored_steps=unscored_steps,
            )
            records = metrics.read_all()
            summary = summarise_run(records)
            if metrics.unreadable_lines:
                # Said rather than absorbed: a summary computed from fewer records than
                # the file has lines is a different summary, and this is the only place
                # anyone would find out.
                summary["unreadable_metric_lines"] = metrics.unreadable_lines
            metrics.close()

        return TrainResult(
            steps_completed=self.step,
            final_train_loss=last_loss,
            best_val_loss=self.best_val_loss,
            best_val_step=self.best_val_step,
            final_val_loss=final_val_loss,
            tokens_seen=meter.total_tokens,
            elapsed_seconds=meter.elapsed,
            tokens_per_second=meter.average_tokens_per_second,
            peak_vram_bytes=peak,
            checkpoint_path=latest_checkpoint,
            run_dir=self.run_dir,
            precision=self.precision_note,
            device=str(self.device),
            diverged=diverged,
            validation_skipped=val_skipped,
            summary=summary,
            loss_mask=train_batcher.has_loss_mask,
            scored_tokens=scored_tokens,
            unscored_steps=unscored_steps,
        )

    def _optimizer_step(self, batcher: TokenBatcher, config: TrainConfig) -> StepOutcome:
        """One optimizer step: ``grad_accum`` micro-batches, then apply.

        Returns the mean loss over the scored positions, the pre-clip gradient norm,
        and how many positions that mean covers. The gradient norm is worth reporting:
        a run whose loss looks fine while its gradient norm climbs is about to diverge,
        and that is visible one or two hundred steps before the loss shows it.

        A non-finite loss returns early without applying the step, so the weights
        in memory stay at the last good value.
        """
        self.model.train()
        lr = self.schedule(self.step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self.optimizer.zero_grad(set_to_none=True)
        # The whole step's data is read before the first backward pass. Without a mask
        # every micro-batch scores the same number of positions and its share of the
        # effective batch is 1/grad_accum; with one, the shares differ, and a
        # micro-batch's share is not knowable until every micro-batch has been read.
        # Averaging the means instead would weight a micro-batch that scored 3 tokens
        # the same as one that scored 300. The reads are memmap slices -- the same data
        # the loop read one batch at a time, held for the length of one step.
        batches = [
            # Each micro-batch of a step gets its own slice of the epoch order, so
            # the effective batch is grad_accum distinct batches rather than the
            # same one summed.
            batcher.batch(self.step * config.grad_accum + micro)
            for micro in range(config.grad_accum)
        ]
        scored = sum(batch.trained_tokens for batch in batches)
        if scored == 0:
            # Every window this step drew lies inside a prompt. There is nothing to
            # learn from, so nothing is applied and the caller is told, rather than
            # reporting a loss of 0 for a step that scored no token.
            return StepOutcome(loss=float("nan"), grad_norm=None, scored_tokens=0, applied=False)

        total = 0.0
        for batch in batches:
            inputs = torch.from_numpy(batch.inputs).to(self.device, non_blocking=True).long()
            targets = torch.from_numpy(batch.targets).to(self.device, non_blocking=True).long()
            mask = self._mask_tensor(batch)

            with self._autocast():
                _, loss, _ = self.model(inputs, targets, loss_mask=mask)
            assert loss is not None
            # Weight before backward so the accumulated gradient is the mean over the
            # effective batch, not the sum. Summing would multiply the effective
            # learning rate by grad_accum. The weight is the micro-batch's share of
            # the step's scored positions, which is 1/grad_accum exactly when every
            # micro-batch scored the same count -- the unmasked case, unchanged.
            count = batch.trained_tokens
            scaled = loss * (count / scored)
            if self.scaler is not None:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()
            total += float(loss.detach()) * count

        mean_loss = total / scored
        if not math.isfinite(mean_loss):
            # Return without stepping. The gradients are already polluted, but the
            # weights are not: this leaves the model, and therefore any checkpoint
            # written from here, holding the last good state. Applying the step
            # first would put NaN into every parameter and make the checkpoint on
            # disk worthless as well as the run.
            return StepOutcome(loss=mean_loss, grad_norm=None, scored_tokens=scored, applied=False)

        grad_norm: float | None = None
        if self.scaler is not None:
            # Unscale before clipping, or the clip threshold is measured against
            # gradients that are thousands of times too large and never triggers.
            self.scaler.unscale_(self.optimizer)
        if config.grad_clip > 0:
            grad_norm = float(nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip))
        if grad_norm is not None and not math.isfinite(grad_norm) and self.scaler is None:
            # Gradients overflowed while the loss was still finite. Skipping the
            # step keeps the weights usable; applying it would put inf into them
            # and the loss would only become NaN on the *next* forward pass, by
            # which point the last good state is gone.
            #
            # Only when there is no scaler. Under fp16 an inf gradient is how
            # GradScaler discovers its scale is too high: it happens routinely in
            # the first few steps, the scaler skips the step itself, and treating
            # it as divergence would abort every fp16 run at step one.
            return StepOutcome(
                loss=float("nan"), grad_norm=grad_norm, scored_tokens=scored, applied=False
            )
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        return StepOutcome(loss=mean_loss, grad_norm=grad_norm, scored_tokens=scored, applied=True)

    def _use_loss_mask(self, config: TrainConfig) -> bool:
        """Whether this run scores only the tokens the dataset marks as targets.

        ``None`` means "apply it if there is one", which is what makes a chat corpus
        and a plain-text corpus both do the right thing unasked. ``True`` on a dataset
        without a mask is a usage error rather than a run that trains on everything:
        someone who asked for it in writing should not get the opposite silently. The
        same three-state rule as ``data prepare --loss-mask``, deliberately.
        """
        if config.loss_mask is True and not self.dataset.has_loss_mask:
            raise DatasetError(
                "--loss-mask was given, but this dataset has no loss mask.",
                hint=(
                    "Re-run `trainai data prepare` with --jsonl-messages-field <field> "
                    "to prepare a chat corpus with a mask, or drop --loss-mask to train "
                    "on every token."
                ),
                details={"dataset": str(self.dataset.root)},
            )
        return config.loss_mask is not False

    def _mask_tensor(self, batch: Batch) -> torch.Tensor | None:
        """The batch's loss mask on the device, or ``None`` when it has none.

        ``uint8`` on the wire and left as ``uint8`` here: the model casts it to the
        logits' dtype, so widening it twice would move four times the bytes for a
        tensor that is only ever multiplied by.
        """
        if batch.loss_mask is None:
            return None
        return torch.from_numpy(batch.loss_mask).to(self.device, non_blocking=True)

    def _autocast(self) -> Any:
        if self.dtype == torch.float32:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.dtype)

    @torch.no_grad()
    def evaluate(self, batcher: TokenBatcher, max_batches: int) -> float:
        """Mean loss per token over up to ``max_batches`` sequential batches.

        Sequential from token 0 rather than sampled, so two validation losses from
        different steps are computed over exactly the same text and are therefore
        comparable. A sampled validation set makes the curve noisy for a reason
        that has nothing to do with the model.

        Weighted by target positions, not by batches. ``sequential_batches`` keeps the
        ragged final batch rather than dropping the tail of the split, so averaging
        per-batch losses gives that short batch a full batch's weight -- on the 1.1 MB
        Shakespeare validation split, six sequences out of thirty-eight carried half of
        the number. That shifted the reported loss by 0.012 nats and, because the best
        checkpoint is chosen on this value, could pick a different checkpoint than the
        data supports. It is also the value ``trainai eval`` recomputes, and the two
        agreeing is the point.

        On a masked dataset the positions counted are the *scored* ones, so validation
        measures the same thing training optimises. Scoring every token here while
        training only the replies would compare the model against a distribution it was
        never trained on, and the number would move for reasons the run cannot act on.
        """
        self.model.eval()
        weighted = 0.0
        positions = 0
        for batch in batcher.sequential_batches(max_batches=max_batches):
            inputs = torch.from_numpy(batch.inputs).to(self.device).long()
            targets = torch.from_numpy(batch.targets).to(self.device).long()
            mask = self._mask_tensor(batch)
            count = batch.trained_tokens
            if count == 0:
                # Nothing to score in this batch, and the model would return a loss of
                # 0 for it. Weighting that in would drag the mean towards zero.
                continue
            with self._autocast():
                _, loss, _ = self.model(inputs, targets, loss_mask=mask)
            assert loss is not None
            # The model's loss is a mean over the positions it scored, so the weight is
            # that count -- which differs on the last batch, and on every batch of a
            # masked split.
            weighted += float(loss) * count
            positions += count
        self.model.train()
        if positions == 0:
            raise TrainingError(
                "The validation split produced no batches."
                if not batcher.has_loss_mask
                else "No token of the validation split is a training target, so there is "
                "nothing to score it on.",
                hint="Re-prepare the dataset with a larger --val-fraction."
                if not batcher.has_loss_mask
                else (
                    "The held-out documents are all context: they have no assistant "
                    "replies. Re-prepare with a larger --val-fraction, or train with "
                    "--no-loss-mask to score every token."
                ),
                details={"max_batches": max_batches, "loss_mask": batcher.has_loss_mask},
            )
        return weighted / positions

    def _open_validation(
        self, config: TrainConfig
    ) -> tuple[TokenBatcher | None, Any | None, ValidationSkipped | None]:
        """Open the validation split, or return why there is none.

        The reason is returned, not logged and dropped. Every caller that wants to
        explain the absence -- the training log, ``metrics.jsonl`` and the result panel
        -- gets the same words, and none of them has to guess.
        """
        if config.eval_every <= 0:
            return (
                None,
                None,
                ValidationSkipped(
                    "--eval-every is 0, so no held-out loss was measured.",
                    "Pass --eval-every N to measure the validation split every N steps.",
                ),
            )
        if self.dataset.tokens("val") == 0:
            return (
                None,
                None,
                ValidationSkipped(
                    "This dataset has no validation split.",
                    "Re-prepare it with `trainai data prepare --val-fraction 0.1` to get "
                    "a held-out loss.",
                ),
            )

        # Sized to what the split holds, not to the training batch size. Validation is a
        # forward pass with no gradient, so a short final batch is just a short batch --
        # which is why `evaluate_split` shrinks it the same way, and why
        # `windows_available` exists. Passing the training batch size straight through
        # meant the *training* batch size decided whether a held-out loss existed at
        # all: measured on a 45-token validation split at --seq-len 16 (one window),
        # --batch-size 1 reported a val loss of 4.9795 and --batch-size 4 reported
        # nothing, on the same data at the same context.
        rows = max(
            1,
            min(config.batch_size, windows_available(self.dataset, "val", seq_len=config.seq_len)),
        )
        try:
            batcher, stream = open_split(
                self.dataset,
                "val",
                seq_len=config.seq_len,
                batch_size=rows,
                seed=config.seed,
                loss_mask=self._use_loss_mask(config),
            )
        except DatasetFormatError:
            # A damaged shard is not "this corpus is small"; it is a broken dataset, and
            # the train split is likely damaged too. Continuing would train to
            # completion on a dataset `trainai data inspect --verify` calls corrupt.
            raise
        except DatasetError as error:
            # A validation split too small for one window at this sequence length is a
            # real situation on a small corpus, so the run continues -- but with the
            # loader's own reason and hint, which name the context that would fit.
            # Broad `except Exception` here also meant a bug anywhere inside
            # `open_split` came out as "no validation split".
            return None, None, ValidationSkipped(str(error), error.hint)
        return batcher, stream, None

    def _save(self, directory: Path, metrics: MetricsWriter, *, is_best: bool) -> Path:
        path = save_checkpoint(
            directory,
            step=self.step,
            model=self.model,
            optimizer=self.optimizer,
            train_config=self.train_config,
            scaler=self.scaler,
            metrics={
                "best_val_loss": self.best_val_loss,
                "best_val_step": self.best_val_step,
            },
            dataset=self._dataset_identity(),
            rng=capture_rng(),
            is_best=is_best,
            keep=self.train_config.keep_checkpoints,
        )
        metrics.write("checkpoint", self.step, path=str(path), is_best=is_best)
        return path

    def _peak_vram(self) -> int:
        if self.device.type != "cuda":
            return 0
        return int(torch.cuda.max_memory_allocated(self.device))

    # -- reporting ---------------------------------------------------------- #
    #
    # A ``describe`` used to live here: model shape, training config and precision
    # joined with pipes. Nothing called it. Every panel that reports a run builds its
    # own rows from ``model_config.describe`` and ``train_config.describe`` -- see
    # ``cli/train.py``'s ``_run_rows`` -- because a table of labelled rows is what
    # those panels print, and a single pipe-joined line is not a row. It went the same
    # way as ``describe_presets``, ``describe_report`` and ``describe_plan`` before it,
    # and for the same reason: a formatting helper no caller wants is a format nobody
    # has agreed to, and it drifts from the panels that do the work without a test
    # able to notice.

    def memory_note(self) -> str:
        """Measured peak allocation, or a statement that there is nothing to measure."""
        if self.device.type != "cuda":
            return f"no VRAM accounting on {self.device.type}"
        return f"peak {fmt_bytes(self._peak_vram())} allocated"
