"""Training configuration.

Every knob a run needs, validated on construction, recorded in the checkpoint. The
validation here is not defensive box-ticking: each check corresponds to a way a run
can waste hours and then produce nothing, and the hint names the flag to change.

One decision worth stating up front. ``batch_size`` is the *micro*-batch -- what
goes through the GPU at once -- and ``grad_accum`` is how many of those are summed
before an optimizer step. The product is the effective batch, and it is the
effective batch that the learning rate has to match. Conflating the two is the
most common way a training script gets a wrong learning rate: halving the
micro-batch to fit in VRAM silently halves the effective batch too, unless
accumulation makes up the difference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from trainai.errors import ConfigError
from trainai.serialise import checked_fields

__all__ = ["Precision", "ScheduleName", "TrainConfig"]

ScheduleName = Literal["cosine", "linear", "constant"]
Precision = Literal["auto", "bf16", "fp16", "fp32"]

#: Warmup as a share of total steps when it is not given explicitly. Enough to get
#: Adam's second-moment estimate off the ground before the learning rate is high
#: enough to do damage with it.
DEFAULT_WARMUP_RATIO = 0.02

#: Warmup is never shorter than this, because a 200-step run with 2% warmup would
#: get four steps of it, which is the same as none.
MIN_WARMUP_STEPS = 20


@dataclass(frozen=True)
class TrainConfig:
    """How to train. Frozen: a checkpoint records it and a resume must not differ.

    Args:
        steps: Optimizer steps to run. Not micro-steps: with ``grad_accum`` 4, one
            step reads four batches.
        batch_size: Sequences per forward pass. Sized to fit VRAM.
        grad_accum: Forward passes summed per optimizer step. The effective batch
            is ``batch_size * grad_accum``.
        seq_len: Tokens per sequence. Must not exceed the model's context.
        lr: Peak learning rate, reached at the end of warmup.
        min_lr_ratio: Floor as a fraction of ``lr``, for the decaying schedules.
            0.1 is standard; 0 decays to nothing and wastes the last steps.
        warmup_steps: Steps spent ramping from 0 to ``lr``. ``None`` derives it
            from ``DEFAULT_WARMUP_RATIO``.
        schedule: How the learning rate falls after warmup.
        weight_decay: AdamW decay, applied to matrices only. Norm weights and
            embeddings are excluded; see ``GPT.parameter_groups``.
        beta1, beta2, eps: AdamW parameters. ``beta2`` 0.95 rather than 0.999 --
            at a few thousand steps, 0.999 averages over more history than the run
            contains.
        grad_clip: Global gradient-norm clip. 0 disables it.
        eval_every: Optimizer steps between validation passes. 0 disables
            validation entirely, which means no held-out number at all.
        eval_batches: Validation batches per pass. Fixed and sequential from token
            0, so successive validation losses are comparable.
        checkpoint_every: Steps between checkpoints. 0 saves only at the end.
        keep_checkpoints: How many step checkpoints to retain, newest first. The
            best-so-far and the final one are always kept.
        log_every: Steps between metric records.
        seed: Seeds parameter initialisation, dropout, and batch order.
        precision: ``auto`` picks bf16 on hardware that supports it, fp16 with a
            gradient scaler otherwise, and fp32 on CPU.
        device: ``auto`` prefers CUDA.
    """

    steps: int
    batch_size: int = 8
    grad_accum: int = 1
    seq_len: int = 256
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int | None = None
    schedule: ScheduleName = "cosine"
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 20
    checkpoint_every: int = 500
    keep_checkpoints: int = 3
    log_every: int = 10
    seed: int = 1234
    precision: Precision = "auto"
    device: str = "auto"

    def __post_init__(self) -> None:
        self._positive("steps", self.steps)
        self._positive("batch_size", self.batch_size)
        self._positive("grad_accum", self.grad_accum)
        self._positive("seq_len", self.seq_len)
        self._positive("eval_batches", self.eval_batches)
        self._positive("log_every", self.log_every)

        if self.seq_len < 2:
            raise ConfigError(
                f"--seq-len must be at least 2, got {self.seq_len}.",
                hint="A sequence needs one input token and one target token.",
                details={"seq_len": self.seq_len},
            )
        if self.lr <= 0:
            raise ConfigError(
                f"--lr must be positive, got {self.lr}.",
                hint="3e-4 is a reasonable starting point for a model of this size.",
                details={"lr": self.lr},
            )
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ConfigError(
                f"--min-lr-ratio must be between 0 and 1, got {self.min_lr_ratio}.",
                hint="0.1 keeps the last steps useful; 1.0 makes the schedule constant.",
                details={"min_lr_ratio": self.min_lr_ratio},
            )
        if self.warmup_steps is not None:
            if self.warmup_steps < 0:
                raise ConfigError(
                    f"--warmup-steps cannot be negative, got {self.warmup_steps}.",
                    hint="Use 0 for no warmup.",
                    details={"warmup_steps": self.warmup_steps},
                )
            if self.warmup_steps >= self.steps:
                raise ConfigError(
                    f"--warmup-steps {self.warmup_steps} is not less than --steps "
                    f"{self.steps}, so the learning rate would never finish rising.",
                    hint=(
                        f"Use at most {max(1, self.steps // 10)} warmup steps for a "
                        f"{self.steps}-step run, or raise --steps."
                    ),
                    details={"warmup_steps": self.warmup_steps, "steps": self.steps},
                )
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ConfigError(
                f"AdamW betas must be in [0, 1), got ({self.beta1}, {self.beta2}).",
                hint="The defaults are 0.9 and 0.95.",
                details={"beta1": self.beta1, "beta2": self.beta2},
            )
        if self.weight_decay < 0:
            raise ConfigError(
                f"--weight-decay cannot be negative, got {self.weight_decay}.",
                hint="Use 0 to disable it; 0.1 is the default.",
                details={"weight_decay": self.weight_decay},
            )
        if self.grad_clip < 0:
            raise ConfigError(
                f"--grad-clip cannot be negative, got {self.grad_clip}.",
                hint="Use 0 to disable clipping; 1.0 is the default.",
                details={"grad_clip": self.grad_clip},
            )
        if self.eval_every < 0 or self.checkpoint_every < 0 or self.keep_checkpoints < 0:
            raise ConfigError(
                "--eval-every, --checkpoint-every and --keep-checkpoints cannot be negative.",
                hint="Use 0 to disable the corresponding behaviour.",
                details={
                    "eval_every": self.eval_every,
                    "checkpoint_every": self.checkpoint_every,
                    "keep_checkpoints": self.keep_checkpoints,
                },
            )
        if self.precision not in ("auto", "bf16", "fp16", "fp32"):
            raise ConfigError(
                f"Unknown --precision {self.precision!r}.",
                hint="Use one of: auto, bf16, fp16, fp32.",
                details={"precision": self.precision},
            )
        if self.schedule not in ("cosine", "linear", "constant"):
            raise ConfigError(
                f"Unknown --schedule {self.schedule!r}.",
                hint="Use one of: cosine, linear, constant.",
                details={"schedule": self.schedule},
            )

    # -- derived ------------------------------------------------------------ #

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum

    @property
    def tokens_per_step(self) -> int:
        return self.effective_batch_size * self.seq_len

    @property
    def total_tokens(self) -> int:
        """Tokens the run will process. Not the corpus size -- epochs repeat."""
        return self.tokens_per_step * self.steps

    @property
    def resolved_warmup_steps(self) -> int:
        """Warmup length, derived when not given.

        Clamped below ``steps`` so the peak learning rate is always reached: a
        schedule that only ever warms up is indistinguishable from a much lower
        constant rate, and nothing in the output would say so.
        """
        if self.warmup_steps is not None:
            return self.warmup_steps
        derived = max(MIN_WARMUP_STEPS, int(self.steps * DEFAULT_WARMUP_RATIO))
        return min(derived, max(1, self.steps // 4))

    @property
    def min_lr(self) -> float:
        return self.lr * self.min_lr_ratio

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # ``warmup_steps`` stays raw -- None when it was never given -- so that
        # to_dict/from_dict is a faithful round trip. Writing the resolved value here
        # instead would silently pin an adaptive default: a plan.json round-tripped
        # through from_dict would carry warmup_steps=20, and overriding --steps 20 on
        # top of it then fails validation for a warmup the user never chose.
        payload["resolved_warmup_steps"] = self.resolved_warmup_steps
        payload["effective_batch_size"] = self.effective_batch_size
        payload["tokens_per_step"] = self.tokens_per_step
        payload["total_tokens"] = self.total_tokens
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainConfig:
        """Rebuild from :meth:`to_dict`, ignoring the derived extras.

        Strict about what it accepts, for the same reason
        :meth:`~trainai.model.ModelConfig.from_dict` is, but with a failure that is
        harder to see. A model rebuilt from the wrong shape eventually fails a weight
        load; a *run* rebuilt from the wrong hyperparameters just continues, at the
        wrong values, reporting nothing. Measured on a real checkpoint: dropping ``lr``
        resumed at 0.0003 instead of the recorded 0.000424, dropping ``batch_size``
        resumed at 8 instead of 32, and dropping ``seed`` resumed at 1234 instead of 99
        -- which changes the batch order, so the "bitwise identical" resume this
        module's own docstring promises was silently false.

        Wrong types were not caught either. ``steps: 100.5`` was accepted, making
        :attr:`total_tokens` a float; ``grad_accum: true`` was accepted as 1, because
        ``bool`` is an ``int``; and ``lr: "fast"`` surfaced as ``'<=' not supported
        between instances of 'str' and 'int'``, a Python operator error naming no field.

        ``schedule`` and ``precision`` are checked for *presence* here but not for type,
        because ``__post_init__`` already refuses their values with a message that lists
        what is allowed. Filtering to the declared fields is what lets a checkpoint
        written by an older version load: ``compile_model`` was a field that nothing
        read, and a config dict that still carries it is dropped rather than refused.
        """
        return cls(
            **checked_fields(
                cls,
                raw,
                subject="training configuration",
                incomplete_hint=(
                    "This checkpoint or plan file is incomplete. Re-create it with "
                    "`trainai train`, which records the full configuration. Filling in "
                    "a default would resume this run on different hyperparameters than "
                    "it was trained with, and nothing in the output would say so."
                ),
            )
        )

    def describe(self) -> str:
        batch = f"{self.batch_size}"
        if self.grad_accum > 1:
            batch += f"x{self.grad_accum}={self.effective_batch_size}"
        return (
            f"{self.steps} steps, batch {batch}, ctx {self.seq_len}, "
            f"lr {self.lr:g} {self.schedule} after {self.resolved_warmup_steps} warmup"
        )

    @staticmethod
    def _positive(name: str, value: int) -> None:
        if value <= 0:
            flag = "--" + name.replace("_", "-")
            raise ConfigError(
                f"{flag} must be positive, got {value}.",
                hint=f"Set {flag} to at least 1.",
                details={name: value},
            )
