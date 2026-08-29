"""Learning-rate schedules, as pure functions of the step number.

This is deliberately not :mod:`torch.optim.lr_scheduler`. Those classes carry
mutable state -- a ``last_epoch`` counter advanced by ``step()`` -- and that state
has to be saved, restored, and advanced exactly as often on resume as it was
originally. Calling ``step()`` once too many or too few times shifts the entire
remaining schedule, and nothing reports it: the run continues at a plausible
learning rate that is simply not the one it should be on.

A schedule that is ``lr(step)`` cannot have that bug. There is no state to save,
resuming needs only the step number, and the learning rate at step 4,000 is the
same whether the run reached it in one go or across three restarts. The
:mod:`trainai.data.loader` design is the same idea applied to batch order.
"""

from __future__ import annotations

import math

from trainai.errors import ConfigError
from trainai.train.config import ScheduleName, TrainConfig

__all__ = ["LearningRateSchedule", "lr_at"]


def lr_at(
    step: int,
    *,
    peak_lr: float,
    total_steps: int,
    warmup_steps: int,
    min_lr: float = 0.0,
    schedule: ScheduleName = "cosine",
) -> float:
    """Learning rate for ``step``, counting from 0.

    Warmup is linear from 0 to ``peak_lr`` over ``warmup_steps``. After that the
    rate falls to ``min_lr`` by ``total_steps`` according to ``schedule``.

    Steps beyond ``total_steps`` return ``min_lr`` rather than extrapolating below
    it, so a run extended past its planned length keeps training at the floor
    instead of at a negative rate.
    """
    if step < 0:
        raise ConfigError(
            f"Step must not be negative, got {step}.",
            hint="This is a TrainAI bug in the training loop; please report it.",
            details={"step": step},
        )

    if warmup_steps > 0 and step < warmup_steps:
        # (step + 1) / warmup so that step 0 is not exactly zero: a first step with
        # no learning rate at all is a wasted forward and backward pass.
        return peak_lr * (step + 1) / warmup_steps

    if schedule == "constant":
        return peak_lr

    decay_steps = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / decay_steps
    progress = min(1.0, max(0.0, progress))

    if schedule == "cosine":
        # Half a cosine from 1 to 0. Spends longer near the peak and longer near
        # the floor than a straight line, which is where the useful learning is.
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    elif schedule == "linear":
        factor = 1.0 - progress
    else:  # pragma: no cover - TrainConfig validates the name
        raise ConfigError(
            f"Unknown schedule {schedule!r}.",
            hint="Use one of: cosine, linear, constant.",
            details={"schedule": schedule},
        )

    return min_lr + (peak_lr - min_lr) * factor


class LearningRateSchedule:
    """A :func:`lr_at` closure over one :class:`TrainConfig`.

    Stateless, so it needs nothing from a checkpoint. Constructed from the config
    the checkpoint already records, which is why a resumed run cannot drift onto a
    different schedule than the one it started on.
    """

    def __init__(self, config: TrainConfig) -> None:
        self._peak = config.lr
        self._total = config.steps
        self._warmup = config.resolved_warmup_steps
        self._min = config.min_lr
        self._schedule: ScheduleName = config.schedule

    def __call__(self, step: int) -> float:
        return lr_at(
            step,
            peak_lr=self._peak,
            total_steps=self._total,
            warmup_steps=self._warmup,
            min_lr=self._min,
            schedule=self._schedule,
        )

    def peak_step(self) -> int:
        """The step at which the learning rate first reaches its peak."""
        return max(0, self._warmup - 1)

    def describe(self) -> str:
        if self._schedule == "constant":
            return f"constant {self._peak:g} after {self._warmup} warmup steps"
        return (
            f"{self._schedule} from {self._peak:g} to {self._min:g} "
            f"over steps {self._warmup}-{self._total}"
        )
