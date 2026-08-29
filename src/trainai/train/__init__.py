"""Training: configuration, schedule, checkpoints, metrics, and the loop.

Imports torch at module scope. The CLI defers importing this package until a
command actually needs to train.
"""

from __future__ import annotations

from trainai.train.budget import (
    EPOCHS_BEFORE_MEMORISING,
    TOKENS_PER_PARAMETER_TARGET,
    DataBudget,
)
from trainai.train.checkpoint import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    Checkpoint,
    RngState,
    capture_rng,
    find_checkpoint,
    list_checkpoints,
    load_checkpoint,
    pointer_fallback_reason,
    restore_rng,
    save_checkpoint,
)
from trainai.train.config import Precision, ScheduleName, TrainConfig
from trainai.train.loop import (
    Trainer,
    TrainResult,
    check_configs_agree,
    precision_for,
    resolve_device,
    resolve_precision,
)
from trainai.train.metrics import (
    MetricsWriter,
    ThroughputMeter,
    TrainingDisplay,
    perplexity,
    summarise_run,
)
from trainai.train.schedule import LearningRateSchedule, lr_at

__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "EPOCHS_BEFORE_MEMORISING",
    "TOKENS_PER_PARAMETER_TARGET",
    "Checkpoint",
    "DataBudget",
    "LearningRateSchedule",
    "MetricsWriter",
    "Precision",
    "RngState",
    "ScheduleName",
    "ThroughputMeter",
    "TrainConfig",
    "TrainResult",
    "Trainer",
    "TrainingDisplay",
    "capture_rng",
    "check_configs_agree",
    "find_checkpoint",
    "list_checkpoints",
    "load_checkpoint",
    "lr_at",
    "perplexity",
    "pointer_fallback_reason",
    "precision_for",
    "resolve_device",
    "resolve_precision",
    "restore_rng",
    "save_checkpoint",
    "summarise_run",
]
