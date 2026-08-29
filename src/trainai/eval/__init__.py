"""Evaluation: measuring a trained model on held-out text."""

from trainai.eval.perplexity import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_BATCHES,
    DatasetCheck,
    EvalReport,
    SplitResult,
    dataset_check,
    evaluate,
    evaluate_split,
)

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_BATCHES",
    "DatasetCheck",
    "EvalReport",
    "SplitResult",
    "dataset_check",
    "evaluate",
    "evaluate_split",
]
