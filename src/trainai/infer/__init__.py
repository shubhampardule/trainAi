"""Inference: loading a finished run and generating text from it.

Kept separate from ``trainai.train`` because the two have different needs from the
same directory -- training writes checkpoints and needs an optimizer, inference reads
one checkpoint and needs a tokenizer -- and because ``trainai chat`` should not pay
for importing the trainer.
"""

from trainai.infer.session import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_WHICH,
    WHICH_CHOICES,
    Finish,
    InferenceSession,
    RunLayout,
    StreamPiece,
    locate_run,
)

__all__ = [
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_REPETITION_PENALTY",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_K",
    "DEFAULT_TOP_P",
    "DEFAULT_WHICH",
    "WHICH_CHOICES",
    "Finish",
    "InferenceSession",
    "RunLayout",
    "StreamPiece",
    "locate_run",
]
