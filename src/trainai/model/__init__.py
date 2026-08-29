"""The model: configuration and the transformer itself.

Unlike :mod:`trainai.data`, this package imports torch at module scope. Anything
that imports it has already decided to pay for that, and the CLI defers the import
until a command needs a model.
"""

from __future__ import annotations

from trainai.model.config import PRESETS, ModelConfig, preset
from trainai.model.gpt import GPT, Attention, Block, FeedForward, RMSNorm, RotaryEmbedding

__all__ = [
    "GPT",
    "PRESETS",
    "Attention",
    "Block",
    "FeedForward",
    "ModelConfig",
    "RMSNorm",
    "RotaryEmbedding",
    "preset",
]
