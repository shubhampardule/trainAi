"""Export: turning a run into a model directory that other tools can load.

Separate from ``trainai.train`` and ``trainai.infer`` because it shares almost
nothing with them -- no optimizer, no sampling loop, just a state dict, a config
format, and the verification that the two agree.

The public entry point is :func:`export_run`. The two formats:

``hf``
    A directory ``transformers`` loads as ``LlamaForCausalLM``. This is the default
    because it is the one that makes the model *usable* outside this project.

``safetensors``
    The same weights under TrainAI's own names, with the ``ModelConfig`` that built
    them. For sharing a model without the optimizer state a checkpoint carries, or
    for loading the weights without any framework's opinion about them.
"""

from trainai.export.bundle import (
    DEFAULT_DTYPE,
    DEFAULT_FORMAT,
    DTYPE_CHOICES,
    FORMAT_CHOICES,
    LOGITS_TOLERANCE,
    ExportedFile,
    ExportResult,
    VerifyCheck,
    export_run,
)

__all__ = [
    "DEFAULT_DTYPE",
    "DEFAULT_FORMAT",
    "DTYPE_CHOICES",
    "FORMAT_CHOICES",
    "LOGITS_TOLERANCE",
    "ExportResult",
    "ExportedFile",
    "VerifyCheck",
    "export_run",
]
