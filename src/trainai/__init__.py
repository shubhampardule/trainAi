"""TrainAI -- train small language models from scratch on your own hardware.

TrainAI takes a folder of text, works out what your machine can *actually*
train, and runs the training for you. It does not require you to write a
training loop, pick a learning rate, or guess whether a model will fit in VRAM.

Import-time contract
--------------------
Importing ``trainai`` must stay cheap. In particular this package does **not**
import ``torch`` at module scope, because ``import torch`` costs several seconds
and ``trainai --help`` has to feel instant. Torch is imported lazily inside the
functions that need it. Please preserve this when adding modules: anything that
touches torch belongs behind a function-local import or in a submodule that the
CLI imports only when the relevant command runs.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
