"""Hardware detection and empirical capacity planning.

This subpackage answers three questions, in order:

1. :mod:`trainai.hardware.probe` -- *what do I actually have?*
2. :mod:`trainai.hardware.benchmark` -- *what does this machine actually do when
   I run a real training step on it?*
3. :mod:`trainai.hardware.planner` -- *given that, what should we train?*

Step 2 is not optional garnish. On Windows/WDDM, allocating more VRAM than the
card has does **not** raise ``torch.cuda.OutOfMemoryError``; the driver pages to
system RAM and throughput collapses by up to 45x while appearing to succeed.
Any purely analytical VRAM estimate will happily approve such a configuration.
TrainAI therefore measures instead of predicting.
"""

from __future__ import annotations

from trainai.hardware.probe import GPUInfo, HardwareProfile, probe_hardware

__all__ = ["GPUInfo", "HardwareProfile", "probe_hardware"]
