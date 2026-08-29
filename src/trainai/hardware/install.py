"""Work out which PyTorch build this machine needs, and say so.

The problem this solves is narrow and common. PyPI's default ``torch`` wheel is
CPU-only on Windows and on Linux it varies by release; a user with a perfectly good
NVIDIA card installs TrainAI, gets the CPU build, and trains twenty to a hundred
times slower than their hardware allows. From inside PyTorch that situation
is indistinguishable from having no GPU: ``torch.cuda.is_available()`` returns False
either way. So TrainAI checks the *system* for a GPU, compares that against what
PyTorch can actually see, and names the gap.

Two things this module deliberately does not do.

It does not claim to know the current list of PyTorch wheel channels. Those move
with every release, and a hardcoded list that has gone stale produces a command
that fails with a 404 -- worse than no command. What it does instead is read the
CUDA version the *driver* reports and pick the closest channel from a table that
says when it was written, alongside the pytorch.org selector, which is the
authoritative source. The suggestion is a shortcut, not a promise.

It does not install anything by itself unless asked twice. Downloading two and a
half gigabytes into someone's environment, possibly the system Python, is not a
thing a diagnostic command should do on its own initiative.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from trainai.hardware.probe import describe_unusable_gpu, detect_gpu_vendors

__all__ = ["InstallAdvice", "advise_install", "detect_driver_cuda_version", "in_virtualenv"]

#: The pytorch.org selector. Always shown, because it is the source of truth and
#: this module's table is a convenience that can go stale.
PYTORCH_SELECTOR = "https://pytorch.org/get-started/locally/"

#: CUDA wheel channels published by PyTorch, newest first, as of August 2026.
#: Used to pick the highest channel a given driver can run. If PyTorch has since
#: added or retired channels this list is out of date -- which is why every message
#: that uses it also points at the selector above.
_CUDA_CHANNELS: tuple[tuple[tuple[int, int], str], ...] = (
    ((12, 8), "cu128"),
    ((12, 6), "cu126"),
    ((12, 4), "cu124"),
    ((12, 1), "cu121"),
    ((11, 8), "cu118"),
)

#: ROCm wheel channel, same caveat as the CUDA table.
_ROCM_CHANNEL = "rocm6.2"

#: Matches the CUDA version out of ``nvidia-smi``'s header, in both the spellings
#: NVIDIA has used. Recent drivers print ``CUDA UMD Version: 13.3`` where older
#: ones printed ``CUDA Version: 12.4``, and ``nvidia-smi -q`` now marks the old key
#: as "Deprecated; will be removed in CUDA 14.0". A regex pinned to the old
#: spelling matches nothing on a current driver and fails silently, which is how
#: this was found: driver 610.88 reported CUDA 13.3 and TrainAI concluded it could
#: not determine a version at all.
_CUDA_VERSION_RE = re.compile(r"CUDA(?:\s+\w+)?\s+Version\s*:\s*(\d+)\.(\d+)")

_TIMEOUT_SECONDS = 10


@dataclass
class InstallAdvice:
    """What to do about this machine's PyTorch install.

    Attributes:
        situation: One-line description of what was found.
        needs_action: Whether anything is actually wrong. False means the install
            already matches the hardware.
        command: The pip command to run, or ``None`` when there is nothing to run.
        notes: Caveats and explanations, each one a complete sentence.
        detected_vendors: GPU vendors found on the system.
        driver_cuda_version: What the NVIDIA driver reports supporting, if any.
        torch_build: How the installed PyTorch was built (``cuda 12.6``, ``cpu``...).
    """

    situation: str
    needs_action: bool
    command: str | None = None
    notes: list[str] = field(default_factory=list)
    detected_vendors: list[str] = field(default_factory=list)
    driver_cuda_version: tuple[int, int] | None = None
    torch_build: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "situation": self.situation,
            "needs_action": self.needs_action,
            "command": self.command,
            "notes": self.notes,
            "detected_vendors": self.detected_vendors,
            "driver_cuda_version": (
                f"{self.driver_cuda_version[0]}.{self.driver_cuda_version[1]}"
                if self.driver_cuda_version
                else None
            ),
            "torch_build": self.torch_build,
            "selector": PYTORCH_SELECTOR,
        }


def in_virtualenv() -> bool:
    """Whether Python is running inside a virtual environment.

    ``sys.prefix != sys.base_prefix`` covers venv and virtualenv; conda sets
    ``CONDA_PREFIX`` without changing the prefixes, so it is checked separately.
    Installing a 2.5 GB wheel into the system Python is worth refusing by default.
    """
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return True
    return bool(os.environ.get("CONDA_PREFIX"))


def detect_driver_cuda_version() -> tuple[int, int] | None:
    """The maximum CUDA version this NVIDIA driver supports, from ``nvidia-smi``.

    This is a measurement, not a guess: the driver reports what it can run, which
    is what decides which wheel channel will work. Returns ``None`` when
    ``nvidia-smi`` is absent, fails, or prints something unexpected -- all of which
    are ordinary and none of which should raise.

    ``encoding`` and ``errors`` are stated rather than left to ``text=True``, which
    decodes with the locale's codec and ``errors=None`` -- strict. ``nvidia-smi``
    draws a box-art table and prints the GPU's marketing name, so a byte the locale
    cannot decode is not exotic, and the consequences are platform-specific and both
    bad: on POSIX the ``UnicodeDecodeError`` is raised on this thread, past the
    handler below, and takes ``doctor`` and ``setup`` down with a traceback; on
    Windows the decode happens in a reader thread, so the traceback is printed
    straight at the user by the threading machinery and ``result.stdout`` comes back
    as ``None``. ``errors="replace"`` is the right trade here because the only thing
    read out of this output is a version number matched by regex -- refusing to
    report the driver over one odd byte elsewhere in the table would be worse.
    """
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - driver-level
        return None
    match = _CUDA_VERSION_RE.search(result.stdout or "")
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _channel_for_driver(driver: tuple[int, int]) -> str | None:
    """Highest published channel the driver can run, or ``None`` if it is too old.

    A driver newer than every entry in the table is fine and picks the top one:
    NVIDIA drivers run older CUDA runtimes, so a CUDA 13.3 driver runs a cu128
    wheel. What the table cannot know is whether PyTorch has since published a
    *newer* channel, which is why the selector URL travels with every suggestion.
    """
    for required, channel in _CUDA_CHANNELS:
        if driver >= required:
            return channel
    return None


def _torch_build() -> tuple[str | None, bool]:
    """How the installed PyTorch was built, and whether it can see an accelerator.

    Returns ``(description, has_accelerator)``. ``description`` is ``None`` when
    torch is not installed at all.
    """
    try:
        import torch
    except Exception:
        return None, False

    version = getattr(torch, "version", None)
    cuda = getattr(version, "cuda", None)
    hip = getattr(version, "hip", None)
    has_accelerator = False
    try:
        has_accelerator = bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - driver-level failure
        has_accelerator = False
    if not has_accelerator:
        xpu = getattr(torch, "xpu", None)
        with suppress(Exception):
            has_accelerator = bool(xpu is not None and xpu.is_available())
    if not has_accelerator:
        mps = getattr(torch.backends, "mps", None)
        with suppress(Exception):
            has_accelerator = bool(mps is not None and mps.is_available())

    if hip:
        return f"ROCm {hip}", has_accelerator
    if cuda:
        return f"CUDA {cuda}", has_accelerator
    return "CPU-only", has_accelerator


def advise_install(vendors: list[str] | None = None) -> InstallAdvice:
    """Diagnose the PyTorch install against the hardware and say what to do.

    Args:
        vendors: GPU vendors present on the system. Passed in when a hardware probe
            has already run, so the detection is not repeated.
    """
    found = sorted(vendors) if vendors is not None else detect_gpu_vendors()
    build, has_accelerator = _torch_build()
    driver = detect_driver_cuda_version() if "nvidia" in found else None
    system = platform.system()

    if build is None:
        return InstallAdvice(
            situation="PyTorch is not installed.",
            needs_action=True,
            command=_install_command(found, driver, system),
            notes=[
                "TrainAI declares torch as a dependency, so this usually means the "
                "install did not finish. Re-run it, or use the command below to get "
                "the build that matches this machine.",
                f"The authoritative source for these commands is {PYTORCH_SELECTOR}.",
            ],
            detected_vendors=found,
            driver_cuda_version=driver,
        )

    if has_accelerator:
        return InstallAdvice(
            situation=f"PyTorch ({build}) can already use this machine's accelerator.",
            needs_action=False,
            notes=[
                "Nothing to change. Run `trainai doctor` for what the accelerator can "
                "actually do, including how much VRAM is free right now."
            ],
            detected_vendors=found,
            driver_cuda_version=driver,
            torch_build=build,
        )

    if not found:
        return InstallAdvice(
            situation=f"No GPU detected. PyTorch ({build}) will train on the CPU.",
            needs_action=False,
            notes=[
                "CPU training works and is roughly 20-100x slower than a modern GPU. "
                "For a first run on a small corpus that is survivable; for anything "
                "larger it is not.",
                "If this machine does have a GPU, TrainAI failed to detect it. Check "
                "that the vendor driver is installed, then report it as a bug.",
            ],
            detected_vendors=found,
            torch_build=build,
        )

    # The interesting case: hardware is present, PyTorch cannot use it.
    return _advise_unusable_gpu(found, driver, system, build)


def _advise_unusable_gpu(
    found: list[str], driver: tuple[int, int] | None, system: str, build: str
) -> InstallAdvice:
    """A GPU is present and PyTorch cannot see it. Say why, and what to run."""
    notes: list[str] = []
    command = _install_command(found, driver, system)

    if "nvidia" in found:
        if driver is None:
            notes.append(
                "An NVIDIA GPU was detected but `nvidia-smi` did not report a CUDA "
                "version, which usually means the driver is not installed or is too "
                "old. Install the current driver from nvidia.com first; no PyTorch "
                "wheel can work without it."
            )
        else:
            channel = _channel_for_driver(driver)
            notes.append(
                f"The driver reports supporting CUDA up to {driver[0]}.{driver[1]}, "
                + (
                    f"so the {channel} wheel is the newest one it can run."
                    if channel
                    else "which predates every PyTorch channel in this table. Update the driver."
                )
            )
            notes.append(
                "The installed PyTorch is a CPU-only build. That is the default wheel "
                "on some platforms, so this happens to people who did nothing wrong."
            )

    if "amd" in found:
        if system == "Linux":
            notes.append(
                "AMD GPUs need a ROCm build of PyTorch, and ROCm itself installed at "
                "the system level. Check that your card is on AMD's supported list "
                "before spending time on it -- consumer RDNA cards are supported "
                "unevenly."
            )
        elif system == "Windows":
            notes.append(
                "PyTorch has no ROCm build for Windows, so an AMD GPU cannot be used "
                "for training here directly. The options are WSL2 with ROCm, or CPU "
                "training. TrainAI will run on the CPU without complaint; it will just "
                "be slow, and it will tell you how slow."
            )
        else:
            notes.append(
                "AMD GPUs are only usable through ROCm, which is Linux-only. This "
                "machine will train on the CPU."
            )

    if "intel" in found and not ("nvidia" in found or "amd" in found):
        notes.append(
            "Intel GPUs need an XPU build of PyTorch. TrainAI detects and reports "
            "XPU devices, but has no automated coverage on them, so treat any "
            "throughput figure as unverified."
        )

    if "apple" in found:
        notes.append(
            "Apple Silicon works through Metal (MPS) and the default PyTorch wheel "
            "includes it. If MPS is unavailable the wheel is probably an x86 build "
            "running under Rosetta; reinstall Python as arm64."
        )

    notes.append(f"The authoritative source for install commands is {PYTORCH_SELECTOR}.")

    return InstallAdvice(
        situation=describe_unusable_gpu(found, build),
        needs_action=command is not None,
        command=command,
        notes=notes,
        detected_vendors=found,
        driver_cuda_version=driver,
        torch_build=build,
    )


def _install_command(found: list[str], driver: tuple[int, int] | None, system: str) -> str | None:
    """The pip command for this hardware, or ``None`` when no wheel would help.

    ``--upgrade --force-reinstall`` because the usual situation is a CPU-only torch
    already sitting in the environment at the same version number, which a plain
    install would leave alone.
    """
    base = "pip install --upgrade --force-reinstall torch"

    if "nvidia" in found and driver is not None:
        channel = _channel_for_driver(driver)
        if channel is not None:
            return f"{base} --index-url https://download.pytorch.org/whl/{channel}"
        return None

    if "amd" in found and system == "Linux":
        return f"{base} --index-url https://download.pytorch.org/whl/{_ROCM_CHANNEL}"

    if "intel" in found:
        return f"{base} --index-url https://download.pytorch.org/whl/xpu"

    if "apple" in found:
        return base

    return None
