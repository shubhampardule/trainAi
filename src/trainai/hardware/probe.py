"""Detect what the machine actually has, without guessing.

Everything in here degrades gracefully: a hardware probe that raises is worse
than one that reports "unknown", because the probe runs before we can possibly
know whether the user's environment is exotic. Failures are recorded in
:attr:`HardwareProfile.warnings` rather than propagated.

``torch`` is imported lazily so that this module stays importable (and testable)
on a machine with no torch at all.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
from collections.abc import Iterable, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "GPUInfo",
    "HardwareProfile",
    "describe_unusable_gpu",
    "detect_gpu_vendors",
    "gpu_pronoun",
    "probe_hardware",
    "vendor_labels",
]

#: Fraction of *free* VRAM we are willing to hand to a training run by default.
#: The remainder absorbs allocator fragmentation, the CUDA context, cuDNN
#: workspaces, and whatever the desktop compositor decides to do mid-run.
DEFAULT_VRAM_SAFETY_FRACTION = 0.85

#: Compute capability at which NVIDIA hardware gained bf16 and TF32. Only
#: meaningful for real CUDA devices -- see :func:`_probe_cuda_devices` for why it
#: must never be applied to a ROCm device.
_NVIDIA_BF16_CAPABILITY = (8, 0)

#: PCI vendor ids, for detecting a GPU that PyTorch cannot see. This is the
#: diagnostic that matters most to a new user: "you have an NVIDIA card and the
#: CPU-only PyTorch wheel" is a fixable situation, and without a system-level
#: check TrainAI can only say "no GPU found", which sounds like a hardware
#: problem rather than an install one.
_PCI_VENDORS = {
    "0x10de": "nvidia",
    "0x1002": "amd",
    "0x1022": "amd",
    "0x8086": "intel",
}

#: Where Linux exposes the DRM cards. A module constant rather than a literal so the
#: card loop can be tested on a machine that is not Linux, with a directory a test
#: builds -- including the malformed cards that motivated the loop's error handling,
#: which no real machine can be asked to produce on demand.
_DRM_CLASS_PATH = "/sys/class/drm"

#: Where the cgroup filesystem is mounted. A module constant for the same reason as
#: ``_DRM_CLASS_PATH``: no CI runner can be asked to impose a specific CPU or memory
#: quota on demand, so the readers below are tested against a directory a test builds.
_CGROUP_ROOT = "/sys/fs/cgroup"

#: A cgroup v1 "no limit" is a sentinel, not an absence: the kernel reports
#: ``PAGE_COUNTER_MAX`` scaled by the page size, which lands around 8 EiB. Anything at
#: or above this is read as unlimited rather than as a quota no machine could honour.
_CGROUP_UNLIMITED_AT_OR_ABOVE = 1 << 62


@dataclass(frozen=True)
class GPUInfo:
    """A single accelerator, as reported by the driver at probe time."""

    index: int
    name: str
    total_vram_bytes: int
    #: VRAM the driver says is *currently* free. This is the number that matters:
    #: on a laptop running a desktop session, a 4 GiB card typically has ~3.2 GiB
    #: free before we allocate anything.
    free_vram_bytes: int
    compute_capability: tuple[int, int] | None = None
    multiprocessor_count: int | None = None
    supports_bf16: bool = False
    #: Which runtime this device is reached through: ``cuda``, ``rocm``, ``xpu`` or
    #: ``mps``. ROCm devices are addressed through the ``torch.cuda`` API but are not
    #: CUDA devices, and several numbers mean different things on them. On ``mps`` the
    #: two memory figures are a driver ceiling and a derived headroom rather than a
    #: card's own VRAM -- see :func:`_probe_mps_device`.
    backend: str = "cuda"

    @property
    def used_vram_bytes(self) -> int:
        return max(0, self.total_vram_bytes - self.free_vram_bytes)

    @property
    def capability_str(self) -> str:
        """Human-readable capability, labelled by what it actually is.

        On CUDA this is the compute capability. On ROCm the same fields carry the
        gfx architecture, which is a different thing with a confusingly similar
        shape, so it is labelled rather than presented as a capability.
        """
        if self.compute_capability is None:
            return "unknown"
        major, minor = self.compute_capability
        if self.backend == "rocm":
            return f"gfx{major}{minor}"
        return f"{major}.{minor}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["compute_capability"] = self.capability_str
        return data


@dataclass(frozen=True)
class HardwareProfile:
    """A snapshot of the machine, taken immediately before planning or training.

    Serialised into every run's ``plan.json`` so that a result can be interpreted
    later, on a different machine, by a different person.
    """

    # --- software ---
    platform_summary: str
    os_name: str
    python_version: str
    torch_version: str | None
    cuda_version: str | None

    # --- compute ---
    device_type: str  # "cuda" | "xpu" | "mps" | "cpu" -- the torch device string
    gpus: list[GPUInfo] = field(default_factory=list)
    #: Which runtime is in use: ``cuda``, ``rocm``, ``xpu``, ``mps`` or ``cpu``.
    #: Distinct from ``device_type`` because ROCm devices are addressed as
    #: ``cuda`` by PyTorch while behaving differently in ways that matter.
    backend: str = "cpu"
    #: GPU vendors present according to the *system*, whether or not PyTorch can
    #: see them. A non-empty list here with an empty ``gpus`` is the signature of
    #: an install problem rather than a hardware one.
    detected_vendors: list[str] = field(default_factory=list)

    # --- host ---
    cpu_name: str | None = None
    cpu_count_logical: int = 1
    cpu_count_physical: int | None = None
    total_ram_bytes: int = 0
    available_ram_bytes: int = 0
    free_disk_bytes: int = 0

    # --- what a container will actually let this process have ---
    #: Cores this process may use, when a cgroup quota or a CPU affinity mask allows
    #: fewer than the machine has. ``None`` means unconstrained. Fractional because a
    #: cgroup quota is: ``cpu.max`` of ``150000 100000`` is one and a half cores.
    cpu_limit: float | None = None
    #: The cgroup memory limit in bytes, when one is set and is below the host's RAM.
    #: ``None`` means unconstrained.
    ram_limit_bytes: int | None = None
    #: Bytes charged against that limit right now. ``None`` when there is no limit, or
    #: when the limit is readable but the usage is not.
    ram_limit_used_bytes: int | None = None

    # --- behavioural flags that change how we plan ---
    #: True on Windows with an NVIDIA consumer GPU. Under the WDDM driver model
    #: the OS lets CUDA oversubscribe VRAM and pages the excess to system RAM,
    #: so exceeding VRAM produces a silent 5-45x slowdown instead of an
    #: ``OutOfMemoryError``. Measured on an RTX 2050: a configuration peaking at
    #: 6.55 GiB "succeeded" on a 4 GiB card at 1/23rd of the expected speed.
    #: When this is True, TrainAI must reject configurations on *measured memory
    #: and throughput*, never on the absence of an OOM exception.
    silent_vram_spillover: bool = False
    supports_bf16: bool = False
    supports_tf32: bool = False
    #: torch.compile needs Triton, which is frequently absent on Windows.
    torch_compile_available: bool = False

    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    @property
    def primary_gpu(self) -> GPUInfo | None:
        """The GPU we would train on, or ``None`` when running on CPU."""
        return self.gpus[0] if self.gpus else None

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpus)

    @property
    def has_unusable_gpu(self) -> bool:
        """A GPU exists on this machine but PyTorch cannot use it.

        Almost always the CPU-only PyTorch wheel. Worth distinguishing from "no
        GPU" because it is fixable in one command, and because the two produce the
        same symptom -- training on the CPU -- with entirely different remedies.
        """
        return bool(self.detected_vendors) and not self.gpus

    @property
    def usable_cpu_count(self) -> int:
        """Cores this process may actually use. Never zero, never above the machine's.

        Read this rather than :attr:`cpu_count_logical` anywhere the number is going to
        be reported as what the run has, because inside a container the two differ and
        only this one is true. Truncated rather than rounded: half a core is not a core
        you can put a thread on, and a quota below one still leaves one thread to run.
        """
        if self.cpu_limit is None:
            return self.cpu_count_logical
        return max(1, min(self.cpu_count_logical, int(self.cpu_limit)))

    @property
    def usable_available_ram_bytes(self) -> int:
        """RAM this process may still allocate, honouring a container limit.

        The lower of what the host has free and what is left of the cgroup limit. Both
        matter: a 64 GiB limit on a host with 2 GiB free does not give you 64 GiB, and
        2 GiB free on a host does not help if the cgroup will kill you at 512 MiB.
        """
        if self.ram_limit_bytes is None:
            return self.available_ram_bytes
        headroom = max(0, self.ram_limit_bytes - (self.ram_limit_used_bytes or 0))
        if not self.available_ram_bytes:
            return headroom
        return min(self.available_ram_bytes, headroom)

    @property
    def is_constrained(self) -> bool:
        """Whether a container quota makes this process smaller than its machine."""
        return self.cpu_limit is not None or self.ram_limit_bytes is not None

    def vram_budget_bytes(self, safety_fraction: float = DEFAULT_VRAM_SAFETY_FRACTION) -> int:
        """How many bytes a training run may use before we call it unsafe.

        Based on *free* VRAM, not total. Returns 0 when there is no GPU.
        """
        gpu = self.primary_gpu
        if gpu is None:
            return 0
        return int(gpu.free_vram_bytes * safety_fraction)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["gpus"] = [gpu.to_dict() for gpu in self.gpus]
        return data


# --------------------------------------------------------------------------- #
# Probing
# --------------------------------------------------------------------------- #
def probe_hardware(*, disk_path: str | os.PathLike[str] | None = None) -> HardwareProfile:
    """Inspect the current machine and return a :class:`HardwareProfile`.

    Args:
        disk_path: Where the run will write. Free space is measured for this
            location's filesystem, since that is the one that can fill up.
    """
    warnings: list[str] = []

    cpu_name, logical, physical = _probe_cpu()
    total_ram, available_ram = _probe_ram(warnings)
    free_disk = _probe_disk(disk_path, warnings)
    cpu_limit = _probe_cpu_limit(logical)
    ram_limit, ram_limit_used = _probe_ram_limit(total_ram)

    torch_version: str | None = None
    cuda_version: str | None = None
    device_type = "cpu"
    backend = "cpu"
    gpus: list[GPUInfo] = []
    supports_bf16 = False
    supports_tf32 = False

    torch = _try_import_torch(warnings)
    if torch is not None:
        torch_version = str(torch.__version__)
        cuda_version = getattr(torch.version, "cuda", None)
        hip_version = getattr(torch.version, "hip", None)
        device_type, backend, gpus, supports_bf16, supports_tf32 = _probe_accelerators(
            torch, warnings, available_ram_bytes=available_ram
        )
        if backend == "rocm" and hip_version:
            # Report the version of the runtime actually in use, not an absent
            # CUDA version that would read as "no GPU support compiled in".
            cuda_version = f"ROCm {hip_version}"

    vendors = detect_gpu_vendors()
    if vendors and not gpus and backend == "cpu":
        warnings.append(
            describe_unusable_gpu(vendors) + " This is almost always the CPU-only "
            "PyTorch wheel rather than a hardware problem. Run `trainai setup` for "
            "the install command that matches this machine."
        )

    is_windows = platform.system() == "Windows"
    # WDDM oversubscription is a property of the Windows driver model, not of one
    # vendor, so the flag is set for any Windows GPU. It only ever makes TrainAI
    # more careful -- measure rather than trust the absence of an OOM -- so
    # applying it to AMD, where it has not been measured, errs in the safe
    # direction. The measurement behind it was taken on an NVIDIA card.
    silent_spillover = is_windows and backend in ("cuda", "rocm")

    return HardwareProfile(
        platform_summary=_platform_summary(),
        os_name=platform.system(),
        python_version=platform.python_version(),
        torch_version=torch_version,
        cuda_version=cuda_version,
        device_type=device_type,
        backend=backend,
        detected_vendors=vendors,
        gpus=gpus,
        cpu_name=cpu_name,
        cpu_count_logical=logical,
        cpu_count_physical=physical,
        total_ram_bytes=total_ram,
        available_ram_bytes=available_ram,
        free_disk_bytes=free_disk,
        cpu_limit=cpu_limit,
        ram_limit_bytes=ram_limit,
        ram_limit_used_bytes=ram_limit_used,
        silent_vram_spillover=silent_spillover,
        supports_bf16=supports_bf16,
        supports_tf32=supports_tf32,
        torch_compile_available=_probe_torch_compile(),
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# Individual probes -- each one swallows its own failure
# --------------------------------------------------------------------------- #
def _try_import_torch(warnings: list[str]) -> Any | None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is a hard dependency
        warnings.append(f"PyTorch could not be imported ({exc.__class__.__name__}: {exc}).")
        return None
    return torch


def _probe_accelerators(
    torch: Any, warnings: list[str], *, available_ram_bytes: int = 0
) -> tuple[str, str, list[GPUInfo], bool, bool]:
    """Return ``(device_type, backend, gpus, supports_bf16, supports_tf32)``.

    ``available_ram_bytes`` is only used by the Apple branch, where the GPU has no
    memory of its own and the ceiling on what it may allocate is partly a fact about
    system RAM. Every other backend reads its own device memory and ignores it.
    """
    try:
        cuda_available = torch.cuda.is_available()
    except Exception as exc:  # pragma: no cover - driver-level failure
        warnings.append(f"CUDA availability check failed: {exc}")
        cuda_available = False

    if cuda_available:
        # A ROCm build of PyTorch reports CUDA as available and exposes AMD GPUs
        # through the whole torch.cuda API. torch.version.hip is the only reliable
        # way to tell the two apart, and telling them apart matters: several
        # numbers under torch.cuda mean different things on ROCm.
        is_rocm = bool(getattr(torch.version, "hip", None))
        backend = "rocm" if is_rocm else "cuda"
        gpus = _probe_cuda_devices(torch, warnings, backend=backend)

        bf16 = _probe_bf16(torch, gpus, backend, warnings)
        # TF32 is an NVIDIA tensor-core format. There is no such thing on AMD, so
        # this must be False on ROCm rather than inferred from the gfx number.
        tf32 = (not is_rocm) and any(
            g.compute_capability is not None and g.compute_capability >= _NVIDIA_BF16_CAPABILITY
            for g in gpus
        )
        if is_rocm:
            warnings.append(
                "ROCm detected. TrainAI's hardware handling is written for it but has "
                "no automated coverage on AMD hardware, so treat throughput figures "
                "as unverified. Precision support is read from the runtime rather "
                "than inferred from the architecture number."
            )
        return "cuda", backend, gpus, bf16, tf32

    # Intel discrete and integrated GPUs, through torch.xpu.
    xpu_gpus = _probe_xpu_devices(torch, warnings)
    if xpu_gpus:
        warnings.append(
            "Intel XPU detected. TrainAI has no automated coverage on Intel GPUs; "
            "treat performance numbers on this device as unverified."
        )
        return "xpu", "xpu", xpu_gpus, _probe_xpu_bf16(torch), False

    # Apple Silicon. Reported with a memory ceiling but no peak counter: see
    # _probe_mps_device for which of those numbers are real and which are not.
    try:
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            warnings.append(
                "Apple MPS detected. TrainAI has no automated MPS test coverage yet; "
                "treat performance numbers on this device as unverified. Training "
                "will run in fp32, because autocast on MPS is not something this "
                "project has been able to verify."
            )
            mps_gpus = _probe_mps_device(torch, warnings, available_ram_bytes=available_ram_bytes)
            return "mps", "mps", mps_gpus, False, False
    except Exception:  # pragma: no cover
        pass

    return "cpu", "cpu", [], False, False


def _probe_bf16(torch: Any, gpus: list[GPUInfo], backend: str, warnings: list[str]) -> bool:
    """Whether bf16 is usable, asked of the runtime rather than inferred.

    ``torch.cuda.is_bf16_supported()`` queries the driver and works on both CUDA
    and ROCm builds, so it is the primary source on both. The difference is the
    fallback: on CUDA, compute capability 8.0+ is a correct rule, and on ROCm there
    is no equivalent rule to fall back to -- so if the runtime cannot answer, the
    honest result is "no", which sends training to fp32. Slower and correct beats
    faster and wrong.
    """
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception as exc:  # pragma: no cover - driver-level failure
        if backend == "rocm":
            warnings.append(
                f"Could not ask the ROCm runtime about bf16 support ({exc}); assuming "
                "it is unavailable and using fp32. There is no architecture number "
                "that reliably implies bf16 on AMD, so guessing here would risk "
                "training in a format the hardware does not have."
            )
            return False
        warnings.append(f"Could not query bf16 support ({exc}); falling back to capability.")
        return any(
            g.compute_capability is not None and g.compute_capability >= _NVIDIA_BF16_CAPABILITY
            for g in gpus
        )


def _probe_mps_device(
    torch: Any, warnings: list[str], *, available_ram_bytes: int = 0
) -> list[GPUInfo]:
    """Apple Silicon as a single device, with a ceiling that is honest about its source.

    The GPU has no memory of its own. What ``torch.mps`` exposes is:

    ``recommended_max_memory()``
        Metal's ``recommendedMaxWorkingSetSize`` -- a driver-declared ceiling on one
        process's GPU allocation, typically around 75% of installed RAM. It is a
        property of the machine, not a reading of what is free right now.
    ``driver_allocated_memory()``
        A live reading of what this process has actually taken.
    ``current_allocated_memory()``
        The same, counted by the allocator rather than the driver.

    What it does **not** expose is any peak counter: there is no
    ``max_memory_allocated`` and no ``reset_peak_memory_stats``, which is why
    :func:`trainai.hardware.benchmark` still reports ``memory_measured=False`` here.
    Sampling the current allocation and calling it a peak would be a fabricated number
    in a field whose whole purpose is to say whether a number was measured.

    The free figure is the **lower** of the driver ceiling minus what torch holds, and
    what the OS says is actually free. Taking the minimum is the point: on unified
    memory the ceiling ignores every other process on the machine, so a 24 GiB Mac with
    a browser open would otherwise be planned as though 18 GiB were waiting. Before
    this, ``gpus`` was empty on Apple, ``primary_gpu`` was ``None``, and
    ``vram_budget_bytes()`` returned 0 -- so the planner's over-budget check, which is
    gated on a positive budget, never ran and a rung that could not fit was found out
    by the OOM rather than by the plan.
    """
    mps = getattr(torch, "mps", None)
    if mps is None:  # pragma: no cover - only on a torch built without MPS
        return []

    ceiling = 0
    reader = getattr(mps, "recommended_max_memory", None)
    if reader is not None:
        try:
            ceiling = int(reader())
        except Exception as exc:  # pragma: no cover - depends on the driver
            warnings.append(f"Could not read the Apple GPU memory ceiling: {exc}")
    if ceiling <= 0:
        # Without a ceiling there is no honest budget to state, and inventing one from
        # total RAM would be a guess presented as a driver figure. Stay at "no opinion",
        # which is what the planner did on every Apple machine until now.
        warnings.append(
            "This PyTorch does not report an Apple GPU memory ceiling, so the plan "
            "cannot say whether a model fits before it runs. It is still chosen by "
            "measurement: a size that does not survive a real step is not offered."
        )
        return []

    held = 0
    for name in ("driver_allocated_memory", "current_allocated_memory"):
        probe = getattr(mps, name, None)
        if probe is None:
            continue
        try:
            held = int(probe())
            break
        except Exception:  # pragma: no cover - depends on the driver
            continue

    free = max(0, ceiling - held)
    if available_ram_bytes > 0:
        free = min(free, available_ram_bytes)

    return [
        GPUInfo(
            index=0,
            name="Apple Silicon (unified memory)",
            total_vram_bytes=ceiling,
            free_vram_bytes=free,
            backend="mps",
        )
    ]


def _probe_xpu_devices(torch: Any, warnings: list[str]) -> list[GPUInfo]:
    """Intel GPUs via ``torch.xpu``, absent from PyTorch builds without XPU support."""
    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        return []
    try:
        if not xpu.is_available():
            return []
        count = int(xpu.device_count())
    except Exception as exc:  # pragma: no cover - depends on the build
        warnings.append(f"Intel XPU availability check failed: {exc}")
        return []

    gpus: list[GPUInfo] = []
    for index in range(count):
        total = 0
        name = f"xpu:{index}"
        try:
            props = xpu.get_device_properties(index)
            total = int(getattr(props, "total_memory", 0))
            name = str(getattr(props, "name", name))
        except Exception as exc:  # pragma: no cover
            warnings.append(f"Could not read properties of XPU device {index}: {exc}")
        free = total
        try:
            free_bytes, total_bytes = xpu.mem_get_info(index)
            free, total = int(free_bytes), int(total_bytes)
        except Exception:
            warnings.append(
                f"Could not read free memory for XPU device {index}; planning will "
                "fall back to total memory, which may overcommit."
            )
        gpus.append(
            GPUInfo(
                index=index,
                name=name,
                total_vram_bytes=total,
                free_vram_bytes=free,
                backend="xpu",
            )
        )
    return gpus


def _probe_xpu_bf16(torch: Any) -> bool:
    xpu = getattr(torch, "xpu", None)
    checker = getattr(xpu, "is_bf16_supported", None) if xpu is not None else None
    if checker is None:
        return False
    try:
        return bool(checker())
    except Exception:  # pragma: no cover
        return False


def _probe_cuda_devices(torch: Any, warnings: list[str], *, backend: str = "cuda") -> list[GPUInfo]:
    gpus: list[GPUInfo] = []
    try:
        count = torch.cuda.device_count()
    except Exception as exc:  # pragma: no cover
        warnings.append(f"Could not count CUDA devices: {exc}")
        return gpus

    for index in range(count):
        try:
            props = torch.cuda.get_device_properties(index)
        except Exception as exc:  # pragma: no cover
            warnings.append(f"Could not read properties of CUDA device {index}: {exc}")
            continue

        total = int(getattr(props, "total_memory", 0))
        # mem_get_info reflects what the driver will actually give us right now,
        # including VRAM taken by the desktop and other applications. It needs a
        # CUDA context, which is why we accept its ~200-300 MiB cost here: the
        # planner would otherwise plan against a number that does not exist.
        free = total
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            free, total = int(free_bytes), int(total_bytes)
        except Exception as exc:
            warnings.append(
                f"Could not read free VRAM for device {index} ({exc}); "
                "planning will fall back to total VRAM, which may overcommit."
            )

        capability: tuple[int, int] | None = None
        major, minor = getattr(props, "major", None), getattr(props, "minor", None)
        if major is not None and minor is not None:
            capability = (int(major), int(minor))

        # Per-device bf16. On CUDA, capability 8.0+ is a correct rule. On ROCm the
        # same two fields carry the gfx architecture, where no such rule exists:
        # gfx906 reports (9, 0) and has no bf16, gfx1030 reports (10, 3) and has
        # none either, while gfx908 also reports (9, 0) and does. Applying the
        # CUDA rule there would tell an RX 6900 XT owner they have bf16 and then
        # autocast into a format their card lacks. So on ROCm the per-device flag
        # is left to the runtime-wide answer from _probe_bf16 instead of guessed
        # from a number that does not mean what it looks like.
        device_bf16 = False
        if backend == "cuda" and capability is not None:
            device_bf16 = capability >= _NVIDIA_BF16_CAPABILITY
        elif backend == "rocm":
            try:
                device_bf16 = bool(torch.cuda.is_bf16_supported())
            except Exception:  # pragma: no cover - driver-level failure
                device_bf16 = False

        gpus.append(
            GPUInfo(
                index=index,
                name=str(getattr(props, "name", f"cuda:{index}")),
                total_vram_bytes=total,
                free_vram_bytes=free,
                compute_capability=capability,
                multiprocessor_count=getattr(props, "multi_processor_count", None),
                supports_bf16=device_bf16,
                backend=backend,
            )
        )
    return gpus


# --------------------------------------------------------------------------- #
# System-level GPU detection
#
# Answers "is there a GPU in this machine at all", independently of whether the
# installed PyTorch can use it. That distinction is the whole point: "no GPU
# found" and "you installed the CPU-only wheel" look identical from inside torch
# and have completely different fixes.
# --------------------------------------------------------------------------- #
#: How each vendor slug is spelled for a reader, and the article it takes. The
#: slugs themselves are identifiers -- lowercase, stable, and part of the ``--json``
#: payload -- so they are not renamed; this is only the spelling used in prose.
#: Every article here is "an" because the letters are read aloud ("an NVIDIA GPU",
#: "an AMD GPU"). That is data rather than a rule because the set is closed, and a
#: rule about vowel sounds is a thing to get wrong later.
_VENDOR_PROSE = {
    "nvidia": ("NVIDIA", "an"),
    "amd": ("AMD", "an"),
    "intel": ("Intel", "an"),
    "apple": ("Apple", "an"),
}


def vendor_labels(vendors: Iterable[str]) -> list[str]:
    """The vendor slugs as they are spelled for a reader, in the order given."""
    return [_VENDOR_PROSE.get(vendor, (vendor, "a"))[0] for vendor in vendors]


def gpu_pronoun(vendors: Sequence[str]) -> str:
    """``"it"`` for one vendor's GPU, ``"them"`` for two vendors' GPUs."""
    return "it" if len(vendors) < 2 else "them"


def describe_unusable_gpu(vendors: Sequence[str], build: str | None = None) -> str:
    """The one sentence for "the hardware is here and PyTorch cannot see it".

    Written in one place because two commands print it -- ``doctor`` as a probe
    warning, ``setup`` as the situation -- and because it has to agree with itself.
    A laptop with switchable graphics reports two vendors, which is what turned an
    unlabelled ``"/".join`` into "A amd/nvidia GPU is present ... cannot use it".

    ``build`` names the installed PyTorch when the caller knows it, which ``setup``
    does and the probe does not.
    """
    labels = vendor_labels(vendors)
    if not labels:
        # No caller reaches this: both guard on a non-empty vendor list. It says
        # something true rather than "A unknown GPU" if one ever stops.
        subject = "A GPU is present"
    elif len(labels) == 1:
        _, article = _VENDOR_PROSE.get(vendors[0], (labels[0], "a"))
        subject = f"{article.capitalize()} {labels[0]} GPU is present"
    else:
        subject = f"{', '.join(labels[:-1])} and {labels[-1]} GPUs are present"
    cannot = f"PyTorch ({build}) cannot" if build else "PyTorch cannot"
    return f"{subject} but {cannot} use {gpu_pronoun(vendors)}, so training would run on the CPU."


def detect_gpu_vendors() -> list[str]:
    """GPU vendors present according to the system, sorted and de-duplicated.

    Returns any of ``"nvidia"``, ``"amd"``, ``"intel"``, ``"apple"``. Never raises:
    every probe here is best-effort, and an empty list means "nothing detected",
    not "nothing present".

    Deliberately cheap. Vendor tools on PATH and Linux sysfs answer instantly;
    nothing here shells out to a slow enumeration like PowerShell's CIM provider,
    because this runs on the way to `trainai doctor` finishing in a few seconds.
    """
    found: set[str] = set()

    # Vendor management tools are the strongest signal: their presence means a
    # driver is installed, which is what actually matters.
    if shutil.which("nvidia-smi"):
        found.add("nvidia")
    if shutil.which("rocm-smi") or shutil.which("rocminfo"):
        found.add("amd")
    if shutil.which("xpu-smi"):
        found.add("intel")

    system = platform.system()
    if system == "Darwin" and platform.machine() in ("arm64", "aarch64"):
        # Every Apple Silicon Mac has an integrated GPU that Metal can use.
        found.add("apple")
    elif system == "Linux":
        found.update(_linux_pci_vendors())
    elif system == "Windows":
        found.update(_windows_registry_vendors())

    return sorted(found)


def _linux_pci_vendors() -> set[str]:
    """Read PCI vendor ids straight out of sysfs. No subprocess, no parsing.

    ``encoding="ascii"`` because the file holds one hex id (``0x10de\\n``) and nothing
    else, and because leaving it off would have decoded with whatever the locale
    happens to be. The matching ``UnicodeDecodeError`` is caught next to ``OSError``
    for a reason that is not cosmetic: it is not an ``OSError``, so it used to escape
    to the outer handler and end the loop, and a driver-less card enumerated before a
    real one would take the real one's vendor down with it. Per-card failures have to
    stay per-card.
    """
    found: set[str] = set()
    try:
        for card in sorted(Path(_DRM_CLASS_PATH).glob("card[0-9]*")):
            vendor_file = card / "device" / "vendor"
            try:
                vendor = vendor_file.read_text(encoding="ascii").strip().lower()
            except (OSError, UnicodeDecodeError):
                continue
            name = _PCI_VENDORS.get(vendor)
            if name:
                found.add(name)
    except Exception:  # pragma: no cover - sysfs layout varies
        pass
    return found


def _windows_registry_vendors() -> set[str]:
    """Read display adapters from the registry.

    The registry rather than WMI or PowerShell: ``wmic`` is removed from recent
    Windows builds, and ``Get-CimInstance`` costs about a second of process
    startup. This is a handful of key reads.
    """
    found: set[str] = set()
    try:
        import winreg

        key_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as root:
            index = 0
            while True:
                try:
                    subkey_name = winreg.EnumKey(root, index)
                except OSError:
                    break
                index += 1
                if not subkey_name.isdigit():
                    continue
                try:
                    with winreg.OpenKey(root, subkey_name) as subkey:
                        description = str(winreg.QueryValueEx(subkey, "DriverDesc")[0]).lower()
                except OSError:
                    continue
                if "nvidia" in description or "geforce" in description:
                    found.add("nvidia")
                elif "amd" in description or "radeon" in description:
                    found.add("amd")
                elif "intel" in description:
                    found.add("intel")
    except Exception:  # pragma: no cover - registry layout varies
        pass
    return found


def _read_cgroup_value(*parts: str) -> str | None:
    """The stripped contents of one cgroup file, or ``None`` if it is unusable.

    ``encoding="ascii"`` because these files hold decimal integers, the word ``max``, and
    nothing else. ``ValueError`` is in the handler and ``UnicodeDecodeError`` is not,
    because the second is a subclass of the first -- naming both would read as though the
    tuple needed widening when it does not, and this is exactly the relationship that let
    an undecodable byte escape ``except OSError`` in the sysfs card walk. Every failure
    here is ordinary: on Windows and macOS the path does not exist at all.
    """
    try:
        return Path(_CGROUP_ROOT, *parts).read_text(encoding="ascii").strip() or None
    except (OSError, ValueError):
        return None


def _cgroup_cpu_quota() -> float | None:
    """Cores this cgroup may use, or ``None`` when it is not capped.

    Both cgroup versions, because both are still in the field: v2 states it as
    ``"<quota> <period>"`` in ``cpu.max`` (or the literal ``max``), v1 as two files
    holding microseconds, with a quota of ``-1`` meaning uncapped.
    """
    raw = _read_cgroup_value("cpu.max")
    if raw is not None:
        fields = raw.split()
        if fields and fields[0] != "max":
            with suppress(ValueError, IndexError, ZeroDivisionError):
                quota, period = int(fields[0]), int(fields[1]) if len(fields) > 1 else 100_000
                if quota > 0 and period > 0:
                    return quota / period

    quota_raw = _read_cgroup_value("cpu", "cpu.cfs_quota_us")
    period_raw = _read_cgroup_value("cpu", "cpu.cfs_period_us")
    if quota_raw is not None and period_raw is not None:
        with suppress(ValueError, ZeroDivisionError):
            quota, period = int(quota_raw), int(period_raw)
            if quota > 0 and period > 0:
                return quota / period
    return None


def _cgroup_memory_limit() -> tuple[int | None, int | None]:
    """``(limit_bytes, used_bytes)`` for this cgroup, either or both ``None``.

    v2 spells them ``memory.max`` / ``memory.current``, v1 ``memory.limit_in_bytes`` /
    ``memory.usage_in_bytes``. An uncapped v1 limit is a sentinel near 8 EiB rather than
    an absence, so it is filtered here -- see ``_CGROUP_UNLIMITED_AT_OR_ABOVE``.
    """
    for limit_file, used_file in (
        (("memory.max",), ("memory.current",)),
        (("memory", "memory.limit_in_bytes"), ("memory", "memory.usage_in_bytes")),
    ):
        raw = _read_cgroup_value(*limit_file)
        if raw is None or raw == "max":
            continue
        try:
            limit = int(raw)
        except ValueError:
            continue
        if limit <= 0 or limit >= _CGROUP_UNLIMITED_AT_OR_ABOVE:
            continue
        used: int | None = None
        used_raw = _read_cgroup_value(*used_file)
        if used_raw is not None:
            with suppress(ValueError):
                used = max(0, int(used_raw))
        return limit, used
    return None, None


def _cpu_affinity_count() -> int | None:
    """How many cores this process is *scheduled* on, or ``None`` if that is unknowable.

    Its own function so a test can control it. It has to be controllable: it reads the
    real machine, ``os.sched_getaffinity`` does not exist on Windows, and a CI runner's
    affinity is whatever the runner happens to have -- a fake cgroup filesystem alone
    does not isolate the quota branch from it. GitHub's two-core Linux runners proved
    that by failing a quota test that had nothing to do with affinity.
    """
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is None:
        return None
    with suppress(OSError, ValueError):
        allowed = len(affinity(0))
        if allowed > 0:
            return allowed
    return None


def _probe_cpu_limit(logical: int) -> float | None:
    """Cores available to *this process*, when that is fewer than the machine has.

    Two independent narrowings, and the tighter wins. A cgroup CPU quota is what a
    container runtime writes to ``cpu.max`` when it is given a core count; a CPU
    affinity mask is what ``taskset`` and most HPC schedulers set. Neither is visible
    to ``os.cpu_count()`` or to ``psutil.cpu_count()``, both of which report the
    machine: psutil's Linux backend reads ``SC_NPROCESSORS_ONLN`` and ``/proc/cpuinfo``
    and contains no cgroup handling at all. So a four-core container on a
    sixty-four-core host used to have ``trainai doctor`` tell it that it had
    sixty-four, and that number went into ``events.jsonl`` as part of the run's record
    of what it ran on.

    ``None`` when nothing narrows it, so an unconstrained machine carries no field
    claiming a limit it does not have.

    One case is deliberately not handled: a process in a *nested* cgroup on a host reads
    the root of the mount, not its own subtree, and so reports unconstrained. Finding the
    real subtree means parsing ``/proc/self/cgroup`` and resolving it against the mount,
    which is where this stops being a measurement and starts being an inference. Failing
    towards "no limit" is the safe direction: it never invents a cap that is not there.
    """
    candidates: list[float] = []

    quota = _cgroup_cpu_quota()
    if quota is not None:
        candidates.append(quota)

    allowed = _cpu_affinity_count()
    if allowed is not None:
        candidates.append(float(allowed))

    if not candidates:
        return None
    limit = min(candidates)
    return limit if limit < logical else None


def _probe_cpu() -> tuple[str | None, int, int | None]:
    logical = os.cpu_count() or 1
    physical: int | None = None
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True) or logical
    except Exception:
        # psutil is a declared dependency, but never let its absence be fatal.
        pass

    return _probe_cpu_name(), logical, physical


def _probe_cpu_name() -> str | None:
    """Best-effort human-readable CPU model name.

    ``platform.processor()`` is useless on Windows -- it returns strings like
    ``"AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD"``. The readable name
    lives in the registry, so we read it there and fall back progressively.
    """
    system = platform.system()

    if system == "Windows":  # pragma: no cover - platform specific
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if name:
                    return " ".join(str(name).split())
        except (OSError, ImportError, ValueError):
            pass

    elif system == "Linux":  # pragma: no cover - platform specific
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass

    elif system == "Darwin":  # pragma: no cover - platform specific
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                # Stated, not left to the locale: a strict decode raises
                # UnicodeDecodeError, which is a ValueError and so escapes the
                # handler below. See detect_driver_cuda_version for the full note.
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=False,
            )
            if result.returncode == 0 and (result.stdout or "").strip():
                return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass

    name = platform.processor() or platform.machine()
    return name.strip() or None


def _probe_ram_limit(total_ram: int) -> tuple[int | None, int | None]:
    """``(limit_bytes, used_bytes)`` when a container caps memory below the host's RAM.

    ``psutil.virtual_memory()`` reads ``/proc/meminfo``, which is the *host's* -- its
    Linux backend contains no cgroup handling whatsoever. So an 8 GiB container on a
    512 GiB host used to be told it had 512 GiB, and a user who believed it planned a run
    the kernel then OOM-killed with no warning from ``doctor`` that the number was not
    theirs. Nothing in TrainAI sizes a run from RAM today, but the figure is printed by
    ``doctor`` and recorded in ``events.jsonl`` as part of what the run ran on, and a
    reproducibility record that names the wrong machine is worse than one that says
    "unknown".

    A limit at or above the host's RAM is not a constraint and is reported as ``None``,
    so the extra fields appear only when they mean something. The nested-cgroup caveat in
    :func:`_probe_cpu_limit` applies here too.
    """
    limit, used = _cgroup_memory_limit()
    if limit is None:
        return None, None
    if total_ram and limit >= total_ram:
        return None, None
    return limit, used


def _probe_ram(warnings: list[str]) -> tuple[int, int]:
    """Return ``(total_bytes, available_bytes)``, or ``(0, 0)`` if unknowable."""
    try:
        import psutil

        memory = psutil.virtual_memory()
        return int(memory.total), int(memory.available)
    except Exception:
        pass

    # Stdlib fallbacks, so `trainai doctor` still works without psutil.
    if platform.system() == "Windows":  # pragma: no cover - platform specific
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys), int(status.ullAvailPhys)
        except Exception:
            pass
    else:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            total = os.sysconf("SC_PHYS_PAGES") * page_size
            available = total
            if hasattr(os, "sysconf_names") and "SC_AVPHYS_PAGES" in os.sysconf_names:
                available = os.sysconf("SC_AVPHYS_PAGES") * page_size
            return int(total), int(available)
        # ``AttributeError`` because ``os.sysconf`` does not exist on every platform,
        # only on the POSIX-ish ones. The line below already guards ``sysconf_names``
        # with ``hasattr`` for that reason, and guarding the second call while calling
        # the first bare is not a position anything can defend: measured, with psutil
        # absent and this branch taken, the missing name ends the whole run with
        # ``AttributeError: module 'os' has no attribute 'sysconf'`` -- out of here,
        # out of ``probe_hardware``, and out of whichever command asked. RAM being
        # unknowable is a warning, which is what the two lines below are for.
        except (AttributeError, OSError, ValueError):
            pass

    warnings.append("System RAM could not be determined; RAM-based checks are disabled.")
    return 0, 0


def _probe_disk(path: str | os.PathLike[str] | None, warnings: list[str]) -> int:
    target = os.fspath(path) if path is not None else os.getcwd()
    # Walk up until we find something that exists -- the output directory is
    # often the thing we are about to create.
    probe_target = os.path.abspath(target)
    while probe_target and not os.path.exists(probe_target):
        parent = os.path.dirname(probe_target)
        if parent == probe_target:
            break
        probe_target = parent
    try:
        return int(shutil.disk_usage(probe_target).free)
    except OSError as exc:
        # The path plainly, not ``!r``. On the platform this is most often read on,
        # repr doubles every separator -- ``'C:\\Users\\me\\runs'`` -- and a user
        # checking whether TrainAI was given the right directory should not have to
        # decide whether the extra backslashes are in the path or in the printing.
        # Every other path this project puts in front of someone is bare.
        warnings.append(f"Free disk space for {target} could not be determined ({exc}).")
        return 0


def _probe_torch_compile() -> bool:
    """Whether ``torch.compile`` can realistically be used on this machine.

    Inductor needs Triton. On Windows, Triton is usually not installed, so
    ``torch.compile`` fails at first call rather than at import. We check for the
    module up front and keep compilation opt-in.
    """
    try:
        return importlib.util.find_spec("triton") is not None
    except (ImportError, ValueError):  # pragma: no cover
        return False


def _platform_summary() -> str:
    system = platform.system()
    if system == "Windows":
        release, version, _, _ = platform.win32_ver()
        return f"Windows {release} ({version})"
    if system == "Darwin":  # pragma: no cover - platform specific
        return f"macOS {platform.mac_ver()[0]} ({platform.machine()})"
    return f"{system} {platform.release()} ({platform.machine()})".strip()
