"""Hardware portability: tested against machines this project does not have.

TrainAI is developed on one GPU. "The code path is generic" and "I have proven it
works on your card" are different claims, and only the first can be earned here. So
every hardware-dependent decision is a pure function of stated facts, and this
module feeds it the facts of real machines nobody in this repository owns: a 24 GiB
4090, a bf16-less GTX 1080, an RX 6900 XT on ROCm, an Intel Arc, an Apple M2, a
box with two GPUs, and a machine with no accelerator at all.

What that does and does not prove:

* It **does** prove the *logic* is right for those machines -- that a 1080 gets fp16
  with a gradient scaler rather than bf16, that an RX 6900 XT is not told it has
  bf16, that a 24 GiB card's budget is computed from free VRAM rather than total.
* It does **not** prove any of them runs. Kernels, driver quirks and throughput are
  beyond reach without the hardware, and the README says so.

The RX 6900 XT case is the reason this file exists. On ROCm, PyTorch reports the
gfx architecture through the same ``major``/``minor`` fields that carry CUDA compute
capability, so ``capability >= (8, 0)`` -- correct on NVIDIA -- is True for gfx1030,
which has no bf16 matrix support. Autocasting into a format the card lacks is a
silent-wrong-answer bug, and no amount of testing on an NVIDIA laptop would find it.
"""

from __future__ import annotations

import os
import platform
import shutil
import stat
import sys
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from conftest import flat
from trainai.errors import ExitCode, TrainingError, UsageError
from trainai.hardware.install import (
    PYTORCH_SELECTOR,
    advise_install,
    detect_driver_cuda_version,
    in_virtualenv,
)
from trainai.hardware.probe import (
    DEFAULT_VRAM_SAFETY_FRACTION,
    GPUInfo,
    HardwareProfile,
    _cgroup_cpu_quota,
    _cgroup_memory_limit,
    _cpu_affinity_count,
    _linux_pci_vendors,
    _platform_summary,
    _probe_cpu,
    _probe_cpu_limit,
    _probe_cpu_name,
    _probe_disk,
    _probe_ram,
    _probe_ram_limit,
    _windows_registry_vendors,
    describe_unusable_gpu,
    detect_gpu_vendors,
    probe_hardware,
    vendor_labels,
)
from trainai.train.config import PRECISION_CHOICES
from trainai.train.loop import precision_for

GIB = 1024**3


def gpu(
    name: str,
    *,
    total_gib: float,
    free_gib: float | None = None,
    capability: tuple[int, int] | None = None,
    bf16: bool = False,
    backend: str = "cuda",
    index: int = 0,
    sms: int | None = None,
) -> GPUInfo:
    return GPUInfo(
        index=index,
        name=name,
        total_vram_bytes=int(total_gib * GIB),
        free_vram_bytes=int((free_gib if free_gib is not None else total_gib * 0.92) * GIB),
        compute_capability=capability,
        multiprocessor_count=sms,
        supports_bf16=bf16,
        backend=backend,
    )


def machine(
    label: str,
    *,
    gpus: list[GPUInfo],
    backend: str,
    os_name: str = "Linux",
    bf16: bool = False,
    tf32: bool = False,
    vendors: list[str] | None = None,
    ram_gib: float = 32.0,
) -> HardwareProfile:
    return HardwareProfile(
        platform_summary=label,
        os_name=os_name,
        python_version="3.12.0",
        torch_version="2.6.0",
        cuda_version="12.6" if backend == "cuda" else ("ROCm 6.2" if backend == "rocm" else None),
        device_type={"rocm": "cuda"}.get(backend, backend),
        backend=backend,
        detected_vendors=vendors if vendors is not None else [],
        gpus=gpus,
        cpu_name="Synthetic CPU",
        cpu_count_logical=16,
        cpu_count_physical=8,
        total_ram_bytes=int(ram_gib * GIB),
        available_ram_bytes=int(ram_gib * 0.6 * GIB),
        free_disk_bytes=200 * GIB,
        silent_vram_spillover=(os_name == "Windows" and backend in ("cuda", "rocm")),
        supports_bf16=bf16,
        supports_tf32=tf32,
        torch_compile_available=(os_name == "Linux"),
    )


# --------------------------------------------------------------------------- #
# The machines. Names and capabilities are real; the numbers are plausible.
# --------------------------------------------------------------------------- #
def rtx_4090() -> HardwareProfile:
    return machine(
        "Linux-x86_64 / RTX 4090",
        gpus=[gpu("NVIDIA GeForce RTX 4090", total_gib=24, capability=(8, 9), bf16=True, sms=128)],
        backend="cuda",
        bf16=True,
        tf32=True,
        vendors=["nvidia"],
    )


def rtx_3070_windows() -> HardwareProfile:
    return machine(
        "Windows-11 / RTX 3070",
        gpus=[
            gpu(
                "NVIDIA GeForce RTX 3070",
                total_gib=8,
                free_gib=6.9,
                capability=(8, 6),
                bf16=True,
                sms=46,
            )
        ],
        backend="cuda",
        os_name="Windows",
        bf16=True,
        tf32=True,
        vendors=["nvidia"],
    )


def rtx_2050_windows() -> HardwareProfile:
    """The development machine, as a fixture, so its behaviour is pinned too."""
    return machine(
        "Windows-11 / RTX 2050",
        gpus=[
            gpu(
                "NVIDIA GeForce RTX 2050",
                total_gib=4,
                free_gib=3.23,
                capability=(8, 6),
                bf16=True,
                sms=16,
            )
        ],
        backend="cuda",
        os_name="Windows",
        bf16=True,
        tf32=True,
        vendors=["nvidia"],
        ram_gib=16,
    )


def gtx_1080_ti() -> HardwareProfile:
    """Pascal: no bf16, no TF32. Must get fp16 with a gradient scaler."""
    return machine(
        "Linux-x86_64 / GTX 1080 Ti",
        gpus=[
            gpu("NVIDIA GeForce GTX 1080 Ti", total_gib=11, capability=(6, 1), bf16=False, sms=28)
        ],
        backend="cuda",
        bf16=False,
        tf32=False,
        vendors=["nvidia"],
    )


def tesla_v100() -> HardwareProfile:
    """Volta: tensor cores, but fp16 only."""
    return machine(
        "Linux-x86_64 / Tesla V100",
        gpus=[gpu("Tesla V100-SXM2-16GB", total_gib=16, capability=(7, 0), bf16=False, sms=80)],
        backend="cuda",
        bf16=False,
        tf32=False,
        vendors=["nvidia"],
    )


def rx_7900_xtx() -> HardwareProfile:
    """RDNA3 on ROCm. The runtime reports bf16; the gfx number is 11.0."""
    return machine(
        "Linux-x86_64 / RX 7900 XTX",
        gpus=[
            gpu(
                "AMD Radeon RX 7900 XTX",
                total_gib=24,
                capability=(11, 0),
                bf16=True,
                backend="rocm",
            )
        ],
        backend="rocm",
        bf16=True,
        vendors=["amd"],
    )


def rx_6900_xt() -> HardwareProfile:
    """RDNA2 on ROCm: gfx1030 reports (10, 3) and has no bf16. The trap case."""
    return machine(
        "Linux-x86_64 / RX 6900 XT",
        gpus=[
            gpu(
                "AMD Radeon RX 6900 XT",
                total_gib=16,
                capability=(10, 3),
                bf16=False,
                backend="rocm",
            )
        ],
        backend="rocm",
        bf16=False,
        vendors=["amd"],
    )


def mi210() -> HardwareProfile:
    """CDNA2: gfx90a reports (9, 0) and does have bf16."""
    return machine(
        "Linux-x86_64 / MI210",
        gpus=[
            gpu("AMD Instinct MI210", total_gib=64, capability=(9, 0), bf16=True, backend="rocm")
        ],
        backend="rocm",
        bf16=True,
        vendors=["amd"],
    )


def arc_a770() -> HardwareProfile:
    return machine(
        "Linux-x86_64 / Arc A770",
        gpus=[gpu("Intel(R) Arc(TM) A770 Graphics", total_gib=16, backend="xpu")],
        backend="xpu",
        bf16=True,
        vendors=["intel"],
    )


def apple_m2() -> HardwareProfile:
    """A 24 GiB M2, with the memory figures the probe actually derives on Apple.

    Not a card's VRAM. The total is Metal's declared working-set ceiling, which is
    roughly 75% of installed RAM, and the usable figure is that ceiling capped by what
    the OS says is free -- here 14.4 GiB of RAM available against an 18 GiB ceiling, so
    RAM is the binding constraint and the lower number is the honest one.

    This fixture used to be ``gpus=[]`` "by design", which is what the probe returned
    before it read ``torch.mps.recommended_max_memory()``. The consequence was that
    ``primary_gpu`` was ``None`` on every Apple machine, ``vram_budget_bytes()`` returned
    0, and the planner's over-budget check -- gated on a positive budget -- never ran.
    """
    return machine(
        "macOS-14 / Apple M2",
        gpus=[
            gpu(
                "Apple Silicon (unified memory)",
                total_gib=18.0,
                free_gib=24.0 * 0.6,
                backend="mps",
            )
        ],
        backend="mps",
        os_name="Darwin",
        vendors=["apple"],
        ram_gib=24,
    )


def dual_4090() -> HardwareProfile:
    return machine(
        "Linux-x86_64 / 2x RTX 4090",
        gpus=[
            gpu("NVIDIA GeForce RTX 4090", total_gib=24, capability=(8, 9), bf16=True, index=0),
            gpu("NVIDIA GeForce RTX 4090", total_gib=24, capability=(8, 9), bf16=True, index=1),
        ],
        backend="cuda",
        bf16=True,
        tf32=True,
        vendors=["nvidia"],
    )


def cpu_only() -> HardwareProfile:
    return machine("Linux-x86_64 / no GPU", gpus=[], backend="cpu", vendors=[])


def nvidia_with_cpu_wheel() -> HardwareProfile:
    """The single most common broken setup: good card, CPU-only PyTorch."""
    return machine(
        "Windows-11 / RTX 3060 with CPU-only torch",
        gpus=[],
        backend="cpu",
        os_name="Windows",
        vendors=["nvidia"],
    )


ALL_MACHINES = {
    "rtx_4090": rtx_4090,
    "rtx_3070_windows": rtx_3070_windows,
    "rtx_2050_windows": rtx_2050_windows,
    "gtx_1080_ti": gtx_1080_ti,
    "tesla_v100": tesla_v100,
    "rx_7900_xtx": rx_7900_xtx,
    "rx_6900_xt": rx_6900_xt,
    "mi210": mi210,
    "arc_a770": arc_a770,
    "apple_m2": apple_m2,
    "dual_4090": dual_4090,
    "cpu_only": cpu_only,
    "nvidia_with_cpu_wheel": nvidia_with_cpu_wheel,
}


# --------------------------------------------------------------------------- #
# Precision: the decision that goes silently wrong
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("label", "expected_dtype", "expected_scaler"),
    [
        ("rtx_4090", torch.bfloat16, False),
        ("rtx_3070_windows", torch.bfloat16, False),
        ("rtx_2050_windows", torch.bfloat16, False),
        ("gtx_1080_ti", torch.float16, True),
        ("tesla_v100", torch.float16, True),
        ("rx_7900_xtx", torch.bfloat16, False),
        ("rx_6900_xt", torch.float16, True),
        ("mi210", torch.bfloat16, False),
        ("arc_a770", torch.bfloat16, False),
        ("apple_m2", torch.float32, False),
        ("cpu_only", torch.float32, False),
    ],
)
def test_auto_precision_matches_the_hardware(
    label: str, expected_dtype: torch.dtype, expected_scaler: bool
) -> None:
    profile = ALL_MACHINES[label]()
    device = profile.primary_gpu

    dtype, needs_scaler, note = precision_for(
        "auto",
        backend=profile.backend,
        supports_bf16=profile.supports_bf16,
        capability=device.compute_capability if device else None,
    )

    assert dtype == expected_dtype, note
    assert needs_scaler == expected_scaler, note
    assert note


def test_an_rdna2_card_is_not_told_it_has_bf16() -> None:
    """The bug this file was written for, stated on its own.

    gfx1030 reports ``(10, 3)`` through the same fields that carry CUDA compute
    capability. The NVIDIA rule ``capability >= (8, 0)`` is True for it, and wrong.
    """
    profile = rx_6900_xt()
    capability = profile.primary_gpu.compute_capability
    assert capability is not None
    assert capability >= (8, 0), "the fixture must reproduce the misleading number"

    dtype, needs_scaler, note = precision_for(
        "auto",
        backend="rocm",
        supports_bf16=profile.supports_bf16,
        capability=capability,
    )

    assert dtype == torch.float16
    assert needs_scaler
    assert "8.0" not in note, "the NVIDIA capability rule must not appear in a ROCm message"


def test_a_cdna_card_that_looks_older_still_gets_bf16() -> None:
    """gfx90a reports (9, 0) -- lower than gfx1030's (10, 3) -- and does have bf16."""
    dtype, needs_scaler, _ = precision_for(
        "auto", backend="rocm", supports_bf16=True, capability=(9, 0)
    )

    assert dtype == torch.bfloat16
    assert not needs_scaler


def test_tf32_is_never_claimed_on_non_nvidia_hardware() -> None:
    """TF32 is an NVIDIA tensor-core format. There is no AMD or Intel equivalent."""
    for label in ("rx_7900_xtx", "rx_6900_xt", "mi210", "arc_a770", "apple_m2", "cpu_only"):
        assert not ALL_MACHINES[label]().supports_tf32, label


@pytest.mark.parametrize("label", list(ALL_MACHINES))
def test_explicit_fp32_is_honoured_everywhere(label: str) -> None:
    profile = ALL_MACHINES[label]()

    dtype, needs_scaler, _ = precision_for(
        "fp32", backend=profile.backend, supports_bf16=profile.supports_bf16
    )

    assert dtype == torch.float32
    assert not needs_scaler


@pytest.mark.parametrize("label", ["gtx_1080_ti", "tesla_v100", "rx_6900_xt"])
def test_requesting_bf16_on_hardware_without_it_is_refused(label: str) -> None:
    """Refused rather than silently downgraded: the user asked for a specific thing."""
    profile = ALL_MACHINES[label]()

    with pytest.raises(TrainingError) as caught:
        precision_for(
            "bf16",
            backend=profile.backend,
            supports_bf16=False,
            capability=profile.primary_gpu.compute_capability,
        )

    assert "--precision fp16" in (caught.value.hint or "")


@pytest.mark.parametrize("label", ["apple_m2", "cpu_only"])
def test_half_precision_on_a_backend_without_it_falls_back_and_says_why(label: str) -> None:
    profile = ALL_MACHINES[label]()

    dtype, _, note = precision_for("bf16", backend=profile.backend, supports_bf16=False)

    assert dtype == torch.float32
    assert "CUDA or ROCm" in note


@pytest.mark.parametrize("label", ["rtx_4090", "rx_7900_xtx", "mi210", "arc_a770"])
def test_requesting_bf16_where_it_exists_is_granted_without_a_scaler(label: str) -> None:
    """The control for the refusal above, on one card per accelerator backend.

    bf16 has the dynamic range of fp32, so the scaler that fp16 needs would be pure
    overhead here -- and the note has to say "requested" rather than name the hardware,
    because the user is being told their ask was honoured, not what was inferred.
    """
    profile = ALL_MACHINES[label]()

    dtype, needs_scaler, note = precision_for(
        "bf16",
        backend=profile.backend,
        supports_bf16=True,
        capability=profile.primary_gpu.compute_capability,
    )

    assert dtype == torch.bfloat16
    assert needs_scaler is False
    assert note == "bf16 (requested)"


@pytest.mark.parametrize("label", ["rtx_4090", "gtx_1080_ti", "rx_6900_xt", "arc_a770"])
def test_requesting_fp16_is_granted_on_every_accelerator_and_always_paired_with_a_scaler(
    label: str,
) -> None:
    """fp16 is never refused -- every accelerator here has it -- and never unscaled.

    The pairing is the point. fp16's smallest normal value is about 6e-5, and gradients
    live below that, so an unscaled fp16 run trains on zeros and reports a loss that
    barely moves. A card with bf16 is included deliberately: asking for fp16 on a 4090
    still gets the scaler, because the ask is honoured rather than second-guessed.
    """
    profile = ALL_MACHINES[label]()

    dtype, needs_scaler, note = precision_for(
        "fp16", backend=profile.backend, supports_bf16=profile.supports_bf16
    )

    assert dtype == torch.float16
    assert needs_scaler is True
    assert note == "fp16 with gradient scaling (requested)"


@pytest.mark.parametrize("label", list(ALL_MACHINES))
def test_a_precision_that_does_not_exist_is_refused_on_every_machine(label: str) -> None:
    """Every machine, because the fall-through this closes was per-backend.

    ``fp64`` matched no branch, so the CPU path returned "fp32 (on CPU)" and a bf16 card
    returned "bf16 (supported by this cuda device)" -- each of them a note describing
    ``auto``, printed for a user who asked for something else. The value is checked once,
    before any of that, so the answer does not depend on the hardware.
    """
    profile = ALL_MACHINES[label]()

    with pytest.raises(UsageError) as caught:
        precision_for("fp64", backend=profile.backend, supports_bf16=profile.supports_bf16)

    assert caught.value.exit_code == ExitCode.USAGE
    assert ", ".join(PRECISION_CHOICES) in (caught.value.hint or "")


def test_a_misspelt_precision_is_not_answered_with_the_one_auto_would_have_picked() -> None:
    """The shape of the bug, rather than one value of it.

    ``bf6`` is one character from ``bf16``, and on a card that supports bf16 the old
    fall-through gave it *exactly what bf16 would have given*: the same dtype, the same
    note, no error. A run could be asked for a precision that does not exist and report a
    plausible one, which is indistinguishable from having worked.
    """
    profile = ALL_MACHINES["rtx_4090"]()
    assert profile.supports_bf16, "this test needs a machine where auto and bf16 agree"

    with pytest.raises(UsageError) as caught:
        precision_for("bf6", backend=profile.backend, supports_bf16=True)

    assert "bf16" in (caught.value.hint or ""), "a near miss this close should be named"


# --------------------------------------------------------------------------- #
# VRAM budgeting
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label", [k for k, v in ALL_MACHINES.items() if v().has_gpu])
def test_the_budget_is_a_fraction_of_free_vram_not_total(label: str) -> None:
    """Planning against total VRAM overcommits by whatever the desktop is using."""
    profile = ALL_MACHINES[label]()
    device = profile.primary_gpu
    assert device is not None

    budget = profile.vram_budget_bytes()

    assert budget == int(device.free_vram_bytes * DEFAULT_VRAM_SAFETY_FRACTION)
    assert budget < device.total_vram_bytes


def test_a_bigger_card_gets_a_proportionally_bigger_budget() -> None:
    small = rtx_2050_windows().vram_budget_bytes()
    large = rtx_4090().vram_budget_bytes()

    assert large > small * 5


@pytest.mark.parametrize("label", ["cpu_only", "nvidia_with_cpu_wheel"])
def test_a_machine_with_no_usable_gpu_has_no_vram_budget(label: str) -> None:
    assert ALL_MACHINES[label]().vram_budget_bytes() == 0


def test_apple_gets_a_budget_from_the_metal_ceiling_rather_than_no_opinion() -> None:
    """``apple_m2`` used to be in the list above, and that was the bug.

    A budget of 0 is not "unlimited", it is "no opinion": the planner's over-budget
    check is gated on ``budget_bytes > 0``, so on Apple it never ran and an oversized
    rung was found out by the OOM instead of by the plan. The ceiling is a driver
    figure, so there is a real number to use here.
    """
    profile = apple_m2()
    device = profile.primary_gpu

    assert device is not None
    assert device.backend == "mps"
    assert profile.vram_budget_bytes() == int(device.free_vram_bytes * DEFAULT_VRAM_SAFETY_FRACTION)
    assert 0 < profile.vram_budget_bytes() < device.total_vram_bytes


def test_apples_usable_memory_is_capped_by_free_ram_not_just_the_ceiling() -> None:
    """The ceiling ignores every other process; unified memory means that matters.

    A 24 GiB M2 declares an 18 GiB working set, but if the OS has 14.4 GiB free then
    14.4 is the number a plan may spend. Reporting the ceiling would promise a browser's
    worth of memory that is not there.
    """
    profile = apple_m2()
    device = profile.primary_gpu

    assert device is not None
    assert device.free_vram_bytes == profile.available_ram_bytes
    assert device.free_vram_bytes < device.total_vram_bytes


def test_multi_gpu_budgets_from_the_first_card_only() -> None:
    """v0.1 trains on one GPU. Summing both cards' VRAM would promise twice the room."""
    profile = dual_4090()

    assert len(profile.gpus) == 2
    assert profile.vram_budget_bytes() == int(
        profile.gpus[0].free_vram_bytes * DEFAULT_VRAM_SAFETY_FRACTION
    )


# --------------------------------------------------------------------------- #
# Reporting the right thing for the right vendor
# --------------------------------------------------------------------------- #
def test_a_rocm_device_labels_its_architecture_rather_than_faking_a_capability() -> None:
    """ "gfx1030" is honest. "10.3" reads as a CUDA compute capability and is not one."""
    assert rx_6900_xt().primary_gpu.capability_str == "gfx103"
    assert rtx_4090().primary_gpu.capability_str == "8.9"


def test_a_device_with_no_capability_reports_unknown() -> None:
    assert arc_a770().primary_gpu.capability_str == "unknown"


def test_windows_gpus_are_flagged_for_silent_spillover() -> None:
    """WDDM oversubscription is a driver-model property, so it applies to AMD too.

    The flag only ever makes TrainAI more careful -- measure rather than trust the
    absence of an OOM -- so applying it where it has not been measured errs safely.
    """
    assert rtx_3070_windows().silent_vram_spillover
    assert rtx_2050_windows().silent_vram_spillover
    assert not rtx_4090().silent_vram_spillover  # same card, Linux
    assert not apple_m2().silent_vram_spillover


def test_a_gpu_that_torch_cannot_use_is_distinguished_from_having_none() -> None:
    """Same symptom -- CPU training -- entirely different fix."""
    broken = nvidia_with_cpu_wheel()
    genuinely_none = cpu_only()

    assert broken.has_unusable_gpu
    assert not broken.has_gpu
    assert not genuinely_none.has_unusable_gpu
    assert not genuinely_none.has_gpu


@pytest.mark.parametrize("label", list(ALL_MACHINES))
def test_every_profile_serialises_to_json(label: str) -> None:
    """Profiles go into run records that are read on other machines by other people."""
    import json

    payload = json.loads(json.dumps(ALL_MACHINES[label]().to_dict()))

    assert payload["backend"]
    assert payload["device_type"]
    assert isinstance(payload["gpus"], list)


@pytest.mark.parametrize("label", list(ALL_MACHINES))
def test_no_profile_makes_a_claim_it_cannot_support(label: str) -> None:
    """Cross-checks between fields that must not contradict each other."""
    profile = ALL_MACHINES[label]()

    if not profile.gpus:
        assert profile.vram_budget_bytes() == 0
        assert not profile.supports_tf32
    if profile.supports_tf32:
        assert profile.backend == "cuda", "TF32 is NVIDIA-only"
        assert profile.supports_bf16, "every TF32-capable NVIDIA GPU also has bf16"
    for device in profile.gpus:
        assert device.free_vram_bytes <= device.total_vram_bytes
        assert device.backend == ("cuda" if profile.backend == "rocm" else profile.backend) or (
            device.backend == profile.backend
        )


# --------------------------------------------------------------------------- #
# Install advice, for hardware and platforms not present here
# --------------------------------------------------------------------------- #
def patch_environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    vendors: list[str],
    system: str,
    torch_cuda: str | None,
    torch_hip: str | None = None,
    accelerator: bool,
    driver: tuple[int, int] | None = None,
) -> None:
    """Pretend to be another machine, as far as the install advisor can tell."""
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr("trainai.hardware.install.detect_driver_cuda_version", lambda: driver)
    monkeypatch.setattr(
        "trainai.hardware.install._torch_build",
        lambda: (
            ("ROCm " + torch_hip)
            if torch_hip
            else (("CUDA " + torch_cuda) if torch_cuda else "CPU-only"),
            accelerator,
        ),
    )
    monkeypatch.setattr("trainai.hardware.install.detect_gpu_vendors", lambda: vendors)


def test_nvidia_with_a_cpu_wheel_gets_a_cuda_index_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_environment(
        monkeypatch,
        vendors=["nvidia"],
        system="Windows",
        torch_cuda=None,
        accelerator=False,
        driver=(12, 6),
    )

    advice = advise_install(["nvidia"])

    assert advice.needs_action
    assert advice.command is not None
    assert "download.pytorch.org/whl/cu126" in advice.command
    assert "CPU-only build" in " ".join(advice.notes)


def test_the_channel_follows_the_driver_not_a_fixed_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An old driver must not be handed a wheel it cannot load."""
    for driver, expected in (((11, 8), "cu118"), ((12, 1), "cu121"), ((12, 8), "cu128")):
        patch_environment(
            monkeypatch,
            vendors=["nvidia"],
            system="Linux",
            torch_cuda=None,
            accelerator=False,
            driver=driver,
        )
        advice = advise_install(["nvidia"])
        assert advice.command is not None
        assert expected in advice.command, f"driver {driver}"


def test_a_driver_newer_than_the_table_gets_the_newest_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NVIDIA drivers run older CUDA runtimes, so this is correct, not a fallback."""
    patch_environment(
        monkeypatch,
        vendors=["nvidia"],
        system="Linux",
        torch_cuda=None,
        accelerator=False,
        driver=(13, 3),
    )

    advice = advise_install(["nvidia"])

    assert advice.command is not None
    assert "cu128" in advice.command
    assert PYTORCH_SELECTOR in " ".join(advice.notes)


def test_a_driver_older_than_every_channel_offers_no_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A command that 404s is worse than no command."""
    patch_environment(
        monkeypatch,
        vendors=["nvidia"],
        system="Linux",
        torch_cuda=None,
        accelerator=False,
        driver=(10, 2),
    )

    advice = advise_install(["nvidia"])

    assert advice.command is None
    assert "Update the driver" in " ".join(advice.notes)


def test_an_nvidia_card_with_no_driver_says_to_install_the_driver_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_environment(
        monkeypatch,
        vendors=["nvidia"],
        system="Linux",
        torch_cuda=None,
        accelerator=False,
        driver=None,
    )

    advice = advise_install(["nvidia"])

    assert "driver" in " ".join(advice.notes).lower()
    assert advice.command is None


def test_amd_on_linux_gets_a_rocm_index_url(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_environment(
        monkeypatch, vendors=["amd"], system="Linux", torch_cuda=None, accelerator=False
    )

    advice = advise_install(["amd"])

    assert advice.command is not None
    assert "rocm" in advice.command
    notes = " ".join(advice.notes)
    assert "ROCm" in notes
    # The published channel covers AMD's official matrix, which lists no consumer
    # RX 6000 card. Sending someone with one there and stopping is the failure mode:
    # the wheel installs, and every kernel launch raises. So the fallback index has
    # to be in the same note as the command it qualifies.
    assert "repo.amd.com" in notes


def test_amd_on_windows_is_told_the_truth_rather_than_given_a_broken_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Windows ROCm wheel exists now, but only AMD publishes one, and per gfx target.

    This assertion used to read ``"no ROCm build for Windows" in notes``, which was
    true when it was written and is not any more: AMD ships ``win_amd64`` wheels down
    to gfx1034. The command still has to stay ``None``, because the pip extra names
    the card's architecture and TrainAI cannot read that without the working PyTorch
    this advice exists to install.
    """
    patch_environment(
        monkeypatch, vendors=["amd"], system="Windows", torch_cuda=None, accelerator=False
    )

    advice = advise_install(["amd"])

    assert advice.command is None
    notes = " ".join(advice.notes)
    assert "no ROCm build for Windows" not in notes
    assert "repo.amd.com" in notes
    assert "gfxNNNN" in notes
    assert "WSL2" in notes


def test_intel_gets_an_xpu_index_url(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_environment(
        monkeypatch, vendors=["intel"], system="Linux", torch_cuda=None, accelerator=False
    )

    advice = advise_install(["intel"])

    assert advice.command is not None
    assert "whl/xpu" in advice.command


def test_an_amd_card_on_a_platform_rocm_does_not_reach_says_so_and_offers_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """macOS with an AMD card: a real machine, and the only honest answer is "CPU".

    Every other vendor branch ends in a command. This one cannot: ROCm is Linux and
    Windows only, so there is no wheel to name, and the note has to say that outright
    rather than leave the user looking for a command that was never printed.
    """
    patch_environment(
        monkeypatch, vendors=["amd"], system="Darwin", torch_cuda=None, accelerator=False
    )

    advice = advise_install(["amd"])

    assert advice.command is None
    notes = " ".join(advice.notes)
    assert "Linux and Windows" in notes, notes
    assert "train on the CPU" in notes, notes


# --------------------------------------------------------------- how the install is read
#
# `patch_environment` replaces `_torch_build` wholesale, which is right for the advice
# tests above and leaves the reader itself untested. It is the one function in the
# module that touches the real torch, and the `import torch` inside its body -- late on
# purpose, because torch's absence is a case it reports rather than a failure it
# propagates -- is also what makes it substitutable: `sys.modules` is consulted first,
# so a stub placed there is what the function imports.


def fake_torch(
    *,
    cuda: str | None = None,
    hip: str | None = None,
    cuda_available: bool = False,
    xpu_available: bool | None = None,
    mps_available: bool | None = None,
) -> types.ModuleType:
    """A ``torch`` carrying only what ``_torch_build`` reads.

    ``None`` for ``xpu_available`` or ``mps_available`` leaves the attribute off
    altogether, which is the difference between a wheel built without that backend and
    one that has it and finds no card.
    """
    module = types.ModuleType("torch")
    module.version = types.SimpleNamespace(cuda=cuda, hip=hip)
    module.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    module.backends = types.ModuleType("torch.backends")
    if xpu_available is not None:
        module.xpu = types.SimpleNamespace(is_available=lambda: xpu_available)
    if mps_available is not None:
        module.backends.mps = types.SimpleNamespace(is_available=lambda: mps_available)
    return module


def build_with(module: Any) -> tuple[str | None, bool]:
    """Read the build with ``module`` standing in for torch, for exactly one call.

    The swap is undone the moment the call returns rather than at teardown: the real
    torch is imported at the top of this file and used by most of the suite, and there
    is no reason to leave a stub in ``sys.modules`` for longer than the one function
    that has to see it. ``None`` is how a missing torch is spelled -- an import that
    finds ``None`` under its name raises ``ImportError``.
    """
    from trainai.hardware.install import _torch_build

    original = sys.modules["torch"]
    sys.modules["torch"] = module
    try:
        return _torch_build()
    finally:
        sys.modules["torch"] = original


def test_a_cpu_only_wheel_is_read_as_cpu_only() -> None:
    assert build_with(fake_torch()) == ("CPU-only", False)


def test_a_cuda_wheel_is_named_by_the_version_it_was_built_against() -> None:
    """The string reaches the user, and `doctor` prints it beside the driver's version."""
    assert build_with(fake_torch(cuda="12.6", cuda_available=True)) == ("CUDA 12.6", True)


def test_a_rocm_wheel_is_named_by_its_hip_version() -> None:
    """`torch.version.cuda` is not the tell on ROCm -- `hip` is, and it is read first.

    A ROCm build reaches the GPU through the CUDA API, so several of the attributes
    around this one answer as if the card were NVIDIA. The description has to say ROCm
    anyway, because it is what the user compares against the wheel they installed.
    """
    assert build_with(fake_torch(hip="6.2.41133", cuda_available=True)) == ("ROCm 6.2.41133", True)


def test_an_intel_card_is_found_even_though_the_wheel_names_no_version() -> None:
    """The accelerator flag and the wheel description answer different questions.

    An XPU wheel sets neither ``version.cuda`` nor ``version.hip``, so the description
    falls through to the same "CPU-only" a plain wheel gets. The flag is the part
    ``advise_install`` acts on -- True here means it says the install already works and
    prints no command, which is right, and would be wrong if it read the description.
    """
    assert build_with(fake_torch(xpu_available=True)) == ("CPU-only", True)


def test_an_intel_build_with_no_card_in_the_machine_reports_no_accelerator() -> None:
    """The control for the test above: the flag follows `is_available`, not the attribute.

    Reading ``torch.xpu`` as the answer on its own would report an accelerator on every
    XPU wheel, card or no card, and send a user with none of the hardware into a
    training run that then falls back to the CPU without saying so.
    """
    assert build_with(fake_torch(xpu_available=False)) == ("CPU-only", False)


def test_apple_metal_is_found_through_the_backends_module() -> None:
    """MPS hangs off `torch.backends`, not `torch`, and the default wheel includes it."""
    assert build_with(fake_torch(mps_available=True)) == ("CPU-only", True)
    assert build_with(fake_torch(mps_available=False)) == ("CPU-only", False)


def test_a_torch_that_answers_nothing_is_described_rather_than_raised() -> None:
    """Regression. A file named `torch.py` beside the script imports as an empty module.

    It is a beginner's mistake with a name this popular, and it lands in the one command
    written for people whose install is wrong. ``torch.backends`` used to be read
    without a guard while every attribute around it had one, so ``trainai setup`` died
    with ``AttributeError: module 'torch' has no attribute 'backends'`` instead of
    reporting what it found.
    """
    assert build_with(types.ModuleType("torch")) == ("CPU-only", False)


def test_torch_missing_altogether_is_a_description_of_none() -> None:
    """`None` is what `advise_install` turns into "PyTorch is not installed."

    Distinct from ``"CPU-only"``: one means re-run the install, the other means the
    install finished and picked the wrong wheel. Collapsing them would send a user
    whose torch never installed a command to change channels.
    """
    assert build_with(None) == (None, False)


# ------------------------------------------------------- where a 2.5 GB wheel would land
#
# `setup --install` refuses to run pip outside a virtual environment unless told twice,
# because the alternative is 2.5 GB landing in the system Python. Until now the only
# thing asserted about that gate was that it returns a bool.


def test_a_virtual_environment_is_recognised_by_its_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "prefix", "/tmp/env")
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    monkeypatch.delenv("CONDA_PREFIX", raising=False)

    assert in_virtualenv() is True


def test_conda_is_recognised_although_it_leaves_the_prefixes_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """conda activates by environment variable, so the prefix test alone misses it.

    Reading a conda environment as the system Python would make ``setup --install``
    refuse the one install it should have been happy to do.
    """
    monkeypatch.setattr(sys, "prefix", "/usr")
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    monkeypatch.setenv("CONDA_PREFIX", "/opt/conda/envs/trainai")

    assert in_virtualenv() is True


def test_the_system_python_is_not_mistaken_for_an_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The answer that matters: this is the case the confirmation prompt exists for."""
    monkeypatch.setattr(sys, "prefix", "/usr")
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    monkeypatch.delenv("CONDA_PREFIX", raising=False)

    assert in_virtualenv() is False


def test_an_empty_conda_prefix_is_not_an_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A variable set to nothing is how a deactivated shell can leave it behind."""
    monkeypatch.setattr(sys, "prefix", "/usr")
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    monkeypatch.setenv("CONDA_PREFIX", "")

    assert in_virtualenv() is False


# --------------------------------------------------------------------------- #
# The one sentence both commands print
# --------------------------------------------------------------------------- #
def test_the_situation_reads_as_english_for_a_switchable_graphics_laptop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two vendors is the common case on a gaming laptop, not an exotic one.

    Found by the wheel rehearsal: installing from PyPI on Windows gets a CPU-only
    torch, so this sentence is the first thing such a user reads -- and on a machine
    with integrated Radeon graphics beside the NVIDIA card it read "A amd/nvidia GPU
    is present but PyTorch (CPU-only) cannot use it".
    """
    patch_environment(
        monkeypatch, vendors=["amd", "nvidia"], system="Windows", torch_cuda=None, accelerator=False
    )

    situation = advise_install(["amd", "nvidia"]).situation

    assert situation.startswith("AMD and NVIDIA GPUs are present")
    assert "cannot use them" in situation
    assert "amd" not in situation


@pytest.mark.parametrize(
    ("vendors", "expected"),
    [
        (["nvidia"], "An NVIDIA GPU is present but PyTorch cannot use it,"),
        (["amd"], "An AMD GPU is present but PyTorch cannot use it,"),
        (["intel"], "An Intel GPU is present but PyTorch cannot use it,"),
        (["apple"], "An Apple GPU is present but PyTorch cannot use it,"),
        (["amd", "nvidia"], "AMD and NVIDIA GPUs are present but PyTorch cannot use them,"),
        (
            ["amd", "intel", "nvidia"],
            "AMD, Intel and NVIDIA GPUs are present but PyTorch cannot use them,",
        ),
    ],
)
def test_every_vendor_combination_is_spelled_and_numbered_correctly(
    vendors: list[str], expected: str
) -> None:
    """Article, spelling, plural and pronoun, for every list the detector can return.

    Parametrized rather than spot-checked because the four vendors all take "an" --
    they are read as letters -- which looks like a typo and will be "corrected" by
    someone eventually.
    """
    assert describe_unusable_gpu(vendors).startswith(expected)


def test_the_build_is_named_when_the_caller_knows_it() -> None:
    """``setup`` knows which torch is installed; the probe does not."""
    assert "PyTorch (CPU-only) cannot use it" in describe_unusable_gpu(["nvidia"], "CPU-only")
    assert "PyTorch cannot use it" in describe_unusable_gpu(["nvidia"])


def test_an_unknown_slug_is_printed_rather_than_swallowed() -> None:
    """A vendor the prose table has not heard of still has to produce a sentence."""
    assert describe_unusable_gpu(["mystery"]).startswith("A mystery GPU is present")


def test_the_json_payload_keeps_the_lowercase_slugs() -> None:
    """The labels are for reading. Anything scripted matches on the slug."""
    assert vendor_labels(["amd", "nvidia"]) == ["AMD", "NVIDIA"]
    assert advise_install(["amd", "nvidia"]).to_dict()["detected_vendors"] == ["amd", "nvidia"]


def test_apple_silicon_without_mps_suspects_a_rosetta_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_environment(
        monkeypatch, vendors=["apple"], system="Darwin", torch_cuda=None, accelerator=False
    )

    advice = advise_install(["apple"])

    assert "Rosetta" in " ".join(advice.notes)


def test_a_working_install_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_environment(
        monkeypatch,
        vendors=["nvidia"],
        system="Linux",
        torch_cuda="12.6",
        accelerator=True,
        driver=(12, 6),
    )

    advice = advise_install(["nvidia"])

    assert not advice.needs_action
    assert advice.command is None


def test_no_gpu_at_all_reports_the_speed_penalty_rather_than_a_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_environment(monkeypatch, vendors=[], system="Linux", torch_cuda=None, accelerator=False)

    advice = advise_install([])

    assert not advice.needs_action
    assert advice.command is None
    assert "20-100x slower" in " ".join(advice.notes)


def test_a_missing_torch_still_produces_a_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("trainai.hardware.install._torch_build", lambda: (None, False))
    monkeypatch.setattr("trainai.hardware.install.detect_driver_cuda_version", lambda: (12, 6))

    advice = advise_install(["nvidia"])

    assert advice.needs_action
    assert advice.command is not None


def test_every_command_reinstalls_rather_than_skipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The usual case is a CPU-only torch at the same version number, which a plain
    install would consider already satisfied."""
    for vendors, system, driver in (
        (["nvidia"], "Linux", (12, 6)),
        (["amd"], "Linux", None),
        (["intel"], "Linux", None),
    ):
        patch_environment(
            monkeypatch,
            vendors=vendors,
            system=system,
            torch_cuda=None,
            accelerator=False,
            driver=driver,
        )
        advice = advise_install(vendors)
        assert advice.command is not None
        assert "--force-reinstall" in advice.command, vendors


def test_advice_serialises_to_json(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    patch_environment(
        monkeypatch,
        vendors=["nvidia"],
        system="Linux",
        torch_cuda=None,
        accelerator=False,
        driver=(12, 6),
    )

    payload = json.loads(json.dumps(advise_install(["nvidia"]).to_dict()))

    assert payload["driver_cuda_version"] == "12.6"
    assert payload["selector"] == PYTORCH_SELECTOR


# --------------------------------------------------------------------------- #
# The driver probe, against an `nvidia-smi` that prints something undecodable
# --------------------------------------------------------------------------- #
def fake_nvidia_smi(root: Path, payload: bytes) -> Path:
    """A runnable stand-in for ``nvidia-smi`` that writes ``payload`` to stdout.

    An executable rather than a patched ``subprocess.run``, because the whole point
    is to exercise the real decode: patching the call would test the mock's codec.
    A shim script on each platform, since the bytes have to come out exactly and no
    shell builtin emits an arbitrary byte portably.
    """
    emitter = root / "emit.py"
    emitter.write_text(
        "import sys; sys.stdout.buffer.write(" + repr(payload) + ")",
        encoding="utf-8",
        newline="\n",
    )
    if os.name == "nt":
        shim = root / "nvidia-smi.bat"
        shim.write_text(
            f'@echo off\r\n"{sys.executable}" "{emitter}" %*\r\n', encoding="utf-8", newline=""
        )
    else:
        shim = root / "nvidia-smi"
        shim.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{emitter}" "$@"\n', encoding="utf-8", newline="\n"
        )
        shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return shim


#: 0x81 is undefined in cp1252 and an invalid UTF-8 lead byte, so it is undecodable
#: under both of the locales this suite runs in. A real ``nvidia-smi`` reaches this
#: territory through its box-art table and the card's marketing name.
_UNDECODABLE_SMI = b"NVIDIA-SMI 550.54  Driver Version: 550.54  CUDA Version: 12.6 \x81\n"


def test_the_driver_version_survives_an_undecodable_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression, and the reason the call states its encoding.

    ``text=True`` alone decodes with the locale's codec and ``errors=None`` --
    strict -- and ``UnicodeDecodeError`` is a ``ValueError``, so it does not reach the
    ``(OSError, subprocess.SubprocessError)`` handler around this call. On POSIX it
    propagated out of ``subprocess.run`` and took ``doctor`` and ``setup`` down with a
    traceback; on Windows it was raised in ``Popen._readerthread`` instead, so the
    threading machinery printed the traceback at the user and ``result.stdout`` came
    back as ``None``, losing the reading silently.

    The version is still found, which is the part that matters: the bad byte is
    replaced rather than fatal, so the wheel-channel advice is still made from a
    measurement.
    """
    shim = fake_nvidia_smi(tmp_path, _UNDECODABLE_SMI)
    monkeypatch.setattr("trainai.hardware.install.shutil.which", lambda _name: str(shim))

    assert detect_driver_cuda_version() == (12, 6)


def test_an_nvidia_smi_that_prints_nothing_useful_reports_no_driver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The negative control: "no version found" must stay distinguishable from a crash."""
    shim = fake_nvidia_smi(tmp_path, b"command not found\n")
    monkeypatch.setattr("trainai.hardware.install.shutil.which", lambda _name: str(shim))

    assert detect_driver_cuda_version() is None


def test_no_nvidia_smi_on_the_path_reports_no_driver_without_running_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ordinary case on every machine without an NVIDIA driver, including this one.

    ``shutil.which`` is checked first so that the absence is answered without a
    ``subprocess`` call at all -- a spawn that fails costs a process launch on every
    ``doctor`` and ``setup`` run, and on Windows can put a "not recognized" line on
    stderr that the user sees.
    """
    monkeypatch.setattr("trainai.hardware.install.shutil.which", lambda _name: None)
    monkeypatch.setattr(
        "trainai.hardware.install.subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("nvidia-smi was launched with nothing to launch"),
    )

    assert detect_driver_cuda_version() is None


# --------------------------------------------------------------------------- #
# The Linux sysfs card walk
# --------------------------------------------------------------------------- #
def fake_drm(root: Path, cards: dict[str, bytes | None]) -> Path:
    """Build a `/sys/class/drm` lookalike. ``None`` means the card has no vendor file.

    Real sysfs cannot be asked for a malformed card on demand, and the bug these
    tests pin was in exactly that path, so the directory has to be constructed.
    """
    drm = root / "drm"
    for name, vendor in cards.items():
        device = drm / name / "device"
        device.mkdir(parents=True)
        if vendor is not None:
            (device / "vendor").write_bytes(vendor)
    return drm


def test_the_card_walk_reads_the_vendor_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    drm = fake_drm(tmp_path, {"card0": b"0x10de\n", "card1": b"0x8086\n"})
    monkeypatch.setattr("trainai.hardware.probe._DRM_CLASS_PATH", str(drm))

    assert _linux_pci_vendors() == {"nvidia", "intel"}


def test_one_undecodable_card_does_not_hide_the_others(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression, and the reason the read states an encoding.

    ``UnicodeDecodeError`` is a ``ValueError``, not an ``OSError``. With a bare
    ``except OSError: continue`` it escaped past the per-card handler to the
    function-wide one, so a single unreadable card ended the loop and every card
    enumerated after it was lost -- `doctor` then reported no GPU vendor at all,
    which reads as a hardware problem rather than one bad sysfs node.

    ``card0`` sorts first on purpose: if the failure aborts the loop, the NVIDIA card
    behind it disappears and the result is empty.
    """
    drm = fake_drm(tmp_path, {"card0": b"\xff\xfe\x00garbage", "card1": b"0x10de\n"})
    monkeypatch.setattr("trainai.hardware.probe._DRM_CLASS_PATH", str(drm))

    assert _linux_pci_vendors() == {"nvidia"}


def test_a_card_with_no_vendor_file_does_not_hide_the_others(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    drm = fake_drm(tmp_path, {"card0": None, "card1": b"0x1002\n"})
    monkeypatch.setattr("trainai.hardware.probe._DRM_CLASS_PATH", str(drm))

    assert _linux_pci_vendors() == {"amd"}


def test_an_unknown_vendor_id_is_not_invented(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    drm = fake_drm(tmp_path, {"card0": b"0xdead\n"})
    monkeypatch.setattr("trainai.hardware.probe._DRM_CLASS_PATH", str(drm))

    assert _linux_pci_vendors() == set()


def test_a_missing_drm_directory_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every non-Linux machine and every headless container looks like this."""
    monkeypatch.setattr("trainai.hardware.probe._DRM_CLASS_PATH", str(tmp_path / "absent"))

    assert _linux_pci_vendors() == set()


# --------------------------------------------------------------------------- #
# Which vendors the machine says are present
# --------------------------------------------------------------------------- #
def fake_machine(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tools: tuple[str, ...] = (),
    system: str = "Linux",
    machine: str = "x86_64",
    drm: Path | str = "/nonexistent",
) -> None:
    """Everything ``detect_gpu_vendors`` reads, so its answer is not this machine's.

    Left alone the function reports whatever is installed on the machine running the
    suite, and that is why seven of its lines had never been read: ``rocm-smi`` and
    ``xpu-smi`` are on nobody's development PATH, and the platform fork can only take
    one of its three branches per run. Measured here, the missing lines are the Linux
    card walk and the two AMD/Intel tools; on a Linux runner the same file instead
    leaves the whole registry walk unread. Neither list describes the code, so the
    facts are supplied rather than sampled.
    """
    monkeypatch.setattr(
        "trainai.hardware.probe.shutil.which",
        lambda name: f"/usr/bin/{name}" if name in tools else None,
    )
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr("trainai.hardware.probe._DRM_CLASS_PATH", str(drm))


def fake_winreg(adapters: dict[str, str | None]) -> types.ModuleType:
    """A ``winreg`` that answers for a named set of display adapters.

    Keys are the subkey names the display class key enumerates, in order; values are
    the ``DriverDesc`` each one holds, or ``None`` for a subkey that has no such value.
    Enumeration ends the way the real one does, by raising ``OSError`` once it runs
    out, and the module is injected into ``sys.modules`` so the walk can be read on any
    platform -- ``import winreg`` inside the function is the only seam it has.
    """
    names = list(adapters)

    class Key:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self) -> Key:
            return self

        def __exit__(self, *_: object) -> None:
            return None

    def enum_key(key: Key, index: int) -> str:
        if index >= len(names):
            raise OSError(259, "No more data is available")
        return names[index]

    def query_value_ex(key: Key, value: str) -> tuple[str, int]:
        description = adapters[key.name]
        if description is None:
            raise OSError(2, f"{value} does not exist under {key.name}")
        return description, 1

    module = types.ModuleType("winreg")
    module.HKEY_LOCAL_MACHINE = "HKLM"  # type: ignore[attr-defined]
    module.OpenKey = lambda parent, path: Key(path)  # type: ignore[attr-defined]
    module.EnumKey = enum_key  # type: ignore[attr-defined]
    module.QueryValueEx = query_value_ex  # type: ignore[attr-defined]
    return module


@pytest.mark.parametrize(
    ("tool", "vendor"),
    [
        ("nvidia-smi", "nvidia"),
        ("rocm-smi", "amd"),
        ("rocminfo", "amd"),
        ("xpu-smi", "intel"),
    ],
)
def test_a_vendors_own_tool_on_the_path_is_enough(
    monkeypatch: pytest.MonkeyPatch, tool: str, vendor: str
) -> None:
    """A management tool means a driver, which is the thing that decides the answer.

    ``rocm-smi`` and ``rocminfo`` are both checked because a ROCm install can carry
    either one: ``rocminfo`` ships with the runtime and ``rocm-smi`` with the
    management stack, and a container that installs one and not the other would
    otherwise report no AMD GPU on a machine that has one.
    """
    fake_machine(monkeypatch, tools=(tool,))

    assert detect_gpu_vendors() == [vendor]


@pytest.mark.parametrize("machine", ["arm64", "aarch64"])
def test_apple_silicon_needs_no_tool_on_the_path(
    monkeypatch: pytest.MonkeyPatch, machine: str
) -> None:
    """There is no ``metal-smi``, and there does not need to be.

    Both spellings of the architecture are accepted because Python reports ``arm64``
    on macOS and ``aarch64`` under an emulated or containerised Linux userland on the
    same silicon.
    """
    fake_machine(monkeypatch, system="Darwin", machine=machine)

    assert detect_gpu_vendors() == ["apple"]


def test_an_intel_mac_is_not_given_an_apple_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """The architecture is load-bearing, not decoration.

    Pre-2020 Macs have an integrated GPU too, but Metal on them is not something
    PyTorch can train through, so claiming one would turn `doctor` into an advert for
    an install that cannot help. The elif chain also means a missed Darwin match falls
    through to nothing rather than to the Linux card walk.
    """
    fake_machine(monkeypatch, system="Darwin", machine="x86_64")

    assert detect_gpu_vendors() == []


def test_the_platform_decides_which_enumeration_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Linux reads sysfs, Windows reads the registry, and neither runs on the other.

    The point of pinning both from one test is that the branches are mutually
    exclusive: a Linux runner covering the card walk says nothing about the registry
    walk still compiling, and this is the fork where that goes unnoticed.
    """
    drm = fake_drm(tmp_path, {"card0": b"0x1002\n"})
    monkeypatch.setitem(sys.modules, "winreg", fake_winreg({"0000": "Intel(R) Arc(TM) A770"}))

    fake_machine(monkeypatch, system="Linux", drm=drm)
    assert detect_gpu_vendors() == ["amd"]

    fake_machine(monkeypatch, system="Windows", drm=drm)
    assert detect_gpu_vendors() == ["intel"]


def test_one_unreadable_adapter_key_does_not_hide_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registry walk's version of the bug the card walk already carries a test for.

    That class key holds more than adapters: ``Properties`` and ``Configuration``
    subkeys sit beside the numbered ones, and a numbered one can exist with no
    ``DriverDesc`` at all -- a device that was uninstalled, or one whose value the
    process may not read. The unreadable key sorts *between* the two real adapters on
    purpose: if its ``OSError`` broke the loop instead of skipping the key, Intel would
    vanish and the machine would look like it had only an NVIDIA card.

    ``Properties`` is given a description naming a third vendor so that the numeric
    check is doing something here. The real one holds no ``DriverDesc``, so the walk
    would drop it on the value read anyway -- the guard is what means it never gets
    that far, whatever else that class key comes to hold.
    """
    monkeypatch.setitem(
        sys.modules,
        "winreg",
        fake_winreg(
            {
                "Properties": "AMD platform device, not a display adapter",
                "0000": "NVIDIA GeForce RTX 4090",
                "0001": None,
                "0002": "Intel(R) Arc(TM) A770",
            }
        ),
    )

    assert _windows_registry_vendors() == {"nvidia", "intel"}


@pytest.mark.parametrize(
    ("description", "vendor"),
    [
        ("NVIDIA GeForce RTX 4090", "nvidia"),
        ("GeForce GTX 1080", "nvidia"),
        ("AMD Radeon RX 6900 XT", "amd"),
        ("Radeon (TM) Graphics", "amd"),
        ("Intel(R) UHD Graphics 770", "intel"),
        ("Microsoft Basic Display Adapter", None),
    ],
)
def test_the_driver_description_is_matched_on_either_of_its_names(
    monkeypatch: pytest.MonkeyPatch, description: str, vendor: str | None
) -> None:
    """``DriverDesc`` is marketing copy, so both the maker and the brand are matched.

    An OEM machine can carry "GeForce GTX 1080" with no "NVIDIA" in the string and
    "Radeon (TM) Graphics" with no "AMD", which is why neither vendor is matched on its
    company name alone. The last case is the one that has to stay unclaimed: the
    fallback adapter Windows installs when no display driver is present is not a GPU
    anyone can train on.
    """
    monkeypatch.setitem(sys.modules, "winreg", fake_winreg({"0000": description}))

    assert _windows_registry_vendors() == ({vendor} if vendor else set())


def test_a_gpu_with_no_nameable_vendor_still_makes_a_sentence() -> None:
    """No caller reaches this, and the sentence is here so that it stays true if one does.

    Both callers guard on a non-empty vendor list, so the subject is written from
    labels that are always there. Were that to lapse, the alternative reads "A  GPU is
    present" or worse -- and this is a message shown to someone whose expensive card
    is not being used, which is not the moment to look broken.
    """
    assert describe_unusable_gpu([]).startswith("A GPU is present but PyTorch cannot use it")


# --------------------------------------------------------------------------- #
# Container limits: the machine TrainAI is given, not the one it is on
# --------------------------------------------------------------------------- #
def fake_cgroup(root: Path, files: dict[str, str]) -> Path:
    """Build a cgroupfs lookalike. Keys are relative paths, so v1 uses ``cpu/cpu.max``.

    A directory rather than a container, because no CI runner can be asked to impose a
    specific CPU or memory quota on demand, and because both cgroup versions have to be
    covered on a machine that can only have one. The files are the whole interface: what
    the kernel puts in them is a decimal integer or the word ``max``, which a test can
    reproduce exactly.
    """
    cgroup = root / "cgroup"
    for name, contents in files.items():
        target = cgroup / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents, encoding="ascii", newline="\n")
    return cgroup


def use_cgroup(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    files: dict[str, str],
    *,
    affinity: int | None = None,
) -> None:
    """Point the probe at a fake cgroupfs, and pin the affinity mask while doing it.

    Pinning affinity is not tidiness. ``_cpu_affinity_count`` reads the real machine, so
    without this a quota test also depends on how many cores the runner has: GitHub's
    two-core Linux runners turned ``cpu.max`` of sixteen cores into a limit of two and
    failed a test about the quota branch. ``None`` is "no affinity narrowing", which is
    what every Windows machine reports anyway.
    """
    monkeypatch.setattr("trainai.hardware.probe._CGROUP_ROOT", str(fake_cgroup(root, files)))
    monkeypatch.setattr("trainai.hardware.probe._cpu_affinity_count", lambda: affinity)


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        # cgroup v2: `docker run --cpus=4` on a default 100ms period.
        ({"cpu.max": "400000 100000\n"}, 4.0),
        # A fractional quota is real and must not round up to a core it does not have.
        ({"cpu.max": "150000 100000\n"}, 1.5),
        ({"cpu.max": "50000 100000\n"}, 0.5),
        # A period other than the default, which Kubernetes does set.
        ({"cpu.max": "200000 50000\n"}, 4.0),
        # v2 with no cap. The file exists; that is not the same as a limit.
        ({"cpu.max": "max 100000\n"}, None),
        # cgroup v1, still the default on plenty of hosts.
        ({"cpu/cpu.cfs_quota_us": "200000\n", "cpu/cpu.cfs_period_us": "100000\n"}, 2.0),
        # v1 spells "no quota" as -1, which must not become a negative core count.
        ({"cpu/cpu.cfs_quota_us": "-1\n", "cpu/cpu.cfs_period_us": "100000\n"}, None),
        # Nothing mounted at all: every Windows and macOS machine.
        ({}, None),
        # Malformed, which a probe must survive rather than raise on.
        ({"cpu.max": "\n"}, None),
        ({"cpu.max": "banana pear\n"}, None),
        ({"cpu.max": "400000 0\n"}, None),
        ({"cpu/cpu.cfs_quota_us": "200000\n", "cpu/cpu.cfs_period_us": "0\n"}, None),
    ],
)
def test_the_cpu_quota_is_read_from_either_cgroup_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, files: dict[str, str], expected: float | None
) -> None:
    use_cgroup(monkeypatch, tmp_path, files)

    assert _cgroup_cpu_quota() == expected


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        # cgroup v2: `docker run -m 8g`.
        (
            {"memory.max": "8589934592\n", "memory.current": "1073741824\n"},
            (8589934592, 1073741824),
        ),
        # The limit is readable, the usage is not. Report what is known.
        ({"memory.max": "8589934592\n"}, (8589934592, None)),
        ({"memory.max": "max\n"}, (None, None)),
        # cgroup v1.
        (
            {
                "memory/memory.limit_in_bytes": "2147483648\n",
                "memory/memory.usage_in_bytes": "512\n",
            },
            (2147483648, 512),
        ),
        # v1's "unlimited" is a sentinel near 8 EiB, not an absent file. Reading it as a
        # limit would report a machine with eight exabytes of RAM.
        ({"memory/memory.limit_in_bytes": "9223372036854771712\n"}, (None, None)),
        ({}, (None, None)),
        ({"memory.max": "not a number\n"}, (None, None)),
        ({"memory.max": "0\n"}, (None, None)),
    ],
)
def test_the_memory_limit_is_read_from_either_cgroup_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    files: dict[str, str],
    expected: tuple[int | None, int | None],
) -> None:
    use_cgroup(monkeypatch, tmp_path, files)

    assert _cgroup_memory_limit() == expected


def test_a_quota_looser_than_the_machine_is_not_reported_as_a_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A limit above what the host has is not a constraint, and must not read as one.

    Container runtimes routinely set a quota at or above the host's size. Reporting it
    would put a `cpu_limit` field on every such machine claiming a cap that changes
    nothing, and `doctor` would tell an unconstrained user they were constrained.
    """
    use_cgroup(monkeypatch, tmp_path, {"cpu.max": "1600000 100000\n"})

    assert _probe_cpu_limit(logical=8) is None
    assert _probe_cpu_limit(logical=32) == 16.0


@pytest.mark.parametrize(
    ("quota", "affinity", "expected"),
    [
        # An affinity mask alone, with nothing mounted: `taskset -c 0,1` on a bare host.
        (None, 2, 2.0),
        # A quota alone, with every core schedulable.
        ({"cpu.max": "400000 100000\n"}, 8, 4.0),
        # Both, and the tighter one wins whichever side it is on.
        ({"cpu.max": "400000 100000\n"}, 2, 2.0),
        ({"cpu.max": "100000 100000\n"}, 4, 1.0),
        # Neither narrows anything: an ordinary machine, and no field claiming a cap.
        (None, None, None),
        (None, 8, None),
        ({"cpu.max": "800000 100000\n"}, 8, None),
    ],
)
def test_the_tighter_of_the_quota_and_the_affinity_mask_wins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    quota: dict[str, str] | None,
    affinity: int | None,
    expected: float | None,
) -> None:
    """The two narrowings are independent, and reporting only one under-reports.

    A cgroup quota and a CPU affinity mask come from different places -- a container
    runtime and `taskset` or an HPC scheduler -- and either can be the binding one. The
    mask is the half this machine cannot exercise honestly: `os.sched_getaffinity` does
    not exist on Windows, which is why the probe reads it through a function a test can
    stand in for.
    """
    use_cgroup(monkeypatch, tmp_path, quota or {}, affinity=affinity)

    assert _probe_cpu_limit(logical=8) == expected


def test_the_affinity_count_answers_for_the_machine_it_is_actually_on() -> None:
    """The one assertion about affinity that cannot be faked, and so must be run.

    Every other affinity test replaces this function, which would make a rename or a
    raise invisible. Here it runs for real, on whichever platform the suite is on: a
    positive count where `os.sched_getaffinity` exists, `None` where it does not.
    """
    allowed = _cpu_affinity_count()

    if hasattr(os, "sched_getaffinity"):
        assert allowed is not None and allowed >= 1
    else:
        assert allowed is None


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        ({0, 1, 2, 3}, 4),
        ({3}, 1),
        # An empty mask should not be possible, and must not become a cap of zero cores
        # if it happens: `doctor` would print "capped at 0", which is not a machine.
        (set(), None),
    ],
)
def test_the_affinity_reader_turns_a_mask_into_a_core_count(
    monkeypatch: pytest.MonkeyPatch, mask: set[int], expected: int | None
) -> None:
    """`raising=False` because Windows has no `sched_getaffinity` to replace.

    Installing it is the only way to reach the POSIX half of this reader from here, and
    the half is worth reaching: it is what `taskset` and every HPC scheduler set.
    """
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: mask, raising=False)

    assert _cpu_affinity_count() == expected


def test_a_memory_limit_at_or_above_the_host_is_not_reported_as_a_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_cgroup(monkeypatch, tmp_path, {"memory.max": str(16 * GIB)})

    assert _probe_ram_limit(total_ram=16 * GIB) == (None, None)
    assert _probe_ram_limit(total_ram=64 * GIB) == (16 * GIB, None)
    # Host RAM unknown: report the limit rather than discard the only figure there is.
    assert _probe_ram_limit(total_ram=0) == (16 * GIB, None)


def test_an_undecodable_cgroup_file_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The same `UnicodeDecodeError`-is-a-`ValueError` trap as the sysfs card walk.

    A cgroup file will not really hold `0xff`, but the handler either catches it or the
    probe raises out of `doctor`, and "will not really" is not a thing to rely on when
    the alternative is one name in an except clause.
    """
    cgroup = fake_cgroup(tmp_path, {"cpu.max": "max 100000\n"})
    (cgroup / "memory.max").write_bytes(b"\xff\xfe garbage\n")
    monkeypatch.setattr("trainai.hardware.probe._CGROUP_ROOT", str(cgroup))

    assert _cgroup_memory_limit() == (None, None)
    assert _cgroup_cpu_quota() is None


#: A four-core, 8 GiB container on the 16-core, 32 GiB synthetic host. The shape of a
#: Codespace, a CI runner, a Kubernetes pod, or `docker run --cpus=4 -m 8g`.
def constrained() -> HardwareProfile:
    return replace(
        cpu_only(),
        cpu_limit=4.0,
        ram_limit_bytes=8 * GIB,
        ram_limit_used_bytes=6 * GIB,
    )


@pytest.mark.parametrize(
    ("limit", "logical", "expected"),
    [
        (None, 16, 16),
        (4.0, 16, 4),
        # Truncated, not rounded: 3.9 cores does not give you four threads' worth of
        # throughput, and reporting four would overstate the machine.
        (3.9, 16, 3),
        # Below one core there is still one thread, and it must never be zero -- a zero
        # would divide by nothing in anything that later sizes work from this.
        (0.5, 16, 1),
        (0.0, 16, 1),
        # A limit above the machine cannot raise the count.
        (64.0, 16, 16),
    ],
)
def test_the_usable_core_count_never_exceeds_the_machine_or_falls_to_zero(
    limit: float | None, logical: int, expected: int
) -> None:
    profile = replace(cpu_only(), cpu_count_logical=logical, cpu_limit=limit)

    assert profile.usable_cpu_count == expected


def test_usable_ram_is_the_lower_of_the_host_and_the_limit() -> None:
    """Both directions matter, and each has a plausible machine behind it."""
    host_is_tighter = replace(constrained(), available_ram_bytes=1 * GIB)
    assert host_is_tighter.usable_available_ram_bytes == 1 * GIB

    # 8 GiB limit, 6 GiB already charged: 2 GiB left, whatever the host has spare.
    limit_is_tighter = replace(constrained(), available_ram_bytes=20 * GIB)
    assert limit_is_tighter.usable_available_ram_bytes == 2 * GIB

    # Charged past the limit -- possible with page cache accounting. Not a negative.
    overcharged = replace(constrained(), ram_limit_used_bytes=9 * GIB)
    assert overcharged.usable_available_ram_bytes == 0

    # A limit read without its usage: report the limit, not nothing.
    no_usage = replace(constrained(), ram_limit_used_bytes=None, available_ram_bytes=20 * GIB)
    assert no_usage.usable_available_ram_bytes == 8 * GIB


def test_a_container_whose_host_ram_is_unreadable_still_has_the_limit_to_go_on() -> None:
    """0 is `_probe_ram`'s "could not tell", and `min` would read it as "none left".

    The two numbers being compared do not mean the same kind of thing. A cgroup limit is
    read from a file and is either there or absent. The host's free RAM comes from psutil
    or `sysconf`, and when neither answers -- a stripped container image, a platform with
    no `os.sysconf` -- the probe appends a warning and returns 0, which is a sentinel and
    not a measurement of an empty machine.

    Taking the minimum of the two would then turn a container with 2 GiB of headroom into
    one with nothing, on a machine where the limit was read perfectly well. That is not a
    smaller plan, it is a refused one, and the reason would be a RAM probe failing rather
    than any shortage of RAM. The sibling above covers the case where both are real.
    """
    unreadable = replace(constrained(), available_ram_bytes=0)

    assert unreadable.usable_available_ram_bytes == 2 * GIB


def test_an_unconstrained_machine_reports_the_machine() -> None:
    """The negative control. Every existing profile must be untouched by all of this."""
    for label, build in ALL_MACHINES.items():
        profile = build()
        assert profile.cpu_limit is None, label
        assert profile.ram_limit_bytes is None, label
        assert profile.is_constrained is False, label
        assert profile.usable_cpu_count == profile.cpu_count_logical, label
        assert profile.usable_available_ram_bytes == profile.available_ram_bytes, label


def test_the_run_record_names_the_cores_the_run_actually_had() -> None:
    """`summarise_for_log` goes into `events.jsonl`, which is the record of the run.

    Sixteen cores on a four-core container is not a rounding error in a log line, it is
    a reproducibility record naming a machine the run was never on.
    """
    from trainai.cli.doctor import summarise_for_log

    assert "4 threads" in summarise_for_log(constrained())
    assert "16 threads" in summarise_for_log(cpu_only())


def test_the_run_record_names_the_card_a_gpu_run_was_on() -> None:
    """The other half of ``summarise_for_log``, and the half that matters for a real run.

    Both existing assertions pass a CPU profile, so only the no-GPU branch had ever run.
    This is the line that goes into ``events.jsonl`` for every run that used a card, and
    it is the record someone reads months later to answer "what was this trained on" --
    so the card, its free and total VRAM, its compute capability and the torch version
    all have to be in it. Free *and* total, because the same card with 2 GiB free and
    with 22 GiB free plans two different runs.
    """
    from trainai.cli.doctor import summarise_for_log

    line = summarise_for_log(rtx_4090())

    assert "RTX 4090" in line
    assert "22.1 GiB free of 24.0 GiB" in line
    assert "cc 8.9" in line
    assert "torch 2.6.0" in line
    assert "threads" not in line, "the CPU form was used for a machine with a card"


def test_doctor_reports_the_container_not_the_host(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = render_doctor(constrained(), capsys)

    assert "capped at 4" in report, "the core count does not say it is capped"
    assert "container limit of 8.00 GiB" in report, "the RAM row does not name the limit"
    assert "2.00 GiB available" in report, "the headroom shown is not what is left of the limit"
    assert "32.0 GiB on the host" in report, "the host figure is not kept as context"
    assert "will not match the same hardware run unconstrained" in report


def test_doctor_says_nothing_about_containers_on_an_ordinary_machine(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The false-positive gate: an unconstrained report must be unchanged by this work."""
    report = render_doctor(cpu_only(), capsys)

    assert "container" not in report.lower()
    assert "capped" not in report.lower()


# --------------------------------------------------------------------------- #
# The machine that will not answer
# --------------------------------------------------------------------------- #
def fake_sysconf(monkeypatch: pytest.MonkeyPatch, values: dict[str, int]) -> None:
    """``os.sysconf`` answering for a named set of variables and no others.

    Supplied rather than sampled, for the same reason the vendor tools are: this
    branch is the one taken on every platform except Windows, so on the machine this
    was written on ``os.sysconf`` does not exist at all and on a Linux runner it
    returns that runner's RAM. Neither makes a test. Unknown names raise ``ValueError``,
    which is what the real one does.
    """

    def sysconf(name: str) -> int:
        if name not in values:
            raise ValueError(f"unrecognized configuration name {name}")
        return values[name]

    monkeypatch.setattr(os, "sysconf", sysconf, raising=False)
    monkeypatch.setattr(os, "sysconf_names", dict.fromkeys(values, 0), raising=False)


def test_the_cpu_count_survives_psutil_being_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """psutil is a declared dependency, and its absence still must not be fatal.

    What is lost is the physical core count, which only psutil knows -- so the answer
    is ``None`` rather than a guess at logical/2, because hyperthreading is not
    universal and a wrong physical count feeds straight into the dataloader worker
    plan. What is kept is the logical count, which the stdlib has.
    """
    monkeypatch.setitem(sys.modules, "psutil", None)

    name, logical, physical = _probe_cpu()

    assert logical == (os.cpu_count() or 1)
    assert physical is None
    assert name is None or isinstance(name, str)


def test_ram_is_read_from_sysconf_when_psutil_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """`trainai doctor` has to work on a machine where the install went wrong.

    Page counts rather than bytes, which is the part worth pinning: ``SC_PHYS_PAGES``
    times ``SC_PAGE_SIZE`` is the total, and it is ``SC_AVPHYS_PAGES`` -- not the
    total -- that gives what is free.
    """
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    fake_sysconf(
        monkeypatch,
        {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 4 * 1024 * 1024, "SC_AVPHYS_PAGES": 1024 * 1024},
    )
    warnings: list[str] = []

    total, available = _probe_ram(warnings)

    assert (total, available) == (16 * GIB, 4 * GIB)
    assert warnings == []


def test_sysconf_without_the_available_pages_name_reports_the_total_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``SC_AVPHYS_PAGES`` is not in POSIX, so it is checked for before it is asked for.

    Reporting the total as available overstates what is free, which is the safe
    direction to be wrong in only because the planner budgets from a fraction of it and
    the run is measured before it is offered. Reporting zero would read as a machine
    with no memory left and refuse to plan anything at all.
    """
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    fake_sysconf(monkeypatch, {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 2 * 1024 * 1024})
    warnings: list[str] = []

    assert _probe_ram(warnings) == (8 * GIB, 8 * GIB)
    assert warnings == []


def test_a_platform_with_no_sysconf_at_all_warns_rather_than_ending_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The measured escape this handler was widened for.

    ``os.sysconf`` is not on every platform -- it is absent on Windows, and the line
    above already guards ``sysconf_names`` with ``hasattr`` for exactly that reason.
    Guarding the second call while calling the first bare cannot be defended, and it
    was not theoretical: measured on Windows with psutil absent and this branch
    reached, the run ended with ``AttributeError: module 'os' has no attribute
    'sysconf'`` -- out of the probe, out of ``probe_hardware``, and out of whichever
    command asked, which for `doctor` means the one command whose job is to explain a
    broken environment dying on it.

    ``delattr`` rather than a platform check, so the condition is the same wherever
    this runs.
    """
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.delattr(os, "sysconf", raising=False)
    warnings: list[str] = []

    assert _probe_ram(warnings) == (0, 0)
    assert warnings == ["System RAM could not be determined; RAM-based checks are disabled."]


def test_a_sysconf_that_does_not_know_the_names_warns_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other way this branch fails: the call exists, the variable does not."""
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    fake_sysconf(monkeypatch, {})
    warnings: list[str] = []

    assert _probe_ram(warnings) == (0, 0)
    assert "RAM-based checks are disabled" in warnings[0]


def test_an_unheard_of_platform_still_names_its_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three platforms get a good name; everything else gets the stdlib's.

    Windows, Linux and Darwin each have a place the readable model name lives, and
    none of those places exists on a BSD or an AIX. ``platform.processor()`` there is
    worse -- an architecture string rather than a model -- but it is true, and the
    alternative is a profile that reports no CPU on a machine that plainly has one.
    """
    monkeypatch.setattr(platform, "system", lambda: "FreeBSD")
    monkeypatch.setattr(platform, "processor", lambda: "amd64")

    assert _probe_cpu_name() == "amd64"


def test_a_platform_that_names_nothing_reports_no_cpu_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``processor()`` returns the empty string on more machines than it does not.

    It is documented to, so the machine name is tried next, and if that is empty too
    the answer is ``None``. Not the empty string: `doctor` prints the name when it has
    one, and an empty line under "CPU" reads as a probe that broke rather than a
    platform that does not say.
    """
    monkeypatch.setattr(platform, "system", lambda: "FreeBSD")
    monkeypatch.setattr(platform, "processor", lambda: "  ")
    monkeypatch.setattr(platform, "machine", lambda: "")

    assert _probe_cpu_name() is None


def test_the_platform_line_falls_back_to_release_and_machine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows and macOS have prettier version strings; nothing else needs one."""
    monkeypatch.setattr(platform, "system", lambda: "FreeBSD")
    monkeypatch.setattr(platform, "release", lambda: "14.1-RELEASE")
    monkeypatch.setattr(platform, "machine", lambda: "amd64")

    assert _platform_summary() == "FreeBSD 14.1-RELEASE (amd64)"


def test_the_walk_up_for_a_disk_stops_at_the_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Free space is asked about a directory that does not exist yet, on purpose.

    ``--out runs/2026-09-05/checkpoints`` names three directories the run is about to
    create, so the probe walks up until it finds something real and measures that
    filesystem. What stops the walk if *nothing* on the way up exists is the root
    comparing equal to its own parent -- without that the loop never ends. On Windows
    an unmounted drive letter reaches this for real; here nothing is said to exist, so
    the condition is the same on every platform.
    """
    monkeypatch.setattr(os.path, "exists", lambda path: False)
    warnings: list[str] = []

    free = _probe_disk(tmp_path / "runs" / "later", warnings)

    assert free > 0, "the root of a filesystem the suite is running on has a size"
    assert warnings == []


def test_free_space_that_cannot_be_read_is_a_warning_naming_the_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A directory can exist and still refuse to be measured.

    A path inside a container mount that has gone away, or one the process may stat but
    not query. Zero rather than a raise, because the planner reads this as "no free
    space known" and says so, and the path is named because the one thing the user
    needs is which location could not be measured -- not guessable when `train` was
    given an output directory and a cache directory.

    The path goes in bare. It was interpolated with ``!r`` until this test, which on
    Windows -- the platform this is most often read on -- printed
    ``'C:\\\\Users\\\\me\\\\runs'`` and left the reader deciding whether the doubled
    separators were in their path or in the printing. Nothing else in the project
    quotes a path that way.
    """

    def refuse(path: str) -> object:
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(shutil, "disk_usage", refuse)
    warnings: list[str] = []

    assert _probe_disk(tmp_path, warnings) == 0
    assert len(warnings) == 1
    assert str(tmp_path) in warnings[0]
    assert "\\\\" not in warnings[0]
    assert "Permission denied" in warnings[0]


def test_detection_is_fast_enough_to_run_before_every_command() -> None:
    """`doctor` and `setup` both call this; a slow probe makes the CLI feel broken."""
    import time

    start = time.perf_counter()
    vendors = detect_gpu_vendors()
    elapsed = time.perf_counter() - start

    assert isinstance(vendors, list)
    assert all(v in {"nvidia", "amd", "intel", "apple"} for v in vendors)
    assert elapsed < 5.0, f"vendor detection took {elapsed:.1f}s"


def test_detection_never_raises_whatever_this_machine_is() -> None:
    """The probe runs before we can know anything about the user's environment."""
    assert isinstance(detect_gpu_vendors(), list)
    assert isinstance(in_virtualenv(), bool)
    driver = detect_driver_cuda_version()
    assert driver is None or (isinstance(driver, tuple) and len(driver) == 2)


def test_this_machines_profile_is_internally_consistent() -> None:
    """Whatever hardware the suite is running on, the profile must not contradict itself."""
    profile = probe_hardware()

    assert profile.backend in {"cuda", "rocm", "xpu", "mps", "cpu"}
    assert profile.device_type in {"cuda", "xpu", "mps", "cpu"}
    if profile.backend == "rocm":
        assert profile.device_type == "cuda", "ROCm is addressed through the cuda API"
        assert not profile.supports_tf32, "TF32 does not exist on AMD"
    if profile.has_gpu:
        assert profile.vram_budget_bytes() > 0
        for device in profile.gpus:
            assert device.free_vram_bytes <= device.total_vram_bytes
    else:
        assert profile.vram_budget_bytes() == 0


def test_a_profile_can_be_edited_without_mutating_the_original() -> None:
    """Frozen dataclasses, because a profile is recorded in run files as evidence."""
    original = rtx_4090()

    modified = replace(original, os_name="Windows")

    assert original.os_name == "Linux"
    assert modified.os_name == "Windows"


def test_the_fixture_set_covers_every_backend() -> None:
    """A guard on this file: if a backend is added, it needs a machine here."""
    covered = {ALL_MACHINES[label]().backend for label in ALL_MACHINES}

    assert covered == {"cuda", "rocm", "xpu", "mps", "cpu"}


def test_the_fixture_set_covers_both_bf16_answers_on_every_gpu_backend() -> None:
    """Both branches of the decision that goes silently wrong."""
    answers: dict[str, set[bool]] = {}
    for label in ALL_MACHINES:
        profile = ALL_MACHINES[label]()
        if profile.backend in ("cuda", "rocm"):
            answers.setdefault(profile.backend, set()).add(profile.supports_bf16)

    assert answers["cuda"] == {True, False}
    assert answers["rocm"] == {True, False}


def test_details_of_any_hardware_error_are_json_safe() -> None:
    """Profiles and errors both end up in run records and, later, in the web UI."""
    import json

    with pytest.raises(TrainingError) as caught:
        precision_for("bf16", backend="cuda", supports_bf16=False, capability=(6, 1))

    assert json.loads(json.dumps(caught.value.to_dict()))["hint"]


def test_a_synthetic_profile_matches_the_shape_of_a_real_one() -> None:
    """Guards the fixtures: a field added to HardwareProfile must appear here too,
    or these tests would be exercising a stale shape."""
    real = probe_hardware().to_dict()
    synthetic: dict[str, Any] = rtx_4090().to_dict()

    assert set(real) == set(synthetic)


# --------------------------------------------------------------------------- #
# What `doctor` says about machines it is not running on
# --------------------------------------------------------------------------- #
def render_doctor(profile: HardwareProfile, capsys: pytest.CaptureFixture[str]) -> str:
    """Print `doctor`'s report for a synthetic profile and return the plain text.

    Uses pytest's capture rather than a private Console, because the notes go
    through the shared console in :mod:`trainai.console` -- swapping only the
    doctor module's reference would silently miss them, which is how the first
    version of this helper produced a passing test that checked nothing.
    """
    from trainai.cli import doctor as doctor_module

    for title, rows in (
        ("System", doctor_module._system_rows(profile)),
        ("Compute", doctor_module._compute_rows(profile)),
        ("Host", doctor_module._host_rows(profile)),
    ):
        doctor_module.console.print(title)
        for key, value in rows:
            doctor_module.console.print(f"{key}: {value}")
    doctor_module._print_notes(profile)
    # Flattened, because every assertion below is on a phrase: a note is wrapped at the
    # console width, so `not implemented` arrives as `not` and `implemented` on two
    # lines on any terminal narrower than this one.
    return flat(capsys.readouterr().out)


def test_doctor_tells_a_cpu_only_wheel_user_it_is_an_install_problem(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ "cpu" alone reads as a hardware limit. It is not, and the fix is one command."""
    report = render_doctor(nvidia_with_cpu_wheel(), capsys)

    assert "NVIDIA" in report
    assert "trainai setup" in report
    assert "cannot use it" in report


def test_doctor_names_two_vendors_as_a_reader_would_write_them(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A switchable-graphics laptop reports two vendors, and this row is prose.

    Found by installing the wheel into a clean venv and running ``doctor`` on this
    machine, which has integrated Radeon graphics beside the NVIDIA card: the
    slugs were printed raw, so the sentence read "A amd/nvidia GPU is present ...
    cannot use it". The slugs stay lowercase in ``--json``; only the prose changes.
    """
    report = render_doctor(
        machine("switchable graphics", gpus=[], backend="cpu", vendors=["amd", "nvidia"]),
        capsys,
    )

    assert "AMD, NVIDIA" in report
    assert "cannot use them" in report
    assert "amd" not in report


def test_doctor_does_not_offer_setup_when_there_is_genuinely_no_gpu(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = render_doctor(cpu_only(), capsys)

    assert "CPU only" in report
    assert "trainai setup" not in report


def test_doctor_does_not_report_tf32_on_amd(capsys: pytest.CaptureFixture[str]) -> None:
    """Reporting "TF32 no" implies the card fell short of something it cannot have."""
    report = render_doctor(rx_7900_xtx(), capsys)

    assert "TF32" not in report
    assert "gfx110" in report
    assert "ROCm" in report


def test_doctor_flags_spillover_on_windows_and_not_on_linux(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert "WDDM" in render_doctor(rtx_3070_windows(), capsys)
    assert "WDDM" not in render_doctor(rtx_4090(), capsys)


def test_doctor_renders_every_synthetic_machine_without_raising(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The report is the first thing a user sees; it must not crash on odd hardware."""
    for label in ALL_MACHINES:
        report = render_doctor(ALL_MACHINES[label](), capsys)
        assert report.strip(), label


def test_doctor_mentions_multi_gpu_is_not_used(capsys: pytest.CaptureFixture[str]) -> None:
    report = render_doctor(dual_4090(), capsys)

    assert "2 GPUs detected" in report
    assert "not implemented" in report


def test_doctor_says_unknown_rather_than_zero_when_ram_cannot_be_read(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A machine whose RAM the probe could not read, which is not a machine with no RAM.

    ``0.00 B available`` would be a measurement, and a wrong one -- the row exists to
    say how much there is, and the honest answer when nothing was read is that nobody
    knows. Every synthetic machine has a RAM figure, so this branch had never rendered.

    The two neighbouring notes are asserted absent for the same reason: both are
    thresholds on a number, and a machine that reports no number must not be told it is
    short of something. ``usable_available_ram_bytes`` on this profile is 0, which is
    below the 4 GiB threshold -- so the guard that keeps the note quiet is on
    ``total_ram_bytes``, and this is what says so.
    """
    unreadable = replace(cpu_only(), total_ram_bytes=0, available_ram_bytes=0)

    report = render_doctor(unreadable, capsys)

    assert "RAM: unknown" in report
    assert "0.00 B" not in report, "an unread figure was printed as a measurement of zero"
    assert "of system RAM is available" not in report, (
        "a machine with no reading was told it is low"
    )


def test_doctor_names_the_two_resources_it_finds_short(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The free-VRAM note and the free-disk note, neither of which any machine triggers.

    Both are advice rather than observation, which is why they are worth a test: the
    VRAM note says closing a browser will let TrainAI recommend a larger model, because
    on this project's own measurements the planner's answer moves with free VRAM and not
    with total. The disk note names the two things that fill a disk here, since "low
    disk" on its own does not tell anyone how much a run will need.

    Both thresholds are asserted from both sides in the same test. Every machine in the
    catalogue is above them -- 92% of VRAM free, 200 GiB of disk -- so a note printed
    unconditionally would appear on all thirteen, and one printed never would look
    identical to this test passing.
    """

    def freeing(profile: HardwareProfile, *, vram_gib: float, disk_gib: float) -> HardwareProfile:
        card = replace(profile.gpus[0], free_vram_bytes=int(vram_gib * GIB))
        return replace(profile, gpus=[card], free_disk_bytes=int(disk_gib * GIB))

    roomy = render_doctor(rtx_4090(), capsys)

    assert "of VRAM is free" not in roomy
    assert "of disk is free" not in roomy

    tight = render_doctor(freeing(rtx_4090(), vram_gib=1.5, disk_gib=2.0), capsys)

    assert "Only 1.50 GiB of VRAM is free" in tight
    assert "Closing your browser" in tight
    assert "recommend a larger model" in tight
    assert "Only 2.00 GiB of disk is free" in tight
    # The whole sentence, not its tail: "several GiB" alone passed with the two things
    # that fill a disk here replaced by "Files", which is the part that says how much a
    # run will need and of what.
    assert "Tokenised datasets and checkpoints can each run to several GiB" in tight


def test_doctor_does_not_comment_on_torch_compile(capsys: pytest.CaptureFixture[str]) -> None:
    """It used to say training was "somewhat slower" without Triton. It is not.

    Nothing in ``src`` calls ``torch.compile``, so its absence costs nothing and its
    presence buys nothing. A note either way invents a cost, and staying silent on a
    machine that *has* Triton would leave the user assuming compilation happened.
    Both machines are checked because only one of them has it.
    """
    windows = rtx_3070_windows()
    linux = rtx_4090()
    assert not windows.torch_compile_available, "fixture drifted; this test needs a machine without"
    assert linux.torch_compile_available, "fixture drifted; this test needs a machine with"

    for profile in (windows, linux):
        report = render_doctor(profile, capsys)
        assert "compile" not in report.lower(), report
        assert "Triton" not in report


def test_nothing_promises_torch_compile_unless_something_calls_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The guard behind the note above.

    ``compile_model`` was a documented ``TrainConfig`` field offering to "try
    ``torch.compile``" that nothing read, and ``doctor`` told users training was
    slower without Triton. Both were removed. This checks the API surface rather
    than scanning for words, so the comments explaining *why* they are gone do not
    trip it: no configuration field offering compilation, and no machine whose report
    mentions it. If compilation is wired up for real, the ``torch.compile(`` call
    site lifts the restriction.
    """
    from trainai.train.config import TrainConfig

    root = Path(__file__).resolve().parent.parent / "src" / "trainai"
    implemented = any(
        "torch.compile(" in path.read_text(encoding="utf-8") for path in root.rglob("*.py")
    )
    if implemented:
        return  # real now; the promise is earned

    offered = [name for name in TrainConfig.__dataclass_fields__ if "compile" in name]
    assert not offered, f"TrainConfig offers {offered} but nothing calls torch.compile"

    for label, machine in ALL_MACHINES.items():
        report = render_doctor(machine(), capsys)
        assert "compile" not in report.lower(), f"{label} report mentions compiling"


# --------------------------------------------------------------------------- #
# The documents count these machines out loud
# --------------------------------------------------------------------------- #
#: Enough of the number line for a fixture list that has grown by one at a time.
NUMBER_WORDS = {
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
}

#: Documents that state the size of this fixture list in words.
COUNTING_DOCUMENTS = ("README.md", "docs/hardware-support.md")


@pytest.mark.parametrize("relative", COUNTING_DOCUMENTS)
def test_the_documents_count_these_machines_correctly(relative: str) -> None:
    """Regression in spirit: the README claimed CI had never run, months after it had.

    A number written out in prose is the kind of claim that goes stale silently, and both
    of these documents lean on this one -- it is the evidence offered for supporting
    hardware nobody here owns. Adding a fixture without touching the prose turns that
    evidence into an understatement; deleting one turns it into a lie.

    Asserted as "the right word is present and no neighbouring word is" rather than by
    parsing the sentence, because the sentence should stay free to be rewritten.
    """
    root = Path(__file__).resolve().parent.parent
    count = len(ALL_MACHINES)
    assert count in NUMBER_WORDS, f"extend NUMBER_WORDS past {count}"

    text = (root / relative).read_text(encoding="utf-8").lower()
    expected = NUMBER_WORDS[count]

    assert f"{expected} machines" in text, (
        f"{relative} does not say it tests {expected} machines, and "
        f"tests/test_hardware_portability.py has {count}"
    )
    wrong = sorted(
        word
        for number, word in NUMBER_WORDS.items()
        if number != count and f"{word} machines" in text
    )
    assert not wrong, f"{relative} also claims {wrong} machines"


def test_the_readme_counts_the_amd_cards_it_offers_as_evidence() -> None:
    """The ROCm claim is the one a reader is most likely to be checking.

    AMD is the largest "should work, nobody has run it" gap in the project, so the number
    of AMD profiles is the specific figure the README puts up in its place. It is derived
    from the backend here rather than from a second hand-written list.
    """
    root = Path(__file__).resolve().parent.parent
    rocm = sum(1 for build in ALL_MACHINES.values() if build().backend == "rocm")
    readme = (root / "README.md").read_text(encoding="utf-8").lower()
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}

    assert f"{words[rocm]} amd cards" in readme, (
        f"the README does not say it tests {words[rocm]} AMD cards, and there are {rocm} "
        "profiles with a rocm backend"
    )


# --------------------------------------------------------------- asynchronous backends


class _Recorder:
    """Stands in for ``torch.mps`` / ``torch.xpu``, which this machine does not have."""

    def __init__(self) -> None:
        self.calls = 0

    def synchronize(self, *args: object) -> None:
        self.calls += 1


@pytest.mark.parametrize("backend", ["mps", "xpu"])
def test_the_training_loop_waits_for_an_asynchronous_backend(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timed step must measure work, not queue depth, on every async backend.

    CUDA is not the only one. MPS and XPU also return before the kernels finish, so a
    ``perf_counter`` pair around an unsynchronised step on an M-series Mac times the
    dispatch -- and every tokens/second figure and ETA the loop prints comes from that
    pair. Before this test, the loop synchronised CUDA only, so the one number a user
    on unverified hardware would quote back was the one that had never been measured.
    """
    from trainai.train.loop import synchronize

    recorder = _Recorder()
    monkeypatch.setattr(torch, backend, recorder, raising=False)

    synchronize(torch.device(backend))

    assert recorder.calls == 1, f"the loop did not wait for {backend}"


def test_waiting_for_a_device_never_raises() -> None:
    """A driver-level failure in a timing call must not end a training run.

    The next real operation will report it with a better message than "synchronize
    failed at step 8,412".
    """
    from trainai.train.loop import synchronize

    class Exploding:
        def synchronize(self, *args: object) -> None:
            raise RuntimeError("driver went away")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch, "mps", Exploding(), raising=False)
        synchronize(torch.device("mps"))

    synchronize(torch.device("cpu"))


def test_the_benchmark_and_the_loop_agree_on_what_a_step_took() -> None:
    """One helper, not two: the benchmark predicts what the loop then has to reproduce."""
    from trainai.hardware import benchmark
    from trainai.train.loop import synchronize

    recorder = _Recorder()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch, "mps", recorder, raising=False)
        benchmark._synchronize(torch.device("mps"))
        synchronize(torch.device("mps"))

    assert recorder.calls == 2, "the benchmark and the loop take different paths"


# ------------------------------------------------------------- Apple's memory reporting


class _FakeMPS:
    """Stands in for ``torch.mps``, whose real numbers this machine cannot produce.

    Every value is a constructor argument so a test states the machine it is describing.
    Passing ``None`` for a reader removes the attribute, which is how an older torch or a
    build without MPS actually looks.
    """

    def __init__(
        self,
        *,
        ceiling: int | None,
        held: int = 0,
        ceiling_raises: bool = False,
        held_reader: str = "driver_allocated_memory",
    ) -> None:
        self._ceiling = ceiling
        self._held = held
        self._ceiling_raises = ceiling_raises
        if ceiling is None and not ceiling_raises:
            # An older torch simply does not have the function. Shadowing the method with
            # None on the instance is what ``getattr(mps, name, None)`` sees as absent.
            self.recommended_max_memory = None  # type: ignore[assignment]
        # Exactly one of the two held-memory readers, because that is what a given torch
        # looks like rather than a machine with both. The probe tries them in order.
        absent = (
            "current_allocated_memory"
            if held_reader == "driver_allocated_memory"
            else "driver_allocated_memory"
        )
        setattr(self, absent, None)

    def recommended_max_memory(self) -> int:
        if self._ceiling_raises:
            raise RuntimeError("Metal query failed")
        assert self._ceiling is not None
        return self._ceiling

    def driver_allocated_memory(self) -> int:
        return self._held

    def current_allocated_memory(self) -> int:
        return self._held


def _mps_torch(fake: _FakeMPS | None) -> Any:
    """A ``torch`` stand-in exposing only what the Apple branch of the probe reads."""

    class FakeTorch:
        pass

    stub = FakeTorch()
    if fake is not None:
        stub.mps = fake  # type: ignore[attr-defined]
    return stub


def test_apple_memory_is_the_ceiling_minus_what_torch_already_holds() -> None:
    from trainai.hardware.probe import _probe_mps_device

    warnings: list[str] = []
    gpus = _probe_mps_device(
        _mps_torch(_FakeMPS(ceiling=18 * GIB, held=2 * GIB)),
        warnings,
        available_ram_bytes=64 * GIB,
    )

    assert len(gpus) == 1
    assert gpus[0].backend == "mps"
    assert gpus[0].total_vram_bytes == 18 * GIB
    assert gpus[0].free_vram_bytes == 16 * GIB
    assert warnings == []


def test_an_older_torch_that_names_held_memory_differently_is_still_read() -> None:
    """``driver_allocated_memory`` is the newer name; the older one is still read.

    Both are tried, in order, rather than the newer one alone -- and a torch that has
    only ``current_allocated_memory`` is not a hypothetical, it is any build from before
    the driver counter was exposed. Reading neither is not a warning either: ``held``
    stays 0 and the ceiling is handed out whole, so a process already holding 2 GiB is
    planned as though it held nothing, and the ceiling is spent twice.

    The sibling above covers the newer name. Between them the order of the two is fixed
    rather than incidental, which is the part worth keeping: the driver's figure is the
    larger of the two -- it counts cached blocks the allocator has not handed back -- so
    preferring it is the conservative reading, and swapping them would quietly raise
    every Apple budget by whatever torch is caching.
    """
    from trainai.hardware.probe import _probe_mps_device

    warnings: list[str] = []
    gpus = _probe_mps_device(
        _mps_torch(
            _FakeMPS(ceiling=18 * GIB, held=2 * GIB, held_reader="current_allocated_memory")
        ),
        warnings,
        available_ram_bytes=64 * GIB,
    )

    assert gpus[0].free_vram_bytes == 16 * GIB, "the absent reader ended the search"
    assert warnings == [], "a torch without the newer name is working, not broken"


def test_apple_memory_never_promises_more_than_the_os_has_free() -> None:
    """The unified-memory case, and the reason the minimum is taken.

    Metal's ceiling is a property of the machine and knows nothing about the browser.
    An 18 GiB ceiling on a box with 3 GiB free is a 3 GiB budget, and a plan that spent
    the ceiling would be planning against memory another process is holding.
    """
    from trainai.hardware.probe import _probe_mps_device

    gpus = _probe_mps_device(
        _mps_torch(_FakeMPS(ceiling=18 * GIB, held=0)), [], available_ram_bytes=3 * GIB
    )

    assert gpus[0].free_vram_bytes == 3 * GIB
    assert gpus[0].total_vram_bytes == 18 * GIB


def test_apple_without_a_readable_ceiling_keeps_no_opinion_and_says_so() -> None:
    """No ceiling means no budget -- and inventing one from total RAM is the wrong fix.

    Returning an empty list restores the old behaviour deliberately: the planner falls
    back to choosing by measurement, which is what it did on every Apple machine before
    this. The difference from before is that the user is told, instead of a silent 0
    that reads as "fits fine".
    """
    from trainai.hardware.probe import _probe_mps_device

    for fake in (_FakeMPS(ceiling=None), _FakeMPS(ceiling=None, ceiling_raises=True)):
        warnings: list[str] = []

        assert _probe_mps_device(_mps_torch(fake), warnings, available_ram_bytes=64 * GIB) == []
        assert warnings, "an unreadable ceiling was not reported"


def test_apple_reports_zero_usable_rather_than_a_negative_number() -> None:
    """A driver holding more than its own ceiling is possible; a negative budget is not."""
    from trainai.hardware.probe import _probe_mps_device

    gpus = _probe_mps_device(
        _mps_torch(_FakeMPS(ceiling=8 * GIB, held=12 * GIB)), [], available_ram_bytes=64 * GIB
    )

    assert gpus[0].free_vram_bytes == 0


def test_a_torch_without_mps_at_all_yields_no_apple_device() -> None:
    from trainai.hardware.probe import _probe_mps_device

    assert _probe_mps_device(_mps_torch(None), [], available_ram_bytes=64 * GIB) == []


def test_apple_peak_memory_is_still_not_claimed_as_measured() -> None:
    """The budget got real; the peak did not, and conflating them would be the regression.

    ``torch.mps`` has ``recommended_max_memory`` and ``current_allocated_memory`` but no
    ``max_memory_allocated`` and no ``reset_peak_memory_stats``. So a plan on Apple can
    now say whether a model fits, and still cannot say what it actually peaked at --
    those are different claims and the plan keeps them apart.
    """
    from trainai.hardware.benchmark import _peak_memory

    peak, reserved, measured = _peak_memory(torch.device("mps"))

    assert (peak, reserved, measured) == (0, 0, False)
    assert not hasattr(getattr(torch, "mps", object()), "max_memory_allocated")


# ------------------------------------------------------------- Intel's memory reporting


class _FakeXPUDevice:
    """One device's properties block, as ``torch.xpu.get_device_properties`` returns it."""

    def __init__(self, name: str, total_memory: int) -> None:
        self.name = name
        self.total_memory = total_memory


class _FakeXPU:
    """Stands in for ``torch.xpu``, which this machine does not have.

    Built like :class:`_FakeMPS`: every number is a constructor argument, so a test
    states the machine it is describing rather than patching a global. ``free=None``
    makes ``mem_get_info`` raise, which is how a torch build without the newer memory
    query actually behaves, and ``bf16=None`` removes the query altogether, which is
    how an older one looks.
    """

    def __init__(
        self,
        *,
        devices: list[_FakeXPUDevice],
        free: list[int] | None = None,
        available: bool = True,
        bf16: bool | None = False,
    ) -> None:
        self._devices = devices
        self._free = free
        self._available = available
        self._bf16 = bool(bf16)
        if bf16 is None:
            self.is_bf16_supported = None  # type: ignore[assignment]

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return len(self._devices)

    def get_device_properties(self, index: int) -> _FakeXPUDevice:
        return self._devices[index]

    def mem_get_info(self, index: int) -> tuple[int, int]:
        if self._free is None:
            raise RuntimeError("this torch build has no XPU memory query")
        return self._free[index], self._devices[index].total_memory

    def is_bf16_supported(self) -> bool:
        return self._bf16


def _xpu_torch(fake: _FakeXPU | None) -> Any:
    """A ``torch`` stand-in exposing only what the Intel branch of the probe reads."""

    class FakeTorch:
        pass

    stub = FakeTorch()
    if fake is not None:
        stub.xpu = fake  # type: ignore[attr-defined]
    return stub


def test_intel_memory_is_read_from_the_runtime_not_the_properties_block() -> None:
    """Free VRAM comes from ``mem_get_info``, which knows what is already allocated.

    ``total_memory`` on the properties block is the card's size and never changes. A
    plan built from it on a card already holding another process's model would promise
    memory that is not there, which is the same overcommit the Apple branch avoids.
    """
    from trainai.hardware.probe import _probe_xpu_devices

    warnings: list[str] = []
    gpus = _probe_xpu_devices(
        _xpu_torch(
            _FakeXPU(
                devices=[
                    _FakeXPUDevice("Intel(R) Arc(TM) A770 Graphics", 16 * GIB),
                    _FakeXPUDevice("Intel(R) Arc(TM) A750 Graphics", 8 * GIB),
                ],
                free=[12 * GIB, 8 * GIB],
            )
        ),
        warnings,
    )

    assert [g.index for g in gpus] == [0, 1]
    assert [g.name for g in gpus] == [
        "Intel(R) Arc(TM) A770 Graphics",
        "Intel(R) Arc(TM) A750 Graphics",
    ]
    assert [g.total_vram_bytes for g in gpus] == [16 * GIB, 8 * GIB]
    assert [g.free_vram_bytes for g in gpus] == [12 * GIB, 8 * GIB]
    assert {g.backend for g in gpus} == {"xpu"}
    assert warnings == []


def test_intel_without_a_free_memory_query_falls_back_to_total_and_says_so() -> None:
    """The fallback is the card's full size, and that is exactly why it is warned about.

    Planning against total memory on a card that is already busy overcommits. The
    number is still the best available, so the probe keeps it -- but a user reading a
    plan that turns out not to fit deserves to have been told which number it came
    from, rather than discovering it as an out-of-memory error at step 1.
    """
    from trainai.hardware.probe import _probe_xpu_devices

    warnings: list[str] = []
    gpus = _probe_xpu_devices(
        _xpu_torch(_FakeXPU(devices=[_FakeXPUDevice("Intel(R) Arc(TM) A770", 16 * GIB)])),
        warnings,
    )

    assert gpus[0].total_vram_bytes == 16 * GIB
    assert gpus[0].free_vram_bytes == 16 * GIB, "the fallback is total, not zero"
    assert len(warnings) == 1
    assert "overcommit" in warnings[0], warnings


def test_a_torch_without_xpu_at_all_yields_no_intel_device() -> None:
    """The common case: every mainstream PyTorch wheel has no ``torch.xpu``."""
    from trainai.hardware.probe import _probe_xpu_devices

    warnings: list[str] = []

    assert _probe_xpu_devices(_xpu_torch(None), warnings) == []
    assert warnings == [], "an absent backend is not a problem worth reporting"


def test_an_xpu_build_with_no_intel_card_present_yields_no_intel_device() -> None:
    """A build *with* XPU support on a machine without the hardware is not an error.

    Distinguished from the test above on purpose: that one is a missing module, this one
    is a present module answering "no". Both must return empty, and neither is a warning
    -- a user on an NVIDIA box running an XPU-capable wheel has nothing to fix.
    """
    from trainai.hardware.probe import _probe_xpu_devices

    warnings: list[str] = []
    fake = _FakeXPU(devices=[_FakeXPUDevice("unused", 16 * GIB)], available=False)

    assert _probe_xpu_devices(_xpu_torch(fake), warnings) == []
    assert warnings == []


def test_intel_bf16_support_is_read_from_the_runtime() -> None:
    """Read, not inferred from the device name.

    The Apple branch hardcodes ``False`` and the NVIDIA branch infers from compute
    capability; Intel exposes a direct query, so the probe asks instead of guessing.
    """
    from trainai.hardware.probe import _probe_xpu_bf16

    devices = [_FakeXPUDevice("Intel(R) Arc(TM) A770", 16 * GIB)]

    assert _probe_xpu_bf16(_xpu_torch(_FakeXPU(devices=devices, bf16=True))) is True
    assert _probe_xpu_bf16(_xpu_torch(_FakeXPU(devices=devices, bf16=False))) is False


def test_an_older_torch_without_the_bf16_query_reports_no_bf16() -> None:
    """Absent query means no claim, and no claim has to mean ``False``.

    Defaulting to ``True`` would hand the training loop an autocast dtype the runtime
    may not implement, and the failure would surface as a kernel error mid-run rather
    than as a plan that chose fp32.
    """
    from trainai.hardware.probe import _probe_xpu_bf16

    devices = [_FakeXPUDevice("Intel(R) Arc(TM) A770", 16 * GIB)]

    assert _probe_xpu_bf16(_xpu_torch(_FakeXPU(devices=devices, bf16=None))) is False
    assert _probe_xpu_bf16(_xpu_torch(None)) is False, "no torch.xpu at all"


# ------------------------------------------------------------------- which branch runs
#
# The four probes above are each tested on their own, and until now nothing tested the
# dispatch that chooses between them. That gap is not academic: `_probe_accelerators`
# returns the `device_type` and `backend` every later decision reads, and a routing
# mistake would leave all four probes passing. It is also invisible from this machine,
# which has a CUDA card and therefore takes the first branch every time -- and from CI,
# which has none and never reaches the numbers.
#
# The `HardwareInfo` builder at the top of this file already encodes the answer for
# ROCm -- `device_type={"rocm": "cuda"}.get(backend, backend)` -- so every planner test
# is written against an asymmetry that nothing checked the probe actually produces.


class _FakeCUDADevice:
    """One device's properties block, as ``torch.cuda.get_device_properties`` returns it.

    ``major``/``minor`` are the compute capability on NVIDIA and the gfx architecture on
    ROCm, which is the whole difficulty the dispatch exists to handle: the same two
    fields, read the same way, meaning different things.
    """

    def __init__(
        self, name: str, total_memory: int, major: int, minor: int, sm_count: int = 0
    ) -> None:
        self.name = name
        self.total_memory = total_memory
        self.major = major
        self.minor = minor
        self.multi_processor_count = sm_count


class _FakeCUDA:
    """Stands in for ``torch.cuda``, which a ROCm build also answers to."""

    def __init__(
        self,
        *,
        devices: list[_FakeCUDADevice] | None = None,
        available: bool = True,
        bf16: bool = True,
        free: list[int] | None = None,
        free_raises: Exception | None = None,
    ) -> None:
        self._devices = devices or []
        self._available = available
        self._bf16 = bf16
        self._free = free
        self._free_raises = free_raises

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return len(self._devices)

    def get_device_properties(self, index: int) -> _FakeCUDADevice:
        return self._devices[index]

    def mem_get_info(self, index: int) -> tuple[int, int]:
        if self._free_raises is not None:
            raise self._free_raises
        total = self._devices[index].total_memory
        return (total if self._free is None else self._free[index], total)

    def is_bf16_supported(self) -> bool:
        return self._bf16


class _FakeMPSBackend:
    """``torch.backends.mps``, which answers only the availability question.

    Distinct from :class:`_FakeMPS`, which stands in for ``torch.mps``: the dispatch
    reads the first to decide, and :func:`_probe_mps_device` reads the second for the
    numbers. A machine needs both to be routed to Apple and then measured.
    """

    def __init__(self, *, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _Untouchable:
    """Any attribute access fails, which is how a test proves a branch was not taken.

    It fails through ``pytest.fail`` rather than by raising ``AssertionError`` on purpose.
    ``_probe_xpu_devices`` treats a torch that answers badly as a machine without an Intel
    card -- it catches ``Exception`` and returns no devices -- so an ``AssertionError`` from
    here would be swallowed and the reach into Intel would go unreported. ``pytest.fail``
    raises an ``OutcomeException``, which derives from ``BaseException`` and passes straight
    through that guard.
    """

    def __init__(self, label: str) -> None:
        self._label = label

    def __getattr__(self, name: str) -> Any:
        pytest.fail(f"the {self._label} branch was consulted: .{name}", pytrace=False)


def _routing_torch(
    *,
    cuda: _FakeCUDA | None = None,
    hip: str | None = None,
    xpu: Any = None,
    mps_backend: _FakeMPSBackend | None = None,
    mps: _FakeMPS | None = None,
) -> Any:
    """A ``torch`` exposing exactly what the dispatch reads, and nothing it does not.

    ``torch.backends`` is always present: leaving it off would send the Apple check into
    the ``except Exception`` guard and route to CPU for the wrong reason, which would
    make a passing test out of a broken dispatch.
    """

    class FakeVersion:
        pass

    class FakeBackends:
        pass

    class FakeTorch:
        pass

    version = FakeVersion()
    version.hip = hip  # type: ignore[attr-defined]
    backends = FakeBackends()
    if mps_backend is not None:
        backends.mps = mps_backend  # type: ignore[attr-defined]
    stub = FakeTorch()
    stub.version = version  # type: ignore[attr-defined]
    stub.backends = backends  # type: ignore[attr-defined]
    stub.cuda = cuda or _FakeCUDA(available=False)  # type: ignore[attr-defined]
    if xpu is not None:
        stub.xpu = xpu  # type: ignore[attr-defined]
    if mps is not None:
        stub.mps = mps  # type: ignore[attr-defined]
    return stub


def _route(torch: Any, **kwargs: Any) -> tuple[Any, list[str]]:
    """``_probe_accelerators`` with its warnings list, since both are the answer."""
    from trainai.hardware.probe import _probe_accelerators

    warnings: list[str] = []
    return _probe_accelerators(torch, warnings, **kwargs), warnings


AMPERE = _FakeCUDADevice("NVIDIA GeForce RTX 3090", 24 * GIB, major=8, minor=6, sm_count=82)
#: gfx908, an MI100. (9, 0) clears the NVIDIA bf16 rule, and the number means nothing here.
GFX908 = _FakeCUDADevice("AMD Instinct MI100", 32 * GIB, major=9, minor=0, sm_count=120)


def test_a_cuda_build_routes_to_cuda_and_reports_tf32_on_an_ampere_card() -> None:
    """The ordinary case, and the baseline the ROCm tests below differ from by one field."""
    (device_type, backend, gpus, bf16, tf32), warnings = _route(
        _routing_torch(cuda=_FakeCUDA(devices=[AMPERE]))
    )

    assert (device_type, backend) == ("cuda", "cuda")
    assert [gpu.name for gpu in gpus] == ["NVIDIA GeForce RTX 3090"]
    assert (bf16, tf32) == (True, True)
    assert warnings == [], f"an NVIDIA card is the supported case and needs no caveat: {warnings}"


def test_a_rocm_build_keeps_the_cuda_device_type_and_changes_only_the_backend() -> None:
    """``torch.version.hip`` is the only reliable way to tell the builds apart.

    A ROCm build reports CUDA as available and serves AMD cards through the entire
    ``torch.cuda`` API, so the device type stays ``"cuda"`` -- that is what selects the
    code path, and the path is shared. ``backend`` is what carries "this is AMD" to
    everything that has to treat it differently.

    The asymmetry is worth pinning here rather than trusting: the ``HardwareInfo``
    builder at the top of this file hard-codes it for every planner test, so if the
    probe ever produced ``("rocm", "rocm")`` the fixtures would keep agreeing with
    themselves and disagreeing with the machine.
    """
    (device_type, backend, gpus, _, _), warnings = _route(
        _routing_torch(cuda=_FakeCUDA(devices=[GFX908]), hip="6.2.41134")
    )

    assert (device_type, backend) == ("cuda", "rocm")
    assert [gpu.backend for gpu in gpus] == ["rocm"]
    assert any("ROCm detected" in warning for warning in warnings), warnings


def test_rocm_reports_no_tf32_however_high_the_architecture_number_reads() -> None:
    """TF32 is an NVIDIA tensor-core format. There is no AMD equivalent to detect.

    This is the test the dispatch most needs, because the mistake is silent and the
    numbers invite it: gfx908 fills ``major``/``minor`` with ``(9, 0)``, which clears the
    ``>= (8, 0)`` NVIDIA rule comfortably. Drop the ``not is_rocm`` guard and an MI100
    is reported as a TF32 device, and whatever acts on that flag asks for a format the
    card has never had.

    bf16 is the contrast, and it is why this is a routing question rather than a
    capability one: the same card *does* report bf16, because that answer comes from the
    runtime rather than from the architecture number.
    """
    (_, backend, _, bf16, tf32), _ = _route(
        _routing_torch(cuda=_FakeCUDA(devices=[GFX908], bf16=True), hip="6.2.41134")
    )

    assert backend == "rocm"
    assert bf16 is True, "the runtime was asked and said yes"
    assert tf32 is False, "an architecture number that clears an NVIDIA rule is not TF32"


def test_an_intel_card_is_routed_to_xpu_only_when_cuda_is_unavailable() -> None:
    """Intel is reached by falling through, so what comes before it decides.

    ``tf32`` is ``False`` by construction on this path rather than by measurement, and it
    has to be: there is no query to ask, and inferring it from the Arc generation would
    be the ROCm mistake again in a different vendor's numbers.
    """
    xpu = _FakeXPU(devices=[_FakeXPUDevice("Intel(R) Arc(TM) A770", 16 * GIB)], bf16=True)

    (device_type, backend, gpus, bf16, tf32), warnings = _route(_routing_torch(xpu=xpu))

    assert (device_type, backend) == ("xpu", "xpu")
    assert [gpu.backend for gpu in gpus] == ["xpu"]
    assert (bf16, tf32) == (True, False)
    assert any("Intel XPU detected" in warning for warning in warnings), warnings


def test_a_discrete_nvidia_card_wins_over_the_intel_graphics_beside_it() -> None:
    """The laptop case: an NVIDIA GPU and Intel integrated graphics in one machine.

    Both are real and both are visible to torch, so the order in the dispatch is the
    whole answer, and returning early is how it is expressed. ``_Untouchable`` makes that
    a fact rather than a reading of the source: if the Intel branch is ever moved above
    the CUDA one, this fails with the attribute it reached for instead of quietly
    planning a training run on integrated graphics.
    """
    (device_type, backend, _, _, _), _ = _route(
        _routing_torch(cuda=_FakeCUDA(devices=[AMPERE]), xpu=_Untouchable("Intel"))
    )

    assert (device_type, backend) == ("cuda", "cuda")


def test_apple_is_routed_to_mps_and_claims_neither_bf16_nor_tf32() -> None:
    """Both flags are hard ``False`` here, and neither is a measurement.

    The probe's own warning says autocast on MPS is not something this project has
    verified, so the honest report is no bf16 -- and a plan that trains in fp32 on Apple
    is the intended consequence, not a gap. Routing that returned ``True`` for either
    would send a run into a path nobody has checked, on the strength of nothing.
    """
    (device_type, backend, gpus, bf16, tf32), warnings = _route(
        _routing_torch(
            mps_backend=_FakeMPSBackend(available=True),
            mps=_FakeMPS(ceiling=18 * GIB, held=2 * GIB),
        )
    )

    assert (device_type, backend) == ("mps", "mps")
    assert [gpu.backend for gpu in gpus] == ["mps"]
    assert (bf16, tf32) == (False, False)
    assert any("Apple MPS detected" in warning for warning in warnings), warnings


def test_an_xpu_build_with_no_intel_card_falls_through_to_apple() -> None:
    """Presence of ``torch.xpu`` is not presence of a device, and the dispatch tests the
    device list rather than the module.

    An Intel-enabled wheel installed on a Mac is not a contrived machine -- it is what a
    single "install PyTorch with every backend" build looks like -- and routing on the
    import would send it to a backend with no hardware under it.
    """
    (device_type, _, _, _, _), _ = _route(
        _routing_torch(
            xpu=_FakeXPU(devices=[]),
            mps_backend=_FakeMPSBackend(available=True),
            mps=_FakeMPS(ceiling=18 * GIB),
        )
    )

    assert device_type == "mps"


def test_a_machine_with_no_accelerator_reports_cpu_and_says_nothing_about_it() -> None:
    """The CPU fall-through, which every branch above has to decline first.

    The empty warning list is half the test. Each accelerator branch appends a caveat,
    so a fall-through that had touched one would leave its warning behind for a machine
    that has no GPU to caveat -- and CPU is a supported configuration here, not a
    degraded one worth a message.
    """
    (device_type, backend, gpus, bf16, tf32), warnings = _route(
        _routing_torch(xpu=_FakeXPU(devices=[]), mps_backend=_FakeMPSBackend(available=False))
    )

    assert (device_type, backend) == ("cpu", "cpu")
    assert gpus == []
    assert (bf16, tf32) == (False, False)
    assert warnings == [], f"nothing to warn about on a CPU-only machine: {warnings}"


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        # A torch old enough not to have the call at all, and a driver that has it and
        # fails it. The probe cannot tell the two apart and does not need to: either way
        # the free figure is unavailable and total is the only number left.
        pytest.param(AttributeError, "module 'torch.cuda' has no attribute", id="absent"),
        pytest.param(RuntimeError, "CUDA error: initialization error", id="fails"),
    ],
)
def test_cuda_without_a_readable_free_figure_falls_back_to_total_and_says_so(
    failure: type[Exception], message: str
) -> None:
    """The fallback overcommits by construction, which is why it is warned about.

    ``mem_get_info`` is what makes the CUDA budget honest -- it reports what the driver
    will hand over right now, desktop compositor and other processes included -- and
    ``total_memory`` is the card's size, which is true and unhelpful. Planning against
    the sticker figure on a card already holding something else promises memory that is
    not there. The number is still the best one available so the probe keeps it, and the
    exception text goes in the warning: a user reading a plan that then does not fit
    should be able to see which number it was built from.

    The exact sibling for Intel is
    :func:`test_intel_without_a_free_memory_query_falls_back_to_total_and_says_so`, and
    the pair is deliberate -- the two backends read different APIs to answer the same
    question, and both degrade the same way.
    """
    (_, _, gpus, _, _), warnings = _route(
        _routing_torch(cuda=_FakeCUDA(devices=[AMPERE], free_raises=failure(message)))
    )

    assert gpus[0].total_vram_bytes == 24 * GIB
    assert gpus[0].free_vram_bytes == 24 * GIB, "the fallback is total, not zero"
    assert len(warnings) == 1, warnings
    assert "device 0" in warnings[0], "an eight-GPU box needs to know which one"
    assert message in warnings[0], warnings
    assert "overcommit" in warnings[0], warnings


# ------------------------------------------------------------ assembling a whole profile
#
# Everything above tests one probe. `probe_hardware` calls a dozen of them and then makes
# two decisions of its own with what comes back, and neither is reachable from a machine
# this suite has ever run on: one needs a ROCm build, the other needs a GPU card sitting
# next to a PyTorch that cannot use it. Both are stated here rather than sampled, for the
# same reason the vendor detector is -- see `fake_machine` -- and both are what a user
# reads first when something is wrong, in `doctor`'s report.


def probe_with(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    torch: Any,
    vendors: list[str],
) -> HardwareProfile:
    """``probe_hardware`` against a stated PyTorch and a stated set of installed cards.

    ``sys.modules`` is the seam because ``_try_import_torch`` imports inside the function
    -- which is what makes an unimportable torch a warning instead of a crash -- so there
    is no module attribute to reach for. ``detect_gpu_vendors`` is patched rather than
    called for the reason the whole vendor section exists: it reads this machine, so on
    the developer's box it answers ``["amd", "nvidia"]`` and in CI ``[]``, and a test
    whose subject is what the answer is *used for* cannot also depend on it.

    The CPU, RAM and disk probes are left reading the real machine. They have their own
    tests above, and none of the three feeds either decision under test here.
    """
    # Read into `torch_version` and reported, but not part of either decision below.
    torch.__version__ = "2.5.1"
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr("trainai.hardware.probe.detect_gpu_vendors", lambda: vendors)
    return probe_hardware(disk_path=tmp_path)


def test_a_rocm_build_reports_its_runtime_version_instead_of_no_cuda_at_all(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``torch.version.cuda`` is genuinely empty on ROCm, and printing that misleads.

    The wheel was not built against CUDA, so the field is None and truthful; what the
    machine is actually running is in ``torch.version.hip``. `doctor` prints this line,
    and an AMD user with a working card reading "CUDA: not available" underneath a
    detected GPU has been told their PyTorch has no GPU support compiled in -- which is
    wrong, and is the one thing they would then go and try to fix.

    The routing sibling above pins that ``backend`` becomes ``"rocm"`` while
    ``device_type`` stays ``"cuda"``; this is the reporting half of the same fork, and it
    is the half that reaches a person.
    """
    profile = probe_with(
        monkeypatch,
        tmp_path,
        torch=_routing_torch(cuda=_FakeCUDA(devices=[GFX908]), hip="6.2.41134"),
        vendors=["amd"],
    )

    assert (profile.device_type, profile.backend) == ("cuda", "rocm")
    assert profile.cuda_version == "ROCm 6.2.41134"


def test_a_card_the_installed_torch_cannot_use_is_named_along_with_the_fix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The commonest broken install there is, and it looks exactly like having no GPU.

    ``pip install torch`` gives a CPU-only wheel on Linux with no index URL, so the card
    is present, the driver is fine, and training silently runs on the CPU at a fraction of
    the speed. The two situations -- no GPU, and a GPU torch cannot see -- produce the
    same symptom and have completely different remedies, so the profile carries the
    difference in ``has_unusable_gpu`` and the warning names the vendor and the command.

    The second half is the guard: a machine that really has no GPU must not be handed an
    install instruction for one. That is the ``vendors and`` in the condition, and without
    it every CPU-only CI runner in the world gets told to go and fix its PyTorch.
    """
    stranded = probe_with(
        monkeypatch,
        tmp_path,
        torch=_routing_torch(
            xpu=_FakeXPU(devices=[]), mps_backend=_FakeMPSBackend(available=False)
        ),
        vendors=["nvidia"],
    )

    assert (stranded.backend, stranded.gpus) == ("cpu", [])
    assert stranded.has_unusable_gpu is True
    advice = [warning for warning in stranded.warnings if "trainai setup" in warning]
    assert len(advice) == 1, stranded.warnings
    assert "An NVIDIA GPU is present" in advice[0]
    assert "CPU-only" in advice[0], "the likely cause is the whole value of the message"

    cpu_only_machine = probe_with(
        monkeypatch,
        tmp_path,
        torch=_routing_torch(
            xpu=_FakeXPU(devices=[]), mps_backend=_FakeMPSBackend(available=False)
        ),
        vendors=[],
    )

    assert cpu_only_machine.has_unusable_gpu is False
    assert not [w for w in cpu_only_machine.warnings if "trainai setup" in w], (
        cpu_only_machine.warnings
    )
