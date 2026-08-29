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
import stat
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from conftest import flat
from trainai.errors import TrainingError
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
    _probe_cpu_limit,
    _probe_ram_limit,
    describe_unusable_gpu,
    detect_gpu_vendors,
    probe_hardware,
    vendor_labels,
)
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
    """MPS exposes no per-device VRAM, so gpus is empty by design."""
    return machine(
        "macOS-14 / Apple M2",
        gpus=[],
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


@pytest.mark.parametrize("label", ["apple_m2", "cpu_only", "nvidia_with_cpu_wheel"])
def test_a_machine_with_no_usable_gpu_has_no_vram_budget(label: str) -> None:
    assert ALL_MACHINES[label]().vram_budget_bytes() == 0


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
    assert "ROCm" in " ".join(advice.notes)


def test_amd_on_windows_is_told_the_truth_rather_than_given_a_broken_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PyTorch publishes no ROCm build for Windows. Saying so beats a 404."""
    patch_environment(
        monkeypatch, vendors=["amd"], system="Windows", torch_cuda=None, accelerator=False
    )

    advice = advise_install(["amd"])

    assert advice.command is None
    notes = " ".join(advice.notes)
    assert "no ROCm build for Windows" in notes
    assert "WSL2" in notes


def test_intel_gets_an_xpu_index_url(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_environment(
        monkeypatch, vendors=["intel"], system="Linux", torch_cuda=None, accelerator=False
    )

    advice = advise_install(["intel"])

    assert advice.command is not None
    assert "whl/xpu" in advice.command


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
# The real machine, whatever it is
# --------------------------------------------------------------------------- #
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
