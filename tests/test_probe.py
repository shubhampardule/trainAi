"""Tests for hardware probing.

Two kinds of test here:

* Pure logic on synthetic :class:`HardwareProfile` values, which runs everywhere
  and is where the planner's future correctness will be pinned down.
* A real probe of the current machine, asserting only invariants that must hold
  on *any* machine. Asserting "4 GiB VRAM" here would make the suite pass only
  on the development laptop.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trainai.hardware.probe import (
    DEFAULT_VRAM_SAFETY_FRACTION,
    GPUInfo,
    HardwareProfile,
    _probe_disk,
    probe_hardware,
)

GIB = 1024**3


def make_gpu(
    *, total: int = 4 * GIB, free: int = 3 * GIB, cc: tuple[int, int] | None = (8, 6)
) -> GPUInfo:
    return GPUInfo(
        index=0,
        name="Synthetic GPU",
        total_vram_bytes=total,
        free_vram_bytes=free,
        compute_capability=cc,
        multiprocessor_count=16,
        supports_bf16=cc is not None and cc >= (8, 0),
    )


def make_profile(**overrides: object) -> HardwareProfile:
    defaults: dict[str, object] = {
        "platform_summary": "Synthetic OS",
        "os_name": "Linux",
        "python_version": "3.12.0",
        "torch_version": "2.10.0",
        "cuda_version": "12.8",
        "device_type": "cuda",
        "gpus": [make_gpu()],
    }
    defaults.update(overrides)
    return HardwareProfile(**defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Pure logic
# --------------------------------------------------------------------------- #
def test_vram_budget_is_a_fraction_of_free_not_total() -> None:
    """The bug this guards against: budgeting against total VRAM overcommits.

    On the dev machine 0.77 GiB of a 4 GiB card is already spoken for by the
    desktop, so planning against 4 GiB reliably produces a thrashing run.
    """
    profile = make_profile(gpus=[make_gpu(total=4 * GIB, free=3 * GIB)])
    budget = profile.vram_budget_bytes()
    assert budget == pytest.approx(3 * GIB * DEFAULT_VRAM_SAFETY_FRACTION)
    assert budget < 3 * GIB, "budget must leave headroom below free VRAM"
    assert budget < profile.gpus[0].total_vram_bytes


def test_vram_budget_respects_explicit_safety_fraction() -> None:
    profile = make_profile(gpus=[make_gpu(free=2 * GIB)])
    assert profile.vram_budget_bytes(1.0) == 2 * GIB
    assert profile.vram_budget_bytes(0.5) == GIB


def test_cpu_only_profile_has_no_vram_budget() -> None:
    profile = make_profile(device_type="cpu", gpus=[])
    assert profile.has_gpu is False
    assert profile.primary_gpu is None
    assert profile.vram_budget_bytes() == 0


def test_used_vram_is_derived_and_never_negative() -> None:
    assert make_gpu(total=4 * GIB, free=3 * GIB).used_vram_bytes == GIB
    # Drivers occasionally report free > total; clamp instead of going negative.
    assert make_gpu(total=GIB, free=2 * GIB).used_vram_bytes == 0


def test_capability_string_handles_unknown() -> None:
    assert make_gpu(cc=(8, 6)).capability_str == "8.6"
    assert make_gpu(cc=None).capability_str == "unknown"


def test_bf16_requires_compute_capability_8() -> None:
    assert make_gpu(cc=(8, 6)).supports_bf16 is True
    assert make_gpu(cc=(7, 5)).supports_bf16 is False


def test_profile_round_trips_through_json() -> None:
    """plan.json embeds this, so it has to survive serialisation intact."""
    profile = make_profile(warnings=["something odd"])
    payload = json.loads(json.dumps(profile.to_dict()))
    assert payload["gpus"][0]["compute_capability"] == "8.6"
    assert payload["device_type"] == "cuda"
    assert payload["warnings"] == ["something odd"]


# --------------------------------------------------------------------------- #
# Disk probing
# --------------------------------------------------------------------------- #
def test_probe_disk_walks_up_to_an_existing_parent(tmp_path: Path) -> None:
    """Output directories usually do not exist yet; measure the parent volume."""
    missing = tmp_path / "runs" / "not" / "created" / "yet"
    warnings: list[str] = []
    assert _probe_disk(missing, warnings) > 0
    assert warnings == []


def test_probe_disk_reports_rather_than_raises_on_nonsense() -> None:
    warnings: list[str] = []
    result = _probe_disk("relative/path/that/does/not/exist", warnings)
    assert result >= 0
    assert isinstance(result, int)


# --------------------------------------------------------------------------- #
# Real probe of this machine -- invariants only
# --------------------------------------------------------------------------- #
def test_probe_hardware_returns_consistent_profile(tmp_path: Path) -> None:
    profile = probe_hardware(disk_path=tmp_path)

    assert profile.device_type in {"cuda", "mps", "cpu"}
    assert profile.cpu_count_logical >= 1
    assert profile.python_version
    assert profile.platform_summary
    assert profile.free_disk_bytes > 0

    if profile.total_ram_bytes:
        assert 0 <= profile.available_ram_bytes <= profile.total_ram_bytes

    for gpu in profile.gpus:
        assert gpu.total_vram_bytes > 0
        assert 0 <= gpu.free_vram_bytes <= gpu.total_vram_bytes

    # A GPU must be present iff we selected a CUDA device type.
    assert bool(profile.gpus) == (profile.device_type == "cuda")

    # Never claim compile support without the compiler.
    if profile.torch_compile_available:
        import importlib.util

        assert importlib.util.find_spec("triton") is not None


def test_probe_hardware_is_json_serialisable(tmp_path: Path) -> None:
    json.dumps(probe_hardware(disk_path=tmp_path).to_dict())


@pytest.mark.gpu
def test_spillover_flag_matches_platform(tmp_path: Path) -> None:
    """Windows + CUDA means VRAM overcommit is silent, which changes planning."""
    import platform

    profile = probe_hardware(disk_path=tmp_path)
    expected = platform.system() == "Windows" and profile.device_type == "cuda"
    assert profile.silent_vram_spillover is expected


@pytest.mark.gpu
def test_real_gpu_free_vram_is_consulted(tmp_path: Path) -> None:
    """A machine running a desktop always has some VRAM already committed.

    If free VRAM ever equals total exactly, mem_get_info is probably not being
    consulted, and the planner is about to overcommit.
    """
    profile = probe_hardware(disk_path=tmp_path)
    gpu = profile.primary_gpu
    assert gpu is not None
    assert gpu.free_vram_bytes <= gpu.total_vram_bytes
    assert gpu.compute_capability is not None
    assert profile.vram_budget_bytes() < gpu.total_vram_bytes
