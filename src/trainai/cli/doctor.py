"""``trainai doctor`` -- print a truthful hardware report.

This is the first command a new user runs, so it does double duty: it tells them
what they have, and it tells them up front about the thing most likely to waste
their time: silent VRAM spillover.

What it deliberately does not do is comment on ``torch.compile``. The probe reports
whether it is usable as a fact about the machine, but TrainAI never calls it, so a
note saying training is "somewhat slower" without it would be inventing a cost that
does not exist.
"""

from __future__ import annotations

from trainai import __version__
from trainai.console import (
    DASH,
    console,
    emit_json,
    fmt_bytes,
    fmt_int,
    print_bullets,
    print_kv,
)
from trainai.hardware.probe import (
    DEFAULT_VRAM_SAFETY_FRACTION,
    HardwareProfile,
    gpu_pronoun,
    probe_hardware,
    vendor_labels,
)

__all__ = ["run_doctor"]

_YES = "[green]yes[/]"
_NO = "[red]no[/]"


def run_doctor(*, json_output: bool = False, disk_path: str = ".") -> HardwareProfile:
    """Probe the machine and print the report. Returns the profile for reuse."""
    profile = probe_hardware(disk_path=disk_path)

    if json_output:
        emit_json(profile.to_dict())
        return profile

    console.print()
    console.print(f"[bold]TrainAI {__version__}[/] {DASH} hardware report")
    console.print()
    print_kv("System", _system_rows(profile))
    print_kv("Compute", _compute_rows(profile))
    print_kv("Host", _host_rows(profile))
    _print_notes(profile)
    return profile


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _system_rows(profile: HardwareProfile) -> list[tuple[str, str]]:
    torch_desc = profile.torch_version or "[red]not installed[/]"
    if profile.torch_version and profile.cuda_version:
        torch_desc = f"{profile.torch_version}  [dim](CUDA {profile.cuda_version})[/]"
    elif profile.torch_version:
        torch_desc = f"{profile.torch_version}  [dim](CPU build)[/]"
    return [
        ("Platform", profile.platform_summary),
        ("Python", profile.python_version),
        ("PyTorch", torch_desc),
    ]


def _compute_rows(profile: HardwareProfile) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [("Device", f"[bold]{profile.device_type}[/]")]
    if profile.backend == "rocm":
        rows[0] = ("Device", "[bold]cuda[/]  [dim](AMD via ROCm)[/]")
    elif profile.backend != profile.device_type:  # pragma: no cover - defensive
        rows[0] = ("Device", f"[bold]{profile.device_type}[/]  [dim]({profile.backend})[/]")

    if not profile.has_gpu:
        if profile.has_unusable_gpu:
            vendors = ", ".join(vendor_labels(profile.detected_vendors))
            rows.append(
                (
                    "GPU present",
                    f"[yellow]{vendors}[/] {DASH} detected on this machine, but PyTorch "
                    f"cannot use {gpu_pronoun(profile.detected_vendors)}. "
                    "Run [bold]trainai setup[/] for the fix.",
                )
            )
        rows.append(
            (
                "Training speed",
                "[yellow]CPU only[/] " + DASH + " expect roughly 20-100x slower than a modern GPU",
            )
        )
        return rows

    for gpu in profile.gpus:
        label = f"GPU {gpu.index}"
        rows.append((label, f"[bold]{gpu.name}[/]"))
        used = gpu.used_vram_bytes
        if gpu.backend == "mps":
            # Neither number is a card's VRAM: the total is Metal's declared working-set
            # ceiling and the free figure is that ceiling minus what torch holds, capped
            # by what the OS says is actually free. Labelling it "VRAM ... free" would
            # read as a driver measurement of a dedicated pool, and there is no such
            # pool -- the GPU is spending the same RAM as everything else on the machine.
            rows.append(
                (
                    "  GPU memory",
                    f"[bold]{fmt_bytes(gpu.free_vram_bytes)}[/] usable of a "
                    f"{fmt_bytes(gpu.total_vram_bytes)} ceiling  [dim](unified with "
                    "system RAM)[/]",
                )
            )
        else:
            vram = (
                f"[bold]{fmt_bytes(gpu.free_vram_bytes)}[/] free of "
                f"{fmt_bytes(gpu.total_vram_bytes)}"
            )
            if used > 0:
                vram += f"  [dim]({fmt_bytes(used)} already in use)[/]"
            rows.append(("  VRAM", vram))
        rows.append(
            (
                "  Capability",
                f"{gpu.capability_str}   bf16 {_YES if gpu.supports_bf16 else _NO}"
                + (
                    ""
                    # TF32 is an NVIDIA tensor-core format; there is no AMD, Intel or
                    # Apple equivalent, so reporting "TF32 no" there would imply the
                    # hardware fell short of something rather than that the concept
                    # does not apply.
                    if profile.backend != "cuda"
                    else f"   TF32 {_YES if profile.supports_tf32 else _NO}"
                ),
            )
        )
        if gpu.multiprocessor_count:
            rows.append(("  SM count", str(gpu.multiprocessor_count)))

    budget = profile.vram_budget_bytes()
    rows.append(
        (
            "Training budget",
            f"[bold]{fmt_bytes(budget)}[/]  "
            f"[dim]({DEFAULT_VRAM_SAFETY_FRACTION:.0%} of free VRAM; the rest absorbs "
            f"fragmentation and driver overhead)[/]",
        )
    )
    return rows


def _host_rows(profile: HardwareProfile) -> list[tuple[str, str]]:
    cores = str(profile.cpu_count_logical)
    if profile.cpu_count_physical:
        cores = f"{profile.cpu_count_physical} physical / {profile.cpu_count_logical} logical"
    if profile.cpu_limit is not None:
        cores = f"{cores}, capped at {profile.cpu_limit:g}"
    rows = [("CPU", f"{profile.cpu_name or 'unknown'}  [dim]({cores})[/]")]

    if profile.ram_limit_bytes is not None:
        # The limit is the number that decides whether this run survives, so it leads,
        # and the host figure follows in dim as context rather than as the answer.
        rows.append(
            (
                "RAM",
                f"[bold]{fmt_bytes(profile.usable_available_ram_bytes)}[/] available "
                f"within a container limit of {fmt_bytes(profile.ram_limit_bytes)}"
                + (
                    f"  [dim]({fmt_bytes(profile.total_ram_bytes)} on the host)[/]"
                    if profile.total_ram_bytes
                    else ""
                ),
            )
        )
    elif profile.total_ram_bytes:
        rows.append(
            (
                "RAM",
                f"[bold]{fmt_bytes(profile.available_ram_bytes)}[/] available "
                f"of {fmt_bytes(profile.total_ram_bytes)}",
            )
        )
    else:
        rows.append(("RAM", "[dim]unknown[/]"))

    if profile.free_disk_bytes:
        rows.append(("Disk", f"[bold]{fmt_bytes(profile.free_disk_bytes)}[/] free"))
    return rows


def _print_notes(profile: HardwareProfile) -> None:
    """Print the caveats that actually change what the user should expect."""
    notes: list[str] = []

    if profile.silent_vram_spillover:
        notes.append(
            "[yellow]Windows/WDDM:[/] exceeding VRAM here does [bold]not[/] raise an "
            "out-of-memory error. The driver pages to system RAM and training silently "
            "runs 5-45x slower. TrainAI measures real memory use and rejects such "
            "configurations instead of trusting a formula."
        )

    if profile.has_gpu and not profile.supports_bf16:
        notes.append(
            "[yellow]No bf16 support[/] on this GPU (needs compute capability 8.0+). "
            "TrainAI will use fp16 with gradient scaling, which is slightly less stable."
        )

    gpu = profile.primary_gpu
    if gpu is not None and gpu.free_vram_bytes < 3 * 1024**3:
        notes.append(
            f"Only {fmt_bytes(gpu.free_vram_bytes)} of VRAM is free. Closing your browser "
            "and other GPU-accelerated apps before training will let TrainAI recommend a "
            "larger model."
        )

    if profile.total_ram_bytes and profile.usable_available_ram_bytes < 4 * 1024**3:
        rest = (
            " That is the container's limit, not the host's, and exceeding it gets the "
            "process killed rather than slowed down."
            if profile.ram_limit_bytes is not None
            else ""
        )
        notes.append(
            f"Only {fmt_bytes(profile.usable_available_ram_bytes)} of system RAM is "
            f"available. Dataset preparation streams from disk, so this is usually "
            f"survivable, but close what you can.{rest}"
        )

    if profile.is_constrained:
        notes.append(
            "[yellow]Running under a container limit.[/] TrainAI reports what this "
            "process may use, which is less than the machine has. Timings here will not "
            "match the same hardware run unconstrained."
        )

    if profile.free_disk_bytes and profile.free_disk_bytes < 5 * 1024**3:
        notes.append(
            f"Only {fmt_bytes(profile.free_disk_bytes)} of disk is free. Tokenised "
            "datasets and checkpoints can each run to several GiB."
        )

    if len(profile.gpus) > 1:
        notes.append(
            f"{len(profile.gpus)} GPUs detected, but TrainAI v0.1 trains on one GPU "
            f"(GPU 0). Multi-GPU training is not implemented and is not simulated."
        )

    notes.extend(f"[dim]probe warning:[/] {warning}" for warning in profile.warnings)

    print_bullets(
        "Notes",
        notes,
        empty="[green]Nothing to flag.[/] This machine looks straightforward.",
    )


def summarise_for_log(profile: HardwareProfile) -> str:
    """One-line summary for ``events.jsonl`` and run READMEs."""
    gpu = profile.primary_gpu
    if gpu is None:
        return (
            f"{profile.device_type} / {profile.cpu_name or 'unknown CPU'} "
            f"({fmt_int(profile.usable_cpu_count)} threads)"
        )
    return (
        f"{gpu.name}, {fmt_bytes(gpu.free_vram_bytes)} free of "
        f"{fmt_bytes(gpu.total_vram_bytes)}, cc {gpu.capability_str}, "
        f"torch {profile.torch_version}"
    )
