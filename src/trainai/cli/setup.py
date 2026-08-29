"""``trainai setup`` -- check that the install matches the hardware, and fix it.

The situation this exists for: someone with a perfectly good NVIDIA card installs
TrainAI, gets the CPU-only PyTorch wheel, and trains a hundred times slower than
their machine allows. Nothing raises. ``trainai doctor`` reports "cpu", which reads
as a hardware limitation rather than an install one.

So this command checks the *system* for a GPU independently of what PyTorch can
see, and names the gap when there is one.

On ``--install``: it prints the exact command, refuses to run outside a virtual
environment unless told otherwise, and asks for confirmation. Downloading two and a
half gigabytes into a possibly-shared Python is not something a diagnostic command
should decide to do on its own, and a wrong index URL leaves an environment worse
than it found it.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

from trainai.console import (
    DASH,
    console,
    emit_json,
    err_console,
    print_bullets,
    print_command,
    print_kv,
    rule,
)
from trainai.errors import UsageError
from trainai.hardware.install import (
    PYTORCH_SELECTOR,
    InstallAdvice,
    advise_install,
    in_virtualenv,
)
from trainai.hardware.probe import detect_gpu_vendors, vendor_labels

__all__ = ["run_setup"]


def run_setup(
    *,
    install: bool = False,
    allow_global: bool = False,
    assume_yes: bool = False,
    json_output: bool = False,
) -> InstallAdvice:
    """Report on the PyTorch install, and optionally repair it.

    Args:
        install: Run the recommended command instead of only printing it.
        allow_global: Permit installing outside a virtual environment.
        assume_yes: Skip the confirmation prompt. Only meaningful with ``install``.
        json_output: Emit the advice as JSON and print nothing else.

    Returns:
        The :class:`InstallAdvice`, so the web interface can reuse this.
    """
    vendors = detect_gpu_vendors()
    advice = advise_install(vendors)

    if json_output:
        payload: dict[str, Any] = advice.to_dict()
        payload["in_virtualenv"] = in_virtualenv()
        payload["python"] = sys.version.split()[0]
        emit_json(payload)
        return advice

    rule("Setup")
    print_kv("Environment", _environment_rows(advice))
    _print_situation(advice)

    if advice.command is None:
        if advice.needs_action:
            console.print("[yellow]No install command would help here.[/] See the notes above.")
        return advice

    console.print("[bold cyan]Command for this machine[/]")
    print_command(advice.command, style="bold")
    console.print()

    if not install:
        console.print(
            "[dim]Run it yourself, or re-run this command with [/][bold]--install[/]"
            "[dim] to have TrainAI run it after confirming.[/]"
        )
        return advice

    _run_install(advice, allow_global=allow_global, assume_yes=assume_yes)
    return advice


def _environment_rows(advice: InstallAdvice) -> list[tuple[str, str]]:
    vendors = (
        ", ".join(vendor_labels(advice.detected_vendors))
        if advice.detected_vendors
        else "none detected"
    )
    rows = [
        ("Python", f"{sys.version.split()[0]}  [dim]{sys.executable}[/]"),
        (
            "Virtual env",
            "[green]yes[/]"
            if in_virtualenv()
            else "[yellow]no[/]  [dim](installing here would touch the system Python)[/]",
        ),
        ("PyTorch", advice.torch_build or "[red]not installed[/]"),
        ("GPUs in system", vendors),
    ]
    if advice.driver_cuda_version is not None:
        major, minor = advice.driver_cuda_version
        rows.append(
            (
                "NVIDIA driver",
                f"supports CUDA up to [bold]{major}.{minor}[/]  [dim](read from nvidia-smi)[/]",
            )
        )
    return rows


def _print_situation(advice: InstallAdvice) -> None:
    colour = "yellow" if advice.needs_action else "green"
    console.print("[bold cyan]Situation[/]")
    console.print(f"  [{colour}]{advice.situation}[/]", highlight=False)
    console.print()
    print_bullets("Notes", advice.notes, empty="[green]Nothing to add.[/]")


def _run_install(advice: InstallAdvice, *, allow_global: bool, assume_yes: bool) -> None:
    """Run the recommended command, with the two guards that matter."""
    assert advice.command is not None

    if not in_virtualenv() and not allow_global:
        raise UsageError(
            "Refusing to install into the system Python.",
            hint=(
                "Create a virtual environment first (`python -m venv .venv` then "
                "activate it) and re-run, or pass --allow-global if you really want "
                "to change the system install. A 2.5 GB PyTorch wheel in the system "
                "Python affects every other project on this machine."
            ),
            details={"python": sys.executable, "command": advice.command},
        )

    if not assume_yes:
        console.print(
            f"[bold]About to run:[/] {advice.command}\n"
            f"[dim]This downloads roughly 2.5 GB and replaces the PyTorch in "
            f"{sys.prefix}.[/]"
        )
        answer = console.input("Continue? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            console.print("[yellow]Nothing was installed.[/]")
            return

    # Run pip as a module of *this* interpreter, so it cannot install into a
    # different Python than the one that will import torch afterwards.
    arguments = advice.command.split()
    if arguments[0] == "pip":
        arguments = [sys.executable, "-m", "pip", *arguments[1:]]

    console.print(f"[dim]{DASH * 3} running pip {DASH * 3}[/]")
    result = subprocess.run(arguments, check=False)
    console.print()
    if result.returncode != 0:
        raise UsageError(
            f"The install command exited with status {result.returncode}.",
            hint=(
                "pip's output is above. A 404 on the index URL usually means the "
                f"wheel channel has moved; check {PYTORCH_SELECTOR} for the current "
                "one. Nothing else in your environment was changed."
            ),
            details={"command": " ".join(arguments), "returncode": result.returncode},
        )

    console.print(
        "[green]Installed.[/] Run [bold]trainai doctor[/] to confirm the GPU is now visible."
    )
    err_console.print(
        "[dim]The PyTorch already loaded into this process is the old one; the new "
        "build takes effect in the next command.[/]"
    )
