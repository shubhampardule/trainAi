"""``trainai plan`` -- measure this machine and print a training command to review.

The report is split into three sections on purpose: **Measured**, **Derived from your
data**, and **Rules of thumb**. Those are three different kinds of claim, and running
them together in one table is how tools end up implying that a guessed learning rate
is as trustworthy as a read allocator counter. A user who wants to argue with the plan
needs to know which numbers came from where.

The output ends with the exact ``trainai train`` command, because a recommendation the
user cannot read, edit and run is not a recommendation -- it is a black box with a
progress bar.
"""

from __future__ import annotations

from pathlib import Path

from trainai.console import (
    DASH,
    console,
    emit_json,
    fmt_bytes,
    fmt_count,
    fmt_duration,
    fmt_int,
    print_bullets,
    print_command,
    print_kv,
    rule,
)
from trainai.data.binarize import DatasetManifest, verify_dataset
from trainai.hardware.benchmark import BenchmarkResult, describe
from trainai.hardware.planner import (
    PLAN_FILENAME,
    Candidate,
    TrainingPlan,
    parse_duration,
    parse_vram,
    plan_training,
)
from trainai.hardware.probe import DEFAULT_VRAM_SAFETY_FRACTION, HardwareProfile, probe_hardware

__all__ = ["run_plan"]


def run_plan(
    data: str,
    *,
    device: str = "auto",
    time_budget: str | None = None,
    max_preset: str | None = None,
    seq_len: int | None = None,
    precision: str = "auto",
    json_output: bool = False,
    out: str | None = None,
    verify: bool = False,
    safety_fraction: float = DEFAULT_VRAM_SAFETY_FRACTION,
    max_vram: str | None = None,
    warmup_steps: int = 2,
    measure_steps: int = 3,
    seed: int = 1234,
) -> TrainingPlan:
    """Measure candidate configurations and report the largest one that works."""
    quiet = json_output
    # Both flags are parsed before anything is read or probed. A mistyped size is a usage
    # error, and `--verify` re-hashes gigabytes while `probe_hardware` initialises CUDA --
    # neither is a thing to make someone wait through to be told they typed "lots".
    horizon = parse_duration(time_budget) if time_budget else None
    cap = parse_vram(max_vram) if max_vram else None
    dataset = verify_dataset(data, deep=verify) if verify else DatasetManifest.load(data)
    hardware = probe_hardware(disk_path=".")

    if not quiet:
        rule("Planning")
        print_kv("Machine", _hardware_rows(hardware, safety_fraction, cap))
        console.print(
            "[dim]Measuring candidates by running real training steps. Each one takes a "
            "few seconds and briefly allocates VRAM.[/]"
        )

    plan = plan_training(
        dataset,
        hardware=hardware,
        device=device,
        dataset_path=Path(data).as_posix(),
        time_budget_seconds=horizon,
        max_preset=max_preset,
        seq_len=seq_len,
        precision=precision,
        safety_fraction=safety_fraction,
        max_vram_bytes=cap,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        seed=seed,
        on_candidate=None if quiet else _announce_candidate,
        on_result=None if quiet else _announce_result,
    )

    target = Path(out) if out else Path(PLAN_FILENAME)
    written = plan.write(target)

    if quiet:
        emit_json(plan.to_dict())
        return plan

    _report(plan, written)
    return plan


# --------------------------------------------------------------------------- #
# Live progress -- a measurement pass is long enough that silence looks like a hang
# --------------------------------------------------------------------------- #
def _announce_candidate(candidate: Candidate) -> None:
    console.print(f"[dim]  measuring {candidate.describe()}[/]")


def _announce_result(candidate: Candidate, result: BenchmarkResult) -> None:
    colour = "green" if result.ok else "yellow"
    console.print(f"    [{colour}]{DASH} {describe(result)}[/]")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _report(plan: TrainingPlan, written: Path) -> None:
    rule("Recommendation")
    print_kv("Plan", _recommendation_rows(plan))
    print_kv("Measured", _measured_rows(plan))
    print_kv("Derived from your data", _derived_rows(plan))
    print_kv("Rules of thumb", _rules_rows(plan))
    _print_rejected(plan)
    print_bullets(
        "Worth knowing",
        list(plan.budget.warnings()) + list(plan.notes),
        empty="[green]Nothing to flag.[/]",
    )

    rule("Next")
    console.print(f"[dim]Plan written to {written.as_posix()}. Run it with:[/]")
    console.print()
    print_command(plan.train_command())
    console.print()
    console.print("[dim]Or apply the whole plan without retyping it:[/]")
    print_command(
        f"trainai train --data {plan.dataset_path} --plan {written.as_posix()}",
        style="dim bold",
    )


def _hardware_rows(
    hardware: HardwareProfile, safety_fraction: float, cap: int | None = None
) -> list[tuple[str, str]]:

    rows = [("Platform", hardware.platform_summary)]
    gpu = hardware.primary_gpu
    if gpu is None:
        rows.append(
            (
                "GPU",
                "[yellow]none usable[/]  [dim](planning for the CPU: throughput will be "
                "measured, memory will not)[/]",
            )
        )
        return rows
    rows.append(
        (
            "GPU",
            f"[bold]{gpu.name}[/]  {fmt_bytes(gpu.free_vram_bytes)} free of "
            f"{fmt_bytes(gpu.total_vram_bytes)}",
        )
    )
    device_budget = hardware.vram_budget_bytes(safety_fraction)
    if cap is not None and cap < device_budget:
        rows.append(
            (
                "Memory budget",
                f"[bold]{fmt_bytes(cap)}[/]  [dim](--max-vram, below the "
                f"{fmt_bytes(device_budget)} this device would have allowed)[/]",
            )
        )
    else:
        rows.append(
            (
                "Memory budget",
                f"[bold]{fmt_bytes(device_budget)}[/]  "
                f"[dim]({safety_fraction:.0%} of what is free, leaving room for the driver "
                "and the desktop)[/]",
            )
        )
    if hardware.silent_vram_spillover:
        rows.append(
            (
                "Note",
                "[yellow]this platform pages silently past VRAM[/]  [dim](so a config "
                "that does not fit gets slow rather than failing)[/]",
            )
        )
    return rows


def _recommendation_rows(plan: TrainingPlan) -> list[tuple[str, str]]:
    model = plan.model_config
    config = plan.train_config
    batch = str(config.batch_size)
    if config.grad_accum > 1:
        batch += (
            f" x {config.grad_accum} accumulated = [bold]{config.effective_batch_size}[/]"
            " sequences per step"
        )
    else:
        batch += " sequences per step"
    return [
        ("Model", f"[bold]{plan.preset_name}[/]  {DASH}  {model.describe()}"),
        (
            "Parameters",
            f"[bold]{fmt_count(model.parameter_count)}[/]  "
            f"[dim]({fmt_count(model.non_embedding_parameter_count)} outside the "
            "embedding)[/]",
        ),
        (
            "Steps",
            f"[bold]{fmt_int(config.steps)}[/] of {fmt_count(config.tokens_per_step)} "
            f"tokens  [dim]({fmt_count(config.total_tokens)} in total)[/]",
        ),
        ("Batch", batch),
        ("Sequence", f"{fmt_int(config.seq_len)} tokens"),
        (
            "Time to finish",
            f"[bold]{fmt_duration(plan.estimated_seconds)}[/]  "
            "[dim](measured step time times step count, not an extrapolation)[/]",
        ),
        ("Limited by", f"[bold]{plan.regime}[/]  {DASH}  {plan.regime_detail}"),
    ]


def _measured_rows(plan: TrainingPlan) -> list[tuple[str, str]]:
    measured = plan.measurement
    rows = [
        (
            "Throughput",
            f"[bold]{fmt_count(measured.tokens_per_second)}[/] tokens/s  "
            f"[dim]({measured.step_seconds:.3f}s per step, {measured.steps_measured} "
            "steps timed after warmup)[/]",
        )
    ]
    if measured.memory_measured:
        rows.append(
            (
                "Peak VRAM",
                f"[bold]{fmt_bytes(measured.peak_bytes)}[/] of the "
                f"{fmt_bytes(measured.budget_bytes)} budget  "
                f"[dim]({measured.budget_fraction:.0%} used)[/]",
            )
        )
        if measured.alloc_retries:
            rows.append(
                (
                    "Allocator",
                    f"[yellow]{measured.alloc_retries} retries[/]  [dim](the allocator "
                    "had to ask the driver twice: this configuration is close to the "
                    "edge)[/]",
                )
            )
    else:
        rows.append(
            (
                "Peak VRAM",
                "[yellow]not measured[/]  [dim](this device exposes no allocator "
                "counters, so the fit was not verified)[/]",
            )
        )
    rows.append(("Device", f"{measured.device or 'unknown'}  {DASH}  {measured.precision}"))
    return rows


def _derived_rows(plan: TrainingPlan) -> list[tuple[str, str]]:
    budget = plan.budget
    return [
        (
            "Tokens",
            f"[bold]{fmt_int(budget.train_tokens)}[/] train, "
            f"{fmt_int(budget.val_tokens)} validation",
        ),
        (
            "Epochs",
            f"[bold]{budget.epochs:.2f}[/] passes over the corpus  "
            f"[dim]({budget.steps_per_epoch:.0f} steps per pass)[/]",
        ),
        (
            "Tokens/parameter",
            f"[bold]{budget.tokens_per_parameter:.1f}[/] available  "
            f"[dim](from-scratch training usually wants around 20; this size would use "
            f"{fmt_count(budget.compute_optimal_tokens)} tokens)[/]",
        ),
    ]


def _rules_rows(plan: TrainingPlan) -> list[tuple[str, str]]:
    config = plan.train_config
    batches = "batch" if config.eval_batches == 1 else "batches"
    return [
        (
            "Learning rate",
            f"[bold]{config.lr:g}[/]  [dim]({config.schedule} down to "
            f"{config.min_lr:g} after {fmt_int(config.resolved_warmup_steps)} warmup "
            "steps; nothing here measured whether this is right for your text)[/]",
        ),
        (
            "Evaluation",
            f"every {fmt_int(config.eval_every)} steps, {fmt_int(config.eval_batches)} "
            f"{batches}  [dim](scaled to the step count so the best checkpoint has "
            "something to be best among)[/]",
        ),
        (
            "Checkpoints",
            f"every {fmt_int(config.checkpoint_every)} steps, keeping {config.keep_checkpoints}",
        ),
    ]


def _print_rejected(plan: TrainingPlan) -> None:
    rejected = plan.rejected
    if not rejected:
        console.print("[dim]Nothing was ruled out: every candidate that was measured fitted.[/]")
        return
    print_bullets(
        "Why not something bigger",
        [
            f"[bold]{record.candidate.describe()}[/]  {DASH}  {record.result.detail}"
            for record in rejected
        ],
    )
