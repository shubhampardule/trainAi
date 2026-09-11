"""Tests for ``trainai plan`` -- the command that measures the machine and recommends.

Coverage said `src/trainai/cli/plan.py` was at 56%: the planner *library* is tested
hard in `test_planner.py`, and the command wrapping it was not tested at all. What was
uncovered is everything the user actually sees -- the machine table, the live
measurement lines, where `plan.json` is written, and the JSON output another program
would read.

The measurement function is replaced with `fake_measure`, exactly as in
`test_planner.py`. The real search runs, the real report is rendered, and nothing
allocates VRAM or spends seconds per candidate. `probe_hardware` is replaced for the
same reason the hardware tests replace it: a report asserted against whatever GPU the
test happened to run on is not a test of the report.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from conftest import flat, unwrapped
from test_hardware_portability import ALL_MACHINES
from test_planner import GIB, card, dataset, fake_measure
from trainai.cli.plan import run_plan
from trainai.data.binarize import DatasetManifest
from trainai.errors import DatasetError
from trainai.hardware.planner import PLAN_FILENAME, plan_training

#: Big enough that the ladder has somewhere to go, small enough to stay quick.
CORPUS = (400_000_000, 4_000_000)


class Planning:
    """What the command was asked to do, recorded rather than performed."""

    def __init__(self) -> None:
        self.verified: list[tuple[str, bool]] = []
        self.kwargs: dict[str, Any] = {}
        #: The measurement function the command will be given. Assign before calling
        #: ``run_plan`` to model a different device -- a card too small to fit a step
        #: without accumulation, one with no allocator counters, or one under pressure.
        self.measure: Any = None


@pytest.fixture
def planning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Planning:
    """Fix the machine, the manifest and the measurement; keep the real search.

    All three read the world outside the test -- a GPU, a directory on disk, and a
    stopwatch. Pinned together rather than one at a time: a report that is deterministic
    in two of the three still says something different on the next machine.
    """
    monkeypatch.chdir(tmp_path)
    recorded = Planning()
    manifest = dataset(*CORPUS)

    monkeypatch.setattr("trainai.cli.plan.probe_hardware", lambda disk_path=".": card(8))
    monkeypatch.setattr(DatasetManifest, "load", staticmethod(lambda path: manifest))

    def verify(path: str, deep: bool = False) -> DatasetManifest:
        recorded.verified.append((path, deep))
        return manifest

    monkeypatch.setattr("trainai.cli.plan.verify_dataset", verify)

    def planned(*args: Any, **kwargs: Any) -> Any:
        recorded.kwargs = dict(kwargs)
        kwargs["measure"] = recorded.measure or fake_measure(total_vram=8 * GIB)
        return plan_training(*args, **kwargs)

    monkeypatch.setattr("trainai.cli.plan.plan_training", planned)
    return recorded


# --------------------------------------------------------------------------- #
# What it writes
# --------------------------------------------------------------------------- #
def test_the_plan_lands_beside_the_report_by_default(
    capsys: pytest.CaptureFixture[str], planning: Planning, tmp_path: Path
) -> None:
    plan = run_plan("data/prepared")

    written = tmp_path / PLAN_FILENAME
    assert written.exists(), f"nothing was written to {PLAN_FILENAME}"
    assert json.loads(written.read_text(encoding="utf-8"))["preset"] == plan.preset_name
    assert PLAN_FILENAME in unwrapped(capsys.readouterr().out), (
        "a file written without being named is a file the user will not find"
    )


def test_out_puts_the_plan_where_it_was_told(
    capsys: pytest.CaptureFixture[str], planning: Planning, tmp_path: Path
) -> None:
    run_plan("data/prepared", out="runs/mine/plan.json")

    assert (tmp_path / "runs" / "mine" / "plan.json").exists()
    assert not (tmp_path / PLAN_FILENAME).exists(), "--out must not also write the default"


def test_json_writes_the_plan_as_well_as_printing_it(
    capsys: pytest.CaptureFixture[str], planning: Planning, tmp_path: Path
) -> None:
    """`--json` is quiet, not inert. A caller that gets the plan on stdout still wants the file."""
    plan = run_plan("data/prepared", json_output=True)

    printed = capsys.readouterr()
    payload = json.loads(printed.out)
    assert payload["preset"] == plan.preset_name
    assert payload["train"]["steps"] == plan.train_config.steps
    assert (tmp_path / PLAN_FILENAME).exists()
    assert "Planning" not in printed.out, "--json must emit nothing but the document"


def test_json_and_the_written_file_are_the_same_document(
    capsys: pytest.CaptureFixture[str], planning: Planning, tmp_path: Path
) -> None:
    """``--json`` prints a plan *and* writes one, so the two have to agree.

    This replaces a test that asserted ``describe_plan(plan) == plan.to_dict()`` against a
    ``describe_plan`` whose body was ``return plan.to_dict()`` -- true by construction, and
    green for any plan whatsoever. What it was standing in for is this: a script reads
    stdout, a person opens the file, and nothing until now checked that they see the same
    thing. The document is compared whole rather than key by key, because a field added
    later would otherwise be outside the claim.
    """
    plan = run_plan("data/prepared", json_output=True)
    printed = json.loads(capsys.readouterr().out)

    written = json.loads((tmp_path / PLAN_FILENAME).read_text(encoding="utf-8"))
    assert printed == written, "stdout and the file disagree about the same plan"
    assert printed == plan.to_dict(), "and neither matches the object they came from"
    assert printed["preset"] == plan.preset_name, "the test is vacuous on an empty document"


# --------------------------------------------------------------------------- #
# What it prints
# --------------------------------------------------------------------------- #
def test_the_report_names_the_machine_it_measured(
    capsys: pytest.CaptureFixture[str], planning: Planning
) -> None:
    run_plan("data/prepared")

    report = flat(capsys.readouterr().out)
    assert "Synthetic" in report, "the GPU it planned for has to be named"
    assert "Memory budget" in report
    assert "pages silently past VRAM" in report, (
        "this synthetic machine is Windows, where a config that does not fit gets slow "
        "instead of failing -- the report has to say so"
    )


def test_a_machine_with_no_usable_gpu_is_planned_for_anyway(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, planning: Planning
) -> None:
    """The CPU path says what it can and cannot measure, rather than refusing."""
    monkeypatch.setattr(
        "trainai.cli.plan.probe_hardware", lambda disk_path=".": ALL_MACHINES["cpu_only"]()
    )

    run_plan("data/prepared", device="cpu")

    report = flat(capsys.readouterr().out)
    assert "none usable" in report
    assert "memory will not" in report, "an unmeasured budget must not read as a measured one"
    assert "Memory budget" not in report, "there is no VRAM budget to state on a CPU"


def test_each_candidate_is_announced_while_it_is_measured(
    capsys: pytest.CaptureFixture[str], planning: Planning
) -> None:
    """A measurement pass is long enough that silence looks like a hang."""
    run_plan("data/prepared")

    report = flat(capsys.readouterr().out)
    assert "measuring" in report
    assert "Measuring candidates by running real training steps" in report


def test_the_announcements_are_silent_under_json(
    capsys: pytest.CaptureFixture[str], planning: Planning
) -> None:
    run_plan("data/prepared", json_output=True)

    assert "measuring" not in capsys.readouterr().out


def test_the_report_separates_measured_from_derived_from_guessed(
    capsys: pytest.CaptureFixture[str], planning: Planning
) -> None:
    """Three headings, because they are three different kinds of claim.

    Running a measured throughput and a rule-of-thumb learning rate together in one
    table is how a tool ends up implying the guess is as trustworthy as the measurement.
    """
    run_plan("data/prepared")

    report = flat(capsys.readouterr().out)
    for heading in ("Measured", "Derived from your data", "Rules of thumb"):
        assert heading in report, f"the {heading!r} section is missing"


def test_the_recommended_command_is_printed_whole_at_a_narrow_width(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, planning: Planning
) -> None:
    """Through the command, not just through `_report`: this is what the user copies."""
    from rich.console import Console

    narrow = Console(width=60)
    monkeypatch.setattr("trainai.console.console", narrow)
    monkeypatch.setattr("trainai.cli.plan.console", narrow)

    plan = run_plan("data/prepared")

    command = plan.train_command()
    assert len(command) > 60
    assert command in capsys.readouterr().out


def test_the_report_says_when_nothing_was_ruled_out(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, planning: Planning
) -> None:
    """On a big card with a small corpus the ladder is never cut short, and silence there
    would read as "something was rejected and not shown"."""
    monkeypatch.setattr("trainai.cli.plan.probe_hardware", lambda disk_path=".": card(80))

    run_plan("data/prepared", max_preset="tiny")

    assert "Nothing was ruled out" in flat(capsys.readouterr().out)


def test_the_report_says_why_it_did_not_go_bigger(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, planning: Planning
) -> None:
    monkeypatch.setattr("trainai.cli.plan.probe_hardware", lambda disk_path=".": card(2, 1.6))

    run_plan("data/prepared")

    report = flat(capsys.readouterr().out)
    assert "Why not something bigger" in report


def test_a_platform_that_fails_loudly_gets_no_spillover_note(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, planning: Planning
) -> None:
    """The note is a warning about Windows, not a decoration.

    On Linux an over-budget config raises rather than paging, so printing the note there
    would tell the user to distrust a number that is in fact reliable.
    """
    monkeypatch.setattr(
        "trainai.cli.plan.probe_hardware", lambda disk_path=".": ALL_MACHINES["rtx_4090"]()
    )

    run_plan("data/prepared")

    report = flat(capsys.readouterr().out)
    assert "Memory budget" in report
    assert "pages silently" not in report


def test_a_device_with_no_allocator_counters_says_the_fit_was_unverified(
    capsys: pytest.CaptureFixture[str], planning: Planning
) -> None:
    """A peak of "0 B" would read as a measurement. There was no measurement.

    The assertions name the row's own wording rather than the phrase "not measured",
    because the planner already emits a note containing that phrase: asserting on it
    passed while the row itself printed a measured-looking zero. The mutation gate is what
    caught that, and the check is written this way so it cannot happen again.
    """
    planning.measure = fake_measure(total_vram=8 * GIB, memory_measured=False)

    run_plan("data/prepared")

    report = flat(capsys.readouterr().out)
    assert "this device exposes no allocator counters" in report
    assert "% used" not in report, "a fraction of the budget is a claim about a measured peak"


def test_allocator_retries_are_reported_as_being_close_to_the_edge(
    capsys: pytest.CaptureFixture[str], planning: Planning
) -> None:
    """A fitting config that made the allocator ask twice fits, but only just.

    It is the one number here that predicts a run failing later on a machine where
    something else claims VRAM in the meantime, so it is worth a row of its own.
    """
    inner = fake_measure(total_vram=8 * GIB)
    planning.measure = lambda *args, **kwargs: replace(inner(*args, **kwargs), alloc_retries=4)

    run_plan("data/prepared")

    report = flat(capsys.readouterr().out)
    assert "Allocator" in report
    assert "4 retries" in report


def test_an_accumulated_batch_is_reported_as_the_effective_one(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, planning: Planning
) -> None:
    """On a small card the batch that fits is not the batch being trained with.

    Printing only the micro-batch would understate the effective batch by the accumulation
    factor, and the effective one is what the learning rate was chosen for.
    """
    big = dataset(4_000_000_000, 40_000_000)
    monkeypatch.setattr(DatasetManifest, "load", staticmethod(lambda path: big))
    monkeypatch.setattr("trainai.cli.plan.probe_hardware", lambda disk_path=".": card(2, 1.6))
    planning.measure = fake_measure(total_vram=int(1.6 * GIB))

    plan = run_plan("data/prepared", time_budget="10000h")

    assert plan.train_config.grad_accum > 1, "the test is vacuous without accumulation"
    report = flat(capsys.readouterr().out)
    assert "accumulated" in report
    assert f"{plan.train_config.effective_batch_size} sequences per step" in report


# --------------------------------------------------------------------------- #
# What it passes through
# --------------------------------------------------------------------------- #
def test_a_time_budget_reaches_the_planner_in_seconds(
    capsys: pytest.CaptureFixture[str], planning: Planning
) -> None:
    run_plan("data/prepared", time_budget="90m")

    assert planning.kwargs["time_budget_seconds"] == 90 * 60


def test_no_time_budget_is_passed_as_none_rather_than_zero(
    capsys: pytest.CaptureFixture[str], planning: Planning
) -> None:
    """Zero would be a budget of no time at all, which is a different instruction."""
    run_plan("data/prepared")

    assert planning.kwargs["time_budget_seconds"] is None


def test_a_bad_duration_is_refused_before_anything_is_measured(
    capsys: pytest.CaptureFixture[str], planning: Planning, tmp_path: Path
) -> None:
    from trainai.errors import UsageError

    with pytest.raises(UsageError):
        run_plan("data/prepared", time_budget="soon", verify=True)

    assert planning.kwargs == {}, "the refusal has to come before the measurement pass"
    assert planning.verified == [], "and before re-hashing gigabytes of shards"
    assert not (tmp_path / PLAN_FILENAME).exists()


def test_a_vram_cap_reaches_the_planner_as_bytes(
    capsys: pytest.CaptureFixture[str], planning: Planning
) -> None:
    """The flag takes text and the planner takes bytes, so this one is not a pass-through."""
    run_plan("data/prepared", max_vram="2GB")

    assert planning.kwargs["max_vram_bytes"] == 2 * GIB


def test_no_vram_cap_is_passed_as_none_rather_than_zero(
    capsys: pytest.CaptureFixture[str], planning: Planning
) -> None:
    """Zero would be a budget of no memory at all, which is a different instruction."""
    run_plan("data/prepared")

    assert planning.kwargs["max_vram_bytes"] is None


def test_a_bad_vram_cap_is_refused_before_anything_is_read_or_probed(
    capsys: pytest.CaptureFixture[str], planning: Planning, tmp_path: Path
) -> None:
    """A mistyped size is a usage error, and `--verify` re-hashes gigabytes."""
    from trainai.errors import UsageError

    with pytest.raises(UsageError):
        run_plan("data/prepared", max_vram="plenty", verify=True)

    assert planning.kwargs == {}, "the refusal has to come before the measurement pass"
    assert planning.verified == [], "and before re-hashing gigabytes of shards"
    assert not (tmp_path / PLAN_FILENAME).exists()


def test_the_machine_table_shows_a_cap_as_the_budget_and_says_what_was_given_up(
    capsys: pytest.CaptureFixture[str], planning: Planning
) -> None:
    """Printing the device's own limit under a cap would contradict the plan below it."""
    run_plan("data/prepared", max_vram="2GB")

    report = flat(capsys.readouterr().out)
    assert "Memory budget 2.00 GiB" in report
    assert "--max-vram, below the 5.44 GiB this device would have allowed" in report
    assert "85% of what is free" not in report, (
        "the device's own limit is not the budget when a cap is lower than it"
    )


def test_the_machine_table_shows_the_devices_own_limit_when_a_cap_is_above_it(
    capsys: pytest.CaptureFixture[str], planning: Planning
) -> None:
    """A cap never raises the budget, so a generous one must not be printed as one."""
    run_plan("data/prepared", max_vram="64GB")

    report = flat(capsys.readouterr().out)
    assert "Memory budget 5.44 GiB" in report
    assert "85% of what is free" in report
    assert "64.0 GiB" not in report.split("Worth knowing")[0], (
        "the machine table states the budget that applied; the note explains the cap"
    )


def test_no_cap_prints_the_budget_as_a_fraction_of_free_vram(
    capsys: pytest.CaptureFixture[str], planning: Planning
) -> None:
    """The negative control for the row: the ordinary path must not mention the flag."""
    run_plan("data/prepared")

    report = flat(capsys.readouterr().out)
    assert "Memory budget 5.44 GiB" in report
    assert "85% of what is free" in report
    assert "--max-vram" not in report


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_preset", "small"),
        ("seq_len", 512),
        ("precision", "fp32"),
        ("safety_fraction", 0.75),
        ("warmup_steps", 5),
        ("measure_steps", 7),
        ("seed", 99),
        ("device", "cpu"),
    ],
)
def test_every_flag_reaches_the_planner(
    capsys: pytest.CaptureFixture[str], planning: Planning, name: str, value: Any
) -> None:
    """A flag accepted and then dropped is worse than one that does not exist."""
    run_plan("data/prepared", **{name: value})

    assert planning.kwargs[name] == value


def test_the_dataset_path_recorded_in_the_plan_is_posix(
    capsys: pytest.CaptureFixture[str], planning: Planning
) -> None:
    """The plan is a document that gets shared; a Windows path in it does not travel.

    The path is built with the platform separator rather than a literal ``"data\\\\prepared"``,
    because a backslash is an ordinary filename character on Linux: hardcoding one would
    assert Windows behaviour on the four Ubuntu jobs and fail there. Written this way the
    test bites exactly where the bug can exist and stays true where it cannot.
    """
    plan = run_plan(os.path.join("data", "prepared"))

    assert plan.dataset_path == "data/prepared"
    assert "\\" not in plan.dataset_path


# --------------------------------------------------------------------------- #
# --verify
# --------------------------------------------------------------------------- #
def test_without_verify_the_shards_are_not_read(
    capsys: pytest.CaptureFixture[str], planning: Planning
) -> None:
    run_plan("data/prepared")

    assert planning.verified == [], "planning must not checksum gigabytes by default"


def test_verify_checks_the_shards_deeply(
    capsys: pytest.CaptureFixture[str], planning: Planning
) -> None:
    run_plan("data/prepared", verify=True)

    assert planning.verified == [("data/prepared", True)]


def test_a_dataset_that_fails_verification_stops_the_command(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, planning: Planning
) -> None:
    def refuse(path: str, deep: bool = False) -> DatasetManifest:
        raise DatasetError("A shard checksum does not match.", hint="Re-run data prepare.")

    monkeypatch.setattr("trainai.cli.plan.verify_dataset", refuse)

    with pytest.raises(DatasetError):
        run_plan("data/prepared", verify=True)

    assert planning.kwargs == {}, "a corrupt dataset must not be measured against"
