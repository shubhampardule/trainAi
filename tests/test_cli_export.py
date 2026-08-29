"""CLI tests for ``trainai export``.

The export itself -- the name mapping, the config fields, and the parity check against
a real ``transformers`` load -- is covered in :mod:`tests.test_export_hf`. What these
check is the wiring and what the user is told: that the flags reach the library, that
``--json`` is machine-readable and silent otherwise, that a refusal produces
``ExitCode.EXPORT`` and a sentence rather than a traceback, and above all that a check
which did not run is printed as not having run.

``test_a_skipped_check_is_printed_as_not_checked`` is the one worth reading. A report
that says "verified" while the parity check was skipped for want of ``transformers`` is
the exact overstatement this project is built to avoid, so the absence is asserted from
the terminal output rather than trusted to the library.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from conftest import flat
from trainai.cli.main import app, main
from trainai.errors import ExitCode


def cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    """Drive the real entry point, argv and all, and return its exit code."""
    monkeypatch.setattr(sys, "argv", ["trainai", *args])
    return main()


# --------------------------------------------------------------------------- #
# The ordinary path
# --------------------------------------------------------------------------- #
def test_a_run_and_an_out_directory_are_enough(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_trained_run: Any,
    tmp_path: Path,
) -> None:
    out = tmp_path / "exported"

    assert cli(monkeypatch, "export", str(cli_trained_run.run), "--out", str(out)) == ExitCode.OK

    printed = flat(capsys.readouterr().out)
    assert "model.safetensors" in printed
    assert "config.json" in printed
    assert "weights in fp32" in printed
    assert (out / "model.safetensors").is_file()
    assert (out / "config.json").is_file()
    assert (out / "README.md").is_file()


def test_the_report_names_every_check(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_trained_run: Any,
    tmp_path: Path,
) -> None:
    assert (
        cli(monkeypatch, "export", str(cli_trained_run.run), "--out", str(tmp_path / "e"))
        == ExitCode.OK
    )

    printed = flat(capsys.readouterr().out)
    assert "weights round-trip" in printed
    assert "tokenizer round-trip" in printed
    assert "logits parity" in printed


def test_the_report_says_it_is_a_base_model(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_trained_run: Any,
    tmp_path: Path,
) -> None:
    """The same warning the run's own README carries, at the moment of exporting."""
    assert (
        cli(monkeypatch, "export", str(cli_trained_run.run), "--out", str(tmp_path / "e"))
        == ExitCode.OK
    )

    assert "continues text rather than answering questions" in flat(capsys.readouterr().out)


def test_a_skipped_check_is_printed_as_not_checked(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_trained_run: Any,
    tmp_path: Path,
) -> None:
    """A check that did not run must never read as a pass.

    Driven through the ``safetensors`` format, whose parity check is genuinely not
    applicable, so this asserts the presentation on every machine rather than only
    on the ones without ``transformers`` installed.
    """
    assert (
        cli(
            monkeypatch,
            "export",
            str(cli_trained_run.run),
            "--out",
            str(tmp_path / "own"),
            "--format",
            "safetensors",
        )
        == ExitCode.OK
    )

    printed = flat(capsys.readouterr().out)
    assert "logits parity not checked" in printed
    assert "not applicable" in printed


def test_no_verify_says_the_checks_were_skipped(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_trained_run: Any,
    tmp_path: Path,
) -> None:
    assert (
        cli(
            monkeypatch,
            "export",
            str(cli_trained_run.run),
            "--out",
            str(tmp_path / "e"),
            "--no-verify",
        )
        == ExitCode.OK
    )

    printed = flat(capsys.readouterr().out)
    assert printed.count("not checked") == 3
    assert "--no-verify" in printed


# --------------------------------------------------------------------------- #
# Flags reaching the library
# --------------------------------------------------------------------------- #
def test_json_is_parseable_and_nothing_else_is_printed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_trained_run: Any,
    tmp_path: Path,
) -> None:
    out = tmp_path / "exported"

    assert (
        cli(monkeypatch, "export", str(cli_trained_run.run), "--out", str(out), "--json")
        == ExitCode.OK
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["format"] == "hf"
    assert payload["dtype"] == "fp32"
    assert payload["parameters"] > 0
    assert payload["total_bytes"] > 0
    assert {item["name"] for item in payload["files"]} >= {"model.safetensors", "config.json"}
    assert len(payload["checks"]) == 3
    assert payload["source"]["step"] > 0
    assert payload["model"]["n_layer"] >= 1


@pytest.mark.parametrize("name", ["fp16", "bf16"])
def test_dtype_reaches_the_library(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_trained_run: Any,
    tmp_path: Path,
    name: str,
) -> None:
    assert (
        cli(
            monkeypatch,
            "export",
            str(cli_trained_run.run),
            "--out",
            str(tmp_path / name),
            "--dtype",
            name,
            "--json",
        )
        == ExitCode.OK
    )

    assert json.loads(capsys.readouterr().out)["dtype"] == name


def test_format_reaches_the_library(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_trained_run: Any,
    tmp_path: Path,
) -> None:
    out = tmp_path / "own"

    assert (
        cli(
            monkeypatch,
            "export",
            str(cli_trained_run.run),
            "--out",
            str(out),
            "--format",
            "safetensors",
            "--json",
        )
        == ExitCode.OK
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["format"] == "safetensors"
    assert (out / "trainai-model.json").is_file()
    assert not (out / "config.json").exists(), "a HF config here would be a lie"


def test_which_latest_reaches_the_library(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_trained_run: Any,
    tmp_path: Path,
) -> None:
    assert (
        cli(
            monkeypatch,
            "export",
            str(cli_trained_run.run),
            "--out",
            str(tmp_path / "latest"),
            "--which",
            "latest",
            "--json",
        )
        == ExitCode.OK
    )

    assert json.loads(capsys.readouterr().out)["source"]["which"] == "latest"


def test_a_single_checkpoint_file_can_be_exported(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_trained_run: Any,
    tmp_path: Path,
) -> None:
    one = sorted((cli_trained_run.run / "checkpoints").glob("step-*.pt"))[0]

    assert (
        cli(monkeypatch, "export", str(one), "--out", str(tmp_path / "one"), "--json")
        == ExitCode.OK
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["source"]["which"] == "explicit"
    assert Path(payload["source"]["checkpoint"]).name == one.name


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #
def test_an_unknown_format_exits_with_usage(
    monkeypatch: pytest.MonkeyPatch, cli_trained_run: Any, tmp_path: Path
) -> None:
    code = cli(
        monkeypatch,
        "export",
        str(cli_trained_run.run),
        "--out",
        str(tmp_path / "e"),
        "--format",
        "onnx",
    )

    assert code == ExitCode.USAGE


def test_an_unknown_dtype_exits_with_usage(
    monkeypatch: pytest.MonkeyPatch, cli_trained_run: Any, tmp_path: Path
) -> None:
    code = cli(
        monkeypatch,
        "export",
        str(cli_trained_run.run),
        "--out",
        str(tmp_path / "e"),
        "--dtype",
        "int4",
    )

    assert code == ExitCode.USAGE


def test_exporting_over_an_existing_export_exits_with_export_and_explains(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_trained_run: Any,
    tmp_path: Path,
) -> None:
    out = tmp_path / "twice"
    assert cli(monkeypatch, "export", str(cli_trained_run.run), "--out", str(out)) == ExitCode.OK
    capsys.readouterr()

    code = cli(monkeypatch, "export", str(cli_trained_run.run), "--out", str(out))

    assert code == ExitCode.EXPORT
    printed = flat(capsys.readouterr().err)
    assert "already holds an export" in printed
    assert "--force" in printed


def test_force_replaces_an_existing_export(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_trained_run: Any,
    tmp_path: Path,
) -> None:
    out = tmp_path / "twice"
    assert cli(monkeypatch, "export", str(cli_trained_run.run), "--out", str(out)) == ExitCode.OK
    capsys.readouterr()

    code = cli(
        monkeypatch,
        "export",
        str(cli_trained_run.run),
        "--out",
        str(out),
        "--dtype",
        "bf16",
        "--force",
        "--json",
    )

    assert code == ExitCode.OK
    assert json.loads(capsys.readouterr().out)["dtype"] == "bf16"


def test_a_directory_that_is_not_an_export_is_refused_not_deleted(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cli_trained_run: Any,
    tmp_path: Path,
) -> None:
    """``--force`` must not be a way to delete a directory the user cares about."""
    occupied = tmp_path / "documents"
    occupied.mkdir()
    (occupied / "thesis.txt").write_text("years of work", encoding="utf-8", newline="\n")

    code = cli(monkeypatch, "export", str(cli_trained_run.run), "--out", str(occupied), "--force")

    assert code == ExitCode.EXPORT
    assert "not a TrainAI export" in flat(capsys.readouterr().err)
    assert (occupied / "thesis.txt").read_text(encoding="utf-8") == "years of work"


def test_a_run_that_does_not_exist_exits_with_usage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code = cli(monkeypatch, "export", str(tmp_path / "nowhere"), "--out", str(tmp_path / "e"))

    assert code == ExitCode.USAGE
    assert "Nothing at" in flat(capsys.readouterr().err)


def test_out_is_required(monkeypatch: pytest.MonkeyPatch, cli_trained_run: Any) -> None:
    """Defaulting the destination would put a 500 MB directory somewhere unasked."""
    assert cli(monkeypatch, "export", str(cli_trained_run.run)) == ExitCode.USAGE


# --------------------------------------------------------------------------- #
# The CLI's literal defaults must not drift from the library's
# --------------------------------------------------------------------------- #
def test_export_flag_defaults_match_the_library() -> None:
    """``main.py`` spells its defaults out so ``--help`` does not import torch."""
    import typer.main

    from trainai import export as export_lib
    from trainai import infer
    from trainai.cli._click import is_group

    expected = {
        "--format": export_lib.DEFAULT_FORMAT,
        "--dtype": export_lib.DEFAULT_DTYPE,
        "--which": infer.DEFAULT_WHICH,
    }

    group = typer.main.get_command(app)
    assert is_group(group)
    command = group.commands["export"]

    for flag, want in expected.items():
        param = next(p for p in command.params if flag in p.opts)
        found = param.default.value if hasattr(param.default, "value") else param.default
        assert found == want, f"export {flag}: CLI has {found!r}, library {want!r}"


def test_the_help_lists_every_format_the_library_accepts() -> None:
    """A format the CLI does not mention is a format nobody finds."""
    import typer.main

    from trainai import export as export_lib
    from trainai.cli._click import is_group

    group = typer.main.get_command(app)
    assert is_group(group)
    command = group.commands["export"]
    param = next(p for p in command.params if "--format" in p.opts)

    for name in export_lib.FORMAT_CHOICES:
        assert name in (param.help or ""), f"--format help does not mention {name}"
