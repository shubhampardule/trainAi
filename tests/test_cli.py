"""CLI smoke tests.

These assert the contract in :func:`trainai.cli.main.main`: exit codes are
stable, expected failures print a hint and no traceback, and ``--help`` never
advertises a command that does not exist.
"""

from __future__ import annotations

import bz2
import json
import sqlite3
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from conftest import flat, unwrapped
from trainai import __version__
from trainai.cli.main import app, main
from trainai.errors import DatasetEmptyError, ExitCode

runner = CliRunner()


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help_lists_doctor() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in result.stdout


def test_no_args_shows_help_rather_than_crashing() -> None:
    result = runner.invoke(app, [])
    # no_args_is_help exits with click's "no command" code, not a traceback.
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Usage" in result.stdout


def test_every_advertised_command_is_registered() -> None:
    """Guards the README promise: if --help lists it, it works.

    A Typer group with no subcommands would be advertised but unusable, so an
    empty group counts as a failure here.
    """
    from trainai.cli._click import is_group

    command = typer_to_click(app)
    assert is_group(command)
    for name, sub in command.commands.items():
        if is_group(sub):
            assert sub.commands, f"command group '{name}' is advertised but empty"


def typer_to_click(typer_app: object) -> object:
    import typer.main

    return typer.main.get_command(typer_app)  # type: ignore[arg-type]


def test_unknown_command_is_a_usage_error() -> None:
    result = runner.invoke(app, ["definitely-not-a-command"])
    assert result.exit_code != 0


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_short_and_long_help(flag: str) -> None:
    assert runner.invoke(app, [flag]).exit_code == 0


# --------------------------------------------------------------------------- #
# main(): the exit-code contract
# --------------------------------------------------------------------------- #
def test_main_returns_zero_for_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["trainai", "--version"])
    assert main() == ExitCode.OK


def test_main_propagates_nonzero_exit_codes_from_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a failing command must not report success.

    In non-standalone mode Typer *returns* the exit code instead of raising it.
    An earlier version of main() discarded that return value, so every usage
    error and every Ctrl-C exited 0 and any wrapping shell script concluded that
    training had succeeded.
    """
    monkeypatch.setattr(sys, "argv", ["trainai", "no-such-command"])
    assert main() != ExitCode.OK


def test_main_reports_missing_option_value_as_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["trainai", "doctor", "--path"])
    assert main() != ExitCode.OK


def test_main_maps_trainai_error_to_its_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(**_: object) -> None:
        raise DatasetEmptyError("nothing to train on", hint="point --path at some text")

    monkeypatch.setattr("trainai.cli.doctor.run_doctor", boom)
    monkeypatch.setattr(sys, "argv", ["trainai", "doctor"])

    assert main() == ExitCode.DATASET

    error = flat(capsys.readouterr().err)
    assert "nothing to train on" in error
    assert "point --path at some text" in error
    assert "Traceback" not in error


def test_main_reports_keyboard_interrupt_as_130(monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupt(**_: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("trainai.cli.doctor.run_doctor", interrupt)
    monkeypatch.setattr(sys, "argv", ["trainai", "doctor"])
    assert main() == ExitCode.INTERRUPTED


def test_main_reports_an_interrupt_typer_did_not_see(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ``except KeyboardInterrupt`` in ``main`` itself, which the test above never runs.

    A Ctrl-C inside a command arrives as the integer 130 -- Typer catches it first and
    returns the code -- so that handler is for the one landing *outside* Typer's own try
    block, between the call and the return. It is reached by replacing the app with a
    callable that raises, and it has to answer the same as the ordinary route: the same
    code and the same sentence on stderr, since the user pressed the same key.
    """

    def interrupt(**_: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("trainai.cli.main.app", interrupt)

    assert main() == ExitCode.INTERRUPTED
    assert "Interrupted." in capsys.readouterr().err


def test_main_treats_an_abort_as_an_interrupt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``Abort`` is what click raises when a confirmation prompt gets Ctrl-C or EOF.

    It is not a ``KeyboardInterrupt`` -- click has already caught that and re-raised its
    own -- but it means the same thing, so it exits 130 and says the same sentence. The
    class comes from :mod:`trainai.cli._click`, which is the one that matches whichever
    package the installed Typer raises it from.
    """
    from trainai.cli._click import Abort

    def abort(**_: object) -> None:
        raise Abort()

    monkeypatch.setattr("trainai.cli.main.app", abort)

    assert main() == ExitCode.INTERRUPTED
    assert "Interrupted." in capsys.readouterr().err


def test_main_returns_the_code_of_an_exit_that_was_raised_rather_than_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Exit`` is returned by Typer and raised by click; ``main`` has to take both.

    ``click.Command.main`` raises it even in non-standalone mode, and a Typer that ever
    stopped translating it would send it here. The code has to come through unchanged --
    ``Exit`` means "stop with this code", not "something went wrong" -- and it must not
    be mistaken for an interrupt or reported as one.
    """
    from trainai.cli._click import Exit

    def leave(**_: object) -> None:
        raise Exit(code=7)

    monkeypatch.setattr("trainai.cli.main.app", leave)

    assert main() == 7


def test_main_lets_unexpected_exceptions_escape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Our bugs must keep their traceback, so they get reported instead of hidden."""

    def bug(**_: object) -> None:
        raise ZeroDivisionError("this is a TrainAI bug, not user error")

    monkeypatch.setattr("trainai.cli.doctor.run_doctor", bug)
    monkeypatch.setattr(sys, "argv", ["trainai", "doctor"])
    with pytest.raises(ZeroDivisionError):
        main()


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #
def test_doctor_runs_and_reports_a_device() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "hardware report" in result.stdout
    assert any(word in result.stdout for word in ("cuda", "cpu", "mps"))


def test_doctor_json_is_machine_readable(tmp_path: object) -> None:
    result = runner.invoke(app, ["doctor", "--json", "--path", str(tmp_path)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["device_type"] in {"cuda", "mps", "cpu"}
    assert "gpus" in payload
    assert payload["cpu_count_logical"] >= 1


# --------------------------------------------------------------------------- #
# data prepare / data inspect
#
# These go through main() rather than CliRunner, because the thing worth checking
# is the exit code contract: a corpus TrainAI refuses must exit 3, not raise.
# --------------------------------------------------------------------------- #
def cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["trainai", *args])
    return main()


def test_data_help_lists_both_subcommands() -> None:
    result = runner.invoke(app, ["data", "--help"])

    assert result.exit_code == 0
    assert "prepare" in result.stdout
    assert "inspect" in result.stdout


def test_prepare_writes_a_usable_dataset(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    many_document_corpus: Path,
    tmp_path: Path,
) -> None:
    out = tmp_path / "prepared"

    code = cli(
        monkeypatch,
        "data",
        "prepare",
        str(many_document_corpus),
        "--out",
        str(out),
        "--vocab-size",
        "512",
    )

    assert code == ExitCode.OK, capsys.readouterr()
    assert (out / "manifest.json").exists()
    assert (out / "tokenizer.json").exists()
    assert list(out.glob("*.bin"))

    from trainai.data import verify_dataset

    manifest = verify_dataset(out, deep=True)
    assert manifest.total_tokens > 0
    assert "Wrote" in capsys.readouterr().out


def test_prepare_json_emits_only_the_manifest(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    many_document_corpus: Path,
    tmp_path: Path,
) -> None:
    """``--json`` has to be pipeable, so the progress display must be suppressed."""
    code = cli(
        monkeypatch,
        "data",
        "prepare",
        str(many_document_corpus),
        "--out",
        str(tmp_path / "prepared"),
        "--vocab-size",
        "512",
        "--json",
    )

    assert code == ExitCode.OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["format"] == "trainai-dataset"
    assert payload["totals"]["tokens"] > 0


def test_preparing_twice_with_the_same_seed_reproduces_the_dataset(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    many_document_corpus: Path,
    tmp_path: Path,
) -> None:
    """The M1 promise, checked through the command a user actually types."""

    def prepare_into(name: str, seed: str) -> dict[str, object]:
        code = cli(
            monkeypatch,
            "data",
            "prepare",
            str(many_document_corpus),
            "--out",
            str(tmp_path / name),
            "--vocab-size",
            "512",
            "--seed",
            seed,
            "--json",
        )
        assert code == ExitCode.OK
        return json.loads(capsys.readouterr().out)

    first = prepare_into("a", "1234")
    second = prepare_into("b", "1234")
    other = prepare_into("c", "7")

    def checksums(payload: dict[str, object]) -> list[str]:
        splits = payload["splits"]
        assert isinstance(splits, dict)
        return [s["sha256"] for name in ("train", "val") for s in splits[name]["shards"]]

    assert checksums(first) == checksums(second)
    assert first["content_hash"] == second["content_hash"]
    assert first["content_hash"] != other["content_hash"]
    assert first["totals"] == other["totals"]  # same text, different split


def test_prepare_refuses_to_overwrite_a_dataset_without_force(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    many_document_corpus: Path,
    tmp_path: Path,
) -> None:
    out = tmp_path / "prepared"
    args = ("data", "prepare", str(many_document_corpus), "--out", str(out), "--vocab-size", "512")
    assert cli(monkeypatch, *args) == ExitCode.OK
    before = (out / "manifest.json").read_bytes()
    capsys.readouterr()

    code = cli(monkeypatch, *args)

    assert code == ExitCode.USAGE
    assert "--force" in capsys.readouterr().err
    assert (out / "manifest.json").read_bytes() == before


def test_prepare_force_replaces_the_dataset(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    many_document_corpus: Path,
    tmp_path: Path,
) -> None:
    out = tmp_path / "prepared"
    args = ("data", "prepare", str(many_document_corpus), "--out", str(out), "--vocab-size", "512")
    assert cli(monkeypatch, *args) == ExitCode.OK
    capsys.readouterr()

    code = cli(monkeypatch, *args, "--force")

    assert code == ExitCode.OK
    assert "replacing" in capsys.readouterr().out


def test_prepare_keeps_nothing_when_the_corpus_changes_between_the_two_passes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    many_document_corpus: Path,
    tmp_path: Path,
) -> None:
    """The rollback nobody had run: a corpus that grows mid-preparation.

    Preparation reads the corpus twice, minutes apart -- once to measure it and train
    the tokenizer, once to encode it. The manifest's corpus report comes from the first
    pass, so if the second pass sees a different number of documents the manifest no
    longer describes its own shards, and a dataset that misdescribes itself is worse
    than no dataset. The code answers that by deleting what it wrote and saying
    "Nothing was kept", which is a promise about the filesystem, so this asserts on the
    filesystem rather than on the message alone.

    The corpus is grown from inside ``validate_corpus``, which runs between the two
    passes: appending a record to the existing ``.jsonl`` rather than adding a new file,
    because the file list is fixed before the first pass and a new file would simply not
    be read.
    """
    from trainai.cli import data as data_cli

    out = tmp_path / "prepared"
    corpus_file = many_document_corpus / "documents.jsonl"
    measure_then_validate = data_cli.validate_corpus

    def grow_the_corpus(*args: Any, **kwargs: Any) -> Any:
        with corpus_file.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"text": "A document that arrived mid-run. " * 8}) + "\n")
        return measure_then_validate(*args, **kwargs)

    monkeypatch.setattr(data_cli, "validate_corpus", grow_the_corpus)

    code = cli(
        monkeypatch,
        *("data", "prepare", str(many_document_corpus)),
        *("--out", str(out), "--vocab-size", "512"),
    )

    assert code == ExitCode.DATASET
    message = flat(capsys.readouterr().err)
    assert "corpus changed while it was being prepared" in message
    assert "Nothing was kept" in message
    left = sorted(path.name for path in out.iterdir()) if out.exists() else []
    assert left == [], f"the rollback left {left} behind"


def test_discarding_a_dataset_removes_its_own_files_and_leaves_the_rest(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    many_document_corpus: Path,
    tmp_path: Path,
) -> None:
    """Deleting "exactly the files this run wrote" is a claim about what survives it.

    The rollback names its shards from the manifest rather than globbing the output
    directory, and this is the difference that makes visible: a file the run did not
    write survives it. Deleting by pattern would be the easier implementation and would
    take the bystander with it -- and since the only caller runs on a failure path a
    user is already unhappy about, that would be a bad moment to also lose their notes.
    """
    from trainai.cli.data import _discard
    from trainai.data.binarize import DatasetManifest

    out = tmp_path / "prepared"
    assert (
        cli(
            monkeypatch,
            *("data", "prepare", str(many_document_corpus)),
            *("--out", str(out), "--vocab-size", "512"),
        )
        == ExitCode.OK
    )
    capsys.readouterr()

    bystander = out / "notes.txt"
    bystander.write_text("kept by hand, not written by the run\n", encoding="utf-8", newline="\n")
    manifest = DatasetManifest.load(out)
    shards = [path for split in ("train", "val") for path in manifest.shard_paths(split)]
    assert shards, "the fixture produced no shards, so the test would prove nothing"
    assert all(path.exists() for path in shards)

    _discard(manifest, out)

    assert [path.name for path in shards if path.exists()] == []
    assert not (out / "manifest.json").exists()
    assert not (out / "tokenizer.json").exists()
    assert bystander.exists(), "the rollback deleted a file it did not write"


def test_prepare_can_reuse_a_tokenizer_from_another_dataset(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    many_document_corpus: Path,
    tmp_path: Path,
) -> None:
    """The prerequisite for fine-tuning, and the only way to get it.

    Every ``data prepare`` without this flag trains its own tokenizer, so every second
    dataset has a vocabulary no existing checkpoint's embedding matrix matches. Two
    facts are asserted together because either alone would be satisfied by a bug: the
    fingerprint must be *identical* (or the base model's weights are meaningless) and
    the content hash must *differ* (or nothing was reused -- the same corpus was
    prepared twice and the test proves nothing).
    """
    import json

    base = tmp_path / "base"
    tuned = tmp_path / "tuned"
    corpus_b = tmp_path / "corpus_b"
    corpus_b.mkdir()
    (corpus_b / "other.jsonl").write_text(
        "".join(
            json.dumps({"text": f"A different document number {n} with its own words."}) + "\n"
            for n in range(200)
        ),
        encoding="utf-8",
        newline="\n",
    )

    assert (
        cli(
            monkeypatch,
            *("data", "prepare", str(many_document_corpus), "--out", str(base)),
            *("--vocab-size", "512"),
        )
        == ExitCode.OK
    )
    capsys.readouterr()

    code = cli(
        monkeypatch,
        *("data", "prepare", str(corpus_b), "--out", str(tuned), "--tokenizer", str(base)),
    )

    assert code == ExitCode.OK
    out = flat(capsys.readouterr().out)
    assert "reused, not trained" in out

    first = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    second = json.loads((tuned / "manifest.json").read_text(encoding="utf-8"))
    assert second["tokenizer_fingerprint"] == first["tokenizer_fingerprint"]
    assert second["vocab_size"] == first["vocab_size"]
    assert second["content_hash"] != first["content_hash"]


def test_prepare_refuses_vocab_size_alongside_a_reused_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    many_document_corpus: Path,
    tmp_path: Path,
) -> None:
    """Refused rather than ignored.

    The loaded file's vocabulary is a fact about that file, so there is nothing for a
    second number to do. Silently dropping a flag the user typed is how someone ends up
    with a dataset that is not the one they asked for -- and here they would only find
    out when the fine-tune refused the checkpoint, several commands later.
    """
    base = tmp_path / "base"
    assert (
        cli(
            monkeypatch,
            *("data", "prepare", str(many_document_corpus), "--out", str(base)),
            *("--vocab-size", "512"),
        )
        == ExitCode.OK
    )
    capsys.readouterr()

    code = cli(
        monkeypatch,
        *("data", "prepare", str(many_document_corpus), "--out", str(tmp_path / "two")),
        *("--tokenizer", str(base), "--vocab-size", "900"),
    )

    assert code == ExitCode.USAGE
    assert "cannot be combined with --tokenizer" in flat(capsys.readouterr().err)
    assert not (tmp_path / "two").exists()


def test_prepare_says_where_a_missing_tokenizer_should_be(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    many_document_corpus: Path,
    tmp_path: Path,
) -> None:
    code = cli(
        monkeypatch,
        *("data", "prepare", str(many_document_corpus), "--out", str(tmp_path / "out")),
        *("--tokenizer", str(tmp_path / "nowhere.json")),
    )

    assert code == ExitCode.DATASET
    assert "No tokenizer at" in flat(capsys.readouterr().err)


def test_prepare_refuses_to_write_into_the_corpus(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    many_document_corpus: Path,
) -> None:
    code = cli(
        monkeypatch,
        "data",
        "prepare",
        str(many_document_corpus),
        "--out",
        str(many_document_corpus / "prepared"),
    )

    assert code == ExitCode.USAGE
    assert "inside the corpus" in flat(capsys.readouterr().err)


def test_prepare_rejects_contradictory_document_bounds(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    many_document_corpus: Path,
    tmp_path: Path,
) -> None:
    code = cli(
        monkeypatch,
        "data",
        "prepare",
        str(many_document_corpus),
        "--out",
        str(tmp_path / "prepared"),
        "--min-doc-chars",
        "500",
        "--max-doc-chars",
        "100",
    )

    assert code == ExitCode.USAGE
    assert "--max-doc-chars" in capsys.readouterr().err


def test_inspect_rejects_a_minimum_document_length_below_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    many_document_corpus: Path,
) -> None:
    """The other half of the guard, checked through ``inspect`` on purpose.

    ``test_prepare_rejects_contradictory_document_bounds`` covers the second branch
    of the same function. Both commands build their options through the same
    ``_ingest_options``, so a guard tested only through ``prepare`` says nothing about
    whether the read-only command validates at all. ``--min-doc-chars 0`` is the
    interesting value rather than a negative one: zero reads as "no minimum", and what
    it would really do is admit empty documents to a corpus report that then divides
    by their length.
    """
    code = cli(monkeypatch, "data", "inspect", str(many_document_corpus), "--min-doc-chars", "0")

    assert code == ExitCode.USAGE
    error = flat(capsys.readouterr().err)
    assert "--min-doc-chars must be at least 1" in error
    assert "got 0" in error


def test_prepare_on_a_missing_path_exits_three(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code = cli(
        monkeypatch,
        "data",
        "prepare",
        str(tmp_path / "not-here"),
        "--out",
        str(tmp_path / "prepared"),
    )

    assert code == ExitCode.DATASET
    error = unwrapped(capsys.readouterr().err)
    assert "not-here" in error
    assert "Traceback" not in error


def test_prepare_on_an_unusable_corpus_explains_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    corpus = tmp_path / "tiny"
    corpus.mkdir()
    (corpus / "tiny.txt").write_text("hello world\n", encoding="utf-8", newline="\n")
    out = tmp_path / "prepared"

    code = cli(monkeypatch, "data", "prepare", str(corpus), "--out", str(out))

    assert code == ExitCode.DATASET
    assert "megabyte" in capsys.readouterr().err
    assert not (out / "manifest.json").exists()


def test_prepare_blames_the_vocab_size_and_not_the_corpus(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    many_document_corpus: Path,
    tmp_path: Path,
) -> None:
    """A ``--vocab-size`` below the floor used to be reported as an empty corpus.

    ``train_tokenizer`` refuses the size before it reads anything, so the handler that
    lets corpus validation speak first was re-validating a measurement of nothing --
    and an empty measurement always fails as ``empty_corpus``. Measured on a 27.7 KiB
    corpus, ``--vocab-size 257`` printed "No usable documents were found." two lines
    below a panel reporting the size of those documents, and the message naming the
    real problem was discarded.
    """
    code = cli(
        monkeypatch,
        "data",
        "prepare",
        str(many_document_corpus),
        "--out",
        str(tmp_path / "prepared"),
        "--vocab-size",
        "257",
    )

    captured = capsys.readouterr()
    assert code == ExitCode.DATASET
    assert "--vocab-size 258" in captured.err
    assert "No usable documents" not in captured.err
    assert "Traceback" not in captured.err


def test_prepare_refuses_a_min_frequency_of_zero_instead_of_ignoring_it(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    many_document_corpus: Path,
    tmp_path: Path,
) -> None:
    """``--min-frequency 0`` used to train exactly as if 1 had been passed.

    The Rust trainer clamps it, so the flag quietly did something other than what its
    help text promises. Refusing it is also what makes the "no merges" hint honest:
    that hint names 1 as the lowest threshold, so 0 must not be accepted below it.
    """
    code = cli(
        monkeypatch,
        "data",
        "prepare",
        str(many_document_corpus),
        "--out",
        str(tmp_path / "prepared"),
        "--min-frequency",
        "0",
    )

    captured = capsys.readouterr()
    assert code == ExitCode.DATASET
    assert "--min-frequency 1" in captured.err
    assert "No usable documents" not in captured.err
    assert "Traceback" not in captured.err


def test_prepare_still_blames_a_genuinely_empty_corpus_on_the_corpus(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The negative control for the test above, and the regression it risks.

    An empty corpus reaches the tokenizer's "no merges were learned", which explains
    nothing useful, so validation must still be allowed to replace it. The distinction
    is whether the trainer read anything: here it did and found nothing, which is a
    fact about the corpus.
    """
    corpus = tmp_path / "empty"
    corpus.mkdir()
    (corpus / "empty.txt").write_text("", encoding="utf-8", newline="\n")

    code = cli(monkeypatch, "data", "prepare", str(corpus), "--out", str(tmp_path / "prepared"))

    captured = capsys.readouterr()
    assert code == ExitCode.DATASET
    assert "No usable documents" in captured.err
    assert "no merges" not in captured.err


# --------------------------------------------------------------------------- #
# Tabular data
#
# A CSV is the most likely first corpus a normal user has, and training a
# language model on one produces a fluent nonsense generator. These pin the two
# ways out -- name the prose column, or say explicitly that you meant it -- and
# the refusal in between.
# --------------------------------------------------------------------------- #
def numeric_table(path: Path, *, rows: int = 400) -> None:
    """A table of measurements: high delimiter regularity, mostly digits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["hour,temp_c,humidity,pressure_mb"]
    for index in range(rows):
        lines.append(f"{index % 24},{9.4 + index % 17:.1f},{0.5 + (index % 40) / 100:.2f},1015.3")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def review_table(path: Path, *, rows: int = 300, column: str = "note") -> None:
    """A table whose prose column holds real sentences, and nothing else does.

    The column is named by the caller for the same reason ``review_database`` takes
    one: ``note`` is not a name TrainAI recognises, so it is the right default for
    the tests about naming a column by hand, and ``review`` is, so it is what the
    test about resolving one by convention has to pass.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    subjects = ("the harbour", "the orchard", "the lantern", "the bridge")
    lines = [f"stars,{column}"]
    for index in range(rows):
        subject = subjects[index % len(subjects)]
        lines.append(
            f"{index % 5 + 1},"
            f'"Visit number {index}. {subject} was quieter than the guidebook '
            f'promised, and the walk back took longer than we had planned for."'
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def test_prepare_refuses_a_csv_without_naming_a_column(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The dead end that used to say "No readable text files found"."""
    corpus = tmp_path / "csvcorpus"
    numeric_table(corpus / "weather.csv")
    out = tmp_path / "prepared"

    code = cli(monkeypatch, "data", "prepare", str(corpus), "--out", str(out))

    assert code == ExitCode.DATASET
    error = capsys.readouterr().err
    assert "--csv-text-column" in error  # the way out, if a column holds prose
    assert "humidity" in error  # the columns it actually found
    assert "regression" in error  # the right tool, if they do not
    assert not (out / "manifest.json").exists()


def test_prepare_reads_the_named_csv_column_and_records_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    corpus = tmp_path / "reviews"
    review_table(corpus / "reviews.csv")
    out = tmp_path / "prepared"

    code = cli(
        monkeypatch,
        "data",
        "prepare",
        str(corpus),
        "--out",
        str(out),
        "--csv-text-column",
        "note",
        "--vocab-size",
        "512",
    )

    captured = capsys.readouterr()
    assert code == ExitCode.OK, captured
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ingest"]["options"]["csv_text_column"] == "note"

    # The column is not echoed back during prepare -- the Read block is for
    # choices TrainAI made, not for repeating a flag the user typed. It has to
    # survive into the dataset, though, so that months later the prepared
    # directory still says which column it was built from.
    assert cli(monkeypatch, "data", "inspect", str(out)) == ExitCode.OK
    assert "note" in capsys.readouterr().out


def test_inspect_names_the_csv_column_it_chose_itself(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A CSV read with no ``--csv-text-column`` at all, which is the common case.

    ``test_inspect_reports_a_table_and_column_it_chose_itself`` covers the database
    rows, and they are a different block built from a different counter: a fix to one
    leaves the other silent. The choice here is made from a list of conventional
    names, so it is the choice most likely to be wrong -- a file with both ``review``
    and ``summary`` columns gets ``review`` and nothing on screen would say so unless
    this row exists.
    """
    corpus = tmp_path / "reviews"
    review_table(corpus / "reviews.csv", column="review")

    code = cli(monkeypatch, "data", "inspect", str(corpus))

    captured = capsys.readouterr()
    assert code == ExitCode.OK, captured
    shown = flat(captured.out)
    assert "CSV column" in shown
    assert "chosen automatically" in shown
    # The file as well as the column: a corpus of several CSVs can resolve a
    # different column in each, and the row is per file for that reason.
    assert "reviews.csv:review" in unwrapped(captured.out)


def review_database(
    path: Path, *, rows: int = 300, extra_table: bool = True, column: str = "note"
) -> None:
    """The same reviews, in a database -- plus a second table, so a choice is needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subjects = ("the harbour", "the orchard", "the lantern", "the bridge")
    connection = sqlite3.connect(path)
    try:
        connection.execute(f'CREATE TABLE reviews (stars INT, "{column}" TEXT)')
        if extra_table:
            connection.execute("CREATE TABLE authors (name TEXT)")
            connection.execute("INSERT INTO authors VALUES ('Ada')")
        connection.executemany(
            "INSERT INTO reviews VALUES (?, ?)",
            [
                (
                    index % 5 + 1,
                    f"Visit number {index}. {subjects[index % len(subjects)]} was quieter "
                    "than the guidebook promised, and the walk back took longer than we "
                    "had planned for.",
                )
                for index in range(rows)
            ],
        )
        connection.commit()
    finally:
        connection.close()


def test_prepare_refuses_a_database_with_several_tables(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    corpus = tmp_path / "reviews.db"
    review_database(corpus)
    out = tmp_path / "prepared"

    code = cli(monkeypatch, "data", "prepare", str(corpus), "--out", str(out))

    assert code == ExitCode.DATASET
    error = capsys.readouterr().err
    assert "--db-table" in error  # the way out
    assert "authors" in error  # the tables it actually found
    assert not (out / "manifest.json").exists()


def test_prepare_reads_the_named_database_table_and_records_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    corpus = tmp_path / "reviews.db"
    review_database(corpus)
    out = tmp_path / "prepared"

    code = cli(
        monkeypatch,
        "data",
        "prepare",
        str(corpus),
        "--out",
        str(out),
        "--db-table",
        "reviews",
        "--csv-text-column",
        "note",
        "--vocab-size",
        "512",
    )

    captured = capsys.readouterr()
    assert code == ExitCode.OK, captured
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ingest"]["options"]["db_table"] == "reviews"
    assert manifest["ingest"]["options"]["csv_text_column"] == "note"

    assert cli(monkeypatch, "data", "inspect", str(out)) == ExitCode.OK
    assert "reviews" in capsys.readouterr().out


CHAT_TURNS = [
    {"role": "user", "content": "Hi there, could you help me with something small?"},
    {
        "role": "assistant",
        "content": (
            "Of course. Tell me what you are working on and I will do what I can to "
            "help you get it finished."
        ),
    },
    {"role": "user", "content": "I need a sentence about the weather in a coastal town."},
    {
        "role": "assistant",
        "content": (
            "The wind came off the harbour all morning and the shopfronts along the "
            "front rattled until the rain finally arrived."
        ),
    },
]


def chat_corpus(directory: Path, conversations: int) -> Path:
    """A typed-conversation corpus: the same turns, varied enough to tokenize."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "chat.jsonl"
    lines = []
    for index in range(conversations):
        turns = [dict(turn) for turn in CHAT_TURNS]
        turns[2]["content"] = f"{turns[2]['content']} Attempt number {index}."
        lines.append(json.dumps({"messages": turns}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def test_prepare_reads_typed_conversations_and_reports_the_trained_share(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """End to end, because the share is the only visible sign the mask exists.

    A corpus prepared with and without ``--jsonl-messages-field`` has the same token
    count, so nothing else in the report would change if chat mode silently did
    nothing. The percentage is what a user checks the flag by.
    """
    corpus = tmp_path / "chats"
    chat_corpus(corpus, 40)
    out = tmp_path / "prepared"

    code = cli(
        monkeypatch,
        "data",
        "prepare",
        str(corpus),
        "--out",
        str(out),
        "--jsonl-messages-field",
        "messages",
        "--vocab-size",
        "512",
    )

    captured = capsys.readouterr()
    assert code == ExitCode.OK, captured
    output = flat(captured.out)
    assert "Chat records" in output
    assert "40 rendered" in output
    assert "are assistant replies" in output
    # The consequence is asserted, not just the share: a user who reads the percentage
    # needs to know whether it describes what the loss will average over. It does --
    # the mask is on by default for a typed corpus, and the trainer applies it -- so the
    # row says both that the file was written and what training does with it.
    assert "written beside every shard" in output
    assert "scores targets only" in output
    assert (out / "train_00000.mask.bin").exists()

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ingest"]["options"]["jsonl_messages_field"] == "messages"
    assert manifest["ingest"]["options"]["jsonl_field"] is None
    # The same block reaches `--json` on prepare and on inspect, which both emit
    # to_dict(): a script that has to build a prompt reads it from here.
    assert manifest["chat"]["version"] == 1
    assert manifest["chat"]["trained_roles"] == ["assistant"]
    assert (
        manifest["splits"]["train"]["shards"][0]["mask_bytes"]
        == (manifest["splits"]["train"]["shards"][0]["tokens"])
    )

    assert cli(monkeypatch, "data", "inspect", str(out), "--verify") == ExitCode.OK
    shown = flat(capsys.readouterr().out)
    assert "Chat messages field" in shown
    assert "tokens are targets" in shown
    assert "scores those tokens only" in shown
    # The layout, beside the mask and separate from it: this is the row a user reads to
    # find out what a prompt has to look like at inference time.
    assert "Chat template" in shown
    assert "System:, User:, Assistant:" in shown
    assert "trained on assistant" in shown


def test_inspect_reports_the_chat_template_even_with_no_loss_mask(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The template and the mask are independent, and this is the case that shows it.

    --no-loss-mask removes the mask, not the ``User:``/``Assistant:`` text in the
    shards. A model trained on this dataset still has to be prompted in that layout, so
    the row that says so cannot be hung off the mask.
    """
    corpus = tmp_path / "chats"
    chat_corpus(corpus, 40)
    out = tmp_path / "prepared"
    assert (
        cli(
            monkeypatch,
            "data",
            "prepare",
            str(corpus),
            "--out",
            str(out),
            "--jsonl-messages-field",
            "messages",
            "--no-loss-mask",
            "--vocab-size",
            "512",
        )
        == ExitCode.OK
    )
    capsys.readouterr()

    assert cli(monkeypatch, "data", "inspect", str(out)) == ExitCode.OK

    shown = flat(capsys.readouterr().out)
    assert "Chat template" in shown
    assert "trained on assistant" in shown
    assert "Loss mask" not in shown


def test_inspect_of_a_chat_corpus_says_the_mask_was_only_measured(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The third state of the Loss mask row, and the only one nothing else reaches.

    ``inspect`` reads a corpus and writes no dataset, so it can report the assistant
    share but has no file to point at. The two tests either side of this one --
    ``test_prepare_reads_typed_conversations_and_reports_the_trained_share`` and
    ``test_no_loss_mask_writes_a_plain_dataset_and_says_so`` -- pin the two states a
    ``prepare`` run can produce, both of which are about a file on disk.

    Collapsing this state into either of those is what the row exists to prevent: one
    would promise a mask that was never written, the other would say ``--no-loss-mask``
    to a user who never passed it.
    """
    corpus = tmp_path / "chats"
    chat_corpus(corpus, 40)

    code = cli(monkeypatch, "data", "inspect", str(corpus), "--jsonl-messages-field", "messages")

    captured = capsys.readouterr()
    assert code == ExitCode.OK, captured
    shown = flat(captured.out)
    assert "are assistant replies" in shown
    assert "measured only" in shown
    assert "`trainai data prepare` writes it to disk" in shown
    # Not the wording of either other state.
    assert "written beside every shard" not in shown
    assert "--no-loss-mask was passed" not in shown


def test_no_loss_mask_writes_a_plain_dataset_and_says_so(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Opting out has to be visible, or the absent file looks like a bug."""
    corpus = tmp_path / "chats"
    chat_corpus(corpus, 40)
    out = tmp_path / "prepared"

    code = cli(
        monkeypatch,
        "data",
        "prepare",
        str(corpus),
        "--out",
        str(out),
        "--jsonl-messages-field",
        "messages",
        "--no-loss-mask",
        "--vocab-size",
        "512",
    )

    captured = capsys.readouterr()
    assert code == ExitCode.OK, captured
    assert "--no-loss-mask was passed" in flat(captured.out)
    assert list(out.glob("*.mask.bin")) == []
    assert "mask" not in (out / "manifest.json").read_text(encoding="utf-8")


def test_loss_mask_without_typed_conversations_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A mask over prose would mark every token, which is a mask that means nothing.

    Refused rather than silently written as all ones: the user asked for the prompts
    to be excluded from the loss, and this corpus has no prompts to exclude.
    """
    corpus = tmp_path / "prose"
    corpus.mkdir()
    (corpus / "a.txt").write_text(
        "The harbour clock measured every tide. " * 200, encoding="utf-8", newline="\n"
    )
    out = tmp_path / "prepared"

    code = cli(monkeypatch, "data", "prepare", str(corpus), "--out", str(out), "--loss-mask")

    assert code == ExitCode.USAGE
    assert "typed conversations" in flat(capsys.readouterr().err)
    assert not (out / "manifest.json").exists()


def test_prepare_refuses_both_field_flags_at_once(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """They describe different records, and either choice trains on the wrong text."""
    corpus = tmp_path / "chats"
    chat_corpus(corpus, 4)
    out = tmp_path / "prepared"

    code = cli(
        monkeypatch,
        "data",
        "prepare",
        str(corpus),
        "--out",
        str(out),
        "--jsonl-field",
        "text",
        "--jsonl-messages-field",
        "messages",
    )

    assert code == ExitCode.USAGE
    assert "cannot both be given" in flat(capsys.readouterr().err)
    assert not (out / "manifest.json").exists()


def test_prepare_refuses_a_misspelt_encoding(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The whole reason the check exists: this is where the typo is made.

    ``--encoding`` is declared as TEXT and reaches the reader untouched, so before
    it was checked this command printed ``LookupError: unknown encoding: uft-8`` and
    a traceback through ``<frozen codecs>``. Reached from here rather than from the
    constructor because the exit code is half the fix -- a usage error is 2, and a
    crash is not -- and because a corpus that is perfectly readable must be left
    alone: nothing is written, and the mistake is in the command, not the data.
    ``test_an_encoding_that_is_no_codec_at_all_is_refused_where_it_was_typed`` is
    the same refusal one layer down, with the message and the near miss on it.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text(
        "some plain and perfectly readable text\n", encoding="utf-8", newline="\n"
    )
    out = tmp_path / "prepared"

    code = cli(
        monkeypatch, "data", "prepare", str(corpus), "--out", str(out), "--encoding", "uft-8"
    )

    err = flat(capsys.readouterr().err)
    assert code == ExitCode.USAGE
    assert "--encoding was given as 'uft-8'" in err
    assert "Did you mean utf-8?" in err
    assert "LookupError" not in err and "Traceback" not in err
    assert not out.exists(), "a usage error wrote an output directory"


def test_inspect_reports_a_table_and_column_it_chose_itself(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """One table, one conventional column name: nothing to ask, so it must be told.

    A guess made on the user's behalf that never reaches the screen is the failure
    mode these two rows exist to prevent.
    """
    corpus = tmp_path / "corpus.db"
    review_database(corpus, extra_table=False, column="review")

    code = cli(monkeypatch, "data", "inspect", str(corpus))

    captured = capsys.readouterr()
    assert code == ExitCode.OK, captured
    shown = flat(captured.out)
    assert "DB table" in shown
    assert "DB column" in shown
    assert "chosen automatically" in shown


def test_prepare_refuses_a_table_that_was_renamed_to_txt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The regression that matters: renaming the file used to make the check vanish."""
    corpus = tmp_path / "renamed"
    numeric_table(corpus / "weather.txt")
    out = tmp_path / "prepared"

    code = cli(monkeypatch, "data", "prepare", str(corpus), "--out", str(out))

    assert code == ExitCode.DATASET
    error = flat(capsys.readouterr().err)
    assert "comma-separated fields" in error
    assert "--allow-tabular" in error
    assert not (out / "manifest.json").exists()


def test_allow_tabular_prepares_and_records_the_override(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The override downgrades the finding and writes it down. It never deletes it."""
    corpus = tmp_path / "renamed"
    numeric_table(corpus / "weather.txt")
    out = tmp_path / "prepared"

    code = cli(
        monkeypatch,
        "data",
        "prepare",
        str(corpus),
        "--out",
        str(out),
        "--allow-tabular",
        "--vocab-size",
        "512",
        # One small file is one document, and the content-hash split cannot hold
        # a fraction of one out. Irrelevant to what this test is about.
        "--val-fraction",
        "0",
    )

    assert code == ExitCode.OK, capsys.readouterr()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    codes = [issue["code"] for issue in manifest["validation"]["issues"]]
    assert "tabular_override" in codes
    assert "looks_like_a_table" not in codes


def test_inspect_reports_on_a_raw_corpus_without_writing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    many_document_corpus: Path,
) -> None:
    before = sorted(p.name for p in many_document_corpus.iterdir())

    code = cli(monkeypatch, "data", "inspect", str(many_document_corpus))

    assert code == ExitCode.OK
    output = flat(capsys.readouterr().out)
    assert "Documents" in output
    assert "estimate only" in output  # the one number that is not measured
    assert sorted(p.name for p in many_document_corpus.iterdir()) == before


def test_inspect_of_a_raw_corpus_can_show_a_sample(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    many_document_corpus: Path,
) -> None:
    code = cli(monkeypatch, "data", "inspect", str(many_document_corpus), "--sample", "40")

    assert code == ExitCode.OK
    assert "First document" in capsys.readouterr().out


def test_inspect_names_the_codec_rather_than_calling_everything_gzipped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A bz2 corpus reported as "gzipped" is a small lie that costs someone an hour."""
    corpus = tmp_path / "squashed"
    corpus.mkdir()
    with bz2.open(corpus / "doc.txt.bz2", "wt", encoding="utf-8", newline="\n") as handle:
        handle.write("prose that is long enough to inspect. " * 40000)

    code = cli(monkeypatch, "data", "inspect", str(corpus))

    assert code == ExitCode.OK
    output = capsys.readouterr().out
    assert "bz2-compressed" in output
    assert "gzipped" not in output


def test_inspect_of_an_archive_names_it_and_qualifies_the_size(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A member's size is what it unpacks to, so "on disk" would be a wrong label.

    The mix counts archive *files*, not members: three documents out of one zip is
    "from 1 zip", which is the thing the user pointed at.
    """
    path = tmp_path / "papers.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("a.txt", "b.txt", "c.txt"):
            archive.writestr(name, "prose that is long enough to inspect. " * 15000)

    code = cli(monkeypatch, "data", "inspect", str(path))

    assert code == ExitCode.OK
    output = flat(capsys.readouterr().out)
    assert "from 1 zip" in output
    assert "archive members counted unpacked" in output
    assert "Size on disk" not in output


def test_inspect_of_an_archive_names_what_it_passed_over(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """An archive is opaque: a member dropped without a name on screen is invisible."""
    path = tmp_path / "papers.zip"
    inner = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("deep.txt", "deep\n")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("doc.txt", "prose that is long enough to inspect. " * 40000)
        archive.writestr("scan.pdf", b"%PDF-1.4")
        archive.writestr("inner.zip", inner.read_bytes())

    code = cli(monkeypatch, "data", "inspect", str(path))

    assert code == ExitCode.OK
    captured = capsys.readouterr().out
    output, tokens = flat(captured), unwrapped(captured)
    assert "scan.pdf" in tokens
    assert "inner.zip" in tokens
    assert "rather than opened" in output


def test_inspect_of_a_word_document_reports_it_as_docx(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The file mix names the kind, so "1 docx" has to appear without a table entry.

    The report is also the only place the split note's grammar shows up, and a
    document long enough to be cut is what makes it fire.
    """
    word = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    paragraph = "<w:p><w:r><w:t>prose that is long enough to inspect.</w:t></w:r></w:p>"
    path = tmp_path / "report.docx"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr(
            "word/document.xml",
            f'<w:document xmlns:w="{word}"><w:body>{paragraph * 40000}</w:body></w:document>',
        )

    code = cli(monkeypatch, "data", "inspect", str(path))

    assert code == ExitCode.OK
    output = capsys.readouterr().out
    flowed = " ".join(output.split())
    assert "1 docx" in flowed
    assert "Size on disk" in flowed
    assert "longer than the per-document limit" in flowed


def test_inspect_of_a_file_with_no_extension_says_it_assumed_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The assumption is TrainAI's, so it belongs on screen rather than in the result."""
    path = tmp_path / "corpus"
    path.write_text(
        "prose that is long enough to inspect. " * 40000, encoding="utf-8", newline="\n"
    )

    code = cli(monkeypatch, "data", "inspect", str(path))

    assert code == ExitCode.OK
    output = capsys.readouterr().out
    assert "Read as text" in output
    assert "no extension" in output


def test_inspect_names_the_files_a_walk_passed_over(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """One file read out of a directory of three has to account for the other two."""
    corpus = tmp_path / "mixed"
    corpus.mkdir()
    (corpus / "corpus.txt").write_text(
        "prose that is long enough to inspect. " * 40000, encoding="utf-8", newline="\n"
    )
    (corpus / "scan.pdf").write_bytes(b"%PDF-1.4")
    (corpus / "NOTICE").write_text("a licence notice\n", encoding="utf-8", newline="\n")

    code = cli(monkeypatch, "data", "inspect", str(corpus))

    assert code == ExitCode.OK
    output = capsys.readouterr().out
    # Collapsed, because the report wraps to the terminal width and a phrase this
    # long is split across lines at an arbitrary point.
    flowed = " ".join(output.split())
    assert "Passed over" in flowed
    assert "scan.pdf" in flowed
    assert "NOTICE" in flowed
    assert "1 of them has no extension at all" in flowed


def test_inspect_counts_the_passed_over_files_it_did_not_name(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The row names three files and counts the rest, and this is the "rest".

    ``test_inspect_names_the_files_a_walk_passed_over`` has two of them, so the tail
    of that row never runs there: the list is complete and the count is redundant. A
    directory of five is the case where truncating matters, and the difference has to
    stay visible -- a row reading "5" over a list of three looks like a bug in the
    count, which is why the missing two are said out loud rather than left implied.

    The whole row is asserted rather than the phrase, because ``validate._first_few``
    writes the same ``, and 2 more`` into the finding printed further down. An
    assertion on the phrase alone passes with the row's count deleted entirely, which
    is what a mutation run showed the first version of this test doing.
    """
    corpus = tmp_path / "scans"
    corpus.mkdir()
    (corpus / "corpus.txt").write_text(
        "prose that is long enough to inspect. " * 40000, encoding="utf-8", newline="\n"
    )
    for name in ("a.pdf", "b.pdf", "c.pdf", "d.pdf", "e.pdf"):
        (corpus / name).write_bytes(b"%PDF-1.4")

    code = cli(monkeypatch, "data", "inspect", str(corpus))

    captured = capsys.readouterr()
    assert code == ExitCode.OK, captured
    shown = flat(captured.out)
    assert "Passed over 5 a.pdf, b.pdf, c.pdf, and 2 more" in shown
    # The two the count stands in for are not among the names: the walk is sorted, so
    # which three were kept is not a coincidence of dict ordering.
    assert "e.pdf" not in shown


def test_inspect_names_the_files_it_could_not_decode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """``--on-error skip`` keeps going, so the report is the only record of what it lost.

    ``test_on_error_skip_records_the_file_and_keeps_reading`` in ``test_data_ingest``
    checks the counter; this checks that the counter reaches the screen. Without the
    flag the same corpus is a refusal, and that half is asserted here too, because a
    row saying "1 skipped" is only reassuring if the default is still to stop.

    The bad file is raw invalid UTF-8 with no byte-order mark. A ``\\xff\\xfe`` prefix
    would not fail at all: ``_effective_encoding`` honours a BOM when the requested
    codec is utf-8, so the file would decode as UTF-16 and this test would pass while
    proving nothing.
    """
    corpus = tmp_path / "mixed"
    corpus.mkdir()
    (corpus / "aaa_good.txt").write_text(
        "prose that is long enough to inspect. " * 2000, encoding="utf-8", newline="\n"
    )
    (corpus / "zzz_bad.txt").write_bytes(b"a readable prefix, then: \xff and more text\n" * 50)

    assert cli(monkeypatch, "data", "inspect", str(corpus)) == ExitCode.DATASET
    assert "zzz_bad.txt" in unwrapped(capsys.readouterr().err)

    code = cli(monkeypatch, "data", "inspect", str(corpus), "--on-error", "skip")

    captured = capsys.readouterr()
    assert code == ExitCode.OK, captured
    shown = flat(captured.out)
    assert "Files skipped" in shown
    assert "could not be decoded" in shown
    assert "zzz_bad.txt" in unwrapped(captured.out)


def test_inspect_counts_short_and_malformed_records_it_dropped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Two counters, two rows, and both are omitted when they are zero.

    ``test_data_ingest`` covers the counting -- ``test_on_error_skip_counts_malformed_records``
    for the second of these -- and nothing covered the reporting. They are asserted
    together because a JSON Lines corpus is where both happen at once, and because
    each is a different kind of loss: a document below ``--min-doc-chars`` was read
    and rejected, a malformed record was never read at all. A user comparing "Documents
    kept" against their line count needs both numbers to make the arithmetic close.
    """
    corpus = tmp_path / "jsonl"
    corpus.mkdir()
    lines = [
        json.dumps({"text": f"Document number {index}. " + "long enough to keep. " * 30})
        for index in range(30)
    ]
    lines += [json.dumps({"text": "tiny"}) for _ in range(5)]
    lines += ["{not json at all", json.dumps({"text": 17})]
    (corpus / "docs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    code = cli(
        monkeypatch,
        "data",
        "inspect",
        str(corpus),
        "--on-error",
        "skip",
        "--min-doc-chars",
        "100",
    )

    captured = capsys.readouterr()
    assert code == ExitCode.OK, captured
    shown = flat(captured.out)
    assert "Documents kept 30" in shown
    assert "Too short 5 dropped" in shown
    assert "Malformed records 2 skipped" in shown


def test_inspect_reports_the_lost_bytes_a_corpus_already_holds(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """U+FFFD in the file is not a decode failure, which is why it needs saying.

    The corpus below is valid UTF-8: someone else's reader already replaced the bytes
    it could not read, and TrainAI would train on the replacement character as a
    token like any other. ``test_inspect_names_the_files_it_could_not_decode`` is the
    opposite case -- bytes this reader cannot decode -- and it fails loudly, so the
    quiet one is the one a user needs told about.
    """
    corpus = tmp_path / "lossy"
    corpus.mkdir()
    # Built with chr(), not written as the character: this file is ASCII by convention.
    lost = chr(0xFFFD)
    (corpus / "a.txt").write_text(
        f"prose with a lost byte {lost} in it. " * 40000, encoding="utf-8", newline="\n"
    )

    code = cli(monkeypatch, "data", "inspect", str(corpus))

    captured = capsys.readouterr()
    assert code == ExitCode.OK, captured
    shown = flat(captured.out)
    assert "Character mix" in shown
    assert "U+FFFD" in shown
    assert "40,000 U+FFFD" in shown


def test_inspect_says_when_the_duplicate_check_hit_its_memory_cap(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A duplicate rate measured over part of the corpus is a lower bound, not a rate.

    The cap is two million hashes, so reaching this on a real corpus means a corpus
    of two million documents -- which is why the cap is lowered here instead. What is
    being tested is the reporting, not the number: past the cap new documents stop
    being remembered, so a duplicate of one of them is counted as unique and the rate
    can only be understated. ``test_data_analyze``'s advice cases build a report with
    ``duplicate_check_truncated=True`` by hand; this drives the flag from the code
    that actually sets it, so the two cannot drift apart.
    """
    monkeypatch.setattr("trainai.data.analyze._DEDUP_CAP", 4)
    corpus = tmp_path / "dupes"
    corpus.mkdir()
    lines = [
        json.dumps({"text": f"Document number {index}. " + "long enough to keep. " * 30})
        for index in range(12)
    ]
    (corpus / "docs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    code = cli(monkeypatch, "data", "inspect", str(corpus))

    captured = capsys.readouterr()
    assert code == ExitCode.OK, captured
    shown = flat(captured.out)
    assert "Duplicates" in shown
    assert "check stopped at its memory cap" in shown
    assert "this is a lower bound" in shown


def test_prepare_refuses_a_binary_file_with_no_extension(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Reading it as text would succeed, which is what makes the guard necessary."""
    path = tmp_path / "corpus"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 40)

    code = cli(monkeypatch, "data", "prepare", str(path), "--out", str(tmp_path / "out"))

    assert code == ExitCode.DATASET
    assert "looks like a binary file" in capsys.readouterr().err


def test_inspect_exits_three_on_a_corpus_that_cannot_be_trained_on(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """So it works as a check in a script, not only as something to read."""
    corpus = tmp_path / "tiny"
    corpus.mkdir()
    (corpus / "tiny.txt").write_text("too short\n", encoding="utf-8", newline="\n")

    code = cli(monkeypatch, "data", "inspect", str(corpus))

    assert code == ExitCode.DATASET
    assert "megabyte" in capsys.readouterr().err


def test_inspect_reports_on_a_prepared_dataset(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    many_document_corpus: Path,
    tmp_path: Path,
) -> None:
    out = tmp_path / "prepared"
    assert (
        cli(
            monkeypatch,
            "data",
            "prepare",
            str(many_document_corpus),
            "--out",
            str(out),
            "--vocab-size",
            "512",
        )
        == ExitCode.OK
    )
    capsys.readouterr()

    code = cli(monkeypatch, "data", "inspect", str(out), "--layout")

    assert code == ExitCode.OK
    output = capsys.readouterr().out
    assert "trainai-dataset" in output
    assert "Content hash" in output
    assert "train_00000.bin" in output  # the layout section
    # Without --verify the report must say what it did and did not check.
    assert "--verify" in output
    # A plain corpus grows no rows: neither of the two that a chat corpus adds. The
    # negative half of the pair, so a template recorded for every dataset -- which
    # would make "rendered as chat" unanswerable -- fails here rather than reading as
    # a harmless extra line.
    assert "Chat template" not in output
    assert "Loss mask" not in output


def test_inspect_verify_catches_a_corrupted_shard(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    many_document_corpus: Path,
    tmp_path: Path,
) -> None:
    out = tmp_path / "prepared"
    assert (
        cli(
            monkeypatch,
            "data",
            "prepare",
            str(many_document_corpus),
            "--out",
            str(out),
            "--vocab-size",
            "512",
        )
        == ExitCode.OK
    )
    shard = next(iter(sorted(out.glob("train_*.bin"))))
    payload = bytearray(shard.read_bytes())
    payload[8] ^= 0x01
    shard.write_bytes(bytes(payload))
    capsys.readouterr()

    assert cli(monkeypatch, "data", "inspect", str(out)) == ExitCode.OK  # sizes only
    capsys.readouterr()

    code = cli(monkeypatch, "data", "inspect", str(out), "--verify")

    assert code == ExitCode.DATASET
    assert "checksum" in capsys.readouterr().err


def test_inspect_json_is_machine_readable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    many_document_corpus: Path,
) -> None:
    code = cli(monkeypatch, "data", "inspect", str(many_document_corpus), "--json")

    assert code == ExitCode.OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "corpus"
    assert payload["corpus"]["documents"] > 0
    assert payload["validation"]["ok"] is True


def test_inspect_json_of_a_prepared_dataset_emits_the_manifest_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    many_document_corpus: Path,
    tmp_path: Path,
) -> None:
    """The other branch of ``--json``, and it is a different payload entirely.

    ``test_inspect_json_is_machine_readable`` covers a raw corpus, which reports
    ``kind: corpus`` and a measurement. Point the same command at a prepared
    directory and it returns the manifest -- ``inspect`` chooses by whether a
    manifest is there, not by a flag, so ``--json`` has two shapes and a script that
    parses one of them will not parse the other.

    Asserted by parsing the whole of stdout rather than by searching it, because the
    contract is that a table was *not* printed alongside: the same reason
    ``test_prepare_json_emits_only_the_manifest`` exists for the writing command.
    ``checked`` says which verification actually ran, since the default is the size
    check and only ``--verify`` re-hashes.
    """
    out = tmp_path / "prepared"
    assert (
        cli(
            monkeypatch,
            "data",
            "prepare",
            str(many_document_corpus),
            "--out",
            str(out),
            "--vocab-size",
            "512",
        )
        == ExitCode.OK
    )
    capsys.readouterr()

    code = cli(monkeypatch, "data", "inspect", str(out), "--json")

    captured = capsys.readouterr()
    assert code == ExitCode.OK, captured
    payload = json.loads(captured.out)
    assert payload["format"] == "trainai-dataset"
    assert payload["checked"] == "sizes"
    assert payload["totals"]["tokens"] > 0
    assert payload["splits"]["train"]["shards"]

    assert cli(monkeypatch, "data", "inspect", str(out), "--json", "--verify") == ExitCode.OK
    assert json.loads(capsys.readouterr().out)["checked"] == "checksums"


def test_inspect_reports_tokens_that_straddle_a_reply_boundary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The row that keeps the trained share honest when the corpus is unkind.

    A token overlapping a span edge is scored on characters the mask calls context,
    so the share printed above it is approximate. It is a property of the corpus and
    the tokenizer, not a failure -- and it does not happen here:
    ``test_the_straddling_count_is_zero_on_the_chat_template`` in
    ``test_data_binarize`` proves the template ends its spans on a whitespace run,
    which is always a token boundary, so every corpus this repo can build reports
    zero.

    The count is therefore written into the manifest, which is where the row reads it
    from in any case. Editing it is safe: ``verify_dataset`` checks shard sizes and
    checksums, and ``totals`` is not part of either, so the dataset is still valid --
    only its reported straddle count changed.
    """
    corpus = tmp_path / "chats"
    chat_corpus(corpus, 40)
    out = tmp_path / "prepared"
    assert (
        cli(
            monkeypatch,
            "data",
            "prepare",
            str(corpus),
            "--out",
            str(out),
            "--jsonl-messages-field",
            "messages",
            "--vocab-size",
            "512",
        )
        == ExitCode.OK
    )
    capsys.readouterr()

    path = out / "manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    # Asserted, not assumed: if the template ever stopped ending spans on a
    # whitespace run this test would be editing a number that was already non-zero,
    # and would still pass while testing nothing.
    assert document["totals"]["straddling_tokens"] == 0
    document["totals"]["straddling_tokens"] = 7
    path.write_text(json.dumps(document, indent=2), encoding="utf-8", newline="\n")

    assert cli(monkeypatch, "data", "inspect", str(out)) == ExitCode.OK

    shown = flat(capsys.readouterr().out)
    assert "7 tokens straddle a reply boundary" in shown
    assert "each is counted as a target" in shown


# --------------------------------------------------------------------------- #
# The CLI's literal defaults must not drift from the library's
# --------------------------------------------------------------------------- #
def test_data_flag_defaults_match_the_library() -> None:
    """``main.py`` spells its defaults out so ``--help`` stays free of heavy imports.

    That duplication is only safe if something checks it, which is this.
    """
    from trainai.cli._click import is_group
    from trainai.data import DEFAULT_SHARD_TOKENS, DEFAULT_VAL_FRACTION, IngestOptions

    ingest = IngestOptions()
    expected = {
        ("prepare", "--val-fraction"): DEFAULT_VAL_FRACTION,
        ("prepare", "--shard-tokens"): DEFAULT_SHARD_TOKENS,
        ("prepare", "--encoding"): ingest.encoding,
        ("prepare", "--jsonl-field"): ingest.jsonl_field,
        ("prepare", "--jsonl-messages-field"): ingest.jsonl_messages_field,
        ("prepare", "--csv-text-column"): ingest.csv_text_column,
        ("prepare", "--db-table"): ingest.db_table,
        ("prepare", "--min-doc-chars"): ingest.min_doc_chars,
        ("prepare", "--max-doc-chars"): ingest.max_doc_chars,
        ("prepare", "--on-error"): ingest.on_error,
        # None, not False: the flag has three states, and a False default would turn
        # "decide from the corpus" into "off" for every typed-conversation run.
        ("prepare", "--loss-mask"): None,
        ("inspect", "--encoding"): ingest.encoding,
        ("inspect", "--jsonl-field"): ingest.jsonl_field,
        ("inspect", "--jsonl-messages-field"): ingest.jsonl_messages_field,
        ("inspect", "--csv-text-column"): ingest.csv_text_column,
        ("inspect", "--db-table"): ingest.db_table,
        ("inspect", "--min-doc-chars"): ingest.min_doc_chars,
        ("inspect", "--max-doc-chars"): ingest.max_doc_chars,
        ("inspect", "--on-error"): ingest.on_error,
    }

    group = typer_to_click(app)
    assert is_group(group)
    data_group = group.commands["data"]
    assert is_group(data_group)

    for (command, flag), want in expected.items():
        params = data_group.commands[command].params
        param = next(p for p in params if flag in p.opts)
        found = param.default.value if hasattr(param.default, "value") else param.default
        assert found == want, f"data {command} {flag}: CLI has {found!r}, library {want!r}"


def test_run_prepare_defaults_match_the_cli_flags() -> None:
    """The programmatic entry point and the command line must agree, too."""
    import inspect as inspect_module

    from trainai.cli._click import is_group
    from trainai.cli.data import run_prepare

    signature = inspect_module.signature(run_prepare)
    group = typer_to_click(app)
    assert is_group(group)
    data_group = group.commands["data"]
    assert is_group(data_group)
    command = data_group.commands["prepare"]

    for parameter in signature.parameters.values():
        if parameter.default is inspect_module.Parameter.empty:
            continue
        flag = "--" + parameter.name.replace("_", "-")
        matches = [p for p in command.params if flag in p.opts]
        if not matches:
            continue  # json_output is spelled --json; checked separately below
        found = matches[0].default
        found = found.value if hasattr(found, "value") else found
        assert found == parameter.default, (
            f"{flag}: CLI {found!r}, run_prepare {parameter.default!r}"
        )


def test_run_train_defaults_match_the_cli_flags() -> None:
    """The same duplication as ``prepare``, for the two commands that build a run.

    ``train`` and ``finetune`` each spell every default as a typer literal and then hand
    the values to ``run_train`` / ``run_finetune``, which spell them again. Nothing
    checked that the two agreed, and the failure mode is invisible: a flag whose CLI
    default is ``False`` where the library's is ``None`` does not error, it just makes
    the un-typed case mean something else. ``--loss-mask`` is exactly that shape -- three
    states, where ``False`` would silently mean "score every token" on every chat run.
    """
    import inspect as inspect_module

    from trainai.cli._click import is_group
    from trainai.cli.train import run_finetune, run_train

    group = typer_to_click(app)
    assert is_group(group)

    for name, function in (("train", run_train), ("finetune", run_finetune)):
        command = group.commands[name]
        for parameter in inspect_module.signature(function).parameters.values():
            if parameter.default is inspect_module.Parameter.empty:
                continue
            flag = "--" + parameter.name.replace("_", "-")
            matches = [p for p in command.params if flag in p.opts]
            if not matches:
                continue  # --json, --preset, --context and --from are spelled otherwise
            found = matches[0].default
            found = found.value if hasattr(found, "value") else found
            assert found == parameter.default, (
                f"{name} {flag}: CLI {found!r}, library {parameter.default!r}"
            )


def test_the_loss_mask_flag_defaults_to_deciding_from_the_dataset() -> None:
    """Asserted by name, because the value that matters here is ``None``.

    ``TrainConfig.loss_mask`` is ``None`` for "apply the mask when the dataset has one",
    which is what makes a chat dataset and a plain-text dataset both do the right thing
    with no flag. A CLI default of ``False`` would turn that into "never", and every
    number the run reported would still look reasonable.
    """
    from trainai.cli._click import is_group
    from trainai.train.config import TrainConfig

    group = typer_to_click(app)
    assert is_group(group)
    for name in ("train", "finetune"):
        params = group.commands[name].params
        param = next(p for p in params if "--loss-mask" in p.opts)
        assert param.default is None
        assert "--no-loss-mask" in param.secondary_opts
    assert TrainConfig(steps=10).loss_mask is None


# --------------------------------------------------------------------------- #
# The wiring between a command and the function behind it
# --------------------------------------------------------------------------- #
#: Where the implementations live. Each command in ``cli/main.py`` is a Typer callback
#: that collects flags and forwards them to one of these, imported inside the callback
#: so that ``--help`` does not pay for torch.
IMPLEMENTATION_MODULES = (
    "chat",
    "data",
    "doctor",
    "eval",
    "export",
    "plan",
    "quickstart",
    "setup",
    "train",
)


def implementations() -> dict[str, Any]:
    """Every ``run_*`` entry point in ``trainai.cli``, by name.

    Names are unique across the package, which is what lets a call site in
    ``cli/main.py`` be resolved from the call alone. Filtered by ``__module__`` because
    several of these import each other -- ``quickstart`` calls four of them by hand --
    so a plain scan of module contents sees the same function under two homes.
    """
    import importlib

    found: dict[str, Any] = {}
    for name in IMPLEMENTATION_MODULES:
        module = importlib.import_module(f"trainai.cli.{name}")
        for attribute, value in vars(module).items():
            if (
                not attribute.startswith("run_")
                or getattr(value, "__module__", "") != module.__name__
            ):
                continue
            assert attribute not in found, f"two modules define {attribute}"
            found[attribute] = value
    return found


def forwarding_calls() -> list[tuple[int, str, list[str], int]]:
    """Every ``run_*(...)`` call in ``cli/main.py``, as ``(line, name, keywords, positional)``.

    Read from the source rather than by calling anything: the mismatch this finds is a
    :exc:`TypeError` raised at the moment the user runs the command, and every one of
    these commands does real work, so reaching it by running them is not the cheap path.
    """
    import ast

    source = Path("src/trainai/cli/main.py")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    calls: list[tuple[int, str, list[str], int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if not node.func.id.startswith("run_"):
            continue
        keywords = [keyword.arg for keyword in node.keywords if keyword.arg is not None]
        calls.append((node.lineno, node.func.id, keywords, len(node.args)))
    return calls


def test_there_are_forwarding_calls_to_check() -> None:
    """Guards the walk itself: a "no mismatches" assertion passes on an empty list.

    One call per command, and the CLI has eleven leaf commands that do work.
    """
    calls = forwarding_calls()
    assert len(calls) >= 10, f"the walk sees almost no run_* calls: {calls}"
    assert {name for _line, name, _kw, _n in calls} <= implementations().keys()


def test_every_flag_a_command_forwards_is_one_its_implementation_accepts() -> None:
    """Regression: ``trainai quickstart`` could not run at all, on any input.

    ``--jsonl-messages-field`` was added to the ``quickstart`` callback and forwarded to
    ``run_quickstart``, which did not take it, so every invocation died on a
    ``TypeError`` -- a traceback, since only ``TrainAIError`` is rendered without one.
    Nothing caught it because the whole quickstart suite calls ``run_quickstart``
    directly: the callback that assembles the arguments was never executed by a test, and
    per-module coverage is satisfied by the other ten commands going through that file.

    Checked statically because the alternative is running eleven commands that each
    prepare a corpus and train a model, and because a signature mismatch is knowable
    without running anything.
    """
    import inspect

    known = implementations()
    problems: list[str] = []
    for line, name, keywords, _positional in forwarding_calls():
        parameters = inspect.signature(known[name]).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            continue
        for keyword in keywords:
            if keyword not in parameters:
                problems.append(f"main.py:{line} passes {keyword}= to {name}(), which has no such")
    assert problems == [], "\n".join(problems)


def test_every_argument_an_implementation_requires_is_one_the_command_passes() -> None:
    """The mirror of the check above, and the same ``TypeError`` from the other side.

    A new required argument on a ``run_*`` function breaks its command until every call
    site is updated, and the call sites are in a different file from the signature.
    """
    import inspect

    known = implementations()
    problems: list[str] = []
    for line, name, keywords, positional in forwarding_calls():
        parameters = list(inspect.signature(known[name]).parameters.values())
        # The positional arguments cover the first ``positional`` parameters that can
        # take one; anything keyword-only has to be named or defaulted.
        positional_capable = [
            p
            for p in parameters
            if p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        covered = {p.name for p in positional_capable[:positional]} | set(keywords)
        for parameter in parameters:
            if parameter.default is not inspect.Parameter.empty:
                continue
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                continue
            if parameter.name not in covered:
                problems.append(f"main.py:{line} calls {name}() without {parameter.name}")
    assert problems == [], "\n".join(problems)
