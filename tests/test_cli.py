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


def review_table(path: Path, *, rows: int = 300) -> None:
    """A table whose ``note`` column holds real sentences, and nothing else does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subjects = ("the harbour", "the orchard", "the lantern", "the bridge")
    lines = ["stars,note"]
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
        ("prepare", "--csv-text-column"): ingest.csv_text_column,
        ("prepare", "--db-table"): ingest.db_table,
        ("prepare", "--min-doc-chars"): ingest.min_doc_chars,
        ("prepare", "--max-doc-chars"): ingest.max_doc_chars,
        ("prepare", "--on-error"): ingest.on_error,
        ("inspect", "--encoding"): ingest.encoding,
        ("inspect", "--jsonl-field"): ingest.jsonl_field,
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
