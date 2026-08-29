"""Tests for :mod:`trainai.data.ingest`.

Three contracts are guarded here, because breaking any of them silently produces
a dataset that is wrong rather than a run that fails:

* Discovery is deterministic. Document order decides tokenizer merges, which
  decides shard contents, which decides every checksum in the manifest.
* A decoding failure names the file and the byte offset. "This file is not UTF-8"
  is not actionable; "byte 30 of broken.txt" is.
* Every document that is dropped is counted. A file skipped without a number
  attached to it is the one failure mode a user cannot notice.
"""

from __future__ import annotations

import bz2
import csv
import gzip
import io
import json
import lzma
import re
import sqlite3
import struct
import subprocess
import sys
import tarfile
import threading
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any, get_args

import pytest

from trainai.data import ingest as ingest_module
from trainai.data.ingest import (
    _BOMS,
    _JSON_MAX_BYTES,
    CUT_BY_MAX_DOC_CHARS,
    Document,
    IngestOptions,
    Ingestor,
    IngestStats,
    Kind,
    SourceFile,
    describe_supported_formats,
)
from trainai.errors import DatasetDecodeError, DatasetFormatError, DatasetNotFoundError


def ingest(root: Path, **options: object) -> tuple[list[Document], IngestStats]:
    """Read every document under ``root``, returning them with the run's counters."""
    ingestor = Ingestor(IngestOptions(**options))  # type: ignore[arg-type]
    return list(ingestor.documents(ingestor.discover(root))), ingestor.stats


def write(path: Path, text: str) -> Path:
    """Write text with byte-for-byte identical results on every platform."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def jsonl(path: Path, texts: list[str], field: str = "text") -> Path:
    return write(path, "\n".join(json.dumps({field: text}) for text in texts) + "\n")


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def test_discovers_text_files_in_a_stable_order(tmp_path: Path) -> None:
    for name in ("zebra.txt", "apple.txt", "middle.md"):
        write(tmp_path / name, "some text\n")

    first = [source.relative for source in Ingestor().discover(tmp_path)]
    second = [source.relative for source in Ingestor().discover(tmp_path)]

    assert first == sorted(first)
    assert first == second


def test_discovers_nested_files_with_posix_relative_paths(tmp_path: Path) -> None:
    write(tmp_path / "sub" / "deeper" / "doc.txt", "nested text\n")

    sources = Ingestor().discover(tmp_path)

    assert [source.relative for source in sources] == ["sub/deeper/doc.txt"]


def test_accepts_a_single_file_as_the_corpus(tmp_path: Path) -> None:
    path = write(tmp_path / "only.txt", "just this one\n")

    sources = Ingestor().discover(path)

    assert [source.relative for source in sources] == ["only.txt"]


def test_hidden_files_and_directories_are_ignored(tmp_path: Path) -> None:
    write(tmp_path / "visible.txt", "keep me\n")
    write(tmp_path / ".hidden.txt", "skip me\n")
    write(tmp_path / ".git" / "notes.txt", "skip me too\n")

    sources = Ingestor().discover(tmp_path)

    assert [source.relative for source in sources] == ["visible.txt"]


def test_unsupported_suffixes_are_not_discovered(tmp_path: Path) -> None:
    write(tmp_path / "keep.txt", "text\n")
    write(tmp_path / "skip.xlsx", "not really a spreadsheet\n")
    (tmp_path / "skip.pdf").write_bytes(b"%PDF-1.4")

    sources = Ingestor().discover(tmp_path)

    assert [source.relative for source in sources] == ["keep.txt"]


def test_missing_path_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotFoundError) as caught:
        Ingestor().discover(tmp_path / "not-here")

    assert "not-here" in str(caught.value)
    assert caught.value.hint


def test_empty_directory_lists_the_supported_formats(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotFoundError) as caught:
        Ingestor().discover(tmp_path)

    assert describe_supported_formats() in (caught.value.hint or "")


def test_a_directory_of_json_is_discovered(tmp_path: Path) -> None:
    """A folder of .json files used to be refused outright; now it is read."""
    write(tmp_path / "records.json", '[{"text": "a"}]')

    sources = Ingestor().discover(tmp_path)

    assert [(source.relative, source.kind) for source in sources] == [("records.json", "json")]


def test_single_unsupported_file_names_its_suffix(tmp_path: Path) -> None:
    path = write(tmp_path / "data.xlsx", "not really a spreadsheet\n")

    with pytest.raises(DatasetNotFoundError) as caught:
        Ingestor().discover(path)

    assert caught.value.details["suffix"] == ".xlsx"


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def test_reads_plain_text(tmp_corpus: Path) -> None:
    documents, stats = ingest(tmp_corpus)

    assert stats.files_read == 2
    assert stats.documents_emitted == len(documents) == 2
    assert all(document.text for document in documents)


def test_reads_gzipped_text(tmp_path: Path) -> None:
    payload = "compressed prose, still prose\n" * 5
    with gzip.open(tmp_path / "squashed.txt.gz", "wt", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)

    documents, stats = ingest(tmp_path)

    assert [document.text for document in documents] == [payload]
    assert stats.files_read == 1


@pytest.mark.parametrize(
    ("suffix", "opener"),
    [(".gz", gzip.open), (".bz2", bz2.open), (".xz", lzma.open)],
)
def test_every_codec_reads_to_the_same_text(tmp_path: Path, suffix: str, opener: object) -> None:
    """gzip, bz2 and xz are three spellings of the same corpus, not three corpora."""
    payload = "the same prose regardless of how it was squashed\n" * 4
    with opener(tmp_path / f"doc.txt{suffix}", "wt", encoding="utf-8", newline="\n") as handle:  # type: ignore[operator]
        handle.write(payload)

    documents, stats = ingest(tmp_path)

    assert [document.text for document in documents] == [payload]
    assert stats.files_read == 1


def test_the_codec_is_recorded_on_the_source(tmp_path: Path) -> None:
    """The manifest and the file-mix line both read this, so the name has to be right."""
    with bz2.open(tmp_path / "doc.txt.bz2", "wt", encoding="utf-8", newline="\n") as handle:
        handle.write("squashed with bzip2\n")

    (source,) = Ingestor().discover(tmp_path)

    assert source.compression == "bz2"
    assert source.compressed is True


@pytest.mark.parametrize("suffix", [".markdown", ".rst", ".org", ".log"])
def test_text_aliases_are_discovered_as_text(tmp_path: Path, suffix: str) -> None:
    write(tmp_path / f"doc{suffix}", "prose under a different extension\n")

    (source,) = Ingestor().discover(tmp_path)

    assert source.kind == "text"


def test_reads_jsonl_with_an_explicit_field(tmp_path: Path) -> None:
    jsonl(
        tmp_path / "docs.jsonl",
        [f"document number {n}" for n in range(3)],
        field="body",
    )

    documents, _ = ingest(tmp_path, jsonl_field="body")

    assert [document.text for document in documents] == [
        "document number 0",
        "document number 1",
        "document number 2",
    ]


def test_autodetects_the_jsonl_text_field_and_records_the_choice(tmp_path: Path) -> None:
    jsonl(tmp_path / "docs.jsonl", ["auto-detected from content"], field="content")

    documents, stats = ingest(tmp_path)

    assert [document.text for document in documents] == ["auto-detected from content"]
    assert stats.resolved_jsonl_fields == {"docs.jsonl": "content"}


def test_jsonl_ordinal_counts_records(tmp_path: Path) -> None:
    jsonl(tmp_path / "docs.jsonl", [f"record {n} is long enough" for n in range(4)])

    documents, _ = ingest(tmp_path)

    assert [document.ordinal for document in documents] == [0, 1, 2, 3]
    assert "docs.jsonl:2" in documents[2].location


def test_honours_a_declared_encoding(tmp_path: Path) -> None:
    (tmp_path / "latin.txt").write_bytes("caf\xe9 na\xefve\n".encode("latin-1"))

    documents, _ = ingest(tmp_path, encoding="latin-1")

    assert documents[0].text == "caf\xe9 na\xefve\n"


def test_byte_order_mark_is_honoured_over_the_default_encoding(tmp_path: Path) -> None:
    (tmp_path / "bom.txt").write_bytes(b"\xff\xfe" + "utf-16 text\n".encode("utf-16-le"))

    documents, _ = ingest(tmp_path)

    assert documents[0].text == "utf-16 text\n"


#: Every distinct codec named by :data:`trainai.data.ingest._BOMS`, taken from the table
#: rather than restated, so adding a BOM there brings it under this test automatically.
BOM_CODECS = sorted({name for _bom, name in _BOMS})

#: Text with a character above ASCII, because a wrong wide codec can survive plain ASCII
#: and fail later. Newline-free so no line-ending rule can hide a difference.
BOM_TEXT = "First line of a corpus. Second line, long enough to matter. Accents: caf\xe9."


@pytest.mark.parametrize("codec", BOM_CODECS)
def test_the_byte_order_mark_hint_names_two_things_that_both_work(
    tmp_path: Path, codec: str
) -> None:
    """The refusal offers two fixes, and both are followed here rather than described.

    ``_effective_encoding`` refuses when a file's BOM contradicts an explicit
    ``--encoding``, and the hint says to drop the flag or pass the codec it names. Both
    halves depend on facts that are easy to break from a distance: dropping the flag only
    works while the default is a codec ``_same_codec`` treats as narrow, and passing the
    named codec only works while every name in ``_BOMS`` is one Python accepts. Neither is
    visible from the line that writes the hint.

    So each half is parsed out of the message the user sees and then followed, separately.
    A table entry of ``"utf16"``, a changed default, or a hint that echoes back the
    encoding the user already passed fails here instead of shipping advice that does not
    work. Parsing both halves is not pedantry: a first version of this test read only one
    codec out of the message, and a mutation that made the second half name the *requested*
    encoding -- the value that had just failed -- passed it.
    """
    directory = tmp_path / codec
    directory.mkdir()
    # Encoded by the codec itself, so the BOM is the one it really emits.
    (directory / "corpus.txt").write_bytes(BOM_TEXT.encode(codec))

    with pytest.raises(DatasetDecodeError) as caught:
        ingest(directory, encoding="latin-1")

    hint = caught.value.hint or ""
    drop_half = re.search(r"Drop --encoding to let TrainAI use ([\w-]+)", hint)
    flag_half = re.search(r"pass --encoding ([\w-]+)", hint)
    assert drop_half and flag_half, f"the hint is missing one of its two fixes: {hint}"
    assert drop_half.group(1) == codec, f"the hint says the default becomes {drop_half.group(1)}"
    assert flag_half.group(1) == codec, (
        f"the hint says to pass --encoding {flag_half.group(1)}, which is not the codec "
        f"the mark identifies; passing back what the user gave is the advice that failed"
    )

    dropped, _ = ingest(directory)
    passed, _ = ingest(directory, encoding=flag_half.group(1))

    assert "".join(document.text for document in dropped) == BOM_TEXT, (
        "dropping --encoding is half the advice, and it has to decode the file exactly"
    )
    assert "".join(document.text for document in passed) == BOM_TEXT, (
        f"--encoding {flag_half.group(1)} is the other half, and it has to decode exactly"
    )
    assert "\ufeff" not in "".join(document.text for document in dropped), (
        "the byte-order mark must not survive into the training text"
    )


# --------------------------------------------------------------------------- #
# Failure reporting
# --------------------------------------------------------------------------- #
def test_invalid_utf8_names_the_file_and_the_byte_offset(tmp_path: Path) -> None:
    (tmp_path / "broken.txt").write_bytes(b"a valid prefix of text, then: \xff more")

    with pytest.raises(DatasetDecodeError) as caught:
        ingest(tmp_path)

    message = str(caught.value)
    assert "broken.txt" in message
    assert "30" in message, message
    assert "--encoding" in (caught.value.hint or "")


def test_on_error_skip_records_the_file_and_keeps_reading(tmp_path: Path) -> None:
    write(tmp_path / "aaa_good.txt", "perfectly readable\n" * 5)
    (tmp_path / "zzz_broken.txt").write_bytes(b"prefix \xff suffix")

    documents, stats = ingest(tmp_path, on_error="skip")

    assert len(documents) == 1
    assert stats.files_read == 1
    assert stats.files_skipped == 1
    assert stats.skipped_files[0]["path"] == "zzz_broken.txt"


def test_malformed_jsonl_names_the_line(tmp_path: Path) -> None:
    write(
        tmp_path / "docs.jsonl",
        json.dumps({"text": "a good first record"}) + "\nthis line is not json\n",
    )

    with pytest.raises(DatasetFormatError) as caught:
        ingest(tmp_path)

    assert "line 2" in str(caught.value)
    assert "--on-error skip" in (caught.value.hint or "")


def test_on_error_skip_counts_malformed_records(tmp_path: Path) -> None:
    write(
        tmp_path / "docs.jsonl",
        json.dumps({"text": "a good first record"}) + "\nnot json\n",
    )

    documents, stats = ingest(tmp_path, on_error="skip")

    assert len(documents) == 1
    assert stats.records_skipped_malformed == 1


def test_missing_jsonl_field_names_the_fields_that_are_present(tmp_path: Path) -> None:
    """The fix is to change --jsonl-field, so the error has to say to what."""
    jsonl(tmp_path / "docs.jsonl", ["no text field here"], field="headline")

    with pytest.raises(DatasetFormatError) as caught:
        ingest(tmp_path, jsonl_field="text")

    assert "headline" in (caught.value.hint or "")
    assert caught.value.details["available_fields"] == ["headline"]


def test_no_recognised_field_lists_what_was_tried(tmp_path: Path) -> None:
    jsonl(tmp_path / "docs.jsonl", ["nothing recognisable here"], field="headline")

    with pytest.raises(DatasetFormatError) as caught:
        ingest(tmp_path)

    assert "--jsonl-field" in (caught.value.hint or "")
    assert caught.value.details["available_fields"] == ["headline"]
    assert "text" in caught.value.details["tried"]


# --------------------------------------------------------------------------- #
# Whole-file JSON
#
# The one format here that cannot stream, and the one whose top-level shape is
# genuinely ambiguous. Every case below is either "this reading is the only
# reading" or "there are two readings, so refuse" -- never a guess.
# --------------------------------------------------------------------------- #
def json_file(path: Path, payload: object) -> Path:
    return write(path, json.dumps(payload))


def test_reads_a_json_array_of_records(tmp_path: Path) -> None:
    json_file(tmp_path / "docs.json", [{"text": "first"}, {"text": "second"}])

    documents, stats = ingest(tmp_path)

    assert [document.text for document in documents] == ["first", "second"]
    assert stats.files_read == 1
    assert stats.resolved_jsonl_fields == {"docs.json": "text"}


def test_reads_a_json_array_of_bare_strings(tmp_path: Path) -> None:
    json_file(tmp_path / "docs.json", ["first", "second"])

    documents, _ = ingest(tmp_path)

    assert [document.text for document in documents] == ["first", "second"]


def test_a_json_field_can_be_named_explicitly(tmp_path: Path) -> None:
    json_file(tmp_path / "docs.json", [{"headline": "first"}, {"headline": "second"}])

    documents, _ = ingest(tmp_path, jsonl_field="headline")

    assert [document.text for document in documents] == ["first", "second"]


def test_a_json_object_wrapping_one_array_is_unwrapped(tmp_path: Path) -> None:
    """``{"data": [...]}`` is how most API dumps arrive, and it is unambiguous."""
    json_file(tmp_path / "docs.json", {"data": [{"text": "first"}, {"text": "second"}], "count": 2})

    documents, _ = ingest(tmp_path)

    assert [document.text for document in documents] == ["first", "second"]


def test_a_json_object_with_several_arrays_lists_them_rather_than_guessing(tmp_path: Path) -> None:
    json_file(tmp_path / "docs.json", {"train": [{"text": "a"}], "test": [{"text": "b"}]})

    with pytest.raises(DatasetFormatError) as caught:
        ingest(tmp_path)

    assert caught.value.details["arrays"] == ["test", "train"]
    assert "test" in (caught.value.hint or "") and "train" in (caught.value.hint or "")


def test_a_json_record_holding_a_list_is_refused_rather_than_unwrapped(tmp_path: Path) -> None:
    """The dangerous shape: unwrapping trains on the tags and discards the text."""
    json_file(tmp_path / "docs.json", {"text": "the real document", "tags": ["news", "politics"]})

    with pytest.raises(DatasetFormatError) as caught:
        ingest(tmp_path)

    assert caught.value.details["text_field"] == "text"
    assert caught.value.details["arrays"] == ["tags"]
    # The refusal has to be reported as ambiguity, not as a successful read of
    # two one-word documents, which is what the naive rule would have produced.
    assert "ambiguous" in str(caught.value)


def test_a_json_object_with_no_array_says_to_wrap_it(tmp_path: Path) -> None:
    json_file(tmp_path / "docs.json", {"title": "a", "n": 1})

    with pytest.raises(DatasetFormatError) as caught:
        ingest(tmp_path)

    assert caught.value.details["keys"] == ["n", "title"]
    assert "array" in (caught.value.hint or "")


def test_a_json_scalar_is_refused_naming_its_type(tmp_path: Path) -> None:
    json_file(tmp_path / "docs.json", "just a string")

    with pytest.raises(DatasetFormatError) as caught:
        ingest(tmp_path)

    assert caught.value.details["type"] == "str"


def test_malformed_json_reports_the_line_and_keeps_the_jsonl_advice(tmp_path: Path) -> None:
    write(tmp_path / "docs.json", '[{"text": "a"},')

    with pytest.raises(DatasetFormatError) as caught:
        ingest(tmp_path)

    assert caught.value.details["line"] == 1
    assert ".jsonl" in (caught.value.hint or "")


def test_a_bad_json_element_is_reported_as_an_element_not_a_line(tmp_path: Path) -> None:
    """A .json array has no lines, so "line 3" would send the user hunting."""
    json_file(tmp_path / "docs.json", [{"text": "good"}, {"nope": 1}])

    with pytest.raises(DatasetFormatError) as caught:
        ingest(tmp_path)

    assert "element 1" in str(caught.value)
    assert caught.value.details["element"] == 1
    assert caught.value.details["available_fields"] == ["nope"]


def test_on_error_skip_drops_a_bad_json_element_and_counts_it(tmp_path: Path) -> None:
    json_file(tmp_path / "docs.json", [{"text": "good"}, {"nope": 1}, {"text": "also good"}])

    documents, stats = ingest(tmp_path, on_error="skip")

    assert [document.text for document in documents] == ["good", "also good"]
    assert stats.records_skipped_malformed == 1


def test_an_oversized_json_is_refused_with_the_streaming_alternative(tmp_path: Path) -> None:
    """The honest constraint: no record separator means no streaming."""
    path = json_file(tmp_path / "docs.json", [{"text": "a"}])
    source = Ingestor().discover(path)[0]
    huge = replace(source, size_bytes=_JSON_MAX_BYTES + 1)

    with pytest.raises(DatasetFormatError) as caught:
        Ingestor()._read_json_source(huge)

    assert caught.value.details["limit_bytes"] == _JSON_MAX_BYTES
    assert caught.value.details["measured"] == "declared"
    assert ".jsonl" in (caught.value.hint or "")


def test_a_compressed_json_is_bounded_by_what_it_expands_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The size on disk bounds nothing: 269:1 on repetitive JSON is easy to hit.

    Checking only the declared size would let a file well under the ceiling
    decompress to tens of gigabytes, which is the failure the ceiling exists to
    prevent in the first place.
    """
    payload = json.dumps([{"text": "the same sentence over and over. " * 8}] * 64)
    path = tmp_path / "expands.json.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    ceiling = 4096
    # The premise of the test: the declared-size check cannot be what fires.
    assert path.stat().st_size < ceiling < len(payload)
    monkeypatch.setattr(ingest_module, "_JSON_MAX_BYTES", ceiling)

    with pytest.raises(DatasetFormatError) as caught:
        ingest(tmp_path)

    assert caught.value.details["measured"] == "decompressed"
    assert caught.value.details["bytes_read"] > ceiling
    assert "expands to more than" in str(caught.value)


def test_a_compressed_json_reads_the_same(tmp_path: Path) -> None:
    payload = json.dumps([{"text": "first"}, {"text": "second"}])
    with gzip.open(tmp_path / "docs.json.gz", "wt", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)

    documents, _ = ingest(tmp_path)

    assert [document.text for document in documents] == ["first", "second"]


def test_utf16_json_is_read_rather_than_refused(tmp_path: Path) -> None:
    """Unlike JSONL and CSV, nothing here is split on the newline byte."""
    (tmp_path / "docs.json").write_bytes(
        json.dumps([{"text": "first"}, {"text": "second"}]).encode("utf-16")
    )

    documents, _ = ingest(tmp_path)

    assert [document.text for document in documents] == ["first", "second"]


# --------------------------------------------------------------------------- #
# Document bounds
# --------------------------------------------------------------------------- #
def test_min_doc_chars_drops_short_documents_and_counts_them(tmp_path: Path) -> None:
    jsonl(tmp_path / "docs.jsonl", ["tiny", "long enough to keep", "no"])

    documents, stats = ingest(tmp_path, min_doc_chars=10)

    assert [document.text for document in documents] == ["long enough to keep"]
    assert stats.documents_skipped_short == 2


def test_blank_documents_are_always_dropped(tmp_path: Path) -> None:
    write(tmp_path / "blank.txt", "   \n\n\t\n")

    documents, stats = ingest(tmp_path)

    assert documents == []
    assert stats.documents_skipped_short == 1


def test_long_file_is_split_without_losing_a_character(tmp_path: Path) -> None:
    paragraph = "Sentences that fill a paragraph nicely enough. " * 4
    payload = "\n\n".join(paragraph for _ in range(40))
    write(tmp_path / "long.txt", payload)

    documents, stats = ingest(tmp_path, max_doc_chars=1024)

    assert len(documents) > 1
    assert stats.long_files_split == 1
    assert "".join(document.text for document in documents) == payload
    assert all(len(document.text) <= 1024 for document in documents)


def test_split_pieces_end_at_a_boundary_rather_than_mid_word(tmp_path: Path) -> None:
    paragraph = "A paragraph of a few words. " * 10
    payload = "\n\n".join(paragraph for _ in range(20))
    write(tmp_path / "long.txt", payload)

    documents, _ = ingest(tmp_path, max_doc_chars=1024)

    for document in documents[:-1]:
        assert document.text.endswith(("\n", " ")), repr(document.text[-20:])


def test_byte_offsets_advance_through_the_file(tmp_path: Path) -> None:
    paragraph = "Offsets must be usable in an error message. " * 8
    payload = "\n\n".join(paragraph for _ in range(20))
    write(tmp_path / "long.txt", payload)

    documents, _ = ingest(tmp_path, max_doc_chars=512)

    offsets = [document.byte_offset for document in documents]
    assert offsets[0] == 0
    assert offsets == sorted(offsets)
    assert len(set(offsets)) == len(offsets)


def test_stats_account_for_every_record_offered(tmp_path: Path) -> None:
    texts = ["ok, keep this one", "x", "also keep this one", "y"]
    jsonl(tmp_path / "docs.jsonl", texts)

    documents, stats = ingest(tmp_path, min_doc_chars=5)

    assert len(documents) + stats.documents_skipped_short == len(texts)
    assert stats.documents_emitted == len(documents)


def test_options_round_trip_through_to_dict() -> None:
    """The manifest records these verbatim, so the mapping must be complete."""
    options = IngestOptions(
        encoding="latin-1",
        jsonl_field="body",
        csv_text_column="review_text",
        max_doc_chars=4096,
        min_doc_chars=8,
        on_error="skip",
    )

    assert IngestOptions(**options.to_dict()) == options


# --------------------------------------------------------------------------- #
# CSV and TSV
#
# One column, named, and nothing joined. The tests below pin that decision from
# both sides: the column is found when it can be found unambiguously, and the
# error names the columns when it cannot -- because a reader that guesses picks
# one column out of twelve and produces a corpus that looks plausible.
# --------------------------------------------------------------------------- #
def csv_file(path: Path, header: str, rows: list[str], delimiter: str = ",") -> Path:
    return write(path, delimiter.join(header.split(",")) + "\n" + "\n".join(rows) + "\n")


REVIEW_ROWS = [
    "0,5,The room was quiet and the harbour view made up for the stairs.",
    "1,3,Breakfast was cold but the staff could not have been kinder.",
    "2,4,Ten minutes from the station and quieter than I expected.",
]


def test_reads_a_csv_column_named_explicitly(tmp_path: Path) -> None:
    csv_file(tmp_path / "hotels.csv", "id,stars,note", REVIEW_ROWS)

    documents, stats = ingest(tmp_path, csv_text_column="note")

    assert [document.text for document in documents] == [
        row.split(",", 2)[2] for row in REVIEW_ROWS
    ]
    assert stats.files_read == 1
    assert stats.documents_emitted == 3


def test_autodetects_a_conventional_csv_column_and_records_the_choice(tmp_path: Path) -> None:
    csv_file(tmp_path / "hotels.csv", "id,stars,review", REVIEW_ROWS)

    documents, stats = ingest(tmp_path)

    assert len(documents) == 3
    assert stats.resolved_csv_columns == {"hotels.csv": "review"}


def test_a_csv_row_becomes_a_document_with_its_row_number(tmp_path: Path) -> None:
    csv_file(tmp_path / "hotels.csv", "id,stars,review", REVIEW_ROWS)

    documents, _ = ingest(tmp_path)

    assert [document.ordinal for document in documents] == [0, 1, 2]
    assert "hotels.csv:1" in documents[1].location


def test_a_single_column_csv_needs_no_name(tmp_path: Path) -> None:
    """Nothing to choose between, so there is nothing to ask about."""
    write(tmp_path / "lines.csv", "sentence\nfirst line of text\nsecond line of text\n")

    documents, _ = ingest(tmp_path)

    assert [document.text for document in documents] == [
        "first line of text",
        "second line of text",
    ]


def test_the_named_column_is_matched_through_case_and_padding(tmp_path: Path) -> None:
    """CSV headers are written by people, who put spaces and capitals in them."""
    write(tmp_path / "hotels.csv", "Id, Review Text \nx,some genuine prose here\n")

    documents, _ = ingest(tmp_path, csv_text_column="review text")

    assert [document.text for document in documents] == ["some genuine prose here"]


def test_a_tsv_is_split_on_tabs(tmp_path: Path) -> None:
    write(tmp_path / "hotels.tsv", "id\treview\n1\tprose, with a comma in it\n")

    documents, _ = ingest(tmp_path)

    assert [document.text for document in documents] == ["prose, with a comma in it"]


def test_a_gzipped_csv_reads_the_same(tmp_path: Path) -> None:
    body = 'id,review\n1,"compressed prose, still prose"\n'
    with gzip.open(tmp_path / "hotels.csv.gz", "wt", encoding="utf-8", newline="\n") as handle:
        handle.write(body)

    documents, stats = ingest(tmp_path)

    assert [document.text for document in documents] == ["compressed prose, still prose"]
    assert stats.csv_rows_with_extra_fields == 0


def test_a_row_with_more_fields_than_the_header_is_counted(tmp_path: Path) -> None:
    """Because the alternative is handing back a fragment and saying nothing.

    An unquoted delimiter inside a field is the ordinary way a CSV row grows a
    column, and the text extracted from that row is then only the part before it.
    Nothing is dropped, so no skip counter moves and no error is raised -- which
    is exactly why this needs a counter of its own, and why ``validate_corpus``
    turns it into a warning instead of leaving it in the manifest to be found
    afterwards.
    """
    write(tmp_path / "hotels.csv", "id,review\n1,fine prose\n2,split, in two\n3,also fine\n")

    documents, stats = ingest(tmp_path)

    assert [document.text for document in documents] == ["fine prose", "split", "also fine"]
    assert stats.csv_rows_with_extra_fields == 1
    assert stats.records_skipped_malformed == 0
    assert stats.to_dict()["csv_rows_with_extra_fields"] == 1


def test_a_trailing_delimiter_on_every_row_is_counted_but_harmless(tmp_path: Path) -> None:
    """The reason the extra field is counted and warned about, not refused."""
    write(tmp_path / "hotels.csv", "id,review\n1,first,\n2,second,\n")

    documents, stats = ingest(tmp_path)

    assert [document.text for document in documents] == ["first", "second"]
    assert stats.csv_rows_with_extra_fields == 2


def test_a_quoted_field_spanning_lines_is_one_document(tmp_path: Path) -> None:
    write(tmp_path / "hotels.csv", 'id,review\n1,"first line\nsecond line"\n2,another\n')

    documents, _ = ingest(tmp_path)

    assert [document.text for document in documents] == ["first line\nsecond line", "another"]


def test_byte_offsets_point_at_the_row_that_starts_there(tmp_path: Path) -> None:
    """A quoted field pulls extra lines, so the offset has to survive that."""
    header = "id,review\n"
    first = '1,"first line\nsecond line"\n'
    write(tmp_path / "hotels.csv", header + first + "2,another\n")

    documents, _ = ingest(tmp_path)

    assert documents[0].byte_offset == len(header)
    assert documents[1].byte_offset == len(header) + len(first)


def test_no_recognised_column_lists_the_columns_that_are_there(tmp_path: Path) -> None:
    """The refusal at the centre of this: twelve columns of numbers, no guessing."""
    header = "Formatted Date,Summary,Temperature (C),Humidity,Pressure (millibars)"
    write(tmp_path / "weather.csv", header + "\n2006-04-01,Partly cloudy,9.5,0.89,1015.1\n")

    with pytest.raises(DatasetFormatError) as caught:
        ingest(tmp_path)

    hint = caught.value.hint or ""
    assert "--csv-text-column" in hint
    assert "'Summary'" in hint, hint
    assert "'Pressure (millibars)'" in hint, hint
    assert caught.value.details["available_columns"] == header.split(",")


def test_summary_is_not_auto_selected(tmp_path: Path) -> None:
    """Named for the bug it prevents.

    A column called "Summary" holds words, so a generous candidate list picks it
    -- and on the weather CSV that means training on 96,000 repetitions of
    "Partly cloudy" while eleven other columns go unread. Guessing wrong here is
    worse than asking, so the list stays short.
    """
    write(tmp_path / "weather.csv", "Date,Summary,Temp\n2006-04-01,Partly cloudy,9.5\n")

    with pytest.raises(DatasetFormatError):
        ingest(tmp_path)


def test_naming_a_column_that_is_not_there_names_the_ones_that_are(tmp_path: Path) -> None:
    csv_file(tmp_path / "hotels.csv", "id,stars,note", REVIEW_ROWS)

    with pytest.raises(DatasetFormatError) as caught:
        ingest(tmp_path, csv_text_column="review_text")

    assert "'note'" in (caught.value.hint or "")
    assert caught.value.details["requested_column"] == "review_text"


def test_a_semicolon_file_called_csv_says_so_specifically(tmp_path: Path) -> None:
    """One absurd column is diagnosable; a sniffer's wrong guess is not.

    Taking the delimiter from the extension is what makes this error possible:
    the whole header arrives as a single field, which is a fact worth reporting
    rather than a dataset worth building.
    """
    write(tmp_path / "hotels.csv", "id;stars;review\n1;5;a genuinely fine room\n")

    with pytest.raises(DatasetFormatError) as caught:
        ingest(tmp_path)

    assert "semicolon" in str(caught.value).lower()
    assert "Save As" in (caught.value.hint or "")


def test_utf16_csv_is_refused_with_an_instruction(tmp_path: Path) -> None:
    """Rows are found by splitting on the newline byte, which UTF-16 makes wrong."""
    body = "id,review\n1,text that decodes fine but splits wrong\n"
    (tmp_path / "hotels.csv").write_bytes(b"\xff\xfe" + body.encode("utf-16-le"))

    with pytest.raises(DatasetFormatError) as caught:
        ingest(tmp_path)

    assert "UTF-8" in (caught.value.hint or "")
    assert "utf-16" in str(caught.value).lower()


def test_a_row_too_short_for_the_column_names_the_line(tmp_path: Path) -> None:
    write(tmp_path / "hotels.csv", "id,stars,review\n1,5,a fine room\n2,4\n")

    with pytest.raises(DatasetFormatError) as caught:
        ingest(tmp_path)

    assert "line 3" in str(caught.value)
    assert "--on-error skip" in (caught.value.hint or "")


def test_on_error_skip_counts_a_short_row_and_keeps_reading(tmp_path: Path) -> None:
    write(tmp_path / "hotels.csv", "id,stars,review\n1,5,a fine room\n2,4\n3,5,also fine\n")

    documents, stats = ingest(tmp_path, on_error="skip")

    assert [document.text for document in documents] == ["a fine room", "also fine"]
    assert stats.records_skipped_malformed == 1


def test_blank_lines_between_rows_are_not_records(tmp_path: Path) -> None:
    write(tmp_path / "hotels.csv", "id,review\n1,first row of prose\n\n2,second row\n")

    documents, stats = ingest(tmp_path)

    assert len(documents) == 2
    assert stats.records_skipped_malformed == 0


def test_a_header_with_no_rows_is_not_an_error(tmp_path: Path) -> None:
    """Zero documents is a number the report can state; an exception is not."""
    write(tmp_path / "hotels.csv", "id,review\n")

    documents, stats = ingest(tmp_path)

    assert documents == []
    assert stats.files_read == 1


def test_a_csv_column_holding_numbers_still_reads(tmp_path: Path) -> None:
    """Ingest does not judge. The corpus measurement is what catches this.

    Refusing a numeric column here would put the same decision in two places,
    and the one in ``validate`` is better informed: it sees the whole corpus
    rather than one cell, and it reports the measurement behind its verdict.
    """
    rows = [f"{n},{n * 1.5}" for n in range(30)]
    write(tmp_path / "readings.csv", "id,temperature\n" + "\n".join(rows) + "\n")

    documents, _ = ingest(tmp_path, csv_text_column="temperature")

    assert len(documents) == 30


# --------------------------------------------------------------------------- #
# Archives
#
# The riskiest part of this module: a member is not a path on disk, so both
# discovery and opening take a second route, and an archive can be built to be
# hostile. What is pinned here is that the route produces the *same* documents as
# the loose files would, in an order that does not depend on how the archive was
# written, and that everything skipped is counted.
# --------------------------------------------------------------------------- #
def zip_of(path: Path, members: list[tuple[str, bytes | str]]) -> Path:
    """Write a zip with members in exactly the order given -- the order matters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members:
            archive.writestr(name, data)
    return path


def tar_of(path: Path, members: list[tuple[str, bytes | str]], mode: str = "w") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode) as archive:  # type: ignore[call-overload]
        for name, data in members:
            raw = data.encode("utf-8") if isinstance(data, str) else data
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    return path


def test_a_zip_of_text_files_reads_as_those_files(tmp_path: Path) -> None:
    zip_of(tmp_path / "papers.zip", [("b/second.txt", "second\n"), ("a/first.txt", "first\n")])

    documents, stats = ingest(tmp_path)

    assert [document.text for document in documents] == ["first\n", "second\n"]
    assert stats.archives_opened == 1
    assert stats.files_read == 2


def test_a_member_is_named_by_archive_and_path(tmp_path: Path) -> None:
    """The report has to point at something the user can find again."""
    zip_of(tmp_path / "papers.zip", [("a/first.txt", "first\n")])

    sources = Ingestor().discover(tmp_path)

    assert [source.relative for source in sources] == ["papers.zip::a/first.txt"]
    assert sources[0].member == "a/first.txt"
    assert sources[0].archive == "zip"
    assert sources[0].path == tmp_path / "papers.zip"


def test_member_order_does_not_depend_on_how_the_archive_was_written(tmp_path: Path) -> None:
    """The determinism guarantee, at the one place archives could break it.

    ``namelist()`` gives insertion order, which is a property of the tool that
    built the archive. Two archives of the same files must give the same document
    order, or every checksum downstream depends on who zipped it.
    """
    members: list[tuple[str, bytes | str]] = [
        ("b/second.txt", "second\n"),
        ("a/first.txt", "first\n"),
        ("c/third.txt", "third\n"),
    ]
    forward = zip_of(tmp_path / "forward" / "corpus.zip", members)
    reverse = zip_of(tmp_path / "reverse" / "corpus.zip", list(reversed(members)))

    with zipfile.ZipFile(reverse) as archive:  # the orders really do differ on disk
        assert archive.namelist() != [name for name, _ in members]

    assert [d.text for d in ingest(forward)[0]] == [d.text for d in ingest(reverse)[0]]


def test_a_tar_gives_the_same_documents_as_a_zip_of_the_same_files(tmp_path: Path) -> None:
    members: list[tuple[str, bytes | str]] = [("b/second.txt", "second\n"), ("a/f.txt", "first\n")]
    as_zip = zip_of(tmp_path / "z" / "corpus.zip", members)
    as_tar = tar_of(tmp_path / "t" / "corpus.tar", members)
    as_tgz = tar_of(tmp_path / "g" / "corpus.tar.gz", members, mode="w:gz")

    texts = [d.text for d in ingest(as_zip)[0]]

    assert texts == [d.text for d in ingest(as_tar)[0]]
    assert texts == [d.text for d in ingest(as_tgz)[0]]


@pytest.mark.parametrize(
    ("suffix", "mode"),
    [
        (".tar", "w"),
        (".tgz", "w:gz"),
        (".tar.gz", "w:gz"),
        (".tar.bz2", "w:bz2"),
        (".tbz2", "w:bz2"),
        (".txz", "w:xz"),
    ],
)
def test_every_tar_spelling_is_recognised(tmp_path: Path, suffix: str, mode: str) -> None:
    path = tar_of(tmp_path / f"corpus{suffix}", [("doc.txt", "in a tar\n")], mode=mode)

    documents, stats = ingest(path)

    assert [document.text for document in documents] == ["in a tar\n"]
    assert stats.archives_opened == 1


def test_jsonl_and_csv_members_read_line_by_line(tmp_path: Path) -> None:
    """The readers iterate the stream by line, which a member stream must support."""
    records = "\n".join(json.dumps({"text": f"record {n}"}) for n in range(3))
    zip_of(
        tmp_path / "mixed.zip",
        [("records.jsonl", records + "\n"), ("rows.csv", "id,text\n1,row one\n2,row two\n")],
    )

    documents, stats = ingest(tmp_path)

    assert [document.text for document in documents] == [
        "record 0",
        "record 1",
        "record 2",
        "row one",
        "row two",
    ]
    assert stats.files_read == 2


def test_a_member_with_its_own_codec_is_decompressed(tmp_path: Path) -> None:
    """Two layers: the archive's compression, and a .gz on the member's own name."""
    zip_of(tmp_path / "papers.zip", [("doc.txt.gz", gzip.compress(b"compressed twice\n"))])

    documents, stats = ingest(tmp_path)

    assert [document.text for document in documents] == ["compressed twice\n"]
    assert stats.files_read == 1


def test_a_utf16_member_is_decoded_from_its_byte_order_mark(tmp_path: Path) -> None:
    """Proves the second open -- the one that reads the BOM -- works on a member."""
    zip_of(tmp_path / "papers.zip", [("wide.txt", "wide text\n".encode("utf-16"))])

    documents, _ = ingest(tmp_path)

    assert [document.text for document in documents] == ["wide text\n"]


def test_hidden_and_macos_metadata_members_are_skipped(tmp_path: Path) -> None:
    """A zip made on a macOS desktop is full of this, and none of it is text."""
    zip_of(
        tmp_path / "papers.zip",
        [
            ("__MACOSX/._doc.txt", b"\x00\x05\x16\x07resource fork"),
            (".hidden/secret.txt", "not this\n"),
            ("doc.txt", "this one\n"),
        ],
    )

    documents, stats = ingest(tmp_path)

    assert [document.text for document in documents] == ["this one\n"]
    # Skipped by the same rule as a dotfile on disk, so not counted as ignored.
    assert stats.archive_members_ignored == 0


def test_a_directory_entry_in_an_archive_is_not_a_document(tmp_path: Path) -> None:
    zip_of(tmp_path / "papers.zip", [("folder/", b""), ("folder/doc.txt", "text\n")])

    documents, stats = ingest(tmp_path)

    assert [document.text for document in documents] == ["text\n"]
    assert stats.archive_members_ignored == 0


def test_a_symlink_member_is_not_read_as_its_target_path(tmp_path: Path) -> None:
    """A zip symlink stores the *target path* as its bytes.

    Reading it would put a file name into the corpus as though it were a document,
    which is silent nonsense rather than a failure.
    """
    path = tmp_path / "papers.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("doc.txt", "real text\n")
        link = zipfile.ZipInfo("link.txt")
        link.create_system = 3  # Unix, which is where the mode bits are meaningful
        link.external_attr = 0o120777 << 16
        archive.writestr(link, "doc.txt")

    documents, _ = ingest(tmp_path)

    assert [document.text for document in documents] == ["real text\n"]


def test_a_nested_archive_is_reported_rather_than_opened(tmp_path: Path) -> None:
    """One level only. Nesting is how the classic 4.5 PB bomb reaches its size."""
    inner = zip_of(tmp_path / "inner" / "inner.zip", [("deep.txt", "deep\n")])
    zip_of(tmp_path / "outer.zip", [("doc.txt", "shallow\n"), ("inner.zip", inner.read_bytes())])

    documents, stats = ingest(tmp_path / "outer.zip")

    assert [document.text for document in documents] == ["shallow\n"]
    assert stats.nested_archives_ignored == 1
    assert stats.ignored_nested_archives == ["outer.zip::inner.zip"]
    assert stats.archive_members_ignored == 0


def test_an_unreadable_member_is_counted_and_named(tmp_path: Path) -> None:
    """An archive is opaque, so a member dropped without a name is invisible."""
    zip_of(
        tmp_path / "papers.zip",
        [("doc.txt", "text\n"), ("scan.pdf", b"%PDF-1.4"), ("sheet.xlsx", b"PK junk")],
    )

    documents, stats = ingest(tmp_path)

    assert [document.text for document in documents] == ["text\n"]
    assert stats.archive_members_ignored == 2
    assert stats.ignored_archive_members == ["papers.zip::scan.pdf", "papers.zip::sheet.xlsx"]


def test_an_archive_holding_nothing_readable_lists_what_it_holds(tmp_path: Path) -> None:
    path = zip_of(tmp_path / "papers.zip", [("scan.pdf", b"%PDF-1.4"), ("photo.jpg", b"\xff\xd8")])

    with pytest.raises(DatasetNotFoundError) as caught:
        Ingestor().discover(path)

    assert "nothing TrainAI can read" in str(caught.value)
    assert "scan.pdf" in (caught.value.hint or "")
    assert caught.value.details["members"] == ["photo.jpg", "scan.pdf"]


def test_an_unreadable_archive_in_a_walk_is_not_an_error(tmp_path: Path) -> None:
    """Naming a file is an instruction; finding one in a tree is not."""
    write(tmp_path / "corpus.txt", "real text\n")
    zip_of(tmp_path / "photos.zip", [("photo.jpg", b"\xff\xd8")])

    documents, stats = ingest(tmp_path)

    assert [document.text for document in documents] == ["real text\n"]
    assert stats.archive_members_ignored == 1


def test_a_corrupt_archive_says_so_instead_of_raising_zipfile_errors(tmp_path: Path) -> None:
    path = tmp_path / "papers.zip"
    path.write_bytes(b"PK\x03\x04 and then nothing that parses")

    with pytest.raises(DatasetFormatError) as caught:
        Ingestor().discover(path)

    assert "could not be read as a zip archive" in str(caught.value)
    assert caught.value.hint


def test_too_many_members_is_refused_before_anything_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound is on discovery cost: every member becomes an object first."""
    zip_of(tmp_path / "many.zip", [(f"doc_{n}.txt", "text\n") for n in range(12)])
    monkeypatch.setattr(ingest_module, "_ARCHIVE_MAX_MEMBERS", 10)

    with pytest.raises(DatasetFormatError) as caught:
        Ingestor().discover(tmp_path / "many.zip")

    assert caught.value.details == {"path": "many.zip", "members": 12, "limit": 10}
    assert "Unpack it" in (caught.value.hint or "")


def test_a_huge_declared_expansion_is_refused_from_the_table_of_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refused without decompressing anything -- the size is declared, not measured."""
    zip_of(tmp_path / "big.zip", [("doc.txt", "x" * 4096)])
    monkeypatch.setattr(ingest_module, "_ARCHIVE_MAX_DECLARED_BYTES", 1024)

    with pytest.raises(DatasetFormatError) as caught:
        Ingestor().discover(tmp_path / "big.zip")

    assert caught.value.details["declared_bytes"] == 4096
    assert "compression bomb" in (caught.value.hint or "")


def test_a_member_size_is_what_it_unpacks_to(tmp_path: Path) -> None:
    """The number the reader will see, not the member's share of the archive."""
    text = "highly compressible. " * 500
    path = zip_of(tmp_path / "papers.zip", [("doc.txt", text)])

    sources = Ingestor().discover(path)

    assert sources[0].size_bytes == len(text.encode("utf-8"))
    assert sources[0].size_bytes > path.stat().st_size


def test_the_archive_is_closed_when_the_document_stream_ends(tmp_path: Path) -> None:
    """On Windows an open handle blocks deleting the file, so this is load-bearing."""
    path = zip_of(tmp_path / "papers.zip", [("a.txt", "one\n"), ("b.txt", "two\n")])
    ingestor = Ingestor()

    list(ingestor.documents(ingestor.discover(path)))

    path.unlink()  # PermissionError on Windows if a handle is still open


def test_abandoning_the_document_stream_still_closes_the_archive(tmp_path: Path) -> None:
    """A caller that stops early -- sampling does -- must not leak a handle."""
    path = zip_of(tmp_path / "papers.zip", [("a.txt", "one\n"), ("b.txt", "two\n")])
    ingestor = Ingestor()

    stream = ingestor.documents(ingestor.discover(path))
    next(stream)
    stream.close()

    path.unlink()


def test_a_plain_directory_gains_no_archive_counters(tmp_path: Path) -> None:
    """Negative control: a guard accidentally inverted fails here, not silently."""
    write(tmp_path / "corpus.txt", "real text\n")

    _, stats = ingest(tmp_path)

    assert stats.archives_opened == 0
    assert stats.archive_members_ignored == 0
    assert stats.nested_archives_ignored == 0
    assert stats.ignored_archive_members == []
    assert stats.ignored_nested_archives == []


def test_the_supported_formats_sentence_names_archives() -> None:
    described = describe_supported_formats()

    assert ".zip" in described
    assert ".tar.gz" in described


# --------------------------------------------------------------------------- #
# Files with no extension
#
# The asymmetry is the whole design: naming a file is an instruction, finding one
# in a tree is a guess. Both halves are asserted, and so is the guard that stops
# a binary file being read as text -- UTF-8 decodes NUL happily, so nothing else
# in the pipeline would object.
# --------------------------------------------------------------------------- #
def test_a_file_with_no_extension_is_read_when_pointed_at_directly(tmp_path: Path) -> None:
    path = write(tmp_path / "corpus", "a corpus in a file called corpus\n")

    documents, stats = ingest(path)

    assert [d.text for d in documents] == ["a corpus in a file called corpus\n"]
    assert stats.assumed_text_files == ["corpus"]
    assert stats.files_read == 1


def test_a_file_with_no_extension_is_still_not_selected_by_a_walk(tmp_path: Path) -> None:
    """A stray README joining the training data is an invisible corpus change."""
    write(tmp_path / "corpus.txt", "real text\n")
    write(tmp_path / "README", "not part of the corpus\n")

    documents, stats = ingest(tmp_path)

    assert [d.source for d in documents] == ["corpus.txt"]
    assert stats.files_unrecognized == 1
    assert stats.files_without_extension == 1
    assert stats.unrecognized_files == ["README"]
    assert stats.assumed_text_files == []


def test_a_walk_names_what_it_passed_over(tmp_path: Path) -> None:
    """Counted *and named*: a silent drop is the failure this replaces."""
    write(tmp_path / "corpus.txt", "real text\n")
    write(tmp_path / "scan.pdf", "%PDF-1.4\n")
    write(tmp_path / "notes", "no extension\n")

    _, stats = ingest(tmp_path)

    assert stats.files_unrecognized == 2
    assert stats.files_without_extension == 1
    assert stats.unrecognized_files == ["notes", "scan.pdf"]


def test_the_kept_names_are_capped_but_the_count_is_not(tmp_path: Path) -> None:
    write(tmp_path / "corpus.txt", "real text\n")
    for index in range(30):
        write(tmp_path / f"file{index:02d}.bin", "x")

    _, stats = ingest(tmp_path)

    assert stats.files_unrecognized == 30
    assert len(stats.unrecognized_files) == ingest_module._MAX_NAMES_KEPT


def test_an_extensionless_file_may_still_be_compressed(tmp_path: Path) -> None:
    """A codec says how the bytes are packed, not what they are."""
    path = tmp_path / "corpus.gz"
    path.write_bytes(gzip.compress(b"compressed and nameless\n"))

    documents, stats = ingest(path)

    assert [d.text for d in documents] == ["compressed and nameless\n"]
    assert stats.assumed_text_files == ["corpus.gz"]


def test_a_file_with_an_unreadable_extension_is_still_refused(tmp_path: Path) -> None:
    """Extensionless is not the same as unrecognized. ``.pdf`` still fails loudly."""
    path = write(tmp_path / "scan.pdf", "%PDF-1.4\n")

    with pytest.raises(DatasetNotFoundError) as caught:
        ingest(path)

    assert "not a format TrainAI can read" in str(caught.value)


def test_a_dotted_name_that_is_not_an_extension_is_refused(tmp_path: Path) -> None:
    """``v1.2`` carries an extension; it is simply not one TrainAI reads."""
    path = write(tmp_path / "release.2", "text\n")

    with pytest.raises(DatasetNotFoundError):
        ingest(path)


def test_a_binary_file_with_no_extension_is_refused_rather_than_read(tmp_path: Path) -> None:
    """UTF-8 decodes NUL happily, so without this the corpus fills with U+0000."""
    path = tmp_path / "corpus"
    path.write_bytes(b"PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

    with pytest.raises(DatasetFormatError) as caught:
        ingest(path)

    assert "looks like a binary file" in str(caught.value)
    assert caught.value.details["nul_at_byte"] == 7


def test_the_binary_guard_survives_on_error_skip(tmp_path: Path) -> None:
    """It fires at discovery, so there is nothing left to fall back to."""
    path = tmp_path / "corpus"
    path.write_bytes(b"\x00" * 64)

    with pytest.raises(DatasetFormatError):
        ingest(path, on_error="skip")


def test_a_compressed_binary_file_is_refused_on_its_decompressed_bytes(tmp_path: Path) -> None:
    """The guard reads through the codec; the packed bytes say nothing about this."""
    path = tmp_path / "corpus.gz"
    path.write_bytes(gzip.compress(b"text\x00more"))

    with pytest.raises(DatasetFormatError) as caught:
        ingest(path)

    assert caught.value.details["nul_at_byte"] == 4


def test_utf16_with_no_extension_is_read_rather_than_called_binary(tmp_path: Path) -> None:
    """Every ASCII character in UTF-16 carries a NUL, so this is the false positive."""
    path = tmp_path / "corpus"
    path.write_bytes("wide text\n".encode("utf-16"))

    documents, _ = ingest(path)

    assert [d.text for d in documents] == ["wide text\n"]


def test_utf16_with_no_extension_and_no_bom_is_read_when_the_encoding_says_so(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corpus"
    path.write_bytes("wide text\n".encode("utf-16-le"))

    documents, _ = ingest(path, encoding="utf-16-le")

    assert [d.text for d in documents] == ["wide text\n"]


def test_a_text_file_full_of_nul_bytes_is_unaffected(tmp_path: Path) -> None:
    """Scoped deliberately: a named .txt is the user's assertion, and is not re-judged.

    Widening the guard to every text file would change what existing corpora do,
    which is a decision of its own rather than a side effect of this one.
    """
    path = write(tmp_path / "weird.txt", "text")
    path.write_bytes(b"text\x00more text here\n")

    documents, _ = ingest(path)

    assert documents[0].text == "text\x00more text here\n"


def test_an_extensionless_member_of_an_archive_is_passed_over(tmp_path: Path) -> None:
    """An archive is a walk, not an instruction about any one member."""
    path = zip_of(tmp_path / "papers.zip", [("a.txt", "kept\n"), ("LICENSE", "not kept\n")])

    documents, stats = ingest(path)

    assert [d.source for d in documents] == ["papers.zip::a.txt"]
    assert stats.archive_members_ignored == 1
    assert stats.ignored_archive_members == ["papers.zip::LICENSE"]
    assert stats.assumed_text_files == []


def test_a_directory_of_readable_files_gains_no_new_counters(tmp_path: Path) -> None:
    """Negative control: an inverted guard fails here instead of changing every corpus."""
    write(tmp_path / "a.txt", "one\n")
    write(tmp_path / "b.jsonl", '{"text": "two"}\n')

    _, stats = ingest(tmp_path)

    assert stats.files_unrecognized == 0
    assert stats.files_without_extension == 0
    assert stats.unrecognized_files == []
    assert stats.assumed_text_files == []


def test_hidden_files_are_not_reported_as_passed_over(tmp_path: Path) -> None:
    """They were already excluded by name, and naming them would be noise."""
    write(tmp_path / "corpus.txt", "real text\n")
    write(tmp_path / ".gitignore", "data/\n")
    write(tmp_path / ".git" / "HEAD", "ref: refs/heads/main\n")

    _, stats = ingest(tmp_path)

    assert stats.files_unrecognized == 0
    assert stats.unrecognized_files == []


def test_a_directory_of_nothing_readable_names_what_it_held(tmp_path: Path) -> None:
    """Otherwise the user cannot tell a wrong path from an unreadable one."""
    write(tmp_path / "a.py", "print('hello')\n")
    write(tmp_path / "b.py", "print('world')\n")

    with pytest.raises(DatasetNotFoundError) as caught:
        ingest(tmp_path)

    message = str(caught.value)
    assert "2 file(s), none of them a readable format" in message
    assert "a.py, b.py" in message
    assert caught.value.details["files_unrecognized"] == 2


def test_a_fruitless_walk_says_how_many_it_did_not_list(tmp_path: Path) -> None:
    for index in range(14):
        write(tmp_path / f"f{index:02d}.py", "x\n")

    with pytest.raises(DatasetNotFoundError) as caught:
        ingest(tmp_path)

    assert "and 4 more" in str(caught.value)


# --------------------------------------------------------------------------- #
# Word documents
#
# Fixtures here are built by hand, so they test the reader against this file's
# idea of the format rather than against Word's. What pins them to reality is a
# separate check run against four real Word documents -- with pictures, tables,
# hyperlinks and footnotes: for each one, the non-whitespace characters the reader
# returned were exactly the concatenation of every ``w:t`` element in
# ``word/document.xml``, and a strict superset of what ``python-docx`` extracted,
# which omits table cells. The cases below fix the details that check cannot see:
# what is left out on purpose, and what happens to a file that is not really a
# Word document at all.
# --------------------------------------------------------------------------- #
_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
_PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
_MAIN_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"


def para(*runs: str) -> str:
    """One ``w:p``, from raw run XML or from plain text."""
    inner = "".join(f"<w:r><w:t>{run}</w:t></w:r>" if "<" not in run else run for run in runs)
    return f"<w:p>{inner}</w:p>"


def docx(
    path: Path,
    body: str,
    *,
    main_part: str = "word/document.xml",
) -> Path:
    """A minimal package with the same shape as one Word writes."""
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{_W}" xmlns:mc="{_MC}"><w:body>{body}</w:body></w:document>'
    )
    types = (
        '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.'
        'openxmlformats.org/package/2006/content-types"><Default Extension="xml" '
        'ContentType="application/xml"/></Types>'
    )
    rels = (
        f'<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="{_PKG}">'
        f'<Relationship Id="rId1" Type="{_MAIN_REL}" Target="{main_part}"/></Relationships>'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("[Content_Types].xml", types)
        package.writestr("_rels/.rels", rels)
        package.writestr(main_part, document)
    return path


def test_a_word_document_yields_its_paragraphs(tmp_path: Path) -> None:
    path = docx(tmp_path / "report.docx", para("First paragraph.") + para("Second paragraph."))

    documents, stats = ingest(path)

    assert [d.text for d in documents] == ["First paragraph.\nSecond paragraph."]
    assert stats.files_read == 1


def test_a_word_document_is_classified_as_docx_not_as_an_archive(tmp_path: Path) -> None:
    """A .docx *is* a zip. Treating it as one would hand back its XML as documents."""
    docx(tmp_path / "report.docx", para("Prose, not markup."))

    sources = Ingestor().discover(tmp_path)

    assert [(s.relative, s.kind) for s in sources] == [("report.docx", "docx")]


def test_an_empty_paragraph_is_a_blank_line(tmp_path: Path) -> None:
    """What Word's own plain-text export does: one newline per paragraph, no more."""
    path = docx(tmp_path / "a.docx", para("Heading") + "<w:p/>" + para("Body"))

    documents, _ = ingest(path)

    assert documents[0].text == "Heading\n\nBody"


def test_breaks_tabs_and_hard_hyphens_inside_a_paragraph_are_kept(tmp_path: Path) -> None:
    body = para(
        "<w:r><w:t>one</w:t><w:br/><w:t>two</w:t><w:tab/><w:t>three</w:t>"
        "<w:noBreakHyphen/><w:t>four</w:t></w:r>"
    )
    path = docx(tmp_path / "a.docx", body)

    documents, _ = ingest(path)

    assert documents[0].text == "one\ntwo\tthree-four"


def test_table_cells_are_read_as_lines(tmp_path: Path) -> None:
    """A table's contents are prose the author wrote, so they belong in the corpus."""
    row = f"<w:tr><w:tc>{para('Practice')}</w:tc><w:tc>{para('Description')}</w:tc></w:tr>"
    path = docx(tmp_path / "a.docx", para("Before") + f"<w:tbl>{row}</w:tbl>" + para("After"))

    documents, _ = ingest(path)

    assert documents[0].text == "Before\nPractice\nDescription\nAfter"


def test_field_codes_and_deleted_text_are_left_out(tmp_path: Path) -> None:
    """Text that is in the file but not in the document.

    ``w:instrText`` is a field instruction -- ``HYPERLINK "http://..."`` and the
    like -- and ``w:delText`` is text struck out under tracked changes. An
    insertion is the mirror case and *is* part of the document, so it stays.
    """
    body = para(
        '<w:r><w:instrText> HYPERLINK "http://example.com" </w:instrText></w:r>'
        "<w:r><w:t>Visible.</w:t></w:r>"
        "<w:del><w:r><w:delText>Struck out.</w:delText></w:r></w:del>"
        "<w:ins><w:r><w:t> Added.</w:t></w:r></w:ins>"
    )
    path = docx(tmp_path / "a.docx", body)

    documents, _ = ingest(path)

    assert documents[0].text == "Visible. Added."


def test_a_text_box_is_read_once_not_twice(tmp_path: Path) -> None:
    """``mc:AlternateContent`` holds the same content twice, in two markups.

    Word writes a text box as a modern ``mc:Choice`` and an older ``mc:Fallback``
    saying the same thing. A walk that visits both puts every text box into the
    training data twice, which teaches the model to repeat itself.
    """
    box = f"<w:txbxContent>{para('Boxed text.')}</w:txbxContent>"
    body = para(
        "<w:r><w:t>Around it.</w:t></w:r>"
        f"<mc:AlternateContent><mc:Choice Requires='wps'>{box}</mc:Choice>"
        f"<mc:Fallback>{box}</mc:Fallback></mc:AlternateContent>"
    )
    path = docx(tmp_path / "a.docx", body)

    documents, _ = ingest(path)

    assert documents[0].text.count("Boxed text.") == 1


def test_a_long_word_document_is_cut_into_documents(tmp_path: Path) -> None:
    path = docx(tmp_path / "book.docx", "".join(para("word " * 40) for _ in range(20)))

    documents, stats = ingest(path, max_doc_chars=500)

    assert len(documents) > 1
    assert all(len(d.text) <= 500 for d in documents)
    assert stats.long_files_split == 1
    # Cutting must not lose or invent characters, and each piece must say where it
    # started -- an offset into the extracted prose, the file itself being XML.
    assert "".join(d.text for d in documents) == ingest(path)[0][0].text
    running = 0
    for ordinal, document in enumerate(documents):
        assert (document.ordinal, document.byte_offset) == (ordinal, running)
        running += len(document.text.encode("utf-8"))


def test_a_word_document_with_no_text_yields_no_documents(tmp_path: Path) -> None:
    """A document of nothing but pictures is empty, not broken."""
    path = docx(tmp_path / "pictures.docx", "<w:p><w:r><w:drawing/></w:r></w:p>")

    documents, stats = ingest(path)

    assert documents == []
    assert stats.files_read == 1


def test_deep_nesting_does_not_overflow_the_stack(tmp_path: Path) -> None:
    """A recursive walk would raise RecursionError here; expat parses this happily."""
    depth = 5_000
    body = (
        ("<w:sdt><w:sdtContent>" * depth)
        + para("Down there.")
        + ("</w:sdtContent></w:sdt>" * depth)
    )
    path = docx(tmp_path / "deep.docx", body)

    documents, _ = ingest(path)

    assert documents[0].text == "Down there."


def test_a_word_document_inside_a_zip_yields_prose_not_xml(tmp_path: Path) -> None:
    """The ordering trap: the member is a zip too, and must not be opened as one."""
    inner = docx(tmp_path / "report.docx", para("Prose from inside an archive."))
    archive = tmp_path / "papers.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        package.write(inner, "reports/report.docx")

    documents, stats = ingest(archive)

    assert [d.text for d in documents] == ["Prose from inside an archive."]
    assert [d.source for d in documents] == ["papers.zip::reports/report.docx"]
    assert stats.archive_members_ignored == 0
    assert stats.nested_archives_ignored == 0


def test_a_compressed_word_document_reads(tmp_path: Path) -> None:
    plain = docx(tmp_path / "plain.docx", para("Through a codec."))
    path = tmp_path / "report.docx.gz"
    path.write_bytes(gzip.compress(plain.read_bytes()))

    documents, _ = ingest(path)

    assert [d.text for d in documents] == ["Through a codec."]


def test_the_main_part_may_be_named_by_the_relationships(tmp_path: Path) -> None:
    """Word always writes word/document.xml; the format does not require it."""
    path = docx(tmp_path / "a.docx", para("Named elsewhere."), main_part="word/main.xml")

    documents, _ = ingest(path)

    assert [d.text for d in documents] == ["Named elsewhere."]


def test_a_legacy_doc_renamed_to_docx_says_so(tmp_path: Path) -> None:
    """The likeliest cause of "not a zip" for this extension, so it gets named."""
    path = tmp_path / "old.docx"
    path.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + bytes(512))

    with pytest.raises(DatasetFormatError) as caught:
        ingest(path)

    assert "OLE compound file" in str(caught.value)
    assert "password-protected" in str(caught.value)
    assert caught.value.details["format"] == "ole"


def test_a_docx_that_is_not_a_zip_at_all_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path / "plain.docx", "This is just text in a misnamed file.\n")

    with pytest.raises(DatasetFormatError) as caught:
        ingest(path)

    assert "could not be read as a .docx" in str(caught.value)
    assert "zip archive of XML" in (caught.value.hint or "")


def test_a_zip_renamed_to_docx_names_what_it_holds_instead(tmp_path: Path) -> None:
    path = tmp_path / "corpus.docx"
    with zipfile.ZipFile(path, "w") as package:
        package.writestr("notes.txt", "some text\n")

    with pytest.raises(DatasetFormatError) as caught:
        ingest(path)

    assert "no word/document.xml part" in str(caught.value)
    assert "notes.txt" in (caught.value.hint or "")
    assert "rename it .zip" in (caught.value.hint or "")


def test_a_document_type_declaration_is_refused(tmp_path: Path) -> None:
    """Entity expansion turns parsing a file into running a program.

    600 bytes of nested entity definitions expand to gigabytes. This interpreter's
    expat caps the amplification, but that depends on the libexpat a given Python
    was built against, and OOXML forbids a DTD anyway -- so it is refused here
    rather than left to somebody else's build.
    """
    path = tmp_path / "bomb.docx"
    with zipfile.ZipFile(path, "w") as package:
        package.writestr(
            "word/document.xml",
            '<?xml version="1.0"?>\n'
            '<!DOCTYPE d [<!ENTITY a "aaaaaaaaaa">'
            ' <!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">]>\n'
            f'<w:document xmlns:w="{_W}"><w:body><w:p><w:r><w:t>&b;</w:t></w:r>'
            "</w:p></w:body></w:document>",
        )

    with pytest.raises(DatasetFormatError) as caught:
        ingest(path)

    assert "document type declaration" in str(caught.value)
    assert caught.value.details["part"] == "word/document.xml"


def test_prose_mentioning_a_doctype_is_not_mistaken_for_one(tmp_path: Path) -> None:
    """The check cannot misfire on text: a raw ``<`` cannot appear in character data."""
    path = docx(tmp_path / "a.docx", para("Every HTML file starts with &lt;!DOCTYPE html&gt;."))

    documents, _ = ingest(path)

    assert documents[0].text == "Every HTML file starts with <!DOCTYPE html>."


def test_a_damaged_document_part_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "torn.docx"
    with zipfile.ZipFile(path, "w") as package:
        package.writestr("word/document.xml", f'<w:document xmlns:w="{_W}"><w:body>')

    with pytest.raises(DatasetFormatError) as caught:
        ingest(path)

    assert "not valid XML" in str(caught.value)


def test_a_document_part_over_the_ceiling_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bound on memory: the parsed tree costs 8-9x the XML it came from."""
    monkeypatch.setattr(ingest_module, "_DOCX_MAX_XML_BYTES", 256)
    path = docx(tmp_path / "big.docx", "".join(para("padding") for _ in range(50)))

    with pytest.raises(DatasetFormatError) as caught:
        ingest(path)

    assert "over the 256-byte limit" in str(caught.value)
    assert caught.value.details["part"] == "word/document.xml"
    assert "save it as plain text" in (caught.value.hint or "")


def test_a_part_that_under_reports_its_size_is_refused_not_truncated(tmp_path: Path) -> None:
    """The one untrusted number here is one ``zipfile`` checks for us.

    Rewriting a member's uncompressed size to 8 bytes while it holds 5,000 does not
    yield 8 bytes of XML: ``zipfile`` caps the read at the declared size and then
    fails the CRC. So there is no second ceiling check on what arrived -- what
    matters is that a lying package is refused rather than silently truncated, and
    that is what this pins.
    """
    path = tmp_path / "liar.docx"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr(
            "word/document.xml",
            f'<w:document xmlns:w="{_W}"><w:body>{para("x" * 5_000)}</w:body></w:document>',
        )
    # One member, so the first local and central headers are both its own. The
    # uncompressed size sits at offset 22 of the local header and 24 of the central.
    data = bytearray(path.read_bytes())
    for signature, offset in ((b"PK\x03\x04", 22), (b"PK\x01\x02", 24)):
        struct.pack_into("<I", data, data.index(signature) + offset, 8)
    path.write_bytes(bytes(data))

    with pytest.raises(DatasetFormatError) as caught:
        ingest(path)

    assert "could not be read" in str(caught.value)
    assert "Bad CRC-32" in caught.value.details["reason"]


def test_a_compressed_docx_that_expands_past_the_package_ceiling_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Here the declared size really is untrusted: it is the *compressed* size.

    A ``.docx.gz`` reports its size on disk, which says nothing about how much has
    to be held in memory to read it, so the buffer is measured as it fills.
    """
    monkeypatch.setattr(ingest_module, "_DOCX_MAX_PACKAGE_BYTES", 4096)
    plain = tmp_path / "plain.docx"
    docx(plain, para("padding"))
    with zipfile.ZipFile(plain, "a", zipfile.ZIP_STORED) as package:
        package.writestr("word/media/image1.bin", " " * 200_000)
    path = tmp_path / "report.docx.gz"
    path.write_bytes(gzip.compress(plain.read_bytes()))
    assert path.stat().st_size < 4096, "the point is a declared size under the ceiling"

    with pytest.raises(DatasetFormatError) as caught:
        ingest(path)

    assert "expands to more than 4,096 bytes" in str(caught.value)
    assert caught.value.details["measured"] == "decompressed"


def test_a_docx_from_an_archive_over_the_package_ceiling_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only reachable from an archive or a codec: a plain file is never buffered."""
    monkeypatch.setattr(ingest_module, "_DOCX_MAX_PACKAGE_BYTES", 128)
    inner = docx(tmp_path / "report.docx", "".join(para("padding") for _ in range(50)))
    archive = tmp_path / "papers.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        package.write(inner, "report.docx")

    with pytest.raises(DatasetFormatError) as caught:
        ingest(archive)

    assert "not a plain file on disk" in str(caught.value)
    assert caught.value.details["measured"] == "declared"
    assert "Unpack it" in (caught.value.hint or "")


def test_a_plain_word_document_is_not_buffered_whole(tmp_path: Path) -> None:
    """The ceiling that applies to buffering must not apply to a file on disk."""
    path = docx(tmp_path / "report.docx", para("Read in place."))
    ingestor = Ingestor()

    def refuse(source: object) -> bytearray:
        raise AssertionError("a plain .docx on disk should be read without buffering")

    ingestor._docx_buffered = refuse  # type: ignore[method-assign]

    documents = list(ingestor.documents(ingestor.discover(path)))

    assert [d.text for d in documents] == ["Read in place."]


def test_the_supported_formats_sentence_names_docx() -> None:
    assert ".docx" in describe_supported_formats()


# --------------------------------------------------------------------------- #
# Databases
#
# Fixtures here are built by sqlite3 itself, so the file on disk is a real
# database and not this file's idea of one. What that leaves untested is
# everything about *other* people's databases, and the two things worth naming
# are both measured rather than asserted: reading 40 MB of text out of a cursor
# peaked at 4.5 KB of traced memory, and a database held open by another process
# under ``BEGIN EXCLUSIVE`` raises "database is locked" after the timeout. The
# cases below pin what a fixture can pin -- above all that the row order is fixed
# by the reader and not by whatever SQLite felt like returning.
# --------------------------------------------------------------------------- #
def db(
    path: Path,
    statements: list[str],
    rows: dict[str, list[tuple[object, ...]]] | None = None,
) -> Path:
    """A real SQLite database, built by ``sqlite3``.

    A connection alone does not make a file -- ``sqlite3.connect`` on a new path
    leaves it empty at zero bytes, with no header. It takes one committed object
    for the ``SQLite format 3`` magic to appear.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        for statement in statements:
            connection.execute(statement)
        for table, values in (rows or {}).items():
            placeholders = ", ".join("?" * len(values[0]))
            connection.executemany(f'INSERT INTO "{table}" VALUES ({placeholders})', values)
        connection.commit()
    finally:
        connection.close()
    return path


def test_a_database_with_one_table_reads_one_document_per_row(tmp_path: Path) -> None:
    path = db(
        tmp_path / "one.db",
        ["CREATE TABLE posts (id INTEGER PRIMARY KEY, body TEXT)"],
        {"posts": [(5, "fifth"), (1, "first"), (3, "third")]},
    )

    documents, stats = ingest(path)

    # Inserted 5, 1, 3 and read back 1, 3, 5: an INTEGER PRIMARY KEY *is* the
    # rowid, so ordering by the rowid orders by it. Not insertion order -- the
    # stored order, which is what survives a copy of the file.
    assert [d.text for d in documents] == ["first", "third", "fifth"]
    assert [d.ordinal for d in documents] == [0, 1, 2]
    assert [d.byte_offset for d in documents] == [0, 0, 0]
    assert stats.files_read == 1
    assert stats.documents_emitted == 3


def test_the_chosen_table_and_column_are_reported(tmp_path: Path) -> None:
    """Both were guesses, so both have to be visible in the report."""
    path = db(
        tmp_path / "one.db",
        ["CREATE TABLE posts (id INT, body TEXT)"],
        {"posts": [(1, "prose")]},
    )

    _, stats = ingest(path)

    assert stats.resolved_db_tables == {"one.db": "posts"}
    assert stats.resolved_db_columns == {"one.db": "body"}


def test_an_index_does_not_change_the_row_order(tmp_path: Path) -> None:
    """The reason ``ORDER BY`` is written out at all.

    A plain ``SELECT`` on this table happens to return stored order today, index
    or no index -- measured. It is free to stop doing that, and nothing about the
    file would change while every checksum in the manifest did.
    """
    path = db(
        tmp_path / "notes.db",
        ["CREATE TABLE notes (body TEXT)"],
        {"notes": [("zulu",), ("alpha",), ("mike",)]},
    )
    before, _ = ingest(path)

    connection = sqlite3.connect(path)
    connection.execute("CREATE INDEX notes_body ON notes(body)")
    connection.commit()
    connection.close()
    after, _ = ingest(path)

    assert [d.text for d in before] == ["zulu", "alpha", "mike"]
    assert [d.text for d in after] == [d.text for d in before]


def test_a_single_column_table_needs_no_column_named(tmp_path: Path) -> None:
    """One column is not a choice, whatever it is called."""
    path = db(
        tmp_path / "one.db",
        ["CREATE TABLE t (whatever TEXT)"],
        {"t": [("alpha",), ("beta",)]},
    )

    documents, stats = ingest(path)

    assert [d.text for d in documents] == ["alpha", "beta"]
    assert stats.resolved_db_columns == {"one.db": "whatever"}


def test_several_tables_are_refused_and_listed(tmp_path: Path) -> None:
    path = db(
        tmp_path / "many.db",
        ["CREATE TABLE posts (body TEXT)", "CREATE TABLE users (name TEXT)"],
        {"posts": [("hello",)], "users": [("ada",)]},
    )

    with pytest.raises(DatasetFormatError) as caught:
        ingest(path)

    assert "holds 2 tables" in str(caught.value)
    assert caught.value.details["tables"] == ["posts", "users"]
    assert "--db-table" in (caught.value.hint or "")


def test_a_named_table_is_read_ignoring_case(tmp_path: Path) -> None:
    path = db(
        tmp_path / "many.db",
        ["CREATE TABLE posts (body TEXT)", "CREATE TABLE users (name TEXT)"],
        {"posts": [("hello",)], "users": [("ada",)]},
    )

    documents, stats = ingest(path, db_table="POSTS")

    assert [d.text for d in documents] == ["hello"]
    # Named, so not a guess, so nothing to report.
    assert stats.resolved_db_tables == {}


def test_a_table_name_that_is_not_there_lists_the_ones_that_are(tmp_path: Path) -> None:
    path = db(
        tmp_path / "many.db",
        ["CREATE TABLE posts (body TEXT)", "CREATE TABLE users (name TEXT)"],
        {"posts": [("hello",)], "users": [("ada",)]},
    )

    with pytest.raises(DatasetFormatError) as caught:
        ingest(path, db_table="comments")

    assert "has no table named 'comments'" in str(caught.value)
    assert caught.value.details["available_tables"] == ["posts", "users"]


def test_a_conventional_column_name_is_resolved_among_several(tmp_path: Path) -> None:
    path = db(
        tmp_path / "one.db",
        ["CREATE TABLE t (id INT, body TEXT, score INT)"],
        {"t": [(1, "the body", 9)]},
    )

    documents, stats = ingest(path)

    assert [d.text for d in documents] == ["the body"]
    assert stats.resolved_db_columns == {"one.db": "body"}


def test_several_columns_with_no_conventional_name_are_refused(tmp_path: Path) -> None:
    path = db(
        tmp_path / "weather.db",
        ["CREATE TABLE t (city TEXT, summary TEXT, temp REAL)"],
        {"t": [("Oslo", "cold", 1.0)]},
    )

    with pytest.raises(DatasetFormatError) as caught:
        ingest(path)

    assert "has 3 columns" in str(caught.value)
    assert caught.value.details["available_columns"] == ["city", "summary", "temp"]
    assert "--csv-text-column" in (caught.value.hint or "")
    assert "HistGradientBoostingRegressor" in (caught.value.hint or "")


def test_a_column_is_named_with_the_same_flag_a_csv_uses(tmp_path: Path) -> None:
    """One question -- which column holds the text -- so one flag, not two."""
    path = db(
        tmp_path / "weather.db",
        ["CREATE TABLE t (city TEXT, summary TEXT, temp REAL)"],
        {"t": [("Oslo", "cold", 1.0), ("Cairo", "hot", 35.0)]},
    )

    documents, _ = ingest(path, csv_text_column="SUMMARY")

    assert [d.text for d in documents] == ["cold", "hot"]


def test_a_column_name_that_is_not_there_lists_the_columns(tmp_path: Path) -> None:
    path = db(
        tmp_path / "weather.db",
        ["CREATE TABLE t (city TEXT, summary TEXT, temp REAL)"],
        {"t": [("Oslo", "cold", 1.0)]},
    )

    with pytest.raises(DatasetFormatError) as caught:
        ingest(path, csv_text_column="text")

    assert "has no column named 'text'" in str(caught.value)
    assert caught.value.details["available_columns"] == ["city", "summary", "temp"]


def test_a_full_text_index_is_one_table_not_six(tmp_path: Path) -> None:
    """FTS5 stores itself in five extra tables that ``sqlite_master`` calls tables.

    Without the shadow-table rule this database would be reported as ambiguous
    and the user told to choose between ``notes_idx`` and ``notes_data``.
    """
    path = tmp_path / "fts.db"
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE VIRTUAL TABLE notes USING fts5(body)")
    except sqlite3.OperationalError as exc:  # pragma: no cover - FTS5 is normally built in
        pytest.skip(f"this SQLite has no FTS5: {exc}")
    finally:
        connection.close()
    db(path, [], {"notes": [("the first note",), ("the second note",)]})

    documents, stats = ingest(path)

    assert [d.text for d in documents] == ["the first note", "the second note"]
    assert stats.resolved_db_tables == {"fts.db": "notes"}


def test_a_virtual_table_beside_a_real_one_is_still_ambiguous(tmp_path: Path) -> None:
    """The shadow rule narrows the choice. It must not make one."""
    path = tmp_path / "both.db"
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE VIRTUAL TABLE notes USING fts5(body)")
    except sqlite3.OperationalError as exc:  # pragma: no cover - FTS5 is normally built in
        pytest.skip(f"this SQLite has no FTS5: {exc}")
    finally:
        connection.close()
    db(path, ["CREATE TABLE other (body TEXT)"])

    with pytest.raises(DatasetFormatError) as caught:
        ingest(path)

    assert caught.value.details["tables"] == ["notes", "other"]


def test_a_view_is_not_chosen_automatically(tmp_path: Path) -> None:
    path = db(
        tmp_path / "views.db",
        [
            "CREATE TABLE posts (body TEXT, score INT)",
            "CREATE VIEW popular AS SELECT body FROM posts WHERE score > 2",
        ],
        {"posts": [("kept", 9), ("dropped", 1)]},
    )

    documents, stats = ingest(path)

    assert [d.text for d in documents] == ["kept", "dropped"]
    assert stats.resolved_db_tables == {"views.db": "posts"}


def test_naming_a_view_is_refused_because_it_has_no_row_order(tmp_path: Path) -> None:
    path = db(
        tmp_path / "views.db",
        [
            "CREATE TABLE posts (body TEXT, score INT)",
            "CREATE VIEW popular AS SELECT body FROM posts WHERE score > 2",
        ],
        {"posts": [("kept", 9), ("dropped", 1)]},
    )

    with pytest.raises(DatasetFormatError) as caught:
        ingest(path, db_table="popular")

    assert "is a view rather than a table" in str(caught.value)
    assert caught.value.details["view"] == "popular"


def test_a_database_of_nothing_but_views_names_them(tmp_path: Path) -> None:
    path = db(tmp_path / "views.db", ["CREATE VIEW v AS SELECT 1 AS body"])

    with pytest.raises(DatasetFormatError) as caught:
        ingest(path)

    assert "holds no table TrainAI can read" in str(caught.value)
    assert caught.value.details["views"] == ["v"]


def test_a_database_holding_nothing_at_all_says_so(tmp_path: Path) -> None:
    """Reachable: creating a table and dropping it leaves a real, empty database."""
    path = db(tmp_path / "blank.db", ["CREATE TABLE t (body TEXT)", "DROP TABLE t"])

    with pytest.raises(DatasetFormatError) as caught:
        ingest(path)

    assert "holds no table" in str(caught.value)
    assert "declares no tables at all" in (caught.value.hint or "")


def test_a_without_rowid_table_is_ordered_by_its_primary_key(tmp_path: Path) -> None:
    """No rowid to order by, so the key is used -- in its declared order, b then a."""
    path = db(
        tmp_path / "wr.db",
        ["CREATE TABLE t (a TEXT, b INT, body TEXT, PRIMARY KEY (b, a)) WITHOUT ROWID"],
        {"t": [("z", 2, "second"), ("y", 1, "first"), ("x", 3, "third")]},
    )

    documents, _ = ingest(path, csv_text_column="body")

    assert [d.text for d in documents] == ["first", "second", "third"]


def test_a_column_named_rowid_does_not_become_the_row_order(tmp_path: Path) -> None:
    """``ORDER BY rowid`` here would sort by the declared column, which is not the order."""
    path = db(
        tmp_path / "shadow.db",
        ["CREATE TABLE t (rowid TEXT, body TEXT)"],
        {"t": [("b", "stored first"), ("a", "stored second")]},
    )

    documents, _ = ingest(path)

    assert [d.text for d in documents] == ["stored first", "stored second"]


def test_a_table_with_every_rowid_spelling_shadowed_falls_back_to_its_key(
    tmp_path: Path,
) -> None:
    path = db(
        tmp_path / "shadow.db",
        ["CREATE TABLE t (rowid TEXT, _rowid_ TEXT, oid TEXT, body TEXT, k INT PRIMARY KEY)"],
        {"t": [("1", "1", "1", "second", 2), ("1", "1", "1", "first", 1)]},
    )

    documents, _ = ingest(path, csv_text_column="body")

    assert [d.text for d in documents] == ["first", "second"]


def test_a_table_with_no_row_order_left_is_refused(tmp_path: Path) -> None:
    """Every spelling of the rowid hidden, and no key. Nothing to reproduce."""
    path = db(
        tmp_path / "shadow.db",
        ["CREATE TABLE t (rowid TEXT, _rowid_ TEXT, oid TEXT, body TEXT)"],
        {
            "t": [
                (
                    "1",
                    "1",
                    "1",
                    "somewhere",
                )
            ]
        },
    )

    with pytest.raises(DatasetFormatError) as caught:
        ingest(path, csv_text_column="body")

    assert "has no row order TrainAI can pin down" in str(caught.value)
    assert caught.value.details["table"] == "t"
    assert "export the table to .csv" in (caught.value.hint or "")


def test_names_that_need_quoting_are_read(tmp_path: Path) -> None:
    """Identifiers reach SQL by hand, so the quoting has to be right."""
    path = db(
        tmp_path / "awkward.db",
        ['CREATE TABLE "my table" ("the ""body""" TEXT)'],
        {"my table": [("quoted",)]},
    )

    documents, stats = ingest(path)

    assert [d.text for d in documents] == ["quoted"]
    assert stats.resolved_db_tables == {"awkward.db": "my table"}


def test_a_hash_in_the_file_name_leaves_no_stray_database_behind(tmp_path: Path) -> None:
    """The measured bug this reader was nearly shipped with.

    Pasting the path after ``file:`` turns ``hash#1.db`` into the URI
    ``file:.../hash`` with a fragment -- which discards ``?mode=ro`` along with
    the rest of the name, so SQLite opened read-write-create and left a new empty
    file called ``hash`` in the user's directory.
    """
    path = db(
        tmp_path / "hash#1.db",
        ["CREATE TABLE t (body TEXT)"],
        {"t": [("read me",)]},
    )

    documents, _ = ingest(path)

    assert [d.text for d in documents] == ["read me"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["hash#1.db"]


def test_a_percent_and_a_space_in_the_file_name_are_read(tmp_path: Path) -> None:
    """``%`` is the other character a hand-built URI gets wrong: it starts an escape."""
    path = db(
        tmp_path / "per cent%20.db",
        ["CREATE TABLE t (body TEXT)"],
        {"t": [("read me",)]},
    )

    documents, _ = ingest(path)

    assert [d.text for d in documents] == ["read me"]


def test_the_row_order_is_written_into_the_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whitebox, deliberately, because today's SQLite makes this unobservable.

    A plain ``SELECT`` returns stored order on this build, index or no index --
    measured. So deleting the ``ORDER BY`` changes nothing any behavioural test
    can see, which is precisely why it has to be asserted directly: SQLite is
    *free* to return index order instead, and on the day it does, every checksum
    in the manifest changes while the file on disk is untouched. The clause is the
    guarantee, so the clause is what this pins.
    """
    path = db(
        tmp_path / "posts.db",
        ["CREATE TABLE posts (body TEXT)"],
        {"posts": [("first",), ("second",)]},
    )
    statements: list[str] = []
    connect = sqlite3.connect

    class Recorder:
        """Only ``execute`` and ``close`` are used, and ``Connection`` is a C type."""

        def __init__(self, connection: sqlite3.Connection) -> None:
            self._connection = connection

        def execute(self, statement: str, *args: object) -> sqlite3.Cursor:
            statements.append(statement)
            return self._connection.execute(statement, *args)

        def close(self) -> None:
            self._connection.close()

    monkeypatch.setattr(
        ingest_module.sqlite3,
        "connect",
        lambda *args, **kwargs: Recorder(connect(*args, **kwargs)),
    )

    documents, _ = ingest(path)

    assert [d.text for d in documents] == ["first", "second"]
    # The catalogue query and the rowid probe are also SELECTs; the read is the
    # only one that asks for the text column.
    reads = [statement for statement in statements if statement.startswith('SELECT "body"')]
    assert reads == ['SELECT "body" FROM "posts" ORDER BY rowid']


def test_a_db_that_is_not_a_database_is_refused_on_its_header(tmp_path: Path) -> None:
    path = tmp_path / "Thumbs.db"
    path.write_bytes(b"not a database at all\n")

    with pytest.raises(DatasetFormatError) as caught:
        ingest(path)

    assert "is not a SQLite database" in str(caught.value)
    assert "SQLite format 3" in str(caught.value)
    assert "thumbnail caches" in (caught.value.hint or "")


def test_a_right_header_over_a_broken_body_passes_sqlites_own_words_on(tmp_path: Path) -> None:
    """The magic check is a better message, not a validation. This is the rest of it."""
    path = tmp_path / "truncated.db"
    path.write_bytes(b"SQLite format 3\x00" + b"\xff" * 4096)

    with pytest.raises(DatasetFormatError) as caught:
        ingest(path)

    assert "could not be read as a database" in str(caught.value)
    assert caught.value.details["reason"] == "file is not a database"


def test_a_walk_reads_a_real_database_and_names_the_one_that_is_not(tmp_path: Path) -> None:
    """The conservative half of the rule: a stray ``Thumbs.db`` must not fail the run.

    Pointed at directly it is an error, because naming a file is an instruction.
    Found in a tree it is somebody else's junk -- passed over, and named.
    """
    write(tmp_path / "corpus.txt", "plain prose here, long enough to keep\n")
    (tmp_path / "Thumbs.db").write_bytes(b"not a database at all\n")
    db(
        tmp_path / "posts.db",
        ["CREATE TABLE posts (body TEXT)"],
        {"posts": [("from the database",)]},
    )

    documents, stats = ingest(tmp_path)

    assert sorted(d.text for d in documents) == [
        "from the database",
        "plain prose here, long enough to keep\n",
    ]
    assert stats.unrecognized_files == ["Thumbs.db"]
    assert stats.files_unrecognized == 1


def test_a_compressed_database_is_refused_with_the_reason(tmp_path: Path) -> None:
    plain = db(
        tmp_path / "posts.db",
        ["CREATE TABLE posts (body TEXT)"],
        {"posts": [("hello",)]},
    )
    path = tmp_path / "backup.db.gz"
    path.write_bytes(gzip.compress(plain.read_bytes()))

    with pytest.raises(DatasetFormatError) as caught:
        ingest(path)

    assert "is a compressed database, which TrainAI cannot read in place" in str(caught.value)
    assert caught.value.details["compression"] == "gzip"
    assert "seeked that way" in (caught.value.hint or "")


def test_a_walk_passes_over_a_compressed_database(tmp_path: Path) -> None:
    """Same asymmetry as ``Thumbs.db``: a backup beside a corpus must not fail the run."""
    write(tmp_path / "corpus.txt", "plain prose here, long enough to keep\n")
    # Never opened, so the bytes inside do not matter -- the extension is the
    # whole of what a walk goes by, which is the point being pinned.
    (tmp_path / "backup.db.gz").write_bytes(gzip.compress(b"SQLite format 3\x00" + b"\x00" * 64))

    documents, stats = ingest(tmp_path)

    assert [d.text for d in documents] == ["plain prose here, long enough to keep\n"]
    assert stats.unrecognized_files == ["backup.db.gz"]


def test_a_database_inside_an_archive_is_counted_as_an_ignored_member(tmp_path: Path) -> None:
    """A mixed archive still prepares. The database in it is named, not fatal."""
    plain = db(tmp_path / "posts.db", ["CREATE TABLE t (body TEXT)"], {"t": [("unread",)]})
    archive = tmp_path / "papers.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        package.write(plain, "posts.db")
        package.writestr("notes.txt", "prose in the same archive, long enough\n")

    documents, stats = ingest(archive)

    assert [d.text for d in documents] == ["prose in the same archive, long enough\n"]
    assert stats.archive_members_ignored == 1
    assert stats.ignored_archive_members == ["papers.zip::posts.db"]


def test_an_archive_holding_only_a_database_says_why(tmp_path: Path) -> None:
    plain = db(tmp_path / "posts.db", ["CREATE TABLE t (body TEXT)"], {"t": [("unread",)]})
    archive = tmp_path / "papers.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        package.write(plain, "posts.db")

    with pytest.raises(DatasetNotFoundError) as caught:
        ingest(archive)

    assert "holds nothing TrainAI can read" in str(caught.value)
    assert "Unpack one level first" in (caught.value.hint or "")


def test_a_database_reached_as_an_archive_member_is_refused(tmp_path: Path) -> None:
    """Unreachable through ``discover`` -- which is why the guard is worth a test.

    Without it, ``source.path`` is the *archive*, and the reader would open a zip
    file as a database and blame the user's data for the result.
    """
    plain = db(tmp_path / "posts.db", ["CREATE TABLE t (body TEXT)"], {"t": [("unread",)]})
    archive = tmp_path / "papers.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        package.write(plain, "posts.db")
    source = SourceFile(
        path=archive,
        relative="papers.zip::posts.db",
        kind="sqlite",
        compressed=False,
        size_bytes=plain.stat().st_size,
        archive="zip",
        member="posts.db",
    )

    with pytest.raises(DatasetFormatError) as caught:
        list(Ingestor().documents([source]))

    assert "is a database inside an archive" in str(caught.value)
    assert caught.value.details["archive"] == "zip"


def test_a_cell_that_is_not_text_is_refused_naming_its_row(tmp_path: Path) -> None:
    """Well targeted: a TEXT column would have converted a number on the way out."""
    path = db(
        tmp_path / "blobs.db",
        ["CREATE TABLE t (body BLOB)"],
        {"t": [(b"\x00\xff",), ("prose",)]},
    )

    with pytest.raises(DatasetFormatError) as caught:
        ingest(path)

    assert "row 1 holds bytes in column 'body', not text" in str(caught.value)
    assert caught.value.details["type"] == "bytes"


def test_a_cell_that_is_not_text_can_be_skipped_and_counted(tmp_path: Path) -> None:
    path = db(
        tmp_path / "blobs.db",
        ["CREATE TABLE t (body BLOB)"],
        {"t": [(b"\x00\xff",), ("prose",)]},
    )

    documents, stats = ingest(path, on_error="skip")

    assert [d.text for d in documents] == ["prose"]
    assert stats.records_skipped_malformed == 1


def test_a_null_cell_is_an_empty_document_not_a_broken_one(tmp_path: Path) -> None:
    """A missing value is missing, not malformed -- it drops out as too short."""
    path = db(
        tmp_path / "nulls.db",
        ["CREATE TABLE t (body TEXT)"],
        {"t": [(None,), ("real prose",)]},
    )

    documents, stats = ingest(path)

    assert [d.text for d in documents] == ["real prose"]
    assert stats.documents_skipped_short == 1
    assert stats.records_skipped_malformed == 0


def test_a_database_can_be_read_across_threads(tmp_path: Path) -> None:
    """Documents leave this module as a generator, and the BPE trainer moves it.

    Measured: ``tokenizers.train_from_iterator`` pulls one corpus from seven
    distinct worker threads, none of them the thread that started the read, and
    never two at once. sqlite3's default refuses the second thread outright, so
    without ``check_same_thread=False`` every database corpus dies partway through
    tokenizer training with a message about thread ids. This drives the same shape
    directly: start on this thread, finish on another.
    """
    path = db(
        tmp_path / "posts.db",
        ["CREATE TABLE posts (body TEXT)"],
        {"posts": [(f"row {index}",) for index in range(5)]},
    )
    ingestor = Ingestor()
    documents = ingestor.documents(ingestor.discover(path))
    first = next(documents)
    rest: list[str] = []
    failure: list[BaseException] = []

    def drain() -> None:
        try:
            rest.extend(document.text for document in documents)
        except BaseException as exc:  # re-raised on the main thread
            failure.append(exc)

    worker = threading.Thread(target=drain)
    worker.start()
    worker.join()

    assert not failure, failure
    assert [first.text, *rest] == [f"row {index}" for index in range(5)]


def test_reading_a_database_does_not_write_to_it(tmp_path: Path) -> None:
    """The ordinary case -- a database in SQLite's default rollback-journal mode.

    Nothing changes and nothing is left beside it. A write-ahead-log database is
    the case where that is not the whole story; the next test covers it.
    """
    path = db(
        tmp_path / "posts.db",
        ["CREATE TABLE posts (body TEXT)"],
        {"posts": [("hello there",)]},
    )
    before = path.read_bytes()

    documents, _ = ingest(path)

    assert [d.text for d in documents] == ["hello there"]
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["posts.db"]


def test_a_database_with_an_unmerged_log_is_read_in_full_and_left_alone(
    tmp_path: Path,
) -> None:
    """What ``mode=ro`` is for, and what ``immutable=1`` would have cost.

    A database whose newest rows are still in its ``-wal`` sidecar, which is how
    one looks while any program has it open. Measured, on a copy of that pair:

    - Opened read-write, SQLite checkpoints the log into the main file on close
      and **the user's database is rewritten** -- a different sha256 for a file
      TrainAI was only asked to read.
    - Opened ``immutable=1``, the log is skipped, so the rows in it are invisible
      and the corpus silently comes out short.
    - Opened ``mode=ro``, every row is there and the main file is untouched.

    The one cost is what this test also pins: a read-only connection cannot clean
    up the ``-shm`` and ``-wal`` it needs to read a log, so those are left behind.
    """
    origin = db(
        tmp_path / "origin.db",
        ["PRAGMA journal_mode=WAL", "CREATE TABLE posts (body TEXT)"],
    )
    # Holding the writer open is what stops SQLite merging the log on close.
    writer = sqlite3.connect(origin)
    try:
        writer.executemany(
            "INSERT INTO posts VALUES (?)", [(f"row {index}",) for index in range(200)]
        )
        writer.commit()
        log = Path(str(origin) + "-wal")
        assert log.stat().st_size > 0, "the point is rows that are not in the main file yet"

        path = tmp_path / "posts.db"
        path.write_bytes(origin.read_bytes())
        Path(str(path) + "-wal").write_bytes(log.read_bytes())
        before = path.read_bytes()

        documents, _ = ingest(path)

        assert [d.text for d in documents] == [f"row {index}" for index in range(200)]
        assert path.read_bytes() == before
        assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("posts.db")) == [
            "posts.db",
            "posts.db-shm",
            "posts.db-wal",
        ]
    finally:
        writer.close()


def test_the_supported_formats_sentence_names_databases() -> None:
    sentence = describe_supported_formats()

    assert ".sqlite" in sentence
    assert ".db" in sentence


def test_a_plain_text_directory_gains_no_database_counters(tmp_path: Path) -> None:
    """The negative control: a guard accidentally inverted fails here, loudly."""
    write(tmp_path / "corpus.txt", "prose and nothing else, long enough to keep\n")

    documents, stats = ingest(tmp_path)

    assert len(documents) == 1
    assert stats.resolved_db_tables == {}
    assert stats.resolved_db_columns == {}
    assert stats.unrecognized_files == []
    assert stats.records_skipped_malformed == 0


# --------------------------------------------------------------------------- #
# Interpreters built without an optional codec
# --------------------------------------------------------------------------- #
# ``bz2``, ``lzma`` and ``sqlite3`` each need a system library present when CPython
# was compiled. A Python built without one still ships the wrapper module, whose
# import then fails on its missing C accelerator -- so these are absent in the field
# rather than in theory: "No module named '_lzma'" is the best-known pyenv install
# failure, and some slim container images ship without any of the three.
def _block_modules(names: set[str]) -> str:
    """A program preamble that makes ``names`` unimportable, as such a build does."""
    return (
        "import builtins, sys\n"
        f"BLOCKED = {names!r}\n"
        "for name in list(sys.modules):\n"
        "    if name.split('.')[0] in BLOCKED or name.lstrip('_') in BLOCKED:\n"
        "        del sys.modules[name]\n"
        "real = builtins.__import__\n"
        "def guarded(name, *args, **kwargs):\n"
        "    if name.lstrip('_') in BLOCKED:\n"
        "        raise ImportError('No module named ' + repr(name))\n"
        "    return real(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded\n"
    )


def test_the_package_imports_without_any_optional_codec(tmp_path: Path) -> None:
    """The regression that mattered: a plain .txt corpus on a stripped interpreter.

    Run in a subprocess because the bug was in the *import* of
    :mod:`trainai.data.ingest`, which this process has already done -- monkeypatching
    the module attributes cannot see it. This is the only test that goes red if the
    guarded imports at the top of ingest.py become plain ones again, which is what
    makes them a guard rather than decoration.
    """
    write(tmp_path / "corpus.txt", "prose that has nothing to do with any codec\n" * 3)
    program = _block_modules({"bz2", "lzma", "sqlite3"}) + (
        "from trainai.data import IngestOptions, Ingestor\n"
        "ingestor = Ingestor(IngestOptions())\n"
        f"sources = ingestor.discover({str(tmp_path)!r})\n"
        "print(len(list(ingestor.documents(sources))))\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == "1"


@pytest.mark.parametrize(
    ("module", "library", "suffix", "compress"),
    [
        ("lzma", "liblzma", ".xz", lzma.compress),
        ("bz2", "libbz2", ".bz2", bz2.compress),
    ],
)
def test_a_corpus_needing_an_absent_codec_names_the_library(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module: str,
    library: str,
    suffix: str,
    compress: Any,
) -> None:
    """Naming the library is the point: "no module named _lzma" is unactionable."""
    (tmp_path / f"corpus.txt{suffix}").write_bytes(compress(b"prose behind a codec\n"))
    monkeypatch.setattr(ingest_module, module, None)

    with pytest.raises(DatasetFormatError) as caught:
        ingest(tmp_path)

    assert module in str(caught.value)
    assert library in (caught.value.hint or "")
    assert caught.value.details["library"] == library


def test_a_database_on_an_interpreter_without_sqlite3_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "corpus.db"
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE posts (body TEXT)")
        connection.execute("INSERT INTO posts VALUES ('a row of prose')")
        connection.commit()
    finally:
        connection.close()
    monkeypatch.setattr(ingest_module, "sqlite3", None)

    with pytest.raises(DatasetFormatError) as caught:
        ingest(path)

    assert "sqlite3" in str(caught.value)
    assert caught.value.details["library"] == "libsqlite3"


def test_an_absent_codec_does_not_stop_an_uncompressed_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The codec is consulted per file that needs it, not once per run."""
    write(tmp_path / "corpus.txt", "plain prose, no codec involved\n" * 3)
    for module in ("bz2", "lzma", "sqlite3"):
        monkeypatch.setattr(ingest_module, module, None)

    documents, stats = ingest(tmp_path)

    assert stats.files_read == 1
    assert len(documents) == 1


# --------------------------------------------------------------------------- #
# Compressed corpora that are corrupt or truncated
# --------------------------------------------------------------------------- #
# A decompressing stream does not fail when it is opened. It fails part-way through
# being read, so a handler wrapped around ``open`` never sees it. Measured before the
# fix: every codec crossed with every reader kind escaped as a traceback -- twelve
# combinations out of twelve, including plain ``.gz``, which predates the bz2/xz work.
def _truncate(path: Path) -> None:
    blob = path.read_bytes()
    path.write_bytes(blob[: len(blob) // 2])


def _scramble(path: Path) -> None:
    blob = path.read_bytes()
    path.write_bytes(blob[:8] + b"\x00" * 30 + blob[38:])


@pytest.mark.parametrize(
    ("suffix", "compress"),
    [(".gz", gzip.compress), (".bz2", bz2.compress), (".xz", lzma.compress)],
)
@pytest.mark.parametrize("damage", [_truncate, _scramble], ids=["truncated", "scrambled"])
def test_a_corrupt_compressed_corpus_is_a_clean_error(
    tmp_path: Path, suffix: str, compress: Any, damage: Any
) -> None:
    """``lzma.LZMAError`` and ``zlib.error`` are not OSErrors, so the obvious
    ``except OSError`` misses them and the traceback reaches the user."""
    path = tmp_path / f"corpus.txt{suffix}"
    path.write_bytes(compress(b"prose that is long enough to matter. " * 200))
    damage(path)

    with pytest.raises(DatasetFormatError) as caught:
        ingest(tmp_path)

    expected = {".gz": "gzip", ".bz2": "bz2", ".xz": "lzma"}[suffix]
    assert "corpus.txt" in str(caught.value)
    assert "downloaded completely" in (caught.value.hint or "")
    assert caught.value.details["compression"] == expected


def test_a_truncated_corpus_is_reported_as_truncated(tmp_path: Path) -> None:
    """Truncation is the likely case -- an interrupted download -- so it is named."""
    path = tmp_path / "corpus.txt.gz"
    path.write_bytes(gzip.compress(b"prose that is long enough to matter. " * 200))
    _truncate(path)

    with pytest.raises(DatasetFormatError) as caught:
        ingest(tmp_path)

    assert caught.value.details["truncated"] is True
    assert "ended before the compressed data did" in str(caught.value)


def test_a_bad_member_inside_a_valid_zip_is_a_clean_error(tmp_path: Path) -> None:
    """The zip opens and lists its contents fine; only reading the member fails."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("corpus.txt", "prose that is long enough to matter. " * 200)
    raw = bytearray(buffer.getvalue())
    for offset in range(40, 70):
        raw[offset] ^= 0xFF
    (tmp_path / "corpus.zip").write_bytes(bytes(raw))

    with pytest.raises(DatasetFormatError) as caught:
        ingest(tmp_path)

    assert caught.value.details["archive"] == "zip"


def test_a_corrupt_compressed_corpus_is_fatal_even_when_skipping(tmp_path: Path) -> None:
    """The deliberate half of the fix.

    By the time the stream fails, documents from earlier in the file have already
    been yielded and counted, so skipping the remainder would train on however much
    of the corpus happened to arrive and report success. That is the silent-wrong-
    corpus failure this module exists to prevent, and every other corrupt container
    here -- a bad tar, a bad .docx, a shredded database -- is already fatal.
    """
    path = tmp_path / "corpus.txt.gz"
    path.write_bytes(gzip.compress(b"prose that is long enough to matter. " * 200))
    _truncate(path)

    with pytest.raises(DatasetFormatError):
        ingest(tmp_path, on_error="skip")


def test_an_io_failure_on_a_plain_file_is_not_called_corruption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The negative control for the wrapper's scope.

    An uncompressed read is deliberately left unwrapped: nothing is being
    decompressed, so an OSError there is a real I/O failure, and describing it to the
    user as a corrupt corpus would be a lie. Widening the wrapper to every source
    turns this red.
    """
    write(tmp_path / "corpus.txt", "prose\n" * 5)
    real_open = Path.open

    def failing(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.name == "corpus.txt":
            raise OSError("simulated disk failure")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing)

    with pytest.raises(OSError, match="simulated disk failure"):
        ingest(tmp_path)


# --------------------------------------------------------------------------- #
# Which kinds --max-doc-chars can cut
#
# CUT_BY_MAX_DOC_CHARS is read outside this module: the empty-split errors in
# trainai.data.binarize use it to decide whether lowering the limit is advice worth
# giving. It used to be given unconditionally, and on a one-row CSV it was inert --
# measured, `--max-doc-chars 2000` came back reporting the same 1 document and the
# same refusal. So the set has to stay true, and the only way to know it is true is
# to lower the limit on one corpus per kind and count.
# --------------------------------------------------------------------------- #
#: Long enough that a 500-character limit must cut it several times over.
CUTTABLE_TEXT = "This is a sentence about the weather in a small town. " * 80


def one_long_record(root: Path, kind: str) -> Path:
    """A corpus of exactly one document of :data:`CUTTABLE_TEXT`, in the given format.

    Field and column names are ones the readers resolve on their own, so no format
    needs an option here and every kind is exercised through the same call.
    """
    root.mkdir(parents=True, exist_ok=True)
    if kind == "text":
        write(root / "corpus.txt", CUTTABLE_TEXT)
    elif kind == "jsonl":
        jsonl(root / "corpus.jsonl", [CUTTABLE_TEXT])
    elif kind == "json":
        write(root / "corpus.json", json.dumps([{"text": CUTTABLE_TEXT}]))
    elif kind == "csv":
        with (root / "corpus.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["text"])
            writer.writerow([CUTTABLE_TEXT])
    elif kind == "docx":
        docx(root / "corpus.docx", para(CUTTABLE_TEXT))
    elif kind == "sqlite":
        db(root / "corpus.db", ["CREATE TABLE rows (text TEXT)"], {"rows": [(CUTTABLE_TEXT,)]})
    else:  # pragma: no cover - the parametrization below turns this into a failure
        pytest.fail(f"no corpus builder for the {kind!r} reader; CUT_BY_MAX_DOC_CHARS may be stale")
    return root


@pytest.mark.parametrize("kind", sorted(get_args(Kind)))
def test_max_doc_chars_cuts_exactly_the_kinds_that_say_they_are_cut(
    tmp_path: Path, kind: str
) -> None:
    """One corpus per kind, read twice, and the count decides -- not the reader's source.

    Parametrized over ``Kind`` itself so a new reader arrives here with no corpus
    builder and fails loudly, rather than quietly inheriting whichever answer
    :data:`CUT_BY_MAX_DOC_CHARS` happens to give it.
    """
    root = one_long_record(tmp_path / kind, kind)
    assert {source.kind for source in Ingestor().discover(root)} == {kind}, (
        "the corpus must exercise the kind it is named for"
    )

    whole, _ = ingest(root, max_doc_chars=1 << 14)
    lowered, _ = ingest(root, max_doc_chars=500)

    assert len(whole) == 1, "one record, one document, before the limit does anything"
    cut = len(lowered) > 1
    assert cut is (kind in CUT_BY_MAX_DOC_CHARS), (
        f"{kind}: lowering --max-doc-chars gave {len(lowered)} documents, but "
        f"CUT_BY_MAX_DOC_CHARS says it "
        f"{'can' if kind in CUT_BY_MAX_DOC_CHARS else 'cannot'} be cut"
    )
    if cut:
        assert "".join(document.text for document in lowered) == CUTTABLE_TEXT, (
            "cutting must divide the text, not sample it"
        )
