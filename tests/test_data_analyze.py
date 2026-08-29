"""Tests for :mod:`trainai.data.analyze` and :mod:`trainai.data.validate`.

``analyze`` measures and ``validate`` judges, and the split matters: a
measurement that quietly became a verdict, or a verdict with no measurement
behind it, is how a tool ends up telling users things that are not true.

So these tests check two families of claim:

* Every number the report exposes is the number it says it is -- the median is a
  length some document actually has, the estimate is labelled as an estimate.
* Every issue carries a code, a message, and a hint, and the code maps to the
  exception type the CLI will exit on.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest

from conftest import cli_command, cli_flag_owners, cli_long_flags
from trainai.data.analyze import CorpusMeter, DatasetReport, analyze_documents
from trainai.data.ingest import Document, Ingestor, IngestStats
from trainai.data.validate import (
    MIN_TRAINABLE_CHARS,
    MIN_VOCAB_SIZE,
    SMALL_CORPUS_CHARS,
    TABLE_MIN_LINES,
    TABLE_ROW_SHARE,
    ValidationResult,
    validate_corpus,
)
from trainai.errors import DatasetEmptyError, DatasetError, DatasetFormatError


def cps(*codepoints: int) -> str:
    """A string from explicit Unicode codepoints.

    Non-ASCII test data is spelled out this way rather than written as literal
    characters, so this file stays pure ASCII and no editor or terminal can
    re-encode the very input a measurement is being asserted about.
    """
    return "".join(chr(codepoint) for codepoint in codepoints)


#: "cafe" with an acute accent: four characters, five UTF-8 bytes.
CAFE = cps(0x0063, 0x0061, 0x0066, 0x00E9)
#: U+FFFD, the marker a failed decode leaves behind.
REPLACEMENT = cps(0xFFFD)
#: Cyrillic "abv".
CYRILLIC = cps(0x0430, 0x0431, 0x0432)


def docs(*texts: str) -> list[Document]:
    return [Document(text, "source.txt", index, 0) for index, text in enumerate(texts)]


def prose(chars: int) -> str:
    """Plain English text of about ``chars`` characters."""
    unit = "The harbour clock measured every tide that crossed the bridge. "
    return (unit * (chars // len(unit) + 1))[:chars]


def codes(result: ValidationResult) -> set[str]:
    return {issue.code for issue in result.issues}


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def test_counts_documents_and_characters() -> None:
    report = analyze_documents(docs("abc", "de"))

    assert report.documents == 2
    assert report.total_chars == 5
    assert report.total_utf8_bytes == 5


def test_utf8_bytes_exceed_chars_for_non_ascii() -> None:
    report = analyze_documents(docs(CAFE))

    assert report.total_chars == 4
    assert report.total_utf8_bytes == 5
    assert report.non_ascii_chars == 1
    assert report.non_ascii_ratio == pytest.approx(0.25)


def test_percentiles_name_lengths_a_document_actually_has() -> None:
    """Nearest-rank, not interpolated: "the median document is 412 chars" must be true."""
    lengths = [10, 20, 30, 40, 50]
    report = analyze_documents(docs(*["x" * n for n in lengths]))

    assert report.min_chars == 10
    assert report.max_chars == 50
    assert report.median_chars in lengths
    assert report.p10_chars in lengths
    assert report.p90_chars in lengths
    assert report.mean_chars == pytest.approx(30.0)


def test_duplicate_documents_are_counted_once_each() -> None:
    report = analyze_documents(docs("same text", "same text", "different"))

    assert report.documents == 3
    assert report.unique_documents == 2
    assert report.duplicate_documents == 1
    assert report.duplicate_rate == pytest.approx(1 / 3)
    assert not report.duplicate_check_truncated


def test_whitespace_and_control_characters_are_measured() -> None:
    report = analyze_documents(docs("a b\tc\n\x00\x01"))

    assert report.whitespace_chars == 3
    assert report.control_chars == 2
    assert report.replacement_chars == 0


def test_replacement_characters_are_counted_separately() -> None:
    report = analyze_documents(docs("lost bytes: " + REPLACEMENT * 2))

    assert report.replacement_chars == 2


def test_script_shares_are_ordered_largest_first() -> None:
    report = analyze_documents(docs("latin text " * 20 + CYRILLIC))

    shares = report.script_shares
    assert list(shares) == sorted(shares, key=lambda name: -shares[name])
    assert report.dominant_script == "latin"
    assert "cyrillic" in shares


def test_script_sample_size_is_reported_alongside_the_shares() -> None:
    report = analyze_documents(docs(prose(5000)))

    assert 0 < report.script_sample_chars <= 5000
    assert report.to_dict()["scripts"]["sampled_chars"] == report.script_sample_chars


def test_token_estimate_is_only_an_estimate() -> None:
    report = analyze_documents(docs(prose(4000)))

    assert report.rough_token_estimate == 1000
    assert "rough_token_estimate" in report.to_dict()


def test_largest_sources_are_ranked_by_size() -> None:
    documents = [
        Document("small", "small.txt", 0, 0),
        Document("x" * 100, "big.txt", 0, 0),
    ]
    report = analyze_documents(documents)

    assert report.sources == 2
    assert [entry["path"] for entry in report.largest_sources] == ["big.txt", "small.txt"]


def test_meter_yields_every_document_unchanged() -> None:
    """``data prepare`` measures and tokenizes from one stream, so this must be lossless."""
    original = docs("first", "second", "third")
    meter = CorpusMeter()

    passed_through = list(meter.measure(original))

    assert passed_through == original
    assert meter.report().documents == 3


def test_report_is_cached_so_reading_it_twice_is_free() -> None:
    meter = CorpusMeter()
    list(meter.measure(docs("some text")))

    assert meter.report() is meter.report()


def test_empty_corpus_reports_zeroes_rather_than_dividing_by_zero() -> None:
    report = analyze_documents([])

    assert report.documents == 0
    assert report.duplicate_rate == 0.0
    assert report.whitespace_ratio == 0.0
    assert report.non_ascii_ratio == 0.0
    assert report.script_shares == {}
    assert report.dominant_script is None
    assert report.to_dict()["documents"] == 0


def test_to_dict_is_json_safe() -> None:
    import json

    report = analyze_documents(docs(prose(2000), CAFE))

    assert json.loads(json.dumps(report.to_dict()))["documents"] == 2


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #
def test_no_documents_is_an_error_and_stops_further_checks() -> None:
    result = validate_corpus(analyze_documents([]))

    assert not result.ok
    assert codes(result) == {"empty_corpus"}


def test_tiny_corpus_is_an_error_that_names_the_size() -> None:
    report = analyze_documents(docs(prose(MIN_TRAINABLE_CHARS - 1)))

    result = validate_corpus(report)

    assert not result.ok
    assert "corpus_too_small" in codes(result)
    assert str(report.total_chars) in result.errors[0].message.replace(",", "")


def test_small_but_usable_corpus_warns_rather_than_fails() -> None:
    report = analyze_documents(docs(prose(MIN_TRAINABLE_CHARS * 5)))

    result = validate_corpus(report)

    assert result.ok
    assert "small_corpus" in codes(result)


def test_large_enough_corpus_is_not_flagged_as_small() -> None:
    report = analyze_documents(docs(*[prose(20_000) for _ in range(60)]))

    result = validate_corpus(report)

    assert result.ok
    assert "small_corpus" not in codes(result)
    assert report.total_chars > SMALL_CORPUS_CHARS


def test_vocabulary_too_large_for_the_corpus_suggests_a_smaller_one() -> None:
    report = analyze_documents(docs(prose(20_000)))

    result = validate_corpus(report, vocab_size=32_768)

    issue = next(i for i in result.issues if i.code == "vocab_too_large_for_corpus")
    assert issue.level == "warning"
    assert "--vocab-size" in issue.hint
    assert issue.details["suggested_vocab_size"] < 32_768
    # `< 32_768` alone was satisfied by 256, which is what this suggestion used to be
    # and which the tokenizer refuses. The lower bound is the half that matters.
    assert issue.details["suggested_vocab_size"] >= MIN_VOCAB_SIZE


def test_the_suggested_vocabulary_floor_matches_the_tokenizer() -> None:
    """``validate`` duplicates the tokenizer's floor; the copy must not drift.

    The import is inside the test on purpose: :mod:`trainai.data.tokenizer` loads the
    Rust tokenizers library at module scope, and :mod:`trainai.data.validate` is on the
    ``trainai --help`` path, which ``test_conventions`` keeps free of it. So the two
    constants cannot be one constant, and this test is what stands in for that.
    """
    from trainai.data import tokenizer as tokenizer_module

    assert MIN_VOCAB_SIZE == tokenizer_module.MIN_VOCAB_SIZE
    assert MIN_VOCAB_SIZE == tokenizer_module.BYTE_ALPHABET_SIZE + 2


@pytest.mark.parametrize("total_chars", [1_100, 5_000, 20_000, 30_000, 60_000, 500_000])
def test_the_suggested_vocabulary_size_is_one_the_tokenizer_accepts(total_chars: int) -> None:
    """Follow the advice literally, on the same corpus that produced it.

    This is the test the bug needed. The suggestion was
    ``max(256, _round_to_power_of_two(chars // 100))``, and both halves capped at 256:
    the ``max`` for a corpus under 25,600 characters, and ``1 << 8`` for every value
    from 256 to 511. 256 is one below the tokenizer's floor, so on any corpus small
    enough to reach either cap the warning named a size that then failed with
    "vocab_size=256 is too small".
    """
    from trainai.data.tokenizer import train_tokenizer

    text = prose(total_chars)
    report = analyze_documents(docs(text))

    result = validate_corpus(report, vocab_size=32_768)

    issue = next(i for i in result.issues if i.code == "vocab_too_large_for_corpus")
    suggested = issue.details["suggested_vocab_size"]
    assert str(suggested) in issue.hint, f"hint does not name {suggested}: {issue.hint}"
    trained = train_tokenizer([text], vocab_size=suggested)
    assert trained.vocab_size > 256


@pytest.mark.parametrize("total_chars", [1, 50, 99])
def test_a_corpus_too_short_to_divide_still_gets_a_suggestion(total_chars: int) -> None:
    """Under 100 characters the supportable size floors to zero, and ``1 << -1`` raises.

    The corpus is refused for other reasons, but the issue list is built before anything
    is raised, so this arithmetic runs. A validator that crashes while explaining why a
    corpus is unusable is worse than the corpus.
    """
    report = analyze_documents(docs(prose(total_chars)))

    result = validate_corpus(report, vocab_size=32_768)

    assert not result.ok
    issue = next(i for i in result.issues if i.code == "vocab_too_large_for_corpus")
    assert issue.details["suggested_vocab_size"] == MIN_VOCAB_SIZE


def test_a_suggestion_that_cannot_clear_the_warning_says_so() -> None:
    """At the floor the warning is unavoidable, and pretending otherwise is the lie.

    A corpus under 25,800 characters cannot support even the smallest legal vocabulary
    at 100 characters per entry, so re-running with the suggested size warns again. The
    hint has to name the real problem -- the amount of text -- rather than imply the
    suggestion clears it.
    """
    report = analyze_documents(docs(prose(20_000)))

    issue = next(
        i
        for i in validate_corpus(report, vocab_size=32_768).issues
        if i.code == "vocab_too_large_for_corpus"
    )

    assert issue.details["suggested_vocab_size"] == MIN_VOCAB_SIZE
    assert "persist" in issue.hint
    assert "More text" in issue.hint
    # And the claim is true: taking the advice really does warn again.
    again = validate_corpus(report, vocab_size=issue.details["suggested_vocab_size"])
    assert "vocab_too_large_for_corpus" in codes(again)


def test_a_suggestion_that_does_clear_the_warning_does_not_claim_otherwise() -> None:
    """The negative control for the branch above, so an inverted test is caught."""
    text = prose(600_000)
    report = analyze_documents(docs(text))

    issue = next(
        i
        for i in validate_corpus(report, vocab_size=32_768).issues
        if i.code == "vocab_too_large_for_corpus"
    )

    assert issue.details["suggested_vocab_size"] > MIN_VOCAB_SIZE
    assert "persist" not in issue.hint
    cleared = validate_corpus(report, vocab_size=issue.details["suggested_vocab_size"])
    assert "vocab_too_large_for_corpus" not in codes(cleared)


def test_vocabulary_the_corpus_can_support_is_not_flagged() -> None:
    report = analyze_documents(docs(*[prose(20_000) for _ in range(60)]))

    result = validate_corpus(report, vocab_size=4096)

    assert "vocab_too_large_for_corpus" not in codes(result)


def test_binary_masquerading_as_text_is_rejected() -> None:
    payload = "".join(chr(byte % 32) for byte in range(4000))
    report = analyze_documents(docs(payload))

    result = validate_corpus(report)

    assert not result.ok
    assert any("control" in code or "binary" in code for code in codes(result)), codes(result)


def test_heavy_duplication_is_flagged() -> None:
    report = analyze_documents(docs(*[prose(2000)] * 20))

    result = validate_corpus(report)

    assert "heavy_duplication" in codes(result), codes(result)


def test_skipped_files_become_a_visible_issue() -> None:
    report = analyze_documents(docs(*[prose(20_000) for _ in range(60)]))
    stats = IngestStats(
        files_read=3,
        files_skipped=1,
        skipped_files=[{"path": "broken.txt", "reason": "could not be decoded"}],
    )

    result = validate_corpus(report, stats)

    issue = next(i for i in result.issues if "skip" in i.code)
    assert "broken.txt" in issue.message or "broken.txt" in str(issue.details)


def test_split_files_produce_a_note_naming_the_limit() -> None:
    report = analyze_documents(docs(*[prose(20_000) for _ in range(60)]))
    stats = IngestStats(files_read=1, long_files_split=1)

    result = validate_corpus(report, stats, max_doc_chars=16_384)

    issue = next(i for i in result.issues if i.code == "documents_were_split")
    assert issue.level == "note"
    assert "16,384" in issue.message


def test_csv_rows_with_extra_fields_become_a_warning() -> None:
    """A row wider than the header may have handed back a truncated document."""
    report = analyze_documents(docs(*[prose(20_000) for _ in range(60)]))
    stats = IngestStats(files_read=1, csv_rows_with_extra_fields=7)

    result = validate_corpus(report, stats)

    issue = next(i for i in result.issues if i.code == "csv_rows_with_extra_fields")
    assert issue.level == "warning"
    assert "7" in issue.message
    assert "unquoted" in issue.hint


def test_every_issue_carries_a_hint() -> None:
    """A ``TrainAIError`` without an actionable hint is treated as a bug here."""
    report = analyze_documents(docs(prose(2000), prose(2000), "x"))

    result = validate_corpus(report, IngestStats(files_skipped=1), vocab_size=32_768)

    assert result.issues
    for issue in result.issues:
        assert issue.hint.strip(), issue.code
        assert issue.message.strip(), issue.code
        assert issue.level in {"error", "warning", "note"}


def test_raise_if_failed_is_silent_when_there_are_only_warnings() -> None:
    report = analyze_documents(docs(prose(MIN_TRAINABLE_CHARS * 5)))

    validate_corpus(report).raise_if_failed()  # must not raise


def test_raise_if_failed_maps_the_code_to_a_specific_exception() -> None:
    result = validate_corpus(analyze_documents([]))

    with pytest.raises(DatasetEmptyError):
        result.raise_if_failed()


def test_raise_if_failed_carries_the_other_errors_along() -> None:
    """Three problems should take one run to discover, not three."""
    payload = "".join(chr(byte % 32) for byte in range(500))
    result = validate_corpus(analyze_documents(docs(payload)))

    assert len(result.errors) > 1, codes(result)
    with pytest.raises(DatasetError) as caught:
        result.raise_if_failed()
    assert caught.value.details["other_errors"]
    assert caught.value.details["code"] == result.errors[0].code


def test_to_dict_counts_errors_and_warnings() -> None:
    result = validate_corpus(analyze_documents(docs(prose(MIN_TRAINABLE_CHARS * 5))))

    payload = result.to_dict()
    assert payload["ok"] is True
    assert payload["errors"] == 0
    assert payload["warnings"] == len(result.warnings)
    assert len(payload["issues"]) == len(result.issues)


def test_the_shared_fixture_corpus_is_trainable(tmp_corpus: Path) -> None:
    """If the fixture corpus were rejected, every test built on it would be suspect."""
    ingestor = Ingestor()
    report = analyze_documents(ingestor.documents(ingestor.discover(tmp_corpus)))

    result = validate_corpus(report, ingestor.stats)

    assert result.ok, [issue.code for issue in result.errors]
    assert report.documents == 2
    assert report.total_chars > MIN_TRAINABLE_CHARS


# --------------------------------------------------------------------------- #
# rows pretending to be prose
#
# This detector exists because a 15.5 MiB weather CSV renamed to .txt prepared
# with no complaint at all, and the model trained on it emitted July temperatures
# for a January date. Its whole value is in the thresholds, so both sides are
# pinned: the corpora it must catch, and -- more important -- the ordinary ones
# it must leave alone. A threshold accidentally set to zero should fail a test,
# not quietly flag every corpus in the world.
# --------------------------------------------------------------------------- #
def measurements(rows: int = 400) -> str:
    """Rows of numbers: twelve comma-separated fields, mostly digits."""
    return "".join(
        f"2006-04-0{n % 9 + 1},{n % 24}:00,{n % 30}.4,{n % 21}.8,0.{n % 99:02d},"
        f"{n % 15}.9,{n % 360},{n % 16}.1,0.0,101{n % 9}.4,{n % 5},{n % 7}\n"
        for n in range(rows)
    )


def review_rows(rows: int = 400) -> str:
    """The other shape: regular rows whose last column is real prose."""
    return "".join(
        f"{n},{n % 5 + 1},The room was quiet and the harbour view made up for the stairs.\n"
        for n in range(rows)
    )


def measured(text: str) -> DatasetReport:
    return analyze_documents(docs(text))


def chunked(text: str, size: int = 2000) -> DatasetReport:
    """Measure ``text`` as the stream of documents real ingest would produce.

    Only the first 2,048 characters of each document are sampled, so measuring a
    long file as one document samples its opening and nothing else -- which would
    let a test about a table buried in prose pass without the table ever being
    looked at. Cutting at line boundaries, as ``_cut_point`` does, spreads the
    sample over the whole input.
    """
    pieces: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        current += line
        if len(current) >= size:
            pieces.append(current)
            current = ""
    if current:
        pieces.append(current)
    return analyze_documents(docs(*pieces))


def test_a_table_of_numbers_is_an_error() -> None:
    result = validate_corpus(measured(measurements()))

    assert "looks_like_a_table" in codes(result)
    assert not result.ok


def test_the_table_error_states_the_measurement_that_produced_it() -> None:
    """A verdict a user cannot check is a verdict they have to take on faith."""
    result = validate_corpus(measured(measurements()))

    issue = next(i for i in result.issues if i.code == "looks_like_a_table")
    assert "12 comma-separated fields" in issue.message
    assert issue.details["modal_fields"] == 12
    assert issue.details["row_share"] >= 0.80
    assert issue.details["digit_share"] >= 0.40


def test_a_table_of_numbers_exits_on_the_dataset_format_error() -> None:
    result = validate_corpus(measured(measurements()))

    with pytest.raises(DatasetFormatError):
        result.raise_if_failed()


def test_rows_whose_last_column_is_prose_warn_rather_than_fail() -> None:
    """The other user situation: the file is usable, one column at a time."""
    result = validate_corpus(measured(review_rows()))

    assert "rows_not_prose" in codes(result)
    assert "looks_like_a_table" not in codes(result)
    assert result.ok


def test_tab_separated_numbers_are_caught_too() -> None:
    text = "".join(f"{n}\t{n * 2}\t{n * 3}\t{n * 4}\n" for n in range(400))

    result = validate_corpus(measured(text))

    assert "looks_like_a_table" in codes(result)


def test_a_file_that_is_only_a_markdown_table_is_flagged() -> None:
    text = "| name | role | team |\n|---|---|---|\n" + "".join(
        f"| person {n} | engineer | platform |\n" for n in range(400)
    )

    result = validate_corpus(measured(text))

    assert "rows_not_prose" in codes(result)


def lined_prose(lines: int = 400) -> str:
    """Prose broken into lines, with commas, the way real writing has them.

    ``prose`` above is one unbroken line, and a delimiter measurement cannot say
    much about a corpus with one line in it. What this detector actually has to
    survive is prose that *does* carry several commas on many of its lines, which
    is most prose.

    Deliberately harsher than the real thing: two of the four sentences below
    carry exactly two commas, so this fixture measures 0.55 where the complete
    works of Shakespeare measure 0.15. That is the narrowest margin the threshold
    has anywhere, and it is worth knowing that the remaining gap is not the only
    thing standing between prose and a stopped run -- a false positive here is a
    warning, because the error also requires the text to be 40% digits.
    """
    units = (
        "The harbour clock measured every tide, and the bridge kept its own time.\n",
        "She wrote it down, twice, before the ferry left again.\n",
        "Nothing moved on the water.\n",
        "By morning the fog had taken the far bank, the cranes, and the light.\n",
    )
    return "".join(units[n % len(units)] for n in range(lines))


def test_prose_is_not_flagged() -> None:
    """The false-positive gate. Measured on tinyshakespeare: about 15%."""
    report = chunked(lined_prose(4_000))

    assert report.sampled_lines > TABLE_MIN_LINES
    assert report.table_row_share < TABLE_ROW_SHARE
    assert not ({"looks_like_a_table", "rows_not_prose"} & codes(validate_corpus(report)))


def test_prose_containing_a_markdown_table_is_not_flagged() -> None:
    """Documentation is full of tables. Only a file that is *mostly* one counts.

    This is the case the share is weighted by length for. The paragraphs here are
    unwrapped, so each contributes a single line to the sample while the table
    contributes twenty-two; counted by line the file is 96% table and gets
    stopped, counted by character it is under 1%, which is what it is.
    """
    table = "| flag | meaning | default |\n|---|---|---|\n" + "".join(
        f"| --flag-{n} | does a thing | off |\n" for n in range(20)
    )
    text = prose(40_000) + "\n\n" + table + "\n\n" + prose(40_000)

    report = chunked(text)

    assert report.table_row_share < TABLE_ROW_SHARE
    assert not ({"looks_like_a_table", "rows_not_prose"} & codes(validate_corpus(report)))


def test_the_row_share_is_weighted_by_length_not_by_line_count() -> None:
    """Twenty short rows do not outvote one long paragraph.

    Pinned on its own, because the whole finding hangs on it and the change that
    breaks it is a plausible simplification: counting one per line instead of one
    per character makes the table below 20 lines of 21, or 95%, and reports a
    paragraph with an option table under it as a table.
    """
    table = "".join(f"| a{n} | b | c |\n" for n in range(20))

    report = analyze_documents(docs(table + prose(1_500)))

    assert report.sampled_lines == 21
    assert report.table_modal_fields == 5
    assert report.table_row_share < 0.25


def test_source_code_is_not_flagged() -> None:
    """Python is full of commas. Measured at about 10%, nowhere near the threshold."""
    report = chunked(Path(__file__).read_text(encoding="utf-8") * 4)

    assert report.table_row_share < TABLE_ROW_SHARE
    assert not ({"looks_like_a_table", "rows_not_prose"} & codes(validate_corpus(report)))


def test_a_handful_of_regular_lines_is_a_coincidence_not_a_table() -> None:
    """Five identical lines are 100% regular and mean nothing. TABLE_MIN_LINES."""
    text = "a,b,c\n" * 5 + prose(MIN_TRAINABLE_CHARS * 2)

    report = measured(text)

    assert report.sampled_lines < TABLE_MIN_LINES or report.table_row_share < TABLE_ROW_SHARE
    assert "looks_like_a_table" not in codes(validate_corpus(report))


def test_allow_tabular_downgrades_the_finding_but_does_not_delete_it() -> None:
    """The override must leave a trace: findings are recorded in the manifest."""
    result = validate_corpus(measured(measurements()), allow_tabular=True)

    assert "tabular_override" in codes(result)
    assert "looks_like_a_table" not in codes(result)
    assert result.ok


def test_allow_tabular_does_not_invent_a_finding_on_a_normal_corpus() -> None:
    result = validate_corpus(measured(prose(SMALL_CORPUS_CHARS * 2)), allow_tabular=True)

    assert "tabular_override" not in codes(result)


def test_the_advice_names_the_rename_when_the_table_is_not_a_csv() -> None:
    """A .txt file gets told to rename it; --csv-text-column is inert on a .txt."""
    report = measured(measurements())
    report.largest_sources = [{"path": "weather.txt", "chars": report.total_chars}]

    issue = next(i for i in validate_corpus(report).issues if i.code == "looks_like_a_table")

    assert "rename" in (issue.hint or "").lower()


def test_the_advice_names_the_flag_when_the_table_is_already_a_csv() -> None:
    report = measured(measurements())
    report.largest_sources = [{"path": "weather.csv", "chars": report.total_chars}]

    issue = next(i for i in validate_corpus(report).issues if i.code == "looks_like_a_table")

    assert "--csv-text-column" in (issue.hint or "")


def test_the_advice_names_the_flag_when_the_table_came_from_a_database() -> None:
    """``--csv-text-column`` names the text column of a table wherever it came from.

    Telling the owner of a ``.db`` to "rename it back to .csv" is advice for a
    file they do not have.
    """
    report = measured(measurements())
    report.largest_sources = [{"path": "weather.db", "chars": report.total_chars}]

    issue = next(i for i in validate_corpus(report).issues if i.code == "looks_like_a_table")

    assert "--csv-text-column" in (issue.hint or "")
    assert "rename" not in (issue.hint or "").lower()


# --------------------------------------------------------------------------- #
# vocabulary saturation
#
# The second, cheaper signal for the same problem, and the one that catches a
# corpus of rows that is too *large* for the CHARS_PER_VOCAB_ENTRY check to
# notice. Twelve megabytes of weather readings built 698 of 8,192 tokens.
# --------------------------------------------------------------------------- #
def test_a_tokenizer_that_ran_out_of_text_is_a_warning() -> None:
    report = analyze_documents(docs(prose(SMALL_CORPUS_CHARS * 2)))

    result = validate_corpus(report, vocab_size=700, requested_vocab_size=8192)

    assert "vocabulary_saturated" in codes(result)
    assert result.ok


def test_the_saturation_warning_states_both_numbers() -> None:
    report = analyze_documents(docs(prose(SMALL_CORPUS_CHARS * 2)))

    result = validate_corpus(report, vocab_size=698, requested_vocab_size=8192)

    issue = next(i for i in result.issues if i.code == "vocabulary_saturated")
    assert "698" in issue.message
    assert "8,192" in issue.message


def test_a_mild_shortfall_is_left_to_the_tokenizer_section() -> None:
    """`data prepare` already prints the shortfall; a finding would be noise."""
    report = analyze_documents(docs(prose(SMALL_CORPUS_CHARS * 2)))

    result = validate_corpus(report, vocab_size=7000, requested_vocab_size=8192)

    assert "vocabulary_saturated" not in codes(result)


def test_a_vocabulary_that_filled_completely_is_not_flagged() -> None:
    report = analyze_documents(docs(prose(SMALL_CORPUS_CHARS * 2)))

    result = validate_corpus(report, vocab_size=8192, requested_vocab_size=8192)

    assert "vocabulary_saturated" not in codes(result)


def test_saturation_is_silent_when_the_requested_size_is_unknown() -> None:
    """Callers that do not pass it get today's behaviour, not a guess."""
    report = analyze_documents(docs(prose(SMALL_CORPUS_CHARS * 2)))

    result = validate_corpus(report, vocab_size=300)

    assert "vocabulary_saturated" not in codes(result)


def test_an_ordinary_corpus_produces_none_of_the_new_findings() -> None:
    """The negative control: a corpus that should sail through, does.

    Deliberately asserted as a set difference rather than one code at a time, so
    that a future finding added with a broken threshold fails here too.
    """
    report = analyze_documents(docs(prose(SMALL_CORPUS_CHARS * 2)))

    result = validate_corpus(
        report,
        vocab_size=8192,
        requested_vocab_size=8192,
        allow_tabular=False,
    )

    new = {"looks_like_a_table", "rows_not_prose", "tabular_override", "vocabulary_saturated"}
    assert not (codes(result) & new)


# --------------------------------------------------------------------------- #
# advice that can be followed
#
# `validate.py` runs under two commands with different flags -- `trainai data
# inspect` and `trainai data prepare` -- and its hints name flags. Three of them
# named a flag the command printing it does not have: `likely_binary` offered
# --path, `looks_like_a_table` offered --allow-tabular, and `mixed_scripts`
# offered --vocab-size. Followed from `data inspect`, each exits 2 with "No such
# option"; --path exists only on `trainai doctor`, where it prints a hardware
# report.
#
# Grepping the repo for the spellings found all three, because all three are real
# somewhere. So this checks against the click commands themselves, and it drives
# the real `validate_corpus` rather than reading the module's text, because
# several hints are assembled at runtime from helper functions.
# --------------------------------------------------------------------------- #
INSPECT = "data inspect"
PREPARE = "data prepare"


def _leaf(*names: str) -> Any:
    """One CLI command by path, e.g. ``_leaf("data", "inspect")``."""
    return cli_command(*names)


def _long_flags(command: Any) -> set[str]:
    """Every long option a command accepts, click's own answer.

    Includes the negative half of a boolean flag, which click keeps in
    ``secondary_opts``, and the help option, which it adds at parse time rather than
    as a declared parameter -- both were missing from a first version of this, which
    then reported four real spellings as nonexistent. Shared with the repo-wide check
    in ``test_conventions``, so the two cannot drift apart and disagree about what
    the CLI accepts.
    """
    return cli_long_flags(command)


def _flag_owners() -> dict[str, set[str]]:
    """Which commands accept each long flag, keyed as ``"data prepare"``."""
    return cli_flag_owners()


def report_with(**fields: Any) -> DatasetReport:
    """A report with exactly the measurements a case needs.

    Built by field rather than by measuring a corpus: the point here is to reach
    every finding, and several of them need combinations no single fixture produces.
    """
    return DatasetReport(**fields)


#: A corpus that trips nothing on its own, for cases that are about the ingest
#: counters or a single ratio rather than about size.
PLAIN = {"documents": 5, "total_chars": 10_000, "median_chars": 2_000}

#: (command, label, report, stats, validate kwargs). The kwargs are the ones that
#: command really passes: `_inspect_corpus` passes no vocabulary and never overrides
#: the tabular check, `run_prepare` passes both. That difference is the whole reason
#: this test exists, so it is spelled out per case rather than defaulted.
ADVICE_CASES: list[tuple[str, str, DatasetReport, IngestStats | None, dict[str, Any]]] = [
    (INSPECT, "an empty corpus", report_with(documents=0), None, {}),
    (
        INSPECT,
        "a corpus too small to train on",
        report_with(documents=1, total_chars=500, median_chars=500),
        None,
        {},
    ),
    (
        INSPECT,
        "a binary file among the text",
        report_with(
            **PLAIN,
            control_chars=1_000,
            largest_sources=[{"path": "junk.txt", "bytes": 4_096}],
        ),
        None,
        {},
    ),
    (
        INSPECT,
        "stray control codes, short duplicated documents, lost bytes",
        report_with(
            documents=10,
            total_chars=10_000,
            median_chars=10,
            control_chars=50,
            whitespace_chars=8_000,
            duplicate_documents=6,
            duplicate_check_truncated=True,
            replacement_chars=5,
        ),
        None,
        {},
    ),
    (
        INSPECT,
        "some duplication but not much",
        report_with(documents=10, total_chars=10_000, median_chars=1_000, duplicate_documents=3),
        None,
        {},
    ),
    (
        INSPECT,
        "three writing systems in near-equal share",
        report_with(**PLAIN, script_counts={"latin": 40, "cyrillic": 35, "greek": 25}),
        None,
        {},
    ),
    (
        INSPECT,
        "one non-Latin writing system",
        report_with(**PLAIN, script_counts={"cyrillic": 100}),
        None,
        {},
    ),
    (
        INSPECT,
        "everything ingest passed over",
        report_with(**PLAIN),
        IngestStats(
            files_read=4,
            files_skipped=2,
            skipped_files=[{"path": "broken.csv", "error": "decode failed"}],
            records_skipped_malformed=3,
            csv_rows_with_extra_fields=4,
            documents_skipped_short=5,
            long_files_split=1,
            archive_members_ignored=2,
            ignored_archive_members=["papers.zip::a.pdf", "papers.zip::b.bin"],
            nested_archives_ignored=1,
            ignored_nested_archives=["papers.zip::inner.zip"],
            files_unrecognized=3,
            files_without_extension=1,
            unrecognized_files=["notes.pdf", "sheet.xlsx", "LICENSE"],
            assumed_text_files=["corpus"],
        ),
        {"max_doc_chars": 16_384},
    ),
    (
        INSPECT,
        "a table of measurements renamed to .txt",
        report_with(
            **PLAIN,
            script_counts={"digits": 70, "latin": 30},
            table_delimiter=",",
            table_modal_fields=12,
            table_row_share=0.95,
            sampled_lines=40,
            largest_sources=[{"path": "weather.txt", "bytes": 4_096}],
        ),
        None,
        {},
    ),
    (
        INSPECT,
        "a .csv whose prose sits in one column",
        report_with(
            **PLAIN,
            script_counts={"latin": 80, "digits": 20},
            table_delimiter=",",
            table_modal_fields=6,
            table_row_share=0.95,
            sampled_lines=40,
            largest_sources=[{"path": "reviews.csv", "bytes": 4_096}],
        ),
        None,
        {},
    ),
    (
        PREPARE,
        "a table prepared anyway",
        report_with(
            **PLAIN,
            script_counts={"digits": 70, "latin": 30},
            table_delimiter=",",
            table_modal_fields=12,
            table_row_share=0.95,
            sampled_lines=40,
            largest_sources=[{"path": "weather.csv", "bytes": 4_096}],
        ),
        None,
        {"vocab_size": 4096, "requested_vocab_size": 4096, "allow_tabular": True},
    ),
    (
        PREPARE,
        "a vocabulary the corpus cannot support, at the tokenizer's floor",
        report_with(**PLAIN),
        None,
        {"vocab_size": 32_768, "requested_vocab_size": 32_768},
    ),
    (
        PREPARE,
        "a vocabulary the corpus cannot support, above the floor",
        report_with(documents=5, total_chars=100_000, median_chars=20_000),
        None,
        {"vocab_size": 32_768, "requested_vocab_size": 32_768},
    ),
    (
        PREPARE,
        "a tokenizer that ran out of distinct text",
        report_with(documents=50, total_chars=SMALL_CORPUS_CHARS * 2, median_chars=40_000),
        None,
        {"vocab_size": 300, "requested_vocab_size": 8192},
    ),
]


def _findings(case: tuple[str, str, DatasetReport, IngestStats | None, dict[str, Any]]):
    _command, _label, report, stats, kwargs = case
    return validate_corpus(report, stats, **kwargs).issues


@pytest.mark.parametrize("case", ADVICE_CASES, ids=[c[1] for c in ADVICE_CASES])
def test_no_finding_names_a_flag_its_own_command_lacks(
    case: tuple[str, str, DatasetReport, IngestStats | None, dict[str, Any]],
) -> None:
    """Advice printed by a command must be advice that command can be given.

    A flag the command lacks is allowed only if the finding says whose it is, which
    is what makes "pass --allow-tabular to `trainai data prepare`" acceptable from
    `data inspect` and a bare "pass --allow-tabular" not.
    """
    command, label, *_ = case
    accepted = _long_flags(_leaf(*command.split()))
    owners = _flag_owners()
    issues = _findings(case)

    assert issues, f"the case {label!r} reached no finding at all"

    for issue in issues:
        text = f"{issue.message} {issue.hint}"
        for flag in sorted(set(re.findall(r"--[a-z][a-z0-9-]+", text))):
            if flag in accepted:
                continue
            elsewhere = sorted(owners.get(flag, ()))
            # The command has to be named as an invocation, not as a word. A first
            # version looked for the bare owner name and a mutation slipped --force
            # past it, because `small_corpus` says "It will still train" and `train`
            # is a command.
            attributed = [owner for owner in elsewhere if owner and f"trainai {owner}" in text]
            assert attributed, (
                f"`trainai {command}` prints {issue.code} advising {flag}, which it does "
                f"not accept and does not attribute to a command that does. Following it "
                f"exits 2 with 'No such option'. "
                f"{f'The flag belongs to: {elsewhere}.' if elsewhere else 'No command has it.'} "
                f"The finding said: {text}"
            )


def test_the_advice_check_reaches_every_finding_validate_can_produce() -> None:
    """Coverage, so the check above cannot silently stop applying.

    The list of findings is read out of the module's syntax tree rather than kept by
    hand here, so a finding added with a flag in its hint and no case above fails
    this test instead of going unchecked.
    """
    tree = ast.parse(Path("src/trainai/data/validate.py").read_text(encoding="utf-8"))
    declared = {
        node.args[1].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ValidationIssue"
        and len(node.args) > 1
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    }
    assert len(declared) > 20, f"the syntax walk found only {declared}, so it is not working"

    reached = {issue.code for case in ADVICE_CASES for issue in _findings(case)}

    assert declared - reached == set(), (
        f"no case in ADVICE_CASES reaches {sorted(declared - reached)}, so the flags "
        "those findings name are unchecked"
    )
    assert reached - declared == set(), (
        f"{sorted(reached - declared)} was produced but is not a literal in the module -- "
        "the syntax walk needs updating"
    )
