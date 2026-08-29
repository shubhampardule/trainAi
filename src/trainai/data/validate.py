"""Decide whether a measured corpus can be trained on, and what is worrying.

The split between this module and :mod:`trainai.data.analyze` is deliberate:
analysis measures, validation judges. Judgement involves thresholds, thresholds
are arguable, and putting them in one file with the reasoning next to each number
makes them reviewable instead of scattered.

Three levels:

``error``
    Preparation stops. The corpus cannot produce a trainable dataset.
``warning``
    Preparation continues, but the result will probably disappoint. Surfaced
    before the user spends hours on it, not after.
``note``
    Something happened that the user should know about. Not a problem.

Every issue carries a ``hint`` naming a concrete next action, because "your
dataset has a high duplicate rate" without "here is what to do" is just an
unhelpful observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from trainai.data.analyze import DatasetReport
from trainai.data.ingest import CSV_SUFFIXES, SQLITE_SUFFIXES, IngestStats, strip_compression
from trainai.errors import DatasetEmptyError, DatasetError, DatasetFormatError

__all__ = ["Level", "ValidationIssue", "ValidationResult", "validate_corpus"]

Level = Literal["error", "warning", "note"]

# --------------------------------------------------------------------------- #
# Thresholds. Each one gets a sentence saying why it is where it is.
# --------------------------------------------------------------------------- #

# Below this there is no dataset. Roughly 250 tokens: not enough to fill one
# training sequence, let alone learn from. Chosen low enough that the tiny
# fixtures used by the test suite are still legal.
MIN_TRAINABLE_CHARS = 1_000

# Below roughly a megabyte, even a 2M-parameter model sees each token many times
# and will reproduce the corpus rather than generalise from it. Still trainable,
# and useful for a first run, so this warns rather than fails.
SMALL_CORPUS_CHARS = 1 << 20

# A byte-level BPE can always *build* a vocabulary this large, but merges learned
# from a handful of occurrences are noise. One hundred characters per vocabulary
# entry is a low bar that catches "32k vocab on a 200 KB file".
CHARS_PER_VOCAB_ENTRY = 100

# The smallest --vocab-size the tokenizer accepts: 256 byte values, the end-of-text
# token, and room for one merge. Duplicated from ``trainai.data.tokenizer.MIN_VOCAB_SIZE``
# rather than imported, because that module loads the Rust tokenizers library at module
# scope and this one is imported by ``trainai --help``, which a convention test keeps
# free of it. ``test_the_suggested_vocabulary_floor_matches_the_tokenizer`` cross-checks
# the two, so the copy cannot drift.
#
# It is a named constant because it was the bare literal ``256`` in the suggestion
# below, one short of the real floor -- so the warning told people to run
# ``--vocab-size 256``, which the tokenizer then refused.
MIN_VOCAB_SIZE = 258

# Control characters other than tab, newline and carriage return. Real prose has
# essentially none; a few per million happen. Percentages mean the file is not
# text, whatever its extension claims.
BINARY_CONTROL_RATIO = 0.05
SUSPICIOUS_CONTROL_RATIO = 0.001

# Above this, most of what the model sees is repeated text, which inflates the
# apparent token count and makes validation loss meaningless.
HEAVY_DUPLICATE_RATE = 0.50
NOTABLE_DUPLICATE_RATE = 0.20

# Prose runs 15-20% whitespace. Much higher means tables, logs, or indentation
# that dominates the token budget.
HIGH_WHITESPACE_RATIO = 0.60

# Documents shorter than this are mostly end-of-text markers by token count.
SHORT_DOCUMENT_CHARS = 32

# Below this share, no single writing system dominates, which has real
# consequences for how the vocabulary gets spent.
MIXED_SCRIPT_SHARE = 0.60

# Row regularity: the share of sampled text sitting on lines that all carry the
# same number of fields. Prose never puts the same number of commas on every
# line; a table always does.
#
# Measured through the real ingest path on this machine. Tables: a weather CSV
# renamed to .txt 1.00, a TSV of numbers 1.00, a CSV of reviews 1.00, a markdown
# file that is only a table 1.00, and the same weather data rendered as sentences
# 0.97. Legitimate corpora: the complete works of Shakespeare 0.15, this
# project's Python 0.10, its markdown with tables included 0.08, and a file of
# unwrapped paragraphs with a 22-line option table pasted into it 0.00. With
# nothing legitimate above 0.15 and nothing tabular below 0.97, the exact
# threshold hardly matters, which is the property to want in a check that can
# stop a run.
#
# The margin is thinner than that against prose built to be awkward: a fixture
# whose every other sentence carries exactly two commas reaches 0.55. Which is
# the reason the digit fork below exists rather than this threshold alone
# deciding -- prose that trips this check gets a warning, never a stopped run.
TABLE_ROW_SHARE = 0.80

# A share computed over a handful of lines says nothing, however lopsided it
# looks: a five-line file whose every line happens to hold two commas scores a
# perfect 1.00. Twenty is the smallest number of lines where this is a
# measurement rather than a coincidence.
TABLE_MIN_LINES = 20

# Digits as a share of sampled characters, which separates the two tabular cases.
# Measured: a raw weather CSV 0.66 and a TSV of numbers 1.00, against 0.24 for
# that weather data written as sentences, 0.14 for a markdown table of names and
# 0.11 for a CSV of reviews. Above this the columns are measurements and a
# language model is the wrong tool; below it, some column holds real prose.
TABLE_DIGIT_SHARE = 0.40

# A tokenizer that reaches only a fraction of the vocabulary it was asked for has
# run out of distinct text to learn merges from. Weather data rendered as
# sentences produced 698 of 8,192 requested -- 9% -- because 4.4M tokens of it
# are the same forty words and a lot of numbers. This catches templated and
# tabular corpora that the character-count check above lets through, because they
# are large; they are simply not varied.
VOCAB_SATURATION_SHARE = 0.25


@dataclass(frozen=True)
class ValidationIssue:
    """One finding. ``code`` is stable and safe to match on in scripts."""

    level: Level
    code: str
    message: str
    hint: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
            "details": self.details,
        }


@dataclass(frozen=True)
class ValidationResult:
    """The full verdict, in the order the checks ran."""

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.level == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.level == "warning")

    @property
    def notes(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.level == "note")

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        """Raise the first error as a :class:`~trainai.errors.DatasetError`.

        The remaining errors travel in ``details`` so that a user with three
        problems learns about all three from one run instead of three.
        """
        if self.ok:
            return
        first, *rest = self.errors
        exception = _EXCEPTION_FOR_CODE.get(first.code, DatasetError)
        raise exception(
            first.message,
            hint=first.hint,
            details={
                **first.details,
                "code": first.code,
                "other_errors": [i.to_dict() for i in rest],
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "issues": [i.to_dict() for i in self.issues],
        }


_EXCEPTION_FOR_CODE = {
    "empty_corpus": DatasetEmptyError,
    "corpus_too_small": DatasetEmptyError,
    "looks_like_a_table": DatasetFormatError,
}


def validate_corpus(
    report: DatasetReport,
    stats: IngestStats | None = None,
    *,
    vocab_size: int | None = None,
    max_doc_chars: int | None = None,
    allow_tabular: bool = False,
    requested_vocab_size: int | None = None,
) -> ValidationResult:
    """Judge a measured corpus.

    Args:
        report: Output of :func:`~trainai.data.analyze.analyze_documents`.
        stats: Ingest counters, so skipped files become visible issues.
        vocab_size: The tokenizer vocabulary that was actually built, if known.
        max_doc_chars: The ingest split limit, quoted in the note about files that
            were cut into several documents.
        allow_tabular: Downgrade the tabular-data error to a warning. The finding
            is kept either way, so a dataset prepared this way records it.
        requested_vocab_size: The vocabulary the user asked for. Compared against
            ``vocab_size`` to spot a corpus with too little distinct text to fill
            it, which is what a table looks like after tokenization.
    """
    issues: list[ValidationIssue] = []
    add = issues.append

    # -- hard failures ----------------------------------------------------- #
    if report.documents == 0:
        add(
            ValidationIssue(
                "error",
                "empty_corpus",
                "No usable documents were found.",
                "Check that the files contain text. Empty files, and documents shorter "
                "than --min-doc-chars, are dropped before this point.",
                {"documents": 0},
            )
        )
        # Every subsequent check divides by the document count or the character
        # count, so there is nothing further to say.
        return ValidationResult(tuple(issues))

    if report.total_chars < MIN_TRAINABLE_CHARS:
        add(
            ValidationIssue(
                "error",
                "corpus_too_small",
                f"The corpus is {report.total_chars:,} characters, which is too little "
                "to train any language model.",
                "A first experiment wants at least a megabyte of text. The complete works "
                "of Shakespeare is about 1.1 MB and is the usual starting point.",
                {"total_chars": report.total_chars, "minimum": MIN_TRAINABLE_CHARS},
            )
        )

    if report.control_ratio > BINARY_CONTROL_RATIO:
        add(
            ValidationIssue(
                "error",
                "likely_binary",
                f"{report.control_ratio:.1%} of the characters are control codes, so at "
                "least one of these files is not text.",
                # No flag here, because the corpus is a positional argument on both
                # commands that reach this check: `trainai data inspect PATH` and
                # `trainai data prepare CORPUS`. This said "point --path at a
                # directory", and neither accepts --path -- both exit 2 with "No such
                # option". `trainai doctor` does have one, so a user who went looking
                # for the flag found it and got a hardware report.
                "Remove the non-text files, or point TrainAI at a directory containing "
                "only the text you want to train on." + _largest_source_suffix(report),
                {
                    "control_ratio": report.control_ratio,
                    "control_chars": report.control_chars,
                    "largest_sources": report.largest_sources,
                },
            )
        )
    elif report.control_ratio > SUSPICIOUS_CONTROL_RATIO:
        add(
            ValidationIssue(
                "warning",
                "control_characters",
                f"{report.control_chars:,} control characters "
                f"({report.control_ratio:.2%}) are present.",
                "Usually harmless, but it can mean a stray binary file or a mangled "
                "export. Worth a look if the trained model produces odd characters.",
                {"control_chars": report.control_chars},
            )
        )

    # -- rows pretending to be prose --------------------------------------- #
    tabular = _tabular_issue(report, allow_tabular=allow_tabular)
    if tabular is not None:
        add(tabular)

    # -- size and shape ---------------------------------------------------- #
    if MIN_TRAINABLE_CHARS <= report.total_chars < SMALL_CORPUS_CHARS:
        add(
            ValidationIssue(
                "warning",
                "small_corpus",
                f"The corpus is {report.total_chars:,} characters, which is small.",
                "Expect the model to reproduce phrases from the training text rather "
                "than write new ones. It will still train; treat the output as a sign "
                "the pipeline works, not as a language model you can use.",
                {"total_chars": report.total_chars, "threshold": SMALL_CORPUS_CHARS},
            )
        )

    if vocab_size and report.total_chars < vocab_size * CHARS_PER_VOCAB_ENTRY:
        # No `max(256, ...)` wrapper any more: 256 is one below the tokenizer's floor,
        # so it turned this warning into advice the tokenizer then refused. The floor
        # lives in `_round_to_power_of_two`.
        supportable = report.total_chars // CHARS_PER_VOCAB_ENTRY
        affordable = _round_to_power_of_two(supportable)
        if affordable > supportable:
            # The floor was hit, so this warning will fire again at the suggested size.
            # Say so, rather than implying the suggestion clears it -- what is actually
            # wrong at this point is the amount of text, not the vocabulary.
            advice = (
                f"--vocab-size {affordable} is the smallest the tokenizer accepts, and "
                f"even that is more than {report.total_chars:,} characters supports, so "
                "this warning will persist. More text is the only real fix."
            )
        else:
            advice = f"Try --vocab-size {affordable}."
        add(
            ValidationIssue(
                "warning",
                "vocab_too_large_for_corpus",
                f"A vocabulary of {vocab_size:,} on {report.total_chars:,} characters "
                f"gives fewer than {CHARS_PER_VOCAB_ENTRY} characters per token.",
                f"{advice} A vocabulary larger than the corpus can "
                "support wastes parameters on embeddings for tokens the model barely "
                "sees, and those parameters come out of the layers that do the work.",
                {
                    "vocab_size": vocab_size,
                    "total_chars": report.total_chars,
                    "suggested_vocab_size": affordable,
                },
            )
        )

    saturation = _vocab_saturation_issue(vocab_size, requested_vocab_size)
    if saturation is not None:
        add(saturation)

    if report.median_chars < SHORT_DOCUMENT_CHARS:
        add(
            ValidationIssue(
                "warning",
                "very_short_documents",
                f"The median document is {report.median_chars} characters.",
                "Each document ends with an end-of-text token, so with documents this "
                "short a large share of training signal is 'a document ended here'. "
                "Consider joining related lines into longer documents.",
                {"median_chars": report.median_chars, "documents": report.documents},
            )
        )

    if report.whitespace_ratio > HIGH_WHITESPACE_RATIO:
        add(
            ValidationIssue(
                "warning",
                "mostly_whitespace",
                f"{report.whitespace_ratio:.0%} of the corpus is whitespace.",
                "Tables, logs and deeply indented code spend most of the token budget "
                "on layout. If that is not what you want the model to learn, strip the "
                "indentation first.",
                {"whitespace_ratio": report.whitespace_ratio},
            )
        )

    # -- duplication ------------------------------------------------------- #
    if report.duplicate_rate > HEAVY_DUPLICATE_RATE:
        add(
            ValidationIssue(
                "warning",
                "heavy_duplication",
                f"{report.duplicate_rate:.0%} of documents are exact duplicates of "
                "another document.",
                "Deduplicate before training. Duplicated text inflates the token count, "
                "so the run looks larger than it is, and it leaks into the validation "
                "split, so validation loss stops measuring generalisation.",
                {
                    "duplicate_rate": report.duplicate_rate,
                    "duplicate_documents": report.duplicate_documents,
                    "unique_documents": report.unique_documents,
                },
            )
        )
    elif report.duplicate_rate > NOTABLE_DUPLICATE_RATE:
        add(
            ValidationIssue(
                "note",
                "some_duplication",
                f"{report.duplicate_rate:.0%} of documents are exact duplicates.",
                "Not enough to be a problem by itself. Worth removing if validation "
                "loss looks better than the samples suggest it should.",
                {"duplicate_rate": report.duplicate_rate},
            )
        )

    if report.duplicate_check_truncated:
        add(
            ValidationIssue(
                "note",
                "duplicate_check_truncated",
                "Duplicate detection stopped tracking new documents after its memory "
                "cap, so the reported rate is a lower bound.",
                "No action needed. The real duplicate rate is at least the number shown.",
                {"unique_documents_tracked": report.unique_documents},
            )
        )

    # -- text quality ------------------------------------------------------ #
    if report.replacement_chars:
        add(
            ValidationIssue(
                "warning",
                "replacement_characters",
                f"{report.replacement_chars:,} Unicode replacement characters (U+FFFD) "
                "are already in the text.",
                "Something upstream of TrainAI decoded these files with the wrong "
                "encoding and lost the original bytes. If the source is still available, "
                "re-export it as UTF-8; TrainAI cannot recover what is already gone.",
                {"replacement_chars": report.replacement_chars},
            )
        )

    shares = report.script_shares
    dominant = report.dominant_script
    if dominant and shares.get(dominant, 0.0) < MIXED_SCRIPT_SHARE and len(shares) > 1:
        top = ", ".join(f"{name} {share:.0%}" for name, share in list(shares.items())[:4])
        add(
            ValidationIssue(
                "note",
                "mixed_scripts",
                f"No single writing system dominates: {top}.",
                # `trainai data inspect` produces this note, and it has neither the flag
                # nor the report: it exits 2 on --vocab-size and prints no tokenizer
                # numbers, because it does not build a tokenizer. So both are attributed
                # to the command that does have them.
                "Multilingual corpora need a larger vocabulary than monolingual ones to "
                "reach the same characters-per-token. `trainai data prepare` reports "
                "characters per token; if it is under three, raise its --vocab-size.",
                {"script_shares": {k: round(v, 4) for k, v in shares.items()}},
            )
        )
    elif dominant and dominant not in {"latin", "digits"}:
        add(
            ValidationIssue(
                "note",
                "non_latin_script",
                f"The corpus is mostly {dominant}.",
                "Byte-level BPE handles this correctly, but non-Latin scripts use more "
                "bytes per character, so expect fewer characters per token than the "
                "numbers usually quoted for English.",
                {"dominant_script": dominant, "share": round(shares[dominant], 4)},
            )
        )

    # -- what ingestion did ------------------------------------------------ #
    if stats is not None:
        issues.extend(_ingest_issues(stats, max_doc_chars))

    return ValidationResult(tuple(issues))


def _first_few(names: list[str], total: int, *, keep: int = 3) -> str:
    """``a, b, c, and 7 more`` -- names for a message, without a wall of them.

    ``total`` is the exact count, which may exceed what ``names`` holds: ingest
    caps the names it keeps but never the counting.
    """
    shown = names[:keep]
    rest = total - len(shown)
    listed = ", ".join(shown) or "(unnamed)"
    return f"{listed}, and {rest:,} more" if rest > 0 else listed


def _ingest_issues(stats: IngestStats, max_doc_chars: int | None = None) -> list[ValidationIssue]:
    found: list[ValidationIssue] = []

    if stats.files_skipped:
        found.append(
            ValidationIssue(
                "warning",
                "files_skipped",
                f"{stats.files_skipped} file(s) were skipped and contributed nothing.",
                "These were requested with --on-error skip. Drop that flag to see the "
                "exact failure for the first one.",
                {"files_skipped": stats.files_skipped, "skipped": stats.skipped_files[:10]},
            )
        )

    if stats.records_skipped_malformed:
        found.append(
            ValidationIssue(
                "warning",
                "records_skipped",
                f"{stats.records_skipped_malformed:,} malformed JSON record(s) were skipped.",
                "Drop --on-error skip to see the first bad record with its line number, "
                "or check that every line has the field named by --jsonl-field.",
                {"records_skipped": stats.records_skipped_malformed},
            )
        )

    if stats.csv_rows_with_extra_fields:
        found.append(
            ValidationIssue(
                "warning",
                "csv_rows_with_extra_fields",
                f"{stats.csv_rows_with_extra_fields:,} CSV row(s) held more fields than the "
                "header names.",
                "A trailing delimiter on every line is harmless, but a delimiter inside an "
                "unquoted field splits a value and the extracted text is only the part before "
                "it. Quote fields that contain the delimiter, or check the row count is what "
                "you expect.",
                {"csv_rows_with_extra_fields": stats.csv_rows_with_extra_fields},
            )
        )

    if stats.documents_skipped_short:
        found.append(
            ValidationIssue(
                "note",
                "short_documents_dropped",
                f"{stats.documents_skipped_short:,} document(s) were dropped for being "
                "shorter than --min-doc-chars.",
                "Lower --min-doc-chars to keep them, if short fragments are meaningful "
                "in your data.",
                {"documents_skipped_short": stats.documents_skipped_short},
            )
        )

    if stats.archive_members_ignored:
        # Named in the *message*, not only in the details. An archive is opaque --
        # the user cannot see what is inside it without listing it themselves -- so
        # a member dropped silently is a corpus quietly smaller than the one they
        # pointed at, with nothing on screen to say so.
        found.append(
            ValidationIssue(
                "note",
                "archive_members_ignored",
                f"{stats.archive_members_ignored:,} file(s) inside the archive were not a "
                "format TrainAI reads and were passed over: "
                f"{_first_few(stats.ignored_archive_members, stats.archive_members_ignored)}.",
                "Ignore this if those files were not meant to be part of the corpus. "
                "Otherwise convert them to a readable format -- the error from pointing "
                "TrainAI straight at one lists what those are.",
                {
                    "archive_members_ignored": stats.archive_members_ignored,
                    "members": stats.ignored_archive_members[:10],
                },
            )
        )

    if stats.nested_archives_ignored:
        found.append(
            ValidationIssue(
                "note",
                "nested_archives_ignored",
                f"{stats.nested_archives_ignored} archive(s) inside the archive were "
                "reported rather than opened: "
                f"{_first_few(stats.ignored_nested_archives, stats.nested_archives_ignored)}.",
                "TrainAI reads one level of archive, which is what stops a nested "
                "compression bomb. Unpack the inner one and point at the result to "
                "include it.",
                {
                    "nested_archives_ignored": stats.nested_archives_ignored,
                    "archives": stats.ignored_nested_archives[:10],
                },
            )
        )

    if stats.files_unrecognized:
        # A directory walk selects by extension, so what it passed over is invisible
        # unless it is said out loud. The hint splits because the fix does: a .pdf
        # has to be converted, while a file with no extension is one command away.
        bare = stats.files_without_extension
        named = stats.files_unrecognized - bare
        fix = ""
        if named:
            fix = (
                "Ignore this if they were not meant to be part of the corpus; otherwise "
                "convert them -- the error from pointing TrainAI straight at one lists "
                "which formats it reads. "
            )
        if bare:
            fix += (
                f"{bare:,} of them {'has' if bare == 1 else 'have'} no extension at all: "
                "point TrainAI straight at one and it is read as plain text whatever it "
                "is called."
            )
        found.append(
            ValidationIssue(
                "note",
                "files_unrecognized",
                f"{stats.files_unrecognized:,} file(s) in the corpus directory were not a "
                "format TrainAI reads and were passed over: "
                f"{_first_few(stats.unrecognized_files, stats.files_unrecognized)}.",
                fix.strip(),
                {
                    "files_unrecognized": stats.files_unrecognized,
                    "files_without_extension": bare,
                    "files": stats.unrecognized_files[:10],
                },
            )
        )

    if stats.assumed_text_files:
        found.append(
            ValidationIssue(
                "note",
                "assumed_text",
                f"{_first_few(stats.assumed_text_files, len(stats.assumed_text_files))} has no "
                "extension, and was read as plain text.",
                "TrainAI took the file being named as the instruction. Rename it .txt "
                "(or .jsonl, .csv) to have it selected inside a directory too.",
                {"files": stats.assumed_text_files},
            )
        )

    if stats.long_files_split:
        # Taken from the ingest counter rather than by comparing the longest
        # document against the limit: the splitter cuts at the last paragraph
        # break before the limit, so the longest document is always shorter than
        # --max-doc-chars and that comparison would never be true.
        limit = f" (--max-doc-chars {max_doc_chars:,})" if max_doc_chars else ""
        found.append(
            ValidationIssue(
                "note",
                "documents_were_split",
                f"{stats.long_files_split} file(s) were longer than the per-document "
                f"limit{limit} and were cut into several documents.",
                "Expected for corpora that are one large file. Cuts land on paragraph "
                "breaks, and the token stream the model trains on is unaffected.",
                {
                    "long_files_split": stats.long_files_split,
                    "max_doc_chars": max_doc_chars,
                },
            )
        )

    return found


_DELIMITER_NAMES = {",": "comma", "\t": "tab", ";": "semicolon", "|": "pipe"}


def _tabular_issue(report: DatasetReport, *, allow_tabular: bool) -> ValidationIssue | None:
    """The finding for a corpus whose lines are rows rather than sentences.

    Three outcomes, because the two ways a table reaches TrainAI need different
    advice and the override needs to leave a trace:

    * mostly digits -- ``looks_like_a_table``, an error. These are measurements,
      and there is no way to train a useful language model on measurements.
    * some column holds prose -- ``rows_not_prose``, a warning naming the flag
      that extracts it.
    * the error case with ``--allow-tabular`` -- ``tabular_override``, a warning,
      so the manifest of the resulting dataset records that the check was
      overruled rather than recording nothing at all.
    """
    if report.table_delimiter is None or report.sampled_lines < TABLE_MIN_LINES:
        return None
    if report.table_row_share < TABLE_ROW_SHARE:
        return None

    delimiter = _DELIMITER_NAMES.get(report.table_delimiter, repr(report.table_delimiter))
    digits = report.digit_share
    shape = (
        f"{report.table_row_share:.0%} of the text sits on lines holding exactly "
        f"{report.table_modal_fields} {delimiter}-separated fields"
    )
    details = {
        "delimiter": report.table_delimiter,
        "modal_fields": report.table_modal_fields,
        "row_share": round(report.table_row_share, 4),
        "digit_share": round(digits, 4),
        "sampled_lines": report.sampled_lines,
    }

    if digits < TABLE_DIGIT_SHARE:
        return ValidationIssue(
            "warning",
            "rows_not_prose",
            f"{shape}, so this corpus is rows rather than prose.",
            f"Only {digits:.0%} of the characters are digits, so one of those columns "
            "probably holds real text. Training on whole rows spends most of the token "
            f"budget on the column layout instead. {_column_advice(report)}",
            details,
        )

    measured = f"{shape}, and {digits:.0%} of the characters are digits."
    if allow_tabular:
        return ValidationIssue(
            "warning",
            "tabular_override",
            f"{measured} This is a table, and --allow-tabular was given, so preparation continued.",
            "Recorded in the dataset manifest, so the resulting dataset says it was "
            "prepared this way. Judge the model by reading its samples rather than by "
            "its validation loss: a table is highly predictable, so the loss will look "
            "excellent while the numbers it generates are wrong.",
            details,
        )

    return ValidationIssue(
        "error",
        "looks_like_a_table",
        f"{measured} This is a table of measurements, not text.",
        "A language model predicts the next token, and to a tokenizer 9.47 and 9.48 are "
        "unrelated strings -- it cannot learn that they are close numbers. Trained on "
        "this it produces rows in exactly the right shape with values that are "
        "confidently wrong.\n"
        f"  {_column_advice(report)}\n"
        "  If you want to predict one column from the others, that is regression rather "
        "than language modelling. scikit-learn's HistGradientBoostingRegressor does it "
        "in seconds on a CPU and reports its own error.\n"
        "  To turn the table into sentences yourself, see examples/weather_to_text.py, "
        "which shows the decisions that involves.\n"
        # The flag is named with its command: only `trainai data prepare` takes it, and
        # `trainai data inspect` raises this same error. `data inspect --allow-tabular`
        # exits 2 with "No such option", so an unqualified "pass --allow-tabular" was
        # advice half the callers could not follow.
        "  To train on it exactly as it is, pass --allow-tabular to "
        "`trainai data prepare`.",
        details,
    )


def _column_advice(report: DatasetReport) -> str:
    """How to reach a single column, which depends on what the files are called.

    Worth the branch: the commonest way a table reaches TrainAI is a CSV renamed
    to ``.txt``, and telling that user to "pass --csv-text-column" sends them to a
    flag that their file's extension makes inert. The rename is the missing step,
    so it is the one named.
    """
    if _has_tabular_suffix(report):
        return (
            "If one column holds real prose, train on that column alone with "
            "--csv-text-column NAME. (If you already named one, name a different one: "
            "the column you chose is itself rows.)"
        )
    return (
        "If this is a spreadsheet export that was renamed to .txt, rename it back: "
        "TrainAI reads .csv and .tsv directly, and --csv-text-column NAME then trains "
        "on a single column of it. Otherwise extract that column before preparing."
    )


def _has_tabular_suffix(report: DatasetReport) -> bool:
    """Whether the corpus is already in a format ``--csv-text-column`` applies to.

    A database counts, because the flag names the text column of a table wherever
    that table came from. Without this, a ``.db`` whose chosen column turned out to
    be rows of numbers would be told to "rename it back to .csv", which is advice
    for a file it is not.
    """
    for entry in report.largest_sources:
        name, _ = strip_compression(str(entry.get("path", "")).lower())
        if name.endswith(CSV_SUFFIXES) or name.endswith(SQLITE_SUFFIXES):
            return True
    return False


def _vocab_saturation_issue(achieved: int | None, requested: int | None) -> ValidationIssue | None:
    """A tokenizer that ran out of distinct text before filling its vocabulary.

    Distinct from ``vocab_too_large_for_corpus``, which compares the vocabulary
    against the corpus's *size*. This compares it against the corpus's *variety*,
    and so catches the large-but-repetitive corpus that the size check waves
    through -- 12M characters of weather readings is not a small corpus, it is a
    narrow one.

    Only the severe case is reported. A mild shortfall is already spelled out in
    the tokenizer section of ``data prepare``, and repeating it as a finding would
    be noise.
    """
    if not achieved or not requested or achieved >= requested:
        return None
    share = achieved / requested
    if share >= VOCAB_SATURATION_SHARE:
        return None
    return ValidationIssue(
        "warning",
        "vocabulary_saturated",
        f"The tokenizer could build only {achieved:,} of the {requested:,} tokens "
        f"requested ({share:.0%}), having run out of distinct text to merge.",
        f"Nothing is broken; the model will be sized for the {achieved:,} tokens that "
        "exist. But a corpus with this little variety is usually templated, logged or "
        "tabular, so check that it is what you meant to train on, and expect the model "
        "to reproduce it rather than generalise from it.",
        {
            "vocab_size": achieved,
            "requested_vocab_size": requested,
            "share": round(share, 4),
        },
    )


def _largest_source_suffix(report: DatasetReport) -> str:
    if not report.largest_sources:
        return ""
    names = ", ".join(entry["path"] for entry in report.largest_sources[:3])
    return f" The largest files are: {names}."


def _round_to_power_of_two(value: int) -> int:
    """Largest power of two not exceeding ``value``, floored at :data:`MIN_VOCAB_SIZE`.

    Vocabulary sizes are conventionally powers of two, and suggesting 3,847 as a
    replacement for 32,768 would look like a bug. The floor is the exception, and it
    is deliberately not a power of two: it used to be 256, which the tokenizer refuses,
    so every ``value`` below 512 -- not merely below 256 -- produced advice that could
    not be followed. ``max`` rather than an early return, because ``1 << 8`` is 256 for
    any ``value`` from 256 to 511.
    """
    if value <= 0:
        return MIN_VOCAB_SIZE
    return max(MIN_VOCAB_SIZE, 1 << (value.bit_length() - 1))
