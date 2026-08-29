"""Measure a corpus.

This module answers "what did the user actually give us?" -- size, document
length distribution, how much is duplicated, which writing systems appear, and
whether it looks like text at all. :mod:`trainai.data.validate` turns those
measurements into a verdict; this module only measures.

Three numbers are explicit about their basis, because a figure presented as exact
when it is not is worse than no figure:

* ``rough_token_estimate`` is named as an estimate. The real token count is known
  only after tokenization, and :mod:`trainai.data.binarize` records it.
* Duplicate detection is by 64-bit hash, so it can in principle report a
  collision as a duplicate. At one million documents that chance is about three
  in a hundred thousand. ``duplicate_check_truncated`` says when the hash set hit
  its memory cap and stopped growing.
* ``script_sample_chars`` reports how much text the writing-system breakdown was
  computed over. Scanning every character of a multi-gigabyte corpus costs
  minutes for a number nobody needs to four decimal places, so a bounded,
  deterministic sample is used: the head of every document, breadth first.
* ``row_sample_chars`` is the basis of the row-regularity measurement, which
  shares that same sample, and ``sampled_lines`` says how many lines that was.
  A share computed over eleven lines means nothing, so both are reported
  alongside it rather than left implicit.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from trainai.data.ingest import Document

__all__ = ["CorpusMeter", "DatasetReport", "analyze_documents"]

# Characters the script breakdown is computed over, and how much of any one
# document contributes. Breadth beats depth: a thousand documents at 2 KiB each
# describes a corpus better than one document at 2 MiB.
_SCRIPT_SAMPLE_CHAR_BUDGET = 8 << 20
_SCRIPT_SAMPLE_PER_DOC = 2048

# Cap on the duplicate-detection hash set. Two million entries is roughly 130 MiB
# of Python set overhead, which is the most this check is worth spending.
_DEDUP_CAP = 2_000_000

# Used only for the clearly-labelled pre-tokenization estimate. English text
# under a byte-level BPE lands near this; other languages do not, which is
# exactly why the field name says "rough".
_ROUGH_CHARS_PER_TOKEN = 4.0

_WHITESPACE_RE = re.compile(r"\s")
# Control characters excluding tab, newline and carriage return. A high count
# here usually means a binary file wearing a .txt extension.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# U+FFFD, the replacement character. Its presence means someone already lost
# bytes to a bad decode upstream of TrainAI.
_REPLACEMENT_RE = re.compile(re.escape(chr(0xFFFD)))

# Writing-system buckets, given as explicit codepoint ranges rather than as
# literal characters in a regex. Three reasons: the hex is checkable against a
# Unicode chart, this file stays pure ASCII (so no editor or terminal can mangle
# it), and a wrong range is visible in review. This is not the real Unicode script
# property -- the standard library ships no such table -- but it is enough to tell
# a user "this corpus is 12% Han" so they can reason about vocabulary size.
_SCRIPT_RANGES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    (
        "latin",
        (
            (0x0041, 0x005A),  # A-Z
            (0x0061, 0x007A),  # a-z
            (0x00C0, 0x024F),  # Latin-1 Supplement .. Latin Extended-B
            (0x1E00, 0x1EFF),  # Latin Extended Additional
        ),
    ),
    ("greek", ((0x0370, 0x03FF), (0x1F00, 0x1FFF))),
    ("cyrillic", ((0x0400, 0x052F),)),
    ("hebrew", ((0x0590, 0x05FF),)),
    ("arabic", ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF))),
    ("devanagari", ((0x0900, 0x097F),)),
    ("bengali", ((0x0980, 0x09FF),)),
    ("thai", ((0x0E00, 0x0E7F),)),
    (
        "han",
        (
            (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
            (0x4E00, 0x9FFF),  # CJK Unified Ideographs
            (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
        ),
    ),
    ("kana", ((0x3040, 0x30FF), (0x31F0, 0x31FF))),
    ("hangul", ((0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF))),
    ("digits", ((0x0030, 0x0039),)),
    ("emoji_symbols", ((0x2600, 0x27BF), (0x1F000, 0x1FAFF))),
)


def _char_class(ranges: tuple[tuple[int, int], ...]) -> re.Pattern[str]:
    body = "".join(f"{re.escape(chr(lo))}-{re.escape(chr(hi))}" for lo, hi in ranges)
    return re.compile(f"[{body}]")


_SCRIPT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, _char_class(ranges)) for name, ranges in _SCRIPT_RANGES
)

# Delimiters the row-regularity measurement considers. Prose contains commas
# irregularly; a table contains the same number on every line. That difference is
# what tells them apart, and measuring it beats trusting the file extension --
# a CSV renamed to .txt is precisely the case worth catching.
_TABLE_DELIMITERS = (",", "\t", ";", "|")

# A line needs this many delimiters (so three or more fields) before it counts
# toward a delimiter's tally. Ordinary writing with one comma per line therefore
# cannot score at all, which is what keeps prose out of the measurement.
_TABLE_MIN_DELIMITERS = 2


@dataclass
class DatasetReport:
    """What the corpus is. Every field is measured, not assumed."""

    documents: int = 0
    total_chars: int = 0
    total_utf8_bytes: int = 0

    min_chars: int = 0
    max_chars: int = 0
    mean_chars: float = 0.0
    p10_chars: int = 0
    median_chars: int = 0
    p90_chars: int = 0

    whitespace_chars: int = 0
    non_ascii_chars: int = 0
    control_chars: int = 0
    replacement_chars: int = 0

    unique_documents: int = 0
    duplicate_documents: int = 0
    duplicate_check_truncated: bool = False

    script_counts: dict[str, int] = field(default_factory=dict)
    script_sample_chars: int = 0

    # Row regularity, measured over the same sample as the scripts above.
    # ``table_row_share`` is the fraction of sampled *characters* sitting on
    # lines that carry exactly ``table_modal_fields`` fields separated by
    # ``table_delimiter``.
    #
    # Characters rather than lines, because lines are not equal-sized units and
    # weighting them equally answers the wrong question. A file of long unwrapped
    # paragraphs with a small table pasted into it has a handful of very long
    # lines and twenty short ones; counted by line it is 95% table, counted by
    # character it is 5%. The second number is the true one, and it is also the
    # one that matters -- what is being asked is how much of the text the model
    # will train on is row layout.
    table_delimiter: str | None = None
    table_modal_fields: int = 0
    table_row_share: float = 0.0
    sampled_lines: int = 0
    row_sample_chars: int = 0

    sources: int = 0
    largest_sources: list[dict[str, Any]] = field(default_factory=list)

    @property
    def rough_token_estimate(self) -> int:
        """Order-of-magnitude token count, before any tokenizer exists.

        Used only to decide whether the corpus is worth tokenizing at all and to
        size a progress bar. The exact count comes out of binarization.
        """
        return int(self.total_chars / _ROUGH_CHARS_PER_TOKEN)

    @property
    def duplicate_rate(self) -> float:
        return self.duplicate_documents / self.documents if self.documents else 0.0

    @property
    def whitespace_ratio(self) -> float:
        return self.whitespace_chars / self.total_chars if self.total_chars else 0.0

    @property
    def non_ascii_ratio(self) -> float:
        return self.non_ascii_chars / self.total_chars if self.total_chars else 0.0

    @property
    def control_ratio(self) -> float:
        return self.control_chars / self.total_chars if self.total_chars else 0.0

    @property
    def script_shares(self) -> dict[str, float]:
        """Script counts as fractions of the sampled characters, largest first."""
        total = sum(self.script_counts.values())
        if not total:
            return {}
        ordered = sorted(self.script_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return {name: count / total for name, count in ordered if count}

    @property
    def dominant_script(self) -> str | None:
        return next(iter(self.script_shares), None)

    @property
    def digit_share(self) -> float:
        """Fraction of sampled characters that are ASCII digits.

        Named separately from the rest of the script breakdown because it is the
        one bucket used as evidence rather than as information: text that is
        mostly digits is measurement data, and a language model models tokens as
        unrelated symbols, so it cannot learn that 9.47 and 9.48 are close.
        """
        return self.script_shares.get("digits", 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "total_chars": self.total_chars,
            "total_utf8_bytes": self.total_utf8_bytes,
            "rough_token_estimate": self.rough_token_estimate,
            "length_chars": {
                "min": self.min_chars,
                "p10": self.p10_chars,
                "median": self.median_chars,
                "p90": self.p90_chars,
                "max": self.max_chars,
                "mean": round(self.mean_chars, 1),
            },
            "character_mix": {
                "whitespace_ratio": round(self.whitespace_ratio, 4),
                "non_ascii_ratio": round(self.non_ascii_ratio, 4),
                "control_ratio": round(self.control_ratio, 6),
                "replacement_chars": self.replacement_chars,
            },
            "duplicates": {
                "unique_documents": self.unique_documents,
                "duplicate_documents": self.duplicate_documents,
                "rate": round(self.duplicate_rate, 4),
                "truncated": self.duplicate_check_truncated,
            },
            "scripts": {
                "shares": {k: round(v, 4) for k, v in self.script_shares.items()},
                "sampled_chars": self.script_sample_chars,
            },
            "rows": {
                "delimiter": self.table_delimiter,
                "modal_fields": self.table_modal_fields,
                "row_share": round(self.table_row_share, 4),
                "sampled_lines": self.sampled_lines,
                "sampled_chars": self.row_sample_chars,
            },
            "sources": self.sources,
            "largest_sources": self.largest_sources,
        }


class CorpusMeter:
    """Accumulates measurements over a document stream.

    Exposed rather than hidden because ``data prepare`` measures and tokenizes in
    a single pass over the files: it wraps the ingest stream in :meth:`measure`,
    hands the result to the tokenizer trainer, and reads :meth:`report` when the
    tokenizer has finished consuming it. Reading the corpus twice would double
    the cost of the slowest stage of preparation.
    """

    def __init__(self) -> None:
        self._lengths: list[int] = []
        self._total_chars = 0
        self._utf8_bytes = 0
        self._whitespace = 0
        self._non_ascii = 0
        self._control = 0
        self._replacement = 0
        self._hashes: set[int] = set()
        self._duplicates = 0
        self._dedup_truncated = False
        self._script_counts: dict[str, int] = dict.fromkeys((n for n, _ in _SCRIPT_PATTERNS), 0)
        self._script_sampled = 0
        self._table_counts: dict[str, Counter[int]] = {d: Counter() for d in _TABLE_DELIMITERS}
        self._sampled_lines = 0
        self._row_sample_chars = 0
        self._per_source_chars: dict[str, int] = {}
        self._report: DatasetReport | None = None

    def measure(self, documents: Iterable[Document]) -> Iterator[Document]:
        """Yield every document onward, measuring each as it passes."""
        for document in documents:
            self.add(document)
            yield document

    def add(self, document: Document) -> None:
        text = document.text
        length = len(text)

        self._lengths.append(length)
        self._total_chars += length
        self._utf8_bytes += len(text.encode("utf-8", errors="ignore"))
        self._whitespace += _count(_WHITESPACE_RE, text)
        # Encoding to ASCII with errors="ignore" drops every non-ASCII character,
        # so the length of the result *is* the ASCII character count -- and the
        # scan happens in C rather than in a Python loop over ord(c).
        self._non_ascii += length - len(text.encode("ascii", errors="ignore"))
        self._control += _count(_CONTROL_RE, text)
        self._replacement += _count(_REPLACEMENT_RE, text)
        self._per_source_chars[document.source] = (
            self._per_source_chars.get(document.source, 0) + length
        )

        digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).digest()
        key = int.from_bytes(digest, "big")
        if key in self._hashes:
            self._duplicates += 1
        elif len(self._hashes) < _DEDUP_CAP:
            self._hashes.add(key)
        else:
            self._dedup_truncated = True

        if self._script_sampled < _SCRIPT_SAMPLE_CHAR_BUDGET:
            sample = text[:_SCRIPT_SAMPLE_PER_DOC]
            self._script_sampled += len(sample)
            for name, pattern in _SCRIPT_PATTERNS:
                self._script_counts[name] += _count(pattern, sample)
            self._measure_rows(sample, truncated=len(sample) < length)

    def _measure_rows(self, sample: str, *, truncated: bool) -> None:
        """Tally delimiter regularity over the script sample, weighted by length.

        Each line contributes its own length rather than a vote, so the result
        answers "how much of this text is rows?" instead of "how many of its
        lines are?". Those differ by a lot when line lengths do: only the first
        2,048 characters of a document are sampled, so a document that is one
        unwrapped paragraph offers a single line, and twenty short table lines
        elsewhere in the corpus then outvote it twenty to one. By length the same
        table is a few hundred characters against a few thousand, which is what
        it is -- and length is the right unit anyway, because it is characters
        that become the tokens the model trains on.

        Only whole lines are measured, since a line cut in half has half the
        delimiters and would read as an irregular row in an otherwise perfectly
        regular table. A truncated sample is therefore trimmed back to its last
        line break rather than having its final line dropped: dropping it would
        also drop the only line of a one-line document, which is exactly the
        shape a single CSV column produces.

        A sample holding no line break at all is inside a line longer than the
        sample budget. That is not a table row, whatever else it is, so its
        characters count toward the denominator without being measured.
        """
        if truncated:
            end = sample.rfind("\n")
            if end < 0:
                self._sampled_lines += 1
                self._row_sample_chars += len(sample)
                return
            sample = sample[: end + 1]

        for line in sample.splitlines():
            if not line.strip():
                continue
            width = len(line)
            self._sampled_lines += 1
            self._row_sample_chars += width
            for delimiter in _TABLE_DELIMITERS:
                found = line.count(delimiter)
                if found >= _TABLE_MIN_DELIMITERS:
                    self._table_counts[delimiter][found] += width

    def report(self) -> DatasetReport:
        """Finalise and return the report. Cached, so calling it twice is free."""
        if self._report is not None:
            return self._report

        report = DatasetReport(
            documents=len(self._lengths),
            total_chars=self._total_chars,
            total_utf8_bytes=self._utf8_bytes,
            whitespace_chars=self._whitespace,
            non_ascii_chars=self._non_ascii,
            control_chars=self._control,
            replacement_chars=self._replacement,
            unique_documents=len(self._hashes),
            duplicate_documents=self._duplicates,
            duplicate_check_truncated=self._dedup_truncated,
            script_counts=dict(self._script_counts),
            script_sample_chars=self._script_sampled,
            sampled_lines=self._sampled_lines,
            row_sample_chars=self._row_sample_chars,
            sources=len(self._per_source_chars),
            largest_sources=_largest_sources(self._per_source_chars),
        )

        delimiter, modal, share = _strongest_row_shape(self._table_counts, self._row_sample_chars)
        report.table_delimiter = delimiter
        report.table_modal_fields = modal
        report.table_row_share = share

        if report.documents:
            lengths = np.sort(np.asarray(self._lengths, dtype=np.int64))
            report.min_chars = int(lengths[0])
            report.max_chars = int(lengths[-1])
            report.mean_chars = self._total_chars / report.documents
            report.p10_chars = _percentile(lengths, 0.10)
            report.median_chars = _percentile(lengths, 0.50)
            report.p90_chars = _percentile(lengths, 0.90)

        self._report = report
        return report


def analyze_documents(documents: Iterable[Document]) -> DatasetReport:
    """Consume a document stream and return its report."""
    meter = CorpusMeter()
    for document in documents:
        meter.add(document)
    return meter.report()


def _count(pattern: re.Pattern[str], text: str) -> int:
    """Number of matches. The scan is in C; the transient list is per-document."""
    return len(pattern.findall(text))


def _percentile(ordered: np.ndarray, q: float) -> int:
    """Nearest-rank percentile of an already-sorted array.

    Nearest-rank rather than interpolated, because "the median document is 412
    characters" should name a length some document actually has.
    """
    if ordered.size == 0:  # pragma: no cover - guarded by the caller
        return 0
    index = min(ordered.size - 1, max(0, round(q * (ordered.size - 1))))
    return int(ordered[index])


def _largest_sources(per_source: dict[str, int], limit: int = 5) -> list[dict[str, Any]]:
    ordered = sorted(per_source.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"path": path, "chars": chars} for path, chars in ordered[:limit]]


def _strongest_row_shape(
    tallies: dict[str, Counter[int]], sampled_chars: int
) -> tuple[str | None, int, float]:
    """The most regular delimiter, its modal field count, and that mode's share.

    The tallies hold characters, not lines, so "modal" here means the field count
    that accounts for the most text rather than the one appearing on the most
    lines. Returns field counts, not delimiter counts: eleven commas make twelve
    fields, and twelve is the number a person reading the message will recognise.

    Every tie -- between two delimiters with equal shares, or between two field
    counts accounting for equal amounts of text -- resolves the same way on every
    run, so the reported shape does not depend on iteration order.
    """
    if not sampled_chars:
        return None, 0, 0.0

    best: tuple[str | None, int, float] = (None, 0, 0.0)
    for delimiter in _TABLE_DELIMITERS:
        tally = tallies[delimiter]
        if not tally:
            continue
        # Largest share of text wins; on a tie, the smaller field count.
        found, chars = max(tally.items(), key=lambda kv: (kv[1], -kv[0]))
        share = chars / sampled_chars
        if share > best[2]:
            best = (delimiter, found + 1, share)
    return best
