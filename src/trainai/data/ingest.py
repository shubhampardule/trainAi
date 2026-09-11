"""Turn a path on disk into a stream of text documents.

Scope of this module: find the files, decode them, and hand back documents. It
knows nothing about tokenizers, tensors, or training. It imports neither torch
nor numpy, which is why its tests run in milliseconds.

Five decisions worth explaining, because each trades convenience for honesty:

**There is no ``errors="replace"`` mode.** A decode error handled by substituting
U+FFFD produces a dataset that trains without complaint and contains garbage.
The choice here is between ``fail`` (default) and ``skip``, and a skip is always
counted and reported. Silent corruption of training data is not on the menu.

**Long text files are cut into documents.** A single 500 MiB ``.txt`` cannot be
held as one Python string, encoded in one call, and kept as one list of token ids
on a machine with 6 GiB of free RAM. So text is emitted in pieces of at most
``max_doc_chars``, cut at the last paragraph break inside the window. This is
visible in the report rather than hidden.

**UTF-16 is refused for the line-oriented formats** -- JSON Lines, CSV and TSV.
Records are found by splitting on the newline *byte*, which is wrong for UTF-16
and UTF-32. Refusing with an instruction beats emitting plausible-looking rubbish.

**A table needs one column named, and will not guess.** A spreadsheet of numbers is
not a text corpus, and concatenating its columns into sentences would mean
inventing units, spelling out missing values and deciding what every field means
-- choices this module has no business making silently. So a CSV, or a table in a
SQLite database, yields one document per row from a single column, named with
``csv_text_column`` or resolved from a short list of conventional names, and
anything else is an error that lists the columns the file actually has. A database
adds one wrinkle to that: SQL defines no row order, so the order is pinned
explicitly here, and a view -- which has nothing to pin it to -- is refused.

**A format is read only where extraction is exact.** ``.docx`` qualifies: the
characters inside its paragraph elements are the characters Word displays. ``.pdf``
does not, because a PDF stores glyphs at positions and the reading order has to be
inferred -- two-column papers interleave, page furniture repeats, hyphenated words
split -- and it fails by producing plausible-looking wrong text rather than by
raising anything. That is the same looks-like-success failure the tabular detector
in :mod:`trainai.data.validate` exists to prevent, so it needs an extraction-quality
gate of its own rather than a reader bolted on here.
"""

from __future__ import annotations

import codecs
import csv
import difflib
import gzip
import io
import json
import tarfile
import zipfile
import zlib
from collections.abc import Callable, Iterable, Iterator
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Literal
from xml.etree import ElementTree

from trainai.data.chat import ChatFormatError, render_conversation
from trainai.errors import (
    DatasetDecodeError,
    DatasetFormatError,
    DatasetNotFoundError,
    UsageError,
)

# ``bz2``, ``lzma`` and ``sqlite3`` are optional at CPython *build* time. Each needs a
# system library present when the interpreter was compiled, and a Python built
# without one still ships the pure-Python wrapper, whose import then fails on the
# missing C accelerator. That is not a rare configuration: "No module named '_lzma'"
# is the best-known pyenv install failure, and some slim container images ship
# without any of the three.
#
# Importing them plainly made every such interpreter unable to import
# :mod:`trainai.data` at all -- measured, blocking any one of the three killed the
# whole package -- so ``trainai data prepare corpus.txt`` on a plain text file died
# with ``ImportError: No module named 'lzma'``: a message about a codec the user had
# not asked for, for a corpus that does not use it. Guarded here instead, and the
# reader that needs a missing one raises :func:`_missing_module` about the file.
#
# ``gzip`` and ``zlib`` are not guarded. zlib is required to build a working pip, so
# an interpreter without it could not have installed this package.
try:
    import bz2
except ImportError:  # pragma: no cover - a Python built without libbz2
    bz2 = None  # type: ignore[assignment]
try:
    import lzma
except ImportError:  # pragma: no cover - a Python built without liblzma
    lzma = None  # type: ignore[assignment]
try:
    import sqlite3
except ImportError:  # pragma: no cover - a Python built without libsqlite3
    sqlite3 = None  # type: ignore[assignment]

__all__ = [
    "ARCHIVE_SUFFIXES",
    "COMPRESSION_SUFFIXES",
    "CSV_SUFFIXES",
    "CUT_BY_MAX_DOC_CHARS",
    "DOCX_SUFFIXES",
    "JSONL_SUFFIXES",
    "JSON_SUFFIXES",
    "SQLITE_SUFFIXES",
    "TEXT_SUFFIXES",
    "Archive",
    "Compression",
    "Document",
    "IngestOptions",
    "IngestStats",
    "Ingestor",
    "Kind",
    "SourceFile",
    "describe_supported_formats",
    "strip_compression",
]

# Extensions read as one document per file. All of these are already text -- no
# markup is stripped, because stripping it means deciding what a heading or a code
# block is worth, and the model sees the characters either way. ``.log`` is here
# because a log file *is* text; that it usually makes a narrow corpus is caught
# after tokenization by ``vocabulary_saturated`` rather than guessed at here.
TEXT_SUFFIXES = (".txt", ".text", ".md", ".markdown", ".rst", ".org", ".log")
JSONL_SUFFIXES = (".jsonl", ".ndjson")
JSON_SUFFIXES = (".json",)
CSV_SUFFIXES = (".csv", ".tsv")
DOCX_SUFFIXES = (".docx",)

# SQLite databases. ``.sqlite3`` is here as well as ``.sqlite`` because it is the
# spelling Django's default ``db.sqlite3`` gave a generation of projects. ``.db``
# is the awkward one: it is generic enough that plenty of files carrying it are not
# databases at all -- Windows scatters ``Thumbs.db`` through picture directories --
# so the magic header is checked before a walk selects one. See
# :meth:`Ingestor._read_sqlite`.
SQLITE_SUFFIXES = (".sqlite", ".sqlite3", ".db")

# Which reader a file asks for. ``docx`` and ``sqlite`` are the two whose bytes are
# not already text, and both earn their place by being typed containers with real
# text in them -- paragraph elements, and a column of a table -- see
# :meth:`Ingestor._read_docx` for where that line is drawn, and why ``.pdf`` falls
# on the other side of it.
Kind = Literal["text", "jsonl", "json", "csv", "docx", "sqlite"]

#: Kinds whose documents ``max_doc_chars`` can cut into more of. Only two: a text file
#: is cut while it decodes (:meth:`Ingestor._read_text`) and a ``.docx`` is cut after it
#: is extracted (:meth:`Ingestor._emit_pieces`). For the other four, one document is one
#: record -- a JSON Lines line, a JSON element, a CSV row, a SQLite row -- and lowering
#: the limit cannot produce more of them.
#:
#: Exported because that distinction leaves this module: the empty-split errors in
#: :mod:`trainai.data.binarize` used to advise lowering ``--max-doc-chars`` whatever the
#: corpus was, and on a one-row CSV following that advice changed nothing at all
#: (measured: still "1 documents", same refusal). A test walks one corpus per kind and
#: asserts this set is exactly the set that splits, so adding a reader that cuts -- or
#: one that does not -- cannot leave the advice stale.
CUT_BY_MAX_DOC_CHARS: frozenset[str] = frozenset({"text", "docx"})

Compression = Literal["none", "gzip", "bz2", "lzma"]

# Codec per trailing suffix. Every opener here is a drop-in for ``Path.open("rb")``
# -- it returns a binary file that reads in blocks *and* iterates by line -- which
# is why the text, JSONL and CSV readers all work through compression without
# knowing it exists. Zstandard is deliberately absent: it reached the standard
# library in Python 3.14 and this project supports 3.10.
_COMPRESSION_BY_SUFFIX: dict[str, Compression] = {
    ".gz": "gzip",
    ".bz2": "bz2",
    ".xz": "lzma",
    ".lzma": "lzma",
}
COMPRESSION_SUFFIXES = tuple(_COMPRESSION_BY_SUFFIX)

# What the system library is called, for the error message. A user told "install
# liblzma and rebuild Python" can act; one told "no module named _lzma" cannot.
_CODEC_LIBRARIES: dict[str, str] = {
    "bz2": "libbz2",
    "lzma": "liblzma",
    "sqlite3": "libsqlite3",
}


def _missing_module(module: str) -> DatasetFormatError:
    """The error for a stdlib module this interpreter was built without."""
    library = _CODEC_LIBRARIES.get(module, module)
    return DatasetFormatError(
        f"this Python has no {module!r} module, so TrainAI cannot read that file.",
        hint=(
            f"{module!r} is part of the standard library but is built only when "
            f"{library} is available. Reinstall or rebuild Python with {library} "
            "present, or convert the corpus to a format that does not need it."
        ),
        details={"module": module, "library": library},
    )


def _codec_module(compression: Compression) -> Any:
    """The module behind a codec, or ``None`` if it is absent or none is needed.

    Deliberately an if-chain reading the module globals at call time rather than a
    table built at import time. A table would snapshot the module objects, which
    makes the guarded imports above no longer the single source of truth -- and
    leaves the absence of one impossible to simulate in a test.
    """
    if compression == "gzip":
        return gzip
    if compression == "bz2":
        return bz2
    if compression == "lzma":
        return lzma
    return None


def _compression_opener(compression: Compression) -> Callable[..., IO[bytes]] | None:
    """The codec's ``open``, or ``None`` for a source that is not compressed.

    Raises if the codec is the one this interpreter was built without, which is the
    only way a caller learns the difference between "nothing to decompress" and
    "cannot decompress".
    """
    if compression == "none":
        return None
    module = _codec_module(compression)
    if module is None:
        raise _missing_module(compression)
    opener: Callable[..., IO[bytes]] = module.open
    return opener


# What a decompressing stream raises when its bytes are not what it expects.
#
# Two of these are not ``OSError`` subclasses. ``zlib.error`` and ``lzma.LZMAError``
# both derive straight from ``Exception``, so the obvious handler -- ``except
# OSError`` -- catches gzip's header check and bz2's "Invalid data stream" and lets a
# corrupt ``.xz`` escape as a traceback. Measured across all three codecs and all
# four reader kinds: every combination escaped, including plain ``.gz``.
#
# ``EOFError`` is the truncation case, which every codec reports that way, and is the
# most likely of the lot in practice: it is what a half-finished download looks like.
_CORRUPT_STREAM_ERRORS: tuple[type[BaseException], ...] = tuple(
    error
    for error in (EOFError, OSError, zlib.error, getattr(lzma, "LZMAError", None))
    if error is not None
)


Archive = Literal["zip", "tar"]

# Container extensions, whose *members* are the corpus. A codec suffix on a tar is
# not stripped first: ``corpus.txt.gz`` is a compressed file, while ``corpus.tar.gz``
# is a container that happens to be compressed, and ``tarfile.open(mode="r:*")``
# decompresses it itself. So every tar spelling lands on one code path.
_ARCHIVE_BY_SUFFIX: dict[str, Archive] = {
    ".zip": "zip",
    ".tar": "tar",
    ".tgz": "tar",
    ".tbz": "tar",
    ".tbz2": "tar",
    ".txz": "tar",
}
ARCHIVE_SUFFIXES = (
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz",
    ".tbz2",
    ".tar.xz",
    ".txz",
)

# Separates an archive from a member in the path a source reports: the corpus
# ``papers.zip`` holding ``a/b.txt`` is reported as ``papers.zip::a/b.txt``. Two
# colons cannot appear in a Windows path, so nothing round-trips ambiguously.
_MEMBER_SEPARATOR = "::"

# Directory prefix macOS puts AppleDouble metadata under. Its contents are
# resource forks, not text, and every file inside it also starts with ``._`` --
# but naming the directory is clearer than relying on the dotfile rule.
_MACOS_METADATA_PREFIX = "__MACOSX/"

# Bounds on one archive, and it is worth being exact about what they protect.
# Nothing is ever extracted to disk, and every reader here is already bounded in
# memory -- text streams in blocks and emits bounded documents, ``.json`` has its
# own ceiling measured on the bytes that arrive -- so a compression bomb cannot
# exhaust memory through this path the way it could through a naive extractor.
# What an archive *can* do is make discovery itself expensive: a table of contents
# with a million entries becomes a million ``SourceFile`` objects before a single
# document is read, and a declared total in the petabytes turns the "Size on disk"
# line of the report into nonsense. Both of those are bounded here, from numbers
# available without decompressing anything.
#
# There is deliberately no per-member compression-ratio ceiling. Measured, gzip
# reaches 269:1 on legitimately repetitive JSON, so any threshold low enough to
# catch a bomb also refuses real corpora -- and a member that does expand hugely
# is read exactly as a large plain file is read, which is already safe. A guard
# that fires on real data while protecting against nothing is worse than none,
# because it advertises safety it does not provide. The real defence against the
# nested-bomb shape is below: archives inside archives are reported, never opened.
_ARCHIVE_MAX_MEMBERS = 20_000
_ARCHIVE_MAX_DECLARED_BYTES = 32 << 30

# How many passed-over names are kept for the report, whether they were members of
# an archive or files in a directory walk. The *counts* stay exact; only the names
# are capped, so a zip of 20,000 unreadable files -- or a directory of them -- does
# not put 20,000 strings into a manifest.
_MAX_NAMES_KEPT = 20

# Field names checked, in order, when a JSONL record's text field is not given
# explicitly. These cover the conventions used by the public text corpora people
# are most likely to already have on disk.
_TEXT_FIELD_CANDIDATES = ("text", "content", "body", "document", "raw_content")

# The same idea for CSV headers, matched case-insensitively because CSV headers
# are written by people. Deliberately a *separate* list from the JSONL one, and
# deliberately short: these are names that mean "this column is the text", not
# domain fields that happen to contain words. A weather CSV has a column called
# "Summary"; auto-selecting it would silently pick one column out of twelve and
# train on 96,000 repetitions of "Partly cloudy". Guessing wrong is worse than
# asking, so anything not on this list has to be named.
_CSV_COLUMN_CANDIDATES = (
    "text",
    "content",
    "body",
    "document",
    "review",
    "review_text",
    "message",
    "comment",
    "abstract",
    "sentence",
)

# Column separator per extension. Taken from the name rather than sniffed: a
# sniffer that guesses wrong yields one giant column and a dataset that looks
# fine, and the failure mode of getting it from the extension is an error message
# listing a single absurd column name, which tells the user exactly what happened.
_CSV_DELIMITERS = {".csv": ",", ".tsv": "\t"}

# Words for those separators, for error messages. A message that prints a tab
# character to say "tab" is a message nobody can read.
_DELIMITER_WORDS = {",": "comma", "\t": "tab", ";": "semicolon", "|": "pipe"}

# Ceiling on one CSV field, raised well above the stdlib's 128 KiB default so a
# genuinely long document in a cell still reads, but not removed: without a bound,
# a single unclosed quotation mark makes the rest of the file one field and the
# corpus silently becomes one document.
_CSV_MAX_FIELD_CHARS = 1 << 24

# Read granularity for text files. Large enough that syscall overhead is
# irrelevant, small enough that the decode buffer stays cheap.
_READ_BLOCK_BYTES = 1 << 20

# Ceiling on a whole-file ``.json``. Unlike every other format here, JSON has no
# record separator to stream on -- ``json.loads`` has to build the entire object
# graph before the first document exists -- so this is a bound on memory, not on
# disk. Measured with ``tracemalloc``: the parsed graph costs 1.1x the file size
# for an array of bare strings, 1.3x for records with one long text field, and
# 3.9x for wide records of many short fields, on top of the decoded text itself.
# At 128 MiB that is a few hundred MiB transient in the ordinary case, and the
# refusal above it points at JSON Lines, which streams with no ceiling at all.
# Enforced against the bytes that actually arrive, not only the size on disk: the
# size on disk of a ``.json.gz`` is not a bound on anything.
_JSON_MAX_BYTES = 128 << 20

# --- .docx ---------------------------------------------------------------- #
# A ``.docx`` is a zip of XML parts. Only one of them holds the prose, and it is
# named in the package relationships; every producer in practice writes it here.
_DOCX_MAIN_PART = "word/document.xml"
_DOCX_RELATIONSHIPS_PART = "_rels/.rels"
_DOCX_MAIN_RELATIONSHIP = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
)
_PACKAGE_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_COMPATIBILITY_NS = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"

# The only element whose character data is prose. Its siblings ``w:instrText``
# (field codes such as ``HYPERLINK "http://..."``) and ``w:delText`` (text struck
# out under tracked changes) are text in the *file* that is not text in the
# *document*, and reading only ``w:t`` excludes both without a special case.
_DOCX_TEXT = _WORD_NS + "t"

# Elements that stand for whitespace Word draws rather than stores. ``w:p`` is
# here because a paragraph break is what separates one line from the next -- one
# newline per paragraph, which is what Word's own plain-text export produces.
_DOCX_BREAKS = {
    _WORD_NS + "p": "\n",
    _WORD_NS + "br": "\n",
    _WORD_NS + "cr": "\n",
    _WORD_NS + "tab": "\t",
    _WORD_NS + "noBreakHyphen": "-",
}

# Subtrees to walk past. ``mc:AlternateContent`` holds the *same* content twice --
# a ``mc:Choice`` in current markup and a ``mc:Fallback`` in the older VML for
# readers that predate it -- so a walk that visits both puts every text box into
# the corpus twice. Word writes the text into both branches; the choice of which
# to drop follows what Word itself reads.
_DOCX_SKIPPED = frozenset({_COMPATIBILITY_NS + "Fallback"})

# Ceiling on the one XML part that is parsed. ElementTree's tree is the cost here,
# not the XML: measured with ``tracemalloc`` on four real documents, the parsed
# tree came to 7.9-9.3x the size of ``word/document.xml``. At 32 MiB that is a few
# hundred MiB transient in the worst case, and 32 MiB of Word XML is a book -- the
# four measured documents ran 4.3-21x more XML than text, so it holds roughly
# 1.5-7 MB of prose. Above the ceiling the refusal points at a plain-text export,
# which streams with no ceiling at all.
_DOCX_MAX_XML_BYTES = 32 << 20

# Ceiling on buffering a whole ``.docx`` package, which is only needed when it does
# not sit on disk as a plain file (see ``_docx_package``). Deliberately larger than
# the part ceiling because a package is mostly pictures: measured on a real 3.2 MB
# document, ``word/document.xml`` was 189 KB of it and the other 3 MB was PNG.
_DOCX_MAX_PACKAGE_BYTES = 64 << 20

# The relationships part is a handful of lines. Bounded anyway, because it is only
# read to find a part name and a zip can declare any size it likes.
_DOCX_MAX_RELATIONSHIPS_BYTES = 1 << 20

# What a ``.doc`` and a password-protected ``.docx`` both start with: they are OLE2
# compound files, not zips. Worth naming, because "not a zip archive" sends someone
# looking for a corrupt download when the file is fine and simply is not a ``.docx``.
_OLE_MAGIC = b"\xd0\xcf\x11\xe0"

# --- .sqlite -------------------------------------------------------------- #
# The first sixteen bytes of every SQLite database file, from the file-format spec.
# Checked because ``.db`` says nothing: a walk that selected a ``Thumbs.db`` would
# fail a whole run over a file the user never meant to include, and a direct point
# at one deserves "this is not a database" rather than sqlite3's own
# "file is not a database", which reads like the file is damaged.
_SQLITE_MAGIC = b"SQLite format 3\x00"

# Spellings of the implicit rowid column, tried in order. Any of them works on an
# ordinary table; a table with a *declared* column of the same name shadows that
# spelling, so the first unshadowed one wins. Measured: with a column called
# ``rowid``, ``ORDER BY rowid`` sorts by the column and ``ORDER BY _rowid_`` gives
# insertion order -- both deterministic, but only one of them is the row order.
_ROWID_ALIASES = ("rowid", "_rowid_", "oid")

# How long to wait for a database another process holds a write lock on. The
# default is 5s; a training run should not sit there that long before saying what
# is wrong, because the answer ("something else has it open") does not change.
_SQLITE_TIMEOUT_SECONDS = 2.0

# Names SQLite keeps for itself, and the prefix by which a virtual table's storage
# tables can be told from real ones. An FTS5 table called ``notes`` stores itself in
# ``notes_config``, ``notes_content``, ``notes_data``, ``notes_docsize`` and
# ``notes_idx``, all of which ``sqlite_master`` reports as ordinary tables -- so a
# database with one searchable table of documents looks like a database with six
# tables, and "which one holds the text" becomes a question the user cannot answer.
# ``PRAGMA table_list`` would report them authoritatively as ``shadow``, but it
# arrived in SQLite 3.37 and this project supports whatever SQLite Python 3.10 was
# built against, so the name prefix is used instead. Measured across fts3, fts4,
# fts5 and rtree: every shadow table is the virtual table's name, an underscore,
# then a suffix.
_SQLITE_INTERNAL_PREFIX = "sqlite_"
_SQLITE_VIRTUAL_PREAMBLE = "CREATE VIRTUAL TABLE"

# Byte-order marks, longest first so that UTF-32-LE is not mistaken for UTF-16-LE
# (its first two bytes are identical).
_BOMS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)

# Codecs whose encoder emits a BOM, mapped to a fixed-endian equivalent that does
# not. Used only for counting bytes, and the counts match either endianness.
_BOMLESS_EQUIVALENT = {"utf-8-sig": "utf-8", "utf-16": "utf-16-le", "utf-32": "utf-32-le"}
_BOM_LENGTHS = {"utf-8-sig": 3, "utf-16": 2, "utf-32": 4}

# Wide codecs cannot be split on a newline byte.
_WIDE_CODECS = frozenset({"utf-16", "utf-16-le", "utf-16-be", "utf-32", "utf-32-le", "utf-32-be"})

# Not the set of encodings --encoding accepts, which is every text codec Python knows.
# Only the ones worth naming back to someone who misspelt one, and the pool a near miss
# is drawn from: a corpus is overwhelmingly one of these, and suggesting koi8-r to
# someone who typed "uft-8" would be worse than suggesting nothing.
_COMMON_ENCODINGS = ("utf-8", "utf-16", "utf-32", "latin-1", "cp1252", "ascii")

OnError = Literal["fail", "skip"]


def strip_compression(name: str) -> tuple[str, Compression]:
    """Split a trailing codec suffix off ``name``.

    Returns the name without it and which codec it named, so callers that care
    about the *format* suffix (``.csv`` in ``reviews.csv.bz2``) do not each
    reimplement the stripping. Case is the caller's business; pass a lowered name.
    """
    for suffix, codec in _COMPRESSION_BY_SUFFIX.items():
        if name.endswith(suffix):
            return name[: -len(suffix)], codec
    return name, "none"


def describe_supported_formats() -> str:
    """One-line summary of readable formats, for use in error hints."""
    plain = ", ".join(TEXT_SUFFIXES + JSONL_SUFFIXES + JSON_SUFFIXES + CSV_SUFFIXES + DOCX_SUFFIXES)
    databases = ", ".join(SQLITE_SUFFIXES)
    codecs_listed = ", ".join(COMPRESSION_SUFFIXES)
    archives_listed = ", ".join(ARCHIVE_SUFFIXES)
    return (
        f"{plain} (each also accepted compressed: {codecs_listed}), "
        f"archives of them: {archives_listed}, "
        f"and SQLite databases: {databases}"
    )


def _archive_kind(name: str) -> Archive | None:
    """Which container ``name`` is, if it is one. Pass a lowered name."""
    stem, _ = strip_compression(name)
    for suffix, kind in _ARCHIVE_BY_SUFFIX.items():
        if name.endswith(suffix) or stem.endswith(suffix):
            return kind
    return None


class _MemberStream(io.BufferedIOBase):
    """A stream out of an archive, closed together with what it sits on top of.

    ``ZipFile.open`` and ``TarFile.extractfile`` hand back a stream that does not own
    the archive it came from, and a member whose own name carries a codec has a
    second stream layered over the first. Closing only the outermost one leaves the
    inner file handle open, so they are held together and closed innermost first.
    That lets every reader in this module keep using ``with self._open(source)``
    without knowing where the bytes came from.

    The archive itself is deliberately *not* owned here: it outlives any one member
    (see ``Ingestor._archive_container``) and is closed when the document stream
    ends.

    Only ``read``, ``read1`` and ``readline`` are implemented; iteration,
    ``readlines`` and the context manager come from :class:`io.IOBase` on top of
    them. ``read1`` is not spare: :class:`io.BufferedIOBase` leaves it raising
    ``UnsupportedOperation``, and it is what :class:`io.TextIOWrapper` calls to read
    a line, so a member wrapped for text would be unreadable without it.

    :meth:`close` closes ``_stream`` first, then ``_owned`` in reverse of the order
    given, then the base class -- a layer is never asked to finish through a stream
    that has already gone.
    """

    def __init__(self, stream: IO[bytes], *owned: Any) -> None:
        self._stream = stream
        self._owned = owned

    def readable(self) -> bool:
        return True

    def read(self, size: int | None = -1) -> bytes:
        return self._stream.read(-1 if size is None else size)

    def read1(self, size: int = -1) -> bytes:
        return self.read(size)

    def readline(self, size: int | None = -1) -> bytes:  # type: ignore[override]
        return self._stream.readline(-1 if size is None else size)

    def close(self) -> None:
        try:
            self._stream.close()
        finally:
            try:
                for owned in reversed(self._owned):
                    owned.close()
            finally:
                super().close()


@dataclass(frozen=True)
class Document:
    """One unit of text, with enough provenance to point at it in an error.

    ``ordinal`` counts within ``source``: the JSONL record number, the CSV row
    number, the ``.json`` array index, the database row number, or the piece number
    for a text file long enough to have been split. ``byte_offset`` is the offset of
    this document's first byte in the *decompressed* file, and is zero for ``.json``
    and for a database, where nothing is read a byte at a time and no per-record
    offset exists to report -- the ordinal is the locator for those. For ``.docx`` it
    is an offset within the extracted prose rather than within the file, because the
    file is XML and an offset into that would point at markup instead of at text.

    ``trained_spans`` is empty for every format and every option except a chat
    corpus read with ``jsonl_messages_field``, where it holds the half-open
    **character** ranges of ``text`` a loss mask should keep -- see
    :mod:`trainai.data.chat`. Empty means "no opinion", not "train on nothing": the
    ordinary case is a document with no mask, and the distinction matters because a
    span list that came out empty by accident would silently zero the loss.
    """

    text: str
    source: str
    ordinal: int
    byte_offset: int
    trained_spans: tuple[tuple[int, int], ...] = ()

    @property
    def location(self) -> str:
        """Human-readable position, for error messages and reports."""
        return f"{self.source}:{self.ordinal} (byte {self.byte_offset:,})"


@dataclass(frozen=True)
class SourceFile:
    """A file selected for ingestion, on disk or inside an archive."""

    path: Path
    relative: str
    kind: Kind
    compressed: bool
    size_bytes: int
    # Which codec ``compressed`` refers to. Kept separate from the flag so the
    # manifest's recorded shape does not change, while the reader still knows
    # whether to reach for gzip, bz2 or lzma.
    compression: Compression = "none"
    # For a member of an archive: which container ``path`` is, and the member's
    # name inside it. ``compression`` then refers to a codec on the *member* name
    # (``notes.txt.gz`` inside a zip), not to the archive's own compression, which
    # zipfile and tarfile handle themselves.
    archive: Archive | None = None
    member: str | None = None

    def to_dict(self) -> dict[str, Any]:
        # Deliberately the same four keys for a member as for a file on disk:
        # ``relative`` already spells out ``papers.zip::a/b.txt``, so the manifest
        # records where a document came from without its shape changing.
        return {
            "path": self.relative,
            "kind": self.kind,
            "compressed": self.compressed,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class _DbObject:
    """One table or view of a database, as ``sqlite_master`` describes it.

    ``sql`` is the statement that created it, which is how a virtual table is
    recognised -- see :func:`_shadow_tables`. It is empty for the handful of internal
    objects SQLite does not record a statement for, none of which reach here.
    """

    name: str
    is_view: bool
    sql: str


@dataclass(frozen=True)
class _DbColumn:
    """One column of a table. ``primary_key`` is its 1-based position in the key, or 0."""

    name: str
    primary_key: int


@dataclass(frozen=True)
class IngestOptions:
    """Everything that changes which documents come out of a given directory.

    Recorded verbatim in the dataset manifest, because it is part of what makes a
    prepared dataset reproducible.

    ``max_doc_chars`` deserves a word, because it is the one default here with a
    consequence that is not obvious. For a corpus that is a single large file, it
    sets the document count -- and the train/val split is chosen per document, so
    it also sets how finely that split can be cut. At 64 KiB, a 1 MB file yields
    18 documents, and a 5% validation share rounds to one document or, for some
    seeds, to none at all. At 16 KiB the same file yields 69, which splits
    reliably. The cost is one extra end-of-text token per cut, about 0.02% of the
    token stream.
    """

    encoding: str = "utf-8"
    jsonl_field: str | None = None
    # The field holding a typed conversation -- a list of {"role", "content"} objects
    # -- instead of a flat string. Naming it switches the reader into chat mode: the
    # text is rendered by :mod:`trainai.data.chat` and the assistant's characters are
    # marked as the trained spans. Mutually exclusive with ``jsonl_field``, which
    # names a field holding text that is trained on in full.
    jsonl_messages_field: str | None = None
    csv_text_column: str | None = None
    # Which table to read from a SQLite database, needed only when it holds more
    # than one. The *column* inside that table is named with ``csv_text_column``,
    # which is deliberate: a SQL table is a table, "which column holds the text" is
    # one question, and two flags asking it would be one flag too many.
    db_table: str | None = None
    max_doc_chars: int = 1 << 14
    min_doc_chars: int = 1
    on_error: OnError = "fail"

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoding": self.encoding,
            "jsonl_field": self.jsonl_field,
            "jsonl_messages_field": self.jsonl_messages_field,
            "csv_text_column": self.csv_text_column,
            "db_table": self.db_table,
            "max_doc_chars": self.max_doc_chars,
            "min_doc_chars": self.min_doc_chars,
            "on_error": self.on_error,
        }

    def __post_init__(self) -> None:
        # Two flags naming two different fields is a contradiction, not a preference:
        # one says "this field holds text, train on all of it" and the other says
        # "this field holds a conversation, train on the replies". Picking either
        # would train on something the user did not ask for, so it stops here rather
        # than in whichever reader happened to look first.
        if self.jsonl_field is not None and self.jsonl_messages_field is not None:
            raise UsageError(
                "--jsonl-field and --jsonl-messages-field cannot both be given.",
                hint=(
                    "They describe different records. --jsonl-field names a field "
                    "holding text, and every token of it is trained on. "
                    "--jsonl-messages-field names a field holding a list of "
                    '{"role": ..., "content": ...} objects, and only the assistant '
                    "replies are trained on. Pick the one that matches the corpus."
                ),
                details={
                    "jsonl_field": self.jsonl_field,
                    "jsonl_messages_field": self.jsonl_messages_field,
                },
            )

        # A codec name is a string the user typed, and every reader here hands it
        # straight to Python. Measured before this check existed, --encoding
        # no-such-codec on a plain .txt corpus produced "LookupError: unknown
        # encoding: no-such-codec" from inside <frozen codecs>: a traceback, with no
        # mention of the flag that caused it and none of our own exit codes. On a
        # BOM'd file it produced a good message by luck, and on an extensionless file
        # it produced a wrong one -- "none of them a readable format" -- because the
        # name had reached the sniffer before anything checked it.
        #
        # The empty string is encoded rather than looked up because codecs.lookup also
        # accepts the bytes-to-bytes codecs -- base64, zlib, rot13 -- which no reader
        # here can use. str.encode rejects those by name too, and on an empty string
        # it has nothing else it can fail on.
        try:
            "".encode(self.encoding)
        except LookupError as exc:
            near = difflib.get_close_matches(self.encoding, _COMMON_ENCODINGS, n=1, cutoff=0.7)
            raise UsageError(
                f"--encoding was given as {self.encoding!r}, which is not a text encoding.",
                hint=" ".join(
                    part
                    for part in (
                        f"Did you mean {near[0]}?" if near else "",
                        f"Common ones are {', '.join(_COMMON_ENCODINGS)}, though any "
                        "codec Python knows by name will do.",
                        "Leave --encoding off to read UTF-8 and honour a byte-order mark.",
                    )
                    if part
                ),
                details={"encoding": self.encoding},
            ) from exc


@dataclass
class IngestStats:
    """Counters describing what ingestion actually did.

    Every rejection is counted here so the summary the user sees adds up. A file
    that was skipped and never mentioned would be the worst possible outcome.
    """

    files_read: int = 0
    files_skipped: int = 0
    documents_emitted: int = 0
    documents_skipped_short: int = 0
    records_skipped_malformed: int = 0
    # CSV rows holding more fields than the header names. Nothing is dropped for
    # this, so it is not a skip -- it is a count of rows whose extracted text may
    # be a fragment, which is why it is reported separately from the skips.
    csv_rows_with_extra_fields: int = 0
    long_files_split: int = 0
    total_bytes_read: int = 0
    archives_opened: int = 0
    # Members of an archive that were passed over, split by why: an extension
    # TrainAI does not read, or a nested archive, which is reported rather than
    # followed. Both are counted *and named*, because a zip is opaque -- the user
    # cannot see what was inside it from a directory listing, so anything dropped
    # silently is invisible. The counts are exact; the name lists are capped.
    archive_members_ignored: int = 0
    nested_archives_ignored: int = 0
    ignored_archive_members: list[str] = field(default_factory=list)
    ignored_nested_archives: list[str] = field(default_factory=list)
    # Files a directory walk passed over: an extension TrainAI does not read, or no
    # extension at all. Counted and named for the same reason ignored members are --
    # a file dropped from a walk without being mentioned is a corpus quietly smaller
    # than the directory the user pointed at. ``files_without_extension`` is a subset
    # of ``files_unrecognized``, counted separately because it has a different fix:
    # point TrainAI straight at one of those and it is read as text.
    files_unrecognized: int = 0
    files_without_extension: int = 0
    unrecognized_files: list[str] = field(default_factory=list)
    # A file read as plain text on the strength of being named, having no extension
    # to classify it by. Only pointing directly at a file produces one, so this holds
    # at most a single name; it exists so the assumption reaches the manifest instead
    # of being made invisibly.
    assumed_text_files: list[str] = field(default_factory=list)
    skipped_files: list[dict[str, Any]] = field(default_factory=list)
    resolved_jsonl_fields: dict[str, str] = field(default_factory=dict)
    resolved_csv_columns: dict[str, str] = field(default_factory=dict)
    # Which table and column a database was read from, when TrainAI worked it out
    # rather than being told. Recorded for the same reason ``resolved_csv_columns``
    # is: a choice made on the user's behalf has to be visible in the report, or the
    # corpus is not the one they think they prepared. This is what makes the shadow
    # table rule safe -- if it ever picks the wrong table, the report says which.
    resolved_db_tables: dict[str, str] = field(default_factory=dict)
    resolved_db_columns: dict[str, str] = field(default_factory=dict)
    # Chat records read with ``jsonl_messages_field``, and the characters inside them.
    # ``chat_chars`` is the denominator for ``chat_trained_chars``: the share is the
    # only number that says whether the mask is doing what the user thinks, and a
    # count without its total cannot be checked against anything. Characters, not
    # tokens -- this module has no tokenizer, and inventing one here to report a
    # rounder-sounding figure would be the estimate this project refuses to make.
    chat_documents: int = 0
    chat_trained_chars: int = 0
    chat_chars: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_read": self.files_read,
            "files_skipped": self.files_skipped,
            "documents_emitted": self.documents_emitted,
            "documents_skipped_short": self.documents_skipped_short,
            "records_skipped_malformed": self.records_skipped_malformed,
            "csv_rows_with_extra_fields": self.csv_rows_with_extra_fields,
            "long_files_split": self.long_files_split,
            "total_bytes_read": self.total_bytes_read,
            "archives_opened": self.archives_opened,
            "archive_members_ignored": self.archive_members_ignored,
            "nested_archives_ignored": self.nested_archives_ignored,
            "ignored_archive_members": self.ignored_archive_members,
            "ignored_nested_archives": self.ignored_nested_archives,
            "files_unrecognized": self.files_unrecognized,
            "files_without_extension": self.files_without_extension,
            "unrecognized_files": self.unrecognized_files,
            "assumed_text_files": self.assumed_text_files,
            "skipped_files": self.skipped_files,
            "resolved_jsonl_fields": self.resolved_jsonl_fields,
            "resolved_csv_columns": self.resolved_csv_columns,
            "resolved_db_tables": self.resolved_db_tables,
            "resolved_db_columns": self.resolved_db_columns,
            "chat_documents": self.chat_documents,
            "chat_trained_chars": self.chat_trained_chars,
            "chat_chars": self.chat_chars,
        }


class Ingestor:
    """Discovers source files and streams documents out of them.

    Stateful on purpose: :attr:`stats` accumulates across a run so the caller can
    report what happened after consuming the stream.
    """

    def __init__(self, options: IngestOptions | None = None) -> None:
        self.options = options or IngestOptions()
        self.stats = IngestStats()
        # The archive currently open for reading, if any. One container is held at a
        # time and reused across its members, because constructing one is not free:
        # measured on a 384-member 1 MiB .tar.gz, opening it per member took 28.2 s
        # against 1.3 s reusing it, since building a tar's member index means
        # decompressing the whole stream. Lifetime is one ``documents()`` call --
        # see the ``finally`` there, which matters on Windows where an open handle
        # blocks deleting the file.
        self._archive_key: tuple[Path, Archive] | None = None
        self._archive_container: Any | None = None

    # --------------------------------------------------------------- discovery
    def discover(self, root: str | Path) -> list[SourceFile]:
        """Find every supported file under ``root``.

        Results are sorted, because document order determines tokenizer merges,
        shard contents, and therefore every checksum in the manifest. An
        unsorted directory walk would make "same input, same seed, same output"
        false on some filesystems and true on others.
        """
        root = Path(root)
        if not root.exists():
            raise DatasetNotFoundError(
                f"{root} does not exist.",
                hint="Check the path. Pass either a single text file or a directory of them.",
                details={"path": str(root)},
            )

        if root.is_file():
            archive = _archive_kind(root.name.lower())
            if archive is not None:
                return self._expand_archive(root, relative=root.name, archive=archive, direct=True)
            source = self._classify(root, relative=root.name)
            if source is None and _is_extensionless(root.name.lower()):
                source = self._assume_text(root, relative=root.name)
            if source is None:
                raise DatasetNotFoundError(
                    f"{root} is not a format TrainAI can read.",
                    hint=f"Supported formats: {describe_supported_formats()}.",
                    details={"path": str(root), "suffix": root.suffix},
                )
            return [source]

        found: list[SourceFile] = []
        for path in sorted(root.rglob("*"), key=lambda p: p.as_posix()):
            if _is_hidden(path, root):
                continue
            try:
                if not path.is_file():
                    continue
            except OSError:  # pragma: no cover - unreadable directory entry
                continue
            relative = path.relative_to(root).as_posix()
            archive = _archive_kind(path.name.lower())
            if archive is not None:
                # An archive found by walking is expanded in place, so its members
                # sit where the archive sat in the sorted order. An unreadable one
                # is not fatal here -- unlike the direct case, the user did not
                # point at it -- but it is counted rather than passed over.
                found.extend(
                    self._expand_archive(path, relative=relative, archive=archive, direct=False)
                )
                continue
            source = self._classify(path, relative=relative)
            if source is not None and source.kind == "sqlite" and not self._walk_reads(source):
                self._note_unrecognized(relative, extensionless=False)
                continue
            if source is not None:
                found.append(source)
            else:
                # Not selected, but not silent either. A walk is a guess about what
                # the user meant, so a stray README joining the training data would
                # be an invisible change to the corpus -- and so would dropping a
                # file called `corpus`. Naming it is the compromise.
                bare = _is_extensionless(path.name.lower())
                self._note_unrecognized(relative, extensionless=bare)

        if not found:
            raise DatasetNotFoundError(
                f"No readable text files found under {root}.{self._what_it_held()}",
                hint=(
                    f"TrainAI looks for {describe_supported_formats()}. "
                    "If your text is in another format, convert it first "
                    "(one document per line in a .jsonl file works well)."
                ),
                details={
                    "path": str(root),
                    "files_unrecognized": self.stats.files_unrecognized,
                    "files": self.stats.unrecognized_files[:10],
                },
            )
        return found

    def _what_it_held(self) -> str:
        """Name a few of the files a fruitless walk passed over, if there were any.

        "No readable text files found" on its own leaves the user to guess whether
        TrainAI looked in the wrong place or simply did not like what it saw. Naming
        what was there settles that in one line -- the same reason the equivalent
        error for an archive lists its members.
        """
        total = self.stats.files_unrecognized
        if not total:
            return ""
        shown = self.stats.unrecognized_files[:10]
        listed = ", ".join(shown)
        if total > len(shown):
            listed += f", and {total - len(shown):,} more"
        return f" It holds {total:,} file(s), none of them a readable format: {listed}."

    def _classify(self, path: Path, *, relative: str) -> SourceFile | None:
        kind, compression = _classify_name(path.name.lower())
        if kind is None:
            return None

        try:
            size = path.stat().st_size
        except OSError:  # pragma: no cover - vanished or unreadable
            return None

        return SourceFile(
            path=path,
            relative=relative,
            kind=kind,
            compressed=compression != "none",
            size_bytes=size,
            compression=compression,
        )

    def _walk_reads(self, source: SourceFile) -> bool:
        """Whether a walk should select this database, rather than pass over it.

        The same asymmetry :meth:`_assume_text` rests on, pointed the other way.
        Finding a file in a tree is a guess; naming one is an instruction. So a walk
        selects a database only when it can actually read it, and the two refusals
        that would otherwise stop a whole run over a file the user never pointed at
        are turned into a pass-over here instead:

        - **It is not a database.** ``.db`` is generic -- Windows leaves a
          ``Thumbs.db`` in picture directories, and plenty of applications use the
          extension for private formats -- so the magic header decides.
        - **It is compressed.** ``sqlite3`` needs a real seekable file, so a
          ``.db.gz`` cannot be read at all.

        Either way the file is counted and named in the report, and pointing straight
        at it gives the specific reason: :meth:`_sqlite_not_a_database` or
        :meth:`_sqlite_needs_a_plain_file`.
        """
        if source.compression != "none":
            return False
        return _has_sqlite_magic(source.path)

    def _assume_text(self, path: Path, *, relative: str) -> SourceFile | None:
        """Read a file with no extension as plain text, because the user named it.

        The asymmetry with :meth:`discover`'s walk is the point. Naming a file is an
        instruction, and a corpus in a file called ``corpus`` should not be a dead
        end; finding one in a directory tree is a guess, and a stray ``README`` or
        ``LICENSE`` joining the training data is exactly the sort of invisible corpus
        change this module refuses to make.
        """
        _, compression = strip_compression(path.name.lower())
        try:
            size = path.stat().st_size
        except OSError:  # pragma: no cover - vanished between exists() and here
            return None

        source = SourceFile(
            path=path,
            relative=relative,
            kind="text",
            compressed=compression != "none",
            size_bytes=size,
            compression=compression,
        )
        self._refuse_if_binary(source)
        self.stats.assumed_text_files.append(relative)
        return source

    def _refuse_if_binary(self, source: SourceFile) -> None:
        """Stop a binary file becoming a corpus of U+0000.

        This guard is load-bearing precisely because nothing downstream would
        object. UTF-8 decodes a NUL byte happily, so reading a ``.png`` as text does
        not fail -- it succeeds, the tokenizer learns merges from the garbage, and
        the only trace is a control-character ratio reported as a neutral
        statistic. Compare the tabular detector: the failure mode worth guarding is
        the one that looks like success.

        A wide encoding is full of NULs by design -- every ASCII character in UTF-16
        carries one -- so those are recognised first and let through, by the same
        byte-order mark the decoder will honour later.
        """
        with self._open(source) as handle:
            block = handle.read(_READ_BLOCK_BYTES)

        if any(block.startswith(bom) for bom, name in _BOMS if name in _WIDE_CODECS):
            return
        if _canonical(self.options.encoding) in _WIDE_CODECS:
            return

        offset = block.find(b"\x00")
        if offset < 0:
            return
        raise DatasetFormatError(
            f"{source.relative} looks like a binary file: byte {offset:,} is NUL, "
            "which text does not contain.",
            hint=(
                "TrainAI read it as plain text because it has no extension to go by. "
                "If it really is text in a wide encoding, pass --encoding utf-16. "
                "If it is not text, it cannot be a corpus."
            ),
            details={"path": source.relative, "nul_at_byte": offset},
        )

    # ----------------------------------------------------------------- archives
    def _expand_archive(
        self, path: Path, *, relative: str, archive: Archive, direct: bool
    ) -> list[SourceFile]:
        """The readable members of one archive, as sources, sorted by member name.

        Members are sorted for the same reason a directory walk is sorted: the same
        corpus has to yield the same documents in the same order, and a zip written
        by one tool lists its entries in a different order from one written by
        another. ``namelist()`` gives insertion order, which is a property of how
        the archive was built rather than of what is in it.

        One level only. A nested archive is counted and named, never opened -- that
        is what stops the classic bomb, whose 4.5 PB exists only through nesting.
        """
        entries = self._archive_entries(path, relative=relative, archive=archive)
        self.stats.archives_opened += 1

        declared = sum(size for _, size in entries)
        if len(entries) > _ARCHIVE_MAX_MEMBERS:
            raise DatasetFormatError(
                f"{relative} holds {len(entries):,} files, over the "
                f"{_ARCHIVE_MAX_MEMBERS:,} TrainAI will read from one archive.",
                hint=(
                    "Unpack it and point TrainAI at the directory, which has no such "
                    "limit, or combine the files into one .jsonl first."
                ),
                details={
                    "path": relative,
                    "members": len(entries),
                    "limit": _ARCHIVE_MAX_MEMBERS,
                },
            )
        if declared > _ARCHIVE_MAX_DECLARED_BYTES:
            raise DatasetFormatError(
                f"{relative} declares {declared:,} bytes once unpacked, over the "
                f"{_ARCHIVE_MAX_DECLARED_BYTES:,} TrainAI will read from one archive.",
                hint=(
                    "That figure comes from the archive's own table of contents. If it "
                    "is honest, unpack the archive and prepare it in parts; if it is "
                    "not, the archive is a compression bomb and should be discarded."
                ),
                details={
                    "path": relative,
                    "declared_bytes": declared,
                    "limit_bytes": _ARCHIVE_MAX_DECLARED_BYTES,
                },
            )

        sources: list[SourceFile] = []
        for member, size in sorted(entries):
            if _is_hidden_member(member):
                continue  # The dotfile rule, and macOS resource forks with it.
            if _archive_kind(member.lower()) is not None:
                self.stats.nested_archives_ignored += 1
                self._note_ignored_member(relative, member, nested=True)
                continue
            kind, compression = _classify_name(member.rsplit("/", 1)[-1].lower())
            if kind == "sqlite":
                # A database inside an archive cannot be read: ``sqlite3`` opens a
                # path and seeks around it, and a member is a forward-only stream.
                # Counted and named like any other passed-over member, so a zip of
                # documents that happens to contain one still prepares.
                self.stats.archive_members_ignored += 1
                self._note_ignored_member(relative, member, nested=False)
                continue
            if kind is None:
                self.stats.archive_members_ignored += 1
                self._note_ignored_member(relative, member, nested=False)
                continue
            sources.append(
                SourceFile(
                    path=path,
                    relative=f"{relative}{_MEMBER_SEPARATOR}{member}",
                    kind=kind,
                    compressed=compression != "none",
                    # The archive's *uncompressed* size for this member, which is
                    # the number worth reporting: it is what the reader will see.
                    size_bytes=size,
                    compression=compression,
                    archive=archive,
                    member=member,
                )
            )

        if not sources and direct:
            listed = ", ".join(member for member, _ in sorted(entries)[:10]) or "(nothing)"
            raise DatasetNotFoundError(
                f"{relative} holds nothing TrainAI can read.",
                hint=(
                    f"It contains: {listed}. TrainAI looks for "
                    f"{describe_supported_formats()}. Two things are never read out of "
                    "an archive, both because they need to seek around a real file: "
                    "another archive, and a database. Unpack one level first."
                ),
                details={
                    "path": relative,
                    "members": [member for member, _ in sorted(entries)],
                },
            )
        return sources

    def _archive_entries(
        self, path: Path, *, relative: str, archive: Archive
    ) -> list[tuple[str, int]]:
        """Every regular file in the archive, as ``(member name, declared size)``.

        Read from the table of contents, so nothing is decompressed to get here.
        Directories, symlinks, devices and anything else that is not a plain file
        are left out: a symlink member's "contents" are the path it points at, and
        putting that in a corpus would be a small silent lie.
        """
        try:
            if archive == "zip":
                with zipfile.ZipFile(path) as container:
                    return [
                        (info.filename, info.file_size)
                        for info in container.infolist()
                        if not info.is_dir() and not _is_zip_symlink(info)
                    ]
            with tarfile.open(path, "r:*") as tar:
                return [(info.name, info.size) for info in tar.getmembers() if info.isfile()]
        except (zipfile.BadZipFile, tarfile.TarError, EOFError, OSError) as exc:
            raise DatasetFormatError(
                f"{relative} could not be read as {'a zip' if archive == 'zip' else 'a tar'} "
                f"archive: {exc}.",
                hint=(
                    "Check the file downloaded completely, and that its extension matches "
                    "what it is. TrainAI does not sniff the contents to find out."
                ),
                details={"path": relative, "archive": archive, "reason": str(exc)},
            ) from exc

    def _note_ignored_member(self, relative: str, member: str, *, nested: bool) -> None:
        """Record an ignored member, keeping the list bounded but the count exact."""
        names = self.stats.ignored_nested_archives if nested else self.stats.ignored_archive_members
        if len(names) < _MAX_NAMES_KEPT:
            names.append(f"{relative}{_MEMBER_SEPARATOR}{member}")

    def _note_unrecognized(self, relative: str, *, extensionless: bool) -> None:
        """Record a file a walk passed over, keeping the list bounded, the count exact."""
        self.stats.files_unrecognized += 1
        if extensionless:
            self.stats.files_without_extension += 1
        if len(self.stats.unrecognized_files) < _MAX_NAMES_KEPT:
            self.stats.unrecognized_files.append(relative)

    # --------------------------------------------------------------- streaming
    def documents(self, sources: Iterable[SourceFile]) -> Iterator[Document]:
        """Yield documents from ``sources``, in the order given.

        An archive stays open across its members and is closed when this generator
        finishes -- including when the caller abandons it early, since Python then
        throws ``GeneratorExit`` in and the ``finally`` still runs.
        """
        try:
            for source in sources:
                emitted_before = self.stats.documents_emitted
                try:
                    yield from self._read(source)
                except DatasetDecodeError:
                    if self.options.on_error == "fail":
                        raise
                    self._record_skip(source, reason="could not be decoded")
                    continue
                self.stats.files_read += 1
                self.stats.total_bytes_read += source.size_bytes
                # Both formats hold one continuous stretch of prose, so both can be
                # cut into several documents by --max-doc-chars, and the note that
                # says so is about the cutting rather than about the extension.
                if source.kind in ("text", "docx") and (
                    self.stats.documents_emitted - emitted_before > 1
                ):
                    self.stats.long_files_split += 1
        finally:
            self._close_archive()

    def _read(self, source: SourceFile) -> Iterator[Document]:
        """Dispatch to the reader for this source's kind, translating stream failures.

        A decompressing stream does not fail when it is opened. It fails part-way
        through being read, once it reaches the bytes that do not decode -- so the
        handlers wrapped around ``open`` never see it, and a truncated ``corpus.txt.gz``
        used to end the run with a ``zlib.error`` traceback. An interrupted download is
        an ordinary thing to have on disk and deserves an ordinary error.

        Fatal rather than skippable, even under ``--on-error skip``, and that is the
        deliberate half of this. By the time the stream fails, documents from earlier
        in the file have already been yielded and counted: skipping the remainder would
        train on however much of the corpus happened to arrive and report success.
        Every other corrupt container here -- a bad tar, a bad ``.docx``, a shredded
        database -- is already fatal, so this matches them.
        """
        reader = self._reader_for(source)
        if source.compression == "none" and source.archive is None:
            # Nothing is being decompressed, so an OSError here is a real I/O failure
            # and has no business being described to the user as a corrupt corpus.
            yield from reader
            return
        try:
            yield from reader
        except _CORRUPT_STREAM_ERRORS as exc:
            raise self._corrupt_stream(source, exc) from exc

    def _reader_for(self, source: SourceFile) -> Iterator[Document]:
        if source.kind == "jsonl":
            return self._read_jsonl(source)
        if source.kind == "json":
            return self._read_json(source)
        if source.kind == "csv":
            return self._read_csv(source)
        if source.kind == "docx":
            return self._read_docx(source)
        if source.kind == "sqlite":
            return self._read_sqlite(source)
        return self._read_text(source)

    def _corrupt_stream(self, source: SourceFile, exc: BaseException) -> DatasetFormatError:
        """The error for a compressed stream that stopped decoding part-way through."""
        truncated = isinstance(exc, EOFError)
        what = "ended before the compressed data did" if truncated else "did not decompress"
        return DatasetFormatError(
            f"{source.relative} {what}: {exc}.",
            hint=(
                "The file is corrupt or was not downloaded completely. Check its size "
                "against the source and fetch it again. TrainAI stops rather than "
                "training on the part that did arrive."
            ),
            details={
                "path": source.relative,
                "compression": source.compression,
                "archive": source.archive,
                "truncated": truncated,
                "reason": str(exc),
            },
        )

    def _record_skip(self, source: SourceFile, *, reason: str) -> None:
        self.stats.files_skipped += 1
        self.stats.skipped_files.append({"path": source.relative, "reason": reason})

    def _keep(self, text: str) -> bool:
        if len(text.strip()) < max(1, self.options.min_doc_chars):
            self.stats.documents_skipped_short += 1
            return False
        return True

    def _open(self, source: SourceFile) -> IO[bytes]:
        if source.archive is not None:
            return self._open_member(source)
        opener = _compression_opener(source.compression)
        if opener is not None:
            return opener(source.path, "rb")
        return source.path.open("rb")

    def _open_member(self, source: SourceFile) -> IO[bytes]:
        """A member of an archive, opened as if it were a file.

        Every reader calls this through ``_open``, and ``_effective_encoding`` calls
        it a second time to read the byte-order mark. Both ``ZipFile.open`` and
        ``TarFile.extractfile`` hand back a fresh stream each time, so the second
        open costs only a seek -- the container itself is reused, see
        ``_archive_container``.

        One member is read at a time, which the callers all do: several
        ``extractfile`` streams from one ``TarFile`` share its underlying file
        object, so reading two of them alternately would interleave garbage.
        """
        member = source.member
        assert member is not None  # set together with `archive` at discovery
        assert source.archive is not None
        container = self._container_for(source.path, source.archive, source)
        if source.archive == "zip":
            try:
                stream: IO[bytes] = container.open(member)
            except (KeyError, zipfile.BadZipFile, OSError) as exc:
                raise self._member_vanished(source, exc) from exc
        else:
            try:
                extracted = container.extractfile(member)
            except (KeyError, tarfile.TarError, OSError) as exc:
                raise self._member_vanished(source, exc) from exc
            if extracted is None:  # pragma: no cover - non-files are filtered above
                raise self._member_vanished(source, KeyError(member))
            stream = extracted

        inner = _compression_opener(source.compression)
        if inner is not None:
            # A codec on the member's own name, inside the archive's compression.
            return _MemberStream(inner(stream, "rb"), stream)
        return _MemberStream(stream)

    def _container_for(self, path: Path, archive: Archive, source: SourceFile) -> Any:
        """The open archive for ``path``, opening it only if it is not already open."""
        key = (path, archive)
        if self._archive_key == key:
            return self._archive_container
        self._close_archive()
        try:
            container: Any = (
                zipfile.ZipFile(path) if archive == "zip" else tarfile.open(path, "r:*")  # noqa: SIM115 - closed by _close_archive
            )
        except (zipfile.BadZipFile, tarfile.TarError, EOFError, OSError) as exc:
            raise self._member_vanished(source, exc) from exc
        self._archive_key, self._archive_container = key, container
        return container

    def _close_archive(self) -> None:
        """Release the held archive. Idempotent, and safe to call when none is held."""
        container, self._archive_container, self._archive_key = self._archive_container, None, None
        if container is not None:
            container.close()

    def _member_vanished(self, source: SourceFile, exc: Exception) -> DatasetFormatError:
        """The archive changed between discovery and reading, or is truncated."""
        return DatasetFormatError(
            f"{source.relative} could not be read from the archive: {exc}.",
            hint=(
                "The archive may have been replaced or truncated since TrainAI listed "
                "it. Re-run to list it again."
            ),
            details={"path": source.relative, "member": source.member, "reason": str(exc)},
        )

    # -- plain text --------------------------------------------------------- #
    def _read_text(self, source: SourceFile) -> Iterator[Document]:
        """Decode a text file incrementally, emitting bounded pieces.

        An incremental decoder is used rather than ``TextIOWrapper`` because it
        keeps the byte offset of a decode failure recoverable. "Invalid byte at
        offset 1,048,573" is a fixable problem; "this file is not UTF-8" is not.
        """
        encoding = self._effective_encoding(source)
        decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
        offset_codec, buffer_offset = _offset_basis(encoding)
        limit = max(1, self.options.max_doc_chars)

        buffer = ""
        consumed = 0  # bytes fed to the decoder so far
        ordinal = 0

        with self._open(source) as handle:
            while True:
                block = handle.read(_READ_BLOCK_BYTES)
                final = not block
                try:
                    buffer += decoder.decode(block, final)
                except UnicodeDecodeError as exc:
                    raise self._decode_error(source, encoding, consumed + exc.start, exc) from exc
                consumed += len(block)

                while len(buffer) > limit:
                    cut = _cut_point(buffer, limit)
                    piece, buffer = buffer[:cut], buffer[cut:]
                    if self._keep(piece):
                        yield Document(piece, source.relative, ordinal, buffer_offset)
                        self.stats.documents_emitted += 1
                        ordinal += 1
                    buffer_offset += len(piece.encode(offset_codec))

                if final:
                    break

        if self._keep(buffer):
            yield Document(buffer, source.relative, ordinal, buffer_offset)
            self.stats.documents_emitted += 1

    def _effective_encoding(self, source: SourceFile) -> str:
        """Honour a byte-order mark when the user did not name a wider encoding.

        A UTF-16 file opened as UTF-8 fails on its second byte with a complaint
        about an invalid start byte, which tells the user nothing useful. Reading
        the BOM turns that into either a correct decode or a precise instruction.
        """
        requested = self.options.encoding
        with self._open(source) as handle:
            prefix = handle.read(4)

        for bom, name in _BOMS:
            if not prefix.startswith(bom):
                continue
            if _same_codec(requested, name) or _same_codec(requested, "utf-8"):
                return name
            raise DatasetDecodeError(
                f"{source.relative} starts with a {name} byte-order mark, "
                f"but --encoding was given as {requested!r}.",
                hint=f"Drop --encoding to let TrainAI use {name}, or pass --encoding {name}.",
                details={"path": source.relative, "detected": name, "requested": requested},
            )
        return requested

    def _decode_error(
        self,
        source: SourceFile,
        encoding: str,
        offset: int,
        exc: UnicodeDecodeError,
    ) -> DatasetDecodeError:
        bad = exc.object[exc.start : exc.end]
        return DatasetDecodeError(
            f"{source.relative} is not valid {encoding}: "
            f"byte {offset:,} begins the invalid sequence {bad.hex(' ') or '??'}.",
            hint=(
                "Re-save the file as UTF-8, or pass the encoding it actually uses "
                "(--encoding latin-1 covers most Western European text). "
                "To drop unreadable files instead of stopping, pass --on-error skip."
            ),
            details={
                "path": source.relative,
                "encoding": encoding,
                "byte_offset": offset,
                "bad_bytes": bad.hex(" "),
            },
        )

    # -- jsonl -------------------------------------------------------------- #
    def _read_jsonl(self, source: SourceFile) -> Iterator[Document]:
        encoding = self._effective_encoding(source)
        if _canonical(encoding) in _WIDE_CODECS:
            raise DatasetFormatError(
                f"{source.relative} appears to be {encoding}, which TrainAI cannot "
                "read as JSON Lines.",
                hint=(
                    "Records are separated by the newline byte, which is ambiguous in "
                    "UTF-16 and UTF-32. Re-save the file as UTF-8."
                ),
                details={"path": source.relative, "encoding": encoding},
            )

        chosen = self.options.jsonl_field
        messages_field = self.options.jsonl_messages_field
        offset = 0
        ordinal = 0

        with self._open(source) as handle:
            for line_number, raw in enumerate(handle, start=1):
                start = offset
                offset += len(raw)
                if not raw.strip():
                    continue

                try:
                    line = raw.decode(encoding)
                except UnicodeDecodeError as exc:
                    raise self._decode_error(source, encoding, start + exc.start, exc) from exc

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    problem = DatasetFormatError(
                        f"{source.relative} line {line_number} is not valid JSON: {exc.msg}.",
                        hint=(
                            "JSON Lines means exactly one complete JSON value per line; a "
                            "pretty-printed array will not work. Use --on-error skip to "
                            "drop bad records instead of stopping."
                        ),
                        details={
                            "path": source.relative,
                            "line": line_number,
                            "byte_offset": start,
                            "reason": exc.msg,
                        },
                    )
                    if self.options.on_error == "fail":
                        raise problem from exc
                    self.stats.records_skipped_malformed += 1
                    continue

                if messages_field is not None:
                    document = self._chat_document(source, record, line_number, start, ordinal)
                    if document is None:
                        continue
                    yield document
                    self.stats.documents_emitted += 1
                    ordinal += 1
                    continue

                text: Any = None
                if isinstance(record, str):
                    text = record
                elif isinstance(record, dict):
                    if chosen is None:
                        chosen = self._resolve_field(source, record, line_number)
                    text = record.get(chosen)

                if not isinstance(text, str):
                    problem = self._bad_record(source, line_number, start, chosen, record, text)
                    if self.options.on_error == "fail":
                        raise problem
                    self.stats.records_skipped_malformed += 1
                    continue

                if self._keep(text):
                    yield Document(text, source.relative, ordinal, start)
                    self.stats.documents_emitted += 1
                    ordinal += 1

    def _resolve_field(
        self,
        source: SourceFile,
        record: dict[str, Any],
        line_number: int,
        *,
        unit: str = "line",
    ) -> str:
        """Pick the text field from the first usable record, then reuse it.

        ``unit`` names what ``line_number`` counts, so a ``.json`` array says
        "element 3" where JSON Lines says "line 3". The candidate list and the
        "here is what this record actually has" hint are shared on purpose: the
        two formats hold the same records and deserve the same advice.
        """
        for candidate in _TEXT_FIELD_CANDIDATES:
            if isinstance(record.get(candidate), str):
                self.stats.resolved_jsonl_fields[source.relative] = candidate
                return candidate
        keys = sorted(record)
        raise DatasetFormatError(
            f"{source.relative} {unit} {line_number} has no obvious text field "
            f"(looked for {', '.join(_TEXT_FIELD_CANDIDATES)}).",
            hint=(
                "Name the field explicitly with --jsonl-field. This record has: "
                f"{', '.join(keys) if keys else '(no keys)'}."
            ),
            details={
                "path": source.relative,
                unit: line_number,
                "available_fields": keys,
                "tried": list(_TEXT_FIELD_CANDIDATES),
            },
        )

    def _bad_record(
        self,
        source: SourceFile,
        line_number: int,
        offset: int,
        chosen: str | None,
        record: Any,
        value: Any,
        *,
        unit: str = "line",
    ) -> DatasetFormatError:
        available: list[str] = []
        if isinstance(record, dict):
            available = sorted(str(key) for key in record)
            what = (
                f"field {chosen!r} is missing"
                if chosen not in record
                else f"field {chosen!r} holds {type(value).__name__}, not a string"
            )
            # Naming the fields that *are* present is the whole fix here: the user
            # picked --jsonl-field from one file and met a record shaped differently.
            present = ", ".join(available) if available else "(no keys)"
            hint = (
                f"Every record needs a string in {chosen!r}. This record has: {present}. "
                "Use --jsonl-field to point at a different field, or --on-error skip to "
                "drop records like this one."
            )
        else:
            what = f"the record is a {type(record).__name__}, not an object or a string"
            hint = (
                "Each record should be a JSON object with a text field, or a bare JSON "
                "string. Use --on-error skip to drop records like this one."
            )
        return DatasetFormatError(
            f"{source.relative} {unit} {line_number}: {what}.",
            hint=hint,
            details={
                "path": source.relative,
                unit: line_number,
                "byte_offset": offset,
                "field": chosen,
                "available_fields": available,
            },
        )

    # -- chat --------------------------------------------------------------- #
    def _chat_document(
        self,
        source: SourceFile,
        record: Any,
        number: int,
        offset: int,
        ordinal: int,
        *,
        unit: str = "line",
    ) -> Document | None:
        """One conversation, rendered and span-marked, or ``None`` if it was skipped.

        Shared by JSON Lines and ``.json`` because they hold the same records. The
        ``None`` return covers both reasons a record does not become a document -- too
        short, or malformed under ``--on-error skip`` -- and both are counted before
        returning, so the caller only has to increment ``documents_emitted``.
        """
        field_name = self.options.jsonl_messages_field
        assert field_name is not None  # only called when the option is set
        problem: DatasetFormatError | None = None
        conversation = None

        if not isinstance(record, dict):
            problem = DatasetFormatError(
                f"{source.relative} {unit} {number}: the record is a "
                f"{type(record).__name__}, not an object with a {field_name!r} field.",
                hint=(
                    "--jsonl-messages-field expects each record to be a JSON object "
                    f"holding a conversation in {field_name!r}. Use --on-error skip to "
                    "drop records like this one."
                ),
                details={"path": source.relative, unit: number, "field": field_name},
            )
        elif field_name not in record:
            keys = sorted(str(key) for key in record)
            problem = DatasetFormatError(
                f"{source.relative} {unit} {number}: field {field_name!r} is missing.",
                hint=(
                    f"Every record needs its conversation in {field_name!r}. This record "
                    f"has: {', '.join(keys) if keys else '(no keys)'}. If the text is "
                    "already flattened into one string, use --jsonl-field instead -- it "
                    "trains on every token, with no mask."
                ),
                details={
                    "path": source.relative,
                    unit: number,
                    "field": field_name,
                    "available_fields": keys,
                },
            )
        else:
            try:
                conversation = render_conversation(record[field_name])
            except ChatFormatError as exc:
                problem = DatasetFormatError(
                    f"{source.relative} {unit} {number}: {exc.problem}.",
                    hint=exc.hint,
                    details={
                        "path": source.relative,
                        unit: number,
                        "byte_offset": offset,
                        "field": field_name,
                        **exc.fields,
                    },
                )

        if problem is not None:
            if self.options.on_error == "fail":
                raise problem
            self.stats.records_skipped_malformed += 1
            return None

        assert conversation is not None
        if not self._keep(conversation.text):
            return None
        self.stats.chat_documents += 1
        self.stats.chat_chars += len(conversation.text)
        self.stats.chat_trained_chars += conversation.trained_chars
        return Document(conversation.text, source.relative, ordinal, offset, conversation.spans)

    # -- json --------------------------------------------------------------- #
    def _read_json(self, source: SourceFile) -> Iterator[Document]:
        """One document per element of a top-level array.

        The odd one out: every other format here streams, and this one cannot.
        JSON has no record separator, so ``json.loads`` must build the whole
        object graph before the first document exists. That is bounded by
        ``_JSON_MAX_BYTES`` rather than worked around, and the refusal above the
        ceiling points at JSON Lines, which streams with no ceiling at all.

        Because the file is parsed as one value, a wide encoding is fine here --
        unlike JSON Lines and CSV, nothing is being split on the newline byte.
        """
        payload = self._read_json_source(source)
        try:
            document = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise DatasetFormatError(
                f"{source.relative} is not valid JSON: {exc.msg} (line {exc.lineno}).",
                hint=(
                    "Check the file parses: "
                    "python -c \"import json,sys; json.load(open(sys.argv[1], encoding='utf-8'))\" "
                    f"{source.relative}. For one record per line, the extension is .jsonl."
                ),
                details={
                    "path": source.relative,
                    "line": exc.lineno,
                    "reason": exc.msg,
                },
            ) from exc

        records = self._json_records(source, document)
        chosen = self.options.jsonl_field
        messages_field = self.options.jsonl_messages_field
        ordinal = 0

        for index, record in enumerate(records):
            if messages_field is not None:
                # byte_offset is 0 here for the same reason as below.
                emitted = self._chat_document(source, record, index, 0, ordinal, unit="element")
                if emitted is None:
                    continue
                yield emitted
                self.stats.documents_emitted += 1
                ordinal += 1
                continue

            text: Any = None
            if isinstance(record, str):
                text = record
            elif isinstance(record, dict):
                if chosen is None:
                    chosen = self._resolve_field(source, record, index, unit="element")
                text = record.get(chosen)

            if not isinstance(text, str):
                # byte_offset is 0 for this format: the file was parsed as one
                # value, so there is no offset for an element to report.
                problem = self._bad_record(source, index, 0, chosen, record, text, unit="element")
                if self.options.on_error == "fail":
                    raise problem
                self.stats.records_skipped_malformed += 1
                continue

            if self._keep(text):
                yield Document(text, source.relative, ordinal, 0)
                self.stats.documents_emitted += 1
                ordinal += 1

    def _read_json_source(self, source: SourceFile) -> str:
        """The whole file as text, refused if it is too large to hold in memory.

        Checked twice, because the two numbers answer different questions. The
        size on disk rejects an oversized plain file before anything is opened;
        the length actually read rejects a *compressed* one, whose size on disk
        says nothing useful -- a 269:1 gzip of repetitive JSON is easy to make,
        and at that ratio a file comfortably under the ceiling decompresses to
        tens of gigabytes. The bound has to sit on the bytes that arrive.
        """
        if source.size_bytes > _JSON_MAX_BYTES:
            raise self._json_too_large(source, source.size_bytes, measured="declared")

        encoding = self._effective_encoding(source)
        # Block by block into a bytearray rather than one bounded ``read``:
        # ``BufferedReader.read(n)`` allocates all n bytes up front, so asking for
        # the ceiling would cost 128 MiB on every file regardless of its size,
        # and joining a list of blocks at the end would hold two copies at once.
        # ``bytearray +=`` amortizes its growth and decodes without a copy.
        buffer = bytearray()
        with self._open(source) as handle:
            while len(buffer) <= _JSON_MAX_BYTES:
                block = handle.read(_READ_BLOCK_BYTES)
                if not block:
                    break
                buffer += block
        if len(buffer) > _JSON_MAX_BYTES:
            raise self._json_too_large(source, len(buffer), measured="decompressed")

        try:
            return buffer.decode(encoding)
        except UnicodeDecodeError as exc:
            raise self._decode_error(source, encoding, exc.start, exc) from exc

    def _json_too_large(
        self, source: SourceFile, size: int, *, measured: Literal["declared", "decompressed"]
    ) -> DatasetFormatError:
        """The one refusal that cannot be softened: it is a bound on memory."""
        if measured == "declared":
            what = f"is {size:,} bytes"
            counted: dict[str, Any] = {"size_bytes": size}
        else:
            # A lower bound, not a size: reading stopped at the ceiling, so how
            # much was behind it is unknown and deliberately not guessed at.
            what = f"expands to more than {_JSON_MAX_BYTES:,} bytes"
            counted = {"bytes_read": size}
        return DatasetFormatError(
            f"{source.relative} {what}, over the "
            f"{_JSON_MAX_BYTES:,}-byte limit for a single .json file.",
            hint=(
                "A .json file has no record separator, so the whole thing has to be "
                "parsed at once. Convert to JSON Lines (one object per line), which "
                "streams with no size limit:\n"
                '  python -c "import json,sys; [print(json.dumps(r)) for r in '
                "json.load(open(sys.argv[1], encoding='utf-8'))]\" in.json > out.jsonl"
            ),
            details={
                "path": source.relative,
                **counted,
                "limit_bytes": _JSON_MAX_BYTES,
                "measured": measured,
            },
        )

    def _json_text_key(self, document: dict[str, Any]) -> str | None:
        """The key that makes this object look like a record rather than a wrapper.

        Checked against the field the user named, if they named one, so
        ``--jsonl-field headline`` is honoured here too.
        """
        named = self.options.jsonl_field
        candidates = (named,) if named else _TEXT_FIELD_CANDIDATES
        for candidate in candidates:
            if candidate and isinstance(document.get(candidate), str):
                return candidate
        return None

    def _json_records(self, source: SourceFile, document: Any) -> list[Any]:
        """The array of records inside a parsed ``.json``.

        A top-level array is the documents. A top-level object is the other
        common shape -- ``{"data": [...]}`` from an API -- and is unwrapped only
        when the choice is unambiguous. With several array-valued keys, picking
        one is the ``Summary``-column mistake in a new costume, so it lists them
        and stops.
        """
        if isinstance(document, list):
            return document

        if isinstance(document, dict):
            arrays = sorted(key for key, value in document.items() if isinstance(value, list))
            if len(arrays) == 1 and not self._json_text_key(document):
                return list(document[arrays[0]])
            if len(arrays) == 1:
                # One array, but the object also holds text of its own, so it
                # reads equally well as a single record. Unwrapping would train on
                # {"text": "...", "tags": [...]} 's *tags* and discard the text,
                # which is the silent-wrong-extraction failure this module exists
                # to refuse. Both readings are named so the fix is obvious.
                text_key = self._json_text_key(document)
                raise DatasetFormatError(
                    f"{source.relative} is a JSON object holding both text in "
                    f"{text_key!r} and one array in {arrays[0]!r}, so whether it is "
                    "one record or a wrapper around many is ambiguous.",
                    hint=(
                        f"If it is one document, put it in an array: [{{...}}]. If "
                        f"{arrays[0]!r} holds the documents, extract it into its own "
                        "file. TrainAI will not guess between them."
                    ),
                    details={
                        "path": source.relative,
                        "arrays": arrays,
                        "text_field": text_key,
                    },
                )
            if not arrays:
                raise DatasetFormatError(
                    f"{source.relative} is a JSON object with no array in it, so there "
                    "is nothing to read as a list of documents.",
                    hint=(
                        "TrainAI reads a .json file that is an array of records, or an "
                        "object wrapping exactly one array. This file's keys are: "
                        f"{', '.join(sorted(map(str, document))) or '(no keys)'}. "
                        "If it is a single document, put it in an array: [{...}]."
                    ),
                    details={"path": source.relative, "keys": sorted(map(str, document))},
                )
            raise DatasetFormatError(
                f"{source.relative} is a JSON object with {len(arrays)} arrays in it, "
                "so which one holds the documents is ambiguous.",
                hint=(
                    "Extract the one you want into its own file, or convert to JSON "
                    f"Lines. The array-valued keys are: {', '.join(arrays)}."
                ),
                details={"path": source.relative, "arrays": arrays},
            )

        raise DatasetFormatError(
            f"{source.relative} is a JSON {type(document).__name__}, not an array of records.",
            hint=(
                "TrainAI reads a .json file that is an array of records -- objects with "
                "a text field, or bare strings -- or an object wrapping exactly one such "
                "array."
            ),
            details={"path": source.relative, "type": type(document).__name__},
        )

    # -- docx --------------------------------------------------------------- #
    def _read_docx(self, source: SourceFile) -> Iterator[Document]:
        """The prose of a Word document, cut into documents like a text file.

        This is the one format here whose bytes are not already text, and it is in
        scope because the extraction is *exact* rather than reconstructed. A
        ``.docx`` is a zip of XML with real paragraph and run elements: the
        characters inside ``w:t`` are the characters Word displays, in the order it
        displays them. Nothing has to be inferred from a visual layout, which is the
        line that keeps ``.pdf`` out -- there, text is recovered from where marks sit
        on a page, and it fails by producing plausible-looking rubbish.

        What is deliberately left out, and why:

        - **Headers and footers.** They are separate parts, and they repeat on every
          page. Including them would stamp the same line through the corpus at
          whatever rate the document happens to break pages -- the page-furniture
          problem that makes naive PDF extraction useless.
        - **Footnotes, endnotes and comments.** Also separate parts, and outside the
          document's reading order, so splicing them in would put a note in the
          middle of a sentence.
        - **Field codes and tracked deletions.** ``w:instrText`` holds instructions
          like ``HYPERLINK "http://..."`` and ``w:delText`` holds text struck out
          under review. Both are in the file; neither is in the document.

        Tables are kept, one line per cell, because their contents are prose the
        author wrote. ``--encoding`` does not apply: an XML part declares its own
        encoding and the parser honours that.
        """
        yield from self._emit_pieces(source, self._docx_text(source))

    def _docx_text(self, source: SourceFile) -> str:
        """Everything a Word document displays, as one string."""
        name, xml = self._docx_main_part(source)
        root = self._docx_parse(source, name, xml)
        return _docx_body_text(root)

    def _docx_main_part(self, source: SourceFile) -> tuple[str, bytes]:
        """The XML part holding the prose, and its name inside the package.

        Only that one part is read. A ``.docx`` is mostly pictures -- measured on a
        real 3.2 MB document, ``word/document.xml`` was 189 KB of it -- so unpacking
        the package to reach the text would cost an order of magnitude more memory
        than the text needs.
        """
        with self._docx_package(source) as package:
            name = self._docx_part_name(source, package)
            declared = package.getinfo(name).file_size
            if declared > _DOCX_MAX_XML_BYTES:
                raise self._docx_part_too_large(source, name, declared)
            # Reading a member decompresses it, so this needs the same errors a codec
            # raises and not just ``zipfile``'s own: measured, a plain ``.docx`` on disk
            # with eight bytes flipped inside its deflate stream ended the run with a
            # ``zlib.error`` traceback. ``_read``'s wrapper does not cover it, because a
            # plain file is deliberately left unwrapped -- there is no outer codec there
            # to blame, and the compression is inside the package rather than around it.
            try:
                with package.open(name) as handle:
                    # Deliberately not checked a second time against what arrived, the
                    # way ``_docx_buffered`` and ``_read_json_source`` check theirs. A
                    # declared size is normally an untrusted claim, but this one
                    # ``zipfile`` enforces itself: it caps the read at ``file_size``
                    # and then fails the CRC. Measured on an entry rewritten to claim
                    # 8 bytes while holding 5,000, stored and deflated alike: BadZipFile,
                    # "Bad CRC-32", never a short read. A second ceiling check here would
                    # be a branch no input can reach.
                    xml = handle.read(declared)
            except (KeyError, zipfile.BadZipFile, *_CORRUPT_STREAM_ERRORS) as exc:
                raise self._docx_part_unreadable(source, name, exc) from exc
        return name, xml

    def _docx_package(self, source: SourceFile) -> zipfile.ZipFile:
        """The ``.docx`` opened as the zip package it is.

        A plain file on disk is opened in place: ``zipfile`` seeks to the central
        directory at its end and pulls out one member, touching nothing else. Any
        other source -- a member of an archive, or a ``.docx.gz`` -- has no usefully
        seekable stream behind it, so it is buffered whole under a ceiling first.
        The buffer is about cost rather than correctness: seeking backwards in a
        decompressing stream re-decompresses it from the start, and ``zipfile`` seeks
        to the end before it reads anything at all.
        """
        plain = source.archive is None and source.compression == "none"
        stream: Any = source.path if plain else io.BytesIO(self._docx_buffered(source))
        try:
            return zipfile.ZipFile(stream)
        except (zipfile.BadZipFile, EOFError, OSError) as exc:
            raise self._docx_not_a_package(source, exc) from exc

    def _docx_buffered(self, source: SourceFile) -> bytearray:
        """A whole ``.docx`` in memory, refused if it is too big to hold."""
        if source.size_bytes > _DOCX_MAX_PACKAGE_BYTES:
            raise self._docx_package_too_large(source, source.size_bytes, measured="declared")
        buffer = bytearray()
        with self._open(source) as handle:
            while len(buffer) <= _DOCX_MAX_PACKAGE_BYTES:
                block = handle.read(_READ_BLOCK_BYTES)
                if not block:
                    break
                buffer += block
        if len(buffer) > _DOCX_MAX_PACKAGE_BYTES:
            raise self._docx_package_too_large(source, len(buffer), measured="decompressed")
        return buffer

    def _docx_part_name(self, source: SourceFile, package: zipfile.ZipFile) -> str:
        """Where the prose lives in this package.

        Every producer in practice writes ``word/document.xml``, and that is tried
        first. The package format does not require the name, though -- it is declared
        as a relationship -- so the declaration is honoured as a fallback rather than
        failing on a conforming file that happens to be spelled differently.
        """
        names = set(package.namelist())
        if _DOCX_MAIN_PART in names:
            return _DOCX_MAIN_PART

        declared = self._docx_declared_part(source, package, names)
        if declared is not None:
            return declared

        listed = ", ".join(sorted(names)[:10]) or "(nothing)"
        raise DatasetFormatError(
            f"{source.relative} is a zip archive, but not a Word document: it has no "
            f"{_DOCX_MAIN_PART} part.",
            hint=(
                f"It contains: {listed}. If it is really a zip of text files, rename it "
                ".zip and TrainAI will read the files inside it."
            ),
            details={"path": source.relative, "parts": sorted(names)[:20]},
        )

    def _docx_declared_part(
        self, source: SourceFile, package: zipfile.ZipFile, names: set[str]
    ) -> str | None:
        """The main part's name as the package's relationships declare it.

        Every way of failing to read the declaration returns ``None`` rather than
        raising, which lands on ``_docx_part_name``'s "no ``word/document.xml``
        part" refusal. That refusal lists the parts the package does hold, so a
        damaged package still shows the user something true; raising from here
        instead would describe a file whose conventional part is *also* missing as
        damaged when the more likely reading is that it was never a Word document.
        """
        if _DOCX_RELATIONSHIPS_PART not in names:
            return None
        if package.getinfo(_DOCX_RELATIONSHIPS_PART).file_size > _DOCX_MAX_RELATIONSHIPS_BYTES:
            return None
        try:
            with package.open(_DOCX_RELATIONSHIPS_PART) as handle:
                rels = handle.read(_DOCX_MAX_RELATIONSHIPS_BYTES)
        except (KeyError, zipfile.BadZipFile, *_CORRUPT_STREAM_ERRORS):
            return None
        root = self._docx_parse(source, _DOCX_RELATIONSHIPS_PART, rels)
        for relationship in root.iter(_PACKAGE_NS + "Relationship"):
            if relationship.get("Type") != _DOCX_MAIN_RELATIONSHIP:
                # Word writes three or four of these -- core properties, extended
                # properties, sometimes a thumbnail -- and the main one is not
                # reliably first, so this skips rather than stops.
                continue
            # Targets in this part are relative to the package root, and a leading
            # slash is permitted; neither is a path to resolve on disk, because
            # nothing here is ever extracted.
            target = (relationship.get("Target") or "").lstrip("/")
            if target in names:
                return target
            # A declaration pointing at a part the package does not hold is not a
            # name to hand back: the caller looks it up with ``getinfo``, outside any
            # handler, so returning it would end the run with a bare ``KeyError``.
        return None

    def _docx_parse(self, source: SourceFile, part: str, xml: bytes) -> ElementTree.Element:
        """Parse one XML part of a ``.docx``, refusing a document type declaration.

        The refusal is the interesting line. A DTD is where XML parsing turns from
        reading a file into running a program: the classic ``billion laughs`` is 600
        bytes of nested entity definitions that expand to gigabytes, and it arrives
        through exactly this path -- a file the user downloaded. This interpreter's
        expat happens to cap the amplification factor, but that protection depends on
        the libexpat a given Python was linked against, and TrainAI supports Python
        3.10 upward, so relying on it would be relying on somebody else's build.

        Refusing outright costs nothing, because a ``.docx`` part is not allowed a DTD
        in the first place (ECMA-376 forbids it). Nor can the check misfire on prose:
        a raw ``<`` cannot appear in XML character data, so the byte sequence being
        searched for cannot occur inside anything an author typed.
        """
        if b"<!DOCTYPE" in xml:
            raise DatasetFormatError(
                f"{source.relative} contains a document type declaration in {part}, "
                "which a Word document does not.",
                hint=(
                    "A DTD lets a small file expand into an enormous one while it is "
                    "being parsed, so TrainAI will not parse one. If this file came from "
                    "somewhere you do not control, treat it as hostile rather than broken."
                ),
                details={"path": source.relative, "part": part},
            )
        try:
            return ElementTree.fromstring(xml)
        except ElementTree.ParseError as exc:
            raise DatasetFormatError(
                f"{source.relative} holds a {part} that is not valid XML: {exc}.",
                hint=(
                    "The file is a zip, but its contents are damaged. Try opening it in "
                    "Word and saving it again, or re-download it."
                ),
                details={"path": source.relative, "part": part, "reason": str(exc)},
            ) from exc

    def _docx_not_a_package(self, source: SourceFile, exc: Exception) -> DatasetFormatError:
        """Not a zip at all -- which for this extension has one likely cause."""
        head = b""
        try:
            with self._open(source) as handle:
                head = handle.read(len(_OLE_MAGIC))
        except OSError:  # pragma: no cover - it opened once already
            pass
        if head.startswith(_OLE_MAGIC):
            return DatasetFormatError(
                f"{source.relative} is not a .docx: it is an OLE compound file, which "
                "means either a legacy .doc or a password-protected document.",
                hint=(
                    "Open it in Word and save it as .docx (or, if it is protected, save "
                    "an unprotected copy). Renaming a .doc does not convert it."
                ),
                details={"path": source.relative, "format": "ole"},
            )
        return DatasetFormatError(
            f"{source.relative} could not be read as a .docx: {exc}.",
            hint=(
                "A .docx is a zip archive of XML. Check the file downloaded completely, "
                "and that its extension matches what it is -- TrainAI does not sniff the "
                "contents to find out."
            ),
            details={"path": source.relative, "reason": str(exc)},
        )

    def _docx_part_unreadable(
        self, source: SourceFile, part: str, exc: Exception
    ) -> DatasetFormatError:
        """The part was listed in the package but would not come out of it."""
        return DatasetFormatError(
            f"{source.relative} lists {part} but it could not be read: {exc}.",
            hint=(
                "The archive's table of contents disagrees with its contents, which "
                "means the file is truncated or damaged. Re-download it."
            ),
            details={"path": source.relative, "part": part, "reason": str(exc)},
        )

    def _docx_part_too_large(self, source: SourceFile, part: str, size: int) -> DatasetFormatError:
        """A bound on memory, for the same reason ``.json`` has one."""
        return DatasetFormatError(
            f"{source.relative} holds a {part} of {size:,} bytes, over the "
            f"{_DOCX_MAX_XML_BYTES:,}-byte limit for one Word document.",
            hint=(
                "An XML part has to be parsed as a whole before any text comes out of "
                "it, and the parsed form costs several times the bytes. Open the document "
                "and save it as plain text (.txt), which TrainAI streams with no size "
                "limit, or split it into chapters."
            ),
            details={
                "path": source.relative,
                "part": part,
                "size_bytes": size,
                "limit_bytes": _DOCX_MAX_XML_BYTES,
            },
        )

    def _docx_package_too_large(
        self, source: SourceFile, size: int, *, measured: Literal["declared", "decompressed"]
    ) -> DatasetFormatError:
        """Too large to buffer -- which only ever applies to a ``.docx`` not on disk."""
        what = (
            f"is {size:,} bytes"
            if measured == "declared"
            else f"expands to more than {_DOCX_MAX_PACKAGE_BYTES:,} bytes"
        )
        return DatasetFormatError(
            f"{source.relative} {what}, over the {_DOCX_MAX_PACKAGE_BYTES:,}-byte limit "
            "for a .docx that is not a plain file on disk.",
            hint=(
                "Reading a .docx means seeking around inside it, which needs the whole "
                "package in memory when it arrives from an archive or through a codec. "
                "Unpack it and point TrainAI at the document itself, which has no such "
                "limit."
            ),
            details={
                "path": source.relative,
                "size_bytes": size,
                "limit_bytes": _DOCX_MAX_PACKAGE_BYTES,
                "measured": measured,
            },
        )

    def _emit_pieces(self, source: SourceFile, text: str) -> Iterator[Document]:
        """Cut one already-extracted string into documents of bounded length.

        Deliberately not shared with :meth:`_read_text`, which does the same cutting
        while decoding block by block so that the byte offset of a decode failure
        stays reportable. A format parsed whole has no such offsets to keep, so it
        gets the plain loop instead of one bent to serve both.
        """
        limit = max(1, self.options.max_doc_chars)
        offset = 0
        ordinal = 0
        while len(text) > limit:
            cut = _cut_point(text, limit)
            piece, text = text[:cut], text[cut:]
            if self._keep(piece):
                yield Document(piece, source.relative, ordinal, offset)
                self.stats.documents_emitted += 1
                ordinal += 1
            offset += len(piece.encode("utf-8"))
        if self._keep(text):
            yield Document(text, source.relative, ordinal, offset)
            self.stats.documents_emitted += 1

    # -- csv and tsv -------------------------------------------------------- #
    def _read_csv(self, source: SourceFile) -> Iterator[Document]:
        """One document per row, taken from a single named column.

        Only one column is read, and joining several is not on offer. Doing that
        would mean deciding what every field means, what unit it is in and what a
        blank cell stands for -- domain judgements this module cannot make, and
        would be making invisibly. ``examples/weather_to_text.py`` shows what those
        judgements cost when someone makes them on purpose.
        """
        encoding = self._effective_encoding(source)
        if _canonical(encoding) in _WIDE_CODECS:
            raise DatasetFormatError(
                f"{source.relative} appears to be {encoding}, which TrainAI cannot "
                "read as a table.",
                hint=(
                    "Rows are separated by the newline byte, which is ambiguous in "
                    "UTF-16 and UTF-32. Re-save the file as UTF-8."
                ),
                details={"path": source.relative, "encoding": encoding},
            )

        delimiter = _csv_delimiter(source)
        # Mutated by the line reader below so the loop can record where each row
        # started. A quoted field may span lines, so the reader pulls as many as
        # it needs before yielding a row; the position after row N-1 is therefore
        # the position where row N begins.
        position = [0]

        def decoded_lines(handle: IO[bytes]) -> Iterator[str]:
            for raw in handle:
                start = position[0]
                try:
                    line = raw.decode(encoding)
                except UnicodeDecodeError as exc:
                    raise self._decode_error(source, encoding, start + exc.start, exc) from exc
                position[0] = start + len(raw)
                yield line

        # Raised for this read only, then restored: the module-level default of
        # 128 KiB rejects a legitimately long document in a cell, while an
        # unbounded limit turns one stray quotation mark into a single field
        # holding the rest of the file. A bound that large means the file is
        # broken, which is worth saying out loud.
        previous_limit = csv.field_size_limit(_CSV_MAX_FIELD_CHARS)
        try:
            with self._open(source) as handle:
                reader = csv.reader(decoded_lines(handle), delimiter=delimiter)
                try:
                    header = next(reader)
                except StopIteration:
                    return  # An empty file is not an error; the report says zero.
                except csv.Error as exc:
                    raise self._csv_unreadable(source, reader, exc) from exc

                index, column = self._resolve_column(source, header, delimiter)
                row_start = position[0]
                ordinal = 0

                while True:
                    try:
                        row = next(reader)
                    except StopIteration:
                        break
                    except csv.Error as exc:
                        raise self._csv_unreadable(source, reader, exc) from exc
                    started, row_start = row_start, position[0]

                    if not row or (len(row) == 1 and not row[0].strip()):
                        continue  # A blank line between rows is not a record.

                    if len(row) > len(header):
                        # More fields than there are column names. Counted, not
                        # refused: a trailing delimiter on every line produces this
                        # and costs nothing. But so does a delimiter sitting inside
                        # an unquoted field, and that one splits a value in two and
                        # hands back a fragment of the document with no other sign
                        # that anything was lost. Silence is the part that is not
                        # acceptable; the warning in ``validate.py`` names both
                        # causes and lets the reader decide which they have.
                        self.stats.csv_rows_with_extra_fields += 1

                    if index >= len(row):
                        problem = self._csv_short_row(
                            source, reader.line_num, started, column, index, row
                        )
                        if self.options.on_error == "fail":
                            raise problem
                        self.stats.records_skipped_malformed += 1
                        continue

                    text = row[index]
                    if self._keep(text):
                        yield Document(text, source.relative, ordinal, started)
                        self.stats.documents_emitted += 1
                        ordinal += 1
        finally:
            csv.field_size_limit(previous_limit)

    def _resolve_column(
        self, source: SourceFile, header: list[str], delimiter: str
    ) -> tuple[int, str]:
        """The index and name of the column to read, or a specific error."""
        columns = [name.strip() for name in header]
        chosen = self.options.csv_text_column

        if chosen is not None:
            wanted = chosen.strip()
            for index, name in enumerate(columns):
                if name == wanted or name.casefold() == wanted.casefold():
                    return index, name
            raise DatasetFormatError(
                f"{source.relative} has no column named {chosen!r}.",
                hint=(
                    f"Its columns are: {_quoted(columns)}. "
                    "Column names are matched ignoring case and surrounding spaces, "
                    "so pass one of these to --csv-text-column."
                ),
                details={
                    "path": source.relative,
                    "requested_column": chosen,
                    "available_columns": columns,
                },
            )

        # ``not any`` rather than ``not columns or not any(columns)``: the second
        # subsumes the first, because ``any([])`` is already False. A file whose first
        # line is blank reaches here as no fields at all, and one whose first line is
        # ``,,`` as three empty ones, and neither names a column.
        if not any(columns):
            raise DatasetFormatError(
                f"{source.relative} has no header row.",
                hint=(
                    "The first line of a CSV must name the columns, because "
                    "--csv-text-column names one of them. Add a header line."
                ),
                details={"path": source.relative},
            )

        # A single column is unambiguous -- unless its name still contains another
        # delimiter, which means the file is not separated the way its extension
        # says and every row is about to become one giant field.
        if len(columns) == 1:
            for other, word in _DELIMITER_WORDS.items():
                if other != delimiter and columns[0].count(other) >= 2:
                    expected = _DELIMITER_WORDS[delimiter]
                    raise DatasetFormatError(
                        f"{source.relative} looks {word}-separated, not "
                        f"{expected}-separated: its whole header read as one column, "
                        f"{columns[0]!r}.",
                        hint=(
                            "TrainAI takes the separator from the file extension "
                            "(.csv is comma, .tsv is tab), because a sniffer that "
                            "guesses wrong produces a dataset that looks fine and is "
                            f"not. Re-export the file with {expected}s -- in a "
                            "spreadsheet that is Save As, CSV UTF-8 -- and TrainAI "
                            "will read it."
                        ),
                        details={
                            "path": source.relative,
                            "expected_delimiter": delimiter,
                            "likely_delimiter": other,
                            "header": columns[0],
                        },
                    )
            self.stats.resolved_csv_columns[source.relative] = columns[0]
            return 0, columns[0]

        lowered = {name.casefold(): index for index, name in reversed(list(enumerate(columns)))}
        for candidate in _CSV_COLUMN_CANDIDATES:
            index = lowered.get(candidate)
            if index is not None:
                self.stats.resolved_csv_columns[source.relative] = columns[index]
                return index, columns[index]

        raise self._csv_needs_a_column(source, columns)

    def _csv_needs_a_column(self, source: SourceFile, columns: list[str]) -> DatasetFormatError:
        """The central refusal: a table is not a corpus until one column is named.

        Shaped after :func:`_json_not_supported` -- a specific and common mistake
        earns the instruction for that mistake instead of a generic list of
        formats. Both branches are spelled out because both are real: sometimes one
        column holds prose and the file is usable, and sometimes the file is
        measurements and a language model is simply the wrong tool.
        """
        return DatasetFormatError(
            f"{source.relative} is a table, and TrainAI will not guess which column "
            "holds the text.",
            hint=(
                f"Its {len(columns)} columns are: {_quoted(columns)}.\n"
                "  If one of them holds real prose -- a review, a message, a "
                "description -- train on that column alone:\n"
                f"    trainai data prepare <path> --csv-text-column <name>\n"
                "  If the columns are measurements, a language model is the wrong "
                "tool: it treats 9.47 and 9.48 as unrelated symbols and cannot learn "
                "that they are close. To predict one column from the others, use "
                "regression -- scikit-learn's HistGradientBoostingRegressor takes "
                "seconds on a CPU and reports its own error.\n"
                "  To turn a table into sentences yourself, see "
                "examples/weather_to_text.py, which shows the decisions that involves."
            ),
            details={
                "path": source.relative,
                "available_columns": columns,
                "tried": list(_CSV_COLUMN_CANDIDATES),
            },
        )

    def _csv_short_row(
        self,
        source: SourceFile,
        line_number: int,
        offset: int,
        column: str,
        index: int,
        row: list[str],
    ) -> DatasetFormatError:
        return DatasetFormatError(
            f"{source.relative} line {line_number} has {len(row)} field(s), so column "
            f"{column!r} (number {index + 1}) is missing.",
            hint=(
                "Rows shorter than the header usually mean an unescaped separator or "
                "an unclosed quotation mark earlier in the file. Use --on-error skip "
                "to drop rows like this one instead of stopping."
            ),
            details={
                "path": source.relative,
                "line": line_number,
                "byte_offset": offset,
                "column": column,
                "fields": len(row),
            },
        )

    def _csv_unreadable(
        self, source: SourceFile, reader: Any, exc: csv.Error
    ) -> DatasetFormatError:
        return DatasetFormatError(
            f"{source.relative} could not be read as a table at line {reader.line_num}: {exc}.",
            hint=(
                "One unclosed quotation mark makes everything after it a single "
                "field, so the reported line is where reading stopped, not "
                "necessarily where the mistake is. Re-export the file from whatever "
                "produced it, which will quote the fields consistently."
            ),
            details={"path": source.relative, "line": reader.line_num, "reason": str(exc)},
        )

    # -- sqlite ------------------------------------------------------------- #
    def _read_sqlite(self, source: SourceFile) -> Iterator[Document]:
        """One document per row of one column of one table.

        The same rule as a CSV -- a table is not a corpus until one column is named
        -- with one addition that SQL forces. **A query has no inherent row order.**
        A plain ``SELECT`` happens to come back in rowid order today, and is free to
        come back in index order tomorrow after someone adds an index; nothing in the
        file changes, and every checksum in the manifest does. This module's whole
        reproducibility guarantee is that document order is fixed, so the order is
        pinned explicitly here: by rowid, or by the primary key of a ``WITHOUT ROWID``
        table, and a table with neither is refused rather than read in whatever order
        arrives.

        The database is opened read-only. A training tool has no business writing to
        a file it was asked to read, and the default open mode would create one if
        the path were wrong -- silently, and empty. ``immutable=1`` would be faster
        still and is deliberately not used: it tells SQLite to skip recovery, which on
        a database with an unmerged write-ahead log means quietly reading stale rows.

        ``--encoding`` does not apply. SQLite stores its own text encoding in the
        file header and ``sqlite3`` decodes accordingly, so there is nothing to guess.
        """
        if source.archive is not None or source.compression != "none":
            raise self._sqlite_needs_a_plain_file(source)
        # Before the magic-header check, because a Python that cannot open any
        # database should say so rather than first pass judgement on the file. Every
        # ``except sqlite3.Error`` below this line is reachable only past this guard.
        if sqlite3 is None:
            raise _missing_module("sqlite3")
        if not _has_sqlite_magic(source.path):
            raise self._sqlite_not_a_database(source)

        # Built with ``as_uri()`` rather than by pasting the path after ``file:``.
        # Measured, and it bites: a database called ``hash#1.db`` spelled the naive
        # way truncates at the ``#``, which throws away ``?mode=ro`` *and* the rest of
        # the name -- so SQLite opened read-write-create and left a brand new empty
        # file called ``hash`` sitting in the user's directory. ``as_uri`` percent-
        # encodes it, and needs an absolute path to do so.
        uri = source.path.resolve().as_uri() + "?mode=ro"
        try:
            connection = sqlite3.connect(
                uri,
                uri=True,
                timeout=_SQLITE_TIMEOUT_SECONDS,
                # Not optional, and not a shortcut. Documents leave this module as a
                # generator, and the Rust BPE trainer that consumes it pulls that
                # generator from its own worker pool -- measured, seven distinct
                # threads over one corpus, none of them the thread that started the
                # read. sqlite3's default refuses to be touched from a second thread
                # at all, so a database corpus would fail during tokenizer training
                # with "SQLite objects created in a thread can only be used in that
                # same thread", which says nothing to the user about their data.
                #
                # Safe because the accesses are serialised, not merely infrequent: a
                # Python generator cannot be advanced from two threads at once (it
                # raises rather than interleaving), so the threads take turns. The
                # measurement counted zero overlapping pulls out of 3,000. SQLite
                # itself is built in serialised mode -- ``sqlite3.threadsafety`` is 3
                # -- so the handle is legal to pass between threads.
                check_same_thread=False,
            )
        except sqlite3.Error as exc:  # pragma: no cover - connect is lazy
            raise self._sqlite_unreadable(source, exc) from exc

        with closing(connection):
            table, is_view = self._resolve_table(source, connection)
            if is_view:
                raise self._sqlite_is_a_view(source, table)
            columns = self._table_columns(source, connection, table)
            column = self._resolve_db_column(source, table, columns)
            order = self._row_order(source, connection, table, columns)
            yield from self._sqlite_rows(source, connection, table, column, order)

    def _sqlite_objects(
        self, source: SourceFile, connection: sqlite3.Connection
    ) -> list[_DbObject]:
        """Every table and view the database declares, in name order."""
        try:
            rows = connection.execute(
                "SELECT type, name, sql FROM sqlite_master "
                "WHERE type IN ('table', 'view') ORDER BY name"
            ).fetchall()
        except sqlite3.Error as exc:
            raise self._sqlite_unreadable(source, exc) from exc
        return [
            _DbObject(name=name, is_view=kind == "view", sql=sql or "")
            for kind, name, sql in rows
            if not name.startswith(_SQLITE_INTERNAL_PREFIX)
        ]

    def _resolve_table(
        self, source: SourceFile, connection: sqlite3.Connection
    ) -> tuple[str, bool]:
        """Which table to read, and whether it turned out to be a view.

        Naming one takes it as given -- including a view, and including a virtual
        table's storage, because an explicit name is an instruction. Only the
        automatic choice is filtered.
        """
        objects = self._sqlite_objects(source, connection)
        chosen = self.options.db_table

        if chosen is not None:
            wanted = chosen.strip()
            for entry in objects:
                if entry.name == wanted or entry.name.casefold() == wanted.casefold():
                    return entry.name, entry.is_view
            raise self._sqlite_no_such_table(source, chosen, objects)

        readable = [entry for entry in objects if not entry.is_view]
        shadowed = _shadow_tables(readable)
        candidates = [entry.name for entry in readable if entry.name not in shadowed]

        if not candidates:
            raise self._sqlite_no_tables(source, objects)
        if len(candidates) > 1:
            raise self._sqlite_needs_a_table(source, candidates)
        self.stats.resolved_db_tables[source.relative] = candidates[0]
        return candidates[0], False

    def _table_columns(
        self, source: SourceFile, connection: sqlite3.Connection, table: str
    ) -> list[_DbColumn]:
        """The columns of ``table``, with their primary-key ordinals.

        ``PRAGMA table_info`` cannot take a bound parameter -- measured, it is a
        syntax error -- so the name is quoted into the statement instead. It came from
        ``sqlite_master`` or was matched against it, so it names something that
        exists; doubling any embedded quote is what makes the spelling of it safe.
        """
        try:
            rows = connection.execute(f"PRAGMA table_info({_sql_quote(table)})").fetchall()
        # Unreachable from any file. ``_sqlite_objects`` has already run a statement on
        # this connection, and SQLite parses the whole schema before it runs the first
        # one -- measured: a ``sqlite_master`` row rewritten to unparseable SQL fails
        # the catalogue query itself, not this pragma. So the only way here is the
        # database changing under the reader between two statements, which is a race
        # rather than an input: another process taking the write lock, or a page
        # damaged in between. Kept because both of those do happen, and the
        # alternative is a raw sqlite3 traceback; the same translation is exercised
        # from the two handlers a file can reach.
        except sqlite3.Error as exc:  # pragma: no cover - see above
            raise self._sqlite_unreadable(source, exc) from exc
        return [_DbColumn(name=name, primary_key=pk) for _, name, _, _, _, pk in rows]

    def _resolve_db_column(self, source: SourceFile, table: str, columns: list[_DbColumn]) -> str:
        """The column to read, or a specific error.

        Deliberately a separate function from :meth:`_resolve_column` rather than a
        generalisation of it. That one carries a delimiter check and a "no header row"
        error, neither of which means anything for a table that declares its columns;
        bending one function to serve both would obscure both. Only
        ``_CSV_COLUMN_CANDIDATES`` is shared, because the conventional names for "the
        column holding the text" are the same wherever the table came from.
        """
        names = [column.name for column in columns]
        chosen = self.options.csv_text_column

        if chosen is not None:
            wanted = chosen.strip()
            for name in names:
                if name == wanted or name.casefold() == wanted.casefold():
                    return name
            raise DatasetFormatError(
                f"{source.relative} table {table!r} has no column named {chosen!r}.",
                hint=(
                    f"Its columns are: {_quoted(names)}. "
                    "Column names are matched ignoring case and surrounding spaces, "
                    "so pass one of these to --csv-text-column."
                ),
                details={
                    "path": source.relative,
                    "table": table,
                    "requested_column": chosen,
                    "available_columns": names,
                },
            )

        if len(names) == 1:
            self.stats.resolved_db_columns[source.relative] = names[0]
            return names[0]

        lowered = {name.casefold(): name for name in reversed(names)}
        for candidate in _CSV_COLUMN_CANDIDATES:
            name = lowered.get(candidate)
            if name is not None:
                self.stats.resolved_db_columns[source.relative] = name
                return name

        raise self._sqlite_needs_a_column(source, table, names)

    def _row_order(
        self,
        source: SourceFile,
        connection: sqlite3.Connection,
        table: str,
        columns: list[_DbColumn],
    ) -> str:
        """An ``ORDER BY`` clause that reproduces the table's stored order.

        Three cases, in order of how common they are:

        - **An ordinary table** has a rowid, and ordering by it gives insertion
          order. Virtual tables have one too -- measured on FTS5.
        - **A declared column can shadow the name.** A table with a column called
          ``rowid`` makes ``ORDER BY rowid`` sort by that column instead, which is
          deterministic but is not the row order, so the next spelling is used.
        - **A ``WITHOUT ROWID`` table** has no rowid at all, and is ordered by its
          primary key -- which it must have, since SQLite refuses to create one
          without.

        Each rowid spelling is *probed* rather than reasoned about, because whether
        one resolves depends on the table's declaration in ways that are cheaper to
        ask than to model. ``LIMIT 0`` reads no rows.
        """
        declared = {column.name.casefold() for column in columns}
        for alias in _ROWID_ALIASES:
            if alias in declared:
                continue
            try:
                connection.execute(f"SELECT {alias} FROM {_sql_quote(table)} LIMIT 0").fetchall()
            except sqlite3.Error:
                continue
            return alias

        key = sorted(
            (column for column in columns if column.primary_key > 0),
            key=lambda column: column.primary_key,
        )
        if key:
            return ", ".join(_sql_quote(column.name) for column in key)

        raise self._sqlite_no_row_order(source, table, [column.name for column in columns])

    def _sqlite_rows(
        self,
        source: SourceFile,
        connection: sqlite3.Connection,
        table: str,
        column: str,
        order: str,
    ) -> Iterator[Document]:
        """Stream one column, ordered, as documents.

        The cursor is iterated rather than fetched: measured with ``tracemalloc``,
        walking 40 MB of text out of a database peaked at 4.5 KB of traced memory, so
        unlike ``.json`` and ``.docx`` there is no ceiling to design here.
        """
        statement = f"SELECT {_sql_quote(column)} FROM {_sql_quote(table)} ORDER BY {order}"
        ordinal = 0
        row_number = 0
        try:
            cursor = connection.execute(statement)
            for (value,) in cursor:
                row_number += 1
                if value is None:
                    # An empty cell, not a broken one. It falls out through the
                    # min-length check like an empty CSV field, and is counted there.
                    value = ""
                elif not isinstance(value, str):
                    problem = self._sqlite_not_text(source, table, column, row_number, value)
                    if self.options.on_error == "fail":
                        raise problem
                    self.stats.records_skipped_malformed += 1
                    continue
                if self._keep(value):
                    yield Document(value, source.relative, ordinal, 0)
                    self.stats.documents_emitted += 1
                    ordinal += 1
        except sqlite3.Error as exc:
            raise self._sqlite_unreadable(source, exc) from exc

    def _sqlite_needs_a_plain_file(self, source: SourceFile) -> DatasetFormatError:
        """A database has to be a real file, and this one is not."""
        what = (
            "a database inside an archive"
            if source.archive is not None
            else "a compressed database"
        )
        return DatasetFormatError(
            f"{source.relative} is {what}, which TrainAI cannot read in place.",
            hint=(
                "SQLite opens a file and seeks around it -- it reads the header, then "
                "the table of contents, then pages scattered through the file -- and "
                "neither a member of an archive nor a decompressing stream can be "
                "seeked that way. Unpack it first and point TrainAI at the database."
            ),
            details={
                "path": source.relative,
                "archive": source.archive,
                "compression": source.compression,
            },
        )

    def _sqlite_not_a_database(self, source: SourceFile) -> DatasetFormatError:
        """The extension says database and the first sixteen bytes disagree."""
        return DatasetFormatError(
            f"{source.relative} is not a SQLite database: it does not start with "
            f"{_SQLITE_MAGIC.decode('ascii').rstrip(chr(0))!r}.",
            hint=(
                "The extension is what TrainAI went by, and .db in particular is used "
                "by plenty of things that are not SQLite -- Windows thumbnail caches, "
                "and any application that liked the abbreviation. If this file is a "
                "corpus in another format, rename it to match and TrainAI will read it."
            ),
            details={"path": source.relative},
        )

    def _sqlite_unreadable(self, source: SourceFile, exc: sqlite3.Error) -> DatasetFormatError:
        """SQLite refused, and its own words are the most useful thing to pass on."""
        return DatasetFormatError(
            f"{source.relative} could not be read as a database: {exc}.",
            hint=(
                "TrainAI opens a database read-only and never writes to it. "
                "'database is locked' means another program holds a write "
                "transaction open -- close it and re-run. 'file is not a database' "
                "means the file is truncated or damaged, since its header was right."
            ),
            details={"path": source.relative, "reason": str(exc)},
        )

    def _sqlite_no_tables(self, source: SourceFile, objects: list[_DbObject]) -> DatasetFormatError:
        listed = ", ".join(f"{entry.name!r} (view)" for entry in objects if entry.is_view)
        return DatasetFormatError(
            f"{source.relative} holds no table TrainAI can read.",
            hint=(
                f"It defines: {listed or '(nothing)'}. A view has no stored row order, "
                "so TrainAI cannot promise the same corpus twice from one; name the "
                "table underneath it with --db-table instead."
                if listed
                else "The database is empty -- it declares no tables at all."
            ),
            details={
                "path": source.relative,
                "views": [entry.name for entry in objects if entry.is_view],
            },
        )

    def _sqlite_needs_a_table(
        self, source: SourceFile, candidates: list[str]
    ) -> DatasetFormatError:
        """Several tables, and picking one for the user would be a guess."""
        return DatasetFormatError(
            f"{source.relative} holds {len(candidates)} tables, and TrainAI will not "
            "guess which one holds the text.",
            hint=(
                f"They are: {_quoted(candidates)}.\n"
                "  Name one, and if it has several columns name the column too:\n"
                "    trainai data prepare <path> --db-table <name> "
                "--csv-text-column <column>\n"
                "  (--csv-text-column names the text column in a table wherever it "
                "came from, a .csv or a database.)"
            ),
            details={"path": source.relative, "tables": candidates},
        )

    def _sqlite_no_such_table(
        self, source: SourceFile, requested: str, objects: list[_DbObject]
    ) -> DatasetFormatError:
        names = [entry.name for entry in objects]
        return DatasetFormatError(
            f"{source.relative} has no table named {requested!r}.",
            hint=(
                f"It defines: {_quoted(names)}. Names are matched ignoring case and "
                "surrounding spaces, so pass one of these to --db-table."
            ),
            details={
                "path": source.relative,
                "requested_table": requested,
                "available_tables": names,
            },
        )

    def _sqlite_is_a_view(self, source: SourceFile, table: str) -> DatasetFormatError:
        """A view was named explicitly. It has no row order to reproduce."""
        return DatasetFormatError(
            f"{source.relative} names {table!r}, which is a view rather than a table.",
            hint=(
                "A view is a stored query, and a query has no inherent row order -- "
                "the same view can hand back the same rows in a different order after "
                "an index is added, which would change every checksum in the manifest "
                "while the file on disk is untouched. Name the table the view reads "
                "from with --db-table, and filter afterwards if you need to."
            ),
            details={"path": source.relative, "view": table},
        )

    def _sqlite_no_row_order(
        self, source: SourceFile, table: str, columns: list[str]
    ) -> DatasetFormatError:
        """Every rowid spelling is shadowed and there is no primary key to fall back on."""
        return DatasetFormatError(
            f"{source.relative} table {table!r} has no row order TrainAI can pin down.",
            hint=(
                f"Its columns are: {_quoted(columns)}. SQLite gives every ordinary "
                f"table a hidden row id readable as {', '.join(_ROWID_ALIASES)} -- but "
                "a declared column of the same name hides each of those spellings, and "
                "this table has no primary key to order by instead. Without a fixed "
                "order the same database would prepare into a different corpus from one "
                "run to the next. Add a primary key, or export the table to .csv."
            ),
            details={"path": source.relative, "table": table, "columns": columns},
        )

    def _sqlite_needs_a_column(
        self, source: SourceFile, table: str, columns: list[str]
    ) -> DatasetFormatError:
        """The CSV refusal, in a database. Same rule, same reasons."""
        return DatasetFormatError(
            f"{source.relative} table {table!r} has {len(columns)} columns, and TrainAI "
            "will not guess which one holds the text.",
            hint=(
                f"They are: {_quoted(columns)}.\n"
                "  If one of them holds real prose -- a review, a message, a "
                "description -- train on that column alone:\n"
                "    trainai data prepare <path> --csv-text-column <name>\n"
                "  (The same flag names the text column in a .csv and in a database "
                "table; it is one question either way.)\n"
                "  If the columns are measurements, a language model is the wrong "
                "tool: it treats 9.47 and 9.48 as unrelated symbols and cannot learn "
                "that they are close. To predict one column from the others, use "
                "regression -- scikit-learn's HistGradientBoostingRegressor takes "
                "seconds on a CPU and reports its own error."
            ),
            details={
                "path": source.relative,
                "table": table,
                "available_columns": columns,
                "tried": list(_CSV_COLUMN_CANDIDATES),
            },
        )

    def _sqlite_not_text(
        self, source: SourceFile, table: str, column: str, row_number: int, value: Any
    ) -> DatasetFormatError:
        """A cell that is not text, in a column being read as text.

        Well targeted, because SQLite would have converted it if it could: a column
        declared ``TEXT`` coerces a stored ``42`` to ``'42'`` on the way out --
        measured. A value arriving as ``bytes`` or a number means the column really
        holds blobs or numbers, so the wrong column is being read.
        """
        return DatasetFormatError(
            f"{source.relative} table {table!r} row {row_number:,} holds "
            f"{type(value).__name__} in column {column!r}, not text.",
            hint=(
                "A column declared TEXT converts numbers to text on the way out, so a "
                "value that arrives as something else means the column really holds "
                "that -- a blob, or measurements. Name the column that holds the prose "
                "with --csv-text-column, or use --on-error skip to drop rows like this "
                "one and count them."
            ),
            details={
                "path": source.relative,
                "table": table,
                "column": column,
                "row": row_number,
                "type": type(value).__name__,
            },
        )


# ----------------------------------------------------------------- helpers -- #
def _csv_delimiter(source: SourceFile) -> str:
    """The column separator, taken from the extension rather than sniffed."""
    name, _ = strip_compression(source.relative.lower())
    for suffix, delimiter in _CSV_DELIMITERS.items():
        if name.endswith(suffix):
            return delimiter
    # Unreachable: only a name ending in CSV_SUFFIXES is read as a table, and
    # test_every_extension_read_as_a_table_has_a_separator_of_its_own holds the two
    # tables to the same keys. Kept as the fallthrough the signature needs, and a
    # comma rather than a raise because a table read with the wrong separator is a
    # bad dataset, not a crash.
    return ","  # pragma: no cover


def _quoted(names: Iterable[str]) -> str:
    """Column names as a readable list, quoted because they often contain spaces."""
    listed = ", ".join(repr(name) for name in names)
    return listed or "(none)"


def _sql_quote(identifier: str) -> str:
    """A table or column name as a SQL identifier literal.

    Doubling the quote character is SQLite's own escape, so any name a database can
    hold round-trips -- including ``the "body"``, measured. Identifiers cannot be
    passed as bound parameters, so this is what makes building the statement safe.
    """
    return '"' + identifier.replace('"', '""') + '"'


def _has_sqlite_magic(path: Path) -> bool:
    """Whether ``path`` starts with the sixteen bytes every SQLite database does."""
    try:
        with path.open("rb") as handle:
            return handle.read(len(_SQLITE_MAGIC)) == _SQLITE_MAGIC
    except OSError:  # pragma: no cover - vanished or unreadable
        return False


def _shadow_tables(tables: list[_DbObject]) -> set[str]:
    """The tables that are a virtual table's private storage, by name prefix.

    A virtual table declares itself with ``CREATE VIRTUAL TABLE`` and keeps its data
    in ordinary tables named after it -- an FTS5 ``notes`` owns ``notes_config``
    through ``notes_idx`` -- and ``sqlite_master`` reports those as tables like any
    other. Left in, a database with one searchable table of documents looks like a
    database with six tables and cannot be read without naming one.

    The prefix rule can only ever *narrow* the candidate list, never wrongly widen
    it, and never empties it: the virtual table itself always survives. Its one
    misfire is a real table genuinely named ``notes_archive`` beside a virtual
    ``notes``, and that is visible rather than silent -- the table TrainAI chose is
    recorded in ``resolved_db_tables`` and printed in the report.
    """
    virtual = [
        entry.name
        for entry in tables
        if entry.sql.lstrip().upper().startswith(_SQLITE_VIRTUAL_PREAMBLE)
    ]
    return {
        entry.name
        for entry in tables
        if any(entry.name != owner and entry.name.startswith(owner + "_") for owner in virtual)
    }


def _cut_point(buffer: str, limit: int) -> int:
    """Where to split an over-long buffer: prefer a paragraph, then a line.

    Cutting mid-sentence is acceptable -- the model sees a continuous token
    stream regardless -- but cutting at a natural boundary keeps the
    per-document length statistics in the dataset report meaningful.
    """
    window = buffer[:limit]
    for separator in ("\n\n", "\n", " "):
        index = window.rfind(separator)
        # Boundaries in the first half are ignored: turning a 64 KiB window into a
        # 200-byte document plus a remainder is worse than an arbitrary cut.
        if index > limit // 2:
            return index + len(separator)
    return limit


def _offset_basis(encoding: str) -> tuple[str, int]:
    """Return a BOM-free codec for counting bytes, and the source's BOM length.

    Re-encoding an emitted piece is how its byte length is recovered, so the
    codec used for that must not prepend a BOM to every piece. The byte counts of
    the fixed-endian variants match either endianness.
    """
    name = _canonical(encoding)
    return _BOMLESS_EQUIVALENT.get(name, encoding), _BOM_LENGTHS.get(name, 0)


def _is_hidden(path: Path, root: Path) -> bool:
    """True for dotfiles and for anything inside a dot-directory below ``root``."""
    try:
        parts = path.relative_to(root).parts
    except ValueError:  # pragma: no cover - path is always under root here
        parts = path.parts
    return any(part.startswith(".") for part in parts)


def _classify_name(name: str) -> tuple[Kind | None, Compression]:
    """Which reader a file name asks for, and any codec on the end of it.

    Shared by files on disk and by archive members, so the two cannot drift apart
    -- a ``notes.txt.gz`` inside a zip has to be read the same way as one beside it.

    A ``.docx`` is classified here rather than left to the archive branch in
    :meth:`Ingestor.discover`, which is what stops one being walked as the zip it
    technically is and handing back its XML as documents.
    """
    stem, compression = strip_compression(name)
    kind: Kind | None = None
    if stem.endswith(JSONL_SUFFIXES):
        kind = "jsonl"
    elif stem.endswith(JSON_SUFFIXES):
        kind = "json"
    elif stem.endswith(CSV_SUFFIXES):
        kind = "csv"
    elif stem.endswith(DOCX_SUFFIXES):
        kind = "docx"
    elif stem.endswith(SQLITE_SUFFIXES):
        kind = "sqlite"
    elif stem.endswith(TEXT_SUFFIXES):
        kind = "text"
    return kind, compression


def _docx_body_text(root: ElementTree.Element) -> str:
    """Walk a Word XML part once, in document order, collecting what it displays.

    Iterative rather than recursive, and one pass rather than one per paragraph,
    for two reasons that are both about what a downloaded file can do:

    - **Depth.** expat parses nesting iteratively, so a document nested ten thousand
      elements deep parses fine and would then overflow the stack of any recursive
      walk over it. An explicit stack has no such limit.
    - **Duplication.** Visiting each element exactly once is what makes the result
      *exactly* the document's text. A tempting shortcut -- iterate every ``w:p``
      and take the ``w:t`` beneath it -- silently doubles anything nested, because a
      text box lives inside a paragraph and holds paragraphs of its own.

    Verified against four real Word documents: the non-whitespace characters this
    returns are exactly the concatenation of every ``w:t`` in the part, none missing
    and none twice.
    """
    out: list[str] = []
    stack = [root]
    while stack:
        element = stack.pop()
        tag = element.tag
        if tag == _DOCX_TEXT:
            # ``xml:space="preserve"`` needs no handling: ElementTree reports
            # character data verbatim, so the spaces Word marked are already here.
            out.append(element.text or "")
            continue
        if tag in _DOCX_SKIPPED:
            continue
        separator = _DOCX_BREAKS.get(tag)
        if separator is not None:
            out.append(separator)
        # Reversed, because popping from the end of the stack would otherwise walk
        # the children backwards and the whole point of this is document order.
        stack.extend(reversed(element))
    # A leading break comes from the first paragraph; trailing ones from the empty
    # paragraphs Word leaves at the end of a section. Neither is content.
    return "".join(out).strip("\n")


def _is_extensionless(name: str) -> bool:
    """True for a name with nothing after the last dot to classify it by.

    A codec suffix is stripped first, so ``corpus.gz`` counts as extensionless: the
    codec says how to decompress, not what the bytes are. A leading dot is not an
    extension either -- ``.bashrc`` has none -- but hidden files never reach here.
    """
    stem, _ = strip_compression(name)
    return "." not in stem.lstrip(".")


def _is_hidden_member(member: str) -> bool:
    """The directory walk's dotfile rule, applied inside an archive.

    ``__MACOSX/`` is named separately because it is the one case where the
    directory is the signal: everything under it is AppleDouble metadata that a
    Mac adds to every zip it makes, and its files start with ``._`` anyway.
    """
    if member.startswith(_MACOS_METADATA_PREFIX):
        return True
    return any(part.startswith(".") for part in member.split("/") if part)


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    """True for a zip entry stored as a symlink rather than as a file.

    Its stored bytes are the target path, not the target's contents, so reading it
    would put a file name into the corpus and call it a document.
    """
    if info.create_system != 3:  # Unix; symlink bits are meaningless otherwise.
        return False
    return (info.external_attr >> 16) & 0o170000 == 0o120000


# Both of these once fell back to the name as typed when the registry did not know
# it, which is what let a misspelt --encoding reach bytes.decode and raise there:
# an unknown name is not a wide codec, so the format guards waved it through.
# IngestOptions rejects the name instead, and the only strings that arrive here now
# are one it accepted or one of the codecs named in _BOMS.
def _canonical(name: str) -> str:
    return codecs.lookup(name).name


def _same_codec(left: str, right: str) -> bool:
    return codecs.lookup(left).name == codecs.lookup(right).name
