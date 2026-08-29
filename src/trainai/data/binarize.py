"""Encode documents to binary token shards with a checksummed manifest.

The output of this module *is* the dataset: a directory of little-endian integer
arrays plus a ``manifest.json`` describing them. Nothing else needs to be
consulted at training time, and the format is simple enough to read with
:func:`numpy.memmap` in one line if TrainAI is unavailable.

Four decisions worth stating, because each one has an alternative that looks
easier and is wrong:

**Explicit little-endian.** Shards are written as ``<u2``/``<u4``, not as the
platform's native order. Native order would make the same corpus hash differently
on a big-endian machine and, worse, would let a shard written on one machine be
read as garbage on another. Two bytes of care removes a class of bug that would be
very hard to diagnose from a loss curve.

**The train/validation split is decided by hashing document content**, not by
position and not by a shuffled index. Two consequences follow, both wanted: the
split is identical on every machine and every run for a given seed, and two
byte-identical documents always land on the same side, so exact duplicates in the
corpus cannot leak from train into validation and quietly flatter the validation
loss.

**The manifest records both a timestamp and a content hash.** The timestamp makes
a dataset directory self-describing; the content hash covers only the reproducible
parts, so "same input, same seed, same output" can be checked without a timestamp
making every comparison fail.

**Shards are hashed as they are written**, incrementally, rather than by reading
the files back afterwards. On a spinning disk with a multi-gigabyte dataset, the
read-back would be the second-slowest part of preparation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Literal

import numpy as np

from trainai import __version__
from trainai.data.analyze import DatasetReport
from trainai.data.ingest import CUT_BY_MAX_DOC_CHARS, Document, IngestOptions
from trainai.data.tokenizer import ByteLevelBPE
from trainai.data.validate import ValidationResult
from trainai.errors import DatasetError, DatasetFormatError, json_literal, json_type_name

__all__ = [
    "DEFAULT_SHARD_TOKENS",
    "DEFAULT_VAL_FRACTION",
    "MANIFEST_NAME",
    "SPLITS",
    "TOKENIZER_NAME",
    "DatasetManifest",
    "ShardInfo",
    "Split",
    "binarize_documents",
    "describe_dataset_layout",
    "token_dtype",
    "verify_dataset",
]

MANIFEST_NAME = "manifest.json"
TOKENIZER_NAME = "tokenizer.json"
FORMAT_NAME = "trainai-dataset"
FORMAT_VERSION = 1

Split = Literal["train", "val"]
SPLITS: tuple[Split, ...] = ("train", "val")

# 64 Mi tokens: 128 MiB per shard at uint16. Small enough that a corrupted or
# interrupted shard is cheap to lose, large enough that a 10 GiB dataset is tens
# of files rather than thousands.
DEFAULT_SHARD_TOKENS = 64 << 20

# 5%, not the 1% common in large-scale work. At the corpus sizes this tool is
# aimed at, 1% of a few megabytes is too little text for validation loss to be
# anything but noise.
DEFAULT_VAL_FRACTION = 0.05

# The ceiling on --val-fraction, exclusive: at or above a half, validation would
# hold more text than training. It is a named constant because it was a bare
# literal in the check and the empty-split hint computed a suggestion that ignored
# it, so the hint told people to pass a value the check then refused.
MAX_VAL_FRACTION = 0.5

# Documents per call into the Rust tokenizer. Batching is what lets it use every
# core; larger batches stop helping and start costing memory.
DEFAULT_BATCH_DOCUMENTS = 512

ProgressCallback = Callable[[int, int], None]
"""Called with ``(documents_done, tokens_written)``. Kept as a plain callable so
this module never imports a UI library."""


def token_dtype(vocab_size: int) -> np.dtype:
    """Smallest little-endian unsigned type that can hold every token id.

    uint16 halves the size of the dataset on disk and in the page cache compared
    with uint32, which matters more than it sounds: the loader's throughput is
    bounded by memory bandwidth, not by arithmetic.
    """
    if vocab_size <= 0:
        raise DatasetError(
            f"vocab_size must be positive, got {vocab_size}.",
            hint="This is a TrainAI bug; a trained tokenizer always has a positive size.",
        )
    return np.dtype("<u2") if vocab_size <= (1 << 16) else np.dtype("<u4")


@dataclass(frozen=True)
class ShardInfo:
    """One shard file. ``sha256`` covers the raw bytes on disk."""

    name: str
    tokens: int
    bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tokens": self.tokens,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


#: What each key the manifest reader uses must hold. Every one of these is written by
#: :meth:`DatasetManifest.to_dict`, so requiring them refuses no manifest TrainAI has
#: ever written -- and a ``float`` key accepts an integer because ``"val_fraction": 0``
#: is what a person types where ``to_dict`` writes ``0.0``.
_MANIFEST_TYPES: dict[str, tuple[type, ...]] = {
    "format_version": (int,),
    "dtype": (str,),
    "vocab_size": (int,),
    "eot_id": (int,),
    "tokenizer_fingerprint": (str,),
    "seed": (int,),
    "val_fraction": (int, float),
    "shard_tokens": (int,),
    "splits": (dict,),
    "totals": (dict,),
    "ingest": (dict,),
    "corpus": (dict,),
    "validation": (dict,),
    "tokenizer": (dict,),
    "created_with": (str,),
    "created_at": (str,),
    "content_hash": (str,),
}

#: The same, for one entry of ``splits``, and for one shard inside it. Both splits are
#: required: ``to_dict`` always writes both, even when ``val`` is empty, so a manifest
#: with one of them absent is not a manifest with an empty split -- it is a file that has
#: lost a block, and reading it as an empty split is a wrong answer given in silence.
_SPLIT_BLOCK_TYPES: dict[str, tuple[type, ...]] = dict.fromkeys(SPLITS, (dict,))
_SPLIT_TYPES: dict[str, tuple[type, ...]] = {"documents": (int,), "shards": (list,)}
_SHARD_TYPES: dict[str, tuple[type, ...]] = {
    "name": (str,),
    "tokens": (int,),
    "bytes": (int,),
    "sha256": (str,),
}

#: What each accepted type tuple is called in a refusal.
_TYPE_NAMES: dict[tuple[type, ...], str] = {
    (int,): "an integer",
    (int, float): "a number",
    (str,): "a string",
    (dict,): "an object",
    (list,): "an array",
}

_REBUILD_HINT = (
    "Re-run `trainai data prepare` to rebuild the dataset; the shards alone are not "
    "enough to recover it."
)


@dataclass(frozen=True)
class _ManifestReader:
    """Reads a manifest's JSON, refusing anything that is not one.

    A small class rather than free functions so that every refusal names the file
    without each call site threading the path through. The rule it enforces is the one
    ``serialise.py`` states for a checkpoint's configuration blocks: **a key the reader
    uses is required, and must hold the right type.** It is applied here for the same
    reason and against the same measurements -- filling a key from a default rebuilds a
    description of a dataset that is not the dataset on disk, and how that surfaces
    depends entirely on which key it was:

    ``dtype``
        Decides how the shard bytes are interpreted. ``<u2`` read as ``<u4`` is not an
        error, it is half as many tokens, each one a different token.
    ``vocab_size``
        Sizes the model's embedding, and is what ``trainai train`` checks a plan
        against.
    ``tokenizer_fingerprint``, ``content_hash``
        Default to ``""``, and an empty one *skips* the resume-safety and
        dataset-identity comparisons in ``eval/perplexity.py`` rather than failing them.
        A defaulted key here silently disables a check.
    ``seed``, ``val_fraction``, ``shard_tokens``
        Reported to the user by ``trainai data inspect`` as facts about how the dataset
        was built. A default is a wrong answer presented as a recorded one.

    Before this, no way a manifest can disagree with its own format was refused.
    Measured on a copy of a real dataset, one edit per case: of twenty-seven edits,
    twenty arrived as a Python traceback and exit 1 -- ``AttributeError`` for a document
    or block that is not an object, ``KeyError`` for an absent key, ``ValueError`` and
    ``TypeError`` for a value of the wrong type -- from a tool that documents exit 3 for
    a dataset it cannot read. The other seven loaded silently, which is worse: a manifest
    missing its ``val`` entry became a dataset with no validation split, and a shard
    whose ``sha256`` was an array became the string ``"[]"``.
    """

    path: Path

    def refuse(self, message: str, **details: Any) -> DatasetFormatError:
        return DatasetFormatError(
            f"{self.path} {message}",
            hint=_REBUILD_HINT,
            details={"path": str(self.path), **details},
        )

    def object_at(self, raw: Any, subject: str) -> dict[str, Any]:
        """``raw`` as a mapping, or a refusal naming what it holds instead."""
        if not isinstance(raw, dict):
            raise self.refuse(
                f"has {subject} that is {json_type_name(raw)}, not an object.",
                subject=subject,
                found=json_type_name(raw),
            )
        return raw

    def value(
        self, holder: dict[str, Any], key: str, types: dict[str, tuple[type, ...]], subject: str
    ) -> Any:
        """``holder[key]``, required to be present and to hold an accepted type."""
        accepted = types[key]
        if key not in holder:
            raise self.refuse(f"is missing {subject}{key}.", missing=f"{subject}{key}")
        found = holder[key]
        # bool is an int subclass, so `isinstance` alone would accept true as a count.
        ok = bool in accepted if isinstance(found, bool) else isinstance(found, accepted)
        if not ok:
            raise self.refuse(
                f"has {subject}{key} as {json_literal(found)}, which is not "
                f"{_TYPE_NAMES[accepted]}.",
                field=f"{subject}{key}",
                found=json_type_name(found),
            )
        return found

    def shard(self, raw: Any, split: str, index: int) -> ShardInfo:
        """One shard entry, checked field by field.

        Built here rather than in a ``ShardInfo.from_dict`` because only the reader knows
        which file and which position in which split to name when an entry is wrong, and
        a refusal that says a shard is malformed without saying which one is barely
        better than the ``KeyError`` it replaces.
        """
        subject = f"splits.{split}.shards[{index}]."
        entry = self.object_at(raw, subject.rstrip("."))
        return ShardInfo(
            name=self.value(entry, "name", _SHARD_TYPES, subject),
            tokens=self.value(entry, "tokens", _SHARD_TYPES, subject),
            bytes=self.value(entry, "bytes", _SHARD_TYPES, subject),
            sha256=self.value(entry, "sha256", _SHARD_TYPES, subject),
        )


@dataclass
class DatasetManifest:
    """Everything needed to train on, verify, or reproduce a prepared dataset."""

    dtype: str
    vocab_size: int
    eot_id: int
    tokenizer_fingerprint: str
    seed: int
    val_fraction: float
    shard_tokens: int
    shards: dict[str, list[ShardInfo]] = field(default_factory=dict)
    documents: dict[str, int] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)
    ingest: dict[str, Any] = field(default_factory=dict)
    corpus: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    tokenizer: dict[str, Any] = field(default_factory=dict)
    created_with: str = f"trainai {__version__}"
    created_at: str = ""
    format: str = FORMAT_NAME
    format_version: int = FORMAT_VERSION
    content_hash: str = ""
    root: Path | None = None

    # -- derived ----------------------------------------------------------- #

    def tokens(self, split: Split) -> int:
        return sum(shard.tokens for shard in self.shards.get(split, ()))

    @property
    def total_tokens(self) -> int:
        return sum(self.tokens(split) for split in SPLITS)

    @property
    def numpy_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    def shard_paths(self, split: Split) -> list[Path]:
        base = self.root or Path()
        return [base / shard.name for shard in self.shards.get(split, ())]

    # -- serialisation ----------------------------------------------------- #

    def reproducible_parts(self) -> dict[str, Any]:
        """The subset of the manifest that must not change between identical runs.

        Deliberately excludes ``created_at``, ``created_with`` and the corpus
        report's timing-free-but-noisy extras, so that the content hash answers
        exactly one question: did the same input produce the same bytes?
        """
        return {
            "format": self.format,
            "format_version": self.format_version,
            "dtype": self.dtype,
            "vocab_size": self.vocab_size,
            "eot_id": self.eot_id,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "seed": self.seed,
            "val_fraction": self.val_fraction,
            "shard_tokens": self.shard_tokens,
            "documents": dict(sorted(self.documents.items())),
            "totals": dict(sorted(self.totals.items())),
            "shards": {
                split: [shard.to_dict() for shard in self.shards.get(split, ())] for split in SPLITS
            },
            "ingest": self.ingest,
        }

    def compute_content_hash(self) -> str:
        canonical = json.dumps(
            self.reproducible_parts(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "format_version": self.format_version,
            "created_with": self.created_with,
            "created_at": self.created_at,
            "content_hash": self.content_hash,
            "dtype": self.dtype,
            "byte_order": "little",
            "bytes_per_token": self.numpy_dtype.itemsize,
            "vocab_size": self.vocab_size,
            "eot_id": self.eot_id,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "tokenizer": self.tokenizer,
            "seed": self.seed,
            "val_fraction": self.val_fraction,
            "shard_tokens": self.shard_tokens,
            "documents": self.documents,
            "totals": self.totals,
            "splits": {
                split: {
                    "documents": self.documents.get(split, 0),
                    "tokens": self.tokens(split),
                    "shards": [shard.to_dict() for shard in self.shards.get(split, ())],
                }
                for split in SPLITS
            },
            "ingest": self.ingest,
            "corpus": self.corpus,
            "validation": self.validation,
        }

    def write(self, directory: str | Path) -> Path:
        """Write ``manifest.json`` atomically. Returns its path.

        Atomic because the manifest is the only thing that says which shards are
        valid. A half-written manifest beside a complete set of shards would turn
        a recoverable interruption into a corrupt dataset.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.content_hash = self.compute_content_hash()
        path = directory / MANIFEST_NAME
        temporary = path.with_name(path.name + ".tmp")
        payload = json.dumps(self.to_dict(), indent=2, ensure_ascii=True, sort_keys=False)
        temporary.write_text(payload + "\n", encoding="utf-8", newline="\n")
        temporary.replace(path)
        self.root = directory
        return path

    @classmethod
    def load(cls, directory: str | Path) -> DatasetManifest:
        """Read a prepared dataset's manifest, refusing by name anything that is not one.

        Every failure here is a :class:`DatasetFormatError`, exit 3, naming the file and
        the key -- never a traceback. See :class:`_ManifestReader` for which keys are
        required and what each one costs when it is silently defaulted instead.
        """
        directory = Path(directory)
        path = directory / MANIFEST_NAME if directory.is_dir() else directory
        if not path.exists():
            raise DatasetFormatError(
                f"No {MANIFEST_NAME} in {directory}",
                hint=(
                    "--data must point at a directory produced by `trainai data prepare`, "
                    "not at a raw corpus. Run `trainai data prepare <corpus> --out "
                    f"{directory}` first."
                ),
                details={"path": str(path)},
            )
        reader = _ManifestReader(path)
        try:
            # UnicodeDecodeError is a ValueError, not an OSError, and was not caught: a
            # manifest saved by an editor that wrote Latin-1 arrived as a traceback.
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatasetFormatError(
                f"{path} could not be read as JSON.",
                hint=(
                    "The manifest is corrupt. Re-run `trainai data prepare` to rebuild "
                    "the dataset; the shards alone are not enough to recover it."
                ),
                details={"path": str(path), "reason": str(exc)},
            ) from exc

        # Valid JSON that is not an object has no `.get`, so the format check below is
        # not the first thing that can fail -- it only looked that way.
        top = reader.object_at(raw, "a manifest")
        if top.get("format") != FORMAT_NAME:
            raise DatasetFormatError(
                f"{path} is not a TrainAI dataset manifest (format={top.get('format')!r}).",
                hint="Point --data at a directory created by `trainai data prepare`.",
                details={"path": str(path), "format": top.get("format")},
            )
        version = reader.value(top, "format_version", _MANIFEST_TYPES, "")
        if version > FORMAT_VERSION:
            raise DatasetFormatError(
                f"{path} was written by a newer TrainAI (format version {version}, "
                f"this build understands {FORMAT_VERSION}).",
                hint=(
                    "Use the TrainAI that wrote it, or re-create this dataset from "
                    "your corpus with `trainai data prepare`, which writes a version "
                    "this build reads."
                ),
                details={"found": version, "supported": FORMAT_VERSION},
            )

        def key(name: str) -> Any:
            return reader.value(top, name, _MANIFEST_TYPES, "")

        blocks = key("splits")
        splits = {
            split: reader.value(blocks, split, _SPLIT_BLOCK_TYPES, "splits.") for split in SPLITS
        }
        shards = {
            split: [
                reader.shard(entry, split, index)
                for index, entry in enumerate(
                    reader.value(splits[split], "shards", _SPLIT_TYPES, f"splits.{split}.")
                )
            ]
            for split in SPLITS
        }
        manifest = cls(
            dtype=key("dtype"),
            vocab_size=key("vocab_size"),
            eot_id=key("eot_id"),
            tokenizer_fingerprint=key("tokenizer_fingerprint"),
            seed=key("seed"),
            val_fraction=float(key("val_fraction")),
            shard_tokens=key("shard_tokens"),
            shards=shards,
            documents={
                split: reader.value(splits[split], "documents", _SPLIT_TYPES, f"splits.{split}.")
                for split in SPLITS
            },
            totals=key("totals"),
            ingest=key("ingest"),
            corpus=key("corpus"),
            validation=key("validation"),
            tokenizer=key("tokenizer"),
            created_with=key("created_with"),
            created_at=key("created_at"),
            format_version=version,
            content_hash=key("content_hash"),
        )
        manifest.root = path.parent
        return manifest

    def tokenizer_path(self) -> Path:
        return (self.root or Path()) / TOKENIZER_NAME


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


class _ShardWriter:
    """Appends token arrays to numbered shard files, hashing as it goes."""

    def __init__(self, directory: Path, prefix: str, dtype: np.dtype, shard_tokens: int) -> None:
        self._directory = directory
        self._prefix = prefix
        self._dtype = dtype
        self._limit = shard_tokens
        self._shards: list[ShardInfo] = []
        self._handle: BinaryIO | None = None
        self._digest = hashlib.sha256()
        self._tokens = 0
        self._index = 0

    @property
    def shards(self) -> tuple[ShardInfo, ...]:
        return tuple(self._shards)

    def write(self, ids: np.ndarray) -> None:
        remaining = ids
        while remaining.size:
            if self._handle is None:
                self._open()
            room = self._limit - self._tokens
            chunk, remaining = remaining[:room], remaining[room:]
            payload = chunk.astype(self._dtype, copy=False).tobytes()
            assert self._handle is not None  # opened above
            self._handle.write(payload)
            self._digest.update(payload)
            self._tokens += int(chunk.size)
            if self._tokens >= self._limit:
                self._close_current()

    def close(self) -> tuple[ShardInfo, ...]:
        if self._handle is not None:
            self._close_current()
        return self.shards

    def _open(self) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        # A `with` block is wrong here by design: the handle stays open across many
        # write() calls and is closed when the shard fills or the writer finishes.
        # binarize_documents() closes every writer in its own finally-equivalent.
        self._handle = open(self._current_path(), "wb")  # noqa: SIM115
        self._digest = hashlib.sha256()
        self._tokens = 0

    def _close_current(self) -> None:
        assert self._handle is not None
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._handle = None
        path = self._current_path()
        self._shards.append(
            ShardInfo(
                name=path.name,
                tokens=self._tokens,
                bytes=path.stat().st_size,
                sha256=self._digest.hexdigest(),
            )
        )
        self._index += 1

    def _current_path(self) -> Path:
        return self._directory / f"{self._prefix}_{self._index:05d}.bin"


def binarize_documents(
    documents: Iterable[Document],
    out_dir: str | Path,
    tokenizer: ByteLevelBPE,
    *,
    seed: int = 0,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    shard_tokens: int = DEFAULT_SHARD_TOKENS,
    batch_documents: int = DEFAULT_BATCH_DOCUMENTS,
    report: DatasetReport | None = None,
    validation: ValidationResult | None = None,
    ingest_options: IngestOptions | None = None,
    sources: list[dict[str, Any]] | None = None,
    progress: ProgressCallback | None = None,
) -> DatasetManifest:
    """Tokenize ``documents`` into shards under ``out_dir`` and write the manifest.

    Every document is followed by the end-of-text token, so the shards are one
    continuous stream that the loader can slice at any offset.

    Args:
        documents: Document stream, consumed once.
        out_dir: Destination directory. Created if absent.
        tokenizer: A trained tokenizer. Its ``tokenizer.json`` is written here too,
            because a dataset without the tokenizer that produced it is unusable.
        seed: Mixed into the split hash. The same seed and corpus always give the
            same split.
        val_fraction: Share of documents held out. ``0`` disables the validation
            split entirely.
        shard_tokens: Tokens per shard file.
        batch_documents: Documents per tokenizer call.
        report: Corpus measurements, recorded verbatim in the manifest.
        validation: Validation verdict, recorded verbatim in the manifest.
        ingest_options: Ingest settings, recorded so the run can be reproduced.
        sources: Source-file records from :meth:`Ingestor.discover`.
        progress: Called with ``(documents_done, tokens_written)`` as work proceeds.

    Raises:
        DatasetError: If ``val_fraction`` is out of range, no documents arrived, or
            the split left the validation set empty.
    """
    out_dir = Path(out_dir)
    if not 0.0 <= val_fraction < MAX_VAL_FRACTION:
        raise DatasetError(
            f"--val-fraction must be at least 0 and below {MAX_VAL_FRACTION}, got {val_fraction}.",
            hint="0.05 holds out 5% for validation, which suits most corpora.",
            details={"val_fraction": val_fraction, "maximum": MAX_VAL_FRACTION},
        )
    if shard_tokens < 1024:
        raise DatasetError(
            f"--shard-tokens must be at least 1024, got {shard_tokens}.",
            hint="Leave it unset unless you have a reason; the default is 64Mi tokens.",
            details={"shard_tokens": shard_tokens},
        )

    dtype = token_dtype(tokenizer.vocab_size)
    seed_key = seed.to_bytes(8, "little", signed=seed < 0)
    # Clear any previous dataset before writing a byte of the new one. Shard names
    # are derived from the shard size, so a re-run with a larger --shard-tokens
    # writes fewer files and would otherwise leave the extra ones behind: valid
    # according to the new manifest, invisible to `verify_dataset`, and still
    # occupying disk. The caller has already decided to replace what is here --
    # `trainai data prepare` refuses without --force -- so nothing is being
    # discarded that the previous manifest still described.
    _remove_shards(out_dir)
    writers = {
        "train": _ShardWriter(out_dir, "train", dtype, shard_tokens),
        "val": _ShardWriter(out_dir, "val", dtype, shard_tokens),
    }
    counts = {"train": 0, "val": 0}
    tokens_written = 0
    documents_done = 0
    eot = tokenizer.eot_id
    # The extremes of the split hash, kept so the two empty-split refusals below can name
    # the fraction that would have worked instead of one that probably would. Sentinels
    # sit outside the range a fraction can take, and neither refusal is reachable without
    # at least one document having updated them.
    lowest_fraction = 1.0
    highest_fraction = -1.0

    try:
        for batch in _batched(documents, batch_documents):
            encoded = tokenizer.encode_batch([document.text for document in batch])
            # Group by split first so each split gets one array per batch rather
            # than one per document; concatenating 512 tiny arrays costs more than
            # the tokenization did.
            grouped: dict[Split, list[list[int]]] = {"train": [], "val": []}
            for document, ids in zip(batch, encoded, strict=True):
                if val_fraction <= 0.0:
                    # No hash is computed when validation is off, so the extremes stay at
                    # their sentinels -- and neither refusal that reads them can fire,
                    # since every document trains.
                    split: Split = "train"
                else:
                    fraction = _split_fraction(document.text, seed_key)
                    lowest_fraction = min(lowest_fraction, fraction)
                    highest_fraction = max(highest_fraction, fraction)
                    split = _side(fraction, val_fraction)
                grouped[split].append([*ids, eot])
                counts[split] += 1
            for split, groups in grouped.items():
                if not groups:
                    continue
                flat = np.fromiter(
                    (token for group in groups for token in group),
                    dtype=dtype,
                    count=sum(len(group) for group in groups),
                )
                writers[split].write(flat)
                tokens_written += int(flat.size)
            documents_done += len(batch)
            if progress is not None:
                progress(documents_done, tokens_written)

        shards = {split: list(writer.close()) for split, writer in writers.items()}
    except BaseException:
        # A partially written dataset with no manifest is worse than nothing: the
        # next run would find stale shards. Close handles, then remove them.
        for writer in writers.values():
            with suppress(Exception):
                writer.close()
        _remove_shards(out_dir)
        raise

    if documents_done == 0:
        _remove_shards(out_dir)
        raise DatasetError(
            "No documents reached the tokenizer, so no dataset was written.",
            hint=(
                "Run `trainai data inspect <corpus>` to see what was found. Empty files "
                "and documents below --min-doc-chars are dropped during ingest."
            ),
            details={"out_dir": str(out_dir)},
        )

    if val_fraction > 0 and counts["val"] == 0:
        _remove_shards(out_dir)
        # The advice here has to be advice the check at the top of this function will
        # accept, it has to be honest about what the split does, and the number it names
        # has to work on the corpus in front of it. It was none of the three.
        #
        # `max(1.0 / documents_done, 0.02)` ignored MAX_VAL_FRACTION, and this error
        # fires most often on a corpus of one or two documents -- exactly where that
        # formula says 1.00 or 0.50, both of which the check then refuses. Following
        # the hint produced "--val-fraction must be at least 0 and below 0.5".
        #
        # It also never looked at the fraction in use, so past about twenty-six
        # documents it named a number at or below what the user had already passed --
        # and called it a raise. And targeting one expected document is a coin flip by
        # construction: measured over 2,000 seeds, `max(1/n, 0.02)` left the validation
        # split empty for 29% to 37% of seeds at every size from 3 to 60 documents.
        #
        # Inverting the distribution fixed the arithmetic but kept the shape of the
        # mistake: a fraction chosen to fail 5% of the time still fails, and it fails on
        # the corpora that get here. Measured on a 3-document text file at the default
        # fraction, it named 0.49 and 0.49 emptied the split again -- two refusals for
        # one mistake. The split is a deterministic function of content and seed, so the
        # answer was never a probability: `lowest_fraction` is the smallest hash the run
        # just produced, and any fraction above it selects that document.
        more_documents = _how_to_get_more_documents(sources)
        suggested = _suggested_val_fraction(lowest_fraction)
        if documents_done < 2:
            hint = (
                "One document cannot land on both sides of a split, whatever "
                f"--val-fraction says. To fix it, {more_documents}. Or use "
                "--val-fraction 0 to train without validation, accepting that there "
                "will be no held-out loss."
            )
        elif suggested is not None:
            hint = (
                "Each document is assigned by hash, independently, so --val-fraction is "
                f"a per-document chance rather than a quota. --val-fraction {suggested:.2f} "
                f"puts at least one of these {documents_done} documents in validation -- "
                f"exactly, for this corpus at --seed {seed}, not as better odds. Change "
                f"either and the number changes with it, so the durable fix is to "
                f"{more_documents}. Use --val-fraction 0 to train without validation, "
                "accepting that there will be no held-out loss."
            )
        else:
            hint = (
                f"No --val-fraction the range check accepts can select any of these "
                f"{documents_done} documents: the closest one sits at "
                f"{lowest_fraction:.2f}, and the ceiling is {MAX_VAL_FRACTION}. Each "
                "document is assigned by hash, independently, and at "
                f"{documents_done} documents this fraction leaves the validation split "
                f"empty {_empty_val_chance(documents_done, val_fraction)} of the time -- "
                "this run is one of those. A different --seed reshuffles at the same "
                f"odds; to stop relying on the odds, {more_documents}. Use "
                "--val-fraction 0 to train without validation, accepting that there "
                "will be no held-out loss."
            )
        raise DatasetError(
            f"The validation split is empty: none of the {documents_done} documents "
            f"was selected at --val-fraction {val_fraction}.",
            hint=hint,
            details={
                "documents": documents_done,
                "val_fraction": val_fraction,
                "lowest_split_fraction": lowest_fraction,
            },
        )

    if counts["train"] == 0:
        _remove_shards(out_dir)
        # The mirror of the check above, and it was missing. `_assign_split` hashes
        # each document independently, so it is a weighted coin per document and not a
        # quota: with few documents it can send *every* one of them to validation.
        # Measured, one document at --val-fraction 0.49 does it for 37 of 60 seeds.
        #
        # Nothing noticed. The dataset was written with a train split of 0 documents
        # and 0 tokens, `data prepare` exited 0 reporting "Wrote ... 601 tokens from 1
        # documents", and `trainai train` then refused it downstream with "a dataset
        # without a train split is a bug", advising a re-run of `data prepare` -- which
        # is deterministic, and reproduces the empty split exactly.
        more_documents = _how_to_get_more_documents(sources)
        lower_to = _val_fraction_that_keeps_a_train_document(highest_fraction)
        if documents_done < 2:
            hint = (
                "One document cannot land on both sides of a split. Use "
                f"--val-fraction 0 to train without validation, or {more_documents}."
            )
        elif lower_to is not None:
            # "Lower --val-fraction" on its own left the user to guess, on a split that
            # gives the same answer to every guess above this number.
            hint = (
                f"--val-fraction {lower_to:.2f} keeps at least one of these "
                f"{documents_done} documents in training -- exactly, for this corpus at "
                f"--seed {seed}. Or {more_documents}. Re-running unchanged will not help: "
                "each document is assigned by hash, so the same corpus at the same --seed "
                "always splits the same way."
            )
        else:
            hint = (
                f"Lower --val-fraction, or {more_documents}. Re-running unchanged will "
                "not help: each document is assigned by hash, so the same corpus at the "
                "same --seed always splits the same way."
            )
        raise DatasetError(
            f"The training split is empty: all {documents_done} documents were "
            f"assigned to validation at --val-fraction {val_fraction}.",
            hint=hint,
            details={
                "documents": documents_done,
                "val_fraction": val_fraction,
                "seed": seed,
                "highest_split_fraction": highest_fraction,
            },
        )

    tokenizer.save(out_dir / TOKENIZER_NAME)
    manifest = DatasetManifest(
        dtype=dtype.str,
        vocab_size=tokenizer.vocab_size,
        eot_id=eot,
        tokenizer_fingerprint=tokenizer.fingerprint(),
        seed=seed,
        val_fraction=val_fraction,
        shard_tokens=shard_tokens,
        shards=shards,
        documents=dict(counts),
        totals={
            "documents": documents_done,
            "tokens": tokens_written,
            "eot_tokens": documents_done,
            "chars": report.total_chars if report else 0,
            "utf8_bytes": report.total_utf8_bytes if report else 0,
        },
        ingest={
            "options": ingest_options.to_dict() if ingest_options else {},
            "sources": sources or [],
        },
        corpus=report.to_dict() if report else {},
        validation=validation.to_dict() if validation else {},
        tokenizer=tokenizer.to_dict(),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    manifest.write(out_dir)
    return manifest


def verify_dataset(directory: str | Path, *, deep: bool = True) -> DatasetManifest:
    """Load a manifest and check the shards match it.

    ``deep=False`` checks only that every shard exists with the recorded size,
    which is instant. ``deep=True`` re-hashes every byte, which is the only way to
    catch silent corruption -- a truncated file has the wrong size, but a file with
    a flipped bit does not.
    """
    manifest = DatasetManifest.load(directory)
    root = manifest.root or Path(directory)
    itemsize = manifest.numpy_dtype.itemsize

    for split in SPLITS:
        for shard in manifest.shards.get(split, ()):
            path = root / shard.name
            if not path.exists():
                raise DatasetFormatError(
                    f"Shard {shard.name} is listed in the manifest but missing.",
                    hint=(
                        "Re-run `trainai data prepare` to rebuild the dataset. Do not "
                        "move or rename files inside a prepared dataset directory."
                    ),
                    details={"path": str(path), "split": split},
                )
            size = path.stat().st_size
            expected = shard.tokens * itemsize
            if size != expected:
                raise DatasetFormatError(
                    f"Shard {shard.name} is {size} bytes; the manifest says {expected} "
                    f"({shard.tokens} tokens x {itemsize} bytes).",
                    hint=(
                        "The file was truncated or partly overwritten, most likely by an "
                        "interrupted copy or a full disk. Re-run `trainai data prepare`."
                    ),
                    details={"path": str(path), "size": size, "expected": expected},
                )
            if deep and _sha256_file(path) != shard.sha256:
                raise DatasetFormatError(
                    f"Shard {shard.name} does not match its recorded checksum.",
                    hint=(
                        "The file's contents changed since preparation. Training on it "
                        "would use data the manifest does not describe. Re-run "
                        "`trainai data prepare`."
                    ),
                    details={"path": str(path), "expected_sha256": shard.sha256},
                )

    return manifest


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _empty_val_chance(documents: int, val_fraction: float) -> str:
    """How often this fraction leaves the validation split empty, in words.

    A document goes to validation when :func:`_split_fraction` puts it below
    ``val_fraction``, and that hash is uniform, so the chance that none of ``documents``
    goes there is exactly ``(1 - f) ** n`` -- algebra, not an estimate. Checked against the
    real function over 2,000 seeds at ten corpus sizes: predicted and measured agreed to
    within a point everywhere (5 documents, 4.6% predicted, 4.8% measured; 60 documents,
    4.6% and 5.0%).

    Only ever quoted about *reshuffling*, never about a fraction to raise to, because it is
    unconditional and every caller here is already conditioned on a failure: every document
    came in at or above the fraction in use. At two documents that emptied the split at
    0.40, a raise to 0.49 fails 72% of the time, not the 26% this gives. Raising is answered
    by measurement instead, in :func:`_suggested_val_fraction`.

    Kept as words rather than a number so the caller cannot print "about a 0% chance" of
    something that just happened, which is where rounding a small probability leads.
    """
    percent = 100.0 * (1.0 - val_fraction) ** documents
    return "under 1%" if percent < 1.0 else f"about {round(percent)}%"


def _suggested_val_fraction(lowest_fraction: float) -> float | None:
    """The lowest ``--val-fraction`` that *this corpus* satisfies, or ``None``.

    ``lowest_fraction`` is the smallest value :func:`_split_fraction` produced during the
    run that just failed. A document goes to validation when its fraction is below
    ``--val-fraction``, so any fraction above that minimum selects at least that one
    document -- not probably, certainly, for this corpus at this seed. Returns the
    smallest two-place value above it, or ``None`` when that is not below
    :data:`MAX_VAL_FRACTION`, where the range check would refuse it.

    This replaced a suggestion computed from the document count alone, which aimed at a
    5% chance of failing again and so could name a number this corpus does not satisfy.
    Measured on a 3-document corpus at the default fraction: it named 0.49, honestly
    quoting "about 13%", and 0.49 emptied the validation split again -- the tool spent the
    user's next attempt on a draw it was holding the answer to. Two earlier formulas
    failed the same way for other reasons: ``max(1.0 / documents, 0.02)`` ignored
    :data:`MAX_VAL_FRACTION` and named values the range check then refused, and named 0.02
    to a user who had passed 0.05 while calling it a raise.

    Rounded to two places, because that is what a person types. ``floor`` then ``+ 0.01``
    rather than ``ceil``: for a minimum that is already a round two-place value ``ceil``
    returns the minimum itself, which does not select it -- the comparison is strict.
    """
    suggested = (math.floor(lowest_fraction * 100.0) + 1) / 100.0
    return suggested if suggested < MAX_VAL_FRACTION else None


def _val_fraction_that_keeps_a_train_document(highest_fraction: float) -> float | None:
    """The highest ``--val-fraction`` that leaves one document in training, or ``None``.

    The mirror of :func:`_suggested_val_fraction`. Every document went to validation, so
    every fraction came in below ``--val-fraction``; a document trains when its fraction
    is at or above it, so anything down to the largest fraction seen keeps that one.

    ``None`` below 0.01, where the answer rounds to zero: 0 is a different instruction --
    it turns validation off rather than shrinking it -- and the hint offers that
    separately, in words, so naming it here as a number would read as a third option that
    happens to be the second.
    """
    highest = math.floor(highest_fraction * 100.0) / 100.0
    return highest if highest >= 0.01 else None


#: What one document is, for each kind ``max_doc_chars`` cannot cut into more. Used to
#: name the real fix in the two empty-split hints, whose advice is otherwise inert on a
#: record-oriented corpus. Kept beside :data:`~trainai.data.ingest.CUT_BY_MAX_DOC_CHARS`
#: in spirit: that set says which readers cut, this says what a record is when they do
#: not, and a test asserts every non-cutting kind appears here.
_ONE_DOCUMENT_IS: dict[str, str] = {
    "jsonl": "one line",
    "json": "one array element",
    "csv": "one row",
    "sqlite": "one database row",
}


def _how_to_get_more_documents(sources: list[dict[str, Any]] | None) -> str:
    """A lowercase fragment naming a way to end up with more documents than this run had.

    Both empty-split hints need one, and before this existed both gave the same answer
    whatever the corpus was: lower ``--max-doc-chars``. Only two readers consult that
    option -- see :data:`~trainai.data.ingest.CUT_BY_MAX_DOC_CHARS` -- a text file being
    cut as it decodes and a ``.docx`` after it is extracted. Everywhere else one document
    is one record, and a lower limit cannot produce more records. Measured on a one-row
    CSV: ``--max-doc-chars 2000`` changed nothing at all, the refusal came back reporting
    the same 1 document, and the tool had spent the user's next attempt for them.

    ``sources`` is the manifest records the CLI passes through
    (:meth:`~trainai.data.ingest.SourceFile.to_dict`), so each carries a ``kind``. A
    library caller may pass ``None`` or an empty list -- documents assembled in Python
    have no readers behind them to ask -- and then neither half can be ruled out, so the
    fragment covers both and says which case each applies to.
    """
    kinds = {str(record.get("kind", "")) for record in sources or ()}
    if kinds & CUT_BY_MAX_DOC_CHARS:
        return "split your text into more documents by lowering --max-doc-chars"
    units = sorted({_ONE_DOCUMENT_IS[kind] for kind in kinds if kind in _ONE_DOCUMENT_IS})
    if not units:
        return (
            "add more documents -- and if they came from text files, lowering "
            "--max-doc-chars splits each file into more"
        )
    which = units[0] if len(units) == 1 else f"{', '.join(units[:-1])} or {units[-1]}"
    return (
        f"add more records to the corpus -- {which} is one document here, and lowering "
        "--max-doc-chars cannot split a record into more"
    )


def _split_fraction(text: str, seed_key: bytes) -> float:
    """Where this document falls in ``[0, 1)``, from its content and the seed alone.

    Factored out of :func:`_assign_split` so the empty-split hints can name the exact
    threshold this corpus needs rather than a probability. The caller keeps the smallest
    and largest value it sees -- two floats, no per-document storage -- which is all the
    advice needs, because the boundary is a comparison against one number.
    """
    digest = hashlib.blake2b(text.encode("utf-8"), key=seed_key, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def _side(fraction: float, val_fraction: float) -> Split:
    """The one place the boundary is decided, so the hints cannot drift from the split.

    Strict ``<``: a document whose fraction equals ``val_fraction`` trains. That is why
    :func:`_suggested_val_fraction` has to name a value strictly *above* the minimum it
    saw rather than the minimum itself.
    """
    return "val" if fraction < val_fraction else "train"


def _assign_split(text: str, seed_key: bytes, val_fraction: float) -> Split:
    """Deterministic content-hash split.

    blake2b keyed with the seed, taken as a 64-bit fraction. Identical text always
    lands on the same side for a given seed, on any platform, in any order.
    """
    if val_fraction <= 0.0:
        return "train"
    return _side(_split_fraction(text, seed_key), val_fraction)


def _batched(items: Iterable[Document], size: int) -> Iterator[list[Document]]:
    batch: list[Document] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _sha256_file(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_shards(directory: Path) -> None:
    """Delete shard files and the manifest, leaving anything else alone.

    The manifest goes too. Re-preparing into an existing directory overwrites its
    shards from the first byte, so a failure part-way through has already
    destroyed the previous dataset; leaving its manifest behind would advertise
    shards that no longer match it. An absent manifest at least says plainly that
    there is no dataset here.

    Deliberately not ``rmtree``: ``--out`` could be pointed at a directory that
    holds other files, and deleting a user's data because their tokenizer failed
    would be indefensible.
    """
    if not directory.exists():
        return
    for pattern in ("train_*.bin", "val_*.bin", MANIFEST_NAME, f"{MANIFEST_NAME}.tmp"):
        for path in directory.glob(pattern):
            with suppress(Exception):
                path.unlink()


def describe_dataset_layout() -> str:
    """Human-readable summary of the on-disk format, for docs and `data inspect`."""
    return (
        "A prepared dataset is a directory containing:\n"
        f"  {MANIFEST_NAME}    what the shards are, with sha256 for each\n"
        f"  {TOKENIZER_NAME}   the tokenizer the shards were encoded with\n"
        "  train_00000.bin  little-endian uint16/uint32 token ids, no header\n"
        "  val_00000.bin    the held-out split, same format\n"
        "\nEach document is followed by the end-of-text token, so the shards form "
        "one continuous token stream."
    )
