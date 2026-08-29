"""Tests for :mod:`trainai.data.binarize` and :mod:`trainai.data.loader`.

These cover the two claims a prepared dataset makes about itself:

* **It is reproducible.** The same corpus, settings and seed produce shards with
  the same sha256 and a manifest with the same content hash. Nothing that varies
  between runs -- a timestamp, a version string -- may leak into that hash.
* **It is exactly the corpus.** Decoding the shards back gives the documents that
  went in, which is the only way to know that nothing was dropped, duplicated or
  reordered between ingest and disk.

Plus the integrity check that only pays for itself when something has already
gone wrong: a flipped bit changes no file size, so only re-hashing finds it.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, get_args

import numpy as np
import pytest

from trainai.data.analyze import analyze_documents
from trainai.data.binarize import (
    MANIFEST_NAME,
    MAX_VAL_FRACTION,
    SPLITS,
    TOKENIZER_NAME,
    DatasetManifest,
    _empty_val_chance,
    _side,
    _suggested_val_fraction,
    _val_fraction_that_keeps_a_train_document,
    binarize_documents,
    describe_dataset_layout,
    token_dtype,
    verify_dataset,
)
from trainai.data.ingest import CUT_BY_MAX_DOC_CHARS, Document, IngestOptions, Ingestor, Kind
from trainai.data.loader import (
    MIN_SEQ_LEN,
    ShardedTokenStream,
    TokenBatcher,
    largest_seq_len,
    open_split,
    window_count,
)
from trainai.data.tokenizer import ByteLevelBPE, train_tokenizer
from trainai.data.validate import validate_corpus
from trainai.errors import DatasetError, DatasetFormatError

VOCAB = 512


def read_corpus(root: Path, **options: object) -> tuple[list[Document], Ingestor]:
    ingestor = Ingestor(IngestOptions(**options))  # type: ignore[arg-type]
    return list(ingestor.documents(ingestor.discover(root))), ingestor


@pytest.fixture(scope="module")
def module_tokenizer() -> ByteLevelBPE:
    """A tokenizer trained once for the whole module; training is the slow part."""
    return train_tokenizer(
        [
            "The harbour clock measured every tide that crossed the bridge. " * 40,
            "Orchards ripen; lanterns kindle. Bridges cross rivers, rivers carry. " * 40,
        ],
        vocab_size=VOCAB,
    )


def prepare(
    out: Path,
    documents: list[Document],
    tokenizer: ByteLevelBPE,
    **kwargs: object,
) -> DatasetManifest:
    return binarize_documents(documents, out, tokenizer, **kwargs)  # type: ignore[arg-type]


@pytest.fixture
def dataset(
    tmp_path: Path, many_document_corpus: Path, module_tokenizer: ByteLevelBPE
) -> tuple[DatasetManifest, list[Document], ByteLevelBPE]:
    """A prepared dataset, with the documents and tokenizer that produced it."""
    documents, _ = read_corpus(many_document_corpus)
    manifest = prepare(tmp_path / "prepared", documents, module_tokenizer, seed=1234)
    return manifest, documents, module_tokenizer


# --------------------------------------------------------------------------- #
# Token width
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("vocab_size", "expected"),
    [(257, "<u2"), (4096, "<u2"), (65536, "<u2"), (65537, "<u4"), (200_000, "<u4")],
)
def test_token_width_is_the_narrowest_that_fits(vocab_size: int, expected: str) -> None:
    assert token_dtype(vocab_size).str == expected


def test_token_width_is_little_endian_regardless_of_host() -> None:
    """The format is fixed, so a big-endian machine reads the same files.

    Compared through ``.str`` rather than ``.byteorder``: numpy reports the native
    order as ``'='``, so on a little-endian host ``byteorder`` would pass whether
    the dtype was pinned or not.
    """
    assert token_dtype(4096).str.startswith("<")
    assert token_dtype(100_000).str.startswith("<")
    assert np.dtype(token_dtype(4096).str).str == "<u2"


def test_a_nonsensical_vocab_size_is_a_trainai_bug_not_a_user_error() -> None:
    with pytest.raises(DatasetError) as caught:
        token_dtype(0)

    assert "bug" in (caught.value.hint or "").lower()


# --------------------------------------------------------------------------- #
# What preparation writes
# --------------------------------------------------------------------------- #
def test_writes_shards_a_manifest_and_the_tokenizer(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    manifest, documents, _ = dataset
    root = manifest.root
    assert root is not None

    assert (root / MANIFEST_NAME).exists()
    assert (root / TOKENIZER_NAME).exists()
    assert manifest.total_tokens > 0
    assert manifest.totals["documents"] == len(documents)
    assert sum(len(manifest.shards[split]) for split in SPLITS) >= 2


def test_the_tokenizer_travels_with_the_dataset(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    """A dataset without the tokenizer that produced it is unusable."""
    manifest, _, tokenizer = dataset

    saved = ByteLevelBPE.load(manifest.tokenizer_path())

    assert saved.fingerprint() == manifest.tokenizer_fingerprint == tokenizer.fingerprint()


def test_every_shard_matches_its_recorded_size_and_checksum(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    manifest, _, _ = dataset

    verified = verify_dataset(manifest.root, deep=True)

    assert verified.content_hash == manifest.content_hash


def test_totals_agree_with_the_per_split_numbers(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    manifest, _, _ = dataset

    assert manifest.total_tokens == sum(manifest.tokens(split) for split in SPLITS)
    assert manifest.totals["tokens"] == manifest.total_tokens
    assert manifest.totals["documents"] == sum(manifest.documents[split] for split in SPLITS)
    assert manifest.totals["eot_tokens"] == manifest.totals["documents"]


def test_one_end_of_text_token_per_document(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    manifest, documents, _ = dataset

    eots = 0
    for split in SPLITS:
        for path in manifest.shard_paths(split):
            ids = np.fromfile(path, dtype=manifest.numpy_dtype)
            eots += int((ids == manifest.eot_id).sum())

    assert eots == len(documents)


def test_manifest_records_the_corpus_report_and_the_verdict(
    tmp_path: Path, many_document_corpus: Path, module_tokenizer: ByteLevelBPE
) -> None:
    documents, ingestor = read_corpus(many_document_corpus)
    report = analyze_documents(documents)
    validation = validate_corpus(report, ingestor.stats)

    manifest = prepare(
        tmp_path / "prepared",
        documents,
        module_tokenizer,
        report=report,
        validation=validation,
        ingest_options=ingestor.options,
        sources=[source.to_dict() for source in ingestor.discover(many_document_corpus)],
    )

    assert manifest.corpus["documents"] == report.documents
    assert manifest.validation["ok"] == validation.ok
    assert manifest.ingest["options"] == ingestor.options.to_dict()
    assert manifest.ingest["sources"][0]["path"] == "documents.jsonl"


def test_manifest_is_json_and_loads_back_identically(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    manifest, _, _ = dataset
    assert manifest.root is not None

    raw = json.loads((manifest.root / MANIFEST_NAME).read_text(encoding="utf-8"))
    reloaded = DatasetManifest.load(manifest.root)

    assert raw["content_hash"] == manifest.content_hash
    assert reloaded.total_tokens == manifest.total_tokens
    assert reloaded.tokenizer_fingerprint == manifest.tokenizer_fingerprint
    assert reloaded.seed == manifest.seed
    assert reloaded.shards == manifest.shards


def test_manifest_is_written_as_pure_ascii(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    """So the same bytes land on disk whatever the platform's default encoding is."""
    manifest, _, _ = dataset
    assert manifest.root is not None

    raw = (manifest.root / MANIFEST_NAME).read_bytes()

    assert all(byte < 128 for byte in raw)
    assert raw.endswith(b"\n")
    assert b"\r\n" not in raw


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def test_same_seed_gives_byte_identical_shards(
    tmp_path: Path, many_document_corpus: Path, module_tokenizer: ByteLevelBPE
) -> None:
    documents, _ = read_corpus(many_document_corpus)

    first = prepare(tmp_path / "a", documents, module_tokenizer, seed=99)
    second = prepare(tmp_path / "b", documents, module_tokenizer, seed=99)

    def checksums(manifest: DatasetManifest) -> list[str]:
        return [s.sha256 for split in SPLITS for s in manifest.shards[split]]

    assert checksums(first) == checksums(second)
    assert first.content_hash == second.content_hash


def test_the_content_hash_ignores_when_the_dataset_was_made(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    """Otherwise "same input, same hash" would be false for every second run."""
    manifest, _, _ = dataset
    before = manifest.compute_content_hash()

    manifest.created_at = "1999-12-31T23:59:59+00:00"
    manifest.created_with = "trainai 99.0.0"

    assert manifest.compute_content_hash() == before
    assert "created_at" not in manifest.reproducible_parts()
    assert "created_with" not in manifest.reproducible_parts()


def test_the_content_hash_notices_a_different_split(
    tmp_path: Path, many_document_corpus: Path, module_tokenizer: ByteLevelBPE
) -> None:
    documents, _ = read_corpus(many_document_corpus)

    first = prepare(tmp_path / "a", documents, module_tokenizer, seed=1)
    second = prepare(tmp_path / "b", documents, module_tokenizer, seed=2)

    assert first.content_hash != second.content_hash
    assert first.total_tokens == second.total_tokens  # the same text, split differently


def test_a_different_seed_moves_documents_between_splits(
    tmp_path: Path, many_document_corpus: Path, module_tokenizer: ByteLevelBPE
) -> None:
    documents, _ = read_corpus(many_document_corpus)

    first = prepare(tmp_path / "a", documents, module_tokenizer, seed=1, val_fraction=0.2)
    second = prepare(tmp_path / "b", documents, module_tokenizer, seed=2, val_fraction=0.2)

    assert first.documents["val"] != second.documents["val"]


def test_document_order_does_not_change_the_split(
    tmp_path: Path, many_document_corpus: Path, module_tokenizer: ByteLevelBPE
) -> None:
    """The split is by content hash, so shuffling the input cannot move a document."""
    documents, _ = read_corpus(many_document_corpus)

    forward = prepare(tmp_path / "a", documents, module_tokenizer, seed=5, val_fraction=0.2)
    backward = prepare(
        tmp_path / "b", list(reversed(documents)), module_tokenizer, seed=5, val_fraction=0.2
    )

    assert forward.documents == backward.documents
    assert forward.total_tokens == backward.total_tokens


def test_identical_documents_cannot_straddle_the_split(
    tmp_path: Path, module_tokenizer: ByteLevelBPE
) -> None:
    """An exact duplicate in train and val would make validation loss meaningless."""
    unique = [
        Document(f"Document {n} says something a little different. " * 4, "src", n, 0)
        for n in range(40)
    ]
    duplicated = [*unique, *unique]

    manifest = prepare(
        tmp_path / "prepared", duplicated, module_tokenizer, seed=3, val_fraction=0.25
    )

    # Every document appears twice, and both copies must land on the same side, so
    # each split holds an even number of documents.
    assert manifest.documents["train"] % 2 == 0
    assert manifest.documents["val"] % 2 == 0
    assert manifest.documents["val"] > 0


# --------------------------------------------------------------------------- #
# The shards are exactly the corpus
# --------------------------------------------------------------------------- #
def decode_documents(manifest: DatasetManifest, tokenizer: ByteLevelBPE) -> list[str]:
    """Every document in the dataset, recovered from the shards."""
    recovered: list[str] = []
    for split in SPLITS:
        ids: list[int] = []
        for path in manifest.shard_paths(split):
            ids.extend(np.fromfile(path, dtype=manifest.numpy_dtype).tolist())
        piece: list[int] = []
        for token in ids:
            if token == manifest.eot_id:
                recovered.append(tokenizer.decode(piece))
                piece = []
            else:
                piece.append(token)
        assert not piece, f"the {split} stream does not end on an end-of-text token"
    return recovered


def test_the_shards_decode_back_to_exactly_the_corpus(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    """The end-to-end integrity claim: nothing dropped, duplicated or corrupted."""
    manifest, documents, tokenizer = dataset

    recovered = decode_documents(manifest, tokenizer)

    assert sorted(recovered) == sorted(document.text for document in documents)


def test_no_document_is_split_across_the_end_of_text_boundary(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    manifest, documents, tokenizer = dataset

    recovered = decode_documents(manifest, tokenizer)

    assert len(recovered) == len(documents)
    assert all(text for text in recovered)


# --------------------------------------------------------------------------- #
# Sharding and splits
# --------------------------------------------------------------------------- #
def test_a_small_shard_size_produces_several_shards_that_stitch_together(
    tmp_path: Path, many_document_corpus: Path, module_tokenizer: ByteLevelBPE
) -> None:
    documents, _ = read_corpus(many_document_corpus)

    manifest = prepare(
        tmp_path / "prepared", documents, module_tokenizer, shard_tokens=2048, val_fraction=0.0
    )

    assert len(manifest.shards["train"]) > 1
    assert sum(s.tokens for s in manifest.shards["train"]) == manifest.tokens("train")
    stitched = np.concatenate(
        [np.fromfile(p, dtype=manifest.numpy_dtype) for p in manifest.shard_paths("train")]
    )
    assert stitched.size == manifest.tokens("train")


def test_val_fraction_zero_writes_no_validation_split(
    tmp_path: Path, many_document_corpus: Path, module_tokenizer: ByteLevelBPE
) -> None:
    documents, _ = read_corpus(many_document_corpus)

    manifest = prepare(tmp_path / "prepared", documents, module_tokenizer, val_fraction=0.0)

    assert manifest.shards["val"] == []
    assert manifest.tokens("val") == 0
    assert manifest.documents["train"] == len(documents)


def documents_numbering(count: int) -> list[Document]:
    """``count`` distinct documents, so the split hash sees ``count`` separate keys."""
    return [
        Document(f"document number {index} with some words in it. " * 4, "src", index, 0)
        for index in range(count)
    ]


#: A fraction small enough that no document is selected for validation at seed 0 --
#: verified for every count in the parametrization below, which is what makes these
#: tests reach the empty-split branch rather than succeed by accident.
NEVER_SELECTS = 0.001


def test_a_val_fraction_that_selects_nothing_is_refused_with_what_to_do(
    tmp_path: Path, module_tokenizer: ByteLevelBPE
) -> None:
    documents = [Document("only one document here, and it is short. " * 4, "src", 0, 0)]

    with pytest.raises(DatasetError) as caught:
        prepare(tmp_path / "prepared", documents, module_tokenizer, val_fraction=0.05)

    assert "validation split is empty" in str(caught.value)
    assert "--val-fraction" in (caught.value.hint or "")
    assert not (tmp_path / "prepared" / MANIFEST_NAME).exists()


@pytest.mark.parametrize("count", [1, 2, 3, 5, 8, 60])
def test_the_empty_split_hint_never_names_a_fraction_the_check_refuses(
    tmp_path: Path, module_tokenizer: ByteLevelBPE, count: int
) -> None:
    """Regression: the hint advised a --val-fraction the validator then rejected.

    ``suggested = max(1.0 / documents_done, 0.02)`` ignored the exclusive 0.5 ceiling
    the range check enforces. This error fires most often on a corpus of one or two
    documents, which is exactly where that formula named 1.00 or 0.50 -- so following
    the advice produced ``--val-fraction must be at least 0 and below 0.5, got 1.0``.
    Measured end to end on a 1,010-character single-document corpus before the fix.

    The assertion is deliberately mechanical: pull every number out of the hint and
    put it back through the public entry point. A hint is documentation that ships in
    an error message, and the only way to know it is true is to run it.
    """
    documents = documents_numbering(count)
    out = tmp_path / "prepared"

    with pytest.raises(DatasetError) as caught:
        prepare(out, documents, module_tokenizer, seed=0, val_fraction=NEVER_SELECTS)

    assert "validation split is empty" in str(caught.value)
    hint = caught.value.hint or ""
    for number in [float(text) for text in re.findall(r"\d+\.\d+", hint)]:
        assert 0.0 <= number < MAX_VAL_FRACTION, f"the hint suggests {number}, which is refused"
        # And not merely in range on paper: the entry point must accept it. It may
        # still report an empty split -- the split is a per-document hash, not a
        # quota -- but it must never come back with the range error.
        try:
            prepare(out, documents, module_tokenizer, seed=0, val_fraction=number)
        except DatasetError as followed:
            assert "must be at least" not in str(followed), (
                f"following the hint's own suggestion of {number} was refused: {followed}"
            )


def test_a_single_document_is_told_that_no_fraction_can_work(
    tmp_path: Path, module_tokenizer: ByteLevelBPE
) -> None:
    """One document cannot land on both sides, so naming any fraction is misleading.

    The old hint named ``1.00`` here -- the worst case of the bug above, since it is
    both refused by the check and impossible in principle.
    """
    with pytest.raises(DatasetError) as caught:
        prepare(
            tmp_path / "prepared",
            documents_numbering(1),
            module_tokenizer,
            seed=0,
            val_fraction=NEVER_SELECTS,
        )

    hint = caught.value.hint or ""
    assert re.findall(r"\d+\.\d+", hint) == [], (
        f"no fraction can work, yet the hint names one: {hint}"
    )
    assert "--max-doc-chars" in hint
    assert "--val-fraction 0" in hint


def test_the_empty_split_hint_names_a_fraction_that_works_on_this_corpus(
    tmp_path: Path, module_tokenizer: ByteLevelBPE
) -> None:
    """Follow the number the hint names, on the corpus that produced it, and it must work.

    The split is an independent keyed hash per document, not a quota, so for years the
    number came from a probability and the hint hedged about it. Two formulas were wrong
    outright: ``max(1/n, 0.02)`` left the validation split empty for 29% to 37% of seeds
    at every size from 3 to 60 documents, and inverting the distribution to target 5%
    still failed on 5% of corpora -- including, measured, a 3-document text file where it
    named 0.49 and 0.49 emptied the split again.

    The hash is deterministic, so the threshold is not a probability at all. The number now
    comes from the smallest fraction the failing run actually produced, and this test is
    the whole claim: pass it back and a dataset is written.
    """
    out = tmp_path / "prepared"
    hint = hint_for(out, module_tokenizer, count=5, seed=0, val_fraction=NEVER_SELECTS, kind=None)

    assert "hash" in hint, "the hint should say why the number is what it is"
    assert "--max-doc-chars" in hint
    offered = re.search(r"--val-fraction (\d+\.\d+) puts at least one", hint)
    assert offered, f"the hint offers no fraction to follow: {hint}"

    manifest = prepare(
        tmp_path / "followed",
        documents_numbering(5),
        module_tokenizer,
        seed=0,
        val_fraction=float(offered.group(1)),
    )

    assert manifest.documents["val"] >= 1
    assert manifest.documents["train"] >= 1


#: Corpus sizes and seeds that empty the validation split at ``NEVER_SELECTS``, which is
#: low enough that all thirty of them do. Several sizes, because the suggestion comes from
#: the smallest hash the run saw and that minimum falls as the corpus grows; several seeds,
#: because it is the seed that decides where the minimum lands. Three of these -- two
#: documents at seeds 2 and 11, three at seed 2 -- have every document above the ceiling,
#: so they take the branch with no fraction to name.
FOLLOWED_SUGGESTION_CASES = [
    (count, seed) for count in (2, 3, 5, 12, 40) for seed in (0, 1, 2, 3, 7, 11)
]


@pytest.mark.parametrize(("count", "seed"), FOLLOWED_SUGGESTION_CASES)
def test_every_fraction_the_empty_split_hint_names_writes_a_dataset(
    tmp_path: Path, module_tokenizer: ByteLevelBPE, count: int, seed: int
) -> None:
    """The generalisation of the test above, across sizes and seeds.

    A number that works on one corpus and not another is the defect this replaced, and one
    corpus cannot show it is gone. So: if the hint names a fraction, following it has to
    produce a dataset with documents on both sides. When it names none, that has to be
    because none exists -- checked by putting the fraction it *did* print back through the
    helper, which must agree there is nothing above it the range check would accept.
    """
    hint = hint_for(
        tmp_path / "empty",
        module_tokenizer,
        count=count,
        seed=seed,
        val_fraction=NEVER_SELECTS,
        kind=None,
    )
    offered = re.search(r"--val-fraction (\d+\.\d+) puts at least one", hint)

    if offered is None:
        closest = re.search(r"sits at (\d+\.\d+)", hint)
        assert closest, (
            f"{count} documents at seed {seed}: no fraction named and no reason given: {hint}"
        )
        assert _suggested_val_fraction(float(closest.group(1))) is None, (
            f"{count} documents at seed {seed}: a legal fraction above {closest.group(1)} "
            "exists, so one should have been named"
        )
        return

    manifest = prepare(
        tmp_path / "followed",
        documents_numbering(count),
        module_tokenizer,
        seed=seed,
        val_fraction=float(offered.group(1)),
    )

    assert manifest.documents["val"] >= 1
    assert manifest.documents["train"] >= 1


#: A seed that sends every document of :func:`documents_numbering` to validation at
#: ``HIGHEST_LEGAL_VAL_FRACTION``, for one, two and three documents alike -- found by
#: enumerating seeds against ``_assign_split``. Hard-coded rather than searched for
#: at run time, so that a change to the split hash, which the manifest's
#: reproducibility guarantee says must not happen, fails these tests loudly.
EMPTIES_TRAIN_SEED = 8

#: The largest fraction the range check accepts, to the two places a person types. It plays
#: two opposite parts below, which is why it is one constant: with ``EMPTIES_TRAIN_SEED``
#: it takes *every* document -- nothing stops a legal fraction from doing that -- and with
#: ``CEILING_UNLUCKY_SEED`` it takes none, leaving no legal raise for the hint to offer.
HIGHEST_LEGAL_VAL_FRACTION = 0.49


@pytest.mark.parametrize("count", [1, 2, 3])
def test_a_val_fraction_that_selects_everything_is_refused(
    tmp_path: Path, module_tokenizer: ByteLevelBPE, count: int
) -> None:
    """Regression: there was no empty-*train* guard at all, only an empty-val one.

    ``_assign_split`` is a weighted coin per document rather than a quota, so a legal
    ``--val-fraction`` can take every document. Measured: one document at 0.49 does
    it for 37 of 60 seeds. The dataset was then written and reported as a success --
    ``data prepare`` exited 0 saying "Wrote ... 601 tokens from 1 documents" with a
    split table reading ``train 0 0 0`` -- and ``trainai train`` refused it several
    minutes later, blaming a TrainAI bug and advising a re-run that reproduces it.
    """
    out = tmp_path / "prepared"

    with pytest.raises(DatasetError) as caught:
        prepare(
            out,
            documents_numbering(count),
            module_tokenizer,
            seed=EMPTIES_TRAIN_SEED,
            val_fraction=HIGHEST_LEGAL_VAL_FRACTION,
        )

    assert "training split is empty" in str(caught.value)
    assert caught.value.details["documents"] == count
    assert caught.value.details["val_fraction"] == HIGHEST_LEGAL_VAL_FRACTION
    # Nothing half-written is left to be mistaken for a dataset, as with every other
    # refusal in this function.
    assert not (out / MANIFEST_NAME).exists()
    assert list(out.glob("*.bin")) == []


def test_the_empty_train_hint_does_not_advise_an_unchanged_rerun(
    tmp_path: Path, module_tokenizer: ByteLevelBPE
) -> None:
    """The old downstream message advised exactly that, and the split is deterministic.

    One document has no fraction that works, so it is told to stop splitting; several
    are told the re-run is pointless without a change, which is the fact the previous
    advice got wrong.
    """
    hints = {}
    for count in (1, 3):
        with pytest.raises(DatasetError) as caught:
            prepare(
                tmp_path / f"n{count}",
                documents_numbering(count),
                module_tokenizer,
                seed=EMPTIES_TRAIN_SEED,
                val_fraction=HIGHEST_LEGAL_VAL_FRACTION,
            )
        hints[count] = caught.value.hint or ""

    assert "--val-fraction 0" in hints[1]
    assert "--max-doc-chars" in hints[1]
    assert "hash" in hints[3]
    assert "unchanged will not help" in hints[3]
    assert "--val-fraction" in hints[3]


def test_the_advice_for_a_single_document_actually_works(
    tmp_path: Path, module_tokenizer: ByteLevelBPE
) -> None:
    """Both single-document refusals point at ``--val-fraction 0``. Prove it succeeds.

    Every other test here asserts that a bad input is refused. This one asserts the
    way out is real, which is the property both of the fixed hints depend on.
    """
    manifest = prepare(
        tmp_path / "prepared",
        documents_numbering(1),
        module_tokenizer,
        seed=EMPTIES_TRAIN_SEED,
        val_fraction=0.0,
    )

    assert manifest.documents["train"] == 1
    assert manifest.tokens("train") > 0
    assert manifest.shards["val"] == []


# --------------------------------------------------------------------------- #
# Advice the corpus can act on
#
# Both empty-split hints used to end with "split your text into more documents by
# lowering --max-doc-chars", whatever the corpus was. Only two readers consult that
# option -- see CUT_BY_MAX_DOC_CHARS, which tests/test_data_ingest.py pins against the
# readers themselves -- and everywhere else one document is one record. Measured before
# the fix on a single-row CSV: following the advice produced "none of the 1 documents
# was selected", the identical refusal, with the count unmoved. The tool had spent the
# user's next attempt for them.
# --------------------------------------------------------------------------- #
#: Every empty-split hint, as the arguments that reach it: both branches of both
#: guards. One document is the case where no fraction can work; several is the case
#: where one might.
EMPTY_SPLIT_CASES = [
    pytest.param(1, 0, NEVER_SELECTS, id="val-empty-one-document"),
    pytest.param(5, 0, NEVER_SELECTS, id="val-empty-several-documents"),
    pytest.param(1, EMPTIES_TRAIN_SEED, HIGHEST_LEGAL_VAL_FRACTION, id="train-empty-one-document"),
    pytest.param(
        3, EMPTIES_TRAIN_SEED, HIGHEST_LEGAL_VAL_FRACTION, id="train-empty-several-documents"
    ),
]


def hint_for(
    out: Path,
    tokenizer: ByteLevelBPE,
    *,
    count: int,
    seed: int,
    val_fraction: float,
    kind: str | None,
) -> str:
    """Trigger an empty-split refusal and return its hint.

    ``kind`` becomes a manifest source record shaped like the one the CLI passes
    (:meth:`~trainai.data.ingest.SourceFile.to_dict`), of which only ``kind`` is read.
    ``None`` passes no sources at all, which is what a library caller assembling
    documents in Python does.
    """
    sources = None if kind is None else [{"relative": f"corpus.{kind}", "kind": kind}]
    with pytest.raises(DatasetError) as caught:
        prepare(
            out,
            documents_numbering(count),
            tokenizer,
            seed=seed,
            val_fraction=val_fraction,
            sources=sources,
        )
    return caught.value.hint or ""


@pytest.mark.parametrize(("count", "seed", "val_fraction"), EMPTY_SPLIT_CASES)
def test_a_record_corpus_is_not_told_to_lower_a_limit_that_cannot_cut_it(
    tmp_path: Path, module_tokenizer: ByteLevelBPE, count: int, seed: int, val_fraction: float
) -> None:
    """A CSV row is one document, so the fix is more rows, and the hint has to say so.

    The negation is still allowed to name the flag -- saying it *cannot* split a record
    is the useful half of that sentence, and a user who was about to reach for it needs
    to be told. What must not survive is the recommendation.
    """
    hint = hint_for(
        tmp_path / "csv",
        module_tokenizer,
        count=count,
        seed=seed,
        val_fraction=val_fraction,
        kind="csv",
    )

    assert "add more records to the corpus" in hint
    assert "one row is one document here" in hint
    assert "cannot split a record into more" in hint
    assert "split your text into more documents" not in hint, (
        f"a CSV row cannot be split by --max-doc-chars, yet the hint advises it: {hint}"
    )


@pytest.mark.parametrize(("count", "seed", "val_fraction"), EMPTY_SPLIT_CASES)
def test_a_text_corpus_is_still_told_to_lower_the_limit(
    tmp_path: Path, module_tokenizer: ByteLevelBPE, count: int, seed: int, val_fraction: float
) -> None:
    """The negative control. Text files are cut as they decode, so here it is the fix.

    Without this, narrowing the advice to nothing at all would pass the test above.
    """
    hint = hint_for(
        tmp_path / "text",
        module_tokenizer,
        count=count,
        seed=seed,
        val_fraction=val_fraction,
        kind="text",
    )

    assert "split your text into more documents by lowering --max-doc-chars" in hint
    assert "add more records" not in hint


@pytest.mark.parametrize(("count", "seed", "val_fraction"), EMPTY_SPLIT_CASES)
def test_a_library_caller_with_no_sources_is_told_which_half_applies(
    tmp_path: Path, module_tokenizer: ByteLevelBPE, count: int, seed: int, val_fraction: float
) -> None:
    """Documents assembled in Python have no readers behind them to ask.

    So neither half can be ruled out, and the honest hint offers both while saying which
    case the flag belongs to -- rather than recommending it flatly, which is what every
    caller used to get.
    """
    hint = hint_for(
        tmp_path / "none",
        module_tokenizer,
        count=count,
        seed=seed,
        val_fraction=val_fraction,
        kind=None,
    )

    assert "add more documents" in hint
    assert "if they came from text files" in hint, "the flag has to arrive conditioned"
    assert "--max-doc-chars" in hint


def test_several_record_kinds_are_all_named(tmp_path: Path, module_tokenizer: ByteLevelBPE) -> None:
    """A corpus can be more than one format, and the hint should not pick a favourite."""
    with pytest.raises(DatasetError) as caught:
        prepare(
            tmp_path / "mixed",
            documents_numbering(1),
            module_tokenizer,
            seed=0,
            val_fraction=NEVER_SELECTS,
            sources=[
                {"relative": "a.csv", "kind": "csv"},
                {"relative": "b.jsonl", "kind": "jsonl"},
            ],
        )

    assert "one line or one row is one document here" in (caught.value.hint or "")


def test_one_cuttable_source_is_enough_to_recommend_the_limit(
    tmp_path: Path, module_tokenizer: ByteLevelBPE
) -> None:
    """Mixed text and CSV: the flag helps the text half, so it stays the advice.

    The alternative -- withholding it because part of the corpus cannot be cut -- would
    hide the fix that works on the files most likely to hold the bulk of the tokens.
    """
    with pytest.raises(DatasetError) as caught:
        prepare(
            tmp_path / "mixed",
            documents_numbering(1),
            module_tokenizer,
            seed=0,
            val_fraction=NEVER_SELECTS,
            sources=[
                {"relative": "a.csv", "kind": "csv"},
                {"relative": "b.txt", "kind": "text"},
            ],
        )

    assert "split your text into more documents by lowering --max-doc-chars" in (
        caught.value.hint or ""
    )


def test_every_kind_that_cannot_be_cut_has_words_for_what_a_document_is(
    tmp_path: Path, module_tokenizer: ByteLevelBPE
) -> None:
    """Otherwise a new record-oriented reader falls back to the vaguest of the three hints.

    That fallback exists for library callers who pass no sources at all, and it is the
    right answer there. Reaching it because a kind was never given words would be a
    silent downgrade for a corpus the tool can see perfectly well.
    """
    for kind in sorted(set(get_args(Kind)) - CUT_BY_MAX_DOC_CHARS):
        hint = hint_for(
            tmp_path / kind,
            module_tokenizer,
            count=1,
            seed=0,
            val_fraction=NEVER_SELECTS,
            kind=kind,
        )
        assert "add more records to the corpus" in hint, f"{kind} fell through to the fallback"


# --------------------------------------------------------------------------- #
# A suggested --val-fraction has to be a raise, and has to be worth making
#
# `min(max(1.0 / documents_done, 0.02), 0.49)` was computed from the document count
# alone and never looked at the fraction in use, so past about twenty-six documents it
# named a value at or below what the user had already passed -- and called it a raise.
# Measured end to end: 60 documents at --val-fraction 0.05, seed 56, was advised to
# raise it to 0.02, which takes the chance of an empty validation split from 5% to 30%.
# At 20 documents it named 0.05, the default, so following it changed nothing at all on
# a split that is deterministic.
#
# Even where it was a raise it aimed at one expected validation document: over 2,000
# seeds that left the split empty for 29% to 37% of seeds at every size from 3 to 60.
# --------------------------------------------------------------------------- #
#: Both sides of the boundary. Sizes and fractions where a raise is available, and the
#: pairs where the fraction in use already meets the target -- which is the case the old
#: formula could not express, and so answered wrongly.
#: Every hash fraction a suggestion might be computed from: the boundaries, the values
#: that round awkwardly, and the region where no legal fraction is left.
#: Boundaries for :func:`_suggested_val_fraction`, whose input is the *smallest* hash a run
#: that emptied the validation split produced. Every document came in at or above the
#: fraction in use, so that minimum lives anywhere in ``[val_fraction, 1)`` -- including
#: above the ceiling, which is the case with no answer. Two-place values and values just
#: either side of one, because the helper rounds.
SUGGESTION_CASES = [
    0.0,
    0.0001,
    0.005,
    0.01,
    0.05,
    0.0999,
    0.1,
    0.30,
    0.3049,
    0.3051,
    0.48,
    0.4899,
    0.49,
    0.4999,
    0.999,
]

#: The same boundaries for the mirror helper, minus everything at or above the ceiling. Its
#: input is the *largest* hash a run that emptied the training split produced, and every
#: document there came in strictly below the fraction in use -- which the range check keeps
#: under ``MAX_VAL_FRACTION``. So a value at or above the ceiling is not a case the helper
#: has to answer; feeding it one would pin behaviour no run can reach.
LOWERING_CASES = [case for case in SUGGESTION_CASES if case < MAX_VAL_FRACTION]


@pytest.mark.parametrize("lowest", SUGGESTION_CASES)
def test_a_suggested_val_fraction_selects_the_document_it_was_computed_from(
    lowest: float,
) -> None:
    """Two properties, on the helper directly so every boundary is cheap to cover.

    It is strictly above the minimum -- ``_side`` compares with ``<``, so equal to the
    minimum keeps that document in training and the suggestion would be inert -- and it is
    inside the range the check at the top of ``binarize_documents`` accepts.

    ``None`` is only correct when no two-place value above the minimum fits under the
    ceiling. That is the case the hint answers in words instead.
    """
    suggested = _suggested_val_fraction(lowest)

    if suggested is None:
        assert (math.floor(lowest * 100.0) + 1) / 100.0 >= MAX_VAL_FRACTION, (
            f"a fraction above {lowest} fits under {MAX_VAL_FRACTION}, so None was wrong"
        )
        return

    assert suggested > lowest, f"{suggested} does not select the document at {lowest}"
    assert 0.0 < suggested < MAX_VAL_FRACTION, "the range check at the top would refuse it"


@pytest.mark.parametrize("fraction", [0.0, 0.01, 0.05, 0.3, 0.49])
def test_a_document_hashing_exactly_to_the_fraction_trains(fraction: float) -> None:
    """The boundary the suggestion is built on, pinned on its own.

    A hash equal to ``--val-fraction`` to the last bit is not something a corpus will
    produce, so nothing end-to-end reaches this and changing ``<`` to ``<=`` passes the
    whole suite. It still matters, because it is a premise elsewhere: the suggestion adds a
    hundredth rather than rounding up *because* the comparison is strict, and ``ceil`` --
    which returns a two-place minimum unchanged -- would be right if it were not. Two edits
    that each read as harmless alone would then leave the named fraction selecting nothing.
    """
    assert _side(fraction, fraction) == "train"


@pytest.mark.parametrize("highest", LOWERING_CASES)
def test_the_lowered_fraction_keeps_the_document_it_was_computed_from(
    highest: float,
) -> None:
    """The mirror, and the boundary goes the other way: at or below is enough.

    ``None`` means the answer rounded to zero, which is a different instruction -- it
    turns validation off rather than shrinking it -- and the hint offers that in words.
    """
    lower_to = _val_fraction_that_keeps_a_train_document(highest)

    if lower_to is None:
        assert highest < 0.01, f"{highest} rounds down to a usable fraction, not None"
        return

    assert lower_to <= highest, f"{lower_to} does not keep the document at {highest}"
    assert 0.0 < lower_to < MAX_VAL_FRACTION, "the range check at the top would refuse it"


def test_the_lowering_helper_is_only_ever_asked_about_fractions_below_the_ceiling(
    tmp_path: Path, module_tokenizer: ByteLevelBPE
) -> None:
    """``LOWERING_CASES`` drops the out-of-range inputs on the strength of an invariant.

    The invariant: a run reaches the empty-training branch only when every document hashed
    strictly below ``--val-fraction``, and the range check has already held that under
    ``MAX_VAL_FRACTION``. So ``highest_split_fraction`` is below the ceiling, always -- and
    ``_val_fraction_that_keeps_a_train_document`` never sees the values trimmed above.

    Left as a comment, that reasoning is how a helper quietly starts naming a fraction the
    range check refuses. So it is asserted here, on the real error, at the highest fraction
    the check accepts.
    """
    with pytest.raises(DatasetError) as caught:
        prepare(
            tmp_path / "train-empty",
            documents_numbering(3),
            module_tokenizer,
            seed=EMPTIES_TRAIN_SEED,
            val_fraction=HIGHEST_LEGAL_VAL_FRACTION,
        )

    highest = caught.value.details["highest_split_fraction"]
    assert highest < HIGHEST_LEGAL_VAL_FRACTION < MAX_VAL_FRACTION
    named = re.search(r"--val-fraction (\d+\.\d+) keeps at least one", caught.value.hint or "")
    assert named, f"a fraction below {highest} exists here, so one must be named"
    assert float(named.group(1)) < MAX_VAL_FRACTION


#: Sixty documents at the default fraction, and a seed that empties the validation split
#: anyway -- which happens for about 5% of seeds, so it is an ordinary outcome rather than
#: a contrived one. Found by enumerating seeds against ``_assign_split``, and hard-coded
#: so a change to the split hash fails this loudly.
UNLUCKY_SEED = 2


def test_the_fraction_the_hint_names_is_never_below_the_one_already_passed(
    tmp_path: Path, module_tokenizer: ByteLevelBPE
) -> None:
    """A number offered as a raise has to be a raise.

    Sixty documents at the default 0.05, with a seed that empties the validation split
    anyway. ``max(1.0 / documents, 0.02)`` named 0.02 here under the word "raising", which
    takes the chance of failing again from 5% to 30%; the formula that replaced it, aiming
    at a 5% failure rate from the document count alone, named 0.05 -- the fraction already
    in use. Neither could do better, because neither looked at the corpus.

    Measuring it removes the whole class: the smallest hash the run produced is at or above
    the fraction in use by definition -- that is *why* the split came out empty -- so a
    suggestion taken from it is strictly higher, always. This asserts that against the real
    function instead of trusting the algebra, and then follows the number.
    """
    hint = hint_for(
        tmp_path / "unlucky",
        module_tokenizer,
        count=60,
        seed=UNLUCKY_SEED,
        val_fraction=0.05,
        kind=None,
    )

    offered = re.search(r"--val-fraction (\d+\.\d+) puts at least one", hint)
    assert offered, f"a legal fraction above 0.05 exists here, so one must be named: {hint}"
    assert float(offered.group(1)) > 0.05, (
        f"the hint offers {offered.group(1)} to a user who passed 0.05, which is not a raise"
    )

    manifest = prepare(
        tmp_path / "followed",
        documents_numbering(60),
        module_tokenizer,
        seed=UNLUCKY_SEED,
        val_fraction=float(offered.group(1)),
    )

    assert manifest.documents["val"] >= 1
    assert manifest.documents["train"] >= 1


def test_a_chance_that_just_happened_is_never_reported_as_zero() -> None:
    """Rounding a small probability to a whole percent is how that gets printed.

    Two hundred documents at 0.3 leaves an empty validation split roughly once in
    10^31 tries, and the branch that reports it is only reached *because* it happened.
    "about a 0% chance" of an event in front of the user is worse than a vague word.
    """
    assert _empty_val_chance(200, 0.3) == "under 1%"
    assert _empty_val_chance(60, 0.05) == "about 5%"
    assert _empty_val_chance(1, 0.05) == "about 95%"


#: Three documents at ``HIGHEST_LEGAL_VAL_FRACTION``, and a seed where all three still hash
#: above it, so the validation split comes out empty with nowhere left to go. Enumerated
#: against ``_split_fraction`` and hard-coded, so a change to the split hash fails this
#: loudly rather than quietly landing the test in a different branch.
CEILING_UNLUCKY_SEED = 2


def test_a_corpus_no_legal_fraction_can_split_is_told_that_instead_of_a_number(
    tmp_path: Path, module_tokenizer: ByteLevelBPE
) -> None:
    """Sometimes there is no number to name, and naming one anyway is the defect.

    ``MAX_VAL_FRACTION`` is exclusive, so 0.49 is as high as advice may go, and here all
    three documents hash above 0.5 -- no fraction the range check accepts selects any of
    them. Two formulas answered anyway: ``max(1/n, 0.02)`` named 0.33, a *decrease* from
    what the user passed, and the distribution-inverting one named 0.49, the fraction
    already in use.

    So this pins the negative case. The hint must say no legal fraction works, and it must
    not name one -- and the reason it gives has to be checkable: the closest document sits
    at a fraction from which :func:`_suggested_val_fraction` itself finds nothing.
    """
    assert HIGHEST_LEGAL_VAL_FRACTION < MAX_VAL_FRACTION, (
        "the range check would refuse it otherwise"
    )

    hint = hint_for(
        tmp_path / "ceiling",
        module_tokenizer,
        count=3,
        seed=CEILING_UNLUCKY_SEED,
        val_fraction=HIGHEST_LEGAL_VAL_FRACTION,
        kind=None,
    )

    assert "No --val-fraction the range check accepts" in hint
    assert f"the ceiling is {MAX_VAL_FRACTION}" in hint
    assert "puts at least one" not in hint, "nothing works here, so nothing may be offered"
    assert "empty about 13% of the time" in hint, (
        "the chance quoted has to be the one at the fraction in use"
    )

    closest = re.search(r"sits at (\d+\.\d+)", hint)
    assert closest, f"the hint gives no reason a fraction cannot be named: {hint}"
    assert _suggested_val_fraction(float(closest.group(1))) is None, (
        f"the hint says nothing can select a document at {closest.group(1)}, but a legal "
        "fraction above that does exist -- so it should have been named"
    )


@pytest.mark.parametrize("val_fraction", [-0.1, 0.5, 1.0])
def test_an_out_of_range_val_fraction_is_refused(
    tmp_path: Path, module_tokenizer: ByteLevelBPE, val_fraction: float
) -> None:
    """At or above a half, validation would hold more text than training.

    The assertion names the range message rather than just ``--val-fraction``, which
    was too loose to pin anything: raising ``MAX_VAL_FRACTION`` to 1.0 as a mutation
    left this test green, because 0.5 on a one-document corpus then sailed past the
    range check and came back with the *empty split* error instead -- a different
    error that also happens to mention ``--val-fraction``.
    """
    documents = [Document("some text here " * 20, "src", 0, 0)]

    with pytest.raises(DatasetError) as caught:
        prepare(tmp_path / "prepared", documents, module_tokenizer, val_fraction=val_fraction)

    assert "must be at least 0 and below" in str(caught.value)
    assert caught.value.details["val_fraction"] == val_fraction


def test_a_tiny_shard_size_is_refused(tmp_path: Path, module_tokenizer: ByteLevelBPE) -> None:
    documents = [Document("some text here " * 20, "src", 0, 0)]

    with pytest.raises(DatasetError) as caught:
        prepare(tmp_path / "p", documents, module_tokenizer, shard_tokens=8, val_fraction=0.0)

    assert "--shard-tokens" in str(caught.value)


def test_no_documents_leaves_nothing_behind(tmp_path: Path, module_tokenizer: ByteLevelBPE) -> None:
    with pytest.raises(DatasetError) as caught:
        prepare(tmp_path / "prepared", [], module_tokenizer)

    assert "data inspect" in (caught.value.hint or "")
    assert not (tmp_path / "prepared" / MANIFEST_NAME).exists()
    assert list((tmp_path / "prepared").glob("*.bin")) == []


def test_progress_callback_reports_monotonic_progress(
    tmp_path: Path, many_document_corpus: Path, module_tokenizer: ByteLevelBPE
) -> None:
    documents, _ = read_corpus(many_document_corpus)
    seen: list[tuple[int, int]] = []

    manifest = prepare(
        tmp_path / "prepared",
        documents,
        module_tokenizer,
        batch_documents=16,
        progress=lambda done, tokens: seen.append((done, tokens)),
    )

    assert seen
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)
    assert [t for _, t in seen] == sorted(t for _, t in seen)
    assert seen[-1] == (manifest.totals["documents"], manifest.totals["tokens"])


def test_a_failure_part_way_through_leaves_no_half_dataset(
    tmp_path: Path, many_document_corpus: Path, module_tokenizer: ByteLevelBPE
) -> None:
    """A partial dataset with no manifest is bad; with a manifest it is worse."""
    documents, _ = read_corpus(many_document_corpus)
    out = tmp_path / "prepared"

    def explode(documents_done: int, tokens_written: int) -> None:
        if documents_done >= 32:
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError):
        prepare(out, documents, module_tokenizer, batch_documents=16, progress=explode)

    assert not (out / MANIFEST_NAME).exists()
    assert list(out.glob("*.bin")) == []


def test_repreparing_over_an_existing_dataset_replaces_it_cleanly(
    tmp_path: Path, many_document_corpus: Path, module_tokenizer: ByteLevelBPE
) -> None:
    documents, _ = read_corpus(many_document_corpus)
    out = tmp_path / "prepared"

    prepare(out, documents, module_tokenizer, shard_tokens=2048, val_fraction=0.0)
    stale = sorted(p.name for p in out.glob("*.bin"))
    second = prepare(out, documents[:20], module_tokenizer, shard_tokens=1 << 20, val_fraction=0.0)

    assert len(stale) > 1
    assert sorted(p.name for p in out.glob("*.bin")) == [s.name for s in second.shards["train"]]
    assert verify_dataset(out, deep=True).total_tokens == second.total_tokens


# --------------------------------------------------------------------------- #
# Integrity checking
# --------------------------------------------------------------------------- #
def test_deep_verify_catches_a_flipped_bit_that_shallow_verify_cannot(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    """The whole reason checksums are stored: a flipped bit changes no file size."""
    manifest, _, _ = dataset
    shard = manifest.shard_paths("train")[0]
    payload = bytearray(shard.read_bytes())
    payload[8] ^= 0x01
    shard.write_bytes(bytes(payload))

    verify_dataset(manifest.root, deep=False)  # size is unchanged, so this passes

    with pytest.raises(DatasetFormatError) as caught:
        verify_dataset(manifest.root, deep=True)
    assert "checksum" in str(caught.value)
    assert caught.value.details["path"].endswith(shard.name)


def test_a_truncated_shard_is_caught_even_without_hashing(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    manifest, _, _ = dataset
    shard = manifest.shard_paths("train")[0]
    shard.write_bytes(shard.read_bytes()[:-64])

    with pytest.raises(DatasetFormatError) as caught:
        verify_dataset(manifest.root, deep=False)

    assert "bytes" in str(caught.value)
    assert "prepare" in (caught.value.hint or "")


def test_a_missing_shard_names_the_file(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    manifest, _, _ = dataset
    shard = manifest.shard_paths("train")[0]
    shard.unlink()

    with pytest.raises(DatasetFormatError) as caught:
        verify_dataset(manifest.root, deep=False)

    assert shard.name in str(caught.value)


def test_a_directory_with_no_manifest_says_what_to_run(tmp_path: Path) -> None:
    with pytest.raises(DatasetFormatError) as caught:
        DatasetManifest.load(tmp_path)

    assert "data prepare" in (caught.value.hint or "")


def test_a_corrupt_manifest_says_the_dataset_must_be_rebuilt(tmp_path: Path) -> None:
    (tmp_path / MANIFEST_NAME).write_text("{not json", encoding="utf-8", newline="\n")

    with pytest.raises(DatasetFormatError) as caught:
        DatasetManifest.load(tmp_path)

    assert "prepare" in (caught.value.hint or "")


def test_a_manifest_that_is_not_utf8_is_refused_rather_than_raised(tmp_path: Path) -> None:
    """``UnicodeDecodeError`` is a ``ValueError``, so ``except OSError`` never saw it.

    A manifest opened and re-saved by an editor that wrote Latin-1 is the realistic way
    to get here, and it arrived as a traceback and exit 1 from a tool that documents exit
    3 for a dataset it cannot read.
    """
    (tmp_path / MANIFEST_NAME).write_bytes(b'{"format": "trainai-dataset\xff"}')

    with pytest.raises(DatasetFormatError) as caught:
        DatasetManifest.load(tmp_path)

    assert MANIFEST_NAME in str(caught.value)
    assert "utf-8" in caught.value.details.get("reason", "")


def edited_manifest(root: Path, edit: Callable[[dict[str, Any]], object]) -> Path:
    """Rewrite a prepared dataset's manifest as ``edit`` returns it.

    Takes a callable rather than a dict of replacements because the interesting cases
    are not all key edits: three of them replace the whole document with something that
    is not an object at all, which is what reached ``raw.get`` and raised
    ``AttributeError``.
    """
    path = root / MANIFEST_NAME
    payload = edit(json.loads(path.read_text(encoding="utf-8")))
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def without(raw: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: value for name, value in raw.items() if name != key}


def with_train(raw: dict[str, Any], train: object) -> dict[str, Any]:
    return {**raw, "splits": {**raw["splits"], "train": train}}


#: A shard entry that is entirely valid, so a case below can make exactly one field wrong.
SHARD = {"name": "train_00000.bin", "tokens": 1, "bytes": 2, "sha256": "abc"}


def shard_field_wrong(field: str) -> Callable[[dict[str, Any]], object]:
    """A manifest whose only train shard has exactly one field of the wrong type."""
    return lambda raw: with_train(raw, {"documents": 1, "shards": [{**SHARD, field: []}]})


#: (what the edit is, the edit, what the refusal must name). Every one of these was a
#: traceback and exit 1 before, across five exception types: ``AttributeError`` for a
#: document or a block that is not an object, ``KeyError`` for an absent key,
#: ``ValueError`` and ``TypeError`` for a value of the wrong type.
MANIFEST_EDITS: list[tuple[str, Callable[[dict[str, Any]], object], str]] = [
    ("the whole document is an array", lambda raw: [], "manifest"),
    ("the whole document is a string", lambda raw: "hello", "manifest"),
    ("the whole document is null", lambda raw: None, "manifest"),
    ("dtype is absent", lambda raw: without(raw, "dtype"), "dtype"),
    ("eot_id is absent", lambda raw: without(raw, "eot_id"), "eot_id"),
    ("content_hash is absent", lambda raw: without(raw, "content_hash"), "content_hash"),
    ("vocab_size is a word", lambda raw: {**raw, "vocab_size": "many"}, "vocab_size"),
    ("vocab_size is a boolean", lambda raw: {**raw, "vocab_size": True}, "vocab_size"),
    ("val_fraction is a word", lambda raw: {**raw, "val_fraction": "half"}, "val_fraction"),
    ("shard_tokens is a word", lambda raw: {**raw, "shard_tokens": "big"}, "shard_tokens"),
    ("format_version is a word", lambda raw: {**raw, "format_version": "two"}, "format_version"),
    ("format_version is null", lambda raw: {**raw, "format_version": None}, "format_version"),
    ("totals is a string", lambda raw: {**raw, "totals": "lots"}, "totals"),
    ("splits is a string", lambda raw: {**raw, "splits": "train"}, "splits"),
    ("a split is missing", lambda raw: {**raw, "splits": {"train": raw["splits"]["train"]}}, "val"),
    ("a split is a string", lambda raw: with_train(raw, "yes"), "splits.train"),
    # A number, not only a string, for every "not an object" case below. `"shards" not in
    # "yes"` is False, so a string reaches the next check and is refused anyway; `"shards"
    # not in 3` raises TypeError. The number is the case that escaped as a traceback, and
    # a suite testing only the string leaves the check that stops it unpinned -- measured:
    # three mutations survived a suite that had only the string cases.
    ("a split is a number", lambda raw: with_train(raw, 3), "splits.train"),
    ("shards is absent", lambda raw: with_train(raw, {"documents": 1}), "splits.train.shards"),
    (
        "shards is a number",
        lambda raw: with_train(raw, {"documents": 1, "shards": 3}),
        "splits.train.shards",
    ),
    (
        "documents is absent",
        lambda raw: with_train(raw, {"shards": []}),
        "splits.train.documents",
    ),
    (
        "a shard entry is a string",
        lambda raw: with_train(raw, {"documents": 1, "shards": ["x"]}),
        "splits.train.shards[0]",
    ),
    (
        "a shard entry is a number",
        lambda raw: with_train(raw, {"documents": 1, "shards": [3]}),
        "splits.train.shards[0]",
    ),
    (
        "a shard entry lacks sha256",
        lambda raw: with_train(raw, {"documents": 1, "shards": [{"name": "a"}]}),
        "splits.train.shards[0].",
    ),
    *(
        (
            f"a shard's {field} is the wrong type",
            # Every field, not one of them: each is read separately, so a suite that
            # checks one leaves the other three free to be read unchecked.
            shard_field_wrong(field),
            f"splits.train.shards[0].{field}",
        )
        for field in ("name", "tokens", "bytes", "sha256")
    ),
]


@pytest.mark.parametrize(
    ("edit", "named"),
    [pytest.param(edit, named, id=label) for label, edit, named in MANIFEST_EDITS],
)
def test_every_way_a_manifest_can_disagree_with_its_format_is_refused(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
    edit: Callable[[dict[str, Any]], object],
    named: str,
) -> None:
    """A manifest is a file users copy between machines, so a damaged one is a refusal.

    Both halves are asserted, and the second is the one worth having: that the message
    names the key. "The manifest is corrupt" for any of twenty-one different edits leaves
    a reader no better off than the traceback did -- the point of refusing is that the
    output says which line to look at.
    """
    manifest, _, _ = dataset
    assert manifest.root is not None
    path = edited_manifest(manifest.root, edit)

    with pytest.raises(DatasetFormatError) as caught:
        DatasetManifest.load(manifest.root)

    assert str(path) in str(caught.value), "a refusal about a file has to name the file"
    assert named in str(caught.value), f"the refusal does not name {named}: {caught.value}"
    assert "prepare" in (caught.value.hint or "")


def test_an_integer_where_a_float_belongs_is_accepted(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    """``"val_fraction": 0`` is what a person types where ``to_dict`` writes ``0.0``.

    The boundary of the check above, stated so that tightening it to ``float`` alone
    fails a test rather than refusing a hand-edited file that says exactly what it means.
    """
    manifest, _, _ = dataset
    assert manifest.root is not None
    edited_manifest(manifest.root, lambda raw: {**raw, "val_fraction": 0})

    assert DatasetManifest.load(manifest.root).val_fraction == 0.0


def test_an_unedited_manifest_still_loads_unchanged(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    """The negative control: a reader this strict must still read what the writer wrote.

    Round-tripped through the same rewrite the cases above use, so that a check which
    refuses the ordinary case -- an accidentally inverted condition, a key required that
    ``to_dict`` does not write -- fails here rather than in one user's terminal.
    """
    manifest, _, _ = dataset
    assert manifest.root is not None
    edited_manifest(manifest.root, lambda raw: raw)

    reloaded = DatasetManifest.load(manifest.root)

    assert reloaded.content_hash == manifest.content_hash
    assert reloaded.shards == manifest.shards
    assert reloaded.documents == manifest.documents


def test_a_manifest_from_another_tool_is_rejected(tmp_path: Path) -> None:
    (tmp_path / MANIFEST_NAME).write_text(
        json.dumps({"format": "someone-elses-format"}), encoding="utf-8", newline="\n"
    )

    with pytest.raises(DatasetFormatError) as caught:
        DatasetManifest.load(tmp_path)

    assert "not a TrainAI dataset manifest" in str(caught.value)


def test_a_manifest_from_a_newer_trainai_names_something_that_can_actually_be_done(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    """The hint used to be "Upgrade TrainAI with ``pip install --upgrade trainai``".

    There is no such command -- README.md says so -- and this was the only advice given
    for a dataset the build cannot read. A dataset, unlike a checkpoint, has a way out
    that needs no newer TrainAI at all: it is derived from a corpus the user still has,
    so `trainai data prepare` re-creates it at a version this build reads.
    """
    manifest, _, _ = dataset
    assert manifest.root is not None
    path = manifest.root / MANIFEST_NAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["format_version"] = manifest.format_version + 1
    path.write_text(json.dumps(raw), encoding="utf-8", newline="\n")

    with pytest.raises(DatasetFormatError) as caught:
        DatasetManifest.load(manifest.root)

    hint = caught.value.hint or ""
    assert "pip install" not in hint, "there is nothing to install from; see README.md"
    assert "trainai data prepare" in hint, (
        "re-creating the dataset is the route that works with the build in hand"
    )


def test_the_layout_description_names_every_file_that_is_written(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    """`data inspect --layout` prints this, so it must match what is on disk."""
    manifest, _, _ = dataset
    assert manifest.root is not None
    described = describe_dataset_layout()

    assert MANIFEST_NAME in described
    assert TOKENIZER_NAME in described
    for path in manifest.root.iterdir():
        stem = path.name.split("_")[0]
        assert stem in described or path.name in described, path.name


# --------------------------------------------------------------------------- #
# loader
# --------------------------------------------------------------------------- #
def test_batches_have_the_requested_shape_and_shifted_targets(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    manifest, _, _ = dataset

    batcher, stream = open_split(manifest, "train", seq_len=32, batch_size=4, seed=0)
    try:
        batch = batcher.batch(0)
    finally:
        stream.close()

    assert batch.inputs.shape == (4, 32)
    assert batch.targets.shape == (4, 32)
    assert np.array_equal(batch.inputs[:, 1:], batch.targets[:, :-1])


def test_the_same_step_always_gives_the_same_batch(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    """Resuming a run must replay the same data, so batch(step) has to be pure."""
    manifest, _, _ = dataset

    first, stream_a = open_split(manifest, "train", seq_len=16, batch_size=2, seed=7)
    second, stream_b = open_split(manifest, "train", seq_len=16, batch_size=2, seed=7)
    try:
        assert np.array_equal(first.batch(5).inputs, second.batch(5).inputs)
        assert np.array_equal(first.batch(5).inputs, first.batch(5).inputs)
        # Out of order access must not change what step 5 contains.
        second.batch(11)
        assert np.array_equal(first.batch(5).inputs, second.batch(5).inputs)
    finally:
        stream_a.close()
        stream_b.close()


def test_a_different_seed_gives_a_different_order(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    manifest, _, _ = dataset

    first, stream_a = open_split(manifest, "train", seq_len=16, batch_size=2, seed=1)
    second, stream_b = open_split(manifest, "train", seq_len=16, batch_size=2, seed=2)
    try:
        assert not np.array_equal(first.batch(0).inputs, second.batch(0).inputs)
    finally:
        stream_a.close()
        stream_b.close()


def test_token_ids_are_always_inside_the_vocabulary(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    """An out-of-range id would index past the embedding table and crash training."""
    manifest, _, _ = dataset

    batcher, stream = open_split(manifest, "train", seq_len=32, batch_size=8, seed=0)
    try:
        for step in range(min(8, batcher.steps_per_epoch)):
            batch = batcher.batch(step)
            assert batch.inputs.min() >= 0
            assert batch.inputs.max() < manifest.vocab_size
    finally:
        stream.close()


def test_sequential_batches_cover_the_split_from_the_start(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    """Evaluation must be deterministic and must not skip the beginning."""
    manifest, _, _ = dataset

    batcher, stream = open_split(manifest, "val", seq_len=16, batch_size=2, seed=0)
    try:
        batches = list(batcher.sequential_batches(max_batches=3))
        first = batcher.batch(0)
    finally:
        stream.close()

    assert len(batches) <= 3
    assert batches[0].inputs.shape == first.inputs.shape


def test_reading_past_the_end_of_the_stream_is_an_error(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    manifest, _, _ = dataset

    _, stream = open_split(manifest, "train", seq_len=16, batch_size=2, seed=0)
    try:
        with pytest.raises(DatasetError) as caught:
            stream.read(stream.total_tokens - 4, 32)
    finally:
        stream.close()

    assert "outside" in str(caught.value)
    assert caught.value.details["total"] == manifest.tokens("train")


@pytest.mark.parametrize(
    ("split", "phrase"),
    [
        ("val", "--val-fraction 0, so there is no held-out data"),
        ("train", "now refuses to write"),
    ],
)
def test_a_missing_split_names_the_cause_rather_than_a_version_to_upgrade_to(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
    split: str,
    phrase: str,
) -> None:
    """The train branch said the shape was one "which newer versions refuse to write".

    This build is the one that refuses -- measured: four documents at --val-fraction
    0.45 exits 3 with "The training split is empty: all 4 documents were assigned to
    validation". So there is no newer version to move to, and the two causes that can
    actually produce such a dataset are the ones worth naming: it predates that guard,
    or it was edited by hand. The comment above the hint said exactly that already,
    three lines from a hint that contradicted it.

    Neither branch had any test at all, which is how the two drifted apart.
    """
    manifest, _, _ = dataset
    without = replace(
        manifest, shards={name: rows for name, rows in manifest.shards.items() if name != split}
    )

    with pytest.raises(DatasetError) as caught:
        open_split(without, split, seq_len=16, batch_size=2, seed=0)  # type: ignore[arg-type]

    hint = caught.value.hint or ""
    assert f"has no {split} split" in str(caught.value)
    assert phrase in hint, hint
    assert "newer version" not in hint, (
        "the build printing this is the one with the guard; upgrading is not the remedy"
    )
    assert "trainai data prepare" in hint


def test_no_document_lands_in_both_splits(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    """The leakage guarantee the content-hash split actually makes.

    Deliberately stated over documents, not over token windows. Whether a 32-token
    window appears in both splits depends on how repetitive the corpus is -- a
    corpus of boilerplate shares windows everywhere -- and no splitter can change
    that. What the splitter guarantees is that a given document goes to exactly one
    side, so the validation set is never scored on text that was trained on.
    """
    manifest, documents, tokenizer = dataset
    if manifest.documents["val"] == 0:
        pytest.skip("no validation split to compare against")

    def documents_in(split: str) -> list[str]:
        ids: list[int] = []
        for path in manifest.shard_paths(split):  # type: ignore[arg-type]
            ids.extend(np.fromfile(path, dtype=manifest.numpy_dtype).tolist())
        out: list[str] = []
        piece: list[int] = []
        for token in ids:
            if token == manifest.eot_id:
                out.append(tokenizer.decode(piece))
                piece = []
            else:
                piece.append(token)
        return out

    train, val = documents_in("train"), documents_in("val")

    assert not set(train) & set(val)
    assert len(train) + len(val) == len(documents)


# --------------------------------------------------------------------------- #
# loader: damaged shards
# --------------------------------------------------------------------------- #
# These build shards by hand rather than preparing a dataset. The loader is pure
# numpy on purpose, so a shard is just bytes on disk and a token count in the
# manifest, and stating the two separately is what lets a mismatch be tested at all.
SHARD_DTYPE = np.dtype(np.uint16)


def write_shard(path: Path, tokens: int) -> Path:
    """A shard of ``tokens`` distinct ids, so a misread shows up as wrong values."""
    path.write_bytes(np.arange(tokens, dtype=SHARD_DTYPE).tobytes())
    return path


def test_an_intact_stream_reads_across_a_shard_boundary(tmp_path: Path) -> None:
    """The negative control for everything below.

    Two shards of different lengths, so a check that used the wrong shard's expected
    size would fail here rather than silently pass.
    """
    first = write_shard(tmp_path / "train_00000.bin", 100)
    second = write_shard(tmp_path / "train_00001.bin", 50)
    stream = ShardedTokenStream(paths=[first, second], tokens=[100, 50], dtype=SHARD_DTYPE)

    assert stream.total_tokens == 150
    across = stream.read(95, 10)

    assert across.tolist() == [95, 96, 97, 98, 99, 0, 1, 2, 3, 4]
    stream.close()


@pytest.mark.parametrize("lost", [1, 2, 3, 64])
def test_a_truncated_shard_is_a_named_error_rather_than_a_numpy_traceback(
    tmp_path: Path, lost: int
) -> None:
    """``lost=1`` and ``lost=3`` used to end training with a numpy traceback, exit 1.

    ``np.memmap`` raises ``ValueError`` -- not ``OSError`` -- when the file length is
    not a multiple of the itemsize, and the loader caught only ``OSError``. An even
    truncation mapped cleanly and was caught afterwards by the token count, which is
    how the gap survived review: half the cases did look handled.
    """
    path = write_shard(tmp_path / "train_00000.bin", 100)
    path.write_bytes(path.read_bytes()[: 200 - lost])
    stream = ShardedTokenStream(paths=[path], tokens=[100], dtype=SHARD_DTYPE)

    with pytest.raises(DatasetFormatError) as caught:
        stream.read(0, 10)

    assert f"is {200 - lost} bytes" in str(caught.value)
    assert "the manifest says 200" in str(caught.value)
    assert caught.value.details["size"] == 200 - lost
    assert caught.value.details["expected"] == 200
    assert "prepare" in (caught.value.hint or "")


def test_an_empty_shard_says_how_large_it_should_have_been(tmp_path: Path) -> None:
    """A zero-byte shard gave "cannot mmap an empty file" -- numpy's words, not ours."""
    path = tmp_path / "train_00000.bin"
    path.write_bytes(b"")
    stream = ShardedTokenStream(paths=[path], tokens=[100], dtype=SHARD_DTYPE)

    with pytest.raises(DatasetFormatError) as caught:
        stream.read(0, 10)

    assert "is 0 bytes" in str(caught.value)
    assert "the manifest says 200" in str(caught.value)


def test_a_shard_larger_than_the_manifest_is_refused_too(tmp_path: Path) -> None:
    """Not only truncation: extra bytes mean the file is not what was described."""
    path = write_shard(tmp_path / "train_00000.bin", 100)
    path.write_bytes(path.read_bytes() + b"\x00\x00")
    stream = ShardedTokenStream(paths=[path], tokens=[100], dtype=SHARD_DTYPE)

    with pytest.raises(DatasetFormatError) as caught:
        stream.read(0, 10)

    assert "is 202 bytes" in str(caught.value)


def test_a_missing_shard_names_the_file_and_what_to_run(tmp_path: Path) -> None:
    stream = ShardedTokenStream(
        paths=[tmp_path / "train_00000.bin"], tokens=[100], dtype=SHARD_DTYPE
    )

    with pytest.raises(DatasetFormatError) as caught:
        stream.read(0, 10)

    assert "train_00000.bin could not be opened" in str(caught.value)
    assert "data inspect" in (caught.value.hint or "")


def test_a_directory_where_a_shard_belongs_is_not_blamed_on_truncation(
    tmp_path: Path,
) -> None:
    """A directory stats as zero bytes on Windows, so the size check would call it
    truncated and advise re-preparing over an interrupted copy. It is neither."""
    path = tmp_path / "train_00000.bin"
    path.mkdir()
    stream = ShardedTokenStream(paths=[path], tokens=[100], dtype=SHARD_DTYPE)

    with pytest.raises(DatasetFormatError) as caught:
        stream.read(0, 10)

    assert "could not be opened" in str(caught.value)
    assert "truncated" not in (caught.value.hint or "")


def test_each_shard_is_checked_against_its_own_token_count(tmp_path: Path) -> None:
    """Two shards of different lengths, only the second damaged.

    A read inside the first still works -- shards are mapped lazily, and a dataset
    should not be refused for a shard nothing has asked for yet. The read that
    reaches the second names the second, with the second's own expected size.
    """
    first = write_shard(tmp_path / "train_00000.bin", 100)
    second = write_shard(tmp_path / "train_00001.bin", 50)
    second.write_bytes(second.read_bytes()[:-1])
    stream = ShardedTokenStream(paths=[first, second], tokens=[100, 50], dtype=SHARD_DTYPE)

    assert stream.read(0, 100).tolist() == list(range(100))

    with pytest.raises(DatasetFormatError) as caught:
        stream.read(95, 10)

    assert "train_00001.bin is 99 bytes" in str(caught.value)
    assert "the manifest says 100" in str(caught.value)


def test_a_shard_that_changes_while_it_is_opened_is_still_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The token-count check after mapping only fires on a race.

    The byte check settles every static file, so this branch is unreachable unless
    the file changes between the ``stat`` and the map -- which a test cannot
    schedule, hence the patched map. It is worth keeping because a stream whose
    shard is shorter than the manifest claims reads past its own boundaries.
    """
    path = write_shard(tmp_path / "train_00000.bin", 100)
    stream = ShardedTokenStream(paths=[path], tokens=[100], dtype=SHARD_DTYPE)
    monkeypatch.setattr(
        "trainai.data.loader.np.memmap",
        lambda *args, **kwargs: np.arange(99, dtype=SHARD_DTYPE),
    )

    with pytest.raises(DatasetFormatError) as caught:
        stream.read(0, 10)

    assert "holds 99 tokens" in str(caught.value)
    assert caught.value.details["found"] == 99


def test_a_value_error_from_the_map_is_still_a_dataset_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``np.memmap`` signals a bad length with ``ValueError``, not ``OSError``.

    The size check above it means no reachable input gets this far, so this is the
    backstop for the bug the size check fixed -- and it is pinned rather than left
    as decoration, because "the handler did not list the exception numpy actually
    raises" is exactly the mistake being repaired.
    """
    path = write_shard(tmp_path / "train_00000.bin", 100)
    stream = ShardedTokenStream(paths=[path], tokens=[100], dtype=SHARD_DTYPE)

    def refuse(*args: object, **kwargs: object) -> np.memmap:
        raise ValueError("Size of available data is not a multiple of the data-type size.")

    monkeypatch.setattr("trainai.data.loader.np.memmap", refuse)

    with pytest.raises(DatasetFormatError) as caught:
        stream.read(0, 10)

    assert "could not be opened" in str(caught.value)
    assert "multiple of the data-type size" in caught.value.details["reason"]


def test_the_loader_and_verify_dataset_agree_word_for_word(
    dataset: tuple[DatasetManifest, list[Document], ByteLevelBPE],
) -> None:
    """One damaged dataset, two code paths, one answer.

    ``--verify`` always reported a truncated shard correctly; training on the same
    dataset printed a numpy traceback. Whoever ran training first got the worse
    answer, so the two messages are pinned together rather than merely both being
    non-empty.
    """
    manifest, _, _ = dataset
    shard = manifest.shard_paths("train")[0]
    shard.write_bytes(shard.read_bytes()[:-1])

    with pytest.raises(DatasetFormatError) as from_verify:
        verify_dataset(manifest.root, deep=False)

    with pytest.raises(DatasetFormatError) as from_loader:
        batcher, stream = open_split(manifest, "train", seq_len=32, batch_size=4)
        try:
            batcher.batch(0)
        finally:
            stream.close()

    assert str(from_loader.value) == str(from_verify.value)
    assert from_loader.value.hint == from_verify.value.hint
    assert shard.name in str(from_loader.value)


# --------------------------------------------------------------------------- #
# loader: the context a split can support
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("total", [5, 7, 22, 107, 129, 1_000, 4_097])
def test_the_largest_supported_context_is_exactly_the_largest(total: int) -> None:
    """``largest_seq_len`` has to be tight in both directions.

    Too high and the suggestion is refused; too low and it wastes context for no
    reason. One below the answer must work and one above must not.
    """
    largest = largest_seq_len(total)

    assert window_count(total, largest) >= 1
    assert window_count(total, largest + 1) == 0


@pytest.mark.parametrize("total", [0, 1, 2, 3, 4])
def test_no_context_is_offered_when_none_can_work(total: int) -> None:
    """Below five tokens the answer is 0, never 1.

    Three and four tokens do yield a window at ``seq_len`` 1 -- arithmetically. The
    batcher refuses a context that small, so returning it would be a suggestion the
    same module rejects, which is the failure mode this function exists to end. Hence
    the range starts at ``MIN_SEQ_LEN``: what matters is that no *acceptable* context
    works, not that no integer does.
    """
    assert largest_seq_len(total) == 0
    assert all(window_count(total, seq_len) == 0 for seq_len in range(MIN_SEQ_LEN, 20))


def test_a_split_too_small_names_a_context_that_actually_works(tmp_path: Path) -> None:
    """The advice is followed here rather than pattern-matched.

    A 107-token split at a context of 64 used to be told, in the same panel, that it
    needed 129 tokens and that 65 would do.
    """
    path = write_shard(tmp_path / "train_00000.bin", 107)
    stream = ShardedTokenStream(paths=[path], tokens=[107], dtype=SHARD_DTYPE)

    with pytest.raises(DatasetError) as caught:
        TokenBatcher(stream, seq_len=64, batch_size=1)

    assert "needs 129" in str(caught.value)
    suggested = int(re.search(r"--seq-len (\d+)", str(caught.value.hint)).group(1))
    assert suggested == caught.value.details["largest_seq_len"]

    batcher = TokenBatcher(stream, seq_len=suggested, batch_size=1)

    assert batcher.windows >= 1
    with pytest.raises(DatasetError):
        TokenBatcher(stream, seq_len=suggested + 1, batch_size=1)
    stream.close()


def test_a_split_too_small_for_any_context_does_not_name_one(tmp_path: Path) -> None:
    """Four tokens support nothing, and saying "use --seq-len 1" would be refused."""
    path = write_shard(tmp_path / "train_00000.bin", 4)
    stream = ShardedTokenStream(paths=[path], tokens=[4], dtype=SHARD_DTYPE)

    with pytest.raises(DatasetError) as caught:
        TokenBatcher(stream, seq_len=64, batch_size=1)

    hint = str(caught.value.hint)

    assert "No --seq-len fits" in hint
    assert re.search(r"--seq-len \d", hint) is None
    assert caught.value.details["largest_seq_len"] == 0
    stream.close()


def test_a_batch_size_above_the_window_count_suggests_one_that_works(tmp_path: Path) -> None:
    """The other numeric suggestion in this module, checked the same way.

    It was already correct; it is pinned so it stays that way, because every other
    computed suggestion in this project has at some point named a value another check
    refused.
    """
    path = write_shard(tmp_path / "train_00000.bin", 300)
    stream = ShardedTokenStream(paths=[path], tokens=[300], dtype=SHARD_DTYPE)

    with pytest.raises(DatasetError) as caught:
        TokenBatcher(stream, seq_len=64, batch_size=10)

    suggested = int(re.search(r"--batch-size (\d+)", str(caught.value.hint)).group(1))
    batcher = TokenBatcher(stream, seq_len=64, batch_size=suggested)

    assert suggested == caught.value.details["windows"]
    assert batcher.steps_per_epoch >= 1
    assert batcher.batch(0).shape == (suggested, 64)
    stream.close()
