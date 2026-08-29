"""Data pipeline: raw files in, binary token shards out.

The stages, in order:

1. :mod:`~trainai.data.ingest` -- find files, decode them, emit documents.
2. :mod:`~trainai.data.analyze` -- measure the corpus.
3. :mod:`~trainai.data.validate` -- decide whether it can be trained on, and say
   what is worrying about it if so.
4. :mod:`~trainai.data.tokenizer` -- train a byte-level BPE tokenizer on it.
5. :mod:`~trainai.data.binarize` -- encode to ``uint16``/``uint32`` memmap shards
   with a checksummed manifest.
6. :mod:`~trainai.data.loader` -- sample batches from those shards.

Only :mod:`~trainai.data.tokenizer` imports a non-stdlib library at module scope
beyond numpy (``tokenizers``); nothing here imports torch. That is what keeps the
data tests fast enough to run on every commit.
"""

from __future__ import annotations

from trainai.data.analyze import CorpusMeter, DatasetReport, analyze_documents
from trainai.data.binarize import (
    DEFAULT_SHARD_TOKENS,
    DEFAULT_VAL_FRACTION,
    MANIFEST_NAME,
    SPLITS,
    DatasetManifest,
    ShardInfo,
    Split,
    binarize_documents,
    describe_dataset_layout,
    token_dtype,
    verify_dataset,
)
from trainai.data.ingest import (
    Document,
    IngestOptions,
    Ingestor,
    IngestStats,
    SourceFile,
    describe_supported_formats,
)
from trainai.data.loader import Batch, ShardedTokenStream, TokenBatcher, open_split
from trainai.data.tokenizer import (
    EOT_TOKEN,
    ByteLevelBPE,
    TokenizerReport,
    train_tokenizer,
)
from trainai.data.validate import ValidationIssue, ValidationResult, validate_corpus

__all__ = [
    "DEFAULT_SHARD_TOKENS",
    "DEFAULT_VAL_FRACTION",
    "EOT_TOKEN",
    "MANIFEST_NAME",
    "SPLITS",
    "Batch",
    "ByteLevelBPE",
    "CorpusMeter",
    "DatasetManifest",
    "DatasetReport",
    "Document",
    "IngestOptions",
    "IngestStats",
    "Ingestor",
    "ShardInfo",
    "ShardedTokenStream",
    "SourceFile",
    "Split",
    "TokenBatcher",
    "TokenizerReport",
    "ValidationIssue",
    "ValidationResult",
    "analyze_documents",
    "binarize_documents",
    "describe_dataset_layout",
    "describe_supported_formats",
    "open_split",
    "token_dtype",
    "train_tokenizer",
    "validate_corpus",
    "verify_dataset",
]
