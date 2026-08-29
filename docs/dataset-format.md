# The prepared-dataset format

`trainai data prepare` writes a directory. This document specifies what is in it,
because that directory is a compatibility surface: the trainer reads it, the
loader memory-maps it, and a dataset prepared by one version of TrainAI should
either load in the next one or be rejected with a clear reason.

Current version: **`trainai-dataset` v1**.

```
data/shakespeare/
  manifest.json      what the shards are, with a sha256 for each
  tokenizer.json     the tokenizer the shards were encoded with
  train_00000.bin    little-endian uint16 (or uint32) token ids, no header
  train_00001.bin    ...continuing the same stream
  val_00000.bin      the held-out split, same format
```

## Why a custom format

The obvious alternative is to depend on `datasets` (Arrow) or to save PyTorch
tensors. Neither fits:

- **Arrow** brings a large dependency tree for a job that needs a flat array of
  integers, and its random access is row-oriented while training reads arbitrary
  windows of a contiguous token stream.
- **`torch.save`** requires torch to read the data, which would make the whole
  data pipeline -- and its tests -- depend on a multi-gigabyte install.

A raw memory-mapped array of token ids is the simplest thing that supports the
one access pattern that matters: "give me `seq_len + 1` tokens starting at
arbitrary offset `n`". The cost is that TrainAI owns the format, which is why it
is written down here and checksummed on disk.

## Shard files

- No header, no padding, no separators. The file is `tokens x itemsize` bytes.
- `uint16` when the vocabulary is 65,536 or smaller, `uint32` above that. The
  manifest records which, as a numpy dtype string (`<u2` / `<u4`).
- **Always little-endian**, whatever the machine that wrote it. A dataset
  prepared on one architecture reads correctly on another.
- Shards of one split concatenate, in the order the manifest lists them, into one
  continuous token stream. A sequence may span a shard boundary; the loader
  stitches across it.
- Every document is followed by one end-of-text token, whose id is in the
  manifest. That token is what marks a document boundary; there is no other
  delimiter, and no other place it appears.

Reading a split without TrainAI needs numpy and nothing else:

```python
import json, numpy as np

manifest = json.load(open("data/shakespeare/manifest.json"))
tokens = np.concatenate(
    [
        np.memmap(f"data/shakespeare/{shard['name']}", dtype=manifest["dtype"], mode="r")
        for shard in manifest["splits"]["train"]["shards"]
    ]
)
```

## `manifest.json`

UTF-8, but written with `ensure_ascii=True`, so the bytes are pure ASCII and
identical on every platform. Trailing newline, LF line endings. Written
atomically (to a temporary file, then renamed), because the manifest is the only
thing that says which shards are valid -- a half-written manifest beside a
complete set of shards would turn an interrupted run into a corrupt dataset.

Fields worth knowing:

| Field | Meaning |
|---|---|
| `format`, `format_version` | `"trainai-dataset"`, `1`. A newer version is refused by name rather than misread; see [below](#what-a-damaged-manifest-is-refused-with). |
| `dtype`, `bytes_per_token`, `byte_order` | How to read the shards. |
| `vocab_size`, `eot_id` | Size the embedding table from `vocab_size`; `eot_id` is the document separator. |
| `tokenizer_fingerprint` | sha256 of `tokenizer.json`'s canonical serialisation. Checkpoints record it too, so a model can never be silently paired with a different tokenizer. |
| `tokenizer` | The tokenizer's own record: `vocab_size`, `requested_vocab_size`, `eot_id`, `eot_token`, `fingerprint`. Three of those repeat top-level keys, and the top-level ones are what the loader and `content_hash` use — this block is the readable copy, and it carries the two facts nothing else does: the requested vocabulary size, which differs from `vocab_size` when the corpus could not fill it, and the end-of-text token as text. |
| `seed`, `val_fraction` | Reproduce the split exactly. |
| `shard_tokens` | Tokens per shard file, from `--shard-tokens` (default 64Mi). **In `content_hash`**, because it decides where the shard boundaries fall and therefore the bytes of every shard. |
| `documents` | Documents per split. The same numbers appear under `splits.<split>.documents`; this is the flat form, and it is the one `content_hash` covers. |
| `splits` | One entry per split — `train` and `val`, always both, even when `val` is empty. Each carries `documents`, `tokens` and `shards[]`; `shards[]` is `name`, `tokens`, `bytes`, `sha256` per shard. This is the block a reader actually loads from, and the only place the per-shard checksums live. |
| `totals` | Documents, tokens, end-of-text tokens, characters, UTF-8 bytes. |
| `corpus` | The full measurement report from `data inspect`, recorded verbatim. |
| `validation` | Every warning and note that was raised at preparation time, so they are still visible later. |
| `ingest` | The options and source files, so the same dataset can be rebuilt. |
| `created_at`, `created_with` | When and by what. Deliberately **excluded** from `content_hash`. |
| `content_hash` | sha256 over the reproducible subset of the manifest. |

### `content_hash`

Answers exactly one question: *did the same input produce the same bytes?*

It covers the format version, dtype, vocabulary, tokenizer fingerprint, seed,
validation fraction, shard size, per-split document counts, totals, the full
shard list including every sha256, and the ingest options. It excludes
`created_at` and `created_with`, so preparing the same corpus twice with the same
settings gives the same hash even though the timestamps differ.

## The train/validation split

Chosen per document, by a keyed hash of the document's text:

```
blake2b(text, key=seed) -> first 64 bits as a fraction in [0, 1)
fraction < val_fraction  ->  validation
```

Three consequences, all deliberate:

1. **Platform-independent.** No RNG whose stream could differ between Python
   versions or numpy versions.
2. **Order-independent.** Shuffling the input files cannot move a document from
   one side to the other.
3. **Exact duplicates cannot leak.** Two byte-identical documents hash the same,
   so they always land on the same side. Validation loss is never measured on
   text that is also in the training set.

The granularity is one document, so `--max-doc-chars` indirectly controls how
finely the split can be cut. For a corpus that is a single large file, see the
note in `IngestOptions`.

## Integrity

`trainai data inspect <dir>` checks that every shard exists at its recorded size.
That is instant and catches truncation -- an interrupted copy, a full disk.

`trainai data inspect <dir> --verify` re-hashes every byte. This is the only way
to catch silent corruption: a truncated file has the wrong size, but a file with
a flipped bit does not. Training on a shard whose contents no longer match the
manifest would use data nothing describes, so this is worth the read.

## Compatibility promise

Within a major version, TrainAI will read any dataset it wrote. If the format
changes incompatibly, `format_version` increases and the old version is either
migrated or rejected by name -- never misread. A manifest from a newer TrainAI is
always refused rather than parsed on a guess, and the refusal names the route that
works with the build in hand: re-create the dataset from the corpus with `trainai
data prepare`, which writes a version this build reads.

### What a damaged manifest is refused with

The manifest is the only thing that says what the shards are, and it is a text
file in a directory users copy between machines and open in editors. So every way
it can disagree with its own format is a `DatasetFormatError`, exit **3**, naming
the file and the key -- never a traceback:

| in the file | the refusal |
|---|---|
| not valid JSON, or not valid UTF-8 | `<path> could not be read as JSON.` |
| valid JSON that is not an object | `has a manifest that is an array, not an object` |
| a key the reader uses, absent | `is missing dtype.` |
| a key at the wrong type | `has vocab_size as "many", which is not an integer` |
| a `splits` entry absent or not an object | `is missing splits.val.` |
| a shard entry wrong | `has splits.train.shards[0].tokens as [], which is not an integer` |

The rule is the one [the checkpoint format](checkpoint-format.md#compatibility-promise)
states, applied to the same reason: **a key the reader does not recognise is
ignored, and a key it does recognise is required and must hold the right type.**
Filling one from a default describes a dataset that is not the dataset on disk,
and how that surfaces depends entirely on which key it was:

| Key defaulted | What that costs |
|---|---|
| `dtype` | Decides how the shard bytes are interpreted. `<u2` read as `<u4` is not an error — it is half as many tokens, each a different one. |
| `vocab_size` | Sizes the model's embedding, and is what `trainai train` checks a plan against. |
| `tokenizer_fingerprint`, `content_hash` | Default to `""`, and an empty one *skips* the tokenizer-match and dataset-identity comparisons in `eval` rather than failing them. A defaulted key here silently disables a check. |
| `seed`, `val_fraction`, `shard_tokens` | Reported by `data inspect` as facts about how the dataset was built. A default is a wrong answer presented as a recorded one. |

A number is accepted where a float belongs -- `"val_fraction": 0` is what a person
types where the writer puts `0.0` -- but `true` is not accepted where a count
belongs, because JSON spells booleans and this is JSON.

None of these is reachable from a manifest TrainAI wrote. Measured on a copy of a
real dataset, one edit per case, against the reader before this was checked: of
twenty-seven edits, **twenty escaped as a Python traceback and exit 1** across five
exception types -- `AttributeError` for a document or block that is not an object,
`KeyError` for an absent key, `ValueError` and `TypeError` for a value of the wrong
type, and `UnicodeDecodeError`, which is a `ValueError` and so `except OSError`
never saw it, for a manifest an editor had re-saved as Latin-1.

The other **seven loaded silently**, and those are the worse half. A manifest
missing its `val` entry entirely, or missing `shards` inside one, loaded as a
dataset with no validation split and reported nothing; `"vocab_size": true` became
`1`; a shard whose `name` or `sha256` was an array became the string `"[]"`, which
is a filename that does not exist and a checksum that matches nothing.


