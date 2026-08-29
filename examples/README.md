# Examples

Small, runnable scripts. Each one does a single thing, uses only the standard
library, and prints what it did.

## `get_tinyshakespeare.py`

Downloads about 1.1 MB of Shakespeare into `data/corpus/`, so there is something
to prepare. Needs network access.

```bash
python examples/get_tinyshakespeare.py
```

Then the two commands that exist today:

```bash
trainai data inspect data/corpus
trainai data prepare data/corpus --out data/shakespeare
```

## What that produces

Measured on this corpus with the default settings (vocabulary 8,192, validation
fraction 0.05, seed 1234). These are counted numbers, not estimates:

| | |
|---|---|
| Input | 1,115,394 characters (1.06 MiB), one file |
| Documents | 69 (the file exceeds `--max-doc-chars 16384`, so it is cut at paragraph breaks) |
| Vocabulary | 8,192 tokens |
| Tokens | 317,284 total: 307,726 train / 9,558 validation |
| Compression | 3.52 characters per token |
| On disk | 620 KiB of shards, in 2 files |
| Wall clock | a few seconds, CPU only |

Re-running with the same seed reproduces the same shard checksums and the same
manifest `content_hash`. Changing the seed changes the split (307,726 / 9,558
becomes something else) but not the total token count, because the same text is
being divided differently.

`data prepare` writes four kinds of file into `--out`:

```
manifest.json    what the shards are, with a sha256 for each
tokenizer.json   the tokenizer the shards were encoded with
train_00000.bin  little-endian uint16 token ids, no header
val_00000.bin    the held-out split, same format
```

See [docs/dataset-format.md](../docs/dataset-format.md) for the format in full.

## A note on scale

A corpus this small is for checking that the pipeline works, not for producing a
useful model. `trainai data inspect` will warn you about it: under a megabyte, a
model of any size will reproduce phrases from the training text rather than write
new ones. That is worth seeing once, and it is not what a language model is for.
