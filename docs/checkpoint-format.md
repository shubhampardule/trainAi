# The checkpoint format

`trainai train` writes checkpoints into `<run>/checkpoints/`. This document
specifies what is in them and what TrainAI promises about them, because a
checkpoint is the only thing standing between a long run and losing it.

Current version: **`trainai-checkpoint` v1**.

```
runs/my-run/
  metrics.jsonl          one JSON object per event, appended and flushed
  checkpoints/
    checkpoints.json     which file is the latest, and which is the best
    step-0000500.pt      a torch.save payload; see below
    step-0001000.pt
```

## Three properties, and what each one cost

**Atomic.** Every checkpoint is written to `step-NNNNNNN.pt.tmp` and then renamed
with `os.replace`, which is atomic on POSIX and on Windows. So a checkpoint either
exists complete or does not exist. Writing in place would mean a crash during the
save — or a full disk, which is how it usually happens — destroys the new
checkpoint *and* the last good one. A leftover `.tmp` file is never mistaken for a
checkpoint: the glob only matches `.pt`.

**Verified.** Each checkpoint records the tokenizer fingerprint and the dataset
content hash it was trained against. Resuming against different data is refused by
name. That failure mode is worth spending code on: the same shard bytes read
through a different tokenizer are a different corpus, the loss curve looks entirely
normal, and the model that comes out is useless in a way nothing reports.

**Exact.** A resumed run is bitwise identical to an uninterrupted one — verified
in `tests/test_train_loop.py`, comparing every parameter *and* both AdamW moment
buffers after 12 + 12 steps against a straight 24. That works because there is
almost nothing to synchronise:

| State | How resume handles it |
|---|---|
| Weights | Restored from the file |
| AdamW moments | Restored from the file |
| fp16 gradient scaler | Restored from the file |
| RNG (torch, CUDA, numpy, python) | Restored from the file |
| Learning rate | **Recomputed** — `lr(step)` is a pure function |
| Batch order | **Recomputed** — `batch(seed, step)` is a pure function |

The last two rows are the design. A stateful `lr_scheduler` has a counter that
must be advanced exactly as many times on resume as it was originally; one call
too many or too few silently shifts the entire remaining schedule, and nothing
reports it. A stateful data iterator has the same problem with worse consequences.
Making both pure functions of the step removes the class of bug rather than
testing for it.

## What is in the file

A single `torch.save` payload. Not safetensors: it holds RNG state and
configuration dicts, not only tensors. `load_checkpoint` reads it with
`weights_only=False`, which means **a checkpoint from an untrusted source should
not be loaded** — it is a pickle. TrainAI only ever loads files it wrote, in a
directory the user owns. For sharing, `trainai export` writes safetensors; see
[export-format.md](export-format.md).

| Key | Contents |
|---|---|
| `format`, `format_version` | `"trainai-checkpoint"`, `1`. A newer version is refused with an upgrade instruction. |
| `step` | Optimizer steps completed. Training resumes *at* this number. |
| `model` | The full `ModelConfig`, with `n_kv_head` and `d_ff` resolved to concrete values. |
| `train` | The full `TrainConfig`, so the schedule cannot drift on resume. |
| `model_state` | Weights. The tied output projection is stored **once**; see below. |
| `optimizer_state` | AdamW's `exp_avg` and `exp_avg_sq` per parameter, plus step counts. |
| `scaler_state` | fp16 gradient-scaler state, or `null`. |
| `rng` | torch CPU, torch CUDA (per device), numpy, and python random state. |
| `metrics` | Best validation loss and the step it was at. |
| `dataset` | Tokenizer fingerprint, content hash, vocabulary, token counts, path. |
| `created_with` | `trainai <version>`. |

### The tied output projection is stored once

With `tie_embeddings` (the default), `state_dict()` lists both
`embed_tokens.weight` and `lm_head.weight`, and they share one storage. PyTorch
does not de-duplicate names — only `parameters()` does. Storing both is avoided
for two measured reasons, neither of them tidiness:

- `load_state_dict` applies the keys in order and the last one wins. Given a state
  dict whose two copies disagree it **silently overwrites the embedding and raises
  nothing**. Verified: a zeroed `lm_head.weight` zeroes the embedding of the loaded
  model. That test is in `tests/test_model_gpt.py`, pinning the behaviour down so
  the reason for this decision stays true.
- `safetensors.save_file` refuses shared storage outright, so the export has to
  drop one of them anyway. It does: a tied export omits `lm_head.weight` and sets
  `tie_word_embeddings: true`.

The saving in bytes is *not* the reason. `torch.save` de-duplicates storage, so
keeping the alias would cost about 1.8 KB of metadata, not a second copy of the
matrix.

On load, `Checkpoint.apply_to` re-ties the projection before loading, so the model
ends up with one tensor again.

### What a mismatched weight is refused with

`apply_to` compares the file's tensors against the model built from the file's own
`model` section, so by the time it looks at weights the shapes it expects are the
ones the checkpoint itself recorded. Four disagreements are refused as
`CheckpointCorruptError`, exit **5**, naming the file:

| in the file | reported in `details` |
|---|---|
| a weight the model does not have | `unexpected` |
| a weight the model needs, absent | `missing` |
| a weight at a different shape | `wrong_shape`, with both shapes |
| an entry that is not a tensor | `wrong_type` |

`lm_head.weight` is never listed as `missing` on a tied checkpoint, because the
format omits it on purpose.

A weight at a different **dtype** is not refused. PyTorch casts it, and a float64
copy of a real checkpoint loads and generates, so refusing it would break a file
that works.

None of the four is reachable from a checkpoint TrainAI wrote; they are what an
edited or truncated file looks like. Before this was checked, the last two escaped
as a `RuntimeError` traceback and exit 1 rather than as a refusal.

### Rotary tables are not stored

The cos/sin table is a `persistent=False` buffer: a pure function of
`(seq_len, head_dim, rope_theta)`, all three of which the checkpoint records.
Storing it would make checkpoints larger and add a way for a checkpoint to
disagree with itself.

## Retention

`--keep-checkpoints N` retains the newest N. Two are never deleted regardless: the
latest, and the best by validation loss. `--keep-checkpoints 0` keeps
*everything* rather than nothing — deleting every checkpoint would leave a run with
no way to resume, which is not a plausible thing for anyone to have asked for.

`checkpoints.json` names the latest and the best. It is a pointer file rather than
a symlink because creating symlinks on Windows needs developer mode or elevation,
and a training run must not require either. It is written atomically, as pure-ASCII
JSON with LF endings.

### What a damaged pointer file does

Nothing fatal, deliberately. This is the opposite of the choice
[`manifest.json`](dataset-format.md#what-a-damaged-manifest-is-refused-with) gets,
and the difference is that this file is *derived*: `latest` is the highest-numbered
`step-*.pt` in the directory, so a reader that cannot use the pointer falls back to
the filenames and gets the same answer. Refusing would turn a damaged convenience
file into a run that cannot resume and a model that cannot be loaded, while the
checkpoints themselves sit intact beside it.

So a pointer file that is not JSON, not an object, or whose `latest`/`best` is not
an object naming a file has that entry discarded and the file rewritten on the next
save. A key this version does not recognise is passed through untouched, because
the file is rewritten in place and dropping one is how a downgrade quietly loses
what a later version recorded. The newest checkpoint is protected from pruning by
its filename, not by the pointer naming it — "never the newest" is promised
unconditionally, and a promise that depends on a file a user can edit is not one.

The one thing here that is **not** derivable is `best`: the filenames record the
order checkpoints were written, not their validation loss. So `--which best`
against a damaged or absent pointer loads the *last* checkpoint, which is a
different one, and `trainai chat` and `trainai eval` say so:

```
Note  step-0000400.pt is the last checkpoint, not the one with the lowest
      validation loss: runs/mine/checkpoints/checkpoints.json is not valid JSON
```

Someone comparing two runs while that went unsaid would conclude their best model
is worse than it is. `--which latest` reports nothing, because there the fallback
names the same file the pointer would have.

Measured on a run directory with three checkpoints, one edit per case, against the
readers before this was checked: of eighteen ways the file can be damaged, pruning
raised a bare `TypeError` or `AttributeError` on ten and the pointer update on
five — and the update runs *after* the checkpoint is safely on disk, so a
hand-edited pointer ended a training run with a traceback over a file the next line
was going to overwrite. `UnicodeDecodeError` is a `ValueError`, so the `except
OSError` on all three readers never saw a pointer an editor had re-saved as
Latin-1, and that one raised in all three.

## Resuming with different settings

Allowed, and reported. `Trainer.config_changes` lists every setting that differs
from the checkpoint's, and the run prints each one:

```
Resumed from step 300 (step-0000300.pt)
  steps changed since the checkpoint: 300 -> 450. The remaining steps follow the
  new value, so this is not a continuation of the original schedule.
```

`--steps` is in that list for a reason that is easy to miss: the learning-rate
schedule is a function of the *total*, so resuming a 300-step run as a 450-step run
does not continue the original curve. It starts following a new one whose first two
thirds already happened at different rates. That is a legitimate thing to want and
a bad thing to have happen silently.

## Compatibility promise

Within a major version, TrainAI reads any checkpoint it wrote. If the format
changes incompatibly, `format_version` increases and the old version is either
migrated or refused by name — never misread. A checkpoint from a newer TrainAI is
always refused rather than parsed on a guess, and the refusal names the only route
that works: use the TrainAI that wrote it. There is no converter, and unlike a
dataset a checkpoint cannot be re-created — it is the output of the training run
that produced it.

That promise has two halves, and they pull in opposite directions:

- **A key the reader does not recognise is ignored.** This is what lets a later
  version add a field without invalidating every checkpoint on disk.
- **A key the reader *does* recognise is required, and must hold the right type.**
  A `model` block missing `n_layer` is refused naming `n_layer`; one whose `n_head`
  is `"four"` is refused quoting `"four"`. Both exit **5** (`ExitCode.CHECKPOINT`)
  and name the file. The same applies to the `train` block.

The second half is not symmetry for its own sake. Filling a missing key from the
dataclass default *rebuilds a different model*, and how that surfaces depends
entirely on which key it was:

| Key dropped from `model` | What used to happen |
|---|---|
| `n_layer`, `d_model`, `n_head` | The default built a differently-shaped model, and the weight load then reported that the checkpoint's weights did not match — accusing the weights, the one part of the file that was still right. |
| `rope_theta`, `norm_eps`, `tie_embeddings` | Nothing. No tensor changes shape, so every weight loaded, no layer complained, and the model generated noise because the rotary base it was rebuilt with was not the one it was trained with. |

The second row is why the check is strict rather than lenient-with-a-warning. It
was measured on a real 4-layer checkpoint, not reasoned about, and it is the whole
argument: a silent wrong answer costs more than a refusal ever does.

### The `train` block fails more quietly still

A model rebuilt at the wrong shape eventually fails a weight load. A *run* rebuilt
from the wrong hyperparameters just continues, at the wrong values, and reports
nothing at all. Measured on a real 600-step checkpoint, before the same strictness
was applied to this block:

| Key dropped from `train` | What the resume used instead |
|---|---|
| `lr` | 0.0003 — the default — in place of the recorded 0.000424. |
| `batch_size` | 8 in place of 32, so a third of the effective batch. |
| `grad_accum` | 1 in place of 2, compounding the line above. |
| `min_lr_ratio` | 0.1 in place of 0.05, so a floor twice as high. |
| `schedule` | `cosine` in place of `linear`. |
| `seed` | 1234 in place of 99 — which reorders every batch. |

The last row is the one that matters most, because it breaks a promise made
[at the top of this page](#three-properties-and-what-each-one-cost) and in
`train/checkpoint.py`'s own docstring: *a continued run is bitwise identical to an
uninterrupted one*. With the seed silently reset, it was not, and no line of output
said so. That claim is only worth making if the file it depends on has to be
complete.

`schedule` and `precision` are the two keys checked for presence but not for type.
Their values are validated where they are used, with a refusal that lists what is
allowed — `Unknown --schedule cosinus. Use one of: cosine, linear, constant.` —
which is more use than *is not a string*, so it is left to do the job.

The one key that may legitimately be absent is `warmup_steps`: its type admits
`null`, meaning *derive it*, and `to_dict` writes it raw so that the round trip
stays faithful. That is the rule in general, not a special case — a key is optional
exactly when its type admits `null`.


