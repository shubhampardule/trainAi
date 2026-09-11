# Plan format (`plan.json`)

`trainai plan` writes one file. `trainai train --plan plan.json` reads it. This page
specifies it, for the same reason the [dataset](dataset-format.md) and
[checkpoint](checkpoint-format.md) formats are specified: a format that is only
implied by the code that writes it becomes impossible to change safely.

A plan is a **record of measurements taken on one machine**, plus the configuration
those measurements justify. It is not portable advice. Applying a plan produced on a
24 GiB card to a 4 GiB card will not fail at load time — it will fail, or silently
page, when training starts. `plan.json` is in `.gitignore` for that reason.

## Compatibility

```json
{ "version": 1 }
```

`version` is checked on load. A plan from a different version is **refused**, with a
message naming the version it carries and the version this build expects. There is no
migration path in `0.x`: re-run `trainai plan`, which takes seconds. Fields may be
added within a version; a reader must ignore keys it does not recognise, which is what
`TrainConfig.from_dict` and `ModelConfig.from_dict` already do.

Ignoring an *unrecognised* key is not the same as tolerating a *missing* one. A `model`
or `train` block that has lost a key it should have is refused naming the key, because
filling it from a default builds a different model, or resumes on a different learning
rate, than the one the plan describes — see the
[compatibility promise](checkpoint-format.md#compatibility-promise), which spells out
what that costs for keys that change no tensor shape, and what it cost for the
hyperparameters that change none of them at all.

Overriding a value therefore does not mean editing this file: explicit flags are applied
on top of both blocks, so `--plan plan.json --lr 1e-4` is the supported way to change
one, and deleting the key is not.

The one cross-check on load is `model.vocab_size` against the dataset's tokenizer.
A mismatch means the plan was made for a different dataset, and applying it would
build an output projection of the wrong width — so it is refused rather than adjusted.

Every way a plan can be unusable — missing, unreadable, not UTF-8 text, not valid JSON,
valid JSON that is not an object, the wrong version, a `model` or `train` block that is
absent or will not load, or the wrong vocabulary — exits **2** (`ExitCode.USAGE`) and
names the file, what was wrong with it, and the `trainai plan` command that would
replace it:

```
badver.json is a version 99 plan and this build reads version 1.

What to do: Re-run `trainai plan --data data/shake` to write one this build reads.
```

A malformed block is named by what it actually is — `is a boolean, not an object` —
rather than only reported as wrong, and an absent block reads differently from one that
is `null`, because those send an editor to two different places.

The file is decoded before it is parsed, so an encoding problem reads as one. A file
that is not UTF-8 is refused as `is not UTF-8 text`, and a UTF-16 BOM is named:

```
plan.json is not UTF-8 text.

What to do: It looks like UTF-16. Re-save it as UTF-8 — in Notepad that is the
encoding dropdown in the Save As dialog — or re-run `trainai plan --data data/shake`
to write a fresh one.
```

A **UTF-8 BOM is accepted**, not reported. It is what Notepad writes by default, this
file is documented as one to open and edit, and it used to be refused as "not valid
JSON" — which sends someone hunting for a syntax error in a file that has none. A
UTF-16 file with no BOM at all is the one case that still reads as bad JSON rather than
bad encoding: ASCII text interleaved with NUL bytes is valid UTF-8, so there is nothing
left to identify it by.

That is a promise about the least-travelled branch, not just the common ones, because
a plan is a file this tool prints in order to be argued with — so a hand-edited one is
ordinary input. It is enforced two ways: every condition is run end to end, and every
refusal in the loader is checked structurally for the file's name and for `path` in its
`details`, so a branch added later cannot quietly drop either.

## What `--plan` actually consumes

Only two blocks:

| Key | Consumed as |
|---|---|
| `model` | `ModelConfig.from_dict` — the architecture |
| `train` | `TrainConfig.from_dict` — steps, batch, learning rate, cadences |

Everything else in the file is there to be read by a human, a script, or a bug report.
`trainai train` ignores it. Explicit command-line flags are applied on top of both
blocks, so `--plan plan.json --steps 500` is supported and the flag wins.

## The blocks

### `dataset`, `preset`

What the plan is *about*, and the two things every other number in the file is
conditional on.

`dataset` is the dataset directory the measurements were taken against, as it was
given to `--data`. It is a path rather than a fingerprint, so it is provenance and
not an integrity check: moving or re-preparing that directory leaves the plan
looking valid while describing measurements of something else. The one guard on load
is `model.vocab_size` against the dataset's tokenizer (see
[Compatibility](#compatibility)), which catches the case that would corrupt training
and no more than that.

`preset` is the name of the preset the accepted candidate came from — the last `ok`
entry in `candidates`. It is recorded because `model` serialises the *resolved*
architecture, so nothing else in the file says which rung of the ladder was reached.
It is also what `train_command` puts after `--preset`; a preset renamed or removed in
a later version leaves an old plan's command naming one that no longer exists, which
is why `model` and not `preset` is what `--plan` consumes.

### `model`, `train`

The serialised form of `ModelConfig` and `TrainConfig`. Two details in `train` are
load-bearing:

- `warmup_steps` is the **raw** field and is `null` when it was never chosen. The
  resolved value is published separately as `resolved_warmup_steps`. Serialising the
  resolved value in place of the raw one pins an adaptive default, which is how a plan
  came to carry `warmup_steps: 20` and then reject `--steps 20` as invalid.
- `resolved_warmup_steps`, `effective_batch_size`, `tokens_per_step` and
  `total_tokens` are **derived** and are written for readability. `from_dict` ignores
  them and recomputes.

`model` deliberately serialises `d_ff` and `n_kv_head` **resolved** rather than raw,
so the round trip is not field-for-field identical. That is intentional: the same
method serialises checkpoints, where the architecture has to reproduce exactly or the
saved weights will not load into it.

### `measurement`

The accepted candidate's `BenchmarkResult`. Read `memory_measured` before reading
`peak_bytes`: on a backend with no allocator counters the peak is `0` because nothing
was measured, not because nothing was used. `budget_fraction` is `0.0` in that case
too, for the same reason.

`step_seconds` is what the time estimate is built from. `alloc_retries` is the
allocator's `num_alloc_retries` delta across the measurement — a non-zero value means
the allocator had to reclaim, which is a fit that is tighter than it looks.

### `provenance`

The same numbers as elsewhere in the file, partitioned by how strong a claim they are.
This exists to be checkable: it is the machine-readable form of the three-section
report `trainai plan` prints.

| Section | Means |
|---|---|
| `measured` | a real step ran on this machine and a counter was read |
| `derived_from_data` | arithmetic over counted tokens in the dataset manifest |
| `rules_of_thumb` | a defensible default that nothing here tested |

The learning rate is in `rules_of_thumb`, and stays there however plausible it looks.

### `candidates`

Every configuration the search touched, in the order it touched them, each with the
`Candidate` shape and its `BenchmarkResult`. The accepted one is the last `ok` entry.
Rejections carry the verdict that stopped them (`over-budget`, `not-enough-data`,
`too-slow`, `throughput-collapse`, `out-of-memory`, `error`, `skipped`) and the
measurement they were rejected on. A rejection with `steps_measured > 0` was run
before being rejected; `skipped` means it was ruled out before being measured, and
says so rather than reporting a peak of zero as a measurement.

### `budget`, `hardware`, `regime`

`budget` is the epochs and tokens-per-parameter arithmetic from
`trainai.train.budget`. `hardware` is the full `trainai doctor` probe, including
`silent_vram_spillover` — which is what put the planner in measure-don't-catch mode.
`regime` is one of `data-limited`, `vram-limited`, `compute-limited`,
`unconstrained`, `blocked`, with `regime_detail` naming the specific thing that
bound the answer.

`blocked` is worth special mention when reading a plan: it means the search stopped
for a reason that was not memory, data or time. Do not read it as "buy a bigger card".

`unconstrained` is worth the opposite caution: it means nothing rejected the largest
preset on the ladder, not necessarily that nothing was rejected. A rung is accepted as
soon as *some* micro-batch on it fits, so a plan can be `unconstrained` and still have
had its batch halved several times to get there. `regime_detail` says which happened,
and `train.grad_accum` above 1 is the same fact in a number.

### `vram_cap_bytes`

What `--max-vram` asked for, or `null` when it was not given. It sits at the top level
rather than inside `provenance.measured` beside the budget it bounded, because it is a
number the user typed and that block is for numbers a counter reported.

`provenance.measured.vram_budget_bytes` is the budget the search actually compared
every measured peak against, which is the smaller of this cap and 85% of free VRAM. A
cap only ever lowers it: sizing against memory the card does not have would mean
measuring candidates it cannot hold, which is the failure the check exists to prevent.
So a cap above what is free changes nothing and is reported in `notes` as having
changed nothing. Comparing the two fields is how a reader tells a small plan on a small
card from a small plan that was asked for.

### `notes`, `train_command`, `estimated_seconds`

`notes` are the plain-language warnings, already rendered. `train_command` is the
fully expanded equivalent `trainai train` invocation, printed so that the plan can be
disagreed with rather than merely obeyed. `estimated_seconds` is
`measurement.step_seconds * train.steps` — training steps only, so it excludes the
dataset reads and validation passes the real run also does. Treat it as a floor.

## Reading it from a script

```bash
trainai plan --data data/mine --json > plan.json
```

`--json` writes the plan to stdout and nothing else — not even the progress lines,
which are suppressed rather than redirected — so it is safe to pipe straight into `jq`
or `json.load`.

When **nothing** on the ladder fits, no plan is written at all: there is no
configuration to record. `trainai plan` exits **4** (`ExitCode.CAPACITY`) and prints
the rejected candidates with the measurement that rejected each one, including a real
measurement of the smallest shape it tried. Machine-readably, those live in the
error's `details["candidates"]`, alongside `vram_budget_bytes` and `train_tokens`.
Exit **0** means a plan was written.
