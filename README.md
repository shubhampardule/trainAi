# TrainAI

**Train a small language model from scratch, on the hardware you already own, without
writing the training code.**

You point TrainAI at a folder of text. It measures your machine, works out what is
realistically trainable on it, explains the trade-offs, and then trains the model —
handling the tokenizer, the data pipeline, the learning-rate schedule, checkpointing,
resuming, evaluation and export for you.

```bash
trainai quickstart ./my-corpus --out my-first-model
```

That is the whole pipeline: prepare the data, measure the machine, train, evaluate,
sample. It stops once to confirm before the long step, quoting the time it *measured*
on your machine rather than a guess, and before each step it prints the individual
command it stands in for — so it is a shortcut *through* the CLI rather than a wizard
around it. Add `--time 30m` if you would rather name the budget than approve an
estimate: the plan is sized to fit it, step count included.

## What you actually get

This first, because it is the thing most tools leave until after you have spent an
evening. Below is real, unedited output from the 4.26M-parameter model this project
trains on 1.1 MB of Shakespeare, in under two minutes on a 4 GiB laptop GPU:

```
KING RICHARD:
Hath count her mortal; none of his siped eyes
That heirs on our pouting eyes to me.
Give him good to a blined, my lord;
For they are purfail'd with commons' royal cheeks,
Or be to opposers be at London's blood!

BUCKINGHAM:
So do I promise you, a body's night.

KING RICHARD III:
O, madam, I am too much to thee.

BUSHY:
What is thy mother and Warwick?

SOMERSET:
He doth not
120 tokens in 1.1s - 107.6 tokens/s
```

It has learned the *form* — speaker headings, line breaks, the register, real
character names. The words "siped", "purfail'd" and "blined" do not exist.

**A model at this scale will not behave like ChatGPT.** It produces fluent-looking
text in the shape and vocabulary of your corpus, and it gets facts wrong constantly.
TrainAI says so before you train rather than after, and `trainai chat` repeats it on
the way in. If what you want is a model that answers questions, you want to fine-tune
an existing one, and this is the wrong tool.

What it *is* good for: understanding how training actually works, on your own data,
with every decision visible and every number measured.

## Install

Requires Python 3.10 or newer. PyTorch is the large part of the download; how large
depends on which build your hardware needs, and `trainai setup` tells you which that is
before you spend it.

> **Not on PyPI yet.** There is no `pip install trainai`, so install from a clone — that
> is the install path this project has actually tested, and nothing has been uploaded to
> any package index. The
> [v0.1.0 release](https://github.com/shubhampardule/trainAi/releases/tag/v0.1.0)
> attaches a built wheel and an sdist for anyone who would rather not clone.

**Windows** (PowerShell):

```powershell
git clone https://github.com/shubhampardule/trainAi
cd trainAi
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

**macOS and Linux**:

```bash
git clone https://github.com/shubhampardule/trainAi
cd trainAi
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Those are the only two spellings that differ, and only for the virtualenv: every
`trainai` command in this README is identical on all three platforms. On Windows, use
`python -m venv` instead if you installed Python from the Microsoft Store, which does
not ship the `py` launcher.

### Then point PyTorch at your hardware

```bash
trainai setup
```

Run this before anything else. The default PyPI `torch` wheel is CPU-only on some
platforms, so a machine with a perfectly good GPU can end up training 20–100× slower
with nothing raising an error — from inside PyTorch it looks identical to having no
GPU at all. `setup` checks the system for a GPU independently of PyTorch, reads the
CUDA version your *driver* reports, and prints the install command that matches. It
only prints; `--install` runs it after confirming, and refuses to touch a system
Python.

What it tells each kind of machine, and how much of that anyone has actually run:

| Your machine | What `setup` does | Run for real? |
|---|---|---|
| **NVIDIA**, Windows or Linux | Reads the driver's CUDA version and prints a `torch` install from the matching `cu###` index | Yes — an RTX 2050 on Windows 11 |
| **Apple Silicon**, macOS | Normally reports there is nothing to change: Metal (MPS) ships in the default wheel, so there is no separate index to point at. It only prints a `torch` reinstall when MPS is *missing* | No — see below |
| **AMD**, Linux | Prints a `torch` install from the ROCm index, and names the fallback index for the many consumer cards that index does not cover | No |
| **AMD**, Windows | Points at AMD's own wheel index, which does publish a Windows ROCm build, and asks you for your card's `gfx` target rather than guessing it | No |
| **Intel Arc** | Prints a `torch` install from the XPU index, and says throughput on it is unverified | No |
| **No GPU** | Says CPU training works and is roughly 20–100× slower than a modern GPU — survivable for a first run on a small corpus, not for anything larger | Yes |

**On macOS specifically**, since it came up: there is no separate download index and no
CUDA-style version to match. The wheel you get from plain `pip` already supports MPS, so
on a normal arm64 Python `trainai setup` should tell you there is nothing to change and
send you on to `trainai doctor`. The one trap is an x86 build of Python running under
Rosetta, where MPS is simply absent — then `setup` prints
`pip install --upgrade --force-reinstall torch` and tells you to reinstall Python as
arm64, which is the actual fix. TrainAI also runs MPS in fp32, because autocast on MPS
has not been verified here.

Nobody working on this project owns a Mac. The macOS code paths exist, are exercised
against a synthetic MPS profile in the test suite, and have never executed on real Apple
hardware. So treat the paragraph above as what the code is written to do, not as
something anyone has watched happen — and if you run it on a Mac and it does something
else, that is a bug worth reporting.

### Then see what the machine can do

```bash
trainai doctor
```

<details>
<summary>What a built wheel actually installs (measured, not assumed)</summary>

The wheel has been built, `twine check --strict`-ed, and installed into a clean
virtualenv from the local file with the pip cache disabled — no checkout on the path,
nothing else on `PATH`, nothing reused from an earlier download. What that run found,
on Windows 11 with Python 3.13:

- **`pip install` of the wheel is 37 packages and no build step.** Eight of those are
  declared; the other twenty-nine are theirs. Every one arrived as a wheel — nothing
  needed a compiler, Node, or a system package. 4m 14s cold, of which `torch` is a
  single 124.1 MB download that unpacks to 502 MiB.
- **The default `torch` was CPU-only** — `2.14.0+cpu`, `torch.version.cuda` `None`,
  `cuda_available` False — on a machine with an NVIDIA card in it. That is exactly the
  trap `trainai setup` exists for: it reads the driver's CUDA version rather than
  asking torch, and prints the `cu128` command for it. This is why `setup` comes first.
- **The suite in the sdist passes against the installed wheel** — 2681 passed, 66
  skipped, 14 deselected, in 3m 20s. The skips are the checks that need a git checkout,
  a GPU, `transformers`, or a `.github/` the sdist deliberately does not ship; each one
  says which. Running it is what found two of those checks failing rather than skipping,
  which no run from a checkout could have.
- **`quickstart` completed on the CPU** with no GPU in the picture, at the 4.7 s/step
  the planner measured there, cutting the step count to fit the time budget and saying
  so rather than quietly running long. That bullet is from an earlier pass; the run
  above stopped at the suite, because a raw corpus to prepare is not in the sdist.

What is still unverified is the *index*: see [What has never been run](#what-has-never-been-run).

</details>

## The pipeline, one command at a time

`quickstart` is a shortcut through these, not a replacement for them. When you want to
change one number, run the step itself:

```bash
trainai setup                                    # is PyTorch built for this hardware?
trainai doctor                                   # what can this machine actually do?
trainai data inspect ./my-corpus                 # what did I actually give it?
trainai data prepare ./my-corpus --out data/mine # analyse + tokenize + binarize
trainai plan --data data/mine                    # measure what this machine can train
trainai train --data data/mine --plan plan.json  # train what it recommended
trainai finetune --data data/more --from runs/mine  # continue it on a second corpus
trainai eval runs/mine                           # measure it on held-out text
trainai chat runs/mine                           # talk to what you made
trainai export runs/mine --out models/mine       # take it elsewhere
```

Every one of those works today. **What is not built is absent from the CLI rather than
present and broken: if `trainai --help` lists it, it works.**

Each command's own report, with the real output it produced and the reasoning behind
what it chose, is in **[docs/walkthrough.md](docs/walkthrough.md)** — the full worked
example on a 1.1 MB corpus, start to finish. The short version of each step:

| Step | What it does that is worth knowing |
|---|---|
| `data prepare` | Reads `.txt` `.md` `.jsonl` `.json` `.csv` `.tsv` `.docx`, each also `.gz`/`.bz2`/`.xz`, plus `.zip`/`.tar.gz` archives, files with no extension, and `.sqlite`/`.db`. Counts what it read rather than estimating. Splits train/validation **by content hash**, so a duplicate document cannot leak across the split. Same input and settings produce byte-identical shards. |
| `data inspect` | Measures a corpus and writes nothing. Exits 3 when a corpus cannot be trained on, so it works as a check in a script. |
| `plan` | Runs **real training steps** at candidate configurations and reads the allocator's own counters, then recommends the largest shape that actually fit — and says what it rejected and why. |
| `train` | Fixed architecture (RMSNorm, RoPE, SwiGLU, grouped-query attention, tied embeddings), so things can be *asserted* rather than hoped: attention cannot see the future, and a resumed run continues bitwise-identically. |
| `finetune` | Continues an existing checkpoint on a different corpus. It has no model flags at all — the shape is read from the checkpoint — and it **refuses** a dataset prepared with a different tokenizer, because that failure is otherwise silent: the loss curve looks ordinary while every token id indexes the wrong embedding row. |
| `eval` | A separate code path from the trainer's validation pass, so it checks that pass rather than restating it. Reports how much of the split it scored, and labels perplexity as vocabulary-dependent. |
| `chat` | Interactive playground where temperature, top-p and the repetition penalty change between completions. A chat-trained checkpoint is prompted in the template it was trained in, and the conversation is remembered, so a follow-up question has the exchange before it behind it. A reply the token limit cut off says so instead of looking finished. Pipes work, so it is scriptable. |
| `export` | Writes a `LlamaForCausalLM` directory — the model *is* a Llama structurally, so this is a rename, not a conversion — then verifies its own output before moving it into place. |

**A spreadsheet is not a corpus, and saying so is the point.** Training a language
model on a table of numbers produces a confident nonsense generator: to a tokenizer
`9.47` and `9.48` are unrelated strings. So `.csv` and `.tsv` are read **one named
column at a time**, and because the usual way a table arrives is a CSV renamed to
`.txt`, the shape is *measured* rather than trusted to the extension — a refusal with
exit code 3, before any GPU time is spent, overridable with `--allow-tabular` and
recorded in the manifest when you do. Details and the thresholds behind them are in
[docs/corpus-formats.md](docs/corpus-formats.md).

A record can also hold a **typed conversation** — a list of `{"role", "content"}`
objects, named with `--jsonl-messages-field` — rendered with one fixed, versioned chat
template, which the dataset records so a reader never has to guess the layout. It is
typed rather than sniffed on purpose: finding the replies by searching flattened text
for `"Assistant:"` mis-marks any reply that contains that string. The share of the
corpus that is replies is reported, and `data prepare` writes the mask itself -- one
`uint8` per token, beside every shard, checksummed in the manifest and checked by
`data inspect --verify`. `trainai train` and `trainai finetune` then **score only the
replies**, weighting each micro-batch by what it scored rather than by
`1/--grad-accum`, naming any step that scored nothing instead of logging a 0.0, and
recording the choice in the checkpoint so `trainai eval` reports the same quantity.
`--no-loss-mask` scores every token deliberately, which is how you measure what the
mask bought. The layout then travels with the run: `trainai chat` prompts a model
trained this way in the template its dataset recorded, cutting the reply where the
model starts writing somebody else's turn, and refuses a template version it does not
render rather than approximating it. `--raw` and `--chat` are there to override that,
for a base-model probe and for a corpus you flattened into those labels yourself.

## Nothing starts until you have seen what it would do

`--dry-run` runs no training step and allocates nothing on the GPU. It takes about five
seconds and tells you what the run would actually produce — including the two warnings
that catch the most common way a from-scratch run is wasted:

```bash
trainai train --data data/shake --out runs/m4 --preset tiny --context 256 \
  --seq-len 256 --steps 600 --batch-size 32 --grad-accum 2 --lr 0.000424 \
  --eval-every 25 --eval-batches 8 --checkpoint-every 25 --seed 1234 --dry-run
```

<details>
<summary>What that prints (captured, and pinned by a test against the renderer)</summary>

```
------------------------------------ Plan -------------------------------------
Dataset
Location          data/shake
Tokens            333,878 train, 10,237 validation
Vocabulary        4,096
Tokenizer         cd9c3e6847fc30c7

Model
Shape             L4 d256 h4 ff704 ctx256 vocab4096
Parameters        4.26M  (3.21M outside the embedding)
Where they are    embedding 1.05M  attention 1.05M  ffn 2.16M  norm 2.30K
Weights           16.3 MiB in fp32  (plus the same again for AdamW's two
                  moments)

Training
Steps             600
Batch             32 x 2 accumulated = 64
Sequence          256 tokens
Tokens            16.4K per step, 9.83M in total
Learning rate     0.000424 peak, cosine to 4.24e-05, 20 warmup steps
Device            cuda  -  bf16 (supported by this cuda device)
Run directory     runs/m4

Data budget
Epochs            29.4 passes over the training split  (20 steps per epoch)
Tokens/parameter  0.08  (from-scratch training usually wants around 20; this
                  model size would want 85.2M tokens)

What to expect
  * This run makes 29.4 passes over the training split (333,878 tokens). Past
    about 4, validation loss usually turns around and starts rising while
    training loss keeps falling: the model is reproducing the text rather than
    learning from it. The best checkpoint will probably be from around step
    163, not the last one. Use fewer --steps, or more text.

  * 0.08 training tokens per parameter (333,878 tokens, 4,262,144 parameters).
    From-scratch training usually wants roughly 20, which for this model size
    would be about 85,242,880 tokens. Expect fluent-looking text in the shape
    of your corpus, and facts that are wrong. A smaller model would generalise
    better on this much data.

Nothing was trained. Remove --dry-run to start, or adjust the flags above
first.
```

That block is not pasted and trusted. A test re-renders the model rows from the real
preset table and asserts every one of them appears here, so a change to the parameter
formula fails the suite rather than quietly making this README wrong.

</details>

The run above was chosen to trigger both warnings, and it did what they said: validation
loss bottomed out at 4.1364 at step 375 and then rose for the remaining 225 steps while
training loss kept falling. Nothing failed. TrainAI reports it as what it is:

```
Best val loss     4.1364 at step 375  (perplexity 62.6)
Final val loss    4.2187  (worse than step 375: the model started memorising,
                  so use the best checkpoint, not the last)
```

The best checkpoint is tracked and kept regardless of where it lands, which is the half
of that warning that is reliable. The predicted *step* is a rule of thumb over epoch
counts and is labelled "probably": it was exact on one run and pessimistic by 2.3× on
another. Both are [in the walkthrough](docs/walkthrough.md#it-will-tell-you-when-a-run-is-badly-proportioned).

## Why this exists

Training a small language model from scratch is not hard because the maths is hard. It
is hard because roughly forty small decisions have to be mutually consistent —
vocabulary size against dataset size, context length against VRAM, batch size against
gradient accumulation, learning rate against batch size, warmup against total steps —
and getting any one of them wrong produces either a crash six hours in, or a model that
silently learns nothing.

TrainAI makes those decisions from measurements of your actual machine and your actual
dataset, shows you what it chose and why, and lets you override anything.

## The part that is actually novel

Most tools estimate VRAM use with a formula, then rely on catching
`torch.cuda.OutOfMemoryError` to detect a configuration that does not fit.

**On Windows that does not work.** Under the WDDM driver model the OS permits CUDA to
oversubscribe VRAM and silently pages the excess to system RAM. Measured on the
development machine (RTX 2050, 4 GiB):

| Model | Batch × context | Measured throughput | Peak VRAM |
|---|---|---|---|
| 13.9M params | 32 × 256 | **61,595 tok/s** | 1.48 GiB |
| 33.8M params | 16 × 512 | **29,923 tok/s** | 2.87 GiB |
| 110.5M params | 8 × 512 | 5,854 tok/s | 4.01 GiB ⚠️ |
| 110.9M params | 8 × 1024 | 1,304 tok/s | 6.55 GiB ❌ |

The last row peaked at **6.55 GiB on a 4 GiB card** and raised no error at all. It just
ran 47× slower than the first row. A formula-plus-`try/except` approach reports that
configuration as working, and the user finds out after wasting an evening.

So the planner runs **real training steps** at candidate configurations and reads real
allocator counters (`allocated_bytes.all.peak`, `num_alloc_retries`) plus real
throughput, then rejects anything that spills. It climbs the preset ladder from the
smallest shape upward and stops at the first failure, so the configuration that would
page 6 GiB onto a 4 GiB card is never run at all. It reports a *measured* time estimate,
never an extrapolated one.

The budget it sizes against is 85% of *free* VRAM, which on a card that is also driving
your desktop is more than you can actually commit; `--max-vram 4GB` lowers it. That flag
only ever lowers — sizing a plan against memory the device does not have would mean
measuring candidates it cannot hold, which is the failure the fit check exists to
prevent — so a cap above what is free is reported as having changed nothing rather than
obeyed. The plan records what was asked for separately from what was measured, so a
small plan on a small card is distinguishable from a small plan that was requested.

It then names which of five regimes you are in — **data-limited**, **vram-limited**,
**compute-limited**, **unconstrained**, or **blocked** — so that, for instance, a driver
fault cannot be reported as a memory limit and send you shopping for a bigger card to
fix a software problem.

## Status

An honest status table, not a roadmap wish-list. Nothing is marked done until it is
implemented **and** covered by a test in the suite CI runs on every push.

| Milestone | Scope | State |
|---|---|---|
| **M0** | Package, CLI skeleton, error taxonomy, hardware probe, CI config | ✅ done |
| **M1** | Dataset ingest, analysis, validation, BPE tokenizer, binarization | ✅ done |
| **M2** | Model + trainer, checkpoints, exact resume, live metrics | ✅ done |
| **M3** | Empirical hardware advisor (`trainai plan`) | ✅ done |
| **M4** | Evaluation, sampling, chat playground, export | ✅ done |
| **M5** | Web interface | ⬜ postponed by design |
| **M6** | Docs site, Docker, packaging polish | ⬜ postponed by design |

`0.1.0` is M0 through M4. The pipeline is complete at that version — the two postponed
milestones are additions to it rather than gaps in it, which is why it is a release and
not a preview.

## Documentation

| | |
|---|---|
| [docs/walkthrough.md](docs/walkthrough.md) | The full worked example: every command, its real output, and where each number came from |
| [docs/corpus-formats.md](docs/corpus-formats.md) | Every input format, what one document is in each, typed conversations, and the tabular-data thresholds |
| [docs/finetuning.md](docs/finetuning.md) | What `finetune` inherits, what it refuses, and the measurement behind its learning rate |
| [docs/dataset-format.md](docs/dataset-format.md) | The on-disk shard and manifest format |
| [docs/plan-format.md](docs/plan-format.md) | What `plan.json` contains |
| [docs/checkpoint-format.md](docs/checkpoint-format.md) | Checkpoint contents, and how exact resume works |
| [docs/export-format.md](docs/export-format.md) | The export layout and the logits-parity measurements |
| [docs/hardware-support.md](docs/hardware-support.md) | Every GPU path, and which of them anyone has actually run |
| [docs/design/dependencies.md](docs/design/dependencies.md) | Why the dependency list is this short |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to work on it |
| [SECURITY.md](SECURITY.md) | What the attack surface is — the corpus parsers — and what reading a checkpoint does and does not run |
| [CHANGELOG.md](CHANGELOG.md) | What changed, and the reasoning behind each change |

## Development

```bash
pytest -m "not gpu"    # fast CPU suite; this is the one CI is configured to run
pytest -m gpu          # requires a CUDA device
ruff check . && ruff format --check .
```

The CPU suite is more than 2,700 tests and runs in a few minutes. It needs no GPU and
no network, and it enforces structural rules that are easy to break by accident: source
files are pure ASCII (only `console.py` is exempt, since the glyphs it defines are the
point), importing the library never imports torch, every text write states its newline
so the bytes on disk do not depend on the platform, every declared optional extra is one
the code actually uses, and every flag and command a message names is one the CLI really
has. It also tests the hardware decisions against synthetic profiles of
thirteen machines, including AMD and Intel GPUs nobody here owns — the planner's search
runs against all of them, which is what makes "it recommends something sane on a 24 GiB
4090" a tested claim rather than an expectation.

CI runs lint, that suite on Ubuntu and Windows across Python 3.10–3.13, a per-module
coverage floor, and a build job — ten jobs, green, none with a GPU and none on macOS.

A second workflow is tag-triggered, and it does the one thing CI cannot: it checks that
the tag, the packaged version and the changelog agree, then installs the **wheel** as a
wheel and runs the suite from the unpacked **sdist**, so a module missing from the wheel
or a fixture missing from the sdist fails there rather than in somebody's install. CI's
`pip install -e .` cannot see either. `v0.1.0` is the first tag, so this release is its
first run on GitHub — its `artifacts` job is the sequence already run by hand above. It
stops at a draft release, and nothing in it publishes to an index.

## Known limitations

Stated explicitly, because the alternative is letting you discover them.

- **Single GPU only.** Multi-GPU/DDP is not implemented, not simulated, and not claimed
  to work.
- **Verified on one GPU.** Developed against an RTX 2050 (4 GiB, compute capability 8.6)
  on Windows 11. NVIDIA CUDA, AMD ROCm, Intel XPU, Apple MPS and CPU are all detected
  and handled by generic code paths, and the *decisions* for each are tested against
  synthetic profiles — but only the NVIDIA and CPU paths have ever run. See
  [docs/hardware-support.md](docs/hardware-support.md) for exactly what that distinction
  means.
- **Full fine-tuning only.** `trainai finetune` continues one of *your own* TrainAI
  checkpoints on a second corpus, and the second corpus has to be prepared with the base
  model's tokenizer. Every parameter is trained: there is no LoRA, no adapter of any
  kind, and no import of pretrained weights from anywhere else. It *is* chat tuning when
  the corpus was prepared with `--jsonl-messages-field`, because the dataset then carries
  a loss mask and `finetune` scores the assistant's replies only; on a flattened corpus
  it trains on every token, prompts included.
- **`torch.compile` is not used.** `doctor` reports whether it is usable as a fact about
  your machine, but no code path calls it, so Triton's absence costs nothing.
- **A rerun is not bitwise on CUDA.** `--seed` fixes initialisation, dropout and batch
  order, which is what its help says and all it claims; it does not make CUDA arithmetic
  deterministic. Three identical runs gave 4.1365, 4.1361 and 4.1360. `--device cpu` *is*
  bit-identical, at 17× the cost.
- **A plan's time estimate covers the training steps only**, so it excludes dataset reads
  and validation passes. On the run above it predicted 14 s against 16 s actual. It is a
  floor, and a close one at this scale; it is not a promise.
- **A plan's learning rate is not measured.** Memory and throughput come from real
  counters; the learning rate is a width-scaling rule of thumb, and the plan labels it as
  one.
- **Throughput figures are from one GPU**, and nothing above 35M parameters has been
  timed.
- **The export's logits-parity check needs `transformers`**, which is deliberately not a
  dependency. Where it is absent the check reports as `not checked` in yellow, never as a
  pass.
- **`eval` scores non-overlapping windows, not a sliding window**, so early tokens in
  each window have less context than a sliding-window evaluation would give them. The
  report says how many tokens fell outside the windows. Comparing this number against one
  computed the other way is not valid.
- **Export is single-file and inference-only.** Everything goes into one
  `model.safetensors`, and it carries no optimizer state, so it cannot be resumed from.

### What has never been run

The list above is about behaviour. This one is about *environments*, and it is separate
because a passing test suite says nothing about a machine the suite has never met. Each
of these code paths exists, is reached by tests against synthetic inputs, and has never
executed against the real thing:

| Never run against | What is exercised instead |
| --- | --- |
| **A container.** No Docker, no Kubernetes, no CI container. | The cgroup v1 and v2 readers in `hardware/probe.py` — CPU quota and memory limit — are tested against a directory the tests build to look like `/sys/fs/cgroup`. A real runtime's layout has never been read. |
| **More than one GPU.** | The planner's refusal to sum two cards' VRAM is tested against a synthetic dual-4090 profile. No two real cards have been enumerated. |
| **An AMD or Intel GPU.** | ROCm gfx-number handling and XPU detection are tested against synthetic profiles of three AMD cards and an Arc A770. Nothing has been trained on either vendor. |
| **macOS, on Apple silicon or otherwise.** | The MPS profile — no per-device VRAM, fp32 only — is synthetic. There is no macOS CI leg, so not one line of this project has run on a Mac. |
| **A GPU in CI.** | The `gpu`-marked tests run only on the development machine's RTX 2050. Hosted runners have no card. |
| **A published install.** | The build job builds an sdist and a wheel and runs `twine check`, and the wheel has been installed into a clean virtualenv from the local file and run end to end — see [Install](#install). The v0.1.0 release attaches both artifacts, so they can now be downloaded. What has never happened is an install from a *package name*: nothing has been uploaded to any index, so every install so far has begun from a path or a URL. |

Where a synthetic profile stands in for hardware, that is stated rather than glossed:
[docs/hardware-support.md](docs/hardware-support.md) lists every one of them and says
which anyone has actually touched.

## License

[Apache-2.0](LICENSE).
