# A worked example, end to end

Every number and every output block below was produced by running the command above
it. The machine was the development machine — an RTX 2050 (4 GiB, compute capability
8.6) on Windows 11, Python 3.13 — and the corpus was the 1.1 MB of Shakespeare that
`examples/get_tinyshakespeare.py` fetches. Nothing here is an illustration.

This document is the long version of what [the README](../README.md) summarises. It
exists because a tool that makes forty decisions for you has to be auditable: the
useful question is not "does it work" but "what did it decide, and on what evidence".
So each section shows the command, what it printed, and where the numbers came from.

Two conventions carried through the whole pipeline, and worth knowing before reading
any of it:

- **Every number is labelled by how strong a claim it is.** *Measured* means a real
  step ran on this machine and a counter was read. *Derived* is arithmetic over
  counted tokens. *Rule of thumb* is the honest label for the learning rate. A tool
  that prints all three in the same typeface is lying about two of them.
- **The unwelcome result is printed as the result.** A validation loss that got worse
  is reported as worse; a check that could not run reports as `not checked`, never as
  a pass.

A note on reproducing these numbers: `--seed` fixes initialisation, dropout and batch
order, and does not make CUDA arithmetic deterministic. The [rerun
section](#training-a-model) measures how far three identical commands drift apart. If
you reproduce the run below and get a validation loss of 4.1360 rather than 4.1364,
you have reproduced it.

---

## Preparing a dataset

This part is finished, so here is what it actually does. Point it at a file or a
folder of `.txt` / `.md` / `.jsonl` / `.json` / `.csv` / `.docx` (each also accepted
compressed with `.gz`, `.bz2` or `.xz`, and a `.zip` or `.tar.gz` of them works
too — or a single file with no extension at all), or at a `.sqlite`/`.db` database:

```bash
python examples/get_tinyshakespeare.py           # ~1.1 MB of text to play with
trainai data prepare data/corpus --out data/shakespeare
```

It reads the corpus twice — once to measure it and train a byte-level BPE
tokenizer from the same stream, once to encode it — and prints what it measured
before it commits to anything:

```
Measured
Documents         69
Characters        1,115,394  (1.06 MiB as UTF-8)
Length (chars)    min 14,537  p10 15,785  median 16,277  p90 16,365  max 16,383
Duplicates        0  (0.0%)
Character mix     whitespace 19%  non-ASCII 0.0%  control 0.0000%
Writing systems   latin 100%  (over 141,312 sampled chars)

Result
Split | Documents |  Tokens | Shards |  On disk
------+-----------+---------+--------+---------
train |        67 | 307,726 |      1 |  601 KiB
val   |         2 |   9,558 |      1 | 18.7 KiB
------+-----------+---------+--------+---------
total |        69 | 317,284 |      2 |  620 KiB

Compression       3.52 chars per token  (measured over the whole corpus)
Content hash      187e02d44842421e  — identical for identical input, settings and seed
```

Those are counted numbers, not estimates. The single figure that *is* an estimate
— the token count `data inspect` shows for a corpus that has not been tokenized
yet — says so on the same line.

**Reproducibility.** The same corpus, settings and seed produce shards with
identical sha256 checksums and a manifest with an identical `content_hash`. The
timestamp in the manifest is deliberately excluded from that hash, so "same input,
same hash" stays true on the second run.

**The train/validation split is by content hash**, not by shuffling: it does not
depend on file order or on any RNG's stream, and two byte-identical documents
always land on the same side, so an exact duplicate cannot leak from train into
validation.

**Bad input is named, not guessed at.** Every dataset error points at a file, a
position, and a fix:

```
Dataset decode error
  broken.txt is not valid utf-8: byte 30 begins the invalid sequence ff.
  What to do: Re-save the file as UTF-8, or pass the encoding it actually uses
  (--encoding latin-1 covers most Western European text). To drop unreadable
  files instead of stopping, pass --on-error skip.
```

**A spreadsheet is not a corpus, and saying so is the point.** A CSV is what a lot
of people have lying around, and training a language model on one produces a
confident nonsense generator: to a tokenizer `9.47` and `9.48` are unrelated
strings, so it cannot learn they are close numbers. It learns the format perfectly
and the numbers not at all.

So TrainAI reads `.csv` and `.tsv` **one named column at a time**, and refuses to
guess which column or to join several into sentences:

```bash
trainai data prepare reviews.csv --out data/reviews --csv-text-column review_text
```

And because the usual way a table arrives is a CSV renamed to `.txt`, the shape is
measured rather than trusted to the extension. Prose never puts the same number of
commas on every line; a table always does:

```
error 100% of the text sits on lines holding exactly 12 comma-separated fields,
and 66% of the characters are digits. This is a table of measurements, not text.
  If this is a spreadsheet export that was renamed to .txt, rename it back:
TrainAI reads .csv and .tsv directly, and --csv-text-column NAME then trains on a
single column of it.
  If you want to predict one column from the others, that is regression rather
than language modelling. scikit-learn's HistGradientBoostingRegressor does it in
seconds on a CPU and reports its own error.
  To train on it exactly as it is, pass --allow-tabular.
```

Exit code 3, before any GPU time is spent. If the table has few digits, one of its
columns probably holds real prose, and that is a warning naming the flag rather than
a refusal. `--allow-tabular` overrides the refusal and records the override in the
dataset's manifest — downgraded, not deleted. Thresholds, the measurements behind
them, and what each format is read as are in
[corpus-formats.md](corpus-formats.md).

To look before you commit, or to check a dataset you already have:

```bash
trainai data inspect ./my-corpus            # measure it; writes nothing
trainai data inspect data/shakespeare       # report on a prepared dataset
trainai data inspect data/shakespeare --verify   # re-hash every shard
```

`data inspect` exits 3 when a corpus cannot be trained on, so it works as a check
in a script. What TrainAI accepts as input is specified in
[corpus-formats.md](corpus-formats.md), and the on-disk format it writes
in [dataset-format.md](dataset-format.md).

---

## Planning a run

This is the part that is unusual, so here is the whole of it. `trainai plan` does not
estimate — it runs real training steps at candidate configurations, reads the
allocator's own counters, and recommends the largest shape that actually fit:

```bash
trainai plan --data data/shake
```

```
---------------------------------- Planning -----------------------------------
Machine
Platform          Windows 11 (10.0.26200)
GPU               NVIDIA GeForce RTX 2050  3.23 GiB free of 4.00 GiB
Memory budget     2.74 GiB  (85% of what is free, leaving room for the driver
                  and the desktop)
Note              this platform pages silently past VRAM  (so a config that
                  does not fit gets slow rather than failing)

Measuring candidates by running real training steps. Each one takes a few
seconds and briefly allocates VRAM.
  measuring tiny (L4 d256 h4 ff704 ctx256 vocab4096), batch 32 x 2, seq 256
    - 94.2K tokens/s, peak 957 MiB, fits
------------------------------- Recommendation --------------------------------
Plan
Model             tiny  -  L4 d256 h4 ff704 ctx256 vocab4096
Parameters        4.26M  (3.21M outside the embedding)
Steps             81 of 16.4K tokens  (1.33M in total)
Batch             32 x 2 accumulated = 64 sequences per step
Sequence          256 tokens
Time to finish    14s  (measured step time times step count, not an
                  extrapolation)
Limited by        data-limited  -  small (L6 d384 h6 ff1024 ctx512 vocab4096),
                  seq 512 was rejected -- 12.2M parameters want about 244M
                  training tokens and this corpus has 334K, so it would
                  memorise rather than learn

Measured
Throughput        94.3K tokens/s  (0.174s per step, 3 steps timed after warmup)
Peak VRAM         957 MiB of the 2.74 GiB budget  (34% used)
Device            cuda  -  bf16 (supported by this cuda device)

Derived from your data
Tokens            333,878 train, 10,237 validation
Epochs            3.97 passes over the corpus  (20 steps per pass)
Tokens/parameter  0.1 available  (from-scratch training usually wants around
                  20; this size would use 85.2M tokens)

Rules of thumb
Learning rate     0.000424  (cosine down to 4.24e-05 after 20 warmup steps;
                  nothing here measured whether this is right for your text)
Evaluation        every 5 steps, 2 batches  (scaled to the step count so the
                  best checkpoint has something to be best among)
Checkpoints       every 10 steps, keeping 3
```

**Every number is labelled by where it came from**, because those are three
different strengths of claim. *Measured* means a real step ran on this machine and a
counter was read. *Derived from your data* is arithmetic over counted tokens.
*Rules of thumb* is the honest label for the learning rate: it is 3e-4 at `d_model`
512 scaled by inverse square root of width, and nothing here tested whether that
suits your text. A tool that prints all three in the same typeface is lying about
two of them.

It then explains what it refused and why, and writes the plan out:

```
Why not something bigger
  * small (L6 d384 h6 ff1024 ctx512 vocab4096), seq 512  -  12.2M parameters
    want about 244M training tokens and this corpus has 334K, so it would
    memorise rather than learn

------------------------------------ Next -------------------------------------
Plan written to plan.json. Run it with:

  trainai train --data data/shake --preset tiny --context 256 --seq-len 256
--steps 81 --batch-size 32 --grad-accum 2 --lr 0.000424 --eval-every 5
--eval-batches 2 --checkpoint-every 10

Or apply the whole plan without retyping it: trainai train --data data/shake
--plan plan.json
```

Both of those produce the same run. The long form is printed because a plan you
cannot read is a plan you cannot disagree with; `--plan` exists because retyping
eleven flags is how a typo gets in. Any flag you pass alongside `--plan` wins, so
`--plan plan.json --steps 500` is a supported thing to want.

**How the search works.** It climbs the preset ladder rather than descending it —
starting at the smallest shape and stopping at the first one that fails — so the
configuration that would page 6 GiB onto a 4 GiB card is never run. When a rung is
over budget the micro-batch is halved and `grad_accum` doubled, which holds the
effective batch constant so the retry has the same learning dynamics rather than
quietly becoming a different experiment. The smallest rung is always measured, so a
"nothing fits" failure still reports a real number.

`--time 2h` adds a third cap on top of memory and data: candidates whose measured
step time cannot finish in the budget are rejected, and the accepted one has its step
count cut to fit. `--max-preset`, `--seq-len`, `--precision` and `--device` pin
whatever you would rather decide yourself. `--json` emits the plan and nothing else.
When nothing fits at all, it exits 4 and shows you the measurement of the smallest
shape it tried, rather than a formula's opinion about it. The written format is
specified in [plan-format.md](plan-format.md).

---

## Training a model

The model is a decoder-only transformer with the modern set of choices — RMSNorm,
RoPE, SwiGLU, grouped-query attention, no biases, tied embeddings — implemented
once, with the shapes fixed by a validated config. Fixing the architecture is what
makes it possible to *assert* things about it rather than hope: that attention
cannot see the future, that a resumed run continues bitwise-identically.

Start with `--dry-run`. It runs no training step and allocates nothing on the GPU —
it does open a CUDA context, because asking the device what precision it supports is
half the point — and it tells you what the run would actually produce. About five
seconds. This is the run the rest of this README uses, measured on the development
machine (RTX 2050, 4 GiB, Windows 11), on the 1.1 MB Shakespeare corpus, at the shape
`trainai plan` recommended:

```bash
trainai train --data data/shake --out runs/m4 --preset tiny --context 256 \
  --seq-len 256 --steps 600 --batch-size 32 --grad-accum 2 --lr 0.000424 \
  --eval-every 25 --eval-batches 8 --checkpoint-every 25 --seed 1234 --dry-run
```

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

Those two warnings are the subject of [the section below](#it-will-tell-you-when-a-run-is-badly-proportioned),
and this run was chosen partly to trigger them. Drop `--dry-run` to train it:

```bash
trainai train --data data/shake --out runs/m4 --preset tiny --context 256 \
  --seq-len 256 --steps 600 --batch-size 32 --grad-accum 2 --lr 0.000424 \
  --eval-every 25 --eval-batches 8 --checkpoint-every 25 --seed 1234
```

| | |
|---|---|
| Model | 4.26M parameters, 4 layers, d256, 4 heads, 256-token context, 4096-token vocabulary |
| Precision | bf16, chosen from what the GPU reports supporting |
| Throughput | **90,300 tokens/s** (9.83M tokens in 1m 48s) |
| Peak VRAM | **957 MiB** of the 3.23 GiB free |
| Best validation loss | **4.1364** at step 375 (perplexity 62.6) |
| Final validation loss | 4.2187 at step 600 — *worse*, and reported as worse |

That last row is the point of the two rows above it. Validation loss fell to
4.1364 at step 375 and then rose for the remaining 225 steps while training loss
kept falling from 3.73 to 3.39. Nothing failed and no error was raised. TrainAI
says so in the result rather than printing the final number as if it were the
best one:

```
Best val loss     4.1364 at step 375  (perplexity 62.6)
Final val loss    4.2187  (worse than step 375: the model started memorising,
                  so use the best checkpoint, not the last)
```

**Exact resume.** Stopping at step 12 of 24 and resuming produces weights and AdamW
moment buffers that are *bitwise identical* to an uninterrupted 24-step run, and an
identical final loss. That works because there is nothing to synchronise: the
learning rate is `lr(step)` and the batch order is `batch(seed, step)`, both pure
functions, so resume restores the weights and continues. There is no scheduler
counter to advance the right number of times. See
[checkpoint-format.md](checkpoint-format.md). It holds on CUDA as well as
on CPU, and there is a test for each.

**A rerun is not bitwise, and the fourth decimal is why.** `--seed` fixes
initialisation, dropout and batch order, which is what its help says and all it
claims. It does not make CUDA arithmetic deterministic. Running the command above
three times on the same machine gave best validation losses of 4.1365, 4.1361 and
4.1360: bit-for-bit identical through step 120, first disagreeing at step 130, and
1.5e-03 apart in training loss by step 600 — one non-deterministic reduction, then
divergence that compounds. So treat the numbers in this section as one measurement
rather than a target to match exactly. If you reproduce this run and get 4.1360, you
have reproduced it.

`--device cpu` does not have this problem: two 200-step runs of the same shape and
seed came out bit-identical, all 38 tensors and every logged loss, well past the step
where the GPU runs had already diverged. It cost 5,271 tokens/s against 90,168 on the
GPU — 17× — so it is a way to pin a number you need pinned, not a way to train.

### It will tell you when a run is badly proportioned

The first real GPU run on this corpus used a 13.8M model for 1,200 steps. That is
32 passes over a 308k-token training split, and here is what actually happened:

| step | train loss | val loss | epochs |
|---|---|---|---|
| 100 | 4.892 | 4.768 | 2.7 |
| 300 | 3.519 | **4.248** | 8.0 |
| 600 | 1.756 | 5.079 | 16.0 |
| 1200 | **0.406** | 5.981 | 31.9 |

The training loss reached 0.41, which looks like success and is not: the model
memorised the corpus. Nothing failed, no error was raised, and the final checkpoint
was worse than the one from step 300.

Two ratios predict that before a single step runs, so TrainAI now computes both and
says so up front — and the epoch-based estimate of where the best checkpoint would
land came out at step 300, which is where it actually was:

```
note: This run makes 31.9 passes over the training split (307,726 tokens). Past
about 4, validation loss usually turns around and starts rising while training loss
keeps falling: the model is reproducing the text rather than learning from it. The
best checkpoint will probably be from around step 300, not the last one. Use fewer
--steps, or more text.
```

**How well does that prediction hold up?** Two runs have tested it, and the answer
is "the direction is right, the step number is a guess":

| Run | Predicted turnaround | Actual best checkpoint |
|---|---|---|
| 13.8M params, 1,200 steps, 32 epochs | step 300 | step 300 |
| 4.26M params, 600 steps, 29 epochs | step 163 | step **375** |

The second one was pessimistic by 2.3×. That is worth stating plainly: the useful
half of the warning is *"the last checkpoint will not be the best one, so use the
best one"*, which was true both times and is what TrainAI acts on — the best
checkpoint is tracked and kept regardless of where it lands. The step estimate is a
rule of thumb about epoch counts and is labelled "probably" for this reason.

`trainai plan` uses the same arithmetic to *choose* a shape from real measurements
instead of only describing the one you asked for.

### Other behaviour worth knowing

- **Precision is resolved from the hardware, not requested and hoped for.** `auto`
  gives bf16 only where compute capability 8.0+ supports it, fp16 with a gradient
  scaler otherwise, fp32 on CPU — and prints which, because "mixed precision
  enabled" that quietly meant fp32 is the sort of claim this project exists to
  avoid.
- **An explicit `--device cuda` is never silently downgraded.** Falling back to CPU
  would turn a twenty-minute run into a twelve-hour one without saying so.
- **Divergence stops the run before it is written down.** A non-finite loss means
  that step's gradients are useless, so the step is not applied: the weights, and
  any checkpoint taken afterwards, stay at the last good value.
- **Gradient accumulation averages, it does not sum.** Getting that wrong
  multiplies the effective learning rate by `--grad-accum` and looks like an
  unstable model rather than an arithmetic mistake.
- **Training into an existing run directory is refused** unless `--resume` or
  `--force` says otherwise, because it would mix two runs' metrics into one file.

Everything measured goes to `<run>/metrics.jsonl`, one JSON object per line,
flushed after every record so a killed process still leaves the interesting part on
disk.

Reading it back never fails. A line that is not readable — a fragment a killed
process left, a line an editor re-saved in the wrong encoding, a value that is not a
number where a loss belongs — is dropped, the complete records around it are still
summarised, and the count of what was dropped is reported as
`unreadable_metric_lines` in the run summary. This file describes what already
happened and no model depends on it, so refusing to read it would throw away a
finished run's summary over a log; being quiet about the dropped lines would report a
summary computed from fewer records than the file has, with nothing saying so.

---

## Measuring what you got

`trainai eval` scores a checkpoint on held-out text. It is a separate code path
from the trainer's own validation pass, which makes it a check on that pass rather
than a restatement of it:

```bash
trainai eval runs/m4 --which best
```

```
Val split
Loss              4.1364 nats/token
Perplexity        62.6
Bits/token        5.9676
Scored            all of it  (38 of 38 sequences, 9,728 predictions, 5 batches)

Measured with
Context           256 tokens
Vocabulary        4,096 tokens  (perplexity is per token of this vocabulary)
Precision         bf16 (supported by this cuda device)
Run recorded      4.1364 at step 375  (0.0000 from what was just measured; the
                  trainer samples --eval-batches of the split, this scored what
                  you asked for)
```

Three things there are deliberate.

**It reconciles itself against the run.** The trainer recorded 4.1364 at step 375;
scoring the whole split independently reproduced it to four decimal places, and the
report prints the difference rather than leaving you to notice it. When the two
*do* disagree — because the trainer samples `--eval-batches` while this scores
everything — the line says which is which.

**It says how much of the split it scored, and what it skipped.** `38 of 38
sequences` and `coverage 1.0`, plus a note that the split's last 471 tokens do not
fill a 256-token sequence and so were not scored. A number quoted over an
unstated fraction of the data is not a measurement.

**Perplexity is labelled as vocabulary-dependent.** 62.6 on a 4,096-token
byte-level BPE is not comparable to 62.6 on a 50,257-token vocabulary, and
perplexity tables that omit the tokenizer are how models get compared wrongly.
Bits per token is printed alongside for the same reason.

Running the same command on the *last* checkpoint instead of the best one
confirms the turnaround from the other side:

| `--which` | Step | Loss | Perplexity | Bits/token |
|---|---|---|---|---|
| `best` | 375 | **4.1364** | **62.6** | **5.9676** |
| `latest` | 600 | 4.2187 | 67.9 | 6.0862 |

`--split train` scores the training split too, `--data` points at a different
dataset, and `--precision fp32` pins the third decimal, which moves under bf16.
`--json` emits the whole thing for a script.

---

## Talking to it

```bash
trainai chat runs/m4 --prompt "KING RICHARD:" --tokens 120 --seed 7 --precision fp32
```

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

That is what 4.26M parameters trained on 1.1 MB of text actually produces, printed
here unedited because it is the single most useful thing to show someone before
they start. It has learned the *form* — speaker headings, line breaks, the register,
real character names — and words like "siped", "purfail'd" and "blined" do not
exist. It is not going to answer your questions, and `chat` says so on the way in
rather than letting you discover it.

With no `--prompt` it is an interactive playground where temperature, top-p and the
repetition penalty can be changed between completions, so you can see what those
do to your own model instead of reading about them. Piped input works too, so
`echo "Once upon" | trainai chat runs/m4` is scriptable. `--seed` with
`--precision fp32` reproduces a sample exactly.

---

## Taking it elsewhere

```bash
trainai export runs/m4 --out models/shakespeare-4m
```

```
Export
Directory         models/shakespeare-4m
Format            hf  -  weights in fp32
Size              16.5 MiB  (7 files)
From              step-0000375.pt  (step 375  -  validation loss 4.1364)
Model             4.26M parameters  -  4 layers  -  4 heads  -  d_model 256  -
                  context 256

Checks
weights round-trip    ok  38 tensors re-read from disk and bit-identical to the
                      model
tokenizer round-trip  ok  fingerprint cd9c3e6847fc matches the run
logits parity         ok  transformers loaded the export; max logit difference
                      3.81e-06 over 16 positions, within the 2e-04 tolerance
```

The result is a `LlamaForCausalLM` directory, because the model **is** a Llama
structurally — RMSNorm, rotary embeddings, SwiGLU, grouped-query attention, no
biases. So the export is a rename, not a conversion. Loading it needs nothing from
this project:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("models/shakespeare-4m")
model = AutoModelForCausalLM.from_pretrained("models/shakespeare-4m")
```

That prints `LlamaForCausalLM` and `4262144` parameters, and generates Shakespeare
of the same quality as above — verified, not assumed.

**Exporting weights is the one operation where a mistake is invisible.** A wrong
`model_type`, a swapped rotary convention, a transposed gate/up pair or a wrong
`rope_theta` all produce a directory that loads without complaint and generates
plausible nonsense. There is no checksum on "do the weights in this file match the
architecture in this config". So the export checks itself before it is moved into
place, and the three lines above are what it found — not a claim that the code has
tests, though it does.

The third check is the one that tests what the export *means*: it loads the
directory back through `transformers` and compares logits against ours. It is also
the check that cannot run everywhere, because `transformers` is deliberately not a
dependency of this project. Where it is absent, it prints as **`not checked`** in
yellow with the reason, never as a pass. And it has a control in the test suite: a
test corrupts `rope_theta`, which leaves every tensor shape intact, and asserts the
export is refused — because a check that cannot fail is not evidence.

Everything else follows from the same idea. An export is written to a staging
directory, verified there, and moved into place only once every check passes, so a
failure leaves no half-written directory that loads far enough to produce garbage.
`--force` replaces an existing export and refuses anything that is not one, so it
cannot be turned into a way to delete a folder you care about. A `README.md` goes
into the directory saying it is a base model that completes text, because that is
the only place the warning survives the directory being zipped and sent to someone
else.

`--format safetensors` writes the same weights under TrainAI's own names for
loading into your own code; `--dtype bf16` halves the file. The format is specified
in [export-format.md](export-format.md).

---
