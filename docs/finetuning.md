# Fine-tuning

`trainai finetune` continues a model you already trained on a second corpus.

```bash
# 1. the base model, trained the normal way
trainai data prepare ./books --out data/books
trainai train --data data/books --out runs/books

# 2. the second corpus, prepared with the BASE MODEL'S tokenizer
trainai data prepare ./letters --out data/letters --tokenizer data/books

# 3. continue the model on it
trainai finetune --data data/letters --from runs/books --out runs/letters
```

Step 2 is the one that is easy to get wrong, and the whole reason this page exists.

## The tokenizer requirement is not a formality

`data prepare` without `--tokenizer` trains a **new** BPE tokenizer on whatever corpus
it was given. That is the right default for a from-scratch run and completely wrong
here, because a token id is a **row index into the embedding matrix**. Id 412 in the
base model's vocabulary is one particular subword; id 412 in a freshly trained
vocabulary is some unrelated one. Feed the second into the first and every lookup
returns a vector trained for different text.

Nothing raises. The loss starts high, falls, and the curve looks entirely ordinary.
You get a run directory, checkpoints, a metrics log and a model that has learned
nothing usable. That is why this is a **refusal** and not a warning:

```
Checkpoint error
The base model was trained with a different tokenizer.
What to do: Every token id in this dataset would index a row of the embedding matrix
that was trained for a different piece of text. Fine-tune against a dataset prepared
with the same tokenizer as the base model -- `trainai data prepare --tokenizer
<the base run's tokenizer.json>` reuses it instead of training a new one.
```

Exit code 5, before the plan is printed and before the model is built. `--dry-run`
refuses it too — a dry run exists to say what the real run would do, and "it would
refuse" is part of that.

A different `content_hash` is **not** checked, because a different corpus is the entire
point of the command.

## What is inherited and what is not

| | Inherited from the base checkpoint |
|---|---|
| Model weights | ✅ yes — this is the command's only job |
| Model shape | ✅ yes, and it cannot be overridden |
| Optimizer moments (AdamW `exp_avg`, `exp_avg_sq`) | ❌ no |
| RNG state | ❌ no |
| Step counter | ❌ no — a fine-tune starts at step 0 |
| Learning-rate schedule | ❌ no — a fresh warmup and decay over the new `--steps` |

The shape is **read, not chosen**: `finetune` has no `--layers`, `--heads`, `--width`,
`--ffn-width`, `--context` or `--preset`, because `Checkpoint.apply_to` requires an exact
match. A flag that parsed could only ever produce a load failure several seconds later,
phrased as a shape mismatch the user did not cause. The base model's `--context` is
therefore also a ceiling on `--seq-len`.

The moments and the RNG are dropped deliberately. They describe the schedule of the run
that produced them, over text this run is not training on; a fine-tune is a new schedule
over different data, and carrying a decayed optimizer state into it means the first steps
move in directions chosen for a corpus that is no longer present. `--resume` is the
command that inherits all of it, and it is a different command for that reason.

The run record's first metrics line carries `finetuned_from` (the checkpoint path) and a
`parent` block with the base model's step, its shape and its dataset identity, so a
tuned run directory can answer "which model is this" without shell history.

## The learning rate

The default is `--lr 3e-5`, one tenth of the from-scratch default. It is measured.

The measurement is worth describing because the first two attempts at it produced an
answer that could not be true. Both said the **highest** rate tested was best — which is
what you always get if you score each candidate only on the corpus being tuned *onto*.
On that corpus, more movement is more improvement, by construction. A measurement that
cannot see the thing it is measuring is not a measurement: the cost of a high rate is
paid on the *base* corpus, and neither sweep looked at it.

The third scored **both** validation splits. A 2-layer, 128-wide model (vocab 600, ctx
128) trained 1,500 steps on a 270,501-token corpus to validation **2.9384**, then
fine-tuned 200 steps on a separate 437,931-token corpus prepared with the same
tokenizer. Validation loss on the tune corpus before any tuning was 4.6463; training on
it from scratch for the same 200 steps reaches 4.5586.

| peak `--lr` | val, tune corpus | val, base corpus |
|---|---|---|
| `3e-4` | 3.4153 | 3.6599 |
| `1e-4` | 3.6714 | 3.3798 |
| `3e-5` | **3.9089** | **3.1883** |
| `1e-5` | 4.1907 | 3.0558 |

(Base model, for reference: 2.9384 on the base corpus.)

Monotone in both directions, with no dominant point. **The choice is a preference, not a
measurement** — the measurement only shows the exchange rate. This default takes the
conservative end: a user who typed `finetune` rather than `train` has said the base model
matters. At `3e-5` the run gains 0.74 nats on the new corpus and gives up 0.25 on the
old; at `3e-4` it gains 1.23 and gives up 0.72.

If you do not care about the base corpus at all, `--lr 3e-4` is the better setting, and
if the two corpora are very different you may want higher still. One measurement, one
model size, two similar prose corpora — treat the table as an illustration of the shape
of the trade, not as a constant. The result panel always names the rate that was used.

## The data budget reads differently

A fine-tune corpus is small relative to the parameter count almost by definition, so the
`20 tokens per parameter` reference from the Chinchilla scaling work fires on essentially
every fine-tune. On a from-scratch run its remedy is real: use a smaller model. Here the
model shape is fixed by the checkpoint, so that advice is not available, and the note
says something useful instead — the weights already encode the base corpus, this run is
adjusting them rather than filling them in, the result is only as good as the base model
at anything the new corpus does not cover, and a lower `--lr` preserves more of it.

The epoch warning is unchanged and still applies. A small tune corpus and a large
`--steps` will memorise it exactly as fast as it would from scratch.

## What this is not

- **Not LoRA**, and not an adapter of any kind. Every parameter is trained.
- **Not instruction or chat tuning by itself.** Fine-tuning on a flattened corpus
  trains the model to predict every token, a prompt included. It becomes chat tuning
  when the corpus is prepared with `--jsonl-messages-field`: the dataset then carries a
  loss mask, and `finetune` scores the assistant's replies only. See
  [corpus-formats.md](corpus-formats.md#the-loss-mask-on-disk).
- **Not an importer.** `--from` takes a TrainAI checkpoint. There is no path from a
  Hugging Face checkpoint into this command, and `trainai export` goes the other way only.
