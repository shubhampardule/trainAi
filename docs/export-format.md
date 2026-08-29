# The export format

`trainai export` turns a finished run into a standalone model directory that
other tools can load. This document specifies what it writes, what it verifies,
and what it deliberately does not claim.

```bash
trainai export runs/my-run --out models/my-model
```

```
models/my-model/
  model.safetensors          the weights, under HuggingFace's names
  config.json                a LlamaForCausalLM config
  generation_config.json     sampling defaults matching `trainai chat`
  tokenizer.json             the tokenizer the model was trained with
  tokenizer_config.json      PreTrainedTokenizerFast
  special_tokens_map.json
  README.md                  what this model is, in the directory itself
```

Two formats are available:

| `--format` | What it is |
|---|---|
| `hf` (default) | A directory `transformers.AutoModelForCausalLM.from_pretrained()` loads. |
| `safetensors` | The same weights under TrainAI's own tensor names, plus `trainai-model.json` holding the `ModelConfig`. For loading into your own code without depending on `transformers`. |

`--dtype` is `fp32` (default), `fp16` or `bf16`. fp32 is the default because it is
what the checkpoint holds, so it is the only lossless choice; the reduced ones
halve the file at a cost the parity check accounts for (see below).

## Why the target is Llama, not GPT-2

The model TrainAI trains *is* a Llama, structurally: RMSNorm, rotary position
embeddings, SwiGLU feed-forwards, grouped-query attention, no biases anywhere,
optionally tied embeddings. So the export is a **rename, not a conversion** —
every tensor crosses unchanged.

This matters because the alternative fails silently. Writing
`"model_type": "gpt2"` would produce a directory that loads without complaint and
generates nonsense: GPT-2 has learned position embeddings, LayerNorm with biases,
a fused `c_attn` in `Conv1D` layout, and a GELU MLP. None of those describe this
model, and **nothing in the format checks that the architecture named in the
config matches the weights in the file.** A wrong `model_type` is a silent wrong
answer, so it gets a measurement rather than an argument.

### The name mapping

| TrainAI | HuggingFace |
|---|---|
| `embed_tokens.weight` | `model.embed_tokens.weight` |
| `final_norm.weight` | `model.norm.weight` |
| `lm_head.weight` | `lm_head.weight` |
| `blocks.N.attn_norm.weight` | `model.layers.N.input_layernorm.weight` |
| `blocks.N.ffn_norm.weight` | `model.layers.N.post_attention_layernorm.weight` |
| `blocks.N.attn.{q,k,v,o}_proj.weight` | `model.layers.N.self_attn.{q,k,v,o}_proj.weight` |
| `blocks.N.ffn.{gate,up,down}_proj.weight` | `model.layers.N.mlp.{gate,up,down}_proj.weight` |

A tensor with no entry in that table is an **error**, not a warning. Renaming a
module in `trainai/model/gpt.py` without updating `trainai/export/hf.py` would
otherwise drop weights from the export and produce a file that loads and is
wrong.

The rotary cos/sin buffers are registered `persistent=False`, so they never
appear in `state_dict()` and need no mapping — they are recomputed from
`rope_theta`.

### Two details that are easy to get wrong

**No q/k permutation.** HF's `rotate_half` splits the head dimension in half and
treats the halves as real and imaginary parts. So does `trainai.model.gpt
.apply_rotary`. The *other* convention in the wild interleaves adjacent pairs, and
converting between them requires permuting the `q_proj` and `k_proj` rows — which
is what the reference Llama-to-HF conversion script does, and what must **not**
happen here.

**Tied embeddings drop a key.** `safetensors` refuses to write two names that
share one storage, so `lm_head.weight` is omitted when the model ties it and
`tie_word_embeddings: true` tells the loader to re-tie. This is the same
arrangement HF's own tied models use.

## What the export verifies about itself

Writing weights is the one operation where a mistake cannot be noticed later,
because the output is plausible either way. So every export re-checks itself
before it is moved into place, and the report names each check:

| Check | What it does | When it runs |
|---|---|---|
| **weights round-trip** | Re-reads the `model.safetensors` just written and compares it, **bit-exactly**, tensor by tensor, against what was handed to the writer. Not approximate: same process, same tensors, so any difference at all is a serialisation fault. | Always |
| **tokenizer round-trip** | Re-loads the written `tokenizer.json` and compares its fingerprint against the run's. A tokenizer that does not match the weights produces fluent text unrelated to what the model computed, and nothing downstream can detect it. | Always |
| **logits parity** | Loads the directory back through `transformers` as a real `LlamaForCausalLM` and compares its logits against ours over 16 positions. Tolerance `2e-4`; the measured difference is always reported, not just the verdict. | Only where `transformers` is installed, and only for `--format hf` |

The parity check is the only one that tests what the export *means* rather than
what it contains. A swapped rotary convention, a wrong `num_key_value_heads`, a
gate/up transposition, or a wrong `rope_theta` all produce a file that loads
cleanly and generates nonsense — and all of them move the logits.

At `--dtype fp16`/`bf16` the exported file genuinely holds different numbers than
the checkpoint. The comparison is therefore made against a copy of the model cast
to the export dtype and back to fp32, so the check is about the mapping rather
than about rounding.

### A skipped check is reported as skipped

`transformers` is **not a dependency of this project** (see
[design/dependencies.md](design/dependencies.md)), so on most machines the parity
check cannot run. `VerifyCheck.ran = False` is a first-class outcome: the terminal
prints a yellow `not checked` line with the reason, and `--json` carries
`"ran": false`. It is never presented as a pass. The same applies to
`--no-verify`, which reports all three checks as skipped rather than omitting
them.

### The check has a control

A verification that passes is only evidence if it can fail. So
`tests/test_export_hf.py` corrupts `rope_theta` — a field no shape depends on, so
every tensor still loads — and asserts the export is refused. Without that test,
the parity check could be comparing something against itself and nobody would
know.

Measured here on the best checkpoint of `runs/m4` (step 375,
`L4 d256 h4 ff704 ctx256 vocab4096`, 4.26M parameters), against
`transformers` 5.3.0. Every `--dtype` the command accepts has a row, because a
missing one reads as untested:

| `--dtype` | Max logit difference over 16 positions | Tolerance |
|---|---|---|
| `fp32` | 4.05e-06 | 2e-04 |
| `fp16` | 5.25e-06 | 2e-04 |
| `bf16` | 4.29e-06 | 2e-04 |

Every row is between thirty-eight and fifty times inside the tolerance, and all
three are close to *each other* — which is the design rather than a coincidence,
and is worth understanding before trusting the check. The reference the export is
compared against is cast to the export dtype and back to fp32 first
(`_reference_model`), so what the check measures is the **mapping**: tensor names,
the rotary convention, the dropped `lm_head`. If it compared against the
un-rounded checkpoint instead, the `fp16` and `bf16` rows would be dominated by
their own rounding rather than by anything about the mapping, and the tolerance
would have to be widened until the check could no longer fail for the reason it
exists.

The spread between the three rows is far too small, on one model measured
over 16 positions, to order the dtypes by. Do not read it as `fp16` being worse
than `bf16`. And these figures are from one model on this development machine and
are not a guarantee for another; the check re-runs wherever `transformers` is
installed and reports its own number rather than this one.

## An export is complete or absent

Everything is written into `<out>.partial-<pid>` beside the destination, verified
there, and moved into place with `os.replace` only once every check passes. A
crash, a full disk, or a failed verification leaves the destination untouched.

The alternative — writing in place — produces a directory holding a `config.json`
and half a tensor file, which loads far enough to produce garbage.

`--force` replaces an existing export, and **only** an existing export. A
directory is eligible for replacement solely if it contains `model.safetensors`
*and* one of `config.json` / `trainai-model.json`. Anything else is refused even
with `--force`, because a flag that deletes a directory must not be able to reach
one the user cares about.

## Provenance travels with the weights

The `README.md` written into the directory states, in the directory itself:

- that this is a **base model that completes text**, not an instruction-following
  assistant;
- parameter count, layers, heads, model dimension, context length, vocabulary;
- the dtype of the weights, the step it was exported from, and the best
  validation loss;
- the tokenizer fingerprint, and that no other tokenizer will produce sensible
  output from these weights;
- which checkpoint of which run it came from.

That file is the only place the message survives the directory being zipped and
sent to someone else, which is the situation it exists for.

## Flags

| Flag | Default | Notes |
|---|---|---|
| `--out`, `-o` | *required* | No default: a defaulted destination would write a large directory somewhere unasked. |
| `--format`, `-f` | `hf` | `hf` or `safetensors`. |
| `--dtype` | `fp32` | `fp32`, `fp16`, `bf16`. |
| `--which` | `best` | `best` or `latest`, when the target is a directory. A checkpoint file can also be named directly. |
| `--tokenizer` | — | Override the tokenizer path, for a run whose dataset moved. |
| `--no-verify` | off | Skips the checks for a very large model where the re-read is slow. The report then says all three were skipped. |
| `--force` | off | Replace an existing export. |
| `--json` | off | The whole result on stdout, nothing else. Errors stay on stderr. |

Exit code **7** (`ExitCode.EXPORT`) means the destination was occupied by
something that is not an export, or a verification failed. Exit code **2**
(`ExitCode.USAGE`) means an unknown `--format`/`--dtype`, or nothing loadable at
the target.
