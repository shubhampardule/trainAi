# Dependency decisions

Every required dependency is a thing a user has to download, a thing that can
break on a Python upgrade, and a thing a contributor has to have working before
they can run the tests. So the list is short and each entry is justified here.

The install must stay a plain `pip install` — no Node, no C compiler, no
system packages.

## Required

| Package | Why it is required |
| --- | --- |
| `torch` | The training runtime. Nothing else in the ecosystem gives the same combination of a stable autograd API, a memory allocator we can *query* (`torch.cuda.memory_stats()`), and fused optimizers. The allocator introspection is not a nice-to-have here; the planner is built on it. |
| `numpy` | Binarized datasets are `numpy` memmaps. Reading a token window from disk without loading the corpus into RAM is the entire point of the format. |
| `tokenizers` | Byte-level BPE training, in Rust. Reimplementing BPE in Python would be slower by orders of magnitude and would be a worse implementation. This is the one piece of the HuggingFace stack we use directly. |
| `safetensors` | The export format. `trainai export` writes `model.safetensors`, so a shared model has no pickle in its loading path and cannot execute code on the machine that downloads it. Checkpoints are *not* safetensors — they are a single `torch.save` payload holding optimizer and RNG state, loaded with `weights_only=False`, which is why they are documented as trusted-input-only in [checkpoint-format.md](../checkpoint-format.md). Sharing goes through the export, and that is the path this dependency protects. |
| `pydantic` | Validating configs that come from disk or from a user's JSON, with error messages that name the offending field. Hand-rolled validation with comparable messages is a lot of code that would need its own tests. |
| `typer` | The CLI. Chosen over bare `click` for the type-hint-driven signatures, and over `argparse` because subcommand groups with good `--help` output in `argparse` is a lot of boilerplate. |
| `rich` | Live training display and formatted tables. Also handles the terminal-capability detection that makes output survive a Windows console. |
| `psutil` | RAM and process memory. There is a stdlib fallback (`ctypes` + `GlobalMemoryStatusEx` on Windows, `os.sysconf` on POSIX) in `hardware/probe.py`, so `psutil` is not load-bearing — it is more accurate and more portable than the fallback, which is worth one small pure-Python-plus-C-extension dependency. |

## Deliberately not required

### `transformers`

TrainAI writes its own ~400-line decoder-only GPT and *exports* to
HuggingFace format instead of training through `transformers`.

1. **The planner needs predictable memory behaviour.** Deciding whether a
   configuration fits requires knowing exactly what the forward and backward
   passes allocate. That means owning the forward pass.
2. **Stability.** Training-from-scratch through the `transformers` API couples
   this project to a large, fast-moving surface that is optimized for a
   different use case (loading pretrained models).
3. **The from-scratch model is the point.** A user who wants to understand what
   they are training should be able to read the whole model in one sitting.

Exporting *to* HF format means the result is usable everywhere the ecosystem
reaches, which is what actually matters for the output. `trainai export` writes a
`LlamaForCausalLM` directory — the architecture is a Llama, so the export is a
rename rather than a conversion.

That export verifies itself by loading the directory back through `transformers`
and comparing logits against ours. Since `transformers` is not required, that
check cannot run everywhere, so it reports itself as **not checked** where the
package is absent rather than implying it passed. The weight and tokenizer
round-trips do not need it and always run.

### `datasets`

Users bring their own text files. TrainAI ingests them and writes its own
binary format: `uint16`/`uint32` memmap shards plus a checksummed
`manifest.json` recording token counts, a sha256 per shard, and the hash of the
tokenizer that produced them.

That format is faster to read than a generic columnar loader, is reproducible
(same input plus same seed gives the same checksums), and adds nothing to the
install. Hub integration can arrive later as an optional extra, where the cost
falls only on people who want it.

### `accelerate`, `deepspeed`, `bitsandbytes`

All exist to solve problems TrainAI v0.1 does not have: multi-GPU
orchestration, sharded optimizers, and quantized training. Adding them would
mean shipping code paths that cannot be tested on the hardware available to the
project. See the roadmap in the [README](../../README.md).

### `torch.compile` / Triton

Not a dependency at all — and not used. Triton is not installed in every environment
(it is absent on the development machine and on Windows generally), so its
availability is feature-detected via `importlib.util.find_spec("triton")` and
reported by `trainai doctor` as a fact about the machine. No code path calls
`torch.compile`, so that fact currently changes nothing about a run.

`TrainConfig` used to carry a `compile_model` flag whose docstring offered to "try
`torch.compile`". Nothing read it. It has been removed rather than left as a switch
that does nothing, on the same reasoning that removed the `web` extra: a promise in
the configuration surface is a promise. When compilation is actually wired up and
measured, the flag comes back with it.

## Optional extras

| Extra | Contents | Why it is optional |
| --- | --- | --- |
| `dev` | `pytest`, `pytest-cov`, `ruff` | Contributors only. |

### Planned, and deliberately not declared yet

`web` — `fastapi`, `uvicorn[standard]`. The web interface is a view over the run
directory, and people using the CLI should not pay for an ASGI server. It will
deliberately have **no** npm build step: static HTML plus vanilla JS plus
server-sent events.

It is **not** in `pyproject.toml`, because [M5](../../README.md) is postponed and
there is no server to run. `pip install trainai[web]` used to install both
packages — several of them compiled — and deliver nothing, which is the kind of
promise this project says it does not make. The extra goes back in the same commit
as the server.

## Adding one

Open the PR with an answer to these:

- What breaks without it, concretely?
- Can it be an optional extra instead?
- Does it need a compiler, Node, or a system package to install? (If yes, the
  answer is no.)
- Does it work on Windows, macOS, and Linux, on every Python version in the CI
  matrix?
