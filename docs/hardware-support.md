# Hardware support

The short version: TrainAI is written to run on whatever accelerator your machine
has, and it has only been *verified* on one of them. This page separates those two
things, because they are different claims and conflating them is how projects end
up promising hardware support they cannot back.

## Run `trainai setup` first

```bash
trainai setup
```

It checks the system for a GPU independently of what PyTorch can see, and tells you
if the two disagree. That case is worth a command of its own because it is common
and invisible:

> PyPI's default `torch` wheel is CPU-only on some platforms. Someone with a
> perfectly good NVIDIA card installs TrainAI, gets the CPU build, and
> trains 20–100× slower than their hardware allows. Nothing raises.
> `torch.cuda.is_available()` returns False — exactly as it would with no GPU at all.

So `setup` reads the GPU vendor from the system (driver tools on PATH, Linux PCI
sysfs, or the Windows display-adapter registry), reads the CUDA version your
*driver* reports, and prints the install command that matches. It prints; it does
not install. `--install` runs it after showing the command, refusing to touch a
system Python without `--allow-global`, and asking for confirmation — a 2.5 GB wheel
going into a shared environment is not a decision a diagnostic should make alone.

The wheel channel comes from your driver, not from a guess:

| Your driver reports | Channel suggested |
|---|---|
| CUDA 11.8 – 12.0 | `cu118` |
| CUDA 12.1 – 12.3 | `cu121` |
| CUDA 12.4 – 12.5 | `cu124` |
| CUDA 12.6 – 12.7 | `cu126` |
| CUDA 12.8 or newer | `cu128` |
| older than 11.8 | no command — update the driver |

That table was written in August 2026 and PyTorch's channels move. Every message
that uses it also prints the [pytorch.org selector][selector], which is the
authoritative source. A stale table that produces a 404 is worse than no
suggestion, so the table is presented as a shortcut and never as a promise.

## What is supported, and how well

| Backend | Detected | Trains | Verified here |
|---|---|---|---|
| NVIDIA / CUDA | yes | yes | **yes** — RTX 2050, 4 GiB, Windows 11 |
| AMD / ROCm (Linux) | yes | should | no hardware to test on |
| Intel / XPU | yes | should | no hardware to test on |
| Apple / MPS | yes | yes, in fp32 | no |
| CPU | yes | yes, slowly | **yes** — CI, Linux and Windows, Python 3.10–3.13 |
| AMD on Windows | yes | no | PyTorch publishes no ROCm build for Windows |
| Multiple GPUs | yes | first one only | multi-GPU is not implemented |

"should" means the code path is generic and tested against synthetic profiles of
that hardware, and that nobody has run it on the real thing. If you do, a report of
what happened is genuinely valuable — including a report that it worked.

## How hardware nobody here owns gets tested

Every hardware-dependent decision is a pure function of stated facts rather than
something that reads global device state. `precision_for(requested, backend=...,
supports_bf16=..., capability=...)` takes the facts and returns the dtype; a thin
wrapper gathers those facts from the live machine. That split is what makes
`tests/test_hardware_portability.py` possible: it feeds the function the real
characteristics of thirteen machines — a 24 GiB 4090, a bf16-less GTX 1080 Ti, a
Tesla V100, an RX 7900 XTX, an RX 6900 XT, an MI210, an Arc A770, an Apple M2, a
dual-4090 box, a CPU-only machine, and a machine with a good card and the wrong
wheel — and asserts what each one should be told.

It proves the logic. It does not prove the kernels run, and no amount of synthetic
testing could.

### The bug that made this necessary

An earlier version decided bf16 support with `compute_capability >= (8, 0)`. That is
correct on NVIDIA. On ROCm, PyTorch reports the **gfx architecture** through the same
`major`/`minor` fields:

| Device | Reports | The `(8, 0)` rule concludes | Actually has bf16 |
|---|---|---|---|
| gfx906 — Vega 20 | `(9, 0)` | yes | **no** |
| gfx908 — MI100 | `(9, 0)` | yes | yes |
| gfx90a — MI210 | `(9, 0)` | yes | yes |
| gfx1030 — RX 6900 XT | `(10, 3)` | yes | **no** |
| gfx1100 — RX 7900 XTX | `(11, 0)` | yes | yes |

An RX 6900 XT owner would have been told they had bf16 and TrainAI would have
autocast into a format their card lacks. Nothing on an NVIDIA laptop would ever
have found it.

The fix is not a better table. There is no architecture number that reliably implies
bf16 on AMD, so TrainAI asks the runtime — `torch.cuda.is_bf16_supported()`, which
works on both CUDA and ROCm builds — and if the runtime cannot answer, concludes
*no* and uses fp32. Slower and correct beats faster and wrong.

The same reasoning applies to TF32, which is an NVIDIA tensor-core format with no
AMD or Intel equivalent. `doctor` omits the row entirely on those backends rather
than printing "TF32: no", which would imply the hardware fell short of something
instead of that the concept does not apply.

## Windows and silent VRAM spillover

Under the WDDM driver model, Windows lets a GPU oversubscribe VRAM and pages the
excess to system RAM. Exceeding VRAM therefore does **not** raise
`torch.cuda.OutOfMemoryError` — throughput collapses instead. Measured on an
RTX 2050: a configuration peaking at 6.55 GiB on a 4 GiB card "succeeded" at 1,304
tokens/s against 61,595 for a configuration that fit.

TrainAI sets a flag for this on any Windows GPU, which makes the planner reject
configurations on *measured* memory and throughput rather than on the absence of an
exception. Concretely, that is two modules:

- `src/trainai/hardware/benchmark.py` builds the real model, runs real optimizer
  steps, and reads `allocated_bytes.all.peak`, `reserved_bytes.all.peak` and
  `num_alloc_retries` from the allocator. It reports whether memory was measured at
  all, so a backend with no counters yields "not measured" rather than a zero that
  would read as "used nothing".
- `src/trainai/hardware/planner.py` compares that measured peak against 85% of *free*
  VRAM, and separately watches for a throughput collapse — a candidate achieving a
  small fraction of the **FLOP rate** that a smaller one already managed on this same
  machine. Compared as FLOPs per second rather than tokens per second, because a larger
  model is legitimately slower per token. That second check is the backstop for the
  case where the allocator's own numbers look fine because the paging happened below
  it. The threshold is deliberately loose: a false positive costs a smaller model than
  you could have had, a false negative costs a run that appears to work and takes
  twenty times as long.

The measurement was taken on NVIDIA; WDDM is a driver model rather than a vendor
feature, so the flag is set for AMD on Windows too. It only ever makes TrainAI more
careful, so applying it where it has not been measured errs in the safe direction.

The same code runs on Linux, where CUDA does raise on an over-allocation and a
try/except would have been enough for the memory decision alone. It is not
platform-dependent anyway, because the same run also produces the throughput the time
estimate is built from — and because a decision procedure that only engages on Windows
is the one nobody tests, on the one platform where it is the only thing that works.

## No GPU at all

Everything works. The CPU suite — more than 1,800 tests, everything not marked
`gpu` — passes
locally, and `.github/workflows/ci.yml` is configured to run exactly that against a
**CPU-only** torch build on Linux and Windows across Python 3.10–3.13, which is the
arrangement that would enforce "works without a GPU" on every commit rather than
leaving it hoped for. Be aware that this has not happened yet: the repository has no
remote, so no CI run has ever executed, and the local runs were on a machine that
does have a CUDA device (the GPU tests were deselected, not made impossible). Data
preparation is CPU work anyway and is not meaningfully slower.

Training is roughly 20–100× slower. For a first run on a megabyte of text that is
survivable — minutes, not hours. For anything larger it is not, and TrainAI says so
rather than letting you find out.

## If TrainAI gets your hardware wrong

That is a bug, and a useful one. `trainai doctor --json` and `trainai setup --json`
print everything the detection concluded; those two outputs plus what the hardware
actually is make a complete report. Detection failures are the class of bug this
project is least able to find on its own.

[selector]: https://pytorch.org/get-started/locally/
