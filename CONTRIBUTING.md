# Contributing to TrainAI

Thanks for considering a contribution. This document covers the setup, the house
rules, and the few conventions that are load-bearing rather than cosmetic.

## Setup

```bash
git clone https://github.com/shubhampardule/trainAi
cd trainAi
python -m venv .venv
```

If you intend to open a pull request, fork first and clone your fork instead — the
URL above is upstream, and you will not be able to push to it.

Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate`
elsewhere), then:

```bash
pip install -e ".[dev]"
```

That installs the CPU-or-GPU torch build pip picks for your platform. If you want
a specific CUDA build, install torch first from
[pytorch.org](https://pytorch.org/get-started/locally/) and then run the editable
install — pip will leave the existing torch alone.

Confirm it worked:

```bash
trainai doctor
```

## Tests

```bash
pytest
```

The default run must pass on a machine with **no GPU, no network, in a few
seconds**. Tests that need more are marked and opt-in:

| Marker | Meaning | How to run |
| --- | --- | --- |
| `gpu` | Needs a CUDA device. Auto-skipped when none is present. | `pytest -m gpu` |
| `slow` | Takes more than a few seconds. | `pytest -m slow` |
| `network` | Downloads something. | `pytest -m network` |

CI runs `pytest -m "not gpu"` on Ubuntu and Windows across Python 3.10–3.13. Nobody's
CI has a GPU, so **the GPU suite is your responsibility to run locally**
if you touch anything under `model/`, `train/`, or `hardware/benchmark.py`. Say in
the PR whether you ran it and on what card.

### Coverage

CI enforces a floor, per module as well as overall:

```bash
pytest -m "not gpu" --cov=trainai --cov-report=xml --cov-report=term-missing
python tools/coverage_floor.py coverage.xml
```

Every module must reach **60%** of its lines and the project **90%**. The per-module
number is the one that matters: this project once sat at 92% overall with
`src/trainai/cli/setup.py` — the command that downloads 2.5 GB and replaces PyTorch in
your environment — at exactly **0%**, because 66 statements cannot move a 6,400-statement
total. A global floor cannot see that; a per-module floor can.

The 60% is deliberately low, and it is a floor rather than a target. Code that reads the
machine is full of platform branches — `hardware/probe.py` measures 70% on Linux and 80%
on Windows, and neither is a gap, since neither job can execute the other's code. The
floor has to clear the lowest real number on every platform in the matrix. Read
`term-missing` for the number you should actually care about.

A module that genuinely cannot be covered goes in `EXEMPT` in
[`tools/coverage_floor.py`](tools/coverage_floor.py) **with the reason**. There is one
entry today. An exemption naming a module that no longer exists fails the gate, so a
rename cannot leave a permanent hole behind.

## Lint and format

```bash
ruff format .
ruff check .
```

Both must be clean before a PR. There is no separate style debate — ruff decides.

## The rules that matter

These are not stylistic preferences. A PR that breaks one of them will be asked
to change regardless of how good the rest is.

**1. Do not claim something works unless you ran it.**
This applies to code, docstrings, README text, and PR descriptions. If you added
a flag you could not test on your hardware, say so in the docs and in the PR. An
untested path labelled untested is fine. An untested path presented as working is
not.

**2. Measure; do not estimate.**
The project's central finding is that on Windows/WDDM, exceeding VRAM does *not*
raise `torch.cuda.OutOfMemoryError` — the driver pages to system RAM and
throughput collapses by 5–45× while appearing to succeed. Any code that decides
whether a configuration fits must read real allocator counters
(`torch.cuda.memory_stats()`) after real training steps. Analytical VRAM formulas
are welcome as a *starting point* for the search, never as the verdict.

**3. Every error carries an actionable hint.**
Raise a subclass of `TrainAIError` (see [`errors.py`](src/trainai/errors.py)) with
a `hint=` that tells the user what to *do*:

```python
raise DatasetEmptyError(
    f"{path} contains no usable text after filtering.",
    hint="Lower --min-doc-chars, or point the corpus argument at a directory of .txt files.",
)
```

A `TrainAIError` raised without a hint is treated as a bug. Bare `ValueError`s
that reach the user are also bugs — those print a full traceback, which is correct
for *our* mistakes and wrong for the user's.

**4. Do not import torch at package import time.**
`trainai --help` must stay instant, so `import trainai` must not pull in torch
(several seconds). Import torch inside the function that needs it. There is a test
for this; don't work around it.

**5. Binary units, always.**
GiB/MiB, never GB/MB. Use the helpers in
[`console.py`](src/trainai/console.py) (`fmt_bytes`, `fmt_duration`,
`fmt_params`) rather than formatting inline. Mixing unit systems in a tool whose
whole job is memory accounting is a correctness bug, not a nitpick.

**6. Plan against free VRAM, not total.**
`torch.cuda.mem_get_info()` free bytes, times a safety fraction. On a 4 GiB
laptop card running a desktop session, roughly 0.8 GiB is already gone before you
allocate anything.

**7. No new required dependency without a reason in the PR.**
Optional extras are the normal answer. The install has to stay a plain
`pip install` with no Node, no compiler, and no system packages.

**8. ASCII fallbacks for non-ASCII output.**
Windows consoles still default to cp1252. Route decorative glyphs through the
`BULLET` / `DASH` / `ELLIPSIS` / `SPINNER` constants in `console.py`. This is not
theoretical: a Braille spinner character crashed `data prepare` mid-run on a
cp1252 console, after the tokenizer had already started training.

**9. Source files are pure ASCII.**
Write non-ASCII data as `chr(0x4F60)` or an escape sequence, never as a literal
character — `console.py` is the single exempt file, because those glyphs are its
purpose. A literal that an editor or terminal re-encodes is a change nobody sees
in review, and in test data it silently alters the very input being asserted
about. There is a test for this.

## Commits and pull requests

Keep commits focused; a subject line under ~72 characters explaining *why*, not
just *what*. In the PR, include:

- what changed and why
- what you ran (`pytest`, `pytest -m gpu`, real training run?) and on what hardware
- anything you could not verify

Small, reviewable PRs get merged. A 2,000-line PR touching six subsystems will
sit.

## Cutting a release

A tag is the one artifact here that cannot be quietly corrected: it is what
`pip install git+https://...@v0.1.0` resolves, so moving one somebody has already
fetched breaks their checkout instead of fixing it. Everything checkable is
therefore checked *before* the tag exists.

1. Rename `## [Unreleased]` in `CHANGELOG.md` to `## [x.y.z] - YYYY-MM-DD` and
   open a fresh empty `## [Unreleased]` above it.
2. Set `version` in `pyproject.toml` to the same `x.y.z`.
3. Run the check, which compares all three and refuses if they disagree:

   ```bash
   python tools/release_check.py --tag vx.y.z
   ```

   It needs Python 3.11+ for `tomllib`; the version in `pyproject.toml` is worth
   parsing properly rather than approximating with a regex. On 3.10 it refuses and
   names an interpreter that works, and the tests that exercise it skip — so a green
   suite on 3.10 has not checked the release path. Cut a release from 3.11 or newer.
4. Tag and push. `.github/workflows/release.yml` runs the same check on the tag,
   builds the sdist and wheel, runs `twine check --strict`, installs the **wheel**
   and runs the suite from the **unpacked sdist** — the two failures an editable
   install cannot see are a module missing from the wheel and a file the tests read
   missing from the sdist — and then creates a **draft** release with the artifacts
   attached and the changelog section as its notes.
5. Read the draft and press publish. That is the last reversible moment.

Nothing publishes to PyPI. No TrainAI artifact has ever been uploaded to an index
and no token is configured, so that step goes in with the commit that actually
claims the name rather than sitting there having never run.

## Where to start

Good first contributions:

- Ingest support for another plain-text-ish format (see `data/ingest.py`)
- Dataset validation checks with clear hints
- Tests for edge cases in the tokenizer or loader
- Running TrainAI on hardware nobody has tested — a report of what broke on your
  AMD card, your Mac, or your 24 GiB desktop GPU is genuinely valuable
- Documentation that fixes something that confused you

Please open an issue before starting anything large (a new architecture, the web
UI, multi-GPU) so we can agree on the shape first.

## Scope

TrainAI v0.1 is deliberately narrow: **training small language models from
scratch on your own text, on consumer hardware.** `trainai finetune` extends that
to continuing one of *your own* checkpoints on a second corpus, and no further:
LoRA and adapters of any kind, importing pretrained weights from elsewhere,
multi-GPU, quantization, and non-text modalities are out of scope for now — not
because they're bad ideas, but because the from-scratch path has to be genuinely
reliable first. See the roadmap in the [README](README.md).

## Code of conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Contributions are licensed under Apache-2.0, matching the project.
