# Security policy

## Supported versions

TrainAI is `0.x`. Only the latest release gets fixes; there are no maintained
branches behind it. Until `1.0`, "upgrade" is the whole patch strategy.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting:
**[Security → Report a vulnerability](https://github.com/shubhampardule/trainAi/security/advisories/new)**.
It opens a private thread visible only to the maintainers, which is the right
place for anything that should not be a public issue yet.

Please do not open a public issue for a suspected vulnerability first.

Include what you would want if you were fixing it: the version
(`trainai --version`), the platform, the exact command, and the smallest input
that shows the problem. If the input is a corpus file, a few kilobytes that
reproduces it is far more useful than a description of one.

This is a small project. Expect a first reply in a few days rather than a few
hours, and no promise of a fix window. If a report is valid you will be credited
in the advisory and the changelog unless you ask otherwise.

## What is in scope

The interesting attack surface is **the corpus**, because that is the one input
TrainAI is designed to accept from wherever the user found it. `trainai data
prepare` parses plain text, JSON Lines, JSON, CSV, TSV, `.docx`, SQLite
databases, four compression codecs and zip/tar archives — every one of those is a
parser reading bytes it did not write. In scope:

- A crafted corpus that crashes TrainAI with a traceback instead of a clear
  error, hangs, or grows memory without bound.
- Anything read out of an archive escaping the paths TrainAI reports — although
  archive members are streamed and **never extracted to disk**, so the classic
  zip-slip has nothing to write through.
- A corpus that makes the report lie: files silently dropped, counts that do not
  add up, or documents ingested that the summary does not account for. This one
  is a correctness bug and a security bug at once, because a corpus quietly
  changing is invisible in the trained model.
- Anything in the installed package making a network request. TrainAI does not
  reach the network at all except in `trainai setup --install`, which runs `pip`
  because that is what the user asked it to do, and in `examples/`, which is
  sample code rather than the package.
- A way to make `trainai setup --install` install something other than the
  PyTorch build it names on screen before asking.
- Credentials, absolute paths from another user's machine, or anything else
  ending up somewhere a user would not expect: the manifest, a run directory, an
  export, or `--json` output.

## What is out of scope

- **A `weights_only=True` bypass in PyTorch itself.** Checkpoints are read with
  torch's restricted unpickler, so a crafted checkpoint reaching arbitrary code
  through it is a torch vulnerability — report it there, and TrainAI ships the fix
  by raising its floor. Getting TrainAI to load a checkpoint **without** that
  restriction, or to load one the user did not point it at, *is* in scope; there is
  one `torch.load` in the package and a test fails if its argument changes.
- **Importing a Hugging Face model or tokenizer someone else published.** The trust
  decision is the user's, at the moment they choose the source, and what arrives
  can be an arbitrary pickle that TrainAI is not the one reading.
- Making your own machine work hard. `trainai train` fills VRAM and pins the
  disk; that is the job.
- Anything requiring an attacker who can already write to the user's run
  directory or replace files in the installed package.

## Hardening that is deliberately not present

Listed so nobody has to read the source to find out:

- Checkpoints are a `torch.save` zip rather than safetensors, because they hold
  optimizer moments, RNG state and configuration and not only tensors. They are
  loaded with `weights_only=True` and a version-1 file is refused rather than
  migrated, so nothing in the load path imports what the file names — but the
  restriction is torch's unpickler doing its job, which is a dependency's guarantee
  and not a property of this code. `trainai export` writes safetensors, and that is
  the path a model *shared with someone else* should travel; the reasoning is in
  [docs/checkpoint-format.md](docs/checkpoint-format.md).
- There is no per-member compression-ratio ceiling on archives. Real,
  legitimately repetitive JSON reaches 269:1 under gzip, so any threshold low
  enough to catch a bomb also refuses real corpora. What is bounded instead is
  the member count and the declared unpacked total, both read from the table of
  contents without decompressing anything, and archives nested inside archives
  are reported rather than opened — which is where the petabyte-class bombs live.
  The reasoning is in `src/trainai/data/ingest.py` beside the constants.
- Nothing is sandboxed. TrainAI runs with your privileges, in your environment,
  on your files.
