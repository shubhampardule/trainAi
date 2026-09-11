# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is `0.x`, the CLI surface and the on-disk run-directory layout
may change between releases. Breaking changes will be listed here explicitly.

## [Unreleased]

Nothing yet.

## [0.1.0] - 2026-09-10

The first release, and the first tag: everything below is what M0 through M4
amounted to, and there is no earlier version for it to be a diff against. See the
roadmap in the [README](README.md) for which milestones are done and which are not.

### Added

- Project skeleton: `src/` layout, [hatchling](https://hatch.pypa.io) build
  backend, Apache-2.0 license, ruff config, pytest config with opt-in `gpu` /
  `slow` / `network` markers.
- `trainai.errors` — the error taxonomy. Every `TrainAIError` carries an
  actionable `hint`, JSON-serializable `details`, and maps to a stable process
  exit code.
- `trainai.console` — shared Rich consoles, binary-unit formatting helpers
  (`fmt_bytes`, `fmt_duration`, `fmt_params`, `fmt_count`, `fmt_int`), error
  rendering without tracebacks for user-facing failures, and ASCII fallbacks for
  consoles that are not UTF-8 (Windows still defaults to cp1252).
- `trainai.hardware.probe` — hardware detection: device type, **free** as well as
  total VRAM, compute capability, bf16/TF32 support, CPU model name, RAM, free
  disk, whether `torch.compile` is usable, and whether the platform can silently
  spill VRAM to system RAM. Probe failures are reported as warnings rather than
  raised.
- `trainai doctor` — prints the probed hardware, the training VRAM budget, and
  anything worth flagging. `--json` emits the same data machine-readably.
- Contributor documentation: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`.
- GitHub Actions CI: ruff check and format verification, plus the non-GPU test
  suite on Ubuntu and Windows across Python 3.10–3.13.
- A coverage floor in CI (`tools/coverage_floor.py`), enforced **per module** as
  well as overall. The per-module floor is the point: the project once measured 92%
  overall with `trainai setup` — the command that downloads 2.5 GB and replaces
  PyTorch in your environment — at 0%, which no global `--cov-fail-under` can
  detect. Exemptions live in the script with a written reason, and an exemption
  naming a module that no longer exists fails the gate.
- One net over every CLI option whose valid values are a closed set
  (`tests/test_cli_choices.py`), because `--precision fp64` was not a
  `--precision`-shaped bug: Typer validates an `Enum` and a `bool` and nothing else, so
  every option declared `TEXT` whose values live in its help text is the same bug
  waiting. Of roughly 120 options, exactly one is a `Choice` the parser can check;
  61 arrive as unchecked strings. Three claims are enforced over all of them. Every
  free-text option is classified — a closed set, keyed to the constant that enforces
  it, or free-form with a written reason — so a new option lands in neither list and
  fails the gate, which asks "is this a closed set?" at the only moment anyone knows.
  Every closed set's values appear in the `--help` text that is the only place most
  people will read them; that check found `plan --max-preset` describing itself as
  "Do not measure anything larger than this preset" and naming none of the four, now
  fixed. And every command refuses a value outside its set end to end, by exit code
  and by a message that lists the real values — the claim being not that a validator
  exists but that the value a user types reaches it, which is what `--precision` had
  looked like all along. An entry naming a flag no command accepts fails too, so the
  lists cannot rot as flags are renamed.
- `SECURITY.md` — the reporting route (GitHub's private advisory flow, not an email
  address nobody reads) and, more usefully, an honest scope. The attack surface is the
  corpus: `data prepare` parses plain text, JSON Lines, JSON, CSV, TSV, `.docx`, SQLite,
  four compression codecs and zip/tar archives, and every one of those is a parser reading
  bytes it did not write. A crafted corpus that crashes, hangs or grows without bound is in
  scope; so is one that makes the *report* lie, because a corpus quietly changing is
  invisible in the trained model. Out of scope, and said so rather than left to be
  discovered: loading a `.pt` checkpoint from a stranger is running their pickle, which is
  why `trainai export` writes safetensors. The hardening that is deliberately absent is
  listed too — pickled checkpoints, no per-member compression-ratio ceiling (real
  repetitive JSON reaches 269:1 under gzip, so any threshold that catches a bomb also
  refuses real corpora; the member count and declared total are bounded instead), and no
  sandbox — so nobody has to read the source to find out what is not defended.
- GitHub issue forms and a pull-request template. The forms ask for the three things that
  decide whether a report is actionable — the version, `trainai doctor`, and the exact
  command — before it is filed rather than in a reply, and blank issues are off so they
  cannot be skipped. One of the three is a hardware report, which is the single most
  valuable thing an outside contributor can send: TrainAI has only ever *run* on NVIDIA
  CUDA and CPU, on one 4 GiB card. The PR template is the three questions
  `CONTRIBUTING.md` already asks, in the file that actually gets filled in.
- A net over the links out of `.github/` (`tests/test_conventions.py`). Issue forms are
  YAML, so a relative path in one resolves against nothing and every link is an absolute
  URL with a hard-coded repository name, branch and path — the one place in this
  repository where renaming a file breaks a link no Markdown tooling will ever look at,
  and a bad `#fragment` is not an error to a browser. It caught the first draft of these
  templates pointing at `README.md#roadmap`, on a README whose heading says "Status". The
  repository name is read from `pyproject.toml`, and a link shape the check cannot resolve
  is reported rather than skipped: the pattern's first draft read the advisory URL as the
  bare repository and cheerfully validated it against the README.
- A release path that stops at a draft: `.github/workflows/release.yml` plus
  `tools/release_check.py`, which compares the tag, `pyproject.toml`'s version and the
  `CHANGELOG.md` section and refuses if any two disagree. A tag is the one artifact here
  that cannot be quietly corrected — it is what `pip install git+…@v0.1.0` resolves, so
  moving one somebody has already fetched breaks their checkout instead of fixing it. So
  the same command is meant to be run *before* the tag exists, and the workflow runs it
  again on the tag push, which is what makes it a gate rather than a note in
  `CONTRIBUTING.md`. Its three refusals are the three cheap mistakes: a tag naming a
  version the package does not build, a tag without the `v` that the workflow's
  `tags: ["v*"]` filter ignores entirely — that one produces no release *at all* rather
  than a failed one, so it looks exactly like success until somebody goes looking for the
  artifacts — and a release whose own notes still sit under `## [Unreleased]`.

  What the release workflow does that CI cannot is test the artifact instead of the
  checkout. CI installs with `pip install -e .`, so it never exercises the files anybody
  would actually download; the release job installs the **wheel** as a wheel and runs the
  suite from the **unpacked sdist**, which are the only conditions under which a module
  missing from the wheel, or a file the tests read missing from the sdist, is visible at
  all. `pyproject.toml` recorded that pair as verified once by hand, and a check performed
  once is a claim with a date on it. Running it for real immediately found one: the
  `.github/` link net above reads a directory the sdist deliberately does not ship, and
  `rglob` on a missing directory returns nothing rather than raising, so the check had
  quietly become vacuous there — it now skips with a stated reason, and a separate floor
  makes zero templates a failure in a checkout.

  The release notes are the changelog section itself rather than a second copy typed by
  hand, and they are written only after every check has passed, so a refusal cannot leave
  notes behind for a later step to publish. Nothing here uploads to an index: no TrainAI
  artifact has ever been published, no token or trusted publisher is configured, and a
  publish step that has never once succeeded is worse than no step — it turns the first
  real release into a debugging session and makes the repository look as though
  `pip install trainai` works. The draft is where it stops, because reading the rendered
  notes and pressing publish is the last reversible moment.

#### Data pipeline (M1)

- `trainai.data.ingest` — discovers and decodes `.txt` / `.text` / `.md` /
  `.jsonl` / `.ndjson`, each also accepted with a `.gz` suffix. Discovery is
  sorted, so document order (and therefore every checksum downstream) does not
  depend on the filesystem. Decoding is incremental, so a failure reports the
  byte offset it happened at; byte-order marks are honoured over the declared
  encoding; over-long files are split at paragraph boundaries; and every document
  that is dropped is counted rather than silently discarded.
- `trainai.data.analyze` — measures the corpus: document count and length
  distribution, character mix, duplicate rate, and a writing-system breakdown over
  a bounded, deterministic sample. Figures that are not exact say so in the field
  name (`rough_token_estimate`) and in the output.
- `trainai.data.validate` — turns those measurements into a verdict with stable
  issue codes, split into errors (too small, binary masquerading as text), warnings
  (vocabulary too large for the corpus, heavy duplication) and notes. Every issue
  carries a hint naming the flag to change.
- `trainai.data.tokenizer` — byte-level BPE via the Rust `tokenizers` library, with
  all 256 byte values seeded into the alphabet so there is no `<unk>` and
  `decode(encode(x)) == x` holds for any string. Trained tokenizers self-check
  against a probe set before being returned, and expose a `fingerprint()` that
  pairs a dataset with the tokenizer that produced it.
- `trainai.data.binarize` — writes little-endian `uint16`/`uint32` memmap shards
  plus a checksummed `manifest.json`. The train/validation split is a keyed hash of
  each document's text, so it is platform-independent, order-independent, and
  cannot leak an exact duplicate across the split. `verify_dataset` checks sizes
  cheaply or re-hashes every byte on request.
- `trainai.data.loader` — deterministic resumable sampling from the shards:
  `batch(step)` is a pure function of `(seed, step)`, so a resumed run replays the
  same data.
- `trainai data prepare` — one command from a folder of text to shards, reading the
  corpus twice (measure and train the tokenizer, then encode) with live progress.
  Refuses to overwrite an existing dataset without `--force`, refuses to write
  inside the corpus directory, and re-checks that both passes saw the same corpus
  before writing the manifest. `--json` emits the manifest and nothing else.
  `--tokenizer PATH` reuses an existing `tokenizer.json` — a file, or the dataset
  directory holding one — instead of training a fresh one. Without it, every
  dataset gets its own vocabulary, and a token id means something different in
  each; a second dataset prepared that way can never be fine-tuned from a
  checkpoint trained on the first, because the ids are row indices into an
  embedding matrix. `--vocab-size` alongside it is refused rather than ignored:
  the loaded file's vocabulary is a fact about the file, and a flag silently
  dropped is how a dataset ends up not being the one that was asked for.
- `trainai data inspect` — reports on either a raw corpus or a prepared dataset,
  writing nothing. `--verify` re-hashes every shard, `--layout` explains the files,
  `--sample N` shows the start of the first document. Exits 3 when a corpus cannot
  be trained on, so it is usable as a check in a script.
- `docs/dataset-format.md` — the on-disk format, specified rather than implied.
- `examples/get_tinyshakespeare.py` — fetches ~1.1 MB of text to try it on.

#### Model and trainer (M2)

- `trainai.model` — a decoder-only transformer: RMSNorm, rotary position
  embeddings, SwiGLU, grouped-query attention, no biases, optional weight tying.
  Attention goes through `scaled_dot_product_attention` so PyTorch can dispatch a
  fused kernel; the naive implementation materialises a `batch x heads x seq x seq`
  matrix, which at batch 8 and 512 tokens is 64 MiB per layer in fp32.
  `ModelConfig.parameter_count` is exact arithmetic over the shapes the model
  allocates, checked against `sum(p.numel())` for every preset and every
  combination of tying and grouping.
- `trainai.train.schedule` — learning-rate schedules as pure functions of the step
  number rather than stateful `lr_scheduler` objects, so a resumed run cannot land
  on a shifted schedule.
- `trainai.train.checkpoint` — checkpoints written atomically, verified on load
  against the tokenizer fingerprint and dataset content hash, and retained with the
  best and newest always protected. The tied output projection is stored once.
- `trainai.train.budget` — the epochs and tokens-per-parameter arithmetic that
  predicts a badly proportioned run before it starts.
- `trainai.train.metrics` — a JSONL event log flushed after every record, plus a
  live progress display sized to fit 80 columns.
- `trainai.train.loop` — the trainer: AdamW with decoupled parameter groups,
  gradient accumulation that averages rather than sums, gradient clipping, autocast
  with a precision resolved from what the GPU reports, sequential validation, and
  exact resume.
- `trainai train` — one command from a prepared dataset to a trained model.
  `--dry-run` reports the shape, the parameter count and the data budget in two
  seconds without touching the GPU. Refuses to train into an existing run directory
  unless `--resume` or `--force` says otherwise.
- `docs/checkpoint-format.md` — the checkpoint format and its compatibility promise.
- `trainai finetune` — continues an existing checkpoint on a different dataset. It is
  not `--resume`: the weights are inherited, and the optimizer moments, the RNG state
  and the step counter deliberately are not, because they belong to the schedule of the
  run that produced them and a fine-tune is a new schedule over different text.
  It has **no model flags at all** — `--layers`, `--width` and the rest do not exist on
  it, because `Checkpoint.apply_to` requires an exact shape match, so a flag that parsed
  could only ever produce a load error phrased as a mismatch the user did not cause.
  A dataset prepared with a different tokenizer is **refused** (exit 5), before the plan
  is printed and before the model is built, and so is `--dry-run` on the same pair: a
  token id is a row index into the embedding matrix, so the same id read through another
  tokenizer selects a vector trained for different text, and nothing about the resulting
  loss curve looks wrong. Prepare the second dataset with
  `data prepare --tokenizer <the base run's dataset>`. The run record carries
  `finetuned_from` and a `parent` block naming the base checkpoint, its step and its
  dataset identity, so a tuned run directory can say what it came from.
  The default `--lr` is `3e-5`, a tenth of the pretraining default, and it is measured
  rather than assumed — see [docs/finetuning.md](docs/finetuning.md) for the sweep, the
  trade it shows, and why the conservative end was taken. The data-budget report changes
  wording for a fine-tune: `20 tokens per parameter` is a from-scratch reference and its
  remedy ("use a smaller model") is not available when the shape is fixed by the
  checkpoint.
- `docs/finetuning.md` — what fine-tuning inherits, what it refuses, and the learning-rate
  measurement behind the default.
- `trainai.data.chat` and `--jsonl-messages-field NAME` — a corpus record may hold a
  **typed conversation**, a list of `{"role", "content"}` objects, instead of a flat
  string. It is rendered with one fixed, versioned template — the same layout the
  `chat.jsonl` this project ships already uses, verified byte-identical on 20,000 of
  20,000 records — and the character spans covering the assistant replies are computed
  and counted. `data prepare` and `data inspect` report how much of the corpus is
  replies; the roles are `system`, `user` and `assistant`, and an unknown role is an
  error listing those three rather than a guess. Works for `.jsonl` and `.json`, and
  cannot be combined with `--jsonl-field`, which means the opposite thing.

  There is one template, and it is versioned, for the same reason the tokenizer
  fingerprint is checked: a model trained with one chat layout and prompted with
  another degrades with no error at all. The input is typed rather than sniffed for the
  same reason a table needs its column named — finding the replies by searching
  flattened text for `"Assistant:"` mis-masks any reply containing that string, and a
  model discussing itself produces many.

  **The loss mask is applied by `trainai train`.** See the entry below for the flag
  and what the loss does with it; the report says on its own `Loss mask` line whether
  the mask was written, so the reply share is never left to be misread.

- The loss mask on disk. For a corpus read with `--jsonl-messages-field`,
  `data prepare` now writes `train_00000.mask.bin` beside `train_00000.bin` -- one
  `uint8` per token, `1` for a training target and `0` for context. The character
  spans covering the assistant replies become token ranges at binarize time, using
  the tokenizer's own per-token offsets rather than a second pass over the text.
  `--no-loss-mask` opts out; `--loss-mask` on a corpus with no typed conversations is
  a usage error rather than a mask of all ones.

  `mask_name`, `mask_bytes` and `mask_sha256` live in the same `shards[]` entry as
  the tokens they describe, so a mask paired with the wrong shard is unrepresentable
  rather than merely checked -- a misaligned mask has no symptom at all: it trains, it
  converges, it is slightly wrong forever. They are present or absent as a group, a
  dataset with a mask on only some shards is refused, and `data inspect --verify`
  re-hashes every mask alongside every shard.

  A token is selected by **intersecting** a span, not by being contained in one: the
  tokenizer reports a token beginning with a space as covering only its text, so
  containment either way would drop or add a token at every boundary. Tokens covering
  both sides of a span edge are counted (`straddling_tokens`) rather than assumed
  away; measured on this repo's 20,000-conversation corpus the count is **zero**, and
  that is by construction -- a span ends on a newline run, which the byte-level
  pre-tokenizer never merges with the word after it.

  A document with no spans is trained on in full: empty spans mean "no opinion", not
  "train on nothing", so a `.txt` file beside a `chat.jsonl` is unaffected. The
  end-of-text token is always a target, because a model never scored on the token that
  ends a document never learns to stop.

  A dataset prepared **without** a mask writes none of the new keys and none of the
  new totals, so its manifest -- and its `content_hash` -- are byte-identical to the
  previous release's. A masked dataset has a different hash from an unmasked one built
  from the same corpus, which is correct: the mask is part of what was prepared.

  `IngestOptions.to_dict()` gained `jsonl_messages_field`, so a dataset prepared with
  this release has a different `content_hash` from one prepared with the previous one,
  even from an identical corpus — the same one-off change the CSV column option caused.
  Nothing on disk is invalidated: a manifest's recorded hash is what is compared, and
  existing datasets keep theirs.

- The loss mask reaches the loss. `trainai train` and `trainai finetune` score only
  the tokens a masked dataset marks as targets, with no flag needed; `--loss-mask`
  requires a masked dataset rather than training on everything quietly, and
  `--no-loss-mask` scores every token deliberately. The choice is recorded in
  `TrainConfig`, so a resume continues the same measurement and `trainai eval` reports
  the same quantity the training curve did.

  The loss is a **weighted mean** over the selected positions, not `ignore_index=-100`.
  A window landing entirely inside a prompt selects nothing, and cross-entropy over
  nothing is `0/0` -- a NaN that reaches every parameter through the backward pass. A
  clamped denominator makes such a window contribute loss 0 and gradient exactly 0.

  Gradient accumulation weights each micro-batch by **its own scored count**, not by
  `1/--grad-accum`: micro-batches of a masked dataset score different numbers of
  tokens, so the fixed factor would make the step's gradient a mean of means. The whole
  step's batches are read first and each is scaled by its share. On an unmasked dataset
  this reduces to `1/--grad-accum` exactly, so no existing run changes.

  A step that scores nothing is written to `metrics.jsonl` as an `unscored` event with
  a null loss, and counted in the result -- not logged as 0.0, which would be a fake
  minimum in the curve, and not as a NaN, which the divergence check would read as a
  diverged run. `trainai eval` applies the mask when the run recorded it and reports
  the scored share, because a perplexity over 41% of the positions is a different
  measurement from one over all of them.

  `TrainConfig.loss_mask` is optional in a recorded config, so a checkpoint or
  `plan.json` written before this release resumes as "apply if the dataset has one".
  No `content_hash` changes: this is a training flag, not an ingest option.

- The dataset records the chat template it was rendered in. `manifest.json` gained a
  `chat` block -- the template `version`, the role `labels`, and `trained_roles` --
  present for a corpus read with `--jsonl-messages-field` and empty for prose.
  `data inspect` shows it as its own `Chat template` row, and `--json` carries it.

  The text in a chat dataset's shards is a **layout**, not just text: a model trained
  on `User: ...\n\nAssistant: ` and then handed a bare question continues the question
  instead of answering it, which reads as a bad model rather than as a format
  mismatch. Recording the layout is what lets anything downstream reproduce it, and
  what `trainai chat` reads to prompt such a run in it (below).

  Recorded off the *rendering*, not off the mask: `--no-loss-mask` removes the mask
  and leaves the role labels in the shards, so such a dataset still records its
  template. One rendered conversation among prose records it too -- the model saw the
  layout, so it can be prompted in it.

  `version` is the contract, so `labels` holds the bare role names and the colon, the
  space and the blank line between turns come from `trainai.data.chat` at that
  version. Recording the punctuation but not the separators would invite a reader to
  treat the summary as the format and still get the turn separator wrong.

  **No `content_hash` changes.** `chat` is excluded from the reproducible subset: the
  rendered text is already in the shards and every shard's sha256 is already in the
  hash, so including the description would only make a dataset prepared before this
  release hash differently from the same corpus prepared today. It is also the one
  manifest key that is **optional on read** -- absent means "no chat template", which
  is what every existing dataset meant by not having it -- and that exception is now
  written down in [docs/dataset-format.md](docs/dataset-format.md). A `chat` that is
  present but not an object is still refused by name.

- The checkpoint carries the chat template. Every checkpoint's `dataset` block gained a
  `chat` key holding what the dataset's manifest recorded, and
  `InferenceSession.chat_template` reads it back -- surfaced in the session report and
  in `--json` as `chat_template`.

  Copied in rather than resolved from the dataset path later, because **a run has to
  stay usable after its dataset is deleted**, which is the same reason the tokenizer is
  copied into the run directory, and datasets are the large thing people delete once
  training is done. A model trained on `User: ...` / `Assistant: ` with nothing on disk
  saying so is a model that answers badly for a reason nobody can look up.

  Recorded from the dataset, not from the run's `--loss-mask` setting: an unmasked run
  on a chat dataset still saw the role labels, so the layout it has to be prompted in is
  the same one.

  **No format-version bump and no resume refusal.** `_check_dataset` compares the
  tokenizer fingerprint and the content hash and nothing else, so a checkpoint written
  before this release resumes unchanged; `chat_template` reports `{}` for it, for a
  prose run, and for a `chat` block that is not an object, without distinguishing them.
  This is what `trainai chat` reads to choose how to prompt a run.

- `trainai.data.chat` gained the two things a generation harness needs from the
  template: `render_prompt(messages)`, which renders the turns so far and then the
  `Assistant:` label for the model to complete, and `TURN_BOUNDARIES`, the strings
  that mark a model having stopped replying and started writing somebody else's turn.
  Both are what `trainai chat` calls, so the layout has one definition (below).

  `render_prompt` is **derived** from `render_conversation`, not assembled again: it
  renders one throwaway reply and cuts the text at the span that marks it. A prompt
  built from its own copy of the label, the colon and the blank-line rule is a prompt
  that can drift from what the shards hold, and the drift is invisible -- the model
  answers, a little worse, for a reason no error names.

  **It stops at the colon, not after the space the shards put there**, and a text prefix
  is not enough to get that right. A byte-level BPE keeps a space with the word after it,
  so the shards hold `Assistant:` and then one `" Hey"` token and never a lone space
  token; a prompt ending in the space asks the model to continue from a token sequence
  that occurs nowhere in its training data, and what comes back is noise rather than a
  slightly worse reply. Measured on the 4,000-conversation corpus in `data/chat-typed`
  with the tokenizer trained on it: every prompt is a *text* prefix of the rendered
  conversation either way, but 4,000 of 4,000 are a *token* prefix when the prompt ends
  at the colon against 0 of 4,000 when it ends after the space. The model writes the
  space itself, as part of its first word.

  A conversation with no reply in it renders as a prompt but is still refused as a
  *corpus record*, which is the one place the two callers legitimately differ: a record
  whose every token is masked out of every step is worth refusing, and a prompt is
  exactly that record. A conversation that already ends with a reply is refused with
  the way to continue it instead.

  `TURN_BOUNDARIES` holds the single-newline form (`\nUser:`) rather than the blank
  line the corpus separates turns with, because a sampling model writes the label after
  one newline too; the shorter form matches both, leaving one trailing newline for the
  caller to strip. Also verified on the real corpus: cutting at the earliest boundary
  recovers exactly the first reply in 4,000 of 4,000 multi-turn conversations.

- `InferenceSession.complete`, `.stream` and `.stream_pieces` gained `stop`, a sequence
  of strings generation ends at. The match is cut out of the result, not left on the end
  of it. This is the other half of the pair above: the strings come from
  `trainai.data.chat`, next to the renderer that put the labels in the shards, because a
  sampler with its own copy of `"\nUser:"` is a second place the layout is written down.

  **Matched on the decoded text, not on token ids.** How `"\nUser:"` tokenizes depends
  on what precedes it, so a token-id comparison would have to enumerate every
  tokenization of it and would quietly miss the ones it did not think of. Verified
  against a real tokenizer on `runs/m4`: 8 stop strings each straddling one of that
  tokenizer's own token boundaries -- so none of them is a token -- all cut the
  continuation at exactly the character the string starts at, and three label-shaped
  strings the model genuinely writes (`\nKING RICHARD III:` and friends) cut at 110, 147
  and 196 characters, with all three passed together cutting at the earliest, 110.

  **A partial match is held back rather than emitted.** The longest suffix of the
  continuation that is the beginning of a stop string is not yielded until the next token
  says whether it completes one, because text already on somebody's terminal cannot be
  taken back. It is flushed if generation ends without completing the match -- a stream
  that drops it truncates the reply at whatever happened to resemble a label, which loses
  text the model wrote and is worse than the bug the hold-back fixes.

  **The tokens that spell a stop string are counted**, unlike the end-of-text token,
  which is a boundary rather than content. They were computed and they took time, so a
  reported tokens/s that leaves them out is a wrong number about the machine; it is only
  the text that is discarded. Generation stops on the token that completes the match and
  does not walk the model again. An empty stop string is refused before anything is
  generated: it matches at position zero, so honouring it would end every generation with
  nothing, and dropping it silently would hide the mistake that computed it.

- `trainai chat` prompts a chat-trained run in the template it was trained in. **The
  checkpoint decides**: a run whose dataset was rendered with `--jsonl-messages-field`
  records the template, so what you type becomes the `user` message, the model generates
  after `Assistant:`, and generation stops where it starts writing somebody else's turn.
  A run that records nothing is prompted with exactly what you type, as before. The mode
  is on the banner and in `--json` (`mode`, `reason`, `stop`, `chat_template`, and
  `model_prompt` beside `prompt`), because a default nobody can see is a default nobody
  can correct.

  **Nothing is sniffed.** Not the prompt -- a question mark is not consent to wrap it --
  and not the corpus. Two flags override the record, for the two things a checkpoint
  cannot know: `--raw`, because a chat model is still a base model worth probing, and
  `--chat`, for a corpus you flattened into `User:`/`Assistant:` text by hand, which
  trains a chat model and records nothing. `/chat` and `/raw` switch mid-session without
  reloading the model, and keep the sampling they were switched with.

  **A recorded template from another version is refused, not applied.** Only `version` is
  compared, which is the documented contract; prompting a model in a layout it was not
  trained in makes it answer worse *without failing*, so the refusal names both versions
  and offers `--chat` or `--raw` to say which you meant. Neither the stop strings nor the
  rendering are written down a second time here -- both come from `trainai.data.chat`,
  next to the renderer that put the labels in the shards.

  `/more` continues the reply rather than the turn after it: generation stops *at* the
  newline that began the next label, so in chat mode the trailing newlines come off before
  the text goes back to the model. Continuing from after that newline asks the model to
  write the label it was just cut at, which stops immediately and produces nothing. In raw
  mode every character is kept, because a trailing newline in prose is a paragraph break
  the model chose. A continuation is passed through unrendered -- it is already model
  text, and wrapping it in a fresh `User:` would ask the model to answer its own reply.

  The context check counts the prompt the model is given, labels included. Counting what
  was typed undercounts a chat prompt by whatever the template costs, and that gap is
  exactly the window where the warning is the only thing between the user and a silent
  truncation; `prompt_truncated_from` reports the rendered count for the same reason.

  No `content_hash` changes and no new checkpoint keys: this reads what the previous four
  releases in this section wrote.

- `trainai chat` keeps the conversation in chat mode, so a follow-up question is a
  follow-up. The corpus this template renders is conversations rather than question/answer
  pairs -- the typed corpus in `data/chat-typed` is 4,000 of 4,000
  `user`/`assistant`/`user`/`assistant` -- so a second question asked with the exchange
  before it in front of it is the layout the model was trained on, and one asked alone is
  the layout that is not. `/new` forgets it and says how much it forgot. `/chat` and
  `/raw` keep it across the switch: probing the model unwrapped and switching back is what
  the pair is for, and losing the conversation to it would make that a one-way door. Raw
  mode accumulates nothing, because prose has no turns.

  **A conversation too long for the context loses whole messages, oldest first, and says
  so.** The sampler truncates by token count, keeping the last `seq_len`, which for a
  conversation cuts wherever the count lands: mid-word, inside a message, leaving a prompt
  whose first label is half a label -- a layout no corpus contains, which still produces a
  reply, so nothing about it looks wrong. Messages come off in pairs where they have to,
  since a corpus conversation begins with a question. The message just typed is never
  dropped; if it alone does not fit, it is truncated and reported as such. Nothing is
  reserved for the answer: a prompt that fills the context leaves the sampler sliding its
  window, which is better than the cut this replaces, and what to reserve is a policy that
  needs measuring rather than guessing.

- `trainai.data.chat` gained `reply_content(generated, *, continuing=False)` -- the
  inverse of rendering, and the reason it is in that module rather than in the harness that
  needs it. Two characters of what a sampler returns belong to the layout: the gap after
  the label, which the model writes itself because the prompt stops at the colon, and the
  separator before the next label, which is where generation stops. Store either and
  rendering the conversation again writes a second one, so the next prompt holds
  `Assistant:  ` or a triple newline -- invisible in the reply on screen, wrong only in
  what the model is given next time, which is the failure that module exists to prevent.
  Exactly one gap comes off rather than an `lstrip`, so a reply that begins with deliberate
  whitespace is not reformatted, and `continuing=True` takes none off at all: a `/more`
  continuation was generated from inside the content, where a leading space is one the
  model meant.

- A generation says **why it ended**, so a reply that finished and a reply that ran out of
  budget stop being the same thing. Both are text that stops, and only one is worth
  continuing; guessing from the token count is wrong in both directions, since a model that
  finishes on its last allowed token reads as cut off and so does one whose stop string
  matched there. `InferenceSession.stream_pieces` now ends with one extra `StreamPiece`
  carrying a `Finish`: `"end-of-text"`, `"stop"` with the string that matched, or
  `"length"`.

  **A piece of its own, not a field on the last real one.** Which token is the last one is
  known only a token later, so attaching the answer to the piece it describes would mean
  holding every piece back by a token -- delaying each character to say something about it
  afterwards, when the point of streaming is that it does not. The extra piece carries no
  new token, repeats the previous count, and flushes whatever text was still held back
  behind a partial stop match.

  **An abandoned stream has no reason at all.** Ctrl-C means the caller stopped asking;
  nothing ended the generation, so `finish` is `None` rather than a reason invented for it.
  In `chat --json` that pair is the whole point: `interrupted` true with `finish` null is
  an abandoned generation, `interrupted` false with a `"length"` finish is one the budget
  cut off, and both are short text that a script could not otherwise tell apart. `Finish`
  is nested under `"finish"` rather than flattened, because `"stop"` at the top level is
  already the list of stop strings that were *configured* and one key cannot be both.

  `trainai chat` prints one yellow line, and only for the token limit: *the model was still
  writing at the N-token limit, so this reply is unfinished*, with `--tokens` named in a
  one-shot run and `/more` and `/tokens` in the interactive one. A reply that ended at
  end-of-text or at somebody else's label is a reply that finished, and a line saying so
  would sit under every well-formed reply this project trains for, which is the same noise
  as no line at all. Nothing is said for a Ctrl-C, which already printed *Stopped.*
  Whether a reason means the reply was cut is `Finish.cut`, a property on the type, so no
  caller compares reason strings and there is no second copy of `"length"` to drift.

  Where two stop strings match at the same position -- `"\nUser"` and `"\nUser:"` both
  begin at the same newline -- the cut is identical and the one *reported* is the one that
  came first in the list the caller passed, rather than whatever a loop happened to reach
  first. No new flags, no new checkpoint keys, and no `content_hash` change.

Measured on the development machine: a 2-layer, 128-wide model (vocab 600, ctx 128)
pretrained 1,500 steps on a 270,501-token corpus to validation 2.9384, then fine-tuned
200 steps on a separate 437,931-token corpus prepared with the same tokenizer. Starting
from the checkpoint, the first training loss was 5.6994 against 6.2837 from scratch, and
validation 5.5632 against 6.0593.

Measured on the development machine (RTX 2050, 4 GiB, Windows 11), 1.1 MB corpus:
a 5.3M-parameter model trained at 54,296 tokens/s with a 400 MiB peak, validation
loss falling 5.797 to 4.475 across ten measurements. Resuming at the halfway point
reproduced the uninterrupted run's weights and AdamW moments bitwise. (Those two
loss figures predate the validation-weighting fix listed under **Fixed** below, so
each is biased by roughly 0.01 nats.)

#### Hardware portability

- `trainai setup` — checks whether the installed PyTorch matches the machine's
  hardware, and prints the install command that does. Catches the case where a good
  GPU is paired with the CPU-only wheel, which trains 20-100x slower and is
  indistinguishable from having no GPU from inside PyTorch. The wheel channel is
  chosen from the CUDA version the *driver* reports, not from a fixed guess, and
  every message carries the pytorch.org selector because the channel list moves.
  `--install` runs the command after showing it, refusing to touch a system Python
  without `--allow-global`, and confirming first.
- `trainai.hardware.probe` now distinguishes CUDA from ROCm (via `torch.version.hip`)
  and detects Intel XPU devices, recording the runtime in a new `backend` field
  alongside the torch device string. It also detects GPU vendors at the *system*
  level — driver tools on PATH, Linux PCI sysfs, the Windows display-adapter
  registry — so "no GPU" and "PyTorch cannot use your GPU" are reported as the
  different problems they are.
- `precision_for` is a pure function of stated hardware facts, with
  `resolve_precision` reduced to gathering those facts from the live machine. That
  split is what allows `tests/test_hardware_portability.py` to assert the right
  behaviour for thirteen machines this project does not have, including a 24 GiB
  4090, a bf16-less GTX 1080 Ti, an RX 7900 XTX, an RX 6900 XT, an MI210, an
  Arc A770, an Apple M2 and a dual-GPU box.
- `docs/hardware-support.md` — what is supported, what is verified, and why those
  are different claims.

#### Empirical hardware advisor (M3)

- `trainai.hardware.benchmark` — measures one candidate configuration by building the
  real model and running real optimizer steps on synthetic token ids, then reading
  `allocated_bytes.all.peak`, `reserved_bytes.all.peak` and `num_alloc_retries` from
  the allocator itself. Every result states whether memory was measured at all, so a
  CPU run reports "not measured" instead of a zero that reads as "used nothing". A
  budget is inert where there are no counters to check it against, which is what lets
  the planner pass the same budget regardless of device.
- `trainai.hardware.planner` — chooses a configuration by measurement. It climbs the
  preset ladder from the smallest shape and stops at the first failure, so a
  configuration that would page gigabytes onto a small card is never run. Three caps
  apply: measured peak against 85% of *free* VRAM, `train_tokens >= parameters x 20`,
  and measured step time against a time budget. Over-budget rungs are retried with the
  micro-batch halved and `grad_accum` doubled, holding the effective batch constant so
  the retry has the same learning dynamics. The result is classified into one of five
  regimes — `data-limited`, `vram-limited`, `compute-limited`, `unconstrained`,
  `blocked` — the last so that a driver fault cannot be reported as a memory limit.
  The measurement function is injectable, which is what makes the search testable
  against synthetic hardware on a machine with no GPU.
- `trainai plan` — prints the machine, then each candidate as it is measured, then the
  recommendation split by provenance: **measured** (a counter was read), **derived
  from your data** (arithmetic over counted tokens), **rules of thumb** (the learning
  rate, labelled because nothing tested it). It lists what it rejected and why, prints
  the equivalent `trainai train` command in full, and writes `plan.json`. `--time`,
  `--max-vram`, `--max-preset`, `--seq-len`, `--precision` and `--device` pin whatever
  you would rather decide yourself; `--json` emits the plan and nothing else. When
  nothing fits, the failure carries a real measurement of the smallest shape that was
  tried.
- `trainai train --plan plan.json` — applies a plan without retyping it. Explicit flags
  still win, so `--plan plan.json --steps 500` works. A plan from an incompatible
  version, or one whose vocabulary size does not match the dataset it is being applied
  to, is refused with an actionable message rather than silently misapplied.
- `docs/plan-format.md` — what `plan.json` contains, which two blocks `--plan` actually
  consumes, and why a plan is a record of one machine's measurements rather than
  portable advice.

Measured on the development machine (RTX 2050, 4 GiB, Windows 11) against the 1.1 MB
Shakespeare corpus: 3.23 GiB free of 4.00 GiB, a 2.74 GiB budget, and `tiny` at
32 x 2 x 256 accepted at 94.3K tokens/s with a 957 MiB peak — 34% of budget. It
rejected `small` for wanting 244M training tokens against a 334K corpus, reported
`data-limited`, and estimated 14 s; the run it printed took 16 s and dropped
validation loss from 6.28 to 5.32.

#### Evaluation, sampling and export (M4)

- `trainai.infer` — loads a finished run for generation. A run stays usable after
  its dataset is deleted, because the trainer now copies `tokenizer.json` into the
  run directory and the session prefers that copy. A tokenizer that does not match
  the weights is **refused, not warned about**: the same token id means different
  strings under different tokenizers, so the output is not degraded but confidently
  wrong, and nothing downstream can notice. Precision is resolved and reported —
  "bf16 (supported by this cuda device)" — because a sample nobody can reproduce is
  an anecdote.
- `GPT.generate` is a thin consumer of `generate_stream`, so the interactive and
  batch paths cannot drift; a test asserts they agree bitwise. Streaming holds back
  a trailing U+FFFD, because byte-level BPE can split a multi-byte character across
  two tokens.
- `trainai.eval` — recomputes loss, perplexity and bits per token over a whole
  split in non-overlapping windows, and states how much of the split it scored and
  what fell outside. Perplexity is per token of one tokenizer, so every report
  carries the vocabulary size and says it is not comparable across tokenizers.
- `trainai eval` — scores a checkpoint and reconciles the result against what the
  trainer recorded, printing the difference rather than leaving it to be noticed.
  `--split`, `--data`, `--which`, `--seq-len`, `--precision`, `--device`, `--json`.
- `trainai chat` — generates from a run: once from `--prompt` or a pipe, or
  interactively, where `/` commands change temperature, top-p, top-k and the
  repetition penalty between completions. `/more` continues the last output rather
  than accumulating a transcript, because a base model has never seen a dialogue
  and a transcript is a format it cannot follow. The banner says it is a base model
  before the first prompt: "what is the capital of France?" getting more questions
  back reads as broken rather than as working exactly as trained.
- `trainai.export` — writes a run out as a standalone model directory. The target
  is `LlamaForCausalLM`, because the architecture *is* a Llama structurally, so the
  export is a rename rather than a conversion. Writing `model_type: "gpt2"` instead
  would produce a directory that loads and generates nonsense, and there is no
  checksum on "do these weights match this architecture".
- Every export verifies itself before it is moved into place, and reports each
  check by name: the weight file is re-read and compared bit-exactly against the
  model, the written tokenizer is re-loaded and fingerprinted, and — where
  `transformers` is installed — the directory is loaded back and its logits
  compared against ours within 2e-4. `transformers` is deliberately not a
  dependency, so where it is absent that check reports as **not checked** rather
  than as a pass. The parity check has a negative control in the test suite: a test
  corrupts `rope_theta`, which leaves every shape intact, and asserts the export is
  refused, because a check that cannot fail is not evidence.
- An export is atomic — staged beside the destination, verified there, moved in
  only once every check passes — and `--force` replaces an export but refuses any
  directory that is not one. A `README.md` goes into the directory stating that it
  is a base model that completes text, since that is the only place the warning
  survives being zipped and shared.
- `trainai export` — `--format hf|safetensors`, `--dtype fp32|fp16|bf16`,
  `--which`, `--tokenizer`, `--no-verify`, `--force`, `--json`. Exit code 7
  (`ExitCode.EXPORT`) for an occupied destination or a failed check.
- `docs/export-format.md` — the exported layout, the full name mapping, what each
  check proves, and why the rotary convention needs no permutation.

Measured on the development machine (RTX 2050, 4 GiB, Windows 11), 1.1 MB
Shakespeare corpus, a 4.26M-parameter model over 600 steps at 90.3K tokens/s:
best validation loss **4.1364 at step 375** (perplexity 62.6), final 4.2187 at step
600 — worse, and reported as worse. `trainai eval` scoring the whole split
independently reproduced 4.1364 to four decimal places. The export's three checks
all passed with a **max logit difference of 3.81e-06** against `transformers`
5.3.0, and the resulting directory loads as `LlamaForCausalLM` with 4,262,144
parameters and generates through `model.generate()`.

#### Tabular data

TrainAI used to prepare a spreadsheet without comment. A 15.5 MiB weather CSV
renamed to `.txt` produced no complaint at all — the corpus measured 66% digits,
which was filed under "Writing systems" as a neutral statistic — and training
produced a model at perplexity 1.6 that generates flawlessly formatted weather rows
with **July temperatures for a January date**. Fluent form, wrong facts, and
nothing in the pipeline said so until a human read the samples.

- **`.csv` and `.tsv` are read directly** (also gzipped), one document per row from
  **one named column**: `--csv-text-column NAME`. Without it, a header meaning
  "this column is the text" is resolved from a short list, and anything else is an
  error listing the columns the file actually has. The list deliberately excludes
  `summary`, `description` and `title` — the weather CSV has a `Summary` column,
  and auto-selecting it would have trained on 96,000 repetitions of "Partly
  cloudy". Multi-line quoted fields are read correctly and byte offsets still point
  at the row's first byte. The separator comes from the extension and is never
  sniffed: a wrong guess yields one giant column and a dataset that looks fine,
  while the extension fails loudly with an error naming the absurd column. A `.csv`
  holding semicolons gets that error specifically.
- **Whole rows are never joined into sentences.** That means deciding what each
  field means, what unit it is in and what a blank cell stands for — domain
  judgements TrainAI has no business making invisibly.
  `examples/weather_to_text.py` makes them explicitly for one dataset and documents
  each one.
- **A corpus of table rows is detected whatever the file is called**, since the
  usual way one arrives is a CSV renamed to `.txt`. Prose never puts the same number
  of commas on every line and a table always does, so the share of text sitting on
  lines of identical field count is measured over the existing bounded sample — no
  extra pass. Weighted by characters rather than by lines, because lines are not
  equal-sized units: a file of unwrapped paragraphs with a 22-line option table in
  it is 96% table by line count and under 1% by character.
- **The verdict forks on the digit share**, because the two situations need
  different advice. Mostly digits (≥ 40%) means machine measurements, and
  `looks_like_a_table` stops preparation with exit code 3 — a tokenizer sees `9.47`
  and `9.48` as unrelated strings, so it cannot learn they are close, and it names
  regression as the right tool instead. Few digits means some column holds prose,
  and `rows_not_prose` warns and points at `--csv-text-column`.
- **`--allow-tabular` overrides the error**, downgrading it to a `tabular_override`
  warning. The finding is downgraded, never deleted: it is written to the dataset's
  `manifest.json`, so a dataset prepared this way says so permanently. Same
  principle as the export reporting a skipped check as skipped.
- **`vocabulary_saturated`** — a second, independent signal, from after
  tokenization. A byte-level BPE that reaches under 25% of the vocabulary it was
  asked for has run out of distinct text to merge; the weather data rendered as
  sentences built 698 of 8,192. This catches templated corpora and logs that are
  not delimiter-regular at all.
- Thresholds live in `validate.py` with the measurements that justify them beside
  each number. Through the real pipeline: nothing tabular below **0.97** (weather
  CSV 1.00, TSV of numbers 1.00, CSV of reviews 1.00, markdown that is only a table
  1.00, weather rendered as sentences 0.97) and nothing legitimate above **0.15**
  (Shakespeare 0.15, this project's Python 0.10, its markdown 0.08, unwrapped
  paragraphs with a table in them 0.00). The threshold is 0.80, over at least 20
  sampled lines.
- **A CSV row wider than its header is counted, not silently truncated.** The
  ordinary cause is a delimiter left unquoted inside a field, which splits the value
  and would otherwise hand back only the part before it — a fragment of the intended
  document with nothing to mark the loss. A `csv_rows_with_extra_fields` warning
  reports the count, in the spirit of "every dropped document is counted"; a trailing
  delimiter on every line trips the same count and is harmless.
- [docs/corpus-formats.md](docs/corpus-formats.md) — new, and the input-side
  counterpart to `docs/dataset-format.md`: what every accepted format is read as,
  how JSONL fields and CSV columns are resolved, why UTF-16 is refused for the
  line-oriented formats, and the detector with its measurements.

#### More formats accepted

Broadening what counts as a corpus, where extraction is exact — the bytes already
are text, or the format is a typed container with a real text field. No new
dependencies: everything here is stdlib.

- **More compression codecs.** Every readable extension was already accepted with a
  `.gz`; now `.bz2`, `.xz` and `.lzma` are too, decompressed while streaming so a
  compressed corpus never has to fit on disk twice. The compression is invisible
  downstream: the same corpus as `.gz`, `.bz2` and `.xz` yields a byte-identical
  document stream, so the codec cannot change a `content_hash`. `data inspect` now
  names which codec it found (`1 bz2-compressed`) rather than calling everything
  "gzipped". `.zst` is deliberately excluded — it reached the standard library in
  Python 3.14 and this project supports 3.10.
- **More text extensions** read as one document per file: `.markdown`, `.rst`,
  `.org` and `.log`, alongside the existing `.txt`, `.text` and `.md`. No markup is
  stripped from any of them. `.log` is included because a log file is text; that it
  usually makes a narrow corpus is measured after tokenization as
  `vocabulary_saturated`, not guessed at from the extension.
- **`.json` is read** instead of refused with an instruction to convert it. A
  `.json` file is an array of records — objects whose text field is resolved exactly
  as for JSON Lines, or bare strings — and each element is one document. An object
  wrapping exactly one array (`{"data": [...]}`, how most API dumps arrive) is
  unwrapped; anything ambiguous is refused with both readings named, including the
  hazard `{"text": "...", "tags": [...]}`, where the obvious "unwrap the single
  array" rule silently trains on the tags and discards the text. Elements are
  reported by array position rather than line number, since a `.json` file has no
  meaningful lines. No new option, so no prepared dataset's `content_hash` moves.
  The whole file is parsed at once because JSON has no record separator, so a file
  over 128 MiB is refused with the one-liner that converts it to streaming JSON
  Lines; the ceiling comes from measured peak memory (1.1×–3.9× file size depending
  on record shape) and is enforced against the bytes that arrive, not only the size
  on disk — a 269:1 gzip of repetitive JSON is easy to make, so a compressed file
  well under the limit could otherwise expand past it. Wide encodings (UTF-16/32)
  are fine here, unlike JSON Lines and CSV, because nothing is being split on the
  newline byte.
- **Archives are read in place**: `.zip`, `.tar`, `.tar.gz`/`.tgz`,
  `.tar.bz2`/`.tbz`/`.tbz2` and `.tar.xz`/`.txz`. A Kaggle download or a research
  corpus can be prepared without unpacking it first, and nothing is ever extracted
  to disk. Members are named `papers.zip::2023/notes.jsonl` everywhere they appear,
  may be any readable format, and may carry their own codec inside the archive.
  Members are read in **sorted order** rather than stored order, because
  `namelist()` returns insertion order — a property of the tool that built the
  archive, not of what is in it — so without sorting the same files zipped by two
  programs would produce different shard checksums. Verified: two zips of the same
  files added in opposite orders give an identical `content_hash` and identical
  per-shard `sha256`, as does the same content as a `.tar.gz`. Directory entries,
  symlinks (a zip symlink's stored bytes are its *target path*, which would enter
  the corpus as a document), dotfiles and `__MACOSX/` metadata are skipped;
  anything else unreadable, and any archive inside the archive, is counted **and
  named** in a note, because an archive is opaque and a member dropped silently is
  a corpus quietly smaller than the one you pointed at. One level only: a zip
  inside a zip is reported, never opened, which is also the real defence against
  42.zip's 4.5 PB. The bounds are 20,000 members and 32 GiB declared unpacked, both
  read from the table of contents. There is deliberately no per-member
  compression-ratio ceiling: gzip reaches 269:1 on legitimately repetitive JSON, so
  any threshold low enough to catch a bomb also refuses real corpora, and every
  reader here is already memory-bounded. A member's reported size is what it
  unpacks to, and `data inspect` labels it as such rather than saying "on disk".
- **A file with no extension is read** when you point straight at it, instead of
  being a dead end — `trainai data prepare corpus` works, and a codec suffix is
  still honoured, so `corpus.gz` does too. A file that *has* an extension TrainAI
  does not read (`scan.pdf`, `release.2`) is still refused; carrying an extension
  and reading past it anyway is how a binary file ends up in a corpus. Inside a
  directory walk these are still **not** selected, deliberately: naming a file is an
  instruction, finding one in a tree is a guess, and a stray `README` joining the
  training data is an invisible change to what the model learns. So no existing
  corpus directory's contents moved.
- **A directory walk is no longer silent about what it passed over.** Every file it
  did not select is counted and named — a `files_unrecognized` note plus a "Passed
  over" row — so `Files read 1` out of a directory of three accounts for the other
  two. Reading an extensionless file is likewise recorded, as an `assumed_text` note
  in the dataset's `manifest.json`, because "TrainAI decided this was text" is not
  something a dataset should be silent about.
- **A binary guard on that new path**, and it is load-bearing precisely because
  nothing else would object: UTF-8 decodes a NUL byte happily, so reading a `.png`
  as text does not fail — it succeeds, the tokenizer learns merges from the garbage,
  and the only trace is a control-character ratio reported as a neutral statistic.
  The same looks-like-success failure the tabular detector exists to prevent. The
  first block is sniffed and a NUL refuses the file, naming the byte, at discovery
  rather than part-way through a run, and on the bytes that arrive so a `corpus.gz`
  of binary is caught too. Wide encodings are the false positive to avoid here —
  every ASCII character in UTF-16 carries a NUL — so a UTF-16 byte-order mark or an
  explicit `--encoding utf-16-le` is recognised first and let through. Scoped to
  extensionless files: naming a file `.txt` is your assertion that it is text, and
  widening this to every text file would change what existing corpora do.
- **Word documents are read**: `.docx`, including one inside a `.zip` or behind a
  codec. A `.docx` is a zip of XML whose paragraphs are real elements, so extraction
  is exact rather than reconstructed from a layout — the line that keeps `.pdf` out
  (see [docs/corpus-formats.md](docs/corpus-formats.md#deliberately-not-supported)).
  One paragraph becomes one line, matching Word's own plain-text export, and table
  cells are read one per line because a technical document keeps real prose in
  tables — more than `python-docx` returns from `.paragraphs`. Verified against four
  real documents with pictures, tables, hyperlinks and footnotes: the non-whitespace
  characters extracted are **exactly** the concatenation of every `w:t` element in
  `word/document.xml`, nothing missed and nothing duplicated, and a strict superset
  of `python-docx`'s output. Duplication is the subtle bug here, not omission: a
  text box lives inside a paragraph and holds paragraphs of its own, and Word stores
  it twice — `mc:Choice` in current markup and `mc:Fallback` in older VML — so the
  extractor is a single depth-first walk that visits each element once and skips the
  fallback, rather than a loop over paragraphs. It is iterative, not recursive,
  because expat parses 20,000-deep nesting happily while a recursive walk raises
  `RecursionError` at 1,000. Field codes (`w:instrText`, i.e. `HYPERLINK "..."`) and
  text struck out under tracked changes (`w:delText`) are excluded as text in the
  file that is not text in the document, while an insertion is kept; headers,
  footers, footnotes, endnotes and comments are excluded as page furniture and
  out-of-flow parts. Only `word/document.xml` is read, not the whole package —
  measured on a real 3.2 MB document that part was 189 KB and the other 3 MB was
  PNG. A **document type declaration in any part is refused outright**: a DTD is
  where XML parsing stops reading a file and starts running a program, and the
  600-byte billion-laughs bomb arrives through exactly this path. This interpreter's
  expat does cap the amplification, but that depends on the libexpat a given Python
  was linked against and TrainAI supports 3.10 upward, so it is not left to somebody
  else's build; the format forbids a DTD anyway, and the check cannot misfire on
  prose because a raw `<` cannot appear in XML character data. Ceilings are measured,
  not guessed: 32 MiB for the XML part (`tracemalloc` put ElementTree's parsed tree
  at 7.9–9.3× the XML across four real documents) and 64 MiB for buffering a package
  that is not a plain file on disk, which is the only case where buffering happens —
  a `.docx` on disk is opened in place. A `.docx` is classified as a Word document
  **before** the archive rule, so neither pointing at one nor finding one inside a
  `.zip` walks it as the container it technically is. A legacy `.doc` renamed to
  `.docx`, a password-protected `.docx` (both OLE compound files, not zips) and a
  plain `.zip` renamed to `.docx` each get their own message naming what the file
  actually is. `--encoding` does not apply, since an XML part declares its own. No
  new option, so no prepared dataset's `content_hash` moves.
- **SQLite databases are read**: `.sqlite`, `.sqlite3` and `.db`. One row of one
  column of one table is one document — the same rule as a CSV, because a database
  table is a table. `--db-table` names the table (a database with exactly one uses
  it), and the column is named with the **existing `--csv-text-column`** rather than
  a second flag, because it means the same thing and resolves the same way, against
  the same candidate list. Both choices are printed and recorded in the manifest.
  **Row order is written into the query**, not inherited from the engine: a plain
  `SELECT` comes back in rowid order today and is free to come back in index order
  tomorrow after someone adds an index, at which point nothing in the file has
  changed and every checksum in the manifest has. So ordinary and virtual tables are
  ordered by rowid, a `WITHOUT ROWID` table by its primary key, and a table with
  neither is refused rather than read in arrival order — with each rowid spelling
  (`rowid`, `_rowid_`, `oid`) probed rather than assumed, since a column of that name
  shadows it. A **view is never chosen automatically and naming one is refused**: a
  view is a stored query, and a query has no row order to pin. A **full-text index
  counts as one table, not six** — an FTS5 table's five shadow tables are filtered by
  name prefix, so a searchable database reads without a flag instead of demanding a
  choice between six names meaning one thing. The database is opened **read-only**,
  which is measured rather than assumed: on a database with an unmerged write-ahead
  log, a read-write open reads every row and then *changes the main file's checksum*
  on close by checkpointing the log into it, while `mode=ro` reads the same rows and
  leaves the file byte-identical. Its one cost is that a read-only connection cannot
  clean up the `-shm`/`-wal` sidecars, so those are left next to the database;
  leftover sidecars are inert and a rewritten corpus file is not. `immutable=1` is
  deliberately not used — it skips recovery, which on that same database means
  quietly reading stale rows and reporting success. Because `.db` is a generic
  extension the `SQLite format 3` header is verified first, so a Windows `Thumbs.db`
  is refused as what it is; the check is a better message, not a validation, and a
  valid header over a damaged body passes SQLite's own words on. No archives and no
  compression: SQLite seeks around a file — header, then catalogue, then scattered
  pages — which a decompressing stream cannot do, so a `.db` inside a `.zip` or a
  `.db.gz` is refused with that reason, and inside an archive walk it is counted and
  named as an ignored member rather than failing the whole archive. A directory walk
  passes over a database it was not told to read, counted and named, since reading
  one needs a table and a column named for it. Memory needed no ceiling here, unlike
  `.json` and `.docx`: the cursor is streamed, and walking 40 MB of text out of a
  database peaked at 4.5 KB of traced memory. The connection is explicitly not
  thread-bound, which is not a shortcut — documents leave the reader as a generator
  and the BPE trainer pulls that generator from its own worker pool, measured at
  seven distinct threads over one corpus and zero overlapping pulls out of 3,000, so
  the default would have failed every database corpus part-way through tokenizer
  training with a message about thread ids. A non-text cell is reported with its row
  number or skipped under `--on-error skip`; a `NULL` is an empty document, not a
  broken one. `--encoding` does not apply, since SQLite stores its text encoding in
  the file header.

#### One command from a corpus to a model

- **`trainai quickstart CORPUS`** — runs the five documented steps in order (prepare,
  plan, train, evaluate, sample) into one `--out` directory, stopping once to confirm
  before the long step and quoting the duration `trainai plan` just **measured** on
  this machine rather than a guess. It is a shortcut *through* the CLI, not a second
  implementation of it: each step calls the same `run_*` function the individual
  command calls, and prints the individual command it stands in for before running it,
  so a user who wants to change one number copies the printed line and stops using
  `quickstart`. Nothing here decides anything the individual commands would not — the
  model shape comes from `plan`'s measurements exactly as it would by hand.
- **An existing dataset is reused; a plan never is.** The asymmetry is the design.
  Preparing the same corpus with the same settings and the same seed produces
  byte-identical shards, so a second pass could only be slower — that step is skipped
  with a note naming `--force`. A plan is a measurement of one machine at one moment,
  and free VRAM moves when a browser opens, so it is always retaken. Training on a
  stale measurement is the failure this project exists to avoid.
- **`--device` and `--precision` default to unset, not to `auto`.** `train` treats
  any value it is given as an override of the plan's *measured* precision, so
  defaulting them to `auto` here — as every other command does — would silently
  discard the measurement on every run. Only the steps with no plan to consult
  (`plan`, `eval`, `chat`) receive `auto`.
- The sample prompt defaults to the corpus's **own opening**, collapsed to one line.
  A base model continues text rather than answering, so the prompt has to look like
  the beginning of something it was trained on; a hard-coded English prompt would be
  the wrong opening for a corpus in another language and would make a working model
  look broken. `--prompt` overrides it.
- The evaluation scores both splits. It costs a second pass and is what makes
  overfitting visible, which on a first run over a small corpus is the most likely
  thing to have gone wrong.
- Three deliberate omissions. **No `--json`**: five reports concatenated is not a
  document, and anything scripted should call the individual commands, each of which
  emits its own object. **No network**: the corpus is the user's to supply, nothing in
  the installed package reaches the network, and a convenience command is a bad place
  to start. **No reused plan**, as above. Declining the confirmation is reported as a
  real outcome — the dataset and the plan are kept, and the one command that resumes
  from there is printed.

#### A memory budget you can set

- **`trainai plan --max-vram 4GB`** — plan against the memory you are willing to spend
  rather than against everything that happens to be free. The budget was `85% of free
  VRAM`, computed inside the library and exposed on no command, so the one case it
  cannot handle had no answer: planning on an idle GPU and then opening a browser, a
  game, or a second job. The plan stays valid-looking and the run OOMs — or on Windows
  pages silently — thousands of steps in. The cap takes `6GB`, `6.5GiB`, `512MB`,
  `2048MiB` or a bare number of gigabytes.
- **`GB` means `GiB`, deliberately.** Every tool a user reads VRAM from — nvidia-smi,
  Task Manager, this project's own `fmt_bytes` — reports binary units, and a card sold
  as "8 GB" holds 8 GiB. Reading `--max-vram 8GB` as 8×10⁹ would report it straight back
  as "7.45 GiB" and look like TrainAI had quietly shaved it. A bare number means
  gigabytes for the same reason a bare `--time` means minutes: it is the unit the
  quantity gets discussed in, and reading `--max-vram 6` as six bytes would reject every
  configuration there is and call it a memory limit. Unlike `--time` it takes one
  quantity rather than a sum — `1h30m` is a natural way to say a duration, `1g512m` is
  not a natural way to say a size.
- **It only ever lowers the budget.** Raising it would size a plan against memory the
  device does not have, which means the search measures candidates it cannot hold — the
  OOM the measurement pass exists to prevent. So a cap above what is free is neither
  obeyed nor refused: it is reported in the plan's notes as having changed nothing,
  because the same `--max-vram 12GB` in a shared script is right on one machine and
  generous on another. A cap on a machine with no VRAM budget at all says that too,
  rather than being silently ignored.
- **The plan records what was asked and what applied, separately.** `vram_cap_bytes` is
  the number the user typed; `provenance.measured.vram_budget_bytes` is what every
  measured peak was compared against. Comparing them is how a reader tells a small plan
  on a small card from a small plan that was asked for. The cap sits at the top level
  rather than under `provenance.measured`, since that block is for numbers a counter
  reported — filing a typed number there is the category error the three-way split
  exists to prevent.
- When a cap fits nothing at all, the capacity failure names the flag — "did not fit the
  2.00 GiB you allowed with `--max-vram`" — instead of telling the user their machine is
  too small for a limit they set themselves, which sends them looking at their card.

### Fixed

- **`trainai plan` said nothing was rejected in the same report that said memory
  shaped the batch.** The `unconstrained` regime's detail read "Every rung fitted, had
  the data behind it, and finished inside the time horizon", and it was printed whenever
  the ladder ran to its end — which is not the same as nothing having been refused. A
  rung is accepted as soon as *some* micro-batch on it fits, so the top preset can be
  reached only by halving the batch and accumulating: measured on synthetic cards over a
  4 B-token corpus, that is 3 rejections at 3 GiB, 2 at 4 GiB and 1 at 6 GiB, each of
  which printed the sentence claiming otherwise. Two rows below it the same report
  already said "Gradient accumulation is in use: 32 sequence(s) at a time, 2 times per
  step … That is what made this shape fit", so the plan contradicted itself, and the
  contradiction pointed the wrong way: "nothing was rejected" is the one reading that
  means a bigger card would change nothing.

  The regime set is unchanged — `unconstrained` is the right name for a ladder that ran
  out, and it is a documented, machine-readable value that `plan.json` consumers key on.
  What was false was the prose, so `regime_detail` is now built from the record rather
  than assumed: it reports the rejections it counted and names the last of them, since
  that is the shape a user would try next, and when the accepted candidate accumulates it
  says memory shaped the batch rather than the model, with the effective batch and the
  factor. "Every candidate fitted first time" is kept for the case it was written for and
  is now reachable only when nothing was refused at all. `docs/plan-format.md` carries the
  matching caution, and `train.grad_accum` above 1 is the same fact as a number.

- **`chat` lost the start of the conversation mid-reply without saying so, and the
  reserve that was supposed to fix it is measurably the wrong fix.** `--tokens` defaults
  to 200 while `tiny`'s context is 256 and a measurement model's is 128, so a reply that
  runs to the limit is the ordinary case rather than a corner one. Past the context the
  sampler's window holds nothing but the model's own output: the question is gone, the
  text still looks fluent, and nothing on screen distinguished that from a reply the
  model simply chose to write.

  `_fit` recorded the open question in its docstring — what to reserve for the answer
  when `--tokens` is larger than the whole context, "a policy that needs measuring, so it
  is not guessed at here". It has now been measured, and the answer is to reserve
  nothing, for a reason that is arithmetic rather than empirical. Reserving `r` means
  fitting the prompt to `seq_len - r`, and the window slides anyway once the reply
  outgrows `r`; with a prompt of `P` in a context of `C`, the window holds `P` tokens of
  conversation until generated token `C - P` and `C - i` from then on, and a larger
  budget never keeps fewer whole messages, so `P` is largest at `r = 0`. Counted over
  216 held-out sessions built from `data/chat-typed`'s validation split, against
  `--tokens 200` on a 128-token context: **0 of 172,800 positions** where any reserve of
  8, 16, 32 or 64 showed the model more of the conversation than reserving nothing. The
  mean fitted prompt falls from 114.0 tokens to 53.0 across that range — the reserve
  spends the context sooner, it does not save it.

  What a reserve does buy is a window whose front is still a message boundary, which is
  the property `_fit` exists for. Teacher-forced against the same 216 held-out replies on
  a 128-context model (`runs/_ctx`, best val loss 0.1169), simulating the slide exactly —
  `slide_caches` re-anchors rotary keys to zero and matches a fresh forward pass over the
  same window to 2.98e-08, so a forward over the last `seq_len` tokens is what generation
  computes: the slide put the window's front mid-message for **11.0%** of generated
  positions at no reserve and **0.0%** from a reserve of 16 up, while the loss moved from
  **0.0058 to 0.0055 nats/token** — perplexity 1.006 against 1.005. That corpus is
  memorised at those numbers, and it is the only chat corpus here, so the honest reading
  is that the alignment is worth an amount this repository cannot measure, against a
  context cost that is certain. Those replies average 11.8 tokens, which is also why the
  loss sweep never reaches the eviction: it takes 128 generated tokens, and the counting
  above is what covers that case.

  So no reserve was added. Instead the slide is reported, after the reply rather than
  before it: once generation ends, `chat` says how many of the reply's tokens had the whole
  prompt in the window and how many were written without its start. Predicting it from
  `--tokens` instead would have printed a warning on every turn of a small model, since the
  budget is an upper bound most replies never reach — a note on every generation is a note
  nobody reads. `--json` carries the same figure as `window_slides_after`, alongside the
  existing `prompt_truncated_from`; the two are mutually exclusive, because a prompt that
  never fitted had its front cut before generation started and describing that loss twice
  is two yellow lines about one thing. Aligning the *slide* to message boundaries would buy
  the layout without the context cost, and wants a corpus where old turns are load-bearing
  before it earns the layering it would take — `data/chat-typed` answers "what is 2 + 2?"
  the same with or without the turns in front of it.

- **CI was red on Python 3.10 from the commit that added the release check.** Both 3.10
  jobs — ubuntu and windows — failed four tests in `tests/test_release_check.py`, while
  3.11, 3.12 and 3.13 passed on both platforms and the local suite was green. One cause:
  `tools/release_check.py` reads `project.version` with `tomllib`, which is stdlib only
  from 3.11, and this project supports 3.10.

  The script's refusal below 3.11 is deliberate and stays. Its own docstring gives the
  reason — a regex is the worse trade for a script whose entire job is catching a version
  that is a near-miss — and the release workflow pins 3.12, so the path that matters is
  never the one that refuses. What was wrong was the tests: they called it on 3.10 and
  asserted the answer they get on 3.12.

  The six tests that parse a real `pyproject.toml` now skip below 3.11, spelling the
  boundary the way the script spells it, which is the guard `tests/test_conventions.py`
  already applies to its four `tomllib` readers. Two of those six were not failing — they
  assert exit 1, and the interpreter refusal is also exit 1, so on 3.10 they passed
  without going near the tag or the notes file they exist to check. Skipping them is what
  makes them honest; leaving them would have been the more comfortable kind of green.

  A skip on two of six matrix jobs is a hole of its own, so one new test asserts something
  on every interpreter instead: below 3.11 the refusal has to name both the interpreter it
  found and one that works — the alternative a contributor gets is `ModuleNotFoundError:
  No module named 'tomllib'` and a guess — and from 3.11 the same call has to return the
  version. One test with two branches rather than two tests each skipped somewhere, so
  neither branch can rot on the interpreter that does not run it.

  Only 3.13 is installed on the development machine, which is how this reached `main`
  twice: the local suite cannot run the failing path at all. The sub-3.11 branch was
  proved by forcing `sys.version_info` below the floor, which reproduces the refusal
  verbatim (`... needs Python 3.11 or newer for tomllib, and this is 3.10`) and confirms
  `main` turns it into exit 1 with a tag that is otherwise correct — the vacuous pass,
  demonstrated rather than argued. The 3.10 jobs are the real gate.

  Two conventions tests now hold the interpreter list itself, because `requires-python`,
  the PyPI classifiers and the CI matrix are three hand-edited copies of one fact and
  nothing compared them. The floor has to equal the oldest version in the matrix — pip
  enforces the floor, and a job running there is the only thing that makes it true — and
  every version the classifiers advertise has to be one CI runs, since that list is what
  PyPI's sidebar shows to people who will never read `pyproject.toml`. Both fail when
  they should: raising the floor to `>=3.11` without pruning the matrix fails the first,
  and dropping `3.10` from the matrix fails both. The floor is compared and the ceiling is
  not, deliberately: classifiers lagging a brand-new Python is a normal few weeks, while
  an advertised version with no job behind it is a claim.

- **Four of the five commands that read a checkpoint told the user to fix it with a flag
  they had not typed.** "Point `--resume` at a file written by `trainai train`" is the
  hint for a file that is not a checkpoint, and it is correct only for `train --resume`.
  `load_checkpoint` is also reached from `finetune --from`, `chat`, `eval` and `export`,
  and it is not told which — so pointing `finetune --from` at a `torch.save(model, path)`
  file diagnosed it precisely and then named the wrong option to correct it with. The hint
  now names the three shapes accepted instead of a flag, which is advice every caller can
  act on and matches what `--resume`'s own help text already said.

  This is the same defect as the hints naming flags that do not exist, below, and it shows
  the limit of the test written for those: it checks each flag a finding names against the
  click parameters of the command that raises it, and `--resume` passes that check because
  it is a real flag on a real command. There is no equivalent test to add here, because
  there is no command to check against — `load_checkpoint` is one function reached from
  five commands, and the only hint that can be right for all five is one that names no
  flag. Verified by pointing each of `finetune --from`, `chat`, `eval` and `export` at a
  `torch.save(model, path)` file: each still exits 5 naming what the file holds, and none
  now mentions an option its own `--help` does not list.

- **The test suite had been red on every push for four commits, on a test that
  asserted a property of the CPU rather than of the code.**
  `test_a_mask_of_ones_gives_exactly_the_unmasked_loss` compared
  `F.cross_entropy(reduction="mean")` against `(per_token * weights).sum() /
  weights.sum()` with `==`. Those are the same arithmetic in two different float32
  accumulation orders, so whether they agree bit-for-bit depends on the vector width
  of whatever machine is running them. They agree exactly on the development machine
  — the gap is `0.0` there, which is why it was written that way and why it could not
  be reproduced locally — and on a hosted runner the pair came out
  `4.196225166320801` against `4.196224689483643`: `4.77e-7` absolute, `1.14e-7`
  relative, **1.9 float32 ulps**. The failing job set moved between runs (Ubuntu
  3.10–3.13 in varying combinations, `windows-latest` py3.12 once) with an identical
  `torch 2.14.0+cpu` in both the passing and the failing Ubuntu jobs, which is what
  identifies the variable as runner hardware rather than Python version.

  The assertion is now a tolerance, `rel=1e-6`, and — because a tolerance on its own
  weakens the claim — the formula is pinned separately by an exact float64 reference
  computed in the test rather than by the other float32 path. The tolerance itself is
  a test subject: `test_the_loss_tolerance_admits_the_gap_a_real_runner_produced`
  holds the measured pair as a witness and asserts it from both sides — that `1e-6`
  admits a gap a real runner produced, that the gap is a rounding gap and not a
  formula error (under 4 ulps), and that the tolerance stays below `1e-4` where a
  formula error would start to hide. Tightening it back to `1e-9` and loosening it to
  `1e-3` both fail. The two code paths in `gpt.py` were deliberately **not** merged
  to make the difference vanish: that would replace a measured agreement between the
  shipped hot path and the masked path with a comparison of one code path against
  itself, which is the weaker claim wearing the stronger operator. The rest of the
  suite was swept for the same class of assertion; the remaining exact float and
  `torch.equal` comparisons all compare a value against itself, against a slice of
  itself, or across a save/reload of identical weights through identical code, where
  bit-equality is a real guarantee.

- **Two documents described a repository that no longer exists.**
  `docs/hardware-support.md` still said the repository had no remote and that therefore no
  CI run had ever happened — a claim `tests/test_conventions.py` has had a net for since
  the same claim went stale in the README, and which every one of that net's six phrasings
  missed, because they are all built around "never" *following* "CI" and that sentence put
  "no" in front of it. It now says what is and is not continuously verified: the CPU
  suite on two platforms and four Python versions is, a GPU is not and cannot be from a
  hosted runner. Two phrasings were added to the net, and the escaped sentence itself is
  pinned in the test — verbatim, where a net that scans Markdown cannot trip over the
  quotation — as the thing they have to keep catching, so an entry added by guess is
  distinguishable from one added by a failure. `CONTRIBUTING.md`'s scope section still
  listed fine-tuning as out of scope, three milestones after `trainai finetune`,
  `docs/finetuning.md` and `tests/test_cli_finetune.py` shipped; it now draws the line
  where the code does — your own checkpoints on a second corpus, and no LoRA, adapters or
  imported weights.

- **`trainai quickstart` could not run at all, on any input.** The CLI accepts
  `--jsonl-messages-field` there and forwarded it to `run_quickstart()`, which did not
  take that parameter, so every invocation died on a raw
  `TypeError: run_quickstart() got an unexpected keyword argument` before doing anything.
  The command a first-time user is pointed at first was the one command that was
  completely broken. It was missed by 2,320
  passing tests because every quickstart test called `run_quickstart()` directly, so the
  argv path in `cli/main.py` had no coverage at all while `main.py`'s ten other commands
  kept the module comfortably above its per-module coverage floor. The flag is now
  threaded through all four places it has to reach — the signature, the `_Reading` record,
  the `data prepare` call, and the reader that takes the sample prompt from the corpus's
  own opening — and the copyable command printed above step 1 shows it, which it did not
  before. Two nets close it: a static one that resolves every `run_*(...)` call in
  `cli/main.py` against the signature it actually calls, in both directions, so a
  forwarded flag no implementation accepts and a required argument no command passes both
  fail; and a registry over `_Reading`'s own fields, so a sixth reading flag cannot arrive
  without a test that it reaches both steps that read the corpus.

- **`--precision fp64` was accepted, silently treated as `auto`, and then reported as
  though that had been the ask.** Typer validates an `Enum` and a `bool` and nothing
  else, so an option declared `TEXT` whose valid values live only in its help text
  arrives as an unchecked string — and `Precision = Literal["auto", "bf16", "fp16",
  "fp32"]` is erased at runtime, so it validated nothing either. `precision_for` had
  branches for cpu/mps, fp32, bf16, fp16 and then a fall-through comment reading
  `# auto`; an unknown value matched none of them and fell through. The failure is worse
  than an ignored flag: on a card that supports bf16, `--precision bf6` — one character
  off — got exactly what `bf16` would have given, the same dtype and the same
  "supported by this cuda device" note, so a run asked for a precision that does not
  exist and answered with a plausible one. A misspelt flag *name* is caught by the
  parser; a misspelt flag *value* had nothing to catch it.

- **`--device gpu` failed with a traceback listing twenty backends TrainAI cannot train
  on.** The string went to `torch.device`, which refuses it with a bare `RuntimeError`
  naming what torch was compiled with — `mkldnn`, `opengl`, `ideep`, `ve`, `fpga`,
  `lazy` — and only `TrainAIError` is rendered without a traceback, so the error escaped
  the CLI boundary raw. TrainAI has a training path for five device names, and five is
  the list the user now gets. `--device cuda:` was a second way in: `"cuda:".partition(":")`
  yields an empty index, which is falsy, so the index check was skipped and torch raised
  again. Validation branches on the separator now, and `cuda:1` — the only way to choose
  between two cards — still works.

- Both are fixed by one shared `trainai.errors.check_choice`, which also replaces the
  three hand-rolled copies of the same raise that already existed (covering `--which`,
  `--split`, and `--format`/`--dtype`) and names a near miss when there is one. The `difflib` cutoff sits at 0.7 rather than the
  default 0.6 deliberately: at 0.6, `--device gpu` is answered with "did you mean xpu?",
  a confident wrong answer to someone holding an NVIDIA card. Listing the real values and
  suggesting nothing is the better failure. Every real near miss still lands —
  `bf6`→`bf16`, `fp64`→`fp16`, `laest`→`latest`, `safetensor`→`safetensors`. The
  precision and device lists are now derived from their `Literal` types with `get_args`
  instead of being written out beside them; there were three copies of the precision list
  and five of the device list, and two `--device` help texts had already drifted, omitting
  `xpu`. Values are checked before the checkpoint is read, so a typo does not cost a
  multi-gigabyte load first.

- **Every real training run printed each data-budget note twice.** `Trainer.run` logs
  `budget.warnings()` when it starts, and the CLI's `Data budget` panel printed the same
  strings as `What to expect` bullets a few lines above it, so a run whose corpus was
  both data-limited and only partly read said four things where two were true. The
  trainer's copy is the one that stays -- it is the layer a caller cannot skip, and a
  corpus quietly being memorised is exactly what `trainai.train.budget` exists to
  announce, so the announcement cannot depend on going through the CLI. `--dry-run`
  keeps the bullets, because it never builds a trainer and would otherwise print no
  advice at all in the one command whose whole purpose is answering whether the run is
  sensible.

- **On Apple Silicon the planner had no memory opinion at all, so an oversized model
  was found out by the crash rather than by the plan.** `probe_hardware` returned an
  empty `gpus` list on MPS, which made `primary_gpu` `None`, which made
  `vram_budget_bytes()` return 0 — and the planner's over-budget check is gated on a
  positive budget, so it never ran on any Mac. The probe now builds one `GPUInfo` from
  `torch.mps.recommended_max_memory()`, Metal's declared working-set ceiling, minus what
  torch already holds, capped by what the OS reports free. The cap is the point: memory
  is unified, so the ceiling knows nothing about the other processes on the machine and
  a 24 GiB Mac with 3 GiB free has a 3 GiB budget. Where no ceiling can be read the
  behaviour is unchanged — no budget, chosen by measurement — but the user is now told
  instead of getting a silent 0 that reads as "fits fine".

  Peak memory is deliberately **still** reported as not measured on Apple. `torch.mps`
  exposes no `max_memory_allocated` and no `reset_peak_memory_stats`, so there is no
  peak to read; sampling the current allocation and calling it a peak would be a
  fabricated number in the one field whose job is to say whether a number was measured.
  A test asserts both halves, so the honest half cannot be quietly "fixed" later.

  `doctor` labels these numbers as a ceiling and usable headroom rather than as
  "VRAM free of total", because there is no dedicated pool to be free of. It also stops
  printing "TF32 no" on every non-NVIDIA backend, which read as the hardware falling
  short of something rather than the concept not applying.

- **`chat` asserted that nothing in the model's training data was a dialogue, which
  this repo itself made false.** The banner and `chat --help` both stated it as a flat
  fact. The repo ships `data/corpus-assistant/chat.jsonl` — 651,448
  `User:`/`Assistant:` examples — so anyone who trained on the assistant corpus was
  told, by the tool, that their dialogue data did not exist. A checkpoint records its
  dataset's `content_hash`, not its contents, so `chat` genuinely cannot tell prose from
  dialogue; it now states that the answer depends on the corpus and leaves the corpus to
  the person who chose it. The test covering the banner had pinned the false phrase, so
  it was asserting the bug; it now asserts the dependency is named and that the old
  claim is absent.

- **`--steps` documented a default the cap did not honour, and nothing reported the
  difference.** The help text read "Defaults to about three passes over the training
  split" and stopped there. The derived count is capped at 20,000 steps, which on this
  project's own 633,422,803-token prepared dataset at the default 2,048 tokens per step
  is 0.065 of one pass — the documented default was wrong by a factor of forty-six, in
  the direction of doing less work than promised, and a user who typed no `--steps` at
  all had no way to find out that a cap rather than their corpus decided how much of
  their data was used. The help text now names the cap and what it means on a large
  corpus, the three-pass target and both bounds are named constants (`DERIVED_EPOCHS`,
  `MAX_DERIVED_STEPS`, `MIN_DERIVED_STEPS`) instead of literals buried in one
  expression, and `DataBudget.warnings()` gained the mirror of the existing memorising
  warning: when a run makes under one full pass it reports how many tokens are never
  read and what `--steps` would reach a full pass. On the dataset above that is
  592,462,803 tokens never read, and `--steps 309,288` to reach one pass. The note
  states the measurement and does not call it a mistake — under one epoch is normal and
  correct when the corpus is large for the compute. `tokens_never_read` and
  `leaves_corpus_unread` are in the budget's JSON. An explicit `--steps` is still never
  capped.

  The negative-control test for `warnings()` had to be corrected to notice this: its
  "well proportioned" run processed 100,000 tokens of a 100,000,000-token split, which
  it got away with because it was written to dodge only the two checks that existed.

- **`trainai setup` told every Windows AMD user their GPU was unusable, which stopped
  being true.** The advice read "PyTorch has no ROCm build for Windows, so an AMD GPU
  cannot be used for training here directly," and offered WSL2 or the CPU. PyTorch's
  own channel is indeed still Linux-only — `download.pytorch.org/whl/rocm7.2` contains
  no `win_amd64` file — but AMD publishes its own multi-architecture index that does,
  including `win_amd64` wheels for RDNA2 targets as small as gfx1034. The pinned ROCm
  channel was also two years stale at `rocm6.2`. Both AMD notes now name the real split,
  which is not Linux versus Windows but which index: the published channel covers only
  AMD's official support matrix, and that matrix lists no consumer RX 6000 card at all,
  so a Linux user with one was being sent to a wheel that installs and then raises on
  the first kernel launch. `setup` still refuses to print a runnable command on Windows.
  The GPU is chosen there by a pip extra naming the card's architecture
  (`torch[device-gfx1034]`), reading that architecture needs the working PyTorch being
  installed, and a guessed target produces a wheel that imports cleanly and fails later —
  so it asks for the one number instead and links the matrix to look it up in. Found by
  checking what an RX 6500M would actually do before claiming anything about it.

- **Every tokens/second figure on an Apple or Intel GPU timed the queue, not the work.**
  The training loop measured a step with a `perf_counter` pair and synchronised the
  device between them for CUDA only. MPS and XPU are asynchronous too: they queue
  kernels and return immediately, so on an M-series Mac the loop timed dispatch, and
  every throughput number, ETA and step duration in `metrics.jsonl` came from that
  measurement. `trainai bench` already had a helper covering all three backends, so the
  two disagreed about what "this step took N seconds" means — the benchmark predicted a
  throughput the loop could not reproduce on the same machine. There is now one
  `synchronize()` in `trainai.train.loop`, used by both. Failures stay suppressed: a
  driver that cannot synchronise will be reported by the next real operation with a
  better message than one from a timing call, and it should not end a training run.
  Found while preparing for a report from an M4 Mac, on hardware this project does not
  have; the fix is pinned by tests that stub `torch.mps` and `torch.xpu`, in the same
  spirit as the rest of `tests/test_hardware_portability.py`.

- **A machine with two GPU vendors was told "A amd/nvidia GPU is present ... cannot use
  it".** `doctor` and `setup` both print the "your hardware is here and PyTorch cannot
  see it" sentence, and both built it by slash-joining the raw vendor slugs. On a laptop
  with integrated Radeon graphics beside an NVIDIA card that is three errors in one
  sentence: identifiers where spellings belong, the wrong article, and a singular noun
  and pronoun for two cards. It now reads "AMD and NVIDIA GPUs are present ... cannot
  use them", from a single `describe_unusable_gpu`, and the two table rows listing
  vendors use the same spellings. The slugs stay lowercase in `--json`, which is what
  anything scripted matches on. Only reachable with a CPU-only PyTorch, so the
  development machine never rendered it; it was found by installing the built wheel into
  a clean virtualenv, where the default PyPI `torch` on Windows is a CPU build.

- **Every command TrainAI printed for the user to copy was broken across lines by the
  terminal.** `trainai plan` ends by printing the exact `trainai train` command it
  recommends, which for a real dataset is around 150 characters. Rich wraps at the
  console width, so at any ordinary terminal size that arrived as three lines — and
  what a user copied out of it ran the first line as a complete command and then
  handed the shell a line beginning `--steps`. Four other sites printed commands the
  same way: the `pip install` line `trainai setup` tells the user to run, both `Next:`
  hints in `trainai data`, and the `transformers.AutoModelForCausalLM.from_pretrained`
  snippet at the end of `trainai export`. All of them now go through one
  `print_command` helper that passes `overflow="ignore", crop=False`, letting the
  terminal soft-wrap so the text stays a single logical line that survives a copy.
  Turning off wrapping alone would not have been enough: Rich then *crops* to the
  console width instead, which is worse, because a cropped command still runs — with
  the last few flags silently missing.
- **`trainai train` printed the checkpoint path with backslashes directly above a
  path with forward slashes.** The Checkpoint row of the result table went through
  `str()` while every other path in the project goes through `as_posix()`, so on
  Windows the same table read `runs\a\b\step-0000075.pt` two rows above
  `runs/a/b/metrics.jsonl`. Now consistent. `trainai train` also ends by naming the
  next two steps (`trainai eval` and `trainai chat`), which every other command that
  leaves something on disk already did.

  Both of these were found by running the documented quickstart end to end and
  reading the output, not by a test. Nothing in the suite asserted how a command was
  printed, only what it contained. There are now tests that assert the recommended
  command survives a 50-column terminal intact, a check that it is not cropped, and a
  census of the call sites so a revert at any one of them fails by name.

- **Inside a container, `doctor` reported the host's CPUs and RAM, not the ones the
  run could actually use.** A 4-core, 8 GiB container on a 64-core, 512 GiB host was
  told it had 64 cores and 512 GiB, and that number was what `events.jsonl` recorded
  as the machine the run ran on — so a run reproduced on the same hardware
  unconstrained would be compared against a record of a machine that never existed.
  Neither of the two sources TrainAI used knows about cgroups: `psutil`'s Linux
  backend contains no cgroup handling at all (`virtual_memory()` reads
  `/proc/meminfo`, `cpu_count_logical()` reads `SC_NPROCESSORS_ONLN` and then
  `/proc/cpuinfo`, all host-wide), and `os.cpu_count()` is the host's count by
  definition. The probe now reads the cgroup limits itself, in both spellings — v2
  `cpu.max` / `memory.max` / `memory.current`, v1 `cpu.cfs_quota_us` +
  `cpu.cfs_period_us` / `memory.limit_in_bytes` / `memory.usage_in_bytes` — and also
  honours a CPU affinity mask, which is the independent narrowing that `taskset` and
  HPC schedulers apply. A v1 "unlimited" is a sentinel near 8 EiB rather than a
  missing file, so it is filtered instead of being reported as a machine with eight
  exabytes of RAM, and a quota looser than the machine is not reported as a limit at
  all. `HardwareProfile` gains `cpu_limit`, `ram_limit_bytes` and
  `ram_limit_used_bytes` (all `None` when unconstrained, so nothing changes on an
  ordinary machine), and the `usable_cpu_count` / `usable_available_ram_bytes`
  properties are what `doctor` and the run record now print. A nested cgroup on a
  host is deliberately not resolved: that needs `/proc/self/cgroup` parsing, which is
  inference rather than measurement, and failing towards "no limit" never invents a
  cap that is not there. **Limitation, stated plainly:** no real container was used to
  test this, and GitHub's `ubuntu-latest` runners are not cgroup-limited, so CI does
  not exercise a real quota either; the coverage is a fake cgroup filesystem the tests
  build and a monkeypatched mount root.
- **One odd byte from `nvidia-smi` cost the driver reading, and on Linux crashed
  `doctor` outright.** `detect_driver_cuda_version()` ran `nvidia-smi` with
  `text=True` and no `encoding=`, which decodes with the platform's locale codec and
  `errors=None` — strict. That output is a box-art table containing the GPU's marketing
  name, so an undecodable byte is not exotic, and the failure was platform-specific and
  bad in both directions. On POSIX the decode happens on the calling thread at the end
  of `Popen._communicate`, so the `UnicodeDecodeError` propagates out of
  `subprocess.run`, past the `except (OSError, subprocess.SubprocessError)` handler —
  a `UnicodeDecodeError` is a `ValueError` — and takes `trainai doctor` and
  `trainai setup` down with a traceback. On Windows the decode happens in a daemon
  reader thread, so the threading machinery prints the traceback straight at the user,
  the buffer stays empty, and `result.stdout` comes back as `None`; the reading is lost
  and TrainAI concludes it cannot determine a CUDA version, which is the same wrong
  answer as having no driver at all. Both call sites now state
  `encoding="utf-8", errors="replace"`. `replace` is the right trade where the only
  thing read out of the output is a version number matched by regex: refusing to report
  the driver over one strange byte elsewhere in the table would be worse. The macOS CPU
  name probe (`sysctl -n machdep.cpu.brand_string`) had the same defect and the same
  fix. Two new tests exercise the real decode through a generated stand-in `nvidia-smi`
  executable rather than a patched `subprocess.run`, since patching the call would test
  the mock's codec instead of the product's.

  Repository-wide, a `subprocess` call that both captures output and asks for `str` must
  now name its encoding, enforced the same way as the `read_text` rule below. Both
  halves are required and both exclude a real call here: `trainai setup` runs pip with
  inherited stdio on purpose so pip's progress stays live, and a call capturing bytes
  reaches no codec. The walk also matches
  `functools.partial(subprocess.run, ...)`, whose call node is `functools.partial` —
  there is one such site in the suite, so a narrower rule would ship with a hole the
  repository already contained.

- **One unreadable GPU in `/sys/class/drm` hid every other GPU on the machine.** The
  Linux vendor probe read each card's PCI id with `Path.read_text()` and no encoding, so
  the bytes were decoded with whatever the locale happened to be, inside a handler
  written as `except OSError: continue`. A `UnicodeDecodeError` is a `ValueError`, not an
  `OSError`, so it escaped the per-card handler to the function-wide one and ended the
  loop — every card enumerated after the bad one was lost, and `trainai doctor` then
  reported no GPU vendor at all. That is the one diagnostic that matters most to a new
  user: "you have an NVIDIA card and the CPU-only PyTorch wheel" is fixable, and "no GPU
  found" reads as a hardware fault. The read now states `encoding="ascii"` — the file
  holds one hex id — and the handler catches both, so one bad sysfs node costs one card.
  The card walk had no tests at all, because the sysfs path was a literal and no real
  machine can be asked for a malformed card on demand; the path is now a module constant
  and five tests cover a good card, an undecodable one ahead of a good one, a card with
  no vendor file, an unknown id, and no `drm` directory.

  Repository-wide, `read_text` without an encoding is now a lint: `test_conventions.py`
  parses every call and requires the keyword, which is the read-side twin of the existing
  `newline=` rule for writes. Text-mode `open` reads are deliberately out of scope, and
  that boundary is measured rather than assumed — all 27 mode-less `open` calls in the
  tree are `InferenceSession.open(...)` or `ZipFile.open(member)`, so a rule covering
  them would be 27 false positives and no findings.

- **`trainai data --help` advertised a `validate` command that does not exist.** The
  group described itself as "Prepare, inspect and validate text datasets", and
  `trainai data validate` exits with "No such command". Validation is real — both
  `prepare` and `inspect` run `validate_corpus` and report what it finds — so the help
  now says "Prepare and inspect text datasets, reporting problems", which describes the
  same capability without naming a command for it.

- **The wheel promised type information it did not ship, and the sdist shipped a test
  suite it could not run.** Three defects in the distribution metadata, none of which
  any test could see, because nothing built the artifacts and looked inside them.

  - `pyproject.toml` declared the `Typing :: Typed` classifier and there was no
    `src/trainai/py.typed`. PEP 561 makes that marker file, not the classifier, what a
    type checker looks for, so mypy and pyright treated every `trainai` import as
    `Any` — silently, which is the whole problem: the annotations are all present and
    the classifier says to expect them, so a downstream user got no types and no
    explanation. Added, and verified in the built wheel as `trainai/py.typed`.
  - The sdist included `tests` but neither `docs` nor `examples`, which that suite
    reads: `test_conventions.py` checks the format documents against the code and lints
    the example scripts. Measured by unpacking the built sdist and running it —
    **6 failed, 123 passed, 49 skipped**, the six on `FileNotFoundError` for
    `docs/export-format.md` and friends. A packager who runs the shipped suite gets a
    failure that says nothing about TrainAI. With both directories included the same
    run is **131 passed, 49 skipped**. The example corpora are gitignored and hatchling
    honours that, so the sdist stays at 595 KB.
  - The version was written in both `pyproject.toml` and `src/trainai/__init__.py` with
    nothing comparing them. Both have to exist — the build backend cannot import the
    package, and `trainai --version` cannot read the metadata — so the duplication is
    not the bug; the absence of a check is. It is edited by hand at release time, once,
    and it is the field every issue report quotes.

  All three are now asserted in `tests/test_conventions.py`, and the marker test is
  written as an equivalence, so removing the classifier while leaving the file also
  fails.

- **The CLI could not start on a current Typer, and every usage error printed a
  traceback instead of exiting 2.** Typer 0.27 vendored click into `typer._click` and
  dropped `click` from its own dependencies. `trainai.cli.main` imported the standalone
  package at module level, and caught `click.exceptions.Abort`, `click.exceptions.Exit`
  and `click.ClickException` — none of which are the classes Typer raises any more.
  Measured on typer 0.27.2: a bad option arrives as
  `typer._click.exceptions.NoSuchOption`, and `isinstance(exc, click.ClickException)` is
  `False`, so the handler never fired. Where `click` was not installed at all — which is
  the ordinary case once Typer stopped requiring it — `import click` ended the process
  before any command ran.

  This was invisible on the development machine, whose typer 0.24.1 still uses the
  standalone package, and it is the first thing CI found on its first execution: **60
  tests failed on all eight matrix cells**, lint and build green.

  - The classes are resolved once, in a new `trainai.cli._click`, from whichever click
    Typer is actually using — the vendored module first, because when both are
    importable it is the vendored classes that get raised.
  - Both layouts are supported rather than pinning `typer<0.27`, which would put this
    project in conflict with anything else in a user's environment that wants a newer
    Typer.
  - `isinstance(command, click.Group)` — the idiom the test suite navigates the CLI
    with — has no equivalent on the new layout, where `TyperGroup` derives straight from
    `Command` and there is no `Group` type at all. Replaced by `is_group()`, which asks
    whether the command holds a `commands` mapping.
  - A test now asserts that no module under `src/trainai/` imports the standalone click,
    and that `click` is absent from the declared dependencies, so the import cannot
    return by working on one contributor's machine.
  - The regression tests name no version: they ask Typer to raise a usage error and
    check that it is the class the shim resolved, which keeps holding through the next
    reshuffle.

- **A plan file that was not UTF-8 ended `--plan` with a traceback, and one saved by
  Notepad was refused as invalid JSON.** `_configs_from_plan` states its own contract in
  its docstring — "every way this can fail becomes a `UsageError` naming the plan file" —
  and the encoding path was outside it, because `UnicodeDecodeError` is a `ValueError`
  and neither the `FileNotFoundError` nor the `json.JSONDecodeError` guard could see it.
  Measured on ten damaged plan files: a stray non-UTF-8 byte and a UTF-16 file each
  **exited 1 with a bare traceback** instead of the documented exit 2.

  - The bytes are now read and decoded before they are parsed, so an encoding problem
    reads as one: `plan.json is not UTF-8 text`, at exit 2, naming the file.
  - A UTF-16 BOM is named in the hint, both byte orders. Nobody chooses UTF-16 on
    purpose — it is one entry in Notepad's Save As dropdown, and the file looks perfectly
    normal in the editor that wrote it, so the refusal says which entry to change.
  - A **UTF-8 BOM is now accepted** rather than refused. It is Notepad's *default* save,
    this file is documented as one to open and argue with, and it was reported as "is not
    valid JSON" — which sends someone hunting for a syntax error in a file that has none.
    `utf-8-sig` is a no-op on a file without a BOM.
  - `OSError` on the read — a permission problem, most plainly — is also a refusal now
    rather than a traceback; only `FileNotFoundError` was caught before.

  A UTF-16 file with **no BOM at all** still reads as bad JSON rather than bad encoding,
  and that is as good as it gets: ASCII text interleaved with NUL bytes is valid UTF-8,
  so nothing is left to identify it by.

- **A damaged `metrics.jsonl` ended a *finished* run with a traceback, and summarised
  a loss that was not a number.** `MetricsWriter.read_all` runs at the very end of
  `Trainer.train`, after the checkpoints are on disk, so anything it raised was a
  traceback over a log thrown at someone whose model was already saved. Measured on a
  three-record file, one edit per case: of fifteen ways it can be damaged, **nine
  ended the run with a bare `AttributeError`, `TypeError` or `UnicodeDecodeError` and
  exit 1**, and two more were summarised wrong in silence.

  - A line holding `3`, `"train"`, `[1,2]`, `null` or `true` was returned from a method
    annotated `list[dict[str, Any]]`, and `summarise_run` then called `.get` on it.
  - A stray byte anywhere in the file raised `UnicodeDecodeError` from the whole-file
    `read_text`. It is a `ValueError`, so the `except json.JSONDecodeError` never saw
    it. Lines are now decoded individually, which confines the damage to its own line;
    `errors="replace"` was rejected because it substitutes characters *inside* a value
    that then still parses, turning a missing line into a wrong number.
  - A `"val_loss": null` or `"val_loss": "3.0"` reached `min` and raised `TypeError:
    '<' not supported between instances of 'NoneType' and 'float'`. A `"loss": "nan"`
    did not raise at all — it was reported as the final loss.
  - `bool` is an `int` subclass, so `"elapsed": true` was a one-second run and
    `"tokens": true` a one-token one.

  Nothing here refuses. This file is the most derived thing a run writes: it describes
  what already happened and no model depends on it, which is the same reasoning
  `checkpoints.json` gets and the opposite of `manifest.json`. Dropped lines are
  counted and reported as `unreadable_metric_lines` in the run summary — a summary
  computed from fewer records than the file has lines is a different summary, and
  nothing else would say so.

  The damage is reachable without a hand edit: the writer appends, so a run killed
  mid-write leaves a fragment with no newline and the resumed run glues its first
  record onto it, mid-file rather than last.

- **A hand-edited `checkpoints.json` ended a training run with a traceback, and turned
  `--which best` into `--which latest` without saying so.** The pointer file beside a
  run's checkpoints has three readers, each with its own guard, and none of them agreed.
  Measured on a run directory with three checkpoints, one edit per case, eighteen ways
  the file can be damaged:

  - **Pruning raised on ten and the pointer update on five**, as a bare `TypeError` or
    `AttributeError` — a document that is not an object reached `raw.get`, and a `best`
    holding a string reached `"step-1.pt".get("file")`. Both of those run *after* the
    checkpoint is safely on disk, so a damaged pointer ended the run with a traceback
    over a file the next line was going to overwrite. `UnicodeDecodeError` is a
    `ValueError`, so the `except OSError` on all three readers never saw a pointer an
    editor had re-saved as Latin-1; that one raised in all three.
  - **`--which best` fell back in silence.** The filenames record the order checkpoints
    were written, not their validation loss, so a damaged pointer answered "the best
    checkpoint" with the *last* one — a different checkpoint. `trainai chat` and
    `trainai eval` now carry a note saying which file was loaded and why. Someone
    comparing two runs while that went unsaid would conclude their best model is worse
    than it is.
  - **`--keep-checkpoints` could delete the checkpoint just written.** "Never the
    newest" was true only because the pointer named it; a pointer whose `best` was
    readable and whose `latest` was not left the new file unprotected, and `keep=1`
    deleted it. It is now protected by its filename, which is what the promise says.

  All three readers now share one that discards what it cannot use and never raises.
  Degrading rather than refusing is the opposite of the choice `manifest.json` gets one
  commit earlier, deliberately: this file is *derived* — `latest` is the highest-numbered
  `step-*.pt` on disk — so nothing in it is unrecoverable, and refusing would turn a
  damaged convenience file into a run that cannot resume while the checkpoints sit intact
  beside it. A key this version does not recognise is passed through, because the file is
  rewritten in place and dropping one is how a downgrade loses what a later version
  recorded.

  Also corrected in `docs/checkpoint-format.md`: the compatibility promise said a
  checkpoint from a newer TrainAI is "refused with an instruction to upgrade", which the
  previous release had already made false — the refusal says to use the TrainAI that
  wrote it, because unlike a dataset a checkpoint cannot be re-created.

- **A damaged `manifest.json` came out as a Python traceback and exit 1, or was read
  wrong in silence.** `DatasetManifest.load` checked exactly one thing — the `format`
  string — and then read every other key with `raw["dtype"]`, `int(...)` and
  `.get(..., default)`. Measured on a copy of a real dataset, one edit per case, twenty-
  seven ways a manifest can disagree with its own format:

  - **Twenty arrived as a traceback**, across five exception types. A manifest holding a
    JSON array reached `raw.get` and raised `AttributeError`; one missing `dtype` raised
    `KeyError`; `"vocab_size": "many"` raised `ValueError`; `"format_version": null`
    raised `TypeError`; and a manifest an editor had re-saved as Latin-1 raised
    `UnicodeDecodeError` — which is a `ValueError`, so the existing
    `except (OSError, json.JSONDecodeError)` never saw it. Exit 1 with
    `trainai/data/binarize.py` in the frame list, from a tool that documents exit **3**
    for a dataset it cannot read.
  - **Seven loaded silently**, which is the worse half. A manifest missing its `val`
    entry, or missing `shards` inside one, loaded as a dataset with no validation split
    and reported nothing. `"vocab_size": true` became `1`, because `bool` is an `int`
    subclass and `int(True)` is not an error. A shard whose `name` or `sha256` was an
    array became the string `"[]"` — a filename that does not exist and a checksum that
    matches nothing.

  Every one is now a `DatasetFormatError`, exit 3, naming the file *and the key*: `is
  missing splits.val.`, `has vocab_size as "many", which is not an integer`, `has
  splits.train.shards[0].tokens as [], which is not an integer`. Naming the key is the
  point — "the manifest is corrupt" for twenty-seven different edits leaves a reader no
  better off than the traceback did.

  The rule applied is the one `serialise.py` already states for a checkpoint's
  configuration blocks: a key the reader uses is required, and must hold the right type.
  It is applied here for the same measured reason. `dtype` decides how the shard bytes
  are interpreted, so `<u2` read as `<u4` is not an error but half as many tokens, each a
  different one. `tokenizer_fingerprint` and `content_hash` default to `""`, and an empty
  one *skips* the tokenizer-match and dataset-identity comparisons in `eval` rather than
  failing them — so a defaulted key there silently disables a check.

  Two boundaries are pinned by tests rather than left to the next reader's judgement: an
  integer **is** accepted where a float belongs, because `"val_fraction": 0` is what a
  person types where the writer puts `0.0`; and `true` is **not** accepted where a count
  belongs, because JSON spells booleans. A negative control asserts that an unedited
  manifest still round-trips with the same `content_hash`, so a check accidentally
  inverted fails a test rather than refusing every dataset on disk.

  `docs/dataset-format.md` gains the table of what is refused and what each defaulted key
  would have cost. Two claims in that file saying a manifest from a newer TrainAI is
  refused "with an upgrade instruction" are also corrected — the hint stopped saying that
  in the previous entry but the document still did.

- **A checkpoint holding a weight at the wrong shape, or an entry that was not a tensor at
  all, came out as a Python traceback and exit 1 instead of a refusal.**
  `Checkpoint.apply_to` called `load_state_dict(strict=False)` and read the missing and
  unexpected keys it returns. But `strict=False` only suppresses *those* two: a size
  mismatch, or a value torch cannot copy, it **raises**, and nothing caught it. Measured
  through the installed CLI on a real 600-step checkpoint with one weight replaced:

  ```
  RuntimeError: Error(s) in loading state_dict for GPT:
          size mismatch for blocks.0.attn.q_proj.weight: copying a param with
          shape torch.Size([3, 5]) from checkpoint, the shape in current model is
          torch.Size([256, 256]).
  ```

  rendered with the internals of `torch.nn.Module` in the frame list, in a tool whose
  contract is that a bad input file exits **5** with no traceback. A string or a list in
  place of a weight did the same. All four ways a checkpoint's weights can disagree with
  the model its own `model` section describes are now one refusal, and `details` names the
  weights rather than only reporting that something was wrong: `missing`, `unexpected`,
  `wrong_type`, and `wrong_shape` — the last carrying both shapes.

  A weight at a different **dtype** is deliberately still accepted: torch casts it, and a
  float64 copy of a real checkpoint loads and generates (measured), so refusing it would
  break a file that works. A test pins that boundary so the next person does not widen the
  check on the assumption that stricter is safer.

  The refusal also **names the file** now, in the message and in `details["path"]`, like
  every other `CheckpointCorruptError` in the module and unlike this one. And a tied
  checkpoint's `lm_head.weight` is no longer listed among the missing weights: the format
  omits it on purpose, so listing it sent someone looking for damage in the one place the
  format guarantees is empty.

  The earlier concern that this branch *blamed the weights for a config problem* is
  obsolete and was measured to be so rather than assumed: `apply_to` refuses a config
  mismatch before it looks at any tensor, so a wrong-shape report can no longer be caused
  by a rebuilt-from-defaults `model` block.

  Two branches had no test at all before this — a weight the model does not have, and the
  tied-alias exclusion — which is how the second went unnoticed. Fifteen single-line
  mutations were applied one at a time; all fifteen are caught, with three
  semantically-identical rewrites surviving as controls, and one boundary case recorded:
  whether a stray non-tensor is reported as `wrong_type` as well as `unexpected` is not
  something any check cares about. `docs/checkpoint-format.md` now states the four
  refusals and the dtype exception.

- **Four messages told the user that a newer TrainAI was the remedy, where upgrading is
  neither possible nor the fix.** Two of them said `pip install --upgrade trainai`, which
  README.md itself describes as not existing — "Not published yet, so there is no
  `pip install trainai` and no clone URL to give you." Both were the *only* advice given
  for an artifact the build could not read, so following them was the user's whole path
  forward, and it went nowhere. The other two said newer versions behave in a way **this
  build already behaves**, which sends someone hunting for a version that is already in
  their hands.

  - A dataset written by a newer format version now says to use the TrainAI that wrote it
    **or re-create the dataset from the corpus with `trainai data prepare`** — which
    always works, because a dataset is derived from a corpus the user still has.
  - A checkpoint written by a newer version now says to use the TrainAI that wrote it, and
    says plainly that there is no converter and that, unlike a dataset, a checkpoint
    cannot be re-created — it is the output of the training run that produced it. The
    asymmetry is the point: identical advice for the two was wrong for at least one.
  - Opening a dataset with no `train` split said the split is one "newer versions refuse
    to write". The accurate explanation was already sitting in a code comment three lines
    above: the dataset was prepared with a `--val-fraction` high enough to send every
    document to validation, which this build refuses. That branch had no test at all,
    which is how the hint and the comment three lines apart drifted.
  - `trainai chat` on a run with no `tokenizer.json` said "Runs trained by a newer version
    keep their own copy; this one does not, and the dataset it names is gone or moved."
    Measured against a run trained a minute earlier by this build, against a dataset still
    on disk: **both halves false.** `trainai train` copies the dataset's tokenizer in, so a
    run lacks one only when the dataset had none beside it either. The message now names
    which of four states it is, reported in `details["reason"]`: no dataset recorded, the
    dataset exists but has no tokenizer, the recorded root is relative and not resolvable
    from here, or the dataset is genuinely gone. The relative case is its own branch
    because the recorded root is the path **as typed on the command line** — measured as
    `runs\_tokprobe_data` — so a naive existence check would report a present dataset as
    "gone or moved" whenever `chat` runs from a different directory, which is the same
    false claim in a new costume.

  Four documentation and docstring sites that opened with "a user runs
  `pip install trainai`" were reworded to "installs TrainAI", since the sentence was about
  getting a CPU-only torch wheel, not about that command.

  `tests/test_conventions.py` grew the check this gap earned:
  `test_no_message_names_an_install_that_does_not_exist` scans every source string and
  every document line for `pip install ... trainai`. The existing flag check could not
  catch it — it asks whether a named flag exists on some command, and `--upgrade` was
  exempt as pip's, so nothing ever asked whether the *package* could be installed. The
  check asserts README.md still says TrainAI is unpublished, and fails with instructions to
  delete it deliberately if that changes, rather than quietly outliving the fact it rests
  on. Its two exemptions must each match exactly one place: an exemption shortened to a
  fragment — `"pip install"` would do it — exempts the whole repository while still
  looking used, and that hole was found by mutating the check rather than by reading it.

  Twenty single-line mutations of the four messages and the new check were applied one at a
  time; all twenty were caught, with three semantically-identical rewrites surviving as
  controls. One boundary case survives and is documented: a recorded dataset root that
  exists but is a *file* reports "gone or moved", which no producer can write. Exit codes
  are unchanged — **3** for the dataset, **5** for the checkpoint, **2** for both loader
  and `chat` refusals. Full suite: 1567 passed, 4 skipped.

- **A checkpoint whose `train` block had lost a key resumed on a different learning rate,
  batch size or seed than it recorded, and nothing in the output said so.**
  `TrainConfig.from_dict` was `cls(**{k: v for k, v in raw.items() if k in allowed})`:
  no completeness check and no type check at all. The previous entry fixed exactly this
  for the `model` block and deliberately left this one for its own commit, because the
  failure is quieter. A model rebuilt at the wrong shape eventually fails the weight
  load; a *run* rebuilt from the wrong hyperparameters simply continues. Measured on a
  real 600-step checkpoint:

  - `lr` gone resumed at **0.0003** in place of the recorded 0.000424.
  - `batch_size` gone resumed at **8** in place of 32, and `grad_accum` at **1** in place
    of 2 — together a sixth of the effective batch the learning rate was chosen for.
  - `seed` gone resumed at **1234** in place of 99, which reorders every batch. That
    breaks the guarantee this module's own docstring makes — *a continued run is bitwise
    identical to an uninterrupted one* — silently, which is the reason this is a defect
    and not a message-quality complaint.
  - `schedule` gone resumed on **cosine** in place of `linear`, and `min_lr_ratio` on a
    floor twice as high as the file's.

  Wrong types were not caught either. `steps: 100.5` was accepted outright, making
  `total_tokens` a float; `grad_accum: true` was accepted as **1**, because `bool` is an
  `int` subclass; `seed: "abc"` and `device: 5` reached the run; and `lr: "fast"` surfaced
  as `'<=' not supported between instances of 'str' and 'int'`, a Python operator error
  naming no field. All of these now read like `The training configuration's lr is "fast",
  which is not a number`, and exit **5** naming the file when the file is a checkpoint.

  The checker itself moved to a new stdlib-only `trainai/serialise.py` and both
  dataclasses use it, rather than the machinery being copy-pasted into a second module —
  which would have been the same "parallel copy rots" mistake the previous entry argued
  against. `ModelConfig.from_dict` switching onto the shared code is its own regression
  guard: all 71 of its tests passed the switch with no edit, exact message text included.
  Two of them are strengthened further down this entry, for the unrelated reason described
  at the end.

  `schedule` and `precision` are required but not type-checked, on purpose. Their values
  are already refused by `__post_init__` with a message that lists the choices —
  `Unknown --schedule cosinus. Use one of: cosine, linear, constant.` — which is more use
  than *is not a string*, so checking the JSON type first would replace a better message
  with a worse one. `tests/test_conventions.py` now proves that exemption rather than
  asserting it: it checks each `Literal` field really is refused by name with a hint, so
  the day `__post_init__` stops doing that, the field does not become the only one that
  is required and never validated.

  Verified through the installed CLI, not only in-process: five doctored checkpoints exit
  **5** naming the file and the key, three doctored plans exit **2** with the
  `trainai plan` hint, an untouched checkpoint resumes and trains to completion, and a
  complete plan still trains. Twenty-six single-line mutations of the shared checker are
  each caught by a test, with three semantically-identical rewrites surviving. The
  strictness itself required no change to any existing test, which is the evidence that no
  file TrainAI writes was ever incomplete: all three producers write `to_dict()`.

  The mutation gate found one hole in the *previous* commit's tests while it was at it.
  Nothing pinned which block a refusal names — renaming the model block's subject to
  "configuration" passed the whole suite. That only became load-bearing when the two
  dataclasses started sharing the code, since a plan file holds one of each and "the
  configuration's `n_head`" sends an editor looking in two places. Both sides now assert
  the block by name.

- **A checkpoint whose `model` block had lost a key was rebuilt from defaults, silently,
  and for some keys the run then trained and generated noise with nothing reported.**
  `ModelConfig.from_dict` computed the set of missing fields and then discarded it for
  every field but `vocab_size`, so anything else absent fell through to the dataclass
  default. Measured on a real 4-layer checkpoint rather than reasoned about:

  - `n_layer` gone rebuilt an **8-layer** model, and the only complaint came later from
    the strict weight load, as `CheckpointCorruptError: The checkpoint's weights do not
    match the model` — accusing the weights, the one part of the file still correct.
  - `rope_theta` or `norm_eps` gone changed **no tensor shape**, so every weight loaded,
    no layer objected, and the model produced noise because the rotary base it was
    rebuilt with was not the base it was trained with. That is the case that made this
    a serious defect rather than a message-quality one.

  Three further holes in the same function, all now closed:

  - `tie_embeddings: "no"` was accepted as **true**, because a non-empty string is
    truthy — a configuration stating the opposite of the file it was read from. A `bool`
    field also no longer accepts `1`: JSON spells booleans, and this file is JSON.
  - `seq_len: 64.5` was accepted and failed hundreds of lines later inside a slice.
  - `"vocab_size" not in raw` was a *substring* test when `raw` was a string, so a
    string containing those characters passed the completeness guard and reached
    `raw.items()` with an `AttributeError`.

  What replaced it derives everything from the dataclass annotations. A field is
  required exactly when its annotation does not admit `None`, which reproduces the
  hand-maintained `- {"n_kv_head", "d_ff"}` exemption without maintaining it, and a
  field added later is checked without anyone remembering. All missing fields are named
  at once (`missing n_layer, rope_theta and seq_len`) rather than one per attempt, which
  turns repairing a file into a single edit. Unknown keys are still ignored — the
  forward-compatibility promise in `docs/plan-format.md` is unchanged, and the two
  halves are now stated together rather than one at a time.

  This also closes the rough edge the previous entry left open by name: a field of the
  wrong type reported the raw operator text `'<=' not supported between instances of
  'str' and 'int'`, naming no field. It now reads `The model configuration's n_head is
  "four", which is not an integer`, quoting the value in the file's own syntax so the
  reader knows what to edit — with the message naming the type and the hint naming the
  literals, since "set it to a boolean" leaves someone who then types `"true"` no
  better off.

  `load_checkpoint` wraps both deserialisers so this stays a **checkpoint** failure.
  `ConfigError`'s exit code is `USAGE` (2), which is right for a hand-written plan and
  wrong for a file TrainAI wrote itself; letting it escape would have quietly moved this
  case from 5 to 2 and broken any script branching on it. The reason is carried through
  verbatim rather than replaced, because "missing `n_layer`" is the whole value of the
  refusal. Verified through the installed CLI: four doctored checkpoints exit **5**
  naming the file and the key, two doctored plans exit **2** with the `trainai plan`
  hint, and an untouched checkpoint still resumes and trains.

  `_json_type` was a local copy in `cli/train.py`; it is now `json_type_name` in
  `errors.py` alongside a new `json_literal`, which is stdlib-only and already imported
  by both callers, so the model layer does not have to reach through `console.py` and
  pull in rich. `json_literal` degrades to the type name for a value too long to quote
  rather than truncating the JSON, which would print an unterminated string and read as
  a second defect.

  The one deliberate gap is asserted rather than trusted: the annotation table is not
  exhaustive over Python, so a field annotated with something it does not cover would be
  loaded *unchecked*. `tests/test_conventions.py` fails on that the day such a field is
  added. Twenty-one single-line mutations of this logic are each caught by a test, with
  two false-alarm controls surviving.

- **A hand-edited `plan.json` crashed `trainai train` with a Python traceback and
  exit 1, and `--plan` loading had no tests at all.** `docs/plan-format.md` promised
  that "all four ways" a plan can be unusable exit **2** and name the file. Measured
  against the code, that promise was wrong in every part of it. There were eight
  branches, not four. Six of seven malformed-block shapes escaped the loader's
  `except (KeyError, TypeError, ValueError)`: `ConfigError` is not a `ValueError`, so a
  block failing a validator surfaced with `config.py`'s hint to re-create it with
  `trainai train` — which makes *checkpoints*, not plans, so the advice was for the
  wrong artifact and the wrong command. A block that was a string or a list reached
  `raw.items()` and raised `AttributeError`, which nothing caught: a forty-three-line
  traceback and exit 1. A plan whose top level was valid JSON but not an object —
  `[1, 2, 3]`, `"small"`, `7`, `null` — had no `.get`, and crashed the same way one
  level up. And the vocabulary-mismatch branch, the one cross-check the document
  singles out, named neither the file nor carried `path` in its machine-readable
  `details`.

  This matters more than a hand-edited file usually would, because this project
  prints `plan.json` specifically "so that the plan can be disagreed with rather than
  merely obeyed". A file the tool invites you to argue with is ordinary input, and
  editing it should not produce a stack trace.

  The loader now shape-checks the document and both blocks before anything reaches a
  deserialiser, catches the deserialiser's own `ConfigError`, reports an *absent*
  block differently from a `null` one, and names a malformed block by what it actually
  is (`is a boolean, not an object` — with `bool` checked before `int`, since it is a
  subclass and would otherwise read as a number). All eighteen shapes now exit 2, name
  the file, and name the `trainai plan` command that would replace it.

  One rough edge is left, deliberately, rather than claimed fixed: a `model` field of
  the wrong type still reports the raw operator text `'<=' not supported between
  instances of 'str' and 'int'` without naming the offending field. Fixing that means
  type-validating inside `ModelConfig.from_dict`, which is also the checkpoint
  deserialiser, so it is a separate change.

  The document now states the contract as a universal ("every way") rather than a
  count, because a count is what rotted. It is enforced two ways: every branch runs
  end to end, and a structural check reads the loader's AST so that every
  `raise UsageError` in it names the file and carries `path` in `details` — a ninth
  branch added later cannot quietly drop either.

- **Six keys the artifacts carry were missing from the documents that specify them,
  and two of the six were silently inside `content_hash`.** These pages are read as
  the list of what is in a file, so a key that is not in them is one nobody knows to
  preserve, look up, or expect. `plan.json` writes fifteen keys and
  `docs/plan-format.md` described thirteen: the two it left out were `dataset` and
  `preset` — the two that say *what the plan is about*, and what every measurement in
  the file is conditional on. `manifest.json` writes twenty-one and
  `docs/dataset-format.md` described seventeen, missing `tokenizer`, `shard_tokens`,
  `documents` and `splits`. `splits` is the block anything reading a dataset actually
  loads from and the only place the per-shard checksums live; the sample snippet in
  that same document indexes `manifest["splits"]["train"]["shards"]`, so the format
  was being *used* in the page that did not define it.
  `shard_tokens` and `documents` are both inside `content_hash`, which turns an
  undocumented key into an undocumented part of the reproducibility guarantee:
  someone comparing two hashes of the same corpus had no way to learn that
  `--shard-tokens` moves one, and the `content_hash` section's prose said "shard size"
  and "per-split document counts" without naming either key.
  `docs/plan-format.md` gains a `dataset`, `preset` block that also states what those
  two do *not* buy — the path is provenance and not an integrity check, so moving or
  re-preparing that directory leaves a plan looking valid while describing measurements
  of something else. `docs/dataset-format.md` gains rows for the four, including which
  of `tokenizer`'s five sub-fields are the readable copy of a top-level key and which
  two facts live nowhere else.
  A new check reads each writer's dict literal with `ast` — the literal is the format
  definition, where a produced artifact is one sample of it, and building a real plan
  would need a GPU — and requires every top-level key to be named as a key in the
  document. `docs/checkpoint-format.md` already named all twelve of its keys and is
  the control. The key has to be the whole content of a backtick span: a rule that
  accepted an identifier *inside* one would have taken this document's dozen mentions
  of `tokenizer.json` as documenting the `tokenizer` block it never described, and a
  mutation confirms that loosening it lets a real deletion through. Sub-keys are
  deliberately not required, so a specification that delegates `model` to
  `ModelConfig` is not pushed into duplicating a schema it does not own.

- **The export format's parity table described a model that does not exist, at two
  numbers that had moved, and left a third of `--dtype` looking untested.** The model
  beside the measurements was "4.26M parameters, 6 layers, d_model 256" — a
  contradiction: 4.26M is the four-layer count, `L6 d256` would be 5.87M, and the
  checkpoint on disk holds `n_layer: 4`. Both quoted differences were stale (fp32
  3.81e-06, measured 4.05e-06; bf16 6.44e-06, measured 4.29e-06), and `fp16` — which
  `--dtype` accepts — had no row at all, so a reader could not tell whether it had been
  measured and found fine or never measured. It now has one (5.25e-06). The paragraph
  under the table also had the *reason* backwards: it said bf16 is the looser row because
  the comparison "absorbs one round-trip through the reduced dtype", when
  `_reference_model` casts the reference to the export dtype and back to fp32 precisely
  so that rounding is **not** what is being measured — the check is about the mapping,
  and if it compared against the un-rounded checkpoint the reduced-dtype rows would be
  dominated by their own rounding and the tolerance would have to be widened until the
  check could no longer fail for the reason it exists. "Two orders of magnitude inside
  the tolerance" was also an overstatement; the measured ratios are thirty-eight to
  fifty times, and the text now says so and warns that a spread this small on one model
  over 16 positions is not an ordering of the dtypes.
  The shape is now written the way `describe` writes it —
  `L4 d256 h4 ff704 ctx256 vocab4096` — which puts it under the shape and
  parameter-count checks added above at no cost, rather than teaching the checker another
  prose dialect. A new check covers the table's frame: every `--dtype` value has a row,
  no row names a dtype the command would refuse, every measured value is inside the
  tolerance it claims to pass, and every tolerance and position count the document states
  — in the table's own columns and in the prose around it — is the one
  `LOGITS_TOLERANCE` and `_PARITY_TOKENS` really hold. Those prose statements are matched
  against whitespace-flowed text, so re-wrapping a paragraph cannot hide one behind a line
  break; a check that fails on a reflow teaches the next reader that the failure means
  "reflow the prose" rather than "a number is wrong". The measurements themselves are
  machine-specific and are not checked; the document says as much, and the export reports
  its own number wherever `transformers` is installed.

- **The README quoted a `--dry-run` plan that no command produced.** The block presented
  as captured output said `vocab8192`, `5.31M` parameters and `Steps 450`, while the
  command printed directly above it asked for 600 steps against a 4,096-token
  vocabulary — and one of its lines, "bf16 (supported by this GPU, no scaler needed)",
  was text the code no longer prints in any configuration. A reader who ran the command
  got a different plan than the one they had just been shown. The block is now the real
  output of the command beside it, captured at 80 columns.
  Checking it turned up that the README quotes model numbers in **three** places, not
  one: the `trainai plan` panel and its "why not something bigger" note above, the
  `--dry-run` plan, and the results table below. Those two other sites were correct, but
  nothing was keeping them that way. Four tests now do, and they cover the whole of
  `README.md` and `docs/`: every `L4 d256 h4 ff704 ctx256 vocab4096`-style shape has to
  be one `ModelConfig.describe` would write (which pins the derived `ff` against
  `d_model`), a shape attributed to a preset has to be that preset's shape and has to
  match the other places the same document names it, every parameter count quoted beside
  a shape is recomputed from that shape, and the `--dry-run` block is compared against
  `_model_rows` — the renderer itself, so a change to the preset table, the parameter
  formula or the count formatting fails here rather than in a reader's terminal. The
  vocabulary comes from a corpus the repository does not carry, so it is read out of the
  block and pinned against that block's own `Vocabulary` row, which is what stops it
  drifting with every number derived from it. Machine-dependent rows — device,
  precision, throughput, VRAM — are deliberately not checked, since no other machine can
  reproduce them.

- **The documented size of the test suite was five hundred tests out of date.** The
  README said "917 tests and runs in about 40 seconds" and
  [docs/hardware-support.md](docs/hardware-support.md) said "917 tests" in the section
  that argues the project works with no GPU at all. Measured on the machine both numbers
  came from, `pytest -m "not gpu"` collects 1,424 and reports 1,420 passed, 4 skipped,
  14 deselected in 55s — and 95s on a cold cache, which the old "40 seconds" would have
  made look like a problem. The README also claimed the suite "enforces two structural
  rules" while it enforces five. Both documents now quote a floor rather than an exact
  count, and a test reads that floor back out of them and fails from either side: if the
  suite no longer clears it, or if the real count has run more than 250 ahead of it. It
  counts `session.items` from the run it is part of, so it costs nothing and cannot
  disagree with what actually ran, and it stands down on a partial run because running
  one file must not fail. Verified by mutating each document in turn: a floor above the
  real count, a floor far below it, and the wording removed from one of the two files are
  all caught, 3/3, against a green control.

- **The empty-split advice named a `--val-fraction` that did not work on the corpus in
  front of it.** When no document was selected for validation, `data prepare` suggested a
  higher fraction chosen from the document count alone, with an honest note about the
  chance it left. On a 3-document text file it named `0.49` and quoted "about 13%", and
  `--val-fraction 0.49` emptied the split again — two refusals for one mistake, the second
  admitting "this run is one of those". Two earlier formulas failed the same way for other
  reasons: one named values the range check then refused, and named `0.02` to a user who
  had passed `0.05` while calling it a raise.

  No probability was ever needed. Each document is assigned by a keyed hash of its text,
  so the run that just failed already holds the answer: the smallest fraction it produced.
  Any value above that selects that document — certainly, for this corpus at this seed —
  and the hint now names it and says so. Measured through the CLI on a 4-document corpus:
  exit 3 naming `--val-fraction 0.33`, then `--val-fraction 0.33` writes 3 train and 1
  validation document at exit 0. The mirror refusal, an empty *training* split, gained the
  same treatment: at `--val-fraction 0.49 --seed 3` it names `0.26`, and `0.26` writes 1
  train and 2 validation documents.

  When the nearest document sits above the ceiling `--val-fraction` accepts, no legal
  value works, and the hint says that rather than naming one. The chance of an empty
  split is still quoted where it is the right tool — for reshuffling with a fresh `--seed`
  — and only there: conditioned on the failure in hand it understates the risk of raising,
  by 72% against 26% at two documents raising from 0.40 to 0.49.

- **The "learned no merges" advice could fail the same way twice.** When tokenizer
  training kept no merges and `--min-frequency` was above 1, the hint said to use
  `--min-frequency 1`. On a corpus that has no two adjacent characters to merge at *any*
  frequency, following that lands on the same error with a different hint. Measured
  through `trainai data prepare` on 2,000 one-character documents: exit 3, then exit 3
  again, the second time as a surprise.

  Telling the two cases apart would mean a second pass over a document stream the API
  promises to consume once, and both reach this branch — so the advice now names the
  threshold to try *and* what it means if trying does not help, in one message. On a
  corpus where the threshold really is the whole story it still works: `data prepare
  --min-frequency 99999999` on the 1.06 MiB corpus prints the hint, and following it with
  `--min-frequency 1` writes 317,284 tokens and exits 0.

  A comment above it cited "every target from 4096 down to 258 gave this same error" on
  that corpus without the `--min-frequency 99999999` that made it true. Measured at the
  default threshold, every one of those targets trains, 258 included. The condition is
  back in the comment; the conclusion it supports — that `--vocab-size` is never the
  cause, so never in the advice — is unchanged and now testable.

  The floor refusal's own advice, "Use --vocab-size 258 or more", is followed literally by
  a new test: 258 trains and yields exactly 258 tokens, and the number named has to be the
  floor the refusal reports, so neither a value below it nor a needlessly large one passes.

- **Three corpus findings advised flags the command printing them does not have.**
  `validate.py` runs under both `trainai data inspect` and `trainai data prepare`, which
  accept different flags, and three of its findings named one that only `prepare` has — or
  that neither has at all.

  `likely_binary` said to "point `--path` at a directory containing only the text". Neither
  command that reaches this check takes `--path`; the corpus is a positional argument on
  both, and both exit 2 with `No such option: --path`. `trainai doctor` does have one, so a
  user who went looking for the flag found it and got a hardware report. The advice now
  names no flag, because there is no flag to name.

  `looks_like_a_table` ended with "To train on it exactly as it is, pass
  `--allow-tabular`". `data inspect` raises that error, and `data inspect --allow-tabular`
  exits 2. It now says which command takes the flag.

  The `mixed_scripts` note said "If the tokenizer report shows fewer than three characters
  per token, raise `--vocab-size`" — from a command that builds no tokenizer, prints no
  such report, and has no such flag. Both are now attributed to `trainai data prepare`,
  which does print a measured characters-per-token figure.

  Measured through the CLI on three corpora: one with a non-text file in it, a weather
  table renamed to `.txt`, and text that is 41% Latin, 32% Cyrillic and 27% Greek. Each
  fixed hint was then followed and worked — a directory holding only the clean file
  inspects cleanly, `data prepare --allow-tabular` writes the dataset and records the
  override in the manifest, and `data prepare` reports 4.89 chars per token.

  `vocab_too_large_for_corpus` names `--vocab-size` too and was left alone: it is gated on
  a vocabulary having been built, which only `prepare` does, so `data inspect` cannot print
  it. Confirmed by inspecting a corpus small enough to trip it.

  A new test drives `validate_corpus` the way each command really calls it and checks every
  flag each finding names against that command's own click parameters, permitting a flag
  the command lacks only where the finding names the command that has it. It covers all 27
  findings the module can produce, with that list read from the module's syntax tree so a
  finding added later cannot slip past uncovered.

- **The vocabulary-mismatch refusal advised a flag that does not exist.** When a model's
  vocabulary was smaller than its dataset's, the error said to build the model with
  `--vocab-size N`. Following that exits 2: `No such option: --vocab-size  Did you mean
  --batch-size?`. The flag is real on `trainai data prepare`, so grepping the repo for it
  found a hit and reported nothing wrong; only running the advice showed it.

  Nor should the flag exist on `train`. That command reads the number out of the dataset
  manifest itself, and the docstring of the function that does it already says `vocab_size`
  is never a flag — so the hint's own second sentence, "the dataset's tokenizer decides
  this, not you", contradicted its first. The guard is also unreachable from the CLI: the
  plan check compares the two vocabularies before the trainer is built and sends the user
  to `trainai plan`. The audience is therefore a library caller, and the lever to name is
  the keyword argument. The hint now names that, the number to pass, and where the number
  comes from.

  The test covering this asserted `"--vocab-size" in hint`, which pinned the wrong advice
  in place. It now asserts the number is named and that no flag is offered at all, and a
  second test checks every refusal `check_configs_agree` can raise against `trainai
  train`'s own parameter list — read from the command, not from the repo's text — so the
  next guard added there is held to the same rule.

- **The `--val-fraction` the empty-split error suggested was not always a raise.** The
  number came from `max(1.0 / documents, 0.02)`, which reads the document count and never
  the fraction in use, so past about twenty-six documents it named a value at or below what
  the user had already passed — under the word *raising*. Measured end to end: 60 documents
  at `--val-fraction 0.05`, seed 56, was advised to raise it to 0.02, which takes the
  chance of an empty validation split from 5% **up** to 30%. At 20 documents it named 0.05,
  the default already in use, so following the advice changed nothing on a split that is
  deterministic.

  Even where it was a raise, it aimed at one expected validation document — a coin flip by
  construction. Over 2,000 seeds that left the split empty for 29% to 37% of seeds at every
  size from 3 to 60 documents. The old hint hedged this as "likely rather than certain",
  which is true and still not a number worth following.

  The suggestion now inverts the real distribution. `_assign_split` sends each document to
  validation independently, so the chance that none of `n` goes there is exactly
  `(1 - f) ** n`; solving it for a 5% target gives the lowest fraction that meets the
  target, rounded up to two places so rounding cannot quietly miss it, and clamped to 0.49
  so the range check at the top cannot refuse the tool's own advice. The formula was
  checked against `_assign_split` itself over 2,000 seeds at ten corpus sizes and agreed
  with the measured rate to within a point everywhere.

  Where no raise exists — the fraction already meets the target, or it is at the ceiling —
  the error now says so instead of inventing one: at 60 documents and `--val-fraction 0.05`
  it reports that the fraction leaves the split empty about 5% of the time, that this run
  is one of those, and that `--seed` reshuffles at the same odds. Small corpora keep an
  honest ceiling: three documents at 0.49 are told 13%, not offered a fraction that does
  not exist. Stated chances are words below one percent, so a hint can never report "about
  a 0% chance" of the thing that just happened.

- **The empty-split errors advised lowering `--max-doc-chars` on corpora it cannot cut.**
  Both refusals — an empty validation split and an empty train split, four hints between
  their branches — ended by suggesting the corpus be split into more documents that way.
  Only two readers consult that option: a text file is cut as it decodes, and a `.docx`
  after it is extracted. For JSON Lines, JSON, CSV and SQLite, one document is one record,
  and no limit produces more records. Measured on a one-row CSV: `--max-doc-chars 2000`
  came back with the same refusal reporting the same *1 documents*, so the tool had spent
  the user's next attempt for them and left them where they started.

  The advice is now chosen from the source kinds the manifest already records. A
  record-oriented corpus is told to add records and told what a record is in its format
  ("one row is one document here"); a corpus with any text or `.docx` source keeps the
  flag, because there it works; a library caller that passes no sources gets both halves
  with the flag conditioned on where it applies. The flag is still named in the
  record-oriented case — as the thing that *cannot* help, which is what a user reaching
  for it needs to hear.

  `trainai.data.ingest.CUT_BY_MAX_DOC_CHARS` is the set the decision reads, and a test
  parametrized over `Kind` itself builds one corpus per format, lowers the limit and
  counts, so a new reader cannot inherit the wrong answer silently.

- **`eval` reported an unverifiable dataset as a verified match.** The check that catches
  a resplit — the dangerous case, where the same corpus prepared with a different `--seed`
  puts training documents into the `val` split and the perplexity comes out flatteringly
  low — compares content hashes, and needs one on both sides. When either was missing it
  returned "same dataset". Measured on a copy of a real dataset with only its
  `content_hash` stripped: the output was indistinguishable from a verified match, path
  printed bare, `"dataset_matches_run": true` in the JSON, the resplit warning absent, and
  the recorded-loss comparison given its green agreement tick.

  The comment justifying it cited "runs trained before the checkpoint recorded one". No
  such run has ever existed: the manifest has written a `content_hash` since the commit
  that introduced manifests, the checkpoint has recorded one since the commit that
  introduced checkpoints, and the dataset `format_version` has never left 1. What the
  branch actually covered was a hand-edited or truncated manifest.

  The check now has three answers instead of two. `EvalReport` gained
  `dataset_check` — `"match"`, `"mismatch"` or `"unknown"` — carried in the JSON beside
  `dataset_matches_run`, which now means a *verified* match and is therefore `false` for
  `"unknown"`. The report prints `(cannot tell: a content hash is missing)` on the dataset
  row, a note saying which claim could not be made and what to do about it, and still
  shows the recorded-loss comparison — that number is probably the right one and dropping
  it would be worse — but never in green. `"unknown"` is deliberately not reported as a
  mismatch either: that would be as unfounded as the match it replaced.

- **Two comments described the code wrongly, in the direction that invites breaking it.**
  `SPLIT_CHOICES` in `eval` was annotated "in report order", but the tuple is
  `("val", "train", "both")` and `--split both` reports train first (verified: the split
  headings land on output lines 10 and 17, train above val). Since `--split` is a plain
  string option rather than a typer choice, that order reaches the user only through the
  refusal message and the machine-readable `choices` list — so anyone trusting the comment
  would have "fixed" the working half. It now says which order is which, and
  `_resolve_splits` says not to reorder itself to match.

  The report order was also only implied by a test asserting both headings appeared
  somewhere in the output, which passes with them reversed; it now asserts train precedes
  val, and a mutation reversing the tuple fails on exactly that line.

  Separately, `model/config.py` documented `ModelConfig.parameter_count` with `:meth:`
  when it is a `@property` — a cross-reference that would not have resolved. Every
  `:meth:` and `:attr:` role in `src/trainai/` was introspected; this was the only one
  wrong.

- **Ctrl-C at the chat prompt ended the session, which contradicted everything `chat`
  says about Ctrl-C.** The banner offers "`/exit` or Ctrl-D to leave" and never lists
  Ctrl-C; `/help` promised outright that "Ctrl-C stops a generation without leaving".
  Mid-generation that was true. At the prompt a single press threw away the session:
  the model came off the device (about 7 seconds to put back, measured on a 2050), along
  with every `/temp`, `/top-p` and `/seed` that had been set and the `/more` history.
  Python's own REPL catches `KeyboardInterrupt`, clears the line and carries on — only
  Ctrl-D breaks its loop — so the keystroke people press to cancel a half-typed line was
  the one that cost the most here.

  A single Ctrl-C at the prompt now discards the line and says
  `Interrupted. Ctrl-C again, Ctrl-D or /exit to leave.`; a second press in a row still
  leaves, so Ctrl-C remains an exit for anyone who reaches for it first, and a terminal
  that raises on every read cannot spin. Any line that arrives resets the count, so two
  strays over a long session do not add up to an exit. `/help` now describes both halves
  instead of the flattering one.

  The test that covered the old behaviour justified it with "which is what every REPL
  does", which is backwards, and it is now the assertion that the session survives.

- **`--dry-run` exited 0 on a run that could not start.** Its documented job is to
  "catch a badly proportioned run before it costs an evening", and the runs it missed
  were the ones that would not have cost an evening at all — they would have failed in
  the first second. The two checks that decide whether a pair of configurations can
  train together lived inline in `Trainer.__init__`, and a dry run returns before
  constructing a `Trainer`:

  ```
  $ trainai train --data data/shake --context 128 --seq-len 512 --steps 10 --dry-run
  Shape             L4 d256 h4 ff704 ctx128 vocab4096
  ...
  Sequence          512 tokens
  ...
  Nothing was trained. Remove --dry-run to start, or adjust the flags above first.
  $ echo $?
  0

  $ trainai train --data data/shake --context 128 --seq-len 512 --steps 10
  --seq-len 512 exceeds the model's context length of 128.
  $ echo $?
  6
  ```

  A full plan, printing `ctx128` on the shape line and `Sequence 512 tokens` three rows
  below it, and exit 0 — so `train --dry-run && train` passed the gate and then failed,
  which is the obvious thing to write and the worst case to get wrong. A dry run that
  approves what the next command refuses is worse than no dry run, because it is
  trusted.

  Both checks moved to `trainai.train.loop.check_configs_agree`, which needs no device
  and reads no shards, and is now called on the dry-run path as well as by
  `Trainer.__init__`. Dry runs refuse the same pairs with the same message and the same
  exit code, before printing any numbers — refusing after printing them would leave
  figures on screen for a run that cannot happen. The `--plan` path is covered too: a
  plan plus an over-long `--seq-len` override is refused under `--dry-run`.

  The guard itself was not untested — two tests covered it through `Trainer`. Nothing
  tested the second caller, which is what let a check pass its tests while one of the
  two paths to it skipped it entirely.

- **`trainai eval` scored at the context the model was *built* for and called it the
  context it was *trained* at.** These are two different numbers for almost every run.
  A model is built for `--context`; it is trained at `--seq-len`, which defaults to
  `min(context, 256)`. So three of the four presets produce a run where they differ, and
  `--preset large` builds for 1,024 and trains at 256. Eval defaulted to the built one,
  which meant the headline number was measured at up to 4x a context the model had never
  seen — and was then printed beside the training curve's loss as though the two were
  the same measurement.

  Measured on a run built for 512 and trained at 128, 300 steps on tinyshakespeare,
  whose best validation loss was **4.8157**:

      before   scored at 512   4.8295    9,216 predictions   3 batches
      after    scored at 128   4.8158    9,984 predictions  10 batches

  The old default was 0.0138 off the curve and scored 768 *fewer* predictions, because
  a longer window leaves a longer unscorable tail. What made it worse than a wrong
  number was the explanation attached to it: the report noticed the gap, coloured it
  yellow, and told the reader "the trainer samples `--eval-batches` of the split, this
  scored what you asked for" — pointing at batch sampling for a discrepancy its own
  default had caused.

  The default is now the trained context, so the number is comparable to the curve
  (0.0001, and that residue is bf16). Scoring past it is still available with an
  explicit `--seq-len` and is no longer silent: the `Context` row names all three
  lengths that matter, the gap against the recorded loss is attributed to the context
  rather than to sampling, and a note says the result is extrapolation. The refusal for
  a context beyond the model's reach no longer misstates which length it is refusing
  against, and offers the comparable setting alongside the largest legal one:

  ```
  Cannot evaluate at 1024 tokens of context: this model was built for 512.

  What to do: Rotary embeddings do extrapolate somewhat, but past its context the
  model has no trained behaviour at all. Use --seq-len 512 or lower, or --seq-len
  128 to stay comparable to the training curve.
  ```

  `--json` gains `model.trained_seq_len` beside each split's `seq_len`, so a reader can
  tell the two cases apart without opening the checkpoint. **This changes the number
  `trainai eval` reports by default** for any run whose `--context` exceeds its
  `--seq-len`; runs where the two match are unaffected, which a test pins. Two existing
  tests had asserted the message said "was trained at" — they passed because their
  fixture had the two lengths equal, which is how the claim survived being tested.

- **Generation past the training context cost a full forward pass every second
  token, while the docstring promised linear cost.** `generate_stream` claimed "cost
  is linear in the number of new tokens rather than quadratic". Once the key/value
  cache filled, it threw the cache away and re-fed the last `seq_len - 1` tokens,
  which bought exactly one cached step before the cache was full again. So the steady
  state past the context was: one full pass over the context, one cached token, repeat.

  Measured through `generate()` on a 12.6M-parameter model, context 256, a 200-token
  prompt and 400 new tokens, best of three runs:

      before  44,287 forward positions (110.72 per new token)   10.13 s    39.5 tok/s
      after      599 forward positions ( 1.50 per new token)     3.54 s   112.9 tok/s

  74x fewer positions and 2.86x faster wall-clock, on CPU. The waste scaled with the
  context, so a bigger model was hit harder: 5.97 positions per token at context 16,
  17.49 at 64, 33.50 at 128, against 1.07 / 1.02 / 1.01 now.

  The cache is now **slid** instead of rebuilt, via a new `GPT.slide_caches()`. A
  cache cannot simply be truncated, because it holds keys with the rotary rotation
  already applied at their absolute positions — the survivors would still be numbered
  from where they were, and the next query would sit past `seq_len`. But `apply_rotary`
  multiplies by `e^(i·theta·position)` and rotations compose, so multiplying every
  surviving key by `e^(-i·theta·drop)` renumbers the whole cache from zero. That is
  exact, not an approximation: a slid cache matched a fresh forward pass over the same
  window to `2.980e-08`, and stayed at `4.768e-07` after 32 successive single-position
  slides, against keys of magnitude 0.36. Values are trimmed but not touched, since
  the rotation applies to queries and keys only.

  **This changes what the model writes past its context**, and would for any fix here.
  Rebuilding recomputed the window with a truncated history, so a cached second-layer
  key lost information that had fallen off the front; sliding keeps it. Measured
  against a fresh pass over the same window: `2.980e-08` at one layer, where the two
  agree exactly, and `1.292e-02` at four layers, which is the retained history. The
  slid model is strictly better informed, and its window is now the full `seq_len`
  rather than `seq_len - 1`.

  Nineteen tests cover it, including greedy generation past the context checked
  logit-by-logit against a cacheless O(n²) reference. The first version of that test
  compared token ids and was worthless: an untrained model decoded greedily emits one
  token forever — measured, 1 distinct token in 25 for every seed tried — so it passed
  regardless of what the cache did. Two mutations survived it; comparing logits against
  a scripted token stream catches both. All nine mutations of the new code are caught,
  including reverting to the rebuild.

- **Attention was not causal when a key/value cache was combined with more than
  one new token.** `Attention.forward` asked the kernel for its own causal mask
  with `is_causal=True`, which is the fast path the training step depends on. But
  `is_causal` means "query *i* may see key *i* and earlier" with both numbered
  from zero, so it is only correct when every key is also a query. With a cache
  the keys start earlier than the queries, and that diagonal lands in the wrong
  place — so the code disabled the flag for cached calls and passed no mask
  instead, which left the attention **unmasked**. Every new token in the chunk
  attended to the ones after it.

  Measured on a 3-layer model over an identical 12-token sequence, comparing a
  single full forward pass against feeding the same tokens as two chunks:

      before  worst |full - chunked| over all 11 split points = 2.891e-01
      after   worst |full - chunked| over all 11 split points = 2.384e-07

  and directly, on a 2-layer model: changing token 7 moved position 4's logits by
  `2.301548e-02`, where causal attention requires exactly `0.0`.

  `forward` now builds a boolean mask for this case from the queries' absolute
  positions (`past .. past + seq - 1`), which composes with the grouped-query head
  expansion. The two paths that were already right are untouched and bit-identical:
  the uncached training forward still uses `is_causal` (max logit delta `0.000e+00`,
  loss `3.078455448` unchanged in all nine digits), and a one-token step against a
  cache still needs no mask (greedy generation identical).

  `generate_stream` feeds one token at a time, so no code path inside TrainAI hit
  this; it was reachable only through `GPT.forward`'s documented `caches` argument,
  which is what chunked prefill and any prompt-reuse scheme would use. No test held
  the bug in place — the existing cache test only ever fed a single token, so this
  was untested ground rather than a wrong assertion. Twelve tests now cover it,
  including one that pins *which* of the three cases builds a mask, so restoring the
  fused kernel's mask for the training path cannot be dropped silently either.

- **The "no merges were learned" tokenizer error gave three pieces of advice, and
  the first one could not work.** The hint read "The corpus is too small or too
  repetitive for this vocabulary size. Lower `--vocab-size`, lower
  `--min-frequency`, or use more text." Every clause in it was wrong or
  unhelpful, reproduced on the real 1.06 MiB tinyshakespeare corpus with
  `--min-frequency 99999999`:

  - `--vocab-size` is a *ceiling* on how many merges may be kept, so lowering it
    cannot create a merge to keep. Measured: every target from 4096 down to 258
    produced the identical error, and 257 was refused by the floor guard with
    "Use `--vocab-size` 258 or more" — so following the advice walked the user
    between two errors and back.
  - "too repetitive" is backwards. Repetition is what *creates* merges:
    `["abc"]` failed and `["abc abc abc"]` succeeded.
  - "for this vocabulary size" named a flag with nothing to do with it.
  - "too small" was printed for a 1.06 MiB corpus, the size the tool's own
    guidance calls enough text ("A first experiment wants at least a megabyte").
  - `--min-frequency` was the only cause and the only fix, and it was buried as
    the second of three options with no number attached.

  The message now names the cause and a threshold that works:

      before  Training produced only 257 tokens, which means no merges were
              learned from the corpus.
              What to do: The corpus is too small or too repetitive for this
              vocabulary size. Lower --vocab-size, lower --min-frequency, or
              use more text.

      after   Training learned no merges: the vocabulary is only the 256 byte
              values and 1 special token(s).
              What to do: No pair of characters occurs 99,999,999 times
              anywhere in the corpus, so there was nothing to merge. Use
              --min-frequency 1, which merges any pair that appears, and raise
              it from there.

  `--min-frequency 1` on that corpus then prepares normally (vocabulary 8,192).
  When `--min-frequency` is already 1 there is nothing lower to suggest, so the
  hint stops naming the flag and asks for more text instead — the old wording
  advised lowering it whatever the value was, including 1.

  A test asserted `"--vocab-size" in hint` for this message, which is why the
  impossible advice went unquestioned; it now asserts the opposite.

- **`--min-frequency 0` was a flag that did nothing.** The Rust BPE trainer
  clamps 0 to 1, so a request to learn merges from zero occurrences trained
  exactly as if 1 had been passed, silently contradicting the flag's own help
  text ("A pair must occur this often to be learned as a merge"). It is now
  refused before the corpus is read, with `Use --min-frequency 1 to merge any
  pair that appears. The default is 2.` This is also what makes the hint above
  honest: it names 1 as the lowest threshold, so 0 must not be accepted below it.

- **`Trainer.evaluate` averaged per-batch validation losses without weighting them,
  so every validation loss this project has published is affected.**
  `sequential_batches` keeps the ragged final batch rather than dropping the tail of
  a split, so that short batch carried a full batch's weight. Measured on the 1.1 MB
  Shakespeare validation split at the run's own settings — six sequences out of
  thirty-eight, so 16% of the data carrying 50% of the mean:

  ```
  unweighted batch mean:  6.961265
  token-weighted mean:    6.949717
  recorded in checkpoint: 6.961310   (4.5e-5 from unweighted, i.e. bf16)
  ```

  0.012 nats. That matters beyond the printed figure, because **the best checkpoint
  is selected on this number**, so a biased mean could keep a checkpoint the data
  does not support. It is also the number `trainai eval` recomputes: after the fix, a
  retrained run and an independent evaluation agree to 0.0 at the same precision.
  Any validation loss printed by an earlier version is biased by roughly this much
  whenever the split did not divide evenly by the batch size — including the M2
  figures quoted above.
- **Evaluation coverage was counted in tokens where it should have been counted in
  windows.** A window of `seq_len + 1` tokens yields only `seq_len` predictions,
  because the first token is context and is never predicted. So scored positions
  over split tokens reported 95% coverage on a split that had in fact been scored
  completely.
- **A re-prepared dataset with a different `--seed` was undetectable.** Moving the
  train/validation boundary leaves the tokenizer fingerprint identical, so every
  existing check passed while the new `val` split held documents the model had
  trained on — measured here as 6.9498 on the run's own dataset against 7.0843 on a
  reseeded copy. The dataset content hash covers the seed and the split fraction, so
  it is now compared and reported first, and the comparison against the trainer's
  recorded loss is dropped rather than shown as a disagreement. Flagged, not
  refused: scoring a model on a different corpus is a real thing to want.
- **Evaluation inherited a batch-size refusal that only makes sense for training.**
  A training batch holding the same window twice makes the gradient smaller than it
  looks; scoring builds no gradient. The 8-row default was turning any validation
  split shorter than 8 sequences into a usage error telling the user to re-prepare
  their dataset. The batch size is now reduced to what the split holds, and reported.
- **`chat` reported tokens/s from string deltas rather than tokens.** `stream` yields
  text and holds back a token that completes only half a character, so any non-ASCII
  output made the reported rate lower than the machine's real one. Generation now
  also exposes `stream_pieces`, yielding `StreamPiece(text, tokens)` per token
  produced, and `stream` is its text-only view.
- **Model output could contain a genuine U+FFFD, which cp1252 cannot encode.** The
  first real `chat` run died with `UnicodeEncodeError` from inside Rich's writer,
  mid-stream, after tokens had already printed. `console.printable()` reduces model
  text to what the stream can encode. Relatedly, `console.emit_json()` is now the
  single JSON path for every command with `ensure_ascii=True` — Rich's `print_json`
  re-serialises with it turned off, so a non-ASCII path would have broken a
  machine-readable stream.
- **The benchmark read its peak-memory counter after wiping it**, which made every
  memory verdict meaningless. The counters were read after the `try` block, and the
  `finally` clause calls `_release`, which calls `reset_peak_memory_stats` so one
  candidate's cached blocks cannot leak into the next. And that call does not zero the
  counter — it sets it to whatever is still live, which on a small model is almost
  exactly the parameter bytes. So a 4.26M-parameter configuration reported a 65 MiB
  peak where the trainer independently reported 957 MiB for the same shape: wrong by a
  factor of fifteen, and plausible enough to go unnoticed. Every over-budget rejection
  was comparing post-release residue against the budget. The counters are now read
  inside the `try`, immediately after the timed loop, and a `gpu` test asserts the peak
  exceeds twice the persistent bytes so it cannot silently become the parameters again.
- **`TrainConfig.to_dict` serialised the resolved warmup in place of the raw field**,
  so `to_dict` / `from_dict` was not a faithful round trip: an unset warmup came back
  pinned to the adaptive default it happened to have. That surfaced as
  `trainai train --plan plan.json --steps 20` failing validation for a warmup of 20
  that nobody had chosen. The raw field now round-trips as itself and the resolved
  value is published alongside it as `resolved_warmup_steps`.
- **The planner's time horizon bounded the larger candidates but not the accepted
  one.** It would reject `small` for needing 6h31m against a 6h default horizon and
  then recommend `tiny` needing 21h, because the horizon was only ever consulted while
  climbing. The horizon now cuts the accepted candidate's step count too, and the note
  distinguishes the default horizon from an explicit `--time`.

- **bf16 support was decided by an NVIDIA-only rule on every backend.** ROCm reports
  the gfx architecture through the same `major`/`minor` fields that carry CUDA
  compute capability, so `capability >= (8, 0)` was True for gfx1030 — an RX 6900 XT
  would have been told it had bf16 and TrainAI would have autocast into a format the
  card lacks. There is no architecture number that reliably implies bf16 on AMD
  (gfx906 has none and reports `(9, 0)`; gfx908 has it and reports the same), so the
  runtime is asked instead, and a runtime that cannot answer yields fp32. TF32, which
  has no AMD or Intel equivalent, is no longer inferred on those backends or reported
  by `doctor` there.
- **`nvidia-smi` parsing matched nothing on current drivers.** Recent drivers print
  `CUDA UMD Version: 13.3` where older ones printed `CUDA Version: 12.4`, and
  `nvidia-smi -q` now marks the old key as deprecated. The regex was pinned to the
  old spelling and failed silently, so TrainAI concluded it could not determine a
  CUDA version on any recent NVIDIA machine. Found on driver 610.88.

- The progress display crashed with `UnicodeEncodeError` on a legacy Windows
  console. Rich downgrades its own box-drawing and progress-bar glyphs there but
  not spinner frames, and the default spinner is Braille (U+280x) — so
  `trainai data prepare` failed part-way through, after the tokenizer had already
  started training. The spinner is now chosen from what the stream can encode.
- The training progress line overflowed 80 columns, so Rich truncated it with
  U+2026 — written as 0x85 on cp1252 and read back as a replacement character,
  losing three of the four numbers the line exists to show. The columns now fit,
  and the gradient norm moved to the metrics file.
- Re-preparing a dataset with a larger `--shard-tokens` left the previous run's
  extra shard files behind: valid according to the new manifest, invisible to
  `verify_dataset`, and still occupying disk. Preparation now clears the previous
  dataset before writing.
- A JSONL record missing the field named by `--jsonl-field` reported only that the
  field was absent. It now lists the fields the record does have, which is the
  information needed to correct the flag.
- `.gitignore` matched `data/` at any depth, which silently excluded the entire
  `src/trainai/data/` package from the repository — everything passed locally, and
  a clone would have installed a package that could not import. The patterns are
  anchored to the repository root, and a test asserts every source file is tracked.
- **`import trainai.data` failed outright on a Python built without libbz2, liblzma
  or libsqlite3.** All three of `bz2`, `lzma` and `sqlite3` are standard-library
  modules that CPython builds only when the matching system library is present at
  compile time; without it the pure-Python wrapper still ships and its import fails
  on the missing C accelerator. `"No module named '_lzma'"` is the best-known pyenv
  install failure, and some slim container images ship without any of the three.
  Measured here by blocking each import in turn: any one of them killed the whole
  package, so `trainai data prepare corpus.txt` on a plain text file died with a
  traceback about a codec the corpus does not use. The three are now imported
  behind a guard, and only a corpus that actually needs one fails — naming the
  library to install rather than the private module that was missing:

  ```
  case                             ingest       data       main     errors
  nothing blocked (control)            ok         ok         ok         ok
  no liblzma                           ok         ok         ok         ok
  no libbz2                            ok         ok         ok         ok
  no libsqlite3                        ok         ok         ok         ok
  none of the three                    ok         ok         ok         ok
  ```

  CI would never have caught this: GitHub's runners have all three libraries.
- **A corrupt or half-downloaded compressed corpus ended the run with a traceback.**
  A decompressing stream does not fail when it is opened — it fails part-way through
  being read, once it reaches the bytes that do not decode, so no handler wrapped
  around `open` ever saw it. The obvious handler would not have been enough either:
  `zlib.error` and `lzma.LZMAError` both derive straight from `Exception`, so
  `except OSError` catches gzip's header check and bz2's "Invalid data stream" and
  lets a corrupt `.xz` through. Measured across every codec crossed with every
  reader kind — 12 of 12 combinations escaped, plus a bad CRC on a member inside a
  valid `.zip`, a corrupt `.tar.gz`, a `.docx` with broken XML and a `.db` with
  shredded pages: 16 of 16 cases now give a clean error naming the file, and
  truncation is reported as truncation rather than as corruption.

  Fatal even under `--on-error skip`, deliberately: by the time the stream fails,
  documents from earlier in the file have already been yielded, so skipping the
  remainder would train on however much of the corpus arrived and report success.
  An I/O error on an *uncompressed* file is still an I/O error and is not relabelled
  as corruption — a negative-control test pins that.
- **Three writers wrote different bytes on Windows than on Linux.** `plan.json`, and
  an export bundle's `config.json` and `README.md`, were written without
  `newline="\n"`, so Python translated every `\n` to `os.linesep` on the way out —
  measured at 144 bytes against 133 for one small `config.json`. The export manifest
  records those byte counts, so the same export reported a different size depending
  on which machine produced it. `tests/conftest.py` has stated this rule since the
  suite began ("this project asserts checksums over exactly those bytes") with
  nothing enforcing it; `test_every_text_write_states_its_newline` now walks the AST
  of `src/`, `tests/` and `examples/` and fails naming the file, line and offending
  value.
- **That guard, as first written, silently skipped most of the calls it was meant to
  check.** It read the mode of every `open` from `args[0]`, which is only right for
  `Path.open`. Three signatures are actually in use: the builtin `open(file, mode)`
  and the codec and archive modules (`gzip`, `bz2`, `lzma`, `tarfile`) put the file
  first; `Path.open(mode)` puts the mode first; `ZipFile.open(name, mode)` writes
  bytes and takes no `newline` at all. So `open(self.path, "a")` in `train/metrics.py`
  was skipped, and so was every `gzip.open(path, "wt")` corpus fixture in the test
  suite — measured, those translate too: `b"one\ntwo\nthree\n"` written through
  `gzip.open(..., "wt")` lands as `b"one\r\ntwo\r\nthree\r\n"` inside the member, 17
  bytes against 14, and 57 compressed bytes against 52. `_mode_of` now distinguishes
  the three, and a parametrized test pins each convention directly rather than
  inferring it from what the repository happens to contain — the repository-wide walk
  cannot pin it, because a `Path.open("w")` in `examples/` keeps a both-kinds
  assertion green even with the reading broken.
- **`--resume` pointed at a `.pt` file from another project crashed with a traceback
  instead of saying so.** The handler for "this is not a TrainAI checkpoint" built
  its own message with `(raw or {}).get("format")` — on the one branch reached
  precisely when `raw` may not be a mapping. Measured, pointing `trainai train
  --resume` at each shape a `.pt` file can hold:

  ```
  torch.save(model)   -> AttributeError: 'Linear' object has no attribute 'get'
  torch.save(tensor)  -> RuntimeError: Boolean value of Tensor ... is ambiguous
  torch.save([1,2,3]) -> AttributeError: 'list' object has no attribute 'get'
  torch.save("hi")    -> AttributeError: 'str' object has no attribute 'get'
  torch.save({})      -> clean CheckpointIncompatibleError
  ```

  `torch.save(model, path)` is how most people save a PyTorch model, so that first
  line is the likeliest way to arrive here at all. Only the dict answered correctly,
  and the existing test for this branch passes a dict — which is why the whole class
  went unnoticed. The user saw a full traceback and exit 1 rather than the intended
  message and exit 5. Now every shape gives the clean error, and the message names
  what the file actually held ("the file holds a Linear"), which is what tells
  someone they pointed at a saved model rather than a checkpoint.
- **The empty-validation-split error advised a `--val-fraction` the validator then
  refused.** The hint computed `max(1.0 / documents, 0.02)` and never checked it
  against the exclusive `0.5` ceiling the range check enforces — and this error fires
  most often on a corpus of one or two documents, which is exactly where that formula
  names `1.00` or `0.50`. Measured end to end on a 1,010-character single-document
  corpus:

  ```
  The validation split is empty: none of the 1 documents was selected
  at --val-fraction 0.05.
  What to do: Raise it to about 1.00, or split your text into more documents ...

  $ trainai data prepare ... --val-fraction 1.00
  --val-fraction must be at least 0 and below 0.5, got 1.0.
  ```

  Two things were wrong. The suggestion is impossible for one document — one document
  cannot land on both sides of a split, whatever number you pass — so that case now
  says so and points at `--max-doc-chars` and `--val-fraction 0` instead of naming a
  fraction at all. And for two or more documents the split is not a quota: it is an
  independent keyed hash per document, so a fraction of `1/n` leaves validation empty
  about 37% of the time, consistent with `(1 - 1/n) ** n`. Measured over 300 seeds:

  ```
  documents  suggested  accepted by the check  usable two-way split
      1        1.00            no                   0/300   ( 0%)
      2        0.50            no                 169/300   (56%)
      3        0.33            yes                190/300   (63%)
      5        0.20            yes                199/300   (66%)
     16        0.06            yes                179/300   (59%)
  ```

  So the number is now capped below the ceiling, offered as making a non-empty split
  "likely rather than certain", and the reliable fix — more documents — is named
  alongside it. The ceiling itself is a constant rather than a bare literal, which is
  what let the two drift apart.
- **A `--val-fraction` that took *every* document wrote a dataset with nothing to
  train on, and reported success.** The empty-validation guard had no counterpart for
  the training split. Since the split is a per-document coin flip, a legal fraction
  can take everything: measured, one document at `--val-fraction 0.49` does it for 37
  of 60 seeds. `trainai data prepare` then exited 0 —

  ```
  |Split | Documents | Tokens | Shards |  On disk|
  |train |         0 |      0 |      0 |      0 B|
  |val   |         1 |    601 |      1 | 1.17 KiB|

  Wrote data/probe2/ds - 601 tokens from 1 documents.
  Next: trainai data inspect data/probe2/ds to review it.
  ```

  — and `trainai train` refused the dataset afterwards with "a dataset without a
  train split is a bug", advising a re-run of `data prepare`, which is deterministic
  and reproduces the empty split exactly. Not a bug, and not fixable by re-running.
  `binarize_documents` now refuses to write such a dataset, naming the cause and what
  to change, so the failure arrives at the command that caused it rather than at the
  next one. The downstream message no longer blames TrainAI nor advises an unchanged
  re-run.
- **Every piece of advice about `--vocab-size` named a size the tokenizer refuses.**
  Byte-level BPE starts from all 256 byte values and reserves one entry for
  `<|endoftext|>`, so 257 entries are filled before anything is learned and 258 is the
  smallest target with room for a single merge. Three places had it wrong, and the
  measured ladder on a 2,130-character corpus is what shows it:

  ```
  --vocab-size 255  refused up front  "257 entries are needed"
  --vocab-size 256  refused up front  "257 entries are needed"
  --vocab-size 257  ACCEPTED, then failed after reading the whole corpus:
                    "Training produced only 257 tokens, no merges were learned"
  --vocab-size 258  works
  ```

  So the corpus was blamed for a budget that could not have worked. The tokenizer's
  floor now sits one entry higher, which turns 257 into an up-front argument error, and
  its hint says 258 rather than 257. The floor also scales with `extra_special_tokens`
  instead of ignoring them.

  The corpus validator's `vocab_too_large_for_corpus` warning suggested
  `max(256, round_to_power_of_two(chars // 100))`, which is 256 — one below the floor —
  for **every** corpus under 51,200 characters, because both halves cap there:
  `1 << 8` is 256 for any value from 256 to 511, so the `max` was not the only
  offender. Following the warning produced "vocab_size=256 is too small". The floor is
  now a named constant, duplicated from the tokenizer rather than imported so that
  `trainai --help` does not load the Rust tokenizers library, and cross-checked by a
  test so the copy cannot drift.

  And where the floor is reached the warning no longer implies the suggestion clears
  it. Under about 25,800 characters no legal vocabulary is small enough, so re-running
  with the suggested size warns again; the hint now says so and names the real problem,
  which is the amount of text. Tests take each suggestion and feed it back to
  `train_tokenizer` on the same corpus, so a suggestion that cannot be followed fails
  the suite rather than reaching a user.
- **`data prepare` reported a tokenizer setting it could not accept as an empty
  corpus.** When tokenizer training fails, `data prepare` runs corpus validation first,
  on the grounds that "no merges were learned" usually means the corpus is too small and
  the corpus error says so better. It does — but it ran that way for *every* tokenizer
  failure, including the ones raised before a single document had been read. An
  unread corpus measures as zero documents, and zero documents always fails validation
  as `empty_corpus`, so a bad `--vocab-size` came out as:

  ```
  Corpus            data/vocabprobe/corpus.txt
  Files             1 - 1 text
  Size on disk      27.7 KiB

  +---------------------------- Dataset empty error ----------------------------+
  |  No usable documents were found.                                            |
  |  What to do: Check that the files contain text. Empty files, and documents   |
  |  shorter than --min-doc-chars, are dropped before this point.               |
  +-----------------------------------------------------------------------------+
  ```

  Two lines under a panel reporting 27.7 KiB of text, about a corpus that was fine,
  advising a check of files that were never opened — and the message naming the actual
  problem was thrown away. Validation now speaks only when the trainer read something,
  which is the condition under which it has anything to say. A corpus that really is
  empty still reads as empty, because the trainer does ask for a document and get
  nothing; that path keeps its message, and a test pins it so the fix cannot swap one
  wrong answer for another.
- **A truncated shard ended training with a numpy traceback, while `--verify` on the
  same dataset explained it properly.** The batch loader caught `OSError` around
  `np.memmap`, but numpy signals both of the failures that matter here with
  `ValueError`: `cannot mmap an empty file`, and `Size of available data is not a
  multiple of the data-type size`. Neither was caught. Measured on a dataset whose
  train shard had one byte cut off:

  ```
  trainai data inspect data/shardprobe/ds --verify
    -> Dataset format error, exit 3
       Shard train_00000.bin is 40809 bytes; the manifest says 40810
       (20405 tokens x 2 bytes).
       What to do: The file was truncated or partly overwritten, most likely by an
       interrupted copy or a full disk. Re-run `trainai data prepare`.

  trainai train --data data/shardprobe/ds --out ... --steps 2
    -> Traceback ... numpy/_core/memmap.py:248
       ValueError: Size of available data is not a multiple of the data-type size.
       exit 1
  ```

  So the user who ran training first got a numpy internal instead of the diagnosis
  the tool already knew how to write. What hid this is that only *odd* truncations
  raised: an even one mapped fine and was caught afterwards by the token count, so
  half the cases genuinely were handled.

  The loader now compares the file's size in bytes against the manifest before
  mapping it, which is the same comparison `verify_dataset` makes and is deliberately
  worded identically — a test asserts the two messages are equal character for
  character, so they cannot drift apart again. The `except` around the map now also
  lists `ValueError` as a backstop. Every failure shape was measured before and after:

  ```
                             before                          after
  intact                     ok                              ok
  truncated by 1 byte        ValueError traceback            format error
  truncated by 2 bytes       format error (token count)      format error (bytes)
  zero bytes                 ValueError traceback            format error
  one byte too long          ValueError traceback            format error
  missing file               format error                    format error
  directory in its place     format error                    format error
  ```

  `ShardedTokenStream` had no tests at all before this; it now has the shapes above,
  each shard checked against its own token count rather than the first one's, and the
  never-reached race backstop pinned with a patched map so it is a guard rather than
  decoration.
- **`trainai eval` contradicted itself about how much data a split needs.** A split
  needs `2 * seq_len + 1` tokens to form one window — `seq_len + 1` for the window
  itself and another `seq_len` reserved so a shuffled epoch's alignment offset cannot
  run off the end. The batcher enforces exactly that. `eval` restated the requirement
  in its own words and got it wrong by about half, so the two appeared in the same
  panel, four lines apart:

  ```
  +-------------------------------- Usage error --------------------------------+
  |  Cannot evaluate the val split: The val split has 107 tokens, which is not  |
  |  enough for even one sequence of 64 (needs 129).                            |
  |                                                                             |
  |  What to do: The val split needs at least 65 tokens to form one window.     |
  |  Re-prepare the dataset with a larger --val-fraction, or evaluate at a      |
  |  shorter --seq-len.                                                         |
  +-----------------------------------------------------------------------------+
  ```

  The user has 107 tokens and is told they need 65, which reads as "this error should
  not be happening". A test asserted the wrong number, so it was pinned rather than
  merely present.

  The requirement is now computed in one place. `largest_seq_len` inverts the window
  arithmetic and the loader's hint names the answer instead of a direction — `Use
  --seq-len 53 or lower` on that 107-token split, which is the largest that works: 53
  scores, 54 does not. `eval` passes the loader's hint through rather than composing a
  second one. Below five tokens no context works at all, and the hint says so instead
  of suggesting `--seq-len 1`, which the batcher refuses — the same mistake in a new
  place. Tests take the suggested number and score with it, so a suggestion that
  cannot be followed fails the suite.

  The wrapper also caught `Exception` and relabelled everything as a usage error about
  split size, so an internal fault would have sent the user to re-prepare a dataset
  that was fine. It catches `DatasetError` now, which is everything `open_split`
  refuses on purpose.
- **The training batch size silently decided whether a held-out loss existed at all,
  and the run then guessed at why there was none.** Two defects that produced one
  symptom. `Trainer._open_validation` opened the validation split at the *training*
  batch size, and `TokenBatcher` refuses a batch it cannot fill — a rule that exists
  for training, where a repeated window shrinks the gradient, and not for validation,
  which is a forward pass with no gradient. `evaluate_split` had always shrunk the
  batch to what the split holds, and `windows_available` exists so a caller can. The
  trainer used neither. Measured on a dataset with a 45-token validation split at
  `--seq-len 16` (one window), same data, same context, four steps:

  ```
  --batch-size 1   Best val loss  5.7584 at step 4
  --batch-size 4   Validation     not measured
  ```

  Whatever went wrong was then discarded: `_open_validation` returned a bare `None`,
  so the result panel had nothing left to explain the absence with and printed one
  fixed string for every cause. On the commonest cause — a validation split that
  exists but is too small for one window at the run's `--seq-len` — both causes it
  named were false:

  ```
  Validation        not measured  (no validation split, or --eval-every 0)
  ```

  That run's dataset had a 45-token validation split and `--eval-every 2`. The loader
  had already computed the context that would fit, and `trainai eval` answers the same
  question with `Use --seq-len 22 or lower`; the run threw the answer away. Nothing was
  written to `metrics.jsonl` either, though a comment there claimed the absence was
  recorded.

  The reason is now carried, in the loader's own words rather than a second wording of
  them, and reaches all three places that report it — a `note:` while the run starts,
  the `start` and `end` events in `metrics.jsonl`, and the result panel:

  ```
  note: No held-out loss: The val split has 45 tokens, which is not enough for
  even one sequence of 64 (needs 129).
        Use --seq-len 22 or lower, or prepare more data. For the validation
  split, raising --val-fraction during `trainai data prepare` also helps.
  ```

  The three causes are now distinguishable: `--eval-every 0` no longer blames the
  dataset, and a dataset prepared with `--val-fraction 0` is told to re-prepare rather
  than to shorten its context. `except Exception` here would have relabelled a bug
  anywhere inside `open_split` as "no validation split" and finished the run green; it
  catches `DatasetError`, and re-raises `DatasetFormatError` so a damaged shard stops
  the run instead of being reported as a small corpus.

  Two consequences elsewhere. The planner capped the recommended micro-batch at the
  validation window count to work around the trainer, which cost the *training* batch
  most of its size: at `--seq-len 256` on a 5,000-token validation split it
  recommended `--batch-size 16 --grad-accum 4` where no validation split at all got
  `--batch-size 64 --grad-accum 1` — four times the launches per step for the same
  effective batch. And it announced "Validation will not run" whenever the micro-batch
  exceeded the window count, which would now be a false claim about a number the run
  does produce; it fires on the window count instead, and names a `--seq-len` that
  works. The export README asserted "the run had no validation split" for every
  missing loss, a cause a checkpoint cannot know; it records the absence and no cause.

  Where validation already worked nothing moved: the same run at `--batch-size 1`
  reports 5.7584 before and after.

- **The walkthrough prepared one dataset and trained from another, so following it
  verbatim failed at the fourth step.** [docs/walkthrough.md](docs/walkthrough.md) wrote
  `trainai data prepare data/corpus --out data/shakespeare` and then, five steps and 110
  lines later, `trainai plan --data data/shake` — the name the recorded transcripts use.
  A reader who typed what the page said got `No manifest.json in data/shake`, an error
  that reads like their own typo rather than the page's. The three commands naming
  `data/shakespeare` now name `data/shake`, which is the spelling the transcripts below
  them cannot be edited to match.

  Neither of the two checks that guard flag and command spelling could see this: both
  paths are real directories and both commands are real, so the mistake is only visible
  as *dataflow*. A recipe page is now checked as a sequence — every directory a `--data`
  or a positional run argument reads must be a directory an **earlier** `--out` on the
  same page wrote — which catches a renamed step, a reordered one, and this. Only fenced
  `bash` blocks count as steps, since a transcript's paths are whatever the machine that
  recorded it used.

- **Two checks read `.github/workflows/ci.yml` unguarded, so the suite failed when run
  from an unpacked sdist.** `test_ci_runs_the_oldest_python_the_package_says_it_supports`
  and `test_every_python_the_classifiers_advertise_is_one_ci_runs` both call
  `matrix_versions()`, which read the workflow directly; the sdist deliberately does not
  ship `.github`, so both raised `FileNotFoundError` there. That is the one place it
  mattered — `.github/workflows/release.yml` runs the suite from the unpacked sdist on
  every tag, so the first real release would have gone red on two tests that have nothing
  to do with the release. Their siblings already skipped for exactly this reason;
  `matrix_versions()` now does too, and deleting `ci.yml` in a checkout still fails
  outright at `test_there_are_github_templates_to_check`, which names that path.

  Found by building the wheel, cold-installing it into a clean virtualenv and running
  the sdist's own suite against it — not by any run from a checkout, where the file is
  always present.

### Changed

- **Four tests that only asserted anything on Windows now assert it everywhere.** Three
  archive tests in `test_data_ingest.py` and one mask-map test in `test_data_binarize.py`
  proved a handle had been released by doing something to the file afterwards —
  `path.unlink()` for the archives, rewriting the bytes for the mask. Both raise
  `PermissionError` on Windows while a map or a handle is open, so both are real proofs
  there. On Linux, unlinking and rewriting an open file are ordinary operations that
  succeed whether the handle leaked or not, so on the four Ubuntu jobs — half the CI
  matrix — the tests passed unconditionally. A leak reaching those jobs first would have
  been caught by nothing.

  The closed state is now read directly, which works on every platform: `ZipFile.close`
  clears `fp`, `TarFile.close` sets `closed`, and the loader's memory maps are tracked by
  weak reference and have to be dead after `close()`, which is the release that `close()`
  actually claims. Each test keeps the Windows operation below the portable assertion,
  because a lock on a file the user is about to re-prepare is the consequence they hit.
  Both helpers refuse to pass vacuously: if nothing was opened at all, they say so rather
  than reporting that everything opened was closed.

  Recording the containers has one trap worth writing down. `tarfile.open` is a bound
  classmethod of the real `TarFile`, captured at import, so patching `tarfile.TarFile`
  intercepts nothing and the tar test would have gone straight back to proving nothing;
  the wrapper goes on `tarfile.open` instead. `np.memmap` is wrapped rather than
  subclassed, since an ndarray subclass brings `__new__` and `__array_finalize__` with it
  and none of that is needed to note what was opened.

  Nothing in `src/` changed. The lifecycle was audited first and is correct: `documents()`
  closes the archive in a `finally`, which covers a caller that abandons the generator
  (sampling does), `discover()` uses `with`, `_MemberStream` closes owned streams
  innermost-first, and `TokenBatcher.close` releases the mask stream as well as the token
  stream. Each of the four was confirmed against a deliberately broken version — the
  `finally` removed, and the mask stream left open — and each failed on the portable
  assertion rather than on the platform-specific line beneath it. A new third archive test
  covers `.tar`, since `_close_archive` is shared and only `.zip` was reaching it.

  Found by scanning every `test_*` function for a body containing no assertion, no
  `raise`, and no `raises`/`warns` context — a test that cannot fail is worse than no
  test, because it holds a coverage number and a name that says the behaviour is checked.

- **Seven names nothing called are gone.** Found by asking, of every module-level
  definition in `src/trainai`, whether the identifier appears anywhere else in the
  repository at all — and reading the answers rather than trusting them, since a private
  helper used only inside its own file is not dead and a name in a docstring is not a use.

  Four functions were defined, exported by nothing, and called by nothing:
  `cli/train.py`'s `describe_presets` — a preset list "for `--help`" that `--help` does
  not use, in a format neither `--preset` nor `--max-preset` prints; `cli/eval.py`'s
  `describe_report` and `cli/plan.py`'s `describe_plan`, each "for the web interface in
  M5" and each a one-line alias for a `to_dict` the caller can reach itself; and
  `probe.py`'s `python_summary`, "useful in bug reports" and absent from `doctor`,
  `doctor --json`, and the issue templates. Three exception classes went with them:
  `HardwareError` and its `NoAcceleratorError` and `InsufficientMemoryError` subclasses
  were raised nowhere, caught nowhere, and shared `ExitCode.CAPACITY` with the
  `CapacityError` that does the work. That is worse than unused — it is an invitation to
  raise the wrong one, since a contributor reaching for `InsufficientMemoryError` on an
  OOM would produce something indistinguishable at the exit code and unreachable by
  anything that handles `CapacityError`. `CapacityError`'s docstring now records what
  used to be there and why "not enough VRAM" is one error rather than two, and why "no
  GPU" is not an error at all when `doctor` reports it, `plan` measures the CPU, and
  `train` runs there.

  `describe_plan` had a test, which is how it survived: `assert describe_plan(plan) ==
  plan.to_dict()` against a body of `return plan.to_dict()` — true by construction, green
  for any plan, and coverage on a line that could not fail. What it was standing in for
  was never checked, so it is now: `--json` both prints a plan and writes one, and the
  test compares the two documents whole.

- **Reading a checkpoint no longer runs it.** `load_checkpoint` was the one place in the
  package that called `torch.load` without `weights_only=True`, which means it imported
  and called whatever the file named. That is not a theoretical exposure: a checkpoint is
  the artifact people copy between machines and attach to issues, and `trainai finetune
  --from` takes its path straight off the command line. The load is now restricted, which
  required a format change — `trainai-checkpoint` is at **version 2**.

  What stood in the way was a single NumPy array. The restricted unpickler will not build
  one, and `np.random.get_state(legacy=True)` returns 624 `uint32` words of MT19937 state
  inside a tuple. Measured rather than assumed, by allowlisting a real checkpoint's
  refusals one at a time until it loaded: **four globals**, all NumPy's, all reachable
  from that one array. With those four allowed the rest of the file loaded untouched — so
  the optimizer moments, the config dicts, the metrics and Python's own `random` tuple
  never needed the unsafe loader at all. Version 2 stores the same 624 words as a tensor,
  which is what the rest of the file is made of anyway.

  A **version-1 file is refused by name**, not migrated, and that costs somebody a
  half-finished run. Reading its random state would need exactly the loader this change
  exists to stop using, so a converter would reintroduce the exposure for every file it
  touched in order to recover one field. Two alternatives were considered and rejected on
  the measurements rather than on taste. Falling back to the unrestricted loader when the
  strict one refuses reads as a kindness and is a hole: the fallback triggers on every
  file the restriction was protecting against. `torch.serialization.safe_globals` is the
  bounded version of that idea, and it needs torch ≥ 2.5 against this project's
  `torch>=2.2` floor, with NumPy global names that move between NumPy versions
  (`numpy._core` versus `numpy.core`) — a permanently fragile branch for a format whose
  installed base is a contributor's working directory, since version 1 shipped in no
  release. The refusal says which of those it is and what to do instead.

  The restriction would have destroyed the best error message in the module. Pointing
  `--resume` at the output of `torch.save(model, path)` is the most common mistake anyone
  makes with PyTorch, and it used to answer *the file holds a `Linear`* — because the
  unrestricted load succeeded and the object could be looked at. Under the restriction
  that load fails first, and the naive result is "could not be read" for every cause at
  once. So the diagnosis now comes from **reading the pickle without executing it**:
  `torch.save` writes a zip whose `data.pkl` member is the pickle, and
  `pickletools.genops` walks its opcodes, listing every global the file *would* have
  imported while importing none of them. Pure stdlib, no torch version to depend on.
  Three shapes are told apart — a saved `nn.Module`, a version-1 checkpoint, and damage,
  the last with the imports named so a reader can tell which it was.

  Two details of that scan are worth recording because they are invisible until they are
  wrong. Which opcode names an import is a property of the pickle's *protocol*, not of its
  contents, so both `GLOBAL` and `STACK_GLOBAL` are read: torch writes protocol 2 today
  and a checkpoint-shaped file from another tool can arrive at 4 or later. And protocol 2
  still spells a builtin the Python 2 way, so the same `dict` reads as `__builtin__.dict`
  there and `builtins.dict` at protocol 5 — which is why the check matches on a
  `torch.nn.` prefix rather than on a table of exact names, a table that would have to
  carry both spellings of every entry and would silently miss whichever one it was not
  written against.

  The whole change is invisible in the happy path — the suite passed unaltered when it
  first landed, which is the gap rather than the achievement, because it means a revert
  would also pass. There are three ways to undo it and they are three different edits, so
  there are three gates. Putting the argument back to `False` is caught by tests that load
  a checkpoint carrying a reduced object and require a refusal. Having `capture_rng` hand
  over `np.random.get_state()` unconverted — a one-line simplification that looks
  obviously right — is caught by reading the pickle a real save produces and requiring no
  NumPy import in it. And adding a *second* `torch.load` somewhere else in the package,
  by someone with no reason to know why the first one is careful, is caught by a source
  scan asserting there is exactly one and that its argument is the literal `True`.

  That third one is why the argument is passed explicitly rather than left to the default.
  The default is only `True` on a torch newer than this project's floor, and — read out of
  torch 2.10's `serialization.py` rather than assumed — the environment variable
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` forces the unsafe loader *only* where the call site
  did not set the argument. An explicit `True` puts the choice beyond the reach of a
  variable somebody exported for another library.

  `SECURITY.md` changed with the code, in both directions. "Checkpoints are pickles, not
  safetensors" left the list of hardening that is deliberately absent, and *getting
  TrainAI to load a checkpoint without the restriction* joined the list of things worth
  reporting. What stays out of scope is a bypass of the restriction inside PyTorch, which
  is torch's to fix and this project's to ship by raising its floor — the honest boundary
  being that the guarantee is a dependency's, while what TrainAI promises is the narrower
  and checkable thing: it does not ask for the unrestricted loader anywhere.

- **The project has a real address, so the placeholder URLs are gone.** `[project.urls]`
  carried `https://github.com/trainai/trainai` under a comment saying so, `tokenizer.py`
  told a user to file a round-trip bug at that same non-existent issues page, the
  CHANGELOG's `[Unreleased]` link resolved nowhere, and `CONTRIBUTING.md` said
  `git clone https://github.com/<your-fork>/trainai`. All four now point at
  `https://github.com/shubhampardule/trainAi`, and `authors` names a maintainer with a
  contact address instead of "TrainAI contributors".

  The README's install note changed with them, and narrowed rather than disappeared. It
  said the project "has not been pushed to a public host or to PyPI, so there is no
  `pip install trainai` and no clone URL to give you". Half of that stops being true the
  moment the repository is public; the other half does not. So it now says **"Not on PyPI
  yet"** and gives the clone, and the platform blocks start with `git clone` instead of
  assuming the reader already has the directory. Nothing has been uploaded to any index,
  so the check that stops any error hint from suggesting `pip install --upgrade trainai`
  stays exactly as it was — only its marker and its one exemption sentence were rewritten
  to the new wording. The "A published install" row under *What has never been run* also
  stays: a wheel has been installed from a local file, and never once from a package name.

- **The README was 887 lines of engineering log, and a reader arriving cold met all of
  it at once.** Everything in it was true and measured, which is why it grew that way:
  each section earned its place by recording something that had actually been run. But
  the effect on a newcomer was that the *first* question — what is this, will it work on
  my machine, what will I get — was answered somewhere around line 700. Installation sat
  below eight sections of captured output.

  It is now 441 lines and ordered by what a reader needs first: what the tool produces
  (a real unedited sample, before anything else, because a 4.26M-parameter model is not
  ChatGPT and that should be the first thing anyone learns), how to install it on each
  platform, the nine commands, then why the planner measures instead of estimating.
  Nothing measured was thrown away — the full worked example moved to
  [docs/walkthrough.md](docs/walkthrough.md), which is where every command's real output
  and the reasoning behind each decision now lives, and it is linked from the top of the
  pipeline section.

  The `--dry-run` plan block stayed in the README rather than moving with the rest,
  deliberately: it is the strongest single piece of evidence that the tool prints what it
  claims, a test re-renders it from the real preset table, and it costs a newcomer
  nothing behind a `<details>`. So did the WDDM oversubscription table, which is the one
  measurement that explains why this project exists at all.

- **The install section now says what to run on each platform, and macOS is answered
  rather than implied.** It gave one `pip install -e ".[dev]"` and left the virtualenv to
  the reader, which is fine on Linux and a guess on the other two. There are now
  PowerShell and POSIX spellings, a note that Microsoft Store Python has no `py`
  launcher, and a table of what `trainai setup` tells each kind of machine — NVIDIA,
  Apple Silicon, AMD on Linux, AMD on Windows, Intel Arc, and no GPU at all — with a
  column for whether anyone has run that path for real. Four of the six say no.

  macOS gets its own paragraph because it is the one where the honest answer is
  counter-intuitive: there is no separate download index and no CUDA version to match,
  MPS ships in the default wheel, so `setup` on a normal arm64 Python should report
  nothing to change. The reinstall command only appears when MPS is *missing*, which
  means an x86 Python under Rosetta. Every row of that table was checked by calling
  `advise_install` with the platform, torch build and driver version patched, rather than
  by reading the source — the Apple row was wrong on the first draft, which is how the
  distinction above got noticed.

- **`/reference/` is now ignored, so the old README can be kept on disk without being
  kept in the repository.** The original 887-line file lives there as
  `reference/README-original.md` for whoever wants to compare drafts. The rule is
  anchored to the repository root and names a directory rather than a pattern, so it can
  never start matching a `reference/` package inside `src/`. Nothing verified depends on
  it: what was worth keeping went to `docs/walkthrough.md`, which is tracked, and
  anything living only under `/reference/` is one `git clean -fdx` from being gone.

- **Documented that a rerun is not bitwise on CUDA, and proved that a resume is.**
  The README states that "a resumed run continues bitwise-identically", and the test
  carrying that milestone pinned `device="cpu"` — so the claim was asserted about the
  project and verified only where runs do not happen. It does hold on CUDA: an
  uninterrupted 24-step run and one resumed from step 12 agree on all 38 weight
  tensors and all 38 pairs of AdamW moment buffers, and there is now a
  `@pytest.mark.gpu` test saying so, including an assertion that it really ran on
  cuda so it cannot quietly become a copy of the CPU test.

  Rerunning is a different claim, and it does not hold. Three runs of the README's
  600-step command, same machine and same `--seed 1234`, gave best validation losses
  of 4.1365, 4.1361 and 4.1360 — bit-identical through step 120, first disagreeing at
  step 130, 1.5e-03 apart in training loss by step 600. `--seed` only ever claimed to
  fix initialisation, dropout and batch order, and it does; CUDA reductions are the
  rest. The README now gives those three numbers where it presents the run, so
  someone who reproduces it and gets 4.1360 can tell that they succeeded. `--device
  cpu` is the escape and was measured too: two 200-step runs bit-identical across all
  38 tensors and every logged loss, at 5,271 tokens/s against 90,168 on the GPU.

- **Re-preparing a corpus produces a different `content_hash` than before the CSV
  work.** `csv_text_column` joined the recorded ingest options, and those options
  are part of what the hash is computed over. Nothing re-verifies a stored hash, so
  **existing prepared datasets keep working** and `eval`'s resplit detection is
  unaffected — it compares a checkpoint's recorded hash against the dataset's stored
  hash, and neither changes. `trainai-dataset` stays at v1: the loader reads named
  keys, so an extra one is not a breaking change. The consequence is only that a
  dataset re-prepared from the same corpus will not report the hash it reported
  before, which is worth knowing if you have one written down.

- **And it moves once more, for the same reason, with the database reader.**
  `db_table` joined the recorded ingest options alongside `csv_text_column`, so a
  corpus re-prepared after this release reports a different `content_hash` again —
  including a plain `.txt` corpus that has nothing to do with databases, because the
  hash covers the options as recorded rather than the ones in use. Everything in the
  note above still holds: nothing re-verifies a stored hash, existing prepared
  datasets keep working, resplit detection is unaffected, and `trainai-dataset` stays
  at v1. This is the last option the M4 format work adds, so the hash settles here.

- The default `--max-doc-chars` is 16,384, down from 65,536. For a corpus that is
  one large file, this value sets the document count, and the train/validation
  split is chosen per document — at 64 KiB a 1 MB corpus yields 18 documents, and a
  5% validation share rounds to one document or, for some seeds, to none. The cost
  is one extra end-of-text token per cut, about 0.02% of the token stream.

- **`trainai doctor` no longer comments on `torch.compile`, and the
  `TrainConfig.compile_model` flag is gone.** Both described a feature that does not
  exist. `doctor` told anyone without Triton that "training works fine without it,
  just somewhat slower" — a real note on a real machine, printed for a cost that is
  exactly zero, because no code path in TrainAI calls `torch.compile`. The
  `compile_model` field offered to "try `torch.compile`" and was read by nothing: it
  serialised into every `plan.json` and every checkpoint config, and setting it did
  the same as leaving it alone. The probe still reports `torch_compile_available` in
  `doctor --json`, because whether the machine *could* compile is a true fact about
  the machine; what is removed is the claim that it matters to a run. Checkpoints and
  `plan.json` files that still carry `compile_model` load unchanged —
  `TrainConfig.from_dict` filters to the declared fields, so the stale key is dropped
  rather than refused. The flag comes back in the same commit as an implementation
  that has been measured, on the same reasoning that removed the `web` extra.

### Notes on what is deliberately absent

Commands that are not implemented are not registered in the CLI, so
`trainai --help` lists only what actually runs. The web interface (M5) and the docs
site, Docker image and packaging polish (M6) are still to come and are tracked in
the README roadmap.

**The `web` extra has been removed.** It declared `fastapi>=0.110` and
`uvicorn[standard]>=0.27` for M5, which the roadmap marks postponed, so
`pip install trainai[web]` installed an ASGI server plus its compiled transitive
dependencies and delivered nothing to run — searched, and there is not one
reference to fastapi, uvicorn or starlette anywhere under `src/`. Registering a
command that does not exist is something this project already refuses to do; an
extra is the same promise in the packaging metadata.
`test_every_declared_extra_is_one_the_code_uses` now fails on any extra whose
packages nothing under `src/` imports, and the reasoning for the eventual shape of
the extra is kept in [the dependency policy](docs/design/dependencies.md) so it
returns with the server rather than being rediscovered.

**SDPA's `enable_gqa` is deliberately not used.** Grouped-query attention expands
the key and value heads with `repeat_interleave` before calling attention, which
allocates a copy; `enable_gqa=True` would skip that copy and looks like the obvious
improvement. Measured, it is the opposite. The memory-efficient kernel — the one
that actually runs on a consumer GPU — refuses mismatched head counts, so SDPA
falls back to the math path and materialises the full
batch x heads x seq x seq attention matrix. One attention call at batch 8, seq 512
on the development RTX 2050: expanded stays fused at +12.0 MiB, `enable_gqa` costs
+189.2 MiB. Over a real training step at the default shape, 1,752 MiB against
2,997 MiB and 0.20 s against 0.35 s per step. cuDNN's kernel does accept it but is
runtime-disabled on this build, and on a card whose flash kernel is available the
trade may go the other way — hence a comment carrying the numbers rather than a
silent choice. The grouping is also now pinned by a test: swapping
`repeat_interleave` for `repeat` keeps every tensor shape, scrambles which key each
query head reads, and used to pass all 1,280 tests.

Three limits are worth stating rather than leaving to be discovered.

The planner's time estimate is measured step time times step count, so it excludes
the dataset reads and validation passes the real run also does — on the Shakespeare
run it predicted 14 s against 16 s actual. Its learning rate is a width-scaling rule
of thumb, not a measurement; the plan labels it as one, and only the loss curve can
settle it.

The export's logits-parity check needs `transformers`, which is not a dependency.
Where it is missing, the weights and the tokenizer are still verified and the parity
check reports itself as not run. The agreement figures quoted above are from
`transformers` 5.3.0 on the development machine.

And the estimate of *which* step the best checkpoint will come from is a rule of
thumb that has been wrong: exact on one run (step 300 predicted, step 300 actual),
pessimistic by 2.3x on another (step 163 predicted, step 375 actual). What it gets
right is that the last checkpoint will not be the best one, which is what TrainAI
acts on — the best is tracked and kept wherever it lands.

One correction to the record: the commit message for `0668603` names the development
GPU as an "RTX 3050 Ti". It is an RTX 2050 (compute capability 8.6, 4 GiB), as stated
everywhere else here.

[Unreleased]: https://github.com/shubhampardule/trainAi/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shubhampardule/trainAi/releases/tag/v0.1.0
