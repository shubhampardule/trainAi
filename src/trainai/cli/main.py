"""The ``trainai`` command-line entry point.

The interesting part of this module is :func:`main`, which converts exceptions
into stable exit codes and renders :class:`~trainai.errors.TrainAIError` without
a traceback. Commands themselves live in sibling modules and are kept thin --
they parse arguments, call into the library, and print. Business logic in a CLI
module is a smell here, because the web interface (M5) has to call the same code.
"""

from __future__ import annotations

from enum import Enum

import typer

from trainai import __version__
from trainai.cli._click import Abort, ClickException, Exit
from trainai.console import console, err_console, render_error
from trainai.errors import ExitCode, TrainAIError

app = typer.Typer(
    name="trainai",
    help=(
        "Train small language models from scratch on your own hardware.\n\n"
        "Start with [bold]trainai doctor[/bold] to see what your machine can do."
    ),
    no_args_is_help=True,
    add_completion=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

data_app = typer.Typer(
    name="data",
    # Not "prepare, inspect and validate": there is no `trainai data validate`. Both
    # commands validate what they read and report what they find, which is the honest
    # way to say it -- naming a verb that is not a command sends the user to a
    # "No such command" error looking for a feature that is really there. Kept to one
    # short sentence because Typer prints this line inside the root `--help` table too,
    # where anything longer is truncated with an ellipsis.
    help="Prepare and inspect text datasets, reporting problems.",
    no_args_is_help=True,
)
app.add_typer(data_app)


class OnErrorPolicy(str, Enum):
    """What to do with a source file that cannot be decoded.

    Lives in this module rather than beside the ingest code because its only job
    is to render the choice list in ``--help``. Keeping it here is what lets
    ``trainai --help`` stay free of numpy and the tokenizers extension: nothing
    under ``trainai.data`` is imported until a command actually runs.
    """

    fail = "fail"
    skip = "skip"


# --------------------------------------------------------------------------- #
# Global state set by the root callback
# --------------------------------------------------------------------------- #
class _Settings:
    """Process-wide CLI flags. Deliberately tiny."""

    verbose: bool = False


settings = _Settings()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"trainai {__version__}")
        raise typer.Exit()


@app.callback()
def root(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the TrainAI version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show structured error details and extra diagnostics.",
    ),
) -> None:
    """TrainAI: dataset in, trained language model out."""
    settings.verbose = verbose


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
@app.command()
def setup(
    install: bool = typer.Option(
        False,
        "--install",
        help="Run the recommended install command, after showing it and confirming.",
    ),
    allow_global: bool = typer.Option(
        False,
        "--allow-global",
        help="Permit installing outside a virtual environment. Refused by default.",
    ),
    assume_yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt for --install."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit the diagnosis as JSON instead of a report."
    ),
) -> None:
    """Check that the installed PyTorch matches this machine's hardware.

    The case this catches: a good GPU plus the CPU-only PyTorch wheel, which trains
    20-100x slower and looks exactly like having no GPU from inside PyTorch. This
    checks the system for a GPU independently, and prints the install command that
    matches what it finds.

    Prints the command by default. [bold]--install[/bold] runs it, after confirming
    and after refusing to touch a system Python.
    """
    from trainai.cli.setup import run_setup

    run_setup(
        install=install,
        allow_global=allow_global,
        assume_yes=assume_yes,
        json_output=json_output,
    )


@app.command()
def doctor(
    json_output: bool = typer.Option(
        False, "--json", help="Emit the hardware profile as JSON instead of a table."
    ),
    path: str = typer.Option(
        ".",
        "--path",
        help="Measure free disk space for this location (where runs will be written).",
    ),
) -> None:
    """Report what this machine can actually train, and flag anything that will bite.

    Run this first. It imports PyTorch and initialises a CUDA context, so it takes
    a few seconds and briefly allocates a small amount of VRAM.
    """
    from trainai.cli.doctor import run_doctor

    run_doctor(json_output=json_output, disk_path=path)


# --------------------------------------------------------------------------- #
# quickstart
#
# Placed after `doctor` in this file, and therefore in `--help`, because that is the
# order a first-time user wants: see what the machine can do, then do it. It takes no
# flag that is not a pass-through to one of the five commands it runs.
# --------------------------------------------------------------------------- #
@app.command()
def quickstart(
    corpus: str = typer.Argument(
        ...,
        metavar="CORPUS",
        help=(
            "Text to train on, in any format `trainai data prepare` accepts. "
            "There is no default: the corpus is yours to supply, and nothing in "
            "TrainAI downloads one."
        ),
    ),
    out: str = typer.Option(
        "quickstart",
        "--out",
        "-o",
        metavar="DIR",
        help="Directory to put the dataset, the plan and the run in.",
    ),
    time_budget: str | None = typer.Option(
        None,
        "--time",
        metavar="DURATION",
        help="How long you are willing to train for: 90s, 45m, 2h, 1h30m, or bare minutes.",
    ),
    steps: int | None = typer.Option(
        None, "--steps", help="Override the step count the plan chose."
    ),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        "-p",
        metavar="TEXT",
        help="Prompt for the sample at the end. Defaults to your corpus's own opening.",
    ),
    tokens: int | None = typer.Option(
        None, "--tokens", "-n", metavar="N", help="Tokens to generate in that sample."
    ),
    encoding: str = typer.Option("utf-8", "--encoding", help="Text encoding of the source files."),
    jsonl_field: str | None = typer.Option(
        None,
        "--jsonl-field",
        help="Field holding the text in .jsonl or .json records. Auto-detected when unset.",
    ),
    jsonl_messages_field: str | None = typer.Option(
        None,
        "--jsonl-messages-field",
        metavar="NAME",
        help=(
            "Field holding a conversation: a list of {role, content} objects, "
            "rendered with TrainAI's chat template."
        ),
    ),
    csv_text_column: str | None = typer.Option(
        None,
        "--csv-text-column",
        metavar="NAME",
        help="Column holding the text in a .csv, .tsv or database table.",
    ),
    db_table: str | None = typer.Option(
        None,
        "--db-table",
        metavar="NAME",
        help="Table to read from a SQLite database. Needed only when it holds more than one.",
    ),
    device: str | None = typer.Option(
        None, "--device", help="auto, cuda, cpu, mps or xpu. Left to the plan when unset."
    ),
    precision: str | None = typer.Option(
        None,
        "--precision",
        help="auto, bf16, fp16 or fp32. Left to what the plan measured when unset.",
    ),
    assume_yes: bool = typer.Option(
        False, "--yes", "-y", help="Do not stop to confirm before the training step."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Redo the dataset and replace an existing run."
    ),
) -> None:
    """Go from a corpus to text a model wrote, in one command.

    Five steps: prepare the corpus, measure this machine, train, score the result, and
    generate a sample. Each one prints the command it stands in for before it runs, so
    this is a shortcut through the CLI rather than a wizard around it -- change one thing
    and you can copy the printed line and carry on by hand.

    It stops once to confirm, just before training, quoting the [bold]measured[/bold]
    time the run will take. There is no [bold]--json[/bold]: use the individual commands
    for anything scripted.
    """
    from trainai.cli.quickstart import run_quickstart

    run_quickstart(
        corpus,
        out=out,
        time_budget=time_budget,
        steps=steps,
        prompt=prompt,
        tokens=tokens,
        encoding=encoding,
        jsonl_field=jsonl_field,
        jsonl_messages_field=jsonl_messages_field,
        csv_text_column=csv_text_column,
        db_table=db_table,
        device=device,
        precision=precision,
        assume_yes=assume_yes,
        force=force,
    )


# --------------------------------------------------------------------------- #
# data prepare / data inspect
#
# The defaults below are literals rather than imports so that `--help` does not
# pay for numpy and the tokenizers extension. Where a default belongs to the
# library, `tests/test_cli.py` asserts the two agree, so the number cannot drift.
# --------------------------------------------------------------------------- #
@data_app.command("prepare")
def data_prepare(
    corpus: str = typer.Argument(
        ...,
        metavar="CORPUS",
        help=(
            "Text to train on: a .txt, .jsonl, .json or .csv file (optionally "
            "compressed), a directory of them, or a .zip/.tar.gz of them."
        ),
    ),
    out: str = typer.Option(
        ..., "--out", "-o", metavar="DIR", help="Where to write the prepared dataset."
    ),
    vocab_size: int = typer.Option(
        8192,
        "--vocab-size",
        help=(
            "Target BPE vocabulary, including the 256 byte tokens and <|endoftext|>. "
            "Minimum 258, leaving room for one merge; useful sizes start around 4096."
        ),
    ),
    min_frequency: int = typer.Option(
        2, "--min-frequency", help="A pair must occur this often to be learned as a merge."
    ),
    val_fraction: float = typer.Option(
        0.05,
        "--val-fraction",
        help="Share of documents held out for validation. 0 disables the held-out split.",
    ),
    seed: int = typer.Option(
        1234,
        "--seed",
        help="Seeds the train/val split. The same seed on the same corpus reproduces it.",
    ),
    encoding: str = typer.Option("utf-8", "--encoding", help="Text encoding of the source files."),
    jsonl_field: str | None = typer.Option(
        None,
        "--jsonl-field",
        help="Field holding the text in .jsonl or .json records. Auto-detected when unset.",
    ),
    jsonl_messages_field: str | None = typer.Option(
        None,
        "--jsonl-messages-field",
        metavar="NAME",
        help=(
            "Field holding a conversation: a list of {role, content} objects, "
            "rendered with TrainAI's chat template."
        ),
    ),
    csv_text_column: str | None = typer.Option(
        None,
        "--csv-text-column",
        metavar="NAME",
        help=(
            "Column holding the text in a .csv, .tsv or database table. "
            "One column, named; nothing is joined."
        ),
    ),
    db_table: str | None = typer.Option(
        None,
        "--db-table",
        metavar="NAME",
        help="Table to read from a SQLite database. Needed only when it holds more than one.",
    ),
    min_doc_chars: int = typer.Option(
        1, "--min-doc-chars", help="Drop documents shorter than this many characters."
    ),
    max_doc_chars: int = typer.Option(
        16384,
        "--max-doc-chars",
        help="Cut longer files into several documents, at paragraph boundaries.",
    ),
    on_error: OnErrorPolicy = typer.Option(
        OnErrorPolicy.fail,
        "--on-error",
        help="What to do with a file that cannot be decoded.",
    ),
    shard_tokens: int = typer.Option(67108864, "--shard-tokens", help="Tokens per shard file."),
    force: bool = typer.Option(
        False, "--force", "-f", help="Replace an existing prepared dataset in --out."
    ),
    allow_tabular: bool = typer.Option(
        False,
        "--allow-tabular",
        help="Prepare a corpus of table rows anyway. Recorded in the manifest.",
    ),
    loss_mask: bool | None = typer.Option(
        None,
        "--loss-mask/--no-loss-mask",
        help="Write a mask marking which tokens are assistant replies. On by default "
        "for a corpus read with --jsonl-messages-field, off otherwise.",
    ),
    tokenizer: str | None = typer.Option(
        None,
        "--tokenizer",
        metavar="PATH",
        help="Reuse an existing tokenizer.json instead of training one. Required to "
        "prepare a fine-tuning dataset: the base model's embedding rows are its "
        "token ids. Point at the file or at the dataset directory holding it.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit the manifest as JSON and print nothing else."
    ),
) -> None:
    """Turn text into token shards a model can train on.

    Reads the corpus twice: once to measure it and train the tokenizer, once to
    encode it. Writes the shards, [bold]tokenizer.json[/bold] and a checksummed
    [bold]manifest.json[/bold] into [bold]--out[/bold]. The same corpus with the
    same settings and seed produces byte-identical shards.

    [bold]--tokenizer[/bold] reuses one instead of training a fresh one. A dataset
    prepared without it gets its own vocabulary, which no existing checkpoint's
    embedding matrix matches -- so fine-tuning an earlier run means preparing the new
    corpus with that run's [bold]tokenizer.json[/bold].
    """
    from trainai.cli.data import run_prepare

    run_prepare(
        corpus,
        out,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        val_fraction=val_fraction,
        seed=seed,
        encoding=encoding,
        jsonl_field=jsonl_field,
        jsonl_messages_field=jsonl_messages_field,
        csv_text_column=csv_text_column,
        db_table=db_table,
        min_doc_chars=min_doc_chars,
        max_doc_chars=max_doc_chars,
        on_error=on_error.value,
        shard_tokens=shard_tokens,
        force=force,
        json_output=json_output,
        allow_tabular=allow_tabular,
        tokenizer=tokenizer,
        loss_mask=loss_mask,
    )


@data_app.command("inspect")
def data_inspect(
    path: str = typer.Argument(
        ...,
        metavar="PATH",
        help="A raw corpus, or a directory produced by `trainai data prepare`.",
    ),
    verify: bool = typer.Option(
        False,
        "--verify",
        help="For a prepared dataset, re-hash every shard instead of only checking sizes.",
    ),
    layout: bool = typer.Option(
        False, "--layout", help="Also explain what each file in a prepared dataset is."
    ),
    sample_chars: int = typer.Option(
        0,
        "--sample",
        metavar="N",
        help="For a raw corpus, print the first N characters of the first document.",
    ),
    encoding: str = typer.Option("utf-8", "--encoding", help="Text encoding of the source files."),
    jsonl_field: str | None = typer.Option(
        None,
        "--jsonl-field",
        help="Field holding the text in .jsonl or .json records. Auto-detected when unset.",
    ),
    jsonl_messages_field: str | None = typer.Option(
        None,
        "--jsonl-messages-field",
        metavar="NAME",
        help=(
            "Field holding a conversation: a list of {role, content} objects, "
            "rendered with TrainAI's chat template."
        ),
    ),
    csv_text_column: str | None = typer.Option(
        None,
        "--csv-text-column",
        metavar="NAME",
        help=(
            "Column holding the text in a .csv, .tsv or database table. "
            "One column, named; nothing is joined."
        ),
    ),
    db_table: str | None = typer.Option(
        None,
        "--db-table",
        metavar="NAME",
        help="Table to read from a SQLite database. Needed only when it holds more than one.",
    ),
    min_doc_chars: int = typer.Option(
        1, "--min-doc-chars", help="Drop documents shorter than this many characters."
    ),
    max_doc_chars: int = typer.Option(
        16384,
        "--max-doc-chars",
        help="Cut longer files into several documents, at paragraph boundaries.",
    ),
    on_error: OnErrorPolicy = typer.Option(
        OnErrorPolicy.fail,
        "--on-error",
        help="What to do with a file that cannot be decoded.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit the report as JSON and print nothing else."
    ),
) -> None:
    """Report on a corpus or a prepared dataset. Writes nothing.

    Which kind it is depends on whether the directory holds a
    [bold]manifest.json[/bold]. For a raw corpus this does the same full read that
    preparation does, minus tokenization, so the numbers are counted rather than
    sampled -- and it exits 3 if the corpus cannot be trained on.
    """
    from trainai.cli.data import run_inspect

    run_inspect(
        path,
        verify=verify,
        layout=layout,
        sample_chars=sample_chars,
        encoding=encoding,
        jsonl_field=jsonl_field,
        jsonl_messages_field=jsonl_messages_field,
        csv_text_column=csv_text_column,
        db_table=db_table,
        min_doc_chars=min_doc_chars,
        max_doc_chars=max_doc_chars,
        on_error=on_error.value,
        json_output=json_output,
    )


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
@app.command()
def plan(
    data: str = typer.Option(
        ...,
        "--data",
        "-d",
        metavar="DIR",
        help="A dataset directory produced by `trainai data prepare`.",
    ),
    time_budget: str | None = typer.Option(
        None,
        "--time",
        metavar="DURATION",
        help=(
            "How long you are willing to train for: [bold]90s[/], [bold]45m[/], "
            "[bold]2h[/], [bold]1h30m[/], or a bare number of minutes. Bounds the model "
            "size and, if needed, cuts the step count."
        ),
    ),
    max_preset: str | None = typer.Option(
        None,
        "--max-preset",
        metavar="NAME",
        help="Stop the ladder at this preset: tiny, small, medium or large.",
    ),
    max_vram: str | None = typer.Option(
        None,
        "--max-vram",
        metavar="SIZE",
        help=(
            "Use at most this much VRAM: [bold]6GB[/], [bold]6.5GiB[/], [bold]512MB[/], or "
            "a bare number of gigabytes. For sharing the GPU with a desktop or another "
            "job. Only ever lowers the budget, never raises it past what the card has."
        ),
    ),
    seq_len: int | None = typer.Option(
        None,
        "--seq-len",
        metavar="N",
        help="Force a sequence length instead of using each preset's own context.",
    ),
    out: str | None = typer.Option(
        None,
        "--out",
        "-o",
        metavar="PATH",
        help="Where to write the plan. Defaults to ./plan.json.",
    ),
    device: str = typer.Option(
        "auto", "--device", help="Measure on this device: auto, cuda, cpu, mps, xpu."
    ),
    precision: str = typer.Option(
        "auto", "--precision", help="Measure at this precision: auto, bf16, fp16, fp32."
    ),
    measure_steps: int = typer.Option(
        3, "--measure-steps", help="Timed steps per candidate, after warmup."
    ),
    warmup_steps: int = typer.Option(
        2,
        "--warmup-steps",
        help="Untimed steps per candidate. The first step pays for kernel autotuning "
        "and AdamW's lazily allocated moments, so timing it would be misleading.",
    ),
    verify: bool = typer.Option(
        False, "--verify", help="Re-hash the dataset shards before planning."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit the plan as JSON instead of a report."
    ),
) -> None:
    """Measure what this machine can train, then print a command to review.

    Runs real training steps at a ladder of candidate configurations and keeps the
    largest one that actually worked, reading the allocator's own counters rather than
    estimating from a formula. Takes a minute or two and briefly allocates VRAM.

    The reported time to finish is the measured step time times the step count. It is
    not extrapolated, and it excludes dataset read time.
    """
    from trainai.cli.plan import run_plan

    run_plan(
        data,
        device=device,
        time_budget=time_budget,
        max_preset=max_preset,
        seq_len=seq_len,
        precision=precision,
        json_output=json_output,
        out=out,
        verify=verify,
        max_vram=max_vram,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
    )


# --------------------------------------------------------------------------- #
# train
#
# Flags are grouped into help panels because there are a lot of them, and a flat
# list of thirty options is not something a first-time user can read. Every one
# defaults to None here and is filled in by `trainai.cli.train`, so "not given"
# is distinguishable from "given the same value as the default" -- which is what
# lets the step count and the sequence length be derived from the dataset.
# --------------------------------------------------------------------------- #
@app.command()
def train(
    data: str = typer.Option(
        ...,
        "--data",
        "-d",
        metavar="DIR",
        help="A dataset directory produced by `trainai data prepare`.",
    ),
    out: str | None = typer.Option(
        None, "--out", "-o", metavar="DIR", help="Run directory. Defaults to runs/<name>."
    ),
    name: str | None = typer.Option(None, "--name", help="Name for the run directory under runs/."),
    resume: str | None = typer.Option(
        None,
        "--resume",
        metavar="PATH",
        help="Continue from a run directory, a checkpoints directory, or a step-*.pt file.",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite an existing run directory."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report the plan -- shape, parameters, epochs -- and exit without training.",
    ),
    verify: bool = typer.Option(
        False, "--verify", help="Re-hash the dataset's shards before starting."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit the result as JSON and print nothing else."
    ),
    model_preset: str | None = typer.Option(
        None,
        "--preset",
        rich_help_panel="Model",
        help=(
            "Starting shape: tiny, small, medium or large. Any flag below overrides it. "
            "Defaults to tiny; use `trainai plan` to pick one from measurements."
        ),
    ),
    plan: str | None = typer.Option(
        None,
        "--plan",
        metavar="PATH",
        rich_help_panel="Model",
        help=(
            "Apply a plan.json written by `trainai plan`, instead of a preset. Any flag "
            "you also give overrides that part of the plan."
        ),
    ),
    n_layer: int | None = typer.Option(
        None, "--layers", rich_help_panel="Model", help="Transformer blocks."
    ),
    n_head: int | None = typer.Option(
        None, "--heads", rich_help_panel="Model", help="Query heads."
    ),
    n_kv_head: int | None = typer.Option(
        None,
        "--kv-heads",
        rich_help_panel="Model",
        help="Key/value heads. Fewer than --heads gives grouped-query attention.",
    ),
    d_model: int | None = typer.Option(
        None, "--width", rich_help_panel="Model", help="Residual stream width (d_model)."
    ),
    d_ff: int | None = typer.Option(
        None,
        "--ffn-width",
        rich_help_panel="Model",
        help="SwiGLU hidden width. Defaults to 8/3 of --width, rounded up to a multiple of 64.",
    ),
    context: int | None = typer.Option(
        None,
        "--context",
        rich_help_panel="Model",
        help="Context length the model is built for.",
    ),
    dropout: float | None = typer.Option(
        None,
        "--dropout",
        rich_help_panel="Model",
        help="0 suits a corpus seen once; raise it when looping over a small one.",
    ),
    tie_embeddings: bool | None = typer.Option(
        None,
        "--tie-embeddings/--no-tie-embeddings",
        rich_help_panel="Model",
        help="Share the input embedding with the output projection. On by default.",
    ),
    steps: int | None = typer.Option(
        None,
        "--steps",
        rich_help_panel="Training",
        help=(
            "Optimizer steps. Defaults to about three passes over the training split, "
            "capped at 20,000 -- on a large corpus that cap means well under one pass, "
            "and --dry-run reports how many you actually get."
        ),
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        "-b",
        rich_help_panel="Training",
        help="Sequences per forward pass. Sized to fit VRAM.",
    ),
    grad_accum: int | None = typer.Option(
        None,
        "--grad-accum",
        rich_help_panel="Training",
        help="Forward passes summed per optimizer step. Effective batch is the product.",
    ),
    seq_len: int | None = typer.Option(
        None,
        "--seq-len",
        rich_help_panel="Training",
        help="Tokens per sequence. Defaults to the model's context, capped at 256.",
    ),
    lr: float | None = typer.Option(
        None, "--lr", rich_help_panel="Training", help="Peak learning rate."
    ),
    min_lr_ratio: float | None = typer.Option(
        None,
        "--min-lr-ratio",
        rich_help_panel="Training",
        help="Floor as a fraction of --lr.",
    ),
    warmup_steps: int | None = typer.Option(
        None,
        "--warmup",
        rich_help_panel="Training",
        help="Steps spent ramping the learning rate up from zero.",
    ),
    schedule: str | None = typer.Option(
        None,
        "--schedule",
        rich_help_panel="Training",
        help="cosine, linear or constant.",
    ),
    weight_decay: float | None = typer.Option(
        None,
        "--weight-decay",
        rich_help_panel="Training",
        help="AdamW decay. Applied to matrices only, never to norms or embeddings.",
    ),
    grad_clip: float | None = typer.Option(
        None,
        "--grad-clip",
        rich_help_panel="Training",
        help="Global gradient-norm clip. 0 disables it.",
    ),
    eval_every: int | None = typer.Option(
        None,
        "--eval-every",
        rich_help_panel="Training",
        help="Steps between validation passes. 0 measures nothing held out.",
    ),
    eval_batches: int | None = typer.Option(
        None,
        "--eval-batches",
        rich_help_panel="Training",
        help="Validation batches per pass.",
    ),
    checkpoint_every: int | None = typer.Option(
        None,
        "--checkpoint-every",
        rich_help_panel="Training",
        help="Steps between checkpoints. 0 saves only at the end.",
    ),
    keep_checkpoints: int | None = typer.Option(
        None,
        "--keep-checkpoints",
        rich_help_panel="Training",
        help="Step checkpoints to retain. The best and the newest are always kept.",
    ),
    log_every: int | None = typer.Option(
        None, "--log-every", rich_help_panel="Training", help="Steps between metric records."
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        rich_help_panel="Training",
        help="Seeds initialisation, dropout and batch order.",
    ),
    loss_mask: bool | None = typer.Option(
        None,
        "--loss-mask/--no-loss-mask",
        rich_help_panel="Training",
        help="Score only the tokens the dataset marks as targets -- the assistant's "
        "replies in a chat corpus. Applied automatically when the dataset has a mask; "
        "--loss-mask requires one, --no-loss-mask scores every token.",
    ),
    precision: str | None = typer.Option(
        None,
        "--precision",
        rich_help_panel="Hardware",
        help="auto, bf16, fp16 or fp32. auto picks what the GPU actually supports.",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        rich_help_panel="Hardware",
        help="auto, cuda, cpu, mps or xpu. An explicit choice is never silently downgraded.",
    ),
) -> None:
    """Train a language model from scratch on a prepared dataset.

    Writes checkpoints and a JSONL metrics log into the run directory. Use
    [bold]--dry-run[/bold] first to see the model shape, the parameter count and how
    many passes over your data the run will make -- that takes two seconds and
    catches a badly proportioned run before it costs an evening.
    """
    from trainai.cli.train import run_train

    run_train(
        data,
        out=out,
        name=name,
        resume=resume,
        force=force,
        dry_run=dry_run,
        json_output=json_output,
        verify=verify,
        model_preset=model_preset,
        plan=plan,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        d_model=d_model,
        d_ff=d_ff,
        context=context,
        dropout=dropout,
        tie_embeddings=tie_embeddings,
        steps=steps,
        batch_size=batch_size,
        grad_accum=grad_accum,
        seq_len=seq_len,
        lr=lr,
        min_lr_ratio=min_lr_ratio,
        warmup_steps=warmup_steps,
        schedule=schedule,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        eval_every=eval_every,
        eval_batches=eval_batches,
        checkpoint_every=checkpoint_every,
        keep_checkpoints=keep_checkpoints,
        log_every=log_every,
        seed=seed,
        loss_mask=loss_mask,
        precision=precision,
        device=device,
    )


# --------------------------------------------------------------------------- #
# finetune
# --------------------------------------------------------------------------- #
@app.command()
def finetune(
    data: str = typer.Option(
        ...,
        "--data",
        "-d",
        metavar="DIR",
        help=(
            "A dataset directory produced by `trainai data prepare --tokenizer "
            "<the base model's tokenizer.json>`. Preparing it without --tokenizer "
            "gives it its own vocabulary, which the base model's embedding rows do "
            "not match, and this command will refuse it."
        ),
    ),
    base: str = typer.Option(
        ...,
        "--from",
        metavar="PATH",
        help="Base model: a run directory, a checkpoints directory, or a step-*.pt file.",
    ),
    out: str | None = typer.Option(
        None, "--out", "-o", metavar="DIR", help="Run directory. Defaults to runs/<name>."
    ),
    name: str | None = typer.Option(None, "--name", help="Name for the run directory under runs/."),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite an existing run directory."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report the plan -- the base model's shape, the schedule, the data budget -- and exit.",
    ),
    verify: bool = typer.Option(
        False, "--verify", help="Re-hash the dataset's shards before starting."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit the result as JSON and print nothing else."
    ),
    steps: int | None = typer.Option(
        None,
        "--steps",
        rich_help_panel="Training",
        help="Optimizer steps. Defaults to about three passes over the training split.",
    ),
    batch_size: int | None = typer.Option(
        None, "--batch-size", "-b", rich_help_panel="Training", help="Sequences per forward pass."
    ),
    grad_accum: int | None = typer.Option(
        None,
        "--grad-accum",
        rich_help_panel="Training",
        help="Forward passes summed per optimizer step. Effective batch is the product.",
    ),
    seq_len: int | None = typer.Option(
        None,
        "--seq-len",
        rich_help_panel="Training",
        help="Tokens per sequence. Cannot exceed the base model's context length.",
    ),
    lr: float | None = typer.Option(
        None,
        "--lr",
        rich_help_panel="Training",
        help=(
            "Peak learning rate. Defaults to 3e-5, a tenth of the from-scratch default: "
            "on this project's own measurement a higher rate learns the new corpus faster "
            "and loses more of the base model, monotonically in both directions."
        ),
    ),
    min_lr_ratio: float | None = typer.Option(
        None, "--min-lr-ratio", rich_help_panel="Training", help="Floor as a fraction of --lr."
    ),
    warmup_steps: int | None = typer.Option(
        None,
        "--warmup",
        rich_help_panel="Training",
        help="Steps spent ramping up from zero. A fine-tune warms up from scratch: "
        "its schedule is new, not a continuation of the base model's.",
    ),
    schedule: str | None = typer.Option(
        None, "--schedule", rich_help_panel="Training", help="cosine, linear or constant."
    ),
    weight_decay: float | None = typer.Option(
        None, "--weight-decay", rich_help_panel="Training", help="AdamW decay."
    ),
    grad_clip: float | None = typer.Option(
        None, "--grad-clip", rich_help_panel="Training", help="Global gradient-norm clip."
    ),
    eval_every: int | None = typer.Option(
        None,
        "--eval-every",
        rich_help_panel="Training",
        help="Steps between validation passes. 0 measures nothing held out.",
    ),
    eval_batches: int | None = typer.Option(
        None, "--eval-batches", rich_help_panel="Training", help="Validation batches per pass."
    ),
    checkpoint_every: int | None = typer.Option(
        None,
        "--checkpoint-every",
        rich_help_panel="Training",
        help="Steps between checkpoints. 0 saves only at the end.",
    ),
    keep_checkpoints: int | None = typer.Option(
        None,
        "--keep-checkpoints",
        rich_help_panel="Training",
        help="Step checkpoints to retain. The best and the newest are always kept.",
    ),
    log_every: int | None = typer.Option(
        None, "--log-every", rich_help_panel="Training", help="Steps between metric records."
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        rich_help_panel="Training",
        help="Seeds dropout and batch order. Initialisation comes from the checkpoint.",
    ),
    loss_mask: bool | None = typer.Option(
        None,
        "--loss-mask/--no-loss-mask",
        rich_help_panel="Training",
        help="Score only the tokens the dataset marks as targets -- the assistant's "
        "replies in a chat corpus. Applied automatically when the dataset has a mask; "
        "--loss-mask requires one, --no-loss-mask scores every token.",
    ),
    precision: str | None = typer.Option(
        None, "--precision", rich_help_panel="Hardware", help="auto, bf16, fp16 or fp32."
    ),
    device: str | None = typer.Option(
        None, "--device", rich_help_panel="Hardware", help="auto, cuda, cpu, mps or xpu."
    ),
) -> None:
    """Continue training an existing model on a different dataset.

    This is not [bold]--resume[/bold]. Resuming continues one run: same data, same
    schedule, the optimizer's moments and the random streams restored so that the
    next step is the step that would have happened anyway. Fine-tuning is the
    opposite claim -- new data, new schedule -- so only the weights come across, and
    the learning rate warms up from zero again.

    There are no model flags: the shape is whatever the checkpoint was built as.
    The one thing that must match is the tokenizer, because a token id is a row
    index into the embedding matrix -- the same id read through a different
    tokenizer selects a vector trained for some other piece of text, and nothing
    about the loss curve would look wrong. Prepare the dataset with
    [bold]data prepare --tokenizer[/bold] and that holds by construction.
    """
    from trainai.cli.train import run_finetune

    run_finetune(
        data,
        base=base,
        out=out,
        name=name,
        force=force,
        dry_run=dry_run,
        json_output=json_output,
        verify=verify,
        steps=steps,
        batch_size=batch_size,
        grad_accum=grad_accum,
        seq_len=seq_len,
        lr=lr,
        min_lr_ratio=min_lr_ratio,
        warmup_steps=warmup_steps,
        schedule=schedule,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        eval_every=eval_every,
        eval_batches=eval_batches,
        checkpoint_every=checkpoint_every,
        keep_checkpoints=keep_checkpoints,
        log_every=log_every,
        seed=seed,
        loss_mask=loss_mask,
        precision=precision,
        device=device,
    )


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
@app.command("eval")
def eval_command(
    run: str = typer.Argument(
        ...,
        metavar="RUN",
        help="A run directory, a checkpoints directory, or a single step-*.pt file.",
    ),
    data: str | None = typer.Option(
        None,
        "--data",
        "-d",
        metavar="DIR",
        help="Dataset to score against. Defaults to the one the checkpoint records.",
    ),
    which: str = typer.Option(
        "best",
        "--which",
        help="Which checkpoint: best (lowest validation loss) or latest.",
    ),
    split: str = typer.Option(
        "val",
        "--split",
        help="val, train, or both. `both` is how you see overfitting.",
    ),
    batch_size: int = typer.Option(
        8,
        "--batch-size",
        "-b",
        help="Sequences per forward pass. Does not change the result, only the speed.",
    ),
    max_batches: int | None = typer.Option(
        None,
        "--max-batches",
        metavar="N",
        help="Stop after N batches per split. Scores the whole split when unset.",
    ),
    seq_len: int | None = typer.Option(
        None,
        "--seq-len",
        metavar="N",
        help="Context to score at. Defaults to the context the run actually trained at, "
        "which is not always the one the model was built for.",
    ),
    device: str = typer.Option("auto", "--device", help="auto, cuda, cpu, mps or xpu."),
    precision: str = typer.Option(
        "auto",
        "--precision",
        help="auto, bf16, fp16 or fp32. fp32 pins the number; reduced precision moves "
        "it in the third decimal.",
    ),
    tokenizer: str | None = typer.Option(
        None,
        "--tokenizer",
        metavar="PATH",
        help="A tokenizer.json to use, for a run whose dataset moved.",
    ),
    verify: bool = typer.Option(
        False, "--verify", help="Re-hash the dataset's shards before scoring."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit the report as JSON and print nothing else."
    ),
) -> None:
    """Measure a trained run on held-out text: loss, perplexity, bits per token.

    Recomputes the same quantity the training curve reports, over a whole split rather
    than the few batches the trainer samples. The report carries the vocabulary size,
    the context and the precision, because a perplexity without those cannot be
    compared to another one -- it is per token of one tokenizer.

    It also states how much of the split it scored. Use [bold]--split both[/bold] to
    see the training and validation numbers side by side.
    """
    from trainai.cli.eval import run_eval

    run_eval(
        run,
        data=data,
        which=which,
        split=split,
        batch_size=batch_size,
        max_batches=max_batches,
        seq_len=seq_len,
        device=device,
        precision=precision,
        tokenizer=tokenizer,
        verify=verify,
        json_output=json_output,
    )


# --------------------------------------------------------------------------- #
# chat
#
# Sampling defaults are literals here for the same reason as everywhere else in
# this file -- `--help` should not import torch -- and `tests/test_cli.py` asserts
# they match `trainai.infer`, so they cannot drift apart silently.
# --------------------------------------------------------------------------- #
@app.command()
def chat(
    run: str = typer.Argument(
        ...,
        metavar="RUN",
        help="A run directory, a checkpoints directory, or a single step-*.pt file.",
    ),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        "-p",
        metavar="TEXT",
        help="Generate once from TEXT and exit, instead of starting the playground.",
    ),
    which: str = typer.Option(
        "best",
        "--which",
        help="Which checkpoint: best (lowest validation loss) or latest.",
    ),
    tokens: int = typer.Option(
        200,
        "--tokens",
        "-n",
        metavar="N",
        help="How many tokens to generate at most.",
    ),
    temperature: float = typer.Option(
        0.8,
        "--temperature",
        "-t",
        metavar="X",
        help="0 is greedy, 0.8 is the usual sample, above 1.2 wanders.",
    ),
    top_k: int | None = typer.Option(
        None, "--top-k", metavar="N", help="Keep only the N most likely tokens. Off by default."
    ),
    top_p: float | None = typer.Option(
        0.95,
        "--top-p",
        metavar="X",
        help="Nucleus sampling threshold. Use 1.0 to disable it.",
    ),
    penalty: float = typer.Option(
        1.1,
        "--penalty",
        metavar="X",
        help="Repetition penalty. 1.0 is off; a small model needs some.",
    ),
    seed: int | None = typer.Option(
        None, "--seed", metavar="N", help="Fix the sample, so the same prompt repeats exactly."
    ),
    ignore_eot: bool = typer.Option(
        False,
        "--ignore-eot",
        help="Keep generating past the end-of-text token instead of stopping there.",
    ),
    device: str = typer.Option("auto", "--device", help="auto, cuda, cpu, mps or xpu."),
    precision: str = typer.Option(
        "auto",
        "--precision",
        help="auto, bf16, fp16 or fp32. fp32 with --seed reproduces a sample exactly.",
    ),
    tokenizer: str | None = typer.Option(
        None,
        "--tokenizer",
        metavar="PATH",
        help="A tokenizer.json to use, for a run whose dataset moved.",
    ),
    chat_format: bool = typer.Option(
        False,
        "--chat",
        help="Wrap what you type in the chat template, even if the run does not record "
        "one. For a corpus flattened into User:/Assistant: text by hand.",
    ),
    raw_format: bool = typer.Option(
        False,
        "--raw",
        help="Send your text to the model unchanged, even if the run was trained on conversations.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="With --prompt, emit the completion as JSON and nothing else."
    ),
) -> None:
    """Generate text from a trained run, once or interactively.

    What a run produces is a [bold]base[/bold] language model: it continues text.
    Whether it answers a question depends on the corpus it was trained on, not on this
    command -- a prose corpus gives you a model that continues prose, so give it the
    beginning of something rather than a request.

    A run trained on conversations records the chat template its dataset was rendered in.
    For those, what you type is wrapped in that template and generation stops where the
    model starts writing the next turn; [bold]--raw[/bold] switches that off and
    [bold]--chat[/bold] forces it on.

    With no [bold]--prompt[/bold] this is an interactive playground where sampling can
    be changed between completions. Piped input is used as the prompt, so
    [bold]echo "Once upon" | trainai chat runs/mine[/bold] works in a script.
    """
    from trainai.cli.chat import run_chat

    run_chat(
        run,
        prompt=prompt,
        which=which,
        max_new_tokens=tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=penalty,
        seed=seed,
        ignore_eot=ignore_eot,
        device=device,
        precision=precision,
        tokenizer=tokenizer,
        chat_format=chat_format,
        raw_format=raw_format,
        json_output=json_output,
    )


# --------------------------------------------------------------------------- #
# export
#
# Defaults are literals for the same reason as everywhere else in this file, and
# `tests/test_cli.py` asserts they match `trainai.export`.
# --------------------------------------------------------------------------- #
@app.command("export")
def export_command(
    run: str = typer.Argument(
        ...,
        metavar="RUN",
        help="A run directory, a checkpoints directory, or a single step-*.pt file.",
    ),
    out: str = typer.Option(
        ...,
        "--out",
        "-o",
        metavar="DIR",
        help="Directory to write the exported model into.",
    ),
    export_format: str = typer.Option(
        "hf",
        "--format",
        "-f",
        help="hf for a directory transformers can load, or safetensors for TrainAI's "
        "own tensor names plus its config.",
    ),
    dtype: str = typer.Option(
        "fp32",
        "--dtype",
        help="fp32, fp16 or bf16. fp32 is what the checkpoint holds, so it is the only "
        "lossless choice; the others halve the file.",
    ),
    which: str = typer.Option(
        "best",
        "--which",
        help="Which checkpoint: best (lowest validation loss) or latest.",
    ),
    tokenizer: str | None = typer.Option(
        None,
        "--tokenizer",
        metavar="PATH",
        help="A tokenizer.json to use, for a run whose dataset moved.",
    ),
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="Re-read the written weights and compare them to the model. On by default.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Replace an existing export in the destination directory."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit the result as JSON and print nothing else."
    ),
) -> None:
    """Write a trained run out as a model directory other tools can load.

    The default format is a HuggingFace [bold]LlamaForCausalLM[/bold] directory --
    weights, config, tokenizer and a README -- because the architecture TrainAI trains
    is structurally a Llama, so nothing has to be converted.

    Every export is verified: the written weights are read back and compared to the
    model, and when [bold]transformers[/bold] is installed the export is loaded and its
    logits compared against ours. The report says which checks ran and which did not.
    """
    from trainai.cli.export import run_export

    run_export(
        run,
        out,
        export_format=export_format,
        dtype=dtype,
        which=which,
        tokenizer=tokenizer,
        verify=verify,
        force=force,
        json_output=json_output,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    """Run the CLI and return a process exit code.

    ``standalone_mode=False`` hands control of exit behaviour to us, which is the
    only way to guarantee the exit codes documented in
    :class:`trainai.errors.ExitCode`.

    One subtlety worth knowing before editing this function: in non-standalone
    mode Typer's ``_main`` *returns* the exit code rather than raising. It maps
    ``KeyboardInterrupt`` to ``Exit(130)`` and then, in its ``except Exit``
    branch, does ``return e.exit_code``. So a Ctrl-C arrives here as the integer
    ``130``, not as an exception. Ignoring the return value -- as an earlier
    version of this function did -- silently reports every failure as success.

    The exception classes come from :mod:`trainai.cli._click` rather than from
    ``click``, because which package defines them depends on the installed Typer.
    Writing ``except click.ClickException`` here made every usage error escape on
    Typer 0.27; see that module for the whole story.
    """
    try:
        result = app(standalone_mode=False)
    except TrainAIError as exc:
        render_error(exc, show_details=settings.verbose)
        return exc.exit_code
    except KeyboardInterrupt:
        # Reachable only if a Ctrl-C lands outside Typer's own try block.
        _report_interrupt()
        return ExitCode.INTERRUPTED
    except (typer.Abort, Abort):
        _report_interrupt()
        return ExitCode.INTERRUPTED
    except (typer.Exit, Exit) as exc:
        # Defensive: click.Command.main raises this even though Typer returns it.
        return int(exc.exit_code)
    except ClickException as exc:
        # Re-raised rather than handled when standalone_mode is False.
        exc.show()
        return int(exc.exit_code)

    if isinstance(result, int):
        if result == ExitCode.INTERRUPTED:
            _report_interrupt()
        return result
    # Commands return None on success.
    return ExitCode.OK


def _report_interrupt() -> None:
    err_console.print("\n[yellow]Interrupted.[/] Nothing was left in a broken state.")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
