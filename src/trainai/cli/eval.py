"""``trainai eval`` -- recompute a run's loss on held-out text, and say what it means.

The number this prints is the same measurement the training curve reports, recomputed
over a whole split rather than the handful of batches the trainer samples. Three things
shape how it is presented:

**Perplexity alone is not comparable to anything.** It is per token of one tokenizer,
so the report carries the vocabulary size, the context, the precision and the exact
checkpoint. Printing "perplexity: 41.2" and nothing else invites a comparison against
someone else's 38.6 that means nothing.

**Coverage is stated, not implied.** ``--max-batches`` and a split whose tail does not
fill a sequence both mean less than the whole split was scored, and the report says how
much. A number labelled "on the validation split" that covered a fifth of it is the
failure this section exists to prevent.

**A ``--data`` that is not the run's own dataset is flagged.** Its ``val`` split is not
necessarily held out from this model, so the report says so and drops the comparison
against the trainer's recorded number, which would otherwise read as disagreement. When
the check cannot be made at all -- a content hash missing on either side -- that is its
own third state, said out loud rather than shown as a pass.

When the run recorded its own best validation loss, the report compares the two. They
should agree closely; a gap that is not explained by precision means one of them is
measuring something else, and seeing it side by side is what makes that visible.
"""

from __future__ import annotations

from pathlib import Path

from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from trainai.console import (
    DASH,
    SPINNER,
    console,
    emit_json,
    fmt_count,
    fmt_int,
    print_bullets,
    print_kv,
    rule,
)
from trainai.data.binarize import DatasetManifest, Split, verify_dataset
from trainai.errors import UsageError, check_choice
from trainai.eval import DEFAULT_BATCH_SIZE, EvalReport, SplitResult, dataset_check, evaluate
from trainai.infer import DEFAULT_WHICH, InferenceSession
from trainai.train.metrics import format_perplexity

__all__ = ["run_eval"]

#: The values ``--split`` accepts, default first, in the order the flag's help text
#: lists them. That is all this order is: it reaches the user only through the refusal
#: message and ``details["choices"]`` below, since ``--split`` is a plain string option
#: rather than a typer choice. It is *not* the order the report uses -- ``both`` prints
#: train before val, which :func:`_resolve_splits` decides and a CLI test pins. A comment
#: here used to read "in report order" and conflate the two, which invited correcting
#: whichever of them was not actually wrong. ``train`` is offered because the gap between
#: the two is the only direct read on overfitting, the thing most likely to be wrong on a
#: small corpus.
SPLIT_CHOICES = ("val", "train", "both")


def run_eval(
    target: str,
    *,
    data: str | None = None,
    which: str = DEFAULT_WHICH,
    split: str = "val",
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_batches: int | None = None,
    seq_len: int | None = None,
    device: str = "auto",
    precision: str = "auto",
    tokenizer: str | None = None,
    verify: bool = False,
    json_output: bool = False,
) -> EvalReport:
    """Load a run, measure it on a prepared dataset, and print the result."""
    quiet = json_output
    splits = _resolve_splits(split)
    session = InferenceSession.open(
        target,
        which=which,
        device=device,
        precision=precision,
        tokenizer=tokenizer,
    )
    dataset = _resolve_dataset(session, data, verify=verify)

    if not quiet:
        rule("Evaluating")
        print_kv("Loaded", _loaded_rows(session, dataset))

    with _progress(quiet=quiet) as progress:
        tasks = {
            name: progress.add_task(f"scoring the {name} split", total=None) for name in splits
        }
        report = evaluate(
            session,
            dataset,
            splits=splits,
            batch_size=batch_size,
            max_batches=max_batches,
            seq_len=seq_len,
            on_batch=lambda name, done, tokens: progress.update(
                tasks[name],
                completed=done,
                total=max_batches,
                description=f"scoring the {name} split {DASH} {fmt_count(tokens)} tokens",
            ),
        )

    if quiet:
        emit_json(report.to_dict())
        return report

    _report(report, session)
    return report


def _resolve_splits(split: str) -> tuple[Split, ...]:
    """Which splits to score, in the order the report prints them.

    ``both`` returns train first so the reader scanning downward subtracts val from
    train, which is the gap they asked for. Do not reorder this to match
    :data:`SPLIT_CHOICES`; those two orders are different things, and a CLI test pins
    this one.
    """
    check_choice(
        split,
        SPLIT_CHOICES,
        "--split",
        hint="`both` scores the training split too, which is how you see overfitting.",
    )
    if split == "both":
        return ("train", "val")
    return (split,)  # type: ignore[return-value]


def _resolve_dataset(
    session: InferenceSession, data: str | None, *, verify: bool
) -> DatasetManifest:
    """The dataset to score against: the one given, or the one the run recorded.

    Defaulting to the recorded path is what makes ``trainai eval runs/mine`` work with
    no other arguments. It is only a default, and it does not stand in for the checks:
    :func:`trainai.eval.evaluate` still refuses a differently-tokenized dataset and
    still flags one whose content hash does not match, so a recorded path that now
    holds something else is caught rather than silently scored.
    """
    if data is not None:
        return verify_dataset(data, deep=verify) if verify else DatasetManifest.load(data)

    recorded = session.checkpoint.dataset.get("root")
    if not recorded:
        raise UsageError(
            "This checkpoint does not record which dataset it was trained on.",
            hint="Pass --data path/to/prepared-dataset.",
            details={"checkpoint": str(session.layout.checkpoint_path)},
        )
    root = Path(str(recorded))
    if not (root / "manifest.json").is_file():
        raise UsageError(
            f"The dataset this run was trained on is no longer at {root}.",
            hint=(
                "Pass --data pointing at it, or at a copy. Evaluation needs the token "
                "shards, which the run directory does not contain."
            ),
            details={"recorded": str(root), "checkpoint": str(session.layout.checkpoint_path)},
        )
    return verify_dataset(root, deep=verify) if verify else DatasetManifest.load(root)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _report(report: EvalReport, session: InferenceSession) -> None:
    rule("Result")
    for result in report.results:
        print_kv(f"{result.split.capitalize()} split", _split_rows(result))
    print_kv("Measured with", _context_rows(report, session))
    print_bullets("Worth knowing", report.notes)


def _loaded_rows(session: InferenceSession, dataset: DatasetManifest) -> list[tuple[str, str]]:
    layout = session.layout
    # Three states, and each gets its own words. Silence means "checked, and it is the
    # run's own dataset" -- so an unanswerable check must not borrow it: a missing hash
    # used to print the bare path, which reads as a clean result from a check that did
    # not run. :func:`trainai.eval.dataset_check` has the measurement.
    check = dataset_check(session, dataset)
    where = Path(str(dataset.root)).as_posix()
    if check == "mismatch":
        shown = f"{where}  [yellow](not the one this run trained on)[/]"
    elif check == "unknown":
        shown = f"{where}  [yellow](cannot tell: a content hash is missing)[/]"
    else:
        shown = where
    rows = [
        (
            "Checkpoint",
            f"[bold]{layout.checkpoint_path.name}[/]  [dim]({layout.which}, step "
            f"{fmt_int(session.step)} of {layout.checkpoints_available} on disk)[/]",
        ),
        (
            "Model",
            f"[bold]{fmt_count(session.model.parameter_count())}[/] parameters  "
            f"{DASH}  {session.model_config.describe()}",
        ),
        (
            "Dataset",
            shown,
        ),
        ("Device", f"{session.device}  {DASH}  {session.precision_note}"),
    ]
    if layout.tokenizer_source != "the run directory":
        rows.append(
            (
                "Tokenizer",
                f"{layout.tokenizer_path.as_posix()}  [dim](from {layout.tokenizer_source})[/]",
            )
        )
    if layout.note:
        rows.append(("Note", f"[yellow]{layout.note}[/]"))
    return rows


def _split_rows(result: SplitResult) -> list[tuple[str, str]]:
    coverage = (
        "[bold]all of it[/]"
        if result.coverage >= 1.0
        else f"[yellow]{result.coverage * 100:.1f}% of it[/]"
    )
    return [
        ("Loss", f"[bold]{result.loss:.4f}[/] nats/token"),
        ("Perplexity", f"[bold]{format_perplexity(result.loss)}[/]"),
        ("Bits/token", f"{result.bits_per_token:.4f}"),
        (
            "Scored",
            f"{coverage}  [dim]({fmt_int(result.windows_scored)} of "
            f"{fmt_int(result.windows_in_split)} sequences, "
            f"{fmt_int(result.tokens_scored)} predictions, {result.batches} batches)[/]",
        ),
    ]


def _context_rows(report: EvalReport, session: InferenceSession) -> list[tuple[str, str]]:
    first = report.results[0]
    trained = report.trained_seq_len
    built = session.model_config.seq_len
    # Three lengths can disagree: what was just scored, what the run trained at, and
    # what the model was built for. Only the first used to be shown, so a run built for
    # 512 and trained at 128 reported "Context 512 tokens" beside a training-curve loss
    # measured at 128 and left the reader to assume they matched.
    context = f"{fmt_int(first.seq_len)} tokens"
    if trained is not None and first.seq_len != trained:
        context += (
            f"  [dim]([yellow]past the {fmt_int(trained)} this run trained at[/]; "
            f"built for {fmt_int(built)})[/]"
        )
    elif trained is not None and trained != built:
        context += f"  [dim](what this run trained at; built for {fmt_int(built)})[/]"
    rows = [
        ("Context", context),
        (
            "Vocabulary",
            f"{fmt_int(report.vocab_size)} tokens  [dim](perplexity is per token of "
            "this vocabulary)[/]",
        ),
        ("Precision", report.precision),
    ]
    recorded = session.best_val_loss
    measured = report.result_for("val")
    # Only worth showing when the two are the same quantity. Against another dataset
    # the difference is dominated by which text was scored, so a "0.42 from what was
    # just measured" line would read as disagreement where there is none. An *unknown*
    # dataset still gets the line, because it is probably the run's own and dropping the
    # most useful row on a missing hash is worse than showing it -- but it never gets to
    # be green, which would assert agreement between two numbers nothing confirmed are
    # measuring the same text.
    if recorded is not None and measured is not None and report.dataset_check != "mismatch":
        gap = abs(measured.loss - recorded)
        # Scoring at a context the run never trained at is a bigger effect than which
        # batches the trainer sampled, so say that instead. Blaming sampling for a gap
        # the context caused sent the reader looking in the wrong place; and a small gap
        # is not agreement when the two numbers measure different things, so it does not
        # get to be green.
        comparable = trained is None or measured.seq_len == trained
        verified = report.dataset_check == "match"
        colour = "green" if gap <= 0.01 and comparable and verified else "yellow"
        if not verified:
            why = "cannot confirm both numbers scored the same text: a content hash is missing"
        elif comparable:
            why = "the trainer samples --eval-batches of the split, this scored what you asked for"
        else:
            why = (
                f"the trainer measured at {fmt_int(trained or 0)} tokens of context, "
                f"this scored at {fmt_int(measured.seq_len)}"
            )
        rows.append(
            (
                "Run recorded",
                f"[bold]{recorded:.4f}[/] at step {fmt_int(session.best_val_step or 0)}  "
                f"[dim]([{colour}]{gap:.4f} from what was just measured[/]; {why})[/]",
            )
        )
    return rows


def _progress(*, quiet: bool) -> Progress:
    """A transient bar. Disabled under ``--json`` so the output stays pipeable."""
    return Progress(
        SpinnerColumn(SPINNER),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        console=console,
        transient=True,
        disable=quiet,
    )
