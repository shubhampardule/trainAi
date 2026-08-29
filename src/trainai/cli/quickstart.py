"""``trainai quickstart`` -- one command from a corpus to text a model wrote.

Five steps, in the order the README teaches them: prepare, plan, train, evaluate,
sample. Each one calls the same ``run_*`` function the individual command calls, and
prints the command it is equivalent to before it runs. That is the whole design
constraint: this is a shortcut *through* the CLI, not a second implementation of it, so
a user who wants to change one number can copy the printed line and stop using this
command. Nothing here decides anything the individual commands would not decide -- the
model shape comes from ``trainai plan``'s measurements, exactly as it would by hand.

Three deliberate omissions.

``--json``: five reports concatenated is not a document. Anything scripted should call
the individual commands, each of which emits its own object.

The network: the corpus is the user's to supply. Nothing in the installed package
reaches the network, and a convenience command is a bad place to start.

A reused plan. The dataset is reused when it is already on disk, because preparing it
twice from the same input gives the same bytes by construction. A plan is not like
that -- it is a measurement of this machine at the moment it ran, and free VRAM moves.
Silently training on a stale measurement is the failure this project exists to avoid, so
the measurement is always retaken.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trainai.cli.chat import run_chat
from trainai.cli.data import run_prepare
from trainai.cli.eval import run_eval
from trainai.cli.plan import run_plan
from trainai.cli.train import run_train
from trainai.console import (
    DASH,
    console,
    fmt_count,
    fmt_duration,
    fmt_int,
    print_command,
    print_kv,
    rule,
)
from trainai.data.binarize import MANIFEST_NAME, DatasetManifest
from trainai.data.ingest import IngestOptions, Ingestor
from trainai.eval.perplexity import EvalReport
from trainai.hardware.planner import TrainingPlan

__all__ = ["run_quickstart"]

#: Steps in the sequence, for the "Step 2 of 5" line. A constant so the two cannot drift.
TOTAL_STEPS = 5

#: How much of the corpus's own opening to use as the sample prompt. Long enough to be a
#: real beginning-of-something, short enough that the printed command stays one line.
PROMPT_CHARS = 48

#: Used only if the corpus's first document is empty after collapsing whitespace, which
#: `data prepare` would already have refused -- so this is a fallback that should be
#: unreachable rather than a default anyone is meant to get.
FALLBACK_PROMPT = "The "


@dataclass(frozen=True)
class _Layout:
    """Where the five steps put things, derived from one ``--out``.

    One directory holding all four artifacts, rather than the README's four separate
    paths, because the point of this command is that the user should not have to invent
    a filing system before their first run.
    """

    root: Path
    dataset: Path
    plan: Path
    run: Path

    @classmethod
    def under(cls, out: str) -> _Layout:
        root = Path(out)
        return cls(root=root, dataset=root / "dataset", plan=root / "plan.json", run=root / "run")


@dataclass(frozen=True)
class _Reading:
    """The flags needed to read the corpus at all, threaded to both steps that read it.

    Deliberately not every ``data prepare`` flag: a first run should not need tuning, and
    the printed command stays copyable only while it is short. Anything else is a reason
    to run ``data prepare`` directly, which the printed command shows how to do.
    """

    encoding: str = "utf-8"
    jsonl_field: str | None = None
    csv_text_column: str | None = None
    db_table: str | None = None

    def flags(self) -> str:
        """The non-default part of this, as it would be typed."""
        pairs = [
            ("--encoding", self.encoding if self.encoding != "utf-8" else None),
            ("--jsonl-field", self.jsonl_field),
            ("--csv-text-column", self.csv_text_column),
            ("--db-table", self.db_table),
        ]
        return "".join(f" {flag} {value}" for flag, value in pairs if value is not None)


def run_quickstart(
    corpus: str,
    *,
    out: str = "quickstart",
    time_budget: str | None = None,
    steps: int | None = None,
    prompt: str | None = None,
    tokens: int | None = None,
    encoding: str = "utf-8",
    jsonl_field: str | None = None,
    csv_text_column: str | None = None,
    db_table: str | None = None,
    device: str | None = None,
    precision: str | None = None,
    assume_yes: bool = False,
    force: bool = False,
) -> None:
    """Run the five steps, stopping for confirmation before the long one.

    ``device`` and ``precision`` default to ``None`` rather than to ``"auto"`` on
    purpose. The plan carries a precision that was *measured* on this machine, and
    ``run_train`` treats any value it is given as an override -- so passing ``"auto"``
    here would quietly discard the measurement in favour of a fresh guess.
    """
    layout = _Layout.under(out)
    reading = _Reading(
        encoding=encoding,
        jsonl_field=jsonl_field,
        csv_text_column=csv_text_column,
        db_table=db_table,
    )

    rule("Quickstart")
    print_kv("Will write", _layout_rows(layout))
    console.print(
        "[dim]Five steps: prepare, plan, train, evaluate, sample. Each prints the "
        "command it stands in for, so nothing here is a black box. Stops to confirm "
        "before training.[/]"
    )

    manifest = _step_prepare(corpus, layout, reading, force=force)
    plan = _step_plan(layout, time_budget=time_budget, device=device, precision=precision)

    if not _confirm(plan, assume_yes=assume_yes):
        _report_stopped(layout, steps=steps)
        return

    _step_train(layout, steps=steps, device=device, precision=precision, force=force)
    report = _step_eval(layout, device=device, precision=precision)
    _step_sample(
        corpus,
        layout,
        reading,
        prompt=prompt,
        tokens=tokens,
        device=device,
        precision=precision,
    )
    _report_done(layout, manifest, report)


# --------------------------------------------------------------------------- #
# The five steps
# --------------------------------------------------------------------------- #
def _step_prepare(
    corpus: str, layout: _Layout, reading: _Reading, *, force: bool
) -> DatasetManifest:
    """Tokenize the corpus, or reuse what is already there.

    Reuse is safe here in a way it is not for the plan: the same corpus with the same
    settings and seed produces byte-identical shards, so a second preparation could only
    reproduce the first one more slowly.
    """
    command = f"trainai data prepare {corpus} --out {layout.dataset.as_posix()}{reading.flags()}"
    prepared = (layout.dataset / MANIFEST_NAME).is_file()
    _announce(
        1, "prepare the corpus", command, reused=layout.dataset if prepared and not force else None
    )
    if prepared and not force:
        return DatasetManifest.load(str(layout.dataset))
    return run_prepare(
        corpus,
        str(layout.dataset),
        encoding=reading.encoding,
        jsonl_field=reading.jsonl_field,
        csv_text_column=reading.csv_text_column,
        db_table=reading.db_table,
        force=force,
    )


def _step_plan(
    layout: _Layout, *, time_budget: str | None, device: str | None, precision: str | None
) -> TrainingPlan:
    """Measure this machine and pick the largest configuration that fits."""
    command = f"trainai plan --data {layout.dataset.as_posix()} --out {layout.plan.as_posix()}"
    if time_budget is not None:
        command += f" --time {time_budget}"
    _announce(2, "measure this machine", command)
    return run_plan(
        str(layout.dataset),
        out=str(layout.plan),
        time_budget=time_budget,
        device=device or "auto",
        precision=precision or "auto",
    )


def _step_train(
    layout: _Layout,
    *,
    steps: int | None,
    device: str | None,
    precision: str | None,
    force: bool,
) -> None:
    command = (
        f"trainai train --data {layout.dataset.as_posix()} "
        f"--plan {layout.plan.as_posix()} --out {layout.run.as_posix()}"
    )
    if steps is not None:
        command += f" --steps {steps}"
    _announce(3, "train the model", command)
    run_train(
        str(layout.dataset),
        out=str(layout.run),
        plan=str(layout.plan),
        steps=steps,
        device=device,
        precision=precision,
        force=force,
    )


def _step_eval(layout: _Layout, *, device: str | None, precision: str | None) -> EvalReport:
    """Score both splits, not just the held-out one.

    ``--split both`` costs a second pass and is what makes overfitting visible, which on
    a first run over a small corpus is the most likely thing to have gone wrong.
    """
    command = f"trainai eval {layout.run.as_posix()} --split both"
    _announce(4, "measure what came out", command)
    return run_eval(
        str(layout.run),
        split="both",
        device=device or "auto",
        precision=precision or "auto",
    )


def _step_sample(
    corpus: str,
    layout: _Layout,
    reading: _Reading,
    *,
    prompt: str | None,
    tokens: int | None,
    device: str | None,
    precision: str | None,
) -> None:
    """Generate once, from the corpus's own opening unless told otherwise.

    A base model continues text rather than answering, so the prompt has to look like the
    beginning of something it was trained on. Taking it from the corpus makes that true
    for any language and any subject matter, which a hard-coded English prompt would not.
    """
    text = prompt if prompt is not None else _prompt_from_corpus(corpus, reading)
    command = f'trainai chat {layout.run.as_posix()} --prompt "{text}"'
    if tokens is not None:
        command += f" --tokens {tokens}"
    _announce(5, "generate text", command)
    if prompt is None:
        console.print(
            "[dim]  The prompt is your corpus's own opening, so a model that fitted it "
            "closely may simply continue with what really follows.[/]"
        )
    run_chat(
        str(layout.run),
        prompt=text,
        **({"max_new_tokens": tokens} if tokens is not None else {}),
        device=device or "auto",
        precision=precision or "auto",
    )


def _prompt_from_corpus(corpus: str, reading: _Reading) -> str:
    """The first document's opening, collapsed to one line.

    Collapsed because it is echoed inside a printed command: a newline there would break
    the copyable line in two, and a double quote would end the argument early.
    """
    options = IngestOptions(
        encoding=reading.encoding,
        jsonl_field=reading.jsonl_field,
        csv_text_column=reading.csv_text_column,
        db_table=reading.db_table,
    )
    ingestor = Ingestor(options)
    documents = ingestor.documents(ingestor.discover(Path(corpus)))
    try:
        first = next((document.text for document in documents), "")
    finally:
        # Only the first document is wanted, so the generator is abandoned partway --
        # closing it is what releases the file handle now rather than at collection.
        documents.close()
    opening = " ".join(first.replace('"', " ").split())[:PROMPT_CHARS]
    return opening or FALLBACK_PROMPT


# --------------------------------------------------------------------------- #
# Narration
# --------------------------------------------------------------------------- #
def _announce(number: int, title: str, command: str, *, reused: Path | None = None) -> None:
    """Head each step with the command it stands in for.

    One of only two ``print_command`` sites in this module -- this one and
    :func:`_print_next` -- so the copyable-command census in
    ``tests/test_conventions.py`` counts two here rather than one per step.
    """
    console.print()
    console.print(f"[bold cyan]Step {number} of {TOTAL_STEPS}[/][dim]  {DASH}  {title}[/]")
    print_command(command, style="dim bold")
    if reused is not None:
        console.print(
            f"[dim]  {reused.as_posix()} is already prepared, so this step was skipped. "
            "Pass [/][bold]--force[/][dim] to redo it.[/]"
        )
    console.print()


def _confirm(plan: TrainingPlan, *, assume_yes: bool) -> bool:
    """Ask before the only step that takes real time.

    Defaults to *yes*, unlike ``trainai setup --install``'s prompt. The difference is
    what the two do: pip replaces a 2.5 GB dependency shared by everything else in the
    environment, while this writes checkpoints into a directory the user just named. The
    prompt is here so nobody is surprised by how long it takes, not because the action
    needs guarding.
    """
    if assume_yes:
        return True
    console.print(
        f"[bold]Next is the long step.[/] Training {plan.preset_name} for "
        f"{fmt_int(plan.train_config.steps)} steps should take about "
        f"[bold]{fmt_duration(plan.estimated_seconds)}[/] on this machine, from the "
        "measurement above."
    )
    answer = console.input("Train now? [Y/n] ").strip().lower()
    return answer in ("", "y", "yes")


def _layout_rows(layout: _Layout) -> list[tuple[str, str]]:
    return [
        ("Dataset", f"{layout.dataset.as_posix()}  [dim](token shards and the tokenizer)[/]"),
        ("Plan", f"{layout.plan.as_posix()}  [dim](what was measured, and what it chose)[/]"),
        ("Run", f"{layout.run.as_posix()}  [dim](checkpoints and the metrics log)[/]"),
    ]


def _report_stopped(layout: _Layout, *, steps: int | None) -> None:
    """Answering "n" is a real outcome, not a cancellation to apologise for.

    The dataset and the plan are both on disk and both still valid, so the only thing
    left to say is the one command that resumes from here.
    """
    console.print()
    console.print("[yellow]Stopped before training.[/] Nothing was trained.")
    command = (
        f"trainai train --data {layout.dataset.as_posix()} "
        f"--plan {layout.plan.as_posix()} --out {layout.run.as_posix()}"
    )
    if steps is not None:
        command += f" --steps {steps}"
    _print_next("The dataset and the plan are kept. Train when you are ready:", [command])


def _report_done(layout: _Layout, manifest: DatasetManifest, report: EvalReport) -> None:
    rule("Done")
    val = report.result_for("val")
    rows = [
        (
            "Dataset",
            f"{layout.dataset.as_posix()}  [dim]({fmt_count(manifest.total_tokens)} tokens, "
            f"vocabulary {fmt_int(manifest.vocab_size)})[/]",
        ),
        ("Run", f"{layout.run.as_posix()}  [dim](step {fmt_int(report.step)})[/]"),
    ]
    if val is not None:
        rows.append(
            (
                "Held-out loss",
                f"[bold]{val.loss:.4f}[/]  [dim](perplexity {val.perplexity:.2f} per token "
                f"of this tokenizer, so comparable only to another run on this "
                f"dataset)[/]",
            )
        )
    print_kv("What you have", rows)
    _print_next(
        "Where to go from here:",
        [
            f"trainai chat {layout.run.as_posix()}",
            f"trainai eval {layout.run.as_posix()} --split both",
            f"trainai export {layout.run.as_posix()} --out {(layout.root / 'model').as_posix()}",
        ],
    )


def _print_next(headline: str, commands: list[str]) -> None:
    console.print(f"[dim]{headline}[/]")
    console.print()
    for command in commands:
        print_command(command, style="dim bold")
    console.print()
