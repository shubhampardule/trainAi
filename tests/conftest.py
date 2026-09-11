"""Shared pytest fixtures and marker handling.

Design rule for this suite: the default run (``pytest -m "not gpu"``) must pass on
a machine with no GPU, in a few seconds, with no network access. Anything that
needs a GPU, real time, or the internet is marked and opt-in. CI runs the default.

Every fixture that writes text passes ``newline="\\n"``. Without it, Python
translates ``\\n`` to ``os.linesep`` on Windows, so the same fixture would put
different bytes on disk depending on the platform -- and this project asserts
checksums over exactly those bytes.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

#: Every character in the Unicode Box Drawing block. Rich draws a panel out of these,
#: and :func:`flat` removes them for the reason given there.
_BOX_DRAWING = {chr(code) for code in range(0x2500, 0x2580)}


def flat(text: str) -> str:
    """Console output reduced to the words in it, for asserting on a message.

    Two things stand between a sentence this project wrote and the bytes on the
    terminal, and neither is the message: Rich wraps at the console width, and a
    refusal is drawn inside a panel whose border is a box-drawing character on both
    ends of every line. Collapsing whitespace alone is not enough -- a phrase that wraps
    inside a panel arrives as ``not a TrainAI | | export``, with the two borders between
    the words, so an assertion on the phrase fails on the panel rather than on the text.

    That failure is not hypothetical and not stable, which is what makes it worth a
    shared helper: the width at which a line wraps depends on how long the temporary
    path in the message is, so a test passed for months and then failed the day
    pytest's run counter grew a digit.

    The ellipsis Rich inserts when it *truncates* is deliberately left in place. That
    one is a word genuinely missing from the output, and hiding it would let a message
    too long to read pass a test that says it is readable.
    """
    return " ".join("".join(" " if ch in _BOX_DRAWING else ch for ch in text).split())


def unwrapped(text: str) -> str:
    """Console output with every space removed, for asserting on a single long token.

    :func:`flat` repairs a *phrase* split across a panel border, because Rich wraps at
    word boundaries and the words themselves survive. It cannot repair a token longer
    than the panel is wide: Rich has nowhere to break but the middle, and a temporary
    path is exactly that token. ``/tmp/pytest-of-runner/pytest-0/test_a0/candidate.json``
    at the default 79 columns arrives as ``candidate.j | | son``, and blanking the border
    leaves ``candidate.j son`` -- still not the filename.

    Widening the console would hide it rather than fix it. Where a line breaks is a
    function of how long that temporary path happens to be, so a width wide enough on
    one platform is a coincidence on the next: this assertion passed on Windows and
    failed on Linux in the same commit, on the same code, for no reason but the length of
    ``/tmp`` versus ``C:\\Users\\RUNNER~1\\AppData\\Local\\Temp``. Deleting the spaces
    instead makes the check independent of the width altogether.

    Use it only for a token that contains no space of its own -- a path, a filename, a
    flag. For prose, :func:`flat` is the right helper, and it keeps the word boundaries
    that make a fused false positive impossible.
    """
    return "".join(flat(text).split())


@functools.lru_cache(maxsize=1)
def cuda_available() -> bool:
    """Whether a usable CUDA device exists. Cached; importing torch is slow."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``@pytest.mark.gpu`` tests when there is no CUDA device."""
    if cuda_available():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


# --------------------------------------------------------------------------- #
# The CLI surface, as click sees it
# --------------------------------------------------------------------------- #
# Several tests check messages against the flags and commands that really exist, and
# they must all ask the same way. Two gotchas made a first version under-report, and
# either would make a check pass while the message it guards is wrong: the negative
# half of a boolean flag lives in ``secondary_opts``, and ``--help`` is added at parse
# time rather than declared, so neither shows up in ``params``. Written once here
# instead of once per test module.
#
# Imported as ``from conftest import cli_command``: pytest puts this directory on
# ``sys.path`` because ``tests/`` has no ``__init__.py``.


def cli_command(*names: str) -> Any:
    """One CLI command by path, e.g. ``cli_command("data", "inspect")``.

    No arguments gives the root group. Typer builds the click tree on demand, so this
    is the only way to see what the CLI actually accepts rather than what a decorator
    was asked for.
    """
    import typer.main

    from trainai.cli._click import is_group
    from trainai.cli.main import app

    command: Any = typer.main.get_command(app)
    for name in names:
        assert is_group(command), f"{name} is not inside a group"
        command = command.commands[name]
    return command


def cli_long_flags(command: Any) -> set[str]:
    """Every long option one command accepts, click's own answer."""
    flags = {
        opt
        for parameter in command.params
        for opt in (*parameter.opts, *parameter.secondary_opts)
        if opt.startswith("--")
    }
    if command.add_help_option:
        names = command.context_settings.get("help_option_names") or ["--help"]
        flags |= {name for name in names if name.startswith("--")}
    return flags


def cli_flag_owners() -> dict[str, set[str]]:
    """Which commands accept each long flag, keyed as ``"data prepare"``.

    The root group's own flags are keyed under ``""``, since they belong to no leaf.
    """
    from trainai.cli._click import is_group

    owners: dict[str, set[str]] = {}

    def walk(command: Any, path: list[str]) -> None:
        for flag in cli_long_flags(command):
            owners.setdefault(flag, set()).add(" ".join(path))
        if is_group(command):
            for name, sub in command.commands.items():
                walk(sub, [*path, name])

    walk(cli_command(), [])
    return owners


def cli_command_paths() -> set[str]:
    """Every invocable command path, e.g. ``{"doctor", "data", "data prepare", ...}``.

    Groups are included: ``trainai data`` runs and prints its own help.
    """
    from trainai.cli._click import is_group

    paths: set[str] = set()

    def walk(command: Any, path: list[str]) -> None:
        if path:
            paths.add(" ".join(path))
        if is_group(command):
            for name, sub in command.commands.items():
                walk(sub, [*path, name])

    walk(cli_command(), [])
    return paths


def cli_group_paths() -> set[str]:
    """The subset of :func:`cli_command_paths` that hold subcommands.

    Needed to tell ``trainai data prepare corpus.txt`` -- a real command with an
    argument -- from ``trainai data prepair``, where the group is real and the leaf is
    a typo. Both leave ``"data"`` as their longest recognised prefix.
    """
    from trainai.cli._click import is_group

    groups: set[str] = set()

    def walk(command: Any, path: list[str]) -> None:
        if not is_group(command):
            return
        if path:
            groups.add(" ".join(path))
        for name, sub in command.commands.items():
            walk(sub, [*path, name])

    walk(cli_command(), [])
    return groups


@pytest.fixture
def tmp_corpus(tmp_path: Path) -> Path:
    """A small, deterministic text corpus on disk.

    Real prose rather than ``"aaaa"``, because tokenizer training and dataset
    statistics both behave differently on degenerate input, and a fixture that
    hides that is worse than no fixture.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "one.txt").write_text(
        "The quick brown fox jumps over the lazy dog.\n"
        "Pack my box with five dozen liquor jugs.\n"
        "How vexingly quick daft zebras jump!\n" * 20,
        encoding="utf-8",
        newline="\n",
    )
    (corpus / "two.txt").write_text(
        "It was the best of times, it was the worst of times.\n"
        "We had everything before us, we had nothing before us.\n" * 20,
        encoding="utf-8",
        newline="\n",
    )
    return corpus


@pytest.fixture
def many_document_corpus(tmp_path: Path) -> Path:
    """A corpus of many separate documents, as JSON Lines.

    ``tmp_corpus`` has two documents, which is too few to say anything about a
    train/val split: at any sane validation fraction the split is all-or-nothing.
    This fixture has enough documents for a 5% holdout to mean something.
    """
    return _write_jsonl_corpus(tmp_path / "jsonl_corpus", count=120)


def _write_jsonl_corpus(directory: Path, *, count: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    subjects = ("rivers", "clocks", "bridges", "harbours", "orchards", "lanterns")
    verbs = ("carry", "measure", "cross", "shelter", "ripen", "kindle")
    lines = []
    for index in range(count):
        subject = subjects[index % len(subjects)]
        verb = verbs[(index * 5) % len(verbs)]
        text = (
            f"Document {index}. The {subject} {verb} what the previous ones did not. "
            "A second sentence keeps this document long enough to be useful, and "
            f"repeats the words {subject} and {verb} so the tokenizer has pairs to merge."
        )
        lines.append(json.dumps({"text": text}))
    (directory / "documents.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    return directory


def _write_messages_corpus(directory: Path, *, count: int) -> Path:
    """The same prose as :func:`_write_jsonl_corpus`, typed as conversations.

    Written as records with a ``messages`` field rather than as flattened
    ``User:``/``Assistant:`` text, because the point of a corpus in this shape is that
    ``data prepare --jsonl-messages-field`` records the chat template it rendered -- and
    flattened text, however identical the shards, records nothing.
    """
    directory.mkdir(parents=True, exist_ok=True)
    subjects = ("rivers", "clocks", "bridges", "harbours", "orchards", "lanterns")
    verbs = ("carry", "measure", "cross", "shelter", "ripen", "kindle")
    lines = []
    for index in range(count):
        subject = subjects[index % len(subjects)]
        verb = verbs[(index * 5) % len(verbs)]
        lines.append(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"What do the {subject} {verb}?"},
                        {
                            "role": "assistant",
                            "content": (
                                f"Answer {index}. The {subject} {verb} what the previous "
                                "ones did not, and a second sentence keeps this reply long "
                                f"enough that the tokenizer sees {subject} and {verb} twice."
                            ),
                        },
                    ]
                }
            )
        )
    (directory / "conversations.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    return directory


@dataclass(frozen=True)
class TrainedRun:
    """Where the three artefacts of a CLI-built run are. See :func:`_train_through_cli`."""

    corpus: Path
    data: Path
    run: Path


def _train_through_cli(root: Path, corpus: Path, *, prepare: tuple[str, ...] = ()) -> TrainedRun:
    """Prepare ``corpus`` and train on it by driving ``main()``, twice through ``sys.argv``.

    Shared by the CLI run fixtures rather than copied into each, because the flags that
    make the run small enough to build in a test are incidental to every one of them --
    what differs is the corpus and the handful of ``prepare`` flags in ``prepare``.
    """
    import sys

    from trainai.cli.main import main
    from trainai.errors import ExitCode

    data = root / "prepared"
    run = root / "run"

    argv = sys.argv
    try:
        sys.argv = [
            "trainai",
            "data",
            "prepare",
            str(corpus),
            "--out",
            str(data),
            "--vocab-size",
            "512",
            *prepare,
            "--json",
        ]
        assert main() == ExitCode.OK
        sys.argv = [
            "trainai",
            "train",
            "--data",
            str(data),
            "--out",
            str(run),
            "--layers",
            "1",
            "--heads",
            "2",
            "--width",
            "32",
            "--context",
            "64",
            "--steps",
            "4",
            "--batch-size",
            "2",
            "--seq-len",
            "32",
            "--warmup",
            "1",
            "--eval-every",
            "2",
            "--eval-batches",
            "2",
            "--checkpoint-every",
            "2",
            "--device",
            "cpu",
            "--json",
        ]
        assert main() == ExitCode.OK
    finally:
        sys.argv = argv

    return TrainedRun(corpus=corpus, data=data, run=run)


@pytest.fixture(scope="session")
def cli_trained_run(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """A corpus, a prepared dataset and a trained run, all built through the CLI.

    Returned as an object with ``.corpus``, ``.data`` and ``.run``. Built by driving
    ``main()`` rather than the library, because the commands that consume this --
    ``eval``, ``chat`` -- are being tested for their wiring, and a run assembled by
    hand would not exercise the parts that put a tokenizer in the run directory or a
    dataset path in the checkpoint.

    Its corpus is **prose**, so the checkpoint records no chat template. That is what
    makes it the right fixture for everything about generation that is not about the
    template, and :func:`cli_chat_run` is the one for the rest.

    Session-scoped: it trains a tokenizer and then a model, which is far too slow to
    repeat per test. Nothing that uses it writes to it.
    """
    root = tmp_path_factory.mktemp("cli_run")
    return _train_through_cli(root, _write_jsonl_corpus(root / "corpus", count=120))


@pytest.fixture(scope="session")
def cli_chat_run(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """The same, from a corpus of typed conversations, so the run records its template.

    Deliberately a second trained run rather than a copy of :func:`cli_trained_run` with
    the checkpoint's ``chat`` block edited in. The block is written by ``data prepare``,
    carried through the manifest, copied by the trainer and read by the inference session;
    an edited copy would test ``trainai chat`` against a value no command had produced,
    which is the one thing worth knowing here.
    """
    root = tmp_path_factory.mktemp("cli_chat_run")
    corpus = _write_messages_corpus(root / "corpus", count=120)
    return _train_through_cli(root, corpus, prepare=("--jsonl-messages-field", "messages"))


@pytest.fixture(scope="session")
def prepared_dataset(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """A real prepared dataset, built once for the whole session.

    Session-scoped because preparing it trains a tokenizer, which is the slowest
    thing in the suite, and because every test that uses it only reads. Tests that
    need to damage a dataset build their own.
    """
    from trainai.data import IngestOptions, Ingestor, binarize_documents, train_tokenizer

    root = tmp_path_factory.mktemp("dataset")
    corpus = _write_jsonl_corpus(root / "corpus", count=300)
    ingestor = Ingestor(IngestOptions())
    documents = list(ingestor.documents(ingestor.discover(corpus)))
    tokenizer = train_tokenizer((d.text for d in documents), vocab_size=512)
    return binarize_documents(
        documents,
        root / "prepared",
        tokenizer,
        seed=1234,
        val_fraction=0.1,
        ingest_options=ingestor.options,
    )


@pytest.fixture(scope="session")
def dataset_without_validation(
    prepared_dataset: Any, tmp_path_factory: pytest.TempPathFactory
) -> Any:
    """The same corpus and tokenizer, re-split with ``--val-fraction 0``.

    A run on this has no held-out loss for a reason no other fixture can produce, and
    that reason has to be told apart from "the split is too small for one window" and
    "--eval-every is 0". Re-binarizing is cheap; the tokenizer is loaded from
    :func:`prepared_dataset` rather than trained again, which is the slow part.
    """
    from trainai.data import IngestOptions, Ingestor, binarize_documents
    from trainai.data.tokenizer import ByteLevelBPE

    root = tmp_path_factory.mktemp("dataset_noval")
    corpus = _write_jsonl_corpus(root / "corpus", count=300)
    ingestor = Ingestor(IngestOptions())
    documents = list(ingestor.documents(ingestor.discover(corpus)))
    assert prepared_dataset.root is not None
    tokenizer = ByteLevelBPE.load(prepared_dataset.root / "tokenizer.json")
    return binarize_documents(
        documents,
        root / "prepared",
        tokenizer,
        seed=1234,
        val_fraction=0.0,
        ingest_options=ingestor.options,
    )


@pytest.fixture(scope="session")
def masked_dataset(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """A prepared dataset of typed conversations, so it carries a loss mask.

    Built from the real chat template and the real span machinery rather than from a
    hand-written mask file, because what the trainer and the evaluator have to agree
    with is what ``data prepare`` actually writes. Session-scoped for the same reason
    :func:`prepared_dataset` is: training the tokenizer is the slow part.

    Sized so that a batch of 4 windows of 32 tokens fits in *both* splits, and so that
    the target share is neither 0 nor 1 -- a mask that selects everything or nothing
    would let a trainer that ignores it pass every test here.
    """
    from trainai.data import IngestOptions, Ingestor, binarize_documents, train_tokenizer
    from trainai.data.chat import render_conversation
    from trainai.data.ingest import Document

    root = tmp_path_factory.mktemp("masked")
    documents = []
    for index in range(240):
        conversation = render_conversation(
            [
                {"role": "user", "content": f"Question {index} about rivers and clocks?"},
                {
                    "role": "assistant",
                    "content": (
                        f"Answer {index}. The rivers carry what the clocks measure, and "
                        "the bridges cross both of them before the lanterns kindle."
                    ),
                },
            ]
        )
        documents.append(Document(conversation.text, "chat.jsonl", index, 0, conversation.spans))
    tokenizer = train_tokenizer((d.text for d in documents), vocab_size=512)
    manifest = binarize_documents(
        documents,
        root / "prepared",
        tokenizer,
        seed=1234,
        val_fraction=0.1,
        loss_mask=True,
        ingest_options=Ingestor(IngestOptions(jsonl_messages_field="messages")).options,
    )
    assert manifest.has_loss_mask
    trained = int(manifest.totals["trained_tokens"])
    assert 0 < trained < manifest.total_tokens
    return manifest
