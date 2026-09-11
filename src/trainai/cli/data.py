"""``trainai data prepare`` and ``trainai data inspect``.

Two commands over one pipeline. ``prepare`` turns a directory of text into token
shards plus a checksummed manifest; ``inspect`` reports on either a raw corpus or
an already-prepared dataset and writes nothing.

Preparation reads the corpus twice, deliberately. The first pass trains the
tokenizer and measures the corpus from the same stream, because no text can be
encoded until the tokenizer exists. The second pass encodes. Reading twice costs
disk bandwidth; holding the decoded corpus in memory instead would cost more RAM
than most machines can spare on a multi-gigabyte dataset.

Two failure modes this module works to avoid:

* Presenting an estimate as a measurement. The document counts, token counts and
  compression ratios printed after preparation are counted over the whole corpus.
  The one figure that is an estimate -- the pre-tokenization token count shown by
  ``inspect`` on a raw corpus -- says so on the same line.
* Destroying a dataset by accident. ``--out`` pointing at an existing prepared
  dataset is an error until ``--force`` says otherwise, because preparation
  overwrites shards from the first byte and there is no undo.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

from rich import box
from rich.markup import escape
from rich.padding import Padding
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from trainai.console import (
    DASH,
    SPINNER,
    console,
    emit_json,
    fmt_bytes,
    fmt_count,
    fmt_int,
    print_bullets,
    print_command,
    print_kv,
    rule,
)
from trainai.data import (
    DEFAULT_SHARD_TOKENS,
    DEFAULT_VAL_FRACTION,
    MANIFEST_NAME,
    SPLITS,
    ByteLevelBPE,
    CorpusMeter,
    DatasetManifest,
    DatasetReport,
    IngestOptions,
    Ingestor,
    IngestStats,
    SourceFile,
    ValidationResult,
    binarize_documents,
    describe_dataset_layout,
    train_tokenizer,
    validate_corpus,
    verify_dataset,
)
from trainai.data.binarize import TOKENIZER_NAME, ShardInfo
from trainai.errors import DatasetError, TokenizerError, UsageError

__all__ = ["run_inspect", "run_prepare"]

#: Default target vocabulary. Large enough for roughly four characters per token
#: on English prose, small enough that embeddings do not dominate a small model:
#: at d_model 512, a 32k vocabulary is 16.8M parameters, which on a 35M budget
#: would be half the model spent on lookup tables. ``validate_corpus`` warns when
#: the corpus is too small to support the size actually requested.
DEFAULT_VOCAB_SIZE = 8192

#: Default split seed. Not zero, so that the number in a manifest reads as a
#: choice someone made rather than as an unset field.
DEFAULT_SEED = 1234

#: Progress descriptions are rebuilt this often, in documents. Updating Rich on
#: every document of a million-document corpus costs more than the tokenizer.
_PROGRESS_EVERY = 64

_LEVEL_LABEL = {
    "error": "[bold red]error[/]",
    "warning": "[yellow]warning[/]",
    "note": "[cyan]note[/]",
}

#: Human words for the codecs ``ingest`` reports, so a file-mix line reads
#: "3 gzipped, 1 xz-compressed" rather than leaking the internal codec names.
_CODEC_WORDS = {
    "gzip": "gzipped",
    "bz2": "bz2-compressed",
    "lzma": "xz-compressed",
}


def _ingest_options(
    *,
    encoding: str,
    jsonl_field: str | None,
    jsonl_messages_field: str | None,
    csv_text_column: str | None,
    db_table: str | None,
    min_doc_chars: int,
    max_doc_chars: int,
    on_error: str,
) -> IngestOptions:
    """Build ingest options, rejecting combinations that can only produce nothing.

    The library checks each value it is given; what it cannot see is that two
    values contradict each other. A maximum below the minimum drops every
    document and then reports an empty corpus, which sends the user looking at
    their files instead of at their flags.
    """
    if min_doc_chars < 1:
        raise UsageError(
            f"--min-doc-chars must be at least 1, got {min_doc_chars}.",
            hint="Use 1 to keep every non-blank document, which is the default.",
            details={"min_doc_chars": min_doc_chars},
        )
    if max_doc_chars < min_doc_chars:
        raise UsageError(
            f"--max-doc-chars ({max_doc_chars:,}) is below --min-doc-chars "
            f"({min_doc_chars:,}), so every document would be dropped.",
            hint=(
                "Raise --max-doc-chars above --min-doc-chars. The default pair, "
                "1 and 16,384, suits most corpora."
            ),
            details={"min_doc_chars": min_doc_chars, "max_doc_chars": max_doc_chars},
        )
    return IngestOptions(
        encoding=encoding,
        jsonl_field=jsonl_field,
        jsonl_messages_field=jsonl_messages_field,
        csv_text_column=csv_text_column,
        db_table=db_table,
        max_doc_chars=max_doc_chars,
        min_doc_chars=min_doc_chars,
        on_error=cast("Any", on_error),
    )


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #
def run_prepare(
    corpus: str,
    out: str,
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = 2,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
    encoding: str = "utf-8",
    jsonl_field: str | None = None,
    jsonl_messages_field: str | None = None,
    csv_text_column: str | None = None,
    db_table: str | None = None,
    min_doc_chars: int = 1,
    max_doc_chars: int = 1 << 14,
    on_error: str = "fail",
    shard_tokens: int = DEFAULT_SHARD_TOKENS,
    force: bool = False,
    json_output: bool = False,
    allow_tabular: bool = False,
    tokenizer: str | None = None,
    loss_mask: bool | None = None,
) -> DatasetManifest:
    """Prepare ``corpus`` into ``out`` and return the manifest that was written."""
    quiet = json_output
    out_dir = Path(out)
    write_mask = _resolve_loss_mask(loss_mask, jsonl_messages_field)
    reuse = _load_reference_tokenizer(tokenizer, vocab_size) if tokenizer else None
    options = _ingest_options(
        encoding=encoding,
        jsonl_field=jsonl_field,
        jsonl_messages_field=jsonl_messages_field,
        csv_text_column=csv_text_column,
        db_table=db_table,
        min_doc_chars=min_doc_chars,
        max_doc_chars=max_doc_chars,
        on_error=on_error,
    )

    replacing = _guard_output(out_dir, corpus, force=force)

    ingestor = Ingestor(options)
    sources = ingestor.discover(corpus)
    if not quiet:
        rule("Corpus")
        print_kv("Input", _input_rows(corpus, out_dir, sources, replacing=replacing))

    meter = CorpusMeter()
    tokenizer_obj = _measure_and_train(
        ingestor,
        sources,
        meter,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        quiet=quiet,
        reuse=reuse,
    )
    report = meter.report()
    validation = validate_corpus(
        report,
        ingestor.stats,
        vocab_size=tokenizer_obj.vocab_size,
        max_doc_chars=max_doc_chars,
        allow_tabular=allow_tabular,
        # A reused tokenizer was not asked to reach a size, so it cannot have
        # undershot one. Passing --vocab-size here would report a shortfall against
        # a target that does not apply to this run.
        requested_vocab_size=tokenizer_obj.vocab_size if reuse else vocab_size,
    )

    if not quiet:
        print_kv("Measured", _corpus_rows(report))
        print_kv("Read", _ingest_rows(ingestor.stats, loss_mask=write_mask))
        print_kv(
            "Tokenizer",
            _reused_tokenizer_rows(tokenizer_obj, tokenizer)
            if reuse
            else _tokenizer_rows(tokenizer_obj, vocab_size),
        )
        _print_issues(validation)
    # Everything above is measurement, so it is printed before the verdict: a user
    # whose corpus is rejected still gets to see what TrainAI saw.
    validation.raise_if_failed()

    if not quiet:
        rule("Encoding")
    manifest = _binarize(
        Ingestor(options),
        sources,
        tokenizer_obj,
        out_dir,
        report=report,
        validation=validation,
        options=options,
        seed=seed,
        val_fraction=val_fraction,
        shard_tokens=shard_tokens,
        quiet=quiet,
        loss_mask=write_mask,
    )
    _check_passes_agree(manifest, report, out_dir)

    if quiet:
        emit_json(manifest.to_dict())
    else:
        _print_result(manifest, report, out_dir)
    return manifest


def _guard_output(out_dir: Path, corpus: str, *, force: bool) -> bool:
    """Check ``--out`` is safe to write to. Returns whether a dataset is replaced.

    Two refusals. Writing into the corpus itself would leave shards among the
    user's source files, where a later run would ignore them silently. Writing
    over an existing dataset is refused because preparation overwrites shards in
    place: by the time it fails, the previous dataset is already gone.
    """
    corpus_path = Path(corpus)
    with suppress(OSError):
        resolved_out = out_dir.resolve()
        resolved_corpus = corpus_path.resolve()
        inside = resolved_out == resolved_corpus or resolved_corpus in resolved_out.parents
        if inside and resolved_corpus.is_dir():
            raise UsageError(
                f"--out {out_dir} is inside the corpus directory {corpus_path}.",
                hint=(
                    "Write the prepared dataset somewhere else, for example "
                    "`--out data/prepared`. Shards left among your source files would "
                    "be silently ignored by the next run and are easy to lose track of."
                ),
                details={"out": str(resolved_out), "corpus": str(resolved_corpus)},
            )

    manifest = out_dir / MANIFEST_NAME
    if not manifest.exists():
        return False
    if force:
        return True
    raise UsageError(
        f"{out_dir} already holds a prepared dataset.",
        hint=(
            "Preparation overwrites shards in place, so it will not run over an "
            "existing dataset by accident. Pass --force to replace it, or choose a "
            "different --out."
        ),
        details={"out_dir": str(out_dir), "manifest": str(manifest)},
    )


def _measure_and_train(
    ingestor: Ingestor,
    sources: list[SourceFile],
    meter: CorpusMeter,
    *,
    vocab_size: int,
    min_frequency: int,
    quiet: bool,
    reuse: ByteLevelBPE | None = None,
) -> ByteLevelBPE:
    """First pass: one read of the corpus, feeding the meter and the BPE trainer.

    No percentage is shown. The stream is consumed by the Rust trainer, which then
    spends an unrelated and unobservable amount of time computing merges; a bar
    that reached 100% and then sat still for a minute would be a lie about what
    the program was doing.

    ``reuse`` skips the merge computation and returns that tokenizer instead. The read
    still happens in full: the meter's report is what corpus validation describes and
    what the shard planner sizes against, so a pass that measured nothing would produce
    a dataset whose own record of its corpus is empty.
    """
    total_bytes = sum(source.size_bytes for source in sources)
    # Whether the trainer ever asked for a document. It is the whole basis on which the
    # handler below decides if corpus validation is entitled to speak: validation can
    # only describe text that was read.
    read_began = False

    with _progress(quiet=quiet) as progress:
        task = progress.add_task(
            "Reading corpus" if reuse is not None else "Reading corpus, training tokenizer",
            total=None,
        )

        def stream() -> Iterator[str]:
            nonlocal read_began
            # Set here rather than inside the loop: a generator body starts running on
            # the first next(), so this is true exactly when the trainer pulls -- and it
            # stays true for a corpus that turns out to be empty, which is a case
            # validation does explain better.
            read_began = True
            for seen, document in enumerate(meter.measure(ingestor.documents(sources)), start=1):
                if seen % _PROGRESS_EVERY == 0:
                    progress.update(
                        task,
                        description=(
                            f"Read {fmt_int(seen)} documents, "
                            f"{fmt_bytes(ingestor.stats.total_bytes_read)}"
                            f" of {fmt_bytes(total_bytes)}"
                        ),
                    )
                yield document.text

        if reuse is not None:
            for _ in stream():
                pass
            return reuse

        try:
            return train_tokenizer(stream(), vocab_size=vocab_size, min_frequency=min_frequency)
        except TokenizerError:
            # "No merges were learned" is what an unusably small corpus looks like
            # from inside the trainer. When that is the real cause, the corpus
            # error explains it better, so let validation speak first.
            #
            # But only then. `train_tokenizer` rejects a --vocab-size below its floor
            # before touching the stream, so this handler used to re-validate a meter
            # that had measured nothing, and an empty measurement always fails as
            # `empty_corpus`. The result was that `--vocab-size 257` on a 27.7 KiB
            # corpus reported "No usable documents were found." -- about a corpus whose
            # size the same command had printed two lines earlier -- and the message
            # naming the real problem was discarded.
            if read_began:
                validate_corpus(
                    meter.report(),
                    ingestor.stats,
                    max_doc_chars=ingestor.options.max_doc_chars,
                ).raise_if_failed()
            raise


def _resolve_loss_mask(requested: bool | None, messages_field: str | None) -> bool:
    """Whether to write mask shards, from the flag and the corpus format.

    Unset means on for a typed corpus and off for everything else, because a mask over
    text with no marked replies is a file of all ones: half the size of the token
    shards, and it says nothing. ``--loss-mask`` without ``--jsonl-messages-field`` is
    that file, so it is refused rather than written -- the request is a
    misunderstanding of what the mask comes from, and silently producing a useless
    file would leave the user believing their prompts are excluded.
    """
    if requested is None:
        return messages_field is not None
    if requested and messages_field is None:
        raise UsageError(
            "--loss-mask needs a corpus of typed conversations, and this run has none.",
            hint=(
                "The mask marks which characters are assistant replies, which only a "
                "record read with --jsonl-messages-field has. Without it every token "
                "would be marked as a target, which is what training already does. See "
                "docs/corpus-formats.md."
            ),
            details={"loss_mask": requested, "jsonl_messages_field": messages_field},
        )
    return requested


def _binarize(
    ingestor: Ingestor,
    sources: list[SourceFile],
    tokenizer: ByteLevelBPE,
    out_dir: Path,
    *,
    report: DatasetReport,
    validation: ValidationResult,
    options: IngestOptions,
    seed: int,
    val_fraction: float,
    shard_tokens: int,
    quiet: bool,
    loss_mask: bool,
) -> DatasetManifest:
    """Second pass: encode every document and write the shards.

    A fresh :class:`Ingestor` is used so that its counters describe this pass
    alone. Reusing the first pass's would double every number it reports.
    """
    with _progress(quiet=quiet) as progress:
        task = progress.add_task("Tokenizing, writing shards", total=report.documents)

        def advance(documents_done: int, tokens_written: int) -> None:
            progress.update(
                task,
                completed=documents_done,
                description=f"Wrote {fmt_count(tokens_written)} tokens",
            )

        return binarize_documents(
            ingestor.documents(sources),
            out_dir,
            tokenizer,
            seed=seed,
            val_fraction=val_fraction,
            shard_tokens=shard_tokens,
            report=report,
            validation=validation,
            ingest_options=options,
            sources=[source.to_dict() for source in sources],
            progress=advance,
            loss_mask=loss_mask,
        )


def _check_passes_agree(manifest: DatasetManifest, report: DatasetReport, out_dir: Path) -> None:
    """Fail if the encoding pass read a different corpus than the measuring pass.

    The two passes open the same files minutes apart. If the document counts
    disagree, something wrote to the corpus in between, and the manifest's corpus
    report -- taken from the first pass -- no longer describes the shards. A
    manifest that misdescribes its own shards is worse than no dataset, so the
    files this run wrote are removed.
    """
    written = int(manifest.totals.get("documents", 0))
    if written == report.documents:
        return
    _discard(manifest, out_dir)
    raise DatasetError(
        f"The corpus changed while it was being prepared: the first pass read "
        f"{report.documents:,} documents, the second {written:,}.",
        hint=(
            "Nothing was kept. Make sure no other program is writing to the input "
            "files, then run `trainai data prepare` again."
        ),
        details={
            "first_pass_documents": report.documents,
            "second_pass_documents": written,
            "out_dir": str(out_dir),
        },
    )


def _discard(manifest: DatasetManifest, out_dir: Path) -> None:
    """Delete exactly the files this run wrote, named from the manifest itself."""
    for split in SPLITS:
        for path in manifest.shard_paths(split):
            with suppress(OSError):
                path.unlink()
    for name in (MANIFEST_NAME, TOKENIZER_NAME):
        with suppress(OSError):
            (out_dir / name).unlink()


# --------------------------------------------------------------------------- #
# inspect
# --------------------------------------------------------------------------- #
def run_inspect(
    path: str,
    *,
    verify: bool = False,
    layout: bool = False,
    sample_chars: int = 0,
    encoding: str = "utf-8",
    jsonl_field: str | None = None,
    jsonl_messages_field: str | None = None,
    csv_text_column: str | None = None,
    db_table: str | None = None,
    min_doc_chars: int = 1,
    max_doc_chars: int = 1 << 14,
    on_error: str = "fail",
    json_output: bool = False,
) -> dict[str, Any]:
    """Report on a raw corpus or a prepared dataset. Returns the JSON payload.

    Which of the two it is depends on whether ``path`` holds a manifest, not on a
    flag: pointing this at the wrong kind of directory is the mistake it exists to
    diagnose, so it should not also require getting a flag right.
    """
    target = Path(path)
    if (target / MANIFEST_NAME).exists():
        return _inspect_dataset(target, verify=verify, layout=layout, json_output=json_output)
    return _inspect_corpus(
        target,
        options=_ingest_options(
            encoding=encoding,
            jsonl_field=jsonl_field,
            jsonl_messages_field=jsonl_messages_field,
            csv_text_column=csv_text_column,
            db_table=db_table,
            min_doc_chars=min_doc_chars,
            max_doc_chars=max_doc_chars,
            on_error=on_error,
        ),
        sample_chars=sample_chars,
        json_output=json_output,
    )


def _inspect_dataset(
    directory: Path, *, verify: bool, layout: bool, json_output: bool
) -> dict[str, Any]:
    """Report on a prepared dataset, checking it against its own manifest."""
    manifest = verify_dataset(directory, deep=verify)
    payload = manifest.to_dict()
    payload["checked"] = "checksums" if verify else "sizes"

    if json_output:
        emit_json(payload)
        return payload

    rule(f"Dataset {DASH} {escape(directory.as_posix())}")
    print_kv("Dataset", _dataset_rows(manifest))
    console.print(_splits_table(manifest))
    console.print()
    print_kv("Tokenizer", _manifest_tokenizer_rows(manifest))
    if manifest.corpus:
        print_kv("Corpus it was built from", _manifest_corpus_rows(manifest.corpus))
    if manifest.ingest.get("options"):
        print_kv("Ingest settings", _ingest_option_rows(manifest.ingest["options"]))
    _print_manifest_issues(manifest.validation)

    if layout:
        console.print("[bold cyan]On-disk layout[/]")
        console.print(Padding(escape(describe_dataset_layout()), (0, 0, 1, 2)), highlight=False)

    if verify:
        shards = sum(len(manifest.shards.get(split, ())) for split in SPLITS)
        # Masks are hashed too, so a count of shards alone would understate what was
        # checked by half on a dataset that has them.
        subject = (
            f"{shards} shard(s) and {shards} loss mask(s)"
            if manifest.has_loss_mask
            else f"{shards} shard(s)"
        )
        console.print(
            f"[green]Verified[/] {subject}: every byte matches the sha256 recorded in the manifest."
        )
    else:
        console.print(
            "[dim]Checked that every shard exists at its recorded size. Pass "
            "[/][bold]--verify[/][dim] to re-hash the contents, which is the only "
            "way to catch a flipped bit.[/]"
        )
    return payload


def _inspect_corpus(
    directory: Path,
    *,
    options: IngestOptions,
    sample_chars: int,
    json_output: bool,
) -> dict[str, Any]:
    """Measure a raw corpus without writing anything, then judge it.

    This does the full read that ``prepare`` does, minus tokenization, so on a
    large corpus it is not instant. That is the point: the numbers are counted,
    not sampled.
    """
    ingestor = Ingestor(options)
    sources = ingestor.discover(directory)
    quiet = json_output

    if not quiet:
        rule(f"Corpus {DASH} {escape(directory.as_posix())}")
        print_kv("Input", _input_rows(str(directory), None, sources, replacing=False))

    meter = CorpusMeter()
    sample = ""
    total_bytes = sum(source.size_bytes for source in sources)
    with _progress(quiet=quiet) as progress:
        task = progress.add_task("Reading corpus", total=total_bytes or None)
        for seen, document in enumerate(meter.measure(ingestor.documents(sources)), start=1):
            if sample_chars > 0 and not sample:
                sample = document.text[:sample_chars]
            if seen % _PROGRESS_EVERY == 0:
                progress.update(
                    task,
                    completed=ingestor.stats.total_bytes_read,
                    description=f"Read {fmt_int(seen)} documents",
                )

    report = meter.report()
    validation = validate_corpus(report, ingestor.stats, max_doc_chars=options.max_doc_chars)
    payload = {
        "kind": "corpus",
        "path": directory.as_posix(),
        "sources": [source.to_dict() for source in sources],
        "ingest": {"options": options.to_dict(), "stats": ingestor.stats.to_dict()},
        "corpus": report.to_dict(),
        "validation": validation.to_dict(),
    }

    if quiet:
        emit_json(payload)
        validation.raise_if_failed()
        return payload

    print_kv("Measured", _corpus_rows(report))
    print_kv("Read", _ingest_rows(ingestor.stats))
    if sample:
        console.print("[bold cyan]First document[/]")
        # Indented as a block: the sample is arbitrary user text and may contain
        # newlines, and unindented continuation lines read as TrainAI's own output.
        console.print(Padding(f"[dim]{escape(sample)}[/]", (0, 0, 1, 2)), highlight=False)
    _print_issues(validation)
    if validation.ok:
        console.print("Next:")
        print_command(
            f"trainai data prepare {escape(directory.as_posix())} --out data/prepared",
            style="bold",
        )
    # Exits 3 when the corpus cannot be trained on, so this is usable as a check
    # in a script. The panel it raises carries the fix.
    validation.raise_if_failed()
    return payload


# --------------------------------------------------------------------------- #
# Row builders
# --------------------------------------------------------------------------- #
def _input_rows(
    corpus: str,
    out_dir: Path | None,
    sources: Sequence[SourceFile],
    *,
    replacing: bool,
) -> list[tuple[str, str]]:
    total = sum(source.size_bytes for source in sources)
    kinds: dict[str, int] = {}
    codecs: dict[str, int] = {}
    # Distinct archive *files* per kind, not members: three files out of one zip is
    # "from 1 zip", which is what the user pointed at.
    archives: dict[str, set[Path]] = {}
    for source in sources:
        kinds[source.kind] = kinds.get(source.kind, 0) + 1
        if source.compression != "none":
            codecs[source.compression] = codecs.get(source.compression, 0) + 1
        if source.archive is not None:
            archives.setdefault(source.archive, set()).add(source.path)
    mix = ", ".join(f"{count} {kind}" for kind, count in sorted(kinds.items()))
    # Named per codec rather than lumped as "compressed": a corpus reported as
    # gzipped when it is really bz2 is a small lie that costs someone an hour.
    for codec, count in sorted(codecs.items()):
        mix += f", {count} {_CODEC_WORDS.get(codec, codec)}"
    for archive, paths in sorted(archives.items()):
        mix += f", from {fmt_int(len(paths))} {archive}"

    # A member's size is what it unpacks to, not its share of the archive on disk,
    # because that is the number the reader will see -- so the label stops claiming
    # "on disk" the moment a member is in the total.
    size_label, size_note = "Size on disk", ""
    if archives:
        size_label = "Size"
        size_note = "  [dim]archive members counted unpacked[/]"

    rows = [
        ("Corpus", escape(Path(corpus).as_posix())),
        ("Files", f"[bold]{fmt_int(len(sources))}[/] {DASH} {mix}"),
        (size_label, f"{fmt_bytes(total)}{size_note}"),
    ]
    if out_dir is not None:
        destination = escape(out_dir.as_posix())
        if replacing:
            destination += "  [yellow](replacing the dataset already there)[/]"
        rows.append(("Output", destination))
    biggest = sorted(sources, key=lambda s: (-s.size_bytes, s.relative))[:3]
    for index, source in enumerate(biggest):
        label = "Largest" if index == 0 else ""
        rows.append((label, f"{escape(source.relative)}  [dim]{fmt_bytes(source.size_bytes)}[/]"))
    return rows


def _corpus_rows(report: DatasetReport) -> list[tuple[str, str]]:
    rows = [
        ("Documents", f"[bold]{fmt_int(report.documents)}[/]"),
        (
            "Characters",
            f"[bold]{fmt_int(report.total_chars)}[/]  "
            f"[dim]({fmt_bytes(report.total_utf8_bytes)} as UTF-8)[/]",
        ),
        (
            "Length (chars)",
            f"min {fmt_int(report.min_chars)}  p10 {fmt_int(report.p10_chars)}  "
            f"median {fmt_int(report.median_chars)}  p90 {fmt_int(report.p90_chars)}  "
            f"max {fmt_int(report.max_chars)}",
        ),
    ]

    duplicates = f"{fmt_int(report.duplicate_documents)}  [dim]({report.duplicate_rate:.1%})[/]"
    if report.duplicate_check_truncated:
        duplicates += "  [yellow](check stopped at its memory cap; this is a lower bound)[/]"
    rows.append(("Duplicates", duplicates))

    mix = (
        f"whitespace {report.whitespace_ratio:.0%}  "
        f"non-ASCII {report.non_ascii_ratio:.1%}  "
        f"control {report.control_ratio:.4%}"
    )
    if report.replacement_chars:
        mix += f"  [yellow]{fmt_int(report.replacement_chars)} U+FFFD[/]"
    rows.append(("Character mix", mix))

    # Only shares that survive rounding to a whole percent. "digits 0%" is noise:
    # it says a category was measured, not that it is worth knowing about.
    shares = [(name, share) for name, share in report.script_shares.items() if share >= 0.005]
    if shares:
        top = "  ".join(f"{name} {share:.0%}" for name, share in shares[:4])
        rows.append(
            (
                "Writing systems",
                f"{top}  [dim](over {fmt_int(report.script_sample_chars)} sampled chars)[/]",
            )
        )
    rows.append(
        (
            "Tokens",
            f"~{fmt_count(report.rough_token_estimate)}  [dim]estimate only, at four "
            "chars per token; `data prepare` counts them exactly[/]",
        )
    )
    return rows


def _ingest_rows(stats: IngestStats, *, loss_mask: bool | None = None) -> list[tuple[str, str]]:
    """Ingest counters. Zero-valued rows are omitted except the two that always matter."""
    rows = [
        (
            "Files read",
            f"{fmt_int(stats.files_read)}  [dim]({fmt_bytes(stats.total_bytes_read)})[/]",
        ),
        ("Documents kept", fmt_int(stats.documents_emitted)),
    ]
    if stats.files_skipped:
        rows.append(("Files skipped", f"[yellow]{fmt_int(stats.files_skipped)}[/]"))
        for skipped in stats.skipped_files[:5]:
            rows.append(
                (
                    "",
                    f"{escape(str(skipped.get('path', '?')))}  [dim]{escape(str(skipped.get('reason', '')))}[/]",
                )
            )
    if stats.files_unrecognized:
        # Distinct from "Files skipped": nothing failed here, these were simply never
        # selected. The row exists so "Files read" adds up against the directory the
        # user pointed at, rather than leaving the difference for them to notice.
        listed = ", ".join(escape(name) for name in stats.unrecognized_files[:3])
        rest = stats.files_unrecognized - min(3, len(stats.unrecognized_files))
        if rest > 0:
            listed += f", and {fmt_int(rest)} more"
        rows.append(("Passed over", f"{fmt_int(stats.files_unrecognized)}  [dim]{listed}[/]"))
    if stats.assumed_text_files:
        rows.append(
            (
                "Read as text",
                f"{', '.join(escape(name) for name in stats.assumed_text_files)}"
                "  [dim](no extension to go by)[/]",
            )
        )
    if stats.documents_skipped_short:
        rows.append(("Too short", f"{fmt_int(stats.documents_skipped_short)} dropped"))
    if stats.records_skipped_malformed:
        rows.append(
            ("Malformed records", f"[yellow]{fmt_int(stats.records_skipped_malformed)} skipped[/]")
        )
    if stats.long_files_split:
        rows.append(
            (
                "Split up",
                f"{fmt_int(stats.long_files_split)} file(s) exceeded --max-doc-chars and "
                "were cut at paragraph boundaries",
            )
        )
    if stats.resolved_jsonl_fields:
        fields = ", ".join(
            f"{escape(path)}:{escape(field)}"
            for path, field in list(stats.resolved_jsonl_fields.items())[:3]
        )
        # "JSON field" rather than "JSONL field": the same resolution serves
        # .jsonl, .ndjson and .json, and labelling a .json file's row "JSONL"
        # reads as though the wrong reader ran.
        rows.append(("JSON field", fields))
    if stats.chat_documents:
        # The share is the point of this row. A rendered conversation looks exactly
        # like prose in every other number here -- same document count, same
        # character count -- so the one figure that says the template understood the
        # records is how much of the text turned out to be replies. A template
        # mistake shows up as a share far too low, or at 100% as a template that
        # marked everything.
        share = stats.chat_trained_chars / max(1, stats.chat_chars)
        rows.append(
            (
                "Chat records",
                f"{fmt_int(stats.chat_documents)} rendered, {share:.0%} of characters "
                f"are assistant replies "
                f"({fmt_int(stats.chat_trained_chars)} of {fmt_int(stats.chat_chars)})",
            )
        )
        # Said here rather than only in the docs, because the alternative is a user
        # who reads the share above and cannot tell whether the prompts are excluded
        # from the loss. They are: `trainai train` applies the mask when the dataset
        # carries one, and --no-loss-mask there scores every token instead.
        #
        # ``loss_mask`` is None when nothing is being written -- `data inspect` reads a
        # corpus and produces no dataset -- so the three cases are genuinely
        # different, and collapsing them would either promise a file that was not
        # written or hide one that was.
        if loss_mask is None:
            state = f"measured only {DASH} `trainai data prepare` writes it to disk"
        elif loss_mask:
            state = f"written beside every shard {DASH} `trainai train` scores targets only"
        else:
            state = f"not written {DASH} --no-loss-mask was passed"
        rows.append(("Loss mask", f"[yellow]{state}[/]"))
    if stats.resolved_csv_columns:
        # Named rather than merely counted: the whole corpus came out of these
        # columns and nothing else in the file was read, so a user who expected a
        # different column should be able to see that here.
        columns = ", ".join(
            f"{escape(path)}:{escape(column)}"
            for path, column in list(stats.resolved_csv_columns.items())[:3]
        )
        rows.append(("CSV column", f"{columns}  [dim](chosen automatically)[/]"))
    for label, resolved in (
        ("DB table", stats.resolved_db_tables),
        ("DB column", stats.resolved_db_columns),
    ):
        # Same reason as the CSV row above, and load-bearing for one guard in
        # particular: the table is chosen automatically after a virtual table's
        # storage tables are filtered out by name, and naming the choice here is
        # what makes a wrong one visible rather than silent.
        if resolved:
            named = ", ".join(
                f"{escape(path)}:{escape(name)}" for path, name in list(resolved.items())[:3]
            )
            rows.append((label, f"{named}  [dim](chosen automatically)[/]"))
    return rows


def _reused_tokenizer_rows(tokenizer: ByteLevelBPE, source: str | None) -> list[tuple[str, str]]:
    """The tokenizer section when ``--tokenizer`` supplied it rather than this run.

    The fingerprint leads, because it is the thing the user is trying to match: it is
    what ``train`` and ``finetune`` compare against a checkpoint, and a run that reused
    the wrong file has no other symptom.
    """
    return [
        ("Vocabulary", f"[bold]{fmt_int(tokenizer.vocab_size)}[/]  [dim](reused, not trained)[/]"),
        ("End-of-text id", str(tokenizer.eot_id)),
        ("Fingerprint", f"[dim]{tokenizer.fingerprint()[:16]}[/]"),
        ("Reused from", f"[dim]{source}[/]"),
    ]


def _load_reference_tokenizer(path: str, vocab_size: int) -> ByteLevelBPE:
    """Load the tokenizer named by ``--tokenizer``, accepting a dataset directory too.

    ``--vocab-size`` alongside it is refused rather than ignored. The loaded file's
    vocabulary is a fact about that file; a second number describing what this run
    would have trained cannot be honoured, and silently dropping a flag the user typed
    is how a dataset ends up not being the one they think they asked for.
    """
    if vocab_size != DEFAULT_VOCAB_SIZE:
        raise UsageError(
            "--vocab-size cannot be combined with --tokenizer.",
            hint=(
                "A reused tokenizer already has its vocabulary; there is nothing for "
                f"--vocab-size {fmt_int(vocab_size)} to change. Drop one of the two."
            ),
            details={"tokenizer": path, "vocab_size": vocab_size},
        )
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / TOKENIZER_NAME
    return ByteLevelBPE.load(candidate)


def _tokenizer_rows(tokenizer: ByteLevelBPE, requested: int) -> list[tuple[str, str]]:
    vocab = f"[bold]{fmt_int(tokenizer.vocab_size)}[/]"
    if tokenizer.vocab_shortfall:
        vocab += (
            f"  [yellow]({fmt_int(tokenizer.vocab_shortfall)} short of the "
            f"{fmt_int(requested)} requested; the corpus had no more repeated pairs "
            "to merge)[/]"
        )
    return [
        ("Vocabulary", vocab),
        ("End-of-text id", str(tokenizer.eot_id)),
        ("Fingerprint", f"[dim]{tokenizer.fingerprint()[:16]}[/]"),
    ]


def _dataset_rows(manifest: DatasetManifest) -> list[tuple[str, str]]:
    rows = [
        ("Format", f"{manifest.format} v{manifest.format_version}"),
        ("Created", f"{manifest.created_at or 'unknown'}  [dim]by {manifest.created_with}[/]"),
        (
            "Content hash",
            f"[dim]{manifest.content_hash[:16]}[/]  [dim]{DASH} identical for identical "
            "input, settings and seed[/]",
        ),
        ("Seed", str(manifest.seed)),
        ("Val fraction", f"{manifest.val_fraction:.3g}"),
        (
            "Token width",
            f"{manifest.dtype}  [dim]({manifest.numpy_dtype.itemsize} bytes per token, "
            "little-endian)[/]",
        ),
    ]
    rows.extend(_loss_mask_rows(manifest))
    rows.extend(_chat_template_rows(manifest))
    return rows


def _chat_template_rows(manifest: DatasetManifest) -> list[tuple[str, str]]:
    """The template the documents were rendered with, or nothing for a plain corpus.

    Worth a row of its own rather than a footnote on the loss mask: the two are
    independent. A chat corpus prepared with --no-loss-mask has a template and no mask,
    and it is the template -- not the mask -- that a model trained on this dataset has
    to be prompted in.
    """
    template = manifest.chat
    if not template:
        return []
    labels = template.get("labels") or {}
    trained = template.get("trained_roles") or []
    return [
        (
            "Chat template",
            f"v{template.get('version', '?')}  [dim]{DASH} "
            f"{', '.join(f'{label}:' for label in labels.values())}[/]",
        ),
        (
            "",
            f"[dim]{DASH} trained on {', '.join(trained) or 'nothing'}; a model trained "
            "on this dataset has to be prompted in the same layout, which trainai chat "
            "does for it[/]",
        ),
    ]


def _loss_mask_rows(manifest: DatasetManifest) -> list[tuple[str, str]]:
    """What the mask on disk says, or nothing at all if there is no mask.

    The share here is over *tokens*, and the one the ingest report gives is over
    characters, so the two will not match exactly and are labelled so that nobody
    tries to reconcile them: a reply of 40 characters is not 40% of a document's
    tokens when the tokenizer compresses prose and prompts differently.
    """
    if not manifest.has_loss_mask:
        return []
    total = manifest.total_tokens
    trained = int(manifest.totals.get("trained_tokens", 0))
    straddling = int(manifest.totals.get("straddling_tokens", 0))
    share = trained / total if total else 0.0
    rows = [
        (
            "Loss mask",
            f"{fmt_int(trained)} of {fmt_int(total)} tokens are targets ({share:.0%})"
            f"  [dim]{DASH} one uint8 per token, beside each shard[/]",
        ),
        (
            "",
            f"[dim]{DASH} `trainai train` scores those tokens only; "
            f"--no-loss-mask scores every token[/]",
        ),
    ]
    if straddling:
        # Zero on this repo's corpus, and reported when it is not: a token covering
        # text on both sides of a span edge is scored on characters the mask calls
        # context. It is a property of the corpus, not a failure, but an unreported
        # one would make the share above look more exact than it is.
        rows.append(
            (
                "",
                f"[yellow]{fmt_int(straddling)} tokens straddle a reply boundary[/] "
                f"[dim]{DASH} each is counted as a target[/]",
            )
        )
    return rows


def _manifest_tokenizer_rows(manifest: DatasetManifest) -> list[tuple[str, str]]:
    tokens = manifest.total_tokens
    chars = int(manifest.totals.get("chars", 0))
    utf8 = int(manifest.totals.get("utf8_bytes", 0))
    rows = [
        ("Vocabulary", f"[bold]{fmt_int(manifest.vocab_size)}[/]"),
        ("End-of-text id", str(manifest.eot_id)),
        ("Fingerprint", f"[dim]{manifest.tokenizer_fingerprint[:16]}[/]"),
    ]
    requested = manifest.tokenizer.get("requested_vocab_size")
    if isinstance(requested, int) and requested > manifest.vocab_size:
        rows.insert(
            1,
            (
                "Requested",
                f"{fmt_int(requested)}  [yellow]({fmt_int(requested - manifest.vocab_size)} "
                "fewer were learnable from this corpus)[/]",
            ),
        )
    if tokens and chars:
        rows.append(
            (
                "Compression",
                f"[bold]{chars / tokens:.2f}[/] chars per token  "
                f"[dim]({utf8 / tokens:.2f} UTF-8 bytes per token, measured over the "
                "whole corpus)[/]",
            )
        )
    return rows


def _manifest_corpus_rows(corpus: dict[str, Any]) -> list[tuple[str, str]]:
    lengths = corpus.get("length_chars", {})
    duplicates = corpus.get("duplicates", {})
    mix = corpus.get("character_mix", {})
    shares = corpus.get("scripts", {}).get("shares", {})
    rows = [
        ("Documents", fmt_int(int(corpus.get("documents", 0)))),
        (
            "Characters",
            f"{fmt_int(int(corpus.get('total_chars', 0)))}  "
            f"[dim]({fmt_bytes(int(corpus.get('total_utf8_bytes', 0)))} as UTF-8)[/]",
        ),
        ("Sources", fmt_int(int(corpus.get("sources", 0)))),
    ]
    if lengths:
        rows.append(
            (
                "Length (chars)",
                f"min {fmt_int(int(lengths.get('min', 0)))}  "
                f"median {fmt_int(int(lengths.get('median', 0)))}  "
                f"max {fmt_int(int(lengths.get('max', 0)))}",
            )
        )
    if duplicates:
        rows.append(
            (
                "Duplicates",
                f"{fmt_int(int(duplicates.get('duplicate_documents', 0)))}  "
                f"[dim]({float(duplicates.get('rate', 0.0)):.1%})[/]",
            )
        )
    if mix:
        rows.append(
            (
                "Character mix",
                f"whitespace {float(mix.get('whitespace_ratio', 0.0)):.0%}  "
                f"non-ASCII {float(mix.get('non_ascii_ratio', 0.0)):.1%}",
            )
        )
    notable = [(name, float(share)) for name, share in shares.items() if float(share) >= 0.005]
    if notable:
        top = "  ".join(f"{name} {share:.0%}" for name, share in notable[:4])
        rows.append(("Writing systems", top))
    return rows


def _ingest_option_rows(options: dict[str, Any]) -> list[tuple[str, str]]:
    """The ingest settings recorded in the manifest, so a run can be repeated."""
    rows = [
        ("Encoding", escape(str(options.get("encoding", "utf-8")))),
        ("JSONL field", escape(str(options.get("jsonl_field") or "auto-detected"))),
        ("CSV column", escape(str(options.get("csv_text_column") or "auto-detected"))),
        ("DB table", escape(str(options.get("db_table") or "auto-detected"))),
        (
            "Document chars",
            f"min {fmt_int(int(options.get('min_doc_chars', 0)))}, "
            f"max {fmt_int(int(options.get('max_doc_chars', 0)))}",
        ),
        ("On decode error", escape(str(options.get("on_error", "fail")))),
    ]
    # Only when it was used. The other rows describe a setting every corpus has;
    # this one says the dataset was built from typed conversations with a loss mask,
    # which is a different kind of fact and worth its own line rather than an
    # "auto-detected" that would be meaningless here.
    messages_field = options.get("jsonl_messages_field")
    if messages_field:
        rows.insert(2, ("Chat messages field", escape(str(messages_field))))
    return rows


def _splits_table(manifest: DatasetManifest) -> Table:
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, header_style="bold cyan")
    table.add_column("Split", no_wrap=True)
    table.add_column("Documents", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Shards", justify="right")
    table.add_column("On disk", justify="right")
    # The mask is one byte per token beside shards of two, so it is half again as much
    # disk. Counting only the token shards would print a total the user can see is
    # wrong by looking at the directory, which is the kind of small dishonesty that
    # makes every other number here worth less.
    masked = manifest.has_loss_mask

    def on_disk(shards: Iterable[ShardInfo]) -> int:
        return sum(shard.bytes + (shard.mask_bytes or 0) for shard in shards)

    for split in SPLITS:
        shards = manifest.shards.get(split, ())
        table.add_row(
            split,
            fmt_int(manifest.documents.get(split, 0)),
            fmt_int(manifest.tokens(split)),
            str(len(shards) * 2 if masked else len(shards)),
            fmt_bytes(on_disk(shards)),
        )
    table.add_section()
    total_shards = sum(len(manifest.shards.get(split, ())) for split in SPLITS)
    total_bytes = on_disk(shard for split in SPLITS for shard in manifest.shards.get(split, ()))
    table.add_row(
        "[bold]total[/]",
        f"[bold]{fmt_int(sum(manifest.documents.get(s, 0) for s in SPLITS))}[/]",
        f"[bold]{fmt_int(manifest.total_tokens)}[/]",
        f"[bold]{total_shards * 2 if masked else total_shards}[/]",
        f"[bold]{fmt_bytes(total_bytes)}[/]",
    )
    return table


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #
def _print_issues(validation: ValidationResult) -> None:
    items = [_finding(issue.level, issue.message, issue.hint) for issue in validation.issues]
    print_bullets(
        "Findings",
        items,
        empty="[green]Nothing to flag.[/] Nothing about this corpus looks unusual.",
    )


def _print_manifest_issues(validation: dict[str, Any]) -> None:
    """Render the validation verdict stored in a manifest at preparation time."""
    items = [
        _finding(
            str(issue.get("level", "")),
            str(issue.get("message", "")),
            str(issue.get("hint", "")),
        )
        for issue in validation.get("issues") or []
    ]
    print_bullets(
        "Findings recorded at preparation time",
        items,
        empty="[green]Nothing was flagged[/] when this dataset was prepared.",
    )


def _finding(level: str, message: str, hint: str) -> str:
    """One finding as a two-line bullet: what it is, then what to do about it.

    The hint carries no extra indentation of its own. ``print_bullets`` already
    aligns wrapped lines under the text column, and a second indent inside the
    string would leave the hint's first line out of step with its own wrapping.
    """
    return f"{_LEVEL_LABEL.get(level, level)} {escape(message)}\n[dim]{escape(hint)}[/]"


# --------------------------------------------------------------------------- #
# Progress
# --------------------------------------------------------------------------- #
def _progress(*, quiet: bool) -> Progress:
    """A progress display that removes itself, and prints nothing when quiet.

    ``transient`` keeps the final report clean; ``disable`` is what makes ``--json``
    safe to pipe, since the bar would otherwise be written to the same stdout.
    """
    return Progress(
        SpinnerColumn(SPINNER),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
        disable=quiet,
    )


def _print_result(manifest: DatasetManifest, report: DatasetReport, out_dir: Path) -> None:
    rule("Result")
    console.print(_splits_table(manifest))
    console.print()
    print_kv("Dataset", _dataset_rows(manifest))
    print_kv("Tokenizer", _manifest_tokenizer_rows(manifest))
    console.print(
        f"Wrote [bold]{escape(out_dir.as_posix())}[/] {DASH} "
        f"{fmt_int(manifest.total_tokens)} tokens from "
        f"{fmt_int(report.documents)} documents."
    )
    console.print("Next: review it with")
    print_command(f"trainai data inspect {escape(out_dir.as_posix())}", style="bold")
