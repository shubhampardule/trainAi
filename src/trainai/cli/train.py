"""``trainai train`` -- build a model, train it, and record what happened.

This module is deliberately thin. Its job is to turn flags into three objects
(:class:`ModelConfig`, :class:`TrainConfig`, and a run directory), report what it is
about to do, and hand off to :class:`~trainai.train.loop.Trainer`. Anything that
looks like a training decision belongs in the library, because the web interface
(M5) has to make the same decisions without going through argument parsing.

Two behaviours worth knowing about before reading the code:

``--dry-run`` builds both configurations, applies every check that does not need a
device or the token shards, prints the shape, the parameter count and the data
budget, and exits without touching the GPU. On a corpus that cannot support the
requested model, that is a two-second answer instead of a wasted evening. It is
also why the configuration checks live in :func:`~trainai.train.loop.check_configs_agree`
rather than inline in ``Trainer``: a dry run that approves a run the next command
refuses is worse than no dry run at all, because it gets trusted.

An existing run directory with checkpoints in it is refused unless ``--resume``
says to continue it or ``--force`` says to overwrite. Training into a directory
that already holds a run silently mixes two runs' metrics into one file and leaves
checkpoints from both, which is unrecoverable after the fact.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from trainai.console import (
    DASH,
    console,
    emit_json,
    fmt_bytes,
    fmt_count,
    fmt_duration,
    fmt_int,
    print_bullets,
    print_command,
    print_kv,
    rule,
)
from trainai.data.binarize import DatasetManifest, verify_dataset
from trainai.errors import ConfigError, UsageError, json_type_name
from trainai.hardware.planner import PLAN_FILENAME, PLAN_VERSION
from trainai.model.config import PRESETS, ModelConfig, preset
from trainai.train.budget import DataBudget
from trainai.train.checkpoint import find_checkpoint, list_checkpoints
from trainai.train.config import TrainConfig
from trainai.train.loop import (
    Trainer,
    TrainResult,
    check_configs_agree,
    resolve_device,
    resolve_precision,
)
from trainai.train.metrics import format_perplexity

__all__ = ["run_train"]

#: Default model shape. The smallest preset, on purpose: it fits on every GPU this
#: project targets and finishes a first run in minutes. It is not a recommendation
#: about what is *best* for the user's hardware -- making that call from real
#: measurements is what M3's `trainai plan` is for, and guessing it here from a
#: table would be exactly the formula-instead-of-measurement mistake this project
#: exists to avoid.
DEFAULT_PRESET = "tiny"


def run_train(
    data: str,
    *,
    out: str | None = None,
    name: str | None = None,
    resume: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    json_output: bool = False,
    verify: bool = False,
    plan: str | None = None,
    # model
    model_preset: str | None = None,
    n_layer: int | None = None,
    n_head: int | None = None,
    n_kv_head: int | None = None,
    d_model: int | None = None,
    d_ff: int | None = None,
    context: int | None = None,
    dropout: float | None = None,
    tie_embeddings: bool | None = None,
    # training
    steps: int | None = None,
    batch_size: int | None = None,
    grad_accum: int | None = None,
    seq_len: int | None = None,
    lr: float | None = None,
    min_lr_ratio: float | None = None,
    warmup_steps: int | None = None,
    schedule: str | None = None,
    weight_decay: float | None = None,
    grad_clip: float | None = None,
    eval_every: int | None = None,
    eval_batches: int | None = None,
    checkpoint_every: int | None = None,
    keep_checkpoints: int | None = None,
    log_every: int | None = None,
    seed: int | None = None,
    precision: str | None = None,
    device: str | None = None,
) -> TrainResult | dict[str, Any]:
    """Train a model on a prepared dataset.

    Returns the :class:`TrainResult`, or the dry-run report when ``dry_run`` is set.
    """
    quiet = json_output
    dataset = verify_dataset(data, deep=verify) if verify else DatasetManifest.load(data)
    model_flags: dict[str, Any] = {
        "n_layer": n_layer,
        "n_head": n_head,
        "n_kv_head": n_kv_head,
        "d_model": d_model,
        "d_ff": d_ff,
        "seq_len": context,
        "dropout": dropout,
        "tie_embeddings": tie_embeddings,
    }
    train_flags: dict[str, Any] = {
        "steps": steps,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "seq_len": seq_len,
        "lr": lr,
        "min_lr_ratio": min_lr_ratio,
        "warmup_steps": warmup_steps,
        "schedule": schedule,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "eval_every": eval_every,
        "eval_batches": eval_batches,
        "checkpoint_every": checkpoint_every,
        "keep_checkpoints": keep_checkpoints,
        "log_every": log_every,
        "seed": seed,
        "precision": precision,
        "device": device,
    }

    if plan is not None:
        # A plan already contains a complete pair of configurations, measured on this
        # machine. Explicit flags still win over it, so a user who disagrees with one
        # number does not have to abandon the whole recommendation.
        model_config, train_config = _configs_from_plan(plan, dataset, data)
        model_config = _with_overrides(model_config, model_flags)
        train_config = _with_overrides(train_config, train_flags)
    else:
        model_config = _build_model_config(
            dataset,
            model_preset=model_preset or DEFAULT_PRESET,
            n_layer=n_layer,
            n_head=n_head,
            n_kv_head=n_kv_head,
            d_model=d_model,
            d_ff=d_ff,
            context=context,
            dropout=dropout,
            tie_embeddings=tie_embeddings,
        )
        train_config = _build_train_config(dataset, model_config, **train_flags)
    run_dir = _resolve_run_dir(out, name, data, resume=resume, force=force)

    # Before the dry-run branch, not after it. These are the two ways the pair of
    # configurations can be impossible, and they used to live only in ``Trainer``,
    # which a dry run returns before constructing -- so `--context 128 --seq-len 512
    # --dry-run` printed a plan reading "ctx128" beside "Sequence 512 tokens" and
    # exited 0, and the same flags without --dry-run exited 6.
    check_configs_agree(dataset, model_config, train_config)

    if dry_run:
        return _report_dry_run(dataset, model_config, train_config, run_dir, quiet=quiet, data=data)

    trainer = Trainer(
        dataset=dataset,
        model_config=model_config,
        train_config=train_config,
        run_dir=run_dir,
        quiet=quiet,
    )
    if resume:
        trainer.resume(_resolve_resume_path(resume))

    if not quiet:
        rule("Training")
        print_kv("Run", _run_rows(data, dataset, run_dir, trainer))
        _print_budget(trainer.budget)

    result = trainer.run()

    if quiet:
        emit_json(result.to_dict())
    else:
        _print_result(result, trainer)
    return result


# --------------------------------------------------------------------------- #
# Turning flags into configuration
# --------------------------------------------------------------------------- #
def _build_model_config(
    dataset: DatasetManifest,
    *,
    model_preset: str,
    n_layer: int | None,
    n_head: int | None,
    n_kv_head: int | None,
    d_model: int | None,
    d_ff: int | None,
    context: int | None,
    dropout: float | None,
    tie_embeddings: bool | None,
) -> ModelConfig:
    """Start from a preset and apply any explicit overrides.

    ``vocab_size`` is never a flag. It is decided by the tokenizer the dataset was
    prepared with, and a model whose embedding does not match cannot represent its
    own training data.
    """
    overrides: dict[str, Any] = {}
    for key, value in (
        ("n_layer", n_layer),
        ("n_head", n_head),
        ("n_kv_head", n_kv_head),
        ("d_model", d_model),
        ("d_ff", d_ff),
        ("seq_len", context),
        ("dropout", dropout),
        ("tie_embeddings", tie_embeddings),
    ):
        if value is not None:
            overrides[key] = value
    return preset(model_preset, vocab_size=dataset.vocab_size, **overrides)


def _with_overrides(config: Any, flags: dict[str, Any]) -> Any:
    """Apply the flags that were actually given, leaving the rest of ``config`` alone.

    Used by ``--plan``: the plan supplies a complete configuration, and this puts the
    user's explicit flags back on top. Unknown keys are dropped rather than raising,
    because the two flag dictionaries are shared with the non-plan path and a model
    flag has no business reaching a :class:`TrainConfig`.
    """
    fields = type(config).__dataclass_fields__
    given = {key: value for key, value in flags.items() if value is not None and key in fields}
    return replace(config, **given) if given else config


def _encoding_hint(raw: bytes, data: str) -> str:
    """What to tell someone whose plan file is not UTF-8.

    A UTF-16 BOM is worth naming, because it is not something a user does on purpose: it
    is what "Save as -> Unicode" produces in Notepad, and the resulting file looks
    perfectly normal in the editor that wrote it. Naming the encoding turns "is not UTF-8
    text" from a fact into an instruction.
    """
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return (
            "It looks like UTF-16. Re-save it as UTF-8 -- in Notepad that is the "
            f"encoding dropdown in the Save As dialog -- or re-run `trainai plan "
            f"--data {data}` to write a fresh one."
        )
    return f"Re-save it as UTF-8, or re-run `trainai plan --data {data}` to write a fresh one."


def _configs_from_plan(
    path: str, dataset: DatasetManifest, data: str
) -> tuple[ModelConfig, TrainConfig]:
    """Rebuild the two configurations a ``trainai plan`` run wrote to disk.

    Every way this can fail becomes a :class:`UsageError` naming the plan file. That is
    the whole contract: a plan is a file on disk that the tool itself prints in order to
    be argued with, so a hand-edited one is ordinary input, and a user with several plans
    needs to know *which* file was refused.

    Keeping that contract is why the document and the two blocks are shape-checked before
    anything is handed to ``from_dict``, and why :class:`ConfigError` is caught. Catching
    an enumeration of builtin exception types did not hold: a block that is a string
    reached ``raw.items()`` and raised ``AttributeError``, which surfaced as a traceback
    and exit 1 instead of the documented exit 2, and a block that fails a validator raised
    ``ConfigError``, which surfaced with a hint about re-creating a *checkpoint* with
    ``trainai train`` -- wrong artifact, wrong command. Neither named the file.

    ``isinstance`` rather than ``.get()`` for the same reason one level up: a file holding
    valid JSON that is not an object has no ``.get``. Absent and ``null`` are also reported
    apart, because "has no ``model`` block" and "its ``model`` block is null" send a user
    editing the file to two different places.

    The vocabulary check is not a formality. A plan is measured against one dataset's
    tokenizer, and a model whose embedding does not match the data it is fed cannot
    represent its own training set -- it would train, slowly, to nothing.

    The bytes are decoded before they are parsed, because the contract above did not hold
    for a file that is not UTF-8. ``UnicodeDecodeError`` is a ``ValueError``, so neither
    the ``FileNotFoundError`` nor the ``json.JSONDecodeError`` guard below ever saw it:
    measured on ten damaged plan files, a stray non-UTF-8 byte and a UTF-16 file each
    ended the command with a bare traceback and exit 1 instead of the documented exit 2.
    UTF-16 is not a hypothetical -- it is one entry in the encoding dropdown of the
    editor most likely to open this file on the platform this project targets first.

    A UTF-8 BOM is *accepted* rather than reported, which is why the encoding is
    ``utf-8-sig``. It was refused as "not valid JSON", which sends someone hunting for a
    syntax error in a file that has none; and unlike ``manifest.json``, this file is
    documented as one to open and argue with, so the editor that adds a BOM by default
    is being used as intended. ``utf-8-sig`` is a no-op on a file without one.
    """
    file = Path(path)
    if file.is_dir():
        file = file / PLAN_FILENAME
    try:
        raw_bytes = file.read_bytes()
    except FileNotFoundError as exc:
        raise UsageError(
            f"No plan file at {file.as_posix()}.",
            hint=f"Run `trainai plan --data {data}` first: it writes {PLAN_FILENAME}.",
            details={"path": str(file)},
        ) from exc
    except OSError as exc:
        raise UsageError(
            f"{file.as_posix()} could not be read.",
            hint="Check the path and the file's permissions.",
            details={"path": str(file), "error": str(exc)},
        ) from exc

    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UsageError(
            f"{file.as_posix()} is not UTF-8 text.",
            hint=_encoding_hint(raw_bytes, data),
            details={"path": str(file), "error": str(exc)},
        ) from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise UsageError(
            f"{file.as_posix()} is not valid JSON.",
            hint=f"Re-run `trainai plan --data {data}` to write a fresh one.",
            details={"path": str(file), "error": str(exc)},
        ) from exc

    if not isinstance(payload, dict):
        raise UsageError(
            f"{file.as_posix()} is {json_type_name(payload)}, not a JSON object.",
            hint=f"Re-run `trainai plan --data {data}` to write a fresh one.",
            details={"path": str(file), "found": json_type_name(payload)},
        )

    version = payload.get("version")
    if version != PLAN_VERSION:
        raise UsageError(
            f"{file.as_posix()} is a version {version!r} plan and this build reads "
            f"version {PLAN_VERSION}.",
            hint=f"Re-run `trainai plan --data {data}` to write one this build reads.",
            details={"path": str(file), "found": version, "expected": PLAN_VERSION},
        )
    for name in ("model", "train"):
        if name not in payload:
            raise UsageError(
                f"{file.as_posix()} has no `{name}` block.",
                hint=f"Re-run `trainai plan --data {data}` to write a fresh one.",
                details={"path": str(file), "block": name, "found": "absent"},
            )
        if not isinstance(payload[name], dict):
            raise UsageError(
                f"The `{name}` block in {file.as_posix()} is "
                f"{json_type_name(payload[name])}, not an object.",
                hint=f"Re-run `trainai plan --data {data}` to write a fresh one.",
                details={
                    "path": str(file),
                    "block": name,
                    "found": json_type_name(payload[name]),
                },
            )

    def build(name: str, builder: Callable[[], Any]) -> Any:
        """Turn any refusal from a deserialiser into one that names the file and block."""
        try:
            return builder()
        except (ConfigError, KeyError, TypeError, ValueError) as exc:
            raise UsageError(
                f"The `{name}` block in {file.as_posix()} cannot be loaded: {exc}",
                hint=f"Re-run `trainai plan --data {data}` to write a fresh one.",
                details={
                    "path": str(file),
                    "block": name,
                    "error": f"{exc.__class__.__name__}: {exc}",
                },
            ) from exc

    model_config: ModelConfig = build("model", lambda: ModelConfig.from_dict(payload["model"]))
    train_config: TrainConfig = build("train", lambda: TrainConfig.from_dict(payload["train"]))

    if model_config.vocab_size != dataset.vocab_size:
        raise UsageError(
            f"{file.as_posix()} is for a vocabulary of {model_config.vocab_size} and "
            f"{data} has {dataset.vocab_size}.",
            hint=(
                "The plan was measured against a different dataset. Re-run "
                f"`trainai plan --data {data}` against this one."
            ),
            details={
                "path": str(file),
                "plan_vocab_size": model_config.vocab_size,
                "dataset_vocab_size": dataset.vocab_size,
                "plan_dataset": payload.get("dataset", ""),
            },
        )
    return model_config, train_config


def _build_train_config(
    dataset: DatasetManifest,
    model_config: ModelConfig,
    **flags: Any,
) -> TrainConfig:
    """Apply the flags that were given, deriving the two that have no useful default.

    ``steps`` and ``seq_len`` are derived rather than fixed: a default step count
    means nothing without knowing the corpus size, and the sequence length has to
    fit the model's context. Everything else has a defensible constant default in
    :class:`TrainConfig`.
    """
    given = {key: value for key, value in flags.items() if value is not None}
    seq_len = int(given.pop("seq_len", min(model_config.seq_len, 256)))
    batch_size = int(given.pop("batch_size", TrainConfig.batch_size))
    grad_accum = int(given.pop("grad_accum", TrainConfig.grad_accum))

    if "steps" in given:
        steps = int(given.pop("steps"))
    else:
        # Default to a run that stays inside the memorising threshold: enough steps
        # to be worth doing, few enough that the model is not shown the same text a
        # dozen times. Derived from the corpus, because a constant cannot be right
        # for both a 1 MB and a 1 GB dataset.
        tokens_per_step = batch_size * grad_accum * seq_len
        train_tokens = max(1, dataset.tokens("train"))
        steps = max(50, min(20_000, int(3.0 * train_tokens / tokens_per_step)))

    return TrainConfig(
        steps=steps,
        batch_size=batch_size,
        grad_accum=grad_accum,
        seq_len=seq_len,
        **given,
    )


def _resolve_run_dir(
    out: str | None,
    name: str | None,
    data: str,
    *,
    resume: str | None,
    force: bool,
) -> Path:
    """Decide where the run lives, refusing to write two runs into one directory."""
    if out is not None:
        run_dir = Path(out)
    else:
        label = name or f"{Path(data).name}-run"
        run_dir = Path("runs") / label

    existing = list_checkpoints(run_dir / "checkpoints")
    if existing and not resume and not force:
        raise UsageError(
            f"{run_dir} already holds a run with {len(existing)} checkpoint(s).",
            hint=(
                f"Continue it with `--resume {(run_dir / 'checkpoints').as_posix()}`, "
                "choose a different --name, or pass --force to overwrite. Training "
                "into an existing run directory would mix two runs' metrics into one "
                "file and leave checkpoints from both."
            ),
            details={"run_dir": str(run_dir), "checkpoints": len(existing)},
        )
    return run_dir


def _resolve_resume_path(resume: str) -> Path:
    """Accept a run directory, its checkpoints directory, or a checkpoint file."""
    path = Path(resume)
    if path.is_file():
        return path
    for candidate in (path, path / "checkpoints"):
        found = find_checkpoint(candidate)
        if found is not None:
            return found
    raise UsageError(
        f"No checkpoint found at {resume}.",
        hint=(
            "Point --resume at a run directory, its checkpoints directory, or a "
            "single step-*.pt file."
        ),
        details={"path": str(path)},
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _report_dry_run(
    dataset: DatasetManifest,
    model_config: ModelConfig,
    train_config: TrainConfig,
    run_dir: Path,
    *,
    quiet: bool,
    data: str,
) -> dict[str, Any]:
    """Describe the run without starting it.

    Uses the analytic parameter count rather than building the model, so this stays
    instant and needs no GPU. That count is checked against a real module by the
    test suite, which is what makes it safe to report here.
    """
    budget = DataBudget(
        train_tokens=dataset.tokens("train"),
        val_tokens=dataset.tokens("val"),
        parameters=model_config.parameter_count,
        non_embedding_parameters=model_config.non_embedding_parameter_count,
        tokens_per_step=train_config.tokens_per_step,
        steps=train_config.steps,
    )
    device = resolve_device(train_config.device)
    _, _, precision_note = resolve_precision(train_config.precision, device)
    payload = {
        "dry_run": True,
        "dataset": data,
        "run_dir": str(run_dir),
        "model": model_config.to_dict(),
        "train": train_config.to_dict(),
        "budget": budget.to_dict(),
        "device": str(device),
        "precision": precision_note,
        "warnings": budget.warnings(),
    }
    if quiet:
        emit_json(payload)
        return payload

    rule("Plan")
    print_kv("Dataset", _dataset_rows(data, dataset))
    print_kv("Model", _model_rows(model_config))
    print_kv("Training", _training_rows(train_config, device, precision_note, run_dir))
    _print_budget(budget)
    console.print(
        "[dim]Nothing was trained. Remove --dry-run to start, or adjust the flags above first.[/]"
    )
    return payload


def _dataset_rows(data: str, dataset: DatasetManifest) -> list[tuple[str, str]]:
    return [
        ("Location", Path(data).as_posix()),
        (
            "Tokens",
            f"[bold]{fmt_int(dataset.tokens('train'))}[/] train, "
            f"{fmt_int(dataset.tokens('val'))} validation",
        ),
        ("Vocabulary", fmt_int(dataset.vocab_size)),
        ("Tokenizer", f"[dim]{dataset.tokenizer_fingerprint[:16]}[/]"),
    ]


def _model_rows(config: ModelConfig) -> list[tuple[str, str]]:
    breakdown = config.breakdown()
    rows = [
        ("Shape", f"[bold]{config.describe()}[/]"),
        (
            "Parameters",
            f"[bold]{fmt_count(config.parameter_count)}[/]  "
            f"[dim]({fmt_count(config.non_embedding_parameter_count)} outside the "
            "embedding)[/]",
        ),
        (
            "Where they are",
            "  ".join(f"{name} {fmt_count(count)}" for name, count in breakdown.items() if count),
        ),
        (
            "Weights",
            f"{fmt_bytes(config.parameter_bytes())} in fp32  "
            f"[dim](plus the same again for AdamW's two moments)[/]",
        ),
    ]
    if config.uses_grouped_query_attention:
        rows.append(
            (
                "Attention",
                f"grouped-query: {config.n_head} query heads share {config.kv_heads} "
                f"key/value heads ({config.kv_groups} to 1)",
            )
        )
    return rows


def _training_rows(
    config: TrainConfig, device: Any, precision_note: str, run_dir: Path
) -> list[tuple[str, str]]:
    batch = f"{config.batch_size}"
    if config.grad_accum > 1:
        batch += f" x {config.grad_accum} accumulated = [bold]{config.effective_batch_size}[/]"
    return [
        ("Steps", f"[bold]{fmt_int(config.steps)}[/]"),
        ("Batch", batch),
        ("Sequence", f"{fmt_int(config.seq_len)} tokens"),
        (
            "Tokens",
            f"{fmt_count(config.tokens_per_step)} per step, "
            f"[bold]{fmt_count(config.total_tokens)}[/] in total",
        ),
        (
            "Learning rate",
            f"{config.lr:g} peak, {config.schedule} to {config.min_lr:g}, "
            f"{fmt_int(config.resolved_warmup_steps)} warmup steps",
        ),
        ("Device", f"[bold]{device}[/]  {DASH}  {precision_note}"),
        ("Run directory", run_dir.as_posix()),
    ]


def _run_rows(
    data: str, dataset: DatasetManifest, run_dir: Path, trainer: Trainer
) -> list[tuple[str, str]]:
    config = trainer.train_config
    return [
        (
            "Dataset",
            f"{Path(data).as_posix()}  [dim]{fmt_int(dataset.tokens('train'))} train tokens[/]",
        ),
        (
            "Model",
            f"[bold]{trainer.model_config.describe()}[/]  "
            f"{fmt_count(trainer.model.parameter_count())} params",
        ),
        (
            "Schedule",
            f"{fmt_int(config.steps)} steps of {fmt_count(config.tokens_per_step)} tokens, "
            f"lr {config.lr:g} {config.schedule}",
        ),
        ("Device", f"{trainer.device}  {DASH}  {trainer.precision_note}"),
        ("Writing to", run_dir.as_posix()),
    ]


def _print_budget(budget: DataBudget) -> None:
    print_kv(
        "Data budget",
        [
            (
                "Epochs",
                f"[bold]{budget.epochs:.1f}[/] passes over the training split  "
                f"[dim]({budget.steps_per_epoch:.0f} steps per epoch)[/]",
            ),
            (
                "Tokens/parameter",
                f"[bold]{budget.tokens_per_parameter:.2f}[/]  "
                f"[dim](from-scratch training usually wants around 20; this model "
                f"size would want {fmt_count(budget.compute_optimal_tokens)} tokens)[/]",
            ),
        ],
    )
    print_bullets(
        "What to expect",
        budget.warnings(),
        empty="[green]Nothing to flag.[/] The data and the model size are in proportion.",
    )


def _print_result(result: TrainResult, trainer: Trainer) -> None:
    rule("Result")
    rows = [
        ("Steps", fmt_int(result.steps_completed)),
        ("Final train loss", f"[bold]{result.final_train_loss:.4f}[/]"),
    ]
    if result.best_val_loss is not None:
        rows.append(
            (
                "Best val loss",
                f"[bold]{result.best_val_loss:.4f}[/] at step "
                f"{fmt_int(result.best_val_step or 0)}  "
                f"[dim](perplexity {format_perplexity(result.best_val_loss)})[/]",
            )
        )
        if result.final_val_loss is not None and result.best_val_step != result.steps_completed:
            rows.append(
                (
                    "Final val loss",
                    f"{result.final_val_loss:.4f}  [yellow](worse than step "
                    f"{fmt_int(result.best_val_step or 0)}: the model started "
                    "memorising, so use the best checkpoint, not the last)[/]",
                )
            )
    else:
        # The reason comes from the trainer, which knows it. This row used to read
        # "no validation split, or --eval-every 0" whatever the cause, and on the
        # common case -- a validation split too small for one window at this
        # --seq-len -- both named causes were false and the real one went unsaid.
        skipped = result.validation_skipped
        detail = skipped.reason if skipped else "no validation split, or --eval-every 0"
        rows.append(("Validation", f"[yellow]not measured[/]  [dim]({detail})[/]"))
        if skipped is not None and skipped.hint:
            rows.append(("", f"[dim]{skipped.hint}[/]"))
    rows.extend(
        [
            (
                "Throughput",
                f"{fmt_count(result.tokens_per_second)} tokens/s  "
                f"[dim]({fmt_int(result.tokens_seen)} tokens in "
                f"{fmt_duration(result.elapsed_seconds)})[/]",
            ),
            ("Peak VRAM", trainer.memory_note()),
            # `as_posix()` rather than `str()`: every other path this project prints
            # goes through it, and mixing the two put `runs\a\b.pt` one row above
            # `runs/a/b.jsonl` in the same table on Windows.
            (
                "Checkpoint",
                result.checkpoint_path.as_posix() if result.checkpoint_path else "none",
            ),
            ("Metrics", (result.run_dir / "metrics.jsonl").as_posix()),
        ]
    )
    print_kv("Measured", rows)
    console.print(
        f"[dim]The loss curve is in {(result.run_dir / 'metrics.jsonl').as_posix()}, "
        "one JSON object per line.[/]"
    )
    console.print()
    # The quickstart names eval and chat as the next two steps, so the command that
    # finishes says what they are. Every other command that leaves something on disk
    # ends with the command that reads it.
    console.print("[dim]Next: score it on held-out text, or try it out:[/]")
    print_command(f"trainai eval {result.run_dir.as_posix()}", style="dim bold")
    print_command(f"trainai chat {result.run_dir.as_posix()}", style="dim bold")


def describe_presets() -> str:
    """One line per preset, for ``--help``."""
    return "  ".join(f"{p.name} (L{p.n_layer} d{p.d_model})" for p in PRESETS)
