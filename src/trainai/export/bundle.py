"""Writing a finished run out as a standalone model directory.

Three properties shape this module, in order of how much they cost:

**An export is either complete or absent.** Everything is written into a temporary
directory beside the destination, verified there, and moved into place only once the
checks pass. A crash, a full disk, or a failed verification leaves the destination
untouched -- rather than a directory that contains a ``config.json`` and half a
tensor file, which loads far enough to produce garbage.

**Every export is verified against the source.** Not "the code that writes it has
tests", though it does: this re-reads the file that was just written and compares it
to the model in memory, tensor by tensor. Writing weights is exactly the operation
where a silent mistake is unrecoverable, because the output is plausible either way.
When ``transformers`` is installed the ``hf`` format additionally gets loaded back
and its logits compared against ours, which is the only check that can catch a wrong
config field or a rotary convention mismatch.

**What was not checked is reported as not checked.** ``transformers`` is not a
dependency of this project, so on most machines the logits comparison cannot run.
The result says so, by name, instead of leaving the impression that a full parity
check happened.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from trainai.data.binarize import TOKENIZER_NAME
from trainai.data.tokenizer import ByteLevelBPE
from trainai.errors import ExportError, UsageError
from trainai.export import hf as hf_layout
from trainai.infer import InferenceSession
from trainai.model.config import ModelConfig
from trainai.model.gpt import GPT

__all__ = [
    "DEFAULT_DTYPE",
    "DEFAULT_FORMAT",
    "DTYPE_CHOICES",
    "FORMAT_CHOICES",
    "ExportResult",
    "ExportedFile",
    "VerifyCheck",
    "export_run",
]

FORMAT_CHOICES = ("hf", "safetensors")
DEFAULT_FORMAT = "hf"

#: Name -> torch dtype. Names match the rest of the CLI (``--precision``).
DTYPE_CHOICES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
DEFAULT_DTYPE = "fp32"

WEIGHTS_NAME = "model.safetensors"
TRAINAI_CONFIG_NAME = "trainai-model.json"

#: How close the exported model's logits must be to ours, in fp32, for the parity
#: check to pass. The two run the same arithmetic in a different order (our attention
#: against HuggingFace's), so bit-equality is not available; 2e-4 is far below the
#: gap between adjacent logits in any trained model and far above reduction-order
#: noise. The measured value is always reported, not just the verdict.
LOGITS_TOLERANCE = 2e-4

#: Tokens used for the parity check. Enough positions to exercise rotary embeddings
#: and attention across the sequence, few enough to be instant on a CPU.
_PARITY_TOKENS = 16


@dataclass(frozen=True)
class ExportedFile:
    """One file in the export, and its size."""

    name: str
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "bytes": self.bytes}


@dataclass(frozen=True)
class VerifyCheck:
    """One verification, including the ones that did not run.

    ``ran=False`` is a first-class outcome. A check that was skipped because
    ``transformers`` is absent is not a pass, and reporting it as one would be the
    exact kind of overstatement this project exists to avoid.
    """

    name: str
    ran: bool
    passed: bool
    detail: str

    @property
    def ok(self) -> bool:
        """True unless this check ran and failed."""
        return self.passed or not self.ran

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ran": self.ran, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class ExportResult:
    """What was written, from what, and what was verified about it."""

    format: str
    out_dir: Path
    dtype: str
    files: list[ExportedFile]
    checks: list[VerifyCheck]
    parameters: int = 0
    source: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return sum(item.bytes for item in self.files)

    @property
    def verified(self) -> list[VerifyCheck]:
        return [check for check in self.checks if check.ran]

    @property
    def skipped(self) -> list[VerifyCheck]:
        return [check for check in self.checks if not check.ran]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "out_dir": str(self.out_dir),
            "dtype": self.dtype,
            "files": [item.to_dict() for item in self.files],
            "total_bytes": self.total_bytes,
            "parameters": self.parameters,
            "checks": [check.to_dict() for check in self.checks],
            "source": self.source,
            "model": self.model,
        }


def export_run(
    target: str | Path,
    out_dir: str | Path,
    *,
    export_format: str = DEFAULT_FORMAT,
    dtype: str = DEFAULT_DTYPE,
    which: str = "best",
    tokenizer: str | Path | None = None,
    verify: bool = True,
    force: bool = False,
) -> ExportResult:
    """Write the model from ``target`` into ``out_dir`` and verify it.

    Args:
        target: A run directory, a checkpoints directory, or a checkpoint file.
        out_dir: Destination directory. Created; must not already hold something
            other than a previous export unless ``force``.
        export_format: ``hf`` for a directory ``transformers`` can load, or
            ``safetensors`` for this project's own names plus its config.
        dtype: ``fp32``, ``fp16`` or ``bf16``. fp32 is the default because it is
            what the checkpoint holds, so it is the only lossless choice.
        which: ``best`` or ``latest``, when ``target`` is a directory.
        tokenizer: Override the tokenizer path, for a run whose dataset moved.
        verify: Run the checks. Off is available for a very large model where the
            re-read is slow; the result then says the checks were skipped.
        force: Replace an existing export at ``out_dir``.

    Raises:
        UsageError: An unknown format or dtype, or nothing loadable at ``target``.
        ExportError: The destination is occupied by something that is not an
            export, a weight has no mapping, or a verification failed.
    """
    resolved_format = _check_choice(export_format, FORMAT_CHOICES, "--format")
    resolved_dtype_name = _check_choice(dtype, tuple(DTYPE_CHOICES), "--dtype")
    torch_dtype = DTYPE_CHOICES[resolved_dtype_name]

    destination = Path(out_dir).expanduser()
    _check_destination(destination, force=force)

    session = InferenceSession.open(target, which=which, device="cpu", tokenizer=tokenizer)

    staging = destination.with_name(f"{destination.name}.partial-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        tensors = _write_everything(session, staging, resolved_format, torch_dtype)
        checks = (
            _verify(session, staging, resolved_format, tensors, torch_dtype)
            if verify
            else [
                VerifyCheck(
                    name=name,
                    ran=False,
                    passed=False,
                    detail="skipped: --no-verify",
                )
                for name in ("weights round-trip", "tokenizer round-trip", "logits parity")
            ]
        )
        failed = [check for check in checks if not check.ok]
        if failed:
            raise ExportError(
                f"The export failed verification: {failed[0].detail}",
                hint=(
                    "Nothing was written to the destination. This is a bug in TrainAI's "
                    "export, not in your run -- please report it with this message."
                ),
                details={"checks": [check.to_dict() for check in checks]},
            )
        files = _measure(staging)
        _move_into_place(staging, destination, force=force)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    return ExportResult(
        format=resolved_format,
        out_dir=destination,
        dtype=resolved_dtype_name,
        files=files,
        checks=checks,
        parameters=session.model.parameter_count(),
        source=_provenance(session),
        model=session.model_config.to_dict(),
    )


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def _write_everything(
    session: InferenceSession,
    staging: Path,
    export_format: str,
    torch_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Write every file of the export into ``staging``; return the tensors written."""
    from safetensors.torch import save_file

    config = session.model_config
    if export_format == "hf":
        tensors = hf_layout.hf_state_dict(session.model, dtype=torch_dtype)
    else:
        tensors = _own_state_dict(session.model, dtype=torch_dtype)

    save_file(tensors, str(staging / WEIGHTS_NAME), metadata={"format": "pt"})
    session.tokenizer.save(staging / "tokenizer.json")

    if export_format == "hf":
        _write_json(
            staging / "config.json",
            hf_layout.hf_config(config, tokenizer=session.tokenizer, dtype=torch_dtype),
        )
        _write_json(
            staging / "generation_config.json",
            hf_layout.hf_generation_config(config, tokenizer=session.tokenizer),
        )
        _write_json(staging / "tokenizer_config.json", hf_layout.hf_tokenizer_config(config))
        _write_json(staging / "special_tokens_map.json", hf_layout.hf_special_tokens_map())
    else:
        _write_json(
            staging / TRAINAI_CONFIG_NAME,
            {
                "format": "trainai-safetensors-1",
                "model": config.to_dict(),
                "dtype": str(torch_dtype).removeprefix("torch."),
                "tokenizer": TOKENIZER_NAME,
                "source": _provenance(session),
            },
        )

    (staging / "README.md").write_text(
        _readme(session, export_format, torch_dtype), encoding="utf-8", newline="\n"
    )
    return tensors


def _own_state_dict(model: GPT, *, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """The model's tensors under this project's own names, cast to ``dtype``.

    Same de-duplication as the HF path: ``safetensors`` will not write two names
    that share one storage, and the config records ``tie_embeddings`` so a loader
    re-creates the alias.
    """
    tied = set(model.alias_state_dict_keys)
    return {
        name: tensor.detach().to(dtype=dtype, device="cpu").contiguous().clone()
        for name, tensor in model.state_dict().items()
        if name not in tied
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    # ``newline="\n"`` is not decoration. Without it Python translates every ``\n``
    # to ``os.linesep``, so the same export writes different bytes on Windows --
    # measured, 144 against 133 for one small config -- and ``_measure`` records
    # those byte counts in the bundle manifest.
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8", newline="\n"
    )


def _measure(staging: Path) -> list[ExportedFile]:
    return sorted(
        (
            ExportedFile(name=item.relative_to(staging).as_posix(), bytes=item.stat().st_size)
            for item in staging.rglob("*")
            if item.is_file()
        ),
        key=lambda item: item.name,
    )


# --------------------------------------------------------------------------- #
# Verifying
# --------------------------------------------------------------------------- #
def _verify(
    session: InferenceSession,
    staging: Path,
    export_format: str,
    tensors: dict[str, torch.Tensor],
    torch_dtype: torch.dtype,
) -> list[VerifyCheck]:
    return [
        _verify_weights(staging, tensors),
        _verify_tokenizer(session, staging),
        _verify_logits(session, staging, export_format, torch_dtype),
    ]


def _verify_weights(staging: Path, tensors: dict[str, torch.Tensor]) -> VerifyCheck:
    """Re-read the written file and compare it to what was handed to the writer.

    Bit-exact, not approximate: this compares a file against the tensors that were
    passed to ``save_file`` in the same process, so any difference at all is a
    serialisation fault rather than rounding.
    """
    from safetensors.torch import load_file

    reloaded = load_file(str(staging / WEIGHTS_NAME))
    if set(reloaded) != set(tensors):
        missing = sorted(set(tensors) - set(reloaded))
        extra = sorted(set(reloaded) - set(tensors))
        return VerifyCheck(
            name="weights round-trip",
            ran=True,
            passed=False,
            detail=f"key mismatch after re-reading: missing {missing}, unexpected {extra}",
        )
    for name, expected in tensors.items():
        found = reloaded[name]
        if found.dtype != expected.dtype or found.shape != expected.shape:
            return VerifyCheck(
                name="weights round-trip",
                ran=True,
                passed=False,
                detail=(
                    f"{name} came back as {found.dtype} {tuple(found.shape)}, "
                    f"expected {expected.dtype} {tuple(expected.shape)}"
                ),
            )
        if not torch.equal(found, expected):
            return VerifyCheck(
                name="weights round-trip",
                ran=True,
                passed=False,
                detail=f"{name} does not match the model it was written from",
            )
    return VerifyCheck(
        name="weights round-trip",
        ran=True,
        passed=True,
        detail=f"{len(tensors)} tensors re-read from disk and bit-identical to the model",
    )


def _verify_tokenizer(session: InferenceSession, staging: Path) -> VerifyCheck:
    """The written tokenizer must be the one the model was trained with.

    A tokenizer that does not match the weights produces fluent text that has
    nothing to do with what the model computed, and nothing downstream can detect
    it -- the same failure ``InferenceSession`` refuses to open a run for.
    """
    try:
        written = ByteLevelBPE.load(staging / "tokenizer.json")
    except Exception as error:  # pragma: no cover - a corrupt copy is not reachable here
        return VerifyCheck(
            name="tokenizer round-trip",
            ran=True,
            passed=False,
            detail=f"the written tokenizer.json cannot be loaded: {error}",
        )
    if written.fingerprint() != session.tokenizer.fingerprint():
        return VerifyCheck(
            name="tokenizer round-trip",
            ran=True,
            passed=False,
            detail="the written tokenizer is not the one the model was trained with",
        )
    return VerifyCheck(
        name="tokenizer round-trip",
        ran=True,
        passed=True,
        detail=f"fingerprint {written.fingerprint()[:12]} matches the run",
    )


def _verify_logits(
    session: InferenceSession,
    staging: Path,
    export_format: str,
    torch_dtype: torch.dtype,
) -> VerifyCheck:
    """Load the export with ``transformers`` and compare its logits against ours.

    This is the only check that tests the *meaning* of the export rather than its
    bytes. A swapped rotary convention, a wrong ``num_key_value_heads``, or a
    gate/up transposition all produce a file that loads cleanly and generates
    nonsense; all three move the logits.

    Skipped, and reported as skipped, when ``transformers`` is not installed --
    it is deliberately not a dependency of this project.
    """
    if export_format != "hf":
        return VerifyCheck(
            name="logits parity",
            ran=False,
            passed=False,
            detail=f"not applicable to the {export_format} format",
        )
    try:
        from transformers import LlamaForCausalLM
    except ImportError:
        return VerifyCheck(
            name="logits parity",
            ran=False,
            passed=False,
            detail=(
                "skipped: transformers is not installed, so the export could not be "
                "loaded back. Install it and re-run with --verify to check parity."
            ),
        )

    config = session.model_config
    ids = torch.tensor(
        [
            [
                (index * 7 + 1) % config.vocab_size
                for index in range(min(_PARITY_TOKENS, config.seq_len))
            ]
        ],
        dtype=torch.long,
    )
    reference = _reference_model(session.model, config, torch_dtype)
    _quieten_transformers()

    try:
        loaded = LlamaForCausalLM.from_pretrained(str(staging), dtype=torch.float32)
    except TypeError:  # transformers 4.x spells it torch_dtype
        loaded = LlamaForCausalLM.from_pretrained(str(staging), torch_dtype=torch.float32)
    loaded.eval()

    with torch.no_grad():
        ours, _loss, _caches = reference(ids)
        theirs = loaded(ids).logits.float()

    if ours.shape != theirs.shape:
        return VerifyCheck(
            name="logits parity",
            ran=True,
            passed=False,
            detail=f"shape mismatch: ours {tuple(ours.shape)}, exported {tuple(theirs.shape)}",
        )
    difference = float((ours - theirs).abs().max())
    passed = difference <= LOGITS_TOLERANCE
    verdict = "within" if passed else "above"
    return VerifyCheck(
        name="logits parity",
        ran=True,
        passed=passed,
        detail=(
            f"transformers loaded the export; max logit difference {difference:.2e} "
            f"over {ids.numel()} positions, {verdict} the {LOGITS_TOLERANCE:.0e} tolerance"
        ),
    )


def _quieten_transformers() -> None:
    """Stop ``transformers`` printing over our own output during verification.

    Its weight-loading bar and startup notices go to stderr, so ``--json`` on stdout
    is already safe -- but a progress bar for a check the user did not ask about
    reads as an error, and the export prints its own summary. Best-effort: the
    logging API has moved between versions and a missing helper is not worth
    failing an export over.
    """
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
    except Exception:  # pragma: no cover - depends on the installed transformers
        pass


def _reference_model(model: GPT, config: ModelConfig, torch_dtype: torch.dtype) -> GPT:
    """A copy of ``model`` holding exactly the values the export holds.

    Casting to the export dtype and back to fp32 makes the comparison about the
    *mapping* rather than about precision: at ``--dtype fp16`` the exported file
    genuinely holds different numbers than the checkpoint, and comparing against
    the un-rounded model would report that rounding as a parity failure.
    """
    reference = GPT(config)
    reference.load_state_dict(
        {
            name: tensor.detach().to(torch_dtype).to(torch.float32)
            for name, tensor in model.state_dict().items()
        }
    )
    reference.eval()
    return reference


# --------------------------------------------------------------------------- #
# The destination
# --------------------------------------------------------------------------- #
def _check_destination(destination: Path, *, force: bool) -> None:
    """Refuse to write over anything that is not an empty dir or an old export.

    ``--force`` deletes a directory, so it must not be able to delete a directory
    the user cares about. An export is recognised by the files this module writes;
    anything else has to be removed by hand.
    """
    if destination.exists() and not destination.is_dir():
        raise ExportError(
            f"{destination} exists and is not a directory.",
            hint="Pass --out with a directory path; the export writes several files.",
            details={"path": str(destination)},
        )
    if not destination.is_dir():
        return
    contents = list(destination.iterdir())
    if not contents:
        return
    if not _looks_like_an_export(destination):
        raise ExportError(
            f"{destination} already exists and is not a TrainAI export.",
            hint=(
                "Refusing to delete it. Pass --out somewhere else, or remove the "
                "directory yourself if you meant to replace it."
            ),
            details={
                "path": str(destination),
                "entries": sorted(item.name for item in contents)[:10],
            },
        )
    if not force:
        raise ExportError(
            f"{destination} already holds an export.",
            hint="Pass --force to replace it, or --out to write somewhere else.",
            details={"path": str(destination)},
        )


def _looks_like_an_export(destination: Path) -> bool:
    """Whether ``destination`` holds an export this module wrote.

    Deliberately strict: the weight file *and* one of the two config files, so a
    directory that merely happens to contain a ``model.safetensors`` from another
    tool is not eligible for ``--force``.
    """
    if not (destination / WEIGHTS_NAME).is_file():
        return False
    return (destination / "config.json").is_file() or (destination / TRAINAI_CONFIG_NAME).is_file()


def _move_into_place(staging: Path, destination: Path, *, force: bool) -> None:
    """Swap the verified staging directory in as ``destination``.

    ``os.replace`` will not replace a non-empty directory on Windows, and will not
    replace one at all on some filesystems, so the destination is cleared first:

    * an empty directory (the common case -- people create it before running) is
      simply removed;
    * a previous export is removed only with ``force``, which
      :func:`_check_destination` has already established this is.

    Clearing leaves a moment where neither directory is in place. The staging copy
    still exists on disk throughout, so nothing is lost, but the destination path is
    briefly absent.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_dir():
        if not any(destination.iterdir()):
            destination.rmdir()
        elif force:
            shutil.rmtree(destination)
        else:  # pragma: no cover - _check_destination already refused
            raise ExportError(
                f"{destination} filled up while the export was being written.",
                hint="Re-run with --force, or --out somewhere else.",
                details={"path": str(destination)},
            )
    os.replace(staging, destination)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def _provenance(session: InferenceSession) -> dict[str, Any]:
    """Where the weights came from, in enough detail to find the run again."""
    return {
        "run_dir": str(session.layout.run_dir),
        "checkpoint": str(session.layout.checkpoint_path),
        "which": session.layout.which,
        "step": session.step,
        "best_val_loss": session.best_val_loss,
        "best_val_step": session.best_val_step,
        "tokenizer_fingerprint": session.tokenizer.fingerprint(),
        "vocab_size": session.tokenizer.vocab_size,
    }


def _readme(session: InferenceSession, export_format: str, torch_dtype: torch.dtype) -> str:
    """A README for the export directory.

    An exported model that arrives with no note about what it is gets used as if it
    were an instruction-following assistant. Saying "base model, trained on <this>,
    completes text" in the directory itself is the only place that message survives
    being copied around.
    """
    config = session.model_config
    loss = session.best_val_loss
    loss_line = (
        f"- Best validation loss: {loss:.4f} (at step {session.best_val_step})"
        if loss is not None
        # No cause named: the checkpoint records the number's absence, not the reason for
        # it. This used to assert "the run had no validation split", which was wrong
        # whenever the split existed but was too small for one window at the run's
        # --seq-len -- the commonest way a run ends up without the number.
        else "- Best validation loss: not measured during training"
    )
    parameters = f"{session.model.parameter_count() / 1e6:.1f}M"
    dtype_name = str(torch_dtype).removeprefix("torch.")

    if export_format == "hf":
        usage = (
            "```python\n"
            "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
            "\n"
            'tokenizer = AutoTokenizer.from_pretrained("./")\n'
            'model = AutoModelForCausalLM.from_pretrained("./")\n'
            "\n"
            'ids = tokenizer("Once upon a time", return_tensors="pt")\n'
            "print(tokenizer.decode(model.generate(**ids, max_new_tokens=50)[0]))\n"
            "```\n"
        )
        layout_note = (
            f"This is a `{hf_layout.HF_ARCHITECTURE}` -- the architecture TrainAI trains "
            "is structurally a Llama (RMSNorm, rotary embeddings, SwiGLU, grouped-query "
            "attention, no biases), so the weights are the same tensors under "
            "HuggingFace's names.\n"
        )
    else:
        usage = (
            "```python\n"
            "import json\n"
            "from safetensors.torch import load_file\n"
            "\n"
            f'config = json.load(open("{TRAINAI_CONFIG_NAME}"))["model"]\n'
            f'weights = load_file("{WEIGHTS_NAME}")\n'
            "```\n"
        )
        layout_note = (
            "The tensor names are TrainAI's own, matching `trainai.model.gpt.GPT`. "
            f"`{TRAINAI_CONFIG_NAME}` holds the `ModelConfig` that built them.\n"
        )

    return f"""# Exported language model

A small base language model trained from scratch with TrainAI. **It completes text;
it is not an instruction-following assistant.** Ask it a question and it will
continue the question rather than answer it.

## What it is

- Parameters: {parameters}
- Layers: {config.n_layer}, heads: {config.n_head} (key/value heads: {config.kv_heads})
- Model dimension: {config.d_model}
- Context length: {config.seq_len} tokens
- Vocabulary: {config.vocab_size} tokens, byte-level BPE trained on the same corpus
- Weights: {dtype_name}
- Trained for {session.step} steps
{loss_line}

{layout_note}
## Using it

{usage}
## Provenance

Exported from `{session.layout.checkpoint_path.name}` (`{session.layout.which}`) of the
run at `{session.layout.run_dir.as_posix()}`. Tokenizer fingerprint
`{session.tokenizer.fingerprint()}` -- the tokenizer in this directory is the one the
model was trained with, and no other tokenizer will produce sensible output from
these weights.
"""


def _check_choice(value: str, choices: tuple[str, ...], flag: str) -> str:
    if value not in choices:
        raise UsageError(
            f"Unknown {flag} {value!r}.",
            hint=f"Choose one of: {', '.join(choices)}.",
            details={"given": value, "choices": list(choices)},
        )
    return value
