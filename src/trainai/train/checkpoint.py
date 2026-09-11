"""Checkpoints: written atomically, verified on load, exact on resume.

Three properties, each of which took a specific decision:

**Atomic.** A checkpoint is written to a temporary file and renamed into place.
``os.replace`` is atomic on both POSIX and Windows, so a checkpoint either exists
complete or does not exist. Writing in place would mean that a crash during the
save -- or a full disk, which is how it usually happens -- destroys the last good
checkpoint as well as the new one.

**Verified.** A checkpoint records the tokenizer fingerprint and the dataset
content hash. Resuming against a different dataset is refused by name rather than
producing a model that trains to a plausible loss and emits nonsense. That failure
is expensive to diagnose after the fact and free to prevent here.

**Exact.** Resume restores enough state that the continued run is bitwise identical
to an uninterrupted one. That means the optimizer's moments, the gradient scaler,
and every RNG stream -- but *not* a data-loader position or a scheduler counter,
because neither exists: batch order is a pure function of ``(seed, step)`` and the
learning rate is a pure function of ``step``. There is nothing to get out of sync.

The tied output projection is stored once. ``state_dict()`` names it twice
(``lm_head.weight`` and ``embed_tokens.weight`` share one storage), and
``load_state_dict`` applies both in order with the last one winning -- so a state
dict whose two copies disagree silently overwrites the embedding and raises
nothing. Measured: a zeroed ``lm_head.weight`` zeroes the embedding of the loaded
model. Storing one copy makes that impossible.
"""

from __future__ import annotations

import json
import os
import pickletools
import random
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch

from trainai import __version__
from trainai.errors import (
    CheckpointCorruptError,
    CheckpointError,
    CheckpointIncompatibleError,
    CheckpointNotFoundError,
    ConfigError,
)
from trainai.model.config import ModelConfig
from trainai.model.gpt import GPT
from trainai.train.config import TrainConfig

__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "Checkpoint",
    "RngState",
    "capture_rng",
    "find_checkpoint",
    "list_checkpoints",
    "pointer_fallback_reason",
    "restore_rng",
    "save_checkpoint",
]

CHECKPOINT_FORMAT = "trainai-checkpoint"

#: Bumped to 2 when the random state stopped being stored as a NumPy array.
#:
#: ``torch.load(weights_only=True)`` will not build a NumPy array, and that single
#: refusal is the whole reason every checkpoint used to be loaded with the unpickler
#: that runs whatever the file says. Measured on a real checkpoint rather than
#: reasoned about: allowlisting exactly four globals --
#: ``numpy._core.multiarray._reconstruct``, ``numpy.ndarray``, ``numpy.dtype`` and
#: ``numpy.dtypes.UInt32DType`` -- made the entire file load, so the optimizer
#: moments, the config dicts, the metrics and Python's own RNG tuple never needed it.
#: Version 2 writes the same 624 words as a tensor and keeps everything else, so the
#: safe unpickler is enough.
CHECKPOINT_VERSION = 2

#: Name of the JSON file that points at the latest and best checkpoints. A pointer
#: file rather than a symlink: creating symlinks on Windows needs either developer
#: mode or elevation, and a training run must not require either.
POINTER_NAME = "checkpoints.json"

#: The keys of the pointer file whose value is a checkpoint reference rather than a
#: plain fact. These are the ones :func:`_read_pointer` checks the shape of; anything
#: else in the file is passed through, because the file is rewritten in place.
_POINTER_ENTRIES = ("latest", "best")

_STEP_PREFIX = "step-"
_STEP_DIGITS = 7

_T = TypeVar("_T")


@dataclass
class RngState:
    """Every random stream the training loop touches.

    ``python`` and ``numpy`` are here even though the loop does not currently draw
    from them, because a future addition that does -- a data augmentation, a
    sampled evaluation prompt -- would otherwise break exact resume silently, and
    the cost of carrying them is a few hundred bytes.

    Both are held as plain data rather than as the objects the standard library hands
    back. ``np.random.get_state(legacy=True)`` returns a tuple whose second element is
    a ``uint32`` array, and an array inside a pickle is what forced every checkpoint
    load to trust the file -- see :data:`CHECKPOINT_VERSION`. The array's 624 words go
    in as a tensor, which is what the rest of the file is made of anyway.
    """

    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor] = field(default_factory=list)
    numpy: dict[str, Any] | None = None
    python: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "torch_cpu": self.torch_cpu,
            "torch_cuda": self.torch_cuda,
            "numpy": self.numpy,
            "python": self.python,
        }

    @classmethod
    def from_payload(cls, raw: dict[str, Any]) -> RngState:
        return cls(
            torch_cpu=raw["torch_cpu"],
            torch_cuda=list(raw.get("torch_cuda") or []),
            numpy=_rng_as_plain(raw.get("numpy"), _numpy_rng_to_plain),
            python=_rng_as_plain(raw.get("python"), _python_rng_to_plain),
        )


def _numpy_rng_to_plain(state: tuple[Any, ...]) -> dict[str, Any]:
    """NumPy's legacy MT19937 state with no NumPy object left in it.

    The key array is 624 ``uint32`` words, so a signed 64-bit tensor holds every one
    exactly and there is no overflow to reason about. The round trip is measured rather
    than assumed -- see the test that draws, restores and draws again.
    """
    name, keys, position, has_gauss, cached_gaussian = state
    return {
        "bit_generator": str(name),
        "keys": torch.as_tensor(np.asarray(keys, dtype=np.int64)),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _numpy_rng_from_plain(plain: dict[str, Any]) -> tuple[Any, ...]:
    """Rebuild what ``np.random.set_state`` expects: the tuple, with a ``uint32`` array."""
    keys = plain["keys"]
    words = keys.cpu().numpy() if torch.is_tensor(keys) else np.asarray(keys)
    return (
        str(plain["bit_generator"]),
        words.astype(np.uint32),
        int(plain["position"]),
        int(plain["has_gauss"]),
        float(plain["cached_gaussian"]),
    )


def _python_rng_to_plain(state: tuple[Any, ...]) -> dict[str, Any]:
    """``random.getstate()`` is already plain data; this only names its three parts.

    Kept symmetrical with the NumPy pair deliberately. The bare tuple loads under
    ``weights_only=True`` as it stands -- measured -- so converting it buys nothing on
    its own; what it buys is a reader not having to work out why one of the two streams
    is stored differently from the other.
    """
    version, internal, gauss_next = state
    return {
        "version": int(version),
        "state": [int(word) for word in internal],
        "gauss_next": None if gauss_next is None else float(gauss_next),
    }


def _python_rng_from_plain(plain: dict[str, Any]) -> tuple[Any, ...]:
    """Rebuild what ``random.setstate`` expects, which insists on a tuple."""
    return (
        int(plain["version"]),
        tuple(int(word) for word in plain["state"]),
        plain["gauss_next"],
    )


def _rng_as_plain(
    value: Any, to_plain: Callable[[tuple[Any, ...]], dict[str, Any]]
) -> dict[str, Any] | None:
    """Accept either shape a stored random state can arrive in, and keep only one.

    Version 2 writes the dict; version 1 wrote the tuple the standard library hands
    back. A version-1 NumPy tuple does not survive ``weights_only=True`` and is refused
    long before it reaches here -- but *whether* it survives is a property of torch's
    allowlist, which is not TrainAI's to promise: ``add_safe_globals`` is public API and
    global to the process, so another library in the same program can turn that refusal
    into a load. Normalising both shapes at the one place a file becomes a
    :class:`RngState` means the outcome is a resumed run rather than a ``TypeError``
    raised from a converter handed the wrong kind of object, and it keeps the field
    annotations honest about what the rest of the module will see.
    """
    if value is None or isinstance(value, dict):
        return value
    return to_plain(tuple(value))


def capture_rng() -> RngState:
    """Snapshot the random state of every stream training uses."""
    cuda_states: list[torch.Tensor] = []
    if torch.cuda.is_available():
        cuda_states = list(torch.cuda.get_rng_state_all())
    return RngState(
        torch_cpu=torch.get_rng_state(),
        torch_cuda=cuda_states,
        numpy=_numpy_rng_to_plain(np.random.get_state(legacy=True)),
        python=_python_rng_to_plain(random.getstate()),
    )


def restore_rng(state: RngState) -> None:
    """Put every random stream back where it was.

    CUDA states are only restored when the device count matches. Moving a
    checkpoint between machines is normal and useful; refusing to resume because
    the new box has a different number of GPUs would not be. The dropout stream
    then differs, so resume is no longer bitwise exact -- which is why the mismatch
    is reported rather than passed over.
    """
    torch.set_rng_state(state.torch_cpu.to(torch.uint8).cpu())
    if state.numpy is not None:
        np.random.set_state(_numpy_rng_from_plain(state.numpy))
    if state.python is not None:
        random.setstate(_python_rng_from_plain(state.python))
    if (
        state.torch_cuda
        and torch.cuda.is_available()
        and len(state.torch_cuda) == torch.cuda.device_count()
    ):
        torch.cuda.set_rng_state_all([s.to(torch.uint8).cpu() for s in state.torch_cuda])


@dataclass
class Checkpoint:
    """A loaded checkpoint. Everything needed to continue, and to prove it matches.

    Attributes:
        step: Optimizer steps completed. Training resumes *at* this step.
        model_config: The shape of the model the weights belong to.
        train_config: The run's settings, so the schedule cannot drift.
        model_state: Weights, with the tied alias removed.
        optimizer_state: AdamW moments.
        scaler_state: fp16 gradient-scaler state, or ``None``.
        rng: Random streams.
        metrics: Loss values and counters at the time of saving.
        dataset: Identity of the data it was trained on.
    """

    step: int
    model_config: ModelConfig
    train_config: TrainConfig
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    scaler_state: dict[str, Any] | None
    rng: RngState | None
    metrics: dict[str, Any]
    dataset: dict[str, Any]
    created_with: str = f"trainai {__version__}"
    path: Path | None = None

    def apply_to(self, model: GPT) -> None:
        """Load the weights into ``model``, re-tying the output projection.

        Raises:
            CheckpointIncompatibleError: If the model's shape differs from the one
                the checkpoint was written for.
            CheckpointCorruptError: If the weights disagree with the model that the
                checkpoint's own ``model`` section describes.
        """
        if model.config.to_dict() != self.model_config.to_dict():
            raise CheckpointIncompatibleError(
                "The checkpoint was written for a different model shape.",
                hint=(
                    "Resume with the same model configuration, or start a new run. "
                    "The checkpoint's shape is in its `model` section; run "
                    "`trainai train --resume <path>` without model flags to reuse it."
                ),
                details={
                    "checkpoint": self.model_config.describe(),
                    "model": model.config.describe(),
                },
            )
        state = dict(self.model_state)
        for alias in model.alias_state_dict_keys:
            source = "embed_tokens.weight"
            if alias not in state and source in state:
                state[alias] = state[source]

        # Every way the file's tensors can disagree with a model whose config already
        # matches. `load_state_dict` reports missing and unexpected keys as return values
        # under `strict=False`, but a tensor at the wrong shape -- or an entry that is not
        # a tensor at all -- it *raises*, and a raise escaping this method arrived as a
        # Python traceback and exit 1. Measured on a real 600-step checkpoint with one
        # weight replaced: `RuntimeError: Error(s) in loading state_dict for GPT: size
        # mismatch for blocks.0.attn.q_proj.weight`, with the internals of
        # `torch.nn.Module` in the frame list. Only reachable by editing the file, since
        # the config gate above has already run -- but a checkpoint is a file users copy
        # between machines, and a damaged file is a refusal, not a bug in TrainAI.
        #
        # A dtype that differs is deliberately not checked. Torch casts it, and a float64
        # copy of a real checkpoint loads and generates (measured), so refusing it would
        # turn a file that works into one that does not.
        expected = model.state_dict()
        wrong_type = sorted(
            key for key, value in state.items() if key in expected and not torch.is_tensor(value)
        )
        wrong_shape = sorted(
            [key, list(state[key].shape), list(expected[key].shape)]
            for key in state
            if key in expected
            and torch.is_tensor(state[key])
            and state[key].shape != expected[key].shape
        )

        missing: list[str] = []
        unexpected: list[str] = []
        torch_error: str | None = None
        if not (wrong_type or wrong_shape):
            try:
                loaded = model.load_state_dict(state, strict=False)
            except RuntimeError as exc:
                # The backstop, not a second copy of the checks above: those name what
                # they found, and this turns anything torch objects to that they did not
                # anticipate into the same refusal and the same exit code, rather than a
                # traceback.
                torch_error = str(exc)
            else:
                # A tied alias is expected to be absent from the file; anything else is
                # not.
                unexpected = list(loaded.unexpected_keys)
                missing = [
                    key for key in loaded.missing_keys if key not in model.alias_state_dict_keys
                ]

        if missing or unexpected or wrong_type or wrong_shape or torch_error:
            raise CheckpointCorruptError(
                f"{self.path or 'The checkpoint'} holds weights that do not match the "
                "model its own `model` section describes.",
                hint=(
                    "The file is damaged, was edited, or was written by an incompatible "
                    "version. Resume from an earlier checkpoint, or start again."
                ),
                details={
                    "path": str(self.path) if self.path else None,
                    "missing": missing,
                    "unexpected": unexpected,
                    "wrong_type": wrong_type,
                    "wrong_shape": wrong_shape,
                    "torch_error": torch_error,
                },
            )


def save_checkpoint(
    directory: str | Path,
    *,
    step: int,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_config: TrainConfig,
    scaler: Any | None = None,
    metrics: dict[str, Any] | None = None,
    dataset: dict[str, Any] | None = None,
    rng: RngState | None = None,
    is_best: bool = False,
    keep: int = 3,
) -> Path:
    """Write a checkpoint for ``step`` and return its path.

    Args:
        directory: Where checkpoints live. Created if absent.
        step: Optimizer steps completed.
        model: The model. Its config is recorded, and the tied alias dropped.
        optimizer: Its state dict is recorded in full.
        train_config: Recorded so a resume uses the same schedule.
        scaler: fp16 gradient scaler, if one is in use.
        metrics: Whatever the loop wants to remember -- losses, throughput.
        dataset: Identity of the training data. Checked on resume.
        rng: Random state. Omitting it makes resume approximate rather than exact.
        is_best: Also record this step as the best so far in the pointer file.
        keep: How many step checkpoints to retain. The best and the newest are
            always kept regardless.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    state = {
        key: value
        for key, value in model.state_dict().items()
        if key not in model.alias_state_dict_keys
    }
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_VERSION,
        "created_with": f"trainai {__version__}",
        "step": int(step),
        "model": model.config.to_dict(),
        "train": train_config.to_dict(),
        "model_state": state,
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "rng": rng.to_payload() if rng is not None else None,
        "metrics": metrics or {},
        "dataset": dataset or {},
    }

    path = directory / f"{_STEP_PREFIX}{step:0{_STEP_DIGITS}d}.pt"
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        # Atomic on POSIX and on Windows: the checkpoint either exists complete or
        # not at all, so a crash mid-save cannot take the previous one with it.
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise CheckpointError(
            f"Could not write the checkpoint to {path}.",
            hint=(
                "Check free disk space and that the run directory is writable. "
                "The previous checkpoint is untouched."
            ),
            details={"path": str(path), "reason": str(exc)},
        ) from exc

    _update_pointer(directory, step=step, path=path, is_best=is_best, metrics=metrics or {})
    _prune(directory, keep=keep)
    return path


def _imports_wanted(path: Path) -> list[str]:
    """Every global a torch file's pickle asks to import, importing none of them.

    ``torch.save`` writes a zip archive whose ``data.pkl`` member is the pickle, and
    :func:`pickletools.genops` walks that pickle's opcodes without executing any of
    them. So this can say *why* a refused file was refused -- a saved ``nn.Module``, a
    version-1 random state, something else entirely -- without performing the very load
    the refusal exists to prevent.

    Returns an empty list when there is nothing to report: a file that is not a zip
    (truncated, or torch's pre-1.6 format), one with no ``data.pkl``, or a pickle that
    imports nothing. All three are indistinguishable from damage here, and are reported
    as damage by the caller.

    ``STACK_GLOBAL`` is handled as well as ``GLOBAL`` because the opcode a pickle uses
    depends on its protocol, not on its contents: torch writes protocol 2 today, and a
    file from another tool written at protocol 4 or later names its imports the other
    way.
    """
    try:
        if not zipfile.is_zipfile(path):
            return []
        with zipfile.ZipFile(path) as archive:
            member = next((n for n in archive.namelist() if n.endswith("data.pkl")), None)
            if member is None:
                return []
            data = archive.read(member)
    except (OSError, zipfile.BadZipFile):
        return []

    wanted: list[str] = []
    strings: list[str] = []
    try:
        for opcode, argument, _position in pickletools.genops(data):
            if opcode.name == "GLOBAL":
                wanted.append(str(argument).replace(" ", "."))
            elif opcode.name == "STACK_GLOBAL":
                if len(strings) >= 2:
                    wanted.append(f"{strings[-2]}.{strings[-1]}")
            elif isinstance(argument, str):
                strings.append(argument)
    except Exception:
        # A pickle this cannot even parse tells us nothing, and this function's whole
        # job is to produce a better message than the caller already has -- so it
        # reports what it managed to read and lets the caller fall back to "damaged".
        return list(dict.fromkeys(wanted))
    return list(dict.fromkeys(wanted))


#: Advice for a file that is not a checkpoint, naming no flag on purpose. This function
#: is reached from ``train --resume``, ``finetune --from``, ``chat``, ``eval`` and
#: ``export``, and ``load_checkpoint`` is not told which -- so the hint said
#: "Point --resume at ..." to four callers who had not typed ``--resume``. Naming the
#: three shapes accepted instead is advice every caller can act on, and it matches what
#: ``--resume``'s own help text says.
_WRONG_FILE_HINT = (
    "Point it at something `trainai train` wrote: a run directory, its checkpoints "
    "directory, or a step-*.pt file inside one."
)

#: What a version-1 checkpoint's random state asks for, and nothing else. Measured by
#: allowlisting the refusals of a real checkpoint one at a time until it loaded;
#: ``_codecs.encode`` is how the bit-generator's name is pickled. A refused file whose
#: imports are a subset of these is the old format rather than a hostile one, and the
#: difference is worth a different message.
_VERSION_1_RNG_IMPORTS = frozenset(
    {
        "numpy._core.multiarray._reconstruct",
        "numpy.core.multiarray._reconstruct",
        "numpy.ndarray",
        "numpy.dtype",
        "numpy.dtypes.UInt32DType",
        "_codecs.encode",
        "collections.OrderedDict",
        "torch._utils._rebuild_tensor_v2",
        "torch.ByteStorage",
        "torch.FloatStorage",
    }
)


def _refusal_for(path: Path, reason: str) -> CheckpointError:
    """Name what the file is, when the safe unpickler would not build it.

    Three shapes are worth telling apart, and all three come from reading the pickle
    rather than running it. A saved ``nn.Module`` is the most likely mistake anybody
    makes -- ``torch.save(model, path)`` is how most PyTorch code saves a model -- and
    it earned a message of its own long before this check existed. A version-1
    checkpoint is TrainAI's own and is not damaged. Everything else is reported as
    damage with the imports named, because a file asking for something neither of those
    is either broken or not to be run, and naming the import is what lets someone tell
    which.
    """
    wanted = _imports_wanted(path)
    module = next((name for name in wanted if name.startswith("torch.nn.")), None)
    if module is not None:
        held = module.rsplit(".", 1)[-1]
        return CheckpointIncompatibleError(
            f"{path} is not a TrainAI checkpoint (the file holds a {held}).",
            hint=_WRONG_FILE_HINT,
            details={"path": str(path), "format": None, "holds": held, "imports": wanted},
        )
    if wanted and set(wanted) <= _VERSION_1_RNG_IMPORTS:
        return CheckpointIncompatibleError(
            f"{path} was written before TrainAI stopped trusting checkpoint files "
            "(checkpoint version 1).",
            hint=(
                "Version 1 stored its random state as a NumPy array, which cannot be "
                "read without running code from the file. Resume from a checkpoint this "
                "version wrote, or start the run again -- exact resume is the only thing "
                "the old random state was for."
            ),
            details={"path": str(path), "imports": wanted, "reason": reason},
        )
    return CheckpointCorruptError(
        f"{path} could not be read as a checkpoint.",
        hint=(
            "The file is truncated or damaged, most likely from an interrupted "
            "copy. Resume from an earlier checkpoint in the same directory."
        ),
        details={"path": str(path), "reason": reason, "imports": wanted},
    )


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    expect_dataset: dict[str, Any] | None = None,
) -> Checkpoint:
    """Read a checkpoint, refusing anything it cannot honestly continue.

    Args:
        path: A checkpoint file, or a directory to take the latest from.
        map_location: Where to put the tensors. ``cpu`` is right for loading before
            the model is moved to its device.
        expect_dataset: When given, the checkpoint's dataset identity must match.

    Raises:
        CheckpointNotFoundError: No checkpoint at that path.
        CheckpointCorruptError: Unreadable, or missing required fields.
        CheckpointIncompatibleError: Written by a newer format, or trained on
            different data.
    """
    path = Path(path)
    if path.is_dir():
        found = find_checkpoint(path)
        if found is None:
            raise CheckpointNotFoundError(
                f"No checkpoint found in {path}.",
                hint=(
                    "Point --resume at a run's checkpoints directory that contains "
                    f"{_STEP_PREFIX}*.pt files, or start a new run."
                ),
                details={"path": str(path)},
            )
        path = found
    if not path.exists():
        raise CheckpointNotFoundError(
            f"{path} does not exist.",
            hint="Check the path. `trainai train --resume <run>/checkpoints` uses the latest.",
            details={"path": str(path)},
        )

    try:
        # weights_only=True, which is the point of format version 2. A checkpoint is a
        # file people copy between machines, and `trainai finetune --from` takes its path
        # straight from the command line, so a load that runs whatever the file says is a
        # real exposure rather than a theoretical one. Nothing in a version-2 payload
        # needs the unsafe unpickler -- measured, see CHECKPOINT_VERSION.
        raw = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        # Which of the three refusals this is comes from reading the file's pickle
        # without executing it, so the message can still say "the file holds a Linear".
        raise _refusal_for(path, str(exc)) from exc

    if not isinstance(raw, dict) or raw.get("format") != CHECKPOINT_FORMAT:
        # ``raw`` is deliberately not touched as a mapping here: this branch is
        # reached *because* it may not be one. ``(raw or {}).get("format")`` --
        # what this used to be -- crashed on the likeliest way to arrive here.
        # `torch.save(model, path)` is how most people save a PyTorch model, and
        # pointing --resume at one of those raised `AttributeError: 'Linear' object
        # has no attribute 'get'` from inside this very handler; a saved tensor
        # raised `RuntimeError: Boolean value of Tensor ... is ambiguous` from the
        # `or`. Both replaced the intended message with a traceback and exit 1.
        # Only a dict answered correctly, which is why no test noticed.
        held = type(raw).__name__
        declared = raw.get("format") if isinstance(raw, dict) else None
        raise CheckpointIncompatibleError(
            f"{path} is not a TrainAI checkpoint (the file holds a {held}).",
            hint=_WRONG_FILE_HINT,
            details={"path": str(path), "format": declared, "holds": held},
        )
    version = int(raw.get("format_version", 0))
    if version > CHECKPOINT_VERSION:
        raise CheckpointIncompatibleError(
            f"{path} was written by a newer TrainAI (checkpoint version {version}, "
            f"this build understands {CHECKPOINT_VERSION}).",
            hint=(
                "Use the TrainAI that wrote it. There is no converter, and unlike a "
                "dataset a checkpoint cannot be re-created -- it is the output of the "
                "training run that produced it."
            ),
            details={"found": version, "supported": CHECKPOINT_VERSION},
        )

    for required in ("step", "model", "train", "model_state", "optimizer_state"):
        if required not in raw:
            raise CheckpointCorruptError(
                f"{path} is missing its {required!r} section.",
                hint="The file is incomplete. Resume from an earlier checkpoint.",
                details={"path": str(path), "missing": required},
            )

    dataset = dict(raw.get("dataset") or {})
    if expect_dataset:
        _check_dataset(path, dataset, expect_dataset)

    return Checkpoint(
        step=int(raw["step"]),
        model_config=_rebuilt(path, "model", ModelConfig.from_dict, raw["model"]),
        train_config=_rebuilt(path, "train", TrainConfig.from_dict, raw["train"]),
        model_state=raw["model_state"],
        optimizer_state=raw["optimizer_state"],
        scaler_state=raw.get("scaler_state"),
        rng=RngState.from_payload(raw["rng"]) if raw.get("rng") else None,
        metrics=dict(raw.get("metrics") or {}),
        dataset=dataset,
        created_with=str(raw.get("created_with", "unknown")),
        path=path,
    )


def _rebuilt(path: Path, section: str, builder: Callable[[Any], _T], block: Any) -> _T:
    """Rebuild one configuration section, keeping the failure a *checkpoint* failure.

    ``ModelConfig.from_dict`` raises :class:`ConfigError`, whose exit code is ``USAGE``:
    correct where it is reached through a hand-written plan file, wrong here, where the
    file is one TrainAI produced and the user's command was fine. Scripts branch on the
    exit code, so an incomplete checkpoint has to keep reporting ``CHECKPOINT``.

    The reason is carried through verbatim rather than replaced. "The model
    configuration is missing n_layer" is the whole value of the refusal -- a message
    saying only that the checkpoint is unusable sends the reader back to guessing.
    """
    try:
        return builder(block)
    except ConfigError as exc:
        raise CheckpointCorruptError(
            f"{path} has a {section!r} section that cannot be loaded. {exc.message}",
            hint=exc.hint,
            details={"path": str(path), "section": section, **exc.details},
        ) from exc


def _check_dataset(path: Path, found: dict[str, Any], expected: dict[str, Any]) -> None:
    """Refuse to resume against data the checkpoint was not trained on.

    The tokenizer fingerprint is the one that matters most: the same shard bytes
    read through a different tokenizer are a different corpus, and the loss curve
    would look entirely normal while the model learned nothing usable.
    """
    for key, label in (
        ("tokenizer_fingerprint", "tokenizer"),
        ("content_hash", "dataset"),
    ):
        want, have = expected.get(key), found.get(key)
        if want and have and want != have:
            raise CheckpointIncompatibleError(
                f"This checkpoint was trained on a different {label}.",
                hint=(
                    "Resume against the dataset the run started on, or start a new "
                    "run against this one. Continuing across a change of "
                    f"{label} would produce a model whose loss looks reasonable and "
                    "whose output is not."
                ),
                details={
                    "path": str(path),
                    "field": key,
                    "checkpoint": have[:16],
                    "dataset": want[:16],
                },
            )


# --------------------------------------------------------------------------- #
# Finding checkpoints
# --------------------------------------------------------------------------- #
def list_checkpoints(directory: str | Path) -> list[Path]:
    """Every step checkpoint in ``directory``, oldest first."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(directory.glob(f"{_STEP_PREFIX}*.pt"), key=_step_of)


def find_checkpoint(directory: str | Path, *, which: str = "latest") -> Path | None:
    """The latest or best checkpoint in ``directory``, or ``None`` if there is none.

    Reads the pointer file when it is available and falls back to the filenames
    when it is not, because the pointer is a convenience and the files are the
    truth. A pointer naming a file that no longer exists is ignored rather than
    raised on -- deleting a checkpoint by hand is a reasonable thing to do. See
    :func:`_read_pointer` for what "available" means and why nothing here refuses.
    """
    directory = Path(directory)
    entry = _read_pointer(directory).get(which)
    if isinstance(entry, dict):
        name = entry["file"]  # _read_pointer dropped the entry unless this is a name
        if (directory / name).exists():
            return directory / name
    available = list_checkpoints(directory)
    return available[-1] if available else None


def _read_pointer(directory: Path) -> dict[str, Any]:
    """The pointer file, with any entry that is not usable dropped. Never raises.

    Degrading rather than refusing is the opposite of the choice
    :meth:`DatasetManifest.load` makes for ``manifest.json`` one file over, and the
    difference is that this file is *derived*: ``latest`` is the highest-numbered
    ``step-*.pt`` on disk, which is why the fallback below the callers exists at all.
    Refusing would turn a damaged convenience file into a run that cannot resume and a
    model that cannot be loaded, when the checkpoints themselves are intact -- and
    :func:`_update_pointer` is about to overwrite the file regardless.

    The one thing here that is *not* derivable is ``best``: the filenames record the
    order checkpoints were written, not their validation loss, so falling back answers
    "the best checkpoint" with the last one. That answer is a different checkpoint from
    the one asked for, so it is reported rather than substituted in silence -- by
    :func:`pointer_fallback_reason`, which the callers that honour ``--which best`` use.

    Measured on a run directory with three checkpoints, one edit per case, against the
    reader before this existed: of eighteen ways the file can be damaged, ``_prune``
    raised on ten and :func:`_update_pointer` on five, both as a bare ``TypeError`` or
    ``AttributeError`` -- and ``_update_pointer`` runs *after* the checkpoint is safely
    on disk, so a hand-edited pointer ended a training run with a traceback over a file
    the next line was going to replace. ``UnicodeDecodeError`` is a ``ValueError``, so
    the ``except OSError`` on all three never saw a pointer an editor had re-saved as
    Latin-1, and that one raised in all three.
    """
    path = directory / POINTER_NAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    # Unknown keys pass through untouched: this rewrites the file, and dropping a key
    # a later version added is how a downgrade quietly loses information.
    return {
        key: value
        for key, value in raw.items()
        if key not in _POINTER_ENTRIES or _is_pointer_entry(value)
    }


def _is_pointer_entry(value: Any) -> bool:
    """Whether ``value`` is an entry a reader can use -- an object naming a file."""
    return isinstance(value, dict) and isinstance(value.get("file"), str) and bool(value["file"])


def pointer_fallback_reason(directory: str | Path, *, which: str) -> str | None:
    """Why ``which`` could not be answered from the pointer file, or ``None``.

    Only ever a reason to *report*, never to refuse: :func:`find_checkpoint` has already
    answered from the filenames by the time anyone asks. Whether it is worth reporting is
    the caller's call and depends on ``which`` -- for ``latest`` the fallback names the
    same file the pointer would have, and for ``best`` it names a different checkpoint --
    so that judgement is left where the answer is shown rather than made here.
    """
    directory = Path(directory)
    path = directory / POINTER_NAME
    if not path.exists():
        return f"{path} does not exist"
    entry = _read_pointer(directory).get(which)
    if not isinstance(entry, dict):
        return f"{path} does not record the {which} checkpoint in a form it can be read from"
    if not (directory / entry["file"]).exists():
        return f"{path} names {entry['file']} as {which}, and that file is gone"
    return None


def _step_of(path: Path) -> int:
    try:
        return int(path.stem[len(_STEP_PREFIX) :])
    except ValueError:  # pragma: no cover - glob only matches the prefix
        return -1


def _update_pointer(
    directory: Path,
    *,
    step: int,
    path: Path,
    is_best: bool,
    metrics: dict[str, Any],
) -> None:
    """Record the latest and best checkpoints, atomically."""
    pointer = directory / POINTER_NAME
    current: dict[str, Any] = _read_pointer(directory)

    entry = {"file": path.name, "step": step}
    if "val_loss" in metrics:
        entry["val_loss"] = metrics["val_loss"]
    current["latest"] = entry
    if is_best or "best" not in current:
        current["best"] = entry

    temporary = pointer.with_name(pointer.name + ".tmp")
    temporary.write_text(
        json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    os.replace(temporary, pointer)


def _prune(directory: Path, *, keep: int) -> None:
    """Delete old checkpoints, never the newest and never the best.

    ``keep`` of 0 means keep everything rather than nothing. Deleting every
    checkpoint would leave a run with no way to resume, which is not a plausible
    thing for anyone to have asked for.

    The newest is protected by name, not by trusting the pointer file to name it. The
    two agree on every run this writes -- :func:`_update_pointer` sets ``latest`` to the
    file just saved, one line above the call here -- but "never the newest" is the
    promise :func:`save_checkpoint` documents, and deriving it from a file a user can
    edit means a damaged pointer deletes the checkpoint that was just written.
    """
    if keep <= 0:
        return
    existing = list_checkpoints(directory)
    if not existing:
        return
    pointer = _read_pointer(directory)
    protected = {existing[-1].name} | {
        pointer[which]["file"] for which in _POINTER_ENTRIES if which in pointer
    }

    removable = [p for p in existing if p.name not in protected]
    surplus = len(existing) - keep
    for path in removable[: max(0, surplus)]:
        path.unlink(missing_ok=True)
