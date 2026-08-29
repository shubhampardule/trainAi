"""Loading a finished run and generating text from it.

Training and inference need different things from the same directory, and this module
is where that difference is handled once instead of in each of ``eval``, ``chat`` and
``export``.

**A run must be usable after the dataset is deleted.** The checkpoint records the
tokenizer's *fingerprint*, which is enough to detect a mismatch and not enough to turn
ids back into text. So the trainer copies ``tokenizer.json`` into the run directory,
and this module prefers that copy -- falling back to the dataset the checkpoint names,
and saying plainly what to pass when neither is there. A directory of numbers with no
way to read them is the failure this arrangement exists to prevent.

**A tokenizer that does not match is refused, not warned about.** Token id 412 means
one string under the tokenizer the model was trained with and a different one under
any other, so decoding with the wrong one produces confident nonsense rather than an
error. The fingerprint is checked before anything is generated.

**Precision is resolved and reported, as everywhere else.** Inference under bf16 does
not produce the same logits as fp32, so which one ran is part of the answer rather
than an implementation detail. ``--precision fp32`` is available for when a
reproducible sample matters more than speed.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from trainai.data.binarize import TOKENIZER_NAME
from trainai.data.tokenizer import ByteLevelBPE
from trainai.errors import UsageError
from trainai.model.config import ModelConfig
from trainai.model.gpt import GPT
from trainai.train.checkpoint import (
    Checkpoint,
    find_checkpoint,
    list_checkpoints,
    load_checkpoint,
    pointer_fallback_reason,
)
from trainai.train.config import TrainConfig
from trainai.train.loop import resolve_device, resolve_precision

#: Sampling defaults. Temperature 0.8 with top-p 0.95 is the usual "readable but not
#: repetitive" setting; a small model at temperature 1.0 wanders badly, and at 0.0 it
#: falls into loops within a sentence or two.
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K: int | None = None
DEFAULT_REPETITION_PENALTY = 1.1
DEFAULT_MAX_NEW_TOKENS = 200

#: Which checkpoint to load when the caller does not say. ``best`` rather than
#: ``latest``, because on a small corpus the last checkpoint is routinely worse than
#: the best one -- that is the overfitting the data budget warns about, and defaulting
#: to ``latest`` would silently hand the user the memorised model.
DEFAULT_WHICH = "best"

WHICH_CHOICES = ("best", "latest")


@dataclass(frozen=True)
class StreamPiece:
    """One step of a generation: the new text, and the tokens produced so far.

    ``text`` is empty when that token completed no character. Byte-level BPE can put a
    multi-byte character across two tokens, and the text is held back until it is
    whole -- but the token was still produced, and anything measuring speed has to
    count it. That is the whole reason this type exists instead of a bare ``str``:
    counting yielded strings undercounts tokens on any non-ASCII output, which makes a
    reported tokens/s lower than the machine's real rate.

    ``tokens`` is cumulative, not a delta, so a caller that misses a piece (an
    interrupted stream) still ends up with the right total.
    """

    text: str
    tokens: int


@dataclass(frozen=True)
class RunLayout:
    """Where a run's parts were found, and how.

    Kept as a value rather than left implicit, so a report can say *which* checkpoint
    and *which* tokenizer were used. "the best checkpoint" is not a location.
    """

    checkpoint_path: Path
    tokenizer_path: Path
    run_dir: Path | None
    which: str
    tokenizer_source: str
    checkpoints_available: int = 0
    note: str | None = None
    """Why ``which`` is not what the run recorded, when that happened.

    Only ever set for ``best``. The pointer file is what records which checkpoint had
    the lowest validation loss, and the filenames it falls back to record the order
    they were written -- so a damaged or absent pointer answers ``--which best`` with
    the *last* checkpoint, which is a different one. Carried here so the report can say
    so, because silently handing back another checkpoint is how someone concludes their
    best model is worse than it is.
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "tokenizer_path": str(self.tokenizer_path),
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "which": self.which,
            "tokenizer_source": self.tokenizer_source,
            "checkpoints_available": self.checkpoints_available,
            "note": self.note,
        }


#: What a decoder emits for a byte sequence that is not a complete character. A
#: byte-level BPE token can be half of one, so this shows up mid-stream on any
#: non-ASCII text and means "wait for the next token", not "the model produced junk".
REPLACEMENT_CHAR = "\ufffd"


def _checkpoint_dir_of(target: Path) -> Path | None:
    """The directory holding step files, given a run directory or that directory."""
    for candidate in (target / "checkpoints", target):
        if list_checkpoints(candidate):
            return candidate
    return None


def _locate_checkpoint(target: Path, which: str) -> tuple[Path, Path | None, str, int, str | None]:
    """``(checkpoint_path, run_dir, resolved_which, checkpoints_available, note)``.

    Split out from :func:`locate_run` so that :meth:`InferenceSession.open` can load
    the checkpoint once and hand it to the tokenizer search, instead of the search
    loading its own copy to read one field out of it.
    """
    if target.is_file():
        run_dir = target.parent.parent if target.parent.name == "checkpoints" else target.parent
        return target, run_dir, "explicit", len(list_checkpoints(target.parent)), None

    checkpoint_dir = _checkpoint_dir_of(target)
    if checkpoint_dir is None:
        raise UsageError(
            f"No checkpoints under {target}.",
            hint=(
                "A trained run contains checkpoints/step-*.pt. If training was "
                "interrupted before the first checkpoint, there is nothing to load "
                "yet -- train for at least --checkpoint-every steps."
            ),
            details={"path": str(target), "looked_in": [str(target / "checkpoints")]},
        )
    found = find_checkpoint(checkpoint_dir, which=which)
    if found is None:  # pragma: no cover - _checkpoint_dir_of already proved non-empty
        raise UsageError(
            f"No {which} checkpoint under {target}.",
            hint="Try --which latest.",
            details={"path": str(target), "which": which},
        )
    # Asked after the answer, because the answer is the same either way: the fallback is
    # a good checkpoint, just not the one that was asked for. Only for `best` -- the
    # filenames record the order checkpoints were written, so for `latest` the fallback
    # names the same file the pointer would have, and there is nothing to say.
    reason = pointer_fallback_reason(checkpoint_dir, which=which) if which == "best" else None
    note = (
        f"{found.name} is the last checkpoint, not the one with the lowest validation "
        f"loss: {reason}"
        if reason
        else None
    )
    run_dir = checkpoint_dir.parent if checkpoint_dir.name == "checkpoints" else checkpoint_dir
    return found, run_dir, which, len(list_checkpoints(checkpoint_dir)), note


def _check_target(target: str | Path, which: str) -> Path:
    """Validate ``--which`` and that ``target`` exists, before any loading happens."""
    if which not in WHICH_CHOICES:
        raise UsageError(
            f"--which must be one of {', '.join(WHICH_CHOICES)}, not {which!r}.",
            hint="best is the checkpoint with the lowest validation loss; latest is the last one.",
            details={"which": which, "choices": list(WHICH_CHOICES)},
        )

    target = Path(target)
    if not target.exists():
        raise UsageError(
            f"Nothing at {target}.",
            hint=(
                "Pass a run directory, for example `trainai chat runs/mine`. "
                "`trainai train` prints the path it wrote to when it finishes."
            ),
            details={"path": str(target)},
        )
    return target


def locate_run(target: str | Path, *, which: str = DEFAULT_WHICH) -> RunLayout:
    """Resolve a user-supplied path into a checkpoint and a tokenizer.

    ``target`` may be a run directory (``runs/mine``), a checkpoints directory, or a
    single checkpoint file. All three are things a user will reasonably type, and
    guessing wrong about which one they meant is not an error worth making them fix.

    Raises:
        UsageError: Nothing loadable at that path, or no tokenizer for it.
    """
    path = _check_target(target, which)
    checkpoint_path, run_dir, resolved_which, available, note = _locate_checkpoint(path, which)
    tokenizer_path, source = _locate_tokenizer(run_dir, checkpoint_path, None)
    return RunLayout(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        run_dir=run_dir,
        which=resolved_which,
        tokenizer_source=source,
        checkpoints_available=available,
        note=note,
    )


def _locate_tokenizer(
    run_dir: Path | None, checkpoint_path: Path, checkpoint: Checkpoint | None
) -> tuple[Path, str]:
    """The tokenizer for this run: the run's own copy, or the dataset it names.

    ``checkpoint`` is the already-loaded checkpoint when the caller has one. Without
    it, and only when the run directory has no tokenizer of its own, this loads the
    checkpoint to read the dataset path it recorded -- the alternative is loading
    weights before discovering there is nothing to decode them with.
    """
    if run_dir is not None:
        local = run_dir / TOKENIZER_NAME
        if local.is_file():
            return local, "the run directory"

    dataset_root: str | None = None
    if checkpoint is None:
        try:
            checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        except Exception:
            checkpoint = None
    if checkpoint is not None:
        recorded = checkpoint.dataset.get("root")
        dataset_root = str(recorded) if recorded else None

    if dataset_root:
        candidate = Path(dataset_root) / TOKENIZER_NAME
        if candidate.is_file():
            return candidate, f"the dataset at {dataset_root}"

    searched = [str(run_dir / TOKENIZER_NAME)] if run_dir else []
    if dataset_root:
        searched.append(str(Path(dataset_root) / TOKENIZER_NAME))

    # Which of the four ways to get here this is, rather than one story that is wrong
    # for the other three. This used to read "Runs trained by a newer version keep
    # their own copy; this one does not, and the dataset it names is gone or moved" --
    # both halves false for a run trained a minute earlier by this build against a
    # dataset that was still sitting there: `trainai train` copies the dataset's
    # tokenizer in, so a run lacks one when the dataset had none beside it either,
    # which train reports at the time. Upgrading is never the remedy; the build
    # printing this is the build that copies.
    #
    # The recorded root is the path as it was given to `trainai train`, so it is
    # relative whenever that argument was. Reporting a relative path as "gone or
    # moved" would be the same mistake in a new costume: it is merely not resolvable
    # from *here*.
    if not dataset_root:
        reason, why = (
            "no_dataset_recorded",
            "This run records no dataset, so there is nothing to find one from.",
        )
    elif Path(dataset_root).is_dir():
        reason, why = (
            "dataset_has_no_tokenizer",
            f"The dataset it names, {dataset_root}, has no {TOKENIZER_NAME} either.",
        )
    elif not Path(dataset_root).is_absolute():
        reason, why = (
            "dataset_not_found_from_here",
            f"It names {dataset_root}, a relative path, which does not exist from "
            "here -- so try again from the directory you trained in.",
        )
    else:
        reason, why = (
            "dataset_missing",
            f"The dataset it names, {dataset_root}, is gone or moved.",
        )
    raise UsageError(
        f"No {TOKENIZER_NAME} for this run, so its output cannot be turned into text.",
        hint=(
            f"Pass --tokenizer path/to/{TOKENIZER_NAME}, from the dataset this run "
            f"was trained on. {why}"
        ),
        details={"checkpoint": str(checkpoint_path), "searched": searched, "reason": reason},
    )


@dataclass
class InferenceSession:
    """A loaded model with the tokenizer that matches it.

    Construct with :meth:`open`. Holds the model in eval mode on the resolved device;
    everything that generates text goes through here so that the tokenizer check, the
    device placement and the precision report happen exactly once.
    """

    model: GPT
    tokenizer: ByteLevelBPE
    checkpoint: Checkpoint
    layout: RunLayout
    device: torch.device
    dtype: torch.dtype
    precision_note: str

    @classmethod
    def open(
        cls,
        target: str | Path,
        *,
        which: str = DEFAULT_WHICH,
        device: str = "auto",
        precision: str = "auto",
        tokenizer: str | Path | None = None,
    ) -> InferenceSession:
        """Load a run for generation.

        Args:
            target: A run directory, a checkpoints directory, or a checkpoint file.
            which: ``best`` or ``latest``. Ignored when ``target`` is a file.
            device: ``auto``, or a torch device string. An explicit device is never
                silently downgraded.
            precision: ``auto`` follows the hardware; ``fp32`` for a reproducible
                sample.
            tokenizer: Override the tokenizer path. For a run whose dataset moved.

        Raises:
            UsageError: Nothing loadable, no tokenizer, or a tokenizer that does not
                match the one the model was trained with.
        """
        path = _check_target(target, which)
        checkpoint_path, run_dir, resolved_which, available, note = _locate_checkpoint(path, which)
        resolved_device = resolve_device(device)
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")

        if tokenizer is not None:
            tokenizer_path = Path(tokenizer)
            if not tokenizer_path.is_file():
                raise UsageError(
                    f"No tokenizer file at {tokenizer_path}.",
                    hint=f"Point --tokenizer at a {TOKENIZER_NAME} from a prepared dataset.",
                    details={"path": str(tokenizer_path)},
                )
            source = "--tokenizer"
        else:
            tokenizer_path, source = _locate_tokenizer(run_dir, checkpoint_path, checkpoint)

        layout = RunLayout(
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            run_dir=run_dir,
            which=resolved_which,
            tokenizer_source=source,
            checkpoints_available=available,
            note=note,
        )
        loaded = ByteLevelBPE.load(layout.tokenizer_path)
        _check_tokenizer_matches(loaded, checkpoint, layout)

        dtype, _needs_scaler, note = resolve_precision(precision, resolved_device)
        model = GPT(checkpoint.model_config)
        checkpoint.apply_to(model)
        model.to(resolved_device)
        model.eval()

        return cls(
            model=model,
            tokenizer=loaded,
            checkpoint=checkpoint,
            layout=layout,
            device=resolved_device,
            dtype=dtype,
            precision_note=note,
        )

    # -- what was loaded ---------------------------------------------------- #

    @property
    def model_config(self) -> ModelConfig:
        return self.checkpoint.model_config

    @property
    def train_config(self) -> TrainConfig:
        return self.checkpoint.train_config

    @property
    def step(self) -> int:
        return self.checkpoint.step

    @property
    def best_val_loss(self) -> float | None:
        """The lowest validation loss recorded up to this checkpoint, if any.

        This is what the trainer stores -- ``best_val_loss``, not "this checkpoint's
        loss". For the ``best`` checkpoint the two are the same number; for ``latest``
        they are not, and calling it ``val_loss`` would invite reading a `latest`
        checkpoint's report as a measurement of that checkpoint. ``None`` means the run
        had no validation split large enough to measure, which happens on small
        corpora and is recorded rather than filled in.
        """
        value = self.checkpoint.metrics.get("best_val_loss")
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def best_val_step(self) -> int | None:
        """The step that :attr:`best_val_loss` was measured at, if any."""
        value = self.checkpoint.metrics.get("best_val_step")
        return int(value) if isinstance(value, (int, float)) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "best_val_loss": self.best_val_loss,
            "best_val_step": self.best_val_step,
            "model": self.model_config.to_dict(),
            "layout": self.layout.to_dict(),
            "device": str(self.device),
            "precision": self.precision_note,
            "vocab_size": self.tokenizer.vocab_size,
            "context": self.model_config.seq_len,
            "parameters": self.model.parameter_count(),
        }

    # -- generating --------------------------------------------------------- #

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Token ids for ``prompt``, as ``(1, seq)`` on this session's device.

        An empty prompt becomes a single end-of-text token rather than an empty
        tensor: the model needs something to attend to, and end-of-text is what the
        training data used to mark a document boundary, so it is the id that means
        "start of something".
        """
        ids = self.tokenizer.encode(prompt)
        if not ids:
            ids = [self.tokenizer.eot_id]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def autocast(self) -> Any:
        """The autocast context this session's precision implies.

        Public because evaluation runs its own forward passes and must run them under
        the same precision the session reports -- a loss measured in fp32 while the
        report says bf16 is a wrong number with a correct-looking label.
        """
        if self.dtype == torch.float32 or self.device.type == "cpu":
            return torch.autocast(device_type=self.device.type, enabled=False)
        return torch.autocast(device_type=self.device.type, dtype=self.dtype)

    def _generator(self, seed: int | None) -> torch.Generator | None:
        if seed is None:
            return None
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        return generator

    def complete(
        self,
        prompt: str,
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int | None = DEFAULT_TOP_K,
        top_p: float | None = DEFAULT_TOP_P,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        seed: int | None = None,
        stop_at_eot: bool = True,
    ) -> str:
        """Generate a continuation and return **only the new text**.

        Not prompt-plus-continuation: the caller already has the prompt, and a
        function that hands it back is one that makes every caller slice it off.
        """
        return "".join(
            self.stream(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
                stop_at_eot=stop_at_eot,
            )
        )

    def stream(
        self,
        prompt: str,
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int | None = DEFAULT_TOP_K,
        top_p: float | None = DEFAULT_TOP_P,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        seed: int | None = None,
        stop_at_eot: bool = True,
    ) -> Iterator[str]:
        """Yield the continuation as text, in pieces, as the model produces it.

        Pieces are *string deltas*, not tokens, and never empty -- see
        :meth:`stream_pieces` for why those are not the same thing, and use that
        instead when the token count matters.
        """
        for piece in self.stream_pieces(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
            stop_at_eot=stop_at_eot,
        ):
            if piece.text:
                yield piece.text

    def stream_pieces(
        self,
        prompt: str,
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int | None = DEFAULT_TOP_K,
        top_p: float | None = DEFAULT_TOP_P,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        seed: int | None = None,
        stop_at_eot: bool = True,
    ) -> Iterator[StreamPiece]:
        """Yield one :class:`StreamPiece` per token produced.

        The text of a piece is a *string delta*, not a token. A byte-level BPE token
        can be half of a multi-byte character, so decoding each token on its own yields
        replacement characters mid-word on any non-ASCII text. Decoding the whole
        continuation each step and yielding what is new costs an extra decode per token
        -- microseconds against a forward pass -- and is correct for any input. A token
        that completes no character yields an empty delta and still counts.

        The end-of-text token is a boundary, not content: it stops the stream, is not
        decoded into the output, and is not counted.
        """
        ids = self.encode_prompt(prompt)
        eot = self.tokenizer.eot_id if stop_at_eot else None
        produced: list[int] = []
        emitted = 0

        with self.autocast():
            for token in self.model.generate_stream(
                ids,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eot_id=eot,
                generator=self._generator(seed),
            ):
                value = int(token[0, 0])
                if eot is not None and value == eot:
                    break
                produced.append(value)
                text = self.tokenizer.decode(produced)
                # A trailing incomplete character decodes to U+FFFD, which would be
                # emitted now and contradicted next step. Hold it back instead -- but
                # report the token, so a caller timing the stream counts it.
                if text.endswith(REPLACEMENT_CHAR) or len(text) <= emitted:
                    yield StreamPiece("", len(produced))
                    continue
                yield StreamPiece(text[emitted:], len(produced))
                emitted = len(text)

        # Whatever the hold-back above kept, if generation ended mid-character. No new
        # token, so the count is unchanged from the last piece.
        if produced:
            final = self.tokenizer.decode(produced)
            if len(final) > emitted:
                yield StreamPiece(final[emitted:], len(produced))


def _check_tokenizer_matches(
    tokenizer: ByteLevelBPE, checkpoint: Checkpoint, layout: RunLayout
) -> None:
    """Refuse a tokenizer the model was not trained with.

    Not a warning. Id 412 means one string under this tokenizer and another under
    that one, so a mismatch does not degrade the output -- it produces fluent text
    that has nothing to do with what the model computed, and nothing anywhere says so.
    """
    expected = checkpoint.dataset.get("tokenizer_fingerprint")
    actual = tokenizer.fingerprint()
    if expected and actual != expected:
        raise UsageError(
            "This tokenizer is not the one the model was trained with.",
            hint=(
                "Pass --tokenizer pointing at the tokenizer.json from the dataset this "
                "run used. Decoding with a different tokenizer produces text that looks "
                "fine and means nothing, so it is refused rather than warned about."
            ),
            details={
                "expected_fingerprint": expected,
                "found_fingerprint": actual,
                "tokenizer": str(layout.tokenizer_path),
                "source": layout.tokenizer_source,
            },
        )

    trained_vocab = checkpoint.model_config.vocab_size
    if tokenizer.vocab_size != trained_vocab:
        raise UsageError(
            f"The model has an output layer of {trained_vocab} tokens and this "
            f"tokenizer has {tokenizer.vocab_size}.",
            hint=(
                "These have to agree or the ids do not line up. Use the tokenizer from "
                "the dataset this run was trained on."
            ),
            details={
                "model_vocab_size": trained_vocab,
                "tokenizer_vocab_size": tokenizer.vocab_size,
                "tokenizer": str(layout.tokenizer_path),
            },
        )
