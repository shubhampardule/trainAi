"""Measuring a trained model on held-out text.

Perplexity is the one number people ask for, and it is easy to report in a way that
is not wrong but is not comparable to anything either. Three things this module is
careful about:

**It is a per-token number, so it depends on the tokenizer.** A model with a 4,096-token
vocabulary and one with 50,257 do not produce comparable perplexities on the same text,
because they are not predicting the same units -- a coarser vocabulary means fewer,
harder predictions. Every result here carries the vocabulary size it was measured with,
and the report says so out loud. Comparing across tokenizers requires bits per *byte*,
which this does not compute yet.

**The mean is weighted by tokens, not by batches.** The final batch of a split holds
fewer rows than the others, and averaging per-batch losses gives that short batch the
same weight as a full one. On a small validation split that is a visible error in the
third digit, which is exactly where people look when comparing two checkpoints.

**It says how much of the split it actually scored.** The split is walked in
non-overlapping windows, so the last partial window -- fewer than ``seq_len + 1``
tokens -- is not scored, and ``--max-batches`` can cut it shorter still. Reporting
"perplexity on the validation split" while having covered 60% of it is the kind of
claim that survives until someone reruns it with different settings.

**It says whether the held-out split is actually held out.** Re-preparing the same
corpus with a different ``--seed`` moves the train/val boundary while leaving the
tokenizer fingerprint identical, so the ``val`` split of the new dataset contains
documents the model was trained on. The perplexity comes out lower and looks like a
better model. Scoring a model on another corpus is legitimate, so this is reported
rather than refused -- but it is reported first, because it changes what every other
number means. And when the check cannot run, that is reported too: the comparison needs
a content hash on both sides, and "could not check" is a third answer rather than a
quiet pass.

**It scores what the run optimised.** A dataset prepared from a chat corpus carries a
loss mask, and a run that trained on it minimised cross-entropy over the assistant's
replies alone. Averaging over every token instead would report a different quantity
under the same name -- one that cannot be compared to the training curve or to another
checkpoint of the same run. So the mask is applied when the run's recorded training
config says it was, and the report says how many of the predicted positions that left.

Evaluation is also not free of the precision question: under bf16 the loss differs
from fp32 in the third decimal, so the resolved precision is part of the result rather
than a footnote.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from trainai.data.binarize import DatasetManifest, Split
from trainai.data.loader import open_split, windows_available
from trainai.errors import DatasetError, UsageError
from trainai.infer.session import InferenceSession

#: Whether the dataset being scored is the one the run trained on. Three answers, not
#: two: the check needs both sides to have recorded a content hash, and when one has not
#: the honest report is that it could not be made. Collapsing that into ``"match"`` is
#: what this used to do, and it printed a clean bill of health over a comparison nothing
#: had performed.
DatasetCheck = Literal["match", "mismatch", "unknown"]

#: Batches to score when the caller does not limit it: all of them. Evaluation is a
#: forward pass per batch with no backward, so a validation split costs a fraction of
#: one training step per batch; there is no reason to sample it by default.
DEFAULT_MAX_BATCHES: int | None = None

#: Rows per forward pass. Independent of the batch size the model was trained with --
#: no gradients are held, so this is bounded by activation memory alone. Small enough
#: to fit the 4 GiB card this was developed on at a 1,024-token context.
DEFAULT_BATCH_SIZE = 8


@dataclass(frozen=True)
class SplitResult:
    """What was measured on one split.

    ``loss`` is mean cross-entropy in nats per token; ``perplexity`` is its
    exponential and ``bits_per_token`` is it in base 2. All three are the same
    measurement -- they are all reported because different literature quotes
    different ones, and recomputing between them invites an off-by-a-log error.

    ``batch_size`` is the size actually used, which can be smaller than the one asked
    for: a split holding fewer windows than the batch size is scored in one short batch
    rather than refused. It is reported because a result should say how it was produced,
    not because it changes the number -- it does not.

    Coverage is counted in **windows**, not tokens, because tokens are two different
    units here. ``tokens_scored`` counts target positions, and a window of
    ``seq_len + 1`` tokens yields only ``seq_len`` of them -- the first token is
    context and is never predicted. Dividing scored positions by split tokens would
    therefore report under 100% on a split that was covered completely.

    ``loss_mask`` says whether only the positions a masked dataset marks as targets
    were scored, in which case ``tokens_scored`` is smaller than ``positions_seen``
    -- the target positions the scored windows contained. Both are reported because
    the two together are what make a masked perplexity interpretable: a number over
    41% of the positions is a different measurement from one over all of them, and
    nothing else in the result would say which it is.
    """

    split: str
    loss: float
    tokens_scored: int
    windows_scored: int
    windows_in_split: int
    tokens_in_split: int
    batches: int
    seq_len: int
    batch_size: int
    stopped_early: bool = False
    loss_mask: bool = False
    positions_seen: int = 0

    @property
    def perplexity(self) -> float:
        return math.exp(self.loss)

    @property
    def bits_per_token(self) -> float:
        return self.loss / math.log(2)

    @property
    def coverage(self) -> float:
        """Fraction of the split's windows that were scored, 0.0 to 1.0."""
        if self.windows_in_split <= 0:  # pragma: no cover - open_split refuses this
            return 0.0
        return min(1.0, self.windows_scored / self.windows_in_split)

    @property
    def tokens_outside_windows(self) -> int:
        """Tokens at the end of the split that no window reaches.

        The batcher reserves ``seq_len`` tokens so a shuffled epoch's alignment offset
        cannot run past the end, then divides what is left into whole windows. What
        remains is unreachable at this ``seq_len`` -- shortening the context reduces it.
        Reported because "perplexity on the validation split" should not quietly mean
        "on all but the last few hundred tokens of it".
        """
        return max(0, self.tokens_in_split - self.windows_in_split * (self.seq_len + 1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "loss": self.loss,
            "perplexity": self.perplexity,
            "bits_per_token": self.bits_per_token,
            "tokens_scored": self.tokens_scored,
            "windows_scored": self.windows_scored,
            "windows_in_split": self.windows_in_split,
            "tokens_in_split": self.tokens_in_split,
            "tokens_outside_windows": self.tokens_outside_windows,
            "coverage": self.coverage,
            "batches": self.batches,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "stopped_early": self.stopped_early,
            "loss_mask": self.loss_mask,
            "positions_seen": self.positions_seen,
        }


@dataclass(frozen=True)
class EvalReport:
    """Every split that was measured, plus what it was measured with.

    The context carried alongside the numbers is not decoration. A perplexity without
    the vocabulary size cannot be compared to another one, and a perplexity without the
    precision cannot be reproduced exactly.

    ``dataset_check`` is ``"match"``, ``"mismatch"`` or ``"unknown"``, and
    ``dataset_matches_run`` is true only for ``"match"`` -- a *verified* match, not the
    absence of evidence against one. ``"mismatch"`` is not an error: scoring a model on
    another corpus is a real thing to want, but it means the ``val`` split here is not
    necessarily held out from this model, so the number is "perplexity on this text"
    rather than a generalisation measurement. ``"unknown"`` means a content hash was
    missing on one side and the comparison could not be made; it is reported rather than
    rounded to either answer, because both "this is the dataset" and "this is not the
    dataset" would be claims nothing checked.

    ``trained_seq_len`` is the context the run trained at, which is not always the
    context the model was built for. It is carried separately from each result's
    ``seq_len`` so a reader can tell "scored at the trained context" from "scored
    past it" without having to open the checkpoint.
    """

    results: list[SplitResult]
    step: int
    vocab_size: int
    parameters: int
    device: str
    precision: str
    checkpoint: str
    which: str
    dataset: str | None = None
    dataset_matches_run: bool = True
    dataset_check: DatasetCheck = "match"
    trained_seq_len: int | None = None
    notes: list[str] = field(default_factory=list)

    def result_for(self, split: str) -> SplitResult | None:
        return next((r for r in self.results if r.split == split), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "step": self.step,
            "checkpoint": self.checkpoint,
            "which": self.which,
            "dataset": self.dataset,
            "dataset_matches_run": self.dataset_matches_run,
            "dataset_check": self.dataset_check,
            "splits": [result.to_dict() for result in self.results],
            "model": {
                "parameters": self.parameters,
                "vocab_size": self.vocab_size,
                "trained_seq_len": self.trained_seq_len,
            },
            "device": self.device,
            "precision": self.precision,
            "notes": list(self.notes),
        }


def evaluate_split(
    session: InferenceSession,
    dataset: DatasetManifest,
    split: Split,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_batches: int | None = DEFAULT_MAX_BATCHES,
    seq_len: int | None = None,
    on_batch: Callable[[int, int], None] | None = None,
) -> SplitResult:
    """Mean cross-entropy over one split, walked once in order.

    Args:
        session: The loaded model. Left in eval mode; no gradients are built.
        dataset: The prepared dataset to measure on.
        split: ``train`` or ``val``.
        batch_size: Rows per forward pass. Does not affect the result, and is reduced
            to the number of sequences the split holds when it holds fewer.
        max_batches: Stop after this many batches. ``None`` scores the whole split.
        seq_len: Context to score at. Defaults to the context the run actually trained
            at, which is the only setting whose number is comparable to the training
            curve.
        on_batch: Called with ``(batches_done, tokens_scored)`` after each batch, for
            a progress display.

    The dataset's loss mask is applied when the run's recorded ``TrainConfig.loss_mask``
    is not ``False``, which is the trainer's own rule -- so the number this returns is
    the same quantity the training curve reported. There is no flag to override it,
    deliberately: the two numbers being comparable is the reason to have this function.

    Raises:
        UsageError: The split does not exist, or is too small for one window.
    """
    # Two different numbers, and the difference is the whole point of this block. The
    # model is *built* for ``model_config.seq_len`` -- that is how much context it can
    # be handed without erroring. It was *trained* at ``train_config.seq_len``, which
    # is what `trainai train` fed it, and `--seq-len` there defaults to
    # ``min(context, 256)``. So every preset above tiny produces a run where the two
    # differ: `--preset large` builds for 1,024 and trains at 256.
    #
    # Defaulting to the built context, as this used to, scored three of the four
    # presets at up to 4x a context they had never seen, and then compared the result
    # to the training curve as though it were the same quantity. Measured on a run
    # built for 512 and trained at 128: the curve's best val loss was 4.8157, scoring
    # at 128 gives 4.8158, and scoring at 512 gives 4.8295 over 768 fewer predictions.
    # The report noticed the 0.0138 gap and blamed it on the trainer sampling only
    # --eval-batches of the split, which was not the reason.
    built = session.model_config.seq_len
    trained = min(session.train_config.seq_len, built)
    context = seq_len or trained
    if context > built:
        raise UsageError(
            f"Cannot evaluate at {context} tokens of context: this model was built for {built}.",
            hint=(
                "Rotary embeddings do extrapolate somewhat, but past its context the "
                f"model has no trained behaviour at all. Use --seq-len {built} or lower"
                + (
                    f", or --seq-len {trained} to stay comparable to the training curve."
                    if trained < built
                    else "."
                )
            ),
            details={
                "requested": context,
                "built_for": built,
                "trained_at": trained,
            },
        )

    # The batcher is sized for training, where a batch holding the same window twice
    # would make the gradient smaller than it looks, so it refuses a batch size larger
    # than the split. Scoring has no gradient and walks the split in order, so a short
    # final batch is just a short final batch -- and refusing here would mean
    # `trainai eval` failing with no flags at all on any small validation split. The
    # size used is reported on the result. Zero windows still goes to the batcher: its
    # message about the split being too small for one sequence is the right one.
    rows = max(1, min(batch_size, windows_available(dataset, split, seq_len=context)))

    # Whether to apply the dataset's loss mask is not this function's decision: it is
    # the run's, recorded in the checkpoint's training config. `trainai eval` exists to
    # produce a number comparable to the training curve, and a curve optimised over
    # assistant replies alone is not comparable to a perplexity over every token --
    # measured on data/chat-mask, the masked and unmasked losses of the same checkpoint
    # differ in the first decimal, not the third. So a run trained with --no-loss-mask
    # is scored on every token and a run trained with the mask is scored on the mask.
    # ``loss_mask is not False`` matches the trainer's own rule, and a checkpoint
    # written before masks existed resumes as None, which means "apply if present".
    use_mask = session.train_config.loss_mask is not False

    try:
        batcher, _stream = open_split(
            dataset,
            split,
            seq_len=context,
            batch_size=rows,
            seed=0,
            loss_mask=use_mask,
        )
    except DatasetError as error:
        # The hint is the loader's, not a restatement. This used to say the split
        # "needs at least {context + 1} tokens to form one window", about half the real
        # requirement -- a window is seq_len + 1 tokens and another seq_len is reserved
        # for the alignment offset -- so a user with a 107-token validation split was
        # told it needed 65, two lines under this same panel's message saying it needed
        # 129. The loader computes the largest context that does fit; repeating the
        # arithmetic here is what let the two drift.
        #
        # `DatasetError` and not `Exception`: everything `open_split` refuses raises
        # one, and relabelling anything else as a usage error would blame the split
        # size for a bug somewhere else.
        raise UsageError(
            f"Cannot evaluate the {split} split: {error}",
            hint=error.hint,
            details={"split": split, "seq_len": context, "dataset": str(dataset.root)},
        ) from error

    masked = batcher.has_loss_mask
    weighted = 0.0
    scored = 0
    positions = 0
    windows = 0
    batches = 0
    try:
        with torch.no_grad():
            for batch in batcher.sequential_batches(max_batches=max_batches):
                inputs = torch.from_numpy(batch.inputs).to(session.device).long()
                targets = torch.from_numpy(batch.targets).to(session.device).long()
                mask = (
                    None
                    if batch.loss_mask is None
                    else torch.from_numpy(batch.loss_mask).to(session.device)
                )
                # Counted before the skip below, so the denominator of the scored share
                # covers every window walked rather than only the ones that scored.
                positions += int(targets.numel())
                windows += int(targets.shape[0])
                batches += 1
                # The weight is the number of positions the loss averaged over, which
                # differs on the last batch -- and, under a mask, on every batch. A
                # batch holding no target at all is skipped rather than added with
                # weight 0: the model returns 0.0 for it by a clamped denominator, and
                # multiplying that by 0 is right but relies on the clamp to not be a NaN.
                count = batch.trained_tokens
                if count:
                    with session.autocast():
                        _, loss, _ = session.model(inputs, targets, loss_mask=mask)
                    assert loss is not None
                    weighted += float(loss) * count
                    scored += count
                if on_batch is not None:
                    on_batch(batches, scored)
    finally:
        batcher.close()

    if scored == 0:
        # Reachable two ways. Without a mask it is not: `open_split` refuses a split
        # with no windows and `sequential_batches` yields at least one batch when there
        # is one. With a mask it is real -- a --max-batches cut short over a stretch of
        # prompts can walk windows that hold no assistant token -- so the message says
        # which case the user is in rather than reporting the small-split one for both.
        if masked:
            raise UsageError(
                f"None of the {windows:,} sequences scored in the {split} split contains "
                "a token the loss mask marks as a target.",
                hint=(
                    "Raise --max-batches to reach more of the split, or pass a run "
                    "trained with --no-loss-mask to score every token."
                ),
                details={
                    "split": split,
                    "seq_len": context,
                    "windows_scored": windows,
                    "positions_seen": positions,
                    "loss_mask": True,
                },
            )
        raise UsageError(
            f"The {split} split produced no batches to score.",
            hint=f"At a context of {context} it needs at least {2 * context + 1} tokens.",
            details={"split": split, "seq_len": context},
        )

    return SplitResult(
        split=split,
        loss=weighted / scored,
        tokens_scored=scored,
        windows_scored=windows,
        windows_in_split=batcher.windows,
        tokens_in_split=batcher.total_tokens,
        batches=batches,
        seq_len=context,
        batch_size=rows,
        stopped_early=windows < batcher.windows,
        loss_mask=masked,
        positions_seen=positions,
    )


def evaluate(
    session: InferenceSession,
    dataset: DatasetManifest,
    *,
    splits: Iterable[Split] = ("val",),
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_batches: int | None = DEFAULT_MAX_BATCHES,
    seq_len: int | None = None,
    on_batch: Callable[[str, int, int], None] | None = None,
) -> EvalReport:
    """Measure ``session`` on each of ``splits`` and collect the results.

    Raises:
        UsageError: The dataset is not the one this model was trained on, or a
            requested split cannot be scored.
    """
    _check_dataset_matches(session, dataset)

    results: list[SplitResult] = []
    for split in splits:
        results.append(
            evaluate_split(
                session,
                dataset,
                split,
                batch_size=batch_size,
                max_batches=max_batches,
                seq_len=seq_len,
                on_batch=(lambda done, tokens, s=split: on_batch(s, done, tokens))
                if on_batch is not None
                else None,
            )
        )

    notes: list[str] = []
    check = dataset_check(session, dataset)
    if check == "mismatch":
        # First, because it changes what every number below it means.
        notes.append(
            "This is not the dataset this run was trained on, so its val split is not "
            "necessarily held out from this model -- if it is the same corpus prepared "
            "again with a different --seed, some of it was trained on. Read these as "
            "perplexity on this text, not as a generalisation measurement."
        )
    elif check == "unknown":
        # Also first, and deliberately not silence. The check that would have caught a
        # resplit could not run, so the strongest true statement is that it did not run.
        notes.append(
            "Could not check whether this is the dataset this run trained on: one of "
            "the two content hashes is missing, which means a manifest was edited or "
            "truncated. If it is the same corpus prepared again, its val split may hold "
            "text this model was trained on and these numbers would flatter it. Rerun "
            "`trainai data prepare` to get a manifest that can be checked."
        )
    notes.append(
        f"Perplexity is per token of this run's {session.tokenizer.vocab_size}-token "
        "vocabulary, so it is not comparable to a model with a different tokenizer.",
    )
    for result in results:
        if not result.loss_mask:
            continue
        share = result.tokens_scored / result.positions_seen if result.positions_seen else 0.0
        notes.append(
            f"Scored {result.tokens_scored:,} of the {result.split} split's "
            f"{result.positions_seen:,} predicted positions ({share * 100:.1f}%): this "
            "dataset carries a loss mask and this run trained on it, so the number is "
            "perplexity over the assistant's replies, not over the prompts as well. "
            "Train with --no-loss-mask to get a number over every token."
        )
    trained_context = min(session.train_config.seq_len, session.model_config.seq_len)
    beyond = sorted({result.seq_len for result in results if result.seq_len > trained_context})
    if beyond:
        # Only reachable via an explicit --seq-len now that the default is the trained
        # context. Saying so is the point: the number is still worth having, it is just
        # not the one the training curve reports, and the "Run recorded" comparison
        # below it will show a gap that has nothing to do with sampling.
        notes.append(
            f"Scored at {', '.join(f'{n:,}' for n in beyond)} tokens of context, but "
            f"this run trained at {trained_context:,}. The model was built for "
            f"{session.model_config.seq_len:,}, so it runs, but positions past "
            f"{trained_context:,} were never trained -- read this as extrapolation, "
            "not as a number comparable to the training curve."
        )
    for result in results:
        if result.stopped_early:
            notes.append(
                f"Only {result.windows_scored} of the {result.split} split's "
                f"{result.windows_in_split} sequences were scored ("
                f"{result.coverage * 100:.1f}%) because --max-batches stopped it early."
            )
        elif result.tokens_outside_windows:
            notes.append(
                f"The whole {result.split} split was scored except its last "
                f"{result.tokens_outside_windows:,} tokens, which do not fill a "
                f"sequence of {result.seq_len}."
            )
    if session.dtype != torch.float32:
        notes.append(
            f"Measured under {session.precision_note}. Loss under reduced precision "
            "differs from fp32 in the third decimal; use --precision fp32 to pin it."
        )

    return EvalReport(
        results=results,
        step=session.step,
        vocab_size=session.tokenizer.vocab_size,
        parameters=session.model.parameter_count(),
        device=str(session.device),
        precision=session.precision_note,
        checkpoint=str(session.layout.checkpoint_path),
        which=session.layout.which,
        dataset=str(dataset.root) if dataset.root else None,
        dataset_matches_run=check == "match",
        dataset_check=check,
        trained_seq_len=trained_context,
        notes=notes,
    )


def dataset_check(session: InferenceSession, dataset: DatasetManifest) -> DatasetCheck:
    """Whether ``dataset`` is byte-for-byte the one this run trained on -- or unknown.

    Compared on the content hash, which covers the shard checksums, the split seed and
    the validation fraction -- so it changes when the train/val boundary moves, even if
    the corpus and the tokenizer are the same. That case is the dangerous one: the
    tokenizer fingerprint still matches, so nothing else notices, and the ``val`` split
    now holds documents this model was trained on. The measured perplexity comes out
    lower and looks like a better model.

    ``"unknown"`` when either side recorded no hash, which is a third answer rather than
    a lenient ``"match"``. It used to return the equivalent of ``"match"``, justified in
    a comment by "runs trained before the checkpoint recorded one" -- a class of run that
    has never existed: the dataset manifest has written a ``content_hash`` since the
    commit that introduced manifests, the checkpoint has recorded one since the commit
    that introduced checkpoints, and ``FORMAT_VERSION`` has never left 1. So nothing was
    being kept compatible. What the branch actually covered was a hand-edited or
    truncated manifest, and there the permissive answer was measurably wrong: stripping
    ``content_hash`` from a copy of a dataset produced output identical to a verified
    match, including ``"dataset_matches_run": true`` in the JSON, with the resplit
    warning absent.
    """
    recorded = session.checkpoint.dataset.get("content_hash")
    if not recorded or not dataset.content_hash:
        return "unknown"
    return "match" if str(recorded) == dataset.content_hash else "mismatch"


def _check_dataset_matches(session: InferenceSession, dataset: DatasetManifest) -> None:
    """Refuse to score a model against a dataset it was not trained for.

    Not the same check as the tokenizer one in ``infer.session``: that one is about
    whether ids can be decoded, this one is about whether the number means anything.
    Perplexity on text the model was trained on is a different quantity from
    perplexity on held-out text, and a mismatched dataset silently turns the second
    into the first -- or into a number over a token stream the model has never seen
    the units of. Both produce a plausible float.
    """
    trained_with = session.checkpoint.dataset.get("tokenizer_fingerprint")
    if trained_with and dataset.tokenizer_fingerprint != trained_with:
        raise UsageError(
            "This dataset was tokenized differently from the one the model was "
            "trained on, so a perplexity over it would not mean anything.",
            hint=(
                "Evaluate against the dataset this run was trained on. The run's "
                "checkpoint records which one that was."
            ),
            details={
                "model_tokenizer_fingerprint": trained_with,
                "dataset_tokenizer_fingerprint": dataset.tokenizer_fingerprint,
                "dataset": str(dataset.root),
                "trained_on": session.checkpoint.dataset.get("root"),
            },
        )

    if dataset.vocab_size != session.model_config.vocab_size:
        raise UsageError(
            f"The model has an output layer of {session.model_config.vocab_size} "
            f"tokens and this dataset has a vocabulary of {dataset.vocab_size}.",
            hint="Evaluate against the dataset this run was trained on.",
            details={
                "model_vocab_size": session.model_config.vocab_size,
                "dataset_vocab_size": dataset.vocab_size,
                "dataset": str(dataset.root),
            },
        )
