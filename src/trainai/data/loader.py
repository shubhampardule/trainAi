"""Sample batches from binary token shards.

Pure numpy on purpose: no torch here, so the sampler's determinism and its
train/validation disjointness can be tested in milliseconds on any machine, and
the trainer's only job is ``torch.from_numpy``.

The sampling scheme is the part worth reading. Two properties are needed at once,
and the obvious approaches each give up one of them:

*Reproducibility at an arbitrary step.* Resuming a run at step 12,000 must produce
exactly the batches an uninterrupted run would have produced, without replaying
12,000 draws. So the batch for a step is a pure function of ``(seed, step)``.

*Coverage.* Uniform sampling with replacement -- the usual shortcut -- sees roughly
63% of the corpus per notional epoch and some windows several times. On a
data-limited corpus, which is the normal case for this tool, that waste is real.

What is implemented instead: the stream is cut into non-overlapping windows of
``seq_len + 1`` tokens, and each epoch is a deterministic permutation of those
windows, derived from ``(seed, epoch)``. Any step can be reconstructed from its
number alone, and a per-epoch random offset shifts the window alignment so the
same token boundaries are not reused every pass.

The remainder is handled honestly rather than silently: an epoch is
``windows // batch_size`` full batches, so ``windows % batch_size`` windows -- at
most ``batch_size - 1`` of them -- go unused in any single epoch. Because the
permutation is redrawn each epoch, the leftovers are different windows every time,
so nothing is permanently excluded from training. Partial batches are not padded,
since a padded batch would contribute a gradient computed on tokens that are not
in the corpus.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISREG

import numpy as np

from trainai.data.binarize import MASK_DTYPE, SPLITS, DatasetManifest, ShardInfo, Split
from trainai.errors import DatasetError, DatasetFormatError

__all__ = [
    "Batch",
    "ShardedTokenStream",
    "TokenBatcher",
    "open_split",
    "windows_available",
]


#: Shortest sequence worth sampling: one input token and one target token.
MIN_SEQ_LEN = 2


def window_count(total_tokens: int, seq_len: int) -> int:
    """Non-overlapping sequences ``total_tokens`` yields at ``seq_len``.

    ``seq_len`` tokens are reserved so the per-epoch alignment offset can never push
    the last window past the end of the stream. Shared with :func:`windows_available`
    so a caller that needs the count before opening the shards gets the same answer
    the batcher will.
    """
    return max(0, (total_tokens - seq_len) // (seq_len + 1))


def largest_seq_len(total_tokens: int) -> int:
    """Longest ``seq_len`` at which ``total_tokens`` still yields one window.

    The inverse of :func:`window_count`. A window is ``seq_len + 1`` tokens and
    another ``seq_len`` is reserved for the alignment offset, so a split needs
    ``2 * seq_len + 1`` tokens and the largest context it supports is
    ``(total_tokens - 1) // 2``. Returns 0 when no context works, which is any split
    under ``2 * MIN_SEQ_LEN + 1`` tokens -- deliberately 0 rather than 1, because a
    ``seq_len`` of 1 is refused, and suggesting it would be advice this module rejects.

    It exists because the requirement used to be restated by each caller that wanted
    to explain it, and one of them got it wrong by about half: `trainai eval` told a
    user with a 107-token validation split that it "needs at least 65 tokens", two
    lines under its own message saying it needed 129.
    """
    largest = (total_tokens - 1) // 2
    return largest if largest >= MIN_SEQ_LEN else 0


def windows_available(dataset: str | Path | DatasetManifest, split: Split, *, seq_len: int) -> int:
    """How many sequences a split holds, from the manifest alone.

    Reads no shard files. Exists so a caller can size a batch to what is actually
    there -- evaluation does this, because a validation split smaller than the
    default batch size is scoreable but the batcher, sized for training, refuses it.
    """
    manifest = (
        dataset if isinstance(dataset, DatasetManifest) else DatasetManifest.load(Path(dataset))
    )
    return window_count(manifest.tokens(split), seq_len)


@dataclass(frozen=True)
class Batch:
    """One training batch. ``inputs`` and ``targets`` are ``int64``, shape (B, T).

    ``int64`` because that is what an embedding lookup wants; the shards are
    ``uint16`` on disk and widened here, which costs one small copy per step and
    saves half the disk and page cache.

    ``loss_mask`` is ``uint8`` of the same shape, or ``None`` on a dataset prepared
    without one. It is aligned with ``targets``, not ``inputs``: a mask byte
    describes the token it was written for, and the loss at position *i* scores
    ``targets[i]``. Getting that off by one would train on the last context token of
    every prompt and skip the first token of every reply -- a shift small enough to
    converge and be wrong.
    """

    inputs: np.ndarray
    targets: np.ndarray
    step: int
    epoch: int
    loss_mask: np.ndarray | None = None

    @property
    def trained_tokens(self) -> int:
        """Tokens the loss will actually be computed over.

        Every token when there is no mask, which is what makes an unmasked run and a
        masked one comparable in the same reporting code.
        """
        if self.loss_mask is None:
            return int(self.targets.size)
        return int(self.loss_mask.sum())

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.inputs.shape[0]), int(self.inputs.shape[1]))


class ShardedTokenStream:
    """Several shard files addressed as one contiguous token array.

    Reads that cross a shard boundary are stitched rather than skipped. Skipping
    would be simpler and would silently drop ``seq_len`` tokens at every boundary;
    at 64Mi-token shards that is a rounding error, but "silently drops some of your
    data" is not a sentence this project wants to be true anywhere.
    """

    def __init__(self, paths: list[Path], tokens: list[int], dtype: np.dtype) -> None:
        if len(paths) != len(tokens):
            raise DatasetError(
                "Shard paths and token counts do not match.",
                hint="This is a TrainAI bug in dataset loading; please report it.",
                details={"paths": len(paths), "counts": len(tokens)},
            )
        self._paths = paths
        self._tokens = tokens
        self._dtype = dtype
        # Exclusive prefix sums, so a global offset maps to a shard with one
        # searchsorted rather than a loop.
        self._starts = np.cumsum([0, *tokens], dtype=np.int64)
        self._maps: list[np.memmap | None] = [None] * len(paths)

    @property
    def total_tokens(self) -> int:
        return int(self._starts[-1])

    def read(self, start: int, count: int) -> np.ndarray:
        """Return ``count`` tokens beginning at global offset ``start``."""
        if start < 0 or count < 0 or start + count > self.total_tokens:
            raise DatasetError(
                f"Read of {count} tokens at offset {start} is outside the "
                f"{self.total_tokens}-token stream.",
                hint="This is a TrainAI bug in dataset loading; please report it.",
                details={"start": start, "count": count, "total": self.total_tokens},
            )
        if count == 0:
            return np.empty(0, dtype=self._dtype)

        index = int(np.searchsorted(self._starts, start, side="right") - 1)
        offset = start - int(self._starts[index])
        pieces: list[np.ndarray] = []
        wanted = count
        while wanted > 0:
            shard = self._shard(index)
            take = min(wanted, shard.size - offset)
            pieces.append(np.asarray(shard[offset : offset + take]))
            wanted -= take
            offset = 0
            index += 1
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)

    def close(self) -> None:
        """Release the memory maps.

        Worth calling on Windows, where an open map keeps a lock on the file and a
        later attempt to re-prepare the dataset in place would fail with a
        permission error rather than anything informative.
        """
        self._maps = [None] * len(self._paths)

    def _shard(self, index: int) -> np.memmap:
        existing = self._maps[index]
        if existing is not None:
            return existing
        path = self._paths[index]
        expected_tokens = self._tokens[index]
        itemsize = self._dtype.itemsize
        expected_bytes = expected_tokens * itemsize

        # The size is checked before mapping, not after. ``np.memmap`` raises
        # ValueError -- not OSError -- on an empty file ("cannot mmap an empty file")
        # and on a length that is not a multiple of the itemsize ("Size of available
        # data is not a multiple of the data-type size"), so an OSError-only handler
        # never saw either. Measured: one byte cut off a shard ended `trainai train`
        # with a numpy traceback and exit 1, raised inside numpy/_core/memmap.py,
        # while `trainai data inspect --verify` on the same dataset reported it
        # cleanly. That asymmetry is what this check removes.
        #
        # A truncation of an *even* number of bytes always mapped fine and was caught
        # afterwards by the token count, which is why the gap went unnoticed. Checking
        # bytes up front covers every truncation, and it is the same comparison
        # ``verify_dataset`` makes, deliberately worded the same way.
        try:
            info = path.stat()
        except OSError as exc:
            raise self._unreadable(path, exc) from exc
        # Regular files only. A directory where a shard should be stats as zero bytes
        # on Windows, and the truncation message below would blame an interrupted copy
        # for something that is not a copy at all; the map attempt names it better.
        if S_ISREG(info.st_mode) and info.st_size != expected_bytes:
            raise DatasetFormatError(
                f"Shard {path.name} is {info.st_size} bytes; the manifest says "
                f"{expected_bytes} ({expected_tokens} tokens x {itemsize} bytes).",
                hint=(
                    "The file was truncated or partly overwritten, most likely by an "
                    "interrupted copy or a full disk. Re-run `trainai data prepare`."
                ),
                details={"path": str(path), "size": info.st_size, "expected": expected_bytes},
            )

        try:
            mapped = np.memmap(path, dtype=self._dtype, mode="r")
        except (OSError, ValueError) as exc:
            # ValueError as well as OSError, even though the check above rules out the
            # two cases numpy documents it for: reaching here means the file is not
            # what was just measured, and a named error beats a traceback regardless.
            raise self._unreadable(path, exc) from exc
        if mapped.size != expected_tokens:
            # Not reachable from a static file -- the byte check already settled it.
            # It stays because the file can change between the stat and the map, and
            # a stream whose shard is the wrong length reads past its own boundaries
            # in `read()`. One comparison per shard open is a cheap backstop for a
            # wrong answer.
            raise DatasetFormatError(
                f"Shard {path.name} holds {mapped.size} tokens; the manifest says "
                f"{expected_tokens}.",
                hint=(
                    "The file changed while it was being opened. Re-run `trainai data "
                    "prepare`, and make sure nothing else is writing to the dataset."
                ),
                details={
                    "path": str(path),
                    "found": int(mapped.size),
                    "expected": expected_tokens,
                },
            )
        self._maps[index] = mapped
        return mapped

    @staticmethod
    def _unreadable(path: Path, exc: Exception) -> DatasetFormatError:
        """The error for a shard that cannot be measured or mapped at all."""
        return DatasetFormatError(
            f"Shard {path.name} could not be opened.",
            hint=(
                "The file is missing or unreadable. Run `trainai data inspect "
                "<dataset> --verify` to check the dataset, then re-prepare it if "
                "needed."
            ),
            details={"path": str(path), "reason": str(exc)},
        )


class TokenBatcher:
    """Deterministic batch sampler over one split."""

    def __init__(
        self,
        stream: ShardedTokenStream,
        *,
        seq_len: int,
        batch_size: int,
        seed: int = 0,
        split: str = "train",
        mask_stream: ShardedTokenStream | None = None,
    ) -> None:
        if seq_len < MIN_SEQ_LEN:
            raise DatasetError(
                f"--seq-len must be at least {MIN_SEQ_LEN}, got {seq_len}.",
                hint="A sequence needs at least one input token and one target token.",
                details={"seq_len": seq_len},
            )
        if batch_size < 1:
            raise DatasetError(
                f"--batch-size must be at least 1, got {batch_size}.",
                hint="Use 1 if memory is tight; the trainer reaches larger effective "
                "batches with gradient accumulation.",
                details={"batch_size": batch_size},
            )

        self._stream = stream
        self._mask_stream = mask_stream
        self._seq_len = seq_len
        self._window = seq_len + 1
        self._batch_size = batch_size
        self._seed = seed
        self._split = split

        if mask_stream is not None and mask_stream.total_tokens != stream.total_tokens:
            # Checked here rather than trusted from the manifest, because the two
            # streams are addressed with the *same* offset arithmetic: a mask stream
            # one token shorter does not fail, it reads a mask byte belonging to the
            # next token for every window past the discrepancy. A dataset that trains
            # and converges on the wrong tokens is the failure this whole feature is
            # built to avoid, so the disagreement is refused before the first read.
            raise DatasetFormatError(
                f"The {split} split has {stream.total_tokens:,} tokens but its loss mask "
                f"covers {mask_stream.total_tokens:,}.",
                hint=(
                    "Run `trainai data inspect <dataset> --verify` to check the dataset, "
                    "then re-run `trainai data prepare`."
                ),
                details={
                    "split": split,
                    "tokens": stream.total_tokens,
                    "mask_tokens": mask_stream.total_tokens,
                },
            )

        # Reserve seq_len tokens so the per-epoch alignment offset can never push
        # the last window past the end of the stream.
        self._windows = window_count(stream.total_tokens, seq_len)
        if self._windows < 1:
            # The hint names the longest context that does fit, computed rather than
            # described. Callers used to restate the requirement in their own words --
            # `trainai eval` said "needs at least seq_len + 1 tokens", roughly half the
            # real figure -- so there is one place that knows it now, and it answers
            # with a value rather than a direction.
            usable = largest_seq_len(stream.total_tokens)
            raise DatasetError(
                f"The {split} split has {stream.total_tokens:,} tokens, which is not "
                f"enough for even one sequence of {seq_len} (needs "
                f"{self._window + seq_len:,}).",
                hint=(
                    f"Use --seq-len {usable} or lower, or prepare more data. For the "
                    "validation split, raising --val-fraction during `trainai data "
                    "prepare` also helps."
                )
                if usable
                else (
                    f"No --seq-len fits: even {MIN_SEQ_LEN} needs "
                    f"{2 * MIN_SEQ_LEN + 1} tokens. Prepare more data, or for the "
                    "validation split raise --val-fraction during `trainai data "
                    "prepare`."
                ),
                details={
                    "split": split,
                    "tokens": stream.total_tokens,
                    "seq_len": seq_len,
                    "tokens_needed": self._window + seq_len,
                    "largest_seq_len": usable,
                },
            )
        if self._windows < batch_size:
            raise DatasetError(
                f"The {split} split holds {self._windows} sequence(s) of {seq_len} "
                f"tokens, fewer than the batch size of {batch_size}.",
                hint=(
                    f"Use --batch-size {self._windows} or lower --seq-len. Repeating the "
                    "same sequence inside one batch would make the gradient smaller than "
                    "it looks, so TrainAI refuses rather than doing it quietly."
                ),
                details={
                    "split": split,
                    "windows": self._windows,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                },
            )

        self._steps_per_epoch = self._windows // batch_size
        self._usable_windows = self._steps_per_epoch * batch_size
        self._cached_epoch: int | None = None
        self._cached_order: np.ndarray | None = None
        self._cached_offset = 0

    # -- shape ------------------------------------------------------------- #

    @property
    def total_tokens(self) -> int:
        return self._stream.total_tokens

    @property
    def windows(self) -> int:
        """Non-overlapping sequences available in this split."""
        return self._windows

    @property
    def steps_per_epoch(self) -> int:
        """Full batches in one pass over the split.

        ``windows % batch_size`` windows are left over and skipped in that epoch;
        the permutation is redrawn each epoch, so a different set is skipped next
        time and no window is permanently excluded.
        """
        return self._steps_per_epoch

    @property
    def windows_per_epoch(self) -> int:
        """Windows actually visited in one epoch: ``steps_per_epoch * batch_size``."""
        return self._usable_windows

    @property
    def has_loss_mask(self) -> bool:
        """Whether the batches this batcher yields carry a mask."""
        return self._mask_stream is not None

    def close(self) -> None:
        """Release both streams' memory maps.

        The way to finish with what :func:`open_split` returned: closing only the
        token stream would leave the mask maps open, which on Windows keeps a lock on
        files that `trainai data prepare` may be about to rewrite in place.
        """
        self._stream.close()
        if self._mask_stream is not None:
            self._mask_stream.close()

    # -- sampling ---------------------------------------------------------- #

    def batch(self, step: int) -> Batch:
        """The batch for ``step``. A pure function of ``(seed, step)``."""
        if step < 0:
            raise DatasetError(
                f"step must be non-negative, got {step}.",
                hint="This is a TrainAI bug in dataset loading; please report it.",
                details={"step": step},
            )
        base = step * self._batch_size
        epoch = base // self._usable_windows
        within = base % self._usable_windows
        order, offset = self._epoch_order(epoch)
        indices = order[within : within + self._batch_size]

        rows = np.empty((self._batch_size, self._window), dtype=np.int64)
        masks = self._empty_mask(self._batch_size)
        for row, index in enumerate(indices):
            start = offset + int(index) * self._window
            rows[row] = self._stream.read(start, self._window)
            if masks is not None:
                masks[row] = self._mask_read(start)
        return Batch(
            inputs=rows[:, :-1],
            targets=rows[:, 1:],
            step=step,
            epoch=epoch,
            loss_mask=None if masks is None else masks[:, 1:],
        )

    def sequential_batches(self, *, max_batches: int | None = None) -> Iterator[Batch]:
        """Walk the split once in order, for evaluation.

        In order and without a random offset, so perplexity over a fixed split is
        the same number every time it is measured. The final batch may hold fewer
        than ``batch_size`` rows; dropping it would quietly exclude the tail of the
        validation set from every evaluation.
        """
        for step in range((self._windows + self._batch_size - 1) // self._batch_size):
            if max_batches is not None and step >= max_batches:
                return
            first = step * self._batch_size
            rows_wanted = min(self._batch_size, self._windows - first)
            rows = np.empty((rows_wanted, self._window), dtype=np.int64)
            masks = self._empty_mask(rows_wanted)
            for row in range(rows_wanted):
                start = (first + row) * self._window
                rows[row] = self._stream.read(start, self._window)
                if masks is not None:
                    masks[row] = self._mask_read(start)
            yield Batch(
                inputs=rows[:, :-1],
                targets=rows[:, 1:],
                step=step,
                epoch=0,
                loss_mask=None if masks is None else masks[:, 1:],
            )

    def _empty_mask(self, rows: int) -> np.ndarray | None:
        """Uninitialised mask rows, or ``None`` when this split has no mask."""
        if self._mask_stream is None:
            return None
        return np.empty((rows, self._window), dtype=MASK_DTYPE)

    def _mask_read(self, start: int) -> np.ndarray:
        """The mask window matching the token window at ``start``.

        The same ``(start, window)`` arithmetic as the token read, deliberately: the
        alignment between a token and its mask byte is the offset, and computing it
        twice in two places is how it drifts.
        """
        assert self._mask_stream is not None
        return self._mask_stream.read(start, self._window)

    def _epoch_order(self, epoch: int) -> tuple[np.ndarray, int]:
        """Permutation of window indices for ``epoch``, plus the alignment offset.

        Cached for one epoch. Regenerating it costs a shuffle of ``windows``
        integers, which is milliseconds for a corpus of a few hundred million
        tokens, and it happens once per epoch rather than once per step.
        """
        if self._cached_epoch == epoch and self._cached_order is not None:
            return self._cached_order, self._cached_offset

        # A SeedSequence built from (seed, epoch, split) gives independent streams
        # for train and val without the caller having to manage two seeds.
        entropy = [self._seed, epoch, _split_salt(self._split)]
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        order = rng.permutation(self._windows)
        if self._windows < (1 << 31):
            order = order.astype(np.int32, copy=False)
        offset = int(rng.integers(0, self._seq_len + 1))

        self._cached_epoch = epoch
        self._cached_order = order
        self._cached_offset = offset
        return order, offset


def open_split(
    dataset: str | Path | DatasetManifest,
    split: Split = "train",
    *,
    seq_len: int,
    batch_size: int,
    seed: int = 0,
    loss_mask: bool = True,
) -> tuple[TokenBatcher, ShardedTokenStream]:
    """Open one split of a prepared dataset for batching.

    Returns the batcher and the underlying token stream. Call ``batcher.close()``
    when done -- it releases the mask stream as well, which the returned stream
    knows nothing about.

    ``loss_mask`` is honoured only if the dataset has one; a dataset prepared without
    a mask yields batches whose ``loss_mask`` is ``None``, and every token is scored.
    Passing ``False`` on a masked dataset trains on every token deliberately, which is
    what `trainai train --no-loss-mask` is for.
    """
    manifest = (
        dataset if isinstance(dataset, DatasetManifest) else DatasetManifest.load(Path(dataset))
    )
    if split not in SPLITS:
        raise DatasetError(
            f"Unknown split {split!r}.",
            hint=f"Use one of: {', '.join(SPLITS)}.",
            details={"split": split},
        )
    shards: list[ShardInfo] = list(manifest.shards.get(split, ()))
    if not shards:
        # The train case used to read "a dataset without a train split is a bug" and
        # advise re-running `data prepare`. Neither held: the split is a per-document
        # hash, so a small corpus at a high --val-fraction could legitimately send
        # every document to validation, and re-running with the same seed reproduced
        # it exactly. `binarize_documents` now refuses to write such a dataset, so a
        # train split can only be missing on one prepared before that guard existed
        # or edited by hand -- and the advice names the cause either way.
        raise DatasetError(
            f"The dataset has no {split} split.",
            hint=(
                "It was prepared with --val-fraction 0, so there is no held-out data. "
                "Re-run `trainai data prepare` with a non-zero --val-fraction to "
                "measure validation loss."
            )
            if split == "val"
            else (
                "Re-run `trainai data prepare` with a lower --val-fraction. This "
                "dataset was prepared with one high enough to send every document to "
                "validation, which `trainai data prepare` now refuses to write -- so "
                "this dataset predates that check, or was edited by hand."
            ),
            details={"split": split, "dataset": str(manifest.root)},
        )

    stream = ShardedTokenStream(
        paths=manifest.shard_paths(split),
        tokens=[shard.tokens for shard in shards],
        dtype=manifest.numpy_dtype,
    )
    # `all`, never `any`: DatasetManifest.load already refuses a partly masked
    # dataset, and reading the mask of only the shards that have one would score
    # every token of the shards that do not -- the exact failure the mask exists to
    # stop. The paths come from the same shard entries as the tokens, so they are in
    # shard order by construction rather than by a second sort.
    mask_stream = (
        ShardedTokenStream(
            paths=manifest.mask_paths(split),
            tokens=[shard.tokens for shard in shards],
            dtype=MASK_DTYPE,
        )
        if loss_mask and manifest.has_loss_mask
        else None
    )
    batcher = TokenBatcher(
        stream,
        seq_len=seq_len,
        batch_size=batch_size,
        seed=seed,
        split=split,
        mask_stream=mask_stream,
    )
    return batcher, stream


def _split_salt(split: str) -> int:
    """Stable small integer per split name, so train and val shuffle differently."""
    return sum(ord(c) * (i + 1) for i, c in enumerate(split))
