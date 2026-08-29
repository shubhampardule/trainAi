"""Train and use a byte-level BPE tokenizer.

Byte-level, specifically, and that choice does most of the work here. The initial
alphabet is all 256 byte values, so every possible input has a representation and
there is no unknown token: ``decode(encode(text)) == text`` holds for any string,
including text in scripts absent from the training corpus. A word-level or
character-level tokenizer would need an ``<unk>`` and would silently destroy
whatever it had not seen. Silent destruction of user data is not acceptable, so
:meth:`ByteLevelBPE.verify_roundtrip` runs on every freshly trained tokenizer and
refuses to hand back one that fails.

The BPE implementation is the Rust ``tokenizers`` library. Reimplementing it in
Python would be slower by two orders of magnitude and no more instructive.

This module is the only one under :mod:`trainai.data` that imports a non-stdlib
library other than numpy at module scope, and it still does not import torch.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from trainai.errors import TokenizerError

__all__ = [
    "BYTE_ALPHABET_SIZE",
    "EOT_TOKEN",
    "MIN_VOCAB_SIZE",
    "ROUNDTRIP_PROBES",
    "ByteLevelBPE",
    "TokenizerReport",
    "train_tokenizer",
]

# The end-of-text marker. Trained as a special token so BPE never merges across
# it, and given id 0 by being first in the special-token list -- a fixed, small,
# easily recognised id that shows up as a literal 0 in a hex dump of a shard.
EOT_TOKEN = "<|endoftext|>"

# Byte-level BPE always starts from the 256 single-byte tokens.
BYTE_ALPHABET_SIZE = 256

# The smallest ``vocab_size`` that can produce a tokenizer rather than just the byte
# alphabet: the 256 byte values, the end-of-text token, and room for one merge. A
# target of 257 fills up on the alphabet and the special token alone, so it can only
# ever fail -- and it failed *late*, after reading the whole corpus, with "no merges
# were learned", which blames the corpus for a budget that could not have worked.
# ``trainai.data.validate`` keeps its own copy of this number, because importing this
# module would make ``trainai --help`` load the Rust tokenizers library; the two are
# cross-checked by a test.
MIN_VOCAB_SIZE = BYTE_ALPHABET_SIZE + 2

# Round-trip probes. Written as escape sequences so this file stays pure ASCII and
# cannot be mangled by an editor, a terminal, or a git filter. Each one targets a
# way tokenizers usually break.
ROUNDTRIP_PROBES: tuple[tuple[str, str], ...] = (
    ("ascii", "The quick brown fox jumps over the lazy dog."),
    ("leading_space", " leading space matters"),
    ("trailing_space", "trailing space matters "),
    ("double_space", "two  spaces  between  words"),
    ("tabs_newlines", "a\tb\r\nc\n\nd"),
    ("only_whitespace", "   \n\t  "),
    ("empty", ""),
    ("digits", "3.14159 and 1,000,000 and 0x1F"),
    # Latin-1 supplement and combining marks: naive lowercasing or NFC
    # normalisation would change these.
    ("accented", "na\u00efve caf\u00e9 Stra\u00dfe \u00e9\u0301"),
    ("cjk", "\u4e2d\u6587\u6d4b\u8bd5\uff1a\u4f60\u597d\u4e16\u754c"),
    ("kana", "\u3053\u3093\u306b\u3061\u306f\u30ab\u30bf\u30ab\u30ca"),
    ("hangul", "\ud55c\uad6d\uc5b4 \ud14c\uc2a4\ud2b8"),
    ("cyrillic", "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440"),
    ("arabic_rtl", "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645"),
    # Astral plane: four UTF-8 bytes, and a surrogate-pair in UTF-16 builds.
    ("emoji", "\U0001f600\U0001f680 \U0001f1ec\U0001f1e7"),
    # A grapheme cluster made of several codepoints, plus a variation selector.
    ("zwj_sequence", "\U0001f469\u200d\U0001f4bb \u2764\ufe0f"),
    ("mixed", "Hello \u4e16\u754c 123 \U0001f30d na\u00efve\n\ttab"),
    ("eot_literal", f"before {EOT_TOKEN} after"),
)


@dataclass(frozen=True)
class TokenizerReport:
    """Measured compression of a real sample. No estimates in here."""

    vocab_size: int
    sample_documents: int
    sample_chars: int
    sample_utf8_bytes: int
    sample_tokens: int

    @property
    def chars_per_token(self) -> float:
        return self.sample_chars / self.sample_tokens if self.sample_tokens else 0.0

    @property
    def bytes_per_token(self) -> float:
        return self.sample_utf8_bytes / self.sample_tokens if self.sample_tokens else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "sample_documents": self.sample_documents,
            "sample_chars": self.sample_chars,
            "sample_utf8_bytes": self.sample_utf8_bytes,
            "sample_tokens": self.sample_tokens,
            "chars_per_token": round(self.chars_per_token, 3),
            "bytes_per_token": round(self.bytes_per_token, 3),
        }


class ByteLevelBPE:
    """A trained tokenizer plus the operations TrainAI needs from it."""

    def __init__(self, tokenizer: Tokenizer, *, requested_vocab_size: int | None = None) -> None:
        self._tokenizer = tokenizer
        # What the user asked for, when known. BPE training stops early when the
        # corpus runs out of pairs above --min-frequency, so a request for 32,768
        # can produce 6,000. Everything downstream must size itself from
        # `vocab_size` (what exists) and not from the request; keeping the request
        # around is what lets the CLI tell the user the two differ.
        self._requested_vocab_size = requested_vocab_size
        eot_id = tokenizer.token_to_id(EOT_TOKEN)
        if eot_id is None:
            raise TokenizerError(
                f"The tokenizer has no {EOT_TOKEN} token.",
                hint=(
                    "TrainAI needs an end-of-text token to separate documents. Train a "
                    "tokenizer with `trainai data prepare` rather than supplying one "
                    "from elsewhere, or add the token before loading it."
                ),
                details={"vocab_size": tokenizer.get_vocab_size()},
            )
        self._eot_id = int(eot_id)

    # -- construction ------------------------------------------------------ #

    @classmethod
    def load(cls, path: str | Path) -> ByteLevelBPE:
        """Load ``tokenizer.json``. Raises :class:`TokenizerError` if unusable."""
        path = Path(path)
        if not path.exists():
            raise TokenizerError(
                f"No tokenizer at {path}",
                hint=(
                    "Point --tokenizer at the tokenizer.json inside a prepared dataset "
                    "directory, or run `trainai data prepare` to create one."
                ),
                details={"path": str(path)},
            )
        try:
            tokenizer = Tokenizer.from_file(str(path))
        except Exception as exc:  # the Rust layer raises bare Exception
            raise TokenizerError(
                f"{path} is not a readable tokenizer file.",
                hint=(
                    "The file must be a tokenizers-library tokenizer.json. If it was "
                    "copied or edited by hand, re-create it with `trainai data prepare`."
                ),
                details={"path": str(path), "reason": str(exc)},
            ) from exc
        return cls(tokenizer)

    def save(self, path: str | Path) -> Path:
        """Write ``tokenizer.json`` atomically and return the path.

        Atomically because a tokenizer truncated by a crash or a full disk would
        be indistinguishable from a valid one until something tried to load it,
        which could be hours into a training run.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        self._tokenizer.save(str(temporary))
        temporary.replace(path)
        return path

    # -- identity ---------------------------------------------------------- #

    @property
    def raw(self) -> Tokenizer:
        """The underlying ``tokenizers.Tokenizer``, for callers that need it."""
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        """The number of tokens that exist. Size embeddings from this."""
        return int(self._tokenizer.get_vocab_size())

    @property
    def requested_vocab_size(self) -> int | None:
        """What was asked for at training time, or ``None`` for a loaded tokenizer."""
        return self._requested_vocab_size

    @property
    def vocab_shortfall(self) -> int:
        """How many fewer tokens were learned than requested; 0 if unknown or met."""
        if self._requested_vocab_size is None:
            return 0
        return max(0, self._requested_vocab_size - self.vocab_size)

    @property
    def eot_id(self) -> int:
        return self._eot_id

    def fingerprint(self) -> str:
        """sha256 of the canonical serialisation.

        Recorded in dataset manifests and checkpoints so that a dataset can never
        be silently paired with a different tokenizer -- the failure mode there is
        a model that trains to a plausible loss and emits garbage, which is
        expensive to diagnose after the fact.
        """
        payload = self._tokenizer.to_str()
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "requested_vocab_size": self._requested_vocab_size,
            "eot_id": self.eot_id,
            "eot_token": EOT_TOKEN,
            "fingerprint": self.fingerprint(),
        }

    # -- use --------------------------------------------------------------- #

    def encode(self, text: str, *, add_eot: bool = False) -> list[int]:
        ids = self._tokenizer.encode(text, add_special_tokens=False).ids
        if add_eot:
            ids.append(self._eot_id)
        return ids

    def encode_batch(self, texts: list[str], *, add_eot: bool = False) -> list[list[int]]:
        """Encode many strings at once; the Rust side parallelises this."""
        encoded = self._tokenizer.encode_batch(texts, add_special_tokens=False)
        if add_eot:
            return [[*e.ids, self._eot_id] for e in encoded]
        return [e.ids for e in encoded]

    def decode(self, ids: Iterable[int]) -> str:
        return self._tokenizer.decode(list(ids), skip_special_tokens=False)

    def measure(self, texts: Iterable[str], *, max_chars: int = 1 << 20) -> TokenizerReport:
        """Measure real compression on up to ``max_chars`` of text."""
        sample: list[str] = []
        chars = 0
        for text in texts:
            if chars >= max_chars:
                break
            sample.append(text)
            chars += len(text)
        tokens = sum(len(ids) for ids in self.encode_batch(sample)) if sample else 0
        return TokenizerReport(
            vocab_size=self.vocab_size,
            sample_documents=len(sample),
            sample_chars=chars,
            sample_utf8_bytes=sum(len(t.encode("utf-8")) for t in sample),
            sample_tokens=tokens,
        )

    # -- self-check -------------------------------------------------------- #

    def verify_roundtrip(
        self, probes: Iterable[tuple[str, str]] | None = None
    ) -> tuple[tuple[str, str], ...]:
        """Check ``decode(encode(x)) == x`` on every probe.

        Returns the probes that were checked, so a caller can report the number.
        Raises :class:`TokenizerError` naming the first failure, its codepoints,
        and what came back instead.
        """
        checked = tuple(ROUNDTRIP_PROBES if probes is None else probes)
        for name, text in checked:
            ids = self.encode(text)
            restored = self.decode(ids)
            if restored != text:
                raise TokenizerError(
                    f"The tokenizer does not round-trip the {name!r} probe: "
                    f"{_codepoints(text)} came back as {_codepoints(restored)}.",
                    hint=(
                        "This is a bug in TrainAI, not in your data -- a byte-level BPE "
                        "must reproduce any input exactly. Please report it with the "
                        "probe name above at "
                        "https://github.com/shubhampardule/trainAi/issues"
                    ),
                    details={
                        "probe": name,
                        "expected": text,
                        "got": restored,
                        "token_ids": ids[:64],
                    },
                )
        return checked


def train_tokenizer(
    documents: Iterable[str],
    *,
    vocab_size: int,
    min_frequency: int = 2,
    extra_special_tokens: Iterable[str] = (),
    verify: bool = True,
) -> ByteLevelBPE:
    """Train a byte-level BPE tokenizer on a stream of documents.

    Args:
        documents: Text, consumed once and streamed to the Rust trainer, so a
            corpus larger than memory is fine.
        vocab_size: Target vocabulary, including the 256 byte tokens and the
            special tokens, and at least one more for a merge --
            :data:`MIN_VOCAB_SIZE` with no extra specials. A target, not a
            guarantee: training stops when no pair occurs ``min_frequency``
            times, so a small corpus yields a smaller vocabulary. Compare
            :attr:`ByteLevelBPE.vocab_size` with
            :attr:`ByteLevelBPE.requested_vocab_size` afterwards.
        min_frequency: A pair must occur this often to become a merge. Guards
            against merges learned from a single occurrence. At least 1; the
            trainer treats 0 as 1, which would make the flag a no-op.
        extra_special_tokens: Additional never-merged tokens, after
            :data:`EOT_TOKEN`.
        verify: Run :meth:`ByteLevelBPE.verify_roundtrip` before returning. Only
            turn this off if you are measuring training time.

    Raises:
        TokenizerError: If ``vocab_size`` cannot fit the byte alphabet and the
            special tokens, if ``min_frequency`` is below 1, or if the corpus
            produced no merges, or if the round-trip check fails.
    """
    specials = [EOT_TOKEN, *dict.fromkeys(t for t in extra_special_tokens if t != EOT_TOKEN)]
    filled = BYTE_ALPHABET_SIZE + len(specials)
    floor = filled + 1
    if vocab_size < floor:
        raise TokenizerError(
            f"vocab_size={vocab_size} is too small: the {BYTE_ALPHABET_SIZE} byte values "
            f"and {len(specials)} special token(s) already fill {filled} entries, so "
            f"{floor} is the smallest target with room for a merge.",
            hint=f"Use --vocab-size {floor} or more. Useful sizes start around 4096.",
            details={"vocab_size": vocab_size, "minimum": floor, "filled_by_alphabet": filled},
        )
    # The Rust trainer clamps 0 to 1 silently, so without this a ``--min-frequency 0``
    # is a flag that does not do what its own help text says. It also fixes the floor
    # that the "no merges" hint below promises, so the advice and the code agree.
    if min_frequency < 1:
        raise TokenizerError(
            f"min_frequency={min_frequency} is too small: a pair cannot become a merge "
            "on fewer than one occurrence.",
            hint="Use --min-frequency 1 to merge any pair that appears. The default is 2.",
            details={"min_frequency": min_frequency, "minimum": 1},
        )

    tokenizer = Tokenizer(models.BPE())
    # add_prefix_space=False: a leading space must not be invented, or decode
    # returns a string that differs from the input. use_regex=True applies the
    # GPT-2 pre-tokenization pattern, which keeps merges from spanning word
    # boundaries and punctuation.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    # Affects reported character offsets only, never token ids.
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=specials,
        # Without this, a byte value absent from the corpus would have no token
        # and the tokenizer would not round-trip text containing it.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    tokenizer.train_from_iterator(_non_empty(documents), trainer=trainer)

    trained = tokenizer.get_vocab_size()
    # Against ``filled``, not ``floor``: ``filled`` entries is the alphabet and the
    # specials with nothing learned on top, so anything above it is at least one merge.
    # Comparing against ``floor`` would make a request for exactly ``floor`` impossible
    # to satisfy -- it can produce at most ``floor`` tokens -- and report the corpus as
    # the reason.
    if trained <= filled:
        # ``vocab_size`` is never the cause, so it is never in the advice. It is a
        # ceiling on how many merges may be kept, and lowering a ceiling cannot create
        # something to keep: measured on the 1.06 MiB tinyshakespeare corpus with
        # ``--min-frequency 99999999``, every target from 4096 down to 258 gave this
        # same error, and 257 was refused by the floor guard above with "Use
        # --vocab-size 258 or more" -- a closed loop. The threshold is half of that
        # measurement and cannot be dropped from it: at the default min_frequency the
        # same corpus trains at every one of those targets, 258 included.
        if min_frequency > 1:
            # The second sentence is not a hedge. This branch cannot tell "no pair
            # occurs min_frequency times" from "no pair occurs at all" without a second
            # pass over a stream the docstring promises to consume once, and both
            # reach it. Measured on 2,000 one-character documents via `trainai data
            # prepare`: this hint fired, and following it with --min-frequency 1 hit
            # the branch below -- exit 3 twice, the second time as a surprise. So the
            # advice names what to try and what it means when trying does not help.
            hint = (
                f"No pair of characters occurs {min_frequency:,} times anywhere in the "
                "corpus, so there was nothing to merge. Use --min-frequency 1, which "
                "merges any pair that appears, and raise it from there. If 1 also learns "
                "nothing, the corpus holds no two adjacent characters to merge at all, "
                "and the fix is more text: a first experiment wants about a megabyte."
            )
        else:
            hint = (
                "The corpus holds no two adjacent characters to merge, even counting a "
                "single occurrence. Use more text: a first experiment wants about a "
                "megabyte."
            )
        raise TokenizerError(
            f"Training learned no merges: the vocabulary is only the "
            f"{BYTE_ALPHABET_SIZE} byte values and {len(specials)} special token(s).",
            hint=hint,
            details={
                "vocab_size_requested": vocab_size,
                "vocab_size_trained": trained,
                "min_frequency": min_frequency,
            },
        )

    bpe = ByteLevelBPE(tokenizer, requested_vocab_size=vocab_size)
    if verify:
        bpe.verify_roundtrip()
    return bpe


def _non_empty(documents: Iterable[str]) -> Iterator[str]:
    """Drop empty strings, which the trainer counts as words and gains nothing from."""
    for text in documents:
        if text:
            yield text


def _codepoints(text: str, limit: int = 24) -> str:
    """``'a\\n'`` as ``U+0061 U+000A`` -- readable on any console, in any encoding.

    Printing the raw string would be friendlier but can itself raise
    ``UnicodeEncodeError`` on a legacy Windows code page, and an error message
    that crashes while being displayed is worse than a terse one.
    """
    if not text:
        return "<empty string>"
    shown = " ".join(f"U+{ord(c):04X}" for c in text[:limit])
    return shown + (" ..." if len(text) > limit else "")
