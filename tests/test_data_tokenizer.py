"""Tests for :mod:`trainai.data.tokenizer`.

The property that matters most here is lossless round-tripping. A byte-level BPE
must satisfy ``decode(encode(x)) == x`` for *any* string, including bytes that
never appeared in the training corpus -- otherwise the model trains on text that
differs from what the user supplied, and nothing downstream can detect it.

The second property is reproducibility: the same corpus and settings must produce
a tokenizer with the same fingerprint, because dataset manifests and (later)
checkpoints are paired by that fingerprint.

Non-ASCII probe data is built from ``chr()`` of explicit codepoints rather than
written as literal characters, the same convention :mod:`trainai.data.tokenizer`
follows. Two reasons: this file stays pure ASCII, so no editor or terminal can
re-encode it; and a wrong codepoint is visible in review, where a wrong glyph is
not. A round-trip test whose input was silently altered proves nothing.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from trainai.data.tokenizer import (
    BYTE_ALPHABET_SIZE,
    EOT_TOKEN,
    MIN_VOCAB_SIZE,
    ByteLevelBPE,
    train_tokenizer,
)
from trainai.errors import TokenizerError

VOCAB = 512


def cps(*codepoints: int) -> str:
    """A string from explicit Unicode codepoints."""
    return "".join(chr(codepoint) for codepoint in codepoints)


# Latin-1 letters: e-acute, i-diaeresis, u-diaeresis, sharp s.
E_ACUTE = cps(0x00E9)
I_DIAERESIS = cps(0x00EF)
U_DIAERESIS = cps(0x00FC)
SHARP_S = cps(0x00DF)
CAFE = f"caf{E_ACUTE}"
NAIVE = f"na{I_DIAERESIS}ve"
UBER = f"{U_DIAERESIS}ber"
STRASSE = f"stra{SHARP_S}e"

# Han: "ni hao" (hello). Arabic: "marhaba" (welcome).
NI_HAO = cps(0x4F60, 0x597D)
SHI_JIE = cps(0x4E16, 0x754C)
MARHABA = cps(0x0645, 0x0631, 0x062D, 0x0628, 0x0627)

# Astral plane: grinning face, rocket. These are the surrogate-pair cases.
GRINNING = cps(0x1F600)
ROCKET = cps(0x1F680)

REPLACEMENT = cps(0xFFFD)  # U+FFFD, what a failed decode leaves behind
COMBINING_ACUTE = cps(0x0065, 0x0301)  # 'e' plus a combining acute, not U+00E9
MATHEMATICAL = cps(0x2200, 0x0078, 0x2208, 0x211D)  # for-all x element-of reals
NUL = cps(0x0000)
CONTROLS = cps(0x0001, 0x0002, 0x001F, 0x007F)

_NON_ASCII_LINE = f"{CAFE.title()} {NAIVE} {UBER} {STRASSE}, {NI_HAO}, {MARHABA}. "


def corpus_text() -> list[str]:
    """A corpus with enough repetition to learn merges from, and some non-ASCII."""
    return [
        "The harbour clock measured every tide that crossed the bridge. " * 30,
        "Orchards ripen; lanterns kindle. Bridges cross rivers, rivers carry boats. " * 30,
        _NON_ASCII_LINE * 20,
    ]


@pytest.fixture(scope="module")
def tokenizer() -> ByteLevelBPE:
    """One trained tokenizer shared by the read-only tests in this module."""
    return train_tokenizer(corpus_text(), vocab_size=VOCAB)


# --------------------------------------------------------------------------- #
# Round-tripping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("ascii", "Hello, world!"),
        ("empty", ""),
        ("single space", " "),
        ("leading space", " leading"),
        ("trailing space", "trailing "),
        ("newlines", "line one\nline two\n\n"),
        ("tabs", "a\tb\tc"),
        ("carriage returns", "line one\r\nline two\r\n"),
        ("accented latin", f"{CAFE} {NAIVE} {UBER}"),
        ("han", NI_HAO + SHI_JIE),
        ("arabic", MARHABA),
        ("astral plane", GRINNING + ROCKET),
        ("combining mark", COMBINING_ACUTE),
        ("nul byte", f"before{NUL}after"),
        ("control bytes", CONTROLS),
        ("replacement char", REPLACEMENT),
        ("mathematical", MATHEMATICAL),
        ("mixed", f"Mix: {CAFE} {NI_HAO} {GRINNING} {NUL} tab\there"),
    ],
)
def test_round_trips_every_probe(tokenizer: ByteLevelBPE, name: str, text: str) -> None:
    assert tokenizer.decode(tokenizer.encode(text)) == text, name


def test_round_trips_every_single_byte_value(tokenizer: ByteLevelBPE) -> None:
    """No ``<unk>`` exists, so every byte needs a token even if training never saw it."""
    for byte in range(256):
        text = bytes([byte]).decode("latin-1")
        assert tokenizer.decode(tokenizer.encode(text)) == text, f"byte {byte}"


def test_round_trips_a_string_of_all_byte_values(tokenizer: ByteLevelBPE) -> None:
    text = bytes(range(256)).decode("latin-1")

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_verify_roundtrip_reports_how_many_probes_it_checked(
    tokenizer: ByteLevelBPE,
) -> None:
    checked = tokenizer.verify_roundtrip()

    assert len(checked) > 5
    assert all(isinstance(name, str) and isinstance(text, str) for name, text in checked)


def test_verify_roundtrip_names_the_failing_probe(
    tokenizer: ByteLevelBPE, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The self-check has to fail loudly, so break decoding and confirm that it does.

    There is no input a correct byte-level BPE fails on -- that is the point of the
    check -- so the only way to reach its failure path is to make decode lie.
    """
    monkeypatch.setattr(ByteLevelBPE, "decode", lambda self, ids: "something else")

    with pytest.raises(TokenizerError) as caught:
        tokenizer.verify_roundtrip([("deliberate", "the original text")])

    assert "deliberate" in str(caught.value)
    assert caught.value.details["probe"] == "deliberate"
    assert caught.value.details["got"] == "something else"
    assert "bug in TrainAI" in (caught.value.hint or "")


def test_a_probe_that_decodes_to_nothing_is_named_as_the_empty_string(
    tokenizer: ByteLevelBPE, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure message renders both sides as codepoints, and one side can be empty.

    ``U+0061 U+000A`` rather than the raw text, because printing the string can itself
    raise ``UnicodeEncodeError`` on a legacy Windows code page and an error that crashes
    while being displayed is worse than a terse one. An empty string renders to no
    codepoints at all, so the message would read "came back as ." -- a sentence that
    describes nothing, on the one failure where "the tokenizer produced nothing" is the
    whole finding.

    Reachable only through a broken decode, like the test above: a correct byte-level BPE
    has no input it fails on, which is the reason the self-check exists.
    """
    monkeypatch.setattr(ByteLevelBPE, "decode", lambda self, ids: "")

    with pytest.raises(TokenizerError) as caught:
        tokenizer.verify_roundtrip([("deliberate", "the original text")])

    message = str(caught.value)
    assert "came back as <empty string>" in message
    assert "came back as ." not in message
    # The other side is a real string, so it still renders as codepoints -- which is
    # what says the placeholder is for the empty case and not for every message.
    assert "U+0074 U+0068 U+0065" in message, "'the' as codepoints"


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
def test_vocabulary_is_larger_than_the_byte_alphabet(tokenizer: ByteLevelBPE) -> None:
    assert tokenizer.vocab_size > BYTE_ALPHABET_SIZE
    assert 0 <= tokenizer.eot_id < tokenizer.vocab_size


def test_encode_can_append_the_end_of_text_token(tokenizer: ByteLevelBPE) -> None:
    plain = tokenizer.encode("some text")
    with_eot = tokenizer.encode("some text", add_eot=True)

    assert with_eot == [*plain, tokenizer.eot_id]


def test_the_end_of_text_token_is_never_produced_by_ordinary_text(
    tokenizer: ByteLevelBPE,
) -> None:
    """Otherwise a document boundary could appear in the middle of a document."""
    for text in ("ordinary prose", _NON_ASCII_LINE, bytes(range(256)).decode("latin-1")):
        assert tokenizer.eot_id not in tokenizer.encode(text)


def test_encode_batch_matches_encoding_one_at_a_time(tokenizer: ByteLevelBPE) -> None:
    texts = ["first document", "second document", ""]

    assert tokenizer.encode_batch(texts) == [tokenizer.encode(text) for text in texts]


def test_encode_batch_can_append_the_end_of_text_token(tokenizer: ByteLevelBPE) -> None:
    batch = tokenizer.encode_batch(["a", "b"], add_eot=True)

    assert all(ids[-1] == tokenizer.eot_id for ids in batch)


def test_vocab_size_below_the_byte_alphabet_is_refused() -> None:
    with pytest.raises(TokenizerError) as caught:
        train_tokenizer(corpus_text(), vocab_size=100)

    assert caught.value.details["minimum"] == MIN_VOCAB_SIZE
    assert caught.value.details["filled_by_alphabet"] == BYTE_ALPHABET_SIZE + 1
    assert "--vocab-size" in (caught.value.hint or "")


@pytest.mark.parametrize("vocab_size", [0, 1, 255, BYTE_ALPHABET_SIZE, MIN_VOCAB_SIZE - 1])
def test_every_vocab_size_below_the_floor_is_refused_before_reading_the_corpus(
    vocab_size: int,
) -> None:
    """``MIN_VOCAB_SIZE - 1`` is the case that used to slip through.

    257 passed the up-front check -- it does fit the alphabet and the special token --
    then read the whole corpus and failed with "no merges were learned", which blames
    the corpus for a budget that had no room for a merge in the first place. The
    generator here is the assertion: if the corpus is read, the test fails.
    """
    read = False

    def documents() -> Iterator[str]:
        nonlocal read
        read = True
        yield from corpus_text()

    with pytest.raises(TokenizerError) as caught:
        train_tokenizer(documents(), vocab_size=vocab_size)

    assert not read, f"vocab_size={vocab_size} read the corpus before refusing"
    assert "no merges" not in str(caught.value).lower()
    assert caught.value.details["minimum"] == MIN_VOCAB_SIZE


def test_the_smallest_accepted_vocab_size_actually_trains() -> None:
    """The floor the error message names has to be a floor that works.

    Both halves matter. The message says ``Use --vocab-size 258 or more``, so 258 must
    train; and it must train on a corpus at the small end, because that is where anyone
    reads this message. The check after training compares against the *filled* count
    and not the floor -- comparing against the floor made a request for exactly the
    floor impossible to satisfy, since it can produce at most that many tokens, and
    reported the corpus as the reason.
    """
    trained = train_tokenizer(["small corpora repeat themselves. " * 30], vocab_size=MIN_VOCAB_SIZE)

    assert trained.vocab_size == MIN_VOCAB_SIZE
    assert trained.vocab_shortfall == 0


def test_the_floor_named_in_the_hint_is_the_floor_that_is_enforced() -> None:
    """Follow the hint literally: the number in it must be accepted."""
    with pytest.raises(TokenizerError) as caught:
        train_tokenizer(corpus_text(), vocab_size=1)

    named = [int(match) for match in re.findall(r"\d+", caught.value.hint or "")]
    assert MIN_VOCAB_SIZE in named, f"hint names {named}, not the floor {MIN_VOCAB_SIZE}"
    followed = train_tokenizer(corpus_text(), vocab_size=MIN_VOCAB_SIZE)
    assert followed.vocab_size >= MIN_VOCAB_SIZE


def test_extra_special_tokens_raise_the_floor_they_consume() -> None:
    """The floor is not the constant; it is the constant plus whatever else is reserved."""
    extras = ["<|pad|>", "<|user|>"]
    with pytest.raises(TokenizerError) as caught:
        train_tokenizer(corpus_text(), vocab_size=MIN_VOCAB_SIZE, extra_special_tokens=extras)

    assert caught.value.details["minimum"] == MIN_VOCAB_SIZE + len(extras)
    assert caught.value.details["filled_by_alphabet"] == BYTE_ALPHABET_SIZE + 1 + len(extras)
    trained = train_tokenizer(
        corpus_text(),
        vocab_size=MIN_VOCAB_SIZE + len(extras),
        extra_special_tokens=extras,
    )
    assert trained.vocab_size == MIN_VOCAB_SIZE + len(extras)


def test_corpus_too_poor_to_learn_merges_is_refused() -> None:
    """A corpus with no repeated pairs cannot fill a vocabulary, and says so."""
    with pytest.raises(TokenizerError) as caught:
        train_tokenizer(["ab"], vocab_size=1024, min_frequency=1000)

    assert "no merges" in str(caught.value).lower()
    assert caught.value.details["min_frequency"] == 1000


def test_the_vocab_size_the_floor_hint_names_actually_trains() -> None:
    """Follow the floor refusal literally, on the corpus that produced it.

    The hint says "Use --vocab-size 258 or more", and the guard below it compares the
    trained size against ``filled`` rather than ``floor`` precisely so that the
    smallest number it names is satisfiable -- otherwise the advice would lead to the
    "learned no merges" error instead, and blame the corpus for the floor. Measured on
    the 1.06 MiB tinyshakespeare corpus: 258 trains and yields exactly 258 tokens.
    """
    with pytest.raises(TokenizerError) as caught:
        train_tokenizer(corpus_text(), vocab_size=MIN_VOCAB_SIZE - 1)

    hint = caught.value.hint or ""
    named = [int(match) for match in re.findall(r"--vocab-size (\d+)", hint)]
    assert named, f"hint names no --vocab-size value: {hint!r}"
    # Satisfiable is not enough: a target far above the floor also trains, so the number
    # in the advice has to be the floor the refusal itself reports.
    assert named == [caught.value.details["minimum"]]
    for value in named:
        followed = train_tokenizer(corpus_text(), vocab_size=value)
        assert followed.vocab_size > BYTE_ALPHABET_SIZE + 1


def test_the_no_merges_hint_does_not_send_the_user_to_the_vocab_size_floor() -> None:
    """``--vocab-size`` is a ceiling on merges, so lowering it can never create one.

    This replaces an assertion that required ``--vocab-size`` to appear in this hint,
    which is why the advice went unquestioned for so long. Measured on the 1.06 MiB
    tinyshakespeare corpus with ``--min-frequency 99999999``: every target from 4096
    down to 258 produced this same error, and 257 was refused by the floor guard with
    "Use --vocab-size 258 or more" -- following the advice was a closed loop between
    two errors.
    """
    with pytest.raises(TokenizerError) as caught:
        train_tokenizer(corpus_text(), vocab_size=4096, min_frequency=10_000_000)

    hint = caught.value.hint or ""
    assert "--vocab-size" not in hint
    # Repetition is what *creates* merges, so naming it as the cause is backwards.
    assert "repetitive" not in hint
    # And the message is reachable on a 1.06 MiB corpus, which the tool itself calls
    # enough text, so it must not be reported as too little.
    assert "too small" not in hint


def test_the_no_merges_hint_names_the_threshold_the_user_actually_set() -> None:
    """The one number that caused it has to appear, grouped the way the CLI prints."""
    with pytest.raises(TokenizerError) as caught:
        train_tokenizer(corpus_text(), vocab_size=4096, min_frequency=99_999_999)

    assert "99,999,999" in (caught.value.hint or "")


def test_the_min_frequency_the_hint_names_actually_trains() -> None:
    """Follow the hint literally, on the corpus that produced it, and it must work."""
    with pytest.raises(TokenizerError) as caught:
        train_tokenizer(corpus_text(), vocab_size=4096, min_frequency=10_000_000)

    named = [int(match) for match in re.findall(r"--min-frequency (\d+)", caught.value.hint or "")]
    assert named, f"hint names no --min-frequency value: {caught.value.hint!r}"
    for value in named:
        followed = train_tokenizer(corpus_text(), vocab_size=4096, min_frequency=value)
        assert followed.vocab_size > BYTE_ALPHABET_SIZE + 1


# One character per document. The GPT-2 pre-tokenizer yields one single-character word
# per document, so no pair of adjacent characters exists at *any* frequency -- which is
# the corpus shape the test below exists for, and the one the sibling test above cannot
# reach with a corpus that merges fine.
NO_PAIRS = [chr(ord("a") + index) for index in range(20)]


def test_the_no_merges_advice_does_not_dead_end_on_a_corpus_with_no_pairs() -> None:
    """Following the hint must either work, or have said what it means when it does not.

    This branch cannot distinguish "no pair occurs min_frequency times" from "no pair
    occurs at all" -- telling them apart needs a second pass over a stream the
    docstring promises to consume once -- and both reach it. So the hint's first
    remedy, ``--min-frequency 1``, does not always help, and the requirement is that it
    says so rather than leaving the user to discover it.

    Measured through ``trainai data prepare`` on 2,000 one-character documents: the
    default threshold printed the ``--min-frequency 1`` hint and exited 3; following it
    printed the same "learned no merges" error with the other hint and exited 3 again.
    """
    with pytest.raises(TokenizerError) as first:
        train_tokenizer(NO_PAIRS, vocab_size=1024)

    hint = first.value.hint or ""
    assert "--min-frequency 1" in hint

    with pytest.raises(TokenizerError) as second:
        train_tokenizer(NO_PAIRS, vocab_size=1024, min_frequency=1)

    # The remedy of the error the advice leads to has to be in the advice already.
    remedy = "more text"
    assert remedy in (second.value.hint or ""), (
        f"this test is pinned to the wrong remedy: {second.value.hint!r}"
    )
    assert remedy in hint, (
        f"the hint sends the user to --min-frequency 1, which on this corpus raises "
        f"{second.value} with hint {second.value.hint!r}, and the hint says nothing "
        f"about that: {hint!r}"
    )


def test_a_corpus_with_nothing_to_merge_is_not_blamed_on_the_threshold() -> None:
    """At ``min_frequency=1`` there is nothing lower to suggest, so the advice changes.

    The old hint said "lower --min-frequency" whatever the value was, including 1,
    where the only lower value is 0 -- and 0 is now refused outright.
    """
    with pytest.raises(TokenizerError) as caught:
        train_tokenizer(["a"], vocab_size=1024, min_frequency=1)

    hint = caught.value.hint or ""
    assert "--min-frequency" not in hint
    assert "more text" in hint


@pytest.mark.parametrize("min_frequency", [0, -1])
def test_min_frequency_below_one_is_refused_before_reading_the_corpus(
    min_frequency: int,
) -> None:
    """0 is not a smaller 1: the Rust trainer clamps it, so the flag became a no-op.

    Measured before the guard, ``--min-frequency 0`` trained identically to 1 on the
    1.06 MiB corpus -- a flag silently doing something other than what its help text
    says. The generator is the assertion: refusing only after the corpus is read would
    make the user wait for a full tokenizer pass to be told about a flag value.
    """
    read = False

    def documents() -> Iterator[str]:
        nonlocal read
        read = True
        yield from corpus_text()

    with pytest.raises(TokenizerError) as caught:
        train_tokenizer(documents(), vocab_size=4096, min_frequency=min_frequency)

    assert not read, f"min_frequency={min_frequency} read the corpus before refusing"
    assert caught.value.details["minimum"] == 1
    assert "--min-frequency 1" in (caught.value.hint or "")


@pytest.mark.parametrize("min_frequency", [1, 2])
def test_the_thresholds_that_already_worked_still_work(min_frequency: int) -> None:
    """Negative control: the new floor must not move any value that was accepted."""
    trained = train_tokenizer(corpus_text(), vocab_size=1024, min_frequency=min_frequency)

    assert trained.vocab_size > BYTE_ALPHABET_SIZE + 1


def test_vocab_shortfall_is_reported_rather_than_hidden() -> None:
    """Asking for more than the corpus supports is allowed, but must be visible."""
    trained = train_tokenizer(["repeat repeat repeat " * 50], vocab_size=4096)

    assert trained.requested_vocab_size == 4096
    assert trained.vocab_size < 4096
    assert trained.vocab_shortfall == 4096 - trained.vocab_size


def test_a_loaded_tokenizer_has_no_requested_size_to_compare_against(
    tokenizer: ByteLevelBPE, tmp_path: Path
) -> None:
    tokenizer.save(tmp_path / "tokenizer.json")

    loaded = ByteLevelBPE.load(tmp_path / "tokenizer.json")

    assert loaded.requested_vocab_size is None
    assert loaded.vocab_shortfall == 0


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def test_training_twice_on_the_same_corpus_gives_the_same_fingerprint() -> None:
    first = train_tokenizer(corpus_text(), vocab_size=VOCAB)
    second = train_tokenizer(corpus_text(), vocab_size=VOCAB)

    assert first.fingerprint() == second.fingerprint()
    assert first.vocab_size == second.vocab_size


def test_a_different_vocab_size_gives_a_different_fingerprint() -> None:
    small = train_tokenizer(corpus_text(), vocab_size=300)
    large = train_tokenizer(corpus_text(), vocab_size=VOCAB)

    assert small.fingerprint() != large.fingerprint()


def test_saving_and_loading_preserves_the_fingerprint(
    tokenizer: ByteLevelBPE, tmp_path: Path
) -> None:
    path = tokenizer.save(tmp_path / "tokenizer.json")

    loaded = ByteLevelBPE.load(path)

    assert loaded.fingerprint() == tokenizer.fingerprint()
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.eot_id == tokenizer.eot_id


def test_saving_and_loading_preserves_token_ids(tokenizer: ByteLevelBPE, tmp_path: Path) -> None:
    text = "Round-tripping through disk must not change one id. " + _NON_ASCII_LINE
    path = tokenizer.save(tmp_path / "tokenizer.json")

    loaded = ByteLevelBPE.load(path)

    assert loaded.encode(text) == tokenizer.encode(text)


def test_loading_a_file_that_is_not_a_tokenizer_says_so(tmp_path: Path) -> None:
    path = tmp_path / "not-a-tokenizer.json"
    path.write_text("{}", encoding="utf-8", newline="\n")

    with pytest.raises(TokenizerError) as caught:
        ByteLevelBPE.load(path)

    assert caught.value.hint


def test_a_tokenizer_with_no_end_of_text_token_is_refused_when_it_is_loaded(
    tmp_path: Path,
) -> None:
    """A real tokenizer file, validly trained, that TrainAI cannot use.

    The end-of-text token is how documents are separated in the shards: without one
    there is no boundary between the last sentence of one document and the first of the
    next, and a model trained on that learns to run them together. Every tokenizer this
    project trains has one, so this is the case the hint describes -- a tokenizer
    supplied from elsewhere, pointed at by ``--tokenizer``.

    Trained here rather than hand-written, because what has to be refused is a file the
    library itself considers valid. Refused in the constructor, so it cannot be reached
    through ``load`` alone: every path that builds one goes through it.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    stranger = Tokenizer(models.BPE())
    stranger.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    stranger.train_from_iterator(
        ["the rivers carry what the clocks measure"] * 8,
        trainers.BpeTrainer(vocab_size=300, special_tokens=[], show_progress=False),
    )
    path = tmp_path / "no-eot.json"
    stranger.save(str(path))
    assert stranger.token_to_id(EOT_TOKEN) is None, "the fixture accidentally trained one"

    with pytest.raises(TokenizerError) as caught:
        ByteLevelBPE.load(path)

    assert EOT_TOKEN in str(caught.value)
    assert "separate documents" in (caught.value.hint or "")
    assert "trainai data prepare" in (caught.value.hint or "")
    assert caught.value.details["vocab_size"] == stranger.get_vocab_size()


def test_loading_a_missing_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(TokenizerError) as caught:
        ByteLevelBPE.load(tmp_path / "absent.json")

    assert "absent.json" in str(caught.value)


# --------------------------------------------------------------------------- #
# Offsets: where each token came from
# --------------------------------------------------------------------------- #
def test_offsets_cover_the_string_a_token_at_a_time(tokenizer: ByteLevelBPE) -> None:
    """Every mask byte in a chat dataset is placed by these ranges.

    ``build_loss_mask`` decides whether a token is a target by asking whether its
    character range overlaps a reply's span, so an offset list that is off by one puts
    the boundary of the trained region on the wrong token. There is no test for the
    offsets themselves: the mask tests all go through ``binarize_documents``, which means
    a shift here and a compensating shift there would agree with each other and with no
    text.

    Checked by slicing the original string with each range and rebuilding it, which is
    the property the mask arithmetic relies on and the one a shift breaks.
    """
    text = "The rivers carry what the clocks measure."

    encoded = tokenizer.encode_batch_with_offsets([text])[0]

    assert len(encoded.ids) == len(encoded.offsets)
    assert encoded.ids == tokenizer.encode(text), "the ids are the ordinary ones"
    for start, end in encoded.offsets:
        assert 0 <= start <= end <= len(text)
    assert encoded.offsets[0][0] == 0
    assert encoded.offsets[-1][1] == len(text)
    # Rebuilt from the ranges rather than compared range by range: the GPT-2
    # pre-tokenizer's `trim_offsets` leaves the space before a word outside that word's
    # range, so consecutive ranges are not required to touch -- but together they still
    # have to account for every character.
    rebuilt = "".join(text[start:end] for start, end in encoded.offsets)
    assert rebuilt.replace(" ", "") == text.replace(" ", "")


def test_a_token_without_a_position_is_refused_rather_than_shifting_the_rest(
    tokenizer: ByteLevelBPE,
) -> None:
    """``Encoded`` is two parallel lists, and a length mismatch is silent damage.

    Zipping ids with offsets stops at the shorter one, so a missing offset does not
    raise -- it drops the tail of the document from the mask and shifts nothing visibly.
    The check is in ``__post_init__`` because that is the only place both lists are known
    before anyone indexes them.

    Constructed directly: no encode produces this, which is why the guard exists at the
    boundary rather than as a comment about what the Rust layer promises.
    """
    from trainai.data.tokenizer import Encoded

    intact = Encoded(ids=[1, 2], offsets=[(0, 1), (1, 2)])
    assert intact.ids == [1, 2], "the well-formed case is accepted"

    with pytest.raises(ValueError) as caught:
        Encoded(ids=[1, 2, 3], offsets=[(0, 1), (1, 2)])

    assert "3 ids but 2 offsets" in str(caught.value)
    assert "shift every position after it" in str(caught.value)


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def test_measure_reports_real_compression(tokenizer: ByteLevelBPE) -> None:
    report = tokenizer.measure(corpus_text(), max_chars=1 << 20)

    assert report.sample_documents == 3
    assert report.sample_tokens > 0
    assert report.chars_per_token > 1.0
    assert report.bytes_per_token >= report.chars_per_token


def test_measure_honours_its_character_budget(tokenizer: ByteLevelBPE) -> None:
    report = tokenizer.measure(corpus_text(), max_chars=100)

    assert report.sample_documents == 1
    assert report.sample_chars == len(corpus_text()[0])


def test_measure_of_nothing_does_not_divide_by_zero(tokenizer: ByteLevelBPE) -> None:
    report = tokenizer.measure([])

    assert report.sample_tokens == 0
    assert report.chars_per_token == 0.0
    assert report.bytes_per_token == 0.0


def test_to_dict_carries_what_the_manifest_needs(tokenizer: ByteLevelBPE) -> None:
    payload = tokenizer.to_dict()

    assert payload["vocab_size"] == tokenizer.vocab_size
    assert payload["eot_id"] == tokenizer.eot_id
    assert payload["eot_token"] == EOT_TOKEN
    assert payload["fingerprint"] == tokenizer.fingerprint()


# --------------------------------------------------------------------------- #
# What reaches the trainer
# --------------------------------------------------------------------------- #
#
# ``train_from_iterator`` is Rust, and the generator feeding it resumes under a Rust
# caller -- which is why a coverage trace of the whole suite reports ``_non_empty``
# as never having run even though every tokenizer test drives it. Never-run in the
# trace, not untested by choice: these three tests call it directly.


def test_empty_documents_never_reach_the_trainer() -> None:
    """An empty document is a word boundary the trainer learns nothing from.

    The BPE trainer counts each document it is handed, so empty strings inflate the
    denominator of every frequency it computes without contributing a single pair to
    merge. A corpus assembled from a directory of files, or from a JSONL export with
    blank ``text`` fields, arrives full of them.
    """
    from trainai.data.tokenizer import _non_empty

    kept = list(_non_empty(["first", "", "second", "", "", "third"]))

    assert kept == ["first", "second", "third"]


def test_a_whitespace_only_document_is_not_empty_and_is_kept() -> None:
    """The filter's exact boundary, pinned because moving it changes a fingerprint.

    ``if text:`` keeps ``" "`` and ``"\\n"``; ``if text.strip():`` would not. The
    tempting tidy-up is not a refactor: whitespace-only documents contribute
    whitespace pairs, so dropping them trains a different vocabulary, which is a
    different :meth:`ByteLevelBPE.fingerprint`, which no longer matches the
    checkpoints written against the old one. Changing this is allowed; changing it by
    accident is not.
    """
    from trainai.data.tokenizer import _non_empty

    assert list(_non_empty([" ", "\n", "\t"])) == [" ", "\n", "\t"]


def test_the_filter_pulls_one_document_at_a_time() -> None:
    """Why it is a generator and not a list comprehension.

    ``train_tokenizer`` documents that it consumes its corpus once, as a stream, and
    that promise is what lets a corpus larger than memory be tokenized at all. A
    filter that materialised its input would keep that signature and that docstring
    while quietly loading the whole corpus into RAM -- and every existing test would
    still pass, because they all use corpora small enough to fit.

    So the source records what it has been asked for. Yielding ``"second"`` requires
    pulling the empty string in front of it and discarding it, which is the minimum;
    what must not appear is ``"third"``.
    """
    from trainai.data.tokenizer import _non_empty

    produced: list[str] = []

    def source() -> Iterator[str]:
        for text in ("first", "", "second", "third"):
            produced.append(text)
            yield text

    stream = _non_empty(source())

    assert next(stream) == "first"
    assert produced == ["first"], f"the filter read ahead: {produced}"
    assert next(stream) == "second"
    assert produced == ["first", "", "second"], f"the filter read ahead: {produced}"
