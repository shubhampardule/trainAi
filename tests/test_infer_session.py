"""Tests for :mod:`trainai.infer.session`.

The test that carries this module is
``test_a_tokenizer_from_a_different_dataset_is_refused``. Every other failure here is
loud -- a missing path, a missing checkpoint, an empty directory. A mismatched
tokenizer is the one that is *silent*: the ids are all in range, the model runs, and
fluent text comes out that has nothing to do with what the model computed. Nothing
downstream can catch it, so it is caught here.

The rest of the file exists because these paths are what a user hits on their second
day, not their first: the dataset directory has been deleted, or moved, or the run was
copied to another machine without it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from trainai.data.binarize import TOKENIZER_NAME
from trainai.data.tokenizer import train_tokenizer
from trainai.errors import UsageError
from trainai.infer import InferenceSession, locate_run
from trainai.model.config import ModelConfig
from trainai.train.config import TrainConfig
from trainai.train.loop import Trainer


def train_a_run(dataset: Any, run_dir: Path, **overrides: Any) -> Path:
    """A real short run on the shared fixture dataset, with a checkpoint on disk."""
    settings: dict[str, Any] = {
        "steps": 4,
        "batch_size": 4,
        "seq_len": 64,
        "lr": 1e-3,
        "warmup_steps": 1,
        "eval_every": 2,
        "eval_batches": 2,
        "checkpoint_every": 2,
        "log_every": 4,
        "seed": 7,
        "device": "cpu",
    }
    settings.update(overrides)
    Trainer(
        dataset=dataset,
        model_config=ModelConfig(
            vocab_size=dataset.vocab_size,
            n_layer=2,
            n_head=4,
            d_model=64,
            seq_len=64,
        ),
        train_config=TrainConfig(**settings),
        run_dir=run_dir,
        quiet=True,
    ).run()
    return run_dir


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory: pytest.TempPathFactory, prepared_dataset: Any) -> Path:
    """One trained run, reused. Every test here only reads it."""
    return train_a_run(prepared_dataset, tmp_path_factory.mktemp("infer") / "run")


# --------------------------------------------------------------------------- #
# Finding the pieces
# --------------------------------------------------------------------------- #
def test_a_run_directory_resolves_to_a_checkpoint_and_a_tokenizer(trained_run: Path) -> None:
    layout = locate_run(trained_run)

    assert layout.checkpoint_path.is_file()
    assert layout.tokenizer_path == trained_run / TOKENIZER_NAME
    assert layout.tokenizer_source == "the run directory"
    assert layout.run_dir == trained_run
    assert layout.checkpoints_available >= 1


def test_a_checkpoints_directory_resolves_too(trained_run: Path) -> None:
    """Tab completion lands people inside checkpoints/; that is not a mistake."""
    layout = locate_run(trained_run / "checkpoints")

    assert layout.checkpoint_path.parent == trained_run / "checkpoints"
    assert layout.run_dir == trained_run
    assert layout.tokenizer_path == trained_run / TOKENIZER_NAME


def test_a_single_checkpoint_file_resolves_and_says_it_was_explicit(trained_run: Path) -> None:
    one = sorted((trained_run / "checkpoints").glob("step-*.pt"))[0]

    layout = locate_run(one)

    assert layout.checkpoint_path == one
    assert layout.which == "explicit", "reporting 'best' for a file the user named is a lie"
    assert layout.run_dir == trained_run


def test_best_and_latest_are_both_reachable(trained_run: Path) -> None:
    best = locate_run(trained_run, which="best")
    latest = locate_run(trained_run, which="latest")

    assert best.which == "best"
    assert latest.which == "latest"
    assert best.checkpoint_path.is_file()
    assert latest.checkpoint_path.is_file()


def test_the_default_is_best_not_latest(trained_run: Path) -> None:
    """On a small corpus the last checkpoint is routinely worse than the best one."""
    assert locate_run(trained_run).which == "best"


def test_a_good_run_carries_no_note(trained_run: Path) -> None:
    """The negative control for the two tests below: a note is an exception report, so
    one appearing on an ordinary run would be noise on every load."""
    assert locate_run(trained_run, which="best").note is None
    assert locate_run(trained_run, which="latest").note is None
    assert locate_run(trained_run).to_dict()["note"] is None


@pytest.mark.parametrize(
    ("damage", "named"),
    [
        pytest.param(b"{not json", "checkpoints.json", id="the pointer is not json"),
        pytest.param(b'{"best": 3}\n', "checkpoints.json", id="best is not an object"),
        pytest.param(None, "step-", id="the file it names is gone"),
    ],
)
def test_best_falling_back_to_the_last_checkpoint_says_so(
    trained_run: Path, tmp_path: Path, damage: bytes | None, named: str
) -> None:
    """The pointer file is the only record of which checkpoint had the lowest loss.

    Without it, ``--which best`` gets the *last* checkpoint -- a different checkpoint,
    from a fallback that used to be silent. Someone comparing two runs that way would
    conclude their best model is worse than it is, and nothing in the output would
    disagree.
    """
    copy = tmp_path / "copy"
    shutil.copytree(trained_run, copy)
    pointer = copy / "checkpoints" / "checkpoints.json"
    if damage is None:
        raw = json.loads(pointer.read_text(encoding="utf-8"))
        (copy / "checkpoints" / raw["best"]["file"]).unlink()
    else:
        pointer.write_bytes(damage)

    layout = locate_run(copy, which="best")

    assert layout.checkpoint_path.is_file(), "the fallback still has to produce a checkpoint"
    assert layout.note is not None
    assert named in layout.note
    assert "lowest validation loss" in layout.note


def test_latest_falling_back_says_nothing(trained_run: Path, tmp_path: Path) -> None:
    """The filenames *are* the record of which checkpoint is the latest, so the fallback
    answers with the same file the pointer would have named. Nothing to report."""
    copy = tmp_path / "copy"
    shutil.copytree(trained_run, copy)
    (copy / "checkpoints" / "checkpoints.json").write_bytes(b"{not json")

    assert locate_run(copy, which="latest").note is None


def test_an_unknown_which_is_refused_before_anything_is_read(trained_run: Path) -> None:
    with pytest.raises(UsageError) as caught:
        locate_run(trained_run, which="worst")

    assert "best" in str(caught.value)
    assert caught.value.details["choices"] == ["best", "latest"]


def test_a_path_that_does_not_exist_says_so(tmp_path: Path) -> None:
    with pytest.raises(UsageError) as caught:
        locate_run(tmp_path / "nowhere")

    assert "Nothing at" in str(caught.value)
    assert "trainai train" in (caught.value.hint or "")


def test_a_directory_with_no_checkpoints_names_where_it_looked(tmp_path: Path) -> None:
    empty = tmp_path / "run"
    empty.mkdir()

    with pytest.raises(UsageError) as caught:
        locate_run(empty)

    assert "No checkpoints" in str(caught.value)
    assert "--checkpoint-every" in (caught.value.hint or "")
    assert str(empty / "checkpoints") in caught.value.details["looked_in"]


def test_a_run_without_its_own_tokenizer_falls_back_to_the_dataset(
    trained_run: Path, prepared_dataset: Any, tmp_path: Path
) -> None:
    """Runs from before the trainer copied the tokenizer must still be usable."""
    older = tmp_path / "older"
    shutil.copytree(trained_run, older)
    (older / TOKENIZER_NAME).unlink()

    layout = locate_run(older)

    assert layout.tokenizer_path == Path(prepared_dataset.root) / TOKENIZER_NAME
    assert "the dataset at" in layout.tokenizer_source


def test_no_tokenizer_anywhere_lists_what_it_tried(
    trained_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orphan = tmp_path / "orphan"
    shutil.copytree(trained_run, orphan)
    (orphan / TOKENIZER_NAME).unlink()
    # Point the recorded dataset at somewhere that does not exist, which is what
    # deleting the dataset directory looks like from here.
    monkeypatch.setattr(
        "trainai.infer.session.load_checkpoint",
        lambda path, **kw: _with_dataset_root(path, str(tmp_path / "deleted"), **kw),
    )

    with pytest.raises(UsageError) as caught:
        locate_run(orphan)

    assert "cannot be turned into text" in str(caught.value)
    assert "--tokenizer" in (caught.value.hint or "")
    assert len(caught.value.details["searched"]) == 2


@pytest.mark.parametrize(
    ("state", "reason", "phrase"),
    [
        ("no root recorded", "no_dataset_recorded", "records no dataset"),
        ("the directory is there", "dataset_has_no_tokenizer", "has no tokenizer.json either"),
        ("recorded relative", "dataset_not_found_from_here", "a relative path"),
        ("recorded absolute, gone", "dataset_missing", "is gone or moved"),
    ],
)
def test_no_tokenizer_names_which_of_the_four_states_this_is(
    trained_run: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    reason: str,
    phrase: str,
) -> None:
    """One hint told one story, and it was wrong for three of the four ways here.

    It read "Runs trained by a newer version keep their own copy; this one does not,
    and the dataset it names is gone or moved". Measured against a run this build had
    trained a minute earlier, against a dataset still sitting on disk: there is no
    newer version, this build *is* the one that copies the tokenizer in, and the
    dataset was neither gone nor moved -- it had simply never had a tokenizer.json
    beside it, which `trainai train` reported at the time and this hint did not repeat.

    The relative case is the subtle one and the reason the branch checks
    ``is_absolute`` rather than just ``is_dir``. A checkpoint records the dataset path
    as it was typed, so running `trainai chat` from a different directory leaves a
    perfectly present dataset unresolvable -- and calling that "gone or moved" would be
    the same false claim wearing a new costume.
    """
    orphan = tmp_path / f"orphan-{reason}"
    shutil.copytree(trained_run, orphan)
    (orphan / TOKENIZER_NAME).unlink()

    roots = {
        "no root recorded": None,
        # A real directory that has no tokenizer.json in it, which is what a dataset
        # prepared from a corpus with no tokenizer beside it looks like.
        "the directory is there": str(tmp_path),
        "recorded relative": "runs/a-dataset-that-is-not-here",
        "recorded absolute, gone": str(tmp_path / "deleted"),
    }
    monkeypatch.setattr(
        "trainai.infer.session.load_checkpoint",
        lambda path, **kw: _with_dataset_root(path, roots[state], **kw),
    )

    with pytest.raises(UsageError) as caught:
        locate_run(orphan)

    hint = caught.value.hint or ""
    assert caught.value.details["reason"] == reason
    assert phrase in hint, hint
    assert "--tokenizer" in hint
    assert "newer version" not in hint, (
        "no version of TrainAI keeps a copy the running one does not; this build is "
        "the one that copies"
    )


def _with_dataset_root(path: Path, root: str, **kwargs: Any) -> Any:
    from dataclasses import replace

    from trainai.train.checkpoint import load_checkpoint as real_load

    loaded = real_load(path, **kwargs)
    return replace(loaded, dataset={**loaded.dataset, "root": root})


# --------------------------------------------------------------------------- #
# Opening a session
# --------------------------------------------------------------------------- #
def test_opening_a_run_gives_a_model_in_eval_mode_with_its_tokenizer(
    trained_run: Path, prepared_dataset: Any
) -> None:
    session = InferenceSession.open(trained_run, device="cpu")

    assert not session.model.training, "dropout active during inference is a silent bug"
    assert session.tokenizer.vocab_size == prepared_dataset.vocab_size
    assert session.model_config.vocab_size == prepared_dataset.vocab_size
    assert session.step > 0
    assert isinstance(session.train_config, TrainConfig)


def test_the_session_reports_what_it_loaded_and_how(trained_run: Path) -> None:
    """A sample that cannot be reproduced is an anecdote; the report is what fixes it."""
    described = InferenceSession.open(trained_run, device="cpu", precision="fp32").to_dict()

    assert described["device"] == "cpu"
    assert "fp32" in described["precision"]
    assert described["parameters"] > 0
    assert described["context"] == 64
    assert described["layout"]["tokenizer_source"] == "the run directory"
    json.dumps(described)  # the report has to survive --json


def test_best_val_loss_is_named_for_what_the_checkpoint_actually_stores(
    trained_run: Path,
) -> None:
    """The trainer records ``best_val_loss``, not this checkpoint's own loss.

    Calling the property ``val_loss`` would invite reading a ``latest`` checkpoint's
    number as a measurement of that checkpoint. It is also how this first went wrong:
    the property read a ``val_loss`` key that nothing ever writes, so it returned
    ``None`` for every run that had validation data.
    """
    session = InferenceSession.open(trained_run, device="cpu")

    assert session.best_val_loss is not None
    assert session.best_val_loss > 0
    assert session.best_val_step is not None


def test_a_tokenizer_override_is_accepted_and_reported(
    trained_run: Path, prepared_dataset: Any
) -> None:
    session = InferenceSession.open(
        trained_run,
        device="cpu",
        tokenizer=Path(prepared_dataset.root) / TOKENIZER_NAME,
    )

    assert session.layout.tokenizer_source == "--tokenizer"


def test_a_tokenizer_override_pointing_at_nothing_says_so(
    trained_run: Path, tmp_path: Path
) -> None:
    with pytest.raises(UsageError) as caught:
        InferenceSession.open(trained_run, device="cpu", tokenizer=tmp_path / "nope.json")

    assert "No tokenizer file at" in str(caught.value)


def test_a_tokenizer_from_a_different_dataset_is_refused(trained_run: Path, tmp_path: Path) -> None:
    """The failure this whole module is arranged around.

    Token id 412 means one string under the tokenizer the model was trained with and a
    different one under any other. A mismatch does not degrade the output, it replaces
    it with confident nonsense, and no check downstream can notice. So it is refused
    rather than warned about.
    """
    other = train_tokenizer(
        [f"Entirely different text {i} about kettles and gantries." for i in range(50)],
        vocab_size=300,
    )
    elsewhere = tmp_path / "other-tokenizer.json"
    other.save(elsewhere)

    with pytest.raises(UsageError) as caught:
        InferenceSession.open(trained_run, device="cpu", tokenizer=elsewhere)

    assert "not the one the model was trained with" in str(caught.value)
    assert caught.value.details["expected_fingerprint"] != caught.value.details["found_fingerprint"]


def test_a_tokenizer_of_the_wrong_size_is_refused_even_without_a_fingerprint(
    trained_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint with no recorded fingerprint still cannot take any tokenizer.

    The vocabulary check is the backstop: ids beyond the output layer would index out
    of bounds, and ids inside a smaller one would silently mean something else.
    """
    other = train_tokenizer(
        [f"Different text {i} about kettles." for i in range(50)], vocab_size=300
    )
    elsewhere = tmp_path / "small.json"
    other.save(elsewhere)
    monkeypatch.setattr(
        "trainai.infer.session.load_checkpoint",
        lambda path, **kw: _without_fingerprint(path, **kw),
    )

    with pytest.raises(UsageError) as caught:
        InferenceSession.open(trained_run, device="cpu", tokenizer=elsewhere)

    assert "output layer" in str(caught.value)
    assert caught.value.details["tokenizer_vocab_size"] == other.vocab_size


def _without_fingerprint(path: Path, **kwargs: Any) -> Any:
    from dataclasses import replace

    from trainai.train.checkpoint import load_checkpoint as real_load

    loaded = real_load(path, **kwargs)
    dataset = {k: v for k, v in loaded.dataset.items() if k != "tokenizer_fingerprint"}
    return replace(loaded, dataset=dataset)


# --------------------------------------------------------------------------- #
# Generating
# --------------------------------------------------------------------------- #
def test_complete_returns_only_the_new_text(trained_run: Path) -> None:
    """Returning prompt-plus-continuation makes every caller slice the prompt off."""
    session = InferenceSession.open(trained_run, device="cpu")

    out = session.complete("The bridges", max_new_tokens=8, temperature=0.0)

    assert isinstance(out, str)
    assert not out.startswith("The bridges")


def test_the_same_seed_gives_the_same_text_and_a_different_seed_does_not(
    trained_run: Path,
) -> None:
    session = InferenceSession.open(trained_run, device="cpu")
    kwargs: dict[str, Any] = {"max_new_tokens": 12, "temperature": 1.0}

    assert session.complete("The", seed=5, **kwargs) == session.complete("The", seed=5, **kwargs)
    assert session.complete("The", seed=5, **kwargs) != session.complete("The", seed=6, **kwargs)


def test_streaming_and_complete_produce_the_same_string(trained_run: Path) -> None:
    """The playground and the batch path must not drift; they share one loop."""
    session = InferenceSession.open(trained_run, device="cpu")
    kwargs: dict[str, Any] = {"max_new_tokens": 16, "temperature": 1.0, "seed": 3}

    assert "".join(session.stream("The", **kwargs)) == session.complete("The", **kwargs)


def test_streaming_emits_more_than_one_piece(trained_run: Path) -> None:
    """A "stream" that yields everything at the end is not one."""
    session = InferenceSession.open(trained_run, device="cpu")

    pieces = list(session.stream("The", max_new_tokens=16, temperature=1.0, seed=3))

    assert len(pieces) > 1
    assert all(isinstance(piece, str) for piece in pieces)


def test_streaming_never_emits_a_replacement_character_mid_stream(
    trained_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Byte-level BPE can split a multi-byte character across two tokens.

    Decoding each token alone would emit U+FFFD and then contradict it. The stream
    decodes the whole continuation each step and yields the delta, so a partial
    character is held back until it completes. This drives that with a decoder that
    reports an incomplete character on the first token.
    """
    from trainai.infer.session import REPLACEMENT_CHAR

    session = InferenceSession.open(trained_run, device="cpu")
    real_decode = session.tokenizer.decode
    calls: list[int] = []

    def flaky_decode(ids: Any) -> str:
        ids = list(ids)
        calls.append(len(ids))
        text = real_decode(ids)
        return text + REPLACEMENT_CHAR if len(ids) == 1 else text

    monkeypatch.setattr(session.tokenizer, "decode", flaky_decode)
    pieces = list(session.stream("The", max_new_tokens=4, temperature=0.0))

    assert calls, "the stream must decode as it goes, not once at the end"
    assert REPLACEMENT_CHAR not in "".join(pieces)


def test_stream_pieces_counts_tokens_that_printed_nothing(
    trained_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A held-back token produced no text and was still produced.

    ``stream`` yields string deltas, so counting them undercounts tokens whenever a
    multi-byte character spans two of them -- which would make a reported tokens/s
    lower than the machine's real rate. ``stream_pieces`` carries the count, and this
    drives it with a decoder that reports an incomplete character on the first token.
    """
    from trainai.infer.session import REPLACEMENT_CHAR

    session = InferenceSession.open(trained_run, device="cpu")
    real_decode = session.tokenizer.decode

    def flaky_decode(ids: Any) -> str:
        ids = list(ids)
        text = real_decode(ids)
        return text + REPLACEMENT_CHAR if len(ids) == 1 else text

    monkeypatch.setattr(session.tokenizer, "decode", flaky_decode)
    pieces = list(
        session.stream_pieces("The", max_new_tokens=6, temperature=0.0, stop_at_eot=False)
    )

    assert pieces[0].text == "", "an incomplete character is held back"
    assert pieces[0].tokens == 1, "and the token it came from is still counted"
    assert pieces[-1].tokens == 6
    assert [p.tokens for p in pieces] == sorted(p.tokens for p in pieces), "cumulative"


def test_stream_and_stream_pieces_agree_on_the_text(trained_run: Path) -> None:
    """``stream`` is the text-only view of the same walk, so it cannot drift."""
    session = InferenceSession.open(trained_run, device="cpu")
    kwargs: dict[str, Any] = {"max_new_tokens": 12, "temperature": 1.0, "seed": 7}

    deltas = list(session.stream("The", **kwargs))
    pieces = list(session.stream_pieces("The", **kwargs))

    assert "".join(piece.text for piece in pieces) == "".join(deltas)
    assert [piece.text for piece in pieces if piece.text] == deltas


def test_an_empty_prompt_generates_rather_than_crashing(trained_run: Path) -> None:
    """The model needs something to attend to; end-of-text is "start of something"."""
    session = InferenceSession.open(trained_run, device="cpu")

    ids = session.encode_prompt("")

    assert ids.shape == (1, 1)
    assert int(ids[0, 0]) == session.tokenizer.eot_id
    assert isinstance(session.complete("", max_new_tokens=4, temperature=0.0), str)


def test_generation_builds_no_autograd_graph(trained_run: Path) -> None:
    """A long chat that retains graphs dies of memory, slowly and confusingly."""
    session = InferenceSession.open(trained_run, device="cpu")

    with torch.enable_grad():
        session.complete("The", max_new_tokens=6, temperature=0.0)

    assert all(param.grad is None for param in session.model.parameters())
