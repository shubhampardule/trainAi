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
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from trainai.data.binarize import TOKENIZER_NAME
from trainai.data.tokenizer import train_tokenizer
from trainai.errors import UsageError
from trainai.infer import Finish, InferenceSession, StreamPiece, locate_run
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

    assert "worst" in str(caught.value), "the rejected value has to appear, to be searchable"
    assert "best, latest" in (caught.value.hint or "")
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


def test_a_checkpoint_that_will_not_load_is_not_reported_as_recording_no_dataset(
    trained_run: Path, tmp_path: Path
) -> None:
    """The fifth way here, which used to wear the first one's story.

    Looking for a tokenizer means reading the dataset path out of the checkpoint, and
    that read is allowed to fail without taking the tokenizer report down with it -- a
    traceback from inside ``locate_run`` would say nothing about either problem. But
    "this run records no dataset" is a claim about a file nobody could read. A truncated
    checkpoint almost certainly *does* record one, and the user sent hunting for a
    tokenizer to pass finds their `--tokenizer` refused a second time, by the same
    unreadable file, one round trip later than they could have known.

    So it gets its own reason and says what will not help. `load_checkpoint` diagnoses
    the file properly -- "truncated or damaged, most likely from an interrupted copy" --
    the moment anything actually tries to use it; this only has to stop guessing.

    ``test_no_tokenizer_names_which_of_the_four_states_this_is`` above covers the four
    that are genuinely about the dataset, and its ``no_dataset_recorded`` row is the one
    this used to be indistinguishable from.
    """
    broken = tmp_path / "broken"
    shutil.copytree(trained_run, broken)
    (broken / TOKENIZER_NAME).unlink()
    damaged = sorted(broken.rglob("*.pt"))
    assert damaged, "the fixture run is supposed to have checkpoints on disk"
    for checkpoint in damaged:
        checkpoint.write_bytes(b"an interrupted copy of a real checkpoint")

    with pytest.raises(UsageError) as caught:
        locate_run(broken)

    hint = caught.value.hint or ""
    resolved = Path(caught.value.details["checkpoint"])
    assert caught.value.details["reason"] == "checkpoint_unreadable"
    assert resolved.name in hint, "the file to look at has to be named"
    assert "could not be read" in hint
    assert "--tokenizer will not get this run working" in hint
    assert "records no dataset" not in hint, (
        "a checkpoint nobody could read is not a checkpoint that recorded nothing"
    )


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
    assert described["chat_template"] == {}, "a prose run must not claim a chat layout"
    json.dumps(described)  # the report has to survive --json


def test_a_gpu_session_autocasts_in_the_precision_its_report_claims(
    trained_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The half of ``autocast`` that a CPU session never reaches -- and got wrong.

    ``autocast`` is public because evaluation runs its own forward passes and has to run
    them under the session's precision: a loss measured in fp32 while the report says
    bf16 is a wrong number with a correct-looking label. Every other test in this file
    opens a CPU session, which takes the disabled branch, so the context that actually
    turns autocast *on* had never been built -- and it hardcoded bf16 regardless of what
    the session resolved. ``precision_for`` in ``trainai/train/loop.py`` answers ``auto``
    with fp16 on every CUDA device that predates bf16, and those devices *reject* a bf16
    context rather than downgrading, so ``infer`` and ``eval`` raised on exactly the
    hardware that fallback exists for.

    Recording the call instead of entering a real context, because which dtype torch
    accepts depends on the card in the machine, and this claim is about what the session
    asks for, not what a GPU happens to allow: on a bf16-capable box a hardcoded bf16
    passes every real-context assertion. The CPU rows below do use real torch, since a
    CPU session is the one case every machine has (compare
    ``test_metal_gets_fp32_because_this_project_has_never_verified_mps_autocast`` in
    ``tests/test_train_loop.py``, which pins the resolver end of the same decision).
    """
    cpu_session = InferenceSession.open(trained_run, device="cpu", precision="fp32")
    with cpu_session.autocast():
        assert not torch.is_autocast_enabled("cpu"), "fp32 on CPU must autocast nothing"

    asked: list[dict[str, Any]] = []

    def record(**kwargs: Any) -> Any:
        asked.append(kwargs)
        return nullcontext()

    monkeypatch.setattr(torch, "autocast", record)

    # Half precision off the CPU: the session's own dtype, on the session's own device.
    for device_type, dtype in (
        ("cuda", torch.bfloat16),
        ("cuda", torch.float16),
        ("xpu", torch.float16),
    ):
        asked.clear()
        session = replace(cpu_session, device=torch.device(device_type), dtype=dtype)

        with session.autocast():
            pass

        assert asked == [{"device_type": device_type, "dtype": dtype}], (
            f"a session reporting {dtype} on {device_type} has to autocast in {dtype}"
        )

    # fp32 is refused the enabled context even off the CPU: autocasting to fp32 is a
    # contradiction, and the caller asked for full precision.
    for device_type in ("cuda", "cpu"):
        asked.clear()
        full = replace(cpu_session, device=torch.device(device_type), dtype=torch.float32)

        with full.autocast():
            pass

        assert asked == [{"device_type": device_type, "enabled": False}]

    # ...and so is a CPU session that asked for half precision, which is the other way
    # into the disabled branch.
    asked.clear()
    half_on_cpu = replace(cpu_session, device=torch.device("cpu"), dtype=torch.bfloat16)
    with half_on_cpu.autocast():
        pass
    assert asked == [{"device_type": "cpu", "enabled": False}]


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
# The chat template
# --------------------------------------------------------------------------- #
# The prose control for these lives in ``test_the_session_reports_what_it_loaded_and_how``
# above, which asserts an empty template on the shared prose run -- a session that
# reported a chat layout for every run would wrap plain text in role labels the model
# never saw, and that is the failure worth a control rather than a repeat.
def test_the_chat_template_survives_the_dataset_being_deleted(
    masked_dataset: Any, tmp_path: Path
) -> None:
    """The reason the checkpoint carries the layout at all, stated as a test.

    Datasets are the large thing people delete once a run is done, and a model trained
    on ``User: ...`` / ``Assistant: `` that is handed a bare question continues the
    question instead of answering it -- which reads as a bad model, not as a format
    mismatch. Trained against a *copy* of the fixture, which is then removed: the shared
    dataset is read-only and every other test in this file needs it.
    """
    from trainai.data.binarize import DatasetManifest

    assert masked_dataset.root is not None
    data = tmp_path / "data"
    shutil.copytree(masked_dataset.root, data)
    run = train_a_run(DatasetManifest.load(data), tmp_path / "run", seq_len=32)
    shutil.rmtree(data)

    session = InferenceSession.open(run, device="cpu")

    assert not data.exists(), "otherwise this test proves nothing about a deleted dataset"
    assert session.chat_template == masked_dataset.chat
    assert session.chat_template["version"] == 1
    described = session.to_dict()
    assert described["chat_template"]["trained_roles"] == ["assistant"]
    json.dumps(described)  # the report has to survive --json


def test_the_reported_template_is_a_copy(
    trained_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller that edits what it was handed must not edit the loaded checkpoint.

    Shallow, which is all the top level needs: nothing reaches into ``labels`` to change
    a role name, and a deep copy on every read would be a promise this cannot keep once
    the block is nested further. Injected rather than trained, because what is under test
    is one ``dict()`` call and a second real run would only make it slower to find.
    """
    monkeypatch.setattr(
        "trainai.infer.session.load_checkpoint",
        lambda path, **kw: _with_chat(path, {"version": 1}, **kw),
    )
    session = InferenceSession.open(trained_run, device="cpu")

    session.chat_template.pop("version")

    assert session.chat_template["version"] == 1


def test_a_chat_block_that_is_not_an_object_reports_no_template(
    trained_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hand-edited or third-party checkpoint must not take inference down with it.

    There is nothing useful to do with a ``chat`` block that is not an object, and the
    alternative to reporting "no template" is an AttributeError raised from inside a
    generate call, long after the load that could have explained it.
    """
    monkeypatch.setattr(
        "trainai.infer.session.load_checkpoint",
        lambda path, **kw: _with_chat(path, ["User", "Assistant"], **kw),
    )

    session = InferenceSession.open(trained_run, device="cpu")

    assert session.chat_template == {}
    assert session.to_dict()["chat_template"] == {}


def _with_chat(path: Path, block: Any, **kwargs: Any) -> Any:
    from dataclasses import replace

    from trainai.train.checkpoint import load_checkpoint as real_load

    loaded = real_load(path, **kwargs)
    return replace(loaded, dataset={**loaded.dataset, "chat": block})


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


# --------------------------------------------------------------------------- #
# Stopping at text
# --------------------------------------------------------------------------- #
# A tiny model will not reliably produce any particular string, so what the model
# writes is scripted here and the *cutting* is what is under test. One character per
# token, which is the hard case: every stop string arrives a fraction at a time, so a
# stream that emits the fraction has already shown the user half a label.
def scripted(session: Any, monkeypatch: pytest.MonkeyPatch, script: str) -> None:
    monkeypatch.setattr(session.tokenizer, "decode", lambda ids: script[: len(list(ids))])


def chunked(session: Any, monkeypatch: pytest.MonkeyPatch, chunks: list[str]) -> None:
    """The other hard case: a real token is several characters, not one.

    One step can therefore reveal two stop strings at once, and which of them is cut at
    stops being a question about time and becomes a question about position.
    """
    monkeypatch.setattr(session.tokenizer, "decode", lambda ids: "".join(chunks[: len(list(ids))]))


def test_a_stop_string_ends_the_generation_and_is_not_in_the_text(
    trained_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stop string arrives one character per token, so it is not a token either."""
    session = InferenceSession.open(trained_run, device="cpu")
    scripted(session, monkeypatch, "Hey!\nUser: and another thing")

    out = session.complete("x", max_new_tokens=40, temperature=0.0, stop=("\nUser:",))

    assert out == "Hey!"


def test_the_earliest_stop_string_wins(trained_run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Not the first one listed, not the longest, and not the one that completes first.

    Scripted several characters per token on purpose. With one character per token the
    two stop strings can never arrive in the same step, so cutting at the *last* match
    would pass every other test in this section -- and show the user the text between
    them. A real tokenizer's tokens are several characters, so the step that reveals one
    label can reveal the next as well.
    """
    session = InferenceSession.open(trained_run, device="cpu")
    chunked(session, monkeypatch, ["one ", "two three"])

    out = session.complete("x", max_new_tokens=40, temperature=0.0, stop=("three", "two"))

    assert out == "one "


def test_a_partial_stop_string_is_never_emitted(
    trained_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The test this section exists for: text on a terminal cannot be taken back.

    Checked over every prefix of the stream rather than the final string, because the
    final string is right even in the implementation that shows ``"\\nUse"`` and then
    stops -- the damage is on screen, not in the return value.
    """
    session = InferenceSession.open(trained_run, device="cpu")
    scripted(session, monkeypatch, "Hey!\nUser: hi")
    stop = "\nUser:"

    shown = ""
    for piece in session.stream("x", max_new_tokens=40, temperature=0.0, stop=(stop,)):
        shown += piece
        for size in range(1, len(stop) + 1):
            assert not shown.endswith(stop[:size]), f"showed {shown!r}"
    assert shown == "Hey!"


@pytest.mark.parametrize(
    "script",
    [
        pytest.param("Hey!\nUse it well", id="resolved-mid-stream"),
        pytest.param("Hey!\nUse\n", id="a-fresh-partial-match-begins"),
        pytest.param("Hey!\nUser", id="the-reply-ends-inside-one"),
    ],
)
def test_text_that_only_looked_like_a_stop_string_is_emitted_after_all(
    trained_run: Path, monkeypatch: pytest.MonkeyPatch, script: str
) -> None:
    """The other half of holding back: what is held has to come out if it never matches.

    A stream that drops it truncates the reply at whatever happened to resemble a label,
    which is worse than the bug the hold-back fixes -- it loses text the model wrote.

    The last two scripts are where that goes wrong quietly. One ends *inside* a partial
    match, so the only thing that can emit its tail is the flush after the loop; the
    other starts a new partial match with the same character that resolves the old one,
    so a stream that marks the whole decoded text as emitted when it flushes part of it
    drops one newline and nothing else.
    """
    session = InferenceSession.open(trained_run, device="cpu")
    scripted(session, monkeypatch, script)

    out = session.complete("x", max_new_tokens=40, temperature=0.0, stop=("\nUser:",))

    assert out == script


def test_a_stop_string_does_not_turn_the_stream_into_one_lump(
    trained_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Holding back too much is as wrong as holding back too little, and it looks fine.

    Every assertion above passes for a stream that emits nothing until the end and then
    the whole reply at once -- the text and the counts all come out right. What is lost
    is the only thing streaming is for. None of these ten characters can begin
    ``"\\nUser:"``, so with one character per token each one is due immediately.

    The stream then ends with the finish piece, which is asserted whole here: it holds no
    text, repeats the last token count rather than inventing an eleventh token, and says
    the budget ran out.
    """
    session = InferenceSession.open(trained_run, device="cpu")
    scripted(session, monkeypatch, "Hey there!")

    pieces = list(session.stream_pieces("x", max_new_tokens=10, temperature=0.0, stop=("\nUser:",)))

    assert [piece.text for piece in pieces[:-1]] == list("Hey there!")
    assert [piece.tokens for piece in pieces[:-1]] == list(range(1, 11))
    assert pieces[-1] == StreamPiece("", 10, Finish("length"))


def test_a_stop_string_at_the_very_start_produces_nothing(
    trained_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty reply. A cut at offset zero is falsy, so it is the one a guard drops.

    The model writing the next turn's label immediately is what an under-trained model
    does, and the honest answer is that it said nothing -- not the label rendered to the
    user as if it were a reply.
    """
    session = InferenceSession.open(trained_run, device="cpu")
    scripted(session, monkeypatch, "\nUser: hi")

    pieces = list(session.stream_pieces("x", max_new_tokens=40, temperature=0.0, stop=("\nUser:",)))

    assert "".join(piece.text for piece in pieces) == ""
    assert pieces[-1].tokens == 6, "the tokens it took to say nothing are still reported"


def test_a_generation_cut_short_still_counts_the_tokens_it_cost(
    trained_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unlike end-of-text, the tokens spelling a stop string were computed and took time.

    A reported tokens/s that drops them is a wrong number about the machine, which is
    the same mistake ``StreamPiece`` exists to prevent for held-back characters.
    """
    session = InferenceSession.open(trained_run, device="cpu")
    scripted(session, monkeypatch, "Hey!\nUser: hi")

    pieces = list(session.stream_pieces("x", max_new_tokens=40, temperature=0.0, stop=("\nUser:",)))

    assert "".join(piece.text for piece in pieces) == "Hey!"
    # "Hey!" is 4 characters, so the match completes on the tenth token of "Hey!\nUser:".
    assert pieces[-1].tokens == 10
    assert [piece.tokens for piece in pieces[:-1]] == list(range(1, 11)), "one piece per token"
    assert pieces[-1].finish == Finish("stop", "\nUser:"), "then one more, for the finish"


def test_the_stream_stops_walking_the_model_once_it_has_matched(
    trained_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cutting the text but generating the rest anyway would waste the whole budget."""
    session = InferenceSession.open(trained_run, device="cpu")
    scripted(session, monkeypatch, "Hey!\nUser: " + "wasted " * 40)
    steps = 0
    real_generate = session.model.generate_stream

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal steps
        for token in real_generate(*args, **kwargs):
            steps += 1
            yield token

    monkeypatch.setattr(session.model, "generate_stream", counted)
    session.complete("x", max_new_tokens=200, temperature=0.0, stop=("\nUser:",))

    assert steps == 10, "the token that completed the match, and not one more"


def test_no_stop_strings_generates_exactly_what_it_did_before(trained_run: Path) -> None:
    """The negative control. A hold-back that fires when nothing was asked for would
    silently shorten every completion in the tool, and the tests above would all pass."""
    session = InferenceSession.open(trained_run, device="cpu")
    kwargs: dict[str, Any] = {"max_new_tokens": 16, "temperature": 1.0, "seed": 11}

    assert session.complete("The", **kwargs) == session.complete("The", stop=(), **kwargs)


def test_the_labels_a_chat_model_runs_on_are_usable_as_stop_strings(
    trained_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pairing this is for, checked once so the two halves cannot drift apart.

    A trained model continues past its own reply into the next ``User:``, because it was
    shown whole conversations. The strings come from :mod:`trainai.data.chat`, next to
    the renderer that put them in the shards, rather than from a copy kept here.
    """
    from trainai.data.chat import TURN_BOUNDARIES

    session = InferenceSession.open(trained_run, device="cpu")
    scripted(session, monkeypatch, "2 + 2 = 4.\n\nUser: thanks!\nAssistant: any time")

    out = session.complete("x", max_new_tokens=80, temperature=0.0, stop=TURN_BOUNDARIES)

    # One trailing newline is left for the caller to strip: the boundaries hold the
    # single-newline form, so against a blank line they match at the second newline.
    assert out == "2 + 2 = 4.\n"
    assert out.rstrip("\n") == "2 + 2 = 4."


def test_an_empty_stop_string_is_refused_before_anything_is_generated(
    trained_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It matches at position zero, so honouring it ends every generation with nothing.

    Dropping it silently would hide the mistake that computed it -- a caller assembling
    stop strings from a template with a label missing would see a model that has
    apparently stopped answering.
    """
    session = InferenceSession.open(trained_run, device="cpu")

    def refuse_to_generate(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("validation must happen before the model is walked")

    monkeypatch.setattr(session.model, "generate_stream", refuse_to_generate)

    with pytest.raises(UsageError) as caught:
        session.complete("x", max_new_tokens=4, stop=("\nUser:", ""))

    assert "cannot be empty" in str(caught.value)
    assert caught.value.hint
    assert caught.value.details["stop"] == ["\nUser:", ""]


# --------------------------------------------------------------------------- #
# Why a generation ended
# --------------------------------------------------------------------------- #
# The distinction these pin is invisible in the text: a reply that finished and a reply
# that ran out of budget are both text that stops. Only one of them is worth continuing,
# and guessing from the token count is wrong exactly when the model happened to finish on
# its last allowed token.
def scripted_tokens(session: Any, monkeypatch: pytest.MonkeyPatch, ids: list[int]) -> None:
    """Replace the model's walk with a fixed sequence of token ids.

    Needed for end-of-text: a model this small will not reliably emit it, and the
    behaviour under test is what the *stream* does when it arrives, not whether it does.
    """

    def fake_generate(prompt: Any, max_new_tokens: int, **kwargs: Any) -> Any:
        for value in ids[:max_new_tokens]:
            yield torch.tensor([[value]])

    monkeypatch.setattr(session.model, "generate_stream", fake_generate)


def test_a_generation_the_model_ended_reports_end_of_text(
    trained_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And the end-of-text token is still not counted, which is the older promise."""
    session = InferenceSession.open(trained_run, device="cpu")
    scripted_tokens(session, monkeypatch, [5, 6, session.tokenizer.eot_id, 7])

    pieces = list(session.stream_pieces("x", max_new_tokens=40, temperature=0.0))

    assert pieces[-1].finish == Finish("end-of-text")
    assert pieces[-1].tokens == 2, "the boundary is not content, so it does not count"
    assert not pieces[-1].finish.cut, "the model finished; there is nothing to continue"


def test_a_generation_that_ran_out_of_budget_reports_the_limit(
    trained_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason this exists. The model was mid-sentence and the caller's budget ended.

    ``cut`` is the property callers act on, and it is the difference between "the model
    is done" and "ask for more tokens" -- which the text alone cannot tell them.
    """
    session = InferenceSession.open(trained_run, device="cpu")
    scripted(session, monkeypatch, "a reply that keeps going and going")

    pieces = list(session.stream_pieces("x", max_new_tokens=6, temperature=0.0))

    assert pieces[-1].finish == Finish("length")
    assert pieces[-1].finish.cut
    assert pieces[-1].tokens == 6


def test_the_stop_string_that_matched_is_named(
    trained_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The earliest one, not the first one listed -- the same rule the cut follows.

    A caller that assembled the list from a chat template wants to know which label the
    model started writing. Reporting the first *listed* match would name ``"three"`` here
    while cutting at ``"two"``, so the reason would contradict the text beside it.
    """
    session = InferenceSession.open(trained_run, device="cpu")
    chunked(session, monkeypatch, ["one ", "two three"])

    pieces = list(
        session.stream_pieces("x", max_new_tokens=40, temperature=0.0, stop=("three", "two"))
    )

    assert "".join(piece.text for piece in pieces) == "one "
    assert pieces[-1].finish == Finish("stop", "two")
    assert not pieces[-1].finish.cut, "a model writing the next turn has finished this one"


@pytest.mark.parametrize(
    ("stops", "named"),
    [
        pytest.param(("\nUser", "\nUser:"), "\nUser", id="shorter-first"),
        pytest.param(("\nUser:", "\nUser"), "\nUser:", id="longer-first"),
    ],
)
def test_stop_strings_that_begin_at_the_same_place_are_named_in_the_callers_order(
    trained_run: Path, monkeypatch: pytest.MonkeyPatch, stops: tuple[str, ...], named: str
) -> None:
    """One label is a prefix of the other, so both begin at the same character.

    The cut is the same either way, so this is only about which one is *reported*, and
    the answer has to come from something the caller can see. Their own order is that;
    the order this happens to iterate in is not, and would make the report an accident of
    the loop. Chunked rather than one character per token on purpose: with one character
    the shorter string always completes a step earlier, and the tie never arises.
    """
    session = InferenceSession.open(trained_run, device="cpu")
    chunked(session, monkeypatch, ["Hey!", "\nUser: hi"])

    pieces = list(session.stream_pieces("x", max_new_tokens=40, temperature=0.0, stop=stops))

    assert "".join(piece.text for piece in pieces) == "Hey!"
    assert pieces[-1].finish == Finish("stop", named)


def test_exactly_one_piece_carries_a_finish_and_it_is_the_last(trained_run: Path) -> None:
    """The contract callers rely on, checked against a real walk rather than a script.

    A finish on an earlier piece would make ``for piece in ...: finish = piece.finish``
    report the wrong reason, and one on none of them would report no reason at all.
    """
    session = InferenceSession.open(trained_run, device="cpu")

    pieces = list(session.stream_pieces("The", max_new_tokens=8, temperature=0.0))

    assert [index for index, piece in enumerate(pieces) if piece.finish] == [len(pieces) - 1]
    assert pieces[-1].tokens == pieces[-2].tokens, "the finish piece is not a token"


def test_an_abandoned_stream_reports_no_reason_at_all(trained_run: Path) -> None:
    """A caller that stops iterating -- Ctrl-C in the playground -- never reaches it.

    Reporting a reason there would be inventing one: nothing ended the generation, the
    caller stopped asking. ``None`` is what makes "interrupted" and "finished" different
    in a machine-readable report rather than only in a printed line.
    """
    session = InferenceSession.open(trained_run, device="cpu")

    stream = session.stream_pieces("The", max_new_tokens=8, temperature=0.0)
    first = next(stream)
    stream.close()

    assert first.finish is None


def test_the_reasons_a_caller_acts_on_are_the_ones_it_is_given(trained_run: Path) -> None:
    """``cut`` is the whole public distinction, so it is asserted on its own.

    Folded into the reasons above it would be asserted three times and pinned nowhere:
    each of those tests would still pass if ``cut`` were ``True`` for every reason.
    """
    assert Finish("length").cut
    assert not Finish("stop", "\nUser:").cut
    assert not Finish("end-of-text").cut
    assert Finish("length").to_dict() == {"reason": "length", "stop": None}
    assert Finish("stop", "\nUser:").to_dict() == {"reason": "stop", "stop": "\nUser:"}
