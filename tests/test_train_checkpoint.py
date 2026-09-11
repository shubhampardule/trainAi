"""Tests for :mod:`trainai.train.checkpoint`.

A checkpoint is the only thing standing between a six-hour run and losing it, so
the properties tested here are the ones that matter when something has already gone
wrong: that a half-written file cannot replace a good one, that a damaged file is
refused rather than half-loaded, and that resuming against the wrong dataset is
caught before it produces a model whose loss looks fine and whose output is
nonsense.

Plus the tied-weight decision. ``state_dict()`` names the output projection twice
and ``load_state_dict`` lets the last one win silently, so the checkpoint format
stores it once. The test that pins that down is
``test_the_tied_alias_is_dropped_from_the_file``.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from trainai.errors import (
    CheckpointCorruptError,
    CheckpointError,
    CheckpointIncompatibleError,
    CheckpointNotFoundError,
    ExitCode,
)
from trainai.model.config import ModelConfig
from trainai.model.gpt import GPT
from trainai.train.checkpoint import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    POINTER_NAME,
    RngState,
    _imports_wanted,
    _prune,
    capture_rng,
    find_checkpoint,
    list_checkpoints,
    load_checkpoint,
    pointer_fallback_reason,
    restore_rng,
    save_checkpoint,
)
from trainai.train.config import TrainConfig

CONFIG = ModelConfig(vocab_size=64, n_layer=2, n_head=4, d_model=32, seq_len=16)
TRAIN = TrainConfig(steps=100, batch_size=2, seq_len=16)
DATASET = {
    "tokenizer_fingerprint": "a" * 64,
    "content_hash": "b" * 64,
    "vocab_size": 64,
}


@pytest.fixture
def model() -> GPT:
    torch.manual_seed(0)
    return GPT(CONFIG)


@pytest.fixture
def optimizer(model: GPT) -> torch.optim.Optimizer:
    opt = torch.optim.AdamW(model.parameter_groups(0.1), lr=1e-3)
    # Take one real step so the optimizer has moments to save.
    inputs = torch.randint(0, 64, (2, 8))
    _, loss, _ = model(inputs, inputs)
    assert loss is not None
    loss.backward()
    opt.step()
    return opt


def write(directory: Path, model: GPT, optimizer: torch.optim.Optimizer, **overrides: Any) -> Path:
    settings: dict[str, Any] = {
        "step": 42,
        "model": model,
        "optimizer": optimizer,
        "train_config": TRAIN,
        "dataset": DATASET,
        "rng": capture_rng(),
    }
    settings.update(overrides)
    return save_checkpoint(directory, **settings)


# --------------------------------------------------------------------------- #
# Writing and reading
# --------------------------------------------------------------------------- #
def test_a_checkpoint_round_trips(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    path = write(tmp_path, model, optimizer)

    loaded = load_checkpoint(path)

    assert loaded.step == 42
    assert loaded.model_config.to_dict() == CONFIG.to_dict()
    assert loaded.train_config.to_dict() == TRAIN.to_dict()
    assert loaded.dataset == DATASET
    assert loaded.rng is not None


def test_a_directory_of_checkpoints_loads_the_latest(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    """What `--resume <run>/checkpoints` does, which is how the flag is documented.

    `load_checkpoint` takes a directory as well as a file and picks the newest step in
    it. The refusal for a directory with nothing in it is covered by
    `test_an_empty_directory_is_refused_with_what_to_do`; this is the other side of that
    branch, and it was the untested one -- every other test in this file hands over the
    exact path `save_checkpoint` returned, so the resolution step that users actually go
    through never ran.

    Asserted by step rather than by filename so it is the checkpoint that was chosen,
    not the name that was matched.
    """
    write(tmp_path, model, optimizer, step=7)
    write(tmp_path, model, optimizer, step=41)

    assert load_checkpoint(tmp_path).step == 41
    assert load_checkpoint(tmp_path / "step-0000007.pt").step == 7


def test_the_filename_carries_a_sortable_step(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    """Zero-padded, so a plain sort of the filenames is a sort by step."""
    for step in (1, 20, 300, 4000):
        write(tmp_path, model, optimizer, step=step, keep=0)

    names = [p.name for p in list_checkpoints(tmp_path)]

    assert names == sorted(names)
    assert names[0] == "step-0000001.pt"


def test_loading_restores_weights_exactly(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    path = write(tmp_path, model, optimizer)
    fresh = GPT(CONFIG)

    load_checkpoint(path).apply_to(fresh)

    for (name, original), (_, restored) in zip(
        model.state_dict().items(), fresh.state_dict().items(), strict=True
    ):
        assert torch.equal(original, restored), name


def test_loading_restores_optimizer_moments_exactly(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    path = write(tmp_path, model, optimizer)
    fresh = GPT(CONFIG)
    fresh_optimizer = torch.optim.AdamW(fresh.parameter_groups(0.1), lr=1e-3)

    loaded = load_checkpoint(path)
    loaded.apply_to(fresh)
    fresh_optimizer.load_state_dict(loaded.optimizer_state)

    original_state = optimizer.state_dict()["state"]
    restored_state = fresh_optimizer.state_dict()["state"]
    assert set(original_state) == set(restored_state)
    for key in original_state:
        for field in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(original_state[key][field], restored_state[key][field])


def test_the_tied_alias_is_dropped_from_the_file(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    """One copy on disk, re-tied on load. See the module docstring for why."""
    path = write(tmp_path, model, optimizer)

    loaded = load_checkpoint(path)

    assert "lm_head.weight" not in loaded.model_state
    assert "embed_tokens.weight" in loaded.model_state

    fresh = GPT(CONFIG)
    loaded.apply_to(fresh)
    assert fresh.lm_head.weight is fresh.embed_tokens.weight


def test_an_untied_model_stores_both_matrices(
    tmp_path: Path, optimizer: torch.optim.Optimizer
) -> None:
    untied_config = ModelConfig.from_dict({**CONFIG.to_dict(), "tie_embeddings": False})
    untied = GPT(untied_config)
    opt = torch.optim.AdamW(untied.parameter_groups(0.1), lr=1e-3)

    path = write(tmp_path, untied, opt)
    loaded = load_checkpoint(path)

    assert "lm_head.weight" in loaded.model_state
    assert "embed_tokens.weight" in loaded.model_state


def test_no_rotary_buffers_are_stored(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    path = write(tmp_path, model, optimizer)

    loaded = load_checkpoint(path)

    assert not any("rotary" in key for key in loaded.model_state)


def test_the_scaler_state_is_stored_when_there_is_one(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    path = write(tmp_path, model, optimizer, scaler=scaler)

    assert load_checkpoint(path).scaler_state is not None


def test_no_scaler_means_no_scaler_state(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    path = write(tmp_path, model, optimizer)

    assert load_checkpoint(path).scaler_state is None


# --------------------------------------------------------------------------- #
# Atomicity
# --------------------------------------------------------------------------- #
def test_no_temporary_file_is_left_behind(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    write(tmp_path, model, optimizer)

    assert list(tmp_path.glob("*.tmp")) == []


def test_a_save_that_runs_out_of_disk_keeps_the_previous_checkpoint(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full disk is the way a long run ends, and it must not cost the last checkpoint.

    The save writes to a temporary and renames it, so the previous checkpoint survives
    by construction -- but only if the failure is caught and the temporary is removed.
    Left behind, a half-written `step-0000004.pt.tmp` sits in the run directory forever;
    the sibling above proves it is gone after a *successful* save, and this proves it is
    gone after a failed one, which is the case that creates it.

    `torch.save` is made to fail after writing part of the file, because that is the
    order a real disk fills up in: `os.replace` is not reached, and a test that raised
    before the temporary existed would exercise the handler without proving it cleans
    anything up. The message and its hint are asserted because they are what a person
    reads at the end of a run that took hours -- "the previous checkpoint is untouched"
    is a promise this test is the only thing holding.
    """
    kept = write(tmp_path, model, optimizer, step=3)

    def out_of_disk(payload: Any, target: Any) -> None:
        Path(str(target)).write_bytes(b"half a checkpoint")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(torch, "save", out_of_disk)

    with pytest.raises(CheckpointError) as caught:
        write(tmp_path, model, optimizer, step=4)

    assert "step-0000004.pt" in str(caught.value)
    assert "free disk space" in (caught.value.hint or "")
    assert "No space left on device" in caught.value.details["reason"]
    monkeypatch.undo()

    assert list(tmp_path.glob("*.tmp")) == [], "the half-written temporary was left behind"
    assert not (tmp_path / "step-0000004.pt").exists()
    assert load_checkpoint(kept).step == 3, "the previous checkpoint did not survive"


def test_a_stray_temporary_file_is_not_mistaken_for_a_checkpoint(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    """What an interrupted save leaves. It must not be offered as resumable."""
    write(tmp_path, model, optimizer, step=10)
    (tmp_path / "step-0000099.pt.tmp").write_bytes(b"half a checkpoint")

    found = find_checkpoint(tmp_path)

    assert found is not None
    assert found.name == "step-0000010.pt"
    assert [p.name for p in list_checkpoints(tmp_path)] == ["step-0000010.pt"]


# --------------------------------------------------------------------------- #
# The pointer file
# --------------------------------------------------------------------------- #
def test_the_pointer_names_the_latest_and_the_best(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    write(tmp_path, model, optimizer, step=10, is_best=True, metrics={"val_loss": 2.0})
    write(tmp_path, model, optimizer, step=20, is_best=False, metrics={"val_loss": 3.0})

    pointer = json.loads((tmp_path / POINTER_NAME).read_text(encoding="utf-8"))

    assert pointer["latest"]["step"] == 20
    assert pointer["best"]["step"] == 10
    assert find_checkpoint(tmp_path, which="latest").name == "step-0000020.pt"
    assert find_checkpoint(tmp_path, which="best").name == "step-0000010.pt"


def test_the_pointer_is_written_as_ascii_json_with_lf_endings(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    write(tmp_path, model, optimizer)

    raw = (tmp_path / POINTER_NAME).read_bytes()

    assert all(byte < 128 for byte in raw)
    assert b"\r\n" not in raw
    assert raw.endswith(b"\n")


def test_a_pointer_naming_a_deleted_file_falls_back_to_the_filenames(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    """Deleting a checkpoint by hand is reasonable and must not break resume."""
    write(tmp_path, model, optimizer, step=10, keep=0)
    write(tmp_path, model, optimizer, step=20, keep=0)
    (tmp_path / "step-0000020.pt").unlink()

    assert find_checkpoint(tmp_path).name == "step-0000010.pt"


def test_a_corrupt_pointer_falls_back_to_the_filenames(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    write(tmp_path, model, optimizer, step=7)
    (tmp_path / POINTER_NAME).write_text("{not json", encoding="utf-8", newline="\n")

    assert find_checkpoint(tmp_path).name == "step-0000007.pt"


def test_find_checkpoint_returns_none_for_an_empty_directory(tmp_path: Path) -> None:
    assert find_checkpoint(tmp_path) is None
    assert list_checkpoints(tmp_path) == []
    assert list_checkpoints(tmp_path / "does-not-exist") == []


# --------------------------------------------------------------------------- #
# A damaged pointer file
# --------------------------------------------------------------------------- #
#: A pointer file that is entirely valid, so a case below can damage exactly one thing.
GOOD_POINTER: dict[str, Any] = {
    "latest": {"file": "step-0000020.pt", "step": 20},
    "best": {"file": "step-0000010.pt", "step": 10, "val_loss": 2.0},
}

#: (what the edit is, the bytes on disk). Every one of these was a bare ``TypeError``,
#: ``AttributeError`` or ``UnicodeDecodeError`` from ``_prune`` or ``_update_pointer``
#: before -- and both of those run *after* the checkpoint is on disk, so a hand-edited
#: pointer file ended a training run with a traceback over a file the next line was
#: going to overwrite. Numbers as well as strings for every "not an object" case: the
#: manifest reader one file over was measured to leave three checks unpinned when only
#: the string was tried, because ``"file" not in "text"`` is merely False.
POINTER_DAMAGE: list[tuple[str, bytes]] = [
    ("the document is an array", b"[]\n"),
    ("the document is a string", b'"latest"\n'),
    ("the document is a number", b"3\n"),
    ("the document is null", b"null\n"),
    ("latest is a string", json.dumps({**GOOD_POINTER, "latest": "step-0000020.pt"}).encode()),
    ("latest is an array", json.dumps({**GOOD_POINTER, "latest": []}).encode()),
    ("latest is a number", json.dumps({**GOOD_POINTER, "latest": 20}).encode()),
    ("best is a string", json.dumps({**GOOD_POINTER, "best": "step-0000010.pt"}).encode()),
    ("best is a number", json.dumps({**GOOD_POINTER, "best": 10}).encode()),
    ("latest.file is an array", json.dumps({**GOOD_POINTER, "latest": {"file": []}}).encode()),
    ("latest.file is a number", json.dumps({**GOOD_POINTER, "latest": {"file": 20}}).encode()),
    ("latest.file is empty", json.dumps({**GOOD_POINTER, "latest": {"file": ""}}).encode()),
    ("latest has no file", json.dumps({**GOOD_POINTER, "latest": {"step": 20}}).encode()),
    ("neither key is there", b'{"note": "hand-edited"}\n'),
    ("not json at all", b"{not json"),
    ("an empty file", b""),
    ("truncated mid-object", b'{"latest": {"file": "step-0000020.pt", "st'),
    # UnicodeDecodeError is a ValueError, so `except OSError` never saw this one.
    ("bytes that are not utf-8", b'{"latest": {"file": "step\xff.pt"}}'),
]


def stubs(directory: Path, *steps: int) -> None:
    """Checkpoint files that are not real checkpoints.

    Pruning and the pointer only ever look at filenames, so a stub exercises them for
    the price of a write. The one checkpoint each test below actually saves goes through
    ``save_checkpoint`` for real.
    """
    for step in steps:
        (directory / f"step-{step:07d}.pt").write_bytes(b"not a real checkpoint")


@pytest.mark.parametrize("damage", [pytest.param(raw, id=label) for label, raw in POINTER_DAMAGE])
def test_a_damaged_pointer_never_stops_a_checkpoint_from_being_read(
    tmp_path: Path, damage: bytes
) -> None:
    """The read path, with the damage still on disk.

    Separate from the save tests below because ``save_checkpoint`` repairs the file
    before anything reads it a second time -- so a test that saves first proves nothing
    about the reader. This is where ``latest.file`` holding an array or the empty string
    is caught: ``directory / []`` raises, and ``directory / ""`` is the directory.
    """
    stubs(tmp_path, 10, 20)
    (tmp_path / POINTER_NAME).write_bytes(damage)

    for which in ("latest", "best"):
        found = find_checkpoint(tmp_path, which=which)
        assert found is not None
        # is_file, not exists: `latest.file` of "" makes `directory / name` the directory
        # itself, which exists and is not a checkpoint.
        assert found.is_file(), f"{which} resolved to {found}, which is not a checkpoint"

    reasons = [pointer_fallback_reason(tmp_path, which=w) for w in ("latest", "best")]
    assert any(reasons), "a fallback nobody is told about is the silent half"


@pytest.mark.parametrize("damage", [pytest.param(raw, id=label) for label, raw in POINTER_DAMAGE])
def test_a_damaged_pointer_never_stops_a_checkpoint_from_being_saved(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer, damage: bytes
) -> None:
    """The pointer is derived data, so a damaged one is rebuilt, not refused.

    This is the opposite of the choice ``manifest.json`` gets, deliberately: ``latest``
    is the highest-numbered file on disk, so nothing here is unrecoverable, and refusing
    would end a run over a convenience file the same call is about to rewrite.
    """
    stubs(tmp_path, 10, 20)
    (tmp_path / POINTER_NAME).write_bytes(damage)

    saved = write(tmp_path, model, optimizer, step=30, keep=1)

    assert saved.exists()
    rewritten = json.loads((tmp_path / POINTER_NAME).read_text(encoding="utf-8"))
    assert rewritten["latest"]["file"] == "step-0000030.pt"
    assert find_checkpoint(tmp_path) == saved


@pytest.mark.parametrize("damage", [pytest.param(raw, id=label) for label, raw in POINTER_DAMAGE])
def test_pruning_against_a_damaged_pointer_keeps_the_newest(tmp_path: Path, damage: bytes) -> None:
    """``_prune`` directly, which is the only way to reach it with damage still in place.

    Through ``save_checkpoint`` this cannot happen -- ``_update_pointer`` runs one line
    earlier and repairs the file, so every mutation of the protection here survives a
    test that goes through the public path. The promise "never the newest" is documented
    unconditionally, though, and the only thing that made it true was a pointer file a
    user can edit, so it is pinned against the function that makes it.
    """
    stubs(tmp_path, 10, 20, 30)
    (tmp_path / POINTER_NAME).write_bytes(damage)

    _prune(tmp_path, keep=1)

    kept = [p.name for p in list_checkpoints(tmp_path)]
    assert "step-0000030.pt" in kept, f"the newest was pruned; kept {kept}"


@pytest.mark.parametrize("damage", [pytest.param(raw, id=label) for label, raw in POINTER_DAMAGE])
def test_a_damaged_pointer_never_costs_the_newest_checkpoint(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer, damage: bytes
) -> None:
    """``save_checkpoint`` documents that the newest is always kept.

    It used to be kept only because the pointer named it. A pointer whose ``best`` was
    readable and whose ``latest`` was not left the just-written file unprotected and
    ``keep=1`` deleted it -- measured, for ``latest.file`` holding an array.
    """
    stubs(tmp_path, 10, 20)
    (tmp_path / POINTER_NAME).write_bytes(damage)

    saved = write(tmp_path, model, optimizer, step=30, keep=1)

    assert saved.exists(), "the checkpoint just written was pruned"
    assert list_checkpoints(tmp_path)[-1] == saved


def test_a_pointer_key_this_version_does_not_know_survives_a_rewrite(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    """The file is rewritten in place, and dropping an unknown key is how a downgrade
    quietly loses what a later version recorded."""
    write(tmp_path, model, optimizer, step=10)
    pointer = tmp_path / POINTER_NAME
    raw = json.loads(pointer.read_text(encoding="utf-8"))
    pointer.write_text(
        json.dumps({**raw, "worst": {"file": "x.pt"}}), encoding="utf-8", newline="\n"
    )

    write(tmp_path, model, optimizer, step=20, keep=0)

    assert json.loads(pointer.read_text(encoding="utf-8"))["worst"] == {"file": "x.pt"}


def test_falling_back_for_the_best_checkpoint_is_reportable(tmp_path: Path) -> None:
    """The filenames record the order checkpoints were written, not their loss.

    So a damaged pointer answers "the best checkpoint" with the last one -- a different
    checkpoint from the one asked for, which is why there is something to report at all.
    Reported for both, though: the asymmetry is which of them is worth *showing*, and
    that judgement belongs where the answer is shown, not in a query.
    """
    stubs(tmp_path, 10, 20)

    absent = pointer_fallback_reason(tmp_path, which="best")
    assert absent is not None and POINTER_NAME in absent

    (tmp_path / POINTER_NAME).write_bytes(b"{not json")
    damaged = pointer_fallback_reason(tmp_path, which="best")
    assert damaged is not None and POINTER_NAME in damaged

    (tmp_path / POINTER_NAME).write_text(json.dumps(GOOD_POINTER), encoding="utf-8", newline="\n")
    assert pointer_fallback_reason(tmp_path, which="best") is None
    assert pointer_fallback_reason(tmp_path, which="latest") is None

    (tmp_path / "step-0000010.pt").unlink()
    gone = pointer_fallback_reason(tmp_path, which="best")
    assert gone is not None and "step-0000010.pt" in gone


def test_an_undamaged_pointer_is_unaffected_by_any_of_this(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    """The negative control: a check accidentally inverted fails here rather than
    silently discarding the pointer of every run on disk."""
    write(tmp_path, model, optimizer, step=10, is_best=True, metrics={"val_loss": 2.0}, keep=0)
    write(tmp_path, model, optimizer, step=20, metrics={"val_loss": 3.0}, keep=0)

    raw = json.loads((tmp_path / POINTER_NAME).read_text(encoding="utf-8"))

    assert raw["latest"] == {"file": "step-0000020.pt", "step": 20, "val_loss": 3.0}
    assert raw["best"] == {"file": "step-0000010.pt", "step": 10, "val_loss": 2.0}
    assert find_checkpoint(tmp_path, which="best").name == "step-0000010.pt"
    assert pointer_fallback_reason(tmp_path, which="best") is None


# --------------------------------------------------------------------------- #
# Pruning
# --------------------------------------------------------------------------- #
def test_old_checkpoints_are_pruned_to_the_keep_count(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    for step in range(1, 8):
        write(tmp_path, model, optimizer, step=step, keep=2)

    kept = list_checkpoints(tmp_path)

    assert len(kept) <= 3  # keep=2 plus whatever the pointer protects
    assert kept[-1].name == "step-0000007.pt"


def test_the_best_checkpoint_survives_pruning(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    write(tmp_path, model, optimizer, step=1, is_best=True, keep=1)
    for step in range(2, 8):
        write(tmp_path, model, optimizer, step=step, keep=1)

    names = {p.name for p in list_checkpoints(tmp_path)}

    assert "step-0000001.pt" in names, "the best checkpoint was deleted"
    assert "step-0000007.pt" in names, "the newest checkpoint was deleted"


def test_keep_zero_keeps_everything_rather_than_nothing(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    """Deleting every checkpoint would leave a run with no way to resume."""
    for step in range(1, 5):
        write(tmp_path, model, optimizer, step=step, keep=0)

    assert len(list_checkpoints(tmp_path)) == 4


def test_pruning_an_empty_directory_is_not_an_error(tmp_path: Path) -> None:
    """The guard that stops "never the newest" from asking which file is newest.

    Unreachable through `save_checkpoint`, which prunes one line after writing a file
    and so always has at least one. It is reachable the moment anything else prunes --
    a directory emptied by hand between the save and the prune, or a future caller --
    and without the guard the next line indexes `existing[-1]` on an empty list and a
    completed save dies with an IndexError. Pinned by calling the private function
    directly, because that is the only way in: a test that went through
    `save_checkpoint` would be testing the file it just wrote.
    """
    _prune(tmp_path, keep=3)

    assert list_checkpoints(tmp_path) == []


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #
def test_a_missing_path_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(CheckpointNotFoundError) as caught:
        load_checkpoint(tmp_path / "step-0000001.pt")

    assert "step-0000001.pt" in str(caught.value)
    assert caught.value.hint


def test_an_empty_directory_is_refused_with_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(CheckpointNotFoundError) as caught:
        load_checkpoint(tmp_path)

    assert "step-" in (caught.value.hint or "")


def test_a_file_that_is_not_a_checkpoint_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "step-0000001.pt"
    path.write_bytes(b"this is not a torch file")

    with pytest.raises(CheckpointCorruptError) as caught:
        load_checkpoint(path)

    assert "interrupted" in (caught.value.hint or "")


def test_a_torch_file_that_is_not_ours_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "step-0000001.pt"
    torch.save({"some": "other tool's checkpoint"}, path)

    with pytest.raises(CheckpointIncompatibleError) as caught:
        load_checkpoint(path)

    assert "not a TrainAI checkpoint" in str(caught.value)


@pytest.mark.parametrize(
    ("label", "payload"),
    [
        ("module", lambda: torch.nn.Linear(2, 2)),
        ("tensor", lambda: torch.zeros(4, 4)),
        ("state_dict", lambda: torch.nn.Linear(2, 2).state_dict()),
        ("list", lambda: [1, 2, 3]),
        ("string", lambda: "not a checkpoint"),
        ("none", lambda: None),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_a_torch_file_holding_something_other_than_a_dict_is_refused_cleanly(
    tmp_path: Path, label: str, payload: Callable[[], object]
) -> None:
    """Regression: the handler for "not our checkpoint" crashed on non-dict payloads.

    It built its own error message with ``(raw or {}).get("format")``, on a branch
    reached precisely when ``raw`` may not be a mapping. ``torch.save(model, path)``
    is how most people save a PyTorch model, and pointing ``--resume`` at one raised
    ``AttributeError: 'Linear' object has no attribute 'get'`` from inside the
    handler; a saved tensor raised ``RuntimeError: Boolean value of Tensor with more
    than one value is ambiguous`` from the ``or``. Either way the user got a
    traceback and exit 1 instead of the message and exit code the code intended.

    The neighbouring test above passes a dict, which is the one shape that answered
    correctly -- so the whole class went unnoticed. Hence the parametrization.
    """
    path = tmp_path / "step-0000001.pt"
    torch.save(payload(), path)

    with pytest.raises(CheckpointIncompatibleError) as caught:
        load_checkpoint(path)

    assert "not a TrainAI checkpoint" in str(caught.value)
    assert caught.value.details["holds"] == type(payload()).__name__
    # The type is named in the message, not just the details: "the file holds a
    # Linear" is what tells someone they pointed at a saved model.
    assert type(payload()).__name__ in str(caught.value)


def test_a_newer_format_version_names_something_that_can_actually_be_done(
    tmp_path: Path,
) -> None:
    """The hint used to be "Upgrade TrainAI with ``pip install --upgrade trainai``".

    README.md says outright that TrainAI is not published, so there is no such command
    -- and this hint is the *only* advice given for a checkpoint the build cannot read,
    which made it the user's whole path forward. It went nowhere.

    What is true instead is narrower and worth saying: unlike a dataset, a checkpoint
    cannot be re-created. It is the output of the run that wrote it, so offering
    "re-create it with `trainai train`" would be telling someone to retrain from
    scratch. The only route is the build that produced the file.
    """
    path = tmp_path / "step-0000001.pt"
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "format_version": CHECKPOINT_VERSION + 1,
            "step": 1,
        },
        path,
    )

    with pytest.raises(CheckpointIncompatibleError) as caught:
        load_checkpoint(path)

    hint = caught.value.hint or ""
    assert "pip install" not in hint, "there is nothing to install from; see README.md"
    assert "wrote it" in hint
    assert "cannot be re-created" in hint, (
        "the asymmetry with a dataset is the whole content of this hint: one can be "
        "rebuilt from the corpus and the other cannot be rebuilt from anything"
    )


@pytest.mark.parametrize("missing", ["step", "model", "train", "model_state", "optimizer_state"])
def test_a_missing_section_is_reported_by_name(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer, missing: str
) -> None:
    path = write(tmp_path, model, optimizer)
    payload = torch.load(path, weights_only=False)
    del payload[missing]
    torch.save(payload, path)

    with pytest.raises(CheckpointCorruptError) as caught:
        load_checkpoint(path)

    assert missing in str(caught.value)


def test_a_different_tokenizer_is_refused(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    """The same shards read through a different tokenizer are a different corpus."""
    path = write(tmp_path, model, optimizer)

    with pytest.raises(CheckpointIncompatibleError) as caught:
        load_checkpoint(path, expect_dataset={**DATASET, "tokenizer_fingerprint": "c" * 64})

    assert "tokenizer" in str(caught.value)
    assert caught.value.details["field"] == "tokenizer_fingerprint"


def test_a_different_dataset_is_refused(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    path = write(tmp_path, model, optimizer)

    with pytest.raises(CheckpointIncompatibleError) as caught:
        load_checkpoint(path, expect_dataset={**DATASET, "content_hash": "d" * 64})

    assert caught.value.details["field"] == "content_hash"


def test_the_matching_dataset_is_accepted(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    path = write(tmp_path, model, optimizer)

    assert load_checkpoint(path, expect_dataset=DATASET).step == 42


def test_a_different_model_shape_is_refused_on_apply(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    path = write(tmp_path, model, optimizer)
    deeper = GPT(ModelConfig.from_dict({**CONFIG.to_dict(), "n_layer": 3}))

    with pytest.raises(CheckpointIncompatibleError) as caught:
        load_checkpoint(path).apply_to(deeper)

    assert "L2" in str(caught.value.details["checkpoint"])
    assert "L3" in str(caught.value.details["model"])


def edited_weight(path: Path, key: str, value: Any) -> Path:
    """Put ``value`` at ``key`` in a saved checkpoint's ``model_state``.

    The ``model`` block is left alone on purpose: these tests are about a file whose
    recorded shape still matches, so the config gate in ``apply_to`` passes and the
    weights themselves are the only thing wrong. Anything that also edited the block
    would be refused earlier and would test the gate instead.
    """
    payload = torch.load(path, weights_only=False)
    payload["model_state"][key] = value
    torch.save(payload, path)
    return path


def test_weights_that_do_not_match_the_model_are_refused(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    path = write(tmp_path, model, optimizer)
    payload = torch.load(path, weights_only=False)
    payload["model_state"].pop("final_norm.weight")
    torch.save(payload, path)

    with pytest.raises(CheckpointCorruptError) as caught:
        load_checkpoint(path).apply_to(GPT(CONFIG))

    assert "final_norm.weight" in str(caught.value.details["missing"])
    assert str(path) in str(caught.value), "a refusal about a file has to name the file"
    assert caught.value.details["path"] == str(path)


def test_a_weight_the_model_does_not_have_is_refused_and_named(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    """The other half of the key comparison, which had no test of its own.

    A key too many is as much a sign of a damaged or hand-edited file as a key too few,
    and it is the half that reports through ``unexpected_keys`` rather than through
    ``missing_keys``.
    """
    path = edited_weight(
        write(tmp_path, model, optimizer), "blocks.0.attn.not_a_real_weight", torch.zeros(4)
    )

    with pytest.raises(CheckpointCorruptError) as caught:
        load_checkpoint(path).apply_to(GPT(CONFIG))

    assert "blocks.0.attn.not_a_real_weight" in str(caught.value.details["unexpected"])


def test_the_missing_list_does_not_blame_the_tied_alias(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    """A key the format deliberately omits must not be reported as one the file lost.

    ``lm_head.weight`` is absent from every tied checkpoint on purpose -- see
    ``test_the_tied_alias_is_dropped_from_the_file`` -- so a refusal that lists it sends
    someone looking for damage in the one place the format guarantees is empty. The
    refusal happens either way here, because ``embed_tokens.weight`` really is gone; what
    this pins is which keys the report names.
    """
    path = write(tmp_path, model, optimizer)
    payload = torch.load(path, weights_only=False)
    del payload["model_state"]["embed_tokens.weight"]
    torch.save(payload, path)

    with pytest.raises(CheckpointCorruptError) as caught:
        load_checkpoint(path).apply_to(GPT(CONFIG))

    assert caught.value.details["missing"] == ["embed_tokens.weight"], (
        "the tied alias is not a missing weight, it is one the format never stores: "
        f"{caught.value.details['missing']}"
    )


def test_anything_else_torch_objects_to_is_still_a_refusal(
    tmp_path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The backstop, tested by forcing it, because by construction nothing else reaches it.

    The two known ways ``load_state_dict`` raises are checked before it is called, so this
    ``except`` clause is only there for a third that a future torch invents. That makes it
    the kind of code that quietly stops working: no test would notice it being deleted.
    Simulated rather than provoked, since provoking it would mean knowing the case it
    exists for.
    """
    path = write(tmp_path, model, optimizer)
    fresh = GPT(CONFIG)

    def explode(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("something torch has not thought of yet")

    monkeypatch.setattr(fresh, "load_state_dict", explode)

    with pytest.raises(CheckpointCorruptError) as caught:
        load_checkpoint(path).apply_to(fresh)

    assert caught.value.exit_code == ExitCode.CHECKPOINT
    assert str(path) in str(caught.value)
    assert "not thought of yet" in str(caught.value.details["torch_error"]), (
        "torch's own words are the only description of a case this code does not know, "
        f"so they belong in details: {caught.value.details}"
    )


@pytest.mark.parametrize(
    ("label", "value", "detail"),
    [
        ("a tensor at the wrong shape", torch.zeros(3, 5), "wrong_shape"),
        ("a string where a tensor belongs", "not a tensor", "wrong_type"),
        ("a list where a tensor belongs", [1, 2, 3], "wrong_type"),
    ],
)
def test_a_weight_torch_cannot_copy_is_refused_rather_than_raised(
    tmp_path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    label: str,
    value: Any,
    detail: str,
) -> None:
    """These three escaped as a Python traceback and exit 1, not as a refusal.

    ``load_state_dict(strict=False)`` returns missing and unexpected keys, so the branch
    above worked -- but a tensor at the wrong shape, or an entry that is not a tensor, it
    *raises*, and nothing here caught it. Measured through the installed CLI on a real
    600-step checkpoint with one weight replaced: ``RuntimeError: Error(s) in loading
    state_dict for GPT: size mismatch for blocks.0.attn.q_proj.weight``, rendered with the
    internals of ``torch.nn.Module`` in the frame list and exiting **1**, in a tool whose
    contract is that a bad input file exits **5** with no traceback.
    """
    path = edited_weight(write(tmp_path, model, optimizer), "final_norm.weight", value)

    with pytest.raises(CheckpointCorruptError) as caught:
        load_checkpoint(path).apply_to(GPT(CONFIG))

    assert caught.value.exit_code == ExitCode.CHECKPOINT, label
    assert str(path) in str(caught.value), label
    assert "final_norm.weight" in str(caught.value.details[detail]), (
        f"{label}: details[{detail!r}] should name the weight, not just report that "
        f"something was wrong: {caught.value.details}"
    )


def test_a_weight_at_a_different_dtype_still_loads(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    """The check above stops at what torch refuses, and torch does not refuse a cast.

    Measured: a float64 copy of a real checkpoint loads and generates. Refusing it would
    break a file that works, so this pins the boundary rather than leaving the next person
    to widen the check on the assumption that stricter is safer.
    """
    original = write(tmp_path, model, optimizer)
    weight = torch.load(original, weights_only=False)["model_state"]["final_norm.weight"]
    path = edited_weight(original, "final_norm.weight", weight.to(torch.float64))

    fresh = GPT(CONFIG)
    load_checkpoint(path).apply_to(fresh)

    assert fresh.final_norm.weight.dtype == torch.float32
    assert torch.allclose(fresh.final_norm.weight, weight)


@pytest.mark.parametrize("lost", ["n_layer", "d_model", "rope_theta", "tie_embeddings"])
def test_a_model_key_lost_from_the_file_is_refused_by_name(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer, lost: str
) -> None:
    """Measured on a real checkpoint: the dropped key used to be filled from a default.

    ``n_layer`` gone from a two-layer checkpoint rebuilt an eight-layer model, and the
    only complaint came later, from the weight load, as "the checkpoint's weights do not
    match the model" -- which accuses the weights, the one part of the file that was
    still correct.

    ``rope_theta`` and ``tie_embeddings`` were worse, and are the reason this test
    parametrises over both kinds of key. Neither changes any tensor shape, so every
    weight loaded, nothing was refused at any layer, and the model generated noise
    because the rotary base it was rebuilt with was not the one it was trained with.
    """
    path = write(tmp_path, model, optimizer)
    payload = torch.load(path, weights_only=False)
    del payload["model"][lost]
    torch.save(payload, path)

    with pytest.raises(CheckpointCorruptError) as caught:
        load_checkpoint(path)

    assert lost in str(caught.value), "the refusal has to name the key that is gone"
    assert path.name in str(caught.value), "and the file, since a run holds several"
    assert caught.value.details["section"] == "model"
    assert caught.value.details["missing"] == [lost]


def test_an_unloadable_model_section_stays_a_checkpoint_failure() -> None:
    """The exit code is part of the contract, and the underlying error's is ``USAGE``.

    ``ModelConfig.from_dict`` raises ``ConfigError`` -- exit 2 -- which is right for a
    file the user wrote by hand and wrong for one TrainAI wrote itself. Before the
    strictness went in, this case exited 5 via the weight-load mismatch; letting the
    ``ConfigError`` escape would have quietly moved it to 2 and broken any script
    branching on "the checkpoint is bad".
    """
    assert CheckpointCorruptError("x").exit_code == ExitCode.CHECKPOINT
    assert ExitCode.CHECKPOINT != ExitCode.USAGE


def test_a_model_value_of_the_wrong_type_is_refused_with_the_value(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    """A hand-edited checkpoint is rare; a hand-edited config file loaded by the same
    code path is not. What this replaced was ``'<=' not supported between instances of
    'str' and 'int'`` leaking out of a validator, naming neither the file nor the field.
    """
    path = write(tmp_path, model, optimizer)
    payload = torch.load(path, weights_only=False)
    payload["model"]["n_head"] = "four"
    torch.save(payload, path)

    with pytest.raises(CheckpointCorruptError) as caught:
        load_checkpoint(path)

    message = str(caught.value)
    assert "n_head" in message
    assert '"four"' in message, f"the value to edit has to appear: {message}"
    assert caught.value.details["field"] == "n_head"
    assert caught.value.hint


def test_an_untouched_checkpoint_still_loads(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    """The control: the tests above pass if ``load_checkpoint`` refuses everything."""
    path = write(tmp_path, model, optimizer)

    loaded = load_checkpoint(path)

    assert loaded.model_config.to_dict() == CONFIG.to_dict()
    assert loaded.step == 42


@pytest.mark.parametrize("lost", ["lr", "seed", "batch_size", "schedule"])
def test_a_train_key_lost_from_the_file_is_refused_by_name(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer, lost: str
) -> None:
    """The `train` block fails more quietly than the `model` one, which is why it matters.

    A model rebuilt from the wrong shape eventually fails the weight load. A *run*
    rebuilt from the wrong hyperparameters just resumes, at the wrong values. Measured on
    a real 600-step checkpoint: a dropped ``lr`` resumed at 0.0003 rather than 0.000424,
    a dropped ``batch_size`` at 8 rather than 32, and a dropped ``seed`` at 1234 rather
    than 99 -- which reorders the batches, so the "bitwise identical" resume this
    module's docstring promises was silently untrue and nothing said so.
    """
    path = write(tmp_path, model, optimizer)
    payload = torch.load(path, weights_only=False)
    del payload["train"][lost]
    torch.save(payload, path)

    with pytest.raises(CheckpointCorruptError) as caught:
        load_checkpoint(path)

    assert lost in str(caught.value), "the refusal has to name the key that is gone"
    assert path.name in str(caught.value)
    assert caught.value.details["section"] == "train"
    assert caught.value.details["missing"] == [lost]
    assert caught.value.exit_code == ExitCode.CHECKPOINT


def test_a_train_value_of_the_wrong_type_is_refused_with_the_value(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    """``lr: "fast"`` used to escape as a comparison error naming neither field nor file."""
    path = write(tmp_path, model, optimizer)
    payload = torch.load(path, weights_only=False)
    payload["train"]["lr"] = "fast"
    torch.save(payload, path)

    with pytest.raises(CheckpointCorruptError) as caught:
        load_checkpoint(path)

    message = str(caught.value)
    assert "lr" in message
    assert '"fast"' in message, f"the value to edit has to appear: {message}"
    assert caught.value.details["field"] == "lr"
    assert caught.value.exit_code == ExitCode.CHECKPOINT


def test_a_bad_schedule_keeps_the_message_that_lists_the_choices(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    """``schedule`` is required but not type-checked, on purpose.

    ``__post_init__`` already refuses its value with the three names that are allowed,
    which is strictly more useful than "is not a string". Checking the JSON type first
    would replace that better message with a worse one. What the strictness adds here is
    only the *missing* case, covered above.
    """
    path = write(tmp_path, model, optimizer)
    payload = torch.load(path, weights_only=False)
    payload["train"]["schedule"] = "cosinus"
    torch.save(payload, path)

    with pytest.raises(CheckpointCorruptError) as caught:
        load_checkpoint(path)

    assert "cosinus" in str(caught.value)
    assert "cosine, linear, constant" in (caught.value.hint or "")
    assert caught.value.exit_code == ExitCode.CHECKPOINT


def test_an_untouched_train_block_resumes_on_what_it_recorded(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    """The control for the three above, on the values that were silently reverting.

    Deliberately none of them a default, since a default is what the old behaviour
    substituted -- asserting against ``lr=3e-4`` would pass either way.
    """
    recorded = TrainConfig(
        steps=100, batch_size=2, seq_len=16, lr=4.24e-4, seed=99, schedule="linear"
    )
    path = write(tmp_path, model, optimizer, train_config=recorded)

    train_config = load_checkpoint(path).train_config

    assert (train_config.lr, train_config.seed, train_config.schedule) == (
        4.24e-4,
        99,
        "linear",
    )
    assert (TrainConfig.lr, TrainConfig.seed, TrainConfig.schedule) != (4.24e-4, 99, "linear")


# --------------------------------------------------------------------------- #
# Random state
# --------------------------------------------------------------------------- #
def test_capturing_and_restoring_replays_the_same_numbers() -> None:
    state = capture_rng()
    first = torch.randn(4)

    restore_rng(state)

    assert torch.equal(torch.randn(4), first)


def test_restoring_also_replays_numpy_and_python() -> None:
    import random as python_random

    import numpy as np

    state = capture_rng()
    numbers = (np.random.random(3).tolist(), python_random.random())

    restore_rng(state)

    assert np.random.random(3).tolist() == numbers[0]
    assert python_random.random() == numbers[1]


def test_random_state_survives_a_checkpoint(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    """Without this, a resumed run draws different dropout masks and diverges."""
    path = write(tmp_path, model, optimizer, rng=capture_rng())
    expected = torch.randn(4)

    loaded = load_checkpoint(path)
    assert loaded.rng is not None
    restore_rng(loaded.rng)

    assert torch.equal(torch.randn(4), expected)


def test_a_checkpoint_without_random_state_still_loads(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    """Resume is then approximate rather than exact, which is better than refusing."""
    path = write(tmp_path, model, optimizer, rng=None)

    assert load_checkpoint(path).rng is None


def test_numpy_and_python_random_state_survive_the_file_too(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    """The two streams that changed shape in format version 2, checked through a file.

    The neighbouring test restores from an in-memory :class:`RngState`, which the
    conversion to plain data cannot break -- ``capture_rng`` and ``restore_rng`` would
    still agree with each other even if what they produced were unusable. Only a save
    and a load prove the tensor of 624 words comes back as the ``uint32`` array
    ``np.random.set_state`` insists on.
    """
    import random as python_random

    import numpy as np

    path = write(tmp_path, model, optimizer, rng=capture_rng())
    expected = (np.random.random(3).tolist(), python_random.random())

    loaded = load_checkpoint(path)
    assert loaded.rng is not None
    restore_rng(loaded.rng)

    assert np.random.random(3).tolist() == expected[0]
    assert python_random.random() == expected[1]


def test_a_version_1_shaped_random_state_is_still_restorable() -> None:
    """The tuples version 1 stored, handed to the reader that expects dicts.

    Reaching this through a real file needs ``weights_only=True`` to build a NumPy array,
    which it will not do -- so the shape is fed to ``from_payload`` directly, which is
    the one place a file becomes an :class:`RngState`. It is not a hypothetical shape:
    ``torch.serialization.add_safe_globals`` is public API and process-global, so a
    program that uses TrainAI alongside a library which allowlists NumPy turns the
    version-1 refusal into a successful load. What must not happen then is a ``TypeError``
    out of a converter; restoring the state is both safe and the useful answer.
    """
    import random as python_random

    import numpy as np

    legacy = {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": [],
        "numpy": np.random.get_state(legacy=True),
        "python": python_random.getstate(),
    }
    expected = (np.random.random(3).tolist(), python_random.random())

    state = RngState.from_payload(legacy)
    assert isinstance(state.numpy, dict), "the tuple should be normalised on the way in"
    assert isinstance(state.python, dict)
    restore_rng(state)

    assert np.random.random(3).tolist() == expected[0]
    assert python_random.random() == expected[1]


# --------------------------------------------------------------------------- #
# Loading a checkpoint does not run it
# --------------------------------------------------------------------------- #
# `torch.load` used to be called here with `weights_only=False`, which unpickles
# whatever the file names -- and the file is not always one this machine wrote:
# `trainai finetune --from` takes a path off the command line, `trainai chat` and
# `trainai export` read whatever they are pointed at, and a checkpoint is precisely
# the kind of artifact people copy between machines and post in issue threads.
#
# The whole reason it could not be `True` was one NumPy array in the random state.
# Measured, by allowlisting the refusals of a real checkpoint one at a time: four
# globals, all NumPy's, all reachable from that one array, and with those four allowed
# the rest of the file loaded untouched. So format version 2 stores the same 624 words
# as a tensor and the load is safe.
#
# The change is invisible in the happy path -- the suite passed unaltered when it landed,
# which is why this section exists. It is written to fail from both directions, because
# there are two ways to undo it and they are not the same edit: putting
# `weights_only=False` back is caught by the refusal tests below, and having
# `capture_rng` hand over `np.random.get_state()` unconverted -- a one-line
# simplification that looks obviously right -- is caught by reading the pickle a real
# save produces. `tests/test_conventions.py` holds the third: that the package contains
# exactly one `torch.load`, so a second one cannot be added with the unsafe default.
# --------------------------------------------------------------------------- #
class Detonator:
    """An object whose unpickling needs a global the safe unpickler will not build.

    Not actually dangerous -- ``__reduce__`` returns a harmless call. The dangerous
    version is the same shape with ``os.system`` in it, which is the point: the
    unpickler cannot tell them apart, so it has to refuse both.
    """

    def __reduce__(self) -> tuple[Any, ...]:
        return (dict, ([("detonated", True)],))


def test_a_checkpoint_carrying_an_arbitrary_object_is_refused(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    """The property the whole section exists for, stated as a refusal.

    A well-formed TrainAI checkpoint with one extra field holding a reduced object. It
    loads happily under ``weights_only=False`` -- so under the old code this file was
    read, and its ``__reduce__`` ran, before a single field was validated.
    """
    path = write(tmp_path, model, optimizer)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["metrics"] = {"note": Detonator()}
    torch.save(payload, path)

    assert isinstance(
        torch.load(path, map_location="cpu", weights_only=False)["metrics"]["note"], dict
    ), "the reduce did run under the unpickler this replaced, so the file is a real case"

    with pytest.raises(CheckpointCorruptError) as caught:
        load_checkpoint(path)

    assert caught.value.details["imports"], "the refusal should name what the file wanted"


def test_the_file_a_checkpoint_writes_asks_for_no_numpy(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    """Read straight off the pickle, so it cannot pass by loading successfully here.

    `capture_rng` returning `np.random.get_state(legacy=True)` unconverted is a
    one-line simplification that looks obviously correct and silently reintroduces the
    whole problem. A test that only loads the file would not notice, because this
    machine's torch can load it either way -- what changes is whether it can be loaded
    *safely*.
    """
    path = write(tmp_path, model, optimizer, rng=capture_rng())

    wanted = _imports_wanted(path)

    assert wanted, "a torch file imports its tensor rebuilders, so an empty list is a bug here"
    assert [name for name in wanted if "numpy" in name] == []


def test_a_version_1_checkpoint_is_refused_as_old_rather_than_damaged(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    """The one file this change breaks, and it must not be reported as corruption.

    "The file is truncated or damaged, most likely from an interrupted copy" sends
    somebody looking for a disk problem they do not have. Nothing is wrong with a
    version-1 checkpoint except that reading it means running it.
    """
    import numpy as np

    path = write(tmp_path, model, optimizer, rng=capture_rng())
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["format_version"] = 1
    payload["rng"]["numpy"] = np.random.get_state(legacy=True)
    torch.save(payload, path)

    with pytest.raises(CheckpointIncompatibleError) as caught:
        load_checkpoint(path)

    assert "version 1" in str(caught.value)
    assert "NumPy array" in (caught.value.hint or "")
    assert "truncated" not in (caught.value.hint or "")


def test_a_version_1_checkpoint_with_no_random_state_still_loads(
    tmp_path: Path, model: GPT, optimizer: torch.optim.Optimizer
) -> None:
    """Version 1 is not refused for being version 1. It is refused for what it holds.

    ``save_checkpoint(rng=None)`` wrote a version-1 file with nothing in it the safe
    unpickler objects to, and there is no reason to reject one -- the version gate
    refuses files from a *newer* TrainAI, and this is the other direction.
    """
    path = write(tmp_path, model, optimizer, rng=None)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["format_version"] = 1
    torch.save(payload, path)

    assert load_checkpoint(path).step == 42


def test_a_truncated_file_reports_no_imports_and_reads_as_damage(tmp_path: Path) -> None:
    """Half a checkpoint is not a zip, so there is no pickle to read and nothing to name.

    This is the branch that decides whether the section above degrades gracefully: the
    diagnosis is best-effort, and when it comes back empty the message has to be the
    damage one rather than a claim about what the file wanted.
    """
    whole = tmp_path / "whole.pt"
    torch.save({"format": CHECKPOINT_FORMAT}, whole)
    path = tmp_path / "step-0000001.pt"
    path.write_bytes(whole.read_bytes()[:200])

    assert _imports_wanted(path) == []
    with pytest.raises(CheckpointCorruptError) as caught:
        load_checkpoint(path)
    assert "interrupted" in (caught.value.hint or "")


def test_imports_are_found_in_a_pickle_that_names_them_the_other_way(tmp_path: Path) -> None:
    """``STACK_GLOBAL``, which torch's own files never use and another tool's might.

    Protocol 2 spells an import as one ``GLOBAL`` opcode carrying "module name"; from
    protocol 4 the two strings are pushed and ``STACK_GLOBAL`` joins them. Which one a
    file uses is a property of the protocol it was written at, not of what it contains,
    so a checkpoint-shaped file from another tool can arrive either way -- and the branch
    for it is unreachable through ``torch.save``, which is exactly why it is worth a test
    of its own rather than trusting it to the cases above.

    The two names differ, and that is not a bug in the reader: protocol 2 still writes
    the Python 2 spelling of a builtin, so the same ``dict`` arrives as
    ``__builtin__.dict`` at protocol 2 and ``builtins.dict`` at protocol 5. Pinned here
    because it is the reason the diagnosis matches on a ``torch.nn.`` prefix rather than
    on a table of exact names -- a table would have to carry both spellings of every
    entry, and would silently miss whichever one it was not written against.
    """
    import pickle

    for protocol, expected in ((2, "__builtin__.dict"), (5, "builtins.dict")):
        path = tmp_path / f"proto{protocol}.pt"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("archive/data.pkl", pickle.dumps(Detonator(), protocol=protocol))

        assert _imports_wanted(path) == [expected], f"protocol {protocol} was not read"


def test_a_zip_that_holds_no_pickle_reports_no_imports_and_reads_as_damage(
    tmp_path: Path,
) -> None:
    """A zip, but not a torch one: there is no pickle in it to read.

    `test_a_truncated_file_reports_no_imports_and_reads_as_damage` covers the file that
    is not a zip at all. This is the next step in: the archive opens, the member list is
    readable, and nothing in it ends in data.pkl -- somebody's own zip renamed to .pt, or
    a torch file whose members were repacked. The diagnosis has nothing to say and has to
    say nothing, rather than guessing from the member names.
    """
    path = tmp_path / "step-0000001.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/version", "3\n")

    assert _imports_wanted(path) == []
    with pytest.raises(CheckpointCorruptError) as caught:
        load_checkpoint(path)

    assert "interrupted" in (caught.value.hint or "")
    assert caught.value.details["imports"] == []
    assert "data.pkl" in caught.value.details["reason"]


def test_a_member_whose_bytes_were_overwritten_is_damage_and_not_a_crash(
    tmp_path: Path,
) -> None:
    """The archive is intact enough to open and the member is not intact enough to read.

    A file overwritten in place -- an interrupted copy over an existing checkpoint, a
    sync that wrote half -- keeps the central directory at its tail, so `is_zipfile`
    says yes and the member is listed at its original size. The stored bytes no longer
    match the recorded CRC, and `read` raises rather than returning them.

    The point is that the diagnosis is *best effort*: it exists to make the refusal
    message better, so it must never be the thing that fails. Without the handler this
    file leaves `load_checkpoint` as a raw `zipfile.BadZipFile` from inside an exception
    handler, which is both the wrong type for a caller to catch and a worse message than
    the one it was already about to print.

    The member is a pickle that *does* name an import, so that this test cannot pass by
    accident: if the overwrite ever stopped landing on the payload the read would succeed
    and the assertion below would see `["__main__.Thing"]` rather than the empty list a
    swallowed failure gives.
    """
    payload = b"c__main__\nThing\n."
    path = tmp_path / "step-0000001.pt"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", payload)
    raw = bytearray(path.read_bytes())
    at = raw.find(payload)
    raw[at : at + 8] = b"\xff" * 8
    path.write_bytes(bytes(raw))

    assert zipfile.is_zipfile(path), "the archive stopped being openable, so this proves nothing"
    assert _imports_wanted(path) == []
    with pytest.raises(CheckpointCorruptError):
        load_checkpoint(path)


def test_a_pickle_that_stops_mid_opcode_still_names_what_it_read(tmp_path: Path) -> None:
    """Half a pickle: the imports before the break are worth reporting, and are reported.

    `pickletools.genops` walks opcodes one at a time and raises when the data runs out
    mid-argument, so a truncated pickle raises *after* yielding the opcodes it did read.
    Returning nothing at that point would throw away the only useful thing about a
    damaged file -- the name of what it was going to import, which is what tells someone
    whether they have a broken checkpoint or a file that was never one.

    `Thing` here is not importable and never gets imported: `genops` reads the opcode's
    argument as text and executes nothing, which is the whole reason this diagnosis is
    safe to run on a file that was just refused for being unsafe.
    """
    path = tmp_path / "step-0000001.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", b"c__main__\nThing\n\x80")

    assert _imports_wanted(path) == ["__main__.Thing"]
    with pytest.raises(CheckpointCorruptError) as caught:
        load_checkpoint(path)

    assert caught.value.details["imports"] == ["__main__.Thing"]
