"""Tests for :mod:`trainai.export`.

Two tests carry this module.

``test_transformers_loads_the_export_and_agrees_on_the_logits`` is the only check that
tests what the export *means* rather than what it contains. A wrong tensor name, a
swapped rotary convention, a transposed gate/up pair or a wrong ``rope_theta`` all
produce a directory that loads without complaint and generates plausible nonsense, and
no amount of key-by-key assertion catches that.

``test_a_wrong_config_field_is_caught_by_the_parity_check`` is its control. A
verification that passes is only evidence if it can fail, so that test corrupts one
config field the shapes do not depend on and asserts the export is refused. Without it,
the parity check could be comparing something to itself and nobody would know.

The rest is the promise that an export is either complete or absent: a failed check has
to leave the destination untouched, and ``--force`` must not be able to delete a
directory that is not an export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from trainai.errors import ExportError, UsageError
from trainai.export import export_run
from trainai.export import hf as hf_layout
from trainai.model.config import ModelConfig
from trainai.model.gpt import GPT


def has_transformers() -> bool:
    """Whether a real HuggingFace load is available in this environment.

    ``transformers`` is deliberately not a dependency of this project, so the parity
    tests skip rather than fail where it is absent -- and the export itself reports
    the check as not run, which is asserted separately.
    """
    from importlib.util import find_spec

    return find_spec("transformers") is not None


needs_transformers = pytest.mark.skipif(
    not has_transformers(),
    reason="transformers is not installed; it is not a dependency of this project",
)


@pytest.fixture(scope="module")
def exported(cli_trained_run: Any, tmp_path_factory: pytest.TempPathFactory) -> Any:
    """One ``hf`` export of the shared run, reused. Nothing here writes to it."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Exported:
        result: Any
        path: Path
        run: Path

    out = tmp_path_factory.mktemp("export") / "hf"
    result = export_run(cli_trained_run.run, out)
    return Exported(result=result, path=out, run=cli_trained_run.run)


# --------------------------------------------------------------------------- #
# What gets written
# --------------------------------------------------------------------------- #
def test_the_export_has_everything_a_loader_needs(exported: Any) -> None:
    names = {item.name for item in exported.result.files}

    assert names == {
        "README.md",
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert all(item.bytes > 0 for item in exported.result.files)


def test_the_config_says_llama_and_matches_the_checkpoint(exported: Any) -> None:
    """A wrong ``model_type`` is the one export mistake nothing downstream can catch."""
    config = json.loads((exported.path / "config.json").read_text(encoding="utf-8"))
    model = exported.result.model

    assert config["architectures"] == ["LlamaForCausalLM"]
    assert config["model_type"] == "llama"
    assert config["num_hidden_layers"] == model["n_layer"]
    assert config["num_attention_heads"] == model["n_head"]
    assert config["hidden_size"] == model["d_model"]
    assert config["max_position_embeddings"] == model["seq_len"]
    assert config["vocab_size"] == model["vocab_size"]
    assert config["rms_norm_eps"] == model["norm_eps"]
    assert config["rope_theta"] == model["rope_theta"]
    assert config["hidden_act"] == "silu", "SwiGLU, not GELU"
    assert config["attention_bias"] is False
    assert config["mlp_bias"] is False


def test_the_config_carries_both_spellings_of_the_dtype(exported: Any) -> None:
    """transformers renamed ``torch_dtype`` to ``dtype`` in v5 and 4.x only knows the old
    one, so an export that names only one of them fails to load on half the versions."""
    config = json.loads((exported.path / "config.json").read_text(encoding="utf-8"))

    assert config["dtype"] == "float32"
    assert config["torch_dtype"] == "float32"


def test_the_readme_says_it_is_a_base_model(exported: Any) -> None:
    """The one place that warning survives the directory being zipped and shared."""
    readme = (exported.path / "README.md").read_text(encoding="utf-8")

    assert "not an instruction-following assistant" in readme
    assert "Provenance" in readme
    assert str(exported.result.source["step"]) in readme


def test_the_readme_does_not_invent_a_reason_for_a_missing_loss(
    cli_trained_run: Any, tmp_path: Path
) -> None:
    """A checkpoint records that the number is absent, not why.

    The README used to state "the run had no validation split" for every absence. The
    commonest cause is a split that exists but is too small for one window at the run's
    --seq-len, and this run's cause is a third thing again: --eval-every 0. The export
    cannot tell them apart from a checkpoint, so it must not name one.
    """
    import sys

    from trainai.errors import ExitCode

    run = tmp_path / "unvalidated"
    argv = sys.argv
    try:
        sys.argv = [
            *("trainai", "train", "--data", str(cli_trained_run.data)),
            *("--out", str(run), "--steps", "2", "--warmup", "1"),
            *("--batch-size", "2", "--seq-len", "32", "--eval-every", "0"),
            *("--layers", "1", "--heads", "2", "--width", "32", "--context", "64"),
            *("--device", "cpu", "--json"),
        ]
        from trainai.cli.main import main

        assert main() == ExitCode.OK
    finally:
        sys.argv = argv

    export_run(run, tmp_path / "hf")
    readme = (tmp_path / "hf" / "README.md").read_text(encoding="utf-8")

    assert "Best validation loss: not measured during training" in readme
    assert "no validation split" not in readme, "a cause the checkpoint cannot know"


def test_the_result_survives_json(exported: Any) -> None:
    payload = json.loads(json.dumps(exported.result.to_dict()))

    assert payload["format"] == "hf"
    assert payload["parameters"] > 0
    assert payload["total_bytes"] == exported.result.total_bytes
    assert len(payload["checks"]) == 3


# --------------------------------------------------------------------------- #
# The name mapping
# --------------------------------------------------------------------------- #
def small_model(**overrides: Any) -> GPT:
    settings: dict[str, Any] = {
        "vocab_size": 128,
        "n_layer": 2,
        "n_head": 4,
        "d_model": 64,
        "seq_len": 32,
    }
    settings.update(overrides)
    return GPT(ModelConfig(**settings))


def test_every_weight_is_mapped_and_none_is_invented() -> None:
    """A dropped tensor produces a file that loads and is wrong, so it is an error."""
    model = small_model()

    mapped = hf_layout.hf_state_dict(model, dtype=torch.float32)

    expected = set(model.state_dict()) - set(model.alias_state_dict_keys)
    assert len(mapped) == len(expected)
    assert all(name.startswith(("model.", "lm_head.")) for name in mapped)
    assert "model.embed_tokens.weight" in mapped
    assert "model.norm.weight" in mapped
    for index in range(model.config.n_layer):
        assert f"model.layers.{index}.self_attn.q_proj.weight" in mapped
        assert f"model.layers.{index}.mlp.gate_proj.weight" in mapped
        assert f"model.layers.{index}.input_layernorm.weight" in mapped
        assert f"model.layers.{index}.post_attention_layernorm.weight" in mapped


def test_a_tied_model_omits_the_output_projection() -> None:
    """safetensors refuses shared storage; ``tie_word_embeddings`` re-creates it."""
    model = small_model(tie_embeddings=True)

    mapped = hf_layout.hf_state_dict(model, dtype=torch.float32)

    assert "lm_head.weight" not in mapped
    assert model.config.tie_embeddings is True


def test_an_untied_model_keeps_the_output_projection() -> None:
    model = small_model(tie_embeddings=False)

    mapped = hf_layout.hf_state_dict(model, dtype=torch.float32)

    assert "lm_head.weight" in mapped
    assert mapped["lm_head.weight"].shape == mapped["model.embed_tokens.weight"].shape
    assert model.alias_state_dict_keys == ()


def test_a_weight_with_no_mapping_stops_the_export(monkeypatch: pytest.MonkeyPatch) -> None:
    """Renaming a module in ``model/gpt.py`` must break loudly, not silently."""
    model = small_model()
    real = model.state_dict()
    extra = {**real, "blocks.0.attn.new_thing.weight": real["blocks.0.attn.q_proj.weight"]}
    monkeypatch.setattr(model, "state_dict", lambda *a, **k: extra)

    with pytest.raises(ExportError) as caught:
        hf_layout.hf_state_dict(model, dtype=torch.float32)

    assert "no HuggingFace equivalent" in str(caught.value)
    assert caught.value.details["unmapped"] == ["blocks.0.attn.new_thing.weight"]


def test_the_tensors_are_detached_copies_not_views() -> None:
    """safetensors writes the whole underlying buffer, so a view exports too much."""
    model = small_model()

    mapped = hf_layout.hf_state_dict(model, dtype=torch.float32)

    for name, tensor in mapped.items():
        assert tensor.is_contiguous(), name
        assert not tensor.requires_grad, name
        assert tensor.storage_offset() == 0, name


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def test_every_check_runs_and_passes_on_a_real_export(exported: Any) -> None:
    by_name = {check.name: check for check in exported.result.checks}

    assert by_name["weights round-trip"].passed
    assert by_name["tokenizer round-trip"].passed
    if has_transformers():
        assert by_name["logits parity"].ran, "transformers is installed, so it must have run"
        assert by_name["logits parity"].passed
    else:
        assert not by_name["logits parity"].ran
        assert "transformers is not installed" in by_name["logits parity"].detail


def test_corrupted_weights_are_caught_and_nothing_is_written(
    cli_trained_run: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The round-trip check exists for exactly this, and the export is atomic.

    Zeroing one tensor on the way to disk is what a serialisation fault looks like
    from here. The destination has to come out of it not existing at all -- a
    half-written export that loads is worse than no export.
    """
    import safetensors.torch as st

    real_save = st.save_file

    def sabotage(tensors: dict[str, torch.Tensor], path: str, **kwargs: Any) -> None:
        broken = dict(tensors)
        broken["model.norm.weight"] = torch.zeros_like(broken["model.norm.weight"])
        real_save(broken, path, **kwargs)

    monkeypatch.setattr(st, "save_file", sabotage)
    out = tmp_path / "broken"

    with pytest.raises(ExportError) as caught:
        export_run(cli_trained_run.run, out)

    assert "failed verification" in str(caught.value)
    assert "model.norm.weight" in json.dumps(caught.value.details)
    assert not out.exists(), "a failed export must not leave a loadable directory behind"
    assert not list(tmp_path.glob("broken.partial-*")), "the staging directory is cleaned up"


def test_no_verify_reports_the_checks_as_not_run(cli_trained_run: Any, tmp_path: Path) -> None:
    """Skipping the checks is allowed; claiming they passed is not."""
    result = export_run(cli_trained_run.run, tmp_path / "quick", verify=False)

    assert result.checks
    assert all(not check.ran for check in result.checks)
    assert all("--no-verify" in check.detail for check in result.checks)


@needs_transformers
def test_transformers_loads_the_export_and_agrees_on_the_logits(exported: Any) -> None:
    """The check the whole format rests on, run again here against our own model.

    The export's own parity check does this too. Repeating it in the test suite is
    not redundant: this asserts the tolerance is actually tight, and it is the test
    that fails if a future refactor loosens the check inside the exporter.
    """
    from transformers import AutoModelForCausalLM

    from trainai.infer import InferenceSession

    session = InferenceSession.open(exported.run, device="cpu")
    loaded = AutoModelForCausalLM.from_pretrained(str(exported.path))
    loaded.eval()
    ids = torch.tensor([[3, 11, 47, 5, 2, 90, 13, 8]], dtype=torch.long)

    with torch.no_grad():
        ours, _loss, _caches = session.model(ids)
        theirs = loaded(ids).logits.float()

    assert ours.shape == theirs.shape
    assert float((ours - theirs).abs().max()) < 1e-4


@needs_transformers
def test_the_export_generates_through_the_transformers_api(exported: Any) -> None:
    """What a user will actually do with the directory."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(exported.path))
    model = AutoModelForCausalLM.from_pretrained(str(exported.path))
    encoded = tokenizer("The river", return_tensors="pt")

    out = model.generate(**encoded, max_new_tokens=8, do_sample=False)

    assert out.shape[1] > encoded["input_ids"].shape[1], "it produced no tokens"
    assert isinstance(tokenizer.decode(out[0]), str)


@needs_transformers
def test_a_wrong_config_field_is_caught_by_the_parity_check(
    cli_trained_run: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control for the parity check: prove it can fail.

    ``rope_theta`` is corrupted rather than a dimension, because every shape still
    matches -- the file loads, generates fluent-looking text, and is wrong. That is
    precisely the class of mistake the logits comparison is there to catch, and a
    check that cannot fail is not evidence of anything.
    """
    from trainai.export import bundle

    real_config = bundle.hf_layout.hf_config

    def wrong_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = real_config(*args, **kwargs)
        payload["rope_theta"] = 500_000.0
        return payload

    monkeypatch.setattr(bundle.hf_layout, "hf_config", wrong_config)
    out = tmp_path / "wrong"

    with pytest.raises(ExportError) as caught:
        export_run(cli_trained_run.run, out)

    assert "failed verification" in str(caught.value)
    assert "logit difference" in json.dumps(caught.value.details)
    assert not out.exists()


# --------------------------------------------------------------------------- #
# The other format
# --------------------------------------------------------------------------- #
def test_the_safetensors_format_keeps_our_own_names(cli_trained_run: Any, tmp_path: Path) -> None:
    from safetensors.torch import load_file

    out = tmp_path / "own"
    result = export_run(cli_trained_run.run, out, export_format="safetensors")

    weights = load_file(str(out / "model.safetensors"))
    payload = json.loads((out / "trainai-model.json").read_text(encoding="utf-8"))

    assert "embed_tokens.weight" in weights
    assert "blocks.0.attn.q_proj.weight" in weights
    assert not any(name.startswith("model.layers") for name in weights)
    assert payload["model"]["n_layer"] == result.model["n_layer"]
    assert payload["source"]["step"] == result.source["step"]
    assert (out / "tokenizer.json").is_file(), "weights without their tokenizer are unusable"


def test_the_parity_check_is_not_claimed_for_the_safetensors_format(
    cli_trained_run: Any, tmp_path: Path
) -> None:
    result = export_run(cli_trained_run.run, tmp_path / "own", export_format="safetensors")

    parity = next(check for check in result.checks if check.name == "logits parity")
    assert not parity.ran
    assert "not applicable" in parity.detail


# --------------------------------------------------------------------------- #
# Precision
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("name", "expected"),
    [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)],
)
def test_the_dtype_reaches_the_file(
    cli_trained_run: Any, tmp_path: Path, name: str, expected: torch.dtype
) -> None:
    from safetensors.torch import load_file

    out = tmp_path / name
    export_run(cli_trained_run.run, out, dtype=name)

    weights = load_file(str(out / "model.safetensors"))
    assert all(tensor.dtype == expected for tensor in weights.values())
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["dtype"] == str(
        expected
    ).removeprefix("torch.")


def test_a_reduced_precision_export_is_half_the_size(cli_trained_run: Any, tmp_path: Path) -> None:
    """The reason anyone passes ``--dtype`` at all."""
    full = export_run(cli_trained_run.run, tmp_path / "fp32", dtype="fp32")
    half = export_run(cli_trained_run.run, tmp_path / "bf16", dtype="bf16")

    weights = {item.name: item.bytes for item in full.files}["model.safetensors"]
    smaller = {item.name: item.bytes for item in half.files}["model.safetensors"]
    assert smaller < weights * 0.55


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "flag"),
    [({"export_format": "onnx"}, "--format"), ({"dtype": "int4"}, "--dtype")],
)
def test_an_unknown_choice_is_refused_before_anything_is_loaded(
    cli_trained_run: Any, tmp_path: Path, kwargs: dict[str, Any], flag: str
) -> None:
    out = tmp_path / "nope"

    with pytest.raises(UsageError) as caught:
        export_run(cli_trained_run.run, out, **kwargs)

    assert flag in str(caught.value)
    assert caught.value.details["choices"]
    assert not out.exists()


def test_a_directory_that_is_not_an_export_is_never_deleted(
    cli_trained_run: Any, tmp_path: Path
) -> None:
    """``--force`` deletes a directory, so it must not reach one the user cares about."""
    occupied = tmp_path / "documents"
    occupied.mkdir()
    (occupied / "thesis.txt").write_text("years of work", encoding="utf-8", newline="\n")

    for force in (False, True):
        with pytest.raises(ExportError) as caught:
            export_run(cli_trained_run.run, occupied, force=force)
        assert "not a TrainAI export" in str(caught.value)

    assert (occupied / "thesis.txt").read_text(encoding="utf-8") == "years of work"


def test_an_existing_export_needs_force(cli_trained_run: Any, tmp_path: Path) -> None:
    out = tmp_path / "again"
    export_run(cli_trained_run.run, out)

    with pytest.raises(ExportError) as caught:
        export_run(cli_trained_run.run, out)
    assert "already holds an export" in str(caught.value)
    assert "--force" in (caught.value.hint or "")

    replaced = export_run(cli_trained_run.run, out, dtype="bf16", force=True)
    assert replaced.dtype == "bf16"


def test_force_leaves_nothing_from_the_previous_export_behind(
    cli_trained_run: Any, tmp_path: Path
) -> None:
    """A replaced export must not inherit stale files from the one it replaced."""
    out = tmp_path / "swap"
    export_run(cli_trained_run.run, out)
    (out / "stale-shard.safetensors").write_text("left over", encoding="utf-8", newline="\n")

    export_run(cli_trained_run.run, out, force=True)

    assert not (out / "stale-shard.safetensors").exists()


def test_a_file_where_the_directory_should_be_says_so(cli_trained_run: Any, tmp_path: Path) -> None:
    blocked = tmp_path / "model.bin"
    blocked.write_text("not a directory", encoding="utf-8", newline="\n")

    with pytest.raises(ExportError) as caught:
        export_run(cli_trained_run.run, blocked)

    assert "not a directory" in str(caught.value)
    assert blocked.read_text(encoding="utf-8") == "not a directory"


def test_an_empty_directory_is_fine_to_export_into(cli_trained_run: Any, tmp_path: Path) -> None:
    """Users make the directory first; that is not a collision."""
    out = tmp_path / "empty"
    out.mkdir()

    result = export_run(cli_trained_run.run, out)

    assert result.out_dir == out
    assert (out / "model.safetensors").is_file()


def test_a_run_that_does_not_exist_says_so(tmp_path: Path) -> None:
    with pytest.raises(UsageError) as caught:
        export_run(tmp_path / "nowhere", tmp_path / "out")

    assert "Nothing at" in str(caught.value)
