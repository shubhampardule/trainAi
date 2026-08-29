"""Tests for :mod:`trainai.model.config`.

The claim under test is that :attr:`ModelConfig.parameter_count` is arithmetic, not
an approximation. Everything downstream depends on it: ``train --dry-run`` reports
it without building a model, and M3's planner will decide what fits in measured
VRAM from it. So it is checked against ``sum(p.numel() for p in model.parameters())``
across every preset and every combination of weight tying and grouped-query
attention.

The rest is validation: a shape that cannot be built has to be refused here, with
the arithmetic that makes it impossible, rather than surfacing as a tensor-shape
error inside an attention kernel.
"""

from __future__ import annotations

import json
from typing import get_args, get_type_hints

import pytest

from trainai.errors import ConfigError
from trainai.model.config import (
    FFN_MULTIPLE,
    PRESETS,
    ModelConfig,
    preset,
)
from trainai.model.gpt import GPT


def real_parameter_count(config: ModelConfig) -> int:
    """What the built module actually allocates. ``parameters()`` de-duplicates ties."""
    return sum(p.numel() for p in GPT(config).parameters())


# --------------------------------------------------------------------------- #
# The parameter count is exact
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", [p.name for p in PRESETS])
def test_preset_parameter_count_matches_the_built_model(name: str) -> None:
    config = preset(name, vocab_size=512, seq_len=64)

    assert config.parameter_count == real_parameter_count(config)


@pytest.mark.parametrize("tie", [True, False])
@pytest.mark.parametrize("n_kv_head", [8, 4, 2, 1])
def test_parameter_count_is_exact_for_tying_and_grouped_queries(tie: bool, n_kv_head: int) -> None:
    """Two features that change the count in opposite directions; both are counted."""
    config = ModelConfig(
        vocab_size=333,
        n_layer=3,
        n_head=8,
        d_model=64,
        seq_len=32,
        n_kv_head=n_kv_head,
        tie_embeddings=tie,
    )

    assert config.parameter_count == real_parameter_count(config)


def test_parameter_count_is_exact_for_an_explicit_ffn_width() -> None:
    config = ModelConfig(vocab_size=100, n_layer=2, n_head=2, d_model=32, seq_len=16, d_ff=97)

    assert config.ffn_dim == 97
    assert config.parameter_count == real_parameter_count(config)


def test_weight_tying_saves_exactly_one_embedding_matrix() -> None:
    shared = {"vocab_size": 1000, "n_layer": 2, "n_head": 4, "d_model": 64, "seq_len": 32}
    tied = ModelConfig(**shared, tie_embeddings=True)
    untied = ModelConfig(**shared, tie_embeddings=False)

    assert untied.parameter_count - tied.parameter_count == 1000 * 64


def test_breakdown_sums_to_the_total() -> None:
    for name in (p.name for p in PRESETS):
        config = preset(name, vocab_size=777, seq_len=64)
        assert sum(config.breakdown().values()) == config.parameter_count, name


def test_non_embedding_count_excludes_the_lookup_tables() -> None:
    """The headline number inflates with vocabulary; this one does not."""
    small_vocab = ModelConfig(vocab_size=256, n_layer=4, n_head=4, d_model=128, seq_len=64)
    big_vocab = ModelConfig(vocab_size=32_000, n_layer=4, n_head=4, d_model=128, seq_len=64)

    assert big_vocab.parameter_count > small_vocab.parameter_count
    assert big_vocab.non_embedding_parameter_count == small_vocab.non_embedding_parameter_count


def test_parameter_bytes_is_four_per_parameter_by_default() -> None:
    config = ModelConfig(vocab_size=100, n_layer=1, n_head=2, d_model=16, seq_len=8)

    assert config.parameter_bytes() == config.parameter_count * 4
    assert config.parameter_bytes(bytes_per_parameter=2) == config.parameter_count * 2


# --------------------------------------------------------------------------- #
# Derived shapes
# --------------------------------------------------------------------------- #
def test_ffn_width_is_rounded_up_to_a_multiple() -> None:
    """Kept divisible so every matmul stays on the tensor-core fast path."""
    config = ModelConfig(vocab_size=100, n_layer=1, n_head=4, d_model=384, seq_len=32)

    assert config.ffn_dim % FFN_MULTIPLE == 0
    assert config.ffn_dim >= 8 / 3 * config.d_model


def test_kv_heads_default_to_no_grouping() -> None:
    config = ModelConfig(vocab_size=100, n_layer=1, n_head=6, d_model=48, seq_len=16)

    assert config.kv_heads == 6
    assert config.kv_groups == 1
    assert not config.uses_grouped_query_attention


def test_grouped_query_attention_is_reported() -> None:
    config = ModelConfig(vocab_size=100, n_layer=1, n_head=8, d_model=64, seq_len=16, n_kv_head=2)

    assert config.uses_grouped_query_attention
    assert config.kv_groups == 4


def test_head_dim_divides_the_width() -> None:
    config = ModelConfig(vocab_size=100, n_layer=1, n_head=8, d_model=512, seq_len=16)

    assert config.head_dim == 64
    assert config.head_dim * config.n_head == config.d_model


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_width_not_divisible_by_heads_suggests_a_value_that_works() -> None:
    with pytest.raises(ConfigError) as caught:
        ModelConfig(vocab_size=100, n_layer=1, n_head=7, d_model=100, seq_len=16)

    hint = caught.value.hint or ""
    assert "n_head" in hint
    suggested = int(hint.split("n_head ")[1].split()[0])
    # The suggestion has to actually be buildable, or it is not a hint.
    ModelConfig(vocab_size=100, n_layer=1, n_head=suggested, d_model=100, seq_len=16)


def test_an_odd_head_dimension_is_refused_because_rope_rotates_pairs() -> None:
    with pytest.raises(ConfigError) as caught:
        ModelConfig(vocab_size=100, n_layer=1, n_head=2, d_model=6, seq_len=16)

    assert "even" in (caught.value.hint or "")


def test_more_kv_heads_than_query_heads_is_refused() -> None:
    with pytest.raises(ConfigError) as caught:
        ModelConfig(vocab_size=100, n_layer=1, n_head=4, d_model=64, seq_len=16, n_kv_head=8)

    assert caught.value.details["n_kv_head"] == 8


def test_kv_heads_that_do_not_divide_evenly_are_refused_with_a_divisor() -> None:
    with pytest.raises(ConfigError) as caught:
        ModelConfig(vocab_size=100, n_layer=1, n_head=6, d_model=48, seq_len=16, n_kv_head=4)

    hint = caught.value.hint or ""
    suggested = int(hint.split("n_kv_head ")[1].rstrip("."))
    assert 6 % suggested == 0
    ModelConfig(vocab_size=100, n_layer=1, n_head=6, d_model=48, seq_len=16, n_kv_head=suggested)


@pytest.mark.parametrize(
    "field",
    ["vocab_size", "n_layer", "n_head", "d_model", "seq_len"],
)
def test_zero_is_refused_for_every_size(field: str) -> None:
    settings = {"vocab_size": 100, "n_layer": 2, "n_head": 4, "d_model": 64, "seq_len": 16}
    settings[field] = 0

    with pytest.raises(ConfigError) as caught:
        ModelConfig(**settings)

    assert field in str(caught.value)


@pytest.mark.parametrize("dropout", [-0.1, 1.0, 1.5])
def test_dropout_outside_the_unit_interval_is_refused(dropout: float) -> None:
    with pytest.raises(ConfigError):
        ModelConfig(vocab_size=100, n_layer=1, n_head=2, d_model=16, seq_len=8, dropout=dropout)


def test_every_config_error_carries_a_hint() -> None:
    base = {"vocab_size": 10, "n_layer": 1, "n_head": 2, "d_model": 16, "seq_len": 8}
    bad_settings = [
        {**base, "vocab_size": 0},
        {**base, "n_head": 3},
        {**base, "d_model": 6},
        {**base, "dropout": 2.0},
        {**base, "rope_theta": 0},
        {**base, "norm_eps": 0},
        {**base, "n_kv_head": 0},
        {**base, "d_ff": 0},
    ]
    for settings in bad_settings:
        with pytest.raises(ConfigError) as caught:
            ModelConfig(**settings)
        assert caught.value.hint, settings


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
def test_to_dict_resolves_the_derived_shapes() -> None:
    """A checkpoint has to be readable by a build whose defaults for them differ."""
    config = ModelConfig(vocab_size=100, n_layer=2, n_head=4, d_model=64, seq_len=32)

    payload = config.to_dict()

    assert payload["n_kv_head"] == config.kv_heads
    assert payload["d_ff"] == config.ffn_dim
    assert payload["head_dim"] == config.head_dim
    assert payload["parameter_count"] == config.parameter_count


def test_from_dict_round_trips_the_resolved_shape() -> None:
    """Idempotent on the resolved form, which is what a checkpoint stores.

    Not field-for-field: ``to_dict`` deliberately writes concrete values where the
    dataclass held ``None``, so a build whose default for ``d_ff`` or ``n_kv_head``
    differs still reads the shape the weights were made for.
    """
    for name in (p.name for p in PRESETS):
        original = preset(name, vocab_size=999, seq_len=128, dropout=0.05)
        rebuilt = ModelConfig.from_dict(original.to_dict())

        assert rebuilt.to_dict() == original.to_dict()
        assert rebuilt.parameter_count == original.parameter_count
        assert rebuilt.describe() == original.describe()


def test_to_dict_makes_the_optional_shapes_concrete() -> None:
    """The behaviour the round-trip test relies on, stated on its own."""
    original = ModelConfig(vocab_size=100, n_layer=2, n_head=4, d_model=64, seq_len=32)
    assert original.n_kv_head is None
    assert original.d_ff is None

    rebuilt = ModelConfig.from_dict(original.to_dict())

    assert rebuilt.n_kv_head == original.kv_heads
    assert rebuilt.d_ff == original.ffn_dim
    assert rebuilt != original  # different fields...
    assert rebuilt.to_dict() == original.to_dict()  # ...same shape


def test_from_dict_ignores_the_derived_extras() -> None:
    config = ModelConfig(vocab_size=100, n_layer=2, n_head=4, d_model=64, seq_len=32)
    payload = config.to_dict()
    payload["something_a_later_version_added"] = 17

    assert ModelConfig.from_dict(payload).to_dict() == config.to_dict()


def test_from_dict_without_a_vocab_size_says_the_record_is_incomplete() -> None:
    with pytest.raises(ConfigError) as caught:
        ModelConfig.from_dict({"n_layer": 2})

    assert "vocab_size" in str(caught.value)
    assert caught.value.hint


# --------------------------------------------------------------------------- #
# from_dict is strict, because guessing rebuilds a different model
# --------------------------------------------------------------------------- #
def required_serialised_fields() -> list[str]:
    """Every field a serialised configuration must carry.

    Derived from the dataclass rather than listed, so a field added later is covered
    here without anyone remembering to add it -- and derived from the *dataclass*
    rather than from ``from_dict``'s own helper, so that a mistake in that helper's
    reasoning fails this test instead of agreeing with it.
    """
    hints = get_type_hints(ModelConfig)
    return sorted(
        name for name in ModelConfig.__dataclass_fields__ if type(None) not in get_args(hints[name])
    )


@pytest.mark.parametrize("missing", required_serialised_fields())
def test_from_dict_refuses_a_missing_field_rather_than_defaulting(missing: str) -> None:
    """A dropped key used to fall through to the dataclass default, silently.

    A checkpoint of a four-layer model whose ``n_layer`` had been dropped rebuilt as
    eight layers. Loading its weights then failed with "the checkpoint's weights do not
    match the model", which points at the wrong half of the problem. The keys that
    change no tensor shape at all were worse: a missing ``rope_theta`` or ``norm_eps``
    loaded without complaint and the model generated noise, because the rotary base it
    was trained with is not the one it was rebuilt with.
    """
    payload = preset("tiny", vocab_size=512).to_dict()
    del payload[missing]

    with pytest.raises(ConfigError) as caught:
        ModelConfig.from_dict(payload)

    assert missing in str(caught.value), f"the refusal did not name {missing}"
    assert f"model configuration is missing {missing}." in str(caught.value), (
        "the refusal has to name both the block and the key, with no list punctuation "
        "for a single name"
    )
    assert caught.value.details["missing"] == [missing]
    assert caught.value.hint


def test_from_dict_names_every_missing_field_at_once() -> None:
    """Reporting them one per attempt makes repairing a file a guessing loop."""
    payload = preset("tiny", vocab_size=512).to_dict()
    for name in ("n_layer", "seq_len", "rope_theta"):
        del payload[name]

    with pytest.raises(ConfigError) as caught:
        ModelConfig.from_dict(payload)

    assert caught.value.details["missing"] == ["n_layer", "rope_theta", "seq_len"]
    assert "n_layer, rope_theta and seq_len" in str(caught.value)


#: Each case is (field, value, what the message must say it is not).
WRONG_TYPES: list[tuple[str, object, str]] = [
    ("n_layer", "four", "an integer"),
    ("n_layer", None, "an integer"),
    # bool is an int subclass, so an isinstance check alone accepts true as a width.
    ("n_layer", True, "an integer"),
    ("d_model", [64], "an integer"),
    # Accepted before, and failed hundreds of lines later inside a slice.
    ("seq_len", 64.5, "an integer"),
    ("dropout", "none", "a number"),
    # Accepted before as *true*, because a non-empty string is truthy -- so a file
    # saying "no" produced a configuration saying yes.
    ("tie_embeddings", "no", "a boolean"),
    ("tie_embeddings", 1, "a boolean"),
    ("n_kv_head", "two", "an integer"),
]


@pytest.mark.parametrize(
    "field_name,value,expected", WRONG_TYPES, ids=[f"{f}={v!r}" for f, v, _ in WRONG_TYPES]
)
def test_from_dict_refuses_a_field_of_the_wrong_type(
    field_name: str, value: object, expected: str
) -> None:
    """Named, with the offending value, in the file's own syntax.

    What this replaced was ``'<=' not supported between instances of 'str' and 'int'``
    -- a Python operator error, leaked from a validator, naming no field at all.

    The message also has to say *which* block is wrong. The checker is shared with
    :class:`TrainConfig`, and a plan file holds one of each, so "the configuration's
    n_head" would send an editor looking in two places.
    """
    payload = {**preset("tiny", vocab_size=512).to_dict(), field_name: value}

    with pytest.raises(ConfigError) as caught:
        ModelConfig.from_dict(payload)

    message = str(caught.value)
    assert f"model configuration's {field_name}" in message, message
    assert f"not {expected}" in message, message
    assert json.dumps(value) in message, f"the value itself is what has to be edited: {message}"
    assert caught.value.details["field"] == field_name
    assert caught.value.hint


def test_the_hint_for_a_boolean_names_the_two_literals() -> None:
    """The message says what the value is not; the hint has to say what to write.

    "Set tie_embeddings to a boolean" leaves someone who then types ``"true"`` -- a
    string, and truthy -- exactly where they started.
    """
    payload = {**preset("tiny", vocab_size=512).to_dict(), "tie_embeddings": "no"}

    with pytest.raises(ConfigError) as caught:
        ModelConfig.from_dict(payload)

    assert "not a boolean" in str(caught.value)
    assert "true or false" in (caught.value.hint or "")


@pytest.mark.parametrize("raw", ["vocab_size is here", [1, 2, 3], 7, None, True])
def test_from_dict_refuses_something_that_is_not_an_object(raw: object) -> None:
    """``"vocab_size" not in raw`` was a *substring* test when ``raw`` was a string.

    So a string containing the words passed the completeness guard and reached
    ``raw.items()``, raising ``AttributeError``: a traceback rather than a refusal.
    """
    with pytest.raises(ConfigError) as caught:
        ModelConfig.from_dict(raw)  # type: ignore[arg-type]

    assert "has to be an object" in str(caught.value)
    assert caught.value.hint


def test_from_dict_still_accepts_what_it_should() -> None:
    """The control. Every test above would pass if ``from_dict`` refused everything."""
    config = preset("tiny", vocab_size=512)
    payload = config.to_dict()

    assert ModelConfig.from_dict(payload).to_dict() == payload
    # A field whose type admits None is resolved in __post_init__, so it may be absent.
    assert ModelConfig.from_dict({k: v for k, v in payload.items() if k != "n_kv_head"})
    assert ModelConfig.from_dict({k: v for k, v in payload.items() if k != "d_ff"})
    # A whole number where a float is written: what a person types by hand.
    assert ModelConfig.from_dict({**payload, "rope_theta": 10_000}).rope_theta == 10_000
    # An unknown key is still ignored, which is what lets an older file load.
    assert ModelConfig.from_dict({**payload, "compile_model": True}).to_dict() == payload


def test_to_dict_is_json_safe() -> None:
    import json

    config = preset("small", vocab_size=8192)

    assert json.loads(json.dumps(config.to_dict()))["n_layer"] == config.n_layer


def test_describe_names_the_shape() -> None:
    config = ModelConfig(
        vocab_size=8192, n_layer=6, n_head=8, d_model=512, seq_len=1024, n_kv_head=2
    )

    described = config.describe()

    assert "L6" in described
    assert "d512" in described
    assert "2kv" in described
    assert "ctx1024" in described


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #
def test_presets_are_ordered_smallest_first() -> None:
    counts = [preset(p.name, vocab_size=512, seq_len=64).parameter_count for p in PRESETS]

    assert counts == sorted(counts)


def test_every_preset_builds() -> None:
    for entry in PRESETS:
        config = preset(entry.name, vocab_size=512, seq_len=64)
        assert config.parameter_count > 0
        assert config.head_dim % 2 == 0, entry.name


def test_an_unknown_preset_lists_the_ones_that_exist() -> None:
    with pytest.raises(ConfigError) as caught:
        preset("enormous", vocab_size=512)

    assert "tiny" in (caught.value.hint or "")
    assert caught.value.details["available"]


def test_preset_overrides_win() -> None:
    config = preset("tiny", vocab_size=512, n_layer=9, dropout=0.3)

    assert config.n_layer == 9
    assert config.dropout == 0.3
