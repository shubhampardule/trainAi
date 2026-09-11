"""Tests for :mod:`trainai.train.schedule` and :mod:`trainai.train.config`.

The schedule is a pure function of the step number, and these tests are what makes
that worth having. A stateful scheduler's ``step()`` counter can be advanced once
too many or too few times across a resume, which silently shifts the entire
remaining schedule; there is nothing here that can drift, so the tests assert the
shape of the curve rather than the correctness of a counter.

The configuration tests are all about refusals. Each one corresponds to a setting
that would produce a run which appears to work and does not: a warmup longer than
the run never reaches the peak learning rate, and a schedule extrapolated past its
end goes negative.
"""

from __future__ import annotations

import itertools
from typing import get_args, get_type_hints

import pytest

from trainai.errors import ConfigError
from trainai.train.config import MIN_WARMUP_STEPS, TrainConfig
from trainai.train.schedule import LearningRateSchedule, lr_at

CURVE = {"peak_lr": 1e-3, "total_steps": 1000, "warmup_steps": 100, "min_lr": 1e-4}


# --------------------------------------------------------------------------- #
# Warmup
# --------------------------------------------------------------------------- #
def test_the_first_step_is_not_at_zero_learning_rate() -> None:
    """A first step with no learning rate is a wasted forward and backward pass."""
    assert lr_at(0, **CURVE) > 0


def test_warmup_rises_monotonically_to_the_peak() -> None:
    rates = [lr_at(step, **CURVE) for step in range(100)]

    assert rates == sorted(rates)
    assert len(set(rates)) == len(rates)
    assert rates[-1] == pytest.approx(CURVE["peak_lr"])


def test_warmup_is_linear() -> None:
    quarter = lr_at(24, **CURVE)
    half = lr_at(49, **CURVE)

    assert half == pytest.approx(2 * quarter, rel=1e-9)


def test_zero_warmup_starts_at_the_peak() -> None:
    assert lr_at(0, **{**CURVE, "warmup_steps": 0}) == pytest.approx(CURVE["peak_lr"])


# --------------------------------------------------------------------------- #
# Decay
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("schedule", ["cosine", "linear"])
def test_decay_is_monotonic_and_lands_on_the_floor(schedule: str) -> None:
    rates = [lr_at(step, **CURVE, schedule=schedule) for step in range(100, 1000)]

    assert all(a >= b for a, b in itertools.pairwise(rates))
    assert rates[-1] == pytest.approx(CURVE["min_lr"], abs=2e-6)


def test_cosine_spends_longer_near_the_peak_than_linear() -> None:
    """Which is the reason to prefer it: more steps at a useful learning rate."""
    step = 300

    assert lr_at(step, **CURVE) > lr_at(step, **CURVE, schedule="linear")


def test_linear_halfway_through_decay_is_halfway_down() -> None:
    midpoint = lr_at(550, **CURVE, schedule="linear")

    assert midpoint == pytest.approx((CURVE["peak_lr"] + CURVE["min_lr"]) / 2, abs=2e-5)


def test_constant_stays_at_the_peak_after_warmup() -> None:
    for step in (100, 500, 999, 5000):
        assert lr_at(step, **CURVE, schedule="constant") == CURVE["peak_lr"]


def test_past_the_end_it_holds_the_floor_rather_than_going_negative() -> None:
    """A run extended past its planned length keeps training, it does not invert."""
    for step in (1000, 1001, 10_000):
        assert lr_at(step, **CURVE) == pytest.approx(CURVE["min_lr"])


def test_a_negative_step_is_a_trainai_bug() -> None:
    with pytest.raises(ConfigError) as caught:
        lr_at(-1, **CURVE)

    assert "bug" in (caught.value.hint or "").lower()


def test_the_same_step_always_gives_the_same_rate() -> None:
    """The property that makes resume exact. No state, so nothing to restore."""
    assert lr_at(437, **CURVE) == lr_at(437, **CURVE)


# --------------------------------------------------------------------------- #
# The schedule object
# --------------------------------------------------------------------------- #
def test_the_schedule_object_matches_the_function() -> None:
    config = TrainConfig(steps=1000, lr=1e-3, min_lr_ratio=0.1, warmup_steps=100, schedule="cosine")
    schedule = LearningRateSchedule(config)

    for step in (0, 50, 99, 100, 500, 999, 2000):
        assert schedule(step) == lr_at(step, **CURVE)


def test_the_schedule_reports_where_the_peak_is() -> None:
    schedule = LearningRateSchedule(TrainConfig(steps=1000, warmup_steps=100))

    assert schedule.peak_step() == 99


def test_the_schedule_describes_itself() -> None:
    described = LearningRateSchedule(TrainConfig(steps=1000, warmup_steps=100)).describe()

    assert "cosine" in described
    assert "100" in described


def test_a_constant_schedule_describes_the_peak_as_after_warmup() -> None:
    """The one ``describe`` sentence with no decay in it.

    ``test_the_schedule_describes_itself`` exercises the default cosine branch of
    ``LearningRateSchedule.describe``, which is the only branch every other test
    reaches. A ``constant`` schedule takes the other one on purpose: there is no
    ``min_lr`` to slide toward, so the sentence names the peak and the warmup and
    says nothing about the steps past it -- the ``from ... to ... over ...`` form
    describes a fall that is not going to happen.

    The string is pinned exactly rather than by substring, because the whole point
    of the branch is that it is a different sentence, not the same one with a word
    swapped.
    """
    config = TrainConfig(steps=1000, warmup_steps=100, lr=5e-4, schedule="constant")
    schedule = LearningRateSchedule(config)

    expected = f"constant {config.lr:g} after {config.resolved_warmup_steps} warmup steps"
    assert schedule.describe() == expected


# --------------------------------------------------------------------------- #
# Configuration: derived values
# --------------------------------------------------------------------------- #
def test_effective_batch_is_the_product_of_batch_and_accumulation() -> None:
    config = TrainConfig(steps=10, batch_size=4, grad_accum=8, seq_len=128)

    assert config.effective_batch_size == 32
    assert config.tokens_per_step == 32 * 128
    assert config.total_tokens == 32 * 128 * 10


def test_warmup_is_derived_when_not_given_and_never_swallows_the_run() -> None:
    for steps in (50, 200, 1000, 100_000):
        config = TrainConfig(steps=steps)

        assert 0 < config.resolved_warmup_steps < config.steps, steps
        assert config.resolved_warmup_steps <= max(1, steps // 4)


def test_a_short_run_still_gets_a_usable_warmup() -> None:
    """2% of 200 steps is four, which is the same as no warmup at all."""
    config = TrainConfig(steps=200)

    assert config.resolved_warmup_steps >= min(MIN_WARMUP_STEPS, config.steps // 4)


def test_min_lr_is_a_fraction_of_the_peak() -> None:
    config = TrainConfig(steps=10, lr=1e-3, min_lr_ratio=0.25)

    assert config.min_lr == pytest.approx(2.5e-4)


# --------------------------------------------------------------------------- #
# Configuration: refusals
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("label", "settings"),
    [
        ("zero steps", {"steps": 0}),
        ("zero batch", {"steps": 10, "batch_size": 0}),
        ("zero accumulation", {"steps": 10, "grad_accum": 0}),
        ("sequence of one", {"steps": 10, "seq_len": 1}),
        ("zero learning rate", {"steps": 10, "lr": 0.0}),
        ("negative learning rate", {"steps": 10, "lr": -1e-4}),
        ("min lr ratio above one", {"steps": 10, "min_lr_ratio": 1.5}),
        ("negative warmup", {"steps": 10, "warmup_steps": -1}),
        ("warmup equal to steps", {"steps": 10, "warmup_steps": 10}),
        ("warmup beyond steps", {"steps": 10, "warmup_steps": 50}),
        ("beta at one", {"steps": 10, "beta2": 1.0}),
        ("negative decay", {"steps": 10, "weight_decay": -0.1}),
        ("negative clip", {"steps": 10, "grad_clip": -1.0}),
        ("negative eval interval", {"steps": 10, "eval_every": -1}),
        ("unknown precision", {"steps": 10, "precision": "int4"}),
        ("unknown schedule", {"steps": 10, "schedule": "exponential"}),
        ("zero eval batches", {"steps": 10, "eval_batches": 0}),
        ("zero log interval", {"steps": 10, "log_every": 0}),
    ],
)
def test_bad_settings_are_refused_with_an_actionable_hint(label: str, settings: dict) -> None:
    with pytest.raises(ConfigError) as caught:
        TrainConfig(**settings)

    assert caught.value.hint, label
    assert caught.value.details, label


def test_a_warmup_longer_than_the_run_suggests_a_length_that_fits() -> None:
    """Otherwise the peak learning rate is never reached and nothing says so."""
    with pytest.raises(ConfigError) as caught:
        TrainConfig(steps=100, warmup_steps=100)

    hint = caught.value.hint or ""
    suggested = int(hint.split("at most ")[1].split()[0])
    TrainConfig(steps=100, warmup_steps=suggested)


def test_zero_is_accepted_where_it_means_disabled() -> None:
    config = TrainConfig(
        steps=10,
        warmup_steps=0,
        grad_clip=0.0,
        weight_decay=0.0,
        eval_every=0,
        checkpoint_every=0,
        keep_checkpoints=0,
        min_lr_ratio=0.0,
    )

    assert config.resolved_warmup_steps == 0
    assert config.min_lr == 0.0


# --------------------------------------------------------------------------- #
# Configuration: serialisation
# --------------------------------------------------------------------------- #
def test_to_dict_resolves_the_derived_values() -> None:
    config = TrainConfig(steps=1000, batch_size=4, grad_accum=2, seq_len=128)

    payload = config.to_dict()

    assert payload["resolved_warmup_steps"] == config.resolved_warmup_steps
    assert payload["effective_batch_size"] == 8
    assert payload["tokens_per_step"] == 1024
    assert payload["total_tokens"] == 1_024_000


def test_to_dict_keeps_warmup_steps_raw_so_the_round_trip_is_faithful() -> None:
    """An unset warmup must survive to_dict/from_dict as unset.

    Serialising the resolved value instead pins an adaptive default. That is how a
    plan.json ended up carrying warmup_steps=20, so that applying it with an explicit
    --steps 20 failed validation for a warmup nobody had chosen.
    """
    derived = TrainConfig(steps=1000)
    assert derived.warmup_steps is None
    assert derived.to_dict()["warmup_steps"] is None
    assert TrainConfig.from_dict(derived.to_dict()).warmup_steps is None

    # A warmup that *was* chosen still round-trips as itself.
    chosen = TrainConfig(steps=1000, warmup_steps=7)
    assert TrainConfig.from_dict(chosen.to_dict()).warmup_steps == 7

    # And the round trip does not change what the schedule will do.
    for config in (derived, chosen):
        rebuilt = TrainConfig.from_dict(config.to_dict())
        assert rebuilt.resolved_warmup_steps == config.resolved_warmup_steps


def test_from_dict_round_trips_the_resolved_form() -> None:
    original = TrainConfig(steps=500, lr=1e-3, schedule="linear", precision="fp16")

    rebuilt = TrainConfig.from_dict(original.to_dict())

    assert rebuilt.to_dict() == original.to_dict()


def test_from_dict_ignores_fields_it_does_not_know() -> None:
    payload = TrainConfig(steps=10).to_dict()
    payload["a_field_from_the_future"] = True

    assert TrainConfig.from_dict(payload).steps == 10


def test_a_checkpoint_carrying_the_removed_compile_model_field_still_loads() -> None:
    """``compile_model`` was a field that nothing read, so it was removed.

    Checkpoints and plan.json files written before that still carry it. Dropping the
    key is what keeps them loadable; refusing would strand every existing run over a
    flag that never did anything.
    """
    payload = TrainConfig(steps=10).to_dict()
    assert "compile_model" not in payload, "the field is gone; nothing should serialise it"
    payload["compile_model"] = True

    rebuilt = TrainConfig.from_dict(payload)

    assert rebuilt.steps == 10
    assert not hasattr(rebuilt, "compile_model")


def test_to_dict_is_json_safe() -> None:
    import json

    payload = json.loads(json.dumps(TrainConfig(steps=10).to_dict()))

    assert payload["steps"] == 10


def test_describe_names_the_shape_of_the_run() -> None:
    described = TrainConfig(steps=1000, batch_size=4, grad_accum=8, seq_len=256, lr=3e-4).describe()

    assert "1000 steps" in described
    assert "4x8=32" in described
    assert "cosine" in described


# --------------------------------------------------------------------------- #
# Configuration: from_dict is strict, because guessing resumes a different run
# --------------------------------------------------------------------------- #
def required_serialised_fields() -> list[str]:
    """Every field a file has to carry, derived from the dataclass.

    Derived here rather than imported from :mod:`trainai.serialise`, deliberately: a
    mistake in that module's reasoning about which fields are optional should fail this
    test rather than agree with it. A field is optional exactly when its annotation
    admits ``None``, which means ``__post_init__`` resolves it -- for
    :class:`TrainConfig` that is ``warmup_steps`` and nothing else.
    """
    hints = get_type_hints(TrainConfig)
    return sorted(
        name
        for name in TrainConfig.__dataclass_fields__
        if type(None) not in (get_args(hints[name]) or (hints[name],))
    )


def test_only_the_derived_and_the_three_state_fields_are_optional() -> None:
    """A control on the derivation above, so a widened annotation is noticed.

    Two fields, for two different reasons. ``warmup_steps`` is ``None`` when
    ``__post_init__`` should derive it. ``loss_mask`` is genuinely three-valued --
    ``None`` means "apply the dataset's mask if it has one" -- and the annotation being
    ``bool | None`` is also what lets a checkpoint written before masks existed resume:
    a required new field would be refused as missing, so every run trained before this
    release would become unresumable.
    """
    optional = set(TrainConfig.__dataclass_fields__) - set(required_serialised_fields())

    assert optional == {"warmup_steps", "loss_mask"}


@pytest.mark.parametrize("missing", required_serialised_fields())
def test_from_dict_refuses_a_missing_field_rather_than_defaulting(missing: str) -> None:
    """Every recognised key is required, and the refusal names the one that is gone.

    Filling a missing key from the default resumes a run on hyperparameters it was not
    trained with. Measured on a real 600-step checkpoint: dropping ``lr`` resumed at
    0.0003 rather than the recorded 0.000424, dropping ``batch_size`` at 8 rather than
    32, and dropping ``seed`` at 1234 rather than 99 -- which changes the batch order,
    so the bitwise-identical resume the checkpoint module promises was quietly false.
    None of it appeared in the output.
    """
    payload = TrainConfig(steps=600, lr=4.24e-4, batch_size=32, grad_accum=2, seed=99).to_dict()
    del payload[missing]

    with pytest.raises(ConfigError) as caught:
        TrainConfig.from_dict(payload)

    assert missing in caught.value.message
    assert caught.value.message.endswith(f"training configuration is missing {missing}.")
    assert caught.value.details["missing"] == [missing]
    assert "trainai train" in (caught.value.hint or "")


def test_from_dict_names_every_missing_field_at_once() -> None:
    """Naming one key per attempt makes a truncated file a guessing game."""
    payload = TrainConfig(steps=600).to_dict()
    for key in ("lr", "seed", "schedule"):
        del payload[key]

    with pytest.raises(ConfigError) as caught:
        TrainConfig.from_dict(payload)

    assert caught.value.details["missing"] == ["lr", "schedule", "seed"]
    assert "lr, schedule and seed" in caught.value.message


def test_a_config_recorded_before_masks_existed_resumes_deciding_from_the_dataset() -> None:
    """The compatibility this field's annotation buys, asserted rather than assumed.

    ``loss_mask`` is the first field added to ``TrainConfig`` since checkpoints started
    being written, and ``from_dict`` refuses a missing *required* key by design -- so
    getting the annotation wrong would make every checkpoint from before this release
    unresumable, with an error blaming an incomplete file. ``None`` is also the right
    value to land on: it means "apply the dataset's mask if it has one", which is what
    those runs did, because there was no mask to apply.
    """
    payload = TrainConfig(steps=600).to_dict()
    del payload["loss_mask"]

    assert TrainConfig.from_dict(payload).loss_mask is None
    assert TrainConfig.from_dict({**payload, "loss_mask": False}).loss_mask is False
    assert TrainConfig.from_dict({**payload, "loss_mask": True}).loss_mask is True


WRONG_TYPES = [
    ("steps", 100.5, "an integer"),
    ("batch_size", 2.5, "an integer"),
    ("grad_accum", True, "an integer"),
    ("seed", "abc", "an integer"),
    ("eval_every", None, "an integer"),
    ("lr", "fast", "a number"),
    ("min_lr_ratio", "half", "a number"),
    ("device", 5, "a string"),
    ("warmup_steps", "10", "an integer"),
    ("lr", {"peak": 3e-4}, "a number"),
    ("seq_len", [256], "an integer"),
]


@pytest.mark.parametrize(("field", "value", "expected"), WRONG_TYPES)
def test_from_dict_refuses_a_field_of_the_wrong_type(
    field: str, value: object, expected: str
) -> None:
    """The value is quoted back, because the point is to find it in the file.

    These used to reach the validators or the training loop. ``lr: "fast"`` surfaced as
    ``'<=' not supported between instances of 'str' and 'int'``, which names no field at
    all; ``steps: 100.5`` was accepted outright and made ``total_tokens`` a float; and
    ``grad_accum: true`` was accepted as 1, because ``bool`` is an ``int`` subclass.
    """
    payload = {**TrainConfig(steps=600).to_dict(), field: value}

    with pytest.raises(ConfigError) as caught:
        TrainConfig.from_dict(payload)

    assert f"training configuration's {field}" in caught.value.message
    assert caught.value.message.endswith(f"which is not {expected}.")
    assert caught.value.details["field"] == field


def test_a_true_that_should_be_a_count_is_named_as_a_boolean() -> None:
    """``bool`` is an ``int``, so this is the one wrong type isinstance would allow."""
    with pytest.raises(ConfigError) as caught:
        TrainConfig.from_dict({**TrainConfig(steps=600).to_dict(), "grad_accum": True})

    assert "grad_accum is true" in caught.value.message
    assert caught.value.details["found"] == "a boolean"


def test_a_nested_object_where_a_number_belongs_is_named_as_an_object() -> None:
    """The shape a hand-edited plan file actually takes: a value moved one level down.

    ``"lr": {"peak": 0.0003}`` is what someone writes who thinks the peak lives under
    ``lr`` rather than being it. So the message has to do three things: name the field,
    quote what is there so it can be found in the file, and say what it is -- and say it
    in the file's own words, because ``dict`` is not a word that appears in JSON.

    The table above covers the scalar confusions. This is the branch of
    :func:`trainai.errors.json_type_name` that only a container reaches, and every
    caller of that helper elsewhere asks about a value that is *not* an object, which is
    why it had never run against a real error.
    """
    with pytest.raises(ConfigError) as caught:
        TrainConfig.from_dict({**TrainConfig(steps=600).to_dict(), "lr": {"peak": 3e-4}})

    assert 'lr is {"peak": 0.0003}' in caught.value.message
    assert caught.value.message.endswith("which is not a number.")
    assert caught.value.details["found"] == "an object"


@pytest.mark.parametrize("raw", [None, "cosine", 7, [1, 2], True])
def test_from_dict_refuses_something_that_is_not_an_object(raw: object) -> None:
    """A `train` block of the wrong shape has no ``.items``, and would traceback."""
    with pytest.raises(ConfigError) as caught:
        TrainConfig.from_dict(raw)  # type: ignore[arg-type]

    assert "has to be an object" in caught.value.message
    assert "training configuration" in (caught.value.hint or "")


def test_from_dict_still_accepts_what_it_should() -> None:
    """The control: strictness that refuses a real file is worse than no strictness.

    Every producer writes ``to_dict()``, so a complete round trip is the case that has
    to keep working -- including the integer where a float is annotated, which is what
    a person types and what JSON gives back for ``1.0``.
    """
    original = TrainConfig(
        steps=600, lr=4.24e-4, batch_size=32, grad_accum=2, seed=99, schedule="linear"
    )

    assert TrainConfig.from_dict(original.to_dict()) == original
    assert TrainConfig.from_dict({**original.to_dict(), "grad_clip": 1}).grad_clip == 1
    assert TrainConfig.from_dict({**original.to_dict(), "warmup_steps": None}).warmup_steps is None

    # An optional key may also be absent outright, not merely null: that is what makes
    # it optional, and requiring it would refuse a file nobody wrote wrongly.
    without = {k: v for k, v in original.to_dict().items() if k != "warmup_steps"}
    assert TrainConfig.from_dict(without).warmup_steps is None
