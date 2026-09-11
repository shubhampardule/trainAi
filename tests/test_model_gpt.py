"""Tests for :mod:`trainai.model.gpt`.

The one that matters most is causality. A transformer that can see the next token
still trains -- the loss falls beautifully, faster than it should -- and produces a
model that cannot generate anything, because at generation time the future is not
there to look at. The failure is silent, looks like success, and is only obvious
once you try to use the result. So it is tested four ways: by perturbing a later
token and checking earlier outputs do not move, by checking gradients do not flow
backwards in time, by checking that incremental generation with a key/value
cache matches a full forward pass, and by checking the same for a *chunk* of new
tokens against a cache, where the kernel's own causal mask does not apply and one
has to be built.

Then the properties that define the architecture's pieces: that rotary embeddings
make attention scores depend on relative position, that grouped-query attention
points each query head at the right key/value head, that sampling is reproducible
from a seed, and that the tied output projection really is one tensor.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from trainai.errors import ConfigError
from trainai.model.config import ModelConfig
from trainai.model.gpt import GPT, RMSNorm, RotaryEmbedding, apply_rotary

VOCAB = 64


def small_config(**overrides: object) -> ModelConfig:
    settings: dict = {
        "vocab_size": VOCAB,
        "n_layer": 2,
        "n_head": 4,
        "d_model": 32,
        "seq_len": 16,
    }
    settings.update(overrides)
    return ModelConfig(**settings)  # type: ignore[arg-type]


@pytest.fixture
def model() -> GPT:
    torch.manual_seed(0)
    return GPT(small_config()).eval()


@pytest.fixture
def tokens() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, VOCAB, (2, 12))


# --------------------------------------------------------------------------- #
# Causality
# --------------------------------------------------------------------------- #
def test_no_output_position_depends_on_a_later_token(model: GPT, tokens: torch.Tensor) -> None:
    """Change token t; every output before t must be bit-for-bit unchanged."""
    with torch.no_grad():
        baseline, _, _ = model(tokens)

    for position in range(1, tokens.size(1)):
        altered = tokens.clone()
        altered[:, position] = (altered[:, position] + 7) % VOCAB
        with torch.no_grad():
            changed, _, _ = model(altered)

        assert torch.equal(baseline[:, :position], changed[:, :position]), (
            f"editing token {position} changed an earlier output"
        )


def test_editing_a_token_does_change_its_own_output(model: GPT, tokens: torch.Tensor) -> None:
    """The other half of the causality claim: the mask must not mask too much."""
    with torch.no_grad():
        baseline, _, _ = model(tokens)

    for position in range(tokens.size(1)):
        altered = tokens.clone()
        altered[:, position] = (altered[:, position] + 7) % VOCAB
        with torch.no_grad():
            changed, _, _ = model(altered)

        assert not torch.equal(baseline[:, position], changed[:, position]), (
            f"output {position} ignored a change to its own input"
        )


def test_gradient_does_not_flow_backwards_in_time() -> None:
    """A gradient path from position t to position t+1 is a leak the loss hides."""
    torch.manual_seed(0)
    model = GPT(small_config())
    embeddings = model.embed_tokens(torch.randint(0, VOCAB, (1, 8))).detach().requires_grad_(True)

    hidden = embeddings
    cos, sin = model.rotary(hidden.size(1), 0)
    for block in model.blocks:
        hidden, _ = block(hidden, cos, sin, None)
    logits = model.lm_head(model.final_norm(hidden))
    logits[0, 3].sum().backward()

    grad = embeddings.grad
    assert grad is not None
    assert grad[0, :4].abs().max() > 0, "no gradient reached the earlier positions"
    assert grad[0, 4:].abs().max() == 0, "gradient reached a later position"


@pytest.mark.parametrize("n_kv_head", [4, 2, 1])
def test_cached_generation_matches_a_full_forward_pass(n_kv_head: int) -> None:
    """Also covers grouped-query attention, where the cache is shared across heads."""
    torch.manual_seed(0)
    model = GPT(small_config(n_kv_head=n_kv_head)).eval()
    ids = torch.randint(0, VOCAB, (1, 6))

    with torch.no_grad():
        full, _, _ = model(ids)
        _, _, caches = model(ids[:, :-1])
        stepped, _, _ = model(ids[:, -1:], caches=caches, offset=ids.size(1) - 1)

    assert torch.allclose(full[:, -1], stepped[:, -1], atol=1e-5)


@pytest.mark.parametrize("split", [1, 2, 3, 5])
@pytest.mark.parametrize("n_kv_head", [4, 2])
def test_a_prompt_fed_in_chunks_matches_one_pass_over_all_of_it(split: int, n_kv_head: int) -> None:
    """A cache plus *more than one* new token, which is the case a mask is needed for.

    ``is_causal=True`` cannot express it: the flag means "query i may see key i and
    earlier" with both numbered from zero, so it is only right when every key is also
    a query. With a cache the keys start earlier than the queries and that diagonal
    lands in the wrong place, so the code passes neither the flag nor a mask here --
    and unmasked attention let each new token see the ones after it. Measured drift
    from a single pass over the same tokens, before the mask existed: 2.9e-01 in
    logit space.
    """
    torch.manual_seed(0)
    model = GPT(small_config(n_kv_head=n_kv_head)).eval()
    ids = torch.randint(0, VOCAB, (2, 6))

    with torch.no_grad():
        full, _, _ = model(ids)
        _, _, caches = model(ids[:, :split])
        chunked, _, _ = model(ids[:, split:], caches=caches, offset=split)

    assert torch.allclose(full[:, split:], chunked, atol=1e-5), (
        f"splitting the prompt after {split} tokens moved the logits by "
        f"{(full[:, split:] - chunked).abs().max().item():.3e}"
    )


def test_a_later_token_in_a_cached_chunk_cannot_move_an_earlier_output() -> None:
    """The same check as the uncached one, run across a cache boundary."""
    torch.manual_seed(0)
    model = GPT(small_config()).eval()
    ids = torch.randint(0, VOCAB, (1, 8))

    def in_two_chunks(sequence: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, _, caches = model(sequence[:, :4])
            rest, _, _ = model(sequence[:, 4:], caches=caches, offset=4)
        return rest

    baseline = in_two_chunks(ids)

    for position in range(5, ids.size(1)):
        altered = ids.clone()
        altered[:, position] = (altered[:, position] + 7) % VOCAB
        within_chunk = position - 4

        assert torch.equal(baseline[:, :within_chunk], in_two_chunks(altered)[:, :within_chunk]), (
            f"changing token {position} moved an output before it"
        )


def test_a_prompt_fed_in_three_uneven_chunks_matches_one_pass() -> None:
    """The third chunk starts at a non-zero offset with several new tokens, which is
    where the mask's ``past`` term has to be right rather than merely present."""
    torch.manual_seed(0)
    model = GPT(small_config()).eval()
    ids = torch.randint(0, VOCAB, (2, 9))

    with torch.no_grad():
        full, _, _ = model(ids)
        _, _, caches = model(ids[:, :2])
        _, _, caches = model(ids[:, 2:5], caches=caches, offset=2)
        third, _, _ = model(ids[:, 5:], caches=caches, offset=5)

    assert torch.allclose(full[:, 5:], third, atol=1e-5)


def test_the_mask_is_only_built_for_a_chunk_against_a_cache() -> None:
    """Building one always would be correct but would cost the training path its fused
    kernel, so the three cases are pinned: no cache asks for ``is_causal``, one new
    token needs no mask at all, and only a chunk against a cache materialises one."""
    calls: list[tuple[bool, bool]] = []
    real = torch.nn.functional.scaled_dot_product_attention

    def recording(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        calls.append((kwargs.get("attn_mask") is not None, bool(kwargs.get("is_causal"))))
        return real(*args, **kwargs)  # type: ignore[arg-type]

    torch.manual_seed(0)
    model = GPT(small_config()).eval()
    ids = torch.randint(0, VOCAB, (1, 6))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.nn.functional, "scaled_dot_product_attention", recording)
        with torch.no_grad():
            _, _, caches = model(ids)
            no_cache = calls.copy()

            calls.clear()
            model(ids[:, -1:], caches=caches, offset=ids.size(1))
            one_token = calls.copy()

            calls.clear()
            model(ids[:, -2:], caches=[(k[:, :, :-2], v[:, :, :-2]) for k, v in caches], offset=4)
            a_chunk = calls.copy()

    assert no_cache == [(False, True)] * model.config.n_layer, "no cache should use is_causal"
    assert one_token == [(False, False)] * model.config.n_layer, "one token needs no mask"
    assert a_chunk == [(True, False)] * model.config.n_layer, "a chunk needs a real mask"


def test_the_cache_holds_one_entry_per_layer_with_the_kv_head_count() -> None:
    torch.manual_seed(0)
    config = small_config(n_kv_head=2)
    model = GPT(config).eval()

    with torch.no_grad():
        _, _, caches = model(torch.randint(0, VOCAB, (3, 7)))

    assert len(caches) == config.n_layer
    for keys, values in caches:
        assert keys.shape == (3, config.kv_heads, 7, config.head_dim)
        assert values.shape == keys.shape


# --------------------------------------------------------------------------- #
# Grouped-query attention
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_kv_head", [2, 1])
def test_grouped_attention_gives_each_query_head_its_own_key_value_head(
    n_kv_head: int,
) -> None:
    """Query head ``i`` must read key/value head ``i // groups``.

    Nothing else pins this. Every other grouped test compares one path against
    another -- cached against a full pass, chunked against whole -- and both sides
    share whatever mapping the code picked, so a wrong one agrees with itself.
    Swapping ``repeat_interleave`` for ``repeat`` keeps the shape, scrambles which
    key each query reads, and passed all 1280 tests in the suite.

    So the mapping is checked directly, against the model's own tensors: the cache
    holds the keys *before* expansion, so whatever SDPA received must be those
    keys interleaved.
    """
    seen: list[torch.Tensor] = []
    real = torch.nn.functional.scaled_dot_product_attention

    def recording(query, key, value, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(key.clone())
        return real(query, key, value, **kwargs)

    torch.manual_seed(0)
    config = small_config(n_kv_head=n_kv_head)
    model = GPT(config).eval()
    groups = config.n_head // config.kv_heads
    assert groups > 1, "this test is meaningless without grouping"

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.nn.functional, "scaled_dot_product_attention", recording)
        with torch.no_grad():
            _, _, caches = model(torch.randint(0, VOCAB, (2, 5)))

    assert len(seen) == config.n_layer
    for expanded, (unexpanded, _) in zip(seen, caches, strict=True):
        assert expanded.size(1) == config.n_head
        assert unexpanded.size(1) == config.kv_heads
        for head in range(config.n_head):
            assert torch.equal(expanded[:, head], unexpanded[:, head // groups]), (
                f"query head {head} should read key/value head {head // groups}"
            )


def test_grouped_attention_hands_sdpa_matching_head_counts() -> None:
    """The expansion is what keeps SDPA on a fused kernel, so it has to happen here.

    Passing the unexpanded keys with ``enable_gqa`` instead would be numerically
    identical and drop the memory-efficient kernel, which refuses mismatched head
    counts -- measured at +189.2 MiB against +12.0 MiB for one attention call. This
    fails if anyone makes that trade without measuring it again.
    """
    shapes: list[tuple[int, int, bool]] = []
    real = torch.nn.functional.scaled_dot_product_attention

    def recording(query, key, value, **kwargs):  # type: ignore[no-untyped-def]
        shapes.append((query.size(1), key.size(1), bool(kwargs.get("enable_gqa"))))
        return real(query, key, value, **kwargs)

    torch.manual_seed(0)
    config = small_config(n_kv_head=1)
    model = GPT(config).eval()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.nn.functional, "scaled_dot_product_attention", recording)
        with torch.no_grad():
            model(torch.randint(0, VOCAB, (2, 5)))

    assert shapes == [(config.n_head, config.n_head, False)] * config.n_layer


def test_grouped_attention_matches_a_hand_expanded_reference() -> None:
    """Black-box companion to the mapping test: build the expansion by hand from the
    cache and check the whole block's output, so the pin does not rest only on what
    SDPA was handed."""
    torch.manual_seed(0)
    config = small_config(n_kv_head=2)
    model = GPT(config).eval()
    block = model.blocks[0]
    groups = config.n_head // config.kv_heads

    torch.manual_seed(4)
    x = torch.randn(2, 5, config.d_model)
    cos, sin = model.rotary(5)

    with torch.no_grad():
        got, (keys, values) = block.attn(x, cos, sin)

        batch, seq, _ = x.shape
        q = block.attn.q_proj(x).view(batch, seq, config.n_head, config.head_dim).transpose(1, 2)
        q = apply_rotary(q, cos, sin)
        reference = torch.nn.functional.scaled_dot_product_attention(
            q,
            torch.stack([keys[:, h // groups] for h in range(config.n_head)], dim=1),
            torch.stack([values[:, h // groups] for h in range(config.n_head)], dim=1),
            is_causal=True,
        )
        reference = reference.transpose(1, 2).reshape(batch, seq, config.d_model)
        reference = block.attn.o_proj(reference)

    assert torch.allclose(got, reference, atol=1e-6), (
        f"max diff {(got - reference).abs().max().item():.3e}"
    )


# --------------------------------------------------------------------------- #
# Sliding the cache
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_kv_head", [4, 2, 1])
@pytest.mark.parametrize("drop", [1, 3, 7])
def test_sliding_a_cache_renumbers_it_exactly(n_kv_head: int, drop: int) -> None:
    """The first layer's cache after a slide must equal a fresh pass over the window.

    Layer 0's keys are a function of one token and its position only, so the two
    routes are directly comparable there: the rotary rotation composes, and
    renumbering by ``-drop`` is exact rather than an approximation.
    """
    torch.manual_seed(0)
    model = GPT(small_config(n_kv_head=n_kv_head)).eval()
    ids = torch.randint(0, VOCAB, (2, 12))

    with torch.no_grad():
        _, _, grown = model(ids)
        slid = model.slide_caches(grown, drop)
        _, _, fresh = model(ids[:, drop:])

    assert slid[0][0].shape == fresh[0][0].shape
    assert torch.allclose(slid[0][0], fresh[0][0], atol=1e-5), (
        f"slid keys differ from a fresh pass by {(slid[0][0] - fresh[0][0]).abs().max().item():.3e}"
    )
    # Values carry no position, so the promise for them is exact rather than
    # approximate: they are the originals with the front trimmed off, nothing more.
    # Comparing them against the *fresh* pass instead would not be exact -- nn.Linear
    # picks a different kernel for a shorter input and the last bit moves (measured
    # 2.980e-08 at drop=7), which is the GEMM's doing and not the slide's.
    for (_, slid_values), (_, grown_values) in zip(slid, grown, strict=True):
        assert torch.equal(slid_values, grown_values[:, :, drop:])


def test_sliding_keeps_history_that_a_fresh_pass_would_have_lost() -> None:
    """Deliberate, and the reason sliding is not the same as recomputing the window.

    A cached layer-2 key was produced when layer 1 could still attend to tokens that
    have since fallen off the front, so it carries information a fresh pass over the
    surviving window cannot reconstruct. Sliding is strictly better informed. This is
    pinned so that swapping it back to a recompute cannot pass silently.
    """
    torch.manual_seed(0)
    model = GPT(small_config(n_layer=4)).eval()
    ids = torch.randint(0, VOCAB, (1, 12))

    with torch.no_grad():
        _, _, grown = model(ids)
        slid = model.slide_caches(grown, 4)
        _, _, fresh = model(ids[:, 4:])

    assert torch.allclose(slid[0][0], fresh[0][0], atol=1e-5), "layer 0 must still match"
    deepest = (slid[-1][0] - fresh[-1][0]).abs().max().item()
    assert deepest > 1e-4, f"the deepest layer matched to {deepest:.3e}, so history was lost"


def test_repeated_sliding_does_not_accumulate_error() -> None:
    """Sliding one position at a time is what generation does, thousands of times."""
    torch.manual_seed(0)
    model = GPT(small_config(n_layer=1, seq_len=32)).eval()
    ids = torch.randint(0, VOCAB, (1, 20))

    with torch.no_grad():
        _, _, stepwise = model(ids)
        for _ in range(8):
            stepwise = model.slide_caches(stepwise, 1)
        _, _, fresh = model(ids[:, 8:])

    drift = (stepwise[0][0] - fresh[0][0]).abs().max().item()
    assert drift < 1e-5, f"eight single-position slides drifted by {drift:.3e}"


def test_dropping_nothing_returns_the_cache_untouched() -> None:
    torch.manual_seed(0)
    model = GPT(small_config()).eval()
    with torch.no_grad():
        _, _, caches = model(torch.randint(0, VOCAB, (1, 6)))

    assert model.slide_caches(caches, 0) is caches


@pytest.mark.parametrize("drop", [-1, 6, 99])
def test_sliding_off_more_than_the_cache_holds_is_refused(drop: int) -> None:
    torch.manual_seed(0)
    model = GPT(small_config()).eval()
    with torch.no_grad():
        _, _, caches = model(torch.randint(0, VOCAB, (1, 6)))

    with pytest.raises(ConfigError) as caught:
        model.slide_caches(caches, drop)

    assert "cache holding 6" in str(caught.value)
    assert caught.value.details == {"drop": drop, "length": 6}


def test_sliding_without_a_cache_is_refused() -> None:
    torch.manual_seed(0)
    model = GPT(small_config()).eval()

    with pytest.raises(ConfigError) as caught:
        model.slide_caches(None, 1)

    assert "forward pass" in str(caught.value)


# --------------------------------------------------------------------------- #
# Rotary position embeddings
# --------------------------------------------------------------------------- #
def test_position_zero_is_the_identity_rotation() -> None:
    rotary = RotaryEmbedding(head_dim=8, max_seq_len=32)

    cos, sin = rotary(4, 0)

    assert torch.allclose(cos[0], torch.ones(4))
    assert torch.allclose(sin[0], torch.zeros(4))


def test_rotation_preserves_the_norm_of_every_vector() -> None:
    """It is a rotation, so it must not scale anything."""
    rotary = RotaryEmbedding(head_dim=8, max_seq_len=32)
    cos, sin = rotary(4, 0)
    x = torch.randn(2, 3, 4, 8)

    rotated = apply_rotary(x, cos, sin)

    assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5)


def test_attention_score_depends_only_on_relative_position() -> None:
    """The property rotary embeddings exist for.

    The same query and key at positions (0, 2) must score the same as at (5, 7),
    and differently from (0, 5).
    """
    rotary = RotaryEmbedding(head_dim=8, max_seq_len=32)
    cos, sin = rotary(16, 0)
    query = torch.randn(1, 1, 1, 8)
    key = torch.randn(1, 1, 1, 8)

    def score(at_query: int, at_key: int) -> float:
        rotated_q = apply_rotary(query, cos[at_query : at_query + 1], sin[at_query : at_query + 1])
        rotated_k = apply_rotary(key, cos[at_key : at_key + 1], sin[at_key : at_key + 1])
        return float((rotated_q * rotated_k).sum())

    assert score(0, 2) == pytest.approx(score(5, 7), abs=1e-4)
    assert score(0, 2) != pytest.approx(score(0, 5), abs=1e-4)


def test_an_offset_beyond_the_context_is_refused_with_the_limit() -> None:
    rotary = RotaryEmbedding(head_dim=8, max_seq_len=16)

    with pytest.raises(ConfigError) as caught:
        rotary(4, offset=14)

    assert caught.value.details["max_seq_len"] == 16
    assert "--seq-len" in (caught.value.hint or "")


def test_an_odd_head_dimension_is_refused() -> None:
    with pytest.raises(ConfigError):
        RotaryEmbedding(head_dim=7, max_seq_len=16)


def test_the_rotary_table_is_not_part_of_the_state_dict() -> None:
    """It is a pure function of the config, which the checkpoint already records."""
    model = GPT(small_config())

    assert not any("rotary" in key for key in model.state_dict())
    assert model.buffer_bytes() > 0  # it does exist, it is just not saved


# --------------------------------------------------------------------------- #
# RMSNorm
# --------------------------------------------------------------------------- #
def test_rmsnorm_scales_to_unit_root_mean_square() -> None:
    norm = RMSNorm(8)
    x = torch.randn(4, 8) * 17

    out = norm(x)

    assert torch.allclose(out.pow(2).mean(-1).sqrt(), torch.ones(4), atol=1e-3)


def test_rmsnorm_does_not_subtract_the_mean() -> None:
    """Unlike LayerNorm. A shifted input stays shifted, only rescaled."""
    norm = RMSNorm(8)
    x = torch.randn(1, 8) + 10.0

    out = norm(x)

    assert abs(float(out.detach().mean())) > 0.1


def test_rmsnorm_reduces_in_fp32_even_when_given_half_precision() -> None:
    """Squaring activations is where a half-precision transformer overflows first."""
    norm = RMSNorm(8)
    x = (torch.ones(1, 8) * 300).half()

    out = norm(x)

    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()


def test_a_norm_says_how_wide_it_is_when_printed() -> None:
    """``print(model)`` is the first thing anyone does with a checkpoint they loaded.

    The bundle written by :mod:`trainai.export.bundle` hands users a loaded ``GPT``, and
    a stack of bare ``RMSNorm()`` lines tells them nothing: the width is the one number
    that says which model this is, and ``eps`` is a config field they are allowed to set
    (``ModelConfig.norm_eps``) whose value would otherwise appear nowhere in the printed
    architecture. ``nn.Module`` supplies the parentheses; what goes inside them is this.

    Read off the parameter rather than stored, so it cannot disagree with the tensor.
    """
    assert repr(RMSNorm(8)) == "RMSNorm(dim=8, eps=1e-05)"
    assert repr(RMSNorm(384, eps=1e-6)) == "RMSNorm(dim=384, eps=1e-06)"

    # Every norm in the model, reached the way a user reaches it.
    printed = repr(GPT(small_config()))
    assert "RMSNorm(dim=32, eps=1e-05)" in printed
    assert "RMSNorm()" not in printed


# --------------------------------------------------------------------------- #
# Weight tying
# --------------------------------------------------------------------------- #
def test_tied_head_is_literally_the_embedding_tensor() -> None:
    model = GPT(small_config(tie_embeddings=True))

    assert model.lm_head.weight is model.embed_tokens.weight


def test_untied_head_is_a_separate_tensor() -> None:
    model = GPT(small_config(tie_embeddings=False))

    assert model.lm_head.weight is not model.embed_tokens.weight
    assert model.alias_state_dict_keys == ()


def test_the_tied_alias_is_named_so_checkpoints_can_drop_it() -> None:
    model = GPT(small_config(tie_embeddings=True))
    state = model.state_dict()

    assert model.alias_state_dict_keys == ("lm_head.weight",)
    assert state["lm_head.weight"].data_ptr() == state["embed_tokens.weight"].data_ptr()


def test_loading_a_state_dict_whose_tied_copies_disagree_silently_wins_last() -> None:
    """The measured behaviour that justifies storing the tied weight once.

    ``load_state_dict`` applies both names in order and raises nothing, so a state
    dict with a zeroed ``lm_head.weight`` zeroes the embedding too. This test does
    not assert that the behaviour is good -- it pins it down, so that the reason
    :mod:`trainai.train.checkpoint` drops the alias stays documented and true.
    """
    config = small_config(tie_embeddings=True)
    good = GPT(config)
    damaged = dict(good.state_dict())
    damaged["lm_head.weight"] = torch.zeros_like(damaged["lm_head.weight"])

    loaded = GPT(config)
    loaded.load_state_dict(damaged)

    assert float(loaded.embed_tokens.weight.detach().abs().max()) == 0.0


def test_state_dict_round_trips(model: GPT, tokens: torch.Tensor) -> None:
    reloaded = GPT(model.config)

    reloaded.load_state_dict(model.state_dict())

    with torch.no_grad():
        assert torch.equal(model(tokens)[0], reloaded.eval()(tokens)[0])


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def test_an_untrained_model_scores_about_ln_vocab_on_unrelated_targets() -> None:
    """The sanity check for initialisation: no better and no worse than guessing."""
    torch.manual_seed(0)
    config = ModelConfig(vocab_size=997, n_layer=2, n_head=4, d_model=64, seq_len=32)
    model = GPT(config)
    inputs = torch.randint(0, 997, (4, 32))
    targets = torch.randint(0, 997, (4, 32))

    with torch.no_grad():
        _, loss, _ = model(inputs, targets)

    assert loss is not None
    assert float(loss) == pytest.approx(math.log(997), abs=0.25)


def test_tied_embeddings_make_predicting_the_input_easier_than_chance() -> None:
    """A trap for anyone writing a sanity check with ``targets = inputs``.

    Measured at 5.73 against ln(997) = 6.90. With tying, the residual stream carries
    the input token's embedding and the output projection *is* the embedding matrix,
    so the input token's own logit is inflated. Anyone testing initialisation with
    ``model(x, x)`` will conclude the model is broken, or worse, that it is good.
    """
    torch.manual_seed(0)
    config = ModelConfig(vocab_size=997, n_layer=2, n_head=4, d_model=64, seq_len=32)
    tied = GPT(config)
    untied = GPT(ModelConfig.from_dict({**config.to_dict(), "tie_embeddings": False}))
    inputs = torch.randint(0, 997, (8, 32))

    with torch.no_grad():
        _, tied_loss, _ = tied(inputs, inputs)
        _, untied_loss, _ = untied(inputs, inputs)

    assert tied_loss is not None and untied_loss is not None
    assert float(tied_loss) < math.log(997) - 0.5
    assert float(untied_loss) == pytest.approx(math.log(997), abs=0.25)


def test_loss_is_computed_in_fp32(model: GPT, tokens: torch.Tensor) -> None:
    """The log-sum-exp over thousands of logits is what loses precision in half."""
    with torch.no_grad():
        _, loss, _ = model(tokens, tokens)

    assert loss is not None
    assert loss.dtype == torch.float32
    assert loss.ndim == 0


def test_an_empty_sequence_is_a_trainai_bug_not_a_crash(model: GPT) -> None:
    with pytest.raises(ConfigError) as caught:
        model(torch.zeros((1, 0), dtype=torch.long))

    assert "bug" in (caught.value.hint or "").lower()


# --------------------------------------------------------------------------- #
# The loss mask
# --------------------------------------------------------------------------- #
#: How far the masked reduction may sit from ``reduction="mean"``. Not a taste: see
#: :func:`test_the_loss_tolerance_admits_the_gap_a_real_runner_produced`, which pins it
#: against the gap a real machine produced and against the size of a formula error.
MASKED_LOSS_REL = 1e-6

#: The two values one CI runner produced for the same loss, computed the two ways, on the
#: commit that made this a tolerance instead of an equality. Kept because a tolerance with
#: no witness is a number someone felt was safe; this one is the number that was measured.
#: The gap is 1.9 float32 ulps. Machines that vectorize the fused mean the same way the
#: explicit sum accumulates -- this developer's, and roughly half the hosted runners --
#: return the first value from both paths and cannot observe the difference at all.
OBSERVED_MEAN_AND_WEIGHTED = (4.196224689483643, 4.196225166320801)


def test_the_loss_tolerance_admits_the_gap_a_real_runner_produced() -> None:
    """The tolerance above has to cover float32, and nothing larger.

    Both halves matter. Too tight and the suite fails on a subset of the matrix that
    depends on the runner's vector width, which is what it did for four commits. Too loose
    and it stops distinguishing "the same formula summed in a different order" from "a
    different formula", which is the only thing it is for. The two are separated by four
    orders of magnitude, so a tolerance can sit between them -- but only if someone checks,
    and this is the check.
    """
    plain, masked = OBSERVED_MEAN_AND_WEIGHTED
    ulp = abs(plain) * torch.finfo(torch.float32).eps

    assert plain != masked, "the witness has to be a pair that actually disagreed"
    assert masked == pytest.approx(plain, rel=MASKED_LOSS_REL), (
        "the tolerance no longer admits a gap a real runner produced; tightening it "
        "re-breaks CI on hardware nobody here has"
    )
    assert abs(masked - plain) < 4 * ulp, "the witness is meant to be a rounding gap"
    assert MASKED_LOSS_REL < 1e-4, (
        "a tolerance this loose would accept a formula error as a rounding difference"
    )


def test_a_mask_of_ones_gives_the_unmasked_loss(model: GPT, tokens: torch.Tensor) -> None:
    """The masked path has to be the same formula, which is not the same as the same bits.

    Every existing run trains without a mask, so a masked path that differed in the fourth
    decimal would show up as a "loss curve changed" bug report with no cause. That is the
    claim worth defending, and two separate assertions defend it.

    The first is *not* ``==``, and that was a real bug in this test rather than a
    concession. ``reduction="mean"`` sums the per-token losses inside one fused kernel;
    the masked path multiplies by the weights and calls ``.sum()``. Those are the same
    arithmetic in exact real numbers and different orders of accumulation in float32, so
    whether they agree bit-for-bit is a property of the CPU's vector width, not of this
    code. Asserting equality made the outcome depend on which machine ran it: the two
    values were 4.196225166320801 and 4.196224689483643 -- 1.9 float32 ulps apart -- on
    some GitHub runners and identical on others, so the test failed on a shifting subset
    of an eight-job matrix, on Windows and Linux alike, with nothing in the repository
    having changed. A formula error moves a loss by percent; a summation order moves it by
    parts per ten million, and the tolerance here sits four orders of magnitude below the
    first and above the second.

    The second assertion is the one that pins the formula, and it is exact: the weighted
    mean computed independently in float64, where the accumulation order cannot matter at
    this scale. A wrong denominator or a dropped weight fails it outright.

    The two paths are deliberately *not* merged into one call to make the first assertion
    trivially true. That would replace a measured agreement between the shipped hot path
    and the masked path with a comparison of one code path against itself, which is the
    weaker claim wearing the stronger operator.
    """
    targets = torch.roll(tokens, -1, dims=1)
    ones = torch.ones_like(targets, dtype=torch.uint8)

    with torch.no_grad():
        logits, plain, _ = model(tokens, targets)
        _, masked, _ = model(tokens, targets, loss_mask=ones)

    assert plain is not None and masked is not None
    assert float(masked) == pytest.approx(float(plain), rel=MASKED_LOSS_REL)

    per_token = torch.nn.functional.cross_entropy(
        logits.double().view(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    )
    reference = float(per_token.sum() / per_token.numel())
    assert float(masked) == pytest.approx(reference, rel=MASKED_LOSS_REL)


def test_a_mask_scores_only_the_positions_it_selects(model: GPT, tokens: torch.Tensor) -> None:
    """Checked against cross-entropy over the selected columns, computed separately."""
    targets = torch.roll(tokens, -1, dims=1)
    mask = torch.zeros_like(targets, dtype=torch.uint8)
    mask[0, 3:7] = 1
    mask[1, 9:] = 1

    with torch.no_grad():
        _, masked, _ = model(tokens, targets, loss_mask=mask)
        logits, _, _ = model(tokens)

    selected = mask.reshape(-1).bool()
    reference = torch.nn.functional.cross_entropy(
        logits.float().view(-1, logits.size(-1))[selected], targets.reshape(-1)[selected]
    )
    assert masked is not None
    assert float(masked) == pytest.approx(float(reference), abs=1e-6)


def test_changing_a_masked_out_target_does_not_move_the_loss(
    model: GPT, tokens: torch.Tensor
) -> None:
    """The negative control. A weight that is applied has to be applied everywhere.

    Weighting the per-token losses and then dividing by ``targets.numel()`` instead of
    by the weight sum passes the test above up to a constant and fails nothing else;
    this pins the *targets* the loss reads rather than only the positions it counts.
    """
    targets = torch.roll(tokens, -1, dims=1)
    mask = torch.zeros_like(targets, dtype=torch.uint8)
    mask[:, :4] = 1
    moved = targets.clone()
    moved[:, 4:] = (moved[:, 4:] + 7) % VOCAB

    with torch.no_grad():
        _, before, _ = model(tokens, targets, loss_mask=mask)
        _, after, _ = model(tokens, moved, loss_mask=mask)

    assert before is not None and after is not None
    assert float(before) == float(after)


def test_a_window_with_no_target_gives_loss_zero_and_no_gradient() -> None:
    """The case that makes this a weighted mean instead of ``ignore_index=-100``.

    A window landing entirely inside a prompt selects nothing, and cross-entropy over
    nothing is 0/0. With ``ignore_index`` that is a NaN, and it does not stay local: it
    reaches every parameter through the backward pass and the run is dead from that step
    on, reported as divergence. The clamped denominator makes the window contribute
    exactly nothing instead, which is the truthful answer -- there was nothing to learn
    from it -- and lets the trainer count it and say so.
    """
    torch.manual_seed(0)
    model = GPT(small_config())
    inputs = torch.randint(0, VOCAB, (2, 8))
    targets = torch.randint(0, VOCAB, (2, 8))
    empty = torch.zeros_like(targets, dtype=torch.uint8)

    _, loss, _ = model(inputs, targets, loss_mask=empty)

    assert loss is not None
    assert float(loss.detach()) == 0.0
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is None or float(parameter.grad.abs().max()) == 0.0, name


def test_a_mask_of_the_wrong_shape_is_refused_as_a_bug(model: GPT, tokens: torch.Tensor) -> None:
    """Not a user error: nothing the user types can produce this, only wrong batching.

    Left unchecked, broadcasting does something plausible -- a (1, T) mask silently
    applies row 0's mask to every row of the batch -- and the loss it returns looks
    fine.
    """
    targets = torch.roll(tokens, -1, dims=1)

    with pytest.raises(ConfigError) as caught:
        model(tokens, targets, loss_mask=torch.ones((1, targets.shape[1]), dtype=torch.uint8))

    assert "bug" in (caught.value.hint or "").lower()
    assert str(list(targets.shape)) in str(caught.value)


def test_a_mask_without_targets_is_ignored_rather_than_silently_scoring(model: GPT) -> None:
    """``forward`` with no targets returns no loss, so there is nothing to mask."""
    inputs = torch.randint(0, VOCAB, (2, 8))

    with torch.no_grad():
        _, loss, _ = model(inputs, loss_mask=torch.ones((2, 8), dtype=torch.uint8))

    assert loss is None


# --------------------------------------------------------------------------- #
# Dropout
# --------------------------------------------------------------------------- #
def test_dropout_is_stochastic_in_training_and_off_in_eval(tokens: torch.Tensor) -> None:
    torch.manual_seed(0)
    model = GPT(small_config(dropout=0.2))

    model.train()
    assert not torch.equal(model(tokens)[0], model(tokens)[0])

    model.eval()
    with torch.no_grad():
        assert torch.equal(model(tokens)[0], model(tokens)[0])


def test_no_dropout_is_deterministic_even_in_training_mode(tokens: torch.Tensor) -> None:
    torch.manual_seed(0)
    model = GPT(small_config(dropout=0.0)).train()

    assert torch.equal(model(tokens)[0], model(tokens)[0])


# --------------------------------------------------------------------------- #
# Optimizer groups
# --------------------------------------------------------------------------- #
def test_every_parameter_lands_in_exactly_one_group() -> None:
    model = GPT(small_config())

    groups = model.parameter_groups(weight_decay=0.1)

    in_groups = [p for group in groups for p in group["params"]]
    assert sum(p.numel() for p in in_groups) == model.parameter_count()
    assert len({id(p) for p in in_groups}) == len(in_groups)


def test_norms_and_embeddings_are_not_decayed() -> None:
    """Decay on a norm's scale fights the normalisation; on an embedding it erodes
    the rows of rare tokens on every step, whether or not they appeared."""
    model = GPT(small_config())

    groups = model.parameter_groups(weight_decay=0.1)
    decayed = {id(p) for group in groups if group["weight_decay"] > 0 for p in group["params"]}

    assert id(model.embed_tokens.weight) not in decayed
    for name, parameter in model.named_parameters():
        if parameter.ndim < 2:
            assert id(parameter) not in decayed, name


def test_matrices_are_decayed() -> None:
    model = GPT(small_config())

    groups = model.parameter_groups(weight_decay=0.1)
    decayed = {id(p) for group in groups if group["weight_decay"] > 0 for p in group["params"]}

    assert id(model.blocks[0].attn.q_proj.weight) in decayed
    assert id(model.blocks[0].ffn.down_proj.weight) in decayed


def test_zero_weight_decay_still_produces_usable_groups() -> None:
    model = GPT(small_config())

    groups = model.parameter_groups(weight_decay=0.0)

    assert all(group["weight_decay"] == 0.0 for group in groups)
    assert sum(p.numel() for g in groups for p in g["params"]) == model.parameter_count()


def test_a_frozen_parameter_is_handed_to_neither_group() -> None:
    """Freezing the embedding is the ordinary fine-tune, and under tying it freezes two.

    ``embed_tokens.weight`` *is* the output projection when ``tie_embeddings`` is on --
    see ``test_tied_head_is_literally_the_embedding_tensor`` -- so this one call holds
    both still, which is a third of a small model's parameters.

    What makes the omission load-bearing rather than tidy is that ``parameter_count``
    already excludes frozen parameters by default, and that count is what the run record
    reports. Leave the frozen tensor in a group and the run says it is training one
    number while the optimizer holds another, with nothing in the output disagreeing.
    AdamW itself would not complain: it skips a parameter whose ``grad`` is ``None``, so
    the divergence is silent for as long as the run lasts.

    The control matters as much as the claim: ``parameter_groups`` dropping *everything*
    would satisfy the assertions below and is what
    ``test_every_parameter_lands_in_exactly_one_group`` above is holding down.
    """
    model = GPT(small_config(tie_embeddings=True))
    frozen = model.embed_tokens.weight

    before = model.parameter_groups(weight_decay=0.1)
    assert any(p is frozen for group in before for p in group["params"]), (
        "the embedding has to be in a group while it is trainable, or this proves nothing"
    )

    frozen.requires_grad_(False)
    groups = model.parameter_groups(weight_decay=0.1)

    kept = [p for group in groups for p in group["params"]]
    assert not any(p is frozen for p in kept), "a frozen tensor was handed to the optimizer"
    assert sum(p.numel() for p in kept) == model.parameter_count(), (
        "the groups and the reported trainable count have to be the same set"
    )
    assert model.parameter_count(trainable_only=False) - model.parameter_count() == frozen.numel()
    assert len(kept) == len([p for group in before for p in group["params"]]) - 1, (
        "exactly one parameter left, not a whole group"
    )


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def test_greedy_decoding_is_deterministic_and_keeps_the_prompt(model: GPT) -> None:
    prompt = torch.randint(0, VOCAB, (2, 4))

    first = model.generate(prompt, 6, temperature=0.0)
    second = model.generate(prompt, 6, temperature=0.0)

    assert torch.equal(first, second)
    assert first.shape == (2, 10)
    assert torch.equal(first[:, :4], prompt)


def test_sampling_is_reproducible_from_a_generator(model: GPT) -> None:
    prompt = torch.randint(0, VOCAB, (2, 4))

    def sample(seed: int) -> torch.Tensor:
        return model.generate(
            prompt, 6, temperature=1.0, generator=torch.Generator().manual_seed(seed)
        )

    assert torch.equal(sample(1), sample(1))
    assert not torch.equal(sample(1), sample(2))


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #
# ``generate`` is a thin consumer of ``generate_stream`` so that the interactive path
# and the batch path cannot drift apart. These tests are what makes that claim
# checkable rather than a comment.
def test_streaming_and_batch_generation_agree_bitwise(model: GPT) -> None:
    prompt = torch.randint(0, VOCAB, (2, 4))

    batch = model.generate(prompt, 6, temperature=1.0, generator=torch.Generator().manual_seed(11))
    pieces = list(
        model.generate_stream(
            prompt, 6, temperature=1.0, generator=torch.Generator().manual_seed(11)
        )
    )

    assert torch.equal(torch.cat([prompt, *pieces], dim=1), batch)


def test_streaming_yields_one_token_per_step(model: GPT) -> None:
    prompt = torch.randint(0, VOCAB, (3, 5))

    pieces = list(model.generate_stream(prompt, 7, temperature=0.0))

    assert len(pieces) == 7
    assert all(piece.shape == (3, 1) for piece in pieces)


def test_asking_for_no_tokens_returns_the_window_the_model_would_have_seen() -> None:
    """Zero is the one call where the promised left-truncation is directly visible.

    :meth:`GPT.generate` documents that a prompt longer than the context is truncated
    from the left, and every other test infers that from a suffix. Here the return value
    *is* the truncation, so this is what pins it -- and what says the answer is the last
    ``seq_len`` tokens rather than the first.

    Reachable from a library caller: ``export.bundle`` writes a snippet that calls
    ``model.generate`` directly, and a token budget computed from a limit minus a prompt
    length arrives here as 0 when the prompt fills the budget. The CLI refuses ``--tokens
    0`` before it gets this far (``cli/chat.py``), which is why nothing exercised it.

    Returning the prompt untouched is the obvious implementation and the wrong one: a
    caller who feeds the result back in has it truncated a second time, one round later
    than they could have known about.
    """
    model = GPT(small_config()).eval()
    context = model.config.seq_len

    fits = torch.randint(0, VOCAB, (2, 4))
    assert torch.equal(model.generate(fits, 0), fits), "nothing asked for, nothing appended"
    assert not list(model.generate_stream(fits, 0)), "the streaming path agrees"

    over_long = torch.arange(1, context * 2 + 1).reshape(1, -1) % VOCAB
    kept = model.generate(over_long, 0)

    assert kept.shape == (1, context)
    assert torch.equal(kept, over_long[:, -context:]), "the *last* seq_len tokens, not the first"


def test_streaming_builds_no_autograd_graph(model: GPT) -> None:
    """``@torch.no_grad()`` has to survive being applied to a generator function.

    PyTorch wraps generators so the guard is re-entered on each ``next()`` rather than
    exited at the first yield. If that stopped working, every streamed token would
    retain a graph and a long chat would leak memory until it died.
    """
    prompt = torch.randint(0, VOCAB, (1, 4))

    for piece in model.generate_stream(prompt, 5, temperature=0.0):
        assert not piece.requires_grad
        assert piece.grad_fn is None


def test_streaming_stops_at_eot_like_the_batch_path(model: GPT) -> None:
    prompt = torch.randint(0, VOCAB, (1, 4))
    stop_token = int(model.generate(prompt, 3, temperature=0.0)[0, 5])

    pieces = list(model.generate_stream(prompt, 6, temperature=0.0, eot_id=stop_token))

    assert len(pieces) <= 6
    assert int(pieces[-1][0, 0]) == stop_token, "the stopping token is yielded, then the loop ends"


def test_top_k_of_one_is_greedy(model: GPT) -> None:
    prompt = torch.randint(0, VOCAB, (2, 4))

    assert torch.equal(
        model.generate(prompt, 6, temperature=1.0, top_k=1),
        model.generate(prompt, 6, temperature=0.0),
    )


def test_top_p_keeps_at_least_one_token(model: GPT) -> None:
    """A nucleus smaller than the top token's probability must not empty the set."""
    prompt = torch.randint(0, VOCAB, (1, 4))

    out = model.generate(
        prompt, 4, temperature=1.0, top_p=1e-6, generator=torch.Generator().manual_seed(0)
    )

    assert out.shape == (1, 8)
    assert bool(((out >= 0) & (out < VOCAB)).all())


def test_generation_past_the_training_context_works(model: GPT) -> None:
    """RoPE allows it, so running out of context slides the cache rather than raising."""
    prompt = torch.randint(0, VOCAB, (2, 4))

    out = model.generate(prompt, 30, temperature=0.0)

    assert out.shape == (2, 34)
    assert bool(((out >= 0) & (out < VOCAB)).all())


def test_generating_past_the_context_still_costs_one_position_per_token() -> None:
    """The claim in :meth:`generate_stream`'s docstring, pinned as a number.

    Re-seeding the cache from the tail instead bought one cached step before the
    cache was full again, so every second token cost a full pass over the context.
    Measured on a 12.6M model, context 256, 400 new tokens: 44,287 forward positions
    against 599, and 10.13s against 3.54s. The cheapest way for that to come back is
    for someone to replace the slide with a rebuild, so the cost is asserted here and
    not left to a comment.
    """
    context = 12
    new_tokens = 40
    torch.manual_seed(0)
    model = GPT(small_config(seq_len=context)).eval()
    prompt = torch.randint(0, VOCAB, (1, 4))

    fed: list[int] = []
    real_forward = GPT.forward

    def counting(self, inputs, targets=None, *, caches=None, offset=0):  # type: ignore[no-untyped-def]
        fed.append(inputs.size(1))
        return real_forward(self, inputs, targets, caches=caches, offset=offset)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(GPT, "forward", counting)
        model.generate(prompt, new_tokens, temperature=0.0)

    # The prompt is one pass over 4 positions; every new token is then a single one.
    assert len(fed) == new_tokens
    assert sum(fed) == prompt.size(1) + new_tokens - 1, (
        f"fed {sum(fed)} positions for {new_tokens} tokens: {fed}"
    )
    assert max(fed[1:]) == 1, f"a step after the first fed more than one position: {fed}"


def test_the_slid_window_holds_the_last_seq_len_tokens() -> None:
    """After sliding, one more step must match a fresh pass over that same window."""
    context = 12
    torch.manual_seed(0)
    model = GPT(small_config(n_layer=1, seq_len=context)).eval()
    ids = torch.randint(0, VOCAB, (1, 30))

    with torch.no_grad():
        _, _, caches = model(ids[:, :context])
        for step in range(context, ids.size(1)):
            caches = model.slide_caches(caches, 1)
            logits, _, caches = model(ids[:, step : step + 1], caches=caches, offset=context - 1)
        fresh, _, _ = model(ids[:, -context:])

    assert caches[0][0].size(2) == context, "the window grew or shrank"
    assert torch.allclose(logits[:, -1], fresh[:, -1], atol=1e-5)


def test_generation_past_the_context_matches_a_brute_force_reference() -> None:
    """The whole loop end to end, against an O(n^2) reference that uses no cache at all.

    One layer, because that is where sliding and recomputing agree exactly -- see
    :func:`test_sliding_keeps_history_that_a_fresh_pass_would_have_lost` for why they
    diverge deeper. This is the only check that would notice the cache being slid
    correctly while the position offset is bookkept wrongly; everything else here
    either counts positions or drives ``slide_caches`` by hand.

    It compares *logits*, not token ids, and scripts the token stream rather than
    sampling. An untrained model decoded greedily emits one token forever -- measured,
    1 distinct token in 25 for every seed tried -- so comparing ids would pass no
    matter what the cache did.
    """
    context = 10
    torch.manual_seed(0)
    model = GPT(small_config(n_layer=1, seq_len=context)).eval()
    prompt = torch.randint(0, VOCAB, (1, 6))
    scripted = [3, 17, 40, 5, 61, 12, 33, 8, 55, 21, 47, 2, 38, 14, 59, 7, 26, 50, 11, 44]

    seen: list[torch.Tensor] = []

    def scripted_sample(logits, **_):  # type: ignore[no-untyped-def]
        seen.append(logits.clone())
        token = scripted[len(seen) - 1]
        return torch.full((logits.size(0), 1), token, dtype=torch.long, device=logits.device)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(GPT, "_sample", staticmethod(scripted_sample))
        model.generate(prompt, len(scripted), temperature=0.0)

    assert len(seen) == len(scripted)

    sequence = prompt
    with torch.no_grad():
        for step, token in enumerate(scripted):
            logits, _, _ = model(sequence[:, -context:])
            expected = logits[:, -1, :].float()

            assert torch.allclose(seen[step], expected, atol=1e-5), (
                f"step {step}, absolute position {prompt.size(1) + step}, differs from "
                f"the reference by {(seen[step] - expected).abs().max().item():.3e}"
            )
            sequence = torch.cat((sequence, torch.full((1, 1), token, dtype=torch.long)), dim=1)


def test_every_generated_id_is_inside_the_vocabulary(model: GPT) -> None:
    prompt = torch.randint(0, VOCAB, (3, 5))

    out = model.generate(
        prompt, 12, temperature=1.2, top_k=8, generator=torch.Generator().manual_seed(3)
    )

    assert int(out.min()) >= 0
    assert int(out.max()) < VOCAB


def test_eot_stops_generation_and_keeps_the_batch_rectangular(model: GPT) -> None:
    prompt = torch.randint(0, VOCAB, (2, 4))
    first = model.generate(prompt, 6, temperature=0.0)
    stop_token = int(first[0, 4])

    out = model.generate(prompt, 6, temperature=0.0, eot_id=stop_token)

    assert out.ndim == 2
    assert out.size(0) == 2
    assert out.size(1) <= 10


def test_repetition_penalty_lowers_the_logit_of_produced_tokens() -> None:
    """Division for positive logits, multiplication for negative ones.

    Dividing a negative logit by 1.2 raises it, which would *encourage* the
    repetition it is meant to discourage. Both branches are checked here rather
    than inferred from generated text, because whether a penalty is large enough to
    flip an argmax depends on the model, not on the arithmetic.
    """
    logits = torch.tensor([[5.0, -3.0, 0.5]])
    produced = torch.tensor([[0, 1]])

    adjusted = GPT._penalise_repeats(logits, produced, 2.0)

    assert float(adjusted[0, 0]) == pytest.approx(2.5)  # positive: halved
    assert float(adjusted[0, 1]) == pytest.approx(-6.0)  # negative: pushed further down
    assert float(adjusted[0, 2]) == pytest.approx(0.5)  # not produced: untouched


def test_a_large_repetition_penalty_breaks_a_repeating_loop(model: GPT) -> None:
    """An untrained model greedily repeats one token; the penalty must escape that.

    The penalty has to be large here, and that is honest rather than a fudge: an
    untrained model's favourite logit sits far above the runner-up, so measured on
    this fixture 1.2 and 2.0 change nothing and 10.0 does.
    """
    prompt = torch.randint(0, VOCAB, (1, 6))

    looping = model.generate(prompt, 8, temperature=0.0)
    escaped = model.generate(prompt, 8, temperature=0.0, repetition_penalty=10.0)

    assert len(set(looping[0, 6:].tolist())) == 1  # the loop this is escaping from
    assert len(set(escaped[0, 6:].tolist())) > 1
    assert not torch.equal(looping, escaped)


def test_a_negative_temperature_is_refused(model: GPT) -> None:
    with pytest.raises(ConfigError):
        model.generate(torch.zeros((1, 2), dtype=torch.long), 2, temperature=-1.0)


def test_a_prompt_without_a_batch_dimension_is_refused(model: GPT) -> None:
    with pytest.raises(ConfigError) as caught:
        model.generate(torch.zeros(4, dtype=torch.long), 2)

    assert "unsqueeze" in (caught.value.hint or "")


# --------------------------------------------------------------------------- #
# Learning
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_the_model_can_overfit_a_single_batch() -> None:
    """If it cannot drive one batch to near-zero loss, nothing else is worth testing."""
    torch.manual_seed(0)
    config = ModelConfig(vocab_size=128, n_layer=2, n_head=4, d_model=128, seq_len=32)
    model = GPT(config).train()
    optimizer = torch.optim.AdamW(model.parameter_groups(0.0), lr=3e-3, betas=(0.9, 0.95))
    batch = torch.randint(0, 128, (4, 33))
    inputs, targets = batch[:, :-1], batch[:, 1:]

    first = None
    loss_value = math.inf
    for _ in range(300):
        _, loss, _ = model(inputs, targets)
        assert loss is not None
        loss_value = float(loss.detach())
        first = first if first is not None else loss_value
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    assert first is not None and first > 1.0
    assert loss_value < 0.1, f"loss only reached {loss_value}"


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def test_parameter_count_matches_the_config(model: GPT) -> None:
    assert model.parameter_count() == model.config.parameter_count


def test_summary_is_json_safe(model: GPT) -> None:
    import json

    payload = json.loads(json.dumps(model.summary()))

    assert payload["parameters"] == model.parameter_count()
    assert payload["tied_embeddings"] is True
    assert sum(payload["breakdown"].values()) == payload["parameters"]


@pytest.mark.gpu
def test_the_model_runs_on_cuda(tokens: torch.Tensor) -> None:
    torch.manual_seed(0)
    model = GPT(small_config()).cuda().eval()

    with torch.no_grad():
        logits, loss, _ = model(tokens.cuda(), tokens.cuda())

    assert logits.device.type == "cuda"
    assert loss is not None
    assert torch.isfinite(loss)


@pytest.mark.gpu
def test_bf16_autocast_keeps_the_loss_finite(tokens: torch.Tensor) -> None:
    if not torch.cuda.is_bf16_supported():
        pytest.skip("this GPU does not support bf16")
    torch.manual_seed(0)
    model = GPT(small_config()).cuda().train()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss, _ = model(tokens.cuda(), tokens.cuda())

    assert loss is not None
    assert loss.dtype == torch.float32  # cross-entropy is forced back to fp32
    assert torch.isfinite(loss)


def test_saving_to_disk_and_loading_back_preserves_logits(
    model: GPT, tokens: torch.Tensor, tmp_path: Path
) -> None:
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)

    reloaded = GPT(model.config)
    reloaded.load_state_dict(torch.load(path, weights_only=True))
    reloaded.eval()

    with torch.no_grad():
        assert torch.equal(model(tokens)[0], reloaded(tokens)[0])
