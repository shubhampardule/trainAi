"""The transformer: a decoder-only GPT with the modern set of choices.

RMSNorm, RoPE, SwiGLU, grouped-query attention, no biases, weight tying. Each of
those is here for a reason that holds specifically for small models on consumer
hardware, and the reasons are written next to the code rather than assumed.

What this module does *not* do is try to be a general modelling framework. It is
one architecture, implemented once, with the shapes fixed by
:class:`~trainai.model.config.ModelConfig`. That is what makes it possible to
assert things about it -- that attention cannot see the future, that a resumed run
continues bitwise-identically -- instead of hoping.

Attention uses :func:`torch.nn.functional.scaled_dot_product_attention` with
``is_causal=True`` rather than a hand-written masked matmul. PyTorch dispatches
that to a fused kernel (FlashAttention or the memory-efficient path) when one is
available, which is the difference between a 512-token context fitting on a 4 GiB
card and not fitting: the naive implementation materialises a
``batch x heads x seq x seq`` attention matrix, and at batch 8, 8 heads and 512
tokens that alone is 64 MiB per layer in fp32.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from trainai.errors import ConfigError
from trainai.model.config import ModelConfig

__all__ = ["GPT", "Attention", "Block", "FeedForward", "RMSNorm", "RotaryEmbedding"]


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation.

    LayerNorm subtracts the mean and learns a bias; RMSNorm does neither and
    performs the same in practice. For a small model the saving is real but modest
    (one vector per norm); the reason to prefer it is that it is one fewer
    reduction per norm per step, and there are ``2 * n_layer + 1`` of them.

    The reduction runs in fp32 even under autocast. Squaring activations in fp16
    is where a mixed-precision transformer overflows first: fp16's maximum is
    65504, so an activation of 300 already squares out of range.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(dtype)

    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[0]}, eps={self.eps}"


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings.

    Position is applied by rotating each pair of dimensions in the query and key
    vectors by an angle proportional to the position. The dot product of two
    rotated vectors then depends on the *difference* of their positions, so
    attention becomes relative without any learned position parameters.

    Two consequences that matter here:

    * Nothing is learned, so a model built for 512 tokens can be evaluated at 256
      without retraining, and lengthening the context later does not invalidate
      the weights.
    * There is no position-embedding matrix to size, which on a small model is
      another ``seq_len * d_model`` parameters not spent.

    The cos/sin table is a buffer, not a parameter, and is excluded from the
    state dict: it is a pure function of ``(seq_len, head_dim, theta)``, all three
    of which the checkpoint records, so storing it would make checkpoints larger
    and add a way for them to disagree with themselves.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ConfigError(
                f"Rotary embeddings need an even head dimension, got {head_dim}.",
                hint="Adjust d_model or n_head so that d_model / n_head is even.",
                details={"head_dim": head_dim},
            )
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        cos, sin = self._table(max_seq_len)
        # persistent=False: derived from the config, so not part of the checkpoint.
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _table(self, seq_len: int) -> tuple[Tensor, Tensor]:
        # Frequencies for each dimension pair, computed in float64 so that the
        # table is identical on every device and dtype. Position tables are
        # compared between a checkpoint and a fresh model during resume.
        half = self.head_dim // 2
        inverse_freq = 1.0 / (self.theta ** (torch.arange(0, half, dtype=torch.float64) / half))
        positions = torch.arange(seq_len, dtype=torch.float64)
        angles = torch.outer(positions, inverse_freq)
        return angles.cos().float(), angles.sin().float()

    def forward(self, seq_len: int, offset: int = 0) -> tuple[Tensor, Tensor]:
        """Return the cos/sin slice for positions ``offset .. offset + seq_len``.

        ``offset`` is what makes incremental generation work: token 300 must be
        rotated by the angle for position 300 even when it is the only token in
        the forward pass.
        """
        end = offset + seq_len
        if end > self.max_seq_len:
            raise ConfigError(
                f"Position {end} is beyond the model's context length of {self.max_seq_len}.",
                hint=(
                    "Shorten the input, or train a model with a larger --seq-len. "
                    "Rotary embeddings allow a shorter context than the model was "
                    "built for, not a longer one."
                ),
                details={"requested": end, "max_seq_len": self.max_seq_len},
            )
        return self.cos[offset:end], self.sin[offset:end]


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate the last dimension of ``x`` pairwise.

    Args:
        x: ``(batch, heads, seq, head_dim)``.
        cos, sin: ``(seq, head_dim / 2)``.

    The two halves of the head dimension are treated as the real and imaginary
    parts of ``head_dim / 2`` complex numbers -- the "split" convention, matching
    the reference Llama implementation. Interleaved pairs would work equally well
    but the two are not interchangeable, so the convention is fixed here and
    checked by a test.
    """
    left, right = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)


class Attention(nn.Module):
    """Causal self-attention with optional grouped queries.

    No biases on any projection: at this scale they measurably do nothing, and
    each one is a separate small kernel launch.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.kv_heads = config.kv_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.d_model, config.n_head * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_head * config.head_dim, config.d_model, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch, seq, _ = x.shape

        q = self.q_proj(x).view(batch, seq, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if cache is not None:
            past_k, past_v = cache
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        present = (k, v)

        # ``is_causal=True`` asks the kernel for the mask instead of building one, which
        # is what keeps the training path on the fused kernel. But it means "query i may
        # see key i and earlier" with both numbered from zero, so it is only right when
        # every key is also a query. With a cache the keys start earlier than the
        # queries, and that diagonal lands in the wrong place.
        attn_mask: Tensor | None = None
        is_causal = cache is None and seq > 1
        if cache is not None and seq > 1:
            # More than one new token against a cache. Passing neither a mask nor
            # ``is_causal`` here left the attention *unmasked*, so each new token
            # attended to the ones after it: measured 2.9e-01 of logit error against a
            # single full pass over the same tokens, and a shape that is wrong output
            # rather than a crash. The queries sit at absolute positions
            # ``past .. past + seq - 1``, so the mask is built from those.
            past = k.size(2) - seq
            query_positions = torch.arange(past, past + seq, device=x.device).unsqueeze(1)
            key_positions = torch.arange(past + seq, device=x.device).unsqueeze(0)
            attn_mask = key_positions <= query_positions

        if self.kv_heads != self.n_head:
            # Grouped-query attention: one key/value head serves ``groups`` query
            # heads. This copies -- ``repeat_interleave`` allocates here and now, it is
            # not a view and nothing is deferred to the kernel (measured: the result
            # does not share storage, and at batch 8 the expanded keys and values take
            # 8.0 MiB where the originals took 2.0).
            #
            # The copy is deliberate, because the alternative is worse. ``enable_gqa``
            # skips it, but the memory-efficient kernel -- the one that actually runs on
            # a consumer GPU -- refuses mismatched head counts, so SDPA silently falls
            # back to the math path and materialises the whole batch x heads x seq x seq
            # attention matrix. Measured on an sm_86 laptop card, one attention call at
            # batch 8, seq 512: expanded stays fused at +12.0 MiB, ``enable_gqa`` costs
            # +189.2 MiB. Over a real training step, 1,752 MiB against 2,997 MiB and
            # 0.20s against 0.35s. Paying for the copy is 1.2 GiB cheaper than avoiding
            # it. (cuDNN's kernel does accept ``enable_gqa``, but it is runtime-disabled
            # on this build; on a card whose flash kernel is available the trade may go
            # the other way, so this is measured, not assumed.)
            #
            # Interleave rather than tile: query head ``i`` must read key/value head
            # ``i // groups``. ``repeat`` has the same shape and the wrong mapping.
            groups = self.n_head // self.kv_heads
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attended = attended.transpose(1, 2).reshape(batch, seq, self.n_head * self.head_dim)
        return self.resid_dropout(self.o_proj(attended)), present


class FeedForward(nn.Module):
    """SwiGLU feed-forward network.

    ``down(silu(gate(x)) * up(x))``. Three projections instead of two, with the
    hidden width scaled by 2/3 so the parameter count matches a 4x ReLU MLP. The
    gate is what earns the third matrix: it lets the layer suppress a channel
    based on the input rather than only rectify it.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = config.ffn_dim
        self.gate_proj = nn.Linear(config.d_model, hidden, bias=False)
        self.up_proj = nn.Linear(config.d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    """One transformer block, pre-norm.

    Pre-norm (normalise the input to each sublayer, add the raw output) rather
    than post-norm, because the residual path stays unnormalised from the
    embedding to the final norm. That path is what lets a deep stack train without
    a warmup schedule tuned per depth; post-norm transformers at this depth need
    careful initialisation to avoid diverging in the first hundred steps.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        attended, present = self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + attended
        x = x + self.ffn(self.ffn_norm(x))
        return x, present


class GPT(nn.Module):
    """A decoder-only transformer.

    Args:
        config: The shape. Stored on the module and written into checkpoints.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.rotary = RotaryEmbedding(config.head_dim, config.seq_len, config.rope_theta)
        self.embed_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)

        if config.tie_embeddings:
            # No separate output matrix: the logits are the embedding read
            # backwards. Assigning the weight rather than allocating a second one
            # means there is only ever one tensor, so an optimizer cannot update
            # two copies out of step.
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.lm_head.weight = self.embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        # Residual projections are scaled down by depth. Each block adds its output
        # to the residual stream, so without this the stream's variance grows
        # linearly in n_layer and the final norm has to undo it.
        scaled = 0.02 / math.sqrt(2 * config.n_layer)
        for name, parameter in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(parameter, mean=0.0, std=scaled)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:  # pragma: no cover - no biases in this model
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -- forward ------------------------------------------------------------ #

    def forward(
        self,
        inputs: Tensor,
        targets: Tensor | None = None,
        *,
        caches: list[tuple[Tensor, Tensor]] | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, Tensor | None, list[tuple[Tensor, Tensor]]]:
        """Run the model.

        Args:
            inputs: ``(batch, seq)`` token ids.
            targets: ``(batch, seq)`` token ids to score against. When given, the
                loss is returned; when not, only logits are.
            caches: Per-layer key/value cache from a previous call, for
                incremental generation. ``inputs`` may be more than one token
                against a cache -- a prompt fed in chunks -- and the new tokens are
                masked against each other as well as attending to the cache.
            offset: Position of ``inputs[:, 0]`` in the full sequence. Required
                when using a cache, because rotary embeddings are absolute.

        Returns:
            ``(logits, loss, caches)``. ``loss`` is ``None`` when ``targets`` is.
        """
        _, seq = inputs.shape
        if seq == 0:
            raise ConfigError(
                "The model was given a sequence of length 0.",
                hint="This is a TrainAI bug in batching; please report it.",
                details={"shape": list(inputs.shape)},
            )

        x = self.embed_dropout(self.embed_tokens(inputs))
        cos, sin = self.rotary(seq, offset)

        present: list[tuple[Tensor, Tensor]] = []
        for index, block in enumerate(self.blocks):
            cache = caches[index] if caches is not None else None
            x, layer_cache = block(x, cos, sin, cache)
            present.append(layer_cache)

        x = self.final_norm(x)

        if targets is None:
            return self.lm_head(x), None, present

        logits = self.lm_head(x)
        # Cross-entropy in fp32: under autocast the logits arrive in bf16/fp16,
        # and the log-sum-exp over a vocabulary of thousands is exactly the
        # reduction that loses precision in half.
        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        return logits, loss, present

    # -- reporting ---------------------------------------------------------- #

    def parameter_count(self, *, trainable_only: bool = True) -> int:
        """Real parameter count, counting a tied weight once.

        ``named_parameters`` already de-duplicates shared tensors, which is what
        makes weight tying show up correctly here.
        """
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad or not trainable_only
        )

    def parameter_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.parameters())

    def buffer_bytes(self) -> int:
        return sum(b.numel() * b.element_size() for b in self.buffers())

    @property
    def alias_state_dict_keys(self) -> tuple[str, ...]:
        """State-dict keys that are a second name for another entry.

        With ``tie_embeddings``, ``state_dict()`` lists ``lm_head.weight`` and
        ``embed_tokens.weight`` separately even though they share one storage --
        PyTorch does not de-duplicate names, only :meth:`parameters` does. Storing
        both is worth avoiding for two measured reasons, not for tidiness:

        * ``load_state_dict`` applies the keys in order and the last one wins. Given
          a state dict whose two copies disagree, it silently overwrites the
          embedding and raises nothing. Verified: a zeroed ``lm_head.weight``
          zeroes the embedding of the loaded model.
        * ``safetensors.save_file`` refuses shared storage outright, so the export
          has to drop one of them anyway -- :mod:`trainai.export.hf` reads this
          property to decide which key to omit.

        The saving in bytes is not the point: ``torch.save`` de-duplicates storage,
        so keeping the alias costs about 1.8 KB of metadata, not a second copy.
        """
        return ("lm_head.weight",) if self.config.tie_embeddings else ()

    def summary(self) -> dict[str, Any]:
        """What this model is, for the run record."""
        return {
            "config": self.config.to_dict(),
            "parameters": self.parameter_count(),
            "non_embedding_parameters": self.config.non_embedding_parameter_count,
            "parameter_bytes": self.parameter_bytes(),
            "breakdown": self.config.breakdown(),
            "tied_embeddings": self.config.tie_embeddings,
        }

    # -- optimizer plumbing ------------------------------------------------- #

    def parameter_groups(self, weight_decay: float) -> list[dict[str, Any]]:
        """Split parameters into decayed and undecayed groups.

        Matrices are decayed; norm weights and embeddings are not. Applying
        weight decay to a norm's scale pulls it toward zero, which fights the
        normalisation it exists to perform. Embeddings are excluded because a rare
        token's row receives gradient only when that token appears, but decay
        would shrink it on every single step -- with weight tying that row is also
        the output projection for the token, so decaying it makes the model
        progressively less able to predict rare tokens the longer training runs.
        """
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        seen: set[int] = set()
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if parameter.ndim < 2 or "embed_tokens" in name:
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        groups = [{"params": decay, "weight_decay": weight_decay}]
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        return groups

    # -- generation --------------------------------------------------------- #

    def slide_caches(
        self, caches: list[tuple[Tensor, Tensor]] | None, drop: int
    ) -> list[tuple[Tensor, Tensor]]:
        """Drop the oldest ``drop`` entries from every layer's cache, renumbering the rest.

        A cache cannot simply be truncated. It holds keys with the rotary rotation
        already applied at their absolute positions, so the survivors would still be
        numbered from where they were and the next query would sit past ``seq_len``.

        But :func:`apply_rotary` multiplies by ``e^(i * theta * position)``, and
        rotations compose, so multiplying every surviving key by
        ``e^(-i * theta * drop)`` renumbers the whole cache from zero. That is exact
        rather than an approximation: the slid cache matched a fresh forward pass over
        the same window to 2.980e-08, and re-anchoring is stable under repetition --
        4.768e-07 after 32 successive single-position slides, against keys of
        magnitude 0.36.

        Values are returned trimmed but otherwise untouched: the rotation applies to
        queries and keys only.

        Args:
            caches: Per-layer ``(keys, values)`` as returned by :meth:`forward`.
            drop: How many of the oldest positions to discard.

        Returns:
            New caches holding ``length - drop`` positions, numbered from zero.

        Raises:
            ConfigError: If there is no cache to slide, or ``drop`` is outside
                ``0 .. length - 1``.
        """
        if not caches:
            raise ConfigError(
                "slide_caches() needs the caches from a previous forward pass, got none.",
                hint="Call forward() without a cache first, then slide what it returns.",
                details={"drop": drop},
            )
        length = caches[0][0].size(2)
        if not 0 <= drop < length:
            raise ConfigError(
                f"Cannot drop {drop} positions from a cache holding {length}.",
                hint=(
                    "drop must be between 0 and the cache length minus one. "
                    "To discard the whole cache, pass caches=None to forward() instead."
                ),
                details={"drop": drop, "length": length},
            )
        if drop == 0:
            return caches

        kept = length - drop
        # One angle for every surviving position: they all move back by the same
        # amount, so this is the position-``drop`` row broadcast, not a slice.
        cos, sin = self.rotary(1, drop)
        cos = cos.expand(kept, -1)
        sin = sin.expand(kept, -1)
        return [
            (apply_rotary(keys[:, :, drop:], cos, -sin), values[:, :, drop:])
            for keys, values in caches
        ]

    @torch.no_grad()
    def generate(
        self,
        prompt: Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        eot_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample a continuation of ``prompt`` and return prompt-plus-continuation.

        Args:
            prompt: ``(batch, seq)`` token ids. Truncated from the left if longer
                than the model's context.
            max_new_tokens: How many tokens to append.
            temperature: 0 means greedy (argmax), which is deterministic.
            top_k: Keep only the ``k`` most likely tokens at each step.
            top_p: Keep the smallest set of tokens whose probabilities sum to
                ``p`` (nucleus sampling).
            repetition_penalty: Above 1, divides the logit of tokens already
                present. 1.0 disables it.
            eot_id: Stop early once every sequence in the batch has produced this.
            generator: For reproducible sampling.

        A thin consumer of :meth:`generate_stream`, so there is exactly one
        implementation of the sampling loop and the streaming and batch paths cannot
        drift apart.
        """
        pieces = list(
            self.generate_stream(
                prompt,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eot_id=eot_id,
                generator=generator,
            )
        )
        tokens = prompt[:, -self.config.seq_len :]
        if not pieces:
            return tokens
        return torch.cat([tokens, *pieces], dim=1)

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        eot_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[Tensor]:
        """Yield each sampled token as a ``(batch, 1)`` tensor, as it is produced.

        Same arguments and same sampling as :meth:`generate`. This is the form the
        interactive playground needs: a 30-token/s model that prints nothing for ten
        seconds and then a paragraph feels broken, and the same model printing as it
        goes does not.

        Uses a key/value cache, so each new token costs one forward position rather
        than a pass over the whole sequence. Past the training context the cache is
        slid rather than rebuilt (see :meth:`slide_caches`), so that stays true: the
        model then sees a window of the last ``seq_len`` tokens.
        """
        self.eval()
        if prompt.ndim != 2:
            raise ConfigError(
                f"generate() expects (batch, seq) token ids, got shape {tuple(prompt.shape)}.",
                hint="Add a batch dimension: prompt.unsqueeze(0).",
                details={"shape": list(prompt.shape)},
            )
        if temperature < 0:
            raise ConfigError(
                f"temperature must not be negative, got {temperature}.",
                hint="Use 0 for greedy decoding, 0.8 for a usual sample.",
                details={"temperature": temperature},
            )

        context = self.config.seq_len
        tokens = prompt[:, -context:]
        caches: list[tuple[Tensor, Tensor]] | None = None
        offset = 0
        produced = tokens
        finished = torch.zeros(tokens.size(0), dtype=torch.bool, device=tokens.device)

        step_input = tokens
        for _ in range(max_new_tokens):
            if offset + step_input.size(1) > context:
                # The cache is full. Slide it: discard the oldest positions and
                # renumber the survivors, so a window of the last ``context`` tokens
                # stays live at one forward position per new token.
                #
                # Re-seeding the cache from the tail instead -- which is what this
                # did -- bought exactly one cached step before the cache was full
                # again, so every second token cost a full pass over the context.
                # Measured on a 12.6M model with context 256, generating 400 tokens:
                # 44,287 forward positions (110.72 per token) against 599 (1.50,
                # the prompt included) here, and 9.90s against 3.23s.
                drop = offset + step_input.size(1) - context
                caches = self.slide_caches(caches, drop)
                offset -= drop

            logits, _, caches = self(step_input, caches=caches, offset=offset)
            offset += step_input.size(1)
            next_logits = logits[:, -1, :].float()

            if repetition_penalty != 1.0:
                next_logits = self._penalise_repeats(next_logits, produced, repetition_penalty)
            next_token = self._sample(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            )
            if eot_id is not None:
                # A finished sequence keeps emitting the stop token, so the batch
                # stays rectangular and the caller can trim at the first one.
                # finished is (batch,) and next_token is (batch, 1): without the
                # unsqueeze, torch.where broadcasts them to (batch, batch).
                next_token = torch.where(
                    finished.unsqueeze(1), torch.full_like(next_token, eot_id), next_token
                )
                finished |= next_token.squeeze(1) == eot_id

            produced = torch.cat((produced, next_token), dim=1)
            step_input = next_token
            yield next_token
            if eot_id is not None and bool(finished.all()):
                break

    @staticmethod
    def _penalise_repeats(logits: Tensor, produced: Tensor, penalty: float) -> Tensor:
        """Divide the logit of every token already produced.

        Division rather than subtraction, so the effect is proportional: a token
        the model is only mildly keen on is discouraged mildly. Negative logits are
        multiplied instead, because dividing a negative number by 1.2 raises it.
        """
        gathered = torch.gather(logits, 1, produced)
        adjusted = torch.where(gathered > 0, gathered / penalty, gathered * penalty)
        return logits.scatter(1, produced, adjusted)

    @staticmethod
    def _sample(
        logits: Tensor,
        *,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        generator: torch.Generator | None,
    ) -> Tensor:
        if temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / temperature

        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            threshold = logits.topk(k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        if top_p is not None and 0.0 < top_p < 1.0:
            ordered, indices = logits.sort(dim=-1, descending=True)
            cumulative = ordered.softmax(dim=-1).cumsum(dim=-1)
            # Keep everything up to and including the token that crosses p, so the
            # kept set is never empty even when one token holds more than p.
            drop = cumulative - ordered.softmax(dim=-1) >= top_p
            ordered = ordered.masked_fill(drop, float("-inf"))
            logits = torch.empty_like(logits).scatter(1, indices, ordered)

        probabilities = logits.softmax(dim=-1)
        return torch.multinomial(probabilities, num_samples=1, generator=generator)
