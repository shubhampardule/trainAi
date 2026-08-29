"""Model configuration, with an exact parameter count.

The parameter count matters more here than it looks. The planner (M3) has to
decide whether a configuration fits in measured free VRAM before building
anything, and every number it reports to the user is derived from this count. So
:attr:`ModelConfig.parameter_count` is arithmetic over the shapes the model
actually allocates, not an approximation -- and a test asserts it equals
``sum(p.numel() for p in model.parameters())`` for a spread of configurations. If
the model changes shape and this file does not, that test fails.

Every field is validated on construction. A configuration that cannot be built is
rejected here, with the arithmetic that makes it impossible, rather than surfacing
as a shape error inside an attention kernel forty lines deep.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from trainai.errors import ConfigError
from trainai.serialise import checked_fields

__all__ = ["PRESETS", "ModelConfig", "preset"]

#: Multiple that the SwiGLU hidden dimension is rounded up to. Matrix
#: multiplications on tensor cores want dimensions divisible by 8 in fp16/bf16;
#: 64 costs a few thousand parameters and keeps every kernel on the fast path.
FFN_MULTIPLE = 64

#: SwiGLU has three projections where a ReLU MLP has two, so the usual 4x hidden
#: dimension is scaled by 2/3 to land at the same parameter count.
FFN_EXPANSION = 8 / 3


@dataclass(frozen=True)
class ModelConfig:
    """Shape of the transformer. Frozen, because a checkpoint records it verbatim.

    Args:
        vocab_size: Token count, taken from the dataset manifest. Not a free
            choice: a model whose embedding is smaller than the tokenizer's
            vocabulary cannot represent its own training data.
        n_layer: Transformer blocks.
        n_head: Query heads.
        n_kv_head: Key/value heads. Fewer than ``n_head`` gives grouped-query
            attention, which shrinks the KV cache during generation at a small
            cost in quality. Defaults to ``n_head`` (no grouping).
        d_model: Residual stream width. Must divide evenly into ``n_head``.
        d_ff: SwiGLU hidden width. Defaults to ``8/3 * d_model`` rounded up to a
            multiple of 64.
        seq_len: Maximum context length the model is built for. RoPE lets a model
            run at a shorter length than it was built for without retraining.
        dropout: Applied to attention output and the FFN output. 0 is right for
            from-scratch pretraining on a corpus seen once; raise it only when the
            model is looping over a small corpus many times.
        rope_theta: RoPE base frequency. 10000 is the standard value.
        tie_embeddings: Share the input embedding with the output projection. On a
            small model this is a large fraction of the parameters -- at
            ``vocab_size`` 8192 and ``d_model`` 512 it saves 4.2M -- and it also
            regularises, because every token's embedding gets gradient from both
            reading and predicting it.
        norm_eps: RMSNorm epsilon.
    """

    vocab_size: int
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    seq_len: int = 512
    n_kv_head: int | None = None
    d_ff: int | None = None
    dropout: float = 0.0
    rope_theta: float = 10_000.0
    tie_embeddings: bool = True
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        self._require_positive("vocab_size", self.vocab_size)
        self._require_positive("n_layer", self.n_layer)
        self._require_positive("n_head", self.n_head)
        self._require_positive("d_model", self.d_model)
        self._require_positive("seq_len", self.seq_len)

        if self.d_model % self.n_head:
            raise ConfigError(
                f"d_model {self.d_model} is not divisible by n_head {self.n_head} "
                f"({self.d_model / self.n_head:.2f} dimensions per head).",
                hint=(
                    f"Use n_head {self._largest_divisor(self.d_model, self.n_head)} or "
                    f"d_model {self.n_head * round(self.d_model / self.n_head)}. Every "
                    "head has to get the same number of dimensions."
                ),
                details={"d_model": self.d_model, "n_head": self.n_head},
            )

        if self.head_dim % 2:
            raise ConfigError(
                f"The head dimension is {self.head_dim}, which is odd.",
                hint=(
                    "Rotary position embeddings rotate dimensions in pairs, so the head "
                    "dimension must be even. Adjust d_model or n_head so that "
                    "d_model / n_head is even."
                ),
                details={"head_dim": self.head_dim},
            )

        kv_heads = self.n_kv_head if self.n_kv_head is not None else self.n_head
        self._require_positive("n_kv_head", kv_heads)
        if kv_heads > self.n_head:
            raise ConfigError(
                f"n_kv_head {kv_heads} exceeds n_head {self.n_head}.",
                hint=(
                    "Grouped-query attention shares key/value heads between query "
                    f"heads, so n_kv_head must be at most n_head ({self.n_head})."
                ),
                details={"n_head": self.n_head, "n_kv_head": kv_heads},
            )
        if self.n_head % kv_heads:
            raise ConfigError(
                f"n_head {self.n_head} is not divisible by n_kv_head {kv_heads}.",
                hint=(
                    "Each key/value head is shared by an equal number of query heads. "
                    f"Try n_kv_head {self._largest_divisor(self.n_head, kv_heads)}."
                ),
                details={"n_head": self.n_head, "n_kv_head": kv_heads},
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(
                f"dropout must be at least 0 and below 1, got {self.dropout}.",
                hint="0.0 suits a corpus seen once; 0.1 is a reasonable maximum.",
                details={"dropout": self.dropout},
            )
        if self.rope_theta <= 0:
            raise ConfigError(
                f"rope_theta must be positive, got {self.rope_theta}.",
                hint="Leave it at the default 10000 unless you know why you are changing it.",
                details={"rope_theta": self.rope_theta},
            )
        if self.norm_eps <= 0:
            raise ConfigError(
                f"norm_eps must be positive, got {self.norm_eps}.",
                hint="1e-5 is the default and is almost always right.",
                details={"norm_eps": self.norm_eps},
            )

        if self.d_ff is not None:
            self._require_positive("d_ff", self.d_ff)

    # -- derived shapes ----------------------------------------------------- #

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def kv_heads(self) -> int:
        """Key/value heads, resolved. Equals :attr:`n_head` when not grouping."""
        return self.n_kv_head if self.n_kv_head is not None else self.n_head

    @property
    def kv_groups(self) -> int:
        """Query heads per key/value head."""
        return self.n_head // self.kv_heads

    @property
    def ffn_dim(self) -> int:
        """SwiGLU hidden width, resolved."""
        if self.d_ff is not None:
            return self.d_ff
        target = int(FFN_EXPANSION * self.d_model)
        return FFN_MULTIPLE * ((target + FFN_MULTIPLE - 1) // FFN_MULTIPLE)

    @property
    def uses_grouped_query_attention(self) -> bool:
        return self.kv_heads != self.n_head

    # -- parameter accounting ----------------------------------------------- #

    @property
    def embedding_parameters(self) -> int:
        return self.vocab_size * self.d_model

    @property
    def attention_parameters_per_layer(self) -> int:
        """Query, key, value and output projections. No biases anywhere."""
        q = self.d_model * self.n_head * self.head_dim
        k = self.d_model * self.kv_heads * self.head_dim
        v = k
        o = self.n_head * self.head_dim * self.d_model
        return q + k + v + o

    @property
    def ffn_parameters_per_layer(self) -> int:
        """SwiGLU: two projections up, one down."""
        return 3 * self.d_model * self.ffn_dim

    @property
    def norm_parameters_per_layer(self) -> int:
        """Two RMSNorms per block, each a single weight vector, no bias."""
        return 2 * self.d_model

    @property
    def parameters_per_layer(self) -> int:
        return (
            self.attention_parameters_per_layer
            + self.ffn_parameters_per_layer
            + self.norm_parameters_per_layer
        )

    @property
    def parameter_count(self) -> int:
        """Total trainable parameters. Checked against the real model by a test."""
        total = self.embedding_parameters
        total += self.n_layer * self.parameters_per_layer
        total += self.d_model  # final RMSNorm
        if not self.tie_embeddings:
            total += self.vocab_size * self.d_model
        return total

    @property
    def non_embedding_parameter_count(self) -> int:
        """The parameters that do the work.

        Reported alongside the total because they scale differently: raising
        ``vocab_size`` inflates the headline number without adding any capacity to
        the layers, and on a small model the embedding can be most of the total.
        """
        return (
            self.parameter_count
            - self.embedding_parameters
            - (0 if self.tie_embeddings else self.vocab_size * self.d_model)
        )

    def parameter_bytes(self, *, bytes_per_parameter: int = 4) -> int:
        """Bytes the parameters occupy. fp32 master weights by default."""
        return self.parameter_count * bytes_per_parameter

    def breakdown(self) -> dict[str, int]:
        """Where the parameters are, for reporting. Sums to ``parameter_count``."""
        return {
            "embedding": self.embedding_parameters,
            "attention": self.n_layer * self.attention_parameters_per_layer,
            "ffn": self.n_layer * self.ffn_parameters_per_layer,
            "norm": self.n_layer * self.norm_parameters_per_layer + self.d_model,
            "output": 0 if self.tie_embeddings else self.vocab_size * self.d_model,
        }

    # -- serialisation ------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Every field, plus the resolved shapes, for checkpoints and reports.

        ``kv_heads`` and ``ffn_dim`` are included even though they are derived,
        because a checkpoint has to be readable by a version whose defaults for
        them differ. Resolving them at write time makes the record unambiguous.
        """
        payload = asdict(self)
        payload["n_kv_head"] = self.kv_heads
        payload["d_ff"] = self.ffn_dim
        payload["head_dim"] = self.head_dim
        payload["parameter_count"] = self.parameter_count
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelConfig:
        """Rebuild from :meth:`to_dict`, ignoring the derived extras.

        Strict about what it accepts, because this rebuilds an *architecture* and the
        cost of guessing is a different model. Every field is checked for presence and
        for type before anything is constructed:

        * A **missing** field used to fall through to the dataclass default. A
          checkpoint of a 4-layer model whose ``n_layer`` had been dropped rebuilt as
          8 layers; the strict weight load then reported that the *weights* did not
          match the model, which sends the reader to the wrong file. Worse, a missing
          ``rope_theta`` or ``norm_eps`` changes no tensor shape at all, so the load
          succeeded and the model quietly generated noise.
        * A field of the **wrong type** used to reach the validators, where
          ``n_layer: "four"`` surfaced as ``'<=' not supported between instances of
          'str' and 'int'`` -- a Python operator error that names no field. And two
          wrong types were not caught at all: ``seq_len: 64.5`` was accepted and failed
          much later inside a slice, and ``tie_embeddings: "no"`` was accepted as
          *true*, because a non-empty string is truthy, so the config said the opposite
          of the file.

        What may be absent is read from the annotations rather than a list kept here: a
        field is optional exactly when its type admits ``None``, which is what
        ``__post_init__`` resolves. Unknown keys are still ignored -- that is what lets
        a config written by an older version load, and ``docs/plan-format.md`` promises
        it.
        """
        return cls(
            **checked_fields(
                cls,
                raw,
                subject="model configuration",
                incomplete_hint=(
                    "This checkpoint or config file is incomplete. Re-create it with "
                    "`trainai train`, which records the full configuration. Filling in "
                    "a default would build a different model and load these weights "
                    "into it."
                ),
            )
        )

    def describe(self) -> str:
        """One line, for logs and run directories."""
        shape = f"L{self.n_layer} d{self.d_model} h{self.n_head}"
        if self.uses_grouped_query_attention:
            shape += f"/{self.kv_heads}kv"
        return f"{shape} ff{self.ffn_dim} ctx{self.seq_len} vocab{self.vocab_size}"

    # -- helpers ------------------------------------------------------------ #

    @staticmethod
    def _require_positive(name: str, value: int) -> None:
        if value <= 0:
            raise ConfigError(
                f"{name} must be positive, got {value}.",
                hint=f"Set {name} to at least 1.",
                details={name: value},
            )

    @staticmethod
    def _largest_divisor(value: int, no_larger_than: int) -> int:
        """The largest divisor of ``value`` that is at most ``no_larger_than``."""
        for candidate in range(min(value, no_larger_than), 0, -1):
            if value % candidate == 0:
                return candidate
        return 1  # pragma: no cover - unreachable, 1 divides everything


@dataclass(frozen=True)
class _Preset:
    """A named starting shape. The planner adjusts from these rather than guessing."""

    name: str
    n_layer: int
    n_head: int
    d_model: int
    seq_len: int
    note: str
    n_kv_head: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


#: Shapes to search over, smallest first. Parameter counts in the notes exclude
#: embeddings, since those depend on the tokenizer the user's corpus produced.
#: These are starting points for M3's planner, not promises about what will fit --
#: whether a shape fits is decided by running it and measuring, never from this
#: table.
PRESETS: tuple[_Preset, ...] = (
    _Preset("tiny", 4, 4, 256, 256, "~3M non-embedding. For checking the pipeline runs."),
    _Preset("small", 6, 6, 384, 512, "~11M non-embedding. First real attempt."),
    _Preset("medium", 8, 8, 512, 512, "~25M non-embedding. Needs a few hundred MB of text."),
    _Preset("large", 12, 12, 768, 1024, "~85M non-embedding. Needs a GPU with room to spare."),
)


def preset(name: str, *, vocab_size: int, **overrides: Any) -> ModelConfig:
    """Build a :class:`ModelConfig` from a named preset.

    Raises:
        ConfigError: If ``name`` is not a preset.
    """
    by_name = {p.name: p for p in PRESETS}
    if name not in by_name:
        raise ConfigError(
            f"There is no model preset called {name!r}.",
            hint=f"Choose one of: {', '.join(by_name)}.",
            details={"requested": name, "available": sorted(by_name)},
        )
    chosen = by_name[name]
    settings: dict[str, Any] = {
        "vocab_size": vocab_size,
        "n_layer": chosen.n_layer,
        "n_head": chosen.n_head,
        "d_model": chosen.d_model,
        "seq_len": chosen.seq_len,
        "n_kv_head": chosen.n_kv_head,
        **chosen.extra,
        **overrides,
    }
    return ModelConfig(**settings)
