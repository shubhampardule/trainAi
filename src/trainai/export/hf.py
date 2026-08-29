"""Exporting a run to the HuggingFace Llama layout.

The model TrainAI trains *is* a Llama, structurally: RMSNorm, rotary position
embeddings in the split convention, SwiGLU feed-forwards, grouped-query attention,
no biases anywhere, optionally tied embeddings. So the export is a rename, not a
conversion -- every tensor goes across unchanged, and the result is what
``LlamaForCausalLM`` expects, byte for byte.

That is worth stating plainly because the alternative is worse. Writing
``"model_type": "gpt2"`` would produce a directory that loads and generates
nonsense: GPT-2 has learned position embeddings, LayerNorm with biases, a fused
``c_attn`` in ``Conv1D`` layout, and a GELU MLP. None of those describe this model,
and a loader has no way to tell -- there is no checksum on "does the architecture
in this config match the weights in this file". A wrong ``model_type`` is a silent
wrong answer, which is the kind of failure this project spends the most effort
avoiding.

Two details that are easy to get wrong:

**Rotary convention.** HF's ``rotate_half`` splits the head dimension in half and
treats the halves as real and imaginary parts. So does :func:`trainai.model.gpt
.apply_rotary`. The other convention in the wild interleaves adjacent pairs, and
converting between them requires permuting the ``q_proj`` and ``k_proj`` rows --
which is exactly what the reference Llama-to-HF script does, and exactly what must
*not* happen here. ``tests/test_export_hf.py`` compares logits against a real
``transformers`` load rather than trusting this paragraph.

**Tied embeddings.** ``safetensors`` refuses to write two names sharing one
storage, so ``lm_head.weight`` is omitted when the model ties it, and
``tie_word_embeddings: true`` in the config tells the loader to re-tie on load.
That is the same arrangement HF's own tied models use.
"""

from __future__ import annotations

import re
from typing import Any

import torch

from trainai.data.tokenizer import EOT_TOKEN, ByteLevelBPE
from trainai.errors import ExportError
from trainai.model.config import ModelConfig
from trainai.model.gpt import GPT

__all__ = [
    "HF_ARCHITECTURE",
    "HF_MODEL_TYPE",
    "hf_config",
    "hf_generation_config",
    "hf_special_tokens_map",
    "hf_state_dict",
    "hf_tokenizer_config",
]

#: What a loader will instantiate from the exported directory.
HF_ARCHITECTURE = "LlamaForCausalLM"
HF_MODEL_TYPE = "llama"

#: Our name -> HF's name, for the tensors outside the block stack.
_TOP_LEVEL = {
    "embed_tokens.weight": "model.embed_tokens.weight",
    "final_norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
}

#: Our name -> HF's name, within one block. ``{i}`` is filled with the block index.
_WITHIN_BLOCK = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn.q_proj.weight": "self_attn.q_proj.weight",
    "attn.k_proj.weight": "self_attn.k_proj.weight",
    "attn.v_proj.weight": "self_attn.v_proj.weight",
    "attn.o_proj.weight": "self_attn.o_proj.weight",
    "ffn.gate_proj.weight": "mlp.gate_proj.weight",
    "ffn.up_proj.weight": "mlp.up_proj.weight",
    "ffn.down_proj.weight": "mlp.down_proj.weight",
}

_BLOCK = re.compile(r"^blocks\.(\d+)\.(.+)$")


def hf_state_dict(model: GPT, *, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """The model's tensors under HuggingFace's names, cast to ``dtype``.

    Every tensor is contiguous and detached from the model's storage, because
    ``safetensors`` writes the underlying buffer and a view of a larger tensor
    would take the whole buffer with it.

    Raises:
        ExportError: A tensor in the model has no mapping. Renaming a module in
            ``trainai.model.gpt`` without updating this table would otherwise drop
            weights from the export and produce a file that loads and is wrong.
    """
    source = model.state_dict()
    tied = set(model.alias_state_dict_keys)
    out: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []

    for name, tensor in source.items():
        if name in tied:
            # Shared storage. The config's tie_word_embeddings re-creates it.
            continue
        mapped = _map_name(name)
        if mapped is None:
            unmapped.append(name)
            continue
        out[mapped] = tensor.detach().to(dtype=dtype, device="cpu").contiguous().clone()

    if unmapped:
        raise ExportError(
            f"{len(unmapped)} weight(s) in this model have no HuggingFace equivalent: "
            + ", ".join(sorted(unmapped)[:5]),
            hint=(
                "The model's module names changed without trainai/export/hf.py "
                "changing with them. Exporting anyway would write a file that loads "
                "and produces wrong output, so it stops here."
            ),
            details={"unmapped": sorted(unmapped)},
        )
    return out


def _map_name(name: str) -> str | None:
    if name in _TOP_LEVEL:
        return _TOP_LEVEL[name]
    match = _BLOCK.match(name)
    if match is None:
        return None
    index, rest = match.group(1), match.group(2)
    suffix = _WITHIN_BLOCK.get(rest)
    return None if suffix is None else f"model.layers.{index}.{suffix}"


def hf_config(
    config: ModelConfig, *, tokenizer: ByteLevelBPE, dtype: torch.dtype
) -> dict[str, Any]:
    """``config.json`` for ``LlamaForCausalLM``.

    Hand-written rather than produced by ``transformers.LlamaConfig``, because
    ``transformers`` is deliberately not a dependency of this project -- see
    ``docs/design/dependencies.md``. The field names are a published format; the
    values all come from the checkpoint.
    """
    return {
        "architectures": [HF_ARCHITECTURE],
        "model_type": HF_MODEL_TYPE,
        "vocab_size": config.vocab_size,
        "hidden_size": config.d_model,
        "intermediate_size": config.ffn_dim,
        "num_hidden_layers": config.n_layer,
        "num_attention_heads": config.n_head,
        "num_key_value_heads": config.kv_heads,
        "head_dim": config.head_dim,
        "hidden_act": "silu",
        "max_position_embeddings": config.seq_len,
        "rms_norm_eps": config.norm_eps,
        "rope_theta": config.rope_theta,
        "rope_scaling": None,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "mlp_bias": False,
        "tie_word_embeddings": config.tie_embeddings,
        "use_cache": True,
        "initializer_range": 0.02,
        "bos_token_id": tokenizer.eot_id,
        "eos_token_id": tokenizer.eot_id,
        # Both spellings: `transformers` renamed `torch_dtype` to `dtype` in v5 and
        # still reads the old key, while v4 only knows the old one. Writing both
        # means the export loads on either.
        "dtype": _dtype_name(dtype),
        "torch_dtype": _dtype_name(dtype),
    }


def hf_generation_config(config: ModelConfig, *, tokenizer: ByteLevelBPE) -> dict[str, Any]:
    """``generation_config.json``, carrying the same sampling defaults as ``chat``.

    A base model at temperature 1.0 with no top-p wanders badly, and someone whose
    first contact with the export is ``model.generate()`` should get the settings
    this project would have used, not the library's defaults.
    """
    from trainai.infer import (
        DEFAULT_MAX_NEW_TOKENS,
        DEFAULT_REPETITION_PENALTY,
        DEFAULT_TEMPERATURE,
        DEFAULT_TOP_P,
    )

    return {
        "do_sample": True,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "repetition_penalty": DEFAULT_REPETITION_PENALTY,
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "bos_token_id": tokenizer.eot_id,
        "eos_token_id": tokenizer.eot_id,
    }


def hf_tokenizer_config(config: ModelConfig) -> dict[str, Any]:
    """``tokenizer_config.json`` for the ``tokenizer.json`` written beside it.

    ``PreTrainedTokenizerFast`` is the class that wraps a ``tokenizers`` file
    directly, which is what this project trains and ships. There is no slow
    (sentencepiece or vocab/merges) counterpart to name, and claiming one would
    make ``use_fast=False`` fail confusingly.
    """
    return {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": config.seq_len,
        "bos_token": EOT_TOKEN,
        "eos_token": EOT_TOKEN,
        "unk_token": None,
        "pad_token": None,
        "clean_up_tokenization_spaces": False,
    }


def hf_special_tokens_map() -> dict[str, Any]:
    return {"bos_token": EOT_TOKEN, "eos_token": EOT_TOKEN}


def _dtype_name(dtype: torch.dtype) -> str:
    """``torch.bfloat16`` -> ``'bfloat16'``, which is what the format spells."""
    return str(dtype).removeprefix("torch.")
