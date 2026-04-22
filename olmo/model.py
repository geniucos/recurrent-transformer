"""
Adapted from
[MosaiclML](https://github.com/mosaicml/examples.git) and
[minGPT](https://github.com/karpathy/minGPT.git)
"""

from __future__ import annotations

import logging
import math
import sys
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)

import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from .aliases import PathOrStr
from .beam_search import BeamSearch, Constraint, FinalSequenceScorer, Sampler
from .config import (
    ActivationCheckpointingStrategy,
    ActivationType,
    BlockType,
    CheckpointType,
    FSDPWrapStrategy,
    InitFnType,
    LayerNormType,
    ModelConfig,
    ShardedCheckpointerType,
    TrainConfig,
)
from .exceptions import OLMoConfigurationError
from .initialization import init_normal
from .torch_util import ensure_finite_, get_cumulative_document_lengths

if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")

__all__ = [
    "LayerNormBase",
    "LayerNorm",
    "RMSLayerNorm",
    "RotaryEmbedding",
    "Activation",
    "GELU",
    "ReLU",
    "SwiGLU",
    "OLMoBlock",
    "OLMoSequentialBlock",
    "OLMo",
    "OLMoOutput",
    "OLMoGenerateOutput",
]

log = logging.getLogger(__name__)


def activation_checkpoint_function(cfg: ModelConfig):
    preserve_rng_state = not (
        (cfg.attention_dropout == 0.0) and (cfg.embedding_dropout == 0.0) and (cfg.residual_dropout == 0.0)
    )
    from torch.utils.checkpoint import checkpoint

    return partial(
        checkpoint,
        preserve_rng_state=preserve_rng_state,
        use_reentrant=False,
    )


def should_checkpoint_block(strategy: Optional[ActivationCheckpointingStrategy], block_idx: int) -> bool:
    if strategy is None:
        return False
    elif (
        (strategy == ActivationCheckpointingStrategy.whole_layer)
        or (strategy == ActivationCheckpointingStrategy.one_in_two and block_idx % 2 == 0)
        or (strategy == ActivationCheckpointingStrategy.one_in_three and block_idx % 3 == 0)
        or (strategy == ActivationCheckpointingStrategy.one_in_four and block_idx % 4 == 0)
        or (strategy == ActivationCheckpointingStrategy.one_in_eight and block_idx % 8 == 0)
        or (strategy == ActivationCheckpointingStrategy.two_in_three and block_idx % 3 != 0)
        or (strategy == ActivationCheckpointingStrategy.three_in_four and block_idx % 4 != 0)
    ):
        return True
    else:
        return False


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """
    Cache for attention biases and other things that would normally be stored as buffers.
    We avoid using buffers because we've run into various issues doing so with FSDP.
    In general it appears the way FSDP handles buffers is not well-defined.
    It doesn't shard them but apparently it does synchronize them across processes, which we want to avoid
    since (A) it isn't necessary, and (B) we sometimes have `-inf` in these biases which might get turned into
    NaNs when they're synchronized due to casting or some other issue.
    """


def _non_meta_init_device(config: ModelConfig) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")


class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input
        else:
            return F.dropout(input, self.p, self.training, self.inplace)


class LayerNormBase(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
    ):
        super().__init__()
        self.config = config
        self.eps = config.layer_norm_eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (elementwise_affine is None and self.config.layer_norm_with_affine):
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=config.init_device))
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=config.init_device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig, size: Optional[int] = None, **kwargs) -> LayerNormBase:
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, size=size, low_precision=False, **kwargs)
        elif config.layer_norm_type == LayerNormType.low_precision:
            return LayerNorm(config, size=size, low_precision=True, **kwargs)
        elif config.layer_norm_type == LayerNormType.rms:
            return RMSLayerNorm(config, size=size, **kwargs)
        elif config.layer_norm_type == LayerNormType.tanh:
            return TanhNorm(config, size=size, **kwargs)
        else:
            raise NotImplementedError(f"Unknown LayerNorm type: '{config.layer_norm_type}'")

    def _cast_if_autocast_enabled(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype())
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype())
        else:
            return tensor

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)  # type: ignore


class LayerNorm(LayerNormBase):
    """
    The default :class:`LayerNorm` implementation which can optionally run in low precision.
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        low_precision: bool = False,
        elementwise_affine: Optional[bool] = None,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine)
        self.low_precision = low_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = (
                self._cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
            )
            downcast_bias = self._cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(
                    downcast_x, self.normalized_shape, weight=downcast_weight, bias=downcast_bias, eps=self.eps
                )
        else:
            return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class RMSLayerNorm(LayerNormBase):
    """
    RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                return self.weight * x
        else:
            return x

class TanhNorm(LayerNormBase):
    """
    Tanh instead of norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine)
        self.alpha = nn.Parameter(torch.ones(1,) * config.tanh_norm_alpha, requires_grad=config.tanh_trainable_alpha)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            x = F.tanh(self.alpha * x)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                return self.weight * x
        else:
            return x



class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        # Warm up cache.
        self.get_rotary_embedding(config.max_sequence_length, _non_meta_init_device(config))

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            (pos_sin := self.__cache.get("rope_pos_sin")) is not None
            and (pos_cos := self.__cache.get("rope_pos_cos")) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self.__cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self.__cache["rope_pos_cos"] = pos_cos
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            dim = self.config.d_model // self.config.n_heads
            inv_freq = 1.0 / (
                self.config.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
            )
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = positions.sin()[None, None, :, :], positions.cos()[None, None, :, :]
        self.__cache["rope_pos_sin"] = pos_sin
        self.__cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = q_.shape[-2], k_.shape[-2]  # could be different if layer_past not None
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len, :],
                pos_cos[:, :, key_len - query_len : key_len, :],
                q_,
            )
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)


class Activation(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_multiplier(self) -> float:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig) -> Activation:
        if config.activation_type == ActivationType.gelu:
            return cast(Activation, GELU(approximate="none"))
        elif config.activation_type == ActivationType.relu:
            return cast(Activation, ReLU(inplace=False))
        elif config.activation_type == ActivationType.swiglu:
            return SwiGLU(config)
        else:
            raise NotImplementedError(f"Unknown activation: '{config.activation_type}'")


class GELU(nn.GELU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class ReLU(nn.ReLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SwiGLU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5


def causal_attention_bias(seq_len: int, device: torch.device) -> torch.FloatTensor:
    att_bias = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.float),
        diagonal=1,
    )
    att_bias.masked_fill_(att_bias == 1, torch.finfo(att_bias.dtype).min)
    return att_bias.view(1, 1, seq_len, seq_len)  # type: ignore


def get_causal_attention_bias(cache: BufferCache, seq_len: int, device: torch.device) -> torch.Tensor:
    if (causal_bias := cache.get("causal_attention_bias")) is not None and causal_bias.shape[-1] >= seq_len:
        if causal_bias.device != device:
            causal_bias = causal_bias.to(device)
            cache["causal_attention_bias"] = causal_bias
        return causal_bias
    with torch.autocast(device.type, enabled=False):
        causal_bias = causal_attention_bias(seq_len, device)
    cache["causal_attention_bias"] = causal_bias
    return causal_bias


def alibi_attention_bias(seq_len: int, config: ModelConfig, device: torch.device) -> torch.FloatTensor:
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, 1, seq_len)

    # shape: (1, 1, seq_len, seq_len)
    alibi_bias = alibi_bias - torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, seq_len, 1)
    alibi_bias.abs_().mul_(-1)

    # shape: (n_heads,)
    m = torch.arange(1, config.n_heads + 1, dtype=torch.float, device=device)
    m.mul_(config.alibi_bias_max / config.n_heads)

    # shape: (1, n_heads, seq_len, seq_len)
    return alibi_bias * (1.0 / (2 ** m.view(1, config.n_heads, 1, 1)))  # type: ignore

class OLMoBlock(nn.Module):
    """
    A base class for transformer block implementations.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        assert config.d_model % config.n_heads == 0

        self._activation_checkpoint_fn: Optional[Callable] = None

        # Dropout.
        self.dropout = Dropout(config.residual_dropout)

        # Layer norms.
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        if config.attention_layer_norm:
            assert config.effective_n_kv_heads is not None
            self.k_norm = LayerNormBase.build(
                config,
                size=(config.d_model // config.n_heads) * config.effective_n_kv_heads,
                elementwise_affine=config.attention_layer_norm_with_affine,
            )
            self.q_norm = LayerNormBase.build(config, elementwise_affine=config.attention_layer_norm_with_affine)

        # Make sure QKV clip coefficient is positive, otherwise it's not well-defined.
        if config.clip_qkv is not None:
            assert config.clip_qkv > 0

        # Activation function.
        self.act = Activation.build(config)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        # Attention output projection.
        self.attn_out = nn.Linear(
            config.d_model, config.d_model, bias=config.include_bias, device=config.init_device
        )

        # Feed-forward output projection.
        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_out._is_residual = True  # type: ignore

        # Rotary embeddings.
        if self.config.rope:
            self.rotary_emb = RotaryEmbedding(config, self.__cache)

        self.flash_attn_func = None
        self.flash_attn_varlen_func = None
        if config.flash_attention:
            try:
                from flash_attn import (  # type: ignore
                    flash_attn_func,
                    flash_attn_varlen_func,
                )

                self.flash_attn_func = flash_attn_func
                self.flash_attn_varlen_func = flash_attn_varlen_func
            except ModuleNotFoundError:
                pass

    def reset_parameters(self):
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()

        if self.config.init_fn == InitFnType.normal:
            attn_out_std = ff_out_std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor

        elif self.config.init_fn == InitFnType.mitchell:
            attn_out_std = 1 / (math.sqrt(2 * self.config.d_model * (self.layer_id + 1)))
            ff_out_std = 1 / (math.sqrt(2 * self.ff_out.in_features * (self.layer_id + 1)))
            cutoff_factor = self.config.init_cutoff_factor or 3.0

        elif self.config.init_fn == InitFnType.full_megatron:
            attn_out_std = ff_out_std = self.config.init_std / math.sqrt(2.0 * self.config.n_layers)
            cutoff_factor = self.config.init_cutoff_factor or 3.0

        else:
            raise NotImplementedError(self.config.init_fn)

        init_normal(self.attn_out, std=attn_out_std, init_cutoff_factor=cutoff_factor)
        init_normal(self.ff_out, std=ff_out_std, init_cutoff_factor=cutoff_factor)

    def set_activation_checkpointing(
        self, strategy: Optional[ActivationCheckpointingStrategy], checkpoint_func: Optional[Callable] = None
    ):
        if strategy == ActivationCheckpointingStrategy.fine_grained:
            self._activation_checkpoint_fn = checkpoint_func or activation_checkpoint_function(self.config)
        else:
            self._activation_checkpoint_fn = None

    @classmethod
    def _cast_attn_bias(cls, bias: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        target_dtype = input_dtype
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if bias.device.type == "cuda" and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif bias.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            target_dtype = torch.get_autocast_cpu_dtype()
        elif bias.device.type == "mps":
            target_dtype = torch.get_autocast_dtype("mps")
        if bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
            ensure_finite_(bias, check_neg_inf=True, check_pos_inf=False)
        return bias

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes scaled dot product attention on query, key and value tensors, using an optional
        attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.
        """
        if max_doc_len is not None and cu_doc_lens is not None:
            assert self.flash_attn_varlen_func is not None, "flash-attn is required for document masking"
            assert attn_mask is None, "attn-mask is currently not supported with document masking"
            B, T, D = q.size(0), q.size(2), q.size(3)
            r = self.flash_attn_varlen_func(
                q.transpose(1, 2).view(B * T, -1, D),
                k.transpose(1, 2).view(B * T, -1, D),
                v.transpose(1, 2).view(B * T, -1, D),
                cu_doc_lens,
                cu_doc_lens,
                max_doc_len,
                max_doc_len,
                dropout_p=dropout_p,
                causal=is_causal,
            )
            return r.view(B, T, -1, D).transpose(1, 2)
        elif self.flash_attn_func is not None and attn_mask is None:
            r = self.flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, causal=is_causal
            )
            return r.transpose(1, 2)
        else:
            # torch's sdpa doesn't support GQA, so we're doing this
            assert k.size(1) == v.size(1)
            num_kv_heads = k.size(1)
            num_q_heads = q.size(1)
            if num_q_heads != num_kv_heads:
                assert num_q_heads % num_kv_heads == 0
                k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
                v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)

            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None

        if self.config.rope:
            # Apply rotary embeddings.
            q, k = self.rotary_emb(q, k)

        # patched BUG in olmo under layer_past;
        if attention_bias is not None and (
            self.config.alibi
            or self.config.block_type not in (BlockType.recurrent, BlockType.recurrent_autograd)
        ):
            # Resize and cast attention bias.
            # The current dtype of the attention bias might not match the dtype that the SDP attn function will
            # run in if AMP is enabled, and this can be a problem if some tokens are masked out due to padding
            # as down-casting the attention bias to the autocast precision will result in -infs, which will
            # cause the SDP attn function to produce NaNs.
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
            )

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        att = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=attention_bias is None, # this is actually a bug under generation when alibi is not used
            max_doc_len=max_doc_len,
            cu_doc_lens=cu_doc_lens,
        )

        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection.
        return self.attn_out(att), present

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        raise NotImplementedError

    @classmethod
    def build(cls, layer_id: int, config: ModelConfig, cache: BufferCache) -> OLMoBlock:
        if config.block_type == BlockType.sequential:
            return OLMoSequentialBlock(layer_id, config, cache)
        elif config.block_type == BlockType.llama:
            return OLMoLlamaBlock(layer_id, config, cache)
        elif config.block_type == BlockType.recurrent:
            return OLMoRecurrentBlockTiled(layer_id, config, cache)
        elif config.block_type == BlockType.recurrent_autograd:
            return OLMoRecurrentAutogradBlock(layer_id, config, cache)
        else:
            raise NotImplementedError(f"Unknown block type: '{config.block_type}'")


class OLMoSequentialBlock(OLMoBlock):
    """
    This is a typical transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). To compute it as ``LN(MLP(x + LN(Attention(x))))``,
    use the flag `norm_after`.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Attention input projection. Projects x -> (q, k, v)

        head_dim = config.d_model // config.n_heads
        self.fused_dims = (
            config.d_model,
            config.effective_n_kv_heads * head_dim,
            config.effective_n_kv_heads * head_dim,
        )
        self.att_proj = nn.Linear(
            config.d_model, sum(self.fused_dims), bias=config.include_bias, device=config.init_device
        )
        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

        # Layer norms.
        self.attn_norm = LayerNorm.build(config, size=config.d_model)
        self.ff_norm = LayerNorm.build(config, size=config.d_model)

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.

        if self.config.init_fn == InitFnType.normal:
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor
        elif self.config.init_fn == InitFnType.mitchell:
            std = 1 / math.sqrt(self.config.d_model)
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        elif self.config.init_fn == InitFnType.full_megatron:
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        else:
            raise NotImplementedError(self.config.init_fn)

        if hasattr(self, 'att_proj'): # this is because recurrent inherits it and may delete att_proj
            init_normal(self.att_proj, std, cutoff_factor)
        init_normal(self.ff_proj, std, cutoff_factor)
        # this is for when BlockRecurrent is used, to store the std and cutoff factor for the attention and feedforward output projections
        self.std, self.cutoff_factor = std, cutoff_factor

    def _real_forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #  - for group query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)

        # apply norm before
        if not self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                h = self._activation_checkpoint_fn(self.attn_norm, x)
            else:
                h = self.attn_norm(x)
        else:
            h = x

        qkv = self.att_proj(h)

        if self.config.clip_qkv is not None:
            qkv.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        q, k, v = qkv.split(self.fused_dims, dim=-1)

        # Get attention scores.
        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention,
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                max_doc_len=max_doc_len,
                cu_doc_lens=cu_doc_lens,
            )
        else:
            # att, cache = h, None
            att, cache = self.attention(
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                max_doc_len=max_doc_len,
                cu_doc_lens=cu_doc_lens,
            )

        if self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                att = self._activation_checkpoint_fn(self.attn_norm, att)
            else:
                att = self.attn_norm(att)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x

        if not self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
            else:
                x = self.ff_norm(x)

        x = self.ff_proj(x)

        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = self.ff_out(x)

        if self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
            else:
                x = self.ff_norm(x)

        x = self.dropout(x)
        x = og_x + x

        return x, cache
    
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        with torch.autocast('cuda', enabled=True, dtype=self.config.precision):
            out, cache = self._real_forward(x, attention_bias, layer_past, use_cache, max_doc_len, cu_doc_lens)
        return out, cache

class PreAttentionBlock(nn.Module):
    def __init__(self, obj: OLMoSequentialBlock, should_do_norm_and_permute: bool = True):
        super().__init__()
        self.attn_norm = obj.attn_norm
        self.kv_proj = obj.kv_proj
        self.q_proj = obj.q_proj
        self.clip_qkv = obj.config.clip_qkv
        self.fused_dims = obj.fused_dims
        self.norm_after = obj.config.norm_after

        self.should_do_norm_and_permute = should_do_norm_and_permute
        if self.should_do_norm_and_permute:
            self.head_dim = obj.config.d_model // obj.config.n_heads
            self.effective_n_kv_heads = obj.config.effective_n_kv_heads
            self.n_heads = obj.config.n_heads
            self.q_norm = obj.q_norm
            self.k_norm = obj.k_norm

    @torch.compile(dynamic=False)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_normed = self.attn_norm(x) if not self.norm_after else x # (B, L, D)
        kv = self.kv_proj(x_normed)
        q = self.q_proj(x_normed)
        if self.clip_qkv is not None:
            kv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
            q.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        k_init, v_init = kv.split(self.fused_dims[1:], dim=-1)

        if self.should_do_norm_and_permute:
            # normalize q, k_init and permute to (B, n_heads, L, head_dim)
            if self.q_norm is not None and self.k_norm is not None:
                dtype = k_init.dtype
                q = self.q_norm(q).to(dtype=dtype)
                k_init = self.k_norm(k_init).to(dtype=dtype)
            B = x.shape[0]
            k_init = k_init.view(B, -1, self.effective_n_kv_heads, self.head_dim).transpose(1, 2)
            v_init = v_init.view(B, -1, self.effective_n_kv_heads, self.head_dim).transpose(1, 2)
            q = q.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        return q, k_init, v_init

class PostAttentionBlock(nn.Module):
    def __init__(self, obj: OLMoSequentialBlock):
        super().__init__()
        self.attn_norm = obj.attn_norm
        self.ff_norm = obj.ff_norm
        self.ff_proj = obj.ff_proj
        self.act = obj.act
        self.ff_out = obj.ff_out
        self.dropout = obj.dropout
        self.norm_after = obj.config.norm_after

    def forward(self, x: torch.Tensor, att: torch.Tensor) -> torch.Tensor:
        if self.norm_after:
            att = self.attn_norm(att)
        x = x + self.dropout(att)
        og_x = x
        if not self.norm_after:
            x = self.ff_norm(x)
        x = self.ff_out(self.act(self.ff_proj(x)))
        if self.norm_after:
            x = self.ff_norm(x)
        x = self.dropout(x)
        x = og_x + x
        return x

class OLMoRecurrentBlockBase(OLMoSequentialBlock):
    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache) # initializations of:
        # self.att_proj, self.ff_proj, self.attn_norm, self.ff_norm, self.final_kv_proj
        del self.att_proj
        self.q_proj = self.kv_proj = nn.Linear(
            config.d_model, self.fused_dims[0], bias=config.include_bias, device=config.init_device
        )
        self.kv_proj = nn.Linear(
            config.d_model, sum(self.fused_dims[1:]), bias=config.include_bias, device=config.init_device
        )
        self.pre_attention_block = PreAttentionBlock(self)
        self.post_attention_block = PostAttentionBlock(self)

    def reset_parameters(self):
        super().reset_parameters()
        init_normal(self.q_proj, self.std, self.cutoff_factor)
        init_normal(self.kv_proj, self.std, self.cutoff_factor)
    
    def init_from_sequential_block(self, sequential_block: OLMoSequentialBlock):
        self.kv_proj.weight.data.copy_(sequential_block.att_proj.weight.data[self.fused_dims[0]:, :])
        if sequential_block.att_proj.bias is not None:
            self.kv_proj.bias.data.copy_(sequential_block.att_proj.bias.data[self.fused_dims[0]:])
        
        self.q_proj.weight.data.copy_(sequential_block.att_proj.weight.data[:self.fused_dims[0], :])
        if sequential_block.att_proj.bias is not None:
            self.q_proj.bias.data.copy_(sequential_block.att_proj.bias.data[:self.fused_dims[0]])
    
        self.ff_proj.weight.data.copy_(sequential_block.ff_proj.weight.data)
        if sequential_block.ff_proj.bias is not None:
            self.ff_proj.bias.data.copy_(sequential_block.ff_proj.bias.data)
        
        self.attn_out.weight.data.copy_(sequential_block.attn_out.weight.data)
        if sequential_block.attn_out.bias is not None:
            self.attn_out.bias.data.copy_(sequential_block.attn_out.bias.data)
        
        self.ff_out.weight.data.copy_(sequential_block.ff_out.weight.data)
        if sequential_block.ff_out.bias is not None:
            self.ff_out.bias.data.copy_(sequential_block.ff_out.bias.data)
        
        # self.attn_norm = sequential_block.attn_norm
        # self.ff_norm = sequential_block.ff_norm
        # if hasattr(sequential_block, 'k_norm'):
        #     assert sequential_block.k_norm is not None
        #     self.k_norm = sequential_block.k_norm
        # if hasattr(sequential_block, 'q_norm'):
        #     assert sequential_block.q_norm is not None
        #     self.q_norm = sequential_block.q_norm

    def _real_forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError(
            "OLMoRecurrentBlockBase only holds shared weights for recurrent Modules; use block_type recurrent or recurrent_autograd."
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        assert (layer_past is None) and (use_cache is False) and (max_doc_len is None) and (cu_doc_lens is None), \
               "Recurrent block does not support layer_past, use_cache, max_doc_len, cu_doc_lens"
        with torch.autocast('cuda', enabled=True, dtype=self.config.precision):
            out = self._real_forward(x, attention_bias)
        return out, None

class OLMoRecurrentAutogradBlock(OLMoRecurrentBlockBase):
    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)

    def _real_forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        outputs = []
        B, L, D = x.shape
        q, k_init, v_init = self.pre_attention_block(x)

        final_k = [] # list of (B, n_heads, 1, head_dim)
        final_v = [] # list of (B, n_heads, 1, head_dim)
        head_dim = self.config.d_model // self.config.n_heads
        kv_dim = head_dim * self.config.effective_n_kv_heads
        
        def shuffle_appropriately(k: torch.Tensor) -> torch.Tensor:
            assert k.ndim == 3 and k.shape[0] == B and k.shape[2] == kv_dim
            return k.view(B, -1, self.config.effective_n_kv_heads, head_dim).transpose(1, 2)
        
        def augment_cache(k, v):
            if self.q_norm is not None and self.k_norm is not None:
                dtype = k.dtype
                k = self.k_norm(k).to(dtype=dtype)
            final_k.append(shuffle_appropriately(k))
            final_v.append(shuffle_appropriately(v))
        
        for t in range(L):
            x_t = x[:, t:(t+1), :] # (B, 1, D)
            q_t = q[:, :, t:(t+1), :] # (B, n_heads, 1, head_dim)
            k_t = k_init[:, :, t:(t+1), :] # (B, effective_n_kv_heads, 1, head_dim)
            v_t = v_init[:, :, t:(t+1), :] # (B, effective_n_kv_heads, 1, head_dim)
            assert (x_t.shape== (B, 1, D)) and (q_t.shape == (B, self.config.n_heads, 1, head_dim))
            assert k_t.shape == v_t.shape == (B, self.config.effective_n_kv_heads, 1, head_dim)
            
            # beginning of attention
            curr_attention_bias = \
                attention_bias[:, :, t:(t+1), :(t + 1)] if attention_bias is not None else \
                torch.zeros((B, self.config.n_heads, 1, t + 1), dtype=k_t.dtype, device=k_t.device)

            assert not self.config.rope, "RoPE is not supported for recurrent_autograd block"
            assert self.config.effective_n_kv_heads == self.config.n_heads, \
            "effective_n_kv_heads must be equal to n_heads for recurrent_autograd block"

            big_k = torch.cat(final_k + [k_t], dim=-2) # (B, n_heads, t + 1, head_dim)
            big_v = torch.cat(final_v + [v_t], dim=-2) # (B, n_heads, t + 1, head_dim)

            # the following is equivalent to
            # att_t = F.scaled_dot_product_attention(q_t, big_k, big_v, attn_mask=curr_attention_bias)

            attn_weights = torch.matmul(big_k, q_t.transpose(-2, -1)) / math.sqrt(head_dim) # (B, n_heads, t + 1, 1)
            attn_weights = attn_weights.transpose(-1, -2) + curr_attention_bias # (B, n_heads, 1, t + 1)
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            att_t = torch.matmul(attn_weights, big_v) # (B, n_heads, 1, head_dim)

            att_t = att_t.transpose(1, 2).contiguous().reshape(B, 1, D)
            att_t = self.attn_out(att_t)
            # attention fully computed

            out_t = self.post_attention_block(x_t, att_t)
            outputs.append(out_t)
            assert att_t.shape == out_t.shape == (B, 1, D)

            final_kv = self.kv_proj(self.pre_attention_block.attn_norm(out_t))
            if self.config.clip_qkv is not None:
                final_kv.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            final_k_t, final_v_t = final_kv.split(self.fused_dims[1:], dim=-1)
            assert final_k_t.shape == final_v_t.shape == (B, 1, kv_dim)
            augment_cache(final_k_t, final_v_t)
        
        cat_outs = torch.cat(outputs, dim=1) # (B, T, D)
        return cat_outs

@torch.compile(dynamic=False)
def block_attention_add(
    att_0, max_logit_0, sum_scores_0,
    k, v, q, attention_bias):
    assert (k.shape == v.shape) and (k.shape[:-2] == q.shape[:-2]) and (k.shape[-1] == q.shape[-1])
    n, m = q.shape[-2], k.shape[-2]
    # q's shape: (B, n_heads, n, head_dim)
    # (k, v)'s shape: (B, n_heads, m, head_dim)
    # assert (attention_bias == None or attention_bias.shape[-2:] == (n, m)) # commented out for compilation
    assert (att_0.shape[0] == n) and \
            (max_logit_0.shape == (sum_scores_0.shape))

    attn_weights = q @ k.transpose(-2, -1) # (B, n_heads, n, m)
    if attention_bias is not None:
        attn_weights += attention_bias
    
    max_logit = torch.max(max_logit_0.permute(1, 2, 0), attn_weights.max(dim=-1).values) # (B, n_heads, n)
    attn_weights = torch.exp(attn_weights - max_logit.unsqueeze(-1)) # (B, n_heads, n, m)
    att = attn_weights @ v # (B, n_heads, n, head_dim)

    max_logit = max_logit.permute(2, 0, 1)
    new_scaling = torch.exp(max_logit_0 - max_logit)
    att_0 = att_0 * new_scaling.unsqueeze(-1) + att.permute(2, 0, 1, 3)
    sum_scores_0 = sum_scores_0 * new_scaling + attn_weights.sum(dim=-1).permute(2, 0, 1)
    return att_0, max_logit, sum_scores_0

@torch.compile(fullgraph=True)
def recompute_alphas(final_k, k_init, q, attention_bias):
    alphas = (final_k.permute(1, 2, 0, 3) @ q.permute(1, 2, 3, 0))  # (B, n_heads, L, L)
    alphas.diagonal(dim1=2, dim2=3).copy_(((q * k_init).sum(dim=-1).permute(1, 2, 0)))  # (B, n_heads, L)
    if attention_bias is not None:
        # assert attention_bias.shape == (1, block.config.n_heads, L, L)
        alphas += attention_bias.permute(0, 1, 3, 2)  # (B, n_heads, L, L)
    L = alphas.size(-1)
    lower_mask = torch.ones((L, L), device=alphas.device, dtype=torch.bool).tril(diagonal=-1)
    alphas.masked_fill_(lower_mask, float("-inf")) # (B, n_heads, L, L)
    # ensure we do softmax in float32 to avoid numerical issues
    alphas = torch.softmax(alphas, dim=-2, dtype=torch.float32).to(q.dtype)  # (B, n_heads, L, L), batch, head, key, query
    return alphas


@torch.compile(fullgraph=True)
def recompute_atts(final_v, v_init, alphas):
    # alphas: (B, n_heads, L, L) - batch, head, key, query
    atts = torch.matmul(final_v.permute(1, 2, 3, 0), alphas).permute(3, 0, 1, 2)  # (L, B, n_heads, head_dim)
    atts += (v_init - final_v) * alphas.diagonal(dim1=2, dim2=3).permute(2, 0, 1).unsqueeze(-1)
    return atts


@torch.compile(fullgraph=True)
def mlp_batched_body(all_atts, x, block, dtype):
    with torch.autocast('cuda', enabled=True, dtype=dtype):
        all_atts = all_atts.transpose(0, 1).reshape(x.shape)
        all_atts = block.attn_out(all_atts)
        mlp_outs = block.post_attention_block(x, all_atts)
    return mlp_outs    


class OLMoRecurrentBlockTiledFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, attention_bias, block):
        B, L, D = x.shape

        head_dim = block.config.d_model // block.config.n_heads
        kv_dim = head_dim * block.config.effective_n_kv_heads
        scale = 1.0 / math.sqrt(head_dim)

        q, k_init, v_init = block.pre_attention_block(x) # all (B, n_heads, L, head_dim)
        q = q.permute(2, 0, 1, 3).contiguous() * scale # (L, B, n_heads, head_dim)
        k_init = k_init.permute(2, 0, 1, 3) # (L, B, n_heads, head_dim)
        
        def shuffle_appropriately(k: torch.Tensor) -> torch.Tensor:
            # k : (B, 1, kv_dim) -> (1, B, n_heads, head_dim)
            assert k.ndim == 3 and k.shape[0] == B and k.shape[2] == kv_dim
            return k.view(B, -1, block.config.effective_n_kv_heads, head_dim).permute(1, 0, 2, 3)
        
        # saved for backward pass
        outs = []  # list of (B, 1, D)

        # initialize attention state (atts, max_logit, sum_scores)
        atts = v_init.permute(2, 0, 1, 3).to(x.dtype).contiguous() # (L, B, n_heads, head_dim)
        max_logit = (k_init * q).sum(dim=-1)  # (L, B, n_heads)
        if attention_bias is not None:
            max_logit += attention_bias.diagonal(dim1=-2, dim2=-1).expand(B, -1, -1).permute(2, 0, 1)
        sum_scores = torch.ones((L, B, block.config.n_heads), dtype=x.dtype, device=q.device) # (B, L, n_heads) - this is in high precision

        # no longer needed now that I initialized atts, max_logit, sum_scores
        del k_init, v_init

        # cache for keys and values (L, B, n_heads, head_dim) - position along first axis for contiguous slicing
        final_k = torch.empty((L, B, block.config.n_heads, head_dim), dtype=q.dtype, device=q.device)
        final_v = torch.empty((L, B, block.config.n_heads, head_dim), dtype=q.dtype, device=q.device)

        for t in range(L):
            att_t = (atts[t:(t+1)] / sum_scores[t:(t+1)].unsqueeze(-1)).to(q.dtype) # (1, B, n_heads, head_dim)
            x_t = x[:, t:(t+1), :]
            assert (x_t.shape == (B, 1, D))

            att_t = att_t.transpose(0, 1).reshape(B, 1, D) # (B, 1, D)
            att_t = block.attn_out(att_t)
            out_t = block.post_attention_block(x_t, att_t)
            outs.append(out_t)
            assert att_t.shape == out_t.shape == x_t.shape == (B, 1, D)

            # compute final_kv
            final_kv = block.kv_proj(block.pre_attention_block.attn_norm(out_t))
            if block.config.clip_qkv is not None:
                final_kv.clamp_(min=-block.config.clip_qkv, max=block.config.clip_qkv)
            final_k_t, final_v_t = final_kv.split(block.fused_dims[1:], dim=-1)
            if block.q_norm is not None and block.k_norm is not None:
                dtype = final_k_t.dtype
                final_k_t = block.k_norm(final_k_t).to(dtype=dtype)
            final_k_t = shuffle_appropriately(final_k_t)
            final_v_t = shuffle_appropriately(final_v_t)
            assert final_k_t.shape == final_v_t.shape == (1, B, block.config.n_heads, head_dim)
            # update cache for forward pass
            final_k[t:(t+1)] = final_k_t
            final_v[t:(t+1)] = final_v_t
            
            if t + 1 == L:
                break
            T = (t + 1) & -(t + 1) # largest power of 2 dividing t + 1
            
            # consider the contribution of last T kvs to next T qs
            k_slice = final_k[(t+1-T):(t+1)].permute(1, 2, 0, 3)  # (B, n_heads, T, head_dim)
            v_slice = final_v[(t+1-T):(t+1)].permute(1, 2, 0, 3)  # (B, n_heads, T, head_dim)
            q_slice = q[t+1:(t+1+T)].permute(1, 2, 0, 3) # (B, n_heads, T, head_dim)
            curr_s = slice(t+1, t+1+T)
            atts[curr_s], max_logit[curr_s], sum_scores[curr_s] = block_attention_add(
                atts[curr_s], max_logit[curr_s], sum_scores[curr_s],
                k_slice,
                v_slice,
                q_slice,
                attention_bias[:, :, t+1:(t+1+T), (t+1-T):(t+1)] if attention_bias is not None else None
            )
        
        final_outs = torch.cat(outs, dim=1)
        ctx.save_for_backward(x, final_outs)
        ctx.block = block
        ctx.dtype = q.dtype
        ctx.attention_bias = attention_bias
        return final_outs

    @staticmethod
    def backward(ctx, grad_output):
        saved_list = ctx.saved_tensors
        x, outs = saved_list[0], saved_list[1]
        block = ctx.block
        autocast_ctx = torch.autocast('cuda', enabled=True, dtype=ctx.dtype)
        attention_bias = getattr(ctx, "attention_bias", None)  # (B, n_heads, L, L) or None

        B, L, D = x.shape
        head_dim = block.config.d_model // block.config.n_heads
        kv_dim = head_dim * block.config.effective_n_kv_heads
        scale = 1.0 / math.sqrt(head_dim)

        def shuffle_appropriately(k: torch.Tensor) -> torch.Tensor:
            # k : (B, 1, kv_dim) -> (1, B, n_heads, head_dim)
            assert k.ndim == 3 and k.shape[0] == B and k.shape[2] == kv_dim
            return k.view(B, -1, block.config.effective_n_kv_heads, head_dim).permute(1, 0, 2, 3)

        x = x.detach().requires_grad_(True)
        with autocast_ctx:
            with torch.enable_grad():
                q, k_init, v_init = block.pre_attention_block(x) # all (B, n_heads, L, head_dim)
                q = q.permute(2, 0, 1, 3).contiguous() * scale # (L, B, n_heads, head_dim)
                k_init = k_init.permute(2, 0, 1, 3).contiguous() # (L, B, n_heads, head_dim)
                v_init = v_init.permute(2, 0, 1, 3).contiguous() # (L, B, n_heads, head_dim)

        with autocast_ctx:
            final_ks = [] # list of (1, B, n_heads, head_dim) - with grad_fns
            final_vs = [] # list of (1, B, n_heads, head_dim) - with grad_fns
            outs_detached = [] # list of (B, 1, D)
            for t in range(L):
                out_t = outs[:, t:(t+1), :].detach().requires_grad_(True)
                outs_detached.append(out_t)
                with torch.enable_grad():
                    final_kv = block.kv_proj(block.pre_attention_block.attn_norm(out_t))
                    if block.config.clip_qkv is not None:
                        final_kv.clamp_(min=-block.config.clip_qkv, max=block.config.clip_qkv)
                    final_k_t, final_v_t = final_kv.split(block.fused_dims[1:], dim=-1)
                    if block.q_norm is not None and block.k_norm is not None:
                        dtype = final_k_t.dtype
                        final_k_t = block.k_norm(final_k_t).to(dtype=dtype)
                    final_k_t = shuffle_appropriately(final_k_t)
                    final_v_t = shuffle_appropriately(final_v_t)
                assert final_k_t.shape == final_v_t.shape == (1, B, block.config.n_heads, head_dim)
                # store for backward pass
                final_ks.append(final_k_t)
                final_vs.append(final_v_t)
            final_k = torch.cat(list(map(lambda x: x.detach(), final_ks)), dim=0)  # (L, B, n_heads, head_dim)
            final_v = torch.cat(list(map(lambda x: x.detach(), final_vs)), dim=0)  # (L, B, n_heads, head_dim)

        # because of the blocking thing in the forward pass, we need to recompute the alphas and atts here
        alphas = recompute_alphas(final_k, k_init, q, attention_bias)
        atts = recompute_atts(final_v, v_init, alphas)

        k_grads = torch.zeros(final_k.shape, dtype=x.dtype, device=final_k.device)  # (L, B, n_heads, head_dim)
        v_grads = torch.zeros(final_v.shape, dtype=x.dtype, device=final_v.device)  # (L, B, n_heads, head_dim)
        all_grads = [] # list of (B, 1, D)
        gs = torch.empty((L, B, block.config.n_heads, head_dim), device=q.device, dtype=q.dtype)
        g_dot_atts = torch.empty((L, B, block.config.n_heads), device=q.device, dtype=q.dtype)
        # shuffle final_v to a better shape:
        final_v = final_v.permute(1, 2, 0, 3).contiguous() # (B, n_heads, L, head_dim)

        # do all MLPs in the same autocast context (to aid caching)
        mlp_out_ts = {}
        atts_with_grad = {}
        def redo_forward_mlp_range(t_start, t_end):
            mlp_out_ts.clear()
            atts_with_grad.clear()
            for t in range(t_end - 1, t_start - 1, -1):
                atts_with_grad[t] = atts[t:(t+1)].detach()
            with autocast_ctx:
                with torch.enable_grad():
                    for t in range(t_end - 1, t_start - 1, -1):
                        if t >= 0:
                            x_t = x[:, t:(t+1), :].detach().requires_grad_(False)
                            atts_with_grad[t].requires_grad_(True)
                            att_t_reshaped = atts_with_grad[t].transpose(0, 1).reshape(B, 1, D)
                            att_t_reshaped = block.attn_out(att_t_reshaped)
                            mlp_out_ts[t] = block.post_attention_block(x_t, att_t_reshaped)

        shared_segment = (L + block.config.bwd_mlp_chunks - 1) // block.config.bwd_mlp_chunks
        for t in range(L-1, -1, -1):
            torch.autograd.backward(
                (final_ks[t], final_vs[t]),
                grad_tensors=(k_grads[t:(t+1)], v_grads[t:(t+1)])
            ) # (1, B, n_heads, head_dim), (1, B, n_heads, head_dim)
            
            if not t in mlp_out_ts:
                redo_forward_mlp_range(t - shared_segment + 1, t + 1)
            # reverting the MLP computation
            grad_to_propagate = grad_output[:, t:(t+1), :] + outs_detached[t].grad
            g = torch.autograd.grad(
                mlp_out_ts[t],
                (atts_with_grad[t],),
                grad_outputs=grad_to_propagate
            )[0] # (1, B, n_heads, head_dim)
            all_grads.append(grad_to_propagate)

            # now I need to propagate att's grad to q, k, v, k_init, v_init
            # O(1) ops
            # g_dot_att = (atts[t].unsqueeze(-2) @ g.unsqueeze(-1)).squeeze(-1)  # (1, B, n_heads, 1), <att, g>
            g_dot_atts[t:(t+1)] = (atts_with_grad[t] * g).sum(dim=-1) # (1, B, n_heads)
            gs[t:(t+1)] = g # (1, B, n_heads, head_dim)

            if t == 0:
                continue
            T = t & -t # largest power of 2 dividing t

            # v_grads[:t] += alpha[:t] * g  # (t, B, n_heads, head_dim)
            # contribution of g[t:t+T] to v_grads[t-T:t] -> v_grads[t-T:t] += alpha[t:t+T, t-T:t, :, :] * g[t:t+T]
            if T > 1:
                v_grads[t-T:t] += torch.matmul(
                    alphas[:, :, t-T:t, t:t+T],
                    gs[t:t+T].permute(1, 2, 0, 3)
                ).permute(2, 0, 1, 3)
            else:
                v_grads[t-1] += alphas[:, :, t-1, t:(t+1)] * gs[t]

            if T > 1:
                alphas[:, :, t-T:t, t:t+T] *= (
                    torch.matmul(final_v[:, :, t-T:t, :], gs[t:t+T].permute(1, 2, 3, 0)) - \
                    g_dot_atts[t:t+T].permute(1, 2, 0).unsqueeze(-2)
                ) # (B, n_heads, keys, queries)
            else:
                alphas[:, :, t-1, t] *= (final_v[:, :, t - 1, :] * gs[t]).sum(dim=-1) - g_dot_atts[t]

            # gradients with respect to k_i
            # k_grads[:t].add_(alpha_gvs * (q_t * scale))  # (t, B, n_heads, head_dim)
            # contribution of g[t:t+T] to k_grads[t-T:t]
            if T > 1:
                k_grads[t-T:t] += torch.matmul(
                    alphas[:, :, t-T:t, t:t+T],
                    q[t:t+T].permute(1, 2, 0, 3)
                ).permute(2, 0, 1, 3)
            else:
                k_grads[t-1] += alphas[:, :, t-1, t:(t+1)] * q[t]
        
        # release final_ks, final_vs, mlp_out_ts
        del final_ks, final_vs, mlp_out_ts, final_v, k_grads, v_grads, atts_with_grad

        # compute k_init_grads, v_init_grads and their influence on q_grads:
        v_init_grad = alphas.diagonal(dim1=2, dim2=3).unsqueeze(-1).permute(2, 0, 1, 3) * gs # (L, B, n_heads, head_dim)
        # gvinits = (v_init.permute(1,2,0,3).unsqueeze(-2) @ gs.permute(1, 2, 3, 0).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        gvinits = (v_init * gs).sum(dim=-1) # (L, B, n_heads)
        alpha_selves = alphas.diagonal(dim1=2, dim2=3).permute(2, 0, 1) * (gvinits - g_dot_atts) # (L, B, n_heads)
        k_init_grad = alpha_selves.unsqueeze(-1) * q # (L, B, n_heads, head_dim)
        q_grads = (alpha_selves.unsqueeze(-1)) * k_init # (L, B, n_heads, head_dim)
        
        del gs # no longer needed henceforth

        # compute q_grads
        alphas.diagonal(dim1=2, dim2=3).zero_()
        q_grads += torch.matmul(
            alphas.permute(0, 1, 3, 2),
            final_k.permute(1, 2, 0, 3)
        ).permute(2, 0, 1, 3)
        del alphas, final_k

        # propagate gradients from k_init, v_init, q to x/preAttentionBlock
        torch.autograd.backward((q, k_init, v_init), grad_tensors=(q_grads, k_init_grad, v_init_grad))
        del q_grads, k_init_grad, v_init_grad, q, k_init, v_init

        # recompute MLP to (parallelly) update its params' grads
        all_atts = atts
        big_grads = torch.cat(list(reversed(all_grads)), dim=1)
        del all_grads, atts
        assert all_atts.requires_grad == False
        assert x.requires_grad == True
        with torch.enable_grad():
            mlp_outs = mlp_batched_body(all_atts, x, block, ctx.dtype)
        torch.autograd.backward(mlp_outs, grad_tensors=big_grads)
        
        return x.grad, None, None


class OLMoRecurrentBlockTiled(OLMoRecurrentBlockBase):
    """Checkpointed tiled recurrent attention (custom autograd); same parameters as ``OLMoRecurrentBlockBase``."""

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        assert not self.config.rope, "RoPE is not supported for recurrent block"
        assert self.config.effective_n_kv_heads == self.config.n_heads, (
            "effective_n_kv_heads must be equal to n_heads for recurrent block"
        )

    def _real_forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return OLMoRecurrentBlockTiledFunction.apply(x, attention_bias, self)

class OLMoLlamaBlock(OLMoBlock):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). This block is similar to `OLMoSequentialBlock`
    but some operations have slightly different implementations to imitate the
    behavior of Llama.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache

        # Attention input projection. Projects x -> (q, k, v)
        if config.multi_query_attention:
            q_proj_out_dim = config.d_model
            k_proj_out_dim = config.d_model // config.n_heads
            v_proj_out_dim = config.d_model // config.n_heads
        else:
            q_proj_out_dim = config.d_model
            k_proj_out_dim = config.d_model
            v_proj_out_dim = config.d_model
        self.q_proj = nn.Linear(
            config.d_model, q_proj_out_dim, bias=config.include_bias, device=config.init_device
        )
        self.k_proj = nn.Linear(
            config.d_model, k_proj_out_dim, bias=config.include_bias, device=config.init_device
        )
        self.v_proj = nn.Linear(
            config.d_model, v_proj_out_dim, bias=config.include_bias, device=config.init_device
        )

        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.

        if self.config.init_fn == InitFnType.normal:
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor
        elif self.config.init_fn == InitFnType.mitchell:
            std = 1 / math.sqrt(self.config.d_model)
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        elif self.config.init_fn == InitFnType.full_megatron:
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        else:
            raise NotImplementedError(self.config.init_fn)

        init_normal(self.q_proj, std, cutoff_factor)
        init_normal(self.k_proj, std, cutoff_factor)
        init_normal(self.v_proj, std, cutoff_factor)
        init_normal(self.ff_proj, std, cutoff_factor)

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if max_doc_len is not None or cu_doc_lens is not None:
            raise NotImplementedError(
                f"attention document masking is not implemented for {self.__class__.__name__}"
            )

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        if is_causal:
            assert attn_mask is None

            query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None
            attn_bias = get_causal_attention_bias(self.__cache, key_len, q.device)[:, :, :query_len, :key_len]
        elif attn_mask is not None:
            attn_bias = attn_mask.to(q.dtype)
        else:
            attn_bias = torch.zeros_like(attn_weights)

        attn_weights += attn_bias
        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(q.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout_p)
        return torch.matmul(attn_weights, v)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        x_normed = self.attn_norm(x)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        if self.config.clip_qkv is not None:
            q.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            k.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            v.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        # Get attention scores.
        att, cache = self.attention(
            q,
            k,
            v,
            attention_bias,
            layer_past=layer_past,
            use_cache=use_cache,
            max_doc_len=max_doc_len,
            cu_doc_lens=cu_doc_lens,
        )

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        x = self.ff_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x, cache


class OLMoOutput(NamedTuple):
    logits: Optional[torch.FloatTensor]
    """
    A tensor of shape `(batch_size, seq_len, vocab_size)` representing the log probabilities
    for the next token *before* normalization via (log) softmax.
    """

    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    """
    Attention keys and values from each block.
    """

    hidden_states: Optional[Tuple[torch.Tensor, ...]]
    """
    Hidden states from each block.
    """

    pre_logits: Optional[torch.Tensor]
    """
    Hidden state after final layer norm and before the logit projection.
    """


class OLMoGenerateOutput(NamedTuple):
    token_ids: torch.LongTensor
    """
    The generated token IDs, a tensor of shape `(batch_size, beam_size, max_steps)`.
    These do *not* include the original input IDs.
    """

    scores: torch.FloatTensor
    """
    The scores of the generated sequences, a tensor of shape `(batch_size, beam_size)`.
    """


class OLMoBlockGroup(nn.ModuleList):
    def __init__(self, config: ModelConfig, layer_offset: int, modules: Optional[Iterable[nn.Module]] = None):
        super().__init__(modules)
        self.config = config
        self.layer_offset = layer_offset
        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn = activation_checkpoint_function(self.config)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layers_past: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        for block_idx, block in enumerate(self):
            layer_past = None if layers_past is None else layers_past[block_idx]
            block_idx += self.layer_offset
            if should_checkpoint_block(self.activation_checkpointing_strategy, block_idx):
                # shape: (batch_size, seq_len, d_model)
                x, cache = self._activation_checkpoint_fn(  # type: ignore
                    block,
                    x,
                    attention_bias=attention_bias,
                    layer_past=layer_past,
                    use_cache=use_cache,
                    max_doc_len=max_doc_len,
                    cu_doc_lens=cu_doc_lens,
                )
            else:
                # shape: (batch_size, seq_len, d_model)
                x, cache = block(
                    x,
                    attention_bias=attention_bias,
                    layer_past=layer_past,
                    use_cache=use_cache,
                    max_doc_len=max_doc_len,
                    cu_doc_lens=cu_doc_lens,
                )
            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)
        return x, attn_key_values

    def reset_parameters(self):
        for block in self:
            block.reset_parameters()

    def set_activation_checkpointing(
        self, strategy: Optional[ActivationCheckpointingStrategy], checkpoint_func: Optional[Callable] = None
    ):
        self.activation_checkpointing_strategy = strategy
        for block in self:
            block.set_activation_checkpointing(strategy, checkpoint_func=checkpoint_func)


class OLMo(nn.Module):
    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()

        # Validate config.
        if self.config.alibi and self.config.flash_attention:
            raise OLMoConfigurationError("ALiBi is currently not supported with FlashAttention")

        if self.config.alibi and self.config.rope:
            raise OLMoConfigurationError("ALiBi and RoPE are mutually exclusive")

        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise OLMoConfigurationError("embedding size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 != 0:
                import warnings

                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )

        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn: Callable = activation_checkpoint_function(self.config)

        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise OLMoConfigurationError("n layers must be divisible by block group size")

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # this is super slow so make sure torch won't use it

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.embedding_size or config.vocab_size, config.d_model, device=config.init_device
                ),
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=LayerNorm.build(config),
            )
        )

        blocks = [OLMoBlock.build(i, config, self.__cache) for i in range(config.n_layers)]
        if self.config.block_group_size > 1:
            block_groups = [
                OLMoBlockGroup(config, i, blocks[i : i + config.block_group_size])
                for i in range(0, config.n_layers, config.block_group_size)
            ]
            self.transformer.update({"block_groups": nn.ModuleList(block_groups)})
        else:
            self.transformer.update({"blocks": nn.ModuleList(blocks)})

        if not (self.config.alibi or self.config.rope):
            self.transformer.update(
                {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, device=config.init_device)}
            )
        if not config.weight_tying:
            self.transformer.update(
                {
                    "ff_out": nn.Linear(
                        config.d_model,
                        config.embedding_size or config.vocab_size,
                        bias=config.include_bias,
                        device=config.init_device,
                    )
                }
            )
        if config.embedding_layer_norm:
            self.transformer.update({"emb_norm": LayerNorm.build(config)})

        # When `init_device="meta"` FSDP will call `reset_parameters()` to initialize weights.
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()
        self.__num_fwd_flops: Optional[int] = None
        self.__num_bck_flops: Optional[int] = None

        # Warm up cache.
        if self.config.alibi:
            get_causal_attention_bias(self.__cache, config.max_sequence_length, _non_meta_init_device(config))
            self.get_alibi_attention_bias(config.max_sequence_length, _non_meta_init_device(config))

    def set_activation_checkpointing(
        self, strategy: Optional[ActivationCheckpointingStrategy], checkpoint_func: Optional[Callable] = None
    ):
        self.activation_checkpointing_strategy = strategy
        if self.config.block_group_size != 1:
            for block_group in self.transformer.block_groups:
                block_group.set_activation_checkpointing(strategy, checkpoint_func=checkpoint_func)
        else:
            for block in self.transformer.blocks:
                block.set_activation_checkpointing(strategy, checkpoint_func=checkpoint_func)

    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device  # type: ignore
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else:
            return device

    def reset_parameters(self):
        log.info("Initializing model parameters...")
        # Top-level embeddings / linear layers.

        if self.config.init_fn == InitFnType.normal:
            # Note: We may potentially want to multiply the std by a factor of sqrt(d) in case of `scale_logits`
            # and `weight_tying`. However, we are currently not using either, and may need to rethink the init logic
            # if/when we do want it.
            wte_std = self.config.emb_init_std or self.config.init_std
            wte_cutoff_factor = self.config.init_cutoff_factor
        elif self.config.init_fn == InitFnType.mitchell:
            wte_std = self.config.emb_init_std or 1.0 / math.sqrt(self.config.d_model)
            wte_cutoff_factor = self.config.init_cutoff_factor or 3.0
        elif self.config.init_fn == InitFnType.full_megatron:
            wte_std = self.config.init_std
            if self.config.emb_init_std is not None:
                wte_std = self.config.emb_init_std
            elif self.config.scale_emb_init:
                wte_std *= math.sqrt(self.config.d_model)
            wte_cutoff_factor = self.config.init_cutoff_factor or 3.0
        else:
            raise NotImplementedError(self.config.init_fn)

        init_normal(self.transformer.wte, std=wte_std, init_cutoff_factor=wte_cutoff_factor)

        if hasattr(self.transformer, "wpe"):
            if self.config.init_fn == InitFnType.normal:
                wpe_std = self.config.init_std
                wpe_cutoff_factor = self.config.init_cutoff_factor
            elif self.config.init_fn == InitFnType.mitchell:
                wpe_std = 1 / math.sqrt(self.config.d_model)
                wpe_cutoff_factor = self.config.init_cutoff_factor or 3.0
            elif self.config.init_fn == InitFnType.full_megatron:
                wpe_std = self.config.init_std
                wpe_cutoff_factor = self.config.init_cutoff_factor or 3.0
            else:
                raise NotImplementedError(self.config.init_fn)

            init_normal(self.transformer.wpe, std=wpe_std, init_cutoff_factor=wpe_cutoff_factor)

        # Top-level layer norm.
        self.transformer.ln_f.reset_parameters()  # type: ignore

        # Output weights.
        if hasattr(self.transformer, "ff_out"):
            if self.config.init_fn == InitFnType.normal:
                ff_out_std = self.config.init_std
                ff_out_cutoff_factor = self.config.init_cutoff_factor
            elif self.config.init_fn == InitFnType.mitchell:
                ff_out_std = 1 / math.sqrt(self.config.d_model)
                ff_out_cutoff_factor = self.config.init_cutoff_factor or 3.0
            elif self.config.init_fn == InitFnType.full_megatron:
                ff_out_std = 1 / math.sqrt(self.config.d_model)
                ff_out_cutoff_factor = self.config.init_cutoff_factor or 3.0
            else:
                raise NotImplementedError(self.config.init_fn)

            init_normal(self.transformer.ff_out, ff_out_std, ff_out_cutoff_factor)

        # Let the blocks handle themselves.
        if self.config.block_group_size == 1:
            for block in self.transformer.blocks:
                block.reset_parameters()
        else:
            for block_group in self.transformer.block_groups:
                block_group.reset_parameters()

    def get_alibi_attention_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if (alibi_bias := self.__cache.get("alibi_attention_bias")) is not None and alibi_bias.shape[
            -1
        ] >= seq_len:
            if alibi_bias.device != device:
                alibi_bias = alibi_bias.to(device)
                self.__cache["alibi_attention_bias"] = alibi_bias
            return alibi_bias
        with torch.autocast(device.type, enabled=False):
            alibi_bias = alibi_attention_bias(seq_len, self.config, device)
        self.__cache["alibi_attention_bias"] = alibi_bias
        return alibi_bias

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        return_pre_logits: bool = False,
        return_logits: bool = True,
        doc_lens: Optional[torch.Tensor] = None,
        max_doc_lens: Optional[Sequence[int]] = None,
    ) -> OLMoOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param input_embeddings: A tensor of shape `(batch_size, seq_len, d_model)` with input
            embeddings. When provided, it is treated as the output of the input embedding layer.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked.

            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        :param attention_bias: A tensor of shape `(batch_size, 1, seq_len, seq_len)`,
            `(1, 1, seq_len, seq_len)`, or `(seq_len, seq_len)`. This is used
            to introduce causal or other biases.

            If the tensor is a bool or byte tensor, a `True` or `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.

            If the tensor is a float tensor, it will just be added to the attention
            scores before the softmax.

            The default is causal, which corresponds to a lower-diagonal byte matrix of ones.
        :param past_key_values: Pre-computed keys and values for each attention block.
            Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        :param use_cache: If `True`, return key and value tensors for each block.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        :param doc_lens: Document lengths to use in attention for intra-document masking.
            Shape `(batch_size, max_docs)`.
        :param max_doc_lens: Maximum document length for each instance in the batch.
        """
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        max_doc_len: Optional[int] = None
        cu_doc_lens: Optional[torch.Tensor] = None
        if doc_lens is not None and max_doc_lens is not None:
            max_doc_len = max(max_doc_lens)
            cu_doc_lens = get_cumulative_document_lengths(doc_lens)

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings  # type: ignore

        # Apply embedding layer norm.
        if self.config.embedding_layer_norm:
            x = self.transformer.emb_norm(x)

        if not (self.config.alibi or self.config.rope):
            # Get positional embeddings.
            # shape: (1, seq_len)
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            # shape: (1, seq_len, d_model)
            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = pos_emb + x

        # Apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # Transform the attention mask into what the blocks expect.
        if attention_mask is not None:
            # shape: (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min

        # Merge attention mask with attention bias.
        if (
            attention_bias is not None
            or attention_mask is not None
            or self.config.alibi
            # NOTE (epwalsh): we need to initialize the attn bias in order for attn to work properly
            # with key+value cache. Otherwise `F.scaled_dot_product_attention()` doesn't seem to compute
            # scores correctly.
            or past_key_values is not None
        ):
            if attention_bias is None and self.config.alibi:
                attention_bias = get_causal_attention_bias(
                    self.__cache, past_length + seq_len, x.device
                ) + self.get_alibi_attention_bias(past_length + seq_len, x.device)
            elif attention_bias is None:
                attention_bias = get_causal_attention_bias(self.__cache, past_length + seq_len, x.device)
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)

            # Transform to the right shape and data type.
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)

            # Add in the masking bias.
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                # Might get -infs after adding attention mask, since dtype.min + dtype.min = -inf.
                # `F.scaled_dot_product_attention()` doesn't handle -inf like you'd expect, instead
                # it can produce NaNs.
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

        # decoder layers
        all_hidden_states = []

        # Apply blocks one-by-one.
        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layer_past = None if past_key_values is None else past_key_values[block_idx]
                if should_checkpoint_block(self.activation_checkpointing_strategy, block_idx):
                    # shape: (batch_size, seq_len, d_model)
                    x, cache = self._activation_checkpoint_fn(
                        block,
                        x,
                        attention_bias=attention_bias,
                        layer_past=layer_past,
                        use_cache=use_cache,
                        max_doc_len=max_doc_len,
                        cu_doc_lens=cu_doc_lens,
                    )
                else:
                    # shape: (batch_size, seq_len, d_model)
                    x, cache = block(
                        x,
                        attention_bias=attention_bias,
                        layer_past=layer_past,
                        use_cache=use_cache,
                        max_doc_len=max_doc_len,
                        cu_doc_lens=cu_doc_lens,
                    )

                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.append(cache)
        else:
            for group_idx, block_group in enumerate(self.transformer.block_groups):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layers_past = (
                    None
                    if past_key_values is None
                    else past_key_values[
                        group_idx * self.config.block_group_size : (group_idx + 1) * self.config.block_group_size
                    ]
                )
                x, cache = block_group(
                    x,
                    attention_bias=attention_bias,
                    layers_past=layers_past,
                    use_cache=use_cache,
                    max_doc_len=max_doc_len,
                    cu_doc_lens=cu_doc_lens,
                )
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.extend(cache)

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            x = x[:, -1, :].unsqueeze(1)

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore
        if output_hidden_states:
            # add final hidden state post-final-layernorm, following HuggingFace's convention
            all_hidden_states.append(x)

        pre_logits = x
        logits: Optional[torch.Tensor] = None
        if return_logits:
            # Get logits.
            # shape: (batch_size, seq_len or 1, vocab_size)
            if self.config.weight_tying:
                logits = F.linear(pre_logits, self.transformer.wte.weight, None)  # type: ignore
            else:
                logits = self.transformer.ff_out(pre_logits)  # type: ignore
            if self.config.scale_logits:
                logits.mul_(1 / math.sqrt(self.config.d_model))

        return OLMoOutput(
            logits=logits,
            attn_key_values=attn_key_values,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            pre_logits=pre_logits if return_pre_logits else None,
        )

    def get_fsdp_wrap_policy(self, wrap_strategy: Optional[FSDPWrapStrategy] = None):
        if wrap_strategy is None:
            return None

        # The 'recurse' mode for the wrap function does not behave like you'd expect.
        # Even if we return False, it may still recurse because PyTorch does what it wants,
        # not what you want. This causes issues when, for example, we want to wrap 'ff_out' (a linear layer)
        # but not other linear layers within a block.
        # So we have to explicitly tell PyTorch which linear layers to wrap, and we also just
        # return True in 'recurse' mode for simplicity.
        size_based_module_to_wrap = {self.transformer.wte}
        if hasattr(self.transformer, "ff_out"):
            size_based_module_to_wrap.add(self.transformer.ff_out)

        if wrap_strategy == FSDPWrapStrategy.by_block:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, OLMoBlock)
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_and_size:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, (OLMoBlock,)) or module in size_based_module_to_wrap
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_group:
            if self.config.block_group_size <= 1:
                raise OLMoConfigurationError(
                    "'by_block_group' FSDP wrapping strategy requires block group size greater than 1"
                )

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, OLMoBlockGroup)
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_group_and_size:
            if self.config.block_group_size <= 1:
                raise OLMoConfigurationError(
                    "'by_block_group_and_size' FSDP wrapping strategy requires block group size greater than 1"
                )

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, (OLMoBlockGroup,)) or module in size_based_module_to_wrap
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.size_based:
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

            return size_based_auto_wrap_policy
        elif wrap_strategy in {
            FSDPWrapStrategy.one_in_two,
            FSDPWrapStrategy.one_in_three,
            FSDPWrapStrategy.one_in_four,
            FSDPWrapStrategy.one_in_five,
        }:
            c = {
                FSDPWrapStrategy.one_in_two: 2,
                FSDPWrapStrategy.one_in_three: 3,
                FSDPWrapStrategy.one_in_four: 4,
                FSDPWrapStrategy.one_in_five: 5,
            }[wrap_strategy]

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, OLMoBlock) and module.layer_id % c == 0
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        else:
            raise NotImplementedError(wrap_strategy)

    def num_params(self, include_embedding: bool = True) -> int:
        """
        Get the total number of parameters.
        """
        params = (np for np in self.named_parameters())
        if not include_embedding:
            params = filter(  # type: ignore
                lambda np: ".wte." not in np[0] and ".wpe." not in np[0],
                params,
            )
        return sum(p.numel() for _, p in params)

    @property
    def num_fwd_flops(self):
        if self.__num_fwd_flops:
            return self.__num_fwd_flops

        # embedding table is just a lookup in the forward pass
        n_params = self.num_params(include_embedding=False)
        # the number of parameters is approximately the number of multiply-accumulates (MAC) in the network
        # each MAC has 2 FLOPs - we multiply by 2 ie 2 * n_param
        # this gets us FLOPs / token
        params_flops_per_token = 2 * n_params
        # there are 2 FLOPS per mac; there is A=Q*K^T and out=A*V ops (ie mult by 2)
        attn_flops_per_token = (
            self.config.n_layers * 2 * 2 * (self.config.d_model * self.config.max_sequence_length)
        )
        self.__num_fwd_flops = params_flops_per_token + attn_flops_per_token
        return self.__num_fwd_flops

    @property
    def num_bck_flops(self):
        if self.__num_bck_flops:
            return self.__num_bck_flops

        n_params = self.num_params()
        params_flops_per_token = 4 * n_params
        attn_flops_per_token = self.config.n_layers * 8 * (self.config.d_model * self.config.max_sequence_length)
        self.__num_bck_flops = params_flops_per_token + attn_flops_per_token
        return self.__num_bck_flops

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        max_steps: int = 10,
        beam_size: int = 1,
        per_node_beam_size: Optional[int] = None,
        sampler: Optional[Sampler] = None,
        min_steps: Optional[int] = None,
        final_sequence_scorer: Optional[FinalSequenceScorer] = None,
        constraints: Optional[List[Constraint]] = None,
    ) -> OLMoGenerateOutput:
        """
        Generate token IDs using beam search.

        Note that by default ``beam_size`` is set to 1, which is greedy decoding.

        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param attention_mask: A optional tensor of shape `(batch_size, seq_len)`, the same
            as for the forward method.
        :param attention_bias: A tensor of shape
            `(batch_size, 1, seq_len + tokens_to_generate, seq_len + tokens_to_generate)`,
            the same as for the forward method except only one shape is excepted here.

        For an explanation of the other arguments, see :class:`BeamSearch`.
        """
        beam_search = BeamSearch(
            self.config.eos_token_id,
            max_steps=max_steps,
            beam_size=beam_size,
            per_node_beam_size=per_node_beam_size,
            sampler=sampler,
            min_steps=min_steps,
            final_sequence_scorer=final_sequence_scorer,
            constraints=constraints,
        )

        # Validate inputs.
        batch_size, seq_len = input_ids.shape
        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, seq_len)
        if attention_bias is not None:
            assert len(attention_bias.shape) == 4
            assert attention_bias.shape[:2] == (batch_size, 1)
            assert (
                seq_len + beam_search.max_steps
                <= attention_bias.shape[2]
                == attention_bias.shape[3]
                <= self.config.max_sequence_length
            )

        tokens_generated = 0

        def flatten_past_key_values(
            past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        ) -> Dict[str, torch.Tensor]:
            out = {}
            for i, (key, value) in enumerate(past_key_values):
                out[f"past_key_{i}"] = key
                out[f"past_value_{i}"] = value
            return out

        def unflatten_past_key_values(
            past_key_values: Dict[str, torch.Tensor],
        ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
            out = []
            for i in range(self.config.n_layers):
                past_key = past_key_values[f"past_key_{i}"]
                past_value = past_key_values[f"past_value_{i}"]
                out.append((past_key, past_value))
            return out

        def step(
            last_predictions: torch.Tensor, state: dict[str, torch.Tensor]
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            nonlocal tokens_generated

            attention_mask = state.get("attention_mask")
            attention_bias = state.get("attention_bias")

            if tokens_generated > 0:
                past_key_values = unflatten_past_key_values(state)
                input_ids = last_predictions.unsqueeze(1)
                if attention_mask is not None:
                    group_size = input_ids.shape[0]
                    attention_mask = torch.cat((attention_mask, attention_mask.new_ones((group_size, 1))), dim=-1)
            else:
                past_key_values = None
                input_ids = state["input_ids"]

            tokens_generated += 1

            # Run forward pass of model to get logits, then normalize to get log probs.
            output = self(
                input_ids,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                past_key_values=past_key_values,
                use_cache=True,
                last_logits_only=True,
            )
            log_probs = F.log_softmax(output.logits[:, -1, :], dim=-1)

            # Create new state.
            state = flatten_past_key_values(output.attn_key_values)
            if attention_mask is not None:
                state["attention_mask"] = attention_mask
            if attention_bias is not None:
                state["attention_bias"] = attention_bias

            return log_probs, state

        initial_preds = input_ids.new_zeros((batch_size,))  # This is arbitrary, we won't use this.
        state: dict[str, torch.Tensor] = {"input_ids": input_ids}
        if attention_mask is not None:
            state["attention_mask"] = attention_mask
        if attention_bias is not None:
            state["attention_bias"] = attention_bias
        with torch.no_grad():
            token_ids, scores = beam_search.search(initial_preds, state, step)

        return OLMoGenerateOutput(
            token_ids=token_ids,  # type: ignore[arg-type]
            scores=scores,  # type: ignore[arg-type]
        )

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: PathOrStr, device: str = "cpu", checkpoint_type: Optional[CheckpointType] = None
    ) -> OLMo:
        """
        Load an OLMo model from a checkpoint.
        """
        from .util import resource_path

        # Guess checkpoint type.
        if checkpoint_type is None:
            try:
                if resource_path(checkpoint_dir, "model.pt").is_file():
                    checkpoint_type = CheckpointType.unsharded
                else:
                    checkpoint_type = CheckpointType.sharded
            except FileNotFoundError:
                checkpoint_type = CheckpointType.sharded

        # Load config.
        config_path = resource_path(checkpoint_dir, "config.yaml")
        model_config = ModelConfig.load(config_path, key="model", validate_paths=False)

        if checkpoint_type == CheckpointType.unsharded:
            # Initialize model (always on CPU to start with so we don't run out of GPU memory).
            model_config.init_device = "cpu"
            model = OLMo(model_config)

            # Load state dict directly to target device.
            state_dict_path = resource_path(checkpoint_dir, "model.pt")
            state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(model._make_state_dict_compatible(state_dict)[0])
            model = model.to(torch.device(device))
        else:
            train_config = TrainConfig.load(config_path)
            if train_config.sharded_checkpointer == ShardedCheckpointerType.olmo_core:
                from olmo_core.distributed.checkpoint import (  # type: ignore
                    load_model_and_optim_state,
                )

                model_config.init_device = device
                model = OLMo(model_config)
                load_model_and_optim_state(checkpoint_dir, model)
            else:
                # train_config.sharded_checkpointer == ShardedCheckpointerType.torch_new
                from .checkpoint import load_model_state

                # Initialize model on target device. In this case the state dict is loaded in-place
                # so it's not necessary to start on CPU if the target device is a GPU.
                model_config.init_device = device
                model = OLMo(model_config)

                # Load state dict in place.
                load_model_state(checkpoint_dir, model)

        return model.eval()

    def _make_state_dict_compatible(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Set[str]]]:
        """
        Handles some cases where the state dict is valid yet may need to be transformed in order to
        be loaded.

        This modifies the state dict in-place and also returns it, along with a mapping of original key
        names to new key names in cases where the keys were simply renamed. That mapping can be used
        to make a corresponding optimizer state dict compatible as well.
        """
        import re
        from fnmatch import fnmatch

        new_keys_to_og_keys: Dict[str, str] = {}

        # Remove "_fsdp_wrapped_module." prefix from all keys. We don't want this prefix when the model is
        # not wrapped in FSDP. And when the model is wrapped in FSDP, loading this state dict will still work
        # fine without the prefixes. This also simplifies the other steps below.
        for key in list(state_dict.keys()):
            state_dict[(new_key := key.replace("_fsdp_wrapped_module.", ""))] = state_dict.pop(key)
            new_keys_to_og_keys[new_key] = key

        # For backwards compatibility prior to fixing https://github.com/allenai/LLM/issues/222
        if self.config.block_type == BlockType.sequential:
            for key in list(state_dict.keys()):
                if fnmatch(key, "transformer.*.norm.weight"):
                    tensor = state_dict.pop(key)
                    state_dict[(new_key := key.replace("norm.weight", "attn_norm.weight"))] = tensor
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    state_dict[(new_key := key.replace("norm.weight", "ff_norm.weight"))] = tensor.clone()
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    del new_keys_to_og_keys[key]
                elif fnmatch(key, "transformer.*.norm.bias"):
                    tensor = state_dict.pop(key)
                    state_dict[(new_key := key.replace("norm.bias", "attn_norm.bias"))] = tensor
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    state_dict[(new_key := key.replace("norm.bias", "ff_norm.bias"))] = tensor.clone()
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    del new_keys_to_og_keys[key]

        # For loading a state dict that was saved with a different `block_group_size`.
        if "transformer.block_groups.0.0.attn_out.weight" in state_dict.keys():
            state_dict_block_group_size = len(
                [k for k in state_dict.keys() if fnmatch(k, "transformer.block_groups.0.*.attn_out.weight")]
            )
        else:
            state_dict_block_group_size = 1
        if self.config.block_group_size != state_dict_block_group_size:
            log.info(
                f"Regrouping state dict blocks from group size {state_dict_block_group_size} to "
                f"group size {self.config.block_group_size}"
            )
            # For simplicity we're first going to flatten out the block groups in the state dict (if necessary)
            # and then (re-)group them into the right block sizes.
            if state_dict_block_group_size > 1:
                for key in list(state_dict.keys()):
                    if (m := re.match(r"transformer.block_groups\.(\d+)\.(\d+)\..*", key)) is not None:
                        group_idx, group_block_idx = int(m.group(1)), int(m.group(2))
                        block_idx = (group_idx * state_dict_block_group_size) + group_block_idx
                        state_dict[
                            (
                                new_key := key.replace(
                                    f"block_groups.{group_idx}.{group_block_idx}.", f"blocks.{block_idx}."
                                )
                            )
                        ] = state_dict.pop(key)
                        new_keys_to_og_keys[new_key] = new_keys_to_og_keys.pop(key)

            if self.config.block_group_size > 1:
                # Group the state dict blocks into the right block size.
                for key in list(state_dict.keys()):
                    if (m := re.match(r"transformer.blocks\.(\d+)\..*", key)) is not None:
                        block_idx = int(m.group(1))
                        group_idx, group_block_idx = (
                            block_idx // self.config.block_group_size,
                            block_idx % self.config.block_group_size,
                        )
                        state_dict[
                            (
                                new_key := key.replace(
                                    f"blocks.{block_idx}.", f"block_groups.{group_idx}.{group_block_idx}."
                                )
                            )
                        ] = state_dict.pop(key)
                        new_keys_to_og_keys[new_key] = new_keys_to_og_keys.pop(key)

        og_keys_to_new: Dict[str, Set[str]] = defaultdict(set)
        for new_key, og_key in new_keys_to_og_keys.items():
            og_keys_to_new[og_key].add(new_key)

        return state_dict, og_keys_to_new
