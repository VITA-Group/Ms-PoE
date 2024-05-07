from typing import Optional, Tuple

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    rotate_half,
    repeat_kv,
)
from transformers.utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

__all__ = ['MsPoELlamaForCausalLM', "MsPoEMistralForCausalLM", "MsPoEQwen2ForCausalLM", "MsPoEGemmaForCausalLM"]


def _make_causal_mask(
        bsz: int, tgt_len: int, past_key_values_length: int, dtype: torch.dtype, device: torch.device):
    """
    Make causal mask used for bi-directional self-attention.
    """
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def apply_rotary_pos_emb_single_scaling(x, cos, sin, position_ids):
    cos = cos[:, position_ids]  # [head, bs, seq_len, dim]
    sin = sin[:, position_ids]  # [head, bs, seq_len, dim]

    cos = cos.transpose(0, 1)  # [bs, head, seq_len, dim]
    sin = sin.transpose(0, 1)  # [bs, head, seq_len, dim]

    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def sample_rotary_emb(cos, sin, num_key_value_groups):
    cos = cos[::num_key_value_groups, ...]  # [head, bs, seq_len, dim]
    sin = sin[::num_key_value_groups, ...]  # [head, bs, seq_len, dim]
    return cos, sin


### Positional Scaling
class MsPoELlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, min_cratio=1, max_cratio=3, num_heads=32, max_position_embeddings=2048, base=10000,
                 device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.min_ratio = min_cratio
        self.max_ratio = max_cratio
        self.num_heads = num_heads

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        min_ratio = self.min_ratio
        max_ratio = self.max_ratio
        num_heads = self.num_heads
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype).repeat(num_heads, 1)
        compress_ratio = torch.arange(num_heads, device=device, dtype=self.inv_freq.dtype)
        compress_ratio = min_ratio + (max_ratio - min_ratio) * (compress_ratio / num_heads)
        compress_ratio = compress_ratio.unsqueeze(-1)

        t = t / compress_ratio
        freqs = torch.einsum("ki,j->kij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


### Positional Scaling
class MsPoELlamaLinearScalingRotaryEmbedding(nn.Module):
    def __init__(self, dim, min_cratio=1, max_cratio=3, num_heads=32, max_position_embeddings=2048, base=10000,
                 device=None, scaling_factor=1.0):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.min_ratio = min_cratio
        self.max_ratio = max_cratio
        self.num_heads = num_heads
        self.scaling_factor = scaling_factor
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        min_ratio = self.min_ratio
        max_ratio = self.max_ratio
        num_heads = self.num_heads
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype).repeat(num_heads, 1)
        compress_ratio = torch.arange(num_heads, device=device, dtype=self.inv_freq.dtype)
        compress_ratio = min_ratio + (max_ratio - min_ratio) * (compress_ratio / num_heads)
        compress_ratio = compress_ratio.unsqueeze(-1)
        t = t / self.scaling_factor
        t = t / compress_ratio
        freqs = torch.einsum("ki,j->kij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class MsPoELlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx):
        super().__init__()
        # 如果没有rope_scaling，则设为None
        if not hasattr(config, "rope_scaling"):
            config.rope_scaling = None

        self.layer_idx = layer_idx
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, "rope_theta", None)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.compress_ratio_min = config.compress_ratio_min
        self.compress_ratio_max = config.compress_ratio_max

        self.enable_head_metrics = True
        self.head_type = config.head_type
        self.head_order = None

        self._init_rope()

    def _head_wise_statistics(self, query_states, key_states, q_len, kv_seq_len, bsz, attention_mask):

        # qk均截断至4k长度以下，防止OOM
        if q_len > 4096:
            q_len = 4096
            kv_seq_len = 4096
            query_states = query_states[:, :, :4096]
            key_states = key_states[:, :, :4096]

        query_states_new = query_states
        key_states_new = repeat_kv(key_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states_new, key_states_new.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)  # , dtype=torch.float32).to(query_states.dtype)

        if len(attn_weights.shape) == 4:
            attn_weights = attn_weights.squeeze(0)

        head_orders = self._calculate_outlier(attn_weights)

        return head_orders

    def _calculate_outlier(self, attn_weights):
        # attn_weights: [num_heads, q_len, kv_seq_len]
        average = attn_weights.mean(-1).unsqueeze(-1)
        outlier = - (attn_weights > 3 * average).to(attn_weights.dtype).mean(-1)[:, -1]
        head_orders = outlier.argsort()

        if self.head_type == "normal":
            head_orders = np.arange(self.num_heads)
            head_orders = self.num_heads - head_orders - 1

        return head_orders

    def _init_rope(self):
        if not hasattr(self.config, "rope_scaling"):
            self.config.rope_scaling = None
        if self.config.rope_scaling is None:
            self.rotary_emb = MsPoELlamaRotaryEmbedding(
                self.head_dim,
                min_cratio=self.compress_ratio_min,
                max_cratio=self.compress_ratio_max,
                num_heads=self.num_heads,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = MsPoELlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    min_cratio=self.compress_ratio_min,
                    max_cratio=self.compress_ratio_max,
                    num_heads=self.num_heads,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                assert False  # not implemented
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        hidden_state_set_layers = getattr(self.config, "hidden_state_set_layers", {})
        hidden_state_set_values = getattr(self.config, "hidden_state_set_values", 0)
        hidden_states_for_qk = hidden_states.clone()  # 复制hidden_states
        if str(self.layer_idx) in hidden_state_set_layers:
            # 将指定维度设为hidden_state_set_values
            hidden_states_for_qk[:, :, hidden_state_set_layers[str(self.layer_idx)]] = hidden_state_set_values

        query_states = self.q_proj(hidden_states_for_qk)
        key_states = self.k_proj(hidden_states_for_qk)
        value_states = self.v_proj(hidden_states_for_qk)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # # remake causal mask
        past_key_values_length = past_key_value[0][0].shape[-2] - 1 if past_key_value is not None and q_len == 1 else 0
        attention_mask = _make_causal_mask(
            bsz=bsz,
            tgt_len=q_len,
            past_key_values_length=past_key_values_length,
            dtype=query_states.dtype,
            device=query_states.device,
        )

        kv_seq_len = key_states.shape[-2]
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        position_length = kv_seq_len
        if not position_ids.nelement() > 1:
            if position_length < position_ids.item() + 1:
                position_length = position_ids.item() + 1

        cos, sin = self.rotary_emb(value_states, seq_len=position_length)

        if self.enable_head_metrics:
            self.head_order = self._head_wise_statistics(query_states, key_states, q_len, kv_seq_len, bsz,
                                                         attention_mask)
            self.enable_head_metrics = False

        cos = cos[self.head_order, :, :]
        sin = sin[self.head_order, :, :]
        query_states = apply_rotary_pos_emb_single_scaling(query_states, cos, sin, position_ids)

        cos, sin = sample_rotary_emb(cos, sin, self.num_key_value_groups)
        key_states = apply_rotary_pos_emb_single_scaling(key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            # # reuse k, v, self_attention
            # key_states = torch.cat([past_key_value[0], key_states], dim=2)
            # value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # # key/value are already rotated
        # past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MsPoELlamaFlashAttention(MsPoELlamaAttention):
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        hidden_state_set_layers = getattr(self.config, "hidden_state_set_layers", {})
        hidden_state_set_values = getattr(self.config, "hidden_state_set_values", 0)
        hidden_states_for_qk = hidden_states.clone()  # 复制hidden_states
        if str(self.layer_idx) in hidden_state_set_layers:
            # 将指定维度设为hidden_state_set_values
            hidden_states_for_qk[:, :, hidden_state_set_layers[str(self.layer_idx)]] = hidden_state_set_values

        query_states = self.q_proj(hidden_states_for_qk)
        key_states = self.k_proj(hidden_states_for_qk)
        value_states = self.v_proj(hidden_states_for_qk)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # # remake causal mask
        past_key_values_length = past_key_value[0][0].shape[-2] - 1 if past_key_value is not None and q_len == 1 else 0
        # attention_mask = _make_causal_mask(
        #     bsz=bsz,
        #     tgt_len=q_len,
        #     past_key_values_length=past_key_values_length,
        #     dtype=query_states.dtype,
        #     device=query_states.device,
        # )

        kv_seq_len = key_states.shape[-2]
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        position_length = kv_seq_len
        if not position_ids.nelement() > 1:
            if position_length < position_ids.item() + 1:
                position_length = position_ids.item() + 1

        cos, sin = self.rotary_emb(value_states, seq_len=position_length)

        if self.enable_head_metrics:
            self.head_order = self._head_wise_statistics(query_states, key_states, q_len, kv_seq_len, bsz,
                                                         attention_mask)
            self.enable_head_metrics = False

        cos = cos[self.head_order, :, :]
        sin = sin[self.head_order, :, :]
        query_states = apply_rotary_pos_emb_single_scaling(query_states, cos, sin, position_ids)

        cos, sin = sample_rotary_emb(cos, sin, self.num_key_value_groups)
        key_states = apply_rotary_pos_emb_single_scaling(key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            # # reuse k, v, self_attention
            # key_states = torch.cat([past_key_value[0], key_states], dim=2)
            # value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # # key/value are already rotated
        # past_key_value = (key_states, value_states) if use_cache else None

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = 0.0
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask=None, query_length=q_len, dropout=dropout_rate
        )

        # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
            self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        self._flash_attn_uses_top_left_mask = False
        self.is_causal = True

        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class MsPoEMistralAttention(MsPoELlamaAttention):
    pass


class MsPoEMistralFlashAttention(MsPoELlamaFlashAttention):
    pass


class MsPoEGemmaAttention(MsPoELlamaAttention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.compress_ratio_min = config.compress_ratio_min
        self.compress_ratio_max = config.compress_ratio_max

        self.enable_head_metrics = True
        self.head_type = config.head_type
        self.head_order = None

        self._init_rope()


class MsPoEGemmaFlashAttention(MsPoELlamaFlashAttention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.compress_ratio_min = config.compress_ratio_min
        self.compress_ratio_max = config.compress_ratio_max

        self.enable_head_metrics = True
        self.head_type = config.head_type
        self.head_order = None

        self._init_rope()


class MsPoEQwen2Attention(MsPoELlamaAttention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()


class MsPoEQwen2FlashAttention(MsPoELlamaFlashAttention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()


from transformers import GemmaForCausalLM, Qwen2ForCausalLM, \
    MistralForCausalLM, LlamaForCausalLM


class MsPoELlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            if layer_idx in config.apply_layers:
                if self.config._attn_implementation == "eager" or self.config._attn_implementation == "sdap":
                    self.model.layers[layer_idx].self_attn = MsPoELlamaAttention(config, layer_idx)
                elif self.config._attn_implementation == "flash_attention_2":
                    self.model.layers[layer_idx].self_attn = MsPoELlamaFlashAttention(config, layer_idx)
                else:
                    raise ValueError(f"Unknown attention implementation {self.config._attn_implementation}")

    def _reset(self):
        for layer_idx in self.config.apply_layers:
            self.model.layers[layer_idx].self_attn.enable_head_metrics = True
            self.model.layers[layer_idx].self_attn.head_order = None


class MsPoEMistralForCausalLM(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            if layer_idx in config.apply_layers:
                if self.config._attn_implementation == "eager" or self.config._attn_implementation == "sdap":
                    self.model.layers[layer_idx].self_attn = MsPoEMistralAttention(config, layer_idx)
                elif self.config._attn_implementation == "flash_attention_2":
                    self.model.layers[layer_idx].self_attn = MsPoEMistralFlashAttention(config,
                                                                                        layer_idx)  # MsPoELlamaAttention(config,layer_idx)
                else:
                    raise ValueError(f"Unknown attention implementation {self.config._attn_implementation}")

    def _reset(self):
        for layer_idx in self.config.apply_layers:
            self.model.layers[layer_idx].self_attn.enable_head_metrics = True
            self.model.layers[layer_idx].self_attn.head_order = None


class MsPoEGemmaForCausalLM(GemmaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            if layer_idx in config.apply_layers:
                if self.config._attn_implementation == "eager" or self.config._attn_implementation == "sdap":
                    self.model.layers[layer_idx].self_attn = MsPoEGemmaAttention(config, layer_idx)
                elif self.config._attn_implementation == "flash_attention_2":
                    self.model.layers[layer_idx].self_attn = MsPoEGemmaFlashAttention(config,
                                                                                      layer_idx)  # MsPoELlamaAttention(config,layer_idx)
                else:
                    raise ValueError(f"Unknown attention implementation {self.config._attn_implementation}")

    def _reset(self):
        for layer_idx in self.config.apply_layers:
            self.model.layers[layer_idx].self_attn.enable_head_metrics = True
            self.model.layers[layer_idx].self_attn.head_order = None


class MsPoEQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            if layer_idx in config.apply_layers:
                if self.config._attn_implementation == "eager" or self.config._attn_implementation == "sdap":
                    self.model.layers[layer_idx].self_attn = MsPoEQwen2FlashAttention(config,
                                                                                      layer_idx)  # MsPoELlamaAttention(config,layer_idx)
                elif self.config._attn_implementation == "flash_attention_2":
                    self.model.layers[layer_idx].self_attn = MsPoEQwen2FlashAttention(config, layer_idx)
                else:
                    raise ValueError(f"Unknown attention implementation {self.config._attn_implementation}")

    def _reset(self):
        for layer_idx in self.config.apply_layers:
            self.model.layers[layer_idx].self_attn.enable_head_metrics = True
            self.model.layers[layer_idx].self_attn.head_order = None
