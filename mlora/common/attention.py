from mlora.utils import is_package_available
from typing import Tuple, Optional

import torch.nn.functional as F
import torch
import math

_xformers_available = is_package_available("xformers")
_flash_attn_available = is_package_available("flash_attn")


def precompute_rope_angle(dim: int, seq_len: int,
                          theta: float = 10000.0,
                          device: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2,
                      dtype=torch.int64).to(device=device, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, dtype=torch.int64).to(
        device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    emb.requires_grad_(False)

    # cos(angle), sin(angle)
    return (emb.cos(), emb.sin())


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (xq * cos) + (rotate_half(xq) * sin)
    k_embed = (xk * cos) + (rotate_half(xk) * sin)
    return q_embed, k_embed


def get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(
        seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


@torch.jit.script
def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                                 attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    attention_score = torch.matmul(
        query, key.transpose(2, 3)) / math.sqrt(query.size(-1))
    if attention_mask is not None:
        attention_score = attention_score + attention_mask
    attention_score = F.softmax(
        attention_score, dim=-1, dtype=torch.float32).to(value.dtype)
    attention_score = torch.matmul(attention_score, value)
    attention_score = attention_score.transpose(1, 2).contiguous()
    return attention_score
