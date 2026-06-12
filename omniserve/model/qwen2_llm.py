"""From-scratch Qwen2 language-model forward (the text decoder of Qwen2-VL).

This replaces transformers' `Qwen2VLForConditionalGeneration` language stack so
the engine controls the forward — the prerequisite for varlen/packed prefill,
mixed batching, and fused kernels that HF's forward can't express. The vision
encoder (ViT) is left to HF for now; its embeddings are spliced into the text
embeddings before this forward runs.

Architecture (Qwen2-VL-2B): hidden 1536, 28 layers, 12 query heads / 2 KV heads
(GQA), head_dim 128, SwiGLU intermediate 8960, RMSNorm eps 1e-6, rope_theta 1e6,
M-RoPE sections [16,24,24], tied embeddings, q/k/v have bias, o does not.

Correctness is the bar: token-identical to HF (verified in
scripts/proto_custom_llm.py). Optimizations come after.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Qwen2Config:
    hidden_size: int = 1536
    num_layers: int = 28
    num_heads: int = 12
    num_kv_heads: int = 2
    head_dim: int = 128
    intermediate_size: int = 8960
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    mrope_section: List[int] = field(default_factory=lambda: [16, 24, 24])


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_mrope(q, k, cos, sin, mrope_section):
    """M-RoPE: cos/sin are [3, B, L, head_dim] (one per t/h/w position axis).
    Each frequency band is driven by one axis per the doubled mrope_section."""
    sec = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(sec, dim=-1))], dim=-1).unsqueeze(1)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(sec, dim=-1))], dim=-1).unsqueeze(1)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class Attention(nn.Module):
    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.nh, self.nkv, self.hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.nh * self.hd, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=True)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.hidden_size, bias=False)
        self.mrope_section = cfg.mrope_section

    def forward(self, x, cos, sin, attn_mask, cache=None, layer_idx=0):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.nkv, self.hd).transpose(1, 2)
        q, k = apply_mrope(q, k, cos, sin, self.mrope_section)
        if cache is not None:
            k, v = cache.update(k, v, layer_idx)   # returns the full K/V window
        # GQA: expand KV heads to match query heads
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, L, self.nh * self.hd)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, attn_mask, cache=None, layer_idx=0):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, attn_mask, cache, layer_idx)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen2LLM(nn.Module):
    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # tied (saves ~0.5 GB)
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rope(self, position_ids):
        # position_ids: [3, B, L] -> cos/sin [3, B, L, head_dim]
        inv = self.inv_freq.to(position_ids.device)
        freqs = position_ids[..., None].float() * inv  # [3, B, L, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    def forward(self, inputs_embeds, position_ids, attn_mask, cache=None):
        cos, sin = self.rope(position_ids)
        cos, sin = cos.to(inputs_embeds.dtype), sin.to(inputs_embeds.dtype)
        h = inputs_embeds
        for i, layer in enumerate(self.layers):
            h = layer(h, cos, sin, attn_mask, cache, i)
        h = self.norm(h)
        return self.lm_head(h)


def load_from_hf(model: Qwen2LLM, hf_state_dict: dict) -> None:
    """Copy weights from a Qwen2VL state_dict into the custom model. Handles both
    the raw safetensors layout (`model.layers...`) and the loaded-model layout
    (`model.language_model.layers...`)."""
    sd = hf_state_dict
    P = "model.language_model." if "model.language_model.embed_tokens.weight" in sd else "model."
    model.embed_tokens.weight.data.copy_(sd[P + "embed_tokens.weight"])
    model.norm.weight.data.copy_(sd[P + "norm.weight"])
    # lm_head is tied to embed_tokens (same tensor), so it's already loaded
    for i, layer in enumerate(model.layers):
        lp = f"{P}layers.{i}."
        a = layer.self_attn
        for name, mod in [("q_proj", a.q_proj), ("k_proj", a.k_proj), ("v_proj", a.v_proj)]:
            mod.weight.data.copy_(sd[lp + f"self_attn.{name}.weight"])
            mod.bias.data.copy_(sd[lp + f"self_attn.{name}.bias"])
        a.o_proj.weight.data.copy_(sd[lp + "self_attn.o_proj.weight"])
        layer.mlp.gate_proj.weight.data.copy_(sd[lp + "mlp.gate_proj.weight"])
        layer.mlp.up_proj.weight.data.copy_(sd[lp + "mlp.up_proj.weight"])
        layer.mlp.down_proj.weight.data.copy_(sd[lp + "mlp.down_proj.weight"])
        layer.input_layernorm.weight.data.copy_(sd[lp + "input_layernorm.weight"])
        layer.post_attention_layernorm.weight.data.copy_(sd[lp + "post_attention_layernorm.weight"])
