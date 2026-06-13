"""从零实现的 Qwen2 语言模型 forward(Qwen2-VL 的文本解码器)。

它替代 transformers 的 `Qwen2VLForConditionalGeneration` 语言部分,让引擎自己控制
forward——这是做 varlen/packed prefill、混合 batching、融合 kernel 的前提(HF 的
forward 表达不了这些)。vision 编码器(ViT)由 model/qwen2_vit.py 自己实现;它的
embedding 在本 forward 跑之前被拼进文本 embedding。

架构(Qwen2-VL-2B):hidden 1536,28 层,12 query 头 / 2 KV 头(GQA),head_dim 128,
SwiGLU intermediate 8960,RMSNorm eps 1e-6,rope_theta 1e6,M-RoPE section [16,24,24],
tied embeddings,q/k/v 带 bias、o 不带。

正确性是底线:和 HF 逐 token 一致(scripts/proto_custom_llm.py 验证)。优化在后面。
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

    def forward(self, x, residual=None):
        # 用融合 Triton kernel:无残差 -> rms_norm;有残差 -> add_rms_norm
        # (残差 add 折进 norm,返回 (normed, new_residual))。
        from ..kernels.fused_ops import add_rms_norm, rms_norm
        if residual is None:
            return rms_norm(x, self.weight, self.eps)
        return add_rms_norm(x, residual, self.weight, self.eps)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_mrope(q, k, cos, sin, mrope_section):
    """M-RoPE:cos/sin 是 [3, B, L, head_dim](t/h/w 三个位置轴各一份)。
    按加倍后的 mrope_section,每个频率带由其中一个轴驱动。"""
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
        self.q_size = self.nh * self.hd
        self.kv_size = self.nkv * self.hd
        # 融合 QKV:q/k/v 合成一个 GEMM(照搬 vLLM 的 QKVParallelLinear),算完 split
        self.qkv_proj = nn.Linear(cfg.hidden_size, self.q_size + 2 * self.kv_size, bias=True)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.hidden_size, bias=False)
        self.mrope_section = cfg.mrope_section

    def forward(self, x, cos, sin, attn_mask, cache=None, layer_idx=0):
        B, L, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(B, L, self.nh, self.hd).transpose(1, 2)
        k = k.view(B, L, self.nkv, self.hd).transpose(1, 2)
        v = v.view(B, L, self.nkv, self.hd).transpose(1, 2)
        q, k = apply_mrope(q, k, cos, sin, self.mrope_section)
        # decode 走 FlashAttention(GQA-native,不物化 KV,比 SDPA+repeat_interleave
        # 快 ~11x);prefill 没有 flash_decode 的 cache,走下面的 SDPA。
        if cache is not None and hasattr(cache, "flash_decode"):
            out = cache.flash_decode(q[:, :, 0, :].contiguous(), k[:, :, 0, :], v[:, :, 0, :],
                                     layer_idx, self.hd ** -0.5)     # [B, nh, hd]
            return self.o_proj(out.reshape(B, L, self.nh * self.hd))
        if cache is not None and hasattr(cache, "flash_prefill"):
            # 打包 prefill 走 flash varlen(段内因果,O(sum L²),不再 O(total²) SDPA)
            out = cache.flash_prefill(q[0].transpose(0, 1), k[0].transpose(0, 1),
                                      v[0].transpose(0, 1), layer_idx, self.hd ** -0.5)
            return self.o_proj(out.reshape(B, L, self.nh * self.hd))
        if cache is not None:
            k, v = cache.update(k, v, layer_idx)   # 返回完整的 K/V 窗口
        # GQA:把 KV 头物化扩到 query 头数(prefill 路径用 SDPA)
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, L, self.nh * self.hd)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        # 融合 gate+up:合成一个 GEMM(照搬 vLLM 的 MergedColumnParallelLinear)
        self.gate_up_proj = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        from ..kernels.fused_ops import silu_mul
        return self.down_proj(silu_mul(self.gate_up_proj(x)))   # silu(gate)*up 一个 kernel


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, residual, cos, sin, attn_mask, cache=None, layer_idx=0):
        # 残差穿层传递(照搬 vLLM):add 折进 RMSNorm,层内不写显式的 x = x + ...
        if residual is None:
            residual = x
            x = self.input_layernorm(x)
        else:
            x, residual = self.input_layernorm(x, residual)
        x = self.self_attn(x, cos, sin, attn_mask, cache, layer_idx)
        x, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(x)
        return x, residual


class Qwen2LLM(nn.Module):
    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # tied 共享(省 ~0.5 GB)
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
        residual = None
        for i, layer in enumerate(self.layers):
            h, residual = layer(h, residual, cos, sin, attn_mask, cache, i)
        h, _ = self.norm(h, residual)   # 最终 norm 折进最后一次残差 add
        return self.lm_head(h)


def load_from_hf(model: Qwen2LLM, hf_state_dict: dict) -> None:
    """把 Qwen2VL 的 state_dict 权重拷进自定义模型。同时支持两种命名:raw
    safetensors 布局(`model.layers...`)和加载后模型布局(`model.language_model.layers...`)。"""
    sd = hf_state_dict
    # 踩坑:raw safetensors 用 `model.layers.` 前缀,而 transformers 加载后的
    # state_dict() 是 `model.language_model.layers.`(新版重构了结构)。两种都支持。
    P = "model.language_model." if "model.language_model.embed_tokens.weight" in sd else "model."
    model.embed_tokens.weight.data.copy_(sd[P + "embed_tokens.weight"])
    model.norm.weight.data.copy_(sd[P + "norm.weight"])
    # lm_head 和 embed_tokens 是 tied(同一个张量),所以已经一起加载好了
    for i, layer in enumerate(model.layers):
        lp = f"{P}layers.{i}."
        a = layer.self_attn
        # 融合 QKV:把 q/k/v 的权重和 bias 沿输出维 concat 进一个 qkv_proj
        a.qkv_proj.weight.data.copy_(torch.cat([
            sd[lp + "self_attn.q_proj.weight"], sd[lp + "self_attn.k_proj.weight"],
            sd[lp + "self_attn.v_proj.weight"]], dim=0))
        a.qkv_proj.bias.data.copy_(torch.cat([
            sd[lp + "self_attn.q_proj.bias"], sd[lp + "self_attn.k_proj.bias"],
            sd[lp + "self_attn.v_proj.bias"]], dim=0))
        a.o_proj.weight.data.copy_(sd[lp + "self_attn.o_proj.weight"])
        # 融合 gate+up:concat 进一个 gate_up_proj(gate 在前、up 在后)
        layer.mlp.gate_up_proj.weight.data.copy_(torch.cat([
            sd[lp + "mlp.gate_proj.weight"], sd[lp + "mlp.up_proj.weight"]], dim=0))
        layer.mlp.down_proj.weight.data.copy_(sd[lp + "mlp.down_proj.weight"])
        layer.input_layernorm.weight.data.copy_(sd[lp + "input_layernorm.weight"])
        layer.post_attention_layernorm.weight.data.copy_(sd[lp + "post_attention_layernorm.weight"])
