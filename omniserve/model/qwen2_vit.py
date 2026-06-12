"""从零实现的 Qwen2-VL 视觉编码器(ViT),和 HF 对齐。

拿掉了性能路径上最后一个大的 transformers 依赖。架构:
patch embed(对 2x14x14 patch 做 Conv3d)-> 32 个 block(LayerNorm,QKV 注意力 +
按 (h,w) patch 位置应用的 2D rotary,quick-GELU MLP)-> patch merger(把
spatial_merge_size^2 个相邻 patch 合并,过 MLP)-> hidden_size 1536。

输出和 HF 的 `model.visual` 逐 bit 一致(scripts/proto_custom_vit.py 验证)。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_vision(q, k, cos, sin):
    orig = q.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q.to(orig), k.to(orig)


class VisionBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.attn_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_proj = nn.Linear(dim, dim, bias=True)
        self.fc1 = nn.Linear(dim, mlp_dim, bias=True)
        self.fc2 = nn.Linear(mlp_dim, dim, bias=True)
        self.n_heads = n_heads
        self.hd = dim // n_heads

    def attn(self, x, cos, sin, cu_seqlens):
        S = x.shape[0]
        qkv = self.attn_qkv(x).reshape(S, 3, self.n_heads, self.hd).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]            # [S, nh, hd]
        q, k = apply_rotary_vision(q, k, cos, sin)
        # 每张图段内部做全注意力(cu_seqlens 划分各张图)
        outs = []
        for i in range(len(cu_seqlens) - 1):
            a, b = cu_seqlens[i], cu_seqlens[i + 1]
            qs = q[a:b].transpose(0, 1).unsqueeze(0)   # [1, nh, L, hd]
            ks = k[a:b].transpose(0, 1).unsqueeze(0)
            vs = v[a:b].transpose(0, 1).unsqueeze(0)
            o = F.scaled_dot_product_attention(qs, ks, vs)
            outs.append(o.squeeze(0).transpose(0, 1))  # [L, nh, hd]
        out = torch.cat(outs, dim=0).reshape(S, -1)
        return self.attn_proj(out)

    def forward(self, x, cos, sin, cu_seqlens):
        x = x + self.attn(self.norm1(x), cos, sin, cu_seqlens)
        x = x + self.fc2(quick_gelu(self.fc1(self.norm2(x))))
        return x


class Qwen2VIT(nn.Module):
    def __init__(self, embed_dim=1280, depth=32, n_heads=16, in_chans=3,
                 patch_size=14, temporal_patch_size=2, spatial_merge_size=2,
                 mlp_ratio=4, out_hidden=1536, rope_theta=10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.spatial_merge_size = spatial_merge_size
        self.patch_embed = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size), bias=False)
        self.blocks = nn.ModuleList([
            VisionBlock(embed_dim, n_heads, embed_dim * mlp_ratio) for _ in range(depth)])
        merged = embed_dim * spatial_merge_size ** 2
        self.merger_ln_q = nn.LayerNorm(embed_dim, eps=1e-6)
        self.merger_fc1 = nn.Linear(merged, merged, bias=True)
        self.merger_fc2 = nn.Linear(merged, out_hidden, bias=True)
        self._merged = merged
        self._in = (in_chans, temporal_patch_size, patch_size)
        # rotary 频率必须保持 fp32——对模块 .half() 会丢精度,误差会在 32 层里累积放大。
        self._rot_dim = (embed_dim // n_heads) // 2
        self._rope_theta = rope_theta

    def _rot_pos_emb(self, grid_thw, device):
        m = self.spatial_merge_size
        pos_ids = []
        for t, h, w in grid_thw.tolist():
            hp = torch.arange(h).unsqueeze(1).expand(-1, w)
            hp = hp.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
            wp = torch.arange(w).unsqueeze(0).expand(h, -1)
            wp = wp.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hp, wp], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0).to(device)
        max_grid = int(grid_thw[:, 1:].max())
        # 踩坑:inv_freq 必须在 CPU 上用 fp32 算。一开始在 GPU 上算,和 HF 在 CPU
        # 初始化差了 ~9.4e-7(ULP 级),这个微小差异被 32 层 fp16 残差链指数放大成
        # 2.5 的输出误差。改成 CPU+fp32 后整个 ViT 和 HF bit 级一致(误差 0.0)。
        inv_freq = (1.0 / (self._rope_theta ** (
            torch.arange(0, self._rot_dim, 2, dtype=torch.float32) / self._rot_dim))).to(device)
        seq = torch.arange(max_grid, device=device, dtype=torch.float32)
        freqs = torch.outer(seq, inv_freq)             # [max_grid, rot_dim/2], fp32
        rpe = freqs[pos_ids].flatten(1)                # [S, rot_dim]
        emb = torch.cat((rpe, rpe), dim=-1)            # [S, head_dim]
        return emb.cos(), emb.sin()

    @torch.inference_mode()
    def forward(self, pixel_values, grid_thw):
        c, tp, ps = self._in
        x = pixel_values.view(-1, c, tp, ps, ps).to(self.patch_embed.weight.dtype)
        x = self.patch_embed(x).view(-1, self.embed_dim)
        # 踩坑:cos/sin 全程保持 fp32(不要 .to(fp16)),否则同样会在深层链放大
        cos, sin = self._rot_pos_emb(grid_thw, x.device)
        cu = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(0)
        cu = F.pad(cu, (1, 0), value=0).to(torch.int32)
        for blk in self.blocks:
            x = blk(x, cos, sin, cu)
        x = self.merger_fc2(F.gelu(self.merger_fc1(self.merger_ln_q(x).view(-1, self._merged))))
        return x


def load_vit_from_state_dict(vit: Qwen2VIT, sd: dict) -> None:
    """Load weights from a HF checkpoint state_dict (keys prefixed 'visual.')."""
    g = lambda k: sd["visual." + k]
    vit.patch_embed.weight.data.copy_(g("patch_embed.proj.weight"))
    for i, blk in enumerate(vit.blocks):
        p = f"blocks.{i}."
        blk.norm1.weight.data.copy_(g(p + "norm1.weight")); blk.norm1.bias.data.copy_(g(p + "norm1.bias"))
        blk.norm2.weight.data.copy_(g(p + "norm2.weight")); blk.norm2.bias.data.copy_(g(p + "norm2.bias"))
        blk.attn_qkv.weight.data.copy_(g(p + "attn.qkv.weight")); blk.attn_qkv.bias.data.copy_(g(p + "attn.qkv.bias"))
        blk.attn_proj.weight.data.copy_(g(p + "attn.proj.weight")); blk.attn_proj.bias.data.copy_(g(p + "attn.proj.bias"))
        blk.fc1.weight.data.copy_(g(p + "mlp.fc1.weight")); blk.fc1.bias.data.copy_(g(p + "mlp.fc1.bias"))
        blk.fc2.weight.data.copy_(g(p + "mlp.fc2.weight")); blk.fc2.bias.data.copy_(g(p + "mlp.fc2.bias"))
    vit.merger_ln_q.weight.data.copy_(g("merger.ln_q.weight")); vit.merger_ln_q.bias.data.copy_(g("merger.ln_q.bias"))
    vit.merger_fc1.weight.data.copy_(g("merger.mlp.0.weight")); vit.merger_fc1.bias.data.copy_(g("merger.mlp.0.bias"))
    vit.merger_fc2.weight.data.copy_(g("merger.mlp.2.weight")); vit.merger_fc2.bias.data.copy_(g("merger.mlp.2.bias"))
