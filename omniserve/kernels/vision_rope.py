"""融合的 ViT 2D rotary kernel。

原来的 apply_rotary_vision 是好几个 op:q/k 各 upcast 到 fp32、rotate_half(一次 cat)、
乘 cos、乘 sin、相加、downcast——每个都是单独的小 kernel,nsys 显示这些碎片 elementwise
占 ViT ~一成多。这里融成一个 Triton kernel(一次读 q/k,在寄存器里算完旋转,一次写回)。

ViT 的 rotary:cos/sin 是 [S, hd],且 emb=cat(rpe,rpe) 所以前后半相同。对每个 (patch, head)
的向量,前半 out1 = q1*cos - q2*sin,后半 out2 = q2*cos + q1*sin(rotate_half 的展开)。
fp32 计算保精度,按 q/k 的 dtype 写回。原地写(q/k 来自刚算的 qkv,可覆盖)。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_one(base, cos, sin, cols, half, mask):
    q1 = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
    q2 = tl.load(base + half + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(base + cols, (q1 * cos - q2 * sin).to(base.dtype.element_ty), mask=mask)
    tl.store(base + half + cols, (q2 * cos + q1 * sin).to(base.dtype.element_ty), mask=mask)


@triton.jit
def _vis_rope(Q, K, COS, SIN, n_heads, hd, half, BLOCK: tl.constexpr):
    pid = tl.program_id(0)               # 覆盖 S*n_heads 个向量
    s = pid // n_heads
    cols = tl.arange(0, BLOCK)
    mask = cols < half
    cos = tl.load(COS + s * hd + cols, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(SIN + s * hd + cols, mask=mask, other=0.0).to(tl.float32)
    _rope_one(Q + pid * hd, cos, sin, cols, half, mask)
    _rope_one(K + pid * hd, cos, sin, cols, half, mask)


def vision_rope(q, k, cos, sin):
    """q,k: [S, n_heads, hd];cos,sin: [S, hd]。原地旋转 q,k 并返回。"""
    S, nh, hd = q.shape
    half = hd // 2
    q = q.contiguous()
    k = k.contiguous()
    _vis_rope[(S * nh,)](q, k, cos.contiguous(), sin.contiguous(), nh, hd, half,
                         BLOCK=triton.next_power_of_2(half))
    return q, k
