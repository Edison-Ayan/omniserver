"""融合的 RMSNorm Triton kernel + Qwen2RMSNorm 的 drop-in 替代。

HF 的 `Qwen2RMSNorm` 跑成好几个 PyTorch op(upcast、平方、求均值、rsqrt、乘、downcast、
乘权重)——每个都是单独的 kernel,各从 HBM 读写一遍激活。RMSNorm 是访存受限的,所以
把整件事融合进一个 Triton kernel(一次读、一次写)是教科书式的优化。

它和 HF 的数值对齐:在 fp32 里 reduce 以保稳定,再乘权重、按输入 dtype 存。针对 LLM
栈(dim 1536);ViT 用的是 LayerNorm 不是 RMSNorm,所以不动它。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd(X, W, Y, stride_row, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptr = X + row * stride_row
    y_ptr = Y + row * stride_row
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(y_ptr + cols, y.to(Y.dtype.element_ty), mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """y = x / sqrt(mean(x^2) + eps) * weight, fused. x: (..., N)."""
    n = x.shape[-1]
    x2d = x.reshape(-1, n)
    x2d = x2d.contiguous() if not x2d.is_contiguous() else x2d
    out = torch.empty_like(x2d)
    block = triton.next_power_of_2(n)
    _rmsnorm_fwd[(x2d.shape[0],)](
        x2d, weight, out, x2d.stride(0), n, eps, BLOCK=block,
        num_warps=max(1, min(16, block // 256)),
    )
    return out.reshape(x.shape)


class FusedRMSNorm(torch.nn.Module):
    """transformers Qwen2RMSNorm 的 drop-in 替代,底层用 Triton kernel。"""

    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        self.weight = weight  # 共享原参数(不拷贝)
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, self.variance_epsilon)


def patch_llm_rmsnorm(model) -> int:
    """把语言模型里的每个 Qwen2RMSNorm 换成 FusedRMSNorm。
    返回替换的模块数。ViT(LayerNorm)不动。"""
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2RMSNorm

    n = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, Qwen2RMSNorm):
                eps = getattr(child, "variance_epsilon", 1e-6)
                setattr(module, name, FusedRMSNorm(child.weight, eps))
                n += 1
    return n
