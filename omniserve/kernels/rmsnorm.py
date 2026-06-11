"""Fused RMSNorm Triton kernel + a drop-in replacement for Qwen2RMSNorm.

HF's `Qwen2RMSNorm` runs as several PyTorch ops (upcast, square, mean, rsqrt,
multiply, downcast, weight multiply) — each a separate kernel reading/writing the
activation from HBM. RMSNorm is memory-bound, so fusing the whole thing into one
Triton kernel (one read, one write) is the textbook win.

It matches HF's numerics: reduce in fp32 for stability, then scale by the weight
and store in the input dtype. Targets the LLM stack (dim 1536); the ViT uses
LayerNorm, not RMSNorm, so it is left untouched.
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
    """Drop-in for transformers' Qwen2RMSNorm, backed by the Triton kernel."""

    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        self.weight = weight  # share the original parameter (no copy)
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, self.variance_epsilon)


def patch_llm_rmsnorm(model) -> int:
    """Replace every Qwen2RMSNorm in the language model with FusedRMSNorm.
    Returns the number of modules swapped. The ViT (LayerNorm) is left alone."""
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2RMSNorm

    n = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, Qwen2RMSNorm):
                eps = getattr(child, "variance_epsilon", 1e-6)
                setattr(module, name, FusedRMSNorm(child.weight, eps))
                n += 1
    return n
