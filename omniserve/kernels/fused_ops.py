"""自写 forward 用的融合 Triton 算子(照搬 vLLM 的层内融合套路)。

- add_rms_norm:把「残差 add」折进 RMSNorm 一个 kernel(残差穿层传递,少两次单独的
  add kernel/访存)。对应 vLLM 的 fused_add_rms_norm。
- rms_norm:无残差版(第一层 / 最终 norm 用)。
- silu_mul:SwiGLU 的 `silu(gate) * up` 一个 kernel。对应 vLLM 的 SiluAndMul,配合
  gate/up 合并成一个 GEMM 使用。

数值上对齐原始 PyTorch:残差 add 在 fp16 里做(和 `x = x + sub` 一致),RMSNorm 的
reduce 在 fp32 里做。验证和未融合版逐 token 一致。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------- RMSNorm(无残差) ----------------
@triton.jit
def _rms_fwd(X, W, Y, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * stride + cols, (x * rstd * w).to(Y.dtype.element_ty), mask=mask)


def rms_norm(x, weight, eps):
    n = x.shape[-1]
    x2d = x.reshape(-1, n).contiguous()
    out = torch.empty_like(x2d)
    block = triton.next_power_of_2(n)
    _rms_fwd[(x2d.shape[0],)](x2d, weight, out, x2d.stride(0), n, eps, BLOCK=block,
                              num_warps=max(1, min(16, block // 256)))
    return out.reshape(x.shape)


# ---------------- 残差 add + RMSNorm 融合 ----------------
@triton.jit
def _add_rms_fwd(X, RES, W, Y, NRES, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(RES + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    r = r + x
    # 先按输入 dtype 落一遍新残差,再用它做 norm —— 和「fp16 残差 add 后再 norm」一致
    r16 = r.to(NRES.dtype.element_ty)
    tl.store(NRES + row * stride + cols, r16, mask=mask)
    rf = r16.to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(rf * rf, axis=0) / N + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * stride + cols, (rf * rstd * w).to(Y.dtype.element_ty), mask=mask)


def add_rms_norm(x, residual, weight, eps):
    """返回 (normed, new_residual);new_residual = residual + x。"""
    n = x.shape[-1]
    x2d = x.reshape(-1, n).contiguous()
    res2d = residual.reshape(-1, n).contiguous()
    out = torch.empty_like(x2d)
    nres = torch.empty_like(res2d)
    block = triton.next_power_of_2(n)
    _add_rms_fwd[(x2d.shape[0],)](x2d, res2d, weight, out, nres, x2d.stride(0), n, eps,
                                  BLOCK=block, num_warps=max(1, min(16, block // 256)))
    return out.reshape(x.shape), nres.reshape(residual.shape)


# ---------------- 残差 add + RMSNorm + FP8 量化 融合 ----------------
@triton.jit
def _add_rms_fp8_fwd(X, RES, W, Y, SCALE, NRES, stride, N, eps, BLOCK: tl.constexpr):
    """在 add_rms 的同一个 per-row program 里顺手把 normed 结果量化成 FP8(per-token scale)。
    norm 本来就按行 reduce,行内 amax 几乎免费 —— 省掉下游 FP8 GEMM 那笔独立的量化 launch。"""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(RES + row * stride + cols, mask=mask, other=0.0).to(tl.float32) + x
    r16 = r.to(NRES.dtype.element_ty)
    tl.store(NRES + row * stride + cols, r16, mask=mask)         # 新残差(fp16)穿层传递
    rf = r16.to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(rf * rf, axis=0) / N + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = rf * rstd * w                                            # fp32 normed
    scale = tl.maximum(tl.max(tl.abs(y), axis=0) / 448.0, 1e-12)  # per-token scale(行内 amax)
    tl.store(Y + row * N + cols, (y / scale).to(Y.dtype.element_ty), mask=mask)  # 直接吐 fp8
    tl.store(SCALE + row, scale)


def add_rms_norm_fp8(x, residual, weight, eps):
    """= add_rms_norm,但 normed 输出直接量化成 FP8(e4m3, per-token scale)。
    返回 (fp8_out [M,N], scale [M,1] fp32, new_residual)。供 gate_up 等 FP8 GEMM 免量化消费。"""
    n = x.shape[-1]
    x2d = x.reshape(-1, n).contiguous()
    res2d = residual.reshape(-1, n).contiguous()
    m = x2d.shape[0]
    out = torch.empty(m, n, device=x2d.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty(m, 1, device=x2d.device, dtype=torch.float32)
    nres = torch.empty_like(res2d)
    block = triton.next_power_of_2(n)
    _add_rms_fp8_fwd[(m,)](x2d, res2d, weight, out, scale, nres, x2d.stride(0), n, eps,
                           BLOCK=block, num_warps=max(1, min(16, block // 256)))
    return out, scale, nres.reshape(residual.shape)


# ---------------- SiLU(gate) * up 融合 ----------------
@triton.jit
def _silu_mul_fwd(GU, Y, s_in, s_out, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    g = tl.load(GU + row * s_in + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(GU + row * s_in + N + cols, mask=mask, other=0.0).to(tl.float32)
    out = (g * tl.sigmoid(g)) * u
    tl.store(Y + row * s_out + cols, out.to(Y.dtype.element_ty), mask=mask)


def silu_mul(gate_up):
    """gate_up: [..., 2N](gate 在前、up 在后)-> [..., N]。"""
    n = gate_up.shape[-1] // 2
    gu = gate_up.reshape(-1, 2 * n).contiguous()
    out = torch.empty(gu.shape[0], n, device=gu.device, dtype=gu.dtype)
    block = triton.next_power_of_2(n)
    _silu_mul_fwd[(gu.shape[0],)](gu, out, gu.stride(0), out.stride(0), n,
                                  BLOCK=block, num_warps=max(1, min(16, block // 256)))
    return out.reshape(*gate_up.shape[:-1], n)
