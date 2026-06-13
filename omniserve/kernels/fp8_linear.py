"""FP8(e4m3)权重量化 Linear,用 sm_89 原生 FP8 tensor core 砍 decode 的权重带宽。

decode 是访存受限(读 fp16 权重 ~4.4GB/步,nsys 显示 GEMM 占 ~82%)。FP8 权重 1 字节
(fp16 的一半)+ FP8 tensor core → 大 GEMM 实测最高 ~5x。难点:_scaled_mm 要求两个操作数
都是 FP8,所以激活也要动态量化——用融合 Triton kernel(amax+cast 一遍)把这个开销压住。

per-channel(权重每输出通道一个 scale)+ per-token(激活每行一个 scale)的 RowWise 量化,
_scaled_mm 输出 bf16 再转 fp16。FP8 e4m3 有 ~3-4% 的精度地板,所以是有损的(看质量验证)。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import nn

E4M3 = torch.float8_e4m3fn


@triton.jit
def _quant_fwd(X, Y, S, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0          # e4m3 最大 ~448
    tl.store(S + row, s)
    tl.store(Y + row * stride + cols, (x / s).to(Y.dtype.element_ty), mask=mask)


def quant_fp8(x):
    """把 [..., K] 动态量化成 FP8(per-token scale),融合成一个 kernel。"""
    shape = x.shape
    x2d = x.reshape(-1, shape[-1]).contiguous()
    m, k = x2d.shape
    y = torch.empty(m, k, device=x.device, dtype=E4M3)
    s = torch.empty(m, device=x.device, dtype=torch.float32)
    _quant_fwd[(m,)](x2d, y, s, x2d.stride(0), k, BLOCK=triton.next_power_of_2(k))
    return y, s.view(m, 1)


class FP8Linear(nn.Module):
    """nn.Linear 的 FP8 替代:权重 e4m3(per-channel scale),激活动态 FP8。"""

    def __init__(self, weight: torch.Tensor, bias=None):
        super().__init__()
        wscale = (weight.abs().amax(1, keepdim=True) / 448.0).float()  # [N,1]
        self.register_buffer("wq", (weight / wscale).to(E4M3))         # [N,K]
        self.register_buffer("wscale", wscale.t().contiguous())        # [1,N]
        self.bias = bias

    def forward(self, x):
        shape = x.shape
        xq, xs = quant_fp8(x)
        out = torch._scaled_mm(xq, self.wq.t(), scale_a=xs, scale_b=self.wscale,
                               out_dtype=torch.bfloat16)
        out = out.to(torch.float16)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape[:-1], -1)
