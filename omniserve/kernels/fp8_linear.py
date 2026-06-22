"""FP8(W8A8,e4m3)Linear,用 sm_89 原生 FP8 tensor core(2x 算力)。

⚠️ 甜区是 **prefill 大 GEMM(算力受限)**,不是 decode——这条在 P3 初版里搞反了,P5 第18阶段
micro-bench 纠正:FP8 吃 2x 算力,gate_up M=512 实测 **2.16x**、M=1024 1.78x,明显赢 W4A16
(~1.1x);而 decode 带宽受限、M 小,激活在线量化开销吃掉收益(qkv/o M=24 仅 0.45x),且 FP8
权重 1B 不如 int4 0.5B 省字节 → **decode 该用 Marlin W4A16,prefill 大 GEMM 用 FP8**。

难点:_scaled_mm 要求两个操作数都是 FP8,激活要动态量化——用融合 Triton kernel(amax+cast 一遍)
把这个固定开销压住(大 M 时被摊薄)。per-channel(权重每输出通道一 scale)+ per-token(激活每行
一 scale)的 RowWise 量化,_scaled_mm 输出 bf16 再转 fp16。真实模型实测 rel ~0.03(约为无校准
RTN-int4 的 1/3);RowWise 相比 per-tensor 在文本激活上几乎无额外收益(outlier 不重)。
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
    """nn.Linear 的 FP8 替代:权重 e4m3,激活动态 FP8。

    默认 **per-tensor**(权重/激活各一个标量 scale,`_scaled_mm` 直接出 fp16):文本激活 outlier
    不重(项目已测 rowwise≈per-tensor,rel 0.0143 vs 0.0147),而 per-tensor 快得多——micro-bench
    隔离显示 rowwise(per-token + bf16→fp16 转换)在 prefill M 上**比 fp16 还慢 0.63–0.83x**,
    per-tensor 才是 **1.0–1.77x**。`rowwise=True` 留给真有重 outlier 的场景(代价是 prefill 反亏)。"""

    def __init__(self, weight: torch.Tensor, bias=None, hadamard: bool = False,
                 rowwise: bool = False):
        super().__init__()
        self.hadamard = hadamard
        self.rowwise = rowwise
        if hadamard:                                          # 离线吸收 W·H(旋转后再量化)
            from .hadamard import rotate
            weight = rotate(weight)
        self.N, self.K = weight.shape
        if rowwise:                                           # 权重 per-channel
            wscale = (weight.abs().amax(1, keepdim=True) / 448.0).float()  # [N,1]
            self.register_buffer("wq", (weight / wscale).to(E4M3))        # [N,K]
            self.register_buffer("wscale", wscale.t().contiguous())       # [1,N]
        else:                                                 # 权重 per-tensor(标量 scale)
            wscale = (weight.abs().max() / 448.0).float()                 # 标量
            self.register_buffer("wq", (weight / wscale).to(E4M3))        # [N,K]
            self.register_buffer("wscale", wscale)
        self.bias = bias

    @classmethod
    def from_linear(cls, linear: nn.Linear, hadamard: bool = False,
                    rowwise: bool = False) -> "FP8Linear":
        """从 fp16 nn.Linear 构造,和 MarlinInt4Linear(linear) 用法对齐(per-op 替换用)。"""
        return cls(linear.weight.data, linear.bias, hadamard=hadamard, rowwise=rowwise)

    def forward(self, x):
        shape = x.shape
        if self.hadamard:                                    # 激活在线旋转 a·H
            from .hadamard import rotate
            x = rotate(x)
        if self.rowwise:                                     # per-token 激活,_scaled_mm 只支持 bf16 输出
            xq, xs = quant_fp8(x)
            out = torch._scaled_mm(xq, self.wq.t(), scale_a=xs, scale_b=self.wscale,
                                   out_dtype=torch.bfloat16).to(torch.float16)
        else:                                                # per-tensor 激活,直接出 fp16(省转换)
            x2d = x.reshape(-1, self.K)
            sa = (x2d.abs().max() / 448.0).float()
            xq = (x2d / sa).to(E4M3)
            out = torch._scaled_mm(xq, self.wq.t(), scale_a=sa, scale_b=self.wscale,
                                   out_dtype=torch.float16)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape[:-1], self.N)


class StageFP8Linear(nn.Module):
    """按激活 M 维自适应精度(roofline 驱动的逐前向切换):
      M ≥ 阈值(prefill / chunk / mixed,算力受限)→ FP8 W8A8(吃 2x 算力);
      M <  阈值(decode,带宽受限,小 GEMM FP8 反而慢)→ fp16。
    存 **fp16 权重**(decode 直接用、显存中性),FP8 路径**在线量化权重**(大 M 把这点开销摊掉)。
    把"qkv/o 该 prefill FP8、decode fp16"那条 micro-bench 结论直接编码进 kernel——prefill/chunk/
    mixed_step(M=n+L)自动走 FP8、纯 decode(M=batch)自动走 fp16,不需要调度器告诉它现在哪个阶段。
    ⚠️ 因为 decode 要 fp16 权重,这条**不省 qkv/o 显存**(是吞吐导向,不是显存导向)。"""

    def __init__(self, linear: nn.Linear, m_threshold: int = 64):
        super().__init__()
        self.weight = linear.weight                            # 复用 fp16 权重,不复制(显存中性)
        self.bias = linear.bias
        self.N, self.K = linear.weight.shape
        self.m_threshold = m_threshold

    @classmethod
    def from_linear(cls, linear: nn.Linear, m_threshold: int = 64) -> "StageFP8Linear":
        return cls(linear, m_threshold)

    def forward(self, x):
        shape = x.shape
        M = 1
        for s in shape[:-1]:
            M *= s
        if M < self.m_threshold:                              # decode:带宽受限,fp16 更快
            return torch.nn.functional.linear(x, self.weight, self.bias)
        # prefill / chunk:算力受限,FP8 per-tensor(直接出 fp16,避开 rowwise 的转换/量化开销)
        x2d = x.reshape(-1, self.K)
        sa = (x2d.abs().max() / 448.0).float()
        xq = (x2d / sa).to(E4M3)
        wscale = (self.weight.abs().max() / 448.0).float()                  # 标量,在线量化权重
        wq = (self.weight / wscale).to(E4M3)
        out = torch._scaled_mm(xq, wq.t(), scale_a=sa, scale_b=wscale, out_dtype=torch.float16)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape[:-1], self.N)


def apply_fp8(model, scope: str = "gate_up") -> int:
    """把 decoder 里指定算子换成 FP8Linear(对齐 marlin_linear.quantize_llm_marlin)。返回替换数。
       scope:
         "gate_up" = 只 gate_up(prefill 甜区、最稳,默认);
         "mlp"     = gate_up + down;
         "decoder" = 全四个(qkv/o/gate_up/down)。
    ⚠️ 纯 FP8 全量化只是积木;真正的 per-op 最优(gate_up=FP8 + qkv/o=fp16 + decode-MLP=W4A16)
       在 per-op 策略层做混合,FP8 主要落在 prefill 大 GEMM 上。"""
    n = 0
    for layer in model.layers:
        if scope == "decoder":
            layer.self_attn.qkv_proj = FP8Linear.from_linear(layer.self_attn.qkv_proj)
            layer.self_attn.o_proj = FP8Linear.from_linear(layer.self_attn.o_proj)
            n += 2
        layer.mlp.gate_up_proj = FP8Linear.from_linear(layer.mlp.gate_up_proj)
        n += 1
        if scope in ("mlp", "decoder"):
            layer.mlp.down_proj = FP8Linear.from_linear(layer.mlp.down_proj)
            n += 1
    return n
