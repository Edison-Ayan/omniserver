"""为什么 FP8 prefill 端到端倒亏?隔离 FP8Linear 的 rowwise 实现开销 vs per-tensor。

端到端测出:prefill 阶段 mixed(FP8 MLP)反而比 fp16 慢。怀疑不是 GEMM 本身,而是当前
FP8Linear 的 rowwise 路径:per-token 激活量化(Triton)+ _scaled_mm 强制 bf16 输出 + 再转 fp16
(在 gate_up 的大输出 [M,17920] 上是巨量访存)。而当初 micro-bench 那个"FP8 prefill 1.77x"用的
是 per-tensor(量化一次 amax、输出直接 fp16),开销小得多。

本脚本在 prefill M 下,对每个真实形状对比:
  fp16 Linear / FP8-rowwise(现状)/ FP8-per-tensor(候选)/ rowwise 但不转 fp16(隔离转换开销)
随机权重(GEMM 速度只看形状)。结论决定:把 FP8Linear 的文本路径换成 per-tensor 是否能救回 prefill。
"""

from __future__ import annotations

import os
import sys
from statistics import median

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omniserve.kernels.fp8_linear import quant_fp8  # noqa: E402  (per-token Triton 量化)

E4M3 = torch.float8_e4m3fn
SHAPES = {"qkv": (1536, 2048), "o": (1536, 1536), "gate_up": (1536, 17920), "down": (8960, 1536)}
M_LIST = [512, 1024, 2048]


def cuda_us(fn, iters=50, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record()
        torch.cuda.synchronize()
        s.append(a.elapsed_time(b))
    return median(s) * 1000.0


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  sm_{p.major}{p.minor}\n")
    print(f"{'GEMM':8} {'M':>5} | {'fp16':>7} {'row+cvt':>8} {'row(bf16)':>9} {'per-tensor':>10} | "
          f"{'row↑':>5} {'pt↑':>5}")
    for name, (K, N) in SHAPES.items():
        W = torch.randn(N, K, device="cuda", dtype=torch.float16)
        lin = torch.nn.Linear(K, N, bias=False).half().cuda()
        lin.weight.data.copy_(W)
        # rowwise(现状 FP8Linear):权重 per-channel,激活 per-token,输出 bf16
        ws_r = (W.abs().amax(1, keepdim=True) / 448.0).float()
        wq_r = (W / ws_r).to(E4M3)
        ws_rt = ws_r.t().contiguous()
        # per-tensor:权重/激活各一个标量 scale,输出直接 fp16
        ws_t = (W.abs().max() / 448.0).float()
        wq_t = (W / ws_t).to(E4M3)

        def f_fp16(x):
            return lin(x)

        def f_row_cvt(x):                              # 现状:rowwise + bf16→fp16 转换
            xq, xs = quant_fp8(x)
            o = torch._scaled_mm(xq, wq_r.t(), scale_a=xs, scale_b=ws_rt, out_dtype=torch.bfloat16)
            return o.to(torch.float16)

        def f_row_bf16(x):                             # rowwise 但保 bf16(隔离转换开销)
            xq, xs = quant_fp8(x)
            return torch._scaled_mm(xq, wq_r.t(), scale_a=xs, scale_b=ws_rt, out_dtype=torch.bfloat16)

        def f_pt(x):                                   # per-tensor,直接出 fp16
            sa = (x.abs().max() / 448.0).float()
            xq = (x / sa).to(E4M3)
            return torch._scaled_mm(xq, wq_t.t(), scale_a=sa, scale_b=ws_t, out_dtype=torch.float16)

        for M in M_LIST:
            x = torch.randn(M, K, device="cuda", dtype=torch.float16)
            t16 = cuda_us(lambda: f_fp16(x))
            trc = cuda_us(lambda: f_row_cvt(x))
            trb = cuda_us(lambda: f_row_bf16(x))
            tpt = cuda_us(lambda: f_pt(x))
            print(f"{name:8} {M:>5} | {t16:7.1f} {trc:8.1f} {trb:9.1f} {tpt:10.1f} | "
                  f"{t16 / trc:4.2f}x {t16 / tpt:4.2f}x")


if __name__ == "__main__":
    main()
