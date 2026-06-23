"""能不能融合 kernel 降低 FP8 激活量化开销?探因表明:qkv 的 FP8 GEMM(scaled_mm)本身比 fp16
快,是激活量化的 ~25us 启动开销(eager:amax→div→cast 多发)把它拖成净亏。本脚本测两条融合:

  eager   : 现状 per-tensor 量化(x.abs().max() → x/s → .to(fp8),多次 launch)
  compile : torch.compile 把 amax+scale+cast 融成更少 kernel
  free    : 量化折进上游 RMSNorm 的理论上限(量化≈0,只算 scaled_mm)——产出激活的 norm 直接吐 fp8

对每个算子看:量化步本身的耗时(eager vs compile),以及 端到端 FP8(各量化方式)能否压过 fp16。
随机权重(速度只看形状),per-tensor FP8(和默认 FP8Linear 同档)。
"""

from __future__ import annotations

import os
import sys
from statistics import median

import torch

E4M3 = torch.float8_e4m3fn
SHAPES = {"qkv": (1536, 2048), "o": (1536, 1536), "gate_up": (1536, 17920), "down": (8960, 1536)}
M_LIST = [24, 64, 256]


def cuda_us(fn, iters=50, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        s.append(a.elapsed_time(b))
    return median(s) * 1000.0


def quant_eager(x):
    s = (x.abs().max() / 448.0).float()
    return (x / s).to(E4M3), s


quant_compiled = torch.compile(quant_eager, mode="max-autotune-no-cudagraphs")


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  sm_{p.major}{p.minor}  torch {torch.__version__}\n")
    print(f"{'GEMM':8} {'M':>4} | {'fp16':>7} {'量化eager':>9} {'量化compile':>11} | "
          f"{'FP8 eager':>9} {'FP8 compile':>11} {'FP8 free*':>9} | {'最优/fp16':>9}")
    for name, (K, N) in SHAPES.items():
        W = torch.randn(N, K, device="cuda", dtype=torch.float16)
        lin = torch.nn.Linear(K, N, bias=False).half().cuda(); lin.weight.data.copy_(W)
        ws = (W.abs().max() / 448.0).float(); wq = (W / ws).to(E4M3)
        for M in M_LIST:
            x = torch.randn(M, K, device="cuda", dtype=torch.float16)
            xq0, _ = quant_eager(x)

            def mm(xq, sa):
                return torch._scaled_mm(xq, wq.t(), scale_a=sa, scale_b=ws, out_dtype=torch.float16)

            t16 = cuda_us(lambda: lin(x))
            tqe = cuda_us(lambda: quant_eager(x))
            tqc = cuda_us(lambda: quant_compiled(x))
            t_eager = cuda_us(lambda: mm(*quant_eager(x)))
            t_comp = cuda_us(lambda: mm(*quant_compiled(x)))
            sa0 = (x.abs().max() / 448.0).float()
            t_free = cuda_us(lambda: mm(xq0, sa0))            # 量化≈0 的上限(scaled_mm only)
            best = min(t_eager, t_comp, t_free)
            print(f"{name:8} {M:>4} | {t16:7.1f} {tqe:9.1f} {tqc:11.1f} | "
                  f"{t_eager:9.1f} {t_comp:11.1f} {t_free:9.1f} | {t16 / best:8.2f}x")
    print("\n* free = 量化折进上游 RMSNorm 的理论上限(只剩 scaled_mm);>1 即该融合后 FP8 能赢 fp16")


if __name__ == "__main__":
    main()
