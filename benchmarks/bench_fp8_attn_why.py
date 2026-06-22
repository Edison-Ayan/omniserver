"""探因:FP8 为什么加速不了 attention(qkv/o)的投影 GEMM?

假设:FP8 只加速**算力受限的大 GEMM**(吃 2x tensor core)。qkv/o 投影 N 小(1536/2048),
- decode(M 小):算术强度低 → **带宽受限**(读权重),FP8 的算力优势无处可用,反被激活量化 +
  scaled_mm 的固定开销拖慢;
- prefill(M 大):虽转算力受限,但 N 小 → 省下的绝对算力少,而固定开销和大算子一样 → 不划算。
对比 gate_up(N=17920)很快进入算力受限,FP8 全程赢。

两张表验证:
  表1 交叉点:扫 M,每个算子 FP8(per-tensor,出 fp16)相对 fp16 的加速比,看从哪个 M 开始 >1。
  表2 decode(M=24)开销拆解:激活量化 / scaled_mm / fp16 matmul —— 看固定开销吃掉多少。
随机权重(GEMM 速度只看形状)。per-tensor FP8(和修复后的 FP8Linear 同档)。
"""

from __future__ import annotations

import os
import sys
from statistics import median

import torch

E4M3 = torch.float8_e4m3fn
SHAPES = {"qkv": (1536, 2048), "o": (1536, 1536), "gate_up": (1536, 17920), "down": (8960, 1536)}
M_SWEEP = [8, 16, 24, 32, 64, 128, 256, 512, 1024, 2048]


def cuda_us(fn, iters=50, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        s.append(a.elapsed_time(b))
    return median(s) * 1000.0


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  sm_{p.major}{p.minor}\n")

    # ---- 表1:扫 M 找交叉点(FP8/fp16 加速比,>1 即 FP8 更快)----
    print("=== 表1 FP8(per-tensor)相对 fp16 加速比,扫 M(<1=更慢)===")
    print(f"{'M':>5} | " + " ".join(f"{n:>8}" for n in SHAPES) + "   ← 算术强度(MAC/byte, fp16)")
    cross = {n: None for n in SHAPES}
    for M in M_SWEEP:
        row, ai_row = [], []
        for name, (K, N) in SHAPES.items():
            W = torch.randn(N, K, device="cuda", dtype=torch.float16)
            lin = torch.nn.Linear(K, N, bias=False).half().cuda(); lin.weight.data.copy_(W)
            ws = (W.abs().max() / 448.0).float(); wq = (W / ws).to(E4M3)
            x = torch.randn(M, K, device="cuda", dtype=torch.float16)

            def f16(): return lin(x)

            def f8():
                sa = (x.abs().max() / 448.0).float(); xq = (x / sa).to(E4M3)
                return torch._scaled_mm(xq, wq.t(), scale_a=sa, scale_b=ws, out_dtype=torch.float16)
            r = cuda_us(f16) / cuda_us(f8)
            row.append(r)
            if r >= 1.0 and cross[name] is None:
                cross[name] = M
            # 算术强度:2MNK FLOP / (MK+KN+MN)*2 byte;小 M 时被权重 KN 主导 → 低 → 带宽受限
            ai_row.append(2 * M * N * K / ((M * K + K * N + M * N) * 2))
        ai = SHAPES["qkv"][1]
        print(f"{M:>5} | " + " ".join(f"{r:7.2f}x" for r in row) +
              "   AI:" + " ".join(f"{a:.0f}" for a in ai_row))
    print("交叉点(FP8 开始变快的 M):" +
          "  ".join(f"{n}={cross[n] or '>2048'}" for n in SHAPES))

    # ---- 表2:decode(M=24)开销拆解 ----
    print(f"\n=== 表2 decode M=24 开销拆解(us)===")
    print(f"{'GEMM':8} {'fp16':>8} {'act量化':>8} {'scaled_mm':>10} {'FP8合计':>9} {'FP8/fp16':>9}")
    M = 24
    for name, (K, N) in SHAPES.items():
        W = torch.randn(N, K, device="cuda", dtype=torch.float16)
        lin = torch.nn.Linear(K, N, bias=False).half().cuda(); lin.weight.data.copy_(W)
        ws = (W.abs().max() / 448.0).float(); wq = (W / ws).to(E4M3)
        x = torch.randn(M, K, device="cuda", dtype=torch.float16)
        t16 = cuda_us(lambda: lin(x))
        tq = cuda_us(lambda: (x / (x.abs().max() / 448.0)).to(E4M3))
        xq = (x / (x.abs().max() / 448.0)).to(E4M3); sa = (x.abs().max() / 448.0).float()
        tmm = cuda_us(lambda: torch._scaled_mm(xq, wq.t(), scale_a=sa, scale_b=ws, out_dtype=torch.float16))

        def f8():
            s = (x.abs().max() / 448.0).float(); q = (x / s).to(E4M3)
            return torch._scaled_mm(q, wq.t(), scale_a=s, scale_b=ws, out_dtype=torch.float16)
        t8 = cuda_us(f8)
        print(f"{name:8} {t16:8.1f} {tq:8.1f} {tmm:10.1f} {t8:9.1f} {t8/t16:8.2f}x")


if __name__ == "__main__":
    main()
