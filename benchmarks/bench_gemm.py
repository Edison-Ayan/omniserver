"""GEMM micro-bench:per-op × per-M 的精度对比(P5 混合精度量化 子任务1)。

对 Qwen2-VL-2B decoder 的四个 GEMM 形状(qkv / o / gate_up / down),在不同 M
(24=decode 并发、512/1024=prefill prompt 长度)下,对比三档精度的延迟:

  - fp16  : 基线,F.linear,fp16 tensor core。
  - W4A16 : Marlin int4 权重 + fp16 激活(MarlinInt4Linear,在线 RTN)。kernel 内反量化
            权重→fp16 算 → 只省权重读取字节,不省算力。
  - FP8   : W8A8,权重+激活都 e4m3,torch._scaled_mm 在 sm_89 FP8 tensor core 上真算
            (算力 2x)。⚠️ 激活动态量化(absmax→scale→cast)计入计时,这是 W8A8 的
            固定开销,decode 上会吃掉收益、prefill 上被大 M 摊薄。

thesis 预测(待验证):
  - decode(M 小,带宽受限)→ W4A16 香(0.5B 权重 < FP8 1B),FP8 因激活量化开销吃亏。
  - prefill(M 大,算力受限)→ FP8 香(吃 2x 算力,激活量化开销被大 M 摊薄)。
  - 小 GEMM(qkv/o,固定开销主导)→ 两种量化都不划算,该保 fp16。

用法:
  conda activate vllm_bench
  python bench_gemm.py                 # 全部形状 × M
  python bench_gemm.py --check         # 额外打印对 fp16 的相对误差(质量维度)
"""

from __future__ import annotations

import argparse
import os
import sys
from statistics import median

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omniserve.kernels.marlin_linear import MarlinInt4Linear  # noqa: E402

# Qwen2-VL-2B decoder 的四个 GEMM:(名字, K=in_features, N=out_features)
SHAPES = [
    ("qkv",     1536,  2048),   # 小 N:固定开销主导
    ("o",       1536,  1536),   # 小:持平
    ("gate_up", 1536, 17920),   # 大 N(2*8960):量化甜区
    ("down",    8960,  1536),   # 大 K:量化甜区
]
M_LIST = [24, 512, 1024]        # 24=decode 并发;512/1024=prefill prompt
E4M3_MAX = 448.0                # float8_e4m3fn 动态范围上界


def cuda_time_us(fn, iters: int = 100, warmup: int = 20) -> float:
    """CUDA event 计时,返回中位延迟(微秒)。fn 内部应只含被测算子。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))  # ms
    return median(samples) * 1000.0


def to_fp8_per_tensor(t: torch.Tensor):
    """per-tensor 对称量化到 e4m3,返回 (fp8 张量, 反量化 scale)。"""
    amax = t.abs().max().clamp(min=1e-4)
    scale = (amax / E4M3_MAX).to(torch.float32)
    return (t / scale).to(torch.float8_e4m3fn), scale


def make_fp8_runner(weight: torch.Tensor):
    """weight:[N,K] fp16。权重离线量化(静态,不计时);返回含激活在线量化的 forward。"""
    # 权重 per-tensor 量化为 e4m3,转成 _scaled_mm 要的列主序 [K,N]
    w_fp8, w_scale = to_fp8_per_tensor(weight)        # [N,K]
    b = w_fp8.t()                                     # [K,N] 列主序(_scaled_mm 要求)

    def run(x_fp16: torch.Tensor):
        # 激活动态量化:这部分开销计入计时(W8A8 的固定成本)
        a_fp8, a_scale = to_fp8_per_tensor(x_fp16)    # [M,K]
        return torch._scaled_mm(a_fp8, b, scale_a=a_scale, scale_b=w_scale,
                                out_dtype=torch.float16)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="额外打印对 fp16 的相对误差")
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "需要 CUDA"
    dev = "cuda"
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  sm_{p.major}{p.minor}  torch {torch.__version__}\n")

    for name, K, N in SHAPES:
        # fp16 权重一次,三档共享同一份
        lin = torch.nn.Linear(K, N, bias=False).half().to(dev).eval()
        w = lin.weight.data                                   # [N,K]
        marlin = MarlinInt4Linear(lin)                        # W4A16
        fp8_run = make_fp8_runner(w)                          # W8A8

        print(f"=== {name:8s}  K={K:5d}  N={N:5d} ===")
        print(f"{'M':>5} | {'fp16(us)':>9} {'W4A16(us)':>10} {'FP8(us)':>9} | "
              f"{'W4A16↑':>7} {'FP8↑':>6}"
              + ("   W4A16_err  FP8_err" if args.check else ""))
        for M in M_LIST:
            x = torch.randn(M, K, device=dev, dtype=torch.float16)
            t_fp16 = cuda_time_us(lambda: F.linear(x, w), args.iters)
            t_w4 = cuda_time_us(lambda: marlin.forward(x), args.iters)
            t_fp8 = cuda_time_us(lambda: fp8_run(x), args.iters)
            line = (f"{M:>5} | {t_fp16:9.1f} {t_w4:10.1f} {t_fp8:9.1f} | "
                    f"{t_fp16/t_w4:6.2f}x {t_fp16/t_fp8:5.2f}x")
            if args.check:
                ref = F.linear(x, w).float()
                e_w4 = (marlin.forward(x).float() - ref).norm() / ref.norm()
                e_fp8 = (fp8_run(x).float() - ref).norm() / ref.norm()
                line += f"   {e_w4:9.4f}  {e_fp8:7.4f}"
            print(line)
        print()


if __name__ == "__main__":
    main()
