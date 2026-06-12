"""Micro-benchmark:融合 Triton RMSNorm vs transformers 的 Qwen2RMSNorm。

报告引擎里会出现的几种 shape(decode batch、prefill 序列、大 prefill)下的最大数值
误差和加速比。RMSNorm 访存受限,所以单趟融合 kernel 单独看赢很多——但端到端的诚实
数字看 README(它只占总时间的一小块)。

    python bench_rmsnorm.py
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omniserve.kernels.rmsnorm import rmsnorm  # noqa: E402

N = 1536  # Qwen2-VL-2B language-model hidden size
EPS = 1e-6


def _bench(fn, iters=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e6  # microseconds


def main():
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2RMSNorm

    torch.manual_seed(0)
    ref = Qwen2RMSNorm(N, EPS).cuda().half()
    ref.weight.data.normal_(1.0, 0.02)
    w = ref.weight.data

    print(f"RMSNorm N={N}, fp16")
    print(f"{'rows':>8}{'max err':>12}{'HF (us)':>10}{'Triton (us)':>13}{'speedup':>9}")
    for m in (16, 256, 1536, 24 * 350):
        x = torch.randn(m, N, device="cuda", dtype=torch.float16) * 2
        err = (ref(x).float() - rmsnorm(x, w, EPS).float()).abs().max().item()
        t_ref = _bench(lambda: ref(x))
        t_mine = _bench(lambda: rmsnorm(x, w, EPS))
        print(f"{m:>8}{err:>12.2e}{t_ref:>10.1f}{t_mine:>13.1f}{t_ref / t_mine:>8.2f}x")


if __name__ == "__main__":
    main()
