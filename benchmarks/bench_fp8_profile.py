"""FP8 还有哪里能优化?对真实 mixed-FP8 decode 做 torch.profiler,把 CUDA 时间按类归并:
  量化税  : abs/max/div/copy/cast/to(FP8 激活量化的那串 eager op)
  scaled_mm: FP8 GEMM(gate_up/down)
  fp16 mm : qkv/o 的 fp16 GEMM
  attn    : flash attention
  其它    : norm / silu / rope / embed / sample 等

回答:decode 时间花在哪、FP8 量化税占多少(可压上限)、瓶颈是不是在能动的地方。
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omniserve import LLMEngine, Request, SamplingParams, SchedulerConfig  # noqa: E402
from omniserve.runners.qwen_vl_native import NativeQwenVLRunner  # noqa: E402
from omniserve.kernels.mixed_precision import apply_mixed  # noqa: E402

MIXED = {"qkv": "fp16", "o": "fp16", "gate_up": "fp8", "down": "fp8"}


def categorize(name: str) -> str:
    n = name.lower()
    if "scaled_mm" in n or "f8f8" in n or "fp8" in n or "rowwise" in n or "scaled" in n:
        return "scaled_mm(FP8 GEMM)"
    if any(k in n for k in ["reduce", "amax", "abs", "max", "div", "copy", "cast",
                            "convert", "_to_copy", "elementwise"]):
        return "量化税(amax/div/cast)"
    if any(k in n for k in ["flash", "attention", "attn"]):
        return "attention"
    if any(k in n for k in ["gemm", "cutlass", "ampere", "sm80", "sm89", "matmul",
                            "cublas", "gemv", "16x8"]):
        return "fp16 GEMM(qkv/o/lm_head)"
    if any(k in n for k in ["norm", "silu", "rope", "rms", "triton", "embed", "index",
                            "softmax", "argmax", "cat", "rotary"]):
        return "norm/silu/rope/其它"
    return "其它"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("-n", type=int, default=12)
    args = ap.parse_args()
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  sm_{p.major}{p.minor}  torch {torch.__version__}")

    runner = NativeQwenVLRunner(max_running=args.n, prefix_cache=None, quant=None)
    apply_mixed(runner.adapter._llm, MIXED)
    eng = LLMEngine(runner, SchedulerConfig(max_running=args.n, max_prefill_per_step=args.n,
                                            max_prefill_tokens=1 << 20))
    for _ in range(args.n):
        eng.add_request(Request(prompt="Explain LLM inference in detail.", images=[],
                                sampling=SamplingParams(max_new_tokens=args.steps + 20, temperature=0.0)))
    for _ in range(args.n + 5):                                # 过 prefill,进稳定 decode + warmup
        eng.step()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(args.steps):
            eng.step()
    torch.cuda.synchronize()

    evs = [(e.key, e.self_device_time_total / 1000.0) for e in prof.key_averages()
           if e.self_device_time_total > 0]
    total = sum(t for _, t in evs)
    print(f"\n=== {args.steps} decode 步 原始 top kernel(共 {total:.1f} ms,batch={args.n})===")
    for name, t in sorted(evs, key=lambda kv: -kv[1])[:16]:
        print(f"  {t:7.1f} ms  {t / total * 100:5.1f}%  {name[:72]}")


if __name__ == "__main__":
    main()
