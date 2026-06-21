"""P4-下篇 A/B:长 prompt 插进正在跑的 decode,chunked prefill 削平 ITL 尖峰(stall)。

动机:omniserve 每步是「prefill 整段 → decode 各吐一个」。一个长 prompt 到达时,它那一步把
整段一次算完,正在流畅吐字的 decode 序列被迫干等 → inter-token latency(ITL)尖峰。chunked
prefill 把长 prompt 切成每步 ≤budget 个 token,峰值步耗时随之被削平(摊到多步)。

A/B(同一进程、都 warmup 过,排除首次 kernel 编译):
  - baseline : chunk 预算=∞,长 prompt 整段一次 prefill(一个大尖峰)。
  - chunked  : chunk 预算=budget,长 prompt 分块,decode 不被整段 prefill 堵。
报告:稳态 decode 步、注入窗口内的**峰值步**(= 最差 ITL)、峰值/稳态倍数、以及长 prompt
的输出 token 两者**是否逐位一致**(分块正确性)。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from statistics import median

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omniserve import LLMEngine, Request, SamplingParams, SchedulerConfig  # noqa: E402
from omniserve.runners.qwen_vl_native import NativeQwenVLRunner  # noqa: E402

SHORT = "Tell me a short story about a robot."
LONG_BASE = (
    "Carefully analyze the following long technical passage and keep it in context. "
    "Large language model serving systems must juggle prefill and decode under tight latency. "
)


def scenario(runner, n, long_prompt, gen, chunk, long_gen=8):
    """N 个短序列稳定 decode 时注入 1 个长 prompt,返回 (注入窗口每步耗时 ms, 长 prompt 输出 tokens)。"""
    budget = chunk if chunk > 0 else (1 << 20)
    engine = LLMEngine(runner, SchedulerConfig(
        max_running=n + 1, max_prefill_per_step=n + 1, max_prefill_tokens=budget))
    for _ in range(n):
        engine.add_request(Request(prompt=SHORT, images=[],
                                   sampling=SamplingParams(max_new_tokens=gen, temperature=0.0)))
    for _ in range(n + 3):                                   # warmup 到稳定 decode
        engine.step()
    long_req = Request(prompt=long_prompt, images=[],
                       sampling=SamplingParams(max_new_tokens=long_gen, temperature=0.0))
    engine.add_request(long_req)
    times, long_seq = [], None
    while engine.has_unfinished():
        torch.cuda.synchronize(); t0 = time.perf_counter()
        engine.step()
        torch.cuda.synchronize(); times.append((time.perf_counter() - t0) * 1000.0)
        if long_seq is None:
            long_seq = next((s for s in engine.sequences()
                             if s.request.request_id == long_req.request_id), None)
    return times, (long_seq.output_token_ids if long_seq else [])


def summarize(tag, times, n_steady_tail=6):
    steady = median(times[-n_steady_tail:])                  # 尾部纯 decode 步 = 稳态
    peak = max(times)                                        # 注入窗口最差步 = 最差 ITL
    print(f"{tag:10} 稳态 {steady:6.2f} ms | 峰值步 {peak:7.2f} ms | 峰值/稳态 {peak / steady:5.2f}x")
    return steady, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=6)
    ap.add_argument("--long-rep", type=int, default=20)
    ap.add_argument("--gen", type=int, default=80)
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()

    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  sm_{p.major}{p.minor}  torch {torch.__version__}")
    runner = NativeQwenVLRunner(max_running=args.n + 1, prefix_cache=None, quant=None)
    long_prompt = LONG_BASE * args.long_rep
    long_len = len(runner.adapter.encode_prompt(long_prompt, None))
    print(f"N={args.n} 短序列(各生成 {args.gen} tok),注入长 prompt {long_len} tok,"
          f"chunk={args.chunk}\n")

    # 各跑两次:第一次 warmup(含 chunk kernel 首次编译),第二次计时
    for warm in (True, False):
        t_base, tok_base = scenario(runner, args.n, long_prompt, args.gen, chunk=0)
        t_chunk, tok_chunk = scenario(runner, args.n, long_prompt, args.gen, chunk=args.chunk)
        if warm:
            continue
        print("=== 注入长 prompt 后的步耗时 ===")
        _, peak_b = summarize("baseline", t_base)
        _, peak_c = summarize("chunked", t_chunk)
        n_pre = sum(1 for t in t_chunk if t > median(t_chunk[-6:]) * 1.3)
        print(f"\n峰值削平 = baseline {peak_b:.1f}ms → chunked {peak_c:.1f}ms "
              f"= {peak_b / peak_c:.2f}x 降低(长 prompt 分成 ~{n_pre} 块摊开)")
        ok = tok_base == tok_chunk
        print(f"长 prompt 输出一致性: {'✅ 逐位一致' if ok else '❌ 不一致'}  "
              f"base={tok_base}  chunk={tok_chunk}")


if __name__ == "__main__":
    main()
