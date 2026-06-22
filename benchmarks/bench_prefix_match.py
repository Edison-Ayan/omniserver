"""最长前缀匹配:共享前缀 + 不同后缀,复用前缀 KV 只算后缀。A/B(关/开)+ 正确性。

现状 prefix cache 只整段精确匹配 → "同前缀异后缀"(同文档/系统提示 + 不同问题)零复用、每条
全量重算 prefill(已测命中率 0)。本次加 token 块级最长前缀匹配:命中共享前缀 → 恢复其 KV、
只 append-prefill 后缀(复用 chunked prefill 的 append 原语)。

流程:先 warm 一条(缓存共享前缀)→ 再发 K 条变体后缀请求,计时这批。
  off : match_prefix 关掉(模拟现状)→ 每条全量 prefill
  on  : 开 → 每条只算后缀
正确性:on 的首 token 必须和 off 逐条一致(复用前缀 KV 等价于重算)。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omniserve import LLMEngine, Request, SamplingParams, SchedulerConfig  # noqa: E402
from omniserve.runners.qwen_vl_native import NativeQwenVLRunner  # noqa: E402
from omniserve.cache.prefix import PrefixKVCache  # noqa: E402

SHARED_PREFIX = (
    "You are a careful assistant. Read the following passage and answer based only on it. "
    "Passage: Large language model inference has two phases. Prefill processes the whole prompt "
    "in parallel and is compute-bound. Decode generates one token at a time and is bandwidth-bound. "
    "Prefix caching reuses the KV of a shared prompt prefix so it is not recomputed. PagedAttention "
    "manages KV in fixed blocks to cut fragmentation. Continuous batching admits and retires "
    "sequences every model step. Question: "
)
SUFFIXES = [
    "What are the two phases of inference?", "Which phase is compute-bound and why?",
    "How does prefix caching save work?", "What problem does PagedAttention solve?",
    "Explain continuous batching in one line.", "Why is decode bandwidth-bound?",
    "What is reused on a prefix cache hit?", "Summarize the passage in five words.",
]
CFG = dict(max_prefill_per_step=8, max_prefill_tokens=1 << 20)


def measure(runner, pc, warm_prompt, tests, enable_partial):
    pc.clear()
    orig = pc.match_prefix
    if not enable_partial:
        pc.match_prefix = lambda *a, **k: None                  # 模拟现状:无部分匹配
    # warm:单独跑一条,把共享前缀写进 cache
    eng = LLMEngine(runner, SchedulerConfig(max_running=1, **CFG))
    eng.add_request(Request(prompt=warm_prompt, images=[],
                            sampling=SamplingParams(max_new_tokens=1, temperature=0.0)))
    while eng.has_unfinished():
        eng.step()
    h0 = pc.stats.hits
    # 计时:K 条变体后缀
    eng2 = LLMEngine(runner, SchedulerConfig(max_running=len(tests), **CFG))
    for p in tests:
        eng2.add_request(Request(prompt=p, images=[],
                                 sampling=SamplingParams(max_new_tokens=1, temperature=0.0)))
    torch.cuda.synchronize(); t0 = time.perf_counter()
    while eng2.has_unfinished():
        eng2.step()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1000.0
    toks = [s.output_token_ids[0] for s in eng2.sequences()]
    hit_rate = (pc.stats.hits - h0) / len(tests)
    pc.match_prefix = orig
    return wall, hit_rate, toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=7)
    args = ap.parse_args()
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  sm_{p.major}{p.minor}")
    pc = PrefixKVCache()
    runner = NativeQwenVLRunner(max_running=args.k, prefix_cache=pc, quant=None)
    enc = runner.adapter.encode_prompt
    warm = SHARED_PREFIX + SUFFIXES[0]
    tests = [SHARED_PREFIX + SUFFIXES[1 + i % (len(SUFFIXES) - 1)] for i in range(args.k)]
    pre_len, full_len = len(enc(SHARED_PREFIX, None)), len(enc(tests[0], None))
    print(f"K={args.k}  共享前缀 {pre_len} tok / 全长 ~{full_len} tok(前缀占 {pre_len/full_len:.0%})\n")

    for _ in range(2):                                          # warmup + timed
        w_off, hr_off, tok_off = measure(runner, pc, warm, tests, enable_partial=False)
        w_on, hr_on, tok_on = measure(runner, pc, warm, tests, enable_partial=True)
    print(f"{'':14}{'prefill wall(ms)':>17}{'前缀命中率':>11}")
    print(f"{'off(现状)':14}{w_off:>17.1f}{hr_off:>11.0%}")
    print(f"{'on(最长前缀)':14}{w_on:>17.1f}{hr_on:>11.0%}")
    print(f"\n后缀-only prefill 加速 = {w_off / w_on:.2f}x")
    ok = tok_on == tok_off
    print(f"首 token 一致性: {'✅ 逐条一致' if ok else '❌ 不一致'}  on={tok_on}  off={tok_off}")


if __name__ == "__main__":
    main()
