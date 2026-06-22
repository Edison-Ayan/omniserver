"""图文最长前缀匹配:同一张图 + 不同问题,复用图像 + 共享文本前缀的 KV,只算问题后缀。

这是 vision serving 最高频的复用模式(同图多问、同文档多问)。图像 token 占 prompt 大头且 KV 重,
现状精确匹配下"同图不同问"零复用、每条把整张图的 KV 重算一遍。本次让块哈希把图像内容 hash 折进去
(同图 → 同前缀哈希),命中共享前缀 → 恢复其 KV(含整张图)、只 append-prefill 文本问题后缀。

流程:warm 一条(缓存 图+前缀)→ 再发 K 条同图不同问。off(关 match)vs on(开)+ 首 token 一致性。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
from _workload import make_image                                  # noqa: E402
from omniserve import LLMEngine, Request, SamplingParams, SchedulerConfig  # noqa: E402
from omniserve.runners.qwen_vl_native import NativeQwenVLRunner  # noqa: E402
from omniserve.cache.prefix import PrefixKVCache  # noqa: E402

PREFIX = "Look at the image carefully and answer the question. "    # 共享文本前缀(图后)
QUESTIONS = [
    "What shapes are present?", "How many objects are there?",
    "What colors do you see?", "Describe the layout.",
    "Is there a circle?", "What is in the top-left?",
    "Which shape is largest?", "Summarize the image.",
]
CFG = dict(max_prefill_tokens=1 << 20)


def measure(runner, pc, img, warm_prompt, tests, enable_partial):
    pc.clear()
    orig = pc.match_prefix
    if not enable_partial:
        pc.match_prefix = lambda *a, **k: None
    eng = LLMEngine(runner, SchedulerConfig(max_running=1, max_prefill_per_step=1, **CFG))
    eng.add_request(Request(prompt=warm_prompt, images=[img],
                            sampling=SamplingParams(max_new_tokens=1, temperature=0.0)))
    while eng.has_unfinished():
        eng.step()
    h0 = pc.stats.hits
    eng2 = LLMEngine(runner, SchedulerConfig(max_running=len(tests), max_prefill_per_step=len(tests), **CFG))
    for p in tests:
        eng2.add_request(Request(prompt=p, images=[img],
                                 sampling=SamplingParams(max_new_tokens=1, temperature=0.0)))
    torch.cuda.synchronize(); t0 = time.perf_counter()
    while eng2.has_unfinished():
        eng2.step()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1000.0
    toks = [s.output_token_ids[0] for s in eng2.sequences()]
    hr = (pc.stats.hits - h0) / len(tests)
    pc.match_prefix = orig
    return wall, hr, toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=7)
    args = ap.parse_args()
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  sm_{p.major}{p.minor}")
    pc = PrefixKVCache()
    runner = NativeQwenVLRunner(max_running=args.k, prefix_cache=pc, quant=None)
    img = make_image(0)                                          # 同一张图
    warm = PREFIX + QUESTIONS[0]
    tests = [PREFIX + QUESTIONS[1 + i % (len(QUESTIONS) - 1)] for i in range(args.k)]
    full_len = len(runner.adapter.encode_prompt(
        PREFIX + QUESTIONS[1], runner.adapter.preprocess([img])[1]))
    print(f"K={args.k} 条「同图 + 不同问」,每条 prompt(含图像 token)~{full_len} tok\n")

    for _ in range(2):                                           # warmup + timed
        w_off, hr_off, t_off = measure(runner, pc, img, warm, tests, False)
        w_on, hr_on, t_on = measure(runner, pc, img, warm, tests, True)
    print(f"{'':14}{'prefill wall(ms)':>17}{'前缀命中率':>11}")
    print(f"{'off(现状)':14}{w_off:>17.1f}{hr_off:>11.0%}")
    print(f"{'on(图文前缀)':14}{w_on:>17.1f}{hr_on:>11.0%}")
    print(f"\n复用图 KV、只算问题后缀 → prefill {w_off / w_on:.2f}x")
    print(f"首 token 一致性: {'✅ 逐条一致' if t_on == t_off else '❌ 不一致'}\n  on ={t_on}\n  off={t_off}")


if __name__ == "__main__":
    main()
