"""P4 隔离 A/B:打包 varlen prefill vs 逐序列 prefill(纯文本,绕开 ViT 干净测 LLM prefill)。

背景:打包 prefill 的整套机器(调度预算 + PackedPrefillCache + varlen flash)已实现并接进
NativeQwenVLRunner。但从没单独量过"打包 varlen 比逐序列快多少"——收益一直糊在端到端数字里,
待解决问题.md 还停在"块对角 SDPA O(total²) 实测 0.97x、就差 varlen"。本脚本补这一刀。

隔离手段:
  - **纯文本请求**(images=[]),preprocess 返回 None → 不走 ViT,两边都没有 ViT 开销,
    测到的差异纯粹来自 LLM prefill 的 GEMM 批处理 + varlen 注意力 + 每步调度开销。
  - **max_new_tokens=1**:每个序列 prefill 出首 token 即结束,wall ≈ 纯 prefill 时间,decode 不掺。

两个配置(同一批 K 个 prompt):
  - packed     : max_prefill_per_step=K → 一步一次 varlen 前向(sum L tokens 打成一个 batch)。
  - sequential : max_prefill_per_step=1 → K 步,每步一个 prompt 单独前向。

正确性:两边首 token 必须逐序列一致(greedy),否则打包路径有 bug。
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

# 内容各异的 prompt 主题(不同首 token),再各自重复到不同长度,让段长参差。
# 内容各异是关键:若 varlen 段间隔离有 bug(跨段注意力),packed 的首 token 会偏离 sequential。
TOPICS = [
    "Explain how a GPU executes a matrix multiplication kernel and why",
    "Describe the water cycle on Earth, starting from ocean evaporation and",
    "Summarize the plot of a detective novel set in nineteenth century Paris where",
    "List the main differences between TCP and UDP transport protocols and when",
    "Write about the history of coffee cultivation from Ethiopia to the rest of",
    "Discuss why the sky appears blue during the day and turns red at sunset because",
    "Outline the steps a compiler takes to turn source code into machine code and",
    "Compare the climate of a tropical rainforest with that of an arctic tundra and",
]


def make_prompts(k: int):
    """K 个内容各异、长度参差的纯文本 prompt(不同主题 × 重复 1..4 次造段长差异)。"""
    return [(TOPICS[i % len(TOPICS)] + " ") * (1 + (i % 4)) for i in range(k)]


def run(runner, prompts, max_prefill_per_step, trials=3):
    """跑一批 prompt 到首 token,返回 (prefill wall 中位数 ms, 首 token 列表)。"""
    walls, first_tokens = [], None
    for t in range(trials + 1):                      # 第 0 次 warmup 不计时
        engine = LLMEngine(runner, SchedulerConfig(
            max_running=len(prompts), max_prefill_per_step=max_prefill_per_step,
            max_prefill_tokens=1 << 20))
        for p in prompts:
            engine.add_request(Request(
                prompt=p, images=[],
                sampling=SamplingParams(max_new_tokens=1, temperature=0.0)))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while engine.has_unfinished():
            engine.step()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1000.0
        seqs = engine.sequences()
        toks = [s.output_token_ids[0] for s in seqs]
        if t > 0:
            walls.append(wall)
            first_tokens = toks
    return median(walls), first_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=8, help="一批并发到达的 prompt 数")
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  sm_{p.major}{p.minor}  torch {torch.__version__}")
    runner = NativeQwenVLRunner(max_running=args.k, prefix_cache=None, quant=None)
    prompts = make_prompts(args.k)
    lens = [len(runner.adapter.encode_prompt(p, None)) for p in prompts]
    print(f"K={args.k} 纯文本 prompt,段长={lens},total={sum(lens)} tokens\n")

    w_pack, tok_pack = run(runner, prompts, args.k, args.trials)
    w_seq, tok_seq = run(runner, prompts, 1, args.trials)

    print(f"{'配置':12} {'prefill wall(ms)':>18}")
    print(f"{'sequential':12} {w_seq:>18.2f}")
    print(f"{'packed':12} {w_pack:>18.2f}")
    print(f"\n打包加速比 = {w_seq / w_pack:.2f}x  (>1 即打包 varlen prefill 更快)")
    ok = tok_pack == tok_seq
    print(f"首 token 一致性: {'✅ 一致' if ok else '❌ 不一致'}  packed={tok_pack}  seq={tok_seq}")


if __name__ == "__main__":
    main()
