"""attention(qkv/o)量化策略对端到端的影响。四档 × 两种负载。

档位:
  fp16      : 全 fp16(质量基准)
  mixed     : qkv/o=fp16 + gate_up/down=FP8(现状,decode 吞吐最优)
  attn_fp8  : 四算子全 FP8(qkv/o 也 FP8)——省显存,但 decode 慢(小 GEMM FP8 摊不掉量化开销)
  attn_fp8a : qkv/o 按 M 自适应(prefill FP8 / decode fp16)+ gate_up/down=FP8 ——
              想兼得:decode 回 fp16 速度、prefill 拿 FP8 加速

两种负载(同一加载的模型各跑一次,看自适应在哪儿起作用):
  decode 主导:图文 + gen 64(prefill 占比小)→ 期望 attn_fp8a ≈ mixed、attn_fp8 略慢
  prefill 主导:长文本 + gen 1(全是 prefill)→ 期望 attn_fp8a / attn_fp8 明显 > mixed

指标:decode 吞吐、prefill 吞吐、对 fp16 的 token 接受率%、ppl、峰值显存。
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
from _workload import build                                    # noqa: E402
from pareto import run_once, quality, QUALITY_TEXTS            # noqa: E402
from omniserve import LLMEngine, Request, SamplingParams, SchedulerConfig  # noqa: E402
from omniserve.runners.qwen_vl_native import NativeQwenVLRunner  # noqa: E402
from omniserve.kernels.mixed_precision import apply_mixed, DEFAULT_PLAN  # noqa: E402

PLANS = {
    "fp16": None,
    "mixed": DEFAULT_PLAN,
    "attn_fp8": {"qkv": "fp8", "o": "fp8", "gate_up": "fp8", "down": "fp8"},
    "attn_fp8a": {"qkv": "fp8a", "o": "fp8a", "gate_up": "fp8", "down": "fp8"},
}

_TOPICS = [
    "Explain in depth how a modern GPU schedules and executes a large matrix multiplication, "
    "covering warps, tensor cores, shared memory and the roofline model. ",
    "Describe the full water cycle on Earth from ocean evaporation through cloud formation, "
    "precipitation, rivers and groundwater, and how climate change perturbs each stage. ",
    "Summarize the architecture of a transformer language model layer by layer, including "
    "attention, feed-forward, normalization, residuals and rotary position embeddings. ",
    "Compare TCP and UDP in detail: connection setup, reliability, ordering, congestion control, "
    "and which real-world protocols build on each and why. ",
]


def prefill_once(runner, prompts, max_running):
    """长文本 + gen=1:几乎全是 prefill,返回 (wall_s, 总 prompt token 数)。"""
    engine = LLMEngine(runner, SchedulerConfig(
        max_running=max_running, max_prefill_per_step=max_running, max_prefill_tokens=16384))
    for p in prompts:
        engine.add_request(Request(prompt=p, images=[],
                                   sampling=SamplingParams(max_new_tokens=1, temperature=0.0)))
    t0 = time.perf_counter()
    while engine.has_unfinished():
        engine.step()
    wall = time.perf_counter() - t0
    ntok = sum(s.num_prompt_tokens for s in engine.sequences())
    return wall, ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=12)
    ap.add_argument("--max-running", type=int, default=12)
    args = ap.parse_args()

    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  sm_{p.major}{p.minor}  torch {torch.__version__}")
    reqs = build(args.requests)
    long_prompts = [(_TOPICS[i % len(_TOPICS)]) * 6 for i in range(args.max_running)]
    texts_ids, ref_argmax, rows = None, None, []

    for name, plan in PLANS.items():
        print(f"\n===== {name} =====")
        runner = NativeQwenVLRunner(max_running=args.max_running, prefix_cache=None, quant=None)
        if plan is not None:
            apply_mixed(runner.adapter._llm, plan)
        adapter = runner.adapter
        if texts_ids is None:
            texts_ids = [adapter.encode_prompt(t, None) for t in QUALITY_TEXTS]
        argmaxes, accept, ppl = quality(adapter._llm, texts_ids, ref_argmax)
        if name == "fp16":
            ref_argmax = argmaxes
        else:
            del argmaxes

        # decode 主导(图文 gen 64)
        run_once(runner, reqs, args.max_running)
        torch.cuda.reset_peak_memory_stats()
        wall, toks = run_once(runner, reqs, args.max_running)
        dec_tok_s = toks / wall
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        # prefill 主导(长文本 gen 1)
        prefill_once(runner, long_prompts, args.max_running)
        wall_p, ntok_p = prefill_once(runner, long_prompts, args.max_running)
        pre_tok_s = ntok_p / wall_p

        rows.append((name, dec_tok_s, pre_tok_s, 100 - accept, ppl, peak))
        print(f"[{name}] decode {dec_tok_s:.1f} tok/s | prefill {pre_tok_s:.0f} tok/s | "
              f"质量损失 {100 - accept:.2f}% | peak {peak:.0f} MiB")
        del runner, adapter
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'config':10} {'decode t/s':>11} {'prefill t/s':>12} {'质量损失%':>9} {'ppl':>8} {'显存MiB':>8}")
    for name, d, pre, loss, ppl, peak in rows:
        print(f"{name:10} {d:>11.1f} {pre:>12.0f} {loss:>9.2f} {ppl:>8.2f} {peak:>8.0f}")
    dbase, pbase = rows[0][1], rows[0][2]
    print("\nvs fp16  decode: " + "  ".join(f"{r[0]} {r[1] / dbase:.2f}x" for r in rows))
    print("vs fp16  prefill:" + "  ".join(f"{r[0]} {r[2] / pbase:.2f}x" for r in rows))


if __name__ == "__main__":
    main()
