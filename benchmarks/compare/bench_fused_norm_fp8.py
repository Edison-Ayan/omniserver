"""融合 RMSNorm→FP8 对 gate_up 的端到端效果:把 gate_up 的激活量化折进 post_attention_layernorm,
消除那笔独立量化 launch。对比三档(decode 主导 gen 64 + 质量):

  fp16        : 全 fp16
  mixed       : gate_up/down=FP8(per-tensor,量化是独立 launch)+ qkv/o=fp16(现状)
  mixed+fuse  : gate_up=融合 norm→FP8(rowwise,量化折进 norm)+ down=FP8 + qkv/o=fp16

融合收益 = mixed+fuse 相对 mixed 的 decode 提升(只差 gate_up 那笔量化 launch 有没有融掉)。
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
from _workload import build                                    # noqa: E402
from pareto import run_once, quality, QUALITY_TEXTS            # noqa: E402
from omniserve.runners.qwen_vl_native import NativeQwenVLRunner  # noqa: E402
from omniserve.kernels.mixed_precision import apply_mixed, fuse_norm_fp8  # noqa: E402

# mixed+fuse 的 plan:gate_up 留给 fuse_norm_fp8 处理(这里设 fp16=跳过),down FP8,qkv/o fp16
FUSE_PLAN = {"qkv": "fp16", "o": "fp16", "gate_up": "fp16", "down": "fp8"}
MIXED_PLAN = {"qkv": "fp16", "o": "fp16", "gate_up": "fp8", "down": "fp8"}


def setup(name):
    runner = NativeQwenVLRunner(max_running=12, prefix_cache=None, quant=None)
    llm = runner.adapter._llm
    if name == "mixed":
        apply_mixed(llm, MIXED_PLAN)
    elif name == "mixed+fuse":
        apply_mixed(llm, FUSE_PLAN)        # down FP8
        fuse_norm_fp8(llm)                 # gate_up 融合 norm→FP8
    return runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=12)
    args = ap.parse_args()
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  sm_{p.major}{p.minor}")
    reqs = build(args.requests)
    texts_ids, ref, rows = None, None, []

    for name in ["fp16", "mixed", "mixed+fuse"]:
        print(f"\n===== {name} =====")
        runner = setup(name)
        adapter = runner.adapter
        if texts_ids is None:
            texts_ids = [adapter.encode_prompt(t, None) for t in QUALITY_TEXTS]
        argmaxes, accept, ppl = quality(adapter._llm, texts_ids, ref)
        if name == "fp16":
            ref = argmaxes
        run_once(runner, reqs, 12)                              # warmup
        torch.cuda.reset_peak_memory_stats()
        wall, toks = run_once(runner, reqs, 12)
        tok_s = toks / wall
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        rows.append((name, tok_s, 100 - accept, ppl, peak))
        print(f"[{name}] {tok_s:.1f} tok/s | 质量损失 {100 - accept:.2f}% | peak {peak:.0f} MiB")
        del runner, adapter
        gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'config':12} {'decode t/s':>11} {'质量损失%':>9} {'ppl':>8} {'显存MiB':>8}")
    for name, t, loss, ppl, peak in rows:
        print(f"{name:12} {t:>11.1f} {loss:>9.2f} {ppl:>8.2f} {peak:>8.0f}")
    m = next(r for r in rows if r[0] == "mixed")[1]
    f = next(r for r in rows if r[0] == "mixed+fuse")[1]
    print(f"\n融合收益(gate_up 量化折进 norm):mixed+fuse / mixed = {f / m:.3f}x decode")


if __name__ == "__main__":
    main()
