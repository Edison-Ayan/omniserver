"""真杠杆 1:量化 lm_head。profile 显示 lm_head(fp16,466MB 权重)占 decode ~25%,纯权重带宽,
且从没被量化。FP8 化把它的权重读减半 → decode 提速。但它和 embed_tokens 是 tied 的:

  解绑版(本脚本):lm_head 单独建 FP8 副本(+233MB),embed 保 fp16 → 换速度,显存略增。
  (省显存的共享版 —— embed+lm_head 共用一份 FP8 权重 —— 若解绑版赢了再做。)

lm_head 量化直接扰动 logits,所以质量(ppl / 对 fp16 接受率)是关键。三档:
  fp16 / mixed(现状)/ mixed+lmhead_fp8。
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
from omniserve.runners.qwen_vl_native import NativeQwenVLRunner  # noqa: E402
from omniserve.kernels.mixed_precision import apply_mixed      # noqa: E402
from omniserve.kernels.fp8_linear import FP8Linear             # noqa: E402

MIXED = {"qkv": "fp16", "o": "fp16", "gate_up": "fp8", "down": "fp8"}


def setup(name):
    runner = NativeQwenVLRunner(max_running=12, prefix_cache=None, quant=None)
    llm = runner.adapter._llm
    if name != "fp16":
        apply_mixed(llm, MIXED)
    if name == "mixed+lmhead_fp8":
        # 解绑 + FP8 lm_head(rowwise:per-vocab 权重 scale)。embed_tokens 保持 fp16 原权重。
        llm.lm_head = FP8Linear.from_linear(llm.lm_head, rowwise=True)
    return runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=12)
    args = ap.parse_args()
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  sm_{p.major}{p.minor}")
    reqs = build(args.requests)
    texts_ids, ref, rows = None, None, []

    for name in ["fp16", "mixed", "mixed+lmhead_fp8"]:
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
        print(f"[{name}] {tok_s:.1f} tok/s | 质量损失 {100 - accept:.2f}% | ppl {ppl:.2f} | peak {peak:.0f} MiB")
        del runner, adapter
        gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'config':18} {'decode t/s':>11} {'质量损失%':>9} {'ppl':>8} {'显存MiB':>8}")
    for name, t, loss, ppl, peak in rows:
        print(f"{name:18} {t:>11.1f} {loss:>9.2f} {ppl:>8.2f} {peak:>8.0f}")
    m = next(r for r in rows if r[0] == "mixed")
    lh = next(r for r in rows if r[0] == "mixed+lmhead_fp8")
    print(f"\nlm_head FP8 收益:decode {lh[1] / m[1]:.3f}x,显存 {lh[4] - m[4]:+.0f} MiB,"
          f"质量损失 {m[2]:.2f}%→{lh[2]:.2f}%")


if __name__ == "__main__":
    main()
