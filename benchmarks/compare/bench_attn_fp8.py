"""attention(qkv/o)也量化成 FP8 值不值?三档对比:fp16 / mixed(现状)/ attn_fp8(全 FP8)。

现状 DEFAULT_PLAN 把 qkv/o 留 fp16(micro-bench:qkv/o 在 decode M=24 下 FP8 只有 0.44–0.46x,
比 fp16 慢一倍;只有 prefill 大 M 才正收益)。本脚本端到端验证:把 qkv/o 也换 FP8 后,decode
主导的 gen-64 负载上 吞吐/质量/显存 到底怎么变。

  fp16     : 全 fp16(质量基准)
  mixed    : qkv/o=fp16 + gate_up/down=FP8(现状)
  attn_fp8 : 四个算子全 FP8(把 attention 也量化)

指标:decode 端到端 tok/s、对 fp16 的 token 接受率%、perplexity、峰值显存。
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
from omniserve.kernels.mixed_precision import apply_mixed, DEFAULT_PLAN  # noqa: E402

PLANS = {
    "fp16": None,
    "mixed": DEFAULT_PLAN,
    "attn_fp8": {"qkv": "fp8", "o": "fp8", "gate_up": "fp8", "down": "fp8"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=12)
    ap.add_argument("--max-running", type=int, default=12)
    args = ap.parse_args()

    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  sm_{p.major}{p.minor}  torch {torch.__version__}")
    reqs = build(args.requests)
    texts_ids, ref_argmax, rows = None, None, []

    for name, plan in PLANS.items():
        print(f"\n===== {name} =====")
        runner = NativeQwenVLRunner(max_running=args.max_running, prefix_cache=None, quant=None)
        if plan is not None:
            info = apply_mixed(runner.adapter._llm, plan)
            print(f"[{name}] per-op 精度×层数:{info}")
        adapter = runner.adapter
        if texts_ids is None:
            texts_ids = [adapter.encode_prompt(t, None) for t in QUALITY_TEXTS]

        argmaxes, accept, ppl = quality(adapter._llm, texts_ids, ref_argmax)
        if name == "fp16":
            ref_argmax = argmaxes                              # 留作基准(小 index 张量)
        else:
            del argmaxes

        run_once(runner, reqs, args.max_running)               # warmup(不计时)
        torch.cuda.reset_peak_memory_stats()
        wall, toks = run_once(runner, reqs, args.max_running)
        tok_s = toks / wall
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        rows.append((name, tok_s, accept, 100 - accept, ppl, peak))
        print(f"[{name}] {tok_s:.1f} tok/s | accept {accept:.2f}% | ppl {ppl:.3f} | "
              f"peak {peak:.0f} MiB")
        # 8GB 显存:建下一个模型前必须把这一个彻底释放(adapter 也握着 _llm 引用)
        del runner, adapter
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'config':10} {'tok/s':>8} {'质量损失%':>9} {'ppl':>8} {'显存MiB':>8}")
    for name, tok_s, accept, loss, ppl, peak in rows:
        print(f"{name:10} {tok_s:>8.1f} {loss:>9.2f} {ppl:>8.2f} {peak:>8.0f}")
    base = rows[0][1]
    print("\nvs fp16 吞吐:" + "  ".join(f"{r[0]} {r[1] / base:.2f}x" for r in rows))


if __name__ == "__main__":
    main()
