"""P5 子任务2 收口:速度 × 质量 帕累托(fp16 / mixed / marlin / gptq)。

thesis 的核心交付物:不同精度配置不是"谁更好",而是落在一条**速度×质量帕累托**上,由
roofline ⊕ 数值敏感度决定。本脚本对每个配置 measure:

  - 速度:native 引擎端到端 tok/s(和 run_omniserve 同 workload,真实生产路径)。
  - 质量:**teacher-forcing** 下相对 fp16 的 greedy token 接受率 + perplexity。走 omniserve 原生
          Qwen2LLM.forward(cache=None 的全序列 SDPA),完全对齐生产模型;teacher-forcing 喂同一前缀,
          避免自回归分叉,纯测"单步预测对 fp16 的偏离"。

质量用**纯文本** prompt(decoder 量化与图像无关,纯文本激活已能代表 decoder 计算;图文 outlier
的影响见 探索日志 第18阶段边界)。

用法:
  conda activate vllm_bench
  HF_HUB_OFFLINE=1 python pareto.py --requests 16 --out pareto.json
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _workload import MAX_NEW_TOKENS, PROMPT, build, make_image  # noqa: E402
from omniserve import LLMEngine, Request, SamplingParams, SchedulerConfig  # noqa: E402
from omniserve.runners.qwen_vl_native import NativeQwenVLRunner  # noqa: E402

# 质量评估文本(真实中英,长度几百 token,覆盖不同主题)
QUALITY_TEXTS = [
    "Large language model inference has two distinct phases. The prefill phase processes the entire "
    "prompt in parallel and is compute-bound, while the decode phase generates one token at a time "
    "and is memory-bandwidth-bound because it reads the full weight matrix for very little compute.",
    "量化通过牺牲数值精度来换取显存带宽和算力。权重only的 int4(GPTQ/AWQ/Marlin)在 kernel 内把权重"
    "反量化成 fp16 再用 fp16 tensor core 计算,因此只省权重读取带宽、不省算力,适合带宽受限的 decode。",
    "FP8 W8A8 在 Ada 架构上有原生 tensor core,权重和激活都是 e4m3,能同时省带宽和算力,2 倍吞吐,"
    "但需要在线量化激活,固定开销在大 GEMM 上被摊薄,所以甜区是算力受限的 prefill 大矩阵乘。",
    "The roofline model says whether a kernel is bound by bandwidth or by compute depends on its "
    "arithmetic intensity, which for a GEMM is governed by the M dimension. Small M means few "
    "multiply-accumulates per byte read, so you are bandwidth-bound; large M flips you to compute-bound.",
]


def run_once(runner, reqs, max_running):
    """端到端跑一批请求,返回 (wall_s, 总输出 token 数)。"""
    engine = LLMEngine(runner, SchedulerConfig(
        max_running=max_running, max_prefill_per_step=max_running, max_prefill_tokens=16384))
    for r in reqs:
        engine.add_request(Request(
            prompt=PROMPT, images=[make_image(r.image_seed)],
            sampling=SamplingParams(max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)))
    t0 = time.perf_counter()
    while engine.has_unfinished():
        engine.step()
    wall = time.perf_counter() - t0
    toks = sum(s.num_output_tokens for s in engine.sequences())
    return wall, toks


@torch.no_grad()
def tf_logits(llm, ids):
    """teacher-forcing:一次全序列 prefill(cache=None 的 SDPA),返回 [L, vocab] logits。"""
    dev = "cuda"
    x = torch.tensor(ids, device=dev).view(1, -1)
    L = x.shape[1]
    emb = llm.embed_tokens(x)                                          # [1,L,H]
    pos = torch.arange(L, device=dev).view(1, 1, L).expand(3, 1, L)    # M-RoPE 纯文本三轴同
    mask = torch.triu(torch.full((L, L), float("-inf"), device=dev), 1).view(1, 1, L, L).half()
    return llm(emb, pos, mask, cache=None)[0]                          # [L, vocab]


def quality(llm, texts_ids, ref_argmax=None):
    """返回 (本配置各文本的 argmax 列表, 对 fp16 的 token 接受率, perplexity)。"""
    argmaxes, accs, nlls, ntok = [], [], 0.0, 0
    for t, ids in enumerate(texts_ids):
        lg = tf_logits(llm, ids)                                       # [L, vocab]
        am = lg.argmax(-1)                                             # [L]
        argmaxes.append(am)
        # next-token:位置 i 预测 ids[i+1];perplexity 用真实 next token 作 label
        lp = F.log_softmax(lg[:-1].float(), -1)
        tgt = torch.tensor(ids[1:], device=lg.device)
        nlls += -lp[torch.arange(len(tgt), device=lg.device), tgt].sum().item()
        ntok += len(tgt)
        if ref_argmax is not None:                                     # 对 fp16 argmax 的一致率
            r = ref_argmax[t]
            accs.append((am[:-1] == r[:-1]).float().mean().item())
    ppl = float(torch.exp(torch.tensor(nlls / ntok)))
    accept = (sum(accs) / len(accs) * 100) if ref_argmax is not None else 100.0
    return argmaxes, accept, ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--max-running", type=int, default=16)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--gptq-dir", type=str, default="/tmp/gptq_model")
    ap.add_argument("--out", type=str, default="pareto.json")
    args = ap.parse_args()

    configs = [(None, "fp16"), ("mixed", "mixed"), ("marlin", "marlin")]
    if glob.glob(os.path.join(args.gptq_dir, "*.safetensors")):
        configs.append(("gptq", "gptq"))
        print(f"[info] 检测到 GPTQ 权重 {args.gptq_dir},加入 gptq 点")

    reqs = build(args.requests)
    texts_ids, ref_argmax, rows = None, None, []

    for quant, name in configs:
        print(f"\n===== {name} =====")
        runner = NativeQwenVLRunner(max_running=args.max_running, quant=quant,
                                    gptq_dir=args.gptq_dir if quant == "gptq" else None)
        adapter = runner.adapter
        if texts_ids is None:                                         # 只 tokenize 一次(与配置无关)
            texts_ids = [adapter.encode_prompt(t, None) for t in QUALITY_TEXTS]

        # 质量(teacher-forcing)
        argmaxes, accept, ppl = quality(adapter._llm, texts_ids, ref_argmax)
        if name == "fp16":
            ref_argmax = argmaxes

        # 速度(端到端,warmup + trials 取中位)
        run_once(runner, reqs, args.max_running)                     # warmup(不计时)
        torch.cuda.reset_peak_memory_stats()
        speeds = []
        for _ in range(args.trials):
            wall, toks = run_once(runner, reqs, args.max_running)
            speeds.append(toks / wall)
        med = sorted(speeds)[len(speeds) // 2]
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

        rows.append({"config": name, "tok_per_s": round(med, 1), "accept_pct": round(accept, 2),
                     "quality_loss_pct": round(100 - accept, 2), "perplexity": round(ppl, 3),
                     "peak_mem_mib": round(peak)})
        print(f"[{name}] {med:.1f} tok/s | accept {accept:.2f}% | ppl {ppl:.3f} | "
              f"peak {peak:.0f} MiB")
        del runner, adapter
        gc.collect()
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print("\n=== 帕累托数据 ===")
    print(f"{'config':8} {'tok/s':>8} {'质量损失%':>9} {'ppl':>8} {'显存MiB':>8}")
    for r in rows:
        print(f"{r['config']:8} {r['tok_per_s']:>8} {r['quality_loss_pct']:>9} "
              f"{r['perplexity']:>8} {r['peak_mem_mib']:>8}")

    # 画帕累托图(横轴质量损失,纵轴吞吐)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for r in rows:
            ax.scatter(r["quality_loss_pct"], r["tok_per_s"], s=120)
            ax.annotate(r["config"], (r["quality_loss_pct"], r["tok_per_s"]),
                        textcoords="offset points", xytext=(8, 4))
        # matplotlib 默认无中文字体,label 用英文避免乱码
        ax.set_xlabel("Quality loss = 100 - token accept rate vs fp16 (%, lower=better)")
        ax.set_ylabel("End-to-end throughput (tok/s, higher=better)")
        ax.set_title("omniserve P5: per-op mixed-precision speed x quality Pareto\n"
                     "(Qwen2-VL-2B, RTX 4060 8GB)")
        ax.grid(True, alpha=0.3)
        png = os.path.splitext(args.out)[0] + ".png"
        fig.tight_layout(); fig.savefig(png, dpi=120)
        print(f"\n帕累托图已存 {png}")
    except Exception as e:
        print(f"[warn] 画图跳过:{e}")


if __name__ == "__main__":
    main()
