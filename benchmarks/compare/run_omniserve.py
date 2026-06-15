"""把共享 workload 跑在 omniserve 上,把指标写成 JSON。

和 run_vllm.py 对称:预热 + `--trials` 次计时(driver 取中位数),存样本输出。
用 fp16 以匹配 vLLM 的精度。

    python run_omniserve.py --requests 24 --trials 3 --fp16 --out omniserve.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _workload import MAX_NEW_TOKENS, PROMPT, build, make_image  # noqa: E402

from omniserve import LLMEngine, Request, SamplingParams, SchedulerConfig  # noqa: E402
from omniserve.runners.qwen_vl import QwenVLRunner  # noqa: E402


def run_once(runner, reqs, max_running):
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
    seqs = engine.sequences()
    toks = sum(s.num_output_tokens for s in seqs)
    texts = [runner.detokenize(s, s.output_token_ids) for s in seqs[:4]]
    return wall, toks, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=24)
    ap.add_argument("--reuse", type=float, default=0.0)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--vision-cache", action="store_true")
    ap.add_argument("--prefix-cache", action="store_true")
    ap.add_argument("--max-running", type=int, default=32)
    ap.add_argument("--native", action="store_true", help="use the zero-transformers native runner")
    ap.add_argument("--cuda-graph", action="store_true", help="decode 走 CUDA graph(消除 kernel 间隙,~1.2x decode);仅 --native")
    ap.add_argument("--quant", type=str, default=None, choices=["marlin", "gptq"],
                    help="可选 int4 量化扩展(降精度)。marlin=在线 RTN(只 MLP);gptq=加载 vLLM 同款 "
                         "GPTQ-Int4 校准权重(全 decoder,和 vLLM int4 同精度,用于公平对比)。"
                         "默认 fp16 对比不受影响。仅 --native 支持。")
    ap.add_argument("--out", type=str, default="omniserve.json")
    args = ap.parse_args()

    vc = pc = None
    if args.vision_cache:
        from omniserve.cache import VisionEmbeddingCache
        vc = VisionEmbeddingCache(max_entries=args.requests)
    if args.prefix_cache:
        from omniserve.cache import PrefixKVCache
        pc = PrefixKVCache(max_entries=args.requests)

    if args.native:
        from omniserve.runners.qwen_vl_native import NativeQwenVLRunner
        # 原生快路径支持 prefix cache(复用整段 prefill);vision cache 仅 HF runner 支持。
        if vc is not None:
            print("[warn] --vision-cache 在 --native 下不支持(原生 ViT 无 monkeypatch 挂点),已忽略")
        runner = NativeQwenVLRunner(max_running=args.max_running, prefix_cache=pc, quant=args.quant, use_graph=args.cuda_graph)
    else:
        if args.quant is not None:
            print("[warn] --quant 仅 --native 支持,已忽略")
        runner = QwenVLRunner(load_in_4bit=not args.fp16, vision_cache=vc, prefix_cache=pc)
    reqs = build(args.requests, reuse_rate=args.reuse)

    def reset_caches():
        # 每次计时都冷启动,这样 cache 只利用流内复用
        # (否则预热会把它喂满,复用率就失去意义了)。
        if vc is not None:
            vc.clear()
        if pc is not None:
            pc.clear()

    run_once(runner, reqs, args.max_running)  # warmup (not timed)

    torch.cuda.reset_peak_memory_stats()
    trials, sample_text, hit_rate = [], None, None
    for _ in range(args.trials):
        reset_caches()
        wall, toks, texts = run_once(runner, reqs, args.max_running)
        trials.append({"wall_s": wall, "tokens": toks,
                       "tok_per_s": toks / wall, "req_per_s": args.requests / wall})
        if sample_text is None:
            sample_text = texts
        cache_obj = pc or vc
        if cache_obj is not None:
            hit_rate = cache_obj.stats.hit_rate

    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    tag = "omniserve" + ("+vc" if args.vision_cache else "") + ("+pc" if args.prefix_cache else "")
    result = {
        "backend": tag, "n_requests": args.requests, "trials": trials,
        "peak_mem_mib_torch": round(peak), "sample_text": sample_text,
        "hit_rate": hit_rate,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    med = sorted(t["tok_per_s"] for t in trials)[len(trials) // 2]
    print(f"[{tag}] {args.requests} reqs x{args.trials} trials -> median {med:.1f} tok/s")


if __name__ == "__main__":
    main()
