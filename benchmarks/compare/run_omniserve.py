"""Run the shared workload through omniserve and write metrics to JSON.

Mirrors run_vllm.py: warmup pass + `--trials` timed passes (median reported by
the driver), saves sample outputs. fp16 to match vLLM precision.

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
    engine = LLMEngine(runner, SchedulerConfig(max_running=max_running, max_prefill_per_step=1))
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
    ap.add_argument("--out", type=str, default="omniserve.json")
    args = ap.parse_args()

    vc = pc = None
    if args.vision_cache:
        from omniserve.cache import VisionEmbeddingCache
        vc = VisionEmbeddingCache(max_entries=args.requests)
    if args.prefix_cache:
        from omniserve.cache import PrefixKVCache
        pc = PrefixKVCache(max_entries=args.requests)

    runner = QwenVLRunner(load_in_4bit=not args.fp16, vision_cache=vc, prefix_cache=pc)
    reqs = build(args.requests, reuse_rate=args.reuse)

    run_once(runner, reqs, args.max_running)  # warmup (not timed)

    torch.cuda.reset_peak_memory_stats()
    trials, sample_text = [], None
    for _ in range(args.trials):
        wall, toks, texts = run_once(runner, reqs, args.max_running)
        trials.append({"wall_s": wall, "tokens": toks,
                       "tok_per_s": toks / wall, "req_per_s": args.requests / wall})
        if sample_text is None:
            sample_text = texts

    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    tag = "omniserve" + ("+vc" if args.vision_cache else "") + ("+pc" if args.prefix_cache else "")
    result = {
        "backend": tag, "n_requests": args.requests, "trials": trials,
        "peak_mem_mib_torch": round(peak), "sample_text": sample_text,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    med = sorted(t["tok_per_s"] for t in trials)[len(trials) // 2]
    print(f"[{tag}] {args.requests} reqs x{args.trials} trials -> median {med:.1f} tok/s")


if __name__ == "__main__":
    main()
