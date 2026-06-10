"""Run the shared workload through omniserve and write metrics to JSON.

Submits all requests to the engine and drives the step loop to completion, so
the scheduler's continuous batching is exercised the same way a server would.

    python run_omniserve.py --requests 32 --reuse 0.0 --fp16 --out omniserve.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# import the omniserve package (two dirs up: benchmarks/compare -> repo root)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _workload import MAX_NEW_TOKENS, PROMPT, Metrics, build, make_image  # noqa: E402

from omniserve import LLMEngine, Request, SamplingParams, SchedulerConfig  # noqa: E402
from omniserve.qwen_runner import QwenVLRunner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--reuse", type=float, default=0.0)
    ap.add_argument("--fp16", action="store_true", help="fp16 instead of 4-bit (fair vs vLLM)")
    ap.add_argument("--vision-cache", action="store_true")
    ap.add_argument("--max-running", type=int, default=8)
    ap.add_argument("--out", type=str, default="omniserve.json")
    args = ap.parse_args()

    vc = None
    if args.vision_cache:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmarks/
        from vision_cache import VisionEmbeddingCache  # type: ignore
        vc = VisionEmbeddingCache(max_entries=args.requests)

    torch.cuda.reset_peak_memory_stats()
    runner = QwenVLRunner(load_in_4bit=not args.fp16, vision_cache=vc)
    engine = LLMEngine(runner, SchedulerConfig(max_running=args.max_running,
                                               max_prefill_per_step=1))

    reqs = build(args.requests, reuse_rate=args.reuse)
    for r in reqs:
        engine.add_request(Request(
            prompt=PROMPT, images=[make_image(r.image_seed)],
            sampling=SamplingParams(max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)))

    t0 = time.perf_counter()
    while engine.has_unfinished():
        engine.step()
    wall = time.perf_counter() - t0
    total_tokens = sum(s.num_output_tokens for s in engine.sequences())

    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    tag = "omniserve" + ("+vc" if args.vision_cache else "")
    m = Metrics(tag, args.requests, total_tokens, wall, peak)
    d = m.dump(args.out)
    print(f"[{tag}] {args.requests} reqs, {total_tokens} tok, {wall:.1f}s -> "
          f"{d['req_per_s']} req/s, {d['tok_per_s']} tok/s, peak {peak:.0f} MiB")


if __name__ == "__main__":
    main()
