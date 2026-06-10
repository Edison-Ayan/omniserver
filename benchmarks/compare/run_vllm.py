"""Run the shared workload through vLLM and write metrics to JSON.

Runs in its own process (vLLM reserves most of the 8 GB, so it cannot coexist
with the omniserve process). vLLM does continuous batching internally: we submit
all requests at once via `llm.generate` and let it schedule them.

    python run_vllm.py --requests 32 --reuse 0.0 --out vllm.json
"""

from __future__ import annotations

import argparse
import time

import torch

from _workload import MAX_NEW_TOKENS, PROMPT, Metrics, build, make_image

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--reuse", type=float, default=0.0)
    ap.add_argument("--out", type=str, default="vllm.json")
    args = ap.parse_args()

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # Same chat-template text as omniserve uses, with the image placeholder.
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    reqs = build(args.requests, reuse_rate=args.reuse)
    inputs = [{"prompt": prompt_text,
               "multi_modal_data": {"image": make_image(r.image_seed)}} for r in reqs]

    torch.cuda.reset_peak_memory_stats()
    llm = LLM(model=MODEL_ID, max_model_len=2048, gpu_memory_utilization=0.85,
              limit_mm_per_prompt={"image": 1}, enforce_eager=True, dtype="float16")
    sp = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0)

    t0 = time.perf_counter()
    outs = llm.generate(inputs, sp)
    wall = time.perf_counter() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    m = Metrics("vLLM", args.requests, total_tokens, wall, peak)
    d = m.dump(args.out)
    print(f"[vLLM] {args.requests} reqs, {total_tokens} tok, {wall:.1f}s -> "
          f"{d['req_per_s']} req/s, {d['tok_per_s']} tok/s, peak {peak:.0f} MiB")


if __name__ == "__main__":
    main()
