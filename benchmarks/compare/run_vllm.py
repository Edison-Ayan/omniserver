"""把共享 workload 跑在 vLLM 上,把指标写成 JSON。

接近生产的配置:开 CUDA graph(不是 enforce_eager),fp16。先跑一次预热,再跑
`--trials` 次计时,报告每个 trial 的吞吐,好让 driver 取中位数。存几个样本输出,
这样能验证各后端确实产生可比的文本(不只是可比的 token 数量)。

    python run_vllm.py --requests 24 --trials 3 --out vllm.json
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from _workload import MAX_NEW_TOKENS, PROMPT, build, make_image

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=24)
    ap.add_argument("--reuse", type=float, default=0.0)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--mem-util", type=float, default=0.9)  # fp16 权重 ~4.4GB;要给 KV 留余量
    ap.add_argument("--out", type=str, default="vllm.json")
    args = ap.parse_args()

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    reqs = build(args.requests, reuse_rate=args.reuse)
    inputs = [{"prompt": prompt_text,
               "multi_modal_data": {"image": make_image(r.image_seed)}} for r in reqs]

    # 生产配置:CUDA graph 开(不用 enforce_eager)。mem-util 决定 KV 预留;
    # 我们设得适中,并把它当作**预留**报告,而不是实测工作集(见 README 说明)。
    llm = LLM(model=MODEL_ID, max_model_len=2048, gpu_memory_utilization=args.mem_util,
              limit_mm_per_prompt={"image": 1}, dtype="float16")
    sp = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0)

    # 预热(编译 CUDA graph、填好 cache)—— 不计时
    llm.generate(inputs, sp)

    torch.cuda.reset_peak_memory_stats()
    trials = []
    sample_text = None
    for _ in range(args.trials):
        t0 = time.perf_counter()
        outs = llm.generate(inputs, sp)
        wall = time.perf_counter() - t0
        toks = sum(len(o.outputs[0].token_ids) for o in outs)
        trials.append({"wall_s": wall, "tokens": toks,
                       "tok_per_s": toks / wall, "req_per_s": args.requests / wall})
        if sample_text is None:
            sample_text = [o.outputs[0].text for o in outs[:4]]

    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    result = {
        "backend": "vLLM", "n_requests": args.requests, "trials": trials,
        "peak_mem_mib_torch": round(peak), "mem_util": args.mem_util,
        "sample_text": sample_text,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    med = sorted(t["tok_per_s"] for t in trials)[len(trials) // 2]
    print(f"[vLLM] {args.requests} reqs x{args.trials} trials -> median {med:.1f} tok/s")


if __name__ == "__main__":
    main()
