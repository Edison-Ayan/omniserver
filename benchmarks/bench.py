"""Benchmark: vision embedding cache vs. baseline on Qwen2-VL-2B (4-bit).

Sweeps the image reuse rate and reports, baseline vs cached:
  TTFT (p50/p90, ms), decode tok/s, end-to-end req/s, peak VRAM, cache hit rate.

Single point:   python bench.py --reuse 0.75 --requests 16
Full sweep:     python bench.py --sweep --requests 24 --out results.csv

Tuned for an 8 GB GPU: 4-bit NF4, batch size 1 (serving-style sequential).
Run with the weights already in the HF cache (set HF_HUB_OFFLINE=1 to be safe).
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for `omniserve`
from dataclasses import dataclass, field
from threading import Thread
from typing import List, Optional

import torch

from workload import Request, build_workload

from omniserve.cache import VisionEmbeddingCache

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


@dataclass
class RunResult:
    ttft_ms: List[float] = field(default_factory=list)
    decode_tok_s: List[float] = field(default_factory=list)
    peak_mem_mib: float = 0.0
    wall_s: float = 0.0
    n_requests: int = 0
    cache_stats: Optional[dict] = None

    def summary(self) -> dict:
        return {
            "ttft_p50_ms": round(statistics.median(self.ttft_ms), 1) if self.ttft_ms else 0.0,
            "ttft_p90_ms": round(_pct(self.ttft_ms, 90), 1),
            "decode_tok_s": round(statistics.mean(self.decode_tok_s), 1) if self.decode_tok_s else 0.0,
            "req_per_s": round(self.n_requests / self.wall_s, 3) if self.wall_s else 0.0,
            "peak_mem_mib": round(self.peak_mem_mib),
            "hit_rate": (self.cache_stats or {}).get("hit_rate", 0.0),
        }


def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def load_model():
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb, dtype=torch.float16, device_map="cuda"
    ).eval()
    return model, processor


@torch.inference_mode()
def run_stream(model, processor, requests: List[Request], max_new_tokens: int = 48) -> RunResult:
    from transformers import TextIteratorStreamer

    res = RunResult(n_requests=len(requests))
    torch.cuda.reset_peak_memory_stats()
    t_wall = time.perf_counter()

    for req in requests:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": req.prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[req.image], return_tensors="pt").to("cuda")

        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(**inputs, max_new_tokens=max_new_tokens, do_sample=False, streamer=streamer)
        thread = Thread(target=model.generate, kwargs=kwargs)

        t0 = time.perf_counter()
        thread.start()
        first_t, n_tok = None, 0
        for _ in streamer:
            now = time.perf_counter()
            if first_t is None:
                first_t = now
                res.ttft_ms.append((now - t0) * 1000)
            n_tok += 1
        thread.join()
        t_end = time.perf_counter()
        if first_t is not None and n_tok > 1:
            res.decode_tok_s.append((n_tok - 1) / max(t_end - first_t, 1e-6))

    res.wall_s = time.perf_counter() - t_wall
    res.peak_mem_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return res


def run_point(model, processor, reuse: float, n: int, use_cache: bool) -> RunResult:
    requests = build_workload(n_requests=n, reuse_rate=reuse)
    cache = VisionEmbeddingCache(max_entries=n).install(model) if use_cache else None
    try:
        res = run_stream(model, processor, requests)
    finally:
        if cache is not None:
            res.cache_stats = cache.stats.as_dict()
            cache.uninstall()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", type=float, default=0.75)
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", type=str, default="results.csv")
    args = ap.parse_args()

    print(f"loading {MODEL_ID} (4-bit) ...")
    model, processor = load_model()
    grid = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0] if args.sweep else [args.reuse]

    rows = []
    for reuse in grid:
        base = run_point(model, processor, reuse, args.requests, use_cache=False).summary()
        cached = run_point(model, processor, reuse, args.requests, use_cache=True).summary()
        row = {"reuse": reuse}
        row.update({f"base_{k}": v for k, v in base.items()})
        row.update({f"cache_{k}": v for k, v in cached.items()})
        rows.append(row)
        ttft_gain = 100 * (1 - cached["ttft_p50_ms"] / base["ttft_p50_ms"]) if base["ttft_p50_ms"] else 0
        thr_gain = 100 * (cached["req_per_s"] / base["req_per_s"] - 1) if base["req_per_s"] else 0
        print(f"reuse={reuse:>4} | hit={cached['hit_rate']:.2f} | "
              f"TTFT {base['ttft_p50_ms']:.0f}->{cached['ttft_p50_ms']:.0f}ms ({ttft_gain:+.0f}%) | "
              f"req/s {base['req_per_s']:.2f}->{cached['req_per_s']:.2f} ({thr_gain:+.0f}%) | "
              f"peakMiB {base['peak_mem_mib']}/{cached['peak_mem_mib']}")

    if args.sweep:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
