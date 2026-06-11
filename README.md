# omniserve

A lightweight **multimodal LLM inference engine** with continuous batching and a
vision-embedding cache, built to run a vision-language model (Qwen2-VL-2B) on a
**single 8 GB consumer GPU** (RTX 4060 Laptop). Benchmarked against vLLM.

> Goal: demonstrate the core serving-engine techniques — iteration-level
> scheduling, KV-cache management, and multimodal-specific optimizations — in a
> small, readable codebase rather than reproducing a production framework.

## Why
Multimodal serving has a cost that text-only serving does not: the vision
encoder (ViT) re-runs for every request, and identical images (multi-turn chat,
repeated system images) get re-encoded needlessly. On a memory-constrained GPU
this wastes both compute and the activation memory budget. omniserve caches
vision-tower outputs by image content so repeated images skip the encoder.

## Architecture
```
  HTTP (OpenAI-compatible)            server.py        ✓
        │
        ▼
  LLMEngine  ── step loop ──────────  engine.py        ✓
        │
        ▼
  Scheduler  ── continuous batching   scheduler.py     ✓
        │  (admit / decode / retire per iteration)
        ▼
  ModelRunner ─ QwenVLRunner          model_runner.py  (interface ✓, Qwen runner WIP)
        ├─ multimodal preprocessing
        ├─ KV-cache management
        └─ vision embedding cache  ◄── the differentiating optimization
```

## Status
- [x] Request / Sequence data model (`request.py`)
- [x] Continuous-batching scheduler (`scheduler.py`)
- [x] Engine step loop + streaming deltas (`engine.py`)
- [x] Backend interface + `StubRunner` (control plane tested without a GPU)
- [x] 4-bit model load validated on RTX 4060 (peak **2.1 GB** VRAM, see below)
- [x] Vision embedding cache + benchmark harness (real numbers below)
- [x] `QwenVLRunner`: real prefill/decode with per-sequence KV cache — engine
      output is **bit-identical to `model.generate()`**, and concurrent requests
      stay isolated (verified)
- [x] Batched decode: all sequences advance in one forward pass (left-padded KV
      + explicit M-RoPE positions) — token-identical to sequential, **~3x
      throughput** (see below)
- [x] **Preallocated KV pool**: write the new token in place instead of the
      O(L²) cat-based rebuild — kernels/step **15.2k → 3.1k**, **+15%** decode
      (token-identical), and the static shapes a CUDA graph needs
- [x] Rigorous benchmark vs. vLLM (matched fp16, CUDA graphs, median of 3 trials,
      verified-identical outputs) — omniserve reaches **0.34x** of vLLM
- [x] Content-addressed **vision embedding cache** wired into the engine
- [x] **Prefix KV cache** (reuse the whole prefill for repeated prefixes) —
      token-identical, **1.84x** throughput on reuse-heavy traffic (see below)
- [x] **Fused Triton RMSNorm** kernel for the LLM stack (1.8–8.5x vs HF in
      isolation, token-identical, ~1% end-to-end — Amdahl)
- [ ] CUDA-graph capture of the decode step (collapse the remaining ~3.1k
      per-layer kernel launches now that shapes are static)
- [ ] Batched/chunked prefill
- [x] OpenAI-compatible server (`/v1/chat/completions`, vision input + SSE
      streaming) on top of the continuous-batching engine

## Validated on RTX 4060 Laptop (8 GB)
Qwen2-VL-2B-Instruct, 4-bit NF4:

| metric | value |
|---|---|
| load time | 2.5 s |
| VRAM after load | 1453 MiB |
| **peak VRAM** | **2107 MiB** / 8188 |
| decode speed | 15.3 tok/s |

The ~6 GB of headroom is what makes concurrent continuous batching feasible on
this GPU.

## Result: omniserve vs vLLM
Identical workload through each backend, in separate processes (each needs most
of the 8 GB GPU).

**Methodology (for comparability):**
- Same model (Qwen2-VL-2B), same **fp16** precision, same 24 images, same prompt,
  same `max_new_tokens=64`, greedy (temp 0).
- vLLM runs in **production config — CUDA graphs enabled** (not `enforce_eager`).
- One warmup pass, then **3 timed trials; median reported** with min–max spread.
- **Output is verified, not assumed**: on the sampled prompts the two backends'
  generated text is **4/4 exact match** — so the throughput numbers compare the
  same work, not different amounts of it.

| backend | tok/s (median) | min–max | req/s | peak GPU MiB* |
|---|---|---|---|---|
| **vLLM 0.12** (CUDA graphs) | **503.5** | 503–504 | 8.09 | 7248* |
| **omniserve** (batched decode) | 172.5 | 172–174 | 2.78 | 5338 |

**omniserve reaches 0.34x of vLLM throughput (2.9x slower).**

\* vLLM's peak GPU is largely a **pre-reservation** (`gpu_memory_utilization=0.9`),
not a measured requirement — reported for completeness, **not** used to claim a
memory win.

**Where the gap comes from, and how batched decode shrank it.** With naive
per-sequence decode, omniserve's throughput is flat across load (~52 tok/s
whether 4 or 24 concurrent requests) — it cannot exploit concurrency. Batching
the decode step (one forward for all running sequences, left-padded KV +
explicit M-RoPE positions, token-identical to sequential) roughly **3x**'d
omniserve's throughput (≈53 → 172 tok/s). The remaining 2.9x to vLLM is paged
attention, fused/Triton kernels, and a tuned scheduler — none of which omniserve
implements. This is an honest, measured accounting of what a production engine
buys you on this workload.

Reproduce: `cd benchmarks/compare && python compare.py --requests 24 --trials 3`

## Result: caching for repeated multimodal content
Multimodal traffic repeats images (multi-turn chat, repeated system images). Two
ways to exploit that, measured in-engine at 24 concurrent requests, fp16, cold
cache per trial (so the number reflects within-stream reuse, not a primed cache):

| config | reuse 0.0 | reuse 0.9 (92% hit) | speedup from reuse |
|---|---|---|---|
| no cache | 176.9 | 178.5 tok/s | 1.01x |
| + vision cache (skip the ViT) | 175.5 | 255.4 tok/s | **1.46x** |
| + prefix KV cache (skip the whole prefill) | 175.5 | **322.9 tok/s** | **1.84x** |
| vLLM (for reference) | 507.8 | 1372.2 tok/s | 2.70x |

Takeaways, the honest version:
- **At 0% reuse all three are equal** — the caches add no overhead, they only help
  when content repeats.
- The **vision cache** (cache ViT outputs) helps (1.46x) but is limited: the ViT
  is only part of prefill. The **prefix KV cache** reuses the *entire* prefill —
  on an exact (prompt tokens + image content) match it clones the cached KV and
  skips the prefill forward — and wins (1.84x). This is the lever vLLM's prefix
  cache and SGLang's radix cache pull; vLLM scales 2.70x because it also has paged
  attention and fused kernels.
- A measurement note that bit us: an early version warmed up with the same
  workload, priming the cache so every timed request hit *regardless of reuse
  rate* — which made the vision cache look useless. Cold-starting each trial fixed
  it. (Measure, then doubt the measurement.)

Reproduce: `run_omniserve.py --reuse 0.9 --prefix-cache` (or `--vision-cache`).

## Result: vision embedding cache (standalone, batch=1 TTFT)
Sweeping the image reuse rate on Qwen2-VL-2B (4-bit), baseline vs. cached:

![vision cache benchmark](benchmarks/results.png)

| image reuse | TTFT (p50) | throughput | cache hit rate |
|---|---|---|---|
| 0.00 | 184 → 185 ms (≈0%) | +3% | 0.00 |
| 0.50 | 189 → 127 ms (**−33%**) | +8% | 0.50 |
| 0.75 | 188 → **66 ms** (**−65%**) | +13% | 0.75 |
| 1.00 | 190 → **67 ms** (**−65%**) | +21% | 0.94 |

Key points:
- **At 0% reuse the cache adds no measurable cost** (≈0% change) — it only helps
  when there is something to reuse, with no overhead otherwise.
- TTFT improvement grows with reuse and plateaus at ~65% once the vision encoder
  is fully elided from prefill.
- Cache hit rate tracks the reuse rate exactly, confirming correctness.

Reproduce:
```bash
cd benchmarks
HF_HUB_OFFLINE=1 python bench.py --sweep --requests 16 --out results.csv
python plot.py results.csv          # -> results.png
```

## Quickstart
```bash
conda activate vllm_bench        # torch 2.9 + transformers 4.57
# control-plane demo, no GPU/model needed:
python -c "from omniserve import LLMEngine, Request, StubRunner; \
  e=LLMEngine(StubRunner()); e.add_request(Request(prompt='hi')); print(e.run_until_complete())"
# model sanity check (requires the weights in the HF cache):
python scripts/check_model.py
```

## Serving (OpenAI-compatible)
```bash
python -m omniserve.server --port 8000 --prefix-cache   # add --fp16 for fp16
```
Then hit it like any OpenAI vision endpoint (base64 data-URI image):
```bash
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<...>"}},
    {"type": "text", "text": "What is in this image?"}]}],
  "max_tokens": 64, "stream": true }'
```
Streaming (`"stream": true`) returns SSE chunks; omitting it returns a single
`chat.completion`. Concurrent requests are continuously batched by the engine.

## Layout
```
omniserve/
├── omniserve/              # the engine package
│   ├── request.py          # Request / Sequence / SamplingParams data model
│   ├── scheduler.py        # continuous-batching scheduler
│   ├── engine.py           # step loop (schedule → prefill → decode → retire)
│   ├── runners/            # model execution backends
│   │   ├── base.py         # ModelRunner interface + StubRunner (no-GPU)
│   │   └── qwen_vl.py      # QwenVLRunner: real Qwen2-VL + KV cache + batched decode
│   ├── cache/              # multimodal caching components
│   │   ├── vision.py       # vision-embedding cache (skip the ViT for repeated images)
│   │   └── prefix.py       # prefix KV cache (skip the whole prefill for repeated prefixes)
│   ├── kernels/            # hand-written fused kernels
│   │   └── rmsnorm.py      # fused Triton RMSNorm for the LLM stack
│   └── server.py           # OpenAI-compatible HTTP server (vision + SSE)
├── benchmarks/             # vision-cache benchmark, profiler, and compare/ (vs vLLM)
└── scripts/                # standalone checks (check_model.py, batched-decode proto)
```
