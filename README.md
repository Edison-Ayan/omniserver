# omniserve

A lightweight **multimodal LLM inference engine** with continuous batching and a
vision-embedding cache, built to run a vision-language model (Qwen2-VL-2B) on a
**single 8 GB consumer GPU** (RTX 4060 Laptop). Benchmarked against vLLM.

> Goal: demonstrate the core serving-engine techniques — iteration-level
> scheduling, KV-cache management, and multimodal-specific optimizations — in a
> small, readable codebase rather than reproducing a production framework.

🧩 **Fully from-scratch, zero `transformers` dependency.** The model forward
(Qwen2-VL ViT + LLM), M-RoPE positions, image preprocessing and the tokenizer
are all reimplemented and verified token-/bit-identical to Hugging Face. The only
third-party pieces are PyTorch (compute), `tokenizers` (BPE), `safetensors`
(weights) and PIL/numpy — the same split vLLM/SGLang use.

📈 **[OPTIMIZATION.md](OPTIMIZATION.md)** — the profile-first journey from 0.18x
to 0.44x of vLLM, what was deliberately *not* done (and why), and where the
remaining gap is.

🚀 **[运行流程.md](运行流程.md)** — 环境准备、三种启动方式(库 / OpenAI server /
benchmark)、以及一个请求从提交到流式返回的完整运行流程(中文)。

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
  ModelRunner ─ NativeQwenVLRunner    runners/, model/  ✓ (zero transformers)
        ├─ tokenizer + image preprocessing + M-RoPE positions (ours)
        ├─ ViT + LLM forward (ours, token/bit-identical to HF)
        ├─ preallocated KV pool (in-place, no O(L²) rebuild)
        └─ vision + prefix caches  ◄── the multimodal-specific optimizations
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
- [x] **From-scratch model, zero `transformers`**: own ViT (bit-identical),
      LLM forward, M-RoPE positions, image preprocessing, tokenizer — verified
      token-/bit-identical to HF, end-to-end output matches (`model/`)
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
│   ├── model/              # from-scratch model (zero transformers)
│   │   ├── qwen2_llm.py    # LLM forward: RMSNorm, GQA+M-RoPE attn, SwiGLU
│   │   ├── qwen2_vit.py    # ViT: patch embed, 2-D rotary attn, patch merger
│   │   ├── positions.py    # M-RoPE 3-D position ids (get_rope_index)
│   │   ├── preprocess.py   # image preprocessing (smart-resize, patchify)
│   │   └── tokenizer.py    # tokenizer (tokenizers lib) + chat template
│   ├── runners/            # model execution backends
│   │   ├── base.py         # ModelRunner interface + StubRunner (no-GPU)
│   │   ├── qwen_vl.py      # HF-backed runner (reference / A-B)
│   │   └── qwen_vl_native.py  # native runner: our forward, zero transformers
│   ├── cache/              # multimodal caching + KV management
│   │   ├── vision.py       # vision-embedding cache (skip the ViT for repeated images)
│   │   ├── prefix.py       # prefix KV cache (skip the whole prefill for repeated prefixes)
│   │   └── kv_prealloc.py  # preallocated KV pool (in-place write, no O(L²) rebuild)
│   ├── kernels/            # hand-written fused kernels
│   │   └── rmsnorm.py      # fused Triton RMSNorm for the LLM stack
│   └── server.py           # OpenAI-compatible HTTP server (vision + SSE)
├── benchmarks/             # vision-cache benchmark, profiler, and compare/ (vs vLLM)
└── scripts/                # standalone checks + proto verifiers (token/bit-identical to HF)
```

## 踩坑与经验(中文)

做这个项目踩过的、比较有价值的坑,记下来给后来人(也方便面试时讲)。

### 1. ViT 在 fp16 下是「混沌系统」,rotary 频率必须 fp32
从零写的 ViT 一开始和 HF 差了 **7.5**(输出均值才 0.726)。逐层定位发现:patch_embed、每个 block、merger 单独喂 HF 的输入都 **bit 级一致**,但整条 32 层链下来就发散。根因是 rotary 的 `inv_freq`:我在 **GPU** 上算,HF 在 **CPU** 初始化,两者差 **9.4e-7**(ULP 级)。这个微小差异被 32 层 fp16 残差链**指数放大**成 2.5。把 `inv_freq` 改成在 **CPU 上用 fp32 计算**、cos/sin 全程保持 fp32(不要 `.to(fp16)`),整个 ViT 就 **bit 级一致(误差 0.0)**。
> **教训**:深层 fp16 网络对 rotary 频率的精度极度敏感;旋转位置编码的 cos/sin 一律用 fp32 算,且要和参考实现的计算设备/顺序对齐。

### 2. 预分配 KV 把瓶颈从「launch」推到「GPU」,CUDA graph 随之失效
profile 发现 decode 单步发了 **~15000 个 kernel**,大头是 cat-based KV 重建的 `copy_/cat`。换成预分配池(原地写,见 `cache/kv_prealloc.py`)后降到 **~3100 个**,decode **+15%**。但再想上 CUDA graph 时,measure 发现「连续发射 ≈ 每步同步」(比值 1.02)→ 已经 **GPU-bound**,launch 开销没了,**CUDA graph 收益≈0**。
> **教训**:measure-first。一个优化可能会改变瓶颈的性质,让下一个「显然该做」的优化变得没用。

### 3. bnb 4-bit 在带宽受限的 decode 上比 fp16 还慢
decode 是**显存带宽受限**(每步读一遍 4.4GB 权重)。理论上 4-bit 读 1/4 字节该快 4x,实测反而 **0.84x(更慢)**——bitsandbytes 先反量化成 fp16 再做 cuBLAS,**反量化那遍读写吃掉了带宽优势**。要真提速得用**融合 dequant+GEMM kernel(Marlin/AWQ)或 FP8**(sm_89 才有)。
> **教训**:量化降的是字节数,但 kernel 必须把 dequant 和 GEMM 融合,否则白搭。

### 4. benchmark 的 warmup 会污染 cache 命中率
最早测 vision/prefix cache 时,warmup 用了**同一份 workload**,把 cache 预热满了,导致计时阶段**无论复用率多少都全命中**——一度误判「vision cache 没用」。每个计时 trial 前 `cache.clear()` 冷启动才测出真实的复用收益(prefix cache 1.84x)。
> **教训**:benchmark 的 warmup 不能用会改变被测状态的同一份数据。

### 5. tokenizer 的特殊 token 不全在 tokenizer.json 里
`tokenizers` 库加载 `tokenizer.json` 后,`<|image_pad|>`(151655)、`<|video_pad|>` 识别不了——它们在 **`tokenizer_config.json` 的 `added_tokens_decoder`** 里(transformers 会读,raw `tokenizers` 不读)。手动 `add_special_tokens` 补上,且要按 id 顺序加,才能落到正确的 151655/151656。

### 6. safetensors 的权重名前缀 ≠ 加载后 state_dict
raw safetensors 用 `model.layers.N...`,而 transformers **加载后**的 `state_dict()` 是 `model.language_model.layers.N...`(新版重构了结构)。直接读 safetensors 时要按前者匹配(且 lm_head 是 tied,checkpoint 里没有单独的 lm_head)。

### 7. 端到端 gap 主要是「串行 prefill」,不是 decode
对 vLLM 的 0.44x 一度让人以为 decode 差很多。强制定长生成实测:**decode 其实只差 ~1.2x**,真正的大头是 omniserve **一次只 prefill 一个请求**(vLLM 批量 prefill)。短生成 workload 把这个短板放大了。
> **教训**:端到端数字会被某个阶段主导;要分阶段拆解才知道该优化哪。

### 8. 杂项
- **8GB 装不下两个 fp16 模型**:验证自写模块时要先把参照算完、搬到 CPU、释放参照模型,再建自己的。
- **FastAPI + `from __future__ import annotations`**:类型注解变成字符串后,`Request` 参数注解必须在**模块级 import**,否则 FastAPI 当成 query 参数报 422。
- **多个 server 进程并存会 OOM 互踩**:8GB 只够一个模型。
