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

📈 **[OPTIMIZATION.md](OPTIMIZATION.md)** — the profile-first journey: from 0.18x
of vLLM to **1.28x *faster* than vLLM** on truly-cold fp16 traffic, then matching
vLLM-int4 to **0.82x** on reuse-heavy traffic with quantization + CUDA graph — and,
the most valuable part, a measured attribution of *exactly* where the remaining
gap lives (and why it moves between layers as you quantize).

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
- [x] Rigorous **fair** benchmark vs. vLLM (matched fp16, CUDA graphs, cold cache
      per trial on *fresh* images, median of 3 trials, verified-identical outputs)
      — on truly-cold traffic omniserve is **1.28x *faster*** than vLLM (see below;
      the earlier "0.34x" was a prefix-cache artifact, since corrected)
- [x] Content-addressed **vision embedding cache** wired into the engine
- [x] **Prefix KV cache** (reuse the whole prefill for repeated prefixes), wired
      into the **native fast path** — token-identical; reuse-0.9 fp16 throughput
      372 → **1011 tok/s** (~2.7x from reuse), matching vLLM's reuse scaling
- [x] **Fused Triton RMSNorm** kernel for the LLM stack (1.8–8.5x vs HF in
      isolation, token-identical, ~1% end-to-end — Amdahl)
- [x] **From-scratch model, zero `transformers`**: own ViT (bit-identical),
      LLM forward, M-RoPE positions, image preprocessing, tokenizer — verified
      token-/bit-identical to HF, end-to-end output matches (`model/`)
- [x] **FlashAttention** for decode and prefill — reuse vLLM's ABI-matched flash
      binary (zero compilation), GQA-native (kills the KV materialization),
      KV-pool slot as one paged block (no gather)
- [x] **FlashAttention decode-copy fix** — only `contiguous` the *used* KV length
      instead of the whole padded `max_len` (decode 16.4 → 12.6 ms)
- [x] **KV-pool layout** `[B, max_len, nkv, hd]` (flash's block layout, no
      per-layer transpose) + single-dim `index_copy_` writes (decode 12.6 → 11.5 ms)
- [x] **CUDA-graph capture** of the decode step — collapse the per-layer kernel
      launches now that shapes are static (decode 11.5 → **9.5 ms**, token-identical;
      whole decode chain **16.4 → 9.5 ms = 1.72x**)
- [x] **int4 quantization** as an optional extension — load production GPTQ-Int4
      weights (repacked to Marlin) for an apples-to-apples comparison vs. vLLM-int4
- [x] Batched prefill (last-token logits only)
- [ ] varlen / chunked-mixed prefill (needs varlen flash `cu_seqlens`; batched
      prefill currently uses a block-diagonal mask — written, awaiting varlen)
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

## Result: omniserve vs vLLM (fair, apples-to-apples)
Identical workload through each backend, in separate processes (each needs most
of the 8 GB GPU). **The headline number below was earned by first fixing the
benchmark, not the engine** — see the methodology note.

**Methodology (for comparability):**
- Same model (Qwen2-VL-2B), same precision (**fp16** by default), same prompt,
  same `max_new_tokens=64`, greedy (temp 0).
- vLLM runs in **production config — CUDA graphs enabled** (not `enforce_eager`).
- **Cold cache per trial, on *fresh* images.** vLLM has *two* caches on by default
  (prefix-KV *and* an mm/encoder cache, `mm_processor_cache_gb=4`); resetting only
  the prefix cache still reuses the ViT for repeated images. A truly-cold trial
  must feed new images each time. (This is the bug the old "0.34x" hid — see below.)
- One warmup pass, then **3 timed trials; median reported**.
- **Output is verified, not assumed**: greedy text matches between the two
  backends — the throughput numbers compare the same work, not different amounts.

**fp16 (the default; same precision both sides):**

| traffic | omniserve | vLLM | ratio |
|---|---|---|---|
| **truly cold** (reuse 0, fresh images) | **372 tok/s** | 290 | **1.28x — omniserve *faster*** |
| reuse-heavy (reuse 0.9, +prefix cache) | 1011 | 1325 | 0.76x |

On genuinely cold traffic the from-scratch engine's **lean flash + preallocated-KV
decode path is actually faster** than vLLM's general-purpose engine (which is tuned
for large batches). The reuse-heavy gap is decode-bound and is closed further with
quantization + CUDA graph below.

**int4 (optional extension; loads the *same* production GPTQ-Int4 weights vLLM
uses, repacked to Marlin — so this is same-precision, not a quantized-vs-fp16
cheat):**

| traffic | omniserve | vLLM-int4 | ratio |
|---|---|---|---|
| truly cold | 421 | 453 | 0.93x |
| reuse-heavy (+prefix cache) | **1785** | 2186 | **0.82x** |

The reuse-heavy int4 ratio climbed **0.54x → 0.67x → 0.82x** as three decode
optimizations landed (flash decode-copy fix, KV-pool layout, CUDA graph) —
collapsing the decode step **16.4 → 9.5 ms (1.72x)**.

### ⭐ The key finding: the gap *moves between layers* when you quantize
This is the most valuable conclusion of the project:

- **fp16 decode is at the memory-bandwidth floor** (each step reads the ~4.4 GB of
  weights; the 4060's bandwidth sets a ~16 ms floor). That floor is *hardware,
  equal for both engines* — so omniserve's leaner path **wins**.
- **int4 cuts weight reads to ~1/4** (floor ~4 ms). Now omniserve's 9.5 ms and
  vLLM's ~7–8 ms are *both well above* the floor → the bottleneck is no longer
  bandwidth but **kernel-scheduling efficiency**, where vLLM's fused kernels +
  full CUDA graph show. Quantization didn't make omniserve slower — **it made
  vLLM's engineering advantage *visible*.**
- **Prefill mirrors this.** Cold fp16 prefill is ViT-bound (both fp16, ~66 vs
  73 ms/img — close, omniserve's leaner scheduling edges it). Quantizing lets
  vLLM also speed up the *LLM* prefill and overlap it (290 → 453), while omniserve
  stays bottlenecked on **serial per-image ViT** (372 → 421) → it gets overtaken.

So the residual gap is not a bug or a correctness issue — it's vLLM's mature
kernel fusion + CUDA-graph + prefill-overlap engineering, and we can point to
*exactly* which layer it lives in and why. (Full derivation in
[OPTIMIZATION.md](OPTIMIZATION.md).)

\* On memory: vLLM's peak GPU is largely a **pre-reservation**
(`gpu_memory_utilization=0.9`), not a measured requirement — so VRAM is reported
elsewhere but **never** used to rank backends.

Reproduce: `cd benchmarks/compare && python compare.py --requests 24 --trials 3`
(add `--quant gptq` for the int4 comparison).

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

> ⚠️ **This table is an *earlier* snapshot** (prefix cache still on the slow HF
> runner; before the decode optimizations in §7–§8 of OPTIMIZATION.md). It's kept
> because it cleanly isolates the **relative lever** — vision cache (skip ViT) vs
> prefix cache (skip the whole prefill) — measured side-by-side. The **current
> absolute** reuse-0.9 throughput, after wiring the prefix cache into the native
> fast path, is **1011 tok/s** (fp16) / **1785** (int4) — see the headline table
> above, not the 322.9 here.

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

### 2. 「CUDA graph 没用」是阶段性结论,后来被推翻了
profile 发现 decode 单步发了 **~15000 个 kernel**,大头是 cat-based KV 重建的 `copy_/cat`。换成预分配池(原地写,见 `cache/kv_prealloc.py`)后降到 **~3100 个**,decode **+15%**。这时 measure「连续发射 ≈ 每步同步」(比值 1.02)→ 当时判定已经 **CPU-launch 跟得上**,CUDA graph 收益≈0。**但这个结论是阶段性的**:后来把 decode 的拷贝低效(flash 整段 `contiguous`)和 KV layout 修掉、decode 降到 11.5ms 后,发现即便 CPU 不是瓶颈,**GPU 侧仍有 28 层 kernel 之间的调度间隙**——CUDA graph 把这些间隙吃掉,decode **11.5 → 9.5ms(1.2x)**。
> **教训**:measure-first,但也要知道「此刻的瓶颈」会随别的优化而变;"现在没用"不等于"永远没用"。把 CUDA graph 跑通的最大坑见 OPTIMIZATION.md —— capture 时用的 `view` 是局部变量,函数返回后被 GC,它被 captured graph 引用的内部 tensor 内存被复用 → replay 时 illegal access;修复是把 view 存进 `self._graphs` 保活。

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

### 7. 端到端 gap 要分阶段拆,而且「哪阶段是大头」会随精度变
对 vLLM 的 0.44x 一度让人以为 decode 差很多。强制定长生成拆开测:**decode 只差 ~1.2x**,大头是串行 prefill。**后来基准测公平了又翻一层**:fp16 真冷下 decode 其实 omniserve **更快**、prefill 的 ViT 两边也接近(详见 OPTIMIZATION.md 的 attribution)——「prefill 是大头」只在量化后(vLLM 加速 LLM-prefill + 重叠)才成立。
> **教训**:端到端数字会被某个阶段主导;**要分阶段拆解,而且要意识到主导阶段会随精度/基准条件迁移**——别把一次拆解的结论当永久结论。

### 8. 杂项
- **8GB 装不下两个 fp16 模型**:验证自写模块时要先把参照算完、搬到 CPU、释放参照模型,再建自己的。
- **FastAPI + `from __future__ import annotations`**:类型注解变成字符串后,`Request` 参数注解必须在**模块级 import**,否则 FastAPI 当成 query 参数报 422。
- **多个 server 进程并存会 OOM 互踩**:8GB 只够一个模型。
