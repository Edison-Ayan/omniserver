# Optimization log: closing the gap to vLLM, profile-first

All work was on a single **RTX 4060 Laptop (8 GB)** serving **Qwen2-VL-2B**.
Every change was driven by a measurement and verified token-identical to the
unoptimized path. The point of this log is not the absolute numbers (a 4060 will
never beat a datacenter-tuned engine) but the **method**: measure → find the
bottleneck → fix it → re-measure → know when to stop.

## Trajectory (24 concurrent requests, fp16, tok/s, vs vLLM 0.12 ≈ 505 tok/s)

| stage | tok/s | vs vLLM | slower by |
|---|---|---|---|
| sequential decode (naive) | 53 | 0.18x | 5.4x |
| + batched decode | 172 | 0.34x | 2.9x |
| + preallocated KV pool | 220 | 0.44x | 2.3x |
| + layer kernel fusion | 221 | 0.44x | 2.3x |
| + FlashAttention (decode) | 236 | 0.47x | 2.1x |
| + FlashAttention (prefill) + ViT patch_embed matmul | 280 | 0.56x | 1.8x |
| + batched prefill (last-token logits only) | **329** | **0.65x** | **1.5x** |

Structural + kernel changes took the gap from **5.4x to 1.8x** on this
(prefill-heavy) benchmark. But the per-phase picture is the real story:

> **Decode (batch 16, fp16): omniserve is 1.32x *faster* than vLLM.** Measured by
> the decode slope (gen 64→320, which cancels prefill): omniserve 709 tok/s vs
> vLLM 535 tok/s. At small batch, decode is memory-bandwidth-bound (both at the
> weight-read floor), and omniserve's lean flash+preallocated-pool path has less
> overhead than vLLM's general engine (vLLM is tuned for large batches).
>
> **The entire same-precision gap is prefill** — vLLM batches the ViT and overlaps
> prefill with decode (chunked prefill); omniserve runs the ViT per image and
> prefills as a separate phase. The benchmark numbers above are prefill-dominated
> and hide the decode win.

> ⚠️ **更正(2026-06-15,详见 探索日志.md 第 8、9 阶段)**:上面"gap 全在 prefill"被后续
> measure 推翻。直接 head-to-head 测 vLLM 自己的 ViT(66ms/图 vs omniserve 73ms),**没有
> "ViT GEMM 2x 护城河"**。之前的"prefill 慢 3x"是**基准的缓存不对称**:vLLM 默认开
> 两层缓存(prefix KV + mm/encoder `mm_processor_cache_gb=4`),warmup 用同图把 prefill 全
> 缓存住;omniserve 每 trial reset 冷跑。**公平对比(gen=64,24 请求,fp16):**
> 真冷 omniserve **341** vs vLLM **290**(omniserve 快 1.18x);高复用 reuse0.9 omniserve+pc
> **821** vs vLLM **1325**(vLLM 快 1.61x,差在 decode 的 CUDA graph,不在 prefill)。
> 已落地:run_vllm 加 `--no-prefix-cache`+每 trial reset(P0);prefix cache 接到 native
> 快路径(P1,reuse0.9 → 2.4x、零复用零开销、逐 token 一致)。**冷测真冷必须每 trial 用全新图**
> (只关 prefix cache 仍有 mm cache 复用 ViT,会虚高到 498)。

The rest is analyzed below.

> ## Final state (after the fair benchmark + phases 7–9 below)
> The prefill-dominated trajectory table above is *historical*. With the benchmark
> made fair (cold cache on fresh images) and three more decode optimizations
> landed, the honest current numbers are:
>
> | traffic | precision | omniserve | vLLM | ratio |
> |---|---|---|---|---|
> | truly cold | fp16 | **372** | 290 | **1.28x — faster** |
> | reuse 0.9 | fp16 | 1011 | 1325 | 0.76x |
> | truly cold | int4 (GPTQ) | 421 | 453 | 0.93x |
> | reuse 0.9 | int4 (GPTQ) | **1785** | 2186 | **0.82x** |
>
> The decode step went **16.4 → 9.5 ms (1.72x)** via §7–§8. The single most
> valuable result is **§ The attribution** below: quantization moves the bottleneck
> from a *hardware* floor (bandwidth, equal for both) to *engineering* (kernel
> efficiency, vLLM's moat) — which is why omniserve wins at fp16 and trails at int4.

---

## 1. Batched decode — 53 → 172 tok/s (~3x)

**Observation.** Naive decode ran one sequence at a time; throughput was flat
across load (52 tok/s at 4 requests, 53 at 24) — it could not exploit
concurrency. vLLM scaled ~4x over the same range because it batches the decode
step.

**Change.** Advance all running sequences in a *single* forward: left-pad each
sequence's KV to the batch max, pass explicit **M-RoPE** position ids
(`pos = length + rope_delta`) and an attention mask. The M-RoPE handling is the
multimodal-specific wrinkle — positions are 3-D (t/h/w), so the text-only batched
decode path doesn't transfer directly.

**Verification.** Token-identical to per-sequence decode for equal- and
ragged-length batches (`scripts/proto_batched_decode.py`).

## 2. Preallocated KV pool — 172 → 220 tok/s (+28% e2e, +15% decode)

**Observation (profile).** The decode step issued **~15,200 GPU kernel launches**,
dominated by `aten::copy_` (~2000) + `aten::cat` (~1200) + elementwise (~1800) —
all from rebuilding the batched KV cache every step (`to_legacy → pad → cat →
scatter`). The cat also copies the *entire* growing KV each step: O(L) per step,
**O(L²)** per sequence. CPU time ≈ wall time → launch/CPU-bound.

**Change.** Preallocate one fixed `[B, H, max_len, D]` buffer per layer. Write the
new token's KV **in place** at each slot's current length (a scatter); attention
reads a view — no growth, no rebuild. Active sequences hold contiguous slots
`[0, n)`; a finishing sequence is compacted by moving the last slot into the hole
(a new `runner.free()` hook). This is the spirit of PagedAttention without a
custom kernel.

**Result.** Kernels/step **15.2k → 3.1k** (5x fewer); decode **+15%**, end-to-end
**+28%** (it cut both launch overhead and HBM traffic). Token-identical incl.
staggered finishes (`scripts/proto_prealloc_kv.py`). Also produces the **static
shapes** a CUDA graph would need.

## 3. Multimodal caching (reuse-heavy traffic)

Two ways to exploit repeated images (multi-turn chat, repeated system images),
measured in-engine, cold cache per trial:

| at reuse 0.9 (92% hit) | speedup |
|---|---|
| vision embedding cache (skip the ViT) | 1.46x |
| **prefix KV cache (skip the whole prefill)** | **1.84x** |

The vision cache only skips the ViT (a small slice of prefill); the prefix cache
reuses the *entire* prefill KV on an exact (prompt tokens + image content) match —
the lever vLLM's prefix cache / SGLang's radix cache pull.

## 4. Fused Triton RMSNorm — kernel 1.8–8.5x, e2e +1%

A hand-written single-pass RMSNorm replacing HF's multi-op `Qwen2RMSNorm`
(`omniserve/kernels/rmsnorm.py`). 1.8–8.5x faster in isolation, token-identical,
but **+1% end-to-end** — RMSNorm is memory-bound yet a small slice of total time.
A clean illustration of Amdahl's law: a fast kernel on a small fraction barely
moves the needle.

## 5. Full layer-kernel fusion (vLLM-style) — e2e +2%

Once the forward was our own (`omniserve/model/qwen2_llm.py`), we replicated
vLLM's in-layer fusions: **fused QKV** (q/k/v into one GEMM, split after), **fused
gate_up** (one GEMM), **fused SiLU×Mul** (Triton), and **residual-carried
fused add+RMSNorm** (Triton, the add folded into the norm — vLLM never writes an
explicit `x = x + ...`). All token-identical. End-to-end: **+2%** (216 → 221 tok/s).
Amdahl again — see the nsys breakdown below for why.

## 6. FlashAttention for decode — reuse vLLM's binary, decode 1.5x

The nsys analysis (below) said decode's attention (SDPA `fmha`) + the GQA KV
materialization were ~39% of decode, and that the fix was FlashAttention — but
building flash-attn here kept hitting the nvcc / cu130 wall. **The unlock: vLLM
already ships a flash binary** (`vllm.vllm_flash_attn`) that is ABI-matched to
this torch/CUDA — import it, zero compilation.

Integration (`omniserve/cache/kv_prealloc.py::flash_decode`): treat each KV-pool
slot as one paged block (`block_size = max_len`), pass `block_table` + `seqused_k`
so `flash_attn_varlen_func` reads the pool directly (no gather), GQA-native (no KV
materialization). Micro-bench flash vs SDPA+materialize: **11.56x**. In the engine:
decode **29.6 → 19.65 ms (1.5x, 541 → 814 tok/s)**; output 3/4 token-identical to
HF, 1/4 semantically equal (flash's own fp16 numerics, same as vLLM-vs-HF).
**Decode-heavy end-to-end (gen=200): 0.51x → 0.68x of vLLM.**

The honest twist: this did **not** make decode launch-bound (back-to-back ≈ synced,
still **1.0**). FlashAttention removed the attention cost, so now the **GEMMs are
~82% of decode** — bandwidth-bound on weight reads. So CUDA graphs still wouldn't
help (§ below holds); the next lever is a **quantized GEMM** (Marlin/FP8), not graphs.
Reusing a production binary instead of fighting a compiler was the highest-leverage
move of the whole project.

## 7. FlashAttention decode-copy fix — decode 16.4 → 12.6 ms

**Observation.** After §6, the remaining decode "moat" looked like a vLLM kernel
advantage (omniserve 16.4 ms vs vLLM ~11 ms/step). It wasn't. `flash_decode`
re-laid-out the KV every step, every layer: it `transpose + contiguous`'d the
*entire* `[n, nkv, max_len=1024, hd]` pool slot into flash's block layout — but
the real sequence was only ~300 tokens. We were copying **3x of useless padding**
every layer.

**Change.** Copy only `ret_len` (the used length, rounded up to a multiple of 16
so flash's `block_size` divides it) instead of the whole `max_len`.

**Result.** Decode **16.4 → 12.6 ms**, token-identical. This **falsified the "it's
a vLLM kernel moat" story** for ~4 ms of the gap — that 4 ms was *our own* copy
inefficiency. (Measure caught it again.) It helped every config, fp16 and int4:
fp16-cold 341 → **372** (now 1.28x *faster* than vLLM), int4 reuse-0.9 0.54x →
0.67x.

## 8. KV-pool layout + CUDA graph — decode 12.6 → 9.5 ms (and "graphs are useless" overturned)

§2's note said CUDA graphs were useless because decode was GPU-bound. That was a
*stage-specific* truth, and §7 changed the stage. Two changes finished the job:

**8-A. KV-pool layout `[B, max_len, nkv, hd]`.** The old pool was
`[B, nkv, max_len, hd]`, so every layer transposed it into flash's block layout.
Storing it *as* the block layout means decode reads `pool.k[:n]` directly — **no
per-layer transpose** — and KV writes become a single-dim `index_copy_`
(`flat = row*max_len + pos`), which is graph-friendly. Decode **12.6 → 11.5 ms**,
token-identical.

**8-B. CUDA-graph capture.** Even with the CPU keeping up, there are **GPU-side
scheduling gaps between the 28 layers' kernels**. Capturing the decode step into a
CUDA graph and replaying it erases those gaps: decode **11.5 → 9.5 ms (1.2x)**,
token-identical. Whole decode chain: **16.4 → 12.6 (§7) → 11.5 (8-A) → 9.5 ms =
1.72x.**

> **The debugging lesson (worth more than the 1.2x).** Capture kept throwing
> `illegal memory access` on replay. The usual suspects (flash, `index_put`,
> in-place deps, `inference_mode`) were all innocent — capturing a single layer or
> a full forward in isolation (`mg.py`) worked fine, so the bug was in the *engine
> integration path*. Root cause: the `view` used during capture was a **local
> variable**; after the capture function returned it was GC'd, and the internal
> tensors (`_rows` etc.) that the captured graph still referenced had their memory
> reused → illegal access on replay. Fix: keep the view alive
> (`self._graphs[n] = (g, buf, view)`). Two more gotchas: buffer-dependent indices
> (`flat`/`seqused`) must be pre-computed *outside* the graph into static buffers;
> and you cannot capture a graph inside `@torch.inference_mode()`.

## 9. int4 quantization (Marlin / GPTQ) — optional, apples-to-apples

To compare against *vLLM-int4* honestly (not int4-vs-fp16), omniserve loads the
**same production GPTQ-Int4 weights vLLM uses** and repacks them to Marlin
(`omniserve/kernels/marlin_linear.py`): `gptq_marlin_repack` for the packed
qweight + scales, fused qkv/gate_up concatenated along N. Default stays fp16, so
the fp16 comparison is never a quantization cheat.

| traffic | omniserve GPTQ | vLLM-int4 | ratio |
|---|---|---|---|
| truly cold | 421 | 453 | 0.93x |
| reuse 0.9 (+prefix cache) | 1785 | 2186 | **0.82x** (was 0.54x before §7–§8) |

Two debugging notes worth keeping: (1) you must load each layer's **two RMSNorms**
— leave them at the init `ones` and the proj unit-tests still pass but the whole
model outputs garbage (bisect layer-by-layer to find it); (2) **don't pass `bias`
into `gptq_marlin_gemm`** (wrong result, rel-err 0.90) — add it outside the kernel
(rel-err 0.0002).

The same-precision int4 result still trails vLLM, and *why* is the project's punchline:

## ⭐ The attribution — the gap moves between layers when you quantize

This is the single most valuable conclusion. The residual gap is not a bug, not a
correctness issue, and not in a fixed place — **it migrates as precision changes,
and we can say exactly where and why:**

| | decode bottleneck | prefill bottleneck |
|---|---|---|
| **fp16** | weight-read **bandwidth** (a hardware floor, ~16 ms, *equal for both*) → omniserve's leaner path **wins** | **ViT** (both fp16, 66 vs 73 ms/img, close) → omniserve's leaner scheduling edges it |
| **int4** | **kernel efficiency** (weight reads drop to ~1/4, floor ~4 ms; omniserve 9.5 ms / vLLM ~7–8 ms are *both far above* it) → vLLM's fused kernels + CUDA graph show → vLLM **wins** | vLLM also quantizes the **LLM** prefill and overlaps it (290 → 453); omniserve stays bottlenecked on **serial per-image ViT** (372 → 421) → overtaken |

**Quantization didn't make omniserve slower — it made vLLM's engineering advantage
*visible*.** At fp16 both engines sit on the same hardware floor (bandwidth, ViT
compute), so the lean from-scratch engine wins. Quantization removes those equal
floors and exposes the layer where vLLM has years of work omniserve doesn't:
fused decode kernels, a full CUDA graph over the whole step, and overlapped
batched prefill. Being able to name *which* layer the gap lives in, *why* it moved
there, and *what* (kernel fusion + prefill overlap) would close it — is a stronger
result than a single tuned throughput number.

## Why vLLM is fast — the compounding chain (nsys evidence)

We profiled both engines' decode with Nsight Systems (`scripts/prof_decode.py`,
`scripts/prof_vllm_decode.py`). The kernels differ in exactly the places that matter:

| function | omniserve | vLLM (nsys) |
|---|---|---|
| decode attention | SDPA `fmha_cutlassF` + `repeat_interleave` to **materialize GQA KV (18.8%)** | **`flash_fwd_splitkv`** (FlashAttention decode, GQA-native, no materialization) |
| RoPE | PyTorch elementwise chain | `rotary_kernel` (fused) |
| SiLU×Mul | our Triton `_silu_mul_fwd` | `vllm::act_and_mul_kernel` |
| QKV / gate_up | fused (after §5) | fused |

omniserve's decode GPU time: **GEMM 55% + attention 20% + elementwise 19%** (the
19% is mostly the GQA KV materialization) + our fused kernels ~1%.

**The key is that it compounds.** A standalone data point: vLLM with `enforce_eager`
(no CUDA graphs) does **293 tok/s**; with CUDA graphs, **503 tok/s** — graphs alone
are **1.7x**. At the time, on omniserve we measured CUDA graphs as *useless*. The
resolution *as it stood then*:

> Fast kernels (FlashAttention + fused) make each step's GPU time **short enough
> that kernel launch becomes the bottleneck** → the engine is *launch-bound* → CUDA
> graphs erase that launch overhead → +1.7x. omniserve's slower kernels (SDPA + KV
> materialization) keep each step **GPU-bound**, so the CPU already keeps up and
> graphs do nothing.

> **Refined later (§8):** "CPU keeps up ⇒ graphs do nothing" was incomplete. After
> §7's copy fix, a graph still bought **1.2x** by collapsing the **GPU-side gaps
> between layer kernels** — a separate effect from CPU launch overhead. So graphs
> help via *two* mechanisms (launch overhead **and** inter-kernel GPU gaps); only
> the first was absent on omniserve, not the second.

So vLLM's speed is a chain where each link enables the next:

1. **FlashAttention** — GQA-native (kills the 19% KV materialization) and faster than SDPA;
2. **fused CUDA kernels** (rotary, act_and_mul, QKV/gate_up) — shrink GPU time further;
3. **1 + 2 ⇒ launch-bound ⇒ CUDA graphs** add 1.7x;
4. **PagedAttention** — no KV waste ⇒ larger batches ⇒ weight reads amortized further.

When this was first written, omniserve was stuck at **link 1**: SDPA can't give
GQA-native *and* fast (we tried `enable_gqa` — slower, see below), so the step
stayed GPU-bound. **§6–§8 then walked the chain anyway:** link 1 by reusing vLLM's
flash binary (GQA-native, no KV materialization), and link 3 by getting the CUDA
graph to capture (once §7's copy fix and §8-A's layout exposed the inter-kernel
gaps). What omniserve still lacks is vLLM's **fused decode kernels** (norm + rope +
residual + quant collapsed into fewer kernels) and **PagedAttention-scale batching**
— which is exactly why the residual gap only becomes visible under quantization
(see *The attribution* above): at fp16 the bandwidth floor hides it; at int4 it
doesn't.

---

## Things deliberately NOT done (and why)

Knowing when to stop is half the work. Each of these was rejected *with data*.

### CUDA graphs — first measured useless, later done anyway (§8)
After preallocation, a back-to-back (no-sync) decode loop ran at the same speed as
a synced one (ratio **1.02**) → the step was **GPU-bound, not launch-bound**, so
removing CPU/launch overhead bought nothing. **This conclusion didn't survive §7.**
Once the decode-copy fix and KV-layout change cut the step to 11.5 ms, profiling
showed the remaining cost included **GPU-side gaps between the 28 layers' kernels**
— which a captured graph *does* erase (11.5 → 9.5 ms, 1.2x). Lesson: "no launch
overhead to remove" ≠ "no graph benefit"; a graph also collapses inter-kernel GPU
scheduling gaps, and *which* effect dominates depends on the current bottleneck.
(Kept here, not deleted, because the wrong-then-right arc is the point.)

### SDPA `enable_gqa` to drop the KV materialization — measured *slower*
nsys flagged the GQA `repeat_interleave` (materializing `[B,2,L,D]→[B,12,L,D]`
each layer) as 19% of decode. The obvious fix — `F.scaled_dot_product_attention(
..., enable_gqa=True)`, which broadcasts the 2 KV heads in-kernel — made decode
**slower** (40 vs 30 ms/step): it forces SDPA off the fast `fmha` path onto the
math backend. *Materialize + fast kernel* beats *broadcast + slow kernel*. Getting
both (GQA-native *and* fast) needs FlashAttention — vLLM's moat, not in SDPA.

### bitsandbytes 4-bit — measured *slower* than fp16 for decode
Decode is **memory-bandwidth-bound on weight loading** (fp16 2B weights ≈ 4.4 GB
read per step; at the 4060's ~270 GB/s that's a ~16 ms floor). In theory 4-bit
reads 4x fewer bytes and should win. Measured: **4-bit 0.84x of fp16** (370 vs
541 tok/s decode) — *slower*. bitsandbytes dequantizes to fp16 then runs cuBLAS;
the unfused dequant pass eats the bandwidth savings. The fix is a **fused
dequant+GEMM kernel (Marlin/AWQ)** or **FP8**, neither of which HF/bnb ships.

### The fp16 GEMM kernel itself — already near the bandwidth floor
The decode GEMMs run on cuBLAS/CUTLASS (`cutlass_80_wmma_tensorop`). The `80` is
an sm_80 (Ampere) target; the GPU is sm_89 (Ada). For **fp16 this is fine** — Ada's
fp16 tensor cores are the same generation as Ampere's, and the kernel already
reads weights at near peak bandwidth, so there is nothing to optimize on the
kernel (the bottleneck is bytes read, not kernel efficiency). What sm_89 *uniquely*
adds is **FP8 tensor cores** (1 byte/param, hardware-accelerated) — that, not a
different fp16 kernel, is the Ada-specific lever for bandwidth-bound decode.

---

## Where the remaining gap is (and where it isn't)

The gap that was once "2.3x, everywhere" has been **localized**. What used to be
self-inflicted inefficiency is fixed; what's left is vLLM's specialized engineering,
and only at int4:

| gap source | status |
|---|---|
| O(L²) KV rebuild / launch storms | **fixed** — preallocated pool (§2) |
| GQA KV materialization | **fixed** — FlashAttention, GQA-native (§6) |
| decode-copy of padded `max_len` | **fixed** — copy only used length (§7) |
| per-layer KV transpose | **fixed** — pool stored as flash's block layout (§8-A) |
| inter-kernel GPU scheduling gaps | **fixed** — CUDA-graph decode (§8-B) |
| quantized GEMM (apples-to-apples vs vLLM-int4) | **done** — GPTQ→Marlin (§9) |
| **fused decode kernels** (norm+rope+residual+quant in fewer kernels) | **vLLM-only** — ~the int4 decode gap (9.5 vs ~7–8 ms); years of kernel work |
| **overlapped batched prefill** (chunked, ViT batched + prefill/decode overlap) | **vLLM-only** — the int4 cold-prefill gap; omniserve runs ViT serially per image |
| varlen / chunked-mixed prefill (fp16 too) | **omniserve TODO** — block-diagonal path written, needs varlen flash |
| lean C++ forward | not pursued — a rewrite, marginal given the above |

The honest bottom line: **at fp16, on a fair benchmark, the from-scratch engine is
not slower — it's faster on cold traffic** (§ Final state). The residual gap is an
**int4-only** phenomenon and lives in two named places — fused decode kernels and
overlapped prefill — i.e. vLLM's mature kernel/scheduler engineering, not a bug or
a structural flaw. The reuse-heavy multimodal path (prefix/vision cache) remains
omniserve's own differentiated story. Knowing exactly which two layers remain, and
why they only surface under quantization, is the result — not a number to keep
grinding on an 8 GB laptop.

---

## Methodology notes (measurement bugs caught)

Rigor included catching our own mistakes:

- **Warmup primed the cache.** An early reuse benchmark warmed up with the same
  workload, so every timed request hit regardless of reuse rate — making the
  vision cache look useless. Cold-starting each trial fixed it.
- **vLLM was handicapped.** An early comparison ran vLLM with `enforce_eager`
  (no CUDA graphs), flattering omniserve at 0.48x; the honest production-config
  number is lower. Always benchmark the opponent at its best.
- **Memory wasn't comparable.** vLLM's peak VRAM is a `gpu_memory_utilization`
  *reservation*, not a measured requirement — so it's reported but never used to
  rank backends.
