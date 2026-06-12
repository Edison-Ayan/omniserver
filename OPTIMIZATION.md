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
| + preallocated KV pool | **220** | **0.44x** | **2.3x** |

Two self-implemented, profile-driven structural changes took the gap from
**5.4x to 2.3x**. The rest is analyzed below.

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

---

## Things deliberately NOT done (and why)

Knowing when to stop is half the work. Each of these was rejected *with data*.

### CUDA graphs — measured useless after the prealloc pool
After preallocation, a back-to-back (no-sync) decode loop ran at the same speed
as a synced one (ratio **1.02**) → the step is now **GPU-bound, not launch-bound**.
CUDA graphs remove launch/CPU overhead; there is none left to remove. (Before the
pool, at 15k launches, it would have helped — the pool moved the bottleneck.)

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

## Where the remaining 2.3x is

| gap source | vLLM | omniserve | tractable here? |
|---|---|---|---|
| quantized GEMM kernel (decode-dominant) | Marlin / FP8 fused | fp16 / bnb (slower) | hard — needs Marlin or an FP8 path |
| sequential prefill (~48% of time) | chunked/batched | one-at-a-time | medium |
| paged attention | dedicated kernel | preallocated pool (most of the win) | mostly done |
| fully fused kernels (QKV/RoPE/MLP) | yes | RMSNorm only | medium, small payoff |
| lean C++ forward | yes | HF Python forward | hard — a rewrite |

The character of the gap changed over this work: it started as **self-inflicted
inefficiency** (O(L²) KV rebuild, launch storms) that was fixable, and now is
mostly **specialized kernels and a rewritten forward** — years of engineering, not
a bug. The reuse-heavy multimodal path (prefix/vision cache) is where omniserve
has a real, differentiated story rather than chasing vLLM's general-purpose speed.

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
