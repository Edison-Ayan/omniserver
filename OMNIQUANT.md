# OmniQuant: Precision Planning for Multimodal LLM Serving

> Research goal: turn quantization from a single global switch into a
> stage-aware, operator-aware and modality-aware precision planner for
> multimodal serving.

## Thesis

The best quantization policy for multimodal inference is not "all int4" or
"all FP8". It depends on four interacting factors:

- **Serving stage**: prefill has large-M GEMMs and is often compute-bound; decode
  has small-M GEMMs and is often weight-bandwidth-bound.
- **Operator shape**: `gate_up` and `down` are wide MLP GEMMs; `qkv` and `o_proj`
  have different shape and launch/scale overhead behavior.
- **Numerical sensitivity**: attention projections and selected layers may need
  higher precision than MLP-heavy paths.
- **Modality distribution**: image tokens, OCR/chart prompts and long visual
  prefixes may change activation ranges and KV-cache sensitivity.

The target outcome is a Pareto frontier over speed, memory and quality, plus a
planner that can choose a precision plan under a user-visible budget.

## Current Starting Point

omniserve already has the key building blocks:

- GPTQ/Marlin int4 loading for apples-to-apples vLLM-int4 comparison.
- FP8 linear support and `apply_mixed(model, plan)` for per-op precision plans.
- A measured default plan: MLP FP8 + attention fp16.
- Pareto evidence: mixed precision roughly matches GPTQ throughput with much
  lower quality loss, while saving substantial memory versus fp16.
- Hadamard tooling, with current evidence that it helps W4A4-style activation
  quantization but is not useful for the default FP8 path.

This document turns those pieces into a deeper research program.

## Milestone 1: Quantization Pareto Dashboard

Build a repeatable benchmark that produces one table and one plot for every
precision plan.

Precision plans:

- `fp16`: full precision baseline.
- `gptq`: calibrated int4 decoder weights.
- `marlin`: local RTN int4 path, kept mostly as an ablation baseline.
- `mixed`: FP8 MLP + fp16 attention.
- `prefill_fp8_decode_int4`: stage-aware policy, FP8 for large-M prefill MLPs
  and int4 for decode MLPs.
- `layerwise_mixed`: fp16 for sensitive layers, FP8/int4 elsewhere.

Workloads:

- Text-only technical prompts.
- Single-image QA.
- OCR or chart/table prompts.
- Long shared visual-prefix prompts.

Metrics:

- Throughput: output tok/s.
- Latency: TTFT and ITL p50/p95.
- Memory: peak allocated/reserved MiB and KV-cache footprint.
- Quality: token acceptance rate versus fp16, teacher-forcing ppl, and task-level
  answer agreement for visual prompts.

Expected conclusion:

> Global int4 is not the universal optimum. Mixed precision creates better
> speed-quality points because FP8 fits compute-heavy prefill MLPs while int4 is
> strongest when decode becomes weight-bandwidth-bound.

## Milestone 2: Per-Op and Per-Stage Planner

Make precision a first-class configuration object rather than a fixed preset.

Candidate policy:

```text
prefill:
  qkv      fp16
  o_proj   fp16
  gate_up  fp8
  down     fp8 or fp16

decode:
  qkv      fp16 or gptq
  o_proj   fp16 or gptq
  gate_up  int4 or fp8
  down     int4
```

Questions to answer:

- At what M does FP8 overtake W4A16 for each GEMM?
- Which ops are launch/scale-overhead dominated at decode batch sizes?
- Does attention projection quantization hurt visual prompts more than text?
- Can one plan handle both cold multimodal traffic and prefix-reuse-heavy
  traffic, or should the planner switch by workload?

Deliverables:

- A `PrecisionPlan` schema.
- CLI flags for selecting named plans.
- A CSV/JSON output format that records plan, workload, speed, quality and
  memory.
- A short write-up explaining each plan with roofline-style reasoning.

## Milestone 3: Layer-Wise Sensitivity

Measure which layers deserve precision budget.

Experiments:

- Quantize only one layer at a time and measure quality loss.
- Keep only one layer fp16 while quantizing the rest.
- Compare first-N fp16, middle-N fp16 and last-N fp16 policies.
- Separate MLP sensitivity from attention sensitivity.

Workload-specific checks:

- OCR and chart prompts, where exact tokens and spatial relations matter.
- Color/count/location prompts, where visual grounding errors are easy to spot.
- Shared-image multi-question prompts, where prefix reuse makes image-token KV
  quality important.

Expected output:

- A layer sensitivity heatmap.
- A compact policy such as "keep layers 0-3 and 26-27 fp16; FP8 MLP in the
  middle; preserve attention in fp16".
- A comparison against uniform `mixed` and full `gptq`.

## Milestone 4: Modality-Aware Activation Analysis

The goal is to know whether image tokens need different quantization treatment
than text tokens.

Collect activation statistics by token region:

- Text-only prompts.
- Image token spans.
- Text after image spans.
- OCR/chart-heavy prompts.
- Multi-image prompts.

For each selected layer/op, record:

- max/mean/rms activation range.
- outlier ratio.
- per-tensor versus rowwise FP8 error.
- Hadamard benefit or lack of benefit.
- quantization error split by image-token and text-token positions.

Decision rule:

- If per-tensor FP8 remains close to rowwise on visual workloads, keep the fast
  per-tensor path as default.
- If image-heavy prompts show clear outliers, enable rowwise or blockwise only
  for the affected ops/layers.

## Milestone 5: KV-Cache Quantization

This is the most serving-specific extension and directly targets 8 GB GPUs.

Variants:

- Store K/V as fp16 baseline.
- Store V as fp8 or int8, keep K fp16.
- Store both K and V as fp8/int8.
- Store prefix-cache entries in low precision.
- Restore prefix cache directly in low precision versus dequantizing to fp16.

Metrics:

- Decode speed and attention overhead.
- Maximum batch size under fixed memory.
- Long-context memory savings.
- Token acceptance rate and visual-answer agreement.

Key questions:

- Is K more sensitive than V for Qwen2-VL decode?
- Are image-prefix KV entries more sensitive than text-prefix KV entries?
- Does low-precision prefix cache create compounding errors across multi-turn
  visual chats?

## Milestone 6: Vision Tower Quantization

This gives the project a clearly multimodal identity.

Targets:

- ViT patch embedding.
- ViT attention qkv/o.
- ViT MLP.
- Vision projector.
- Vision embedding cache storage format.

Plans:

- FP8 ViT MLP only.
- FP8 ViT attention + MLP.
- FP8 or int8 vision embeddings in cache.
- Hybrid: keep projector fp16 if it is the quality bottleneck.

Metrics:

- ViT latency per image.
- TTFT on cold image traffic.
- Cache memory per image.
- Visual QA agreement against fp16.

## Suggested Execution Order

1. Build the Pareto dashboard around existing plans.
2. Add the explicit `PrecisionPlan` schema and named plan registry.
3. Run per-op/per-stage sweeps to justify the default planner.
4. Run layer-wise sensitivity to find high-value fp16 islands.
5. Add modality-aware activation analysis.
6. Try KV-cache quantization.
7. Quantize the vision tower after the LLM-side evidence is solid.

## Success Criteria

The research is successful if it produces:

- A reproducible Pareto plot showing speed, memory and quality trade-offs.
- A default mixed policy that is clearly better than at least one global policy.
- A stage/op explanation for why that policy wins.
- At least one multimodal-specific result that text-only quantization would not
  reveal, such as image-token activation behavior, visual-prefix KV sensitivity
  or ViT quantization trade-offs.

