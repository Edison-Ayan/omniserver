"""Driver for a rigorous omniserve vs vLLM comparison.

For each backend (separate process, since each needs most of the 8 GB GPU):
  - production config (vLLM uses CUDA graphs, not enforce_eager)
  - one warmup pass + N timed trials -> median throughput + spread
  - true peak VRAM via nvidia-smi polling
  - sample outputs captured so we can check the backends agree on content

Then prints a table and an output-agreement check.

    python compare.py --requests 24 --trials 3

Caveats it is honest about:
  * vLLM peak VRAM is a *reservation* (gpu_memory_utilization), not a measured
    requirement, so the memory column is reported but not used to rank backends.
  * omniserve has no paged attention / fused kernels / chunked prefill; the gap
    to vLLM is the expected cost of those.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def _gpu_used_mib() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        return float(out.decode().split("\n")[0])
    except Exception:
        return 0.0


def run(script: str, extra: list, out: str):
    cmd = [PY, os.path.join(HERE, script), "--out", os.path.join(HERE, out)] + extra
    print(f"\n>>> {script} {' '.join(extra)}")
    baseline = _gpu_used_mib()
    peak = {"v": 0.0}
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak["v"] = max(peak["v"], _gpu_used_mib())
            time.sleep(0.1)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    r = subprocess.run(cmd, cwd=HERE, env=ENV)
    stop.set()
    th.join()
    if r.returncode != 0:
        print(f"!!! {script} failed (exit {r.returncode})")
        return None
    with open(os.path.join(HERE, out)) as f:
        d = json.load(f)
    d["peak_gpu_mib"] = round(peak["v"] - baseline)
    return d


def agg(d):
    tps = [t["tok_per_s"] for t in d["trials"]]
    rps = [t["req_per_s"] for t in d["trials"]]
    return {
        "backend": d["backend"],
        "tok_s_med": statistics.median(tps),
        "tok_s_min": min(tps), "tok_s_max": max(tps),
        "req_s_med": statistics.median(rps),
        "peak_gpu_mib": d.get("peak_gpu_mib", 0),
        "sample_text": d.get("sample_text"),
    }


def norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def output_agreement(a, b):
    """Fraction of sampled prompts where both backends' outputs agree, plus a
    char-level prefix-overlap as a softer signal."""
    if not a or not b:
        return None
    n = min(len(a), len(b))
    exact = sum(norm(a[i]) == norm(b[i]) for i in range(n))
    # average longest common prefix ratio
    def lcp(x, y):
        x, y = norm(x), norm(y)
        m = 0
        for cx, cy in zip(x, y):
            if cx != cy:
                break
            m += 1
        return m / max(1, max(len(x), len(y)))
    prefix = sum(lcp(a[i], b[i]) for i in range(n)) / n
    return exact, n, prefix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=24)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()
    common = ["--requests", str(args.requests), "--trials", str(args.trials)]

    raw = {}
    d = run("run_vllm.py", common, "vllm.json")
    if d:
        raw["vLLM"] = d
    d = run("run_omniserve.py", common + ["--fp16"], "omniserve.json")
    if d:
        raw["omniserve"] = d

    results = [agg(d) for d in raw.values()]

    print("\n" + "=" * 74)
    print(f"Qwen2-VL-2B fp16 | {args.requests} concurrent reqs | {args.trials} trials "
          f"(median) | RTX 4060 8GB")
    print("=" * 74)
    print(f"{'backend':<14}{'tok/s (med)':>14}{'[min-max]':>16}{'req/s':>9}{'peak GPU MiB*':>15}")
    print("-" * 74)
    for r in results:
        rng = f"[{r['tok_s_min']:.0f}-{r['tok_s_max']:.0f}]"
        print(f"{r['backend']:<14}{r['tok_s_med']:>14.1f}{rng:>16}"
              f"{r['req_s_med']:>9.2f}{r['peak_gpu_mib']:>15}")
    if len(results) >= 2:
        fastest = max(results, key=lambda r: r["tok_s_med"])
        print("-" * 74)
        for r in results:
            if r is fastest:
                continue
            rel = r["tok_s_med"] / fastest["tok_s_med"] if fastest["tok_s_med"] else 0
            if rel:
                print(f"{r['backend']:<14} {rel:.2f}x of {fastest['backend']} "
                      f"({1/rel:.1f}x slower)")

    print("\n* peak GPU MiB: for vLLM this is largely a pre-reservation "
          "(gpu_memory_utilization),\n  not a measured requirement — reported, "
          "not used to rank.")

    if "vLLM" in raw and "omniserve" in raw:
        oa = output_agreement(raw["vLLM"]["sample_text"], raw["omniserve"]["sample_text"])
        if oa:
            exact, n, prefix = oa
            print(f"\noutput agreement on {n} sampled prompts: "
                  f"{exact}/{n} exact match, {prefix:.0%} avg prefix overlap")
            print("  (greedy + same weights => should be near-identical; small "
                  "diffs come from kernel/quant numerics)")


if __name__ == "__main__":
    main()
