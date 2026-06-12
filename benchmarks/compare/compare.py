"""严谨的 omniserve vs vLLM 对比的驱动脚本。

对每个后端(独立进程,因为各自都要占掉 8 GB GPU 的大部分):
  - 生产配置(vLLM 用 CUDA graph,不是 enforce_eager)
  - 一次预热 + N 次计时 trial -> 中位数吞吐 + 波动范围
  - 用 nvidia-smi 轮询拿到真实显存峰值
  - 抓取样本输出,这样能检查各后端在内容上是否一致

然后打印一张表和一个输出一致性检查。

    python compare.py --requests 24 --trials 3

它诚实标注的几个 caveat:
  * vLLM 的显存峰值是**预留**(gpu_memory_utilization),不是实测需求,所以显存这列
    只报告、不用来给后端排名。
  * omniserve 没有 paged attention / 融合 kernel / chunked prefill;和 vLLM 的差距
    就是这些东西该有的代价。
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
    """两个后端输出完全一致的样本占比,外加一个字符级前缀重合率作为更软的信号。"""
    if not a or not b:
        return None
    n = min(len(a), len(b))
    exact = sum(norm(a[i]) == norm(b[i]) for i in range(n))
    # 平均最长公共前缀比例
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
