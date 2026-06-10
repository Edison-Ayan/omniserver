"""Driver: run the same workload through each backend (in separate processes,
since each needs most of the 8 GB GPU) and print a comparison table.

    python compare.py --requests 32 --reuse 0.0

Backends are run as subprocesses; each writes a JSON metrics file we collect.
SGLang is included only if --sglang is passed (and installed).
"""

from __future__ import annotations

import argparse
import json
import os
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
    """Run a backend in a subprocess; poll nvidia-smi to capture true peak VRAM
    (works uniformly even when the backend allocates in a child process)."""
    cmd = [PY, os.path.join(HERE, script), "--out", os.path.join(HERE, out)] + extra
    print(f"\n>>> running {script} {' '.join(extra)}")

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
    # override torch-based peak with the externally measured GPU peak (above idle)
    d["peak_mem_mib"] = round(peak["v"] - baseline)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--reuse", type=float, default=0.0)
    ap.add_argument("--sglang", action="store_true")
    args = ap.parse_args()

    common = ["--requests", str(args.requests), "--reuse", str(args.reuse)]
    results = []

    r = run("run_vllm.py", common, "vllm.json")
    if r:
        results.append(r)
    # omniserve in fp16 to match vLLM precision
    r = run("run_omniserve.py", common + ["--fp16"], "omniserve.json")
    if r:
        results.append(r)
    if args.sglang:
        r = run("run_sglang.py", common, "sglang.json")
        if r:
            results.append(r)

    print("\n" + "=" * 72)
    print(f"Qwen2-VL-2B, {args.requests} requests, reuse={args.reuse}, RTX 4060 8GB")
    print("=" * 72)
    hdr = f"{'backend':<14}{'req/s':>9}{'tok/s':>9}{'wall(s)':>9}{'peak MiB':>10}"
    print(hdr)
    print("-" * 72)
    for r in results:
        print(f"{r['backend']:<14}{r['req_per_s']:>9.3f}{r['tok_per_s']:>9.1f}"
              f"{r['wall_s']:>9.1f}{r['peak_mem_mib']:>10.0f}")
    if results:
        fastest = max(results, key=lambda x: x["tok_per_s"])
        print("-" * 72)
        for r in results:
            rel = r["tok_per_s"] / fastest["tok_per_s"] if fastest["tok_per_s"] else 0
            print(f"{r['backend']:<14} {rel:.2f}x of {fastest['backend']} tok/s")


if __name__ == "__main__":
    main()
