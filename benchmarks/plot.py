"""Plot the vision-cache sweep results into the headline figure.

    python plot.py results.csv      # -> results.png

Produces a 1x3 panel: TTFT, throughput, and cache hit rate vs image reuse rate,
baseline vs cached. This is the figure to put in the README / resume writeup.
"""

import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    cols = {k: [float(r[k]) for r in rows] for k in rows[0]}
    return cols


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    out = path.replace(".csv", ".png")
    d = load(path)
    reuse = d["reuse"]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    ax[0].plot(reuse, d["base_ttft_p50_ms"], "o-", label="baseline")
    ax[0].plot(reuse, d["cache_ttft_p50_ms"], "s-", label="+ vision cache")
    ax[0].set_title("TTFT (p50) vs image reuse")
    ax[0].set_xlabel("image reuse rate")
    ax[0].set_ylabel("TTFT (ms)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(reuse, d["base_req_per_s"], "o-", label="baseline")
    ax[1].plot(reuse, d["cache_req_per_s"], "s-", label="+ vision cache")
    ax[1].set_title("Throughput vs image reuse")
    ax[1].set_xlabel("image reuse rate")
    ax[1].set_ylabel("requests / s")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    ax[2].plot(reuse, d["cache_hit_rate"], "s-", color="green")
    ax[2].plot(reuse, reuse, "--", color="gray", label="ideal (=reuse rate)")
    ax[2].set_title("Cache hit rate")
    ax[2].set_xlabel("image reuse rate")
    ax[2].set_ylabel("hit rate")
    ax[2].legend()
    ax[2].grid(alpha=0.3)

    fig.suptitle("omniserve: vision embedding cache on Qwen2-VL-2B (4-bit, RTX 4060 8GB)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
