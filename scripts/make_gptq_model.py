"""生成 omniserve gptq 点所需的 GPTQ-Int4 权重(让 benchmarks/compare/pareto.py 的 gptq 可复现)。

omniserve 的 `quant="gptq"` 从一个目录(默认 /tmp/gptq_model)加载预量化的 GPTQ-Int4 权重,
repack 成 Marlin 跑(见 omniserve/kernels/marlin_linear.py::load_gptq_llm)。这份权重的来源有两条
等价路径,本脚本默认走**可复现**的第一条:

  1)【推荐·可复现】下载官方 `Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4`
     —— 标准 AutoGPTQ 格式(bits=4, group_size=128, sym, desc_act=False),和 vLLM-int4
     对比用的是**同一份权重**,人人下到的一模一样。load_gptq_llm 直接消费。

  2)【自量化·复刻原始 /tmp/gptq_model】用 auto_gptq/gptqmodel 把基座 Qwen2-VL-2B-Instruct
     量化成 int4。原始 /tmp/gptq_model 就是这么来的(config 记录:bits=4, group_size=128,
     sym=True, desc_act=False, damp_percent=0.1, true_sequential=True)。需要装 gptqmodel
     + 校准语料,耗时较长;仅在想精确复刻"自量化"那一份时用。

用法:
  python scripts/make_gptq_model.py --out /tmp/gptq_model            # 下载官方(默认)
  python scripts/make_gptq_model.py --self-quantize --out /tmp/gptq_model  # 自量化复刻

下完/量化完,pareto.py 会自动检测到 *.safetensors 并加入 gptq 点。
"""

from __future__ import annotations

import argparse
import glob
import os

OFFICIAL_REPO = "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4"
BASE_REPO = "Qwen/Qwen2-VL-2B-Instruct"
# 原始 /tmp/gptq_model/quantize_config.json 记录的配置(auto_gptq 签名)
QUANT_CONFIG = dict(bits=4, group_size=128, sym=True, desc_act=False,
                    damp_percent=0.1, true_sequential=True)


def verify(out: str) -> None:
    """快速校验目录里有 GPTQ 必需的张量(每层 proj 的 qweight/scales),load_gptq_llm 才能消费。"""
    from safetensors import safe_open
    files = glob.glob(os.path.join(out, "model*.safetensors"))
    assert files, f"{out} 下没有 *.safetensors"
    keys = set()
    for f in files:
        with safe_open(f, "pt") as h:
            keys.update(h.keys())
    need = ["model.layers.0.self_attn.q_proj.qweight",
            "model.layers.0.self_attn.q_proj.scales",
            "model.layers.0.mlp.down_proj.qweight"]
    miss = [k for k in need if k not in keys]
    assert not miss, f"缺少 GPTQ 张量(布局不兼容 load_gptq_llm):{miss}"
    print(f"[verify] ✅ {len(keys)} 个张量,GPTQ qweight/scales 齐全,可被 load_gptq_llm 消费。")


def download(out: str) -> None:
    from huggingface_hub import snapshot_download
    print(f"[download] {OFFICIAL_REPO} → {out}")
    snapshot_download(repo_id=OFFICIAL_REPO, local_dir=out,
                      allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*"])


def self_quantize(out: str) -> None:
    """用 gptqmodel 把基座模型量化成 int4(复刻原始 /tmp/gptq_model 的配置)。"""
    from datasets import load_dataset
    from gptqmodel import GPTQModel, QuantizeConfig
    print(f"[self-quantize] {BASE_REPO} → int4 {QUANT_CONFIG}")
    qc = QuantizeConfig(bits=QUANT_CONFIG["bits"], group_size=QUANT_CONFIG["group_size"],
                        sym=QUANT_CONFIG["sym"], desc_act=QUANT_CONFIG["desc_act"],
                        damp_percent=QUANT_CONFIG["damp_percent"])
    model = GPTQModel.load(BASE_REPO, qc)
    calib = [x["text"] for x in load_dataset(
        "allenai/c4", "en", data_files="en/c4-train.00000-of-01024.json.gz",
        split="train").select(range(256)) if x["text"].strip()]
    model.quantize(calib)
    model.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/gptq_model")
    ap.add_argument("--self-quantize", action="store_true",
                    help="用 gptqmodel 自量化复刻原始权重(默认走下载官方)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.self_quantize:
        self_quantize(args.out)
    else:
        download(args.out)
    verify(args.out)
    print(f"\n完成。现在可跑:python benchmarks/compare/pareto.py --gptq-dir {args.out}")


if __name__ == "__main__":
    main()
