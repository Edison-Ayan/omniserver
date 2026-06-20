"""Hadamard 旋转对量化精度损失的影响(P5,measure-driven 第一步)。

问题:Hadamard 变换(QuaRot/SpinQuant 的核心)能旋转打散 outlier,让量化更准。但我们 measure 过
纯文本激活 outlier 不重(per-token rowwise 对 FP8 几乎无收益)。所以先用**真实权重 × 真实激活**
量一刀:Hadamard 在我们这到底值不值得,甜区在哪。

计算不变性:y = a Wᵀ = (aH)(WH)ᵀ,H 正交(块对角 Hadamard,块大小 256;1536 和 8960 都整除)。
权重侧 WH 离线吸收(免费),激活侧 aH 在线(这里直接算,微基准只看精度不看速度)。

对每个 decoder GEMM,在真实激活上对比量化误差(rel L2):
  fp16(参考) | W4A16(int4 权重) | FP8-row | FP8-row+H | W4A4(int4 权重+int4 激活) | W4A4+H

预期:① FP8 上 Hadamard 收益小(outlier 本就不重);② **W4A4 直接会乱码(激活 int4 + outlier),
Hadamard 把它从不可用救回可用**——这才是 Hadamard 的甜区,尤其 down(SwiGLU 中间激活)。

用法:
  conda activate vllm_bench
  HF_HUB_OFFLINE=1 python bench_hadamard.py --layer 14
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from bench_gemm_real import grab_real_weights_and_acts, quality  # noqa: E402

BLOCK = 256  # 块对角 Hadamard 的块大小(1536/256=6, 8960/256=35 都整除)


def hadamard_matrix(n: int, device, dtype=torch.float32) -> torch.Tensor:
    """Sylvester 递归构造正交 Hadamard(n 必须是 2 的幂),元素 ±1/√n。"""
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / (n ** 0.5)).to(device=device, dtype=dtype)


def hrot(t: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """在最后一维做块对角 Hadamard 旋转(每 BLOCK 个一块)。t [..., K],K % BLOCK == 0。"""
    *lead, K = t.shape
    r = t.float().reshape(*lead, K // BLOCK, BLOCK) @ H        # fp32 旋转更稳
    return r.reshape(*lead, K).to(t.dtype)


def q_int_sym(t: torch.Tensor, n_bits: int = 4) -> torch.Tensor:
    """对称 per-row(最后一维)整数量化的 fake-quant(量化再反量化,只为测误差)。"""
    qmax = 2 ** (n_bits - 1) - 1                               # int4 -> 7
    amax = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    s = amax / qmax
    return (t / s).round().clamp(-qmax, qmax) * s


def q_fp8_row(t: torch.Tensor):
    """per-row FP8(e4m3)fake-quant。"""
    amax = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    s = (amax / 448.0)
    return (t / s).to(torch.float8_e4m3fn).to(torch.float16) * s.to(torch.float16)


def q_fp8_tensor(t: torch.Tensor):
    """per-tensor FP8(e4m3)fake-quant。per-tensor vs per-row 的差距越大 = outlier 越重。"""
    s = (t.abs().max().clamp(min=1e-4) / 448.0)
    return (t / s).to(torch.float8_e4m3fn).to(torch.float16) * s.to(torch.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--multimodal", action="store_true", help="抓真实图文激活(走 ViT)而非纯文本")
    ap.add_argument("--image-path", type=str, default=None, help="真实照片路径(默认合成几何图)")
    args = ap.parse_args()

    p = torch.cuda.get_device_properties(0)
    tag = ("图文:" + ("真实照片" if args.image_path else "合成图")) if args.multimodal else "纯文本"
    print(f"GPU: {p.name} sm_{p.major}{p.minor} | 块对角 Hadamard,块={BLOCK} | 激活={tag}\n")
    W, X = grab_real_weights_and_acts(args.layer, use_image=args.multimodal, image_path=args.image_path)
    H = hadamard_matrix(BLOCK, "cuda")

    order = ["qkv", "o", "gate_up", "down"]
    print(f"{'GEMM':8} | {'FP8tns':>7} {'FP8row':>7} {'FP8+H':>7} {'粒度gap':>7} | "
          f"{'W4A4':>8} {'W4A4+H':>8} {'救回×':>6}")
    print("-" * 78)
    for name in order:
        w, _ = W[name]                      # [N,K] fp16
        x = X[name]                         # [M,K] fp16 真实激活
        ref = F.linear(x.float(), w.float())

        def rel(out):
            return quality(out, ref)[0]

        # FP8:per-tensor / per-row / per-row+Hadamard(权重+激活都量化)
        r_fp8t = rel(F.linear(q_fp8_tensor(x).float(), q_fp8_tensor(w).float()))
        r_fp8 = rel(F.linear(q_fp8_row(x).float(), q_fp8_row(w).float()))
        xh, wh = hrot(x, H), hrot(w, H)
        r_fp8h = rel(F.linear(q_fp8_row(xh).float(), q_fp8_row(wh).float()))
        gap = r_fp8t / r_fp8 if r_fp8 > 0 else float("inf")   # per-tensor 比 per-row 差几倍 = outlier 程度
        # W4A4:int4 权重 + int4 激活(outlier 重灾);±Hadamard
        r_w4a4 = rel(F.linear(q_int_sym(x).float(), q_int_sym(w).float()))
        r_w4a4h = rel(F.linear(q_int_sym(xh).float(), q_int_sym(wh).float()))
        gain = r_w4a4 / r_w4a4h if r_w4a4h > 0 else float("inf")

        print(f"{name:8} | {r_fp8t:7.4f} {r_fp8:7.4f} {r_fp8h:7.4f} {gap:6.2f}x | "
              f"{r_w4a4:8.4f} {r_w4a4h:8.4f} {gain:5.2f}x")

    print("\n解读:① 粒度gap(FP8tns/FP8row)和 FP8 vs FP8+H 都反映 outlier 重不重——纯文本上两者都≈1/持平;"
          "若图文上 gap 拉大、FP8+H 明显降,说明图文 outlier 重、Hadamard 翻盘。② W4A4 救回× 看激活 int4。"
          "down(SwiGLU 中间激活)最该看。")


if __name__ == "__main__":
    main()
