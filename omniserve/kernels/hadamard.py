"""块对角 Hadamard 旋转 —— 量化前打散 outlier(QuaRot/SpinQuant 思路)。

计算不变性:y = a Wᵀ = (aH)(WH)ᵀ,H 正交。权重侧 WH 离线吸收(免费,在量化前对权重旋转一次);
激活侧 aH 在线(forward 里旋转)。用**块对角** Hadamard(块 BLOCK=256):整体
H = blkdiag(H₂₅₆, …) 仍正交,块小、矩阵乘快;K 必须被 BLOCK 整除
(Qwen2-VL:hidden 1536/256=6、intermediate 8960/256=35 都满足)。

为什么能帮量化:Hadamard 把"少数分量极大"的向量旋转成"各分量幅度接近"的向量(中心极限),
outlier 被打散 → 量化 scale 不再被撑大。measure(bench_hadamard.py):对 FP8 outlier 不重时
**零增益**,对 W4A4 **救回 2–3x(down 最显著)**。所以默认关闭,"必要时"(W4A4 / 图文 outlier 重)才开。

注:当前用块矩阵乘实现(够快);需要时可换 Triton 融合 FWHT(O(n log n))。
"""

from __future__ import annotations

import torch

BLOCK = 256
_CACHE: dict = {}


def hadamard_block(block: int, device) -> torch.Tensor:
    """缓存 Sylvester 递归构造的正交 Hadamard 块(±1/√block,fp32)。block 必须是 2 的幂。"""
    key = (block, str(device))
    H = _CACHE.get(key)
    if H is None:
        m = torch.ones(1, 1)
        while m.shape[0] < block:
            m = torch.cat([torch.cat([m, m], 1), torch.cat([m, -m], 1)], 0)
        H = (m / block ** 0.5).to(device=device, dtype=torch.float32)
        _CACHE[key] = H
    return H


def rotate(t: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """在最后一维做块对角 Hadamard 旋转(fp32 内部更稳),返回与输入同 dtype。"""
    *lead, K = t.shape
    if K % block:
        raise ValueError(f"K={K} 不被 Hadamard 块 {block} 整除")
    H = hadamard_block(block, t.device)
    return (t.float().reshape(*lead, K // block, block) @ H).reshape(*lead, K).to(t.dtype)
