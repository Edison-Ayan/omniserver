"""per-op 混合精度策略(P5 子任务2):按算子选 fp16 / FP8(W8A8)/ marlin(W4A16)。

落地 micro-bench(真实权重 × 真实激活)导出的"最优配置 = roofline ⊕ 数值敏感度"结论,而非
全局单一精度。默认计划(DEFAULT_PLAN):

  gate_up → FP8   :大 N GEMM,FP8 通吃(decode 3.28x / prefill 2.16x),rel ~0.03
  down    → FP8   :decode 1.22x / prefill 1.0–1.33x 不亏 fp16,rel ~0.03(优于 RTN-int4 的 0.10)
  qkv / o → fp16  :小 GEMM 量化都不划算(M=24 FP8 崩 0.45x)+ attention 数值敏感 → 双重保 fp16

即"MLP 用 FP8 + attention 保 fp16"。三档精度可按 plan 自由覆盖,例如 decode 偏好场景可把
down 换成 marlin(W4A16,decode 1.36x 略快,但无校准 RTN 精度 rel 0.10),即一个模型里
FP8(gate_up)+ W4A16(down)+ fp16(qkv/o)三种精度按算子共存。
"""

from __future__ import annotations

# 默认 per-op 精度计划:键=算子,值 ∈ {"fp16", "fp8", "marlin"}
DEFAULT_PLAN = {"qkv": "fp16", "o": "fp16", "gate_up": "fp8", "down": "fp8"}

# 算子名 -> (子模块, 属性)
_ATTR = {
    "qkv": ("self_attn", "qkv_proj"),
    "o": ("self_attn", "o_proj"),
    "gate_up": ("mlp", "gate_up_proj"),
    "down": ("mlp", "down_proj"),
}


def _replace(layer, op: str, prec: str) -> int:
    """把单层的一个算子换成指定精度。fp16=不动,返回是否替换(0/1)。
    精度可带 "+h" 后缀(如 "fp8+h"/"marlin+h")开启 Hadamard 旋转(默认关,outlier 重时才用)。"""
    sub, attr = _ATTR[op]
    parent = getattr(layer, sub)
    lin = getattr(parent, attr)
    had = prec.endswith("+h")
    base = prec[:-2] if had else prec
    if base == "fp16":
        return 0
    if base == "fp8":
        from .fp8_linear import FP8Linear
        setattr(parent, attr, FP8Linear.from_linear(lin, hadamard=had))
    elif base == "fp8a":                          # 按 M 自适应:prefill FP8 / decode fp16
        from .fp8_linear import StageFP8Linear
        setattr(parent, attr, StageFP8Linear.from_linear(lin))
    elif base == "marlin":
        from .marlin_linear import MarlinInt4Linear
        setattr(parent, attr, MarlinInt4Linear(lin, hadamard=had))
    else:
        raise ValueError(f"未知精度 {prec!r}(支持 fp16/fp8/fp8a/marlin,可带 +h 后缀)")
    return 1


def fuse_norm_fp8(model) -> int:
    """把 gate_up 的激活量化**折进 post_attention_layernorm**(消除下游那笔独立量化 launch)。

    micro-bench 探因:FP8 的 scaled_mm 本身常比 fp16 快,拖死它的是独立的激活量化 launch(~25us)。
    norm 本来就按行 reduce,顺手吐 fp8+per-token scale 几乎免费 → gate_up 免量化、bf16 直流进 silu_mul
    (无转换税)。只动 gate_up(输入来自 RMSNorm);o/down 输入非 norm,不动。

    在 **fp16 模型上调用**(gate_up 还是 nn.Linear):把 gate_up 建成 rowwise FP8(per-channel 权重
    scale,配 norm 的 per-token 激活 scale),并置 post_attention_layernorm.fp8_out。返回融合层数。"""
    import torch.nn as nn
    from .fp8_linear import FP8Linear
    n = 0
    for layer in model.layers:
        gu = layer.mlp.gate_up_proj
        if isinstance(gu, nn.Linear):
            layer.mlp.gate_up_proj = FP8Linear.from_linear(gu, rowwise=True)
        elif isinstance(gu, FP8Linear) and not gu.rowwise:
            raise ValueError("gate_up 已是 per-tensor FP8;融合需 rowwise,请在量化前调用 fuse_norm_fp8")
        layer.post_attention_layernorm.fp8_out = True
        n += 1
    return n


def apply_mixed(model, plan: dict | None = None) -> dict:
    """按 per-op plan 替换 model.layers 各 Linear。返回 {算子: (精度, 替换层数)} 供日志。"""
    plan = plan or DEFAULT_PLAN
    counts = {op: 0 for op in plan}
    for layer in model.layers:
        for op, prec in plan.items():
            counts[op] += _replace(layer, op, prec)
    return {op: (plan[op], counts[op]) for op in plan}
