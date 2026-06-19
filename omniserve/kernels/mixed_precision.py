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
    """把单层的一个算子换成指定精度。fp16=不动,返回是否替换(0/1)。"""
    sub, attr = _ATTR[op]
    parent = getattr(layer, sub)
    lin = getattr(parent, attr)
    if prec == "fp16":
        return 0
    if prec == "fp8":
        from .fp8_linear import FP8Linear
        setattr(parent, attr, FP8Linear.from_linear(lin))
    elif prec == "marlin":
        from .marlin_linear import MarlinInt4Linear
        setattr(parent, attr, MarlinInt4Linear(lin))
    else:
        raise ValueError(f"未知精度 {prec!r}(支持 fp16/fp8/marlin)")
    return 1


def apply_mixed(model, plan: dict | None = None) -> dict:
    """按 per-op plan 替换 model.layers 各 Linear。返回 {算子: (精度, 替换层数)} 供日志。"""
    plan = plan or DEFAULT_PLAN
    counts = {op: 0 for op in plan}
    for layer in model.layers:
        for op, prec in plan.items():
            counts[op] += _replace(layer, op, prec)
    return {op: (plan[op], counts[op]) for op in plan}
