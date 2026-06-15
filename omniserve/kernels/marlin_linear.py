"""可选的 int4 量化 Linear —— 复用 vLLM 的 Marlin kernel(omniserve 的扩展特性)。

把 LLM 里划算的大 GEMM(MLP 的 gate_up / down)换成 **int4 权重 + fp16 激活** 的 Marlin
GEMM:decode 读权重的字节降到 1/4(decode 是访存受限,直接转成加速)。micro-bench(M=24,
Qwen2-VL-2B 形状):gate_up 2.42x、down 1.61x;小 GEMM(qkv/o)0.85-0.96x 不划算、lm_head
tied 不动,都保持 fp16。Marlin 用 fp16 激活、**不需要激活量化**,所以小 GEMM 只是持平(不像
FP8 那样崩到 0.26x)——这也是 vLLM decode 量化选 Marlin 的原因。

⚠️ int4 是**降精度(有损)**,所以这是个**默认关闭的扩展特性**:
- fp16 仍是默认路径;
- **不参与和 vLLM 的公平对比**(对比恒为 fp16 vs fp16,量化只作"框架支持量化推理"的能力展示);
- 权重量化用 RTN(marlin_quantize 内部,无 GPTQ 校准)——作为特性展示够用,要更高质量需校准数据。
"""

from __future__ import annotations

import torch
from torch import nn

_MARLIN = None


def _marlin_api():
    """惰性 import vLLM 的 Marlin 工具(只有开了量化才需要,避免无谓依赖)。"""
    global _MARLIN
    if _MARLIN is None:
        from vllm.scalar_type import scalar_types
        from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
            marlin_quantize)
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            apply_gptq_marlin_linear, marlin_make_workspace_new, marlin_make_empty_zp)
        _MARLIN = dict(stype=scalar_types.uint4b8, quantize=marlin_quantize,
                       apply=apply_gptq_marlin_linear, make_ws=marlin_make_workspace_new,
                       make_zp=marlin_make_empty_zp)
    return _MARLIN


class MarlinInt4Linear(nn.Module):
    """int4 Marlin 替代一个 fp16 nn.Linear。在线把 fp16 权重量化打包成 Marlin 布局。"""

    def __init__(self, linear: nn.Linear, group_size: int = 128):
        super().__init__()
        m = _marlin_api()
        dev = linear.weight.device
        N, K = linear.weight.shape                 # nn.Linear 权重布局 [out, in]
        # marlin_quantize 要 [size_k, size_n] = [in, out]
        w = linear.weight.data.t().contiguous()
        _, qw, s, g_idx, sort_idx, _ = m["quantize"](w, m["stype"], group_size, False)
        self.register_buffer("qweight", qw)
        self.register_buffer("scales", s)
        self.register_buffer("g_idx", g_idx)
        self.register_buffer("sort_indices", sort_idx)
        self.register_buffer("zp", m["make_zp"](dev))
        self.register_buffer("workspace", m["make_ws"](dev))
        self.bias = linear.bias                    # gate_up/down 都无 bias,但保留通用性
        self.N, self.K = N, K

    def forward(self, x):
        m = _marlin_api()
        return m["apply"](x, self.qweight, self.scales, self.zp, self.g_idx,
                          self.sort_indices, self.workspace, m["stype"],
                          self.N, self.K, is_k_full=True, bias=self.bias)


def quantize_llm_marlin(model, scope: str = "decoder") -> int:
    """把 LLM 的 Linear 换成 int4 Marlin。返回替换的 GEMM 数。
    - scope="decoder"(默认,**和 vLLM GPTQ 同范围**):每层 qkv/o/gate_up/down 全量化,
      lm_head + ViT 保持 fp16(标准 GPTQ 范围,公平对比用这个)。
    - scope="mlp":只量化划算的大 GEMM(gate_up/down),qkv/o 保持 fp16
      (micro-bench 上 qkv int4 0.85x、o 0.96x,选择性量化精度更高;但范围和 vLLM 不同)。"""
    n = 0
    for layer in model.layers:
        if scope == "decoder":
            layer.self_attn.qkv_proj = MarlinInt4Linear(layer.self_attn.qkv_proj)
            layer.self_attn.o_proj = MarlinInt4Linear(layer.self_attn.o_proj)
            n += 2
        layer.mlp.gate_up_proj = MarlinInt4Linear(layer.mlp.gate_up_proj)
        layer.mlp.down_proj = MarlinInt4Linear(layer.mlp.down_proj)
        n += 2
    return n
