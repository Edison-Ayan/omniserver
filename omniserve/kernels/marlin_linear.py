"""int4 量化 Linear —— 复用 vLLM 的 Marlin kernel(omniserve 的扩展特性)。

两种来源:
1. **MarlinInt4Linear**:在线 RTN 量化 fp16 权重(无校准)。只对 MLP 鲁棒,全量化 qkv/o 会
   乱码。用于"框架能不能在线量化"的轻量展示。
2. **GPTQMarlinLinear / load_gptq_llm**:加载**预量化的 GPTQ 校准权重**(同 vLLM 那份),
   repack 成 Marlin 布局。精度=vLLM,可全量化(qkv/o/gate_up/down)。这是和 vLLM int4
   **同精度公平对比**的路径。

decode 把读权重的字节降到 1/4(decode 访存受限,直接转成加速)。⚠️ int4 是降精度;默认 fp16,
量化是可选扩展。和 vLLM 的对比须同精度(int4 vs int4)。
"""

from __future__ import annotations

import torch
from torch import nn

_MARLIN = None


def _marlin_api():
    """惰性 import vLLM 的 Marlin 工具(只有开了量化才需要)。"""
    global _MARLIN
    if _MARLIN is None:
        from vllm import _custom_ops as ops
        from vllm.scalar_type import scalar_types
        from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
            marlin_quantize)
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            apply_gptq_marlin_linear, marlin_permute_scales, marlin_make_workspace_new,
            marlin_make_empty_zp, marlin_make_empty_g_idx)
        _MARLIN = dict(
            stype=scalar_types.uint4b8, quantize=marlin_quantize, apply=apply_gptq_marlin_linear,
            permute_scales=marlin_permute_scales, make_ws=marlin_make_workspace_new,
            make_zp=marlin_make_empty_zp, make_g=marlin_make_empty_g_idx, repack=ops.gptq_marlin_repack)
    return _MARLIN


class _MarlinBase(nn.Module):
    """持有 Marlin 布局权重 + 跑 apply_gptq_marlin_linear 的公共部分。"""

    def _register(self, mq, ms, bias, N, K, dev):
        m = _marlin_api()
        self.register_buffer("qweight", mq)
        self.register_buffer("scales", ms)
        self.register_buffer("g_idx", m["make_g"](dev))
        self.register_buffer("sort_indices", m["make_g"](dev))
        self.register_buffer("zp", m["make_zp"](dev))
        self.register_buffer("workspace", m["make_ws"](dev))
        self.bias = bias
        self.N, self.K = N, K

    def forward(self, x):
        m = _marlin_api()
        # bias 在 Marlin kernel 外手动加(实测把 bias 传进 gptq_marlin_gemm 结果不对)
        out = m["apply"](x, self.qweight, self.scales, self.zp, self.g_idx, self.sort_indices,
                         self.workspace, m["stype"], self.N, self.K, is_k_full=True)
        return out if self.bias is None else out + self.bias


class MarlinInt4Linear(_MarlinBase):
    """在线 RTN 量化一个 fp16 nn.Linear(无校准)。"""

    def __init__(self, linear: nn.Linear, group_size: int = 128):
        super().__init__()
        m = _marlin_api()
        dev = linear.weight.device
        N, K = linear.weight.shape                 # nn.Linear 权重 [out, in]
        w = linear.weight.data.t().contiguous()    # marlin_quantize 要 [size_k, size_n]
        _, qw, s, _, _, _ = m["quantize"](w, m["stype"], group_size, False)
        self._register(qw, s, linear.bias, N, K, dev)


class GPTQMarlinLinear(_MarlinBase):
    """加载预量化的 GPTQ(sym, desc_act=False)权重,repack 成 Marlin。
    qweight: [K//8, N] int32(AutoGPTQ 布局);scales: [K//group, N] fp16。"""

    def __init__(self, qweight, scales, bias, group_size: int = 128):
        super().__init__()
        m = _marlin_api()
        dev = qweight.device
        K = qweight.shape[0] * 8
        N = qweight.shape[1]
        perm = m["make_g"](dev)                    # desc_act=False -> 不重排
        mq = m["repack"](qweight.contiguous(), perm=perm, size_k=K, size_n=N, num_bits=4)
        ms = m["permute_scales"](scales.contiguous(), size_k=K, size_n=N, group_size=group_size)
        self._register(mq, ms, bias, N, K, dev)


def quantize_llm_marlin(model, scope: str = "decoder") -> int:
    """在线 RTN 量化(无校准)。scope="decoder"=全 decoder linear,"mlp"=只 gate_up/down。
    ⚠️ RTN 全量化 qkv/o 会乱码,要全量化用 load_gptq_llm(校准权重)。"""
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


def load_gptq_llm(model, sd: dict, device, group_size: int = 128) -> int:
    """把 GPTQ checkpoint 的 LLM 权重加载进 omniserve Qwen2LLM:
    - decoder linear(qkv/o/gate_up/down)-> GPTQMarlinLinear(融合 qkv、gate_up,沿 N concat);
    - embed_tokens / norm fp16;lm_head 和 embed tied。
    返回替换的 GEMM 数。和 vLLM 同精度(同一份校准权重)。"""
    g = lambda k: sd[k].to(device)
    model.embed_tokens.weight.data.copy_(g("model.embed_tokens.weight"))
    model.norm.weight.data.copy_(g("model.norm.weight"))
    n = 0
    for i, layer in enumerate(model.layers):
        lp = f"model.layers.{i}."
        # 两个 RMSNorm 是 fp16(不量化),必须从 checkpoint 加载,否则保持初始化的全 1 → 乱码
        layer.input_layernorm.weight.data.copy_(g(lp + "input_layernorm.weight"))
        layer.post_attention_layernorm.weight.data.copy_(g(lp + "post_attention_layernorm.weight"))
        a = layer.self_attn
        # 融合 qkv:沿 N(dim=1)concat q/k/v 的 qweight/scales/bias(各自带 scale,Marlin 支持)
        qkv_qw = torch.cat([g(lp + f"self_attn.{x}_proj.qweight") for x in "qkv"], dim=1)
        qkv_s = torch.cat([g(lp + f"self_attn.{x}_proj.scales") for x in "qkv"], dim=1)
        qkv_b = torch.cat([g(lp + f"self_attn.{x}_proj.bias") for x in "qkv"], dim=0).half()
        a.qkv_proj = GPTQMarlinLinear(qkv_qw, qkv_s, qkv_b, group_size)
        a.o_proj = GPTQMarlinLinear(g(lp + "self_attn.o_proj.qweight"),
                                    g(lp + "self_attn.o_proj.scales"), None, group_size)
        # 融合 gate_up:沿 N concat gate/up
        gu_qw = torch.cat([g(lp + f"mlp.{x}_proj.qweight") for x in ("gate", "up")], dim=1)
        gu_s = torch.cat([g(lp + f"mlp.{x}_proj.scales") for x in ("gate", "up")], dim=1)
        layer.mlp.gate_up_proj = GPTQMarlinLinear(gu_qw, gu_s, None, group_size)
        layer.mlp.down_proj = GPTQMarlinLinear(g(lp + "mlp.down_proj.qweight"),
                                               g(lp + "mlp.down_proj.scales"), None, group_size)
        n += 4
    return n
