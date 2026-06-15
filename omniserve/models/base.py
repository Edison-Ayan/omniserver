"""VLMAdapter —— 模型特定逻辑的适配层。

通用的 `MultimodalRunner` 通过这个接口跑**任意**多模态模型:它只管 prefill/decode 的调度、
KV 池、prefix cache、CUDA graph(这些都和模型无关),把所有"这个模型怎么算"的事委托给 adapter。

加一个新模型 = 写一个新 adapter,实现下面这几件事。引擎/调度器/cache/graph 一行不用动。

adapter 要保证:
- `llm(...)` 遵循 KV 池的 cache 约定(cache 对象有 flash_prefill / flash_decode,见 kv_prealloc);
  decode 注意力走 cache.flash_decode,prefill 走 cache.flash_prefill(packed varlen)。
- `vision_embed` 的输出维度已对齐 LLM 的 hidden(connector/merger 在 adapter 内做完)。
- positions 的语义由模型自定(标准 RoPE 就返回 1D 序列位置;Qwen2-VL 这种 M-RoPE 返回 3D)。
"""

from __future__ import annotations

import abc
from typing import List, Optional, Tuple

import torch
from PIL import Image


class VLMAdapter(abc.ABC):
    # ---- KV 池需要的形状 + 特殊 token(子类在 __init__ 里设好)----
    num_layers: int
    num_kv_heads: int
    head_dim: int
    image_token_id: int
    eos_ids: set
    device: str = "cuda"

    # ---- 输入侧:图像预处理 + 文本 tokenize ----
    @abc.abstractmethod
    def preprocess(self, images: List[Image.Image]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """图像 -> (pixel_values, grids)。无图返回 (None, None)。grids 编码每图的网格尺寸。"""

    @abc.abstractmethod
    def encode_prompt(self, prompt: str, grids) -> List[int]:
        """prompt(+图占位)-> input_ids(含 chat 模板、image token 展开成对应数量的占位)。"""

    # ---- 编码:文本 embedding / 视觉 embedding ----
    @abc.abstractmethod
    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """token id -> 文本 embedding [.., hidden]。decode/graph 路径只用这个(无视觉)。"""

    @abc.abstractmethod
    def vision_embed(self, pixel_values: torch.Tensor, grids) -> torch.Tensor:
        """pixel_values -> 视觉 embedding [N_visual_tokens, hidden](已对齐 LLM hidden)。
        runner 会把它 scatter 到 input_ids==image_token_id 的位置。"""

    # ---- 位置编码(模型自定语义)----
    @abc.abstractmethod
    def prefill_positions(self, input_ids: torch.Tensor, grids) -> Tuple[torch.Tensor, int]:
        """prefill 的 position_ids [3,1,L] 和 rope_delta(decode 时复用)。
        标准 RoPE 的 adapter 可让三轴相同 = 序列位置,delta=0。"""

    @abc.abstractmethod
    def decode_positions(self, write_pos: torch.Tensor, rope_deltas: torch.Tensor) -> torch.Tensor:
        """decode 时每个槽位的 position_ids [3,n,1](由当前写入位置 + 各序列的 rope_delta 推出)。"""

    # ---- 前向 + 反 tokenize ----
    @abc.abstractmethod
    def llm(self, embeds, positions, attn_mask, cache, logits_indices=None) -> torch.Tensor:
        """LLM 前向。cache 是 KV 池适配器(flash_prefill/flash_decode 约定)。
        logits_indices 非空时只算这些位置的 logits(打包 prefill 省显存)。"""

    @abc.abstractmethod
    def detokenize(self, token_ids: List[int]) -> str:
        """token id -> 文本。"""

    # ---- 可选:GPTQ 等需要单独入口时重写 ----
    def supports_quant(self, quant: Optional[str]) -> bool:
        return quant is None
