"""Qwen2-VL 的 vision embedding cache。

在 VLM 里 prefill 成本有两部分:vision 编码器(ViT)把像素变成 vision token,以及
语言模型对 文本+vision token 做注意力。当同一张图在多个请求间重复出现(多轮对话、
重复的 system 图)时,ViT 会被重算,尽管输出完全一样。我们按图像内容做 key 缓存 ViT
输出,这样重复的图直接跳过编码器。

挂载点(对 transformers 4.57 / Qwen2-VL 验证过):模型会调用
    self.visual(pixel_values, grid_thw=image_grid_thw) -> image_embeds
其中 `model.visual` 是 Qwen2VisionTransformerPretrainedModel。我们 monkeypatch 它的
`forward`。LRU 上界让 cache 内存保持平稳。

内容寻址:cache key 是**图像内容**的 hash。runner 在请求到来时算一次(`image_key`),
每次 prefill 前通过 `set_pending` 交给 cache,这样查表是 O(1),不用每次 ViT 调用都
重新 hash 整个像素张量。没有预算 key 时(比如独立 benchmark 直接调模型),回退到对
像素张量做 hash。
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional

import torch
from PIL import Image


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def as_dict(self) -> dict:
        return {"hits": self.hits, "misses": self.misses,
                "evictions": self.evictions, "hit_rate": round(self.hit_rate, 4)}


class VisionEmbeddingCache:
    def __init__(self, max_entries: int = 64):
        self.max_entries = max_entries
        self._store: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.stats = CacheStats()
        self._orig_forward = None
        self._visual = None
        self._pending: Optional[List[str]] = None  # 下次前向用的内容 key

    @staticmethod
    def image_key(image: Image.Image) -> str:
        """图像的内容 hash,在请求 ingest 时算一次。"""
        return hashlib.sha1(image.tobytes()).hexdigest()

    def set_pending(self, keys: Optional[List[str]]) -> None:
        """提供即将到来的前向里各图像的内容 key。
        每张图一个 key,顺序和 processor 排布的一致。"""
        self._pending = keys

    @staticmethod
    def _pixel_key(hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> str:
        h = hashlib.sha1()
        h.update(hidden_states.detach().to(torch.float16).cpu().numpy().tobytes())
        h.update(grid_thw.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def install(self, model) -> "VisionEmbeddingCache":
        visual = model.visual  # 是 model.model.visual 的别名;同一个对象
        self._visual = visual
        self._orig_forward = visual.forward

        def cached_forward(hidden_states, grid_thw, **kwargs):
            # 只在预算 key 能无歧义对应这次调用(单张图)时用它;
            # 否则回退到对像素做 hash。
            key = None
            if self._pending is not None and len(self._pending) == 1 \
                    and grid_thw is not None and grid_thw.shape[0] == 1:
                key = self._pending[0]
            if key is None:
                try:
                    key = self._pixel_key(hidden_states, grid_thw)
                except Exception:
                    return self._orig_forward(hidden_states, grid_thw, **kwargs)

            cached = self._store.get(key)
            if cached is not None:
                self._store.move_to_end(key)
                self.stats.hits += 1
                return cached.to(hidden_states.device)

            out = self._orig_forward(hidden_states, grid_thw, **kwargs)
            self.stats.misses += 1
            self._store[key] = out.detach()
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)
                self.stats.evictions += 1
            return out

        visual.forward = cached_forward
        return self

    def clear(self) -> None:
        """丢掉缓存的 embedding 并重置统计(用于冷启动 benchmark)。
        保持 monkeypatch 仍然装着。"""
        self._store.clear()
        self.stats = CacheStats()

    def uninstall(self):
        if self._visual is not None and self._orig_forward is not None:
            self._visual.forward = self._orig_forward
        self._store.clear()
