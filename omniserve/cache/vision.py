"""Vision embedding cache for Qwen2-VL.

In a VLM the prefill cost has two parts: the vision encoder (ViT) turning pixels
into vision tokens, and the language model attending over text+vision tokens.
When the same image recurs across requests (multi-turn chat, repeated system
images) the ViT is recomputed even though its output is identical. We cache the
ViT output keyed by image content so repeated images skip the encoder.

Hook point (verified against transformers 4.57, Qwen2-VL): the model calls
    self.visual(pixel_values, grid_thw=image_grid_thw) -> image_embeds
where `model.visual` is the Qwen2VisionTransformerPretrainedModel. We monkeypatch
its `forward`. An LRU bound keeps cache memory flat.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass

import torch


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

    @staticmethod
    def _key(hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> str:
        h = hashlib.sha1()
        h.update(hidden_states.detach().to(torch.float16).cpu().numpy().tobytes())
        h.update(grid_thw.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def install(self, model) -> "VisionEmbeddingCache":
        visual = model.visual  # aliased to model.model.visual; same object
        self._visual = visual
        self._orig_forward = visual.forward

        def cached_forward(hidden_states, grid_thw, **kwargs):
            try:
                key = self._key(hidden_states, grid_thw)
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

    def uninstall(self):
        if self._visual is not None and self._orig_forward is not None:
            self._visual.forward = self._orig_forward
        self._store.clear()
