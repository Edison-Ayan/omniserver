"""omniserve 的 Prefix KV cache(前缀 KV 缓存)。

vision cache 只跳过 ViT,而实测显示 ViT 只占 prefill 的一小块——所以在高复用流量下
几乎没用。真正的杠杆(vLLM 的 prefix cache、SGLang 的 radix cache 干的事)是复用
**整段 prefill**:相同 prompt 前缀再次出现时,直接复用它产生的 KV,而不是把整个
prefill 前向重算一遍。

这里是个刻意简单的变体:**精确整段前缀匹配**。key 是 (prompt token id + 图像内容 hash)
的内容 hash。命中时给新序列 clone 缓存的 KV、完全跳过 prefill,直接进 decode。我们
**不做** SGLang 那种 token 级部分前缀(radix 树)匹配——那能覆盖更多情况但复杂得多;
精确匹配已经覆盖了主要的复用模式(同图 + 同 prompt)。

正确性:命中必须是**精确**前缀匹配,否则会用别的请求的 KV 给序列做种子、产生错误输出。
从相同前缀做 greedy 解码是确定性的,所以复用缓存的第一个 token + KV 和重算逐 token 一致。
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional

from .vision import CacheStats


@dataclass
class PrefixEntry:
    """不重跑 prefill 就能恢复一个序列所需的全部信息。"""
    cache: object        # 装着 prefill KV 的 DynamicCache(长度 `length`)
    rope_delta: int      # prefill 时产生的 Qwen2-VL M-RoPE delta
    length: int          # prompt token 数(KV 长度)
    first_token: int     # 从 prefill logits 采样出的 token


class PrefixKVCache:
    def __init__(self, max_entries: int = 64):
        self.max_entries = max_entries
        self._store: "OrderedDict[str, PrefixEntry]" = OrderedDict()
        self.stats = CacheStats()

    @staticmethod
    def prefix_key(prompt_token_ids: List[int], image_keys: Optional[List[str]]) -> str:
        """整段 prompt 前缀的内容 hash:文本 token 加上各图像的内容 hash
        (这样只有文本和图都相同的两个请求才会撞 key)。"""
        h = hashlib.sha1()
        h.update(repr(tuple(prompt_token_ids)).encode())
        if image_keys:
            for k in image_keys:
                h.update(b"|")
                h.update(k.encode())
        return h.hexdigest()

    def get(self, key: str, clone_fn) -> Optional[PrefixEntry]:
        """查一个前缀。命中时返回一个 entry,其 KV 是独立的 clone(这样请求方序列
        可以往自己的 cache 追加而不破坏共享的缓存副本)。`clone_fn(cache) -> cache`
        会深拷贝 KV。"""
        entry = self._store.get(key)
        if entry is None:
            self.stats.misses += 1
            return None
        self._store.move_to_end(key)
        self.stats.hits += 1
        return PrefixEntry(cache=clone_fn(entry.cache), rope_delta=entry.rope_delta,
                           length=entry.length, first_token=entry.first_token)

    def put(self, key: str, entry: PrefixEntry) -> None:
        """存一个 prefill 结果。调用方传入的 entry 里 `cache` 是它不会改的 clone,
        这样缓存副本保持干净。"""
        self._store[key] = entry
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
            self.stats.evictions += 1

    def clear(self) -> None:
        """清空所有 entry 并重置统计(用于冷启动 benchmark)。"""
        self._store.clear()
        self.stats = CacheStats()
