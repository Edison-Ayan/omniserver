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


BLOCK = 16  # 最长前缀匹配的块粒度(对齐到块,避免半块匹配)


def block_hashes(token_ids: List[int]) -> List[str]:
    """累积块哈希:h_i 唯一标识 prompt 的前 (i+1)*BLOCK 个 token 这一段前缀
    (链式:h_i = hash(h_{i-1}, 第 i 块的 token))。只算满块,尾部不足一块不计。
    两个 prompt 的前缀块哈希一路相同 ⟺ 它们的 token 前缀逐位相同(忽略碰撞)。"""
    hs, parent = [], b""
    for i in range(len(token_ids) // BLOCK):
        h = hashlib.sha1(parent)
        h.update(repr(tuple(token_ids[i * BLOCK:(i + 1) * BLOCK])).encode())
        parent = h.digest()
        hs.append(h.hexdigest())
    return hs


@dataclass
class PrefixEntry:
    """不重跑 prefill 就能恢复一个序列所需的全部信息。"""
    cache: object        # 装着 prefill KV 的 DynamicCache(长度 `length`)
    rope_delta: int      # prefill 时产生的 Qwen2-VL M-RoPE delta
    length: int          # prompt token 数(KV 长度)
    first_token: int     # 从 prefill logits 采样出的 token
    block_hashes: Optional[List[str]] = None  # 用于最长前缀匹配的累积块哈希


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

    def match_prefix(self, token_ids: List[int]):
        """最长前缀匹配:在缓存里找和 token_ids 共享最长**块前缀**的 entry。
        返回 (entry, matched_len) —— matched_len 是对齐到块的可复用 token 数:
          == len 整段       → 精确命中(整段 KV 可复用,跳过 prefill);
          0 < matched < len → 部分命中(复用前 matched 个 token 的 KV,只 prefill 后缀);
          0(返回 None)     → 无可复用前缀。
        正确性:块哈希链一路相同 ⟺ token 前缀逐位相同,复用 KV[0:matched] 安全(因果注意力下
        前 matched 个 token 的 KV 不依赖其后的 token)。"""
        bh = block_hashes(token_ids)
        if not bh:
            return None
        best, best_n = None, 0
        for entry in self._store.values():
            ebh = entry.block_hashes
            if not ebh:
                continue
            n = 0
            for a, b in zip(bh, ebh):
                if a != b:
                    break
                n += 1
            if n > best_n:
                best, best_n = entry, n
        if best_n == 0:
            return None
        return best, best_n * BLOCK

    def clear(self) -> None:
        """清空所有 entry 并重置统计(用于冷启动 benchmark)。"""
        self._store.clear()
        self.stats = CacheStats()
