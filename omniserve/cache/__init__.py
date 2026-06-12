"""多模态缓存组件 + KV 管理。

`vision` 是 vision-embedding cache(重复图跳过 ViT);`prefix` 是前缀 KV cache
(重复前缀跳过整段 prefill);`kv_prealloc` 是预分配 KV 池(原地写,无 O(L²) 重建)。
"""

from .kv_prealloc import PreallocatedKVCache
from .prefix import PrefixEntry, PrefixKVCache
from .vision import CacheStats, VisionEmbeddingCache

__all__ = ["VisionEmbeddingCache", "CacheStats", "PrefixKVCache", "PrefixEntry",
           "PreallocatedKVCache"]
