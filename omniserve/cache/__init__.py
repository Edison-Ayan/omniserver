"""Multimodal caching components.

`vision` holds the vision-embedding cache (skip the ViT for repeated images).
Future content-addressed / prefix reuse lives here too.
"""

from .prefix import PrefixEntry, PrefixKVCache
from .vision import CacheStats, VisionEmbeddingCache

__all__ = ["VisionEmbeddingCache", "CacheStats", "PrefixKVCache", "PrefixEntry"]
