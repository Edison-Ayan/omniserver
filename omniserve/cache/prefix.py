"""Prefix KV cache for omniserve.

The vision cache only skips the ViT, which measurements showed is a small slice
of prefill — so on reuse-heavy traffic it barely helps. The real lever (what
vLLM's prefix cache and SGLang's radix cache do) is to reuse the *whole prefill*:
when an identical prompt prefix recurs, reuse the KV it produced instead of
recomputing the entire prefill forward.

This is a deliberately simple variant: **exact whole-prefix matching**. The key
is a content hash of (prompt token ids + image content hashes). On a hit we clone
the cached KV for the new sequence and skip prefill entirely, jumping straight to
decode. We do NOT do SGLang-style token-level partial-prefix (radix tree)
matching — that captures more cases but is much more complex; exact matching
already captures the dominant reuse pattern (same image + same prompt).

Correctness: a hit must be an *exact* prefix match, otherwise we would seed a
sequence with another request's KV and produce wrong output. Greedy decoding from
an identical prefix is deterministic, so reusing the cached first token + KV is
token-identical to recomputing.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional

from .vision import CacheStats


@dataclass
class PrefixEntry:
    """Everything needed to resume a sequence without re-running prefill."""
    cache: object        # a DynamicCache holding the prefill KV (length `length`)
    rope_delta: int      # Qwen2-VL M-RoPE delta produced at prefill
    length: int          # number of prompt tokens (KV length)
    first_token: int     # the token sampled from the prefill logits


class PrefixKVCache:
    def __init__(self, max_entries: int = 64):
        self.max_entries = max_entries
        self._store: "OrderedDict[str, PrefixEntry]" = OrderedDict()
        self.stats = CacheStats()

    @staticmethod
    def prefix_key(prompt_token_ids: List[int], image_keys: Optional[List[str]]) -> str:
        """Content hash of the full prompt prefix: the text tokens plus the
        content hashes of any images (so two requests collide only when both the
        text and the images are identical)."""
        h = hashlib.sha1()
        h.update(repr(tuple(prompt_token_ids)).encode())
        if image_keys:
            for k in image_keys:
                h.update(b"|")
                h.update(k.encode())
        return h.hexdigest()

    def get(self, key: str, clone_fn) -> Optional[PrefixEntry]:
        """Look up a prefix. On a hit, return an entry whose KV is an independent
        clone (so the requesting sequence can grow its cache without corrupting
        the shared cached copy). `clone_fn(cache) -> cache` deep-copies the KV."""
        entry = self._store.get(key)
        if entry is None:
            self.stats.misses += 1
            return None
        self._store.move_to_end(key)
        self.stats.hits += 1
        return PrefixEntry(cache=clone_fn(entry.cache), rope_delta=entry.rope_delta,
                           length=entry.length, first_token=entry.first_token)

    def put(self, key: str, entry: PrefixEntry) -> None:
        """Store a prefill result. The caller passes an entry whose `cache` is a
        clone it will not mutate, so the cached copy stays pristine."""
        self._store[key] = entry
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
            self.stats.evictions += 1
