"""Preallocated batched KV cache (the spirit of PagedAttention without a kernel).

The cat-based DynamicCache grows the KV by `torch.cat` every decode step (O(L) a
step, O(L^2) over a sequence), and our batched decode additionally rebuilt the
whole batched cache each step (legacy round-trip + pad + cat + scatter). This
class preallocates one fixed `[B, H, max_len, D]` buffer per layer and writes the
new token's KV **in place** at each slot's current length — no growth, no rebuild.

Slots: each running sequence owns a row `b`. `lengths[b]` is its valid KV length,
which is also where the next token is written. Different rows are at different
lengths (ragged), so writes are per-row scatters and attention uses a mask.

This implements just enough of the transformers Cache interface for Qwen2-VL's
decode forward (`update`, `get_seq_length`). It is token-identical to the
cat-based path (verified in scripts/proto_prealloc_kv.py).
"""

from __future__ import annotations

import torch


class PreallocatedKVCache:
    def __init__(self, n_layers: int, max_batch: int, n_kv_heads: int,
                 max_len: int, head_dim: int, device, dtype):
        self.n_layers = n_layers
        self.max_len = max_len
        self.k = [torch.zeros(max_batch, n_kv_heads, max_len, head_dim,
                              device=device, dtype=dtype) for _ in range(n_layers)]
        self.v = [torch.zeros(max_batch, n_kv_heads, max_len, head_dim,
                              device=device, dtype=dtype) for _ in range(n_layers)]
        # valid length per slot == write position for the next token
        self.lengths = torch.zeros(max_batch, dtype=torch.long, device=device)
        self._active = 0  # number of slots in use (rows [0, _active) are live)

    # ---- population (prefill) ------------------------------------------------
    def set_slot_prefix(self, slot: int, legacy_kv, length: int) -> None:
        """Copy a prefill result (a to_legacy_cache() tuple) into a slot."""
        for li, (k, v) in enumerate(legacy_kv):
            self.k[li][slot, :, :length, :] = k[0]
            self.v[li][slot, :, :length, :] = v[0]
        self.lengths[slot] = length
        self._active = max(self._active, slot + 1)

    def move_slot(self, src: int, dst: int) -> None:
        """Move a sequence's KV from slot `src` to `dst` (used to compact the pool
        when a sequence in the middle finishes). Copies the full row per layer."""
        for li in range(self.n_layers):
            self.k[li][dst].copy_(self.k[li][src])
            self.v[li][dst].copy_(self.v[li][src])
        self.lengths[dst] = self.lengths[src]

    def view(self, n: int):
        """Return a cache view bound to the first `n` active slots for a forward."""
        return _CacheView(self, n)


class _CacheView:
    """A per-forward adapter passed as `past_key_values`. `update` writes the new
    token in place at each active slot's length and returns the valid KV window."""

    def __init__(self, pool: PreallocatedKVCache, n: int):
        self.pool = pool
        self.n = n
        self.write_pos = pool.lengths[:n].clone()      # position to write this step
        self.ret_len = int(self.write_pos.max().item()) + 1
        self._rows = torch.arange(n, device=pool.lengths.device)

    def update(self, key, value, layer_idx, cache_kwargs=None):
        # key/value: [n, H, 1, D] (single new token per slot)
        k_buf, v_buf = self.pool.k[layer_idx], self.pool.v[layer_idx]
        k_buf[self._rows, :, self.write_pos, :] = key[:, :, 0, :]
        v_buf[self._rows, :, self.write_pos, :] = value[:, :, 0, :]
        if layer_idx == self.pool.n_layers - 1:
            self.pool.lengths[:self.n] = self.write_pos + 1
        return k_buf[:self.n, :, :self.ret_len, :], v_buf[:self.n, :, :self.ret_len, :]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.ret_len - 1

    def get_mask_sizes(self, cache_position, layer_idx: int = 0):
        # (kv_length, kv_offset): the returned KV window has ret_len keys at offset 0
        return self.ret_len, 0

    def get_max_cache_shape(self):
        return self.pool.max_len
