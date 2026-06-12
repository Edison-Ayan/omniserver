"""预分配的批量 KV cache(PagedAttention 的思路,但不用写 kernel)。

cat-based 的 DynamicCache 每个 decode 步用 `torch.cat` 让 KV 增长(每步 O(L),整条
序列 O(L^2)),而我们的批量 decode 还额外每步重建整个批量 cache(legacy 往返 + pad +
cat + scatter)。这个类给每层预分配一个固定的 `[B, H, max_len, D]` buffer,把新 token
的 KV **原地**写到每个槽位的当前长度处——不增长、不重建。

槽位:每个在跑序列占一行 `b`。`lengths[b]` 是它有效的 KV 长度,也是下一个 token 写入的
位置。不同行长度不同(ragged),所以写入是逐行 scatter,注意力用 mask。

它只实现了 Qwen2-VL decode 前向需要的那部分 transformers Cache 接口(`update`、
`get_seq_length`)。和 cat-based 路径逐 token 一致(scripts/proto_prealloc_kv.py 验证)。
"""

from __future__ import annotations

import torch


class PreallocatedKVCache:
    # 经验:cat-based 的 DynamicCache 每步 torch.cat 把整段 KV 拷一遍(每步 O(L),
    # 整条 O(L²)),profile 显示 decode 单步发了 ~15000 个 kernel,大头是这些 copy/cat。
    # 改成预分配固定 buffer + 原地 scatter 写入后,kernel 数降到 ~3100,decode +15%,
    # 而且把瓶颈从 launch-bound 推到了 GPU-bound(详见 OPTIMIZATION.md)。
    def __init__(self, n_layers: int, max_batch: int, n_kv_heads: int,
                 max_len: int, head_dim: int, device, dtype):
        self.n_layers = n_layers
        self.max_len = max_len
        self.k = [torch.zeros(max_batch, n_kv_heads, max_len, head_dim,
                              device=device, dtype=dtype) for _ in range(n_layers)]
        self.v = [torch.zeros(max_batch, n_kv_heads, max_len, head_dim,
                              device=device, dtype=dtype) for _ in range(n_layers)]
        # 每个槽位的有效长度 == 下一个 token 的写入位置
        self.lengths = torch.zeros(max_batch, dtype=torch.long, device=device)
        self._active = 0  # 在用的槽位数(行 [0, _active) 是活的)

    # ---- 填充(prefill)------------------------------------------------------
    def set_slot_prefix(self, slot: int, legacy_kv, length: int) -> None:
        """把一个 prefill 结果(to_legacy_cache() 的 tuple)拷进某个槽位。"""
        for li, (k, v) in enumerate(legacy_kv):
            self.k[li][slot, :, :length, :] = k[0]
            self.v[li][slot, :, :length, :] = v[0]
        self.lengths[slot] = length
        self._active = max(self._active, slot + 1)

    def move_slot(self, src: int, dst: int) -> None:
        """把一个序列的 KV 从槽位 `src` 移到 `dst`(中间某序列跑完时用来紧凑化池)。
        每层拷贝整行。"""
        for li in range(self.n_layers):
            self.k[li][dst].copy_(self.k[li][src])
            self.v[li][dst].copy_(self.v[li][src])
        self.lengths[dst] = self.lengths[src]

    def view(self, n: int):
        """返回一个绑定到前 `n` 个活跃槽位的 cache view,供一次前向使用。"""
        return _CacheView(self, n)

    def prefill_cache(self, slot: int):
        """给原生 forward 用的 cache 适配器,把一个序列的 prefill KV 写进 `slot`
        (cache.update(k, v, layer_idx) 约定)。"""
        return _PrefillWriter(self, slot)

    def packed_prefill_cache(self, slots, seg_lengths):
        """打包式 prefill 的 cache 适配器:一次前向打包了多个序列的 token,
        update 时把每段的 KV 切出来写进对应槽位。块对角 mask 保证段间互不注意。"""
        return _PackedPrefillWriter(self, slots, seg_lengths)


class _PackedPrefillWriter:
    def __init__(self, pool: PreallocatedKVCache, slots, seg_lengths):
        self.pool, self.slots, self.seg = pool, slots, seg_lengths

    def update(self, k, v, layer_idx):
        # k, v: [1, n_kv_heads, total, head_dim](所有段打包在一起,rope 已应用)
        off = 0
        for slot, L in zip(self.slots, self.seg):
            self.pool.k[layer_idx][slot, :, :L, :] = k[0, :, off:off + L, :]
            self.pool.v[layer_idx][slot, :, :L, :] = v[0, :, off:off + L, :]
            off += L
        if layer_idx == self.pool.n_layers - 1:
            for slot, L in zip(self.slots, self.seg):
                self.pool.lengths[slot] = L
        return k, v  # 注意力在打包序列上做,段内/段间由 mask 控制


class _PrefillWriter:
    def __init__(self, pool: PreallocatedKVCache, slot: int):
        self.pool, self.slot = pool, slot

    def update(self, k, v, layer_idx):
        # k, v: [1, n_kv_heads, L, head_dim](rope 已由调用方应用)
        L = k.shape[2]
        self.pool.k[layer_idx][self.slot, :, :L, :] = k[0]
        self.pool.v[layer_idx][self.slot, :, :L, :] = v[0]
        if layer_idx == self.pool.n_layers - 1:
            self.pool.lengths[self.slot] = L
        return k, v  # prefill 注意它自己的 K/V


class _CacheView:
    """作为 `past_key_values` 传入的「单次前向」适配器。`update` 把新 token 原地写到
    每个活跃槽位的长度处,并返回有效的 KV 窗口。"""

    def __init__(self, pool: PreallocatedKVCache, n: int):
        self.pool = pool
        self.n = n
        self.write_pos = pool.lengths[:n].clone()      # 这一步要写入的位置
        self.ret_len = int(self.write_pos.max().item()) + 1
        self._rows = torch.arange(n, device=pool.lengths.device)

    def update(self, key, value, layer_idx, cache_kwargs=None):
        # key/value: [n, H, 1, D](每个槽位一个新 token)
        k_buf, v_buf = self.pool.k[layer_idx], self.pool.v[layer_idx]
        k_buf[self._rows, :, self.write_pos, :] = key[:, :, 0, :]
        v_buf[self._rows, :, self.write_pos, :] = value[:, :, 0, :]
        if layer_idx == self.pool.n_layers - 1:
            self.pool.lengths[:self.n] = self.write_pos + 1
        return k_buf[:self.n, :, :self.ret_len, :], v_buf[:self.n, :, :self.ret_len, :]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.ret_len - 1

    def get_mask_sizes(self, cache_position, layer_idx: int = 0):
        # (kv_length, kv_offset):返回的 KV 窗口有 ret_len 个 key,offset 为 0
        return self.ret_len, 0

    def get_max_cache_shape(self):
        return self.pool.max_len
