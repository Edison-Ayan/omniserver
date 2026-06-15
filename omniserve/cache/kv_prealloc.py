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
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        # 布局 [B, max_len, nkv, hd]:正好是 flash 的 block 布局([num_blocks, block_size,
        # nkv, hd],每槽位一个 block),decode 时直接喂 flash、免 transpose+contiguous;
        # 且 KV 写入可用单维 index_copy_(flat=row*max_len+pos),对 CUDA graph 友好。
        self.k = [torch.zeros(max_batch, max_len, n_kv_heads, head_dim,
                              device=device, dtype=dtype) for _ in range(n_layers)]
        self.v = [torch.zeros(max_batch, max_len, n_kv_heads, head_dim,
                              device=device, dtype=dtype) for _ in range(n_layers)]
        # 每个槽位的有效长度 == 下一个 token 的写入位置
        self.lengths = torch.zeros(max_batch, dtype=torch.long, device=device)
        self._active = 0  # 在用的槽位数(行 [0, _active) 是活的)

    # ---- 填充(prefill)------------------------------------------------------
    def set_slot_prefix(self, slot: int, legacy_kv, length: int) -> None:
        """把一个 prefill 结果(to_legacy_cache() 的 tuple)拷进某个槽位。"""
        for li, (k, v) in enumerate(legacy_kv):
            self.k[li][slot, :length, :, :] = k[0].transpose(0, 1)   # [nkv,L,hd]->[L,nkv,hd]
            self.v[li][slot, :length, :, :] = v[0].transpose(0, 1)
        self.lengths[slot] = length
        self._active = max(self._active, slot + 1)

    def move_slot(self, src: int, dst: int) -> None:
        """把一个序列的 KV 从槽位 `src` 移到 `dst`(中间某序列跑完时用来紧凑化池)。
        每层拷贝整行。"""
        for li in range(self.n_layers):
            self.k[li][dst].copy_(self.k[li][src])
            self.v[li][dst].copy_(self.v[li][src])
        self.lengths[dst] = self.lengths[src]

    # ---- prefix cache 支持:槽位 KV 的快照/恢复/复制 ------------------------
    def snapshot_slot(self, slot: int, length: int):
        """把某槽位前 `length` 个 token 的 KV 克隆出来(脱离池 buffer),供 prefix
        cache 跨 step 保存。返回 (ks, vs):每层一个 [length, nkv, hd] 张量。"""
        ks = [self.k[li][slot, :length, :, :].clone() for li in range(self.n_layers)]
        vs = [self.v[li][slot, :length, :, :].clone() for li in range(self.n_layers)]
        return ks, vs

    def restore_slot(self, slot: int, snapshot, length: int) -> None:
        """把一份快照(snapshot_slot 产物)写进某槽位,跳过整段 prefill。快照本身只读、
        不被修改,所以同一快照能恢复到多个槽位、各自独立 decode。"""
        ks, vs = snapshot
        for li in range(self.n_layers):
            self.k[li][slot, :length, :, :] = ks[li]
            self.v[li][slot, :length, :, :] = vs[li]
        self.lengths[slot] = length
        self._active = max(self._active, slot + 1)

    def copy_slot_kv(self, src: int, dst: int, length: int) -> None:
        """池内槽位到槽位直接拷贝前 `length` 个 token 的 KV(batch 内同前缀去重用,
        不必经过 clone 快照)。"""
        for li in range(self.n_layers):
            self.k[li][dst, :length, :, :] = self.k[li][src, :length, :, :]
            self.v[li][dst, :length, :, :] = self.v[li][src, :length, :, :]
        self.lengths[dst] = length
        self._active = max(self._active, dst + 1)

    def view(self, n: int):
        """返回一个绑定到前 `n` 个活跃槽位的 cache view,供一次前向使用。"""
        return _CacheView(self, n)

    def view_graph(self, n: int, write_pos, flat, seqused):
        """给 CUDA graph 用的静态 view:write_pos / flat / seqused 都是外部 buffer(图外更新)。"""
        return _CacheView(self, n, write_pos=write_pos, flat=flat, seqused=seqused)

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

    def flash_prefill(self, q, k, v, layer_idx, scale):
        """用 FlashAttention varlen 做 prefill 注意力:把打包的各段 prompt 用 cu_seqlens
        分隔,只算段内因果注意力(O(sum L²),不像块对角 SDPA 那样 O(total²) 浪费)。
        q/k/v: [total, nh/nkv, hd]。各段 KV 写进各自池槽位。"""
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func
        pool = self.pool
        # 各段 KV 写进槽位(池布局 [slot, max_len, nkv, hd]);k/v 已是 [L,nkv,hd],直接写、免 transpose
        off = 0
        for slot, L in zip(self.slots, self.seg):
            pool.k[layer_idx][slot, :L, :, :] = k[off:off + L]
            pool.v[layer_idx][slot, :L, :, :] = v[off:off + L]
            off += L
        cu = torch.zeros(len(self.seg) + 1, device=q.device, dtype=torch.int32)
        cu[1:] = torch.tensor(self.seg, device=q.device, dtype=torch.int32).cumsum(0)
        max_seg = max(self.seg)
        out = flash_attn_varlen_func(
            q.contiguous(), k.contiguous(), v.contiguous(),
            max_seqlen_q=max_seg, cu_seqlens_q=cu, max_seqlen_k=max_seg, cu_seqlens_k=cu,
            softmax_scale=scale, causal=True)              # [total, nh, hd]
        if layer_idx == pool.n_layers - 1:
            for slot, L in zip(self.slots, self.seg):
                pool.lengths[slot] = L
        return out

    def update(self, k, v, layer_idx):
        # k, v: [1, n_kv_heads, total, head_dim](所有段打包在一起,rope 已应用)
        off = 0
        for slot, L in zip(self.slots, self.seg):
            self.pool.k[layer_idx][slot, :L, :, :] = k[0, :, off:off + L, :].transpose(0, 1)
            self.pool.v[layer_idx][slot, :L, :, :] = v[0, :, off:off + L, :].transpose(0, 1)
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
        self.pool.k[layer_idx][self.slot, :L, :, :] = k[0].transpose(0, 1)   # [nkv,L,hd]->[L,nkv,hd]
        self.pool.v[layer_idx][self.slot, :L, :, :] = v[0].transpose(0, 1)
        if layer_idx == self.pool.n_layers - 1:
            self.pool.lengths[self.slot] = L
        return k, v  # prefill 注意它自己的 K/V


class _CacheView:
    """作为 `past_key_values` 传入的「单次前向」适配器。`update` 把新 token 原地写到
    每个活跃槽位的长度处,并返回有效的 KV 窗口。"""

    def __init__(self, pool: PreallocatedKVCache, n: int, write_pos=None, flat=None, seqused=None):
        self.pool = pool
        self.n = n
        # graph 模式:write_pos / flat 都是外部 static buffer(图外更新值),不在图内改 lengths。
        # flat 索引(row*max_len+pos)也图外预算 —— 图内只做 index_copy_(0, flat, key),
        # 避免图内中间 tensor(_rows*ml)被 graph 内存池复用搞乱。
        self._graph = write_pos is not None
        if self._graph:
            self.write_pos = write_pos
            self._flat = flat
            self._seqused = seqused
            self.ret_len = pool.max_len
        else:
            self.write_pos = pool.lengths[:n].clone()  # 这一步要写入的位置
            self.ret_len = int(self.write_pos.max().item()) + 1
        self._rows = torch.arange(n, device=pool.lengths.device)

    def _write_kv(self, key, value, layer_idx):
        """新 token KV 写进 [row, write_pos, :, :]。用单维 index_copy_(flat=row*max_len+pos)
        而非 advanced index_put——单维 scatter 对 CUDA graph 友好。graph 模式用图外预算的
        flat buffer(避免图内中间 tensor 被内存池复用搞乱)。"""
        pool = self.pool
        flat = self._flat if self._graph else (self._rows * pool.max_len + self.write_pos)
        nkv, hd = pool.n_kv_heads, pool.head_dim
        pool.k[layer_idx].view(-1, nkv, hd).index_copy_(0, flat, key)
        pool.v[layer_idx].view(-1, nkv, hd).index_copy_(0, flat, value)

    def update(self, key, value, layer_idx, cache_kwargs=None):
        # key/value: [n, H, 1, D](每个槽位一个新 token)。SDPA fallback 用,返回 [n,nkv,L,hd]。
        self._write_kv(key[:, :, 0, :], value[:, :, 0, :], layer_idx)
        if layer_idx == self.pool.n_layers - 1:
            self.pool.lengths[:self.n] = self.write_pos + 1
        k_buf, v_buf = self.pool.k[layer_idx], self.pool.v[layer_idx]
        return (k_buf[:self.n, :self.ret_len].transpose(1, 2),
                v_buf[:self.n, :self.ret_len].transpose(1, 2))

    def flash_decode(self, q, key, value, layer_idx, scale):
        """用 vLLM 自带的 FlashAttention 做 decode 注意力(GQA-native,不物化 KV)。
        q: [n, nh, hd];key/value: [n, nkv, hd](新 token)。新布局 [B,max_len,nkv,hd] 正好是
        flash 的 block 布局,把新 token 原地 index_copy_ 写进池后**直接喂 flash、免 transpose+
        contiguous**(每层都省,decode 大头之一);block_table 让 flash 直接读池,seqused_k 限有效长度。"""
        from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func
        pool, n = self.pool, self.n
        self._write_kv(key, value, layer_idx)                  # graph 友好的单维写入
        kc = pool.k[layer_idx][:n]                             # [n, max_len, nkv, hd] 直接喂,免拷贝
        vc = pool.v[layer_idx][:n]
        cu_q = torch.arange(0, n + 1, device=q.device, dtype=torch.int32)
        seqused_k = self._seqused if self._graph else (self.write_pos + 1).to(torch.int32)
        block_table = self._rows.view(n, 1).to(torch.int32)
        out = flash_attn_varlen_func(
            q.contiguous(), kc, vc, max_seqlen_q=1, cu_seqlens_q=cu_q,
            max_seqlen_k=pool.max_len, seqused_k=seqused_k, block_table=block_table,
            softmax_scale=scale, causal=False)            # [n, nh, hd]
        if layer_idx == pool.n_layers - 1 and not self._graph:
            pool.lengths[:n] = self.write_pos + 1          # graph 模式由 runner 在图外更新
        return out

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.ret_len - 1

    def get_mask_sizes(self, cache_position, layer_idx: int = 0):
        # (kv_length, kv_offset):返回的 KV 窗口有 ret_len 个 key,offset 为 0
        return self.ret_len, 0

    def get_max_cache_shape(self):
        return self.pool.max_len
