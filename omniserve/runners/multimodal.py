"""通用多模态 runner —— 模型无关的 prefill/decode 调度 + KV 池 + prefix cache + CUDA graph。

所有"这个模型怎么算"都委托给 VLMAdapter(见 models/base.py);换模型只换 adapter,本文件一行不动。
prefill 的打包/前缀复用、decode 的批量 flash、CUDA graph 都是通用机制,和模型无关。
"""

from __future__ import annotations

from typing import Dict, List

import torch

from ..cache.kv_prealloc import PreallocatedKVCache
from ..models.base import VLMAdapter
from ..request import Sequence
from .base import ModelRunner


class MultimodalRunner(ModelRunner):
    def __init__(self, adapter: VLMAdapter, max_running: int = 32, max_len: int = 1024,
                 prefix_cache=None, use_graph: bool = False):
        self.adapter = adapter
        self.device = adapter.device
        self.image_token_id = adapter.image_token_id
        self.eos_ids = adapter.eos_ids
        self.prefix_cache = prefix_cache
        self._img_keys: Dict[str, list] = {}
        # decode CUDA graph(消除 28 层 kernel 间隙);capture 用的 view 必须存活,见 _capture_graph。
        self._use_graph = use_graph
        self._graphs: Dict[int, tuple] = {}
        self._pending: Dict[str, dict] = {}
        # 多预留 1 个槽位(索引 max_running)给"正在分块 prefill 的长 prompt",和 decode 区
        # [0, n_dec) 完全隔离;分块算完再一次性 move 进 decode 区。
        self._reserve_slot = max_running
        self._pool = PreallocatedKVCache(
            n_layers=adapter.num_layers, max_batch=max_running + 1, n_kv_heads=adapter.num_kv_heads,
            max_len=max_len, head_dim=adapter.head_dim, device=self.device, dtype=torch.float16)
        self._slot_seq: List[Sequence] = []

    # ---- ModelRunner interface ---------------------------------------------
    def tokenize(self, seq: Sequence) -> None:
        req = seq.request
        pixel_values, grids = self.adapter.preprocess(req.images)
        input_ids = self.adapter.encode_prompt(req.prompt, grids)
        seq.prompt_token_ids = input_ids
        self._pending[seq.request_id] = {
            "ids": torch.tensor([input_ids], device=self.device),
            "pixel_values": pixel_values, "grids": grids}
        if self.prefix_cache is not None and req.images:
            from ..cache.vision import VisionEmbeddingCache
            self._img_keys[seq.request_id] = [
                VisionEmbeddingCache.image_key(im) for im in req.images]
        sp = seq.request.sampling
        for e in self.eos_ids:
            if e not in sp.stop_token_ids:
                sp.stop_token_ids.append(e)

    @torch.inference_mode()
    def prefill(self, seqs: List[Sequence]) -> None:
        """打包式 prefill + 前缀复用(compute 真算 / dup 同 batch 重复拷 KV / hit 跨 step 命中快照)。
        模型相关的只有 embed/vision/positions/llm 四处,全走 adapter。"""
        dev = self.device
        base = len(self._slot_seq)
        slots = [base + i for i in range(len(seqs))]

        # 1) 规划前缀复用
        plan, first_compute, compute_idxs = [], {}, []
        keys = [None] * len(seqs)
        for i, seq in enumerate(seqs):
            if self.prefix_cache is not None:
                keys[i] = self.prefix_cache.prefix_key(
                    seq.prompt_token_ids, self._img_keys.get(seq.request_id))
                if keys[i] in first_compute:
                    plan.append(("dup", first_compute[keys[i]]))
                    self.prefix_cache.stats.hits += 1
                    continue
                entry = self.prefix_cache.get(keys[i], lambda x: x)
                if entry is not None:
                    plan.append(("hit", entry)); continue
                first_compute[keys[i]] = i
            plan.append(("compute", None))
            compute_idxs.append(i)

        # 2) 只对 compute 的序列做 vision encode + 打包 LLM prefill
        L_of, delta_of, first_tok = {}, {}, {}
        if compute_idxs:
            embeds, positions, seg_lens, c_slots = [], [], [], []
            for i in compute_idxs:
                m = self._pending[seqs[i].request_id]
                ids = m["ids"]; L = ids.shape[1]; L_of[i] = L
                emb = self.adapter.embed_tokens(ids).clone()
                if m["pixel_values"] is not None:
                    emb[ids == self.image_token_id] = \
                        self.adapter.vision_embed(m["pixel_values"], m["grids"]).to(emb.dtype)
                pos, delta = self.adapter.prefill_positions(ids, m["grids"])
                embeds.append(emb); positions.append(pos)
                seg_lens.append(L); c_slots.append(slots[i]); delta_of[i] = delta

            packed_emb = torch.cat(embeds, dim=1)
            packed_pos = torch.cat(positions, dim=2)
            last_idx = torch.tensor(
                [sum(seg_lens[:j + 1]) - 1 for j in range(len(seg_lens))], device=dev)
            cache = self._pool.packed_prefill_cache(c_slots, seg_lens)
            logits = self.adapter.llm(packed_emb, packed_pos, None, cache, logits_indices=last_idx)[0]
            for j, i in enumerate(compute_idxs):
                first_tok[i] = self._sample(logits[j:j + 1], seqs[i])
                if keys[i] is not None:
                    from ..cache.prefix import PrefixEntry
                    self.prefix_cache.put(keys[i], PrefixEntry(
                        cache=self._pool.snapshot_slot(slots[i], L_of[i]),
                        rope_delta=delta_of[i], length=L_of[i], first_token=first_tok[i]))

        # 3) 落位
        for i, seq in enumerate(seqs):
            kind = plan[i][0]
            if kind == "compute":
                slot, rope_delta, tok = slots[i], delta_of[i], first_tok[i]
            elif kind == "dup":
                src = plan[i][1]
                self._pool.copy_slot_kv(slots[src], slots[i], L_of[src])
                slot, rope_delta, tok = slots[i], delta_of[src], first_tok[src]
            else:  # hit
                entry = plan[i][1]
                self._pool.restore_slot(slots[i], entry.cache, entry.length)
                slot, rope_delta, tok = slots[i], entry.rope_delta, entry.first_token
            self._pending.pop(seq.request_id, None)
            self._img_keys.pop(seq.request_id, None)
            self._slot_seq.append(seq)
            seq.kv_handle = {"slot": slot, "rope_delta": rope_delta}
            seq.append_token(tok)
            seq.maybe_finish()

    @torch.inference_mode()
    def chunk_prefill(self, seq: Sequence) -> None:
        """长 prompt 分块 prefill:每步算一块,append 到保留槽位 R(和 decode 区隔离),
        算完最后一块时吐首 token、把 KV 从 R 一次性 move 进 decode 区转为普通 decode 序列。
        期间正在跑的 decode 不被整段 prefill 阻塞(stall 被摊到多步)。MVP:纯文本路径。"""
        dev = self.device
        R = self._reserve_slot
        m = self._pending[seq.request_id]
        ids_full = m["ids"]                                   # [1, num_prompt]
        start = seq.num_prefilled
        L = seq.prefill_chunk
        end = start + L
        if start == 0:                                        # 首块:整段 M-RoPE 位置算一次,后续块切片复用
            assert m["pixel_values"] is None, "chunked prefill 暂只支持纯文本长 prompt(无图)"
            pos_full, delta = self.adapter.prefill_positions(ids_full, m["grids"])
            seq.kv_handle = {"slot": R, "rope_delta": delta, "pos_full": pos_full}
        pos_full = seq.kv_handle["pos_full"]
        emb = self.adapter.embed_tokens(ids_full[:, start:end])           # [1, L, hidden]
        pos_chunk = pos_full[:, :, start:end]                             # [3, 1, L]
        cache = self._pool.chunked_prefill_cache(R, start)
        logits = self.adapter.llm(emb, pos_chunk, None, cache, logits_indices=[L - 1])
        seq.num_prefilled = end
        if end < seq.num_prompt_tokens:
            return                                            # 还没算完,下步继续
        # 最后一块:落位到 decode 区(move 一次)+ 吐首 token
        n_dec = len(self._slot_seq)
        self._pool.move_slot(R, n_dec)
        seq.kv_handle = {"slot": n_dec, "rope_delta": seq.kv_handle["rope_delta"]}
        self._slot_seq.append(seq)
        self._pending.pop(seq.request_id, None)
        self._img_keys.pop(seq.request_id, None)
        seq.append_token(self._sample(logits[:, -1, :], seq))
        seq.maybe_finish()

    def decode(self, seqs: List[Sequence]) -> None:
        seqs = sorted(seqs, key=lambda s: s.kv_handle["slot"])
        n = len(seqs)
        assert all(seqs[b].kv_handle["slot"] == b for b in range(n))
        if self._use_graph:
            self._decode_graph(seqs, n)        # CUDA graph 不能在 inference_mode 内 capture/replay
        else:
            with torch.inference_mode():
                self._decode_eager(seqs, n)

    def _decode_eager(self, seqs, n):
        dev = self.device
        view = self._pool.view(n)
        ids = torch.tensor([[s.last_token_id()] for s in seqs], device=dev)
        rope = torch.tensor([s.kv_handle["rope_delta"] for s in seqs], device=dev)
        pos = self.adapter.decode_positions(view.write_pos, rope)
        emb = self.adapter.embed_tokens(ids)
        logits = self.adapter.llm(emb, pos, None, view)[:, -1, :]
        self._emit(seqs, logits)

    def _emit(self, seqs, logits):
        if all(s.request.sampling.greedy for s in seqs):
            toks = logits[:len(seqs)].argmax(dim=-1).tolist()
            for b, seq in enumerate(seqs):
                seq.append_token(toks[b]); seq.maybe_finish()
        else:
            for b, seq in enumerate(seqs):
                seq.append_token(self._sample(logits[b:b + 1], seq)); seq.maybe_finish()

    def _capture_graph(self, n):
        dev = self.device
        ml = self._pool.max_len
        rows = torch.arange(n, device=dev)
        L = self._pool.lengths[:n]
        buf = {"ids": torch.zeros(n, 1, dtype=torch.long, device=dev),
               "pos": torch.zeros(3, n, 1, dtype=torch.long, device=dev),
               "wpos": L.clone(), "flat": (rows * ml + L).clone(),
               "seqused": (L + 1).to(torch.int32), "rows": rows}
        view = self._pool.view_graph(n, buf["wpos"], buf["flat"], buf["seqused"])

        def fwd():
            emb = self.adapter.embed_tokens(buf["ids"])
            return self.adapter.llm(emb, buf["pos"], None, view)[:, -1, :]

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fwd()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            buf["logits"] = fwd()
        # view 必须存活——它的 _rows 等被 captured graph 引用,被 GC 则 replay illegal access。
        self._graphs[n] = (g, buf, view)

    def _decode_graph(self, seqs, n):
        dev = self.device
        if n not in self._graphs:
            self._capture_graph(n)
        g, buf, _view = self._graphs[n]
        ids = torch.tensor([[s.last_token_id()] for s in seqs], device=dev)
        rope = torch.tensor([s.kv_handle["rope_delta"] for s in seqs], device=dev)
        lengths = self._pool.lengths[:n]
        buf["ids"].copy_(ids)
        buf["wpos"].copy_(lengths)
        buf["flat"].copy_(buf["rows"] * self._pool.max_len + lengths)
        buf["seqused"].copy_((lengths + 1).to(torch.int32))
        buf["pos"].copy_(self.adapter.decode_positions(lengths, rope))
        g.replay()
        self._pool.lengths[:n] += 1
        self._emit(seqs, buf["logits"])

    def free(self, seq: Sequence) -> None:
        slot = seq.kv_handle.get("slot")
        if slot is None:
            return
        last = len(self._slot_seq) - 1
        if slot != last:
            self._pool.move_slot(last, slot)
            moved = self._slot_seq[last]
            moved.kv_handle["slot"] = slot
            self._slot_seq[slot] = moved
        self._slot_seq.pop()

    def detokenize(self, seq: Sequence, new_token_ids: List[int]) -> str:
        return self.adapter.detokenize(new_token_ids)

    def _sample(self, logits: torch.Tensor, seq: Sequence) -> int:
        sp = seq.request.sampling
        if sp.greedy:
            return int(logits.argmax(dim=-1).item())
        probs = torch.softmax(logits.float() / sp.temperature, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())
