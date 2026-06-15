"""原生 Qwen2-VL runner —— 零 transformers 依赖。

驱动全套自写实现:tokenizer(`tokenizers` 库)、图像预处理、ViT、M-RoPE 位置、
LLM forward,配合预分配 KV 池。权重直接从 safetensors 分片读。剩下的第三方只有
PyTorch(计算)、`tokenizers`(BPE)、`safetensors`(加载)和 PIL。
"""

from __future__ import annotations

import gc
import glob
import os
from typing import Dict, List

import torch
from safetensors.torch import load_file

from ..cache.kv_prealloc import PreallocatedKVCache
from ..model import Qwen2Config, Qwen2LLM, Qwen2VIT, load_from_hf, load_vit_from_state_dict
from ..model.positions import IMAGE_TOKEN_ID, mrope_position_ids
from ..model.preprocess import preprocess_image
from ..model.tokenizer import Qwen2VLTokenizer
from ..request import Sequence
from .base import ModelRunner

EOS_ID = 151645  # <|im_end|>
DEFAULT_SNAPSHOT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/*")


class NativeQwenVLRunner(ModelRunner):
    def __init__(self, model_dir: str = None, max_running: int = 32, max_len: int = 1024,
                 prefix_cache=None):
        model_dir = model_dir or glob.glob(DEFAULT_SNAPSHOT)[0]
        self.device = "cuda"
        self.tokenizer = Qwen2VLTokenizer(os.path.join(model_dir, "tokenizer.json"))
        self.image_token_id = IMAGE_TOKEN_ID
        self.eos_ids = {EOS_ID}
        # 可选 prefix KV cache:相同(prompt token + 图)复用整段 prefill。原生 runner 走
        # 预分配池,所以缓存的是池槽位的 KV 快照(不是 HF DynamicCache)。
        self.prefix_cache = prefix_cache
        self._img_keys: Dict[str, list] = {}

        sd = {}
        for f in glob.glob(os.path.join(model_dir, "model-*.safetensors")):
            sd.update(load_file(f))

        self.vit = Qwen2VIT()
        load_vit_from_state_dict(self.vit, sd)
        self.vit = self.vit.half().to(self.device).eval()

        self.llm = Qwen2LLM(Qwen2Config()).half()
        load_from_hf(self.llm, sd)
        self.llm = self.llm.to(self.device).eval()
        del sd
        gc.collect()
        torch.cuda.empty_cache()

        self._pending: Dict[str, dict] = {}
        cfg = self.llm.cfg
        self._pool = PreallocatedKVCache(
            n_layers=cfg.num_layers, max_batch=max_running, n_kv_heads=cfg.num_kv_heads,
            max_len=max_len, head_dim=cfg.head_dim, device=self.device, dtype=torch.float16)
        self._slot_seq: List[Sequence] = []

    # ---- ModelRunner interface ---------------------------------------------
    def tokenize(self, seq: Sequence) -> None:
        req = seq.request
        pvs, grids = [], []
        for im in req.images:
            pv, grid = preprocess_image(im)
            pvs.append(pv)
            grids.append(grid[0])
        input_ids = self.tokenizer.encode_prompt(req.prompt, grids)
        ids = torch.tensor([input_ids], device=self.device)
        pixel_values = torch.cat(pvs, 0).to(self.device).half() if pvs else None
        grid_thw = torch.stack(grids).to(self.device) if grids else None
        seq.prompt_token_ids = input_ids
        self._pending[seq.request_id] = {"ids": ids, "pixel_values": pixel_values, "grid_thw": grid_thw}
        # prefix cache 用:对图像内容做一次 hash,作为前缀 key 的一部分
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
        """打包式 prefill:把这一批序列的 prompt token 拼成一次前向,用 FlashAttention
        varlen 让每个 prompt 只注意自己内部。这样权重只读一次、GPU 喂满。每段的 KV 写进
        各自的池槽位。

        开了 prefix cache 时:相同 (prompt token + 图) 的请求复用整段 prefill——
        ① 跨 step 命中缓存的槽位快照(skip ViT+LLM);② 同一 batch 内的重复前缀只算一次,
        其余从已算的兄弟槽位拷贝(vLLM 也是靠前缀缓存吃到高复用收益,这一步把同样的杠杆接到
        原生快路径上)。命中是精确前缀匹配,greedy 确定性 → 复用的首 token+KV 和重算逐 token 一致。"""
        dev = self.device
        base = len(self._slot_seq)
        slots = [base + i for i in range(len(seqs))]   # 每个序列(命中/重算/重复)都占一个槽位

        # 1) 规划每个序列:'compute'(本 batch 首次算这个前缀)/ ('dup', 兄弟下标)/ ('hit', entry)
        plan = []
        first_compute = {}     # prefix_key -> 本 batch 内首个算它的下标
        compute_idxs = []      # 真正需要跑前向的下标
        keys = [None] * len(seqs)
        for i, seq in enumerate(seqs):
            if self.prefix_cache is not None:
                keys[i] = self.prefix_cache.prefix_key(
                    seq.prompt_token_ids, self._img_keys.get(seq.request_id))
                if keys[i] in first_compute:           # 同 batch 内重复前缀,也算命中
                    plan.append(("dup", first_compute[keys[i]]))
                    self.prefix_cache.stats.hits += 1
                    continue
                entry = self.prefix_cache.get(keys[i], lambda x: x)   # 快照只读,无需深拷;计 hit/miss
                if entry is not None:
                    plan.append(("hit", entry)); continue
                first_compute[keys[i]] = i
            plan.append(("compute", None))
            compute_idxs.append(i)

        # 2) 只对 compute 的序列做 ViT + 打包 LLM prefill
        L_of, delta_of, first_tok = {}, {}, {}
        if compute_idxs:
            embeds, positions, seg_lens, c_slots = [], [], [], []
            for i in compute_idxs:
                m = self._pending[seqs[i].request_id]
                ids = m["ids"]; L = ids.shape[1]; L_of[i] = L
                emb = self.llm.embed_tokens(ids).clone()
                if m["pixel_values"] is not None:
                    emb[ids == self.image_token_id] = self.vit(
                        m["pixel_values"], m["grid_thw"]).to(emb.dtype)
                pos, delta = mrope_position_ids(ids, m["grid_thw"]) if m["grid_thw"] is not None \
                    else self._text_positions(ids)
                embeds.append(emb)                 # [1, L, hidden]
                positions.append(pos)              # [3, 1, L]
                seg_lens.append(L); c_slots.append(slots[i])
                delta_of[i] = int(delta.flatten()[0].item())

            packed_emb = torch.cat(embeds, dim=1)          # [1, total, hidden]
            packed_pos = torch.cat(positions, dim=2)       # [3, 1, total]
            # 只取每段最后一个 token 的位置算 logits(避免 [total, vocab] 的巨大 OOM)。
            last_idx = torch.tensor(
                [sum(seg_lens[:j + 1]) - 1 for j in range(len(seg_lens))], device=dev)
            cache = self._pool.packed_prefill_cache(c_slots, seg_lens)
            logits = self.llm(packed_emb, packed_pos, None, cache, logits_indices=last_idx)[0]
            for j, i in enumerate(compute_idxs):
                first_tok[i] = self._sample(logits[j:j + 1], seqs[i])
                # 把算好的槽位快照存进 prefix cache,供后续 step / 别的 batch 复用
                if keys[i] is not None:
                    from ..cache.prefix import PrefixEntry
                    self.prefix_cache.put(keys[i], PrefixEntry(
                        cache=self._pool.snapshot_slot(slots[i], L_of[i]),
                        rope_delta=delta_of[i], length=L_of[i], first_token=first_tok[i]))

        # 3) 落位:compute 直接用;dup 从兄弟槽位拷 KV;hit 从快照恢复。全部跳过重算。
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

    @staticmethod
    def _text_positions(ids):
        L = ids.shape[1]
        pos = torch.arange(L, device=ids.device).view(1, 1, -1).expand(3, 1, -1)
        return pos, torch.zeros(1, 1, device=ids.device)

    @torch.inference_mode()
    def decode(self, seqs: List[Sequence]) -> None:
        seqs = sorted(seqs, key=lambda s: s.kv_handle["slot"])
        n = len(seqs)
        assert all(seqs[b].kv_handle["slot"] == b for b in range(n))
        dev = self.device
        view = self._pool.view(n)
        wpos, ret = view.write_pos, view.ret_len
        ids = torch.tensor([[s.last_token_id()] for s in seqs], device=dev)
        rope = torch.tensor([s.kv_handle["rope_delta"] for s in seqs], device=dev)
        pos = torch.empty(3, n, 1, device=dev, dtype=torch.long)
        pos[:, :, 0] = (wpos + rope).unsqueeze(0)
        mask = torch.full((n, 1, 1, ret), float("-inf"), device=dev, dtype=torch.float16)
        for b in range(n):
            mask[b, 0, 0, : wpos[b] + 1] = 0.0
        emb = self.llm.embed_tokens(ids)
        logits = self.llm(emb, pos, mask, view)[:, -1, :]
        for b, seq in enumerate(seqs):
            seq.append_token(self._sample(logits[b:b + 1], seq))
            seq.maybe_finish()

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
        return self.tokenizer.decode(new_token_ids)

    def _sample(self, logits: torch.Tensor, seq: Sequence) -> int:
        sp = seq.request.sampling
        if sp.greedy:
            return int(logits.argmax(dim=-1).item())
        probs = torch.softmax(logits.float() / sp.temperature, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())
