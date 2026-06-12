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
    def __init__(self, model_dir: str = None, max_running: int = 32, max_len: int = 1024):
        model_dir = model_dir or glob.glob(DEFAULT_SNAPSHOT)[0]
        self.device = "cuda"
        self.tokenizer = Qwen2VLTokenizer(os.path.join(model_dir, "tokenizer.json"))
        self.image_token_id = IMAGE_TOKEN_ID
        self.eos_ids = {EOS_ID}

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
        sp = seq.request.sampling
        for e in self.eos_ids:
            if e not in sp.stop_token_ids:
                sp.stop_token_ids.append(e)

    @torch.inference_mode()
    def prefill(self, seqs: List[Sequence]) -> None:
        """打包式 prefill:把这一批序列的 prompt token 拼成一次前向,用块对角因果
        mask 让每个 prompt 只注意自己内部。这样权重只读一次、GPU 喂满,替代原来
        一个一个 prefill。每段的 KV 写进各自的池槽位。"""
        dev = self.device
        embeds, positions, seg_lens, slots, deltas = [], [], [], [], []
        for seq in seqs:
            p = self._pending.pop(seq.request_id)
            ids = p["ids"]
            L = ids.shape[1]
            emb = self.llm.embed_tokens(ids).clone()
            if p["pixel_values"] is not None:
                vis = self.vit(p["pixel_values"], p["grid_thw"])
                emb[ids == self.image_token_id] = vis.to(emb.dtype)
            pos, delta = mrope_position_ids(ids, p["grid_thw"]) if p["grid_thw"] is not None \
                else self._text_positions(ids)
            embeds.append(emb)                 # [1, L, hidden]
            positions.append(pos)              # [3, 1, L]
            seg_lens.append(L)
            slots.append(len(self._slot_seq) + len(slots))
            deltas.append(int(delta.flatten()[0].item()))

        packed_emb = torch.cat(embeds, dim=1)          # [1, total, hidden]
        packed_pos = torch.cat(positions, dim=2)       # [3, 1, total]
        total = sum(seg_lens)
        # 块对角因果 mask:每段内部是因果,段之间互不可见(保持各 prompt 独立)
        mask = torch.full((total, total), float("-inf"), device=dev, dtype=torch.float16)
        off = 0
        for L in seg_lens:
            mask[off:off + L, off:off + L] = torch.triu(
                torch.full((L, L), float("-inf"), device=dev, dtype=torch.float16), 1)
            off += L
        mask = mask[None, None]                        # [1, 1, total, total]

        cache = self._pool.packed_prefill_cache(slots, seg_lens)
        logits = self.llm(packed_emb, packed_pos, mask, cache)[0]   # [total, vocab]

        # 每段最后一个 token 的 logits -> 该序列的第一个生成 token
        off = 0
        for i, seq in enumerate(seqs):
            L = seg_lens[i]
            self._slot_seq.append(seq)
            seq.kv_handle = {"slot": slots[i], "rope_delta": deltas[i]}
            seq.append_token(self._sample(logits[off + L - 1: off + L], seq))
            seq.maybe_finish()
            off += L

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
