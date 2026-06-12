"""Native Qwen2-VL runner: our own LLM forward + HF's vision encoder.

Unlike QwenVLRunner (which calls transformers' model forward), this drives the
from-scratch `Qwen2LLM` (omniserve/model). That ownership is what later enables
varlen/packed prefill, mixed batching and kernel fusion. The ViT is still HF's;
its embeddings are spliced into the text embeddings.

Memory: two fp16 2B models don't fit in 8 GB, so we load HF once, copy the LLM
weights into our model, then free HF's language stack and keep only its ViT.
"""

from __future__ import annotations

import gc
from typing import Dict, List

import torch

from ..cache.kv_prealloc import PreallocatedKVCache
from ..model import Qwen2Config, Qwen2LLM, load_from_hf
from ..request import Sequence
from .base import ModelRunner

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


class NativeQwenVLRunner(ModelRunner):
    def __init__(self, model_id: str = MODEL_ID, max_running: int = 32, max_len: int = 1024):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        hf = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.float16, device_map="cuda").eval()
        self.device = hf.device
        self.image_token_id = hf.config.image_token_id
        self.eos_ids = {hf.config.eos_token_id}

        # Copy the LLM weights to CPU, then free HF's GPU language stack (keeping
        # only the ViT) BEFORE building our model, so the GPU never holds two full
        # LLMs at once.
        P = "model.language_model."
        llm_weights = {k: v.cpu() for k, v in hf.state_dict().items()
                       if k.startswith(P)}
        self.visual = hf.model.visual
        self._get_rope_index = hf.model.get_rope_index
        hf.model.language_model = None
        hf.lm_head = None
        gc.collect()
        torch.cuda.empty_cache()

        # build our LLM in fp16 on CPU, load weights, then move to GPU
        self.llm = Qwen2LLM(Qwen2Config()).half()
        load_from_hf(self.llm, llm_weights)
        self.llm = self.llm.to(self.device).eval()
        del llm_weights
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
        content = [{"type": "image"} for _ in req.images]
        content.append({"type": "text", "text": req.prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=req.images or None, return_tensors="pt").to(self.device)
        seq.prompt_token_ids = inputs["input_ids"][0].tolist()
        self._pending[seq.request_id] = inputs
        sp = seq.request.sampling
        for e in self.eos_ids:
            if e is not None and e not in sp.stop_token_ids:
                sp.stop_token_ids.append(e)

    @torch.inference_mode()
    def prefill(self, seqs: List[Sequence]) -> None:
        for seq in seqs:
            inputs = self._pending.pop(seq.request_id)
            ids = inputs["input_ids"]
            L = ids.shape[1]

            emb = self.llm.embed_tokens(ids).clone()
            if inputs.get("pixel_values") is not None:
                vis = self.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
                emb[ids == self.image_token_id] = vis.to(emb.dtype)
            pos, rope_delta = self._get_rope_index(
                ids, inputs.get("image_grid_thw"), attention_mask=inputs.get("attention_mask"))
            rope_delta = int(rope_delta.flatten()[0].item())

            causal = torch.full((L, L), float("-inf"), device=self.device,
                                dtype=torch.float16).triu(1)[None, None]
            slot = len(self._slot_seq)
            cache = self._pool.prefill_cache(slot)
            logits = self.llm(emb, pos, causal, cache)[:, -1, :]

            self._slot_seq.append(seq)
            seq.kv_handle = {"slot": slot, "rope_delta": rope_delta}
            seq.append_token(self._sample(logits, seq))
            seq.maybe_finish()

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
        # additive mask [n,1,1,ret]: row b attends to [0, wpos[b]]
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
        return self.tokenizer.decode(new_token_ids, skip_special_tokens=True)

    def _sample(self, logits: torch.Tensor, seq: Sequence) -> int:
        sp = seq.request.sampling
        if sp.greedy:
            return int(logits.argmax(dim=-1).item())
        probs = torch.softmax(logits.float() / sp.temperature, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())
