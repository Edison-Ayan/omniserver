"""QwenVLRunner — the real model backend for Qwen2-VL-2B.

This is where omniserve actually runs the model. Unlike a blackbox
`model.generate()` call, the engine needs prefill and single-token decode as
separate operations so the scheduler can admit/retire sequences every step
(continuous batching). That means we manage the KV cache ourselves.

Qwen2-VL specifics we handle:
  - multimodal prefill: processor turns (text, image) into input_ids +
    pixel_values + image_grid_thw; the model merges vision tokens internally.
  - M-RoPE: Qwen2-VL uses 3D rotary positions. `Qwen2VLModel.forward` computes
    position_ids itself when given `cache_position` and a stored `rope_deltas`,
    so per sequence we stash the `rope_deltas` produced at prefill and restore
    it before each decode step.

Stage 1 (this file): correct prefill + decode with a per-sequence DynamicCache.
Sequences in a scheduler step are executed one at a time inside the runner; the
control plane (batching/streaming) is already real. Stage 2 will fuse multiple
sequences into a single batched forward for throughput.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .base import ModelRunner
from ..request import Sequence

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


class QwenVLRunner(ModelRunner):
    def __init__(self, model_id: str = MODEL_ID, load_in_4bit: bool = True,
                 vision_cache=None, prefix_cache=None, fused_kernels: bool = False,
                 prealloc_kv: bool = True, max_running: int = 32, max_len: int = 1024):
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.model = self._load(model_id, load_in_4bit)
        if fused_kernels:
            from ..kernels import patch_llm_rmsnorm
            patch_llm_rmsnorm(self.model)
        self.device = next(self.model.parameters()).device
        self.eos_ids = {self.model.config.eos_token_id}
        # per-request stash for prompt-stage tensors (pixel_values etc.)
        self._pending: Dict[str, dict] = {}
        # per-request precomputed image content keys (for the caches)
        self._img_keys: Dict[str, list] = {}
        # optional prefix KV cache (reuse the whole prefill for repeated prefixes)
        self.prefix_cache = prefix_cache
        # optional vision embedding cache (installed on the model)
        self.vision_cache = vision_cache
        if vision_cache is not None:
            vision_cache.install(self.model)

        # Preallocated KV pool: write the new token in place instead of the
        # O(L^2) cat-based DynamicCache rebuild each decode step. Slots [0,n) hold
        # the active sequences; finishing a sequence compacts the pool.
        self.prealloc_kv = prealloc_kv
        self._max_len = max_len
        self._pool = None
        self._slot_seq: List = []   # _slot_seq[slot] -> Sequence in that slot
        if prealloc_kv:
            cfg = self.model.config
            from ..cache.kv_prealloc import PreallocatedKVCache
            self._pool = PreallocatedKVCache(
                n_layers=cfg.num_hidden_layers, max_batch=max_running,
                n_kv_heads=cfg.num_key_value_heads, max_len=max_len,
                head_dim=cfg.hidden_size // cfg.num_attention_heads,
                device=self.device, dtype=torch.float16)

    @staticmethod
    def _load(model_id: str, load_in_4bit: bool):
        from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration

        kw = dict(dtype=torch.float16, device_map="cuda")
        if load_in_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
            )
        return Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kw).eval()

    # ---- ModelRunner interface ---------------------------------------------

    def tokenize(self, seq: Sequence) -> None:
        """Build the chat-formatted multimodal inputs and stash the tensors."""
        req = seq.request
        content = []
        for _ in req.images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": req.prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], images=req.images or None, return_tensors="pt").to(self.device)
        seq.prompt_token_ids = inputs["input_ids"][0].tolist()
        self._pending[seq.request_id] = inputs

        # content-address the images once, here: used by the vision cache (O(1)
        # ViT lookup) and the prefix cache (part of the prefix key).
        if (self.vision_cache is not None or self.prefix_cache is not None) and req.images:
            from ..cache.vision import VisionEmbeddingCache
            self._img_keys[seq.request_id] = [
                VisionEmbeddingCache.image_key(im) for im in req.images]

        # make sure per-sequence stop ids include EOS
        sp = seq.request.sampling
        for e in self.eos_ids:
            if e is not None and e not in sp.stop_token_ids:
                sp.stop_token_ids.append(e)

    @staticmethod
    def _clone_cache(cache):
        """Deep-copy a DynamicCache so two sequences can grow it independently."""
        from transformers import DynamicCache
        legacy = [(k.clone(), v.clone()) for (k, v) in cache.to_legacy_cache()]
        return DynamicCache.from_legacy_cache(legacy)

    def _store_prefill(self, seq, cache, length: int, rope_delta: int) -> None:
        """Park a sequence's prefill KV: into a pool slot (prealloc) or keep the
        per-sequence DynamicCache (legacy path)."""
        if self.prealloc_kv:
            slot = len(self._slot_seq)
            self._pool.set_slot_prefix(slot, cache.to_legacy_cache(), length)
            self._slot_seq.append(seq)
            seq.kv_handle = {"slot": slot, "rope_delta": rope_delta}
        else:
            seq.kv_handle = {"cache": cache, "rope_delta": rope_delta, "len": length}

    @torch.inference_mode()
    def prefill(self, seqs: List[Sequence]) -> None:
        from transformers import DynamicCache

        for seq in seqs:
            inputs = self._pending.pop(seq.request_id)
            img_keys = self._img_keys.pop(seq.request_id, None)
            L = inputs["input_ids"].shape[1]

            # Prefix KV cache: identical (prompt tokens + images) -> reuse the
            # whole prefill. A hit clones the cached KV and skips the forward.
            pkey = None
            if self.prefix_cache is not None:
                pkey = self.prefix_cache.prefix_key(seq.prompt_token_ids, img_keys)
                entry = self.prefix_cache.get(pkey, self._clone_cache)
                if entry is not None:
                    self._store_prefill(seq, entry.cache, entry.length, entry.rope_delta)
                    seq.append_token(entry.first_token)
                    seq.maybe_finish()
                    continue

            cache = DynamicCache()
            self.model.model.rope_deltas = None  # force recompute for this seq
            if self.vision_cache is not None:
                self.vision_cache.set_pending(img_keys)
            out = self.model(
                **inputs,
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.arange(L, device=self.device),
            )
            if self.vision_cache is not None:
                self.vision_cache.set_pending(None)
            next_id = self._sample(out.logits[:, -1, :], seq)
            rope_delta = int(self.model.model.rope_deltas.flatten()[0].item())

            # Store a clone for the prefix cache BEFORE the pool consumes `cache`
            # (set_slot_prefix only reads it, but cloning keeps the cached copy
            # independent of any later in-place writes).
            if self.prefix_cache is not None:
                from ..cache.prefix import PrefixEntry
                self.prefix_cache.put(pkey, PrefixEntry(
                    cache=self._clone_cache(cache), rope_delta=rope_delta,
                    length=L, first_token=next_id))

            self._store_prefill(seq, cache, L, rope_delta)
            seq.append_token(next_id)
            seq.maybe_finish()

    def decode(self, seqs: List[Sequence]) -> None:
        if self.prealloc_kv:
            return self._decode_prealloc(seqs)
        return self._decode_cat(seqs)

    @torch.inference_mode()
    def _decode_prealloc(self, seqs: List[Sequence]) -> None:
        """Advance all running sequences one token in a single forward, writing
        each new token's KV in place into the preallocated pool (no per-step
        rebuild). Active sequences occupy contiguous slots [0, n); we order the
        batch by slot so row b == slot b."""
        seqs = sorted(seqs, key=lambda s: s.kv_handle["slot"])
        n = len(seqs)
        assert all(seqs[b].kv_handle["slot"] == b for b in range(n)), \
            "decode slots must be the contiguous prefix [0, n)"
        device = self.device

        view = self._pool.view(n)
        wpos = view.write_pos              # [n] per-slot position of the new token
        ret = view.ret_len
        input_ids = torch.tensor([[s.last_token_id()] for s in seqs], device=device)
        rope = torch.tensor([s.kv_handle["rope_delta"] for s in seqs], device=device)
        attn = torch.zeros(n, ret, device=device, dtype=torch.long)
        for b in range(n):
            attn[b, : wpos[b] + 1] = 1     # new token attends to [0, wpos]
        pos = torch.empty(3, n, 1, device=device, dtype=torch.long)
        pos[:, :, 0] = (wpos + rope).unsqueeze(0)

        out = self.model(input_ids=input_ids, past_key_values=view, use_cache=True,
                         attention_mask=attn, position_ids=pos,
                         cache_position=torch.tensor([ret - 1], device=device))
        logits = out.logits[:, -1, :]
        for b, seq in enumerate(seqs):
            seq.append_token(self._sample(logits[b:b + 1], seq))
            seq.maybe_finish()

    def free(self, seq: Sequence) -> None:
        """Release a finished sequence's pool slot, compacting so active slots
        stay the contiguous prefix [0, n): move the last active sequence into the
        freed slot."""
        if not self.prealloc_kv:
            return
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

    @torch.inference_mode()
    def _decode_cat(self, seqs: List[Sequence]) -> None:
        """Legacy path: rebuild a left-padded batched DynamicCache each step
        (O(L^2) cat). Kept for A/B comparison against the preallocated pool."""
        from transformers import DynamicCache

        B = len(seqs)
        device = self.device
        lens = [s.kv_handle["len"] for s in seqs]
        max_len = max(lens)

        # 1) build a batched, left-padded KV cache from each sequence's cache
        legs = [s.kv_handle["cache"].to_legacy_cache() for s in seqs]
        n_layers = len(legs[0])
        batched = []
        for li in range(n_layers):
            ks, vs = [], []
            for b in range(B):
                k, v = legs[b][li]
                pad = max_len - k.shape[2]
                if pad:
                    k = torch.nn.functional.pad(k, (0, 0, pad, 0))
                    v = torch.nn.functional.pad(v, (0, 0, pad, 0))
                ks.append(k)
                vs.append(v)
            batched.append((torch.cat(ks, 0), torch.cat(vs, 0)))
        bcache = DynamicCache.from_legacy_cache(batched)

        # 2) batched inputs: last token, mask (left-padding), explicit positions
        input_ids = torch.tensor([[s.last_token_id()] for s in seqs], device=device)
        attn = torch.zeros(B, max_len + 1, device=device, dtype=torch.long)
        pos = torch.empty(3, B, 1, device=device, dtype=torch.long)
        for b in range(B):
            attn[b, max_len - lens[b]:] = 1
            pos[:, b, 0] = lens[b] + seqs[b].kv_handle["rope_delta"]

        out = self.model(input_ids=input_ids, past_key_values=bcache, use_cache=True,
                         attention_mask=attn, position_ids=pos,
                         cache_position=torch.tensor([max_len], device=device))
        logits = out.logits[:, -1, :]  # [B, vocab]

        # 3) append ONLY the new token's KV to each sequence's own cache.
        #    The old KV is unchanged, so we copy a single position per layer
        #    instead of rebuilding the whole per-sequence cache (the new token
        #    sits at index max_len in the batched cache after the forward).
        new_legs = bcache.to_legacy_cache()
        for b, seq in enumerate(seqs):
            cache = seq.kv_handle["cache"]
            for li, (k, v) in enumerate(new_legs):
                k_new = k[b:b + 1, :, max_len:max_len + 1, :].contiguous()
                v_new = v[b:b + 1, :, max_len:max_len + 1, :].contiguous()
                cache.update(k_new, v_new, li)
            seq.kv_handle["len"] += 1
            seq.append_token(self._sample(logits[b:b + 1], seq))
            seq.maybe_finish()

    def detokenize(self, seq: Sequence, new_token_ids: List[int]) -> str:
        return self.tokenizer.decode(new_token_ids, skip_special_tokens=True)

    # ---- sampling -----------------------------------------------------------

    def _sample(self, logits: torch.Tensor, seq: Sequence) -> int:
        sp = seq.request.sampling
        if sp.greedy:
            return int(logits.argmax(dim=-1).item())
        probs = torch.softmax(logits.float() / sp.temperature, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())
