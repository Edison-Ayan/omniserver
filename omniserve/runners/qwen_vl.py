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
                 vision_cache=None):
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.model = self._load(model_id, load_in_4bit)
        self.device = next(self.model.parameters()).device
        self.eos_ids = {self.model.config.eos_token_id}
        # per-request stash for prompt-stage tensors (pixel_values etc.)
        self._pending: Dict[str, dict] = {}
        # per-request precomputed image content keys (for the vision cache)
        self._img_keys: Dict[str, list] = {}
        # optional vision embedding cache (installed on the model)
        self.vision_cache = vision_cache
        if vision_cache is not None:
            vision_cache.install(self.model)

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

        # content-address the images once, here, so the vision cache lookup is
        # O(1) at the ViT instead of re-hashing the pixel tensor each call.
        if self.vision_cache is not None and req.images:
            self._img_keys[seq.request_id] = [
                self.vision_cache.image_key(im) for im in req.images]

        # make sure per-sequence stop ids include EOS
        sp = seq.request.sampling
        for e in self.eos_ids:
            if e is not None and e not in sp.stop_token_ids:
                sp.stop_token_ids.append(e)

    @torch.inference_mode()
    def prefill(self, seqs: List[Sequence]) -> None:
        from transformers import DynamicCache

        for seq in seqs:
            inputs = self._pending.pop(seq.request_id)
            cache = DynamicCache()
            L = inputs["input_ids"].shape[1]
            self.model.model.rope_deltas = None  # force recompute for this seq
            if self.vision_cache is not None:
                self.vision_cache.set_pending(self._img_keys.pop(seq.request_id, None))
            out = self.model(
                **inputs,
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.arange(L, device=self.device),
            )
            if self.vision_cache is not None:
                self.vision_cache.set_pending(None)
            next_id = self._sample(out.logits[:, -1, :], seq)
            seq.kv_handle = {
                "cache": cache,
                "rope_delta": int(self.model.model.rope_deltas.flatten()[0].item()),
                "len": L,
            }
            seq.append_token(next_id)
            seq.maybe_finish()

    @torch.inference_mode()
    def decode(self, seqs: List[Sequence]) -> None:
        """Advance all running sequences by one token in a SINGLE batched
        forward pass. Sequences have different KV lengths, so we left-pad each
        cache to the batch max and pass explicit M-RoPE positions + an
        attention mask. Verified token-identical to per-sequence decoding."""
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
