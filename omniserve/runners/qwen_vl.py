"""QwenVLRunner —— Qwen2-VL-2B 的真实模型后端(基于 HF,用于参照/对比)。

这是 omniserve 真正跑模型的地方。和黑盒的 `model.generate()` 不同,引擎需要把
prefill 和单 token decode 拆成两个独立操作,这样调度器才能每步准入/退休序列
(continuous batching)。这意味着 KV cache 由我们自己管。

我们处理的 Qwen2-VL 细节:
  - 多模态 prefill:processor 把 (文本, 图) 变成 input_ids + pixel_values +
    image_grid_thw;模型内部把 vision token 合并进去。
  - M-RoPE:Qwen2-VL 用 3D 旋转位置。给定 `cache_position` 和存好的 `rope_deltas`
    时 `Qwen2VLModel.forward` 会自己算 position_ids,所以我们按序列存好 prefill 时
    产生的 `rope_deltas`,每个 decode 步前恢复它。

Stage 1(本文件):用每序列一个 DynamicCache 做正确的 prefill + decode。一个调度步里
的序列在 runner 内部一个一个执行;控制平面(batching/流式)已经是真的。Stage 2 把多个
序列融进单个批量前向以提吞吐。
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
        # 每请求暂存 prompt 阶段的张量(pixel_values 等)
        self._pending: Dict[str, dict] = {}
        # 每请求预先算好的图像内容 key(给各 cache 用)
        self._img_keys: Dict[str, list] = {}
        # 可选的 prefix KV cache(相同前缀复用整段 prefill)
        self.prefix_cache = prefix_cache
        # 可选的 vision embedding cache(装在模型上)
        self.vision_cache = vision_cache
        if vision_cache is not None:
            vision_cache.install(self.model)

        # 预分配 KV 池:每个 decode 步原地写入新 token,而不是 cat-based DynamicCache
        # 那样 O(L^2) 重建。槽位 [0,n) 放活跃序列;序列跑完时紧凑化池。
        self.prealloc_kv = prealloc_kv
        self._max_len = max_len
        self._pool = None
        self._slot_seq: List = []   # _slot_seq[slot] -> 该槽位里的 Sequence
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
        """构造 chat 格式的多模态输入,并暂存张量。"""
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

        # 在这里对图像做一次内容寻址:给 vision cache 用(O(1) 查 ViT)和
        # prefix cache 用(作为前缀 key 的一部分)。
        if (self.vision_cache is not None or self.prefix_cache is not None) and req.images:
            from ..cache.vision import VisionEmbeddingCache
            self._img_keys[seq.request_id] = [
                VisionEmbeddingCache.image_key(im) for im in req.images]

        # 确保每序列的 stop id 里包含 EOS
        sp = seq.request.sampling
        for e in self.eos_ids:
            if e is not None and e not in sp.stop_token_ids:
                sp.stop_token_ids.append(e)

    @staticmethod
    def _clone_cache(cache):
        """深拷贝一个 DynamicCache,让两个序列能各自独立地往里追加。"""
        from transformers import DynamicCache
        legacy = [(k.clone(), v.clone()) for (k, v) in cache.to_legacy_cache()]
        return DynamicCache.from_legacy_cache(legacy)

    def _store_prefill(self, seq, cache, length: int, rope_delta: int) -> None:
        """存放一个序列的 prefill KV:放进池槽位(预分配),或保留每序列的
        DynamicCache(旧路径)。"""
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

            # Prefix KV cache:相同的(prompt token + 图)-> 复用整段 prefill。
            # 命中时 clone 缓存的 KV,跳过前向。
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
            self.model.model.rope_deltas = None  # 强制为这个序列重算
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

            # 在池消费 `cache` 之前,先给 prefix cache 存一份 clone
            # (set_slot_prefix 只读它,但 clone 能让缓存副本独立于之后的原地写)。
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
        """所有在跑序列在单次前向里前进一个 token,把每个新 token 的 KV 原地写进
        预分配池(不每步重建)。活跃序列占连续槽位 [0, n);我们按槽位给 batch 排序,
        使得 row b == slot b。"""
        seqs = sorted(seqs, key=lambda s: s.kv_handle["slot"])
        n = len(seqs)
        assert all(seqs[b].kv_handle["slot"] == b for b in range(n)), \
            "decode 槽位必须是连续前缀 [0, n)"
        device = self.device

        view = self._pool.view(n)
        wpos = view.write_pos              # [n] 每个槽位新 token 的位置
        ret = view.ret_len
        input_ids = torch.tensor([[s.last_token_id()] for s in seqs], device=device)
        rope = torch.tensor([s.kv_handle["rope_delta"] for s in seqs], device=device)
        attn = torch.zeros(n, ret, device=device, dtype=torch.long)
        for b in range(n):
            attn[b, : wpos[b] + 1] = 1     # 新 token 注意到 [0, wpos]
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
        """释放跑完序列的池槽位,并紧凑化让活跃槽位保持连续前缀 [0, n):
        把最后一个活跃序列移到空出来的槽位。"""
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
        """旧路径:每步重建一个左 padding 的批量 DynamicCache(O(L^2) cat)。
        保留它是为了和预分配池做 A/B 对比。"""
        from transformers import DynamicCache

        B = len(seqs)
        device = self.device
        lens = [s.kv_handle["len"] for s in seqs]
        max_len = max(lens)

        # 1) 从每个序列的 cache 拼出一个左 padding 的批量 KV cache
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

        # 2) 批量输入:最后一个 token、mask(左 padding)、显式 position
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

        # 3) 只把新 token 的 KV 追加到每个序列自己的 cache。
        #    旧 KV 没变,所以每层只拷贝一个位置,而不是重建整个 per-sequence cache
        #    (前向后,新 token 在批量 cache 里位于 index max_len)。
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
