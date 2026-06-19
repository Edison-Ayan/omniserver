"""Qwen2-VL-2B 的 VLMAdapter:ViT + Qwen2 LLM + M-RoPE + 图像切块 + chat 模板。

把原来散在 NativeQwenVLRunner 里的"模型怎么算"全收进这一个 adapter;通用 MultimodalRunner
只认 VLMAdapter 接口。支持 fp16(默认)/ marlin(在线 RTN int4,只 MLP)/ gptq(加载 GPTQ-Int4
校准权重)/ mixed(per-op 混合精度:MLP=FP8 + attention=fp16,见 kernels/mixed_precision.py)。
"""

from __future__ import annotations

import gc
import glob
import os
from typing import List

import torch
from PIL import Image
from safetensors.torch import load_file

from .base import VLMAdapter
from ..model import Qwen2Config, Qwen2LLM, Qwen2VIT, load_from_hf, load_vit_from_state_dict
from ..model.positions import IMAGE_TOKEN_ID, mrope_position_ids
from ..model.preprocess import preprocess_image
from ..model.tokenizer import Qwen2VLTokenizer

EOS_ID = 151645  # <|im_end|>
DEFAULT_SNAPSHOT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/*")


class Qwen2VLAdapter(VLMAdapter):
    def __init__(self, model_dir: str = None, quant: str = None, gptq_dir: str = None):
        self.device = "cuda"
        model_dir = model_dir or glob.glob(DEFAULT_SNAPSHOT)[0]
        self._tokenizer = Qwen2VLTokenizer(os.path.join(model_dir, "tokenizer.json"))
        self.image_token_id = IMAGE_TOKEN_ID
        self.eos_ids = {EOS_ID}
        self.quant = quant

        # gptq:整模型从 GPTQ checkpoint 加载(ViT/embed fp16 + decoder int4,和 vLLM 同权重)
        gptq = quant == "gptq"
        src = (gptq_dir or "/tmp/gptq_model") if gptq else model_dir
        sd = {}
        for f in glob.glob(os.path.join(src, "model*.safetensors")):
            sd.update(load_file(f))

        self._vit = Qwen2VIT()
        load_vit_from_state_dict(self._vit, sd)
        self._vit = self._vit.half().to(self.device).eval()

        self._llm = Qwen2LLM(Qwen2Config()).half().to(self.device)
        if gptq:
            from ..kernels.marlin_linear import load_gptq_llm
            n = load_gptq_llm(self._llm, sd, self.device)
            print(f"[gptq] 加载 {n} 个 GPTQ int4 GEMM(全 decoder,和 vLLM 同权重同精度)")
        else:
            load_from_hf(self._llm, sd)
            if quant == "marlin":
                from ..kernels.marlin_linear import quantize_llm_marlin
                n = quantize_llm_marlin(self._llm, scope="mlp")
                print(f"[marlin] 量化 {n} 个 MLP 大 GEMM 为 int4(在线 RTN,降精度)")
            elif quant == "mixed":
                from ..kernels.mixed_precision import apply_mixed
                plan = apply_mixed(self._llm)
                print(f"[mixed] per-op 混合精度(每算子 精度×层数):{plan}")
        self._llm = self._llm.eval()
        del sd
        gc.collect()
        torch.cuda.empty_cache()

        cfg = self._llm.cfg
        self.num_layers = cfg.num_layers
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim

    # ---- 输入侧 ----
    def preprocess(self, images: List[Image.Image]):
        if not images:
            return None, None
        pvs, grids = [], []
        for im in images:
            pv, grid = preprocess_image(im)
            pvs.append(pv)
            grids.append(grid[0])
        pixel_values = torch.cat(pvs, 0).to(self.device).half()
        grid_thw = torch.stack(grids).to(self.device)
        return pixel_values, grid_thw

    def encode_prompt(self, prompt: str, grids) -> List[int]:
        glist = list(grids) if grids is not None else ()
        return self._tokenizer.encode_prompt(prompt, glist)

    # ---- 编码 ----
    def embed_tokens(self, input_ids):
        return self._llm.embed_tokens(input_ids)

    def vision_embed(self, pixel_values, grids):
        return self._vit(pixel_values, grids).to(torch.float16)

    # ---- 位置(M-RoPE 3D)----
    def prefill_positions(self, input_ids, grids):
        if grids is None:
            L = input_ids.shape[1]
            pos = torch.arange(L, device=input_ids.device).view(1, 1, -1).expand(3, 1, -1)
            return pos, 0
        pos, delta = mrope_position_ids(input_ids, grids)
        return pos, int(delta.flatten()[0].item())

    def decode_positions(self, write_pos, rope_deltas):
        n = write_pos.shape[0]
        pos = torch.empty(3, n, 1, device=write_pos.device, dtype=torch.long)
        pos[:, :, 0] = (write_pos + rope_deltas).unsqueeze(0)
        return pos

    # ---- 前向 + 反 tokenize ----
    def llm(self, embeds, positions, attn_mask, cache, logits_indices=None):
        return self._llm(embeds, positions, attn_mask, cache, logits_indices=logits_indices)

    def detokenize(self, token_ids):
        return self._tokenizer.decode(token_ids)

    def supports_quant(self, quant):
        return quant in (None, "marlin", "gptq", "mixed")
