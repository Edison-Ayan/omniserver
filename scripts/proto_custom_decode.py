"""Verify the from-scratch LLM decode (with KV cache) matches HF greedy gen.

Same memory trick: capture HF's greedy continuation + reused bits on CPU, free
HF, then prefill+decode through our model and compare the token sequences.
"""

import gc

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from omniserve.model import Qwen2Config, Qwen2LLM, load_from_hf

MODEL = "Qwen/Qwen2-VL-2B-Instruct"
N = 20


class SimpleCache:
    """Minimal append-only multi-layer KV cache (cat-based) for checking."""
    def __init__(self, n_layers):
        self.k = [None] * n_layers
        self.v = [None] * n_layers

    def update(self, k, v, i):
        self.k[i] = k if self.k[i] is None else torch.cat([self.k[i], k], dim=2)
        self.v[i] = v if self.v[i] is None else torch.cat([self.v[i], v], dim=2)
        return self.k[i], self.v[i]


def make_img():
    im = Image.new("RGB", (448, 448), (30, 30, 60))
    d = ImageDraw.Draw(im)
    d.rectangle([60, 60, 200, 200], fill=(220, 40, 40))
    d.ellipse([240, 240, 380, 380], fill=(40, 200, 90))
    return im


@torch.inference_mode()
def main():
    proc = AutoProcessor.from_pretrained(MODEL)
    hf = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.float16, device_map="cuda").eval()
    dev = "cuda"
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "Describe this image in detail."}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[make_img()], return_tensors="pt").to(dev)
    input_ids = inp["input_ids"]
    L = input_ids.shape[1]
    image_token_id = hf.config.image_token_id

    ref_ids = hf.generate(**inp, max_new_tokens=N, do_sample=False)[0, L:].cpu().tolist()
    img_embeds = hf.model.visual(inp["pixel_values"], grid_thw=inp["image_grid_thw"]).cpu()
    pos, rope_delta = hf.model.get_rope_index(input_ids, inp["image_grid_thw"],
                                              attention_mask=inp.get("attention_mask"))
    pos, rope_delta = pos.cpu(), int(rope_delta.flatten()[0].item())
    sd_cpu = {k: v.cpu() for k, v in hf.state_dict().items()}
    ids_cpu = input_ids.cpu()
    del hf, inp
    gc.collect(); torch.cuda.empty_cache()

    mine = Qwen2LLM(Qwen2Config()).to(dev).half().eval()
    load_from_hf(mine, sd_cpu)
    cache = SimpleCache(mine.cfg.num_layers)

    ids = ids_cpu.to(dev)
    emb = mine.embed_tokens(ids).clone()
    emb[ids == image_token_id] = img_embeds.to(dev).to(emb.dtype)
    causal = torch.full((L, L), float("-inf"), device=dev, dtype=torch.float16).triu(1)[None, None]
    logits = mine(emb, pos.to(dev), causal, cache)[:, -1, :]

    got = [int(logits.argmax(-1).item())]
    cur = L
    for _ in range(N - 1):
        tok = torch.tensor([[got[-1]]], device=dev)
        e = mine.embed_tokens(tok)
        p = torch.full((3, 1, 1), cur + rope_delta, device=dev, dtype=torch.long)
        logits = mine(e, p, None, cache)[:, -1, :]  # full attention over cache
        got.append(int(logits.argmax(-1).item()))
        cur += 1

    match = sum(a == b for a, b in zip(ref_ids, got))
    print(f"decode token match: {match}/{N} {'✅' if match == N else '❌'}")
    if match != N:
        print(" HF :", ref_ids)
        print(" mine:", got)
    print(" text:", repr(proc.tokenizer.decode(got)))


if __name__ == "__main__":
    main()
