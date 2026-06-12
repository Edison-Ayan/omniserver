"""预分配 KV cache 的正确性原型。

标准:预分配的原地 decode 必须和(已验证的)逐序列 decode 逐 token 一致。
在一个 batch 里测不等长序列。
"""

import torch
from PIL import Image, ImageDraw
from transformers import DynamicCache

from omniserve.cache.kv_prealloc import PreallocatedKVCache
from omniserve.runners.qwen_vl import QwenVLRunner


def img(c):
    im = Image.new("RGB", (448, 448), (20, 20, 20))
    d = ImageDraw.Draw(im)
    d.rectangle([50, 50, 200, 200], fill=c)
    return im


def prefill(r, prompt, image):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    t = r.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = r.processor(text=[t], images=[image], return_tensors="pt").to(r.device)
    cache = DynamicCache()
    L = inp["input_ids"].shape[1]
    r.model.model.rope_deltas = None
    with torch.inference_mode():
        out = r.model(**inp, past_key_values=cache, use_cache=True,
                      cache_position=torch.arange(L, device=r.device))
    nxt = int(out.logits[:, -1, :].argmax(-1).item())
    rd = int(r.model.model.rope_deltas.flatten()[0].item())
    return cache, L, rd, nxt


@torch.inference_mode()
def seq_decode(r, cache, length, rope_delta, first_tok, n):
    toks = [first_tok]
    cur = length
    for _ in range(n):
        r.model.model.rope_deltas = torch.tensor([[rope_delta]], device=r.device)
        last = torch.tensor([[toks[-1]]], device=r.device)
        out = r.model(input_ids=last, past_key_values=cache, use_cache=True,
                      cache_position=torch.tensor([cur], device=r.device))
        cur += 1
        toks.append(int(out.logits[:, -1, :].argmax(-1).item()))
    return toks


@torch.inference_mode()
def prealloc_decode(r, states, n):
    """states:list of {cache(DynamicCache), len, rope_delta, first_tok}。"""
    B = len(states)
    device = r.device
    legs0 = states[0]["cache"].to_legacy_cache()
    n_layers = len(legs0)
    H, D = legs0[0][0].shape[1], legs0[0][0].shape[3]
    max_len = max(s["len"] for s in states) + n + 1

    pool = PreallocatedKVCache(n_layers, B, H, max_len, D, device, legs0[0][0].dtype)
    rope = torch.tensor([s["rope_delta"] for s in states], device=device)
    for b, s in enumerate(states):
        pool.set_slot_prefix(b, s["cache"].to_legacy_cache(), s["len"])

    toks = [[s["first_tok"]] for s in states]
    for _ in range(n):
        view = pool.view(B)
        wpos = view.write_pos                       # [B] 每个槽位新 token 的位置
        ret = view.ret_len
        input_ids = torch.tensor([[t[-1]] for t in toks], device=device)
        attn = torch.zeros(B, ret, device=device, dtype=torch.long)
        for b in range(B):
            attn[b, : wpos[b] + 1] = 1              # 新 token 能看到 [0, wpos]
        pos = torch.empty(3, B, 1, device=device, dtype=torch.long)
        pos[:, :, 0] = (wpos + rope).unsqueeze(0)
        out = r.model(input_ids=input_ids, past_key_values=view, use_cache=True,
                      attention_mask=attn, position_ids=pos,
                      cache_position=torch.tensor([ret - 1], device=device))
        nxt = out.logits[:, -1, :].argmax(-1).tolist()
        for b in range(B):
            toks[b].append(int(nxt[b]))
    return toks


def main():
    r = QwenVLRunner(load_in_4bit=True)
    N = 16
    specs = [("Describe.", (220, 30, 30)),
             ("Describe this image with as much detail as you possibly can.", (30, 30, 220))]
    states, refs = [], []
    for p, c in specs:
        cache, L, rd, ft = prefill(r, p, img(c))
        states.append({"cache": cache, "len": L, "rope_delta": rd, "first_tok": ft})
    for s in states:
        c2 = DynamicCache.from_legacy_cache(s["cache"].to_legacy_cache())
        refs.append(seq_decode(r, c2, s["len"], s["rope_delta"], s["first_tok"], N))
    got = prealloc_decode(r, states, N)
    for b in range(len(specs)):
        ok = refs[b] == got[b]
        print(f" seq{b} (len {states[b]['len']}): {'MATCH' if ok else 'DIFF'}")
        if not ok:
            print("  ref:", refs[b][:10])
            print("  got:", got[b][:10])


if __name__ == "__main__":
    main()
