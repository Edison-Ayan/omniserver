"""Prototype + correctness test for batched decode.

Bar: batched decode must produce token-identical output to the (already
verified) sequential decode path. We test equal-length first, then ragged
lengths with left-padding.
"""

import torch
from PIL import Image, ImageDraw
from transformers import DynamicCache

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
    """Ground-truth: the existing single-sequence decode loop."""
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
def batched_decode(r, states, n):
    """states: list of dicts {cache, len, rope_delta, first_tok}.
    Runs all sequences in ONE forward per step with left-padded KV."""
    B = len(states)
    toks = [[s["first_tok"]] for s in states]
    lens = [s["len"] for s in states]
    device = r.device

    for _ in range(n):
        max_len = max(lens)
        legs = [s["cache"].to_legacy_cache() for s in states]
        n_layers = len(legs[0])

        # build batched, left-padded legacy cache
        batched = []
        for li in range(n_layers):
            ks, vs = [], []
            for b in range(B):
                k, v = legs[b][li]            # [1, H, lens[b], D]
                pad = max_len - k.shape[2]
                if pad:
                    k = torch.nn.functional.pad(k, (0, 0, pad, 0))  # left pad dim=2
                    v = torch.nn.functional.pad(v, (0, 0, pad, 0))
                ks.append(k)
                vs.append(v)
            batched.append((torch.cat(ks, 0), torch.cat(vs, 0)))
        bcache = DynamicCache.from_legacy_cache(batched)

        input_ids = torch.tensor([[t[-1]] for t in toks], device=device)  # [B,1]
        # attention mask over [past max_len + 1 new]; left-padding => zeros on left
        attn = torch.zeros(B, max_len + 1, device=device, dtype=torch.long)
        for b in range(B):
            attn[b, max_len - lens[b]:] = 1
        # explicit M-RoPE positions: new token position = current_len + rope_delta
        pos = torch.empty(3, B, 1, device=device, dtype=torch.long)
        for b in range(B):
            pos[:, b, 0] = lens[b] + states[b]["rope_delta"]

        out = r.model(input_ids=input_ids, past_key_values=bcache, use_cache=True,
                      attention_mask=attn, position_ids=pos,
                      cache_position=torch.tensor([max_len], device=device))
        logits = out.logits[:, -1, :]                 # [B, vocab]
        nxt = logits.argmax(-1).tolist()

        # scatter updated KV back to each sequence's own cache
        new_legs = bcache.to_legacy_cache()
        for b in range(B):
            per = []
            real = lens[b] + 1                        # this seq's true new length
            for li in range(n_layers):
                k, v = new_legs[li]                   # [B, H, max_len+1, D]
                per.append((k[b:b+1, :, max_len + 1 - real:, :].contiguous(),
                            v[b:b+1, :, max_len + 1 - real:, :].contiguous()))
            states[b]["cache"] = DynamicCache.from_legacy_cache(per)
            lens[b] += 1
            toks[b].append(int(nxt[b]))
    return toks


def main():
    r = QwenVLRunner(load_in_4bit=True)
    N = 16

    print("=== Test 1: equal-length sequences ===")
    specs = [("Describe briefly.", (220, 30, 30)),
             ("Describe briefly.", (30, 30, 220))]
    states, refs = [], []
    for p, c in specs:
        cache, L, rd, ft = prefill(r, p, img(c))
        states.append({"cache": cache, "len": L, "rope_delta": rd, "first_tok": ft})
    for s in list(states):
        c2 = DynamicCache.from_legacy_cache(s["cache"].to_legacy_cache())
        refs.append(seq_decode(r, c2, s["len"], s["rope_delta"], s["first_tok"], N))
    got = batched_decode(r, states, N)
    for b in range(len(specs)):
        ok = refs[b] == got[b]
        print(f" seq{b}: {'MATCH' if ok else 'DIFF'}")
        if not ok:
            print("  ref:", refs[b]); print("  got:", got[b])

    print("\n=== Test 2: ragged-length sequences (left padding) ===")
    specs = [("Describe.", (220, 30, 30)),
             ("Describe this image with as much detail as you possibly can.", (30, 30, 220))]
    states, refs = [], []
    for p, c in specs:
        cache, L, rd, ft = prefill(r, p, img(c))
        states.append({"cache": cache, "len": L, "rope_delta": rd, "first_tok": ft})
    for s in list(states):
        c2 = DynamicCache.from_legacy_cache(s["cache"].to_legacy_cache())
        refs.append(seq_decode(r, c2, s["len"], s["rope_delta"], s["first_tok"], N))
    got = batched_decode(r, states, N)
    for b in range(len(specs)):
        ok = refs[b] == got[b]
        print(f" seq{b} (len differ): {'MATCH' if ok else 'DIFF'}")
        if not ok:
            print("  ref:", refs[b][:8]); print("  got:", got[b][:8])


if __name__ == "__main__":
    main()
