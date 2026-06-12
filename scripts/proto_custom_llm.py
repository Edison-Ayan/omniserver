"""Verify the from-scratch Qwen2 LLM forward is token-identical to HF.

Reuses HF's ViT (vision embeds) and get_rope_index (positions) — only the LLM
decoder stack is ours. Two fp16 2B models don't fit in 8 GB, so we capture HF's
reference + inputs on CPU, free the HF model, then build and run ours.
"""

import gc

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from omniserve.model import Qwen2Config, Qwen2LLM, load_from_hf

MODEL = "Qwen/Qwen2-VL-2B-Instruct"


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
        {"type": "image"}, {"type": "text", "text": "What shapes and colors are here?"}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[make_img()], return_tensors="pt").to(dev)
    input_ids = inp["input_ids"]
    L = input_ids.shape[1]
    image_token_id = hf.config.image_token_id

    # Capture HF reference + the bits we reuse, on CPU, then free the HF model.
    ref = hf(**inp).logits[:, -1, :].cpu()
    img_embeds = hf.model.visual(inp["pixel_values"], grid_thw=inp["image_grid_thw"]).cpu()
    pos = hf.model.get_rope_index(input_ids, inp["image_grid_thw"],
                                  attention_mask=inp.get("attention_mask"))[0].cpu()
    sd_cpu = {k: v.cpu() for k, v in hf.state_dict().items()}
    input_ids_cpu = input_ids.cpu()
    print("position_ids shape:", tuple(pos.shape), "(expect [3, 1, L])")

    del hf, inp
    gc.collect()
    torch.cuda.empty_cache()

    # Our model
    mine = Qwen2LLM(Qwen2Config()).to(dev).half().eval()
    load_from_hf(mine, sd_cpu)

    ids = input_ids_cpu.to(dev)
    emb = mine.embed_tokens(ids).clone()
    emb[ids == image_token_id] = img_embeds.to(dev).to(emb.dtype)
    causal = torch.full((L, L), float("-inf"), device=dev, dtype=torch.float16).triu(1)[None, None]
    out = mine(emb, pos.to(dev), causal)[:, -1, :].cpu()

    same = int(ref.argmax(-1).item()) == int(out.argmax(-1).item())
    max_err = (ref.float() - out.float()).abs().max().item()
    r5 = set(ref.topk(5, -1).indices[0].tolist())
    o5 = set(out.topk(5, -1).indices[0].tolist())
    print(f"next-token argmax match: {same}  (HF {ref.argmax(-1).item()} vs ours {out.argmax(-1).item()})")
    print(f"logits max abs err: {max_err:.3f} | top-5 overlap: {len(r5 & o5)}/5")
    print("decoded next token:", repr(proc.tokenizer.decode([int(out.argmax(-1).item())])))
    print("\n✅ LLM forward token-identical" if same else "\n❌ mismatch — debug")


if __name__ == "__main__":
    main()
