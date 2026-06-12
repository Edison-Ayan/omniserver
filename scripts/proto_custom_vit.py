"""验证从零写的 ViT 和 HF 的 model.visual 一致,权重直接用 safetensors 加载
(权重不走 transformers 的 from_pretrained)。"""

import glob
import os

import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

from omniserve.model.qwen2_vit import Qwen2VIT, load_vit_from_state_dict

MODEL = "Qwen/Qwen2-VL-2B-Instruct"
SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/*"))[0]


def load_sd():
    sd = {}
    for f in glob.glob(os.path.join(SNAP, "model-*.safetensors")):
        sd.update(load_file(f))
    return sd


def make_img():
    im = Image.new("RGB", (448, 448), (30, 30, 60))
    d = ImageDraw.Draw(im)
    d.rectangle([60, 60, 200, 200], fill=(220, 40, 40))
    d.ellipse([240, 240, 380, 380], fill=(40, 200, 90))
    return im


@torch.inference_mode()
def main():
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(MODEL)
    inp = proc(text=["<|vision_start|><|image_pad|><|vision_end|>"], images=[make_img()],
               return_tensors="pt").to("cuda")
    pv, grid = inp["pixel_values"], inp["image_grid_thw"]

    hf = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.float16, device_map="cuda").eval()
    ref = hf.model.visual(pv, grid_thw=grid)
    print("HF visual out:", tuple(ref.shape))
    del hf
    torch.cuda.empty_cache()

    sd = load_sd()
    vit = Qwen2VIT().half().to("cuda").eval()
    load_vit_from_state_dict(vit, sd)
    out = vit(pv, grid)

    max_err = (ref.float() - out.float()).abs().max().item()
    rel = max_err / ref.float().abs().mean().item()
    print(f"my ViT out:    {tuple(out.shape)}")
    print(f"max abs err: {max_err:.4f} | mean|ref|: {ref.float().abs().mean():.3f} | rel: {rel:.4f}")
    print("\n✅ ViT output matches HF" if max_err < 0.05 else "\n⚠️ diff — check details")


if __name__ == "__main__":
    main()
