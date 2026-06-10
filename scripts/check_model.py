"""Minimal sanity check: can this machine load Qwen2-VL-2B in 4-bit and
generate from a real image? Reports actual peak VRAM so we stop guessing.

Run:
    python scripts/check_model.py
First run downloads ~weights to the HF cache.
"""

import time

import torch
from PIL import Image, ImageDraw

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


def make_test_image(size=448):
    img = Image.new("RGB", (size, size), (30, 30, 60))
    d = ImageDraw.Draw(img)
    d.rectangle([60, 60, 200, 200], fill=(220, 40, 40))
    d.ellipse([240, 240, 380, 380], fill=(40, 200, 90))
    return img


def mib(x):
    return f"{x / 1024**2:.0f} MiB"


def main():
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2VLForConditionalGeneration,
    )

    assert torch.cuda.is_available(), "no CUDA device visible"
    torch.cuda.reset_peak_memory_stats()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"loading {MODEL_ID} in 4-bit ...")
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    print(f"loaded in {time.perf_counter()-t0:.1f}s | VRAM after load: {mib(torch.cuda.memory_allocated())}")

    image = make_test_image()
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text",
                     "text": "What shapes and colors are in this image?"}],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
    print(f"prompt tokens: {inputs.input_ids.shape[1]} (includes vision tokens)")

    t1 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    dt = time.perf_counter() - t1

    gen = out[0][inputs.input_ids.shape[1]:]
    answer = processor.tokenizer.decode(gen, skip_special_tokens=True)
    n_new = gen.shape[0]

    print("\n--- generation ---")
    print(answer.strip())
    print("------------------")
    print(f"generated {n_new} tokens in {dt:.2f}s -> {n_new/dt:.1f} tok/s")
    print(f"PEAK VRAM: {mib(torch.cuda.max_memory_allocated())}  (budget 8188 MiB)")


if __name__ == "__main__":
    main()
