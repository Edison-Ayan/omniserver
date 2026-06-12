"""nsys 剖析 vLLM 的 decode,和 omniserve 对比。enforce_eager 让 kernel 可见,
强制长生成(ignore_eos)让 decode 主导整个 kernel summary。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "compare"))

from PIL import Image, ImageDraw
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2-VL-2B-Instruct"


def img(s):
    import random
    rng = random.Random(s)
    im = Image.new("RGB", (448, 448), (rng.randint(0, 60),) * 3)
    d = ImageDraw.Draw(im)
    d.rectangle([rng.randint(0, 200), rng.randint(0, 200),
                 rng.randint(220, 440), rng.randint(220, 440)], fill=(rng.randint(0, 255),) * 3)
    return im


def main():
    proc = AutoProcessor.from_pretrained(MODEL)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}]
    ptext = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = [{"prompt": ptext, "multi_modal_data": {"image": img(i)}} for i in range(16)]
    llm = LLM(model=MODEL, max_model_len=2048, gpu_memory_utilization=0.9,
              limit_mm_per_prompt={"image": 1}, dtype="float16", enforce_eager=True)
    sp = SamplingParams(max_tokens=200, temperature=0, ignore_eos=True)
    pass  # 不预热,缩短


    llm.generate(inputs, sp)   # 被 nsys 抓:16×200 token,decode 主导



if __name__ == "__main__":
    main()
