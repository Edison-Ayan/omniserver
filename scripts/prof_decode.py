"""给 nsys 用的 decode 专测脚本:进入稳定 decode 后跑 N 步,nsys 抓 GPU kernel 分布。
    nsys profile -o /tmp/dec python scripts/prof_decode.py
    nsys stats --report cuda_gpu_kern_sum /tmp/dec.nsys-rep
"""
import torch
from PIL import Image, ImageDraw
from omniserve import LLMEngine, Request, SamplingParams, SchedulerConfig
from omniserve.runners.qwen_vl_native import NativeQwenVLRunner


def img(s):
    import random
    rng = random.Random(s)
    im = Image.new("RGB", (448, 448), (rng.randint(0, 60),) * 3)
    d = ImageDraw.Draw(im)
    d.rectangle([rng.randint(0, 200), rng.randint(0, 200),
                 rng.randint(220, 440), rng.randint(220, 440)], fill=(rng.randint(0, 255),) * 3)
    return im


def main():
    r = NativeQwenVLRunner(max_running=16, max_len=512)
    eng = LLMEngine(r, SchedulerConfig(max_running=16, max_prefill_per_step=1))
    for i in range(16):
        eng.add_request(Request(prompt="Describe.", images=[img(i)],
                                sampling=SamplingParams(max_new_tokens=300)))
    for _ in range(18):
        eng.step()                         # 跑完 prefill,进入稳定 decode
    seqs = list(eng.scheduler.running)
    for _ in range(5):
        r.decode(seqs); torch.cuda.synchronize()   # 预热
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(40):                    # 被 nsys 抓的部分:纯 decode
        r.decode(seqs)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
