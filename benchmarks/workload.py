"""多模态推理 benchmark 的 workload 生成器。

产出一个请求流,带**可控的图片复用率**——这是我们用来展示 vision-embedding cache
效果的扫描维度。复用越高 => cache 能跳过的 ViT 前向越多。图片在本地合成,所以
benchmark 不用下数据集,而且相同 seed 产生逐字节相同的图(因此 processor 的
pixel_values 相同 => cache key 相同)。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from PIL import Image, ImageDraw


@dataclass
class Request:
    image: Image.Image
    prompt: str
    image_id: int  # 真实 id,用于核对 cache 命中/未命中的统计


QUESTIONS = [
    "Describe this image in one sentence.",
    "What colors dominate this picture?",
    "How many shapes do you see?",
    "Is there anything unusual here?",
    "What is in the top-left corner?",
    "Summarize the scene.",
    "What mood does this image convey?",
    "List the objects you can identify.",
]


def _make_image(seed: int, size: int = 448) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size), (rng.randint(0, 80),) * 3)
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(3, 8)):
        x0, y0 = rng.randint(0, size - 1), rng.randint(0, size - 1)
        x1, y1 = rng.randint(0, size - 1), rng.randint(0, size - 1)
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        (draw.rectangle if rng.random() < 0.5 else draw.ellipse)(box, fill=color)
    return img


def build_workload(n_requests: int, reuse_rate: float, seed: int = 0) -> List[Request]:
    """reuse_rate ∈ [0,1]:复用已出现过图片的请求占比。
    0 -> 每个请求都是唯一图(cache 无收益);1 -> 全部共用一张图。
    """
    assert 0.0 <= reuse_rate <= 1.0
    rng = random.Random(seed)
    n_unique = max(1, round(n_requests * (1.0 - reuse_rate)))
    pool = [(_make_image(i), i) for i in range(n_unique)]
    requests: List[Request] = []
    for k in range(n_requests):
        img, img_id = pool[k] if k < n_unique else rng.choice(pool)
        requests.append(Request(image=img, prompt=rng.choice(QUESTIONS), image_id=img_id))
    rng.shuffle(requests)
    return requests
