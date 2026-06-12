"""omniserve vs vLLM 对比用的共享 workload + 指标。

每个后端都喂**完全相同**的 (图, prompt) 请求集和相同的解码设置,所以报告数字上的
差异来自 serving 引擎本身,不是来自输入。图片在本地合成(不用数据集)。
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from typing import List

from PIL import Image, ImageDraw

# 所有后端共用的解码设置(greedy,固定预算)。
MAX_NEW_TOKENS = 64
PROMPT = "Describe the shapes and their colors in this image."


@dataclass
class Req:
    idx: int
    image_seed: int


def make_image(seed: int, size: int = 448) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size), (rng.randint(0, 70),) * 3)
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(3, 7)):
        x0, y0 = rng.randint(0, size - 1), rng.randint(0, size - 1)
        x1, y1 = rng.randint(0, size - 1), rng.randint(0, size - 1)
        c = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        (d.rectangle if rng.random() < 0.5 else d.ellipse)(box, fill=c)
    return img


def build(n: int, reuse_rate: float = 0.0, seed: int = 0) -> List[Req]:
    """n 个请求;reuse_rate 控制多少个共用一张图(给 vision-cache 的故事用)。
    reuse_rate=0 -> 全部是不同的图。"""
    rng = random.Random(seed)
    n_unique = max(1, round(n * (1.0 - reuse_rate)))
    reqs = []
    for i in range(n):
        s = i if i < n_unique else rng.randrange(n_unique)
        reqs.append(Req(idx=i, image_seed=s))
    return reqs


@dataclass
class Metrics:
    backend: str
    n_requests: int
    total_tokens: int
    wall_s: float
    peak_mem_mib: float

    @property
    def req_per_s(self) -> float:
        return self.n_requests / self.wall_s if self.wall_s else 0.0

    @property
    def tok_per_s(self) -> float:
        return self.total_tokens / self.wall_s if self.wall_s else 0.0

    def dump(self, path: str):
        d = asdict(self)
        d["req_per_s"] = round(self.req_per_s, 3)
        d["tok_per_s"] = round(self.tok_per_s, 1)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        return d
