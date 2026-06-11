"""Shared workload + metrics for the omniserve vs vLLM comparison.

Every backend is fed the *identical* set of (image, prompt) requests and the
same decoding settings, so differences in the reported numbers come from the
serving engine, not the inputs. Images are synthesized locally (no dataset).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from typing import List

from PIL import Image, ImageDraw

# Decoding settings shared by all backends (greedy, fixed budget).
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
    """n requests; reuse_rate controls how many share an image (for the
    vision-cache story). reuse_rate=0 -> all distinct images."""
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
