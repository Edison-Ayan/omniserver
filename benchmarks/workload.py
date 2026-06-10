"""Workload generator for multimodal inference benchmarking.

Produces a request stream with a *controllable image reuse rate* — the axis we
sweep to show the vision-embedding cache's effect. Higher reuse => more ViT
forward passes the cache can skip. Images are synthesized locally so the
benchmark needs no dataset download, and identical seeds produce byte-identical
images (hence identical processor pixel_values => identical cache keys).
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
    image_id: int  # ground-truth id for verifying cache hit/miss accounting


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
    """reuse_rate in [0,1]: fraction of requests that reuse an already-seen image.
    0 -> every request a unique image (no cache benefit); 1 -> all share one image.
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
