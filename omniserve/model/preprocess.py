"""Qwen2-VL image preprocessing (reimplements the HF image processor).

PIL image -> (pixel_values [num_patches, 1176], grid_thw [1, 3]). Smart-resize to
a multiple of patch*merge=28 within the pixel budget, CLIP-normalize, then patchify
into the temporal/spatial-merge-grouped layout the ViT expects. Verified equal to
HF's processor (scripts/proto_custom_preprocess.py).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
PATCH_SIZE = 14
TEMPORAL_PATCH = 2
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE  # 28
MIN_PIXELS = 56 * 56
MAX_PIXELS = 14 * 14 * 4 * 1280


def smart_resize(h, w, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    h_bar = round(h / factor) * factor
    w_bar = round(w / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return h_bar, w_bar


def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    w, h = img.size
    rh, rw = smart_resize(h, w)
    img = img.resize((rw, rh), Image.BICUBIC)

    arr = np.asarray(img, dtype=np.float32) / 255.0      # [rh, rw, 3]
    arr = (arr - CLIP_MEAN) / CLIP_STD
    arr = arr.transpose(2, 0, 1)                          # [3, rh, rw]

    # repeat the single frame to temporal_patch_size, then patchify
    patches = np.repeat(arr[np.newaxis], TEMPORAL_PATCH, axis=0)  # [T, 3, rh, rw]
    gt, gh, gw = 1, rh // PATCH_SIZE, rw // PATCH_SIZE
    patches = patches.reshape(
        gt, TEMPORAL_PATCH, 3,
        gh // MERGE_SIZE, MERGE_SIZE, PATCH_SIZE,
        gw // MERGE_SIZE, MERGE_SIZE, PATCH_SIZE)
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = patches.reshape(gt * gh * gw, 3 * TEMPORAL_PATCH * PATCH_SIZE * PATCH_SIZE)
    pixel_values = torch.from_numpy(flat)
    grid_thw = torch.tensor([[gt, gh, gw]], dtype=torch.long)
    return pixel_values, grid_thw
