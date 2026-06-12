"""M-RoPE 3-D position ids for Qwen2-VL (reimplements HF's get_rope_index).

Text tokens get sequential positions on all three (t,h,w) axes; image tokens get
their grid (t,h,w) positions offset after the preceding text, and text after an
image resumes from max+1. Covers the engine's case (images, no video, no padding
mask); verified equal to HF (scripts/proto_custom_positions.py).
"""

from __future__ import annotations

import torch

IMAGE_TOKEN_ID = 151655
VISION_START_TOKEN_ID = 151652
SPATIAL_MERGE_SIZE = 2


def mrope_position_ids(input_ids: torch.Tensor, image_grid_thw: torch.Tensor,
                       image_token_id: int = IMAGE_TOKEN_ID,
                       vision_start_token_id: int = VISION_START_TOKEN_ID,
                       spatial_merge_size: int = SPATIAL_MERGE_SIZE):
    """input_ids: [B, L] (no padding). Returns (position_ids [3,B,L], deltas [B,1])."""
    device = input_ids.device
    B, L = input_ids.shape
    position_ids = torch.ones(3, B, L, dtype=torch.long, device=device)
    deltas = []
    for b in range(B):
        ids = input_ids[b]
        tokens = ids.tolist()
        vis_starts = (ids == vision_start_token_id).nonzero().squeeze(1)
        n_images = int((ids[vis_starts + 1] == image_token_id).sum()) if len(vis_starts) else 0

        pieces = []
        st = 0
        img_i = 0
        for _ in range(n_images):
            ed = tokens.index(image_token_id, st)
            t, h, w = (int(x) for x in image_grid_thw[img_i])
            img_i += 1
            gh, gw = h // spatial_merge_size, w // spatial_merge_size
            text_len = ed - st
            st_idx = pieces[-1].max() + 1 if pieces else 0
            # text before this image
            pieces.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
            # the image's 3-D grid positions, offset after that text
            ti = torch.arange(t).view(-1, 1).expand(-1, gh * gw).flatten()
            hi = torch.arange(gh).view(1, -1, 1).expand(t, -1, gw).flatten()
            wi = torch.arange(gw).view(1, 1, -1).expand(t, gh, -1).flatten()
            pieces.append(torch.stack([ti, hi, wi]) + text_len + st_idx)
            st = ed + t * gh * gw
        # trailing text
        if st < len(tokens):
            st_idx = pieces[-1].max() + 1 if pieces else 0
            pieces.append(torch.arange(len(tokens) - st).view(1, -1).expand(3, -1) + st_idx)

        pos = torch.cat(pieces, dim=1).to(device)      # [3, L]
        position_ids[:, b, :] = pos
        deltas.append(int(pos.max()) + 1 - L)
    return position_ids, torch.tensor(deltas, device=device).unsqueeze(1)
